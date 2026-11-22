"""Strictly causal kinematic feature engineering (velocity, acceleration, filters).

All transformations use only current and past observations (t, t-1, ...).
No future lookahead is permitted anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter, savgol_coeffs


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for causal smoothing and kinematic feature extraction."""

    smoother: str = "ema"  # "ema", "dema", "savgol", "linreg", "kalman"
    smooth_window: int = 10  # Span for EMA/DEMA, window for SavGol/LinReg, or noise ratio for Kalman
    vel_lookback: int = 3  # Lookback n for velocity: (s_t - s_{t-n}) / n
    acc_lookback: int = 1  # Lookback m for acceleration: (v_t - v_{t-m}) / m
    vol_window: int = 20  # Rolling window for relative volume & volatility (ATR/std)
    kalman_q: float = 1e-4  # Process noise scale for Kalman filter
    kalman_r: float = 1e-2  # Measurement noise variance for Kalman filter
    reset_daily: bool = True  # Prevent overnight gaps from corrupting intraday derivatives
    price_col: str = "close"  # Input price column for kinematics ("close" or "vwap")


def _causal_ema(x: np.ndarray, span: int) -> np.ndarray:
    """Strictly causal Exponential Moving Average."""
    alpha = 2.0 / (span + 1.0)
    # y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
    b = np.array([alpha], dtype=np.float64)
    a = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)
    zi = np.array([x[0] * (1.0 - alpha)], dtype=np.float64)
    y, _ = lfilter(b, a, x, zi=zi)
    return y


def _causal_dema(x: np.ndarray, span: int) -> np.ndarray:
    """Strictly causal Double Exponential Moving Average (reduced lag)."""
    e1 = _causal_ema(x, span)
    e2 = _causal_ema(e1, span)
    return 2.0 * e1 - e2


def _causal_savgol(x: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    """Strictly causal Savitzky-Golay filter evaluated at the right endpoint (pos = window - 1)."""
    w = max(int(window), polyorder + 2)
    # pos=w-1 evaluates the fitted polynomial at the most recent (rightmost) point
    # savgol_coeffs returns coefficients ordered from oldest to newest when use='conv' is false;
    # for causal FIR filter y[t] = sum_{k=0}^{w-1} h[k] * x[t-k], we need newest-first (use="conv").
    coeffs = savgol_coeffs(w, polyorder=polyorder, deriv=0, pos=w - 1, use="conv")
    # Pad start with x[0] so initial transient matches starting level without lookahead
    padded = np.concatenate([np.full(w - 1, x[0], dtype=np.float64), x])
    y = np.convolve(padded, coeffs, mode="valid")
    return y


def _causal_linreg(x: np.ndarray, window: int) -> np.ndarray:
    """Strictly causal rolling OLS linear regression endpoint value (degree-1 SavGol)."""
    w = max(int(window), 3)
    return _causal_savgol(x, window=w, polyorder=1)


def _causal_kalman_ca(
    x: np.ndarray, q: float = 1e-4, r: float = 1e-2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strictly causal 1D Constant-Acceleration Kalman Filter.

    State vector: [position (log price), velocity, acceleration]^T with dt = 1 bar.
    Returns (pos_est, vel_est, acc_est).
    """
    n = len(x)
    pos_out = np.empty(n, dtype=np.float64)
    vel_out = np.empty(n, dtype=np.float64)
    acc_out = np.empty(n, dtype=np.float64)

    # State transition matrix F (dt = 1)
    # x_{t|t-1} = [p + v + 0.5*a, v + a, a]
    # Process noise Q = q * G * G^T where G = [0.5, 1.0, 1.0]^T
    q00 = 0.25 * q
    q01 = 0.5 * q
    q02 = 0.5 * q
    q11 = 1.0 * q
    q12 = 1.0 * q
    q22 = 1.0 * q

    # Initialize state at first observation
    s0 = float(x[0])
    s1 = 0.0
    s2 = 0.0

    # Covariance matrix P (symmetric 3x3)
    p00, p01, p02 = r, 0.0, 0.0
    p11, p12 = 1e-4, 0.0
    p22 = 1e-5

    pos_out[0] = s0
    vel_out[0] = 0.0
    acc_out[0] = 0.0

    for t in range(1, n):
        # 1. Predict state: s_pred = F @ s
        ps0 = s0 + s1 + 0.5 * s2
        ps1 = s1 + s2
        ps2 = s2

        # 2. Predict covariance: P_pred = F @ P @ F^T + Q
        # First compute FP = F @ P
        fp00 = p00 + p01 + 0.5 * p02
        fp01 = p01 + p11 + 0.5 * p12
        fp02 = p02 + p12 + 0.5 * p22

        fp10 = p01 + p02
        fp11 = p11 + p12
        fp12 = p12 + p22

        fp20 = p02
        fp21 = p12
        fp22 = p22

        # Now P_pred = FP @ F^T + Q (only upper triangle needed)
        pp00 = fp00 + fp01 + 0.5 * fp02 + q00
        pp01 = fp01 + fp02 + q01
        pp02 = fp02 + q02
        pp11 = fp11 + fp12 + q11
        pp12 = fp12 + q12
        pp22 = fp22 + q22

        # 3. Innovation (measurement H = [1, 0, 0])
        innov = x[t] - ps0
        S = pp00 + r
        inv_S = 1.0 / S

        # Kalman gain K = P_pred @ H^T / S
        k0 = pp00 * inv_S
        k1 = pp01 * inv_S
        k2 = pp02 * inv_S

        # 4. Update state
        s0 = ps0 + k0 * innov
        s1 = ps1 + k1 * innov
        s2 = ps2 + k2 * innov

        # 5. Update covariance: P = (I - K H) P_pred
        p00 = pp00 - k0 * pp00
        p01 = pp01 - k0 * pp01
        p02 = pp02 - k0 * pp02
        p11 = pp11 - k1 * pp01
        p12 = pp12 - k1 * pp02
        p22 = pp22 - k2 * pp02

        pos_out[t] = s0
        vel_out[t] = s1
        acc_out[t] = s2

    return pos_out, vel_out, acc_out


def _smooth_segment(
    log_price: np.ndarray, cfg: FeatureConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (smoothed_log_price, velocity, acceleration) on a contiguous price segment."""
    n = len(log_price)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    smoother = cfg.smoother.lower()
    if smoother == "kalman":
        # Scale R by smooth_window^2 so smooth_window acts intuitively as a smoothing knob
        r_eff = cfg.kalman_r * (cfg.smooth_window / 10.0) ** 2
        s, v_raw, a_raw = _causal_kalman_ca(log_price, q=cfg.kalman_q, r=r_eff)
        if cfg.vel_lookback == 1 and cfg.acc_lookback == 1:
            return s, v_raw, a_raw
        # Allow multi-bar lookback on Kalman smoothed price if requested
        v = np.zeros(n, dtype=np.float64)
        vl = max(1, cfg.vel_lookback)
        if n > vl:
            v[vl:] = (s[vl:] - s[:-vl]) / vl
        a = np.zeros(n, dtype=np.float64)
        al = max(1, cfg.acc_lookback)
        if n > al:
            a[al:] = (v[al:] - v[:-al]) / al
        return s, v, a

    if smoother == "ema":
        s = _causal_ema(log_price, cfg.smooth_window)
    elif smoother == "dema":
        s = _causal_dema(log_price, cfg.smooth_window)
    elif smoother == "savgol":
        s = _causal_savgol(log_price, cfg.smooth_window, polyorder=2)
    elif smoother == "linreg":
        s = _causal_linreg(log_price, cfg.smooth_window)
    else:
        raise ValueError(f"Unknown smoother: {cfg.smoother!r}")

    vl = max(1, cfg.vel_lookback)
    al = max(1, cfg.acc_lookback)

    v = np.zeros(n, dtype=np.float64)
    if n > vl:
        v[vl:] = (s[vl:] - s[:-vl]) / vl

    a = np.zeros(n, dtype=np.float64)
    if n > al:
        a[al:] = (v[al:] - v[:-al]) / al

    return s, v, a


def compute_kinematic_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Compute strictly causal kinematic features on an OHLCV DataFrame.

    Returns a DataFrame aligned with `df.index` containing:
      - log_close: ln(close)
      - smooth_log_close: causally smoothed log close
      - smooth_price: exp(smooth_log_close)
      - velocity: smoothed log-price 1st derivative (in bps per bar = * 10,000)
      - acceleration: velocity 1st derivative (in bps per bar^2)
      - rel_volume: volume / causal rolling mean volume
      - vol_momentum: rel_volume * velocity ("mass * velocity")
      - atr_bps: causal rolling ATR as basis points of close
      - ret_std_bps: causal rolling std of 1-min log returns in bps
      - bar_in_day: 0-indexed bar counter within each trading date (for warmup masking)
    """
    close = df["close"].to_numpy(dtype=np.float64)
    if cfg.price_col in df.columns:
        sig_price = df[cfg.price_col].fillna(df["close"]).to_numpy(dtype=np.float64)
    else:
        sig_price = close
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)

    log_close = np.log(sig_price)
    n = len(df)

    smooth_log = np.empty(n, dtype=np.float64)
    velocity = np.empty(n, dtype=np.float64)
    acceleration = np.empty(n, dtype=np.float64)
    bar_in_day = np.empty(n, dtype=np.int32)

    dates = df.index.date

    if cfg.reset_daily:
        # Find contiguous day boundaries
        day_change = np.empty(n, dtype=bool)
        day_change[0] = True
        day_change[1:] = dates[1:] != dates[:-1]
        starts = np.flatnonzero(day_change)
        ends = np.append(starts[1:], n)

        for st, en in zip(starts, ends):
            s_seg, v_seg, a_seg = _smooth_segment(log_close[st:en], cfg)
            smooth_log[st:en] = s_seg
            velocity[st:en] = v_seg
            acceleration[st:en] = a_seg
            bar_in_day[st:en] = np.arange(en - st, dtype=np.int32)
    else:
        smooth_log, velocity, acceleration = _smooth_segment(log_close, cfg)
        # Still compute bar_in_day for reference
        day_change = np.empty(n, dtype=bool)
        day_change[0] = True
        day_change[1:] = dates[1:] != dates[:-1]
        starts = np.flatnonzero(day_change)
        ends = np.append(starts[1:], n)
        for st, en in zip(starts, ends):
            bar_in_day[st:en] = np.arange(en - st, dtype=np.int32)

    # Convert velocity and acceleration to basis points (1 bp = 0.01% = 1e-4)
    velocity_bps = velocity * 10_000.0
    acceleration_bps = acceleration * 10_000.0

    # Causal rolling volume ratio ("mass")
    vol_s = pd.Series(volume, index=df.index)
    vol_ma = vol_s.rolling(cfg.vol_window, min_periods=1).mean().to_numpy(dtype=np.float64)
    rel_volume = np.where(vol_ma > 0.0, volume / vol_ma, 1.0)
    vol_momentum = rel_volume * velocity_bps

    # Causal ATR in basis points of close
    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    tr_bps = (tr / close) * 10_000.0
    atr_bps = pd.Series(tr_bps, index=df.index).rolling(cfg.vol_window, min_periods=1).mean().to_numpy()

    # Causal rolling volatility of 1-min log returns in bps
    log_ret = np.zeros(n, dtype=np.float64)
    log_ret[1:] = (log_close[1:] - log_close[:-1]) * 10_000.0
    ret_std_bps = (
        pd.Series(log_ret, index=df.index)
        .rolling(cfg.vol_window, min_periods=2)
        .std()
        .fillna(0.0)
        .to_numpy()
    )

    out = pd.DataFrame(
        {
            "log_close": log_close,
            "smooth_log_close": smooth_log,
            "smooth_price": np.exp(smooth_log),
            "velocity": velocity_bps,
            "acceleration": acceleration_bps,
            "rel_volume": rel_volume,
            "vol_momentum": vol_momentum,
            "atr_bps": atr_bps,
            "ret_std_bps": ret_std_bps,
            "bar_in_day": bar_in_day,
        },
        index=df.index,
    )
    return out
