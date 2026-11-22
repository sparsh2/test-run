"""Explore economically grounded kinematic variants to test if a durable OOS edge exists on AAPL.

Tests three distinct hypotheses under strict chronological Train / Validation / OOS Test splits
with realistic execution friction (1.5 bps/side = 3.0 bps round-trip) and zero lookahead:

  H1: Higher-Timeframe (5m / 15m) Kinematic Momentum + VWAP Trend Filter (Intraday, Long & Short)
  H2: Kinematic Deceleration / Exhaustion Mean-Reversion (5m / 15m, fade extreme velocity when accel flips)
  H3: Higher-Timeframe (15m / 30m) Long/Cash Kinematic Trend Filter (hold overnight, step to cash in downtrends)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import Trade, compute_metrics
from src.features import FeatureConfig, compute_kinematic_features
from src.fetch_data import data_path_for_symbol, load_data
from src.optimize import DEFAULT_COST_BPS, split_chronological

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def resample_rth_bars(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1-minute RTH bars to higher timeframe (e.g. '5min', '15min', '30min') per day."""
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "vwap" in df_1m.columns:
        # Compute volume-weighted VWAP accurately
        df_tmp = df_1m.copy()
        df_tmp["_pv"] = df_tmp["vwap"] * df_tmp["volume"]
        resampled = df_tmp.resample(rule, origin="start_day", offset="30min").agg(
            {**agg, "_pv": "sum"}
        )
        resampled = resampled.dropna(subset=["open", "close"])
        resampled = resampled[resampled["volume"] > 0]
        resampled["vwap"] = resampled["_pv"] / resampled["volume"]
        resampled = resampled.drop(columns=["_pv"])
    else:
        resampled = df_1m.resample(rule, origin="start_day", offset="30min").agg(agg)
        resampled = resampled.dropna(subset=["open", "close"])

    # Keep only RTH bars
    times = resampled.index.time
    mask = (times >= time(9, 30)) & (times <= time(15, 59))
    return resampled.loc[mask].copy()


def compute_intraday_vwap(df: pd.DataFrame) -> np.ndarray:
    """Strictly causal cumulative intraday VWAP resetting each trading day."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    dates = df.index.date
    cum_pv = pv.groupby(dates).cumsum()
    cum_v = df["volume"].groupby(dates).cumsum()
    return (cum_pv / cum_v.replace(0, np.nan)).bfill().to_numpy(dtype=np.float64)


def run_signal_backtest(
    df: pd.DataFrame,
    target_pos_at_close: np.ndarray,
    cost_bps_per_side: float = DEFAULT_COST_BPS,
    stop_loss_bps: float = 0.0,
    take_profit_bps: float = 0.0,
) -> dict[str, float]:
    """Zero-lookahead execution engine: target_pos_at_close[t] executes at bar t+1 OPEN."""
    n = len(df)
    opens = df["open"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    timestamps = df.index

    cost_rate = cost_bps_per_side / 10_000.0
    sl_frac = stop_loss_bps / 10_000.0
    tp_frac = take_profit_bps / 10_000.0
    use_sl = sl_frac > 0.0
    use_tp = tp_frac > 0.0

    equity_arr = np.empty(n, dtype=np.float64)
    trades: list[Trade] = []

    equity = 1.0
    pos = 0
    entry_price = 0.0
    entry_idx = -1
    pending = 0
    equity_arr[0] = 1.0

    for t in range(n):
        if t > 0 and pending != pos:
            exec_price = opens[t]
            if pos != 0:
                gross = pos * (exec_price - entry_price) / entry_price
                net = gross - 2.0 * cost_rate
                equity *= 1.0 + net
                trades.append(
                    Trade(entry_idx, t, timestamps[entry_idx], timestamps[t], pos, entry_price, exec_price, gross, net, "signal", t - entry_idx)
                )
                pos = 0
            if pending != 0:
                pos = pending
                entry_price = exec_price
                entry_idx = t

        exited_intrabar = False
        if pos != 0 and (use_sl or use_tp):
            if pos == 1:
                sl_p = entry_price * (1.0 - sl_frac) if use_sl else -1.0
                tp_p = entry_price * (1.0 + tp_frac) if use_tp else 1e18
                if use_sl and lows[t] <= sl_p:
                    fill = min(opens[t], sl_p)
                    gross = (fill - entry_price) / entry_price
                    net = gross - 2.0 * cost_rate
                    equity *= 1.0 + net
                    trades.append(Trade(entry_idx, t, timestamps[entry_idx], timestamps[t], 1, entry_price, fill, gross, net, "sl", max(1, t - entry_idx)))
                    pos = 0
                    exited_intrabar = True
                elif use_tp and highs[t] >= tp_p:
                    fill = max(opens[t], tp_p)
                    gross = (fill - entry_price) / entry_price
                    net = gross - 2.0 * cost_rate
                    equity *= 1.0 + net
                    trades.append(Trade(entry_idx, t, timestamps[entry_idx], timestamps[t], 1, entry_price, fill, gross, net, "tp", max(1, t - entry_idx)))
                    pos = 0
                    exited_intrabar = True
            else:
                sl_p = entry_price * (1.0 + sl_frac) if use_sl else 1e18
                tp_p = entry_price * (1.0 - tp_frac) if use_tp else -1.0
                if use_sl and highs[t] >= sl_p:
                    fill = max(opens[t], sl_p)
                    gross = -(fill - entry_price) / entry_price
                    net = gross - 2.0 * cost_rate
                    equity *= 1.0 + net
                    trades.append(Trade(entry_idx, t, timestamps[entry_idx], timestamps[t], -1, entry_price, fill, gross, net, "sl", max(1, t - entry_idx)))
                    pos = 0
                    exited_intrabar = True
                elif use_tp and lows[t] <= tp_p:
                    fill = min(opens[t], tp_p)
                    gross = -(fill - entry_price) / entry_price
                    net = gross - 2.0 * cost_rate
                    equity *= 1.0 + net
                    trades.append(Trade(entry_idx, t, timestamps[entry_idx], timestamps[t], -1, entry_price, fill, gross, net, "tp", max(1, t - entry_idx)))
                    pos = 0
                    exited_intrabar = True

        if pos != 0:
            unreal = pos * (closes[t] - entry_price) / entry_price - cost_rate
            equity_arr[t] = equity * (1.0 + unreal)
        else:
            equity_arr[t] = equity

        if t == n - 1:
            if pos != 0:
                gross = pos * (closes[t] - entry_price) / entry_price
                net = gross - 2.0 * cost_rate
                equity *= 1.0 + net
                equity_arr[t] = equity
                trades.append(Trade(entry_idx, t, timestamps[entry_idx], timestamps[t], pos, entry_price, closes[t], gross, net, "eod", max(1, t - entry_idx)))
            break

        pending = 0 if exited_intrabar else int(target_pos_at_close[t])

    eq_s = pd.Series(equity_arr, index=timestamps)
    bh_s = pd.Series(closes / opens[0], index=timestamps)
    return compute_metrics(eq_s, bh_s, trades, closes, cost_bps_per_side=cost_bps_per_side, compute_oracle=False)


def gen_h1_momentum_vwap(
    df: pd.DataFrame,
    smooth_window: int,
    vel_z_thresh: float,
    acc_z_thresh: float,
    long_only: bool,
    flatten_eod: bool,
) -> np.ndarray:
    """H1: Higher-TF ATR-normalized kinematic momentum gated by intraday VWAP."""
    f_cfg = FeatureConfig(smoother="ema", smooth_window=smooth_window, vel_lookback=2, acc_lookback=1, vol_window=20, reset_daily=flatten_eod)
    feats = compute_kinematic_features(df, f_cfg)
    vwap = compute_intraday_vwap(df)
    closes = df["close"].to_numpy(dtype=np.float64)

    atr = np.maximum(feats["atr_bps"].to_numpy(dtype=np.float64), 1e-3)
    vel_z = feats["velocity"].to_numpy(dtype=np.float64) / atr
    acc_z = feats["acceleration"].to_numpy(dtype=np.float64) / atr
    bar_in_day = feats["bar_in_day"].to_numpy(dtype=np.int32)
    times = df.index.time

    n = len(df)
    target = np.zeros(n, dtype=np.int8)
    pos = 0
    for t in range(n):
        if flatten_eod and times[t] >= time(15, 45):
            pos = 0
        elif flatten_eod and bar_in_day[t] < 3:
            pos = 0
        else:
            above_vwap = closes[t] > vwap[t]
            below_vwap = closes[t] < vwap[t]
            if pos == 0:
                if vel_z[t] > vel_z_thresh and acc_z[t] > acc_z_thresh and above_vwap:
                    pos = 1
                elif (not long_only) and vel_z[t] < -vel_z_thresh and acc_z[t] < -acc_z_thresh and below_vwap:
                    pos = -1
            elif pos == 1:
                if vel_z[t] <= 0.0:
                    pos = 0
            elif pos == -1:
                if vel_z[t] >= 0.0:
                    pos = 0
        target[t] = pos
    return target


def gen_h2_exhaustion_reversion(
    df: pd.DataFrame,
    smooth_window: int,
    vel_z_thresh: float,
    acc_flip_thresh: float,
    max_hold: int,
    long_only: bool,
) -> np.ndarray:
    """H2: Kinematic exhaustion mean-reversion (fade extreme velocity when acceleration reverses)."""
    f_cfg = FeatureConfig(smoother="ema", smooth_window=smooth_window, vel_lookback=2, acc_lookback=1, vol_window=20, reset_daily=True)
    feats = compute_kinematic_features(df, f_cfg)

    atr = np.maximum(feats["atr_bps"].to_numpy(dtype=np.float64), 1e-3)
    vel_z = feats["velocity"].to_numpy(dtype=np.float64) / atr
    acc_z = feats["acceleration"].to_numpy(dtype=np.float64) / atr
    bar_in_day = feats["bar_in_day"].to_numpy(dtype=np.int32)
    times = df.index.time

    n = len(df)
    target = np.zeros(n, dtype=np.int8)
    pos = 0
    held = 0
    for t in range(n):
        if times[t] >= time(15, 45) or bar_in_day[t] < 4:
            pos = 0
            held = 0
        else:
            if pos == 0:
                # Oversold deceleration: strong negative velocity, positive acceleration (turning up)
                if vel_z[t] < -vel_z_thresh and acc_z[t] > acc_flip_thresh:
                    pos = 1
                    held = 0
                # Overbought deceleration: strong positive velocity, negative acceleration (turning down)
                elif (not long_only) and vel_z[t] > vel_z_thresh and acc_z[t] < -acc_flip_thresh:
                    pos = -1
                    held = 0
            else:
                held += 1
                if held >= max_hold:
                    pos = 0
                    held = 0
                elif pos == 1 and vel_z[t] >= 0.0:
                    pos = 0
                    held = 0
                elif pos == -1 and vel_z[t] <= 0.0:
                    pos = 0
                    held = 0
        target[t] = pos
    return target


def gen_h3_swing_trend_filter(
    df: pd.DataFrame,
    smooth_window: int,
    vel_enter: float,
    vel_exit: float,
    allow_short: bool,
) -> np.ndarray:
    """H3: Multi-day higher-TF kinematic trend filter (holds overnight, steps to cash or short in downtrends)."""
    f_cfg = FeatureConfig(smoother="ema", smooth_window=smooth_window, vel_lookback=3, acc_lookback=1, vol_window=20, reset_daily=False)
    feats = compute_kinematic_features(df, f_cfg)
    atr = np.maximum(feats["atr_bps"].to_numpy(dtype=np.float64), 1e-3)
    vel_z = feats["velocity"].to_numpy(dtype=np.float64) / atr
    acc_z = feats["acceleration"].to_numpy(dtype=np.float64) / atr

    n = len(df)
    target = np.zeros(n, dtype=np.int8)
    pos = 0
    for t in range(smooth_window, n):
        if pos == 0:
            if vel_z[t] > vel_enter and acc_z[t] >= 0.0:
                pos = 1
            elif allow_short and vel_z[t] < -vel_enter and acc_z[t] <= 0.0:
                pos = -1
        elif pos == 1:
            if vel_z[t] < vel_exit:
                pos = -1 if (allow_short and vel_z[t] < -vel_enter and acc_z[t] <= 0.0) else 0
        elif pos == -1:
            if vel_z[t] > -vel_exit:
                pos = 1 if (vel_z[t] > vel_enter and acc_z[t] >= 0.0) else 0
        target[t] = pos
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore higher-TF and mean-reversion kinematic hypotheses.")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol (default: AAPL)")
    args = parser.parse_args()
    sym = args.symbol.upper()

    df_1m = load_data(data_path_for_symbol(sym), rth_only=True)
    tr_1m, va_1m, te_1m = split_chronological(df_1m)

    # Precompute resampled splits
    splits = {}
    for tf in ["5min", "15min", "30min"]:
        splits[tf] = (
            resample_rth_bars(tr_1m, tf),
            resample_rth_bars(va_1m, tf),
            resample_rth_bars(te_1m, tf),
        )

    print("=" * 88)
    print("HYPOTHESIS EXPLORATION (STRICT TRAIN -> VAL SELECTION, SINGLE OOS TEST EVALUATION)")
    print("=" * 88)

    # ---------------- H1: Higher-TF Momentum + VWAP ----------------
    h1_rows = []
    for tf in ["5min", "15min"]:
        tr, va, _ = splits[tf]
        for sw in [6, 12, 20]:
            for vz in [0.10, 0.20, 0.35]:
                for az in [0.0, 0.05]:
                    for lo in [False, True]:
                        sig_tr = gen_h1_momentum_vwap(tr, sw, vz, az, lo, flatten_eod=True)
                        m_tr = run_signal_backtest(tr, sig_tr)
                        if m_tr["n_trades"] < 20:
                            continue
                        sig_va = gen_h1_momentum_vwap(va, sw, vz, az, lo, flatten_eod=True)
                        m_va = run_signal_backtest(va, sig_va)
                        # Select by minimum of Train & Val Sharpe (rewards consistency, rejects overfits)
                        score = min(m_tr["sharpe"], m_va["sharpe"])
                        h1_rows.append((score, tf, sw, vz, az, lo, m_tr, m_va))

    h1_rows.sort(key=lambda x: x[0], reverse=True)
    best_h1 = h1_rows[0]
    _, tf, sw, vz, az, lo, m_tr, m_va = best_h1
    _, _, te = splits[tf]
    m_te = run_signal_backtest(te, gen_h1_momentum_vwap(te, sw, vz, az, lo, flatten_eod=True))
    print(f"\n[H1: Higher-TF Momentum + VWAP] Best consistent config: tf={tf}, sw={sw}, vel_z={vz}, acc_z={az}, long_only={lo}")
    for lbl, m in [("Train", m_tr), ("Val  ", m_va), ("OOS  ", m_te)]:
        print(f"  {lbl} | Ret: {m['total_return_pct']:+6.2f}% (B&H {m['bh_return_pct']:+6.2f}%) | Sharpe: {m['sharpe']:5.2f} | DD: {m['max_dd_pct']:6.2f}% | Trades: {int(m['n_trades']):4d} | WR: {m['win_rate_pct']:4.1f}% | PF: {m['profit_factor']:4.2f}")

    # ---------------- H2: Kinematic Exhaustion Mean-Reversion ----------------
    h2_rows = []
    for tf in ["5min", "15min"]:
        tr, va, _ = splits[tf]
        for sw in [4, 8, 14]:
            for vz in [0.15, 0.25, 0.40]:
                for af in [0.02, 0.08]:
                    for mh in [6, 12, 24]:
                        for lo in [False, True]:
                            sig_tr = gen_h2_exhaustion_reversion(tr, sw, vz, af, mh, lo)
                            m_tr = run_signal_backtest(tr, sig_tr)
                            if m_tr["n_trades"] < 20:
                                continue
                            sig_va = gen_h2_exhaustion_reversion(va, sw, vz, af, mh, lo)
                            m_va = run_signal_backtest(va, sig_va)
                            score = min(m_tr["sharpe"], m_va["sharpe"])
                            h2_rows.append((score, tf, sw, vz, af, mh, lo, m_tr, m_va))

    h2_rows.sort(key=lambda x: x[0], reverse=True)
    best_h2 = h2_rows[0]
    _, tf, sw, vz, af, mh, lo, m_tr, m_va = best_h2
    _, _, te = splits[tf]
    m_te = run_signal_backtest(te, gen_h2_exhaustion_reversion(te, sw, vz, af, mh, lo))
    print(f"\n[H2: Kinematic Exhaustion Mean-Reversion] Best consistent config: tf={tf}, sw={sw}, vel_z={vz}, acc_flip={af}, max_hold={mh}, long_only={lo}")
    for lbl, m in [("Train", m_tr), ("Val  ", m_va), ("OOS  ", m_te)]:
        print(f"  {lbl} | Ret: {m['total_return_pct']:+6.2f}% (B&H {m['bh_return_pct']:+6.2f}%) | Sharpe: {m['sharpe']:5.2f} | DD: {m['max_dd_pct']:6.2f}% | Trades: {int(m['n_trades']):4d} | WR: {m['win_rate_pct']:4.1f}% | PF: {m['profit_factor']:4.2f}")

    # ---------------- H3: Multi-Day Kinematic Trend Filter (Long/Cash or Long/Short) ----------------
    h3_rows = []
    for tf in ["15min", "30min"]:
        tr, va, _ = splits[tf]
        for sw in [20, 40, 80, 120]:
            for ve in [0.02, 0.08, 0.15]:
                for vx in [0.0, -0.05, -0.10]:
                    for sh in [False, True]:
                        sig_tr = gen_h3_swing_trend_filter(tr, sw, ve, vx, sh)
                        m_tr = run_signal_backtest(tr, sig_tr)
                        if m_tr["n_trades"] < 6:
                            continue
                        sig_va = gen_h3_swing_trend_filter(va, sw, ve, vx, sh)
                        m_va = run_signal_backtest(va, sig_va)
                        score = min(m_tr["sharpe"], m_va["sharpe"])
                        h3_rows.append((score, tf, sw, ve, vx, sh, m_tr, m_va))

    h3_rows.sort(key=lambda x: x[0], reverse=True)
    best_h3 = h3_rows[0]
    _, tf, sw, ve, vx, sh, m_tr, m_va = best_h3
    _, _, te = splits[tf]
    m_te = run_signal_backtest(te, gen_h3_swing_trend_filter(te, sw, ve, vx, sh))
    print(f"\n[H3: Multi-Day Kinematic Swing Filter] Best consistent config: tf={tf}, sw={sw}, vel_enter={ve}, vel_exit={vx}, allow_short={sh}")
    for lbl, m in [("Train", m_tr), ("Val  ", m_va), ("OOS  ", m_te)]:
        print(f"  {lbl} | Ret: {m['total_return_pct']:+6.2f}% (B&H {m['bh_return_pct']:+6.2f}%) | Sharpe: {m['sharpe']:5.2f} | DD: {m['max_dd_pct']:6.2f}% | Trades: {int(m['n_trades']):4d} | WR: {m['win_rate_pct']:4.1f}% | PF: {m['profit_factor']:4.2f}")


if __name__ == "__main__":
    main()
