"""Strictly causal Bollinger Band Peak & Trough Strategy + multi-period zoomed trade charts.

Instead of a fixed % rebound/pullback, uses causal Bollinger Bands (rolling mean +/- k * std):
  - Mode 'band_reentry':
      * Trough confirmed -> BUY: price dips below Lower Band, then closes back above Lower Band.
      * Peak confirmed -> SELL: price rises above Upper Band, then closes back below Upper Band.
  - Mode 'bb_width':
      * Trailing trough/peak reversal where the rebound/pullback threshold is k * rolling_std
        (automatically widens in high volatility and tightens in quiet markets).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.explore_edge import resample_rth_bars, run_signal_backtest
from src.fetch_data import DEFAULT_OUTPUT_PATH, load_data
from src.optimize import split_chronological
from src.walk_forward import walk_forward_eval

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def compute_bollinger_bands(
    closes: np.ndarray, window: int, k: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Strictly causal rolling Bollinger Bands (ma, upper, lower, std)."""
    s = pd.Series(closes)
    ma = s.rolling(window, min_periods=2).mean().bfill().to_numpy(dtype=np.float64)
    sd = s.rolling(window, min_periods=2).std().fillna(0.0).to_numpy(dtype=np.float64)
    upper = ma + k * sd
    lower = ma - k * sd
    return ma, upper, lower, sd


def gen_bb_peak_trough_signals(
    df: pd.DataFrame,
    bb_window: int = 20,
    k_rebound: float = 1.5,
    k_pullback: float = 2.0,
    mode: str = "band_reentry",
    allow_short: bool = False,
) -> np.ndarray:
    """Generate strictly causal target positions (-1, 0, +1) using Bollinger Band peak/trough rules."""
    closes = df["close"].to_numpy(dtype=np.float64)
    n = len(closes)

    ma, _, _, sd = compute_bollinger_bands(closes, bb_window, 1.0)
    lower = ma - k_rebound * sd
    upper = ma + k_pullback * sd

    target = np.zeros(n, dtype=np.int8)
    state = 0  # 0 = flat/short, 1 = long

    if mode == "band_reentry":
        touched_lower = False
        touched_upper = False
        for t in range(bb_window, n):
            p = closes[t]
            if state == 0:
                if p < lower[t]:
                    touched_lower = True
                elif touched_lower and p >= lower[t]:
                    # Confirmed bounce back inside lower Bollinger Band -> BUY
                    state = 1
                    touched_lower = False
                    touched_upper = False
            else:
                if p > upper[t]:
                    touched_upper = True
                elif touched_upper and p <= upper[t]:
                    # Confirmed pullback back inside upper Bollinger Band -> SELL
                    state = 0
                    touched_upper = False
                    touched_lower = False
            target[t] = 1 if state == 1 else (-1 if allow_short else 0)
    elif mode == "bb_width":
        running_low = float(closes[0])
        running_high = float(closes[0])
        for t in range(bb_window, n):
            p = closes[t]
            rebound_dist = max(k_rebound * sd[t], p * 0.002)
            pullback_dist = max(k_pullback * sd[t], p * 0.002)
            if state == 0:
                if p < running_low:
                    running_low = p
                elif p >= running_low + rebound_dist:
                    state = 1
                    running_high = p
            else:
                if p > running_high:
                    running_high = p
                elif p <= running_high - pullback_dist:
                    state = 0
                    running_low = p
            target[t] = 1 if state == 1 else (-1 if allow_short else 0)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return target


def plot_zoomed_periods(
    res: dict,
    df_15m: pd.DataFrame,
    bb_window_ref: int = 20,
    k_ref: float = 2.0,
) -> list[Path]:
    """Generate 3 zoomed shorter-period (~1-month) charts with Bollinger Bands + exact trades."""
    price = res["price"]
    pos = res["positions"]
    eq = res["equity"]
    bh = res["bh_equity"]
    trades = res["trades"]

    # Pick 3 distinct ~4-week windows across the OOS year:
    # 1. Early selloff/chop, 2. Mid-year rally, 3. Recent autumn chop
    periods = [
        ("bb_zoom_1_selloff.png", "2026-01-05", "2026-02-20", "Period 1: Winter Selloff & Rebound (Jan–Feb 2026)"),
        ("bb_zoom_2_rally.png", "2026-06-22", "2026-08-07", "Period 2: Summer Rally & Volatility (Jun–Aug 2026)"),
        ("bb_zoom_3_recent.png", "2026-10-05", "2026-11-20", "Period 3: Recent Choppy Market (Oct–Nov 2026)"),
    ]

    # Precompute reference Bollinger Bands on full 15m series so window edges have proper warmup
    ma_all, up_all, lo_all, _ = compute_bollinger_bands(
        df_15m["close"].to_numpy(dtype=np.float64), bb_window_ref, k_ref
    )
    bb_df = pd.DataFrame(
        {"ma": ma_all, "upper": up_all, "lower": lo_all}, index=df_15m.index
    )

    saved_paths: list[Path] = []
    for fname, d_start, d_end, title in periods:
        mask = (price.index >= pd.Timestamp(d_start, tz=price.index.tz)) & (
            price.index <= pd.Timestamp(d_end, tz=price.index.tz) + pd.Timedelta(days=1)
        )
        p_sub = price.loc[mask]
        if p_sub.empty:
            continue
        pos_sub = pos.loc[mask]
        eq_sub = eq.loc[mask]
        bh_sub = bh.loc[mask]
        bb_sub = bb_df.reindex(p_sub.index)

        # Normalize equity curves to 0% at the start of this zoomed window
        eq_rel = (eq_sub / eq_sub.iloc[0] - 1.0) * 100.0
        bh_rel = (bh_sub / bh_sub.iloc[0] - 1.0) * 100.0

        ts = p_sub.index.tz_localize(None)
        t_min, t_max = p_sub.index.min(), p_sub.index.max()
        sub_trades = [
            tr for tr in trades if (t_min <= tr["entry_time"] <= t_max) or (t_min <= tr["exit_time"] <= t_max)
        ]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 7.8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.1]}
        )

        # Price + Bollinger Bands
        ax1.plot(ts, p_sub.to_numpy(), color="#111111", lw=1.3, label="AAPL 15-min Close")
        ax1.plot(ts, bb_sub["ma"].to_numpy(), color="#1f77b4", lw=1.0, ls="--", alpha=0.8, label=f"BB Mid ({bb_window_ref})")
        ax1.plot(ts, bb_sub["upper"].to_numpy(), color="#9467bd", lw=0.9, alpha=0.75, label=f"BB Upper/Lower (±{k_ref}σ)")
        ax1.plot(ts, bb_sub["lower"].to_numpy(), color="#9467bd", lw=0.9, alpha=0.75)
        ax1.fill_between(ts, bb_sub["lower"].to_numpy(), bb_sub["upper"].to_numpy(), color="#9467bd", alpha=0.06)

        y_min = float(min(p_sub.min(), bb_sub["lower"].min())) * 0.992
        y_max = float(max(p_sub.max(), bb_sub["upper"].max())) * 1.008
        ax1.fill_between(ts, y_min, y_max, where=(pos_sub.to_numpy() == 1), color="#2ca02c", alpha=0.11, label="In Long")
        ax1.fill_between(ts, y_min, y_max, where=(pos_sub.to_numpy() == -1), color="#d62728", alpha=0.11, label="In Short")
        ax1.set_ylim(y_min, y_max)

        # Trade markers
        labeled = set()
        for tr in sub_trades:
            if t_min <= tr["entry_time"] <= t_max:
                et = tr["entry_time"].tz_localize(None)
                if tr["side"] == 1:
                    lbl = "Buy (Trough)" if "L" not in labeled else None
                    labeled.add("L")
                    ax1.scatter(et, tr["entry_price"], marker="^", s=75, color="#00c853", edgecolor="black", lw=0.7, zorder=6, label=lbl)
                else:
                    lbl = "Short (Peak)" if "S" not in labeled else None
                    labeled.add("S")
                    ax1.scatter(et, tr["entry_price"], marker="v", s=75, color="#d50000", edgecolor="black", lw=0.7, zorder=6, label=lbl)
            if t_min <= tr["exit_time"] <= t_max:
                xt = tr["exit_time"].tz_localize(None)
                if tr["net_ret"] > 0:
                    lbl = "Win Exit" if "W" not in labeled else None
                    labeled.add("W")
                    ax1.scatter(xt, tr["exit_price"], marker="o", s=45, color="#1e88e5", edgecolor="white", lw=0.6, zorder=7, label=lbl)
                else:
                    lbl = "Loss Exit" if "X" not in labeled else None
                    labeled.add("X")
                    ax1.scatter(xt, tr["exit_price"], marker="x", s=55, color="#ff6f00", lw=1.8, zorder=7, label=lbl)

        ax1.set_ylabel("AAPL Price ($)")
        ax1.set_title(
            f"Bollinger Band Peak/Trough — {title} | Window Return: {eq_rel.iloc[-1]:+.2f}% vs B&H {bh_rel.iloc[-1]:+.2f}%"
        )
        ax1.legend(loc="upper left", ncol=4, fontsize=8.5)
        ax1.grid(True, alpha=0.25)

        ax2.plot(ts, eq_rel.to_numpy(), color="#1f77b4", lw=1.8, label="BB Peak/Trough (WFO)")
        ax2.plot(ts, bh_rel.to_numpy(), color="#444444", lw=1.5, ls="--", label="AAPL Buy & Hold")
        ax2.axhline(0.0, color="black", lw=0.7, ls=":")
        ax2.set_ylabel("Window Return (%)")
        ax2.set_xlabel("Date")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, alpha=0.25)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        out_p = OUTPUTS_DIR / fname
        fig.tight_layout()
        fig.savefig(out_p, dpi=150)
        plt.close(fig)
        saved_paths.append(out_p)
        print(f"Saved zoomed chart: {out_p}")

    return saved_paths


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df_1m = load_data(DEFAULT_OUTPUT_PATH, rth_only=True)
    df_15m = resample_rth_bars(df_1m, "15min")
    tr_15m, va_15m, te_15m = split_chronological(df_15m)

    grid_reentry = [
        (w, kr, kp, "band_reentry", sh)
        for w in [15, 25, 40]
        for kr in [1.0, 1.5, 2.0]
        for kp in [1.0, 1.5, 2.0]
        for sh in [False, True]
    ]
    grid_width = [
        (w, kr, kp, "bb_width", sh)
        for w in [15, 25, 40]
        for kr in [1.0, 1.5, 2.5]
        for kp in [1.5, 2.5, 3.5]
        for sh in [False, True]
    ]

    print("=" * 94)
    print("BOLLINGER BAND PEAK & TROUGH STRATEGY (15-min AAPL, 1.5 bps/side friction)")
    print("=" * 94)

    wfo_results = {}
    for label, grid in [
        ("BB Band Re-entry (Long/Cash)", [p for p in grid_reentry if not p[4]]),
        ("BB Band Re-entry (Long/Short)", grid_reentry),
        ("BB Width Reversal (Long/Cash)", [p for p in grid_width if not p[4]]),
        ("BB Width Reversal (Long/Short)", grid_width),
    ]:
        res = walk_forward_eval(
            df_15m,
            grid,
            gen_bb_peak_trough_signals,
            train_days=120,
            step_days=20,
            min_train_trades=4,
            warmup_days=10,
        )
        wfo_results[label] = res
        print(
            f"  {label:30s} | OOS Ret: {res['total_return_pct']:+6.2f}% (B&H {res['bh_return_pct']:+6.2f}%) | "
            f"Sharpe: {res['sharpe']:5.2f} (B&H {res['bh_sharpe']:4.2f}) | "
            f"MaxDD: {res['max_dd_pct']:6.2f}% (B&H {res['bh_max_dd_pct']:6.2f}%) | "
            f"Folds+: {res['pct_pos_folds']:4.0f}% ({res['n_folds']}) | Trades: {res['n_trades']:3d}"
        )

    # Pick the best Sharpe WFO variant to plot zoomed periods
    best_label = max(wfo_results, key=lambda k: wfo_results[k]["sharpe"])
    print(f"\nPlotting 3 zoomed shorter-period charts for best WFO variant: {best_label}")
    plot_zoomed_periods(wfo_results[best_label], df_15m, bb_window_ref=20, k_ref=2.0)


if __name__ == "__main__":
    main()
