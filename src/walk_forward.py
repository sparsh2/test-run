"""Strictly causal Walk-Forward Optimization (WFO) across 2-week and 4-week re-tuning cycles.

For each rolling fold:
  1. Optimize parameters strictly on the preceding `train_days` trading days.
  2. Freeze the winning parameters and trade the next `step_days` trading days (100% OOS).
  3. Stitch all consecutive OOS folds into a single continuous out-of-sample equity curve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.explore_edge import (
    gen_h1_momentum_vwap,
    gen_h2_exhaustion_reversion,
    gen_h3_swing_trend_filter,
    resample_rth_bars,
    run_signal_backtest,
)
from src.fetch_data import DEFAULT_OUTPUT_PATH, load_data
from src.optimize import DEFAULT_COST_BPS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def run_signal_backtest_with_curve(
    df: pd.DataFrame,
    target_pos_at_close: np.ndarray,
    cost_bps_per_side: float = DEFAULT_COST_BPS,
) -> tuple[pd.Series, int]:
    """Return (equity_series_starting_at_1, n_trades) with zero lookahead (bar t signal -> bar t+1 open)."""
    n = len(df)
    opens = df["open"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    cost_rate = cost_bps_per_side / 10_000.0

    equity_arr = np.empty(n, dtype=np.float64)
    equity = 1.0
    pos = 0
    entry_price = 0.0
    pending = 0
    n_trades = 0
    equity_arr[0] = 1.0

    for t in range(n):
        if t > 0 and pending != pos:
            exec_price = opens[t]
            if pos != 0:
                gross = pos * (exec_price - entry_price) / entry_price
                equity *= 1.0 + gross - 2.0 * cost_rate
                n_trades += 1
                pos = 0
            if pending != 0:
                pos = pending
                entry_price = exec_price

        if pos != 0:
            unreal = pos * (closes[t] - entry_price) / entry_price - cost_rate
            equity_arr[t] = equity * (1.0 + unreal)
        else:
            equity_arr[t] = equity

        if t == n - 1:
            if pos != 0:
                gross = pos * (closes[t] - entry_price) / entry_price
                equity *= 1.0 + gross - 2.0 * cost_rate
                equity_arr[t] = equity
                n_trades += 1
            break

        pending = int(target_pos_at_close[t])

    return pd.Series(equity_arr, index=df.index), n_trades


def walk_forward_eval(
    df: pd.DataFrame,
    param_grid: list[tuple],
    signal_fn: Callable,
    train_days: int,
    step_days: int,
    min_train_trades: int,
    warmup_days: int = 10,
) -> dict:
    """Run rolling Walk-Forward Optimization and return stitched OOS equity + metrics."""
    unique_dates = np.array(sorted(pd.unique(df.index.date)))
    n_days = len(unique_dates)
    idx_dates = df.index.date

    # Pre-slice date -> mask indices for speed
    date_to_pos = {}
    for d in unique_dates:
        date_to_pos[d] = np.flatnonzero(idx_dates == d)

    stitched_segments: list[pd.Series] = []
    bh_segments: list[pd.Series] = []
    fold_returns: list[float] = []
    fold_dates: list[pd.Timestamp] = []
    total_trades = 0
    cum_eq = 1.0
    cum_bh = 1.0

    start_i = train_days
    while start_i < n_days:
        end_i = min(start_i + step_days, n_days)
        tr_dates = unique_dates[start_i - train_days : start_i]
        te_dates = unique_dates[start_i:end_i]
        if len(te_dates) < 3:
            break

        tr_idx = np.concatenate([date_to_pos[d] for d in tr_dates])
        te_idx = np.concatenate([date_to_pos[d] for d in te_dates])
        tr_df = df.iloc[tr_idx]
        te_df = df.iloc[te_idx]

        # 1. Optimize strictly on rolling Train window
        best_score = -1e9
        best_params = param_grid[0]
        for params in param_grid:
            sig_tr = signal_fn(tr_df, *params)
            m = run_signal_backtest(tr_df, sig_tr)
            if m["n_trades"] < min_train_trades:
                continue
            if m["sharpe"] > best_score:
                best_score = m["sharpe"]
                best_params = params

        # 2. Evaluate frozen best_params on OOS fold (using past warmup_days for causal smoother init)
        warm_start_i = max(0, start_i - warmup_days)
        warm_dates = unique_dates[warm_start_i:end_i]
        warm_idx = np.concatenate([date_to_pos[d] for d in warm_dates])
        warm_df = df.iloc[warm_idx]

        sig_warm = signal_fn(warm_df, *best_params)
        sig_oos = sig_warm[-len(te_df) :]

        eq_fold, n_tr = run_signal_backtest_with_curve(te_df, sig_oos)
        bh_fold = te_df["close"] / te_df["open"].iloc[0]

        stitched_segments.append(eq_fold * cum_eq)
        bh_segments.append(bh_fold * cum_bh)

        fold_ret = float(eq_fold.iloc[-1] - 1.0)
        fold_returns.append(fold_ret)
        fold_dates.append(te_df.index[0])
        total_trades += n_tr

        cum_eq *= float(eq_fold.iloc[-1])
        cum_bh *= float(bh_fold.iloc[-1])
        start_i = end_i

    full_eq = pd.concat(stitched_segments)
    full_bh = pd.concat(bh_segments)

    daily_eq = full_eq.groupby(full_eq.index.date).last()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe = float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252.0)) if daily_ret.std() > 0 else 0.0
    max_dd = float((full_eq / full_eq.cummax() - 1.0).min() * 100.0)

    bh_daily = full_bh.groupby(full_bh.index.date).last().pct_change().dropna()
    bh_sharpe = float((bh_daily.mean() / bh_daily.std()) * np.sqrt(252.0)) if bh_daily.std() > 0 else 0.0
    bh_dd = float((full_bh / full_bh.cummax() - 1.0).min() * 100.0)

    return {
        "equity": full_eq,
        "bh_equity": full_bh,
        "fold_dates": fold_dates,
        "total_return_pct": float((full_eq.iloc[-1] - 1.0) * 100.0),
        "bh_return_pct": float((full_bh.iloc[-1] - 1.0) * 100.0),
        "sharpe": sharpe,
        "bh_sharpe": bh_sharpe,
        "max_dd_pct": max_dd,
        "bh_max_dd_pct": bh_dd,
        "n_folds": len(fold_returns),
        "pct_pos_folds": float(np.mean([r > 0 for r in fold_returns]) * 100.0),
        "n_trades": total_trades,
    }


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df_1m = load_data(DEFAULT_OUTPUT_PATH, rth_only=True)
    df_5m = resample_rth_bars(df_1m, "5min")
    df_15m = resample_rth_bars(df_1m, "15min")

    # Compact, economically grounded parameter grids for each strategy
    h1_grid = [
        (sw, vz, az, lo, True)
        for sw in [6, 12, 20]
        for vz in [0.15, 0.30]
        for az in [0.0, 0.05]
        for lo in [False, True]
    ]
    h2_grid = [
        (sw, vz, af, mh, lo)
        for sw in [4, 10]
        for vz in [0.20, 0.35]
        for af in [0.02, 0.08]
        for mh in [6, 12]
        for lo in [False, True]
    ]
    h3_grid = [
        (sw, ve, vx, sh)
        for sw in [20, 40, 80, 120]
        for ve in [0.02, 0.08, 0.15]
        for vx in [0.0, -0.05, -0.10]
        for sh in [False, True]
    ]

    train_days = 120  # ~6 months rolling training window
    results = {}

    print("=" * 94)
    print(f"WALK-FORWARD OPTIMIZATION (Rolling {train_days}d Train, 1.5 bps/side friction, 100% OOS stitched)")
    print("=" * 94)

    for step_days, step_lbl in [(20, "4-Week"), (10, "2-Week")]:
        print(f"\n--- Re-tuning every {step_lbl} ({step_days} trading days) ---")
        for name, df_tf, grid, fn, min_tr in [
            ("H1 (15m Mom+VWAP)", df_15m, h1_grid, gen_h1_momentum_vwap, 15),
            ("H2 (5m Exhaustion)", df_5m, h2_grid, gen_h2_exhaustion_reversion, 15),
            ("H3 (15m Swing Filter)", df_15m, h3_grid, gen_h3_swing_trend_filter, 4),
        ]:
            res = walk_forward_eval(df_tf, grid, fn, train_days=train_days, step_days=step_days, min_train_trades=min_tr)
            results[(step_lbl, name)] = res
            print(
                f"  {name:22s} | OOS Ret: {res['total_return_pct']:+6.2f}% (B&H {res['bh_return_pct']:+6.2f}%) | "
                f"Sharpe: {res['sharpe']:5.2f} (B&H {res['bh_sharpe']:4.2f}) | "
                f"MaxDD: {res['max_dd_pct']:6.2f}% (B&H {res['bh_max_dd_pct']:6.2f}%) | "
                f"Folds+: {res['pct_pos_folds']:4.0f}% ({res['n_folds']}) | Trades: {res['n_trades']:4d}"
            )

    # Plot 4-Week WFO stitched equity curves vs Buy & Hold
    fig, ax = plt.subplots(figsize=(13, 6))
    ref = results[("4-Week", "H3 (15m Swing Filter)")]
    bh_d = ref["bh_equity"].groupby(ref["bh_equity"].index.date).last()
    ax.plot(pd.to_datetime(bh_d.index), (bh_d - 1.0) * 100.0, label="AAPL Buy & Hold", color="#333333", lw=2.0, ls="--")

    colors = {"H1 (15m Mom+VWAP)": "#d62728", "H2 (5m Exhaustion)": "#ff7f0e", "H3 (15m Swing Filter)": "#1f77b4"}
    for name in ["H1 (15m Mom+VWAP)", "H2 (5m Exhaustion)", "H3 (15m Swing Filter)"]:
        eq_d = results[("4-Week", name)]["equity"].groupby(results[("4-Week", name)]["equity"].index.date).last()
        ax.plot(pd.to_datetime(eq_d.index), (eq_d - 1.0) * 100.0, label=f"{name} (4w WFO)", color=colors[name], lw=1.9)

    for fd in ref["fold_dates"]:
        ax.axvline(pd.Timestamp(fd.date()), color="gray", ls=":", lw=0.7, alpha=0.45)

    ax.axhline(0.0, color="black", lw=0.7, ls="-")
    ax.set_title("100% Out-of-Sample Walk-Forward Equity Curves (Re-tuned Every 4 Weeks, 120d Rolling Train)")
    ax.set_ylabel("Cumulative OOS Return (%)")
    ax.set_xlabel("Date (dotted vertical lines = 4-week re-tuning boundaries)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    out_png = OUTPUTS_DIR / "walk_forward_equity.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved Walk-Forward plot to {out_png}")


if __name__ == "__main__":
    main()
