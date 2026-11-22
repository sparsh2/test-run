"""Phase 4 & 5: Chronological Train/Val/Test optimization, stability analysis, plots, and report."""

from __future__ import annotations

import argparse
import itertools
import json
import time as pytime
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import BacktestResult, StrategyConfig, run_backtest
from src.features import FeatureConfig, compute_kinematic_features
from src.fetch_data import data_path_for_symbol, load_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_PATH = PROJECT_ROOT / "report.md"

DEFAULT_COST_BPS = 1.5  # 0.015% per side (~$0.03-$0.04/share on AAPL)


def split_chronological(
    df: pd.DataFrame,
    train_frac: float = 10.0 / 18.0,
    val_frac: float = 4.0 / 18.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict chronological split by unique trading dates (~10m Train, ~4m Val, ~4m Test)."""
    dates = np.array(sorted(pd.unique(df.index.date)))
    n_days = len(dates)
    train_end_idx = int(round(n_days * train_frac))
    val_end_idx = int(round(n_days * (train_frac + val_frac)))

    train_dates = set(dates[:train_end_idx])
    val_dates = set(dates[train_end_idx:val_end_idx])
    test_dates = set(dates[val_end_idx:])

    idx_dates = df.index.date
    train_df = df.loc[[d in train_dates for d in idx_dates]].copy()
    val_df = df.loc[[d in val_dates for d in idx_dates]].copy()
    test_df = df.loc[[d in test_dates for d in idx_dates]].copy()
    return train_df, val_df, test_df


def run_train_grid_search(train_df: pd.DataFrame) -> pd.DataFrame:
    """Grid search over smoothers, kinematic lookbacks, deadbands, and risk rules on TRAIN only."""
    smoothers = ["ema", "dema", "savgol", "linreg", "kalman"]
    smooth_windows = [10, 20, 35, 60]
    vel_lookbacks = [1, 3]

    vel_thresh_grid = [1.0, 2.0, 3.5]
    acc_thresh_grid = [0.0, 0.25, 0.6]
    exit_modes = ["vel_cross", "acc_decel"]
    sl_tp_grid = [
        (0.0, 0.0),      # Pure kinematic exit
        (40.0, 80.0),    # 0.40% SL / 0.80% TP
    ]
    min_atr_grid = [0.0, 6.0]

    records: list[dict] = []
    t0 = pytime.time()

    feat_combos = list(itertools.product(smoothers, smooth_windows, vel_lookbacks))
    strat_combos = list(
        itertools.product(vel_thresh_grid, acc_thresh_grid, exit_modes, sl_tp_grid, min_atr_grid)
    )
    total_runs = len(feat_combos) * len(strat_combos)
    print(
        f"Running Train grid search: {len(feat_combos)} feature sets x "
        f"{len(strat_combos)} strategy configs = {total_runs} backtests..."
    )

    done = 0
    for smoother, sw, vl in feat_combos:
        f_cfg = FeatureConfig(
            smoother=smoother,
            smooth_window=sw,
            vel_lookback=vl,
            acc_lookback=1,
            vol_window=20,
            reset_daily=True,
        )
        feats = compute_kinematic_features(train_df, f_cfg)

        for v_th, a_th, ex_mode, (sl, tp), min_atr in strat_combos:
            s_cfg = StrategyConfig(
                vel_thresh=v_th,
                acc_thresh=a_th,
                exit_mode=ex_mode,
                exit_vel_thresh=0.0,
                exit_acc_thresh=0.1,
                min_atr_bps=min_atr,
                min_rel_vol=0.0,
                warmup_bars=max(15, sw // 2),
                min_hold_bars=3,
                stop_loss_bps=sl,
                take_profit_bps=tp,
                flatten_eod=True,
                cost_bps_per_side=DEFAULT_COST_BPS,
            )
            res = run_backtest(
                train_df, f_cfg, s_cfg, features=feats, compute_oracle=False
            )
            m = res.metrics
            records.append(
                {
                    "smoother": smoother,
                    "smooth_window": sw,
                    "vel_lookback": vl,
                    "vel_thresh": v_th,
                    "acc_thresh": a_th,
                    "exit_mode": ex_mode,
                    "stop_loss_bps": sl,
                    "take_profit_bps": tp,
                    "min_atr_bps": min_atr,
                    "train_return_pct": m["total_return_pct"],
                    "train_sharpe": m["sharpe"],
                    "train_max_dd_pct": m["max_dd_pct"],
                    "train_trades": m["n_trades"],
                    "train_win_rate": m["win_rate_pct"],
                    "train_pf": m["profit_factor"],
                    "train_avg_bps": m["avg_trade_bps"],
                }
            )
            done += 1

        if done % (len(strat_combos) * 10) == 0 or done == total_runs:
            elapsed = pytime.time() - t0
            print(f"  Progress: {done}/{total_runs} ({elapsed:.1f}s)")

    return pd.DataFrame(records)


def row_to_configs(row: pd.Series) -> tuple[FeatureConfig, StrategyConfig]:
    """Reconstruct FeatureConfig and StrategyConfig from a grid-search DataFrame row."""
    sw = int(row["smooth_window"])
    f_cfg = FeatureConfig(
        smoother=str(row["smoother"]),
        smooth_window=sw,
        vel_lookback=int(row["vel_lookback"]),
        acc_lookback=1,
        vol_window=20,
        reset_daily=True,
    )
    s_cfg = StrategyConfig(
        vel_thresh=float(row["vel_thresh"]),
        acc_thresh=float(row["acc_thresh"]),
        exit_mode=str(row["exit_mode"]),
        exit_vel_thresh=0.0,
        exit_acc_thresh=0.1,
        min_atr_bps=float(row["min_atr_bps"]),
        min_rel_vol=0.0,
        warmup_bars=max(15, sw // 2),
        min_hold_bars=3,
        stop_loss_bps=float(row["stop_loss_bps"]),
        take_profit_bps=float(row["take_profit_bps"]),
        flatten_eod=True,
        cost_bps_per_side=DEFAULT_COST_BPS,
    )
    return f_cfg, s_cfg


def evaluate_stability_grid(
    val_df: pd.DataFrame,
    base_f: FeatureConfig,
    base_s: StrategyConfig,
    multipliers: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Perturb smooth_window and vel_thresh by +-10% and +-20% on Validation set.

    Returns (sharpe_grid_df, return_grid_df, summary_dict).
    """
    sw_vals = [max(4, int(round(base_f.smooth_window * m))) for m in multipliers]
    vt_vals = [round(base_s.vel_thresh * m, 3) for m in multipliers]

    sharpe_mat = np.zeros((len(sw_vals), len(vt_vals)), dtype=np.float64)
    ret_mat = np.zeros((len(sw_vals), len(vt_vals)), dtype=np.float64)

    for i, sw in enumerate(sw_vals):
        f_cfg = replace(base_f, smooth_window=sw)
        feats = compute_kinematic_features(val_df, f_cfg)
        for j, vt in enumerate(vt_vals):
            s_cfg = replace(base_s, vel_thresh=vt, warmup_bars=max(15, sw // 2))
            res = run_backtest(val_df, f_cfg, s_cfg, features=feats, compute_oracle=False)
            sharpe_mat[i, j] = res.metrics["sharpe"]
            ret_mat[i, j] = res.metrics["total_return_pct"]

    row_labels = [f"{int(m*100)}% (w={sw})" for m, sw in zip(multipliers, sw_vals)]
    col_labels = [f"{int(m*100)}% (v={vt:.2f})" for m, vt in zip(multipliers, vt_vals)]

    sharpe_df = pd.DataFrame(sharpe_mat, index=row_labels, columns=col_labels)
    ret_df = pd.DataFrame(ret_mat, index=row_labels, columns=col_labels)

    center_idx = len(multipliers) // 2
    summary = {
        "center_sharpe": float(sharpe_mat[center_idx, center_idx]),
        "mean_neighbor_sharpe": float(sharpe_mat.mean()),
        "std_neighbor_sharpe": float(sharpe_mat.std()),
        "min_neighbor_sharpe": float(sharpe_mat.min()),
        "mean_neighbor_ret_pct": float(ret_mat.mean()),
        "pct_positive_neighbors": float((ret_mat > 0.0).mean() * 100.0),
    }
    return sharpe_df, ret_df, summary


def plot_equity_curves(
    full_res: BacktestResult,
    train_end_ts: pd.Timestamp,
    val_end_ts: pd.Timestamp,
    out_path: Path,
) -> None:
    """Plot strategy equity curve vs AAPL Buy & Hold with Train/Val/Test split markers."""
    eq = full_res.equity
    bh = full_res.buy_hold_equity
    # Downsample to daily close for clean 18-month visualization
    eq_daily = eq.groupby(eq.index.date).last()
    bh_daily = bh.groupby(bh.index.date).last()
    dates = pd.to_datetime(eq_daily.index)
    dd_daily = (eq_daily / eq_daily.cummax() - 1.0) * 100.0
    bh_dd_daily = (bh_daily / bh_daily.cummax() - 1.0) * 100.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}
    )

    ax1.plot(dates, (eq_daily - 1.0) * 100.0, label="Kinematic Strategy (net)", color="#1f77b4", lw=2.0)
    ax1.plot(dates, (bh_daily - 1.0) * 100.0, label="AAPL Buy & Hold", color="#555555", lw=1.6, alpha=0.85)
    ax1.axhline(0.0, color="black", lw=0.7, ls=":")

    t_end = pd.Timestamp(train_end_ts.date())
    v_end = pd.Timestamp(val_end_ts.date())
    for ax in (ax1, ax2):
        ax.axvline(t_end, color="#d62728", ls="--", lw=1.3, alpha=0.85)
        ax.axvline(v_end, color="#2ca02c", ls="--", lw=1.3, alpha=0.85)

    ylim = ax1.get_ylim()
    y_text = ylim[1] - 0.06 * (ylim[1] - ylim[0])
    ax1.text(dates[int(len(dates) * 0.25)], y_text, "TRAIN (10m)", ha="center", fontsize=10, fontweight="bold", color="#333333")
    ax1.text(t_end + (v_end - t_end) / 2, y_text, "VALIDATION (4m)", ha="center", fontsize=10, fontweight="bold", color="#d62728")
    ax1.text(v_end + (dates[-1] - v_end) / 2, y_text, "OOS TEST (4m)", ha="center", fontsize=10, fontweight="bold", color="#2ca02c")

    ax1.set_ylabel("Cumulative Return (%)")
    ax1.set_title("AAPL 1-Minute Kinematic Strategy vs. Buy & Hold (Train / Validation / Out-of-Sample Test)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(dates, dd_daily, 0.0, color="#1f77b4", alpha=0.35, label="Strategy Drawdown")
    ax2.plot(dates, bh_dd_daily, color="#555555", lw=1.0, alpha=0.7, label="Buy & Hold Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sample_trades_zoom(
    test_df: pd.DataFrame,
    test_res: BacktestResult,
    s_cfg: StrategyConfig,
    out_path: Path,
) -> None:
    """Plot a zoomed 1-day candlestick chart from OOS Test showing smoothed price, v, a, and entries/exits."""
    # Pick a test day with active trades (closest to median-high activity)
    if test_res.trades:
        trade_dates = pd.Series([tr.entry_time.date() for tr in test_res.trades])
        counts = trade_dates.value_counts()
        # Pick a day with 3-8 trades if available, else the busiest day
        good_days = counts[(counts >= 3) & (counts <= 8)]
        chosen_date = good_days.index[0] if len(good_days) > 0 else counts.index[0]
    else:
        chosen_date = sorted(pd.unique(test_df.index.date))[len(pd.unique(test_df.index.date)) // 2]

    day_mask = test_df.index.date == chosen_date
    day_df = test_df.loc[day_mask]
    day_feat = test_res.features.loc[day_mask]
    day_trades = [tr for tr in test_res.trades if tr.entry_time.date() == chosen_date]

    x = np.arange(len(day_df))
    opens = day_df["open"].to_numpy()
    highs = day_df["high"].to_numpy()
    lows = day_df["low"].to_numpy()
    closes = day_df["close"].to_numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 9.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3, 1.3]}
    )

    # Candlesticks
    up = closes >= opens
    down = ~up
    ax1.vlines(x[up], lows[up], highs[up], color="#26a69a", lw=0.9)
    ax1.vlines(x[down], lows[down], highs[down], color="#ef5350", lw=0.9)
    ax1.bar(x[up], closes[up] - opens[up], bottom=opens[up], width=0.65, color="#26a69a", edgecolor="#26a69a", lw=0.4)
    ax1.bar(x[down], opens[down] - closes[down], bottom=closes[down], width=0.65, color="#ef5350", edgecolor="#ef5350", lw=0.4)

    ax1.plot(x, day_feat["smooth_price"].to_numpy(), color="#1a237e", lw=1.6, label="Causal Smoothed Price")

    # Map timestamps to integer x indices for trade markers
    ts_to_x = {ts: i for i, ts in enumerate(day_df.index)}
    labeled = set()
    for tr in day_trades:
        if tr.entry_time in ts_to_x:
            ex = ts_to_x[tr.entry_time]
            if tr.side == 1:
                lbl = "Long Entry" if "L_in" not in labeled else None
                labeled.add("L_in")
                ax1.scatter(ex, tr.entry_price, marker="^", s=85, color="#00c853", edgecolor="black", zorder=5, label=lbl)
            else:
                lbl = "Short Entry" if "S_in" not in labeled else None
                labeled.add("S_in")
                ax1.scatter(ex, tr.entry_price, marker="v", s=85, color="#d50000", edgecolor="black", zorder=5, label=lbl)
        if tr.exit_time in ts_to_x:
            xx = ts_to_x[tr.exit_time]
            lbl = "Exit" if "exit" not in labeled else None
            labeled.add("exit")
            ax1.scatter(xx, tr.exit_price, marker="x", s=70, color="#ff6f00", lw=2.0, zorder=5, label=lbl)

    ax1.set_ylabel("Price ($)")
    ax1.set_title(f"OOS Sample Day Zoom ({chosen_date}) — AAPL 1-Min Candles, Smoothed Price & Kinematic Signals")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.25)

    # Velocity subplot
    ax2.plot(x, day_feat["velocity"].to_numpy(), color="#1f77b4", lw=1.2, label="Velocity (bps/bar)")
    ax2.axhline(s_cfg.vel_thresh, color="#00c853", ls="--", lw=1.0, label=f"+/- vel_thresh ({s_cfg.vel_thresh})")
    ax2.axhline(-s_cfg.vel_thresh, color="#d50000", ls="--", lw=1.0)
    ax2.axhline(0.0, color="black", lw=0.6, ls=":")
    ax2.set_ylabel("Velocity (bps)")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, alpha=0.25)

    # Acceleration subplot
    ax3.plot(x, day_feat["acceleration"].to_numpy(), color="#9467bd", lw=1.1, label="Acceleration (bps/bar²)")
    ax3.axhline(s_cfg.acc_thresh, color="#00c853", ls="--", lw=1.0, label=f"+/- acc_thresh ({s_cfg.acc_thresh})")
    ax3.axhline(-s_cfg.acc_thresh, color="#d50000", ls="--", lw=1.0)
    ax3.axhline(0.0, color="black", lw=0.6, ls=":")
    ax3.set_ylabel("Accel (bps/bar²)")
    ax3.set_xlabel("Time (ET)")
    ax3.legend(loc="upper right", fontsize=8.5)
    ax3.grid(True, alpha=0.25)

    # Format x-axis ticks as HH:MM ET every 30 bars
    tick_idx = np.arange(0, len(day_df), 30)
    ax3.set_xticks(tick_idx)
    ax3.set_xticklabels([day_df.index[i].strftime("%H:%M") for i in tick_idx], rotation=30)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stability_heatmap(
    sharpe_df: pd.DataFrame,
    ret_df: pd.DataFrame,
    title_suffix: str,
    out_path: Path,
) -> None:
    """Plot 2D parameter sensitivity heatmap (Validation Sharpe & Return)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    for ax, mat_df, title, fmt in [
        (ax1, sharpe_df, "Validation Sharpe Ratio", "{:.2f}"),
        (ax2, ret_df, "Validation Total Return (%)", "{:.2f}%"),
    ]:
        data = mat_df.to_numpy(dtype=np.float64)
        vmax = max(abs(data.min()), abs(data.max()), 0.5)
        im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(mat_df.columns)))
        ax.set_yticks(np.arange(len(mat_df.index)))
        ax.set_xticklabels(mat_df.columns, rotation=25, ha="right", fontsize=8.5)
        ax.set_yticklabels(mat_df.index, fontsize=8.5)
        ax.set_xlabel("vel_thresh Perturbation")
        ax.set_ylabel("smooth_window Perturbation")
        ax.set_title(f"{title}\n({title_suffix})")

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                weight = "bold" if (i == 2 and j == 2) else "normal"
                ax.text(j, i, fmt.format(val), ha="center", va="center", color="black", fontsize=9, fontweight=weight)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_friction_and_regime(
    friction_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot execution friction sensitivity and PnL breakdown by volatility regime."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.plot(friction_df["cost_bps_per_side"], friction_df["train_ret_pct"], "o-", label="Train (10m)")
    ax1.plot(friction_df["cost_bps_per_side"], friction_df["val_ret_pct"], "s-", label="Validation (4m)")
    ax1.plot(friction_df["cost_bps_per_side"], friction_df["test_ret_pct"], "^-", label="OOS Test (4m)", lw=2.2)
    ax1.axhline(0.0, color="black", ls=":", lw=0.8)
    ax1.set_xlabel("Execution Cost (bps per side)")
    ax1.set_ylabel("Total Return (%)")
    ax1.set_title("Execution Friction Sensitivity (Selected Parameters)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in regime_df["avg_net_bps"]]
    ax2.bar(regime_df["regime"], regime_df["avg_net_bps"], color=colors, alpha=0.85, edgecolor="black")
    ax2.axhline(0.0, color="black", lw=0.8)
    ax2.set_ylabel("Avg Net Return per Trade (bps)")
    ax2.set_title("Trade PnL by Volatility Regime (Full Period, ATR Tertiles)")
    ax2.grid(True, axis="y", alpha=0.3)
    for i, row in regime_df.iterrows():
        ax2.text(
            i,
            row["avg_net_bps"] + (0.3 if row["avg_net_bps"] >= 0 else -0.6),
            f"n={int(row['n_trades'])}\nWR={row['win_rate_pct']:.1f}%",
            ha="center",
            va="bottom" if row["avg_net_bps"] >= 0 else "top",
            fontsize=8.5,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_regimes(full_res: BacktestResult) -> pd.DataFrame:
    """Break down trade performance by entry ATR tertile (Low / Mid / High volatility)."""
    if not full_res.trades:
        return pd.DataFrame(columns=["regime", "n_trades", "win_rate_pct", "avg_net_bps", "total_net_pct"])

    atr_series = full_res.features["atr_bps"]
    rows = []
    for tr in full_res.trades:
        entry_atr = float(atr_series.iloc[max(0, tr.entry_idx - 1)])
        rows.append({"atr_bps": entry_atr, "net_ret": tr.net_ret, "gross_ret": tr.gross_ret, "side": tr.side})
    tdf = pd.DataFrame(rows)

    q1, q2 = tdf["atr_bps"].quantile([1 / 3, 2 / 3])
    labels = [
        f"Low Vol\n(<{q1:.1f} bps)",
        f"Mid Vol\n({q1:.1f}-{q2:.1f} bps)",
        f"High Vol\n(>{q2:.1f} bps)",
    ]
    tdf["regime"] = pd.cut(tdf["atr_bps"], bins=[-np.inf, q1, q2, np.inf], labels=labels)

    summary = []
    for lbl in labels:
        sub = tdf[tdf["regime"] == lbl]
        if len(sub) == 0:
            summary.append({"regime": lbl, "n_trades": 0, "win_rate_pct": 0.0, "avg_net_bps": 0.0, "total_net_pct": 0.0})
        else:
            summary.append(
                {
                    "regime": lbl,
                    "n_trades": len(sub),
                    "win_rate_pct": float((sub["net_ret"] > 0).mean() * 100.0),
                    "avg_net_bps": float(sub["net_ret"].mean() * 10_000.0),
                    "total_net_pct": float(sub["net_ret"].sum() * 100.0),
                }
            )
    return pd.DataFrame(summary)


def write_report(
    best_f: FeatureConfig,
    best_s: StrategyConfig,
    train_res: BacktestResult,
    val_res: BacktestResult,
    test_res: BacktestResult,
    smoother_summary: pd.DataFrame,
    stability_summary: dict[str, float],
    friction_df: pd.DataFrame,
    session_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    split_info: dict[str, str],
) -> None:
    """Generate concise, evidence-based report.md."""

    def fmt_row(name: str, m: dict[str, float]) -> str:
        return (
            f"| {name} | {m['total_return_pct']:+.2f}% | {m['bh_return_pct']:+.2f}% | "
            f"{m['sharpe']:.2f} | {m['max_dd_pct']:.2f}% | {m['win_rate_pct']:.1f}% | "
            f"{m['profit_factor']:.2f} | {m['avg_trade_bps']:+.2f} bps | {int(m['n_trades']):,} | "
            f"{m['pct_bh_captured']:.1f}% | {m['pct_oracle_captured']:.2f}% |"
        )

    tm, vm, om = train_res.metrics, val_res.metrics, test_res.metrics

    # Verdict logic grounded in actual OOS results
    oos_profitable = om["total_return_pct"] > 0 and om["sharpe"] > 0.3
    beats_bh = om["total_return_pct"] > om["bh_return_pct"]
    if oos_profitable and beats_bh:
        verdict_headline = "Promising Positive Edge — Outperforms Buy & Hold Out-of-Sample"
    elif oos_profitable:
        verdict_headline = "Modest Positive OOS Return, Underperforms Buy & Hold"
    else:
        verdict_headline = "No Durable Positive Edge at 1-Minute Granularity After Realistic Friction"

    lines = [
        "# AAPL 1-Minute Kinematic (Velocity & Acceleration) Strategy — Backtest Report",
        "",
        "## 1. Executive Summary & Verdict",
        f"**Verdict:** **{verdict_headline}**",
        "",
        f"- **Dataset:** 1.5 years of split-adjusted AAPL 1-minute SIP bars (`data/aapl_1min_18m.parquet`), RTH (`09:30–15:59 ET`) baseline with zero lookahead (signals at bar $t$ close execute at bar $t+1$ open).",
        f"- **Chronological Splits:** Train `{split_info['train']}` | Validation `{split_info['val']}` | Out-of-Sample Test `{split_info['test']}`.",
        f"- **Baseline Friction:** `{best_s.cost_bps_per_side:.2f} bps/side` (`{2*best_s.cost_bps_per_side:.2f} bps` round-trip, ~${best_s.cost_bps_per_side/10000*220:.3f}/share per side).",
        f"- **Selected Most-Stable Parameters (from Validation stability check):**",
        f"  - Smoother: `{best_f.smoother}` (`smooth_window={best_f.smooth_window}`, `vel_lookback={best_f.vel_lookback}`, `reset_daily={best_f.reset_daily}`)",
        f"  - Entry/Exit: `vel_thresh={best_s.vel_thresh} bps/bar`, `acc_thresh={best_s.acc_thresh} bps/bar²`, `exit_mode={best_s.exit_mode}`, `min_atr_bps={best_s.min_atr_bps}`, `SL={best_s.stop_loss_bps} bps`, `TP={best_s.take_profit_bps} bps`, `flatten_eod={best_s.flatten_eod}`.",
        "",
        "---",
        "",
        "## 2. Performance Table (Train vs. Validation vs. Out-of-Sample Test)",
        "",
        "| Split | Strategy Return | AAPL Buy & Hold | Ann. Sharpe | Max Drawdown | Win Rate | Profit Factor | Avg Trade | Trades | % of B&H | % of Oracle (0.25% Swings) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        fmt_row("Train (10m)", tm),
        fmt_row("Validation (4m)", vm),
        fmt_row("**OOS Test (4m)**", om),
        "",
        f"*Note:* Theoretical Oracle Ceiling (non-causal ZigZag capturing all $\\ge 0.25\\%$ swings long & short after identical {best_s.cost_bps_per_side:.1f} bps/side costs): "
        f"Train `{tm['oracle_return_pct']:,.1f}%` ({int(tm['oracle_swings']):,} swings), "
        f"Validation `{vm['oracle_return_pct']:,.1f}%` ({int(vm['oracle_swings']):,} swings), "
        f"OOS Test `{om['oracle_return_pct']:,.1f}%` ({int(om['oracle_swings']):,} swings). "
        f"OOS Buy & Hold Sharpe: `{om['bh_sharpe']:.2f}`, Max DD: `{om['bh_max_dd_pct']:.2f}%`.",
        "",
        "---",
        "",
        "## 3. Causal Smoother Comparison (Train Set Best per Smoother)",
        "",
        "| Smoother | Best Window | Best Vel Lookback | Best Train Return | Best Train Sharpe | Trades | Median Grid Sharpe |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in smoother_summary.iterrows():
        lines.append(
            f"| `{r['smoother']}` | {int(r['smooth_window'])} | {int(r['vel_lookback'])} | "
            f"{r['train_return_pct']:+.2f}% | {r['train_sharpe']:.2f} | {int(r['train_trades']):,} | {r['median_sharpe']:.2f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Parameter Stability Analysis (±10% & ±20% Perturbations on Validation)",
        "",
        f"- **Center Validation Sharpe:** `{stability_summary['center_sharpe']:.2f}`",
        f"- **Neighborhood Mean Sharpe (5×5 grid):** `{stability_summary['mean_neighbor_sharpe']:.2f}` (std `{stability_summary['std_neighbor_sharpe']:.2f}`, min `{stability_summary['min_neighbor_sharpe']:.2f}`)",
        f"- **Neighborhood Mean Return:** `{stability_summary['mean_neighbor_ret_pct']:+.2f}%` (`{stability_summary['pct_positive_neighbors']:.0f}%` of perturbed neighbors profitable)",
        "",
        "![Parameter Stability Heatmap](outputs/param_stability_heatmap.png)",
        "",
        "---",
        "",
        "## 5. Visualizations",
        "",
        "### Equity Curve vs. AAPL Buy & Hold",
        "![Equity Curve](outputs/equity_curve.png)",
        "",
        "### Out-of-Sample 1-Day Zoom (Candlesticks, Smoothed Price, Velocity & Acceleration)",
        "![Sample Trades Zoom](outputs/sample_trades_zoom.png)",
        "",
        "---",
        "",
        "## 6. Execution Friction, Session & Regime Diagnostics",
        "",
        "### Execution Friction Sensitivity (Selected Parameters)",
        "",
        "| Cost (bps/side) | Train Return | Val Return | OOS Test Return | OOS Sharpe | OOS Avg Trade |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in friction_df.iterrows():
        lines.append(
            f"| {r['cost_bps_per_side']:.1f} | {r['train_ret_pct']:+.2f}% | {r['val_ret_pct']:+.2f}% | "
            f"{r['test_ret_pct']:+.2f}% | {r['test_sharpe']:.2f} | {r['test_avg_bps']:+.2f} bps |"
        )

    lines += [
        "",
        "### Session Comparison (OOS Test Period, Selected Parameters)",
        "",
        "| Session Variant | OOS Return | OOS Sharpe | OOS Max DD | OOS Trades | Avg Trade |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in session_df.iterrows():
        lines.append(
            f"| {r['variant']} | {r['ret_pct']:+.2f}% | {r['sharpe']:.2f} | {r['max_dd_pct']:.2f}% | "
            f"{int(r['n_trades']):,} | {r['avg_bps']:+.2f} bps |"
        )

    lines += [
        "",
        "### Trade Performance by Volatility Regime (Full Period, Entry ATR Tertiles)",
        "",
        "| Volatility Regime | Trades | Win Rate | Avg Net Trade (bps) | Sum Net Return |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in regime_df.iterrows():
        lbl = str(r["regime"]).replace("\n", " ")
        lines.append(
            f"| {lbl} | {int(r['n_trades']):,} | {r['win_rate_pct']:.1f}% | {r['avg_net_bps']:+.2f} bps | {r['total_net_pct']:+.2f}% |"
        )

    lines += [
        "",
        "![Friction and Regime Diagnostics](outputs/friction_and_regime.png)",
        "",
        "---",
        "",
        "## 7. Diagnosis & Concrete Next Iterations",
        "",
        "### Where the Strategy Makes Money vs. Where It Bleeds",
        "- **Lag vs. Noise Trade-off:** On 1-minute AAPL bars, raw 1st/2nd derivatives are dominated by bid-ask bounce and microstructure noise. Heavy smoothing (`smooth_window` 35–60) suppresses false flips but introduces causal phase lag, causing entries partway into a move and exits after a pullback has already begun.",
        "- **Friction Drag:** Even at `1.5 bps/side` (`3.0 bps` round-trip), high-frequency intraday turnover erodes gross edge rapidly. Notice in the friction table above how sensitively net return shifts between `0.0 bps` (frictionless) and `1.5–2.0 bps/side`.",
        "- **Choppy vs. Trending Regimes:** Kinematic momentum (velocity + acceleration) earns its positive expectancy during directional expansion (higher ATR / sustained trend), while bleeding via whipsaws in low-volatility, mean-reverting midday chop.",
        "",
        "### Recommended Next Iterations",
        "1. **Higher Timeframes (5-min / 15-min bars):** Resampling to 5-min or 15-min bars raises the average move size per trade relative to the fixed spread/slippage cost, improving the signal-to-friction ratio.",
        "2. **Regime Gating (Trend vs. Mean-Reversion Filter):** Gate kinematic entries with a higher-timeframe trend/regime filter (e.g., ADX, Hurst exponent, or distance from intraday VWAP) so the strategy only trades breakouts/trends and stays flat during range-bound midday chop.",
        "3. **Asymmetric / Volatility-Scaled Thresholds:** Replace fixed `vel_thresh` and `SL/TP` in bps with multiples of rolling ATR (`vel / atr_bps`) so thresholds automatically adapt across quiet summer sessions and earnings/macro volatility spikes.",
        "4. **Limit-Order / Microstructure Execution:** Testing passive/midpoint limit entries (lower slippage than crossing the spread at bar $t+1$ open) if live execution is targeted.",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report to {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="1-minute kinematic grid search, validation, and OOS backtest.")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol (default: AAPL)")
    args = parser.parse_args()
    sym = args.symbol.upper()
    data_path = data_path_for_symbol(sym)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading RTH 1-minute {sym} dataset from {data_path}...")
    df_rth = load_data(data_path, rth_only=True)
    df_full_session = load_data(data_path, rth_only=False)

    train_df, val_df, test_df = split_chronological(df_rth)
    split_info = {
        "train": f"{train_df.index.min().date()} to {train_df.index.max().date()} ({len(pd.unique(train_df.index.date))}d)",
        "val": f"{val_df.index.min().date()} to {val_df.index.max().date()} ({len(pd.unique(val_df.index.date))}d)",
        "test": f"{test_df.index.min().date()} to {test_df.index.max().date()} ({len(pd.unique(test_df.index.date))}d)",
    }
    print(f"Splits -> Train: {split_info['train']} | Val: {split_info['val']} | Test: {split_info['test']}")

    # Phase 4A: Train grid search
    grid_df = run_train_grid_search(train_df)
    grid_df.to_csv(OUTPUTS_DIR / "train_grid_results.csv", index=False)

    # Filter out degenerate low-trade configs (< 40 trades over 10 months)
    valid_train = grid_df[grid_df["train_trades"] >= 40].copy()
    if valid_train.empty:
        valid_train = grid_df.copy()

    # Smoother comparison summary
    smoother_rows = []
    for sm, grp in valid_train.groupby("smoother"):
        best_r = grp.sort_values("train_sharpe", ascending=False).iloc[0]
        smoother_rows.append(
            {
                "smoother": sm,
                "smooth_window": best_r["smooth_window"],
                "vel_lookback": best_r["vel_lookback"],
                "train_return_pct": best_r["train_return_pct"],
                "train_sharpe": best_r["train_sharpe"],
                "train_trades": best_r["train_trades"],
                "median_sharpe": grp["train_sharpe"].median(),
            }
        )
    smoother_summary = pd.DataFrame(smoother_rows).sort_values("train_sharpe", ascending=False)
    print("\nTop Train Result per Smoother:")
    print(smoother_summary.to_string(index=False))

    # Phase 4B: Evaluate top 25 diverse Train candidates (top 5 per smoother) on Validation
    top_candidates = (
        valid_train.sort_values("train_sharpe", ascending=False)
        .groupby("smoother", group_keys=False)
        .head(5)
        .reset_index(drop=True)
    )

    print(f"\nEvaluating {len(top_candidates)} top Train candidates on Validation set...")
    val_records = []
    for idx, row in top_candidates.iterrows():
        f_cfg, s_cfg = row_to_configs(row)
        vres = run_backtest(val_df, f_cfg, s_cfg, compute_oracle=False)
        vm = vres.metrics
        val_records.append(
            {
                **row.to_dict(),
                "val_return_pct": vm["total_return_pct"],
                "val_sharpe": vm["sharpe"],
                "val_max_dd_pct": vm["max_dd_pct"],
                "val_trades": vm["n_trades"],
                "val_pf": vm["profit_factor"],
            }
        )
    val_df_res = pd.DataFrame(val_records).sort_values("val_sharpe", ascending=False).reset_index(drop=True)
    val_df_res.to_csv(OUTPUTS_DIR / "validation_candidates.csv", index=False)

    print("\nTop 8 Candidates on Validation:")
    cols_show = ["smoother", "smooth_window", "vel_lookback", "vel_thresh", "acc_thresh", "exit_mode", "train_sharpe", "val_sharpe", "val_return_pct", "val_trades"]
    print(val_df_res[cols_show].head(8).to_string(index=False))

    # Stability check (+-10%, +-20% perturbations) on top 5 Validation candidates
    print("\nRunning +-10%/+-20% neighborhood stability check on top 5 Validation candidates...")
    stability_candidates = []
    best_stability_score = -1e9
    best_bundle = None

    for idx in range(min(5, len(val_df_res))):
        row = val_df_res.iloc[idx]
        f_cfg, s_cfg = row_to_configs(row)
        sh_grid, ret_grid, stab = evaluate_stability_grid(val_df, f_cfg, s_cfg)
        # Stability-adjusted score rewards high mean neighborhood Sharpe and penalizes sharp isolated spikes
        stab_score = stab["mean_neighbor_sharpe"] - 0.25 * stab["std_neighbor_sharpe"]
        print(
            f"  Candidate #{idx+1} ({f_cfg.smoother} w={f_cfg.smooth_window} v={s_cfg.vel_thresh} a={s_cfg.acc_thresh} {s_cfg.exit_mode}): "
            f"center_sh={stab['center_sharpe']:.2f}, mean_nb_sh={stab['mean_neighbor_sharpe']:.2f}, "
            f"std={stab['std_neighbor_sharpe']:.2f}, pos_nb={stab['pct_positive_neighbors']:.0f}%, score={stab_score:.2f}"
        )
        stability_candidates.append((stab_score, f_cfg, s_cfg, sh_grid, ret_grid, stab))
        if stab_score > best_stability_score:
            best_stability_score = stab_score
            best_bundle = (f_cfg, s_cfg, sh_grid, ret_grid, stab)

    assert best_bundle is not None
    best_f, best_s, best_sh_grid, best_ret_grid, best_stab = best_bundle
    print(f"\nSelected Most-Stable Parameters:\n  FeatureConfig: {best_f}\n  StrategyConfig: {best_s}")

    # Save chosen config
    with open(OUTPUTS_DIR / "selected_params.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "feature_config": asdict(best_f),
                "strategy_config": {k: str(v) if isinstance(v, pytime.struct_time) else (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in asdict(best_s).items()},
                "stability_summary": best_stab,
            },
            fp,
            indent=2,
        )

    # Phase 4C: Run ONCE on untouched Out-of-Sample Test Set (plus Train & Val with oracle benchmarks)
    print("\nRunning final evaluation on Train, Validation, and UNTOUCHED Out-of-Sample Test...")
    train_res = run_backtest(train_df, best_f, best_s, compute_oracle=True)
    val_res = run_backtest(val_df, best_f, best_s, compute_oracle=True)
    test_res = run_backtest(test_df, best_f, best_s, compute_oracle=True)
    full_res = run_backtest(df_rth, best_f, best_s, compute_oracle=False)

    for name, res in [("TRAIN", train_res), ("VALIDATION", val_res), ("OOS TEST", test_res)]:
        m = res.metrics
        print(
            f"  [{name:10s}] Ret: {m['total_return_pct']:+6.2f}% (B&H {m['bh_return_pct']:+6.2f}%) | "
            f"Sharpe: {m['sharpe']:5.2f} | MaxDD: {m['max_dd_pct']:6.2f}% | "
            f"Trades: {int(m['n_trades']):4d} | WR: {m['win_rate_pct']:4.1f}% | PF: {m['profit_factor']:4.2f}"
        )

    # Friction sensitivity on fixed selected params
    print("\nRunning friction sensitivity sweep (0.0 to 2.0 bps/side)...")
    friction_rows = []
    train_feats = compute_kinematic_features(train_df, best_f)
    val_feats = compute_kinematic_features(val_df, best_f)
    test_feats = compute_kinematic_features(test_df, best_f)
    for cost in [0.0, 0.5, 1.0, 1.5, 2.0]:
        sc = replace(best_s, cost_bps_per_side=cost)
        r_tr = run_backtest(train_df, best_f, sc, features=train_feats, compute_oracle=False)
        r_va = run_backtest(val_df, best_f, sc, features=val_feats, compute_oracle=False)
        r_te = run_backtest(test_df, best_f, sc, features=test_feats, compute_oracle=False)
        friction_rows.append(
            {
                "cost_bps_per_side": cost,
                "train_ret_pct": r_tr.metrics["total_return_pct"],
                "val_ret_pct": r_va.metrics["total_return_pct"],
                "test_ret_pct": r_te.metrics["total_return_pct"],
                "test_sharpe": r_te.metrics["sharpe"],
                "test_avg_bps": r_te.metrics["avg_trade_bps"],
            }
        )
    friction_df = pd.DataFrame(friction_rows)

    # Session comparison on OOS Test period
    print("Running session comparison (RTH intraday vs RTH overnight vs Extended hours)...")
    _, _, test_full_session = split_chronological(df_full_session)
    session_variants = [
        ("RTH Intraday (flatten 15:55)", test_df, best_f, best_s),
        ("RTH Hold Overnight", test_df, replace(best_f, reset_daily=False), replace(best_s, flatten_eod=False)),
        ("Extended Hours (04:00-20:00, flatten off)", test_full_session, replace(best_f, reset_daily=False), replace(best_s, flatten_eod=False)),
    ]
    session_rows = []
    for label, sdf, fc, sc in session_variants:
        sres = run_backtest(sdf, fc, sc, compute_oracle=False)
        sm = sres.metrics
        session_rows.append(
            {
                "variant": label,
                "ret_pct": sm["total_return_pct"],
                "sharpe": sm["sharpe"],
                "max_dd_pct": sm["max_dd_pct"],
                "n_trades": sm["n_trades"],
                "avg_bps": sm["avg_trade_bps"],
            }
        )
    session_df = pd.DataFrame(session_rows)

    # Regime analysis
    regime_df = analyze_regimes(full_res)

    # Phase 5: Generate plots & report
    print("\nGenerating plots...")
    plot_equity_curves(
        full_res,
        train_end_ts=train_df.index.max(),
        val_end_ts=val_df.index.max(),
        out_path=OUTPUTS_DIR / "equity_curve.png",
    )
    plot_sample_trades_zoom(
        test_df,
        test_res,
        best_s,
        out_path=OUTPUTS_DIR / "sample_trades_zoom.png",
    )
    plot_stability_heatmap(
        best_sh_grid,
        best_ret_grid,
        title_suffix=f"{best_f.smoother}, w={best_f.smooth_window}, v={best_s.vel_thresh}, a={best_s.acc_thresh}",
        out_path=OUTPUTS_DIR / "param_stability_heatmap.png",
    )
    plot_friction_and_regime(
        friction_df,
        regime_df,
        out_path=OUTPUTS_DIR / "friction_and_regime.png",
    )

    write_report(
        best_f=best_f,
        best_s=best_s,
        train_res=train_res,
        val_res=val_res,
        test_res=test_res,
        smoother_summary=smoother_summary,
        stability_summary=best_stab,
        friction_df=friction_df,
        session_df=session_df,
        regime_df=regime_df,
        split_info=split_info,
    )
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
