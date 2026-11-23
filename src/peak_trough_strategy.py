"""Strictly causal Peak & Trough Reversal Strategy (buy right after a trough, sell right after a peak).

Causal confirmation rule (zero lookahead):
  - Trough confirmed -> BUY: Track running low since last peak. Once (smoothed) price rebounds
    by >= `rebound_pct` above the running low, a trough is confirmed -> enter Long at next bar OPEN.
  - Peak confirmed -> SELL: Track running high since entry. Once (smoothed) price pulls back
    by >= `pullback_pct` below the running high, a peak is confirmed -> exit Long (or flip Short)
    at next bar OPEN.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.explore_edge import resample_rth_bars, run_signal_backtest
from src.features import _causal_ema
from src.fetch_data import data_path_for_symbol, load_data
from src.optimize import DEFAULT_COST_BPS, split_chronological
from src.walk_forward import walk_forward_eval

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def gen_peak_trough_signals(
    df: pd.DataFrame,
    rebound_pct: float = 0.01,
    pullback_pct: float = 0.01,
    smooth_window: int = 1,
    allow_short: bool = False,
) -> np.ndarray:
    """Generate strictly causal target positions (-1, 0, +1) at each bar close.

    Parameters:
      - rebound_pct: Fractional bounce above running trough required to confirm a trough & buy (e.g. 0.01 = 1.0%).
      - pullback_pct: Fractional drop below running peak required to confirm a peak & sell (e.g. 0.01 = 1.0%).
      - smooth_window: Causal EMA span applied to close before peak/trough detection (1 = raw close).
      - allow_short: If True, sell after peak flips to Short (-1); if False, sells to Cash (0).
    """
    closes = df["close"].to_numpy(dtype=np.float64)
    prices = _causal_ema(closes, smooth_window) if smooth_window > 1 else closes
    n = len(prices)

    target = np.zeros(n, dtype=np.int8)
    state = 0  # 0 = looking for trough (flat/short), 1 = in long (looking for peak)
    running_low = float(prices[0])
    running_high = float(prices[0])

    for t in range(n):
        p = float(prices[t])
        if state == 0:
            if p < running_low:
                running_low = p
            elif p >= running_low * (1.0 + rebound_pct):
                # Trough confirmed -> BUY
                state = 1
                running_high = p
        else:
            if p > running_high:
                running_high = p
            elif p <= running_high * (1.0 - pullback_pct):
                # Peak confirmed -> SELL
                state = 0
                running_low = p

        if state == 1:
            target[t] = 1
        else:
            target[t] = -1 if allow_short else 0

    return target


def plot_peak_trough_trades(res: dict, title_prefix: str, out_path: Path, symbol: str = "AAPL") -> None:
    """Plot stock price with confirmed trough buys, peak sells, holding shading, and equity curve."""
    price = res["price"]
    pos = res["positions"]
    eq = res["equity"]
    bh = res["bh_equity"]
    trades = res["trades"]
    fold_dates = res["fold_dates"]

    ts = price.index.tz_localize(None)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.4, 1.2]}
    )

    ax1.plot(ts, price.to_numpy(), color="#222222", lw=1.1, label=f"{symbol} 15-min Close")

    y_min, y_max = float(price.min()) * 0.98, float(price.max()) * 1.02
    ax1.fill_between(ts, y_min, y_max, where=(pos.to_numpy() == 1), color="#2ca02c", alpha=0.12, label="Long (Post-Trough)")
    ax1.fill_between(ts, y_min, y_max, where=(pos.to_numpy() == -1), color="#d62728", alpha=0.12, label="Short (Post-Peak)")
    ax1.set_ylim(y_min, y_max)

    l_in_t, l_in_p = [], []
    s_in_t, s_in_p = [], []
    win_ex_t, win_ex_p = [], []
    loss_ex_t, loss_ex_p = [], []

    for tr in trades:
        et = tr["entry_time"].tz_localize(None)
        xt = tr["exit_time"].tz_localize(None)
        if tr["side"] == 1:
            l_in_t.append(et)
            l_in_p.append(tr["entry_price"])
        else:
            s_in_t.append(et)
            s_in_p.append(tr["entry_price"])
        if tr["net_ret"] > 0:
            win_ex_t.append(xt)
            win_ex_p.append(tr["exit_price"])
        else:
            loss_ex_t.append(xt)
            loss_ex_p.append(tr["exit_price"])

    if l_in_t:
        ax1.scatter(l_in_t, l_in_p, marker="^", s=60, color="#00c853", edgecolor="black", lw=0.6, zorder=5, label=f"Buy After Trough ({len(l_in_t)})")
    if s_in_t:
        ax1.scatter(s_in_t, s_in_p, marker="v", s=60, color="#d50000", edgecolor="black", lw=0.6, zorder=5, label=f"Short After Peak ({len(s_in_t)})")
    if win_ex_t:
        ax1.scatter(win_ex_t, win_ex_p, marker="o", s=38, color="#1e88e5", edgecolor="white", lw=0.5, zorder=6, label=f"Winning Exit ({len(win_ex_t)})")
    if loss_ex_t:
        ax1.scatter(loss_ex_t, loss_ex_p, marker="x", s=48, color="#ff6f00", lw=1.6, zorder=6, label=f"Losing Exit ({len(loss_ex_t)})")

    for fd in fold_dates:
        fdt = fd.tz_localize(None)
        ax1.axvline(fdt, color="gray", ls=":", lw=0.7, alpha=0.5)
        ax2.axvline(fdt, color="gray", ls=":", lw=0.7, alpha=0.5)

    ax1.set_ylabel(f"{symbol} Price ($)")
    ax1.set_title(
        f"{title_prefix} — {len(trades)} Out-of-Sample Trades "
        f"(OOS Return: {res['total_return_pct']:+.2f}%, Sharpe: {res['sharpe']:.2f}, Max DD: {res['max_dd_pct']:.2f}%)"
    )
    ax1.legend(loc="upper left", ncol=3, fontsize=8.5)
    ax1.grid(True, alpha=0.25)

    ax2.plot(ts, (eq.to_numpy() - 1.0) * 100.0, color="#1f77b4", lw=1.8, label="Peak/Trough Strategy (WFO)")
    ax2.plot(ts, (bh.to_numpy() - 1.0) * 100.0, color="#444444", lw=1.5, ls="--", label=f"{symbol} Buy & Hold")
    ax2.axhline(0.0, color="black", lw=0.7, ls=":")
    ax2.set_ylabel("Cumulative Return (%)")
    ax2.set_xlabel("Date (dotted vertical lines = 4-week re-tuning boundaries)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved chart to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal Peak & Trough Reversal Strategy.")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol (default: AAPL)")
    args = parser.parse_args()
    sym = args.symbol.upper()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df_1m = load_data(data_path_for_symbol(sym), rth_only=True)
    df_15m = resample_rth_bars(df_1m, "15min")

    tr_15m, va_15m, te_15m = split_chronological(df_15m)

    # Parameter grid: rebound_pct (buy trigger), pullback_pct (sell trigger), smooth_window, allow_short
    param_grid = [
        (reb, pb, sw, sh)
        for reb in [0.005, 0.01, 0.015, 0.02, 0.03]
        for pb in [0.005, 0.01, 0.015, 0.02, 0.03]
        for sw in [1, 4, 10]
        for sh in [False, True]
    ]

    print("=" * 92)
    print(f"CAUSAL PEAK & TROUGH REVERSAL STRATEGY (15-min {sym}, 1.5 bps/side friction)")
    print("=" * 92)

    # 1. Fixed Train -> Val -> OOS Test evaluation
    rows = []
    for params in param_grid:
        m_tr = run_signal_backtest(tr_15m, gen_peak_trough_signals(tr_15m, *params))
        if m_tr["n_trades"] < 6:
            continue
        m_va = run_signal_backtest(va_15m, gen_peak_trough_signals(va_15m, *params))
        score = min(m_tr["sharpe"], m_va["sharpe"])
        rows.append((score, params, m_tr, m_va))

    rows.sort(key=lambda x: x[0], reverse=True)
    _, best_params, m_tr, m_va = rows[0]
    m_te = run_signal_backtest(te_15m, gen_peak_trough_signals(te_15m, *best_params))
    reb, pb, sw, sh = best_params
    print(f"\n[Static Split] Best consistent params: rebound={reb*100:.1f}%, pullback={pb*100:.1f}%, smooth_window={sw}, allow_short={sh}")
    for lbl, m in [("Train", m_tr), ("Val  ", m_va), ("OOS  ", m_te)]:
        print(
            f"  {lbl} | Ret: {m['total_return_pct']:+6.2f}% (B&H {m['bh_return_pct']:+6.2f}%) | "
            f"Sharpe: {m['sharpe']:5.2f} | DD: {m['max_dd_pct']:6.2f}% | "
            f"Trades: {int(m['n_trades']):3d} | WR: {m['win_rate_pct']:4.1f}% | PF: {m['profit_factor']:4.2f}"
        )

    # 2. Rolling 4-Week Walk-Forward Optimization (Long/Cash + Long/Short allowed, and Long-Only)
    print("\n[4-Week Walk-Forward Optimization] (120d rolling Train, 20d OOS step)")
    for label, grid in [
        ("Peak/Trough (Long/Cash only)", [p for p in param_grid if not p[3]]),
        ("Peak/Trough (Long & Short)  ", param_grid),
    ]:
        wfo = walk_forward_eval(
            df_15m,
            grid,
            gen_peak_trough_signals,
            train_days=120,
            step_days=20,
            min_train_trades=4,
        )
        print(
            f"  {label} | OOS Ret: {wfo['total_return_pct']:+6.2f}% (B&H {wfo['bh_return_pct']:+6.2f}%) | "
            f"Sharpe: {wfo['sharpe']:5.2f} (B&H {wfo['bh_sharpe']:4.2f}) | "
            f"MaxDD: {wfo['max_dd_pct']:6.2f}% (B&H {wfo['bh_max_dd_pct']:6.2f}%) | "
            f"Folds+: {wfo['pct_pos_folds']:4.0f}% ({wfo['n_folds']}) | Trades: {wfo['n_trades']:3d}"
        )
        prefix = "" if sym == "AAPL" else f"{sym.lower()}_"
        if "Long/Cash" in label:
            plot_peak_trough_trades(
                wfo,
                title_prefix=f"15-Min {sym} Causal Peak & Trough Strategy (4-Week WFO, Long/Cash)",
                out_path=OUTPUTS_DIR / f"{prefix}peak_trough_wfo_trades.png",
                symbol=sym,
            )
        else:
            plot_peak_trough_trades(
                wfo,
                title_prefix=f"15-Min {sym} Causal Peak & Trough Strategy (4-Week WFO, Long & Short)",
                out_path=OUTPUTS_DIR / f"{prefix}peak_trough_ls_wfo_trades.png",
                symbol=sym,
            )
            plot_zoomed_peak_trough(wfo, label=f"{sym} Peak/Trough (4w WFO, Long & Short)", symbol=sym, prefix=prefix)


def plot_zoomed_peak_trough(
    res: dict,
    label: str = "Peak/Trough (4w WFO)",
    symbol: str = "AAPL",
    prefix: str = "",
) -> list[Path]:
    """Generate the same 3 zoomed shorter-period (~6-week) charts for side-by-side comparison."""
    price = res["price"]
    pos = res["positions"]
    eq = res["equity"]
    bh = res["bh_equity"]
    trades = res["trades"]

    periods = [
        (f"{prefix}pt_zoom_1_selloff.png", "2026-01-05", "2026-02-20", "Period 1: Winter (Jan–Feb 2026)"),
        (f"{prefix}pt_zoom_2_rally.png", "2026-06-22", "2026-08-07", "Period 2: Summer (Jun–Aug 2026)"),
        (f"{prefix}pt_zoom_3_recent.png", "2026-10-05", "2026-11-20", "Period 3: Recent (Oct–Nov 2026)"),
    ]

    saved: list[Path] = []
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

        ax1.plot(ts, p_sub.to_numpy(), color="#111111", lw=1.3, label=f"{symbol} 15-min Close")

        y_min = float(p_sub.min()) * 0.992
        y_max = float(p_sub.max()) * 1.008
        ax1.fill_between(ts, y_min, y_max, where=(pos_sub.to_numpy() == 1), color="#2ca02c", alpha=0.12, label="In Long")
        ax1.fill_between(ts, y_min, y_max, where=(pos_sub.to_numpy() == -1), color="#d62728", alpha=0.12, label="In Short")
        ax1.set_ylim(y_min, y_max)

        labeled = set()
        for tr in sub_trades:
            if t_min <= tr["entry_time"] <= t_max:
                et = tr["entry_time"].tz_localize(None)
                if tr["side"] == 1:
                    lbl = "Buy (After Trough)" if "L" not in labeled else None
                    labeled.add("L")
                    ax1.scatter(et, tr["entry_price"], marker="^", s=75, color="#00c853", edgecolor="black", lw=0.7, zorder=6, label=lbl)
                else:
                    lbl = "Short (After Peak)" if "S" not in labeled else None
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

        ax1.set_ylabel(f"{symbol} Price ($)")
        ax1.set_title(
            f"{label} — {title} | Window Return: {eq_rel.iloc[-1]:+.2f}% vs B&H {bh_rel.iloc[-1]:+.2f}%"
        )
        ax1.legend(loc="upper left", ncol=4, fontsize=8.5)
        ax1.grid(True, alpha=0.25)

        ax2.plot(ts, eq_rel.to_numpy(), color="#1f77b4", lw=1.8, label=label)
        ax2.plot(ts, bh_rel.to_numpy(), color="#444444", lw=1.5, ls="--", label=f"{symbol} Buy & Hold")
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
        saved.append(out_p)
        print(f"Saved zoomed chart: {out_p}")

    return saved


if __name__ == "__main__":
    main()
