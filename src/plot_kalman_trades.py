"""Generate trade visualizations for the best 1-minute Kalman Filter kinematic strategy."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.fetch_data import DEFAULT_OUTPUT_PATH, load_data
from src.optimize import OUTPUTS_DIR, plot_sample_trades_zoom, row_to_configs, split_chronological


def main() -> None:
    grid_df = pd.read_csv(OUTPUTS_DIR / "train_grid_results.csv")
    kalman_rows = grid_df[(grid_df["smoother"] == "kalman") & (grid_df["train_trades"] >= 40)]
    best_k = kalman_rows.sort_values("train_sharpe", ascending=False).iloc[0]
    f_cfg, s_cfg = row_to_configs(best_k)
    print(f"Best Train Kalman Config:\n  {f_cfg}\n  {s_cfg}")

    df_rth = load_data(DEFAULT_OUTPUT_PATH, rth_only=True)
    _, _, test_df = split_chronological(df_rth)
    test_res = run_backtest(test_df, f_cfg, s_cfg, compute_oracle=False)
    m = test_res.metrics
    print(
        f"Kalman OOS Test | Ret: {m['total_return_pct']:+.2f}% (B&H {m['bh_return_pct']:+.2f}%) | "
        f"Sharpe: {m['sharpe']:.2f} | MaxDD: {m['max_dd_pct']:.2f}% | "
        f"Trades: {int(m['n_trades'])} | WR: {m['win_rate_pct']:.1f}% | PF: {m['profit_factor']:.2f}"
    )

    # 1. Full 4-month OOS overview with trades & equity curve
    price = test_df["close"]
    pos = test_res.positions
    eq = test_res.equity
    bh = test_res.buy_hold_equity
    trades = test_res.trades
    ts = price.index.tz_localize(None)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3]}
    )
    ax1.plot(ts, price.to_numpy(), color="#222222", lw=0.9, label="AAPL 1-min Close")

    l_in_t = [tr.entry_time.tz_localize(None) for tr in trades if tr.side == 1]
    l_in_p = [tr.entry_price for tr in trades if tr.side == 1]
    s_in_t = [tr.entry_time.tz_localize(None) for tr in trades if tr.side == -1]
    s_in_p = [tr.entry_price for tr in trades if tr.side == -1]
    win_t = [tr.exit_time.tz_localize(None) for tr in trades if tr.net_ret > 0]
    win_p = [tr.exit_price for tr in trades if tr.net_ret > 0]
    loss_t = [tr.exit_time.tz_localize(None) for tr in trades if tr.net_ret <= 0]
    loss_p = [tr.exit_price for tr in trades if tr.net_ret <= 0]

    ax1.scatter(l_in_t, l_in_p, marker="^", s=30, color="#00c853", alpha=0.85, zorder=5, label=f"Long Entry ({len(l_in_t)})")
    ax1.scatter(s_in_t, s_in_p, marker="v", s=30, color="#d50000", alpha=0.85, zorder=5, label=f"Short Entry ({len(s_in_t)})")
    ax1.scatter(win_t, win_p, marker="o", s=18, color="#1e88e5", alpha=0.75, zorder=6, label=f"Win Exit ({len(win_t)})")
    ax1.scatter(loss_t, loss_p, marker="x", s=24, color="#ff6f00", alpha=0.75, zorder=6, label=f"Loss Exit ({len(loss_t)})")

    ax1.set_ylabel("AAPL Price ($)")
    ax1.set_title(
        f"1-Min Kalman Filter Strategy (OOS Test: Jul–Nov 2026) — {len(trades)} Trades "
        f"(OOS Return: {m['total_return_pct']:+.2f}%, Sharpe: {m['sharpe']:.2f}, Max DD: {m['max_dd_pct']:.2f}%)"
    )
    ax1.legend(loc="upper left", ncol=3, fontsize=8.5)
    ax1.grid(True, alpha=0.25)

    ax2.plot(ts, (eq.to_numpy() - 1.0) * 100.0, color="#d62728", lw=1.8, label="Kalman 1-min Strategy")
    ax2.plot(ts, (bh.to_numpy() - 1.0) * 100.0, color="#444444", lw=1.5, ls="--", label="AAPL Buy & Hold")
    ax2.axhline(0.0, color="black", lw=0.7, ls=":")
    ax2.set_ylabel("Cumulative Return (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    out_overview = OUTPUTS_DIR / "kalman_oos_trades.png"
    fig.tight_layout()
    fig.savefig(out_overview, dpi=150)
    plt.close(fig)
    print(f"Saved {out_overview}")

    # 2. Zoomed 1-day candlestick chart with Kalman smoothed price, velocity, acceleration & trades
    out_zoom = OUTPUTS_DIR / "kalman_sample_day_zoom.png"
    plot_sample_trades_zoom(test_df, test_res, s_cfg, out_zoom)
    print(f"Saved {out_zoom}")


if __name__ == "__main__":
    main()
