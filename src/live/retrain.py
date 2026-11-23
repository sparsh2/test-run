"""Standalone & scheduled 4-week rolling WFO parameter re-tuner using free Alpaca IEX data.

Never closes or disrupts an open position:
  - If symbol is flat (CASH), new params activate immediately.
  - If symbol is currently HOLDING, new params are stored in `pending_params` and activate
    automatically the moment the open trade exits to cash.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.explore_edge import run_signal_backtest
from src.live.broker import fetch_iex_training_15m_bars
from src.live.config import LiveConfig, load_config
from src.live.state import (
    StrategyParams,
    SymbolState,
    load_portfolio,
    queue_retrained_params,
    save_portfolio,
)
from src.peak_trough_strategy import gen_peak_trough_signals

LONG_ONLY_GRID = [
    (reb, pb, sw)
    for reb in [0.005, 0.01, 0.015, 0.02, 0.03]
    for pb in [0.005, 0.01, 0.015, 0.02, 0.03]
    for sw in [1, 4, 10]
]


def is_retrain_due(st: SymbolState, interval_days: int = 28) -> bool:
    """Return True if symbol has never been trained or >= interval_days have elapsed."""
    if not st.last_retrain_at:
        return True
    try:
        last_dt = datetime.fromisoformat(st.last_retrain_at)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400.0
        return elapsed >= interval_days
    except Exception:
        return True


def tune_symbol_iex(
    symbol: str,
    train_trading_days: int = 120,
    cost_bps_per_side: float = 1.5,
) -> StrategyParams:
    """Fetch last ~120 trading days of 15-min IEX bars and find best Long/Cash Peak & Trough params."""
    df_15m = fetch_iex_training_15m_bars(symbol, calendar_days=int(train_trading_days * 1.65) + 15)
    unique_dates = sorted(pd.unique(df_15m.index.date))
    keep_dates = set(unique_dates[-train_trading_days:])
    tr_df = df_15m.loc[[d in keep_dates for d in df_15m.index.date]].copy()

    if len(tr_df) < 100:
        raise RuntimeError(f"Insufficient IEX 15-min training bars for {symbol}: {len(tr_df)}")

    best_sharpe = -1e9
    best_tuple = (0.015, 0.010, 1)
    best_metrics = {"sharpe": 0.0, "total_return_pct": 0.0, "n_trades": 0.0}

    for reb, pb, sw in LONG_ONLY_GRID:
        sig = gen_peak_trough_signals(tr_df, rebound_pct=reb, pullback_pct=pb, smooth_window=sw, allow_short=False)
        m = run_signal_backtest(tr_df, sig, cost_bps_per_side=cost_bps_per_side)
        if m["n_trades"] < 4:
            continue
        if m["sharpe"] > best_sharpe:
            best_sharpe = m["sharpe"]
            best_tuple = (reb, pb, sw)
            best_metrics = m

    reb, pb, sw = best_tuple
    return StrategyParams(
        rebound_pct=reb,
        pullback_pct=pb,
        smooth_window=sw,
        allow_short=False,
        trained_at=datetime.now(timezone.utc).isoformat(),
        train_sharpe=round(float(best_metrics["sharpe"]), 3),
        train_return_pct=round(float(best_metrics["total_return_pct"]), 2),
        train_trades=int(best_metrics["n_trades"]),
    )


def retrain_symbols(
    symbols: list[str] | None = None,
    force: bool = False,
    cfg: LiveConfig | None = None,
) -> dict[str, str]:
    """Run 4-week IEX parameter re-tuning for due (or forced) symbols without interrupting open trades."""
    cfg = cfg or load_config()
    target_syms = [s.upper() for s in (symbols or cfg.symbols)]
    portfolio = load_portfolio(cfg.symbols, initial_cash=cfg.initial_cash_per_symbol)

    outcomes: dict[str, str] = {}
    for sym in target_syms:
        st = portfolio[sym]
        if (not force) and (not is_retrain_due(st, cfg.retrain_interval_days)):
            outcomes[sym] = "skipped_not_due"
            continue

        print(f"[retrain] Tuning {sym} on last {cfg.train_trading_days} trading days of 15-min IEX bars...")
        new_p = tune_symbol_iex(sym, train_trading_days=cfg.train_trading_days, cost_bps_per_side=cfg.cost_bps_per_side)
        status = queue_retrained_params(st, new_p)
        outcomes[sym] = status
        print(
            f"[retrain] {sym}: rebound={new_p.rebound_pct*100:.1f}%, pullback={new_p.pullback_pct*100:.1f}%, "
            f"smooth={new_p.smooth_window} (Train Sh={new_p.train_sharpe:.2f}, Ret={new_p.train_return_pct:+.2f}%) -> {status}"
        )

    save_portfolio(portfolio)
    return outcomes


if __name__ == "__main__":
    retrain_symbols(force=True)
