"""15-minute RTH live trading runner (IEX real-time feed, Long/Cash Peak & Trough, auto 4-week retrain)."""

from __future__ import annotations

import time as pytime
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from src.features import _causal_ema
from src.live.broker import (
    fetch_iex_15m_bars,
    notify_macos,
    submit_long_only_order,
)
from src.live.config import LiveConfig, load_config
from src.live.retrain import is_retrain_due, retrain_symbols
from src.live.state import (
    EquityPoint,
    SymbolState,
    activate_pending_if_flat,
    apply_buy,
    apply_sell,
    load_portfolio,
    save_portfolio,
)

ET = ZoneInfo("America/New_York")
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)


def is_rth_open_now(now_et: datetime | None = None) -> bool:
    """Return True during Mon-Fri 09:30-16:05 ET."""
    now = now_et or datetime.now(ET)
    if now.weekday() >= 5:
        return False
    return RTH_OPEN <= now.time() <= time(16, 5)


def evaluate_symbol_tick(st: SymbolState, cfg: LiveConfig) -> dict:
    """Fetch latest 15-min IEX bars for one symbol, update running peak/trough, and act on BUY/SELL."""
    df = fetch_iex_15m_bars(st.symbol, lookback_days=14)
    if df.empty:
        return {"symbol": st.symbol, "event": "no_data"}

    closes = df["close"].to_numpy(dtype=np.float64)
    sw = max(1, int(st.active_params.smooth_window))
    smooth_arr = _causal_ema(closes, sw) if sw > 1 else closes

    latest_ts = df.index[-1].isoformat()
    latest_close = float(closes[-1])
    latest_smooth = float(smooth_arr[-1])

    st.last_price = latest_close
    st.last_smooth_price = latest_smooth
    if st.benchmark_start_price <= 0:
        st.benchmark_start_price = latest_close

    # Initialize running_low / running_high on first-ever tick
    if not st.holding and st.running_low <= 0:
        st.running_low = latest_smooth
    if st.holding and st.running_high <= 0:
        st.running_high = max(latest_smooth, st.avg_entry_price or latest_smooth)

    # Avoid double-processing the exact same 15-min bar
    if st.last_bar_time == latest_ts:
        return {"symbol": st.symbol, "event": "already_processed", "price": latest_close, "bar": latest_ts}

    st.last_bar_time = latest_ts
    event = "hold"

    if not st.holding:
        # Flat (CASH): track running trough and watch for rebound >= rebound_pct
        if latest_smooth < st.running_low:
            st.running_low = latest_smooth
        elif latest_smooth >= st.running_low * (1.0 + st.active_params.rebound_pct):
            event = "BUY"
            order_note = ""
            if cfg.mode in ("paper", "live"):
                res = submit_long_only_order(
                    st.symbol, side="BUY", notional=st.cash, paper=(cfg.mode == "paper")
                )
                order_note = f"alpaca_order={res['id']}"
            rec = apply_buy(
                st,
                price=latest_close,
                source=cfg.mode,
                timestamp=latest_ts,
                note=order_note or f"trough={st.running_low:.2f}, reb={st.active_params.rebound_pct*100:.1f}%",
            )
            msg = f"BUY {st.symbol} @ ${latest_close:.2f} (qty={rec.qty:.3f}, mode={cfg.mode})"
            print(f"[{latest_ts}] *** {msg} ***")
            if cfg.notify_macos:
                notify_macos(f"BUY Signal: {st.symbol}", msg)
    else:
        # Holding (LONG): track running peak and watch for pullback >= pullback_pct
        if latest_smooth > st.running_high:
            st.running_high = latest_smooth
        elif latest_smooth <= st.running_high * (1.0 - st.active_params.pullback_pct):
            event = "SELL"
            order_note = ""
            if cfg.mode in ("paper", "live"):
                res = submit_long_only_order(st.symbol, side="SELL", paper=(cfg.mode == "paper"))
                order_note = f"alpaca_order={res['id']}"
            rec = apply_sell(
                st,
                price=latest_close,
                source=cfg.mode,
                timestamp=latest_ts,
                note=order_note or f"peak={st.running_high:.2f}, pb={st.active_params.pullback_pct*100:.1f}%",
            )
            msg = f"SELL {st.symbol} @ ${latest_close:.2f} (PnL={rec.pnl_pct:+.2f}%, mode={cfg.mode})"
            print(f"[{latest_ts}] *** {msg} ***")
            if cfg.notify_macos:
                notify_macos(f"SELL Signal: {st.symbol}", msg)

    # Activate any queued 4-week params if flat
    activate_pending_if_flat(st)

    # Record equity snapshot (keep last 2,500 points ~ 4 months of 15-min bars)
    st.equity_curve.append(
        EquityPoint(
            timestamp=latest_ts,
            equity=round(st.current_equity(latest_close), 2),
            price=round(latest_close, 4),
            holding=st.holding,
        )
    )
    if len(st.equity_curve) > 2500:
        st.equity_curve = st.equity_curve[-2500:]

    trig_side, trig_px = st.next_trigger_price()
    return {
        "symbol": st.symbol,
        "event": event,
        "price": latest_close,
        "smooth_price": latest_smooth,
        "holding": st.holding,
        "next_trigger_side": trig_side,
        "next_trigger_price": round(trig_px, 2),
        "bar": latest_ts,
    }


def run_once(symbols: list[str] | None = None, check_retrain: bool = True) -> list[dict]:
    """Run one evaluation tick across all configured symbols."""
    cfg = load_config()
    target_syms = [s.upper() for s in (symbols or cfg.symbols)]
    portfolio = load_portfolio(cfg.symbols, initial_cash=cfg.initial_cash_per_symbol)

    # Auto-retrain outside market hours (or if never trained yet) without interrupting open trades
    if check_retrain:
        due = [
            sym
            for sym in target_syms
            if (not portfolio[sym].last_retrain_at)
            or ((not is_rth_open_now()) and is_retrain_due(portfolio[sym], cfg.retrain_interval_days))
        ]
        if due:
            retrain_symbols(symbols=due, force=False, cfg=cfg)
            portfolio = load_portfolio(cfg.symbols, initial_cash=cfg.initial_cash_per_symbol)

    results = []
    for sym in target_syms:
        res = evaluate_symbol_tick(portfolio[sym], cfg)
        results.append(res)
    save_portfolio(portfolio)
    return results


def seconds_until_next_15m_close(now_et: datetime | None = None) -> float:
    """Compute seconds until the next 15-minute mark + 10s buffer (e.g. 09:45:10, 10:00:10 ET)."""
    now = now_et or datetime.now(ET)
    minute_block = ((now.minute // 15) + 1) * 15
    next_dt = now.replace(minute=0, second=10, microsecond=0) + timedelta(minutes=minute_block)
    return max(5.0, (next_dt - now).total_seconds())


def run_loop() -> None:
    """Continuous 15-minute RTH live loop (safe to leave running 24/7; sleeps cleanly off-hours)."""
    cfg = load_config()
    print(f"Starting 15-min IEX Long/Cash runner | symbols={cfg.symbols} | mode={cfg.mode}")
    # Initial check on startup (trains if needed + grabs latest bar)
    res = run_once(check_retrain=True)
    for r in res:
        print(f"  [startup] {r}")

    while True:
        now_et = datetime.now(ET)
        if is_rth_open_now(now_et):
            wait_s = seconds_until_next_15m_close(now_et)
            print(f"[{now_et.strftime('%H:%M:%S ET')}] Sleeping {wait_s:.0f}s until next 15-min bar close...")
            pytime.sleep(wait_s)
            if is_rth_open_now():
                tick_res = run_once(check_retrain=False)
                for r in tick_res:
                    print(f"  [tick] {r}")
        else:
            # Off-hours: check if 4-week retrain is due, then sleep 15 minutes
            run_once(check_retrain=True)
            print(f"[{now_et.strftime('%Y-%m-%d %H:%M ET')}] Market closed. Next check in 15 min...")
            pytime.sleep(900)


if __name__ == "__main__":
    run_loop()
