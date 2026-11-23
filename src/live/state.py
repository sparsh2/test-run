"""Persistent per-symbol wallet state, safe 4-week parameter handoff, and manual trade ingestion."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.live.config import STATE_DIR, STATE_PATH


@dataclass
class StrategyParams:
    rebound_pct: float = 0.015  # Buy trigger: % bounce above running low
    pullback_pct: float = 0.010  # Sell trigger: % drop below running high
    smooth_window: int = 1  # 1 = raw 15-min close
    allow_short: bool = False  # Strictly Long/Cash only
    trained_at: str = ""
    train_sharpe: float = 0.0
    train_return_pct: float = 0.0
    train_trades: int = 0


@dataclass
class TradeRecord:
    timestamp: str
    action: str  # "BUY" or "SELL"
    price: float
    qty: float
    source: str  # "signal", "paper", "live", or "manual"
    pnl_dollars: float = 0.0
    pnl_pct: float = 0.0
    note: str = ""


@dataclass
class EquityPoint:
    timestamp: str
    equity: float
    price: float
    holding: bool


@dataclass
class SymbolState:
    symbol: str
    holding: bool = False
    qty: float = 0.0
    avg_entry_price: float = 0.0
    cash: float = 10_000.0
    initial_cash: float = 10_000.0
    benchmark_start_price: float = 0.0
    running_low: float = 0.0
    running_high: float = 0.0
    last_price: float = 0.0
    last_smooth_price: float = 0.0
    last_bar_time: str = ""
    active_params: StrategyParams = field(default_factory=StrategyParams)
    pending_params: StrategyParams | None = None
    last_retrain_at: str = ""
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)

    def current_equity(self, price: float | None = None) -> float:
        p = price if (price is not None and price > 0) else self.last_price
        if self.holding and p > 0:
            return self.cash + self.qty * p
        return self.cash

    def next_trigger_price(self) -> tuple[str, float]:
        """Return (label, trigger_price) for the next BUY or SELL event."""
        if self.holding:
            base = self.running_high if self.running_high > 0 else self.avg_entry_price
            return "SELL", base * (1.0 - self.active_params.pullback_pct) if base > 0 else 0.0
        base = self.running_low if self.running_low > 0 else self.last_smooth_price
        return "BUY", base * (1.0 + self.active_params.rebound_pct) if base > 0 else 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _symbol_from_dict(d: dict[str, Any]) -> SymbolState:
    ap = StrategyParams(**d["active_params"]) if d.get("active_params") else StrategyParams()
    pp = StrategyParams(**d["pending_params"]) if d.get("pending_params") else None
    trades = [TradeRecord(**t) for t in d.get("trades", [])]
    eq = [EquityPoint(**e) for e in d.get("equity_curve", [])]
    return SymbolState(
        symbol=d["symbol"],
        holding=bool(d.get("holding", False)),
        qty=float(d.get("qty", 0.0)),
        avg_entry_price=float(d.get("avg_entry_price", 0.0)),
        cash=float(d.get("cash", 10_000.0)),
        initial_cash=float(d.get("initial_cash", 10_000.0)),
        benchmark_start_price=float(d.get("benchmark_start_price", 0.0)),
        running_low=float(d.get("running_low", 0.0)),
        running_high=float(d.get("running_high", 0.0)),
        last_price=float(d.get("last_price", 0.0)),
        last_smooth_price=float(d.get("last_smooth_price", 0.0)),
        last_bar_time=str(d.get("last_bar_time", "")),
        active_params=ap,
        pending_params=pp,
        last_retrain_at=str(d.get("last_retrain_at", "")),
        trades=trades,
        equity_curve=eq,
    )


def load_portfolio(symbols: list[str], initial_cash: float = 10_000.0, path: Path = STATE_PATH) -> dict[str, SymbolState]:
    """Load portfolio state from JSON, initializing any new symbols cleanly."""
    portfolio: dict[str, SymbolState] = {}
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        for sym, d in raw.items():
            portfolio[sym.upper()] = _symbol_from_dict(d)

    changed = False
    for sym in symbols:
        s = sym.upper()
        if s not in portfolio:
            portfolio[s] = SymbolState(symbol=s, cash=initial_cash, initial_cash=initial_cash)
            changed = True
    if changed:
        save_portfolio(portfolio, path)
    return portfolio


def save_portfolio(portfolio: dict[str, SymbolState], path: Path = STATE_PATH) -> None:
    """Atomically persist portfolio state to JSON."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {sym: asdict(st) for sym, st in portfolio.items()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def queue_retrained_params(st: SymbolState, new_params: StrategyParams) -> str:
    """Safely apply or queue newly tuned 4-week parameters without interrupting an open trade.

    - If flat (not holding): activates immediately.
    - If currently holding: stores as `pending_params` and activates right after the open trade closes.
    """
    new_params.allow_short = False  # Enforce Long/Cash only
    st.last_retrain_at = new_params.trained_at or _now_iso()
    if not st.holding:
        st.active_params = new_params
        st.pending_params = None
        return "activated_immediately"
    st.pending_params = new_params
    return "queued_until_flat"


def activate_pending_if_flat(st: SymbolState) -> bool:
    """Activate queued pending_params once the symbol is flat (in cash)."""
    if (not st.holding) and (st.pending_params is not None):
        st.active_params = st.pending_params
        st.pending_params = None
        return True
    return False


def apply_buy(
    st: SymbolState,
    price: float,
    qty: float | None = None,
    source: str = "signal",
    timestamp: str | None = None,
    note: str = "",
) -> TradeRecord:
    """Transition symbol from CASH -> HOLDING (Long)."""
    ts = timestamp or _now_iso()
    if qty is None or qty <= 0:
        qty = st.cash / price if price > 0 else 0.0
        cost = st.cash
    else:
        cost = qty * price

    st.cash = max(0.0, st.cash - cost)
    st.qty = qty
    st.avg_entry_price = price
    st.holding = True
    st.running_high = max(price, st.last_smooth_price or price)
    st.last_price = price
    if st.benchmark_start_price <= 0:
        st.benchmark_start_price = price

    rec = TradeRecord(timestamp=ts, action="BUY", price=price, qty=qty, source=source, note=note)
    st.trades.append(rec)
    return rec


def apply_sell(
    st: SymbolState,
    price: float,
    source: str = "signal",
    timestamp: str | None = None,
    note: str = "",
) -> TradeRecord:
    """Transition symbol from HOLDING -> CASH and activate any queued 4-week params."""
    ts = timestamp or _now_iso()
    qty = st.qty
    entry = st.avg_entry_price
    proceeds = qty * price
    pnl_dollars = (price - entry) * qty if entry > 0 else 0.0
    pnl_pct = ((price - entry) / entry * 100.0) if entry > 0 else 0.0

    st.cash = st.cash + proceeds
    st.qty = 0.0
    st.avg_entry_price = 0.0
    st.holding = False
    st.running_low = min(price, st.last_smooth_price or price)
    st.last_price = price

    rec = TradeRecord(
        timestamp=ts,
        action="SELL",
        price=price,
        qty=qty,
        source=source,
        pnl_dollars=round(pnl_dollars, 2),
        pnl_pct=round(pnl_pct, 3),
        note=note,
    )
    st.trades.append(rec)
    activate_pending_if_flat(st)
    return rec


def set_manual_wallet_state(
    st: SymbolState,
    holding: bool,
    price: float,
    qty: float | None = None,
    cash: float | None = None,
    note: str = "manual wallet override",
) -> str:
    """Ingest a manual trade or override current wallet status cleanly."""
    ts = _now_iso()
    if holding and not st.holding:
        if cash is not None:
            st.cash = cash
        apply_buy(st, price=price, qty=qty, source="manual", timestamp=ts, note=note)
        return f"Logged MANUAL BUY: {st.qty:.4f} {st.symbol} @ ${price:.2f}"
    if (not holding) and st.holding:
        apply_sell(st, price=price, source="manual", timestamp=ts, note=note)
        if cash is not None:
            st.cash = cash
        return f"Logged MANUAL SELL: {st.symbol} @ ${price:.2f} (cash=${st.cash:.2f})"

    # Same holding status -> direct wallet adjustment without fabricating a round-trip
    if holding:
        if qty is not None and qty > 0:
            st.qty = qty
        if price > 0:
            st.avg_entry_price = price
            st.running_high = max(price, st.last_price or price)
        if cash is not None:
            st.cash = cash
        return f"Updated HOLDING wallet: {st.qty:.4f} {st.symbol} @ avg ${st.avg_entry_price:.2f}, cash=${st.cash:.2f}"

    if cash is not None:
        st.cash = cash
    if price > 0:
        st.running_low = price
    activate_pending_if_flat(st)
    return f"Updated CASH wallet: {st.symbol} flat, cash=${st.cash:.2f}"
