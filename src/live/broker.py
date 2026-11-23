"""Alpaca IEX real-time data client, optional Paper/Live Long-Only order execution, and macOS alerts."""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.fetch_data import EASTERN_TZ, filter_rth, load_credentials


def get_iex_data_client() -> StockHistoricalDataClient:
    api_key, secret_key = load_credentials()
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def get_trading_client(paper: bool = True) -> TradingClient:
    api_key, secret_key = load_credentials()
    return TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)


def fetch_iex_15m_bars(symbol: str, lookback_days: int = 14) -> pd.DataFrame:
    """Fetch recent real-time 15-minute RTH bars using free Alpaca DataFeed.IEX (zero SIP)."""
    client = get_iex_data_client()
    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=lookback_days)

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(15, TimeFrameUnit.Minute),
        start=start_utc,
        end=end_utc,
        adjustment=Adjustment.SPLIT,
        feed=DataFeed.IEX,
    )
    df = client.get_stock_bars(req).df
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(EASTERN_TZ)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return filter_rth(df)


def fetch_iex_training_15m_bars(symbol: str, calendar_days: int = 200) -> pd.DataFrame:
    """Fetch ~6+ months of 15-minute RTH bars from free DataFeed.IEX for 4-week WFO re-tuning."""
    client = get_iex_data_client()
    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=calendar_days)

    chunks: list[pd.DataFrame] = []
    cs = start_utc
    while cs < end_utc:
        ce = min(cs + timedelta(days=45), end_utc)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(15, TimeFrameUnit.Minute),
            start=cs,
            end=ce,
            adjustment=Adjustment.SPLIT,
            feed=DataFeed.IEX,
        )
        dfc = client.get_stock_bars(req).df
        if not dfc.empty:
            if isinstance(dfc.index, pd.MultiIndex):
                dfc = dfc.xs(symbol, level="symbol")
            chunks.append(dfc)
        cs = ce

    if not chunks:
        raise RuntimeError(f"No IEX 15-min bars returned for {symbol}.")

    df = pd.concat(chunks)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(EASTERN_TZ)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return filter_rth(df)


def submit_long_only_order(
    symbol: str,
    side: str,
    notional: float | None = None,
    qty: float | None = None,
    paper: bool = True,
) -> dict:
    """Submit a Long/Cash market order to Alpaca (never opens a short position)."""
    tc = get_trading_client(paper=paper)
    if side.upper() == "BUY":
        req = MarketOrderRequest(
            symbol=symbol,
            notional=round(notional, 2) if notional else None,
            qty=round(qty, 4) if (qty and not notional) else None,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = tc.submit_order(req)
        return {"id": str(order.id), "status": str(order.status), "side": "BUY"}

    if side.upper() == "SELL":
        # Close existing long position cleanly (never sells short)
        order = tc.close_position(symbol)
        return {"id": str(order.id), "status": str(order.status), "side": "SELL"}

    raise ValueError(f"Unsupported side: {side}")


def get_broker_position(symbol: str, paper: bool = True) -> dict | None:
    """Fetch current Alpaca open position for `symbol`, or None if flat."""
    tc = get_trading_client(paper=paper)
    try:
        pos = tc.get_open_position(symbol)
        return {
            "qty": float(pos.qty),
            "avg_entry_price": float(pos.avg_entry_price),
            "market_value": float(pos.market_value),
            "unrealized_pl": float(pos.unrealized_pl),
        }
    except Exception:
        return None


def notify_macos(title: str, message: str) -> None:
    """Send a non-blocking macOS desktop notification with sound."""
    safe_t = title.replace('"', "'")
    safe_m = message.replace('"', "'")
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{safe_m}" with title "{safe_t}" sound name "Glass"'],
            check=False,
            capture_output=True,
        )
    except Exception:
        pass
