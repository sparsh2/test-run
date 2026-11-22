"""Fetch and audit 1-minute historical AAPL OHLCV bars from Alpaca (SIP feed)."""

from __future__ import annotations

import os
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "aapl_1min_18m.parquet"
SYMBOL = "AAPL"
LOOKBACK_DAYS = 548  # ~1.5 years
CHUNK_DAYS = 30
EASTERN_TZ = "America/New_York"

RTH_START = time(9, 30)
RTH_END = time(15, 59)


def load_credentials() -> tuple[str, str]:
    """Load Alpaca API credentials from .env or environment variables."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()

    if not api_key or not secret_key:
        raise ValueError(
            "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY. "
            f"Please add them to {PROJECT_ROOT / '.env'} or export them as environment variables."
        )
    return api_key, secret_key


def fetch_historical_bars(
    symbol: str = SYMBOL,
    lookback_days: int = LOOKBACK_DAYS,
    chunk_days: int = CHUNK_DAYS,
) -> pd.DataFrame:
    """Fetch 1-minute OHLCV bars from Alpaca SIP feed in chunks.

    Uses Adjustment.SPLIT and sets end = now(UTC) - 20 minutes to comply
    with free-tier SIP delayed data restrictions.
    """
    api_key, secret_key = load_credentials()
    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    end_utc = datetime.now(timezone.utc) - timedelta(minutes=20)
    start_utc = end_utc - timedelta(days=lookback_days)

    print(f"Fetching {symbol} 1-minute bars (SIP, split-adjusted)...")
    print(f"UTC range: {start_utc.isoformat()} -> {end_utc.isoformat()}")

    chunks: list[pd.DataFrame] = []
    chunk_start = start_utc
    chunk_idx = 0

    while chunk_start < end_utc:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end_utc)
        chunk_idx += 1
        print(
            f"  [Chunk {chunk_idx:02d}] {chunk_start.strftime('%Y-%m-%d %H:%M')} -> "
            f"{chunk_end.strftime('%Y-%m-%d %H:%M')} UTC ...",
            end=" ",
            flush=True,
        )

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=chunk_start,
            end=chunk_end,
            adjustment=Adjustment.SPLIT,
            feed=DataFeed.SIP,
        )
        bars = client.get_stock_bars(request)
        df_chunk = bars.df

        if not df_chunk.empty:
            # Handle MultiIndex (symbol, timestamp) returned by alpaca-py
            if isinstance(df_chunk.index, pd.MultiIndex):
                df_chunk = df_chunk.xs(symbol, level="symbol")
            chunks.append(df_chunk)
            print(f"{len(df_chunk):,} bars")
        else:
            print("0 bars")

        chunk_start = chunk_end

    if not chunks:
        raise RuntimeError(f"No bar data returned for {symbol}.")

    df = pd.concat(chunks)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(EASTERN_TZ)
    df.index.name = "timestamp"

    # Deduplicate overlapping chunk boundaries and sort chronologically
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Filter a DataFrame indexed by Eastern-time timestamps to Regular Trading Hours (09:30-15:59 ET)."""
    if df.index.tz is None:
        raise ValueError("DataFrame index must be timezone-aware (America/New_York).")

    ts_et = df.index.tz_convert(EASTERN_TZ)
    times = ts_et.time
    mask = (times >= RTH_START) & (times <= RTH_END)
    return df.loc[mask].copy()


def audit_data(df: pd.DataFrame) -> None:
    """Print diagnostic statistics for the fetched 1-minute dataset."""
    print("\n" + "=" * 64)
    print("DATA AUDIT SUMMARY")
    print("=" * 64)

    total_rows = len(df)
    print(f"Total rows (full session): {total_rows:,}")
    print(f"Columns:                   {list(df.columns)}")
    print(f"Date range (ET):           {df.index.min()} -> {df.index.max()}")
    print(f"Duplicate timestamps:      {df.index.duplicated().sum():,}")
    print(f"Null values (total):       {int(df.isna().sum().sum()):,}")

    # Session breakdown
    times = df.index.time
    pre_mask = (times >= time(4, 0)) & (times < RTH_START)
    rth_mask = (times >= RTH_START) & (times <= RTH_END)
    post_mask = (times > RTH_END) & (times <= time(20, 0))

    print("\nSession Breakdown (ET):")
    print(f"  Pre-market  (04:00-09:29): {pre_mask.sum():>9,} bars")
    print(f"  RTH         (09:30-15:59): {rth_mask.sum():>9,} bars")
    print(f"  Post-market (16:00-20:00): {post_mask.sum():>9,} bars")

    # RTH missing-minute diagnostics
    rth = filter_rth(df)
    bars_per_day = rth.groupby(rth.index.date).size()
    expected_full_day = 390  # 09:30 through 15:59 inclusive

    full_days = (bars_per_day == expected_full_day).sum()
    short_days = bars_per_day[bars_per_day < expected_full_day]
    # Half-days typically close at 13:00 ET (210 minutes: 09:30-12:59)
    likely_early_close = short_days[short_days <= 215]
    partial_regular_days = short_days[short_days > 215]

    total_missing_regular = int((expected_full_day - partial_regular_days).sum())

    print("\nRegular Trading Hours (RTH) Completeness:")
    print(f"  Trading days observed:             {len(bars_per_day):,}")
    print(f"  Complete 390-bar days:             {full_days:,}")
    print(f"  Likely early-close days (<=215m):  {len(likely_early_close):,}")
    print(f"  Regular days with missing bars:    {len(partial_regular_days):,}")
    print(f"  Missing minutes on regular days:   {total_missing_regular:,}")

    if not partial_regular_days.empty:
        worst = partial_regular_days.nsmallest(5)
        print("  Lowest-count regular days:")
        for dt, count in worst.items():
            print(f"    {dt}: {count} / {expected_full_day} bars (missing {expected_full_day - count})")

    print("=" * 64 + "\n")


def load_data(path: Path = DEFAULT_OUTPUT_PATH, rth_only: bool = False) -> pd.DataFrame:
    """Load saved parquet dataset and optionally filter to RTH."""
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(EASTERN_TZ)
    else:
        df.index = df.index.tz_convert(EASTERN_TZ)
    return filter_rth(df) if rth_only else df


def main() -> None:
    df = fetch_historical_bars()
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DEFAULT_OUTPUT_PATH)
    print(f"\nSaved {len(df):,} rows to {DEFAULT_OUTPUT_PATH}")
    audit_data(df)


if __name__ == "__main__":
    main()
