# Stock Kinematic & Peak/Trough Backtesting Pipeline

A reproducible Python research and backtesting pipeline using **Alpaca SIP 1-minute historical data** to test strictly causal trading strategies (zero lookahead, realistic `1.5 bps/side` execution friction, chronological Train/Val/Test splits, and rolling Walk-Forward Optimization) on **AAPL** or **any US stock/ETF ticker** (`MSFT`, `NVDA`, `SPY`, `QQQ`, etc.).

Full AAPL findings, charts, and next-phase TODOs are documented in [`report.md`](report.md).

---

## 1. Setup

### Prerequisites
- Python 3.12+ and [`uv`](https://docs.astral.sh/uv/)
- Free [Alpaca Paper Trading API](https://alpaca.markets/) keys

### Install Dependencies
```bash
uv sync
```

### Configure API Keys
Copy `.env.example` to `.env` (`.env` is gitignored) and paste your Alpaca keys:
```bash
cp .env.example .env
```
```env
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
```

---

## 2. Running on Any Stock Ticker

Every script accepts `--symbol <TICKER>` (defaults to `AAPL`).

### Step 1: Download 1.5 Years of 1-Minute SIP Data
Downloads split-adjusted 1-minute OHLCV + VWAP bars (548 days ending 20 minutes ago) to `data/<symbol>_1min_18m.parquet` and prints a completeness audit:

```bash
# Default (AAPL)
uv run python -m src.fetch_data

# Any other stock or ETF (e.g., MSFT, NVDA, SPY, QQQ)
uv run python -m src.fetch_data --symbol MSFT
uv run python -m src.fetch_data --symbol SPY --days 548
```

### Step 2: Run Backtests & Strategies

#### A. Causal Peak & Trough Reversal Strategy (Best Performer on AAPL)
Buys right after a confirmed trough (`rebound_pct` off running low) and sells/shorts right after a confirmed peak (`pullback_pct` off running high) on 15-min bars with 4-week rolling Walk-Forward Optimization + zoomed trade charts:
```bash
uv run python -m src.peak_trough_strategy --symbol AAPL
uv run python -m src.peak_trough_strategy --symbol MSFT
```

#### B. Bollinger Band Peak & Trough Strategy
Uses causal Bollinger Bands (`band_reentry` and `bb_width`) instead of a fixed percentage threshold, plus 3 zoomed shorter-period charts:
```bash
uv run python -m src.bollinger_peak_trough --symbol AAPL
uv run python -m src.bollinger_peak_trough --symbol MSFT
```

#### C. Rolling Walk-Forward Optimization (H1, H2, H3 — 2-Week & 4-Week Re-tuning)
Tests 15-min Momentum+VWAP (H1), 5-min Exhaustion Mean-Reversion (H2), and 15-min Multi-Day Swing Filter (H3) under 2-week and 4-week rolling WFO:
```bash
uv run python -m src.walk_forward --symbol AAPL
uv run python -m src.walk_forward --symbol MSFT
```

#### D. Higher-Timeframe & Mean-Reversion Hypothesis Explorer (Static Split)
```bash
uv run python -m src.explore_edge --symbol AAPL
uv run python -m src.explore_edge --symbol MSFT
```

#### E. Full 1-Minute Kinematic Grid Search, Stability Heatmap & Report
Runs 2,880 combinations across `ema`, `dema`, `linreg`, `savgol`, and `kalman` on 1-minute bars (Train/Val/OOS Test), friction sensitivity, and regime diagnostics (~5–6 minutes):
```bash
uv run python -m src.optimize --symbol AAPL
uv run python -m src.optimize --symbol MSFT
```

#### F. Plot 1-Minute Kalman Filter Trades
```bash
uv run python -m src.plot_kalman_trades
```

---

## 3. Project Structure

```text
.
├── data/                         # Downloaded 1-min SIP parquet datasets (<symbol>_1min_18m.parquet)
├── outputs/                      # Generated equity curves, trade charts, heatmaps, and CSVs
├── src/
│   ├── fetch_data.py             # Alpaca SIP 1-min data downloader, RTH filter, and data audit
│   ├── features.py               # Strictly causal EMA, DEMA, SavGol, LinReg, and Kalman features
│   ├── backtest.py               # Zero-lookahead backtesting engine, friction, B&H & ZigZag Oracle
│   ├── optimize.py               # 1-min grid search, +-10/20% stability check, and report generator
│   ├── explore_edge.py           # Higher-TF (5m/15m/30m) momentum, mean-reversion & swing filter tests
│   ├── walk_forward.py           # Rolling 2-week & 4-week Walk-Forward Optimization engine
│   ├── peak_trough_strategy.py   # Causal fixed-% Peak & Trough reversal strategy + zoomed charts
│   ├── bollinger_peak_trough.py  # Causal Bollinger Band Peak & Trough strategy + zoomed charts
│   └── plot_kalman_trades.py     # Trade visualization for the 1-min Kalman Filter strategy
├── report.md                     # Full AAPL experimental findings, charts, and TODOs
├── .env.example                  # Template for Alpaca API credentials
└── pyproject.toml                # Project dependencies managed by uv
```
