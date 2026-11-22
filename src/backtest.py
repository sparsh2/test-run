"""Fast, realistic, zero-lookahead backtesting engine for AAPL 1-minute kinematic strategies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

from src.features import FeatureConfig, compute_kinematic_features


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters governing kinematic entry/exit rules, risk controls, and friction."""

    # Kinematic signal thresholds (in bps/bar and bps/bar^2)
    vel_thresh: float = 1.5  # Minimum |velocity| to enter
    acc_thresh: float = 0.2  # Minimum |acceleration| in same direction to enter
    exit_mode: str = "vel_cross"  # "vel_cross", "acc_decel", or "flip_only"
    exit_vel_thresh: float = 0.0  # Velocity threshold to exit (0.0 = zero-crossing)
    exit_acc_thresh: float = 0.0  # Deceleration threshold when exit_mode == "acc_decel"

    # Regime / noise filters
    min_atr_bps: float = 0.0  # Skip entries when rolling ATR < min_atr_bps
    min_rel_vol: float = 0.0  # Skip entries when relative volume < min_rel_vol
    warmup_bars: int = 15  # Stay flat for first N bars of each trading day
    min_hold_bars: int = 1  # Minimum bars to hold before kinematic exit

    # Risk controls (0.0 disables)
    stop_loss_bps: float = 0.0
    take_profit_bps: float = 0.0

    # Session & execution
    flatten_eod: bool = True  # Flatten before close (no overnight holds)
    eod_no_entry_time: time = time(15, 50)
    eod_flatten_time: time = time(15, 55)
    cost_bps_per_side: float = 1.5  # Slippage + spread per side (1.5 bps = 0.015%)


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int  # +1 long, -1 short
    entry_price: float
    exit_price: float
    gross_ret: float
    net_ret: float
    exit_reason: str
    hold_bars: int


@dataclass
class BacktestResult:
    equity: pd.Series
    buy_hold_equity: pd.Series
    positions: pd.Series
    trades: list[Trade]
    metrics: dict[str, float]
    features: pd.DataFrame


def compute_oracle_ceiling(
    prices: np.ndarray,
    swing_pct: float = 0.0025,
    cost_bps_per_side: float = 1.5,
) -> tuple[float, int]:
    """Compute theoretical ZigZag Oracle return capturing every swing >= swing_pct (long & short).

    This is an idealized non-causal upper bound (with identical transaction costs per leg)
    showing how much directional price movement exists above `swing_pct`.
    Returns (total_return_fraction, num_swings).
    """
    n = len(prices)
    if n < 2:
        return 0.0, 0

    # Identify ZigZag turning points
    pivots: list[float] = [float(prices[0])]
    trend = 0  # 0 unknown, +1 up, -1 down
    extreme = float(prices[0])

    for i in range(1, n):
        p = float(prices[i])
        if trend == 0:
            if p >= extreme * (1.0 + swing_pct):
                trend = 1
                extreme = p
            elif p <= extreme * (1.0 - swing_pct):
                trend = -1
                extreme = p
        elif trend == 1:
            if p > extreme:
                extreme = p
            elif p <= extreme * (1.0 - swing_pct):
                pivots.append(extreme)
                trend = -1
                extreme = p
        else:  # trend == -1
            if p < extreme:
                extreme = p
            elif p >= extreme * (1.0 + swing_pct):
                pivots.append(extreme)
                trend = 1
                extreme = p

    pivots.append(extreme)
    if len(pivots) < 2:
        return 0.0, 0

    cost_rt = 2.0 * cost_bps_per_side / 10_000.0
    equity = 1.0
    swings = 0
    for i in range(len(pivots) - 1):
        p0, p1 = pivots[i], pivots[i + 1]
        gross = abs(p1 - p0) / p0
        net = gross - cost_rt
        if net > 0.0:
            equity *= 1.0 + net
            swings += 1

    return equity - 1.0, swings


def run_backtest(
    df: pd.DataFrame,
    feat_cfg: FeatureConfig,
    strat_cfg: StrategyConfig,
    features: pd.DataFrame | None = None,
    compute_oracle: bool = True,
    oracle_swing_pct: float = 0.0025,
) -> BacktestResult:
    """Run a strictly causal event-driven backtest.

    Execution semantics (zero lookahead):
      - Signals are evaluated at the close of bar t using features[t].
      - Position changes execute at the OPEN of bar t + 1.
      - Stop-loss / take-profit checks use intrabar high/low on bar t while in a position,
        assuming SL triggers before TP if both are touched within the same 1-minute bar.
    """
    if features is None:
        features = compute_kinematic_features(df, feat_cfg)

    n = len(df)
    if n < 2:
        raise ValueError("DataFrame must contain at least 2 bars.")

    opens = df["open"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    timestamps = df.index

    vel = features["velocity"].to_numpy(dtype=np.float64)
    acc = features["acceleration"].to_numpy(dtype=np.float64)
    atr = features["atr_bps"].to_numpy(dtype=np.float64)
    rvol = features["rel_volume"].to_numpy(dtype=np.float64)
    bar_in_day = features["bar_in_day"].to_numpy(dtype=np.int32)

    # Precompute EOD masks once in vectorized form
    times = timestamps.time
    no_entry_mask = times >= strat_cfg.eod_no_entry_time if strat_cfg.flatten_eod else np.zeros(n, dtype=bool)
    flatten_mask = times >= strat_cfg.eod_flatten_time if strat_cfg.flatten_eod else np.zeros(n, dtype=bool)

    cost_rate = strat_cfg.cost_bps_per_side / 10_000.0
    sl_frac = strat_cfg.stop_loss_bps / 10_000.0
    tp_frac = strat_cfg.take_profit_bps / 10_000.0
    use_sl = sl_frac > 0.0
    use_tp = tp_frac > 0.0

    v_th = strat_cfg.vel_thresh
    a_th = strat_cfg.acc_thresh
    ex_v_th = strat_cfg.exit_vel_thresh
    ex_a_th = strat_cfg.exit_acc_thresh
    min_atr = strat_cfg.min_atr_bps
    min_rvol = strat_cfg.min_rel_vol
    warmup = strat_cfg.warmup_bars
    min_hold = strat_cfg.min_hold_bars
    exit_mode = strat_cfg.exit_mode

    equity_arr = np.empty(n, dtype=np.float64)
    pos_arr = np.zeros(n, dtype=np.int8)
    trades: list[Trade] = []

    equity = 1.0
    pos = 0  # Current active position during bar t: -1, 0, +1
    entry_price = 0.0
    entry_idx = -1
    pending_target = 0  # Target position to execute at bar t open (decided at bar t-1 close)
    pending_reason = ""

    equity_arr[0] = equity

    for t in range(n):
        # 1. Execute pending order from bar t-1 at bar t OPEN
        if t > 0 and pending_target != pos:
            exec_price = opens[t]
            # Close existing position if open
            if pos != 0:
                gross = pos * (exec_price - entry_price) / entry_price
                net = gross - 2.0 * cost_rate
                equity *= 1.0 + net
                trades.append(
                    Trade(
                        entry_idx=entry_idx,
                        exit_idx=t,
                        entry_time=timestamps[entry_idx],
                        exit_time=timestamps[t],
                        side=pos,
                        entry_price=entry_price,
                        exit_price=exec_price,
                        gross_ret=gross,
                        net_ret=net,
                        exit_reason=pending_reason,
                        hold_bars=t - entry_idx,
                    )
                )
                pos = 0

            # Open new position if requested
            if pending_target != 0:
                pos = pending_target
                entry_price = exec_price
                entry_idx = t

        # 2. Check intrabar Stop-Loss / Take-Profit while holding position on bar t
        exited_intrabar = False
        if pos != 0 and (use_sl or use_tp):
            if pos == 1:
                sl_price = entry_price * (1.0 - sl_frac) if use_sl else -1.0
                tp_price = entry_price * (1.0 + tp_frac) if use_tp else 1e18
                # Conservative: check SL before TP
                if use_sl and lows[t] <= sl_price:
                    fill = min(opens[t], sl_price)
                    gross = (fill - entry_price) / entry_price
                    net = gross - 2.0 * cost_rate
                    equity *= 1.0 + net
                    trades.append(
                        Trade(
                            entry_idx=entry_idx,
                            exit_idx=t,
                            entry_time=timestamps[entry_idx],
                            exit_time=timestamps[t],
                            side=1,
                            entry_price=entry_price,
                            exit_price=fill,
                            gross_ret=gross,
                            net_ret=net,
                            exit_reason="stop_loss",
                            hold_bars=max(1, t - entry_idx),
                        )
                    )
                    pos = 0
                    exited_intrabar = True
                elif use_tp and highs[t] >= tp_price:
                    fill = max(opens[t], tp_price)
                    gross = (fill - entry_price) / entry_price
                    net = gross - 2.0 * cost_rate
                    equity *= 1.0 + net
                    trades.append(
                        Trade(
                            entry_idx=entry_idx,
                            exit_idx=t,
                            entry_time=timestamps[entry_idx],
                            exit_time=timestamps[t],
                            side=1,
                            entry_price=entry_price,
                            exit_price=fill,
                            gross_ret=gross,
                            net_ret=net,
                            exit_reason="take_profit",
                            hold_bars=max(1, t - entry_idx),
                        )
                    )
                    pos = 0
                    exited_intrabar = True
            else:  # pos == -1
                sl_price = entry_price * (1.0 + sl_frac) if use_sl else 1e18
                tp_price = entry_price * (1.0 - tp_frac) if use_tp else -1.0
                if use_sl and highs[t] >= sl_price:
                    fill = max(opens[t], sl_price)
                    gross = -(fill - entry_price) / entry_price
                    net = gross - 2.0 * cost_rate
                    equity *= 1.0 + net
                    trades.append(
                        Trade(
                            entry_idx=entry_idx,
                            exit_idx=t,
                            entry_time=timestamps[entry_idx],
                            exit_time=timestamps[t],
                            side=-1,
                            entry_price=entry_price,
                            exit_price=fill,
                            gross_ret=gross,
                            net_ret=net,
                            exit_reason="stop_loss",
                            hold_bars=max(1, t - entry_idx),
                        )
                    )
                    pos = 0
                    exited_intrabar = True
                elif use_tp and lows[t] <= tp_price:
                    fill = min(opens[t], tp_price)
                    gross = -(fill - entry_price) / entry_price
                    net = gross - 2.0 * cost_rate
                    equity *= 1.0 + net
                    trades.append(
                        Trade(
                            entry_idx=entry_idx,
                            exit_idx=t,
                            entry_time=timestamps[entry_idx],
                            exit_time=timestamps[t],
                            side=-1,
                            entry_price=entry_price,
                            exit_price=fill,
                            gross_ret=gross,
                            net_ret=net,
                            exit_reason="take_profit",
                            hold_bars=max(1, t - entry_idx),
                        )
                    )
                    pos = 0
                    exited_intrabar = True

        # 3. Record mark-to-market equity at bar t CLOSE
        pos_arr[t] = pos
        if pos != 0:
            unrealized = pos * (closes[t] - entry_price) / entry_price - cost_rate
            equity_arr[t] = equity * (1.0 + unrealized)
        else:
            equity_arr[t] = equity

        # 4. Decide target position at bar t CLOSE (will execute at bar t+1 OPEN)
        if t == n - 1:
            # Close any remaining position at final bar close for clean accounting
            if pos != 0:
                gross = pos * (closes[t] - entry_price) / entry_price
                net = gross - 2.0 * cost_rate
                equity *= 1.0 + net
                equity_arr[t] = equity
                trades.append(
                    Trade(
                        entry_idx=entry_idx,
                        exit_idx=t,
                        entry_time=timestamps[entry_idx],
                        exit_time=timestamps[t],
                        side=pos,
                        entry_price=entry_price,
                        exit_price=closes[t],
                        gross_ret=gross,
                        net_ret=net,
                        exit_reason="end_of_data",
                        hold_bars=max(1, t - entry_idx),
                    )
                )
                pos = 0
            break

        # EOD flatten rule
        if flatten_mask[t]:
            pending_target = 0
            pending_reason = "eod_flatten"
            continue

        # Avoid re-entering on the exact same bar we got stopped out
        if exited_intrabar:
            pending_target = 0
            pending_reason = ""
            continue

        # Raw kinematic signals at bar t close
        vt = vel[t]
        at = acc[t]
        can_enter = (
            (bar_in_day[t] >= warmup)
            and (not no_entry_mask[t])
            and (atr[t] >= min_atr)
            and (rvol[t] >= min_rvol)
        )

        long_sig = can_enter and (vt > v_th) and (at > a_th)
        short_sig = can_enter and (vt < -v_th) and (at < -a_th)

        if pos == 0:
            if long_sig:
                pending_target = 1
                pending_reason = ""
            elif short_sig:
                pending_target = -1
                pending_reason = ""
            else:
                pending_target = 0
                pending_reason = ""
        else:
            held = t - entry_idx + 1
            if pos == 1:
                if short_sig:
                    pending_target = -1
                    pending_reason = "flip_short"
                elif held >= min_hold and exit_mode == "vel_cross" and vt < -ex_v_th:
                    pending_target = 0
                    pending_reason = "vel_exit"
                elif held >= min_hold and exit_mode == "acc_decel" and (vt <= 0.0 or at < -ex_a_th):
                    pending_target = 0
                    pending_reason = "acc_exit"
                else:
                    pending_target = 1
                    pending_reason = ""
            else:  # pos == -1
                if long_sig:
                    pending_target = 1
                    pending_reason = "flip_long"
                elif held >= min_hold and exit_mode == "vel_cross" and vt > ex_v_th:
                    pending_target = 0
                    pending_reason = "vel_exit"
                elif held >= min_hold and exit_mode == "acc_decel" and (vt >= 0.0 or at > ex_a_th):
                    pending_target = 0
                    pending_reason = "acc_exit"
                else:
                    pending_target = -1
                    pending_reason = ""

    equity_s = pd.Series(equity_arr, index=timestamps, name="strategy_equity")
    bh_equity = pd.Series(closes / opens[0], index=timestamps, name="buy_hold_equity")
    pos_s = pd.Series(pos_arr, index=timestamps, name="position")

    metrics = compute_metrics(
        equity=equity_s,
        bh_equity=bh_equity,
        trades=trades,
        closes=closes,
        cost_bps_per_side=strat_cfg.cost_bps_per_side,
        compute_oracle=compute_oracle,
        oracle_swing_pct=oracle_swing_pct,
    )

    return BacktestResult(
        equity=equity_s,
        buy_hold_equity=bh_equity,
        positions=pos_s,
        trades=trades,
        metrics=metrics,
        features=features,
    )


def compute_metrics(
    equity: pd.Series,
    bh_equity: pd.Series,
    trades: list[Trade],
    closes: np.ndarray,
    cost_bps_per_side: float = 1.5,
    compute_oracle: bool = True,
    oracle_swing_pct: float = 0.0025,
) -> dict[str, float]:
    """Compute standard portfolio and trade-level performance metrics."""
    total_ret = float(equity.iloc[-1] - 1.0)
    bh_ret = float(bh_equity.iloc[-1] - 1.0)

    # Daily equity curve for annualized Sharpe (avoids 1-min autocorrelation distortion)
    daily_eq = equity.groupby(equity.index.date).last()
    daily_ret = daily_eq.pct_change().dropna()
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252.0))
    else:
        sharpe = 0.0

    # Buy & Hold daily Sharpe
    bh_daily = bh_equity.groupby(bh_equity.index.date).last().pct_change().dropna()
    bh_sharpe = (
        float((bh_daily.mean() / bh_daily.std()) * np.sqrt(252.0))
        if len(bh_daily) > 1 and bh_daily.std() > 0
        else 0.0
    )

    # Max drawdown
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    max_dd = float(dd.min())

    bh_dd = float((bh_equity / bh_equity.cummax() - 1.0).min())

    # Trade stats
    n_trades = len(trades)
    if n_trades > 0:
        net_rets = np.array([tr.net_ret for tr in trades], dtype=np.float64)
        gross_rets = np.array([tr.gross_ret for tr in trades], dtype=np.float64)
        holds = np.array([tr.hold_bars for tr in trades], dtype=np.float64)
        wins = net_rets[net_rets > 0]
        losses = net_rets[net_rets <= 0]
        win_rate = float(len(wins) / n_trades)
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        avg_trade_bps = float(net_rets.mean() * 10_000.0)
        avg_gross_bps = float(gross_rets.mean() * 10_000.0)
        avg_hold_bars = float(holds.mean())
        long_trades = sum(1 for tr in trades if tr.side == 1)
        short_trades = sum(1 for tr in trades if tr.side == -1)
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_bps = 0.0
        avg_gross_bps = 0.0
        avg_hold_bars = 0.0
        long_trades = 0
        short_trades = 0

    pct_bh_captured = float((total_ret / bh_ret) * 100.0) if abs(bh_ret) > 1e-9 else 0.0

    oracle_ret = 0.0
    oracle_swings = 0
    pct_oracle_captured = 0.0
    if compute_oracle:
        oracle_ret, oracle_swings = compute_oracle_ceiling(
            closes, swing_pct=oracle_swing_pct, cost_bps_per_side=cost_bps_per_side
        )
        pct_oracle_captured = (
            float((total_ret / oracle_ret) * 100.0) if oracle_ret > 1e-9 else 0.0
        )

    return {
        "total_return_pct": total_ret * 100.0,
        "bh_return_pct": bh_ret * 100.0,
        "oracle_return_pct": oracle_ret * 100.0,
        "oracle_swings": float(oracle_swings),
        "pct_bh_captured": pct_bh_captured,
        "pct_oracle_captured": pct_oracle_captured,
        "sharpe": sharpe,
        "bh_sharpe": bh_sharpe,
        "max_dd_pct": max_dd * 100.0,
        "bh_max_dd_pct": bh_dd * 100.0,
        "n_trades": float(n_trades),
        "long_trades": float(long_trades),
        "short_trades": float(short_trades),
        "win_rate_pct": win_rate * 100.0,
        "profit_factor": profit_factor,
        "avg_trade_bps": avg_trade_bps,
        "avg_gross_bps": avg_gross_bps,
        "avg_hold_bars": avg_hold_bars,
    }
