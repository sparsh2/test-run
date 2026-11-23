"""CLI for managing the live 15-min IEX Peak & Trough tracker, manual wallet overrides, and retraining."""

from __future__ import annotations

import argparse

from src.live.config import load_config, save_config
from src.live.retrain import retrain_symbols
from src.live.runner import run_loop, run_once
from src.live.state import load_portfolio, save_portfolio, set_manual_wallet_state


def cmd_status() -> None:
    cfg = load_config()
    portfolio = load_portfolio(cfg.symbols, initial_cash=cfg.initial_cash_per_symbol)
    print("=" * 92)
    print(f"LIVE PORTFOLIO STATUS | mode={cfg.mode} | feed=IEX (real-time free) | symbols={cfg.symbols}")
    print("=" * 92)
    for sym in cfg.symbols:
        st = portfolio[sym]
        eq = st.current_equity()
        ret_pct = (eq / st.initial_cash - 1.0) * 100.0 if st.initial_cash > 0 else 0.0
        trig_side, trig_px = st.next_trigger_price()
        ap = st.active_params
        status_lbl = f"HOLDING {st.qty:.4f} @ ${st.avg_entry_price:.2f}" if st.holding else f"CASH (${st.cash:,.2f})"
        print(f"  {sym:6s} | {status_lbl:30s} | Equity: ${eq:,.2f} ({ret_pct:+.2f}%) | Last: ${st.last_price:.2f}")
        print(
            f"         | Active Params: reb={ap.rebound_pct*100:.1f}%, pb={ap.pullback_pct*100:.1f}%, sw={ap.smooth_window} "
            f"| Next {trig_side} Trigger: ${trig_px:.2f} | Pending Retrain: {'YES' if st.pending_params else 'no'}"
        )
        print(f"         | Last Bar: {st.last_bar_time or 'none'} | Last Retrain: {st.last_retrain_at or 'never'} | Trades: {len(st.trades)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live 15-min IEX Peak & Trough CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Show current wallet status, active params, and next buy/sell triggers")
    sub.add_parser("tick", help="Run a single 15-min IEX evaluation tick now")
    sub.add_parser("run", help="Start the continuous 15-min RTH runner loop")

    p_ret = sub.add_parser("retrain", help="Run 4-week WFO IEX parameter re-tuning now")
    p_ret.add_argument("--symbol", default=None, help="Specific symbol (default: all configured symbols)")
    p_ret.add_argument("--force", action="store_true", help="Force retrain even if < 28 days since last run")

    p_set = sub.add_parser("set-wallet", help="Manually set wallet status or ingest a manual buy/sell trade")
    p_set.add_argument("--symbol", required=True, help="Ticker symbol (e.g. TSLA)")
    p_set.add_argument("--holding", required=True, choices=["yes", "no"], help="Do you currently hold the stock?")
    p_set.add_argument("--price", type=float, required=True, help="Fill/current price ($)")
    p_set.add_argument("--qty", type=float, default=None, help="Share quantity (optional; defaults to full cash on buy)")
    p_set.add_argument("--cash", type=float, default=None, help="Override remaining cash balance ($)")
    p_set.add_argument("--note", default="manual CLI update", help="Optional note")

    p_add = sub.add_parser("add-symbol", help="Add a new stock ticker to live_config.json")
    p_add.add_argument("symbol", help="Ticker symbol to add (e.g. NVDA)")

    p_mode = sub.add_parser("set-mode", help="Switch execution mode (signal_only, paper, live)")
    p_mode.add_argument("mode", choices=["signal_only", "paper", "live"])

    args = parser.parse_args()

    if args.cmd == "status":
        cmd_status()
    elif args.cmd == "tick":
        for r in run_once(check_retrain=True):
            print(r)
        cmd_status()
    elif args.cmd == "run":
        run_loop()
    elif args.cmd == "retrain":
        syms = [args.symbol.upper()] if args.symbol else None
        retrain_symbols(symbols=syms, force=args.force or True)
        cmd_status()
    elif args.cmd == "set-wallet":
        cfg = load_config()
        sym = args.symbol.upper()
        if sym not in cfg.symbols:
            cfg.symbols.append(sym)
            save_config(cfg)
        portfolio = load_portfolio(cfg.symbols, initial_cash=cfg.initial_cash_per_symbol)
        msg = set_manual_wallet_state(
            portfolio[sym],
            holding=(args.holding == "yes"),
            price=args.price,
            qty=args.qty,
            cash=args.cash,
            note=args.note,
        )
        save_portfolio(portfolio)
        print(msg)
        cmd_status()
    elif args.cmd == "add-symbol":
        cfg = load_config()
        sym = args.symbol.upper()
        if sym not in cfg.symbols:
            cfg.symbols.append(sym)
            save_config(cfg)
        load_portfolio(cfg.symbols, initial_cash=cfg.initial_cash_per_symbol)
        print(f"Added {sym}. Configured symbols: {cfg.symbols}")
    elif args.cmd == "set-mode":
        cfg = load_config()
        cfg.mode = args.mode
        save_config(cfg)
        print(f"Updated execution mode to: {cfg.mode}")


if __name__ == "__main__":
    main()
