"""Local FastAPI + Plotly live dashboard for the 15-min IEX Peak & Trough (Long/Cash) strategy."""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.features import _causal_ema
from src.live.broker import fetch_iex_15m_bars
from src.live.config import load_config, save_config
from src.live.retrain import retrain_symbols
from src.live.runner import run_once
from src.live.state import load_portfolio, save_portfolio, set_manual_wallet_state

app = FastAPI(title="Peak & Trough Live IEX Dashboard")


class WalletUpdateReq(BaseModel):
    holding: bool
    price: float
    qty: Optional[float] = None
    cash: Optional[float] = None
    note: str = "manual dashboard update"


class AddSymbolReq(BaseModel):
    symbol: str


@app.get("/api/config")
def api_config() -> dict:
    cfg = load_config()
    return asdict(cfg)


@app.get("/api/state/{symbol}")
def api_symbol_state(symbol: str) -> dict:
    cfg = load_config()
    sym = symbol.upper()
    portfolio = load_portfolio(cfg.symbols, initial_cash=cfg.initial_cash_per_symbol)
    if sym not in portfolio:
        raise HTTPException(status_code=404, detail=f"Symbol {sym} not configured.")

    st = portfolio[sym]
    df = fetch_iex_15m_bars(sym, lookback_days=14)

    bars = {"ts": [], "open": [], "high": [], "low": [], "close": [], "smooth": []}
    live_price = st.last_price
    if not df.empty:
        closes = df["close"].to_numpy(dtype=np.float64)
        sw = max(1, int(st.active_params.smooth_window))
        smooth = _causal_ema(closes, sw) if sw > 1 else closes
        bars = {
            "ts": [t.strftime("%Y-%m-%d %H:%M") for t in df.index],
            "open": np.round(df["open"].to_numpy(), 4).tolist(),
            "high": np.round(df["high"].to_numpy(), 4).tolist(),
            "low": np.round(df["low"].to_numpy(), 4).tolist(),
            "close": np.round(closes, 4).tolist(),
            "smooth": np.round(smooth, 4).tolist(),
        }
        live_price = float(closes[-1])

    eq_now = st.current_equity(live_price)
    strat_ret_pct = (eq_now / st.initial_cash - 1.0) * 100.0 if st.initial_cash > 0 else 0.0
    bh_ret_pct = (
        (live_price / st.benchmark_start_price - 1.0) * 100.0
        if (st.benchmark_start_price > 0 and live_price > 0)
        else 0.0
    )
    unreal_pl = (live_price - st.avg_entry_price) * st.qty if (st.holding and st.avg_entry_price > 0) else 0.0
    unreal_pct = ((live_price / st.avg_entry_price - 1.0) * 100.0) if (st.holding and st.avg_entry_price > 0) else 0.0

    sell_trades = [t for t in st.trades if t.action == "SELL"]
    win_rate = (
        float(np.mean([t.pnl_dollars > 0 for t in sell_trades]) * 100.0) if sell_trades else 0.0
    )

    trig_side, trig_px = st.next_trigger_price()

    return {
        "config": asdict(cfg),
        "symbol": sym,
        "live_price": round(live_price, 4),
        "holding": st.holding,
        "qty": round(st.qty, 4),
        "avg_entry_price": round(st.avg_entry_price, 4),
        "cash": round(st.cash, 2),
        "initial_cash": round(st.initial_cash, 2),
        "equity": round(eq_now, 2),
        "strat_ret_pct": round(strat_ret_pct, 2),
        "bh_ret_pct": round(bh_ret_pct, 2),
        "unrealized_pl": round(unreal_pl, 2),
        "unrealized_pct": round(unreal_pct, 2),
        "running_low": round(st.running_low, 4),
        "running_high": round(st.running_high, 4),
        "next_trigger_side": trig_side,
        "next_trigger_price": round(trig_px, 2),
        "last_bar_time": st.last_bar_time,
        "last_retrain_at": st.last_retrain_at,
        "active_params": asdict(st.active_params),
        "pending_params": asdict(st.pending_params) if st.pending_params else None,
        "win_rate_pct": round(win_rate, 1),
        "closed_trades": len(sell_trades),
        "trades": [asdict(t) for t in reversed(st.trades[-50:])],
        "equity_curve": [asdict(e) for e in st.equity_curve],
        "bars": bars,
    }


@app.post("/api/tick/{symbol}")
def api_tick(symbol: str) -> dict:
    res = run_once(symbols=[symbol.upper()], check_retrain=False)
    return {"ok": True, "result": res}


@app.post("/api/retrain/{symbol}")
def api_retrain(symbol: str) -> dict:
    outcomes = retrain_symbols(symbols=[symbol.upper()], force=True)
    return {"ok": True, "outcomes": outcomes}


@app.post("/api/wallet/{symbol}")
def api_wallet_update(symbol: str, req: WalletUpdateReq) -> dict:
    cfg = load_config()
    sym = symbol.upper()
    portfolio = load_portfolio(cfg.symbols, initial_cash=cfg.initial_cash_per_symbol)
    if sym not in portfolio:
        raise HTTPException(status_code=404, detail=f"Symbol {sym} not configured.")
    msg = set_manual_wallet_state(
        portfolio[sym],
        holding=req.holding,
        price=req.price,
        qty=req.qty,
        cash=req.cash,
        note=req.note,
    )
    save_portfolio(portfolio)
    return {"ok": True, "message": msg}


@app.post("/api/symbols")
def api_add_symbol(req: AddSymbolReq) -> dict:
    cfg = load_config()
    sym = req.symbol.strip().upper()
    if not sym or not sym.isalpha():
        raise HTTPException(status_code=400, detail="Invalid ticker symbol.")
    if sym not in cfg.symbols:
        cfg.symbols.append(sym)
        save_config(cfg)
    load_portfolio(cfg.symbols, initial_cash=cfg.initial_cash_per_symbol)
    return {"ok": True, "symbols": cfg.symbols}


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Peak &amp; Trough Live IEX Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root { --bg:#0d1017; --card:#161b26; --border:#273043; --text:#e8edf7; --muted:#95a2ba; --green:#22c55e; --red:#ef4444; --blue:#3b82f6; --amber:#f59e0b; }
    * { box-sizing: border-box; }
    body { margin:0; padding:18px 22px; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
    .topbar { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:12px; margin-bottom:16px; }
    .title { font-size:20px; font-weight:700; display:flex; align-items:center; gap:10px; }
    .badge { font-size:11px; padding:3px 8px; border-radius:999px; background:#1e293b; border:1px solid #334155; color:#93c5fd; text-transform:uppercase; letter-spacing:.04em; }
    .actions { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
    select, input, button { background:#1d2433; color:var(--text); border:1px solid #33405a; border-radius:7px; padding:6px 10px; font-size:13px; }
    button { cursor:pointer; font-weight:600; }
    button:hover { background:#273147; }
    button.primary { background:#2563eb; border-color:#3b82f6; }
    button.primary:hover { background:#1d4ed8; }
    .kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(185px, 1fr)); gap:12px; margin-bottom:14px; }
    .kpi { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:12px 14px; }
    .kpi .lbl { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }
    .kpi .val { font-size:20px; font-weight:700; }
    .kpi .sub { font-size:12px; color:var(--muted); margin-top:3px; }
    .pos { color:var(--green); } .neg { color:var(--red); } .warn { color:var(--amber); }
    .grid { display:grid; grid-template-columns:2.4fr 1fr; gap:14px; }
    @media (max-width: 1050px) { .grid { grid-template-columns:1fr; } }
    .card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:14px; margin-bottom:14px; }
    .card h3 { margin:0 0 10px 0; font-size:14px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
    #price-chart { width:100%; height:440px; }
    #equity-chart { width:100%; height:240px; }
    .form-row { display:flex; flex-direction:column; gap:4px; margin-bottom:10px; }
    .form-row label { font-size:12px; color:var(--muted); }
    table { width:100%; border-collapse:collapse; font-size:12.5px; }
    th, td { text-align:left; padding:7px 8px; border-bottom:1px solid #222a3a; }
    th { color:var(--muted); font-weight:600; }
    .toast { font-size:12px; margin-top:8px; color:#93c5fd; min-height:16px; }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="title">
      Peak &amp; Trough Live Dashboard
      <span class="badge" id="badge-feed">IEX REAL-TIME</span>
      <span class="badge" id="badge-mode">SIGNAL_ONLY</span>
      <span class="badge">LONG / CASH</span>
    </div>
    <div class="actions">
      <select id="sym-select"></select>
      <input id="new-sym" placeholder="Add ticker (e.g. NVDA)" style="width:155px" />
      <button id="btn-add-sym">+ Add</button>
      <button class="primary" id="btn-tick">Run 15m Tick</button>
      <button id="btn-retrain">Retrain 4w Params</button>
      <button id="btn-refresh">Refresh</button>
    </div>
  </div>

  <div class="kpis">
    <div class="kpi"><div class="lbl">Wallet Status</div><div class="val" id="k-status">—</div><div class="sub" id="k-status-sub">—</div></div>
    <div class="kpi"><div class="lbl">Total Equity</div><div class="val" id="k-eq">—</div><div class="sub" id="k-eq-sub">—</div></div>
    <div class="kpi"><div class="lbl">Next Signal Trigger</div><div class="val" id="k-trig">—</div><div class="sub" id="k-trig-sub">—</div></div>
    <div class="kpi"><div class="lbl">Active 4w Parameters</div><div class="val" id="k-params">—</div><div class="sub" id="k-params-sub">—</div></div>
    <div class="kpi"><div class="lbl">Closed Trades / Win Rate</div><div class="val" id="k-wr">—</div><div class="sub" id="k-wr-sub">—</div></div>
  </div>

  <div class="grid">
    <div>
      <div class="card">
        <h3>Live 15-Min IEX Price, Trigger Levels &amp; Signals</h3>
        <div id="price-chart"></div>
      </div>
      <div class="card">
        <h3>Live Equity Curve vs. Buy &amp; Hold (%)</h3>
        <div id="equity-chart"></div>
      </div>
    </div>

    <div>
      <div class="card">
        <h3>Manual Wallet / Trade Override</h3>
        <div class="form-row">
          <label>Current Wallet Status (Do you hold stock?)</label>
          <select id="w-holding">
            <option value="false">CASH (Flat / Sold)</option>
            <option value="true">HOLDING (Long / Bought)</option>
          </select>
        </div>
        <div class="form-row">
          <label>Fill / Reference Price ($)</label>
          <input id="w-price" type="number" step="0.01" placeholder="e.g. 350.00" />
        </div>
        <div class="form-row">
          <label>Shares Qty (optional — blank uses full cash on buy)</label>
          <input id="w-qty" type="number" step="0.0001" placeholder="optional" />
        </div>
        <div class="form-row">
          <label>Cash Balance Override $ (optional)</label>
          <input id="w-cash" type="number" step="0.01" placeholder="optional" />
        </div>
        <div class="form-row">
          <label>Note</label>
          <input id="w-note" type="text" value="manual trade" />
        </div>
        <button class="primary" id="btn-wallet" style="width:100%">Update Wallet Status</button>
        <div class="toast" id="toast"></div>
      </div>

      <div class="card">
        <h3>Recent Trades</h3>
        <div style="max-height:360px; overflow-y:auto;">
          <table>
            <thead><tr><th>Time</th><th>Side</th><th>Price</th><th>PnL</th><th>Src</th></tr></thead>
            <tbody id="trades-body"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

<script>
  let currentSym = "TSLA";

  function setToast(msg) {
    document.getElementById("toast").textContent = msg;
  }

  async function loadSymbols() {
    const res = await fetch("/api/config");
    const cfg = await res.json();
    document.getElementById("badge-mode").textContent = cfg.mode.toUpperCase();
    const sel = document.getElementById("sym-select");
    sel.innerHTML = "";
    cfg.symbols.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s; opt.textContent = s;
      if (s === currentSym) opt.selected = true;
      sel.appendChild(opt);
    });
    if (!cfg.symbols.includes(currentSym) && cfg.symbols.length > 0) {
      currentSym = cfg.symbols[0];
      sel.value = currentSym;
    }
  }

  async function loadState() {
    const res = await fetch(`/api/state/${currentSym}`);
    const d = await res.json();

    // KPIs
    const kStatus = document.getElementById("k-status");
    kStatus.textContent = d.holding ? `HOLDING (${d.qty} sh)` : "CASH (FLAT)";
    kStatus.className = "val " + (d.holding ? "pos" : "");
    document.getElementById("k-status-sub").textContent = d.holding
      ? `Avg Entry: $${d.avg_entry_price.toFixed(2)} | Unreal: ${d.unrealized_pl >= 0 ? "+" : ""}$${d.unrealized_pl.toFixed(2)} (${d.unrealized_pct >= 0 ? "+" : ""}${d.unrealized_pct.toFixed(2)}%)`
      : `Available Cash: $${d.cash.toLocaleString()}`;

    const kEq = document.getElementById("k-eq");
    kEq.textContent = `$${d.equity.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})}`;
    kEq.className = "val " + (d.strat_ret_pct >= 0 ? "pos" : "neg");
    document.getElementById("k-eq-sub").textContent = `Strat: ${d.strat_ret_pct >= 0 ? "+" : ""}${d.strat_ret_pct.toFixed(2)}% | ${d.symbol} B&H: ${d.bh_ret_pct >= 0 ? "+" : ""}${d.bh_ret_pct.toFixed(2)}%`;

    const kTrig = document.getElementById("k-trig");
    kTrig.textContent = `${d.next_trigger_side} @ $${d.next_trigger_price.toFixed(2)}`;
    kTrig.className = "val " + (d.next_trigger_side === "BUY" ? "pos" : "warn");
    document.getElementById("k-trig-sub").textContent = d.holding
      ? `Running High: $${d.running_high.toFixed(2)} | Live: $${d.live_price.toFixed(2)}`
      : `Running Low: $${d.running_low.toFixed(2)} | Live: $${d.live_price.toFixed(2)}`;

    const ap = d.active_params;
    document.getElementById("k-params").textContent = `Reb ${(ap.rebound_pct*100).toFixed(1)}% / PB ${(ap.pullback_pct*100).toFixed(1)}%`;
    document.getElementById("k-params-sub").textContent = `Smooth=${ap.smooth_window} | Train Sh=${ap.train_sharpe}` + (d.pending_params ? " | PENDING RETRAIN QUEUED" : "");

    document.getElementById("k-wr").textContent = `${d.closed_trades} / ${d.win_rate_pct.toFixed(1)}%`;
    document.getElementById("k-wr-sub").textContent = `Last bar: ${d.last_bar_time ? d.last_bar_time.slice(0, 16) : "—"}`;

    // Pre-fill wallet form price
    const wp = document.getElementById("w-price");
    if (!wp.value && d.live_price > 0) wp.value = d.live_price.toFixed(2);
    document.getElementById("w-holding").value = d.holding ? "true" : "false";

    // Price Chart
    const x = d.bars.ts.map((_, i) => i);
    const tickVals = [], tickText = [];
    for (let i = 0; i < x.length; i += Math.max(1, Math.floor(x.length / 10))) {
      tickVals.push(i); tickText.push(d.bars.ts[i].slice(5));
    }
    const traces = [
      {
        type: "candlestick",
        x, open: d.bars.open, high: d.bars.high, low: d.bars.low, close: d.bars.close,
        text: d.bars.ts, name: `${d.symbol} 15m IEX`,
        increasing: { line: { color: "#22c55e" } }, decreasing: { line: { color: "#ef4444" } },
      }
    ];
    if (ap.smooth_window > 1) {
      traces.push({
        type: "scatter", mode: "lines", x, y: d.bars.smooth, text: d.bars.ts,
        name: `EMA(${ap.smooth_window})`, line: { color: "#60a5fa", width: 1.8 }
      });
    }

    const shapes = [];
    const annotations = [];
    if (d.next_trigger_price > 0 && x.length > 0) {
      const col = d.next_trigger_side === "BUY" ? "#22c55e" : "#f59e0b";
      shapes.push({ type:"line", x0:0, x1:x.length-1, y0:d.next_trigger_price, y1:d.next_trigger_price, line:{color:col, width:1.6, dash:"dash"} });
      annotations.push({ x:x.length-1, y:d.next_trigger_price, xanchor:"right", yanchor:"bottom", text:`Next ${d.next_trigger_side} Trigger: $${d.next_trigger_price.toFixed(2)}`, font:{color:col, size:11}, showarrow:false });
    }
    if (d.holding && d.avg_entry_price > 0 && x.length > 0) {
      shapes.push({ type:"line", x0:0, x1:x.length-1, y0:d.avg_entry_price, y1:d.avg_entry_price, line:{color:"#93c5fd", width:1.2, dash:"dot"} });
      annotations.push({ x:0, y:d.avg_entry_price, xanchor:"left", yanchor:"bottom", text:`Entry: $${d.avg_entry_price.toFixed(2)}`, font:{color:"#93c5fd", size:11}, showarrow:false });
    }

    Plotly.react("price-chart", traces, {
      paper_bgcolor:"#161b26", plot_bgcolor:"#10141d", font:{color:"#e8edf7", size:11},
      margin:{l:56, r:20, t:16, b:46}, shapes, annotations,
      xaxis:{ rangeslider:{visible:false}, tickmode:"array", tickvals:tickVals, ticktext:tickText, gridcolor:"#1f2738" },
      yaxis:{ title:"Price ($)", gridcolor:"#1f2738" },
      legend:{ orientation:"h", y:1.06 }
    }, {responsive:true, displaylogo:false});

    // Equity Chart
    const eqX = d.equity_curve.map((p) => p.timestamp.slice(5, 16));
    const eqY = d.equity_curve.map((p) => ((p.equity / d.initial_cash) - 1) * 100);
    const bhBase = d.equity_curve.length > 0 ? d.equity_curve[0].price : 1;
    const bhY = d.equity_curve.map((p) => ((p.price / bhBase) - 1) * 100);
    Plotly.react("equity-chart", [
      { type:"scatter", mode:"lines", x:eqX, y:eqY, name:"Strategy (%)", line:{color:"#3b82f6", width:2} },
      { type:"scatter", mode:"lines", x:eqX, y:bhY, name:`${d.symbol} B&H (%)`, line:{color:"#94a3b8", width:1.4, dash:"dash"} }
    ], {
      paper_bgcolor:"#161b26", plot_bgcolor:"#10141d", font:{color:"#e8edf7", size:11},
      margin:{l:52, r:16, t:12, b:40},
      xaxis:{ gridcolor:"#1f2738" }, yaxis:{ title:"Return (%)", gridcolor:"#1f2738" },
      legend:{ orientation:"h", y:1.12 }
    }, {responsive:true, displaylogo:false});

    // Trades Table
    const tb = document.getElementById("trades-body");
    tb.innerHTML = d.trades.length === 0
      ? '<tr><td colspan="5" style="color:#95a2ba">No trades logged yet.</td></tr>'
      : d.trades.map((t) => `
          <tr>
            <td>${t.timestamp.slice(5, 16)}</td>
            <td style="color:${t.action==="BUY" ? "#22c55e" : "#f59e0b"}; font-weight:600">${t.action}</td>
            <td>$${Number(t.price).toFixed(2)}</td>
            <td class="${t.pnl_dollars > 0 ? "pos" : (t.pnl_dollars < 0 ? "neg" : "")}">${t.action==="SELL" ? `${t.pnl_pct >= 0 ? "+" : ""}${t.pnl_pct.toFixed(2)}%` : "—"}</td>
            <td>${t.source}</td>
          </tr>`).join("");
  }

  document.getElementById("sym-select").addEventListener("change", (e) => { currentSym = e.target.value; document.getElementById("w-price").value = ""; loadState(); });
  document.getElementById("btn-refresh").addEventListener("click", loadState);
  document.getElementById("btn-tick").addEventListener("click", async () => {
    setToast("Running 15-min IEX tick...");
    await fetch(`/api/tick/${currentSym}`, { method:"POST" });
    setToast("Tick complete.");
    loadState();
  });
  document.getElementById("btn-retrain").addEventListener("click", async () => {
    setToast("Running 4-week IEX WFO retrain (takes ~10s)...");
    const r = await (await fetch(`/api/retrain/${currentSym}`, { method:"POST" })).json();
    setToast("Retrain: " + JSON.stringify(r.outcomes));
    loadState();
  });
  document.getElementById("btn-wallet").addEventListener("click", async () => {
    const body = {
      holding: document.getElementById("w-holding").value === "true",
      price: parseFloat(document.getElementById("w-price").value),
      qty: document.getElementById("w-qty").value ? parseFloat(document.getElementById("w-qty").value) : null,
      cash: document.getElementById("w-cash").value ? parseFloat(document.getElementById("w-cash").value) : null,
      note: document.getElementById("w-note").value || "manual dashboard update",
    };
    if (!body.price || body.price <= 0) { setToast("Enter a valid price."); return; }
    const r = await (await fetch(`/api/wallet/${currentSym}`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body) })).json();
    setToast(r.message || "Updated.");
    loadState();
  });
  document.getElementById("btn-add-sym").addEventListener("click", async () => {
    const sym = document.getElementById("new-sym").value.trim().toUpperCase();
    if (!sym) return;
    await fetch("/api/symbols", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({symbol:sym}) });
    document.getElementById("new-sym").value = "";
    currentSym = sym;
    await loadSymbols();
    await loadState();
    setToast(`Added ${sym}. Click 'Retrain 4w Params' to initialize its parameters.`);
  });

  (async () => { await loadSymbols(); await loadState(); setInterval(loadState, 60000); })();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return DASHBOARD_HTML


def main() -> None:
    cfg = load_config()
    print(f"Starting Live IEX Dashboard at http://{cfg.dashboard_host}:{cfg.dashboard_port}")
    uvicorn.run(app, host=cfg.dashboard_host, port=cfg.dashboard_port, log_level="info")


if __name__ == "__main__":
    main()
