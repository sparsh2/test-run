"""Generate an interactive HTML explorer comparing raw 1-min OHLCV against compute_kinematic_features()."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import FeatureConfig, compute_kinematic_features
from src.fetch_data import data_path_for_symbol, load_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def build_html(
    symbol: str,
    sub: pd.DataFrame,
    window: int,
    out_path: Path,
) -> None:
    smoothers = [
        ("ema", "EMA", "#1f77b4", 3),
        ("dema", "DEMA", "#2ca02c", 3),
        ("savgol", "SavGol", "#17becf", 3),
        ("linreg", "LinReg", "#ff7f0e", 3),
        ("kalman", "Kalman", "#d62728", 1),
    ]

    timestamps = [ts.strftime("%Y-%m-%d %H:%M") for ts in sub.index]
    dates = sorted(pd.unique(sub.index.date))
    day_splits = []
    for d in dates[1:]:
        idx = int(np.flatnonzero(sub.index.date == d)[0])
        day_splits.append({"idx": idx, "label": str(d)})

    log_close = np.log(sub["close"].to_numpy(dtype=np.float64))
    raw_vel = np.zeros(len(sub), dtype=np.float64)
    raw_vel[1:] = (log_close[1:] - log_close[:-1]) * 10_000.0
    for sp in day_splits:
        raw_vel[sp["idx"]] = 0.0
    raw_acc = np.zeros(len(sub), dtype=np.float64)
    raw_acc[1:] = raw_vel[1:] - raw_vel[:-1]
    for sp in day_splits:
        raw_acc[sp["idx"]] = 0.0

    raw_data = {
        "x": list(range(len(sub))),
        "ts": timestamps,
        "open": np.round(sub["open"].to_numpy(), 4).tolist(),
        "high": np.round(sub["high"].to_numpy(), 4).tolist(),
        "low": np.round(sub["low"].to_numpy(), 4).tolist(),
        "close": np.round(sub["close"].to_numpy(), 4).tolist(),
        "vwap": np.round(sub["vwap"].fillna(sub["close"]).to_numpy(), 4).tolist()
        if "vwap" in sub.columns
        else np.round(sub["close"].to_numpy(), 4).tolist(),
        "volume": sub["volume"].to_numpy().tolist(),
        "raw_velocity": np.round(raw_vel, 3).tolist(),
        "raw_acceleration": np.round(raw_acc, 3).tolist(),
    }

    smoother_data = {}
    vol_features = None
    for key, label, color, vl in smoothers:
        f_cfg = FeatureConfig(smoother=key, smooth_window=window, vel_lookback=vl, reset_daily=True)
        feats = compute_kinematic_features(sub, f_cfg)
        smoother_data[key] = {
            "label": f"{label} (w={window}, vl={vl})",
            "short": label,
            "color": color,
            "smooth_price": np.round(feats["smooth_price"].to_numpy(), 4).tolist(),
            "velocity": np.round(feats["velocity"].to_numpy(), 3).tolist(),
            "acceleration": np.round(feats["acceleration"].to_numpy(), 3).tolist(),
        }
        if vol_features is None:
            vol_features = {
                "atr_bps": np.round(feats["atr_bps"].to_numpy(), 2).tolist(),
                "ret_std_bps": np.round(feats["ret_std_bps"].to_numpy(), 2).tolist(),
                "rel_volume": np.round(feats["rel_volume"].to_numpy(), 2).tolist(),
            }

    payload = {
        "symbol": symbol,
        "date_range": f"{dates[0]} to {dates[-1]}",
        "window": window,
        "day_splits": day_splits,
        "raw": raw_data,
        "smoothers": smoother_data,
        "vol": vol_features,
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{symbol} Kinematic Features Explorer ({dates[0]} to {dates[-1]})</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #0f1117;
      --panel: #181c25;
      --border: #2a3040;
      --text: #e6eaf2;
      --muted: #9aa4b8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 16px 20px;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    h1 {{
      margin: 0 0 6px 0;
      font-size: 20px;
      font-weight: 600;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 14px;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      padding: 12px 14px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      margin-bottom: 14px;
      align-items: center;
    }}
    .group {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      padding-right: 14px;
      border-right: 1px solid var(--border);
    }}
    .group:last-child {{ border-right: none; padding-right: 0; }}
    .group-title {{
      font-size: 12px;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    label.chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      cursor: pointer;
      padding: 4px 9px;
      border-radius: 6px;
      background: #222736;
      border: 1px solid #32394c;
      user-select: none;
    }}
    label.chip:hover {{ border-color: #55607e; }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }}
    button {{
      background: #283044;
      color: var(--text);
      border: 1px solid #3b455e;
      border-radius: 6px;
      padding: 5px 10px;
      font-size: 12px;
      cursor: pointer;
    }}
    button:hover {{ background: #34405c; }}
    #chart {{
      width: 100%;
      height: 82vh;
      min-height: 680px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
    }}
  </style>
</head>
<body>
  <h1>{symbol} — <code>compute_kinematic_features()</code> Interactive 2-Day Explorer</h1>
  <div class="subtitle">
    RTH 1-minute bars ({dates[0]} to {dates[-1]}, {len(sub)} bars, smooth_window={window}, reset_daily=True).
    Drag to zoom, double-click to reset, or toggle any line below.
  </div>

  <div class="controls">
    <div class="group">
      <span class="group-title">Raw Input</span>
      <label class="chip"><input type="checkbox" id="chk-close" checked /><span class="swatch" style="background:#ffffff"></span>Raw Close</label>
      <label class="chip"><input type="checkbox" id="chk-candles" /><span class="swatch" style="background:#26a69a"></span>Candles (OHLC)</label>
      <label class="chip"><input type="checkbox" id="chk-vwap" /><span class="swatch" style="background:#e377c2"></span>Bar VWAP</label>
      <label class="chip"><input type="checkbox" id="chk-raw-deriv" /><span class="swatch" style="background:#b0b8c9"></span>Raw Vel &amp; Accel</label>
    </div>

    <div class="group">
      <span class="group-title">Smoothers (Price + Vel + Accel)</span>
      <label class="chip"><input type="checkbox" class="chk-sm" data-key="ema" checked /><span class="swatch" style="background:#1f77b4"></span>EMA</label>
      <label class="chip"><input type="checkbox" class="chk-sm" data-key="dema" /><span class="swatch" style="background:#2ca02c"></span>DEMA</label>
      <label class="chip"><input type="checkbox" class="chk-sm" data-key="savgol" /><span class="swatch" style="background:#17becf"></span>SavGol</label>
      <label class="chip"><input type="checkbox" class="chk-sm" data-key="linreg" /><span class="swatch" style="background:#ff7f0e"></span>LinReg</label>
      <label class="chip"><input type="checkbox" class="chk-sm" data-key="kalman" checked /><span class="swatch" style="background:#d62728"></span>Kalman</label>
    </div>

    <div class="group">
      <span class="group-title">Vol &amp; Mass</span>
      <label class="chip"><input type="checkbox" id="chk-atr" checked /><span class="swatch" style="background:#2ca02c"></span>ATR (bps)</label>
      <label class="chip"><input type="checkbox" id="chk-std" checked /><span class="swatch" style="background:#ff7f0e"></span>Ret Std (bps)</label>
      <label class="chip"><input type="checkbox" id="chk-rvol" checked /><span class="swatch" style="background:#8c96ab"></span>Rel Volume</label>
    </div>

    <div class="group">
      <button id="btn-all">Select All</button>
      <button id="btn-minimal">Minimal (Close + EMA)</button>
      <button id="btn-reset-zoom">Reset Zoom</button>
    </div>
  </div>

  <div id="chart"></div>

  <script>
    const DATA = {json.dumps(payload)};
    const X = DATA.raw.x;
    const TS = DATA.raw.ts;

    const tickVals = [];
    const tickText = [];
    for (let i = 0; i < X.length; i += 45) {{
      tickVals.push(i);
      tickText.push(TS[i].slice(5));
    }}

    function render() {{
      const traces = [];

      if (document.getElementById("chk-candles").checked) {{
        traces.push({{
          type: "candlestick",
          x: X,
          open: DATA.raw.open,
          high: DATA.raw.high,
          low: DATA.raw.low,
          close: DATA.raw.close,
          text: TS,
          name: "OHLC Candles",
          increasing: {{ line: {{ color: "#26a69a" }} }},
          decreasing: {{ line: {{ color: "#ef5350" }} }},
          xaxis: "x",
          yaxis: "y",
        }});
      }}
      if (document.getElementById("chk-close").checked) {{
        traces.push({{
          type: "scatter",
          mode: "lines",
          x: X,
          y: DATA.raw.close,
          customdata: TS,
          hovertemplate: "%{{customdata}}<br>Close: $%{{y:.2f}}<extra></extra>",
          name: "Raw 1-min Close",
          line: {{ color: "#ffffff", width: 1.4 }},
          xaxis: "x",
          yaxis: "y",
        }});
      }}
      if (document.getElementById("chk-vwap").checked) {{
        traces.push({{
          type: "scatter",
          mode: "lines",
          x: X,
          y: DATA.raw.vwap,
          customdata: TS,
          hovertemplate: "%{{customdata}}<br>VWAP: $%{{y:.2f}}<extra></extra>",
          name: "Bar VWAP",
          line: {{ color: "#e377c2", width: 1.2, dash: "dot" }},
          xaxis: "x",
          yaxis: "y",
        }});
      }}
      if (document.getElementById("chk-raw-deriv").checked) {{
        traces.push({{
          type: "scatter",
          mode: "lines",
          x: X,
          y: DATA.raw.raw_velocity,
          customdata: TS,
          hovertemplate: "%{{customdata}}<br>Raw Vel: %{{y:.2f}} bps/bar<extra></extra>",
          name: "Raw Velocity (1m ret)",
          line: {{ color: "rgba(210, 218, 232, 0.55)", width: 1.0 }},
          xaxis: "x",
          yaxis: "y2",
        }});
        traces.push({{
          type: "scatter",
          mode: "lines",
          x: X,
          y: DATA.raw.raw_acceleration,
          customdata: TS,
          hovertemplate: "%{{customdata}}<br>Raw Accel: %{{y:.2f}} bps/bar²<extra></extra>",
          name: "Raw Accel",
          line: {{ color: "rgba(210, 218, 232, 0.45)", width: 1.0 }},
          xaxis: "x",
          yaxis: "y3",
        }});
      }}

      document.querySelectorAll(".chk-sm").forEach((cb) => {{
        if (!cb.checked) return;
        const sm = DATA.smoothers[cb.dataset.key];
        traces.push({{
          type: "scatter",
          mode: "lines",
          x: X,
          y: sm.smooth_price,
          customdata: TS,
          hovertemplate: `%{{customdata}}<br>${{sm.short}} Price: $%{{y:.2f}}<extra></extra>`,
          name: `${{sm.label}} Price`,
          line: {{ color: sm.color, width: 1.8 }},
          xaxis: "x",
          yaxis: "y",
        }});
        traces.push({{
          type: "scatter",
          mode: "lines",
          x: X,
          y: sm.velocity,
          customdata: TS,
          hovertemplate: `%{{customdata}}<br>${{sm.short}} Vel: %{{y:.2f}} bps/bar<extra></extra>`,
          name: `${{sm.short}} Velocity`,
          line: {{ color: sm.color, width: 1.4 }},
          xaxis: "x",
          yaxis: "y2",
        }});
        traces.push({{
          type: "scatter",
          mode: "lines",
          x: X,
          y: sm.acceleration,
          customdata: TS,
          hovertemplate: `%{{customdata}}<br>${{sm.short}} Accel: %{{y:.2f}} bps/bar²<extra></extra>`,
          name: `${{sm.short}} Accel`,
          line: {{ color: sm.color, width: 1.2 }},
          xaxis: "x",
          yaxis: "y3",
        }});
      }});

      if (document.getElementById("chk-atr").checked) {{
        traces.push({{
          type: "scatter",
          mode: "lines",
          x: X,
          y: DATA.vol.atr_bps,
          customdata: TS,
          hovertemplate: "%{{customdata}}<br>ATR: %{{y:.2f}} bps<extra></extra>",
          name: "ATR (bps)",
          line: {{ color: "#2ca02c", width: 1.4 }},
          xaxis: "x",
          yaxis: "y4",
        }});
      }}
      if (document.getElementById("chk-std").checked) {{
        traces.push({{
          type: "scatter",
          mode: "lines",
          x: X,
          y: DATA.vol.ret_std_bps,
          customdata: TS,
          hovertemplate: "%{{customdata}}<br>Ret Std: %{{y:.2f}} bps<extra></extra>",
          name: "Ret Std (bps)",
          line: {{ color: "#ff7f0e", width: 1.3, dash: "dash" }},
          xaxis: "x",
          yaxis: "y4",
        }});
      }}
      if (document.getElementById("chk-rvol").checked) {{
        traces.push({{
          type: "bar",
          x: X,
          y: DATA.vol.rel_volume,
          customdata: TS,
          hovertemplate: "%{{customdata}}<br>Rel Vol: %{{y:.2f}}x<extra></extra>",
          name: "Rel Volume (x)",
          marker: {{ color: "rgba(140, 150, 171, 0.35)" }},
          xaxis: "x",
          yaxis: "y5",
        }});
      }}

      const shapes = [];
      const annotations = [];
      DATA.day_splits.forEach((sp) => {{
        shapes.push({{
          type: "line",
          x0: sp.idx, x1: sp.idx,
          y0: 0, y1: 1,
          xref: "x", yref: "paper",
          line: {{ color: "#9aa4b8", width: 1.2, dash: "dash" }},
        }});
        annotations.push({{
          x: sp.idx + 4,
          y: 0.99,
          xref: "x", yref: "paper",
          text: `Day Open: ${{sp.label}} (reset_daily)`,
          showarrow: false,
          xanchor: "left",
          font: {{ color: "#e6eaf2", size: 11 }},
        }});
      }});

      shapes.push(
        {{ type: "line", x0: 0, x1: X.length - 1, y0: 0, y1: 0, xref: "x", yref: "y2", line: {{ color: "#55607e", width: 1, dash: "dot" }} }},
        {{ type: "line", x0: 0, x1: X.length - 1, y0: 0, y1: 0, xref: "x", yref: "y3", line: {{ color: "#55607e", width: 1, dash: "dot" }} }}
      );

      const layout = {{
        paper_bgcolor: "#181c25",
        plot_bgcolor: "#131620",
        font: {{ color: "#e6eaf2", size: 12 }},
        margin: {{ l: 62, r: 58, t: 22, b: 56 }},
        hovermode: "x unified",
        legend: {{ orientation: "h", y: 1.03, x: 0, font: {{ size: 11 }} }},
        shapes,
        annotations,
        xaxis: {{
          rangeslider: {{ visible: false }},
          tickmode: "array",
          tickvals: tickVals,
          ticktext: tickText,
          tickangle: -20,
          gridcolor: "#232938",
          title: "Time (ET, RTH 09:30–15:59)",
        }},
        yaxis:  {{ domain: [0.58, 1.00], title: "Price ($)", gridcolor: "#232938" }},
        yaxis2: {{ domain: [0.37, 0.54], title: "Velocity (bps)", gridcolor: "#232938" }},
        yaxis3: {{ domain: [0.18, 0.34], title: "Accel (bps/bar²)", gridcolor: "#232938" }},
        yaxis4: {{ domain: [0.00, 0.15], title: "Vol (bps)", gridcolor: "#232938" }},
        yaxis5: {{
          domain: [0.00, 0.15],
          overlaying: "y4",
          side: "right",
          title: "Rel Vol",
          showgrid: false,
          range: [0, 10],
        }},
      }};

      Plotly.react("chart", traces, layout, {{ responsive: true, displaylogo: false }});
    }}

    document.querySelectorAll("input[type=checkbox]").forEach((el) => el.addEventListener("change", render));
    document.getElementById("btn-all").addEventListener("click", () => {{
      document.querySelectorAll("input[type=checkbox]").forEach((el) => {{ if (el.id !== "chk-candles") el.checked = true; }});
      render();
    }});
    document.getElementById("btn-minimal").addEventListener("click", () => {{
      document.querySelectorAll("input[type=checkbox]").forEach((el) => (el.checked = false));
      document.getElementById("chk-close").checked = true;
      document.querySelector('.chk-sm[data-key="ema"]').checked = true;
      document.getElementById("chk-atr").checked = true;
      render();
    }});
    document.getElementById("btn-reset-zoom").addEventListener("click", () => {{
      Plotly.relayout("chart", {{ "xaxis.autorange": true, "yaxis.autorange": true, "yaxis2.autorange": true, "yaxis3.autorange": true, "yaxis4.autorange": true }});
    }});

    render();
  </script>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved interactive explorer to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build interactive 2-day HTML explorer for compute_kinematic_features.")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--start-date", default="2026-09-09", help="First trading date YYYY-MM-DD (default: 2026-09-09)")
    parser.add_argument("--days", type=int, default=2, help="Number of consecutive trading days (default: 2)")
    parser.add_argument("--window", type=int, default=20, help="Smooth window (default: 20)")
    args = parser.parse_args()

    sym = args.symbol.upper()
    df = load_data(data_path_for_symbol(sym), rth_only=True)
    all_dates = sorted(pd.unique(df.index.date))
    start_d = pd.Timestamp(args.start_date).date()
    chosen = [d for d in all_dates if d >= start_d][: args.days]
    if not chosen:
        chosen = all_dates[-args.days :]

    sub = df.loc[[d in set(chosen) for d in df.index.date]].copy()
    out_path = OUTPUTS_DIR / f"kinematic_features_{sym.lower()}_{args.days}day.html"
    build_html(sym, sub, args.window, out_path)


if __name__ == "__main__":
    main()
