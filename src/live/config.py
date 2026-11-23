"""Configuration for the live 15-min IEX Long/Cash Peak & Trough trading system."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "live_config.json"
STATE_DIR = PROJECT_ROOT / "state"
STATE_PATH = STATE_DIR / "portfolio.json"


@dataclass
class LiveConfig:
    """Top-level live trading configuration (multi-stock ready, Long/Cash only, IEX feed)."""

    symbols: list[str] = field(default_factory=lambda: ["TSLA"])
    mode: str = "signal_only"  # "signal_only", "paper", or "live"
    initial_cash_per_symbol: float = 10_000.0
    retrain_interval_days: int = 28  # 4-week rolling WFO cadence
    train_trading_days: int = 120  # ~6 months rolling training window
    cost_bps_per_side: float = 1.5
    notify_macos: bool = True
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8000


def load_config(path: Path = CONFIG_PATH) -> LiveConfig:
    """Load LiveConfig from live_config.json, creating defaults if missing."""
    if not path.exists():
        cfg = LiveConfig()
        save_config(cfg, path)
        return cfg
    raw = json.loads(path.read_text(encoding="utf-8"))
    cfg = LiveConfig(**{k: v for k, v in raw.items() if k in LiveConfig.__dataclass_fields__})
    cfg.symbols = [s.upper() for s in cfg.symbols]
    return cfg


def save_config(cfg: LiveConfig, path: Path = CONFIG_PATH) -> None:
    """Persist LiveConfig to JSON."""
    path.write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")
