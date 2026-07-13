from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MarketContext:
    regime: str = "UNKNOWN"
    benchmark_close: float | None = None
    benchmark_roc20: float | None = None
    breadth_ema50: float | None = None
    breadth_ema200: float | None = None
    reason: str = "Benchmark tidak tersedia"


@dataclass
class SetupPlan:
    ticker: str
    setup: str
    detected: bool
    setup_score: float
    signal_date: Any = None
    zone_created_date: Any = None
    entry_low: float | None = None
    entry_high: float | None = None
    entry: float | None = None
    entry_type: str = "CONDITIONAL"
    trigger: float | None = None
    stop_loss: float | None = None
    tp1: float | None = None
    tp2: float | None = None
    rr1: float | None = None
    rr2: float | None = None
    distance_atr: float | None = None
    zone_age_bars: int | None = None
    valid_until: Any = None
    invalidated: bool = False
    action: str = "NO_SETUP"
    reason: str = ""
    evidence: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["evidence"] = " • ".join(self.evidence)
        result["blockers"] = " • ".join(self.blockers)
        return result


@dataclass
class DownloadReport:
    requested: list[str]
    downloaded: list[str]
    failed: dict[str, str]
    benchmark_ok: bool = False
