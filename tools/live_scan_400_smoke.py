from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from evidence_governance import apply_three_rank_contract
from fast_scan_engine import run_fast_single_scan

UNIVERSE_PATH = Path(__file__).with_name("idx_400_production_universe_2026-07.txt")
EXPECTED_UNIVERSE_SHA256 = "804af9cc17ac2df973c39d22eeac3bacc8388d91ad2f67663928264d4a027120"


def _load_canonical_universe() -> list[str]:
    tokens: list[str] = []
    for line in UNIVERSE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens.extend(part.strip().upper() for part in line.split(",") if part.strip())
    tickers = list(dict.fromkeys(tokens))
    digest = hashlib.sha256("\n".join(tickers).encode("utf-8")).hexdigest()
    if len(tickers) != 400:
        raise AssertionError(f"Canonical production universe must contain exactly 400 unique tickers, got {len(tickers)}")
    if digest != EXPECTED_UNIVERSE_SHA256:
        raise AssertionError(f"Canonical production universe hash changed: {digest}")
    return tickers


TICKERS = _load_canonical_universe()

CONFIG = {
    "period": "3y",
    "evidence_refresh_cap": 20,
    "decision_evidence_cap": 12,
    "evidence_fundamental_cap": 20,
    "evidence_official_cap": 12,
    "evidence_snapshot_cap": 16,
    "evidence_market_cap": 6,
    "evidence_news_cap": 10,
    "execution_verification_cap": 8,
    "daily_market_refresh_limit": 6,
    "macro_external_enabled": True,
    "macro_timeout_seconds": 3,
    "lean_persistence": True,
    "lean_skip_narrative_history": True,
}


def _save(frame: pd.DataFrame, path: str) -> int:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        pd.DataFrame().to_csv(path, index=False)
        return 0
    frame.to_csv(path, index=False)
    return len(frame)


def _three_rank_leaders(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame
    raw = "raw_ranking_score" if "raw_ranking_score" in frame.columns else "research_score"
    return apply_three_rank_contract(
        frame,
        raw_score_col=raw,
        guarded_score_col="ranking_score",
        production_score_col="v9_next_leader_score",
        research_eligible_col="rank_eligible" if "rank_eligible" in frame.columns else None,
        guarded_eligible_col="rank_eligible" if "rank_eligible" in frame.columns else None,
        production_eligible_cols=("real_money_authorization_pass",),
    )


def _three_rank_swings(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame
    return apply_three_rank_contract(
        frame,
        raw_score_col="research_score",
        guarded_score_col="ranking_score",
        production_score_col="v9_swing_score",
        research_eligible_col="rank_eligible" if "rank_eligible" in frame.columns else None,
        guarded_eligible_col="rank_eligible" if "rank_eligible" in frame.columns else None,
        production_eligible_cols=("real_money_authorization_pass",),
    )


result = run_fast_single_scan(TICKERS, config=CONFIG, runtime={})
focus = result.get("focus_screens", {}) or {}
leaders = _three_rank_leaders(focus.get("next_leaders", pd.DataFrame()))
swings = _three_rank_swings(focus.get("swing_ready", pd.DataFrame()))
Path("live_scan_output").mkdir(exist_ok=True)
leader_n = _save(leaders, "live_scan_output/next_leaders.csv")
swing_n = _save(swings, "live_scan_output/swing_ready.csv")
coverage = result.get("scan_coverage_summary", pd.DataFrame())
_save(coverage, "live_scan_output/coverage.csv")

summary = {
    "scanner_version": result.get("scanner_version"),
    "universe_source": str(UNIVERSE_PATH),
    "universe_sha256": EXPECTED_UNIVERSE_SHA256,
    "universe_tickers": len(TICKERS),
    "scan_elapsed_seconds": result.get("scan_elapsed_seconds"),
    "database_transport_state": result.get("database_transport_state"),
    "feature_cache_hits": result.get("feature_cache_hits"),
    "feature_cache_refreshes": result.get("feature_cache_refreshes"),
    "leader_rows": leader_n,
    "swing_rows": swing_n,
    "production_real_money_leaders": int(pd.to_numeric(leaders.get("production_real_money_rank"), errors="coerce").notna().sum()) if not leaders.empty else 0,
    "production_real_money_swings": int(pd.to_numeric(swings.get("production_real_money_rank"), errors="coerce").notna().sum()) if not swings.empty else 0,
}
Path("live_scan_output/summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
print(json.dumps(summary, indent=2, default=str))
if isinstance(leaders, pd.DataFrame) and not leaders.empty:
    cols = [c for c in [
        "ticker", "raw_research_rank", "guarded_decision_priority_rank", "production_real_money_rank",
        "raw_research_score", "guarded_decision_priority_score", "production_real_money_score",
        "real_money_authorization_state", "entry_low", "entry_high", "trigger", "stop_loss", "tp1", "tp2",
    ] if c in leaders.columns]
    print("NEXT_LEADERS_TOP20")
    print(leaders.sort_values("guarded_decision_priority_rank", na_position="last").loc[:, cols].head(20).to_csv(index=False))
if isinstance(swings, pd.DataFrame) and not swings.empty:
    cols = [c for c in [
        "ticker", "raw_research_rank", "guarded_decision_priority_rank", "production_real_money_rank",
        "raw_research_score", "guarded_decision_priority_score", "production_real_money_score",
        "real_money_authorization_state", "entry_low", "entry_high", "trigger_price", "stop_loss", "tp1", "tp2",
    ] if c in swings.columns]
    print("SWING_TOP20")
    print(swings.sort_values("guarded_decision_priority_rank", na_position="last").loc[:, cols].head(20).to_csv(index=False))
