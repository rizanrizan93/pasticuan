from __future__ import annotations

"""Evidence, ranking, runtime-cache and walk-forward integrity primitives.

The module is deliberately fail-closed.  It never upgrades proxy evidence to
DIRECT/OFFICIAL, never activates a calibration without mature OOS observations,
and never conflates research ranking with production authorization.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse
import math
import time

import numpy as np
import pandas as pd

EVIDENCE_GOVERNANCE_VERSION = "1.0.0"


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value or "").strip().upper() in {"1", "TRUE", "YES", "Y", "PASS", "VALID", "VERIFIED", "READY"}


def is_https_url(value: Any) -> bool:
    try:
        parsed = urlparse(str(value or "").strip())
    except ValueError:
        return False
    return parsed.scheme.lower() == "https" and bool(parsed.hostname)


def validate_official_evidence(
    *,
    source_url: Any,
    evidence_date: Any,
    entity_match_verified: Any,
    source_verified: Any,
    source_urls: Sequence[str] | None = None,
    quorum_required: bool = False,
    min_quorum: int = 2,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Validate direct evidence lineage without manufacturing missing fields."""
    urls = [str(url).strip() for url in (source_urls or []) if str(url or "").strip()]
    primary = str(source_url or "").strip()
    if primary and primary not in urls:
        urls.insert(0, primary)
    distinct_https = list(dict.fromkeys(url for url in urls if is_https_url(url)))
    parsed_date = pd.to_datetime(evidence_date, errors="coerce", utc=True)
    now_ts = now if now is not None else pd.Timestamp.now(tz="UTC")
    date_valid = bool(pd.notna(parsed_date) and parsed_date <= now_ts + pd.Timedelta(days=1))
    verified = _truthy(source_verified)
    entity_ok = _truthy(entity_match_verified)
    https_ok = bool(primary and is_https_url(primary))
    quorum_count = len(distinct_https)
    quorum_ok = quorum_count >= max(1, int(min_quorum)) if quorum_required else quorum_count >= 1
    production_valid = bool(verified and https_ok and entity_ok and date_valid and quorum_ok)
    reasons: list[str] = []
    if not verified:
        reasons.append("SOURCE_NOT_VERIFIED")
    if not https_ok:
        reasons.append("HTTPS_SOURCE_MISSING")
    if not entity_ok:
        reasons.append("ENTITY_MATCH_NOT_VERIFIED")
    if not date_valid:
        reasons.append("EVIDENCE_DATE_MISSING_OR_INVALID")
    if not quorum_ok:
        reasons.append("SOURCE_QUORUM_NOT_MET")
    return {
        "evidence_production_valid": production_valid,
        "source_https_verified": https_ok,
        "entity_match_verified": entity_ok,
        "evidence_date_verified": date_valid,
        "source_quorum_count": quorum_count,
        "source_quorum_verified": quorum_ok,
        "evidence_validation_state": "VERIFIED_DIRECT_EVIDENCE" if production_valid else "EVIDENCE_INCOMPLETE_FAIL_CLOSED",
        "evidence_validation_reasons": " | ".join(reasons) or "NONE",
    }


def _rank_from_score(score: pd.Series, eligible: pd.Series) -> pd.Series:
    rank = pd.Series(pd.NA, index=score.index, dtype="Int64")
    work = pd.DataFrame({"score": pd.to_numeric(score, errors="coerce"), "eligible": eligible.fillna(False).astype(bool)}, index=score.index)
    order = work.loc[work["eligible"] & work["score"].notna()].sort_values("score", ascending=False, kind="stable").index
    rank.loc[order] = np.arange(1, len(order) + 1)
    return rank


def apply_three_rank_contract(
    frame: pd.DataFrame,
    *,
    raw_score_col: str,
    guarded_score_col: str,
    production_score_col: str | None = None,
    research_eligible_col: str | None = None,
    guarded_eligible_col: str | None = None,
    production_eligible_cols: Sequence[str] = ("real_money_authorization_pass", "real_money_candidate", "production_ready"),
) -> pd.DataFrame:
    """Expose three non-overlapping ranking contracts on the same universe."""
    if frame is None or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    out = frame.copy()
    raw = pd.to_numeric(out.get(raw_score_col, pd.Series(np.nan, index=out.index)), errors="coerce")
    guarded = pd.to_numeric(out.get(guarded_score_col, raw), errors="coerce")
    prod_col = production_score_col if production_score_col in out.columns else guarded_score_col
    production = pd.to_numeric(out.get(prod_col, guarded), errors="coerce")
    research_eligible = (
        out.get(research_eligible_col, pd.Series(True, index=out.index)).fillna(False).astype(bool)
        if research_eligible_col and research_eligible_col in out.columns else raw.notna()
    )
    guarded_eligible = (
        out.get(guarded_eligible_col, pd.Series(True, index=out.index)).fillna(False).astype(bool)
        if guarded_eligible_col and guarded_eligible_col in out.columns else guarded.notna()
    )
    production_eligible = pd.Series(False, index=out.index)
    found = False
    for column in production_eligible_cols:
        if column in out.columns:
            found = True
            production_eligible |= out[column].map(_truthy)
    if not found and "production_rank_eligible" in out.columns:
        production_eligible = out["production_rank_eligible"].fillna(False).astype(bool)
    production_eligible &= guarded_eligible
    out["raw_research_score"] = raw
    out["guarded_decision_priority_score"] = guarded
    out["production_real_money_score"] = production
    out["raw_research_rank"] = _rank_from_score(raw, research_eligible)
    out["guarded_decision_priority_rank"] = _rank_from_score(guarded, guarded_eligible)
    out["production_real_money_rank"] = _rank_from_score(production, production_eligible)
    out["ranking_contract_state"] = "THREE_RANK_CONTRACT_V1_RAW_GUARDED_PRODUCTION"
    return out


def select_enrichment_shortlist(
    frame: pd.DataFrame,
    *,
    limit: int = 24,
    score_columns: Sequence[str] = ("guarded_decision_priority_score", "ranking_score", "raw_research_score"),
    direct_evidence_columns: Sequence[str] = ("forward_source_quorum_verified", "management_direct_evidence_verified"),
) -> list[str]:
    """Bound expensive enrichment to decision-relevant names and evidence gaps."""
    if frame is None or frame.empty or "ticker" not in frame.columns or limit <= 0:
        return []
    local = frame.copy()
    score = pd.Series(np.nan, index=local.index)
    for column in score_columns:
        if column in local.columns:
            candidate = pd.to_numeric(local[column], errors="coerce")
            score = score.where(score.notna(), candidate)
    score = score.fillna(-1e9)
    missing_direct = pd.Series(0.0, index=local.index)
    for column in direct_evidence_columns:
        if column in local.columns:
            missing_direct += (~local[column].map(_truthy)).astype(float)
    # Missing direct evidence is useful only after a name is research-relevant;
    # it cannot propel a low-quality name into the shortlist by itself.
    percentile = score.rank(pct=True, method="average").fillna(0.0)
    priority = score + 4.0 * missing_direct * percentile
    local = local.assign(_enrichment_priority=priority)
    return local.sort_values(["_enrichment_priority", "ticker"], ascending=[False, True], kind="stable").head(int(limit))["ticker"].astype(str).tolist()


@dataclass
class ProviderNegativeCache:
    """Process-local provider-specific failure cache for expensive enrichment."""
    max_entries: int = 4096
    ttl_seconds: Mapping[str, int] = field(default_factory=lambda: {
        "NOT_FOUND": 86400,
        "AUTH": 86400,
        "PARSE": 21600,
        "RATE_LIMIT": 3600,
        "TIMEOUT": 1800,
        "SERVER": 900,
        "EMPTY": 1800,
        "OTHER": 900,
    })
    _entries: dict[tuple[str, str, str], tuple[float, str]] = field(default_factory=dict)

    def _key(self, provider: Any, family: Any, cache_key: Any) -> tuple[str, str, str]:
        return (str(provider or "UNKNOWN").upper(), str(family or "UNKNOWN").upper(), str(cache_key or "").upper())

    def should_skip(self, provider: Any, family: Any, cache_key: Any) -> bool:
        key = self._key(provider, family, cache_key)
        item = self._entries.get(key)
        if not item:
            return False
        expires_at, _ = item
        if time.monotonic() >= expires_at:
            self._entries.pop(key, None)
            return False
        return True

    def record_failure(self, provider: Any, family: Any, cache_key: Any, failure_class: str = "OTHER") -> None:
        failure = str(failure_class or "OTHER").upper()
        ttl = int(self.ttl_seconds.get(failure, self.ttl_seconds.get("OTHER", 900)))
        self._entries[self._key(provider, family, cache_key)] = (time.monotonic() + max(60, ttl), failure)
        if len(self._entries) > self.max_entries:
            oldest = sorted(self._entries.items(), key=lambda item: item[1][0])[: max(1, len(self._entries) - self.max_entries)]
            for key, _ in oldest:
                self._entries.pop(key, None)

    def record_success(self, provider: Any, family: Any, cache_key: Any) -> None:
        self._entries.pop(self._key(provider, family, cache_key), None)

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        active = {key: value for key, value in self._entries.items() if value[0] > now}
        providers: dict[str, int] = {}
        for provider, _, _ in active:
            providers[provider] = providers.get(provider, 0) + 1
        return {"active_entries": len(active), "providers": providers, "version": EVIDENCE_GOVERNANCE_VERSION}


DEFAULT_GUARDRAIL_PARAMETERS = {
    "anti_chase_penalty": 3.0,
    "technical_floor": 50.0,
    "technical_slope": 0.08,
    "technical_max_penalty": 4.0,
    "flow_floor": 40.0,
    "flow_slope": 0.06,
    "flow_max_penalty": 2.4,
    "distribution_floor": 45.0,
    "distribution_slope": 0.08,
    "distribution_max_penalty": 2.0,
    "distribution_block_penalty": 8.0,
}


def _guarded_score(frame: pd.DataFrame, p: Mapping[str, float]) -> pd.Series:
    raw = pd.to_numeric(frame["raw_ranking_score"], errors="coerce")
    technical = pd.to_numeric(frame.get("technical_readiness_score", np.nan), errors="coerce")
    flow = pd.to_numeric(frame.get("narrative_flow_score", frame.get("silent_accumulation_score", np.nan)), errors="coerce")
    distribution = pd.to_numeric(frame.get("distribution_score", np.nan), errors="coerce")
    anti = frame.get("anti_chase_gate", pd.Series(False, index=frame.index)).map(_truthy)
    block = frame.get("distribution_block", pd.Series(False, index=frame.index)).map(_truthy)
    penalty = anti.astype(float) * float(p["anti_chase_penalty"])
    penalty += (float(p["technical_floor"]) - technical).clip(lower=0).fillna(0) * float(p["technical_slope"])
    penalty += (float(p["flow_floor"]) - flow).clip(lower=0).fillna(0) * float(p["flow_slope"])
    penalty += (distribution - float(p["distribution_floor"])).clip(lower=0).fillna(0) * float(p["distribution_slope"])
    penalty = penalty.clip(upper=float(p["technical_max_penalty"]) + float(p["flow_max_penalty"]) + float(p["distribution_max_penalty"]) + float(p["anti_chase_penalty"]))
    penalty += block.astype(float) * float(p["distribution_block_penalty"])
    return raw - penalty


def _objective(frame: pd.DataFrame, parameters: Mapping[str, float], return_col: str) -> float:
    local = frame.copy()
    local["_score"] = _guarded_score(local, parameters)
    local["_ret"] = pd.to_numeric(local[return_col], errors="coerce")
    local = local.dropna(subset=["_score", "_ret"])
    if len(local) < 12:
        return -1e9
    selected = local.sort_values("_score", ascending=False, kind="stable").head(max(10, min(30, int(math.ceil(len(local) * 0.20)))))
    median_ret = float(selected["_ret"].median())
    hit_rate = float((selected["_ret"] > 0).mean())
    downside = float(selected["_ret"].quantile(0.10))
    return median_ret + 4.0 * (hit_rate - 0.5) + 0.35 * min(0.0, downside)


def calibrate_guardrails_walk_forward(
    outcomes: pd.DataFrame,
    *,
    default_parameters: Mapping[str, float] | None = None,
    return_col: str = "forward_return_20d",
    min_rows: int = 120,
    min_signal_dates: int = 8,
) -> dict[str, Any]:
    """Expanding-window OOS calibration; latest snapshot is never training data by itself."""
    defaults = dict(default_parameters or DEFAULT_GUARDRAIL_PARAMETERS)
    required = {"signal_date", "raw_ranking_score", return_col}
    if outcomes is None or outcomes.empty or not required.issubset(outcomes.columns):
        return {"calibration_state": "INSUFFICIENT_MATURE_OOS_EVIDENCE_KEEP_BASELINE", "active": False, "parameters": defaults, "sample_count": 0, "fold_count": 0}
    local = outcomes.copy()
    local["signal_date"] = pd.to_datetime(local["signal_date"], errors="coerce", utc=True)
    local[return_col] = pd.to_numeric(local[return_col], errors="coerce")
    if "outcome_verified" in local.columns:
        local = local[local["outcome_verified"].map(_truthy)]
    elif "outcome_status" in local.columns:
        local = local[local["outcome_status"].astype(str).str.upper().isin({"RESOLVED", "VERIFIED"})]
    local = local.dropna(subset=["signal_date", "raw_ranking_score", return_col]).sort_values("signal_date")
    dates = list(pd.Index(local["signal_date"].dt.date.unique()).sort_values())
    if len(local) < int(min_rows) or len(dates) < int(min_signal_dates):
        return {"calibration_state": "INSUFFICIENT_MATURE_OOS_EVIDENCE_KEEP_BASELINE", "active": False, "parameters": defaults, "sample_count": int(len(local)), "distinct_signal_dates": len(dates), "fold_count": 0}
    split_points = np.linspace(max(4, len(dates) // 2), len(dates) - 1, num=min(4, max(1, len(dates) // 2)), dtype=int)
    scales = (0.75, 1.0, 1.25)
    fold_rows: list[dict[str, Any]] = []
    chosen_scales: list[float] = []
    for split in sorted(set(int(x) for x in split_points if int(x) < len(dates))):
        train_dates = set(dates[:split])
        validation_date = dates[split]
        train = local[local["signal_date"].dt.date.isin(train_dates)]
        valid = local[local["signal_date"].dt.date.eq(validation_date)]
        if len(train) < 40 or len(valid) < 5:
            continue
        best_scale, best_train = 1.0, -1e9
        for scale in scales:
            candidate = dict(defaults)
            for key in ("anti_chase_penalty", "technical_slope", "flow_slope", "distribution_slope", "distribution_block_penalty"):
                candidate[key] = float(defaults[key]) * float(scale)
            score = _objective(train, candidate, return_col)
            if score > best_train:
                best_scale, best_train = float(scale), float(score)
        candidate = dict(defaults)
        for key in ("anti_chase_penalty", "technical_slope", "flow_slope", "distribution_slope", "distribution_block_penalty"):
            candidate[key] = float(defaults[key]) * best_scale
        oos = _objective(valid, candidate, return_col)
        baseline = _objective(valid, defaults, return_col)
        fold_rows.append({"validation_date": str(validation_date), "chosen_scale": best_scale, "oos_objective": oos, "baseline_objective": baseline, "oos_lift": oos - baseline})
        chosen_scales.append(best_scale)
    if len(fold_rows) < 2:
        return {"calibration_state": "INSUFFICIENT_WALK_FORWARD_FOLDS_KEEP_BASELINE", "active": False, "parameters": defaults, "sample_count": int(len(local)), "distinct_signal_dates": len(dates), "fold_count": len(fold_rows), "folds": fold_rows}
    median_lift = float(np.median([row["oos_lift"] for row in fold_rows]))
    positive_fold_rate = float(np.mean([row["oos_lift"] >= 0 for row in fold_rows]))
    scale = float(np.median(chosen_scales))
    active = bool(median_lift > 0 and positive_fold_rate >= 0.60 and scale != 1.0)
    selected = dict(defaults)
    if active:
        for key in ("anti_chase_penalty", "technical_slope", "flow_slope", "distribution_slope", "distribution_block_penalty"):
            selected[key] = float(defaults[key]) * scale
    return {
        "calibration_state": "OOS_WALK_FORWARD_ACTIVE" if active else "OOS_NO_STABLE_LIFT_KEEP_BASELINE",
        "active": active,
        "parameters": selected,
        "sample_count": int(len(local)),
        "distinct_signal_dates": len(dates),
        "fold_count": len(fold_rows),
        "median_oos_lift": median_lift,
        "positive_fold_rate": positive_fold_rate,
        "selected_penalty_scale": scale if active else 1.0,
        "folds": fold_rows,
    }


__all__ = [
    "DEFAULT_GUARDRAIL_PARAMETERS", "EVIDENCE_GOVERNANCE_VERSION", "ProviderNegativeCache",
    "apply_three_rank_contract", "calibrate_guardrails_walk_forward", "is_https_url",
    "select_enrichment_shortlist", "validate_official_evidence",
]
