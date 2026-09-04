from __future__ import annotations

"""Route PASTICUAN forward factual input through the Shared Evidence Hub.

The Shared Hub supplies facts only. PASTICUAN's existing official_evidence_bridge
continues to derive its own project/future-fundamental model inputs.
"""

from typing import Any, Sequence

import numpy as np
import pandas as pd

from official_evidence_bridge import bridge_project_events
from pasticuan_shared_forward_producer import publish_project_events
from shared_forward_evidence import (
    canonical_rows_to_pasticuan_projects,
    profile_rows,
    read_canonical_forward_rows,
)

PATCH_VERSION = "1.1.0-shared-canonical-forward-consumer-producer"


def _pasticuan_model_input(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Adapt canonical facts without importing shared completeness as model coverage."""
    frame = canonical_rows_to_pasticuan_projects(rows)
    if not frame.empty:
        frame["project_data_coverage"] = np.nan
    return frame


def _attach_profiles(frame: pd.DataFrame, rows: list[dict[str, Any]], universe: Sequence[str]) -> pd.DataFrame:
    profiles = [profile_rows(rows, ticker=ticker) for ticker in universe]
    profile_frame = pd.DataFrame(profiles)
    if frame.empty or profile_frame.empty or "ticker" not in frame.columns:
        return frame
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    profile_frame["ticker"] = profile_frame["ticker"].astype(str).str.upper().str.strip()
    return frame.merge(profile_frame, on="ticker", how="left", suffixes=("", "_shared"))


def install() -> dict[str, str]:
    import resumable_app_engine as engine

    if getattr(engine, "_shared_forward_runtime_patch", "") == PATCH_VERSION:
        return {"patch_version": PATCH_VERSION, "state": "ALREADY_INSTALLED"}

    original = engine._job_forward_quality

    def shared_job_forward_quality(job_id: str, bridge: Any, universe: Sequence[str]) -> pd.DataFrame:
        key = str(job_id or "")
        cached = getattr(engine, "_JOB_FORWARD_CACHE", {}).get(key)
        if cached is not None:
            return cached.copy()

        # Publish newly verified PASTICUAN project facts first. The producer
        # reconciles existing canonical provenance and fails closed on unsafe
        # writes, so a producer failure does not mutate or weaken the ledger.
        producer_audit = publish_project_events()

        canonical_rows, audit = read_canonical_forward_rows(universe, client_id="PASTICUAN")
        state = str(audit.get("state") or "")
        if state == "SHARED_CANONICAL_FORWARD":
            raw_projects = _pasticuan_model_input(canonical_rows)
            # Existing PASTICUAN deterministic model conversion remains local
            # and independent; this does not import EMIR scores or shared
            # contract-completeness as PASTICUAN model coverage.
            frame = bridge_project_events(raw_projects)
            frame = _attach_profiles(frame, canonical_rows, universe)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frame["shared_forward_producer_state"] = str(producer_audit.get("state") or "UNKNOWN")
                frame["shared_forward_producer_rows"] = int(producer_audit.get("rows") or 0)
            cache = getattr(engine, "_JOB_FORWARD_CACHE", None)
            if isinstance(cache, dict):
                cache[key] = frame.copy()
                while len(cache) > 4:
                    cache.pop(next(iter(cache)))
            return frame.copy()

        # During a Shared Hub outage, retain the old fail-soft cache path. Once
        # the canonical table is readable, even an empty result is authoritative
        # and must not silently diverge to a scanner-local fact set.
        return original(job_id, bridge, universe)

    engine._job_forward_quality = shared_job_forward_quality
    engine._shared_forward_runtime_patch = PATCH_VERSION
    return {
        "patch_version": PATCH_VERSION,
        "facts": "SHARED_CANONICAL_FORWARD",
        "producer": "PASTICUAN_PROJECT_EVENTS_STRICT_FACTS_RECONCILED",
        "scoring": "PASTICUAN_INDEPENDENT_OFFICIAL_EVIDENCE_BRIDGE",
        "shared_contract_coverage": "TELEMETRY_ONLY_NOT_MODEL_COVERAGE",
        "fallback": "LOCAL_ONLY_WHEN_SHARED_HUB_UNAVAILABLE",
    }


__all__ = ["PATCH_VERSION", "install", "_pasticuan_model_input"]
