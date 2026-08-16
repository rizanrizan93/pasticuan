from __future__ import annotations

"""Runtime hook: bounded ZAPI confirmation for Super Scanner final decision flow."""

from functools import wraps
from typing import Any

import pandas as pd

from zapi_flow_enrichment import (
    ZAPI_FLOW_ENRICHMENT_VERSION,
    enrich_super_universe,
)

PATCH_VERSION = "1.0.0-super-zapi-flow"


def _canonical(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def _merge_audit_fields(out: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(out, pd.DataFrame) or out.empty or not isinstance(enriched, pd.DataFrame) or enriched.empty:
        return out
    if "ticker" not in out.columns or "ticker" not in enriched.columns:
        return out
    wanted = [
        "ticker",
        "zapi_foreign_flow_score",
        "zapi_foreign_flow_coverage_pct",
        "zapi_foreign_net_participation_1d",
        "zapi_foreign_net_participation_5d",
        "zapi_foreign_net_participation_20d",
        "zapi_foreign_positive_days_ratio_20d",
        "zapi_foreign_state",
        "zapi_accumulation_confirmation_score",
        "zapi_smart_money_confirmation_score",
        "zapi_smc_flow_confirmation_score",
        "zapi_flow_evidence_type",
        "zapi_confirmation_weight_pct",
        "zapi_super_original_silent_score",
        "zapi_super_original_silent_coverage_pct",
        "zapi_super_flow_basis",
        "zapi_flow_meta_state",
    ]
    cols = [column for column in wanted if column in enriched.columns]
    audit = enriched[cols].copy()
    audit["_zapi_key"] = audit["ticker"].map(_canonical)
    audit = audit.drop(columns=["ticker"]).drop_duplicates("_zapi_key", keep="last")
    result = out.copy()
    result["_zapi_key"] = result["ticker"].map(_canonical)
    duplicate = [column for column in audit.columns if column != "_zapi_key" and column in result.columns]
    if duplicate:
        result = result.drop(columns=duplicate)
    return result.merge(audit, on="_zapi_key", how="left").drop(columns=["_zapi_key"])


def _wrap_focus_builder(owner: Any, name: str) -> None:
    original = getattr(owner, name, None)
    if not callable(original) or getattr(original, "__zapi_flow_confirmation_v1__", False):
        return

    @wraps(original)
    def wrapped(universe: pd.DataFrame, *args: Any, **kwargs: Any):
        enriched = universe
        try:
            if isinstance(universe, pd.DataFrame) and not universe.empty:
                enriched = enrich_super_universe(universe)
        except Exception:
            enriched = universe
        out = original(enriched, *args, **kwargs)
        if isinstance(out, pd.DataFrame):
            try:
                out = _merge_audit_fields(out, enriched)
            except Exception:
                pass
        return out

    wrapped.__zapi_flow_confirmation_v1__ = True
    setattr(owner, name, wrapped)


def install() -> dict[str, str]:
    import simple_focus

    _wrap_focus_builder(simple_focus, "build_next_leaders")
    _wrap_focus_builder(simple_focus, "build_swing_ready")
    return {
        "patch_version": PATCH_VERSION,
        "zapi_version": ZAPI_FLOW_ENRICHMENT_VERSION,
        "policy": "BOUNDED_CONFIRMATION_INSIDE_EXISTING_NARRATIVE_FLOW_PILLAR",
        "smc_policy": "PRICE_STRUCTURE_PRIMARY_ZAPI_FLOW_CONFIRMATION_ONLY",
        "identity_policy": "FOREIGN_FLOW_IS_NOT_BROKER_OR_BENEFICIAL_OWNER_IDENTITY",
    }


__all__ = ["PATCH_VERSION", "install"]
