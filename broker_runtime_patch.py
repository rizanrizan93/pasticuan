from __future__ import annotations

from functools import wraps
from typing import Any

import pandas as pd

from public_idx_broker_flow import PUBLIC_CACHE_URL, VERSION, enrich_super_broker

PATCH_VERSION = "1.1.1-pasticuan-owned-public-idx-broker-cache"


def _merge_fields(out: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(out, pd.DataFrame) or out.empty or not isinstance(enriched, pd.DataFrame) or enriched.empty:
        return out
    if "ticker" not in out.columns or "ticker" not in enriched.columns:
        return out
    fields = [
        "ticker", "broker_flow_observed_days", "broker_flow_latest_date", "broker_top_buyer_code",
        "broker_latest_top_buyer_code", "broker_top3_buyer_persistence_20d_pct",
        "broker_top_buyer_net_value_20d", "broker_buyer_concentration_pct",
        "broker_buy_sell_dominance_ratio", "broker_latest_top_buyer_buy_avg",
        "broker_flow_coverage_pct", "broker_flow_source", "broker_flow_provenance",
        "broker_net_score", "broker_accumulation_score", "broker_smart_money_confirmation_score",
        "broker_accumulation_state", "broker_confirmation_weight_pct",
        "broker_pre_confirmation_accumulation_score", "broker_post_confirmation_accumulation_score",
        "broker_accumulation_delta", "broker_flow_version",
    ]
    right = enriched[[c for c in fields if c in enriched.columns]].copy()
    right["_broker_key"] = right["ticker"].astype(str).str.upper().str.removesuffix(".JK")
    right = right.drop(columns=["ticker"]).drop_duplicates("_broker_key")
    left = out.copy()
    left["_broker_key"] = left["ticker"].astype(str).str.upper().str.removesuffix(".JK")
    duplicate = [c for c in right.columns if c != "_broker_key" and c in left.columns]
    if duplicate:
        left = left.drop(columns=duplicate)
    return left.merge(right, on="_broker_key", how="left").drop(columns=["_broker_key"])


def _wrap(owner: Any, name: str) -> None:
    original = getattr(owner, name, None)
    if not callable(original) or getattr(original, "__public_idx_broker_v1__", False):
        return

    @wraps(original)
    def wrapped(universe: pd.DataFrame, *args: Any, **kwargs: Any):
        enriched = universe
        try:
            if isinstance(universe, pd.DataFrame) and not universe.empty:
                enriched = enrich_super_broker(universe)
        except Exception:
            enriched = universe
        out = original(enriched, *args, **kwargs)
        if isinstance(out, pd.DataFrame):
            try:
                out = _merge_fields(out, enriched)
            except Exception:
                pass
        return out

    wrapped.__public_idx_broker_v1__ = True
    setattr(owner, name, wrapped)


def install() -> dict[str, str]:
    import simple_focus
    _wrap(simple_focus, "build_next_leaders")
    _wrap(simple_focus, "build_swing_ready")
    return {
        "patch_version": PATCH_VERSION,
        "broker_flow_version": VERSION,
        "policy": "CANONICAL_IDX_PUBLIC_PARTICIPANT_FLOW_CONFIRMATION_NOT_BENEFICIAL_OWNER_IDENTITY",
        "max_confirmation_weight_pct": "20",
        "cache_policy": "PASTICUAN_OWNED_PUBLIC_PARTICIPANT_CACHE_WITH_DIRECT_IDX_REFRESH",
        "canonical_source": PUBLIC_CACHE_URL,
    }


__all__ = ["PATCH_VERSION", "install"]
