from __future__ import annotations

from functools import wraps
from io import BytesIO
import gzip
from typing import Any

import pandas as pd
import requests

from public_idx_broker_flow import VERSION, enrich_super_broker

PATCH_VERSION = "1.1.0-super-public-idx-broker-canonical-bridge"
CANONICAL_PUBLIC_PARTICIPANT_URL = "https://raw.githubusercontent.com/rizanrizan93/idx-flow-scanner/main/data/cache/idx_public_participant_30d.csv.gz"


def _normalize_canonical_participant_cache(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    required = {"ticker", "trade_date", "participant", "buy_value", "sell_value", "buy_volume", "sell_volume"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    out = frame.copy()
    out["broker_code"] = out["participant"].astype(str).str.strip().str.upper()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.removesuffix(".JK")
    out["buy_value"] = pd.to_numeric(out["buy_value"], errors="coerce").fillna(0.0)
    out["sell_value"] = pd.to_numeric(out["sell_value"], errors="coerce").fillna(0.0)
    out["buy_volume"] = pd.to_numeric(out["buy_volume"], errors="coerce").fillna(0.0)
    out["sell_volume"] = pd.to_numeric(out["sell_volume"], errors="coerce").fillna(0.0)
    out["buy_avg"] = out["buy_value"].div(out["buy_volume"].replace(0.0, pd.NA))
    out["sell_avg"] = out["sell_value"].div(out["sell_volume"].replace(0.0, pd.NA))
    out["net_value"] = out["buy_value"] - out["sell_value"]
    out["net_volume"] = out["buy_volume"] - out["sell_volume"]
    out["gross_value"] = out["buy_value"] + out["sell_value"]
    out["source"] = "IDX_OFFICIAL_PUBLIC_TRADE_DETAIL_PARTICIPANT_FLOW"
    out["source_verified"] = True
    out["provenance_state"] = "VERIFIED_IDX_PUBLIC_TRADE_DETAIL_PARTICIPANT_FLOW_NOT_BENEFICIAL_OWNER"
    out["side"] = out["net_value"].map(lambda x: "TOP_NET_BUYER" if x > 0 else ("TOP_NET_SELLER" if x < 0 else "NEUTRAL"))
    out["net_rank"] = out.groupby(["trade_date", "ticker", "side"])["net_value"].rank(method="first", ascending=False)
    return out.dropna(subset=["ticker", "trade_date", "broker_code"]).reset_index(drop=True)


def _install_canonical_cache_bridge() -> None:
    import public_idx_broker_flow as broker_module
    original = getattr(broker_module, "load_public_cache", None)
    if not callable(original) or getattr(original, "__canonical_idx_participant_bridge_v1__", False):
        return

    @wraps(original)
    def wrapped() -> pd.DataFrame:
        try:
            response = requests.get(
                CANONICAL_PUBLIC_PARTICIPANT_URL,
                timeout=12,
                headers={"User-Agent": "IDX-Scanner-Broker-Bridge/1.0"},
            )
            response.raise_for_status()
            canonical = _normalize_canonical_participant_cache(pd.read_csv(BytesIO(gzip.decompress(response.content))))
            if not canonical.empty:
                return canonical
        except Exception:
            pass
        return original()

    wrapped.__canonical_idx_participant_bridge_v1__ = True
    broker_module.load_public_cache = wrapped


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
    _install_canonical_cache_bridge()
    _wrap(simple_focus, "build_next_leaders")
    _wrap(simple_focus, "build_swing_ready")
    return {
        "patch_version": PATCH_VERSION,
        "broker_flow_version": VERSION,
        "policy": "CANONICAL_IDX_PUBLIC_PARTICIPANT_FLOW_CONFIRMATION_NOT_BENEFICIAL_OWNER_IDENTITY",
        "max_confirmation_weight_pct": "20",
        "cache_policy": "IDX_FLOW_SCANNER_PUBLIC_PARTICIPANT_CACHE_PRIMARY_PASTICUAN_CACHE_FAIL_SOFT_FALLBACK",
        "canonical_source": CANONICAL_PUBLIC_PARTICIPANT_URL,
    }


__all__ = ["PATCH_VERSION", "install"]