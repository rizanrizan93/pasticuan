from __future__ import annotations

"""Issuer sector normalisation and deterministic fail-soft classification.

The production scanner prefers an explicit public-provider/IDX sector.  When
that field is missing, a conservative inference from industry/business text is
used.  A small audited override map covers issuers that were observed missing
sector metadata in the 2026-08-07 production scan; these are public
classifications, not scoring overrides.
"""

from typing import Any, Mapping
import re

import numpy as np
import pandas as pd

CLASSIFIER_VERSION = "9.6.0-sector-normalizer"

# Audited public classifications for issuers that were UNKNOWN in the
# production scan that motivated v9.6.0.  Keep this map intentionally small;
# provider/IDX evidence always has priority when it is available.
_TICKER_SECTOR_OVERRIDES: dict[str, str] = {
    "DGWG.JK": "BASIC MATERIALS",
    "BMSR.JK": "BASIC MATERIALS",
    "HRUM.JK": "ENERGY",
    "INDY.JK": "ENERGY",
    "SSIA.JK": "INFRASTRUCTURE",
}

_ALIAS = {
    "BASIC MATERIALS": "BASIC MATERIALS",
    "BASIC INDUSTRY": "BASIC MATERIALS",
    "MATERIALS": "BASIC MATERIALS",
    "CHEMICALS": "BASIC MATERIALS",
    "ENERGY": "ENERGY",
    "FINANCIALS": "FINANCIALS",
    "FINANCE": "FINANCIALS",
    "BANKING": "FINANCIALS",
    "CONSUMER CYCLICAL": "CONSUMER CYCLICALS",
    "CONSUMER CYCLICALS": "CONSUMER CYCLICALS",
    "CYCLICAL": "CONSUMER CYCLICALS",
    "CONSUMER DISCRETIONARY": "CONSUMER CYCLICALS",
    "CONSUMER NON-CYCLICAL": "CONSUMER NON-CYCLICALS",
    "CONSUMER NON-CYCLICALS": "CONSUMER NON-CYCLICALS",
    "NON-CYCLICAL": "CONSUMER NON-CYCLICALS",
    "CONSUMER STAPLES": "CONSUMER NON-CYCLICALS",
    "INDUSTRIALS": "INDUSTRIALS",
    "INDUSTRIAL": "INDUSTRIALS",
    "INFRASTRUCTURE": "INFRASTRUCTURE",
    "PROPERTIES AND REAL ESTATE": "PROPERTY",
    "PROPERTIES & REAL ESTATE": "PROPERTY",
    "PROPERTY": "PROPERTY",
    "REAL ESTATE": "PROPERTY",
    "TECHNOLOGY": "TECHNOLOGY",
    "HEALTHCARE": "HEALTHCARE",
    "HEALTH CARE": "HEALTHCARE",
    "TRANSPORTATION AND LOGISTIC": "TRANSPORTATION",
    "TRANSPORTATION AND LOGISTICS": "TRANSPORTATION",
    "TRANSPORTATION & LOGISTICS": "TRANSPORTATION",
    "TRANSPORTATION": "TRANSPORTATION",
    # This is an explicit universe classification, but there is deliberately no
    # generic macro transmission weight for diversified holdings. The issuer
    # remains sector-known while its macro score stays evidence-pending.
    "MULTI-SECTOR HOLDINGS": "MULTI-SECTOR HOLDINGS",
    "MULTI SECTOR HOLDINGS": "MULTI-SECTOR HOLDINGS",
}

# Conservative keyword inference.  Order matters: specific phrases precede
# generic words such as mining/construction.
_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("FINANCIALS", (
        "commercial bank", "banking", "bank ", "insurance", "consumer finance",
        "multifinance", "securities brokerage", "asset management",
    )),
    ("ENERGY", (
        "thermal coal", "coal mining", "coal", "oil and gas", "oil & gas",
        "petroleum", "natural gas", "energy services", "energy company",
    )),
    ("BASIC MATERIALS", (
        "agricultural chemicals", "agrochemical", "agro chemical", "fertilizer",
        "pesticide", "chemical", "nickel", "gold mining", "copper", "bauxite",
        "aluminium", "aluminum", "steel", "cement", "paper products",
        "basic materials", "metal mining",
    )),
    ("INFRASTRUCTURE", (
        "heavy construction", "civil engineering", "toll road", "telecommunication infrastructure",
        "tower infrastructure", "water utility", "port management", "airport infrastructure",
        "infrastructure services",
    )),
    ("PROPERTY", (
        "real estate", "property development", "industrial estate", "township",
        "residential development", "shopping mall", "property management",
    )),
    ("TRANSPORTATION", (
        "shipping", "marine transportation", "logistics", "transportation",
        "airline", "trucking", "freight", "port operator",
    )),
    ("TECHNOLOGY", (
        "software", "information technology", "technology", "data center",
        "digital platform", "it services", "computer hardware",
    )),
    ("HEALTHCARE", (
        "hospital", "healthcare", "health care", "pharmaceutical", "medical devices",
        "clinic", "laboratory services",
    )),
    ("CONSUMER NON-CYCLICALS", (
        "food products", "food distribution", "beverage", "tobacco", "supermarket",
        "grocery", "personal care", "household products", "plantation", "poultry",
        "consumer staples",
    )),
    ("CONSUMER CYCLICALS", (
        "retail", "department store", "automotive", "restaurant", "hotel",
        "tourism", "apparel", "consumer discretionary", "media entertainment",
    )),
    ("INDUSTRIALS", (
        "engineering & construction", "engineering and construction", "construction",
        "industrial machinery", "industrial services", "manufacturing services",
        "commercial services",
    )),
)


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text if text.endswith(".JK") else f"{text}.JK"


def canonical_sector(value: Any) -> str:
    text = str(value or "").strip().upper().replace("&", "AND")
    text = re.sub(r"\s+", " ", text)
    if not text or text in {"NAN", "NONE", "UNKNOWN", "N/A", "NA"}:
        return "UNKNOWN"
    if text in _ALIAS:
        return _ALIAS[text]
    for key, mapped in _ALIAS.items():
        if key in text:
            return mapped
    return "UNKNOWN"


def classify_issuer(row: Mapping[str, Any]) -> tuple[str, str, float]:
    ticker = _ticker(row.get("ticker"))
    explicit_candidates = (
        row.get("idx_sector"), row.get("sector"), row.get("sector_name"),
        row.get("profile_sector"), row.get("yahoo_sector"),
    )
    for value in explicit_candidates:
        sector = canonical_sector(value)
        if sector != "UNKNOWN":
            return sector, "EXPLICIT_PROVIDER", 100.0

    if ticker in _TICKER_SECTOR_OVERRIDES:
        return _TICKER_SECTOR_OVERRIDES[ticker], "AUDITED_PUBLIC_OVERRIDE", 95.0

    text_parts = []
    for name in (
        "industry", "industry_name", "subsector", "sub_sector", "sub_industry",
        "business_summary", "company_business", "business_activity", "company_name",
    ):
        value = row.get(name)
        if value is not None and str(value).strip():
            text_parts.append(str(value).strip().lower())
    text = " | ".join(text_parts)
    if text:
        for sector, keywords in _KEYWORDS:
            if any(keyword in text for keyword in keywords):
                return sector, "TEXT_INFERENCE", 70.0
    return "UNKNOWN", "MISSING", 0.0


def normalize_fundamental_classification(frame: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    out = frame.copy()
    sectors: list[str] = []
    sources: list[str] = []
    confidences: list[float] = []
    raw: list[str] = []
    for _, source in out.iterrows():
        row = source.to_dict()
        raw_value = row.get("sector", "")
        sector, source_name, confidence = classify_issuer(row)
        sectors.append(sector)
        sources.append(source_name)
        confidences.append(confidence)
        raw.append(str(raw_value or ""))
    if "sector_raw" not in out.columns:
        out["sector_raw"] = raw
    out["sector"] = sectors
    out["sector_source"] = sources
    out["sector_confidence_pct"] = confidences
    out["sector_classification_version"] = CLASSIFIER_VERSION
    return out


__all__ = [
    "CLASSIFIER_VERSION", "canonical_sector", "classify_issuer",
    "normalize_fundamental_classification",
]
