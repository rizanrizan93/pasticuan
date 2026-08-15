from __future__ import annotations

"""Convert governed official evidence into scanner-consumable rows.

Raw disclosure tables remain factual stores.  This module derives conservative,
deterministic model inputs only after HTTPS lineage, entity match, evidence date
and source quorum pass.  The derived numbers are scanner scores, never issuer-
reported scores, and are explicitly labelled as such.
"""

from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse
import json
import math
import re

import numpy as np
import pandas as pd

OFFICIAL_EVIDENCE_BRIDGE_VERSION = "1.0.0-direct-facts-to-model-inputs"


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _ticker(value: Any) -> str:
    text = _text(value).upper().replace(" ", "")
    return text if text.endswith(".JK") else f"{text}.JK" if text else ""


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return _text(value).upper() in {"1", "TRUE", "YES", "Y", "VERIFIED", "PASS", "VALID"}


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.to_datetime(value, errors="coerce", utc=True)
    return pd.Timestamp(stamp) if pd.notna(stamp) else pd.NaT


def _urls(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        candidates = [str(item).strip() for item in value]
    else:
        candidates = [part.strip() for part in re.split(r"[|\n,]+", _text(value))]
    out: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = urlparse(candidate)
        except ValueError:
            continue
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            continue
        if candidate not in out:
            out.append(candidate)
    return out


def _source_families(value: Any) -> list[str]:
    return list(dict.fromkeys(
        token.strip().upper() for token in re.split(r"[|,]+", _text(value)) if token.strip()
    ))


def _strict_lineage(
    row: Mapping[str, Any],
    *,
    url_fields: Sequence[str],
    date_fields: Sequence[str],
    verified_field: str | None = None,
    quorum_field: str = "source_quorum_verified",
    count_field: str = "source_quorum_count",
    entity_field: str = "entity_match_verified",
    as_of: Any | None = None,
    max_age_days: int = 540,
) -> dict[str, Any]:
    urls: list[str] = []
    for field in url_fields:
        for url in _urls(row.get(field)):
            if url not in urls:
                urls.append(url)
    date = pd.NaT
    for field in date_fields:
        candidate = _timestamp(row.get(field))
        if pd.notna(candidate):
            date = candidate
            break
    now = _timestamp(as_of)
    if pd.isna(now):
        now = pd.Timestamp.now(tz="UTC")
    age_days = float((now - date).total_seconds() / 86400.0) if pd.notna(date) else np.nan
    count = int(max(0.0, _finite(row.get(count_field), 0.0)))
    verified = True if verified_field is None else _truthy(row.get(verified_field))
    quorum = _truthy(row.get(quorum_field)) and count >= 2
    entity = _truthy(row.get(entity_field))
    date_ok = bool(np.isfinite(age_days) and -1.0 <= age_days <= float(max_age_days))
    # Quorum means at least two corroborating sources, but historical rows can
    # store a single canonical URL plus a separately audited quorum count.  Keep
    # both facts visible instead of inventing a second URL.
    valid = bool(verified and quorum and entity and len(urls) >= 1 and date_ok)
    return {
        "valid": valid,
        "urls": urls,
        "source_quorum_count": count,
        "source_quorum_verified": quorum,
        "entity_match_verified": entity,
        "evidence_date": date.isoformat() if pd.notna(date) else "",
        "evidence_age_days": round(age_days, 1) if np.isfinite(age_days) else np.nan,
        "source_https_verified": bool(urls),
    }


def _has(text: str, *tokens: str) -> bool:
    upper = text.upper()
    return any(token.upper() in upper for token in tokens)


def _project_score(row: Mapping[str, Any], lineage: Mapping[str, Any]) -> dict[str, Any]:
    text = " | ".join(_text(row.get(name)) for name in (
        "project_name", "project_names", "project_stage", "project_execution_flags", "review_origin"
    ))
    completion = _finite(row.get("project_completion_pct"), np.nan)
    funding = _finite(row.get("project_funding_secured_pct"), np.nan)
    capex = _finite(row.get("project_capex_idr"), np.nan)
    revenue = _finite(row.get("project_expected_revenue_idr"), np.nan)
    ebitda = _finite(row.get("project_expected_ebitda_idr"), np.nan)

    if _has(text, "OPERATING", "COMMERCIAL OPERATION", "REPORTED_FULL", "BACKLOG", "ORDER_VISIBILITY", "ORDER VISIBILITY"):
        commitment, execution = 88.0, 90.0
    elif _has(text, "SIGNED", "AWARDED", "CONTRACT", "JOINT VENTURE", "JV"):
        commitment, execution = 84.0, 62.0
    elif _has(text, "APPROVED", "BOARD_APPROVED", "FUNDED", "SECURED"):
        commitment, execution = 78.0, 58.0
    elif _has(text, "IN_PROGRESS", "CONSTRUCTION", "COMMISSIONING"):
        commitment, execution = 72.0, 68.0
    elif _has(text, "GUIDANCE", "LAUNCH", "CAPEX"):
        commitment, execution = 68.0, 52.0
    else:
        commitment, execution = 52.0, 42.0

    if np.isfinite(completion):
        execution = max(execution, float(np.clip(30.0 + 0.65 * completion, 30.0, 95.0)))
    if np.isfinite(funding):
        commitment = max(commitment, float(np.clip(45.0 + 0.45 * funding, 45.0, 90.0)))

    if np.isfinite(revenue) and revenue > 0 or np.isfinite(ebitda) and ebitda > 0:
        quantification = 90.0
    elif np.isfinite(capex) and capex > 0:
        quantification = 82.0
    elif np.isfinite(completion) or np.isfinite(funding):
        quantification = 74.0
    elif re.search(r"\b\d+(?:[.,]\d+)?\s*(?:TRILIUN|TRILLION|MILIAR|BILLION|JUTA|MILLION|UNIT|MW|GW|TPA|KTPA)\b", text.upper()):
        quantification = 66.0
    elif re.search(r"\b(?:Q[1-4]|20\d{2})\b", text.upper()):
        quantification = 54.0
    else:
        quantification = 38.0

    quorum_count = int(lineage.get("source_quorum_count", 0) or 0)
    family_count = len(_source_families(row.get("project_source_families")))
    corroboration = min(95.0, 72.0 + 5.0 * max(0, quorum_count - 2) + 5.0 * max(0, family_count - 1))

    pipeline = 0.32 * commitment + 0.32 * execution + 0.20 * quantification + 0.16 * corroboration
    pipeline = float(np.clip(pipeline, 0.0, 88.0))

    if np.isfinite(revenue) and revenue > 0 or np.isfinite(ebitda) and ebitda > 0:
        financial_link = 90.0
    elif _has(text, "BACKLOG", "ORDER", "OFFTAKE", "CAPACITY", "KAPASITAS"):
        financial_link = 80.0
    elif np.isfinite(capex) and capex > 0:
        financial_link = 68.0
    elif _has(text, "JOINT VENTURE", "JV", "NEW PRODUCT", "PRODUCT", "LAUNCH", "NEW MARKET"):
        financial_link = 65.0
    else:
        financial_link = 52.0

    if _has(text, "OPERATING", "REPORTED_FULL", "BACKLOG", "ORDER_VISIBILITY"):
        timing = 86.0
    elif re.search(r"\bQ[1-4][ _-]?20\d{2}\b", text.upper()):
        timing = 74.0
    elif _has(text, "SIGNED", "IN_PROGRESS", "COMMISSIONING", "LAUNCH"):
        timing = 65.0
    else:
        timing = 52.0

    strategic = (
        78.0 if _has(text, "CONTRACT", "BACKLOG", "OFFTAKE", "ORDER")
        else 72.0 if _has(text, "JOINT VENTURE", "JV", "EXPANSION", "CAPACITY", "NEW MARKET", "NEW PRODUCT", "LAUNCH")
        else 62.0 if _has(text, "CAPEX", "GUIDANCE")
        else 50.0
    )
    impact = 0.45 * financial_link + 0.25 * timing + 0.15 * strategic + 0.15 * corroboration
    impact = float(np.clip(impact, 0.0, 85.0))

    observed_points = 30.0  # strict source/quorum/entity lineage
    observed_points += 15.0 if _text(row.get("project_stage")) else 0.0
    observed_points += 15.0 if lineage.get("evidence_date") else 0.0
    observed_points += 20.0 if any(np.isfinite(v) for v in (completion, funding, capex, revenue, ebitda)) or quantification >= 66.0 else 0.0
    observed_points += 20.0 if execution >= 60.0 else 10.0
    derived_coverage = min(100.0, observed_points)
    reported_coverage = _finite(row.get("project_data_coverage"), np.nan)
    coverage = min(derived_coverage, reported_coverage) if np.isfinite(reported_coverage) and reported_coverage > 0 else derived_coverage

    return {
        "project_pipeline_score_observed": round(pipeline, 1),
        "future_fundamental_impact_score_observed": round(impact, 1),
        "project_data_coverage": round(max(0.0, min(100.0, coverage)), 1),
        "official_forward_commitment_score": round(commitment, 1),
        "official_forward_execution_score": round(execution, 1),
        "official_forward_quantification_score": round(quantification, 1),
        "official_forward_corroboration_score": round(corroboration, 1),
        "official_forward_financial_link_score": round(financial_link, 1),
        "official_forward_timing_score": round(timing, 1),
        "official_forward_strategic_score": round(strategic, 1),
        "official_forward_bridge_state": "DETERMINISTIC_MODEL_SCORE_FROM_VERIFIED_DIRECT_FACTS",
        "official_forward_score_disclaimer": "SCANNER_DERIVED_NOT_ISSUER_REPORTED_SCORE",
    }


def bridge_project_events(projects: pd.DataFrame | None, *, as_of: Any | None = None) -> pd.DataFrame:
    if not isinstance(projects, pd.DataFrame) or projects.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for raw in projects.to_dict("records"):
        ticker = _ticker(raw.get("ticker"))
        lineage = _strict_lineage(
            raw,
            url_fields=("project_source_urls", "source_url"),
            date_fields=("last_verified_at", "evidence_date", "event_date", "as_of"),
            verified_field=None,
            quorum_field="project_source_quorum_verified",
            as_of=as_of,
        )
        if not ticker or not lineage["valid"]:
            continue
        scored = _project_score(raw, lineage)
        row = dict(raw)
        row.update(scored)
        row.update({
            "ticker": ticker,
            "project_source_quorum_verified": True,
            "source_quorum_count": lineage["source_quorum_count"],
            "entity_match_verified": lineage["entity_match_verified"],
            "project_source_urls": " | ".join(lineage["urls"]),
            "last_verified_at": lineage["evidence_date"],
            "official_evidence_bridge_version": OFFICIAL_EVIDENCE_BRIDGE_VERSION,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def bridge_management_evidence(
    management_roles: pd.DataFrame | None,
    ownership_events: pd.DataFrame | None,
    corporate_events: pd.DataFrame | None,
    *,
    as_of: Any | None = None,
) -> pd.DataFrame:
    """Build issuer-alignment evidence without assigning management quality.

    A board roster increases evidence coverage but is not a positive score by
    itself.  Insider ownership is passed as an observed fact; the narrative
    engine applies its existing incentive-alignment function.
    """
    roles = management_roles if isinstance(management_roles, pd.DataFrame) else pd.DataFrame()
    ownership = ownership_events if isinstance(ownership_events, pd.DataFrame) else pd.DataFrame()
    corporate = corporate_events if isinstance(corporate_events, pd.DataFrame) else pd.DataFrame()
    tickers = set()
    for frame in (roles, ownership, corporate):
        if not frame.empty and "ticker" in frame.columns:
            tickers.update(_ticker(value) for value in frame["ticker"] if _ticker(value))
    rows: list[dict[str, Any]] = []
    for ticker in sorted(tickers):
        role_rows = []
        if not roles.empty:
            for raw in roles.loc[roles["ticker"].map(_ticker).eq(ticker)].to_dict("records"):
                lineage = _strict_lineage(
                    raw, url_fields=("source_url",), date_fields=("updated_at", "appointment_date", "rups_date", "created_at"),
                    verified_field="verified", as_of=as_of, max_age_days=1460,
                )
                if lineage["valid"]:
                    role_rows.append(raw)
        own_rows = []
        if not ownership.empty:
            for raw in ownership.loc[ownership["ticker"].map(_ticker).eq(ticker)].to_dict("records"):
                lineage = _strict_lineage(
                    raw, url_fields=("source_url",), date_fields=("report_date", "transaction_date", "created_at"),
                    verified_field="verified", as_of=as_of, max_age_days=730,
                )
                if lineage["valid"]:
                    own_rows.append(raw)
        capital_rows = []
        if not corporate.empty:
            for raw in corporate.loc[corporate["ticker"].map(_ticker).eq(ticker)].to_dict("records"):
                lineage = _strict_lineage(
                    raw, url_fields=("source_url",), date_fields=("event_date", "published_at", "created_at"),
                    verified_field="verified", as_of=as_of, max_age_days=730,
                )
                if lineage["valid"]:
                    capital_rows.append(raw)

        latest_ownership_date = pd.NaT
        for raw in own_rows:
            stamp = _timestamp(raw.get("report_date") or raw.get("transaction_date"))
            if pd.notna(stamp) and (pd.isna(latest_ownership_date) or stamp > latest_ownership_date):
                latest_ownership_date = stamp
        insider_pct = 0.0
        insider_observed = False
        for raw in own_rows:
            stamp = _timestamp(raw.get("report_date") or raw.get("transaction_date"))
            if pd.notna(latest_ownership_date) and pd.notna(stamp) and stamp.date() != latest_ownership_date.date():
                continue
            holder_type = _text(raw.get("holder_type")).upper()
            if holder_type not in {"INSIDER", "DIRECTOR", "COMMISSIONER", "MANAGEMENT", "BOARD"}:
                continue
            value = _finite(raw.get("ownership_pct_after"), np.nan)
            if np.isfinite(value):
                insider_pct += max(0.0, value)
                insider_observed = True

        coverage = min(25.0, 5.0 * len(role_rows))
        coverage += 35.0 if insider_observed else min(20.0, 5.0 * len(own_rows))
        coverage += min(40.0, 10.0 * len(capital_rows))
        if coverage <= 0.0:
            continue
        rows.append({
            "ticker": ticker,
            "management_data_coverage": round(min(100.0, coverage), 1),
            "insider_ownership_pct": round(min(100.0, insider_pct), 4) if insider_observed else np.nan,
            "management_role_count_verified": len(role_rows),
            "ownership_event_count_verified": len(own_rows),
            "capital_action_count_verified": len(capital_rows),
            "management_quality_score_observed": np.nan,
            "management_governance_flags": "",
            "management_related_party_risk": "UNKNOWN_NOT_INFERRED",
            "management_evidence_state": "DIRECT_FACTS_NO_QUALITY_SCORE_INFERENCE",
            "official_evidence_bridge_version": OFFICIAL_EVIDENCE_BRIDGE_VERSION,
        })
    return pd.DataFrame(rows)


def _corporate_event_type(value: Any) -> tuple[str, str, int]:
    event = _text(value).upper()
    if any(token in event for token in ("RIGHTS", "HMETD", "DILUTION", "PRIVATE_PLACEMENT", "EQUITY_RAISE")):
        return "DILUTION_OR_EQUITY_RAISE", "CAPITAL_STRUCTURE", -1
    if any(token in event for token in ("INSIDER_BUY", "CONTROLLER_BUY")):
        return "INSIDER_OR_CONTROLLER_BUY", "ISSUER_ALIGNMENT", 1
    if any(token in event for token in ("INSIDER_SELL", "CONTROLLER_SELL")):
        return "INSIDER_OR_CONTROLLER_SELL", "ISSUER_ALIGNMENT", -1
    if "BUYBACK" in event:
        return "BUYBACK", "ISSUER_ALIGNMENT", 1
    if "DIVIDEND" in event:
        return "DIVIDEND_OR_CAPITAL_RETURN", "CAPITAL_RETURN", 1
    if any(token in event for token in ("CAPEX", "EXPANSION", "CAPACITY")):
        return "CAPACITY_OR_EXPANSION", "GROWTH_CATALYST", 1
    if any(token in event for token in ("CONTRACT", "BACKLOG", "OFFTAKE")):
        return "PROJECT_OR_CONTRACT", "GROWTH_CATALYST", 1
    return "OTHER_MATERIAL_EVENT", "OTHER", 0


def corporate_events_to_narrative(corporate_events: pd.DataFrame | None, *, as_of: Any | None = None) -> pd.DataFrame:
    if not isinstance(corporate_events, pd.DataFrame) or corporate_events.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for raw in corporate_events.to_dict("records"):
        ticker = _ticker(raw.get("ticker"))
        lineage = _strict_lineage(
            raw, url_fields=("source_url",), date_fields=("event_date", "published_at", "created_at"),
            verified_field="verified", as_of=as_of, max_age_days=730,
        )
        if not ticker or not lineage["valid"]:
            continue
        event_type, family, sign = _corporate_event_type(raw.get("event_type"))
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), Mapping) else {}
        title = _text(metadata.get("title")) or _text(raw.get("event_type")).replace("_", " ").title()
        event_date = lineage["evidence_date"][:10]
        source_url = lineage["urls"][0]
        hostname = urlparse(source_url).hostname or ""
        materiality = float(np.clip(_finite(raw.get("materiality"), 60.0), 0.0, 100.0))
        event_id = sha256(f"OFFICIAL_CORPORATE|{ticker}|{event_type}|{event_date}|{source_url}".encode("utf-8")).hexdigest()
        rows.append({
            "narrative_event_id": event_id,
            "ticker": ticker,
            "event_date": event_date,
            "detected_at": lineage["evidence_date"],
            "event_type": event_type,
            "event_family": family,
            "headline": title,
            "summary": title,
            "source_url": source_url,
            "source_hostname": hostname,
            "registered_official_domain": hostname,
            "source_state": "OFFICIAL_VERIFIED",
            "source_present": True,
            "source_family": _text(raw.get("source_family")) or "GOVERNED_CORPORATE_EVENT",
            "source_quality_score": 95.0,
            "official_claimed": True,
            "official_verified": True,
            "materiality_score": materiality,
            "impact_direction": "POSITIVE" if sign > 0 else "NEGATIVE" if sign < 0 else "MIXED_OR_NEUTRAL",
            "impact_sign": sign,
            "financial_bridge_score": 70.0 if event_type in {"DILUTION_OR_EQUITY_RAISE", "BUYBACK", "DIVIDEND_OR_CAPITAL_RETURN", "CAPACITY_OR_EXPANSION", "PROJECT_OR_CONTRACT"} else 35.0,
            "content_hash": sha256(json.dumps({k: _text(v) for k, v in raw.items() if k != "metadata"}, sort_keys=True).encode("utf-8")).hexdigest(),
            "event_cluster_key": sha256(f"{ticker}|{event_type}|{event_date}".encode("utf-8")).hexdigest(),
            "detection_time_source": "GOVERNED_OFFICIAL_EVIDENCE",
            "entity_match_state": "VERIFIED_ENTITY_MATCH",
            "event_status": "ACTIVE",
            "requested_event_status": "ACTIVE",
            "lifecycle_evidence_state": "DIRECT_OFFICIAL_QUORUM_VERIFIED",
            "event_active": True,
            "event_evidence_state": "DIRECT_OFFICIAL_QUORUM_VERIFIED",
            "future_detection_invalid": False,
            "novelty_score": 100.0,
            "event_age_days": lineage["evidence_age_days"],
            "narrative_decay_weight": 1.0,
            "catalyst_proximity_score": 80.0,
            "event_strength_score": materiality,
            "signed_event_strength": float(sign) * materiality,
            "official_evidence_bridge_version": OFFICIAL_EVIDENCE_BRIDGE_VERSION,
        })
    return pd.DataFrame(rows)


def combine_project_management(
    cached_forward: pd.DataFrame | None,
    project_events: pd.DataFrame | None,
    management_roles: pd.DataFrame | None,
    ownership_events: pd.DataFrame | None,
    corporate_events: pd.DataFrame | None,
    *,
    as_of: Any | None = None,
) -> pd.DataFrame:
    frames = [
        frame.copy() for frame in (
            cached_forward if isinstance(cached_forward, pd.DataFrame) else pd.DataFrame(),
            bridge_project_events(project_events, as_of=as_of),
            bridge_management_evidence(management_roles, ownership_events, corporate_events, as_of=as_of),
        ) if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


__all__ = [
    "OFFICIAL_EVIDENCE_BRIDGE_VERSION",
    "bridge_project_events",
    "bridge_management_evidence",
    "corporate_events_to_narrative",
    "combine_project_management",
]
