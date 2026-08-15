from __future__ import annotations

"""Point-in-time narrative intelligence for IDX Multibagger and Core Swing.

The engine is deliberately deterministic and source-grounded.  It does not
invent a story when evidence is absent, does not infer beneficial ownership
from broker codes, and keeps attention/retail-adoption fields as market-data
proxies unless direct attention data is supplied.
"""

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping
import json
import math
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

def build_emir_method_profile(**_kwargs: Any) -> dict[str, Any]:
    """Compatibility record for v7 database readers.

    V8 decomposes the public Emir-style framework into separate, auditable
    narrative, issuer-action, market and flow pillars.  The former composite
    Emir score is therefore removed rather than calculated and added again.
    """
    return {
        "emir_method_score": np.nan,
        "emir_method_coverage_pct": 0.0,
        "emir_method_score_state": "REMOVED_FROM_V8_PRODUCTION",
        "emir_method_state": "DISABLED",
        "emir_method_production_eligible": False,
        "emir_method_reliability_pct": 0.0,
        "emir_selection_reason": (
            "Composite Emir score removed; v8 scores its disjoint evidence "
            "families directly."
        ),
        "emir_risk_flags": "COMPOSITE_REMOVED_NO_DOUBLE_COUNTING",
        "emir_position_cap_pct": 0.0,
        "emir_growth_rank_adjustment": 0.0,
        "emir_turnaround_rank_adjustment": 0.0,
        "emir_swing_rank_adjustment": 0.0,
    }


NARRATIVE_ENGINE_VERSION = "2.0.0-pure-event-issuer-action-separation"
NARRATIVE_EVENT_SCHEMA_VERSION = "narrative_events_v2"
NARRATIVE_OUTCOME_SCHEMA_VERSION = "narrative_outcomes_v2"


@dataclass(frozen=True)
class NarrativeConfig:
    enabled: bool = True
    event_lookback_days: int = 180
    half_life_days: float = 45.0
    min_conversion_events: int = 20
    roundtrip_cost_pct: float = 0.0065
    growth_max_adjustment_points: float = 5.0
    turnaround_max_adjustment_points: float = 7.0
    swing_max_adjustment_points: float = 5.0
    emir_growth_max_adjustment_points: float = 14.0
    emir_turnaround_max_adjustment_points: float = 16.0
    emir_swing_max_adjustment_points: float = 18.0

    @classmethod
    def from_scan_config(cls, config: Any | None) -> "NarrativeConfig":
        if config is None:
            return cls()

        def number(name: str, default: float) -> float:
            return _finite(getattr(config, name, default), default)

        return cls(
            enabled=bool(getattr(config, "narrative_enabled", True)),
            event_lookback_days=max(
                30, int(number("narrative_event_lookback_days", 180)),
            ),
            half_life_days=max(
                7.0, number("narrative_half_life_days", 45.0),
            ),
            min_conversion_events=max(
                5, int(number("narrative_min_conversion_events", 20)),
            ),
            roundtrip_cost_pct=max(
                0.0, number("narrative_roundtrip_cost_pct", 0.0065),
            ),
            growth_max_adjustment_points=max(
                0.0, min(
                    12.0,
                    number("narrative_growth_max_adjustment_points", 5.0),
                ),
            ),
            turnaround_max_adjustment_points=max(
                0.0, min(
                    15.0,
                    number(
                        "narrative_turnaround_max_adjustment_points", 7.0,
                    ),
                ),
            ),
            swing_max_adjustment_points=max(
                0.0, min(
                    12.0,
                    number("narrative_swing_max_adjustment_points", 5.0),
                ),
            ),
            emir_growth_max_adjustment_points=max(
                0.0, min(20.0, number("emir_growth_max_adjustment_points", 14.0)),
            ),
            emir_turnaround_max_adjustment_points=max(
                0.0, min(22.0, number("emir_turnaround_max_adjustment_points", 16.0)),
            ),
            emir_swing_max_adjustment_points=max(
                0.0, min(25.0, number("emir_swing_max_adjustment_points", 18.0)),
            ),
        )


_EVENT_RULES: tuple[tuple[str, str, tuple[str, ...], float, int], ...] = (
    (
        "DEFAULT_OR_SOLVENCY",
        "GOVERNANCE_RISK",
        (
            "GAGAL BAYAR", "DEFAULT", "PAILIT", "BANKRUPTCY", "PKPU",
            "NEGATIVE EQUITY", "EKUITAS NEGATIF",
        ),
        100.0,
        -1,
    ),
    (
        "FRAUD_OR_INVESTIGATION",
        "GOVERNANCE_RISK",
        (
            "FRAUD", "KORUPSI", "CORRUPTION", "PENIPUAN",
            "PENYIDIKAN", "INVESTIGATION", "SUAP", "BRIBERY",
        ),
        100.0,
        -1,
    ),
    (
        "SUSPENSION_OR_WATCHLIST",
        "REGULATORY",
        (
            "SUSPENSI", "SUSPENSION", "PEMANTAUAN KHUSUS",
            "SPECIAL MONITORING", "DELISTING",
        ),
        92.0,
        -1,
    ),
    (
        "SHARE_PLEDGE",
        "ISSUER_ALIGNMENT",
        (
            "MENJAMINKAN SAHAM", "PLEDGED SHARES", "SHARE PLEDGE",
            "JAMINAN SAHAM",
        ),
        88.0,
        -1,
    ),
    (
        "INSIDER_OR_CONTROLLER_SELL",
        "ISSUER_ALIGNMENT",
        (
            "PENJUALAN SAHAM DIREKSI", "PENJUALAN SAHAM KOMISARIS",
            "INSIDER SELL", "CONTROLLER SELL", "PEMEGANG SAHAM PENGENDALI MENJUAL",
        ),
        82.0,
        -1,
    ),
    (
        "DILUTION_OR_EQUITY_RAISE",
        "CAPITAL_STRUCTURE",
        (
            "RIGHTS ISSUE", "HMETD", "PRIVATE PLACEMENT",
            "PENAMBAHAN MODAL", "DILUSI",
        ),
        78.0,
        -1,
    ),
    (
        "LEGAL_OR_GOVERNANCE",
        "GOVERNANCE_RISK",
        (
            "GUGATAN", "LAWSUIT", "RESTATEMENT", "PENYAJIAN KEMBALI",
            "DISCLAIMER OPINION", "ADVERSE OPINION", "SANKSI",
        ),
        84.0,
        -1,
    ),
    (
        "INSIDER_OR_CONTROLLER_BUY",
        "ISSUER_ALIGNMENT",
        (
            "PEMBELIAN SAHAM DIREKSI", "PEMBELIAN SAHAM KOMISARIS",
            "INSIDER BUY", "CONTROLLER BUY", "PEMEGANG SAHAM PENGENDALI MEMBELI",
        ),
        86.0,
        1,
    ),
    (
        "BUYBACK",
        "ISSUER_ALIGNMENT",
        ("BUYBACK", "PEMBELIAN KEMBALI SAHAM"),
        82.0,
        1,
    ),
    (
        "STRATEGIC_INVESTOR_OR_MA",
        "CORPORATE_ACTION",
        (
            "STRATEGIC INVESTOR", "INVESTOR STRATEGIS", "AKUISISI",
            "ACQUISITION", "MERGER", "JOINT VENTURE",
        ),
        82.0,
        1,
    ),
    (
        "PROJECT_OR_CONTRACT",
        "GROWTH_CATALYST",
        (
            "KONTRAK", "CONTRACT", "PROYEK", "PROJECT", "OFFTAKE",
            "ORDER BOOK", "TENDER",
        ),
        74.0,
        1,
    ),
    (
        "CAPACITY_OR_EXPANSION",
        "GROWTH_CATALYST",
        (
            "EKSPANSI", "EXPANSION", "KAPASITAS", "CAPACITY",
            "SMELTER", "PABRIK", "PLANT", "COMMISSIONING",
            "COMMERCIAL OPERATION", "BEROPERASI",
        ),
        76.0,
        1,
    ),
    (
        "EARNINGS_INFLECTION",
        "EARNINGS",
        (
            "LABA", "PROFIT", "MARGIN", "EBITDA", "REVENUE",
            "PENDAPATAN", "TURNAROUND", "PEMULIHAN",
        ),
        70.0,
        1,
    ),
    (
        "PRODUCT_OR_NEW_MARKET",
        "GROWTH_CATALYST",
        (
            "PRODUK BARU", "NEW PRODUCT", "PASAR BARU", "NEW MARKET",
            "PELUNCURAN", "LAUNCH",
        ),
        66.0,
        1,
    ),
    (
        "DIVIDEND_OR_CAPITAL_RETURN",
        "CAPITAL_RETURN",
        ("DIVIDEN", "DIVIDEND"),
        58.0,
        1,
    ),
    (
        "REGULATION_OR_POLICY",
        "REGULATORY",
        (
            "REGULASI", "REGULATION", "KEBIJAKAN", "POLICY",
            "LARANGAN EKSPOR", "EXPORT BAN", "INSENTIF",
        ),
        65.0,
        0,
    ),
    (
        "COMMODITY_OR_MACRO",
        "MACRO",
        (
            "HARGA EMAS", "GOLD PRICE", "HARGA NIKEL", "NICKEL PRICE",
            "HARGA BATUBARA", "COAL PRICE", "SUKU BUNGA", "INTEREST RATE",
        ),
        55.0,
        0,
    ),
)

_POSITIVE_ALIGNMENT = {
    "INSIDER_OR_CONTROLLER_BUY": 1.00,
    "BUYBACK": 0.90,
    "CAPACITY_OR_EXPANSION": 0.55,
    "PROJECT_OR_CONTRACT": 0.50,
    "STRATEGIC_INVESTOR_OR_MA": 0.45,
    "EARNINGS_INFLECTION": 0.40,
    "DIVIDEND_OR_CAPITAL_RETURN": 0.30,
}
_NEGATIVE_ALIGNMENT = {
    "INSIDER_OR_CONTROLLER_SELL": 0.90,
    "SHARE_PLEDGE": 0.95,
    "DILUTION_OR_EQUITY_RAISE": 0.65,
    "DEFAULT_OR_SOLVENCY": 1.00,
    "FRAUD_OR_INVESTIGATION": 1.00,
    "LEGAL_OR_GOVERNANCE": 0.85,
    "SUSPENSION_OR_WATCHLIST": 0.85,
}
_FINANCIAL_BRIDGE_TERMS = (
    "PENDAPATAN", "REVENUE", "LABA", "PROFIT", "EBITDA", "MARGIN",
    "ARUS KAS", "CASH FLOW", "FCF", "KAPASITAS", "CAPACITY", "CAPEX",
    "KONTRAK", "CONTRACT", "ORDER BOOK", "OFFTAKE", "VOLUME PRODUKSI",
    "PRODUCTION VOLUME", "COST SAVING", "EFISIENSI",
)
_OFFICIAL_DOMAINS = (
    "idx.id", "idx.co.id", "ojk.go.id", "ksei.co.id", "bi.go.id",
)
_GENERIC_NEWS_DOMAINS = (
    "google.com", "google.co.id", "finance.yahoo.com", "yahoo.com",
    "bloomberg.com", "reuters.com", "cnbcindonesia.com", "kontan.co.id",
    "bisnis.com", "detik.com", "kompas.com", "investing.com",
)
_AMBIGUOUS_TICKERS = {
    "META", "TECH", "BOSS", "CASH", "DATA", "EDGE", "FAST", "FIRE",
    "GOLD", "GOOD", "HOPE", "ICON", "IDEA", "KING", "LIFE", "LINK",
    "LIVE", "MAXI", "NICE", "PACK", "PLAN", "REAL", "SAFE", "STAR",
    "TRUE", "WIFI", "ZONE",
}

_ADVERSE_PHRASE_RULES: tuple[
    tuple[str, str, tuple[str, ...], float], ...
] = (
    (
        "PROJECT_OR_CONTRACT_CANCELLED",
        "GROWTH_CATALYST",
        (
            "KONTRAK DIBATALKAN", "PEMBATALAN KONTRAK", "CONTRACT CANCELLED",
            "CONTRACT TERMINATED", "TERMINASI KONTRAK", "OFFTAKE DIBATALKAN",
        ),
        86.0,
    ),
    (
        "PROJECT_DELAY_OR_COST_OVERRUN",
        "GROWTH_CATALYST",
        (
            "PROYEK DITUNDA", "PROJECT DELAY", "COMMISSIONING DITUNDA",
            "COD MUNDUR", "COST OVERRUN", "CAPEX OVERRUN", "BIAYA MEMBENGKAK",
        ),
        84.0,
    ),
    (
        "EARNINGS_DETERIORATION",
        "EARNINGS",
        (
            "LABA TURUN", "LABA ANJLOK", "LABA MEROSOT", "RUGI BERSIH",
            "PENDAPATAN TURUN", "REVENUE DECLINE", "PROFIT DECLINE",
            "MARGIN TURUN", "MARGIN TERTEKAN", "EBITDA TURUN",
            "GUIDANCE CUT", "GUIDANCE DITURUNKAN",
        ),
        78.0,
    ),
)


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _first_nonempty_value(*values: Any) -> Any:
    """Return the first scalar with a non-empty, non-null text representation."""
    for value in values:
        if _text(value):
            return value
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return _text(value).lower() in {
        "1", "true", "t", "yes", "y", "on", "verified", "official",
    }


def _ticker(value: Any) -> str:
    text = _text(value).upper().replace(" ", "")
    if not text:
        return ""
    return text if text.endswith(".JK") else f"{text}.JK"


def _timestamp(value: Any, default: Any | None = None) -> pd.Timestamp:
    stamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(stamp):
        stamp = pd.to_datetime(default, errors="coerce", utc=True)
    if pd.isna(stamp):
        stamp = pd.Timestamp.now(tz="UTC")
    return pd.Timestamp(stamp)


def _date_text(value: Any, default: Any | None = None) -> str:
    return _timestamp(value, default).date().isoformat()


def _normalised_words(value: Any) -> set[str]:
    text = re.sub(r"[^A-Z0-9 ]+", " ", _text(value).upper())
    return {
        token for token in text.split()
        if len(token) >= 3 and token not in {
            "DAN", "DENGAN", "UNTUK", "YANG", "THE", "AND", "SERTA",
            "SAHAM", "EMITEN", "PERSEROAN", "TBK",
        }
    }


def _stable_hash(parts: Iterable[Any], length: int = 40) -> str:
    body = "|".join(_text(part) for part in parts)
    return sha256(body.encode("utf-8")).hexdigest()[:length]


def _hostname(url: Any) -> str:
    text = _text(url)
    if not text:
        return ""
    try:
        parsed = urlparse(text)
    except ValueError:
        return ""
    host = (parsed.hostname or "").lower().rstrip(".")
    return host


def _host_matches(host: str, domain: str) -> bool:
    clean = str(domain).lower().strip().lstrip(".")
    return bool(host and clean and (host == clean or host.endswith("." + clean)))


def _is_https_url(url: Any) -> bool:
    text = _text(url)
    if not text:
        return False
    try:
        parsed = urlparse(text)
    except ValueError:
        return False
    return parsed.scheme.lower() == "https" and bool(parsed.hostname)


def _source_url_candidates(value: Any) -> list[str]:
    """Normalise URL ledgers from list/JSON/pipe/bullet separated fields."""
    values: list[Any]
    if isinstance(value, (list, tuple, set)):
        values = list(value)
    elif isinstance(value, str):
        text = value.strip()
        values = []
        if text.startswith("["):
            try:
                decoded = json.loads(text)
                if isinstance(decoded, list):
                    values.extend(decoded)
            except Exception:
                pass
        if not values:
            values = re.split(r"\s*(?:\||•|;|\n)\s*", text)
    else:
        values = [value]
    output: list[str] = []
    for item in values:
        url = _text(item).strip("'\"[]() ")
        if _is_https_url(url) and url not in output:
            output.append(url)
    return output


def _source_quality(
    url: Any,
    family: Any,
    official: Any,
    expected_domain: Any = "",
) -> tuple[float, bool]:
    source_url = _text(url)
    host = _hostname(source_url)
    source_family = _text(family).upper()
    domain_official = any(_host_matches(host, domain) for domain in _OFFICIAL_DOMAINS)
    issuer_official = any(
        token in source_family
        for token in ("ISSUER_IR", "ANNUAL_REPORT", "PUBLIC_EXPOSE")
    )
    generic_news = any(_host_matches(host, domain) for domain in _GENERIC_NEWS_DOMAINS)
    issuer_domain_plausible = bool(
        issuer_official and host and not generic_news and _is_https_url(source_url)
    )
    registered_domain = _hostname(expected_domain) or _text(expected_domain).lower()
    issuer_domain_match = bool(
        issuer_domain_plausible
        and registered_domain
        and _host_matches(host, registered_domain)
    )
    # An HTTPS URL on a known regulator/exchange host verifies its own source
    # provenance.  Issuer-domain evidence still requires both a registered
    # domain match and an explicit official claim.
    verified = bool(domain_official or (_truthy(official) and issuer_domain_match))
    if domain_official:
        return (95.0 if verified else 88.0, verified)
    if issuer_domain_plausible:
        return (86.0 if verified else 75.0, verified)
    if _is_https_url(source_url):
        return (62.0, False)
    if source_family in {"FUNDAMENTAL_SNAPSHOT", "PROJECT_REVIEW"}:
        return (55.0, False)
    return (35.0, False)


def _classify_event(
    text: Any,
    explicit_type: Any = "",
    explicit_direction: Any = "",
    explicit_materiality: Any = np.nan,
) -> tuple[str, str, float, int]:
    body = _text(text).upper()
    explicit = re.sub(r"[^A-Z0-9_]+", "_", _text(explicit_type).upper())
    direction_text = _text(explicit_direction).upper()
    direction = (
        1 if direction_text in {"POSITIVE", "BULLISH", "UP", "1"}
        else -1 if direction_text in {"NEGATIVE", "BEARISH", "DOWN", "-1"}
        else 0
    )
    materiality = _finite(explicit_materiality, np.nan)
    if np.isfinite(materiality) and materiality <= 1.0:
        materiality *= 100.0
    materiality_text = _text(explicit_materiality).upper()
    if not np.isfinite(materiality):
        materiality = {
            "CRITICAL": 95.0, "HIGH": 80.0, "MEDIUM": 60.0, "LOW": 35.0,
        }.get(materiality_text, np.nan)
    adverse = next(
        (
            (event_type, family, base_materiality, -1)
            for event_type, family, terms, base_materiality
            in _ADVERSE_PHRASE_RULES
            if any(term in body for term in terms)
        ),
        None,
    )
    matched = adverse or next(
        (
            (event_type, family, base_materiality, base_direction)
            for event_type, family, terms, base_materiality, base_direction
            in _EVENT_RULES
            if any(term in body for term in terms)
        ),
        None,
    )
    if matched is None:
        event_type, family, base_materiality, base_direction = (
            explicit or "OTHER_MATERIAL_EVENT",
            "OTHER",
            45.0,
            0,
        )
    else:
        event_type, family, base_materiality, base_direction = matched
    if explicit:
        event_type = explicit
        family_match = next(
            (rule[1] for rule in _EVENT_RULES if rule[0] == explicit),
            family,
        )
        family = family_match
    final_direction = (
        -1 if adverse is not None
        else direction if direction != 0 else base_direction
    )
    return (
        event_type,
        family,
        max(
            0.0, min(
                100.0,
                materiality if np.isfinite(materiality)
                else base_materiality,
            ),
        ),
        final_direction,
    )


def _financial_bridge_score(
    text: Any,
    event_type: str,
    explicit: Any = np.nan,
    project_row: Mapping[str, Any] | None = None,
) -> float:
    observed = _finite(explicit, np.nan)
    if np.isfinite(observed):
        if observed <= 1.0:
            observed *= 100.0
        return max(0.0, min(100.0, observed))
    body = _text(text).upper()
    term_count = sum(term in body for term in _FINANCIAL_BRIDGE_TERMS)
    score = 30.0 + min(45.0, 12.0 * term_count)
    if re.search(r"(?:RP|IDR|\$|USD)\s*[\d.,]+", body):
        score += 12.0
    if re.search(r"\d+(?:[.,]\d+)?\s*%", body):
        score += 8.0
    if event_type in {
        "PROJECT_OR_CONTRACT", "CAPACITY_OR_EXPANSION",
        "EARNINGS_INFLECTION",
    }:
        score += 8.0
    if project_row:
        direct_values = (
            _finite(project_row.get("project_expected_revenue_idr"), np.nan),
            _finite(project_row.get("project_expected_ebitda_idr"), np.nan),
            _finite(project_row.get("project_capex_idr"), np.nan),
        )
        if any(np.isfinite(value) and value > 0 for value in direct_values):
            score = max(score, 82.0)
    return max(0.0, min(100.0, score))


def _event_record(
    *,
    ticker: Any,
    headline: Any,
    summary: Any = "",
    source_url: Any = "",
    source_family: Any = "",
    event_date: Any = None,
    detected_at: Any = None,
    event_type: Any = "",
    impact_direction: Any = "",
    materiality: Any = np.nan,
    official_verified: Any = False,
    financial_bridge_score: Any = np.nan,
    project_row: Mapping[str, Any] | None = None,
    detection_time_source: str = "CURRENT_SCAN",
    event_status: Any = "ACTIVE",
    resolved_at: Any = None,
    supersedes_event_id: Any = "",
    resolution_source_url: Any = "",
    entity_match_state: Any = "",
    official_domain: Any = "",
) -> dict[str, Any] | None:
    name = _ticker(ticker)
    title = _text(headline)
    detail = _text(summary)
    if not name or not title:
        return None
    event_type_value, family, materiality_score, direction = _classify_event(
        f"{title} {detail}", event_type, impact_direction, materiality,
    )
    detected = _timestamp(detected_at)
    event_day = _date_text(event_date, detected)
    source_text = _text(source_url)
    source_quality, verified = _source_quality(
        source_url, source_family, official_verified, official_domain,
    )
    if (
        detection_time_source.startswith("MANUAL")
        and not any(
            _host_matches(_hostname(source_url), domain)
            for domain in _OFFICIAL_DOMAINS
        )
    ):
        # A manual row cannot certify its own issuer-domain provenance.
        verified = False
        source_quality = min(source_quality, 75.0)
    source_present = bool(_is_https_url(source_text))
    source_state = (
        "SOURCE_IDENTIFIED" if source_present else "MISSING_SOURCE"
    )
    lifecycle = _text(event_status).upper() or "ACTIVE"
    if lifecycle not in {
        "ACTIVE", "RESOLVED", "SUPERSEDED", "REVERSED", "DISPUTED",
    }:
        lifecycle = "ACTIVE"
    requested_lifecycle = lifecycle
    resolution_url = _text(resolution_source_url)
    lifecycle_evidence_state = "ACTIVE_EVENT"
    if lifecycle in {"RESOLVED", "SUPERSEDED", "REVERSED"}:
        _, resolution_verified = _source_quality(
            resolution_url,
            source_family,
            True,
            official_domain,
        )
        if resolution_verified:
            lifecycle_evidence_state = "RESOLUTION_SOURCE_VERIFIED"
        else:
            # A stale negative event cannot be cleared by an unsupported flag.
            lifecycle = "ACTIVE"
            lifecycle_evidence_state = (
                "RESOLUTION_SOURCE_UNVERIFIED_KEPT_ACTIVE"
                if _is_https_url(resolution_url)
                else "RESOLUTION_SOURCE_MISSING_KEPT_ACTIVE"
            )
    elif lifecycle == "DISPUTED":
        lifecycle_evidence_state = "DISPUTED_STILL_RISK_ACTIVE"
    entity_state = _text(entity_match_state).upper() or "NOT_APPLICABLE"
    bridge = _financial_bridge_score(
        f"{title} {detail}",
        event_type_value,
        financial_bridge_score,
        project_row,
    )
    canonical = re.sub(r"\s+", " ", f"{title} {detail}".upper()).strip()
    content_hash = _stable_hash((name, canonical, _text(source_url)), 64)
    event_id = _stable_hash(
        (
            name, event_type_value, event_day,
            re.sub(r"[^A-Z0-9]+", "", title.upper())[:120],
            _text(source_url),
        ),
        48,
    )
    return {
        "narrative_event_id": event_id,
        "ticker": name,
        "event_date": event_day,
        "detected_at": detected.isoformat(),
        "event_type": event_type_value,
        "event_family": family,
        "headline": title[:500],
        "summary": detail[:1500],
        "source_url": _text(source_url)[:1000],
        "source_hostname": _hostname(source_url),
        "registered_official_domain": (
            _hostname(official_domain) or _text(official_domain).lower()
        ),
        "source_state": source_state,
        "source_present": source_present,
        "source_family": _text(source_family).upper() or "UNKNOWN",
        "source_quality_score": round(source_quality, 1),
        "official_claimed": bool(_truthy(official_verified)),
        "official_verified": bool(verified),
        "materiality_score": round(materiality_score, 1),
        "impact_direction": (
            "POSITIVE" if direction > 0
            else "NEGATIVE" if direction < 0 else "MIXED_OR_NEUTRAL"
        ),
        "impact_sign": int(direction),
        "financial_bridge_score": round(bridge, 1),
        "content_hash": content_hash,
        "event_cluster_key": _stable_hash(
            (name, family, event_day[:7], sorted(_normalised_words(title))),
            40,
        ),
        "detection_time_source": detection_time_source,
        "entity_match_state": entity_state,
        "event_status": lifecycle,
        "requested_event_status": requested_lifecycle,
        "lifecycle_evidence_state": lifecycle_evidence_state,
        "resolved_at": (
            _timestamp(resolved_at).isoformat()
            if _text(resolved_at) and lifecycle in {
                "RESOLVED", "SUPERSEDED", "REVERSED",
            }
            else ""
        ),
        "supersedes_event_id": _text(supersedes_event_id),
        "resolution_source_url": resolution_url[:1000],
        "narrative_engine_version": NARRATIVE_ENGINE_VERSION,
    }


def parse_narrative_event_csv(
    source: bytes | Any | pd.DataFrame,
) -> pd.DataFrame:
    """Parse an optional point-in-time event ledger without upgrading trust."""
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    elif isinstance(source, bytes):
        from io import BytesIO
        frame = pd.read_csv(BytesIO(source))
    else:
        frame = pd.read_csv(source)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = {"ticker", "event_date", "headline", "source_url"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Narrative Event CSV kurang kolom: " + ", ".join(missing)
        )
    rows: list[dict[str, Any]] = []
    for raw in frame.to_dict("records"):
        record = _event_record(
            ticker=raw.get("ticker"),
            headline=raw.get("headline"),
            summary=raw.get("summary", ""),
            source_url=raw.get("source_url"),
            source_family=raw.get("source_family", "MANUAL_RESEARCH"),
            event_date=raw.get("event_date"),
            detected_at=raw.get("detected_at") or pd.Timestamp.now(tz="UTC"),
            event_type=raw.get("event_type", ""),
            impact_direction=raw.get("impact_direction", ""),
            materiality=raw.get("materiality_score", raw.get("materiality")),
            official_verified=raw.get("official_verified", False),
            financial_bridge_score=raw.get("financial_bridge_score"),
            detection_time_source="MANUAL_POINT_IN_TIME_IMPORT",
            event_status=raw.get("event_status", "ACTIVE"),
            resolved_at=raw.get("resolved_at"),
            supersedes_event_id=raw.get("supersedes_event_id", ""),
            resolution_source_url=raw.get("resolution_source_url", ""),
            entity_match_state=raw.get("entity_match_state", "MANUAL_RESEARCH"),
            official_domain=raw.get("official_domain", ""),
        )
        if record is not None:
            rows.append(record)
    return pd.DataFrame(rows)


def _events_from_news(news_review: pd.DataFrame | None) -> list[dict[str, Any]]:
    if news_review is None or news_review.empty:
        return []
    rows: list[dict[str, Any]] = []
    for raw in news_review.to_dict("records"):
        ticker = raw.get("ticker")
        detected_at = raw.get("news_reviewed_at")
        provider = raw.get("news_provider", "NEWS_REVIEW")
        payload = raw.get("narrative_events_json")
        items: list[Mapping[str, Any]] = []
        if isinstance(payload, str) and payload.strip():
            try:
                decoded = json.loads(payload)
                if isinstance(decoded, list):
                    items = [item for item in decoded if isinstance(item, Mapping)]
            except Exception:
                items = []
        elif isinstance(payload, list):
            items = [item for item in payload if isinstance(item, Mapping)]
        if not items:
            titles = [
                item.strip()
                for item in _text(raw.get("catalyst_summary")).split(" | ")
                if item.strip()
            ]
            urls = [
                item.strip()
                for item in _text(raw.get("news_sources")).split(" | ")
            ]
            items = [
                {
                    "headline": title,
                    "source_url": urls[index] if index < len(urls) else "",
                    "event_date": detected_at,
                }
                for index, title in enumerate(titles)
            ]
        for item in items:
            body = " ".join(
                part for part in (
                    _text(item.get("headline") or item.get("title")),
                    _text(item.get("summary")),
                ) if part
            ).upper()
            symbol = _ticker(ticker).replace(".JK", "")
            issuer_terms = [
                _text(item.get("issuer_name")),
                _text(item.get("company_name")),
                _text(raw.get("issuer_name")),
                _text(raw.get("company_name")),
            ]
            explicit_entity = _truthy(item.get("entity_verified"))
            issuer_match = any(
                term and term.upper() in body for term in issuer_terms
            )
            symbol_match = bool(symbol and re.search(
                rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])", body,
            ))
            if explicit_entity or issuer_match:
                entity_state = "ENTITY_VERIFIED"
            elif symbol_match and symbol not in _AMBIGUOUS_TICKERS:
                entity_state = "TICKER_CONTEXT_MATCH"
            elif symbol in _AMBIGUOUS_TICKERS:
                entity_state = "AMBIGUOUS_TICKER_UNVERIFIED"
            else:
                entity_state = "PROVIDER_QUERY_CONTEXT_ONLY"
            record = _event_record(
                ticker=ticker,
                headline=item.get("headline") or item.get("title"),
                summary=item.get("summary", ""),
                source_url=item.get("source_url") or item.get("url"),
                source_family=item.get("source_family") or provider,
                event_date=item.get("event_date") or item.get("published_at"),
                detected_at=item.get("detected_at") or detected_at,
                event_type=item.get("event_type", ""),
                impact_direction=item.get(
                    "impact_direction", item.get("sentiment", ""),
                ),
                materiality=item.get(
                    "materiality_score", item.get("materiality"),
                ),
                official_verified=item.get("official_verified", False),
                financial_bridge_score=item.get("financial_bridge_score"),
                detection_time_source="AUTOMATIC_NEWS_REVIEW",
                entity_match_state=entity_state,
                official_domain=(
                    item.get("official_domain")
                    or raw.get("official_domain")
                    or raw.get("issuer_official_domain")
                ),
            )
            if record is not None:
                rows.append(record)
    return rows


def _events_from_projects(
    project_management: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    if project_management is None or project_management.empty:
        return []
    rows: list[dict[str, Any]] = []
    for raw in project_management.to_dict("records"):
        name = (
            _text(raw.get("project_name"))
            or _text(raw.get("project_names"))
        )
        if not name:
            continue
        stage = _text(raw.get("project_stage")).upper()
        title = f"{name} — tahap {stage or 'belum terverifikasi'}"
        summary = " | ".join(
            part for part in (
                f"completion {_finite(raw.get('project_completion_pct'), np.nan):.1f}%"
                if np.isfinite(_finite(raw.get("project_completion_pct"), np.nan))
                else "",
                f"funding {_finite(raw.get('project_funding_secured_pct'), np.nan):.1f}%"
                if np.isfinite(_finite(raw.get("project_funding_secured_pct"), np.nan))
                else "",
                _text(raw.get("project_execution_flags")),
            )
            if part
        )
        source_urls: list[str] = []
        for source_value in (
            raw.get("project_source_urls"), raw.get("source_url"),
        ):
            for candidate in _source_url_candidates(source_value):
                if candidate not in source_urls:
                    source_urls.append(candidate)
        source_url = source_urls[0] if source_urls else ""
        event_type = (
            "CAPACITY_OR_EXPANSION"
            if any(
                term in name.upper()
                for term in ("EXPANS", "KAPASITAS", "SMELTER", "PABRIK", "PLANT")
            )
            else "PROJECT_OR_CONTRACT"
        )
        record = _event_record(
            ticker=raw.get("ticker"),
            headline=title,
            summary=summary,
            source_url=source_url,
            source_family=(
                _first_nonempty_value(
                    raw.get("project_source_families"), "PROJECT_REVIEW",
                )
            ),
            event_date=_first_nonempty_value(
                raw.get("event_date"), raw.get("last_verified_at"),
            ),
            detected_at=_first_nonempty_value(
                raw.get("last_verified_at"), raw.get("as_of"),
            ),
            event_type=event_type,
            impact_direction=(
                "NEGATIVE"
                if any(
                    token in _text(raw.get("project_execution_flags")).upper()
                    for token in ("CRITICAL", "DELAY", "COST OVERRUN")
                )
                else "POSITIVE"
            ),
            materiality=(
                88.0 if stage in {"COMMISSIONING", "OPERATING"}
                else 76.0 if stage in {"CONSTRUCTION", "FINANCING"}
                else 58.0
            ),
            official_verified=raw.get("project_source_quorum_verified", False),
            project_row=raw,
            detection_time_source="PROJECT_FORWARD_REVIEW",
            official_domain=(
                raw.get("official_domain")
                or raw.get("issuer_official_domain")
            ),
        )
        if record is not None:
            rows.append(record)
    return rows


def _events_from_fundamentals(
    fundamentals: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    if fundamentals is None or fundamentals.empty:
        return []
    rows: list[dict[str, Any]] = []
    for raw in fundamentals.to_dict("records"):
        inflection = _finite(raw.get("fundamental_inflection_score"), np.nan)
        coverage = _finite(
            raw.get("fundamental_inflection_coverage_pct"), 0.0,
        )
        if not (np.isfinite(inflection) and coverage >= 50.0):
            continue
        revenue = _finite(raw.get("revenue_growth"), np.nan)
        earnings = _finite(raw.get("earnings_growth"), np.nan)
        statement_reference = _first_nonempty_value(
            raw.get("statement_date"),
            raw.get("latest_statement_date"),
            raw.get("fundamental_history_latest_period"),
            raw.get("period_end"),
        )
        headline = (
            f"Fundamental inflection {inflection:.1f}/100 "
            f"pada laporan {_text(statement_reference) or 'periode belum tercatat'}"
        )
        summary = " | ".join(
            part for part in (
                f"revenue growth {100.0 * revenue:.1f}%"
                if np.isfinite(revenue) else "",
                f"earnings growth {100.0 * earnings:.1f}%"
                if np.isfinite(earnings) else "",
                f"coverage {coverage:.1f}%",
            )
            if part
        )
        source_urls: list[str] = []
        for source_value in (
            raw.get("fundamental_source_urls"),
            raw.get("fundamental_official_source_urls"),
            raw.get("source_url"),
        ):
            for candidate in _source_url_candidates(source_value):
                if candidate not in source_urls:
                    source_urls.append(candidate)
        record = _event_record(
            ticker=raw.get("ticker"),
            headline=headline,
            summary=summary,
            source_url=source_urls[0] if source_urls else "",
            source_family=_first_nonempty_value(
                raw.get("fundamental_source_families"),
                "FUNDAMENTAL_SNAPSHOT",
            ),
            event_date=statement_reference,
            detected_at=_first_nonempty_value(
                raw.get("fundamental_fetched_at"),
                raw.get("database_source_checked_at"),
                raw.get("history_fetched_at"),
            ),
            event_type="EARNINGS_INFLECTION",
            impact_direction=(
                "POSITIVE" if inflection >= 60.0
                else "NEGATIVE" if inflection < 40.0
                else "MIXED"
            ),
            materiality=55.0 + 0.35 * abs(inflection - 50.0),
            official_verified=raw.get("fundamental_official_verified", False),
            financial_bridge_score=min(100.0, 60.0 + 0.35 * coverage),
            detection_time_source="FUNDAMENTAL_POINT_IN_TIME",
            official_domain=(
                raw.get("official_domain")
                or raw.get("issuer_official_domain")
            ),
        )
        if record is not None:
            rows.append(record)
    return rows


def _events_from_market_status(
    market_status: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    if market_status is None or market_status.empty:
        return []
    rows: list[dict[str, Any]] = []
    for raw in market_status.to_dict("records"):
        flags: list[str] = []
        if _truthy(raw.get("suspended")):
            flags.append("SUSPENSI")
        if _truthy(raw.get("special_monitoring")) or _truthy(raw.get("fca")):
            flags.append("PEMANTAUAN KHUSUS")
        if not flags:
            continue
        record = _event_record(
            ticker=raw.get("ticker"),
            headline="Status IDX: " + ", ".join(flags),
            summary=_text(raw.get("special_notation")),
            source_url=raw.get("market_status_source"),
            source_family="IDX_MARKET_STATUS",
            event_date=raw.get("market_status_asof"),
            detected_at=raw.get("market_status_asof"),
            event_type="SUSPENSION_OR_WATCHLIST",
            impact_direction="NEGATIVE",
            materiality=95.0 if "SUSPENSI" in flags else 88.0,
            official_verified=raw.get("market_status_verified", False),
            financial_bridge_score=35.0,
            detection_time_source="IDX_MARKET_STATUS",
        )
        if record is not None:
            rows.append(record)
    return rows


def _normalise_existing_events(
    existing: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    if existing is None or existing.empty:
        return []
    rows: list[dict[str, Any]] = []
    for raw in existing.to_dict("records"):
        record = _event_record(
            ticker=raw.get("ticker"),
            headline=raw.get("headline"),
            summary=raw.get("summary", ""),
            source_url=raw.get("source_url"),
            source_family=raw.get("source_family"),
            event_date=raw.get("event_date"),
            detected_at=raw.get("detected_at"),
            event_type=raw.get("event_type"),
            impact_direction=raw.get("impact_direction"),
            materiality=raw.get("materiality_score"),
            official_verified=raw.get("official_verified", False),
            financial_bridge_score=raw.get("financial_bridge_score"),
            detection_time_source=(
                _text(raw.get("detection_time_source"))
                or "PERSISTED_POINT_IN_TIME"
            ),
            event_status=raw.get("event_status", "ACTIVE"),
            resolved_at=raw.get("resolved_at"),
            supersedes_event_id=raw.get("supersedes_event_id", ""),
            resolution_source_url=raw.get("resolution_source_url", ""),
            entity_match_state=raw.get("entity_match_state", ""),
            official_domain=(
                raw.get("registered_official_domain")
                or raw.get("official_domain")
            ),
        )
        if record is not None:
            # Preserve the durable ID and original detection timestamp.
            record["narrative_event_id"] = (
                _text(raw.get("narrative_event_id"))
                or record["narrative_event_id"]
            )
            rows.append(record)
    return rows


def build_narrative_events(
    *,
    fundamentals: pd.DataFrame | None = None,
    news_review: pd.DataFrame | None = None,
    project_management: pd.DataFrame | None = None,
    market_status: pd.DataFrame | None = None,
    existing_events: pd.DataFrame | None = None,
    as_of: Any | None = None,
    config: NarrativeConfig | None = None,
) -> pd.DataFrame:
    cfg = config or NarrativeConfig()
    now = _timestamp(as_of)
    official_domain_map: dict[str, str] = {}
    if isinstance(fundamentals, pd.DataFrame) and not fundamentals.empty:
        for raw in fundamentals.to_dict("records"):
            ticker = _ticker(raw.get("ticker"))
            domain = (
                _hostname(raw.get("official_domain"))
                or _hostname(raw.get("issuer_official_domain"))
                or _hostname(raw.get("company_website"))
            )
            if ticker and domain:
                official_domain_map[ticker] = domain
    rows = (
        _normalise_existing_events(existing_events)
        + _events_from_news(news_review)
        + _events_from_projects(project_management)
        + _events_from_fundamentals(fundamentals)
        + _events_from_market_status(market_status)
    )
    if not rows:
        return pd.DataFrame(columns=[
            "narrative_event_id", "ticker", "event_date", "detected_at",
            "event_type", "event_family", "headline", "source_url",
            "source_quality_score", "official_verified",
            "materiality_score", "novelty_score",
            "financial_bridge_score", "narrative_decay_weight",
        ])
    frame = pd.DataFrame(rows)
    if "registered_official_domain" not in frame:
        frame["registered_official_domain"] = ""
    frame["registered_official_domain"] = [
        _text(value) or official_domain_map.get(_ticker(ticker), "")
        for ticker, value in zip(
            frame.get("ticker", pd.Series("", index=frame.index)),
            frame["registered_official_domain"],
        )
    ]
    source_recheck = [
        _source_quality(
            row.get("source_url"),
            row.get("source_family"),
            row.get("official_claimed", row.get("official_verified")),
            row.get("registered_official_domain"),
        )
        for row in frame.to_dict("records")
    ]
    frame["source_quality_score"] = [round(value[0], 1) for value in source_recheck]
    frame["official_verified"] = [bool(value[1]) for value in source_recheck]
    frame["detected_at"] = pd.to_datetime(
        frame["detected_at"], errors="coerce", utc=True,
    )
    frame = frame.dropna(subset=["ticker", "detected_at"])
    frame["_source_sort"] = pd.to_numeric(
        frame["source_quality_score"], errors="coerce",
    ).fillna(0.0)
    frame = (
        frame.sort_values(
            ["narrative_event_id", "_source_sort"],
            ascending=[True, False],
            kind="stable",
        )
        .drop_duplicates("narrative_event_id", keep="first")
        .drop(columns="_source_sort")
        .sort_values(["ticker", "detected_at", "narrative_event_id"])
        .reset_index(drop=True)
    )
    novelty: list[float] = []
    seen: dict[tuple[str, str], list[set[str]]] = {}
    for raw in frame.to_dict("records"):
        key = (_text(raw.get("ticker")), _text(raw.get("event_family")))
        words = _normalised_words(raw.get("headline"))
        similarities: list[float] = []
        for prior in seen.get(key, [])[-12:]:
            union = words | prior
            similarities.append(len(words & prior) / len(union) if union else 1.0)
        maximum = max(similarities, default=0.0)
        novelty.append(max(15.0, min(100.0, 100.0 * (1.0 - maximum))))
        seen.setdefault(key, []).append(words)
    frame["novelty_score"] = np.round(novelty, 1)
    raw_age_days = (
        now - frame["detected_at"]
    ).dt.total_seconds().div(86400.0)
    future_detection = raw_age_days.lt(-1.0 / 1440.0)
    age_days = raw_age_days.clip(lower=0.0)
    frame["event_age_days"] = age_days.round(2)
    frame["narrative_decay_weight"] = np.power(
        0.5, age_days / max(1.0, cfg.half_life_days),
    ).clip(0.0, 1.0).round(4)
    frame.loc[future_detection, "narrative_decay_weight"] = 0.0
    frame["catalyst_proximity_score"] = np.select(
        [
            age_days.le(7.0), age_days.le(30.0),
            age_days.le(90.0), age_days.le(cfg.event_lookback_days),
        ],
        [100.0, 82.0, 60.0, 35.0],
        default=10.0,
    )
    frame["future_detection_invalid"] = future_detection.astype(bool)
    source_present = frame.get(
        "source_present", pd.Series(False, index=frame.index),
    ).fillna(False).astype(bool)
    lifecycle_active = frame.get(
        "event_status", pd.Series("ACTIVE", index=frame.index),
    ).fillna("ACTIVE").astype(str).str.upper().isin({"ACTIVE", "DISPUTED"})
    entity_state = frame.get(
        "entity_match_state", pd.Series("NOT_APPLICABLE", index=frame.index),
    ).fillna("NOT_APPLICABLE").astype(str).str.upper()
    entity_valid = ~entity_state.eq("AMBIGUOUS_TICKER_UNVERIFIED")
    frame["event_active"] = (
        age_days.le(cfg.event_lookback_days)
        & ~future_detection
        & source_present
        & lifecycle_active
        & entity_valid
    )
    frame["event_evidence_state"] = np.select(
        [
            future_detection,
            ~source_present,
            ~lifecycle_active,
            ~entity_valid,
            frame["official_verified"].fillna(False).astype(bool),
            pd.to_numeric(
                frame["source_quality_score"], errors="coerce",
            ).fillna(0.0).ge(60.0),
        ],
        [
            "FUTURE_DETECTION_INVALID",
            "MISSING_SOURCE",
            "LIFECYCLE_NOT_ACTIVE",
            "ENTITY_UNVERIFIED",
            "OFFICIAL_VERIFIED",
            "SOURCE_IDENTIFIED",
        ],
        default="UNVERIFIED_OR_LOW_QUALITY",
    )
    signed = pd.to_numeric(
        frame["impact_sign"], errors="coerce",
    ).fillna(0.0)
    raw_strength = (
        0.25 * pd.to_numeric(frame["materiality_score"], errors="coerce").fillna(45.0)
        + 0.25 * pd.to_numeric(frame["source_quality_score"], errors="coerce").fillna(35.0)
        + 0.20 * pd.to_numeric(frame["novelty_score"], errors="coerce").fillna(50.0)
        + 0.20 * pd.to_numeric(frame["financial_bridge_score"], errors="coerce").fillna(30.0)
        + 0.10 * pd.to_numeric(frame["catalyst_proximity_score"], errors="coerce").fillna(35.0)
    )
    frame["event_strength_score"] = raw_strength.clip(0.0, 100.0).round(1)
    frame["signed_event_strength"] = (
        signed
        * raw_strength
        * frame["narrative_decay_weight"]
        * frame["event_active"].astype(float)
        * pd.to_numeric(
            frame["source_quality_score"], errors="coerce",
        ).fillna(0.0)
        / 100.0
    ).round(2)
    frame["narrative_engine_version"] = NARRATIVE_ENGINE_VERSION
    return frame.sort_values(
        ["detected_at", "ticker"], ascending=[False, True], kind="stable",
    ).reset_index(drop=True)


def _price_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty or "Close" not in frame:
        return pd.DataFrame()
    local = frame.copy()
    local.index = pd.to_datetime(local.index, errors="coerce")
    local = local[~local.index.isna()].sort_index()
    local["Close"] = pd.to_numeric(local["Close"], errors="coerce")
    return local.dropna(subset=["Close"])


def _return_between(
    frame: pd.DataFrame,
    start_date: pd.Timestamp,
    target_bars: int,
) -> tuple[float, float, float, str, float]:
    local = _price_frame(frame)
    if local.empty:
        return (np.nan, np.nan, np.nan, "", np.nan)
    # Daily OHLCV cannot prove an executable same-day price after a disclosure
    # timestamp.  Use the first completed trading session strictly after the
    # durable detection date to eliminate same-day close look-ahead.
    positions = np.flatnonzero(local.index.normalize() > start_date.normalize())
    if len(positions) == 0:
        return (np.nan, np.nan, np.nan, "", np.nan)
    start = int(positions[0])
    end = start + int(target_bars)
    if end >= len(local):
        return (
            np.nan, np.nan, np.nan,
            local.index[start].date().isoformat(),
            float(local.iloc[start]["Close"]),
        )
    entry = float(local.iloc[start]["Close"])
    exit_price = float(local.iloc[end]["Close"])
    window = local.iloc[start + 1:end + 1]
    highs = pd.to_numeric(
        window["High"] if "High" in window else window["Close"],
        errors="coerce",
    )
    lows = pd.to_numeric(
        window["Low"] if "Low" in window else window["Close"],
        errors="coerce",
    )
    stock_return = 100.0 * (exit_price / entry - 1.0) if entry > 0 else np.nan
    mfe = 100.0 * (float(highs.max()) / entry - 1.0) if entry > 0 and highs.notna().any() else np.nan
    mae = 100.0 * (float(lows.min()) / entry - 1.0) if entry > 0 and lows.notna().any() else np.nan
    return (
        stock_return, mfe, mae,
        local.index[start].date().isoformat(), entry,
    )


def _benchmark_return(
    benchmark: pd.DataFrame | None,
    anchor_date: Any,
    target_date: Any,
) -> float:
    local = _price_frame(benchmark)
    if local.empty:
        return np.nan
    anchor = pd.to_datetime(anchor_date, errors="coerce")
    target = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(anchor) or pd.isna(target):
        return np.nan
    start_positions = np.flatnonzero(local.index.normalize() >= anchor.normalize())
    end_positions = np.flatnonzero(local.index.normalize() >= target.normalize())
    if len(start_positions) == 0 or len(end_positions) == 0:
        return np.nan
    start = int(start_positions[0])
    end = int(end_positions[0])
    start_price = float(local.iloc[start]["Close"])
    end_price = float(local.iloc[end]["Close"])
    return (
        100.0 * (end_price / start_price - 1.0)
        if start_price > 0 else np.nan
    )


def update_narrative_event_outcomes(
    events: pd.DataFrame | None,
    existing_outcomes: pd.DataFrame | None,
    prepared: Mapping[str, pd.DataFrame],
    *,
    benchmark: pd.DataFrame | None = None,
    as_of: Any | None = None,
    config: NarrativeConfig | None = None,
) -> pd.DataFrame:
    """Resolve only bars that exist after the durable detection timestamp."""
    cfg = config or NarrativeConfig()
    now = _timestamp(as_of)
    existing_map: dict[str, dict[str, Any]] = {}
    if isinstance(existing_outcomes, pd.DataFrame) and not existing_outcomes.empty:
        for raw in existing_outcomes.to_dict("records"):
            outcome_id = _text(raw.get("narrative_outcome_id"))
            if outcome_id:
                existing_map[outcome_id] = dict(raw)
    if events is None or events.empty:
        return pd.DataFrame(list(existing_map.values()))
    rows: list[dict[str, Any]] = []
    for event in events.to_dict("records"):
        if not _truthy(event.get("event_active", True)):
            continue
        event_id = _text(event.get("narrative_event_id"))
        if not event_id:
            continue
        outcome_id = _stable_hash((event_id, "20D_60D"), 48)
        prior = existing_map.get(outcome_id, {})
        ticker = _ticker(event.get("ticker"))
        frame = _price_frame(prepared.get(ticker, pd.DataFrame()))
        if not frame.empty:
            frame = frame[
                frame.index <= now.tz_convert(None)
            ]
        signal_date = _timestamp(event.get("detected_at")).tz_convert(None)
        local = frame
        base = dict(prior)
        base.update({
            "narrative_outcome_id": outcome_id,
            "narrative_event_id": event_id,
            "ticker": ticker,
            "event_type": event.get("event_type"),
            "event_family": event.get("event_family"),
            "impact_sign": int(_finite(event.get("impact_sign"), 0.0)),
            "impact_direction": event.get("impact_direction"),
            "signal_timestamp": _timestamp(event.get("detected_at")).isoformat(),
            "signal_date": signal_date.date().isoformat(),
            "entry_policy": "NEXT_COMPLETED_SESSION_AFTER_DETECTION",
            "roundtrip_cost_pct": 100.0 * cfg.roundtrip_cost_pct,
            "narrative_engine_version": NARRATIVE_ENGINE_VERSION,
        })
        anchor_date = ""
        entry_reference = np.nan
        resolved_horizons = 0
        stock_resolved_horizons = 0
        for horizon in (5, 20, 60):
            stock_return, mfe, mae, anchor, entry = _return_between(
                frame, signal_date, horizon,
            )
            anchor_date = anchor_date or anchor
            entry_reference = (
                entry if not np.isfinite(entry_reference)
                else entry_reference
            )
            if np.isfinite(stock_return):
                stock_resolved_horizons += 1
                start_positions = np.flatnonzero(
                    local.index.normalize() > signal_date.normalize()
                )
                if len(start_positions) == 0:
                    continue
                target_position = int(start_positions[0]) + horizon
                target_date = local.index[int(target_position)]
                bench_return = _benchmark_return(
                    benchmark, anchor, target_date,
                )
                net_excess = (
                    stock_return - bench_return
                    - 100.0 * cfg.roundtrip_cost_pct
                    if np.isfinite(bench_return) else np.nan
                )
                resolved_horizons += int(np.isfinite(net_excess))
                impact_sign = int(_finite(event.get("impact_sign"), 0.0))
                directional_excess = (
                    impact_sign * net_excess
                    if impact_sign in {-1, 1} and np.isfinite(net_excess)
                    else np.nan
                )
                base.update({
                    f"stock_return_{horizon}d_pct": round(stock_return, 3),
                    f"benchmark_return_{horizon}d_pct": (
                        round(bench_return, 3)
                        if np.isfinite(bench_return) else np.nan
                    ),
                    f"net_excess_return_{horizon}d_pct": (
                        round(net_excess, 3)
                        if np.isfinite(net_excess) else np.nan
                    ),
                    f"directional_excess_return_{horizon}d_pct": (
                        round(directional_excess, 3)
                        if np.isfinite(directional_excess) else np.nan
                    ),
                    f"mfe_{horizon}d_pct": round(mfe, 3)
                    if np.isfinite(mfe) else np.nan,
                    f"mae_{horizon}d_pct": round(mae, 3)
                    if np.isfinite(mae) else np.nan,
                    f"converted_{horizon}d": (
                        bool(directional_excess > 0.0)
                        if np.isfinite(directional_excess) else np.nan
                    ),
                })
        base["anchor_date"] = anchor_date
        base["entry_reference"] = entry_reference
        base["outcome_status"] = (
            "RESOLVED_60D" if resolved_horizons == 3
            else "PARTIAL_20D" if resolved_horizons >= 2
            else "PARTIAL_5D" if resolved_horizons == 1
            else "OPEN_BENCHMARK_PENDING"
            if stock_resolved_horizons > 0
            else "OPEN_NO_FORWARD_BARS"
        )
        base["resolved_at"] = (
            now.isoformat() if resolved_horizons == 3
            else _text(prior.get("resolved_at"))
        )
        rows.append(base)
    seen = {row["narrative_outcome_id"] for row in rows}
    rows.extend(
        row for key, row in existing_map.items() if key not in seen
    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["signal_timestamp", "ticker"],
        ascending=[False, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def _last_number(
    frame: pd.DataFrame,
    names: Iterable[str],
    default: float = np.nan,
) -> float:
    if frame is None or frame.empty:
        return default
    for name in names:
        if name in frame:
            values = pd.to_numeric(frame[name], errors="coerce").dropna()
            if len(values):
                return _finite(values.iloc[-1], default)
    return default


def _market_flow_features(
    frame: pd.DataFrame | None,
    silent_profile: Mapping[str, Any] | None,
) -> dict[str, float]:
    local = _price_frame(frame)
    profile = dict(silent_profile or {})
    silent = _finite(profile.get("silent_accumulation_score"), 50.0)
    confidence = max(
        0.0, min(
            100.0,
            _finite(
                profile.get(
                    "silent_accumulation_confidence",
                    profile.get("silent_accumulation_data_coverage"),
                ),
                0.0,
            ),
        ),
    )
    state = _text(profile.get("silent_accumulation_state")).upper()
    effective = 50.0 + (silent - 50.0) * confidence / 100.0
    if state in {"DISTRIBUTION_RISK", "WEAK_OR_DISTRIBUTION"}:
        effective = min(effective, 45.0)
    if local.empty:
        return {
            "silent_raw": silent,
            "silent_confidence": confidence,
            "effective_silent": effective,
            "roc20_pct": 0.0,
            "roc60_pct": 0.0,
            "volume_ratio": 1.0,
            "distance_high_pct": -100.0,
            "flow_data_coverage_pct": 0.0,
            "distribution_days": _finite(
                profile.get("distribution_days20"), 0.0,
            ),
        }
    close = local["Close"]
    roc20 = 100.0 * (close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) > 20 else 0.0
    roc60 = 100.0 * (close.iloc[-1] / close.iloc[-61] - 1.0) if len(close) > 60 else 0.0
    volume_ratio = 1.0
    if "Volume" in local and len(local) >= 30:
        volume = pd.to_numeric(local["Volume"], errors="coerce")
        recent = _finite(volume.tail(10).mean(), np.nan)
        prior = _finite(volume.iloc[-60:-10].mean(), np.nan)
        if np.isfinite(recent) and np.isfinite(prior) and prior > 0:
            volume_ratio = recent / prior
    distance_high = 100.0 * (
        close.iloc[-1] / close.tail(min(252, len(close))).max() - 1.0
    )
    observed = 2 + int("Volume" in local) + int(confidence > 0)
    return {
        "silent_raw": max(0.0, min(100.0, silent)),
        "silent_confidence": confidence,
        "effective_silent": max(0.0, min(100.0, effective)),
        "roc20_pct": round(roc20, 2),
        "roc60_pct": round(roc60, 2),
        "volume_ratio": round(volume_ratio, 3),
        "distance_high_pct": round(distance_high, 2),
        "flow_data_coverage_pct": 25.0 * observed,
        "distribution_days": _finite(
            profile.get("distribution_days20"), 0.0,
        ),
    }


def _linear_component(value: Any, low: float, high: float) -> float:
    observed = _finite(value, np.nan)
    if not np.isfinite(observed) or high <= low:
        return np.nan
    return max(0.0, min(100.0, 100.0 * (observed - low) / (high - low)))


def _operating_narrative_proxy(
    fundamental: Mapping[str, Any] | None,
    project_rows: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Build a hard-fact operating score from structured financial evidence.

    The score becomes production-eligible only when `_structured_financial_lineage`
    confirms identifiable source lineage, a dated reporting period, adequate field
    coverage, and acceptable freshness.  Without that lineage it remains a research
    proxy and has zero production influence.
    """
    fund = dict(fundamental or {})
    projects = [dict(row) for row in (project_rows or [])]
    components: list[tuple[str, float, float]] = []

    revenue_growth = _finite(
        fund.get("history_revenue_cagr_3y", fund.get("revenue_growth")),
        np.nan,
    )
    earnings_growth = _finite(fund.get("earnings_growth"), np.nan)
    roic = _finite(
        fund.get("history_roic_proxy", fund.get("return_on_invested_capital")),
        np.nan,
    )
    cash_conversion = _finite(
        fund.get("history_cash_conversion", fund.get("cash_conversion_ttm")),
        np.nan,
    )
    fcf = _finite(
        fund.get("history_fcf_ttm", fund.get("free_cash_flow")),
        np.nan,
    )
    dilution = _finite(
        fund.get("history_share_dilution_yoy", fund.get("share_dilution_yoy")),
        np.nan,
    )
    net_margin = _finite(fund.get("net_margin"), np.nan)

    if np.isfinite(revenue_growth):
        components.append((
            "REVENUE_GROWTH",
            _linear_component(revenue_growth, -0.10, 0.30),
            0.22,
        ))
    if np.isfinite(earnings_growth):
        components.append((
            "EARNINGS_GROWTH",
            _linear_component(earnings_growth, -0.20, 0.40),
            0.22,
        ))
    if np.isfinite(roic):
        components.append((
            "ROIC", _linear_component(roic, 0.02, 0.20), 0.14,
        ))
    if np.isfinite(cash_conversion):
        conversion_score = (
            85.0 if 0.80 <= cash_conversion <= 1.80
            else 65.0 if 0.50 <= cash_conversion < 0.80
            or 1.80 < cash_conversion <= 2.50
            else 30.0
        )
        components.append(("CASH_CONVERSION", conversion_score, 0.10))
    if np.isfinite(fcf):
        components.append(("FREE_CASH_FLOW", 72.0 if fcf > 0 else 35.0, 0.08))
    if np.isfinite(dilution):
        dilution_score = (
            82.0 if dilution <= 0.0
            else 65.0 if dilution <= 0.05
            else 38.0 if dilution <= 0.12
            else 15.0
        )
        components.append(("DILUTION", dilution_score, 0.08))
    if np.isfinite(net_margin):
        components.append((
            "NET_MARGIN", _linear_component(net_margin, 0.00, 0.20), 0.06,
        ))

    best_project_score = np.nan
    best_project_coverage = 0.0
    verified_project = False
    for row in projects:
        score = _finite(
            row.get("project_pipeline_score_observed",
                    row.get("project_pipeline_score")),
            np.nan,
        )
        coverage = max(0.0, min(100.0, _finite(
            row.get("project_data_coverage",
                    row.get("project_data_coverage_effective")),
            0.0,
        )))
        if np.isfinite(score) and coverage >= best_project_coverage:
            best_project_score = max(0.0, min(100.0, score))
            best_project_coverage = coverage
            verified_project = _truthy(row.get("project_source_quorum_verified"))
    if np.isfinite(best_project_score):
        project_weight = 0.10 * max(0.25, best_project_coverage / 100.0)
        components.append(("PROJECT_EXECUTION", best_project_score, project_weight))

    if not components:
        return {
            "score": np.nan,
            "coverage_pct": 0.0,
            "state": "NOT_SCORED_NO_OPERATING_FACTS",
            "basis": "",
        }
    observed_weight = sum(weight for _, score, weight in components if np.isfinite(score))
    score = sum(score * weight for _, score, weight in components if np.isfinite(score)) / max(observed_weight, 1e-9)
    # 1.00 is the intended full weight.  A verified project adds confidence but
    # the proxy remains research-only even at high coverage.
    coverage = min(85.0, 100.0 * observed_weight + (8.0 if verified_project else 0.0))
    return {
        "score": round(max(0.0, min(100.0, score)), 1),
        "coverage_pct": round(coverage, 1),
        "state": "OPERATING_FACT_PROXY_RESEARCH_ONLY",
        "basis": " | ".join(name for name, _, _ in components),
    }


def _structured_financial_lineage(
    fundamental: Mapping[str, Any] | None,
    operating_proxy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate whether structured financial facts are usable narrative evidence.

    This closes the old false choice between a manually discovered corporate-event
    URL and an empty score.  A dated financial statement with identifiable provider
    lineage is itself public evidence.  Official IDX/XBRL remains highest quality,
    while Yahoo/Twelve references are accepted as lower-tier corroborating sources.
    """
    fund = dict(fundamental or {})
    proxy = dict(operating_proxy or {})
    score = _finite(proxy.get("score"), np.nan)
    proxy_coverage = max(0.0, min(100.0, _finite(proxy.get("coverage_pct"), 0.0)))
    source_urls = _text(fund.get("fundamental_source_urls"))
    official_urls = _text(fund.get("fundamental_official_source_urls"))
    source_families = _text(fund.get("fundamental_source_families"))
    source_count = int(max(0.0, _finite(fund.get("fundamental_source_count"), 0.0)))
    if source_count <= 0 and source_families:
        source_count = len({
            part.strip().upper()
            for part in re.split(r"[|•;,]", source_families)
            if part.strip()
        })
    latest_period = _text(_first_nonempty_value(
        fund.get("fundamental_history_latest_period"),
        fund.get("latest_statement_date"),
        fund.get("statement_date"),
        fund.get("period_end"),
    ))
    history_coverage = max(0.0, min(100.0, _finite(
        fund.get("fundamental_history_coverage",
                 fund.get("fundamental_coverage")),
        0.0,
    )))
    age_days = _finite(fund.get("fundamental_history_age_days"), np.nan)
    grade = _text(fund.get("fundamental_data_grade")).upper()
    official = bool(
        _truthy(fund.get("fundamental_official_verified"))
        or _truthy(fund.get("fundamental_official_reference"))
        or official_urls
    )
    identifiable = bool(source_count > 0 or source_urls or official_urls or source_families)
    dated = bool(latest_period)
    fresh = not np.isfinite(age_days) or age_days <= 450.0
    scoring_eligible = bool(
        np.isfinite(score)
        and proxy_coverage >= 30.0
        and identifiable
        and fresh
        and (history_coverage >= 35.0 or grade in {"A", "B", "C"})
    )
    production_eligible = bool(scoring_eligible and dated)
    grade_bonus = {"A": 12.0, "B": 8.0, "C": 4.0}.get(grade, 0.0)
    source_bonus = min(16.0, 6.0 * source_count)
    official_bonus = 14.0 if official else 0.0
    raw_lineage_coverage = min(82.0, 0.55 * proxy_coverage + 0.20 * history_coverage + source_bonus + grade_bonus + official_bonus)
    # Missing period metadata no longer destroys the research score.  It caps
    # coverage below the Emir production threshold until a dated statement is
    # acquired, preserving comparability without fabricating production-grade evidence.
    lineage_coverage = raw_lineage_coverage if production_eligible else min(32.0, 0.55 * raw_lineage_coverage) if scoring_eligible else 0.0
    missing: list[str] = []
    if not np.isfinite(score) or proxy_coverage < 30.0:
        missing.append("OPERATING_FACT_COVERAGE")
    if not identifiable:
        missing.append("SOURCE_LINEAGE")
    if not dated:
        missing.append("REPORTING_PERIOD")
    if not fresh:
        missing.append("FRESHNESS")
    if history_coverage < 35.0 and grade not in {"A", "B", "C"}:
        missing.append("HISTORY_COVERAGE")
    return {
        "eligible": scoring_eligible,
        "production_eligible": production_eligible,
        "score": score,
        "coverage_pct": round(lineage_coverage, 1),
        "state": (
            "OFFICIAL_STRUCTURED_FINANCIAL_EVIDENCE" if production_eligible and official
            else "IDENTIFIED_STRUCTURED_FINANCIAL_EVIDENCE" if production_eligible
            else "PARTIAL_STRUCTURED_FINANCIAL_EVIDENCE" if scoring_eligible
            else "STRUCTURED_EVIDENCE_INCOMPLETE"
        ),
        "source_count": source_count,
        "official": official,
        "latest_period": latest_period,
        "missing": " | ".join(missing),
    }


def _capital_allocation_alignment_proxy(
    fundamental: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Continuous issuer-alignment proxy from capital allocation outcomes.

    The previous bucketed four-metric model produced large clusters of identical
    scores.  This version uses continuous reinvestment, cash-funding, leverage,
    dilution and operating-growth evidence while remaining coverage-aware.
    """
    fund = dict(fundamental or {})
    components: list[tuple[str, float, float]] = []
    roic = _finite(fund.get("history_roic_proxy", fund.get("roic_proxy")), np.nan)
    cash_conversion = _finite(fund.get("history_cash_conversion", fund.get("cash_conversion_ttm")), np.nan)
    fcf = _finite(fund.get("history_fcf_ttm", fund.get("free_cash_flow")), np.nan)
    ocf = _finite(fund.get("history_ocf_ttm", fund.get("operating_cash_flow")), np.nan)
    revenue = _finite(fund.get("history_revenue_ttm", fund.get("revenue")), np.nan)
    capex = abs(_finite(fund.get("history_capex_ttm", fund.get("capital_expenditure")), np.nan))
    dilution = _finite(fund.get("history_share_dilution_yoy", fund.get("share_dilution_yoy")), np.nan)
    revenue_growth = _finite(fund.get("history_revenue_growth", fund.get("revenue_growth")), np.nan)
    earnings_growth = _finite(fund.get("history_earnings_growth", fund.get("earnings_growth")), np.nan)
    net_debt_ebitda = _finite(fund.get("history_net_debt_ebitda", fund.get("net_debt_ebitda")), np.nan)
    debt_equity = _finite(fund.get("history_debt_equity", fund.get("debt_equity")), np.nan)

    if np.isfinite(roic):
        components.append(("ROIC", _linear_component(roic, 0.02, 0.22), 0.20))
    if np.isfinite(cash_conversion):
        distance = abs(cash_conversion - 1.15)
        score = max(10.0, min(95.0, 95.0 - 42.0 * distance))
        components.append(("CASH_CONVERSION", score, 0.16))
    if np.isfinite(fcf):
        fcf_margin = fcf / revenue if np.isfinite(revenue) and revenue > 0 else np.nan
        score = _linear_component(fcf_margin, -0.05, 0.15) if np.isfinite(fcf_margin) else (72.0 if fcf > 0 else 28.0)
        components.append(("FCF_DISCIPLINE", score, 0.10))
    if np.isfinite(dilution):
        score = max(0.0, min(100.0, 92.0 - 420.0 * max(0.0, dilution) + 80.0 * max(0.0, -dilution)))
        components.append(("DILUTION_DISCIPLINE", score, 0.16))
    growth_values = [v for v in (revenue_growth, earnings_growth) if np.isfinite(v)]
    if growth_values:
        growth = float(np.mean(growth_values))
        components.append(("OPERATING_GROWTH", _linear_component(growth, -0.08, 0.28), 0.10))
    if np.isfinite(capex) and np.isfinite(revenue) and revenue > 0:
        capex_intensity = capex / revenue
        # Reward meaningful but not reckless reinvestment; extreme intensity is
        # penalised unless later corroborated by project evidence.
        score = max(10.0, min(95.0, 92.0 - 260.0 * abs(capex_intensity - 0.10)))
        components.append(("CAPEX_INTENSITY", score, 0.10))
    if np.isfinite(ocf) and np.isfinite(capex) and capex > 0:
        funding = ocf / capex
        score = _linear_component(funding, 0.35, 2.00)
        components.append(("OCF_CAPEX_FUNDING", score, 0.10))
    leverage = net_debt_ebitda if np.isfinite(net_debt_ebitda) else debt_equity
    if np.isfinite(leverage):
        if np.isfinite(net_debt_ebitda):
            score = max(0.0, min(100.0, 95.0 - 22.0 * max(0.0, leverage)))
        else:
            score = max(0.0, min(100.0, 95.0 - 38.0 * max(0.0, leverage)))
        components.append(("LEVERAGE_DISCIPLINE", score, 0.08))
    if not components:
        return {"score": np.nan, "coverage_pct": 0.0, "basis": ""}
    total_weight = sum(weight for _, _, weight in components)
    score = sum(value * weight for _, value, weight in components) / max(total_weight, 1e-9)
    return {
        "score": round(max(0.0, min(100.0, score)), 1),
        "coverage_pct": round(min(92.0, 100.0 * total_weight), 1),
        "basis": " | ".join(name for name, _, _ in components),
    }


def _conversion_statistics(
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
    min_events: int,
) -> dict[str, dict[str, Any]]:
    if outcomes is None or outcomes.empty:
        return {}
    event_family = (
        events.set_index("narrative_event_id")["event_family"].to_dict()
        if events is not None and not events.empty else {}
    )
    local = outcomes.copy()
    local["event_family"] = local.apply(
        lambda row: _text(row.get("event_family"))
        or _text(event_family.get(row.get("narrative_event_id"))),
        axis=1,
    )
    cohort: dict[tuple[str, int], tuple[float, int]] = {}
    for horizon in (5, 20, 60):
        flag = f"converted_{horizon}d"
        value = f"directional_excess_return_{horizon}d_pct"
        if flag not in local:
            local[flag] = False
        if value not in local:
            legacy = pd.to_numeric(
                local.get(f"net_excess_return_{horizon}d_pct"),
                errors="coerce",
            )
            sign = pd.to_numeric(local.get("impact_sign"), errors="coerce")
            local[value] = legacy * sign.where(sign.isin([-1, 1]))
        resolved = local[pd.to_numeric(local.get(value), errors="coerce").notna()]
        for family, group in resolved.groupby("event_family", dropna=False):
            hits = group[flag].map(_truthy).sum()
            count = len(group)
            cohort[(_text(family), horizon)] = (
                100.0 * (hits + 2.0) / (count + 4.0),
                count,
            )
    output: dict[str, dict[str, Any]] = {}
    for ticker, group in local.groupby("ticker", sort=False):
        profile: dict[str, Any] = {}
        for horizon in (5, 20, 60):
            value = f"directional_excess_return_{horizon}d_pct"
            flag = f"converted_{horizon}d"
            values = pd.to_numeric(group.get(value), errors="coerce")
            resolved = group.loc[values.notna()].copy()
            count = len(resolved)
            if count:
                family_rates = [
                    cohort.get((_text(family), horizon), (50.0, 0))[0]
                    for family in resolved["event_family"]
                ]
                prior_rate = float(np.mean(family_rates)) if family_rates else 50.0
                hits = int(resolved[flag].map(_truthy).sum())
                posterior = (
                    100.0
                    * (hits + 4.0 * prior_rate / 100.0)
                    / (count + 4.0)
                )
                expectancy = float(
                    pd.to_numeric(resolved[value], errors="coerce").mean()
                )
            else:
                prior_rate = np.nan
                posterior = np.nan
                expectancy = np.nan
            ready = count >= min_events
            effective = (
                posterior if ready
                else 50.0 + (posterior - 50.0) * count / max(min_events, 1)
                if count and np.isfinite(posterior)
                else np.nan
            )
            profile.update({
                f"narrative_conversion_rate_{horizon}d_pct": (
                    round(posterior, 1) if np.isfinite(posterior) else np.nan
                ),
                f"narrative_conversion_effective_{horizon}d_score": (
                    round(effective, 1) if np.isfinite(effective) else np.nan
                ),
                f"narrative_conversion_expectancy_{horizon}d_pct": (
                    round(expectancy, 3) if np.isfinite(expectancy) else np.nan
                ),
                f"narrative_conversion_resolved_{horizon}d": int(count),
                f"narrative_conversion_state_{horizon}d": (
                    "PRODUCTION_ELIGIBLE_POINT_IN_TIME"
                    if ready else "SHADOW_MIN_SAMPLE_PENDING"
                    if count else "NO_RESOLVED_POINT_IN_TIME_SAMPLE"
                ),
                f"narrative_conversion_cohort_prior_{horizon}d_pct": (
                    round(prior_rate, 1) if np.isfinite(prior_rate) else np.nan
                ),
            })
        output[_ticker(ticker)] = profile
    return output


def _verified_forward_profile(
    rows: Iterable[Mapping[str, Any]],
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    """Select one current, sourced forward record without synthesising a score.

    A parser-produced management proxy or a realised financial trend is not
    forward evidence.  The record must carry a project score/impact, a measured
    project coverage, HTTPS lineage, an explicit source quorum, and a current
    verification timestamp.  Otherwise the Future pillar remains missing.
    """
    candidates: list[tuple[float, int, pd.Timestamp, dict[str, Any]]] = []
    saw_score = False
    saw_lineage_failure = False
    for raw in rows:
        pipeline = _finite(
            raw.get("project_pipeline_score_observed",
                    raw.get("project_pipeline_score")),
            np.nan,
        )
        impact = _finite(
            raw.get("future_fundamental_impact_score_observed",
                    raw.get("future_fundamental_impact_score")),
            np.nan,
        )
        if not (np.isfinite(pipeline) or np.isfinite(impact)):
            continue
        saw_score = True
        coverage = max(0.0, min(100.0, _finite(
            raw.get("project_data_coverage",
                    raw.get("project_data_coverage_effective")),
            0.0,
        )))
        urls: list[str] = []
        for value in (raw.get("project_source_urls"), raw.get("source_url")):
            for url in _source_url_candidates(value):
                if url not in urls:
                    urls.append(url)
        verified_at = pd.to_datetime(
            raw.get("last_verified_at", raw.get("source_checked_at")),
            errors="coerce",
            utc=True,
        )
        age_days = (
            float((as_of - verified_at).total_seconds() / 86400.0)
            if pd.notna(verified_at) else np.nan
        )
        lineage_ok = bool(
            _truthy(raw.get("project_source_quorum_verified"))
            and coverage > 0.0
            and urls
            and np.isfinite(age_days)
            and -1.0 <= age_days <= 450.0
        )
        if not lineage_ok:
            saw_lineage_failure = True
            continue
        payload = {
            "forward_project_pipeline_score": (
                round(max(0.0, min(100.0, pipeline)), 1)
                if np.isfinite(pipeline) else np.nan
            ),
            "forward_future_fundamental_impact_score": (
                round(max(0.0, min(100.0, impact)), 1)
                if np.isfinite(impact) else np.nan
            ),
            "forward_project_data_coverage_pct": round(coverage, 1),
            "forward_source_quorum_verified": True,
            "forward_source_urls": " | ".join(urls),
            "forward_source_families": _text(raw.get("project_source_families")),
            "forward_last_verified_at": pd.Timestamp(verified_at).isoformat(),
            "forward_evidence_state": "VERIFIED_DIRECT_FORWARD_EVIDENCE",
            "forward_evidence_basis": "PROJECT_PIPELINE_AND_IMPACT"
            if np.isfinite(pipeline) and np.isfinite(impact)
            else "PROJECT_PIPELINE" if np.isfinite(pipeline)
            else "FUTURE_FUNDAMENTAL_IMPACT",
        }
        candidates.append((
            coverage,
            int(np.isfinite(pipeline)) + int(np.isfinite(impact)),
            pd.Timestamp(verified_at),
            payload,
        ))
    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]
    return {
        "forward_project_pipeline_score": np.nan,
        "forward_future_fundamental_impact_score": np.nan,
        "forward_project_data_coverage_pct": 0.0,
        "forward_source_quorum_verified": False,
        "forward_source_urls": "",
        "forward_source_families": "",
        "forward_last_verified_at": "",
        "forward_evidence_state": (
            "NOT_SCORED_FORWARD_LINEAGE_INCOMPLETE"
            if saw_score and saw_lineage_failure
            else "NOT_SCORED_DIRECT_FORWARD_EVIDENCE_MISSING"
        ),
        "forward_evidence_basis": "",
    }


def build_narrative_profiles(
    tickers: Iterable[str],
    *,
    prepared: Mapping[str, pd.DataFrame],
    events: pd.DataFrame | None,
    outcomes: pd.DataFrame | None,
    fundamentals: pd.DataFrame | None = None,
    news_review: pd.DataFrame | None = None,
    project_management: pd.DataFrame | None = None,
    silent_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    as_of: Any | None = None,
    config: NarrativeConfig | None = None,
) -> pd.DataFrame:
    cfg = config or NarrativeConfig()
    now = _timestamp(as_of)
    names = list(dict.fromkeys(_ticker(name) for name in tickers if _ticker(name)))
    event_frame = events.copy() if isinstance(events, pd.DataFrame) else pd.DataFrame()
    if not event_frame.empty:
        event_frame["detected_at"] = pd.to_datetime(
            event_frame["detected_at"], errors="coerce", utc=True,
        )
    conversion = _conversion_statistics(
        event_frame, outcomes if isinstance(outcomes, pd.DataFrame) else pd.DataFrame(),
        cfg.min_conversion_events,
    )
    news_map: dict[str, dict[str, Any]] = {}
    if isinstance(news_review, pd.DataFrame) and not news_review.empty:
        for raw in news_review.to_dict("records"):
            news_map[_ticker(raw.get("ticker"))] = raw
    fundamental_map: dict[str, dict[str, Any]] = {}
    if isinstance(fundamentals, pd.DataFrame) and not fundamentals.empty:
        for raw in fundamentals.to_dict("records"):
            fundamental_map[_ticker(raw.get("ticker"))] = raw
    pm_map: dict[str, list[dict[str, Any]]] = {}
    if isinstance(project_management, pd.DataFrame) and not project_management.empty:
        for raw in project_management.to_dict("records"):
            pm_map.setdefault(_ticker(raw.get("ticker")), []).append(raw)
    silent_map = {
        _ticker(name): dict(profile)
        for name, profile in (silent_profiles or {}).items()
    }
    rows: list[dict[str, Any]] = []
    for ticker in names:
        fundamental = fundamental_map.get(ticker, {})
        pm_evidence = pm_map.get(ticker, [])
        forward_evidence = _verified_forward_profile(pm_evidence, now)
        operating_proxy = _operating_narrative_proxy(
            fundamental, pm_evidence,
        )
        structured_financial = _structured_financial_lineage(
            fundamental, operating_proxy,
        )
        all_events = (
            event_frame[event_frame["ticker"].eq(ticker)].copy()
            if not event_frame.empty else pd.DataFrame()
        )
        active = (
            all_events[all_events["event_active"].fillna(False).astype(bool)]
            if not all_events.empty and "event_active" in all_events else all_events
        )
        positive = (
            active[active["impact_sign"].eq(1)].copy()
            if not active.empty else pd.DataFrame()
        )
        negative = (
            active[active["impact_sign"].eq(-1)].copy()
            if not active.empty else pd.DataFrame()
        )
        weights = (
            pd.to_numeric(active.get("narrative_decay_weight"), errors="coerce").fillna(0.0)
            * pd.to_numeric(active.get("source_quality_score"), errors="coerce").fillna(0.0)
            / 100.0
        ) if not active.empty else pd.Series(dtype=float)
        signed = pd.to_numeric(
            active.get("signed_event_strength"), errors="coerce",
        ).fillna(0.0) if not active.empty else pd.Series(dtype=float)
        signed_mean = (
            float(signed.sum() / max(weights.sum(), 1e-9))
            if len(weights) and weights.sum() > 0 else 0.0
        )
        event_narrative_raw = max(0.0, min(100.0, 50.0 + 0.5 * signed_mean))
        cluster_count = int(
            active["event_cluster_key"].nunique()
        ) if not active.empty and "event_cluster_key" in active else len(active)
        source_count = int(
            active.loc[
                active["source_url"].fillna("").astype(str).str.len().gt(0),
                "source_url",
            ].nunique()
        ) if not active.empty else 0
        official_count = int(
            active.loc[
                active["official_verified"].fillna(False).astype(bool),
                "event_cluster_key",
            ].nunique()
        ) if not active.empty and "event_cluster_key" in active else int(
            active["official_verified"].fillna(False).astype(bool).sum()
        ) if not active.empty else 0
        corroborated_clusters = 0
        if not active.empty and "event_cluster_key" in active:
            corroborated_clusters = int(
                active.groupby("event_cluster_key")["source_url"]
                .nunique()
                .ge(2)
                .sum()
            )
        bridge_mean = (
            float(pd.to_numeric(
                active.get("financial_bridge_score"), errors="coerce",
            ).mean())
            if not active.empty else 0.0
        )
        event_evidence_coverage = min(
            100.0,
            12.0 * cluster_count
            + 8.0 * source_count
            + 18.0 * official_count
            + 10.0 * corroborated_clusters
            + 0.20 * bridge_mean,
        )
        structured_eligible = bool(structured_financial.get("eligible"))
        structured_production_eligible = bool(structured_financial.get("production_eligible"))
        structured_score = _finite(structured_financial.get("score"), np.nan)
        structured_coverage = max(0.0, min(100.0, _finite(
            structured_financial.get("coverage_pct"), 0.0,
        )))
        if event_evidence_coverage > 0.0 and structured_eligible:
            event_weight = max(0.35, event_evidence_coverage / 100.0)
            structured_weight = max(0.20, structured_coverage / 100.0)
            narrative_raw = (
                event_narrative_raw * event_weight
                + structured_score * structured_weight
            ) / max(event_weight + structured_weight, 1e-9)
            evidence_coverage = min(
                100.0,
                event_evidence_coverage + 0.45 * structured_coverage,
            )
            narrative_evidence_mode = (
                "EVENT_PLUS_STRUCTURED_FINANCIAL" if structured_production_eligible
                else "EVENT_PLUS_PARTIAL_STRUCTURED_FINANCIAL"
            )
        elif event_evidence_coverage > 0.0:
            narrative_raw = event_narrative_raw
            evidence_coverage = event_evidence_coverage
            narrative_evidence_mode = "SOURCE_EVENT"
        elif structured_eligible:
            narrative_raw = structured_score
            evidence_coverage = structured_coverage
            narrative_evidence_mode = (
                "STRUCTURED_FINANCIAL" if structured_production_eligible
                else "STRUCTURED_FINANCIAL_PARTIAL"
            )
        else:
            narrative_raw = event_narrative_raw
            evidence_coverage = 0.0
            narrative_evidence_mode = "NO_ELIGIBLE_EVIDENCE"
        narrative_raw = max(0.0, min(100.0, narrative_raw))
        narrative_effective = (
            50.0 + (narrative_raw - 50.0) * evidence_coverage / 100.0
        )
        # V8 production separates sourced narrative events from structured
        # financial evidence.  The latter belongs exclusively to the 55%
        # fundamental/future-fundamental pillar and must not be counted again
        # inside narrative.
        narrative_event_effective = (
            50.0
            + (event_narrative_raw - 50.0)
            * event_evidence_coverage / 100.0
        )
        latest = (
            active.sort_values("detected_at", ascending=False).iloc[0]
            if not active.empty else pd.Series(dtype=object)
        )

        alignment_numerator = 0.0
        alignment_weight = 0.0
        direct_alignment_weight = 0.0
        alignment_basis: list[str] = []
        # Pure production alignment excludes project-pipeline, dilution,
        # governance ratios and capital-allocation proxies because those are
        # already represented in the fundamental/future-fundamental pillar.
        production_alignment_numerator = 0.0
        production_alignment_weight = 0.0
        production_alignment_basis: list[str] = []
        production_alignment_coverage_points = 0.0
        positive_alignment_events = 0
        negative_alignment_events = 0
        for event in active.to_dict("records"):
            event_type = _text(event.get("event_type"))
            sign_weight = _POSITIVE_ALIGNMENT.get(
                event_type,
                -_NEGATIVE_ALIGNMENT.get(event_type, 0.0),
            )
            if sign_weight == 0.0:
                continue
            weight = (
                _finite(event.get("narrative_decay_weight"), 0.0)
                * _finite(event.get("source_quality_score"), 0.0)
                / 100.0
                * _finite(event.get("materiality_score"), 0.0)
                / 100.0
            )
            alignment_numerator += sign_weight * weight
            alignment_weight += abs(weight)
            direct_alignment_weight += abs(weight)
            alignment_basis.append(f"EVENT:{event_type}")
            production_alignment_numerator += sign_weight * weight
            production_alignment_weight += abs(weight)
            production_alignment_basis.append(f"EVENT:{event_type}")
            positive_alignment_events += int(sign_weight > 0)
            negative_alignment_events += int(sign_weight < 0)
        pm_coverage = 0.0
        for pm in pm_evidence:
            coverage = _finite(pm.get("management_data_coverage"), 0.0)
            insider = _finite(pm.get("insider_ownership_pct"), np.nan)
            governance = _text(pm.get("management_governance_flags"))
            related = _text(pm.get("management_related_party_risk")).upper()
            if np.isfinite(insider):
                insider_pct = 100.0 * insider if insider <= 1.0 else insider
                # Ownership can align incentives at moderate levels, while very
                # high concentration may introduce entrenchment risk.
                alignment_numerator += (
                    0.30 if 5.0 <= insider_pct <= 60.0
                    else -0.15 if insider_pct > 75.0
                    else 0.05 if insider_pct > 0.0 else 0.0
                )
                alignment_weight += 0.30
                direct_alignment_weight += 0.30
                alignment_basis.append("DIRECT_INSIDER_OWNERSHIP")
                production_alignment_numerator += (
                    0.30 if 5.0 <= insider_pct <= 60.0
                    else -0.15 if insider_pct > 75.0
                    else 0.05 if insider_pct > 0.0 else 0.0
                )
                production_alignment_weight += 0.30
                production_alignment_basis.append("DIRECT_INSIDER_OWNERSHIP")
                production_alignment_coverage_points += 15.0
                pm_coverage += 15.0
            if governance or related == "CRITICAL":
                alignment_numerator -= 0.60
                alignment_weight += 0.60
                direct_alignment_weight += 0.60
                alignment_basis.append("GOVERNANCE_OR_RELATED_PARTY_RISK")
                production_alignment_numerator -= 0.60
                production_alignment_weight += 0.60
                production_alignment_basis.append("GOVERNANCE_OR_RELATED_PARTY_RISK")
                production_alignment_coverage_points += 25.0
                negative_alignment_events += 1
                pm_coverage += 25.0
            management_score = _finite(
                pm.get("management_quality_score_observed",
                       pm.get("management_quality_score")),
                np.nan,
            )
            if np.isfinite(management_score) and coverage > 0.0:
                proxy_weight = 0.40 * min(1.0, coverage / 100.0)
                alignment_numerator += (management_score - 50.0) / 50.0 * proxy_weight
                alignment_weight += proxy_weight
                alignment_basis.append("MANAGEMENT_EXECUTION_PROXY")
                production_alignment_numerator += (
                    (management_score - 50.0) / 50.0 * proxy_weight
                )
                production_alignment_weight += proxy_weight
                production_alignment_basis.append("MANAGEMENT_EXECUTION_PROXY")
                production_alignment_coverage_points += min(25.0, 0.25 * coverage)
                pm_coverage += min(25.0, 0.25 * coverage)
            project_score = _finite(
                pm.get("project_pipeline_score_observed",
                       pm.get("project_pipeline_score")),
                np.nan,
            )
            project_coverage = max(0.0, min(100.0, _finite(
                pm.get("project_data_coverage",
                       pm.get("project_data_coverage_effective")),
                0.0,
            )))
            if np.isfinite(project_score) and project_coverage > 0.0:
                proxy_weight = 0.22 * min(1.0, project_coverage / 100.0)
                alignment_numerator += (project_score - 50.0) / 50.0 * proxy_weight
                alignment_weight += proxy_weight
                alignment_basis.append("PROJECT_EXECUTION_PROXY")
                pm_coverage += min(15.0, 0.15 * project_coverage)
            pm_coverage += min(12.0, 0.12 * coverage)
        production_alignment_raw = (
            50.0
            + 50.0 * production_alignment_numerator
            / production_alignment_weight
            if production_alignment_weight > 0.0 else 50.0
        )
        production_alignment_raw = max(
            0.0, min(100.0, production_alignment_raw),
        )
        production_alignment_coverage = min(
            100.0,
            18.0 * (positive_alignment_events + negative_alignment_events)
            + 15.0 * official_count
            + production_alignment_coverage_points,
        )
        production_alignment_effective = (
            50.0
            + (production_alignment_raw - 50.0)
            * production_alignment_coverage / 100.0
        )
        fundamental_insider = _finite(
            fundamental.get("insider_ownership_pct"), np.nan,
        )
        if np.isfinite(fundamental_insider):
            insider_pct = (
                100.0 * fundamental_insider
                if fundamental_insider <= 1.0 else fundamental_insider
            )
            alignment_numerator += (
                0.18 if 5.0 <= insider_pct <= 60.0
                else -0.10 if insider_pct > 75.0
                else 0.03 if insider_pct > 0.0 else 0.0
            )
            alignment_weight += 0.18
            direct_alignment_weight += 0.18
            alignment_basis.append("DIRECT_INSIDER_OWNERSHIP")
            pm_coverage += 8.0
        dilution = _finite(
            fundamental.get("history_share_dilution_yoy"), np.nan,
        )
        if np.isfinite(dilution) and dilution > 0.05:
            alignment_numerator -= 0.15 if dilution <= 0.12 else 0.30
            alignment_weight += 0.15 if dilution <= 0.12 else 0.30
            direct_alignment_weight += 0.15 if dilution <= 0.12 else 0.30
            alignment_basis.append("OBSERVED_DILUTION_RISK")
            negative_alignment_events += 1
            pm_coverage += 8.0
        governance_risks = [
            _finite(fundamental.get(name), np.nan)
            for name in (
                "governance_overall_risk",
                "governance_board_risk",
                "governance_audit_risk",
                "governance_shareholder_rights_risk",
            )
        ]
        observed_governance = [
            value for value in governance_risks if np.isfinite(value)
        ]
        if observed_governance:
            average_governance_risk = float(np.mean(observed_governance))
            if average_governance_risk >= 7.0:
                alignment_numerator -= 0.18
                alignment_weight += 0.18
                direct_alignment_weight += 0.18
                alignment_basis.append("GOVERNANCE_RISK_SCORE")
                negative_alignment_events += 1
            pm_coverage += min(10.0, 2.5 * len(observed_governance))
        capital_proxy = _capital_allocation_alignment_proxy(fundamental)
        capital_proxy_score = _finite(capital_proxy.get("score"), np.nan)
        capital_proxy_coverage = max(0.0, min(100.0, _finite(
            capital_proxy.get("coverage_pct"), 0.0,
        )))
        if np.isfinite(capital_proxy_score) and capital_proxy_coverage > 0.0:
            proxy_weight = 0.38 * min(1.0, capital_proxy_coverage / 100.0)
            alignment_numerator += (capital_proxy_score - 50.0) / 50.0 * proxy_weight
            alignment_weight += proxy_weight
            alignment_basis.append("CAPITAL_ALLOCATION_PROXY:" + _text(
                capital_proxy.get("basis"),
            ))
            pm_coverage += min(28.0, 0.40 * capital_proxy_coverage)
        alignment_raw = (
            50.0 + 50.0 * alignment_numerator / alignment_weight
            if alignment_weight > 0.0 else 50.0
        )
        alignment_raw = max(0.0, min(100.0, alignment_raw))
        alignment_coverage = min(
            100.0,
            18.0 * (positive_alignment_events + negative_alignment_events)
            + 15.0 * official_count + pm_coverage,
        )
        alignment_effective = (
            50.0 + (alignment_raw - 50.0) * alignment_coverage / 100.0
        )
        alignment_is_scored = alignment_weight > 0.0 and alignment_coverage > 0.0
        alignment_direct = direct_alignment_weight > 0.0
        capital_allocation_grounded = bool(
            structured_financial.get("eligible")
            and np.isfinite(capital_proxy_score)
            and capital_proxy_coverage > 0.0
        )
        alignment_state = (
            "NOT_SCORED_NO_ALIGNMENT_EVIDENCE"
            if not alignment_is_scored
            else "DIRECT_ALIGNED" if alignment_direct and alignment_effective >= 62.0
            else "DIRECT_MISALIGNED_OR_ENTRENCHMENT_RISK"
            if alignment_direct and alignment_effective <= 38.0
            else "DIRECT_MIXED" if alignment_direct
            else "STRUCTURED_CAPITAL_ALLOCATION_ALIGNED"
            if capital_allocation_grounded and alignment_effective >= 62.0
            else "STRUCTURED_CAPITAL_ALLOCATION_RISK"
            if capital_allocation_grounded and alignment_effective <= 38.0
            else "STRUCTURED_CAPITAL_ALLOCATION_MIXED"
            if capital_allocation_grounded
            else "PROXY_ALIGNED_RESEARCH_ONLY" if alignment_effective >= 62.0
            else "PROXY_MISALIGNED_RESEARCH_ONLY" if alignment_effective <= 38.0
            else "PROXY_MIXED_RESEARCH_ONLY"
        )

        flow = _market_flow_features(
            prepared.get(ticker), silent_map.get(ticker),
        )
        news = news_map.get(ticker, {})
        news_complete = (
            _text(news.get("news_review_status")).upper() == "COMPLETE"
            and _truthy(news.get("provider_query_ok"))
        )
        items_reviewed = int(max(0.0, _finite(news.get("items_reviewed"), 0.0)))
        recent_30 = (
            active[
                (
                    now - active["detected_at"]
                ).dt.total_seconds().div(86400.0).le(30.0)
            ]
            if not active.empty else pd.DataFrame()
        )
        attention_coverage = min(
            100.0,
            (45.0 if news_complete else 0.0)
            + min(25.0, 2.5 * items_reviewed)
            + 0.30 * flow["flow_data_coverage_pct"],
        )
        attention_score = max(
            0.0, min(
                100.0,
                22.0
                + min(28.0, 7.0 * len(recent_30))
                + min(22.0, 12.0 * max(0.0, flow["volume_ratio"] - 1.0))
                + min(18.0, max(0.0, flow["roc20_pct"]))
                + (10.0 if flow["distance_high_pct"] >= -8.0 else 0.0),
            ),
        )
        crowding = max(
            0.0, min(
                100.0,
                25.0 * max(0.0, flow["volume_ratio"] - 1.4)
                + 1.3 * max(0.0, flow["roc20_pct"] - 15.0)
                + (20.0 if flow["distance_high_pct"] >= -3.0 else 0.0)
                + min(20.0, 4.0 * max(0, len(recent_30) - 3)),
            ),
        )
        distribution = (
            flow["distribution_days"] >= 4
            or _text(silent_map.get(ticker, {}).get(
                "silent_accumulation_state",
            )).upper() in {"DISTRIBUTION_RISK", "WEAK_OR_DISTRIBUTION"}
        )
        if (not news_complete) and active.empty:
            adoption_stage = "UNKNOWN_ATTENTION_DATA_PENDING"
        elif distribution and (
            flow["roc20_pct"] > 10.0 or crowding >= 55.0
        ):
            adoption_stage = "EXHAUSTION_OR_DISTRIBUTION"
        elif crowding >= 65.0:
            adoption_stage = "CONSENSUS_CROWDED"
        elif (
            narrative_effective >= 58.0
            and flow["effective_silent"] >= 58.0
            and flow["roc20_pct"] >= 5.0
        ):
            adoption_stage = "EXPANSION"
        elif (
            narrative_effective >= 55.0
            and flow["effective_silent"] >= 52.0
        ):
            adoption_stage = "EARLY_DISCOVERY"
        elif narrative_effective >= 55.0:
            adoption_stage = "SEED_STORY_AHEAD_OF_FLOW"
        else:
            adoption_stage = "DORMANT_OR_UNCONVERTED"
        if active.empty:
            adoption_stage = (
                "EARLY_DISCOVERY_STRUCTURED"
                if structured_eligible
                and narrative_effective >= 55.0
                and flow["effective_silent"] >= 52.0
                else "STRUCTURED_OPERATING_STORY"
                if structured_eligible
                else "FLOW_WITHOUT_SOURCED_STORY"
                if flow["effective_silent"] >= 58.0
                and flow["silent_confidence"] >= 40.0
                else "OPERATING_FACT_PROXY_ONLY"
                if np.isfinite(_finite(operating_proxy.get("score"), np.nan))
                else "NO_ACTIVE_STORY"
            )

        contradiction_events = int(
            (
                pd.to_numeric(
                    negative.get("materiality_score"), errors="coerce",
                ).fillna(0.0).ge(75.0)
            ).sum()
        ) if not negative.empty else 0
        official_critical = (
            negative[
                negative["official_verified"].fillna(False).astype(bool)
                & pd.to_numeric(
                    negative["materiality_score"], errors="coerce",
                ).fillna(0.0).ge(85.0)
            ]
            if not negative.empty else pd.DataFrame()
        )
        hard_block = not official_critical.empty
        narrative_flow_raw = (
            0.42 * narrative_effective
            + 0.38 * flow["effective_silent"]
            + 0.20 * alignment_effective
            - 0.12 * abs(
                narrative_effective - flow["effective_silent"]
            )
            - 0.18 * crowding
            - 5.0 * min(3, contradiction_events)
        )
        narrative_flow_raw = max(0.0, min(100.0, narrative_flow_raw))
        convergence_coverage = min(
            evidence_coverage,
            flow["silent_confidence"],
        )
        convergence_effective = (
            50.0
            + (narrative_flow_raw - 50.0)
            * convergence_coverage / 100.0
        )
        operating_proxy_score = _finite(operating_proxy.get("score"), np.nan)
        operating_proxy_coverage = max(0.0, min(100.0, _finite(
            operating_proxy.get("coverage_pct"), 0.0,
        )))
        research_story_score = (
            narrative_effective if evidence_coverage > 0.0
            else operating_proxy_score
        )
        research_story_coverage = (
            evidence_coverage if evidence_coverage > 0.0
            else min(55.0, operating_proxy_coverage)
        )
        alignment_research_score = (
            alignment_effective if alignment_is_scored else 50.0
        )
        if np.isfinite(research_story_score):
            research_flow_raw = (
                0.42 * research_story_score
                + 0.38 * flow["effective_silent"]
                + 0.20 * alignment_research_score
                - 0.12 * abs(
                    research_story_score - flow["effective_silent"]
                )
                - 0.18 * crowding
                - 5.0 * min(3, contradiction_events)
            )
            research_flow_raw = max(0.0, min(100.0, research_flow_raw))
            research_flow_coverage = min(
                research_story_coverage,
                max(
                    flow["silent_confidence"],
                    0.60 * flow["flow_data_coverage_pct"],
                ),
            )
            research_flow_effective = (
                50.0 + (research_flow_raw - 50.0)
                * research_flow_coverage / 100.0
                if research_flow_coverage > 0.0 else np.nan
            )
        else:
            research_flow_raw = np.nan
            research_flow_coverage = 0.0
            research_flow_effective = np.nan
        if hard_block and distribution:
            convergence_state = "DISTRIBUTION_CONTRADICTION"
        elif adoption_stage in {
            "CONSENSUS_CROWDED", "EXHAUSTION_OR_DISTRIBUTION",
        }:
            convergence_state = "CROWDED_REVERSAL_RISK"
        elif (
            narrative_effective >= 58.0
            and flow["effective_silent"] >= 58.0
        ):
            convergence_state = (
                "CONVERGED_EXPANSION"
                if adoption_stage == "EXPANSION"
                else "CONVERGED_EARLY"
            )
        elif narrative_effective >= 58.0 and flow["effective_silent"] < 52.0:
            convergence_state = "STORY_AHEAD_OF_FLOW"
        elif flow["effective_silent"] >= 58.0 and narrative_effective < 52.0:
            convergence_state = "FLOW_AHEAD_OF_STORY"
        elif evidence_coverage < 20.0 and flow["silent_confidence"] < 40.0:
            convergence_state = "INSUFFICIENT_DATA"
        else:
            convergence_state = "MIXED_WATCH"

        conversion_profile = conversion.get(ticker, {})
        conversion_5 = _finite(
            conversion_profile.get(
                "narrative_conversion_effective_5d_score",
            ),
            50.0,
        )
        conversion_20 = _finite(
            conversion_profile.get(
                "narrative_conversion_effective_20d_score",
            ),
            50.0,
        )
        conversion_60 = _finite(
            conversion_profile.get(
                "narrative_conversion_effective_60d_score",
            ),
            50.0,
        )
        conversion_ready = (
            int(conversion_profile.get(
                "narrative_conversion_resolved_20d", 0,
            )) >= cfg.min_conversion_events
        )
        # Positive narrative remains a production input by design, but its
        # score is source/evidence weighted and no longer borrows confidence
        # from Silent Accumulation.  Flow is a separate ranking key.
        production_reliability = min(1.0, evidence_coverage / 100.0)
        conversion_reliability = (
            1.0 if conversion_ready
            else int(conversion_profile.get(
                "narrative_conversion_resolved_20d", 0,
            )) / max(cfg.min_conversion_events, 1)
        )
        swing_conversion_ready = (
            int(conversion_profile.get(
                "narrative_conversion_resolved_5d", 0,
            )) >= cfg.min_conversion_events
        )
        swing_conversion_reliability = (
            1.0 if swing_conversion_ready
            else int(conversion_profile.get(
                "narrative_conversion_resolved_5d", 0,
            )) / max(cfg.min_conversion_events, 1)
        )
        growth_overlay_signal = (
            0.55 * narrative_effective
            + 0.25 * alignment_effective
            + 0.12 * conversion_20
            + 0.08 * conversion_60
        )
        turnaround_overlay_signal = (
            0.48 * narrative_effective
            + 0.25 * alignment_effective
            + 0.17 * conversion_20
            + 0.10 * conversion_60
        )
        swing_overlay_signal = (
            0.50 * narrative_effective
            + 0.20 * alignment_effective
            + 0.30 * conversion_5
        )
        long_overlay_reliability = min(
            production_reliability,
            0.75 + 0.25 * conversion_reliability,
        )
        swing_overlay_reliability = min(
            production_reliability,
            0.75 + 0.25 * swing_conversion_reliability,
        )
        growth_overlay_unit = (
            (growth_overlay_signal - 50.0) / 50.0
            * long_overlay_reliability
        )
        turnaround_overlay_unit = (
            (turnaround_overlay_signal - 50.0) / 50.0
            * long_overlay_reliability
        )
        swing_overlay_unit = (
            (swing_overlay_signal - 50.0) / 50.0
            * swing_overlay_reliability
        )
        if hard_block:
            growth_overlay_unit = min(growth_overlay_unit, -0.75)
            turnaround_overlay_unit = min(
                turnaround_overlay_unit, -0.75,
            )
            swing_overlay_unit = min(swing_overlay_unit, -0.75)
        reason_parts = [convergence_state]
        if evidence_coverage > 0.0:
            reason_parts.append(
                f"narrative {narrative_effective:.1f} "
                f"[{narrative_evidence_mode}]"
            )
        elif np.isfinite(operating_proxy_score):
            reason_parts.append(
                f"operating facts belum production-eligible {operating_proxy_score:.1f}; "
                f"missing {structured_financial.get('missing') or 'source lineage'}"
            )
        else:
            reason_parts.append("story belum terskor; evidence acquisition belum menghasilkan fakta")
        reason_parts.append(
            f"effective Silent Accumulation {flow['effective_silent']:.1f}"
        )
        if alignment_is_scored:
            reason_parts.append(
                f"alignment {alignment_effective:.1f} [{alignment_state}]"
            )
        else:
            reason_parts.append("alignment belum terskor")
        primary_reason = "; ".join(reason_parts)
        primary_risk = (
            _text(official_critical.iloc[0].get("headline"))
            if not official_critical.empty
            else "Crowding/reversal risk meningkat"
            if crowding >= 55.0
            else "Narrative belum memiliki sampel conversion point-in-time"
            if not conversion_ready
            else "Narrative dapat gagal terkonversi menjadi fundamental/flow"
        )
        emir_profile = build_emir_method_profile(
            ticker=ticker,
            frame=prepared.get(ticker),
            active_events=active,
            outcomes=conversion_profile,
            fundamental=fundamental,
            silent_profile={
                **silent_map.get(ticker, {}),
                "effective_silent_accumulation_score": flow["effective_silent"],
            },
            narrative_effective_score=narrative_effective,
            narrative_evidence_coverage_pct=evidence_coverage,
            narrative_evidence_mode=narrative_evidence_mode,
            alignment_effective_score=alignment_effective,
            alignment_coverage_pct=alignment_coverage,
            adoption_stage=adoption_stage,
            crowding_risk_score=crowding,
            hard_block=hard_block,
            growth_max_adjustment_points=cfg.emir_growth_max_adjustment_points,
            turnaround_max_adjustment_points=cfg.emir_turnaround_max_adjustment_points,
            swing_max_adjustment_points=cfg.emir_swing_max_adjustment_points,
        )
        acquisition_missing = _text(structured_financial.get("missing"))
        if narrative_evidence_mode in {"SOURCE_EVENT", "EVENT_PLUS_STRUCTURED_FINANCIAL", "EVENT_PLUS_PARTIAL_STRUCTURED_FINANCIAL"}:
            acquisition_status = "SOURCE_EVENT_ACQUIRED"
        elif narrative_evidence_mode == "STRUCTURED_FINANCIAL":
            acquisition_status = "STRUCTURED_FINANCIAL_ACQUIRED"
        elif narrative_evidence_mode == "STRUCTURED_FINANCIAL_PARTIAL":
            acquisition_status = "STRUCTURED_FINANCIAL_PARTIAL_ACQUIRED"
        elif not fundamental:
            acquisition_status = "NO_FUNDAMENTAL_HISTORY_OR_CACHE"
        elif acquisition_missing:
            acquisition_status = "PARTIAL_MISSING_" + acquisition_missing.replace(" | ", "_")
        else:
            acquisition_status = "NEWS_OR_DISCLOSURE_REFRESH_PENDING"
        row = {
            "ticker": ticker,
            "narrative_engine_version": NARRATIVE_ENGINE_VERSION,
            "narrative_as_of": now.isoformat(),
            "narrative_state": (
                "SOURCE_GROUNDED"
                if narrative_evidence_mode in {"SOURCE_EVENT", "EVENT_PLUS_STRUCTURED_FINANCIAL", "EVENT_PLUS_PARTIAL_STRUCTURED_FINANCIAL"}
                and evidence_coverage >= 40.0
                else "STRUCTURED_FINANCIAL_GROUNDED"
                if narrative_evidence_mode == "STRUCTURED_FINANCIAL"
                else "STRUCTURED_FINANCIAL_PARTIAL"
                if narrative_evidence_mode == "STRUCTURED_FINANCIAL_PARTIAL"
                else "LIMITED_SOURCE_EVIDENCE"
                if evidence_coverage > 0.0
                else "EVIDENCE_ACQUISITION_INCOMPLETE"
            ),
            "narrative_score_state": (
                "SCORED_SOURCE_GROUNDED"
                if narrative_evidence_mode in {"SOURCE_EVENT", "EVENT_PLUS_STRUCTURED_FINANCIAL", "EVENT_PLUS_PARTIAL_STRUCTURED_FINANCIAL"}
                else "SCORED_STRUCTURED_FINANCIAL_EVIDENCE"
                if narrative_evidence_mode == "STRUCTURED_FINANCIAL"
                else "SCORED_PARTIAL_STRUCTURED_FINANCIAL_EVIDENCE"
                if narrative_evidence_mode == "STRUCTURED_FINANCIAL_PARTIAL"
                else "NOT_SCORED_EVIDENCE_ACQUISITION_INCOMPLETE"
            ),
            "narrative_score": (
                round(narrative_raw, 1) if evidence_coverage > 0.0 else np.nan
            ),
            "narrative_effective_score": (
                round(narrative_effective, 1)
                if evidence_coverage > 0.0 else np.nan
            ),
            "narrative_event_score": (
                round(event_narrative_raw, 1)
                if event_evidence_coverage > 0.0 else np.nan
            ),
            "narrative_event_effective_score": (
                round(narrative_event_effective, 1)
                if event_evidence_coverage > 0.0 else np.nan
            ),
            "narrative_event_coverage_pct": round(
                event_evidence_coverage, 1,
            ),
            "narrative_rank_neutral_score": (
                round(narrative_effective, 1) if evidence_coverage > 0.0 else np.nan
            ),
            "narrative_evidence_coverage_pct": round(evidence_coverage, 1),
            "narrative_evidence_mode": narrative_evidence_mode,
            "structured_financial_evidence_state": _text(structured_financial.get("state")),
            "structured_financial_evidence_coverage_pct": round(structured_coverage, 1),
            "structured_financial_source_count": int(structured_financial.get("source_count", 0)),
            "structured_financial_latest_period": _text(structured_financial.get("latest_period")),
            "structured_financial_production_eligible": structured_production_eligible,
            "evidence_acquisition_missing": acquisition_missing,
            "evidence_acquisition_status": acquisition_status,
            "evidence_acquisition_complete": bool(evidence_coverage > 0.0),
            "operating_narrative_proxy_score": (
                round(operating_proxy_score, 1)
                if np.isfinite(operating_proxy_score) else np.nan
            ),
            "operating_narrative_proxy_coverage_pct": round(
                operating_proxy_coverage, 1,
            ),
            "operating_narrative_proxy_state": _text(
                operating_proxy.get("state"),
            ),
            "operating_narrative_proxy_basis": _text(
                operating_proxy.get("basis"),
            ),
            "narrative_context_score": (
                round(narrative_effective, 1)
                if evidence_coverage > 0.0
                else round(operating_proxy_score, 1)
                if np.isfinite(operating_proxy_score) else np.nan
            ),
            "narrative_context_state": (
                "SOURCE_GROUNDED"
                if narrative_evidence_mode in {"SOURCE_EVENT", "EVENT_PLUS_STRUCTURED_FINANCIAL", "EVENT_PLUS_PARTIAL_STRUCTURED_FINANCIAL"}
                else "STRUCTURED_FINANCIAL_GROUNDED"
                if narrative_evidence_mode == "STRUCTURED_FINANCIAL"
                else "STRUCTURED_FINANCIAL_PARTIAL"
                if narrative_evidence_mode == "STRUCTURED_FINANCIAL_PARTIAL"
                else "OPERATING_PROXY_RESEARCH_ONLY"
                if np.isfinite(operating_proxy_score)
                else "NOT_SCORED"
            ),
            "narrative_event_count": int(len(all_events)),
            "narrative_active_event_count": int(len(active)),
            "narrative_missing_source_event_count": int(
                all_events.get(
                    "source_state", pd.Series("", index=all_events.index),
                ).fillna("").astype(str).str.upper().eq("MISSING_SOURCE").sum()
            ) if not all_events.empty else 0,
            "narrative_inactive_lifecycle_event_count": int(
                (
                    ~all_events.get(
                        "event_status",
                        pd.Series("ACTIVE", index=all_events.index),
                    ).fillna("ACTIVE").astype(str).str.upper().isin(
                        {"ACTIVE", "DISPUTED"}
                    )
                ).sum()
            ) if not all_events.empty else 0,
            "narrative_entity_unverified_event_count": int(
                all_events.get(
                    "entity_match_state", pd.Series("", index=all_events.index),
                ).fillna("").astype(str).str.upper().eq(
                    "AMBIGUOUS_TICKER_UNVERIFIED"
                ).sum()
            ) if not all_events.empty else 0,
            "narrative_event_cluster_count": int(cluster_count),
            "narrative_corroborated_cluster_count": int(
                corroborated_clusters,
            ),
            "narrative_positive_event_count": int(len(positive)),
            "narrative_negative_event_count": int(len(negative)),
            "narrative_official_event_count": official_count,
            "latest_narrative_event": _text(latest.get("headline")),
            "latest_narrative_event_type": _text(latest.get("event_type")),
            "latest_narrative_event_date": _text(latest.get("event_date")),
            "narrative_source_quality_score": (
                round(float(pd.to_numeric(
                    active.get("source_quality_score"), errors="coerce",
                ).mean()), 1) if not active.empty else np.nan
            ),
            "narrative_novelty_score": (
                round(float(pd.to_numeric(
                    active.get("novelty_score"), errors="coerce",
                ).mean()), 1) if not active.empty else np.nan
            ),
            "narrative_financial_bridge_score": (
                round(bridge_mean, 1) if not active.empty else np.nan
            ),
            "issuer_alignment_score": (
                round(alignment_raw, 1) if alignment_is_scored else np.nan
            ),
            "issuer_alignment_effective_score": (
                round(alignment_effective, 1)
                if alignment_is_scored else np.nan
            ),
            "issuer_action_alignment_score": (
                round(production_alignment_raw, 1)
                if production_alignment_weight > 0.0 else np.nan
            ),
            "issuer_action_alignment_effective_score": (
                round(production_alignment_effective, 1)
                if production_alignment_weight > 0.0 else np.nan
            ),
            "issuer_action_alignment_coverage_pct": round(
                production_alignment_coverage, 1,
            ),
            "issuer_action_alignment_basis": " | ".join(
                dict.fromkeys(
                    value for value in production_alignment_basis if value
                )
            ),
            "issuer_alignment_rank_neutral_score": (
                round(alignment_effective, 1) if alignment_is_scored else np.nan
            ),
            "issuer_alignment_coverage_pct": round(alignment_coverage, 1),
            "issuer_alignment_state": alignment_state,
            "issuer_alignment_score_state": (
                "SCORED_DIRECT_AND_PROXY"
                if alignment_is_scored and alignment_direct
                else "SCORED_STRUCTURED_CAPITAL_ALLOCATION"
                if alignment_is_scored and capital_allocation_grounded
                else "SCORED_PROXY_RESEARCH_ONLY"
                if alignment_is_scored else "NOT_SCORED_NO_ALIGNMENT_EVIDENCE"
            ),
            "issuer_alignment_evidence_basis": " | ".join(
                dict.fromkeys(value for value in alignment_basis if value)
            ),
            "issuer_alignment_direct_weight": round(
                direct_alignment_weight, 4,
            ),
            "issuer_alignment_total_weight": round(alignment_weight, 4),
            "issuer_alignment_positive_events": positive_alignment_events,
            "issuer_alignment_negative_events": negative_alignment_events,
            "retail_adoption_stage": adoption_stage,
            "retail_adoption_proxy_score": round(attention_score, 1),
            "retail_adoption_proxy_coverage_pct": round(
                attention_coverage, 1,
            ),
            "retail_proxy_disclaimer": (
                "MARKET_ATTENTION_PROXY_NOT_DIRECT_RETAIL_IDENTITY"
            ),
            "narrative_crowding_risk_score": round(crowding, 1),
            "narrative_flow_convergence_score": (
                round(narrative_flow_raw, 1)
                if evidence_coverage > 0.0
                and flow["flow_data_coverage_pct"] > 0.0 else np.nan
            ),
            "narrative_flow_effective_score": (
                round(convergence_effective, 1)
                if convergence_coverage > 0.0 else np.nan
            ),
            "narrative_flow_rank_neutral_score": round(
                convergence_effective, 1,
            ),
            "narrative_flow_convergence_coverage_pct": round(
                convergence_coverage, 1,
            ),
            "narrative_flow_convergence_state": convergence_state,
            "narrative_flow_score_state": (
                "SCORED_SOURCE_GROUNDED"
                if convergence_coverage > 0.0
                and narrative_evidence_mode in {"SOURCE_EVENT", "EVENT_PLUS_STRUCTURED_FINANCIAL"}
                else "SCORED_STRUCTURED_FINANCIAL_GROUNDED"
                if convergence_coverage > 0.0
                and narrative_evidence_mode == "STRUCTURED_FINANCIAL"
                else "STORY_SCORED_FLOW_CONFIDENCE_PENDING"
                if evidence_coverage > 0.0
                else "PROXY_RESEARCH_ONLY"
                if np.isfinite(research_flow_effective)
                else "NOT_SCORED"
            ),
            "narrative_flow_research_score": (
                round(research_flow_effective, 1)
                if np.isfinite(research_flow_effective) else np.nan
            ),
            "narrative_flow_research_raw_score": (
                round(research_flow_raw, 1)
                if np.isfinite(research_flow_raw) else np.nan
            ),
            "narrative_flow_research_coverage_pct": round(
                research_flow_coverage, 1,
            ),
            "narrative_silent_integration_state": (
                "DIAGNOSTIC_CONVERGENCE_SECONDARY_FLOW_KEY"
                if convergence_coverage >= 40.0
                else "SHADOW_PENDING_EVIDENCE"
            ),
            "narrative_production_policy": (
                "OFFICIAL_FIRST_EVENT_OR_DATED_STRUCTURED_FINANCIAL_"
                "SILENT_SEPARATE_RANK_KEY"
            ),
            "narrative_contradiction_count": contradiction_events,
            "narrative_hard_block": bool(hard_block),
            "narrative_primary_reason": primary_reason,
            "narrative_primary_risk": primary_risk,
            "narrative_overlay_reliability_pct": round(
                100.0 * long_overlay_reliability, 1,
            ),
            "narrative_swing_overlay_reliability_pct": round(
                100.0 * swing_overlay_reliability, 1,
            ),
            "narrative_growth_rank_adjustment": round(
                cfg.growth_max_adjustment_points * growth_overlay_unit, 2,
            ),
            "narrative_turnaround_rank_adjustment": round(
                cfg.turnaround_max_adjustment_points
                * turnaround_overlay_unit,
                2,
            ),
            "narrative_swing_rank_adjustment": round(
                cfg.swing_max_adjustment_points * swing_overlay_unit, 2,
            ),
            "narrative_news_collection_state": (
                "COMPLETE" if news_complete
                else "INCOMPLETE_OR_NOT_REQUESTED"
            ),
            "narrative_items_reviewed": items_reviewed,
            "narrative_flow_proxy_score": round(
                flow["effective_silent"], 1,
            ),
        }
        # Keep direct forward evidence as an explicit profile contract.  It is
        # consumed by the Future pillar without borrowing realised growth,
        # profitability, or management outcome proxies from Business Quality.
        row.update(forward_evidence)
        row.update(emir_profile)
        row.update({
            f"narrative_conversion_rate_{horizon}d_pct":
                conversion_profile.get(
                    f"narrative_conversion_rate_{horizon}d_pct", np.nan,
                )
            for horizon in (5, 20, 60)
        })
        row.update({
            f"narrative_conversion_effective_{horizon}d_score":
                conversion_profile.get(
                    f"narrative_conversion_effective_{horizon}d_score",
                    np.nan,
                )
            for horizon in (5, 20, 60)
        })
        row.update({
            f"narrative_conversion_expectancy_{horizon}d_pct":
                conversion_profile.get(
                    f"narrative_conversion_expectancy_{horizon}d_pct",
                    np.nan,
                )
            for horizon in (5, 20, 60)
        })
        row.update({
            f"narrative_conversion_resolved_{horizon}d":
                int(conversion_profile.get(
                    f"narrative_conversion_resolved_{horizon}d", 0,
                ))
            for horizon in (5, 20, 60)
        })
        row.update({
            f"narrative_conversion_state_{horizon}d":
                conversion_profile.get(
                    f"narrative_conversion_state_{horizon}d",
                    "NO_RESOLVED_POINT_IN_TIME_SAMPLE",
                )
            for horizon in (5, 20, 60)
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _sanitize_narrative_profile_placeholders(
    profiles: pd.DataFrame | None,
) -> pd.DataFrame:
    """Convert legacy neutral placeholders into explicit missing values.

    A score of 50 is meaningful only when it was actually estimated.  Older
    caches used 50 as a no-data sentinel, which made every issuer appear to
    have identical narrative and alignment.
    """
    if profiles is None:
        return pd.DataFrame()
    out = profiles.copy()
    if out.empty:
        return out
    narrative_coverage = pd.to_numeric(
        out.get(
            "narrative_evidence_coverage_pct",
            pd.Series(0.0, index=out.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    no_narrative = narrative_coverage.le(0.0)
    for column in (
        "narrative_score", "narrative_effective_score",
        "narrative_event_score", "narrative_event_effective_score",
        "narrative_flow_convergence_score",
        "narrative_flow_effective_score",
    ):
        if column in out:
            out.loc[no_narrative, column] = np.nan
    if "narrative_score_state" not in out:
        out["narrative_score_state"] = "SCORED_SOURCE_GROUNDED"
    out.loc[no_narrative, "narrative_score_state"] = (
        "NOT_SCORED_NO_ACTIVE_SOURCE_EVENT"
    )
    if "narrative_state" in out:
        out.loc[no_narrative, "narrative_state"] = (
            "NOT_SCORED_NO_ACTIVE_SOURCE_EVENT"
        )
    alignment_coverage = pd.to_numeric(
        out.get(
            "issuer_alignment_coverage_pct",
            pd.Series(0.0, index=out.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    no_alignment = alignment_coverage.le(0.0)
    for column in (
        "issuer_alignment_score", "issuer_alignment_effective_score",
    ):
        if column in out:
            out.loc[no_alignment, column] = np.nan
    if "issuer_alignment_score_state" not in out:
        out["issuer_alignment_score_state"] = "SCORED"
    out.loc[no_alignment, "issuer_alignment_score_state"] = (
        "NOT_SCORED_NO_ALIGNMENT_EVIDENCE"
    )
    if "issuer_alignment_state" in out:
        out.loc[no_alignment, "issuer_alignment_state"] = (
            "NOT_SCORED_NO_ALIGNMENT_EVIDENCE"
        )
    for horizon in (5, 20, 60):
        resolved_column = f"narrative_conversion_resolved_{horizon}d"
        resolved = pd.to_numeric(
            out.get(resolved_column, pd.Series(0, index=out.index)),
            errors="coerce",
        ).fillna(0)
        no_sample = resolved.le(0)
        for column in (
            f"narrative_conversion_rate_{horizon}d_pct",
            f"narrative_conversion_effective_{horizon}d_score",
            f"narrative_conversion_expectancy_{horizon}d_pct",
            f"narrative_conversion_cohort_prior_{horizon}d_pct",
        ):
            if column in out:
                out.loc[no_sample, column] = np.nan
        state_column = f"narrative_conversion_state_{horizon}d"
        if state_column in out:
            out.loc[no_sample, state_column] = (
                "NO_RESOLVED_POINT_IN_TIME_SAMPLE"
            )
    return out


def attach_narrative_profiles(
    frame: pd.DataFrame | None,
    profiles: pd.DataFrame | None,
) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    if frame.empty or profiles is None or profiles.empty or "ticker" not in frame:
        return frame.copy()
    out = frame.copy()
    out["_narrative_ticker"] = out["ticker"].map(_ticker)
    overlay = _sanitize_narrative_profile_placeholders(profiles)
    overlay["_narrative_ticker"] = overlay["ticker"].map(_ticker)
    overlay = overlay.drop(columns="ticker", errors="ignore").drop_duplicates(
        "_narrative_ticker", keep="last",
    )
    return out.merge(
        overlay, on="_narrative_ticker", how="left",
    ).drop(columns="_narrative_ticker")


def build_narrative_intelligence(
    *,
    prepared: Mapping[str, pd.DataFrame],
    fundamentals: pd.DataFrame | None = None,
    news_review: pd.DataFrame | None = None,
    project_management: pd.DataFrame | None = None,
    market_status: pd.DataFrame | None = None,
    existing_events: pd.DataFrame | None = None,
    existing_outcomes: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
    silent_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    scan_config: Any | None = None,
    as_of: Any | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = NarrativeConfig.from_scan_config(scan_config)
    if not cfg.enabled:
        profiles = pd.DataFrame({
            "ticker": list(prepared),
            "narrative_state": "DISABLED",
            "narrative_score_state": "DISABLED",
            "narrative_effective_score": np.nan,
            "narrative_evidence_coverage_pct": 0.0,
            "narrative_event_score": np.nan,
            "narrative_event_effective_score": np.nan,
            "narrative_event_coverage_pct": 0.0,
            "issuer_alignment_effective_score": np.nan,
            "issuer_alignment_coverage_pct": 0.0,
            "issuer_action_alignment_score": np.nan,
            "issuer_action_alignment_effective_score": np.nan,
            "issuer_action_alignment_coverage_pct": 0.0,
            "issuer_alignment_state": "DISABLED",
            "issuer_alignment_score_state": "DISABLED",
            "retail_adoption_stage": "DISABLED",
            "narrative_flow_effective_score": np.nan,
            "narrative_flow_score_state": "DISABLED",
            "narrative_hard_block": False,
            "narrative_growth_rank_adjustment": 0.0,
            "narrative_turnaround_rank_adjustment": 0.0,
            "narrative_swing_rank_adjustment": 0.0,
            "stock_universe_familiarity_score": np.nan,
            "stock_universe_familiarity_coverage_pct": 0.0,
            "stock_universe_familiarity_state": "DISABLED",
            "smart_money_behavior_score": np.nan,
            "smart_money_behavior_coverage_pct": 0.0,
            "smart_money_behavior_state": "DISABLED",
            "smart_money_flow_evidence_mode": "DISABLED",
            "narrative_lifecycle_score": np.nan,
            "narrative_lifecycle_state": "DISABLED",
            "flow_preceded_narrative": False,
            "emir_method_score": np.nan,
            "emir_method_coverage_pct": 0.0,
            "emir_method_score_state": "DISABLED",
            "emir_method_state": "DISABLED",
            "emir_method_production_eligible": False,
            "emir_method_reliability_pct": 0.0,
            "emir_selection_reason": "Narrative engine disabled by configuration.",
            "emir_risk_flags": "ENGINE_DISABLED",
            "emir_position_cap_pct": 0.0,
            "emir_growth_rank_adjustment": 0.0,
            "emir_turnaround_rank_adjustment": 0.0,
            "emir_swing_rank_adjustment": 0.0,
        })
        return {
            "events": pd.DataFrame(),
            "outcomes": (
                existing_outcomes.copy()
                if isinstance(existing_outcomes, pd.DataFrame)
                else pd.DataFrame()
            ),
            "profiles": profiles,
            "audit": pd.DataFrame([{
                "state": "DISABLED",
                "detail": "Narrative engine disabled by configuration.",
            }]),
        }
    events = build_narrative_events(
        fundamentals=fundamentals,
        news_review=news_review,
        project_management=project_management,
        market_status=market_status,
        existing_events=existing_events,
        as_of=as_of,
        config=cfg,
    )
    outcomes = update_narrative_event_outcomes(
        events, existing_outcomes, prepared,
        benchmark=benchmark, as_of=as_of, config=cfg,
    )
    profiles = build_narrative_profiles(
        list(prepared),
        prepared=prepared,
        events=events,
        outcomes=outcomes,
        fundamentals=fundamentals,
        news_review=news_review,
        project_management=project_management,
        silent_profiles=silent_profiles,
        as_of=as_of,
        config=cfg,
    )
    profiles = _sanitize_narrative_profile_placeholders(profiles)
    audit = pd.DataFrame([
        {
            "stage": "Universe profiles",
            "rows": len(profiles),
            "state": "READY" if len(profiles) else "EMPTY",
            "meaning": "Every prepared ticker receives an explicit evidence state.",
        },
        {
            "stage": "Point-in-time events",
            "rows": len(events),
            "state": (
                "READY" if len(events) else "NO_EVENT_EVIDENCE"
            ),
            "meaning": "No event is fabricated when sources are absent.",
        },
        {
            "stage": "Resolved conversion 20D",
            "rows": int(pd.to_numeric(
                outcomes.get(
                    "net_excess_return_20d_pct",
                    pd.Series(np.nan, index=outcomes.index),
                ),
                errors="coerce",
            ).notna().sum()) if not outcomes.empty else 0,
            "state": "SHADOW_UNTIL_MIN_SAMPLE",
            "meaning": (
                f"Minimum {cfg.min_conversion_events} resolved events per "
                "ticker before full production influence."
            ),
        },
        {
            "stage": "Source-grounded narrative profiles",
            "rows": int(
                pd.to_numeric(
                    profiles.get(
                        "narrative_evidence_coverage_pct",
                        pd.Series(0.0, index=profiles.index),
                    ),
                    errors="coerce",
                ).fillna(0.0).gt(0.0).sum()
            ) if not profiles.empty else 0,
            "state": "PRODUCTION_SCORED_ONLY_WITH_SOURCE",
            "meaning": "Nilai narrative produksi kosong bila event aktif bersumber tidak tersedia.",
        },
        {
            "stage": "Operating proxy research-only",
            "rows": int(pd.to_numeric(
                profiles.get(
                    "operating_narrative_proxy_score",
                    pd.Series(np.nan, index=profiles.index),
                ), errors="coerce",
            ).notna().sum()) if not profiles.empty else 0,
            "state": "DUE_DILIGENCE_ORDER_ONLY",
            "meaning": "Proxy fakta operasi membedakan emiten tanpa memalsukan narrative publik.",
        },
        {
            "stage": "Emir public-framework reconstruction",
            "rows": int(
                profiles.get(
                    "emir_method_production_eligible",
                    pd.Series(False, index=profiles.index),
                ).fillna(False).astype(bool).sum()
            ) if not profiles.empty else 0,
            "state": "PUBLIC_PROCESS_MODEL_NOT_TRACK_RECORD_REPLICATION",
            "meaning": (
                "Ranks stock-universe familiarity, narrative lifecycle, flow confirmation, "
                "crowding/distribution and risk-based position caps. Broker summary remains "
                "optional direct evidence; OHLCV is labelled as a proxy when absent."
            ),
        },
        {
            "stage": "Official critical contradictions",
            "rows": int(
                profiles.get(
                    "narrative_hard_block",
                    pd.Series(False, index=profiles.index),
                ).fillna(False).astype(bool).sum()
            ) if not profiles.empty else 0,
            "state": "HARD_ALLOCATION_GATE",
            "meaning": "Research remains visible; capital allocation is blocked.",
        },
    ])
    return {
        "events": events,
        "outcomes": outcomes,
        "profiles": profiles,
        "audit": audit,
    }


__all__ = [
    "NARRATIVE_ENGINE_VERSION",
    "NARRATIVE_EVENT_SCHEMA_VERSION",
    "NARRATIVE_OUTCOME_SCHEMA_VERSION",
    "NarrativeConfig",
    "parse_narrative_event_csv",
    "build_narrative_events",
    "update_narrative_event_outcomes",
    "build_narrative_profiles",
    "attach_narrative_profiles",
    "build_narrative_intelligence",
]
