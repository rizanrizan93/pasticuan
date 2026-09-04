from __future__ import annotations

"""Yahoo-direct ownership facts for Phase 5.6.

This module deliberately keeps three evidence families separate:
- aggregate Yahoo ownership concentration metrics;
- named Yahoo institutional/mutual-fund holders when explicitly returned;
- existing IDX/KSEI/issuer ownership evidence, which is never relabelled here.

No free-float, beneficial-owner, broker/bandar, score, rank, or gate inference is
performed in this producer.
"""

from datetime import date, datetime, timezone
import hashlib
import json
import math
from typing import Any, Mapping

from shared_evidence_hub import HubConfig, SupabaseEvidenceBackend

CONCENTRATION_TABLE = "evidence_ownership_concentration_metrics"
SHAREHOLDER_TABLE = "evidence_shareholder_profiles"
CONCENTRATION_PROVIDER = "YAHOO_DIRECT_OWNERSHIP_CONCENTRATION"
INSTITUTIONAL_PROVIDER = "YAHOO_DIRECT_INSTITUTIONAL_HOLDER"
MUTUAL_FUND_PROVIDER = "YAHOO_DIRECT_MUTUAL_FUND_HOLDER"
LINEAGE_STATE = "PUBLIC_PROVIDER_OBSERVED_NOT_IDX_KSEI"

MAJOR_LABELS = {
    "insidersPercentHeld": ("insiders_held_pct", "PERCENT"),
    "institutionsPercentHeld": ("institutions_held_pct", "PERCENT"),
    "institutionsFloatPercentHeld": ("institutions_float_held_pct", "PERCENT"),
    "institutionsCount": ("institutions_count", "COUNT"),
}


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.endswith(".JK"):
        text = text[:-3]
    return text


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _source_url(ticker: str) -> str:
    return f"https://finance.yahoo.com/quote/{_ticker(ticker)}.JK/holders/"


def normalize_major_holders(
    ticker: str,
    major_holders: Any,
    *,
    observed_on: date,
    fetched_at: datetime | None = None,
) -> list[dict[str, Any]]:
    code = _ticker(ticker)
    if not code or major_holders is None or not hasattr(major_holders, "iterrows"):
        return []
    stamp = (fetched_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    source_period = observed_on.replace(day=1).isoformat()
    output: list[dict[str, Any]] = []
    for index, series in major_holders.iterrows():
        label = str(index or "").strip()
        mapping = MAJOR_LABELS.get(label)
        if mapping is None:
            continue
        metric_name, unit = mapping
        value = _number(series.get("Value") if hasattr(series, "get") else None)
        if value is None:
            continue
        canonical = value * 100.0 if unit == "PERCENT" else value
        if unit == "PERCENT" and not 0 <= canonical <= 100:
            continue
        if unit == "COUNT" and canonical < 0:
            continue
        identity = {
            "provider": CONCENTRATION_PROVIDER,
            "ticker": code,
            "source_period": source_period,
            "metric_name": metric_name,
            "metric_value": canonical,
        }
        output.append({
            **identity,
            "observed_on": observed_on.isoformat(),
            "metric_unit": unit,
            "source_record_hash": _hash(identity),
            "source_url": _source_url(code),
            "source_verified": True,
            "official_verified": False,
            "source_authority": "PUBLIC_PROVIDER",
            "lineage_state": LINEAGE_STATE,
            "validation_state": "VALID",
            "fetched_at": stamp,
        })
    return output


def normalize_named_holders(
    ticker: str,
    frame: Any,
    *,
    provider: str,
    holder_category: str,
    observed_on: date,
    fetched_at: datetime | None = None,
) -> list[dict[str, Any]]:
    code = _ticker(ticker)
    if not code or frame is None or not hasattr(frame, "iterrows"):
        return []
    stamp = (fetched_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    output: list[dict[str, Any]] = []
    for _, series in frame.iterrows():
        get = series.get if hasattr(series, "get") else lambda key, default=None: default
        name = str(get("Holder") or "").strip()
        shares = _number(get("Shares"))
        pct = _number(get("pctHeld"))
        raw_date = get("Date Reported")
        try:
            report_date = raw_date.date() if hasattr(raw_date, "date") else date.fromisoformat(str(raw_date)[:10])
        except (TypeError, ValueError):
            report_date = None
        if not name or report_date is None or (shares is None and pct is None):
            continue
        ownership_pct = pct * 100.0 if pct is not None else None
        if shares is not None and shares < 0:
            continue
        if ownership_pct is not None and not 0 <= ownership_pct <= 100:
            continue
        identity_hash = hashlib.sha256(f"{name.casefold()}|{holder_category.casefold()}".encode("utf-8")).hexdigest()
        payload = {
            "provider": provider,
            "ticker": code,
            "source_period": report_date.isoformat(),
            "holder_identity_hash": identity_hash,
            "holder_name": name,
            "shares_held": shares,
            "ownership_percentage": ownership_pct,
            "holder_category": holder_category,
        }
        output.append({
            **payload,
            "observed_on": observed_on.isoformat(),
            "source_profile_hash": _hash(payload),
            "source_url": _source_url(code),
            "source_verified": True,
            "validation_state": "VALID",
            "fetched_at": stamp,
        })
    return output


def validate_concentration_rows(rows: list[Mapping[str, Any]]) -> tuple[bool, str]:
    if not rows:
        return False, "EMPTY_RESPONSE"
    seen: set[str] = set()
    for row in rows:
        name = str(row.get("metric_name") or "")
        value = _number(row.get("metric_value"))
        unit = str(row.get("metric_unit") or "")
        if name not in {value[0] for value in MAJOR_LABELS.values()} or name in seen or value is None:
            return False, "PARSE_FAILURE"
        seen.add(name)
        if unit == "PERCENT" and not 0 <= value <= 100:
            return False, "CONTEXT_REJECTED"
        if unit == "COUNT" and value < 0:
            return False, "CONTEXT_REJECTED"
        if row.get("official_verified") not in {False, 0} or row.get("source_authority") != "PUBLIC_PROVIDER":
            return False, "PROVENANCE_REJECTED"
        if row.get("lineage_state") != LINEAGE_STATE or row.get("validation_state") != "VALID":
            return False, "PROVENANCE_REJECTED"
    return True, "VALID"


class YahooOwnershipEvidence:
    def __init__(
        self,
        client_id: str = "PASTICUAN",
        *,
        config: HubConfig | None = None,
        backend: SupabaseEvidenceBackend | None = None,
    ) -> None:
        self.config = config or HubConfig.from_environment(client_id=client_id)
        self.backend = backend or SupabaseEvidenceBackend(self.config)

    @property
    def ready(self) -> bool:
        return bool(self.config.ready)

    def refresh(self, ticker: str, *, observed_on: date | None = None) -> dict[str, Any]:
        if not self.ready:
            return {"ticker": _ticker(ticker), "state": "ENVIRONMENT_BLOCKED", "rows": 0}
        import yfinance as yf

        code = _ticker(ticker)
        today = observed_on or date.today()
        fetched_at = datetime.now(timezone.utc)
        handle = yf.Ticker(f"{code}.JK")
        major = handle.major_holders
        concentration = normalize_major_holders(code, major, observed_on=today, fetched_at=fetched_at)
        valid, reason = validate_concentration_rows(concentration)
        if not valid:
            return {"ticker": code, "state": reason, "rows": 0, "concentration_rows": 0, "named_rows": 0}

        institutional = normalize_named_holders(
            code, handle.institutional_holders,
            provider=INSTITUTIONAL_PROVIDER,
            holder_category="INSTITUTIONAL_DISCLOSURE",
            observed_on=today,
            fetched_at=fetched_at,
        )
        mutual = normalize_named_holders(
            code, handle.mutualfund_holders,
            provider=MUTUAL_FUND_PROVIDER,
            holder_category="MUTUAL_FUND_DISCLOSURE",
            observed_on=today,
            fetched_at=fetched_at,
        )

        written_concentration = self.backend.upsert_rows(
            CONCENTRATION_TABLE,
            concentration,
            conflict=("provider", "ticker", "source_period", "metric_name"),
        )
        named = institutional + mutual
        written_named: list[dict[str, Any]] = []
        if named:
            written_named = self.backend.upsert_rows(
                SHAREHOLDER_TABLE,
                named,
                conflict=("provider", "ticker", "source_period", "holder_identity_hash"),
            )
        return {
            "ticker": code,
            "state": "REFRESHED",
            "rows": len(written_concentration) + len(written_named),
            "concentration_rows": len(written_concentration),
            "named_rows": len(written_named),
            "institutional_rows": len(institutional),
            "mutual_fund_rows": len(mutual),
            "provider": CONCENTRATION_PROVIDER,
            "policy": "FACTS_ONLY_NO_FREE_FLOAT_OR_KSEI_INFERENCE",
        }


__all__ = [
    "CONCENTRATION_PROVIDER",
    "CONCENTRATION_TABLE",
    "INSTITUTIONAL_PROVIDER",
    "LINEAGE_STATE",
    "MAJOR_LABELS",
    "MUTUAL_FUND_PROVIDER",
    "YahooOwnershipEvidence",
    "normalize_major_holders",
    "normalize_named_holders",
    "validate_concentration_rows",
]
