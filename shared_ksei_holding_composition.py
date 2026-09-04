from __future__ import annotations

"""Official KSEI monthly local/foreign holding composition evidence.

KSEI publishes a monthly ``BalanceposEfekYYYYMMDD.zip`` archive.  We retain a
small scanner-neutral factual projection per equity: issued/security count,
scripless ratio, local share of scripless holdings, and foreign share of
scripless holdings.  These facts are KSEI registration/composition evidence;
they are never labelled regulatory free float or beneficial ownership.
"""

from calendar import monthrange
from datetime import date, datetime, timezone
from hashlib import sha256
from io import BytesIO, StringIO
from typing import Any, Iterable, Mapping
from urllib.parse import urljoin
from zipfile import ZipFile
import csv
import re

import numpy as np
import pandas as pd
import requests

from shared_evidence_hub import HubConfig, SupabaseEvidenceBackend

ARCHIVE_PAGE = "https://web.ksei.co.id/archive_download/holding_composition"
CATEGORY = "ksei-komposisi"
PARSER_VERSION = "phase5.6-ksei-holding-composition-v1"
PROVIDER = "KSEI_MONTHLY_HOLDING_COMPOSITION"
REQUEST_TIMEOUT_SECONDS = 25
USER_AGENT = "Mozilla/5.0 (compatible; PASTICUAN-Phase5.6; official-public-data)"


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def _number(value: Any) -> float:
    text = str(value or "").strip().replace(",", "")
    if not text or text in {"-", "--"}:
        return np.nan
    try:
        number = float(text)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _candidate_urls(now: Any = None) -> list[tuple[str, str]]:
    current = pd.Timestamp.now(tz="Asia/Jakarta") if now is None else pd.Timestamp(now)
    if current.tzinfo is not None:
        current = current.tz_convert("Asia/Jakarta").tz_localize(None)
    first_this_month = current.normalize().replace(day=1)
    candidates: list[tuple[str, str]] = []
    for months_back in (1, 2, 3):
        anchor = first_this_month - pd.DateOffset(months=months_back)
        end = anchor.replace(day=monthrange(anchor.year, anchor.month)[1])
        for offset in range(0, 8):
            stamp = (end - pd.Timedelta(days=offset)).strftime("%Y%m%d")
            candidates.append((f"https://web.ksei.co.id/Download/BalanceposEfek{stamp}.zip", stamp))
    return candidates


def discover_latest_archive(session: requests.Session, *, now: Any = None) -> tuple[str, str]:
    try:
        response = session.get(ARCHIVE_PAGE, headers={"User-Agent": USER_AGENT}, timeout=12)
        if response.ok:
            matches = re.findall(
                r'href=["\']([^"\']*BalanceposEfek(\d{8})\.zip[^"\']*)["\']',
                response.text,
                flags=re.I,
            )
            if matches:
                href, stamp = sorted(matches, key=lambda item: item[1])[-1]
                return urljoin(ARCHIVE_PAGE, href), stamp
    except Exception:
        pass
    for url, stamp in _candidate_urls(now):
        try:
            response = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=10, stream=True)
            ok = response.status_code == 200
            response.close()
            if ok:
                return url, stamp
        except Exception:
            continue
    return "", ""


def _decode_zip(content: bytes) -> tuple[str, str]:
    with ZipFile(BytesIO(content)) as archive:
        names = [name for name in archive.namelist() if "balancepos" in name.lower()]
        if not names:
            names = [name for name in archive.namelist() if name.lower().endswith((".txt", ".csv"))]
        if not names:
            raise ValueError("KSEI_BALANCEPOS_FILE_MISSING")
        raw = archive.read(names[0])
        for encoding in ("utf-8-sig", "latin-1"):
            try:
                return raw.decode(encoding), names[0]
            except UnicodeDecodeError:
                continue
    raise ValueError("KSEI_BALANCEPOS_DECODE_FAILED")


def parse_archive(
    content: bytes,
    *,
    source_url: str,
    observed_on: date,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse one official archive into four compact factual rows per equity."""
    text, file_name = _decode_zip(content)
    reader = csv.reader(StringIO(text), delimiter="|")
    rows = list(reader)
    if len(rows) < 2 or len(rows[0]) < 25:
        raise ValueError("KSEI_BALANCEPOS_SCHEMA_INVALID")
    header = [str(value or "").strip().lower() for value in rows[0]]
    if "code" not in header[1] or "type" not in header[2]:
        raise ValueError("KSEI_BALANCEPOS_HEADER_INVALID")

    file_hash = sha256(content).hexdigest()
    fetched_at = datetime.now(timezone.utc).isoformat()
    snapshots: list[dict[str, Any]] = []
    for values in rows[1:]:
        if len(values) < 25 or str(values[2] or "").strip().upper() != "EQUITY":
            continue
        ticker = _ticker(values[1])
        if not re.fullmatch(r"[A-Z][A-Z0-9]{3,5}", ticker):
            continue
        issued = _number(values[3])
        local_total = _number(values[14])
        foreign_total = _number(values[24])
        if not np.isfinite(local_total):
            local_total = float(np.nansum([_number(value) for value in values[5:14]]))
        if not np.isfinite(foreign_total):
            foreign_total = float(np.nansum([_number(value) for value in values[15:24]]))
        scripless = local_total + foreign_total if np.isfinite(local_total) and np.isfinite(foreign_total) else np.nan
        local_pct = 100.0 * local_total / scripless if np.isfinite(scripless) and scripless > 0 else np.nan
        foreign_pct = 100.0 * foreign_total / scripless if np.isfinite(scripless) and scripless > 0 else np.nan
        scripless_pct = 100.0 * scripless / issued if np.isfinite(issued) and issued > 0 and np.isfinite(scripless) else np.nan
        facts = (
            ("KSEI_SECURITY_NUMBER", "KSEI security number / issued reference", issued, None),
            ("KSEI_SCRIPLESS_TOTAL", "KSEI scripless holdings", scripless, scripless_pct),
            ("KSEI_LOCAL_TOTAL", "KSEI local scripless holdings", local_total, local_pct),
            ("KSEI_FOREIGN_TOTAL", "KSEI foreign scripless holdings", foreign_total, foreign_pct),
        )
        for classification, label, shares, percentage in facts:
            if not np.isfinite(shares) and not np.isfinite(percentage if percentage is not None else np.nan):
                continue
            identity = sha256(f"{CATEGORY}|{ticker}|{classification}".encode("utf-8")).hexdigest()
            snapshots.append({
                "source_file_hash": file_hash,
                "category": CATEGORY,
                "ticker": ticker,
                "holder_identity_hash": identity,
                "holder_name": label,
                "report_date": observed_on.isoformat(),
                "publication_date": observed_on.isoformat(),
                "shares_held": float(shares) if np.isfinite(shares) else None,
                "ownership_percentage": float(percentage) if percentage is not None and np.isfinite(percentage) else None,
                "holder_classification": classification,
                "holder_type": "KSEI_REGISTRATION_COMPOSITION",
                "local_foreign_state": "LOCAL" if classification == "KSEI_LOCAL_TOTAL" else "FOREIGN" if classification == "KSEI_FOREIGN_TOTAL" else None,
                "source_url": source_url,
                "source_verified": True,
                "validation_state": "VALID",
                "fetched_at": fetched_at,
            })
    if not snapshots:
        raise ValueError("KSEI_BALANCEPOS_NO_EQUITY_ROWS")
    file_row = {
        "source_file_hash": file_hash,
        "category": CATEGORY,
        "publication_date": observed_on.isoformat(),
        "report_date": observed_on.isoformat(),
        "source_url": source_url,
        "file_name": file_name,
        "source_verified": True,
        "parser_version": PARSER_VERSION,
        "fetched_at": fetched_at,
        "validation_state": "VALID",
    }
    return file_row, snapshots


def canonical_context(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build latest official KSEI context without inferring regulatory free float."""
    by_ticker: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        if str(row.get("category") or "") != CATEGORY or not bool(row.get("source_verified")):
            continue
        ticker = _ticker(row.get("ticker"))
        classification = str(row.get("holder_classification") or "")
        report_date = str(row.get("report_date") or "")
        if not ticker or classification not in {
            "KSEI_SECURITY_NUMBER", "KSEI_SCRIPLESS_TOTAL", "KSEI_LOCAL_TOTAL", "KSEI_FOREIGN_TOTAL"
        }:
            continue
        current = by_ticker.setdefault(ticker, {"report_date": report_date, "facts": {}})
        if report_date > str(current.get("report_date") or ""):
            current["report_date"] = report_date
            current["facts"] = {}
        if report_date == current["report_date"]:
            current["facts"][classification] = row

    output: dict[str, dict[str, Any]] = {}
    for ticker, bundle in by_ticker.items():
        facts = bundle["facts"]
        issued = facts.get("KSEI_SECURITY_NUMBER", {}).get("shares_held")
        scripless_pct = facts.get("KSEI_SCRIPLESS_TOTAL", {}).get("ownership_percentage")
        local_pct = facts.get("KSEI_LOCAL_TOTAL", {}).get("ownership_percentage")
        foreign_pct = facts.get("KSEI_FOREIGN_TOTAL", {}).get("ownership_percentage")
        values = [issued, scripless_pct, local_pct, foreign_pct]
        present = sum(value is not None for value in values)
        output[ticker] = {
            "ownership_ksei_total_shares": issued,
            "ownership_ksei_scripless_pct": scripless_pct,
            "ownership_ksei_local_pct": local_pct,
            "ownership_ksei_foreign_pct": foreign_pct,
            "ownership_ksei_context_coverage_pct": 25.0 * present,
            "ownership_ksei_observed_on": bundle["report_date"],
            "ownership_ksei_source_authority": "OFFICIAL_KSEI",
            "ownership_ksei_official_verified": True,
            "ownership_ksei_provenance_state": "KSEI_REGISTRATION_COMPOSITION_NOT_REGULATORY_FREE_FLOAT",
            "ownership_ksei_context_state": "CONTEXT_ONLY_NOT_BENEFICIAL_OWNERSHIP_OR_FREE_FLOAT",
        }
    return output


class SharedKseiHoldingComposition:
    def __init__(self, client_id: str = "PASTICUAN", *, backend: Any | None = None, session: requests.Session | None = None):
        self.config = HubConfig.from_environment(client_id=client_id)
        self.backend = backend or (SupabaseEvidenceBackend(self.config) if self.config.ready else None)
        self.session = session or requests.Session()

    @property
    def ready(self) -> bool:
        return self.backend is not None

    def refresh(self, *, now: Any = None) -> dict[str, Any]:
        if not self.ready:
            return {"state": "ENVIRONMENT_BLOCKED", "rows": 0}
        url, stamp = discover_latest_archive(self.session, now=now)
        if not url or not stamp:
            return {"state": "OFFICIAL_ARCHIVE_NOT_FOUND", "rows": 0}
        observed_on = datetime.strptime(stamp, "%Y%m%d").date()
        response = self.session.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        file_row, snapshots = parse_archive(response.content, source_url=url, observed_on=observed_on)
        self.backend.upsert_rows("evidence_ownership_files", [file_row], conflict=("source_file_hash", "category"))
        persisted = 0
        for start in range(0, len(snapshots), 400):
            written = self.backend.upsert_rows(
                "evidence_ownership_snapshots",
                snapshots[start:start + 400],
                conflict=("source_file_hash", "ticker", "holder_identity_hash"),
            )
            persisted += len(written)
        return {
            "state": "REFRESHED" if persisted == len(snapshots) else "PARTIAL",
            "source_url": url,
            "observed_on": observed_on.isoformat(),
            "snapshot_rows": len(snapshots),
            "persisted_rows": persisted,
            "equity_tickers": len({row["ticker"] for row in snapshots}),
            "semantics": "OFFICIAL_KSEI_REGISTRATION_COMPOSITION_NOT_REGULATORY_FREE_FLOAT",
        }


__all__ = [
    "ARCHIVE_PAGE", "CATEGORY", "PARSER_VERSION", "PROVIDER",
    "SharedKseiHoldingComposition", "canonical_context", "discover_latest_archive", "parse_archive",
]
