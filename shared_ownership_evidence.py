from __future__ import annotations

"""Scanner-neutral ownership-file acquisition and factual snapshot normalization.

Rows describe reported holders or published classifications only.  They never
assert beneficial ownership, broker identity, coordinated control, or "bandar".
"""

from datetime import date, datetime, timezone
from io import BytesIO
import hashlib
import math
import os
import re
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlparse

import pandas as pd
import requests

from shared_evidence_hub import (
    EvidenceKey,
    HubConfig,
    MissingReason,
    SharedEvidenceCoordinator,
    SupabaseEvidenceBackend,
)


OWNERSHIP_INDEX_URL = "https://api.zpi.web.id/v1/finance:idx/ownership-files"
CATEGORIES = frozenset({"lima-persen", "satu-persen", "klasifikasi", "tipe"})
PARSER_VERSION = "phase5.6-ownership-v2"
MAX_FILE_BYTES = 50 * 1024 * 1024
INDEX_PAGE_SIZE = 200
MAX_INDEX_PAGES = 3
MAX_FILES_PER_PUBLICATION = 1
REQUEST_TIMEOUT_SECONDS = 30


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _secret(name: str) -> str:
    value = _clean(os.getenv(name, ""))
    if value:
        return value
    try:
        import streamlit as st

        return _clean(st.secrets.get(name, ""))
    except Exception:
        return ""


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", _clean(value).lower()).strip()


def _ticker(value: Any) -> str:
    text = _clean(value).upper().removesuffix(".JK")
    match = re.search(r"(?:^|\b)([A-Z][A-Z0-9]{3,5})(?:\b|$)", text)
    return match.group(1) if match else ""


def _date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _clean(value)
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed.date()


def _number(value: Any, *, percentage: bool = False) -> float | int | None:
    if isinstance(value, bool) or not _clean(value):
        return None
    text = _clean(value).replace("\u00a0", "").replace(" ", "")
    percent_mark = "%" in text
    text = text.replace("%", "")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        tail = text.rsplit(",", 1)[-1]
        text = text.replace(",", ".") if percentage or len(tail) != 3 else text.replace(",", "")
    elif text.count(".") > 1:
        text = text.replace(".", "")
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    # Percent-marked text is already in percentage points ("5,25%" -> 5.25).
    if percent_mark:
        number = number
    return int(number) if number.is_integer() else number


def _official_url(url: str) -> bool:
    parsed = urlparse(_clean(url))
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and any(
        host == domain or host.endswith(f".{domain}") for domain in ("idx.co.id", "ksei.co.id")
    )


def _field(header: Any) -> str:
    name = _key(header)
    aliases = {
        "ticker": ("kode emiten", "kode saham", "stock code", "ticker", "security code"),
        "holder_name": ("nama pemegang saham", "nama investor", "nama pihak", "shareholder name", "holder name"),
        "shares_held": ("jumlah saham", "total saham", "shares held", "number of shares"),
        "ownership_percentage": ("persentase kepemilikan", "% kepemilikan", "ownership percentage", "percentage"),
        "holder_classification": ("klasifikasi investor", "klasifikasi pemegang", "investor classification", "classification"),
        "holder_type": ("tipe investor", "tipe pemegang", "jenis investor", "investor type", "holder type"),
        "local_foreign_state": ("lokal asing", "domestik asing", "local foreign", "domestic foreign"),
        "report_date": ("tanggal posisi", "tanggal laporan", "report date", "position date"),
    }
    for field, candidates in aliases.items():
        if any(candidate in name for candidate in candidates):
            return field
    return ""


def _header_row(frame: pd.DataFrame) -> tuple[int, dict[int, str]] | None:
    for row_number in range(min(40, len(frame))):
        fields = {column: _field(value) for column, value in frame.iloc[row_number].items()}
        fields = {column: field for column, field in fields.items() if field}
        present = set(fields.values())
        if "ticker" in present and present.intersection(
            {"holder_name", "holder_classification", "holder_type"}
        ):
            return row_number, fields
    return None


def parse_ownership_workbook(
    sheets: Mapping[str, pd.DataFrame],
    *,
    category: str,
    publication_date: date,
    source_url: str,
    source_file_hash: str,
) -> list[dict[str, Any]]:
    if category not in CATEGORIES:
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    if not _official_url(source_url):
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    stamp = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for frame in sheets.values():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        header = _header_row(frame)
        if header is None:
            continue
        row_number, columns = header
        for _, raw in frame.iloc[row_number + 1 :].iterrows():
            values = {field: raw.get(column) for column, field in columns.items()}
            ticker = _ticker(values.get("ticker"))
            holder = _clean(values.get("holder_name"))
            classification = _clean(values.get("holder_classification"))
            holder_type = _clean(values.get("holder_type"))
            local_foreign = _clean(values.get("local_foreign_state")).upper()
            if not ticker:
                continue
            if category in {"lima-persen", "satu-persen"} and not holder:
                continue
            if category == "klasifikasi" and not classification:
                continue
            if category == "tipe" and not holder_type:
                continue
            identity_label = holder or classification or holder_type
            shares = _number(values.get("shares_held"))
            percentage = _number(values.get("ownership_percentage"), percentage=True)
            if shares is None and percentage is None:
                continue
            report_date = _date(values.get("report_date"))
            if report_date is None or report_date > publication_date:
                raise RuntimeError(MissingReason.WRONG_PERIOD.value)
            identity_hash = hashlib.sha256(
                "|".join(
                    _key(value) for value in (category, ticker, identity_label)
                ).encode("utf-8")
            ).hexdigest()
            identity = (ticker, identity_hash)
            if identity in seen:
                raise RuntimeError(MissingReason.PARSE_FAILURE.value)
            seen.add(identity)
            rows.append({
                "source_file_hash": source_file_hash,
                "category": category,
                "ticker": ticker,
                "holder_identity_hash": identity_hash,
                "holder_name": holder or None,
                "report_date": report_date.isoformat(),
                "publication_date": publication_date.isoformat(),
                "shares_held": shares,
                "ownership_percentage": percentage,
                "holder_classification": classification or None,
                "holder_type": holder_type or None,
                "local_foreign_state": local_foreign or None,
                "source_url": source_url,
                "source_verified": True,
                "validation_state": "VALID",
                "fetched_at": stamp,
            })
    if not rows:
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    report_dates = {_date(row.get("report_date")) for row in rows}
    if None in report_dates or len(report_dates) != 1:
        raise RuntimeError(MissingReason.WRONG_PERIOD.value)
    return rows


def validate_ownership_rows(rows: Iterable[Mapping[str, Any]], *, category: str) -> tuple[bool, str]:
    records = [dict(row) for row in rows]
    if not records:
        return False, MissingReason.EMPTY_RESPONSE.value
    identities: set[tuple[str, str]] = set()
    report_dates: set[date] = set()
    publication_dates: set[date] = set()
    for row in records:
        identity = (_ticker(row.get("ticker")), _clean(row.get("holder_identity_hash")))
        if not all(identity) or identity in identities or row.get("category") != category:
            return False, MissingReason.PARSE_FAILURE.value
        identities.add(identity)
        shares = row.get("shares_held")
        percentage = row.get("ownership_percentage")
        if shares is not None and (not isinstance(shares, (int, float)) or shares < 0):
            return False, MissingReason.CONTEXT_REJECTED.value
        if percentage is not None and (
            not isinstance(percentage, (int, float)) or not 0 <= percentage <= 100
        ):
            return False, MissingReason.CONTEXT_REJECTED.value
        report_date = _date(row.get("report_date"))
        publication_date = _date(row.get("publication_date"))
        if report_date is None or publication_date is None or report_date > publication_date:
            return False, MissingReason.WRONG_PERIOD.value
        report_dates.add(report_date)
        publication_dates.add(publication_date)
        if not row.get("source_verified") or not _official_url(_clean(row.get("source_url"))):
            return False, MissingReason.CONTEXT_REJECTED.value
    if len(report_dates) != 1 or len(publication_dates) != 1:
        return False, MissingReason.WRONG_PERIOD.value
    return True, "VALID"


def derive_ownership_changes(
    previous: Iterable[Mapping[str, Any]], current: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    prior = [dict(row) for row in previous]
    latest = [dict(row) for row in current]
    if not prior or not latest:
        return []
    prior_dates = {_date(row.get("report_date")) for row in prior}
    latest_dates = {_date(row.get("report_date")) for row in latest}
    categories = {_clean(row.get("category")) for row in prior + latest}
    if len(prior_dates) != 1 or len(latest_dates) != 1 or len(categories) != 1:
        return []
    previous_date = next(iter(prior_dates))
    current_date = next(iter(latest_dates))
    if previous_date is None or current_date is None or current_date <= previous_date:
        return []
    category = next(iter(categories))
    old = {(_ticker(row.get("ticker")), _clean(row.get("holder_identity_hash"))): row for row in prior}
    new = {(_ticker(row.get("ticker")), _clean(row.get("holder_identity_hash"))): row for row in latest}
    changes: list[dict[str, Any]] = []
    for identity in sorted(set(old) | set(new)):
        before, after = old.get(identity), new.get(identity)
        before_shares = before.get("shares_held") if before else None
        after_shares = after.get("shares_held") if after else None
        before_pct = before.get("ownership_percentage") if before else None
        after_pct = after.get("ownership_percentage") if after else None
        delta_shares = after_shares - before_shares if before_shares is not None and after_shares is not None else None
        delta_pct = after_pct - before_pct if before_pct is not None and after_pct is not None else None
        if before is None:
            state = {"satu-persen": "NEW_1PCT_HOLDER", "lima-persen": "NEW_5PCT_HOLDER"}.get(category)
        elif after is None:
            state = "EXITED_REPORTED_HOLDER"
        elif (delta_shares is not None and delta_shares > 0) or (delta_pct is not None and delta_pct > 0):
            state = "INCREASED_REPORTED_HOLDING"
        elif (delta_shares is not None and delta_shares < 0) or (delta_pct is not None and delta_pct < 0):
            state = "REDUCED_REPORTED_HOLDING"
        else:
            state = None
        if state is None:
            continue
        source = after or before or {}
        changes.append({
            "source_file_hash": _clean(latest[0].get("source_file_hash")),
            "previous_source_file_hash": _clean(prior[0].get("source_file_hash")),
            "category": category,
            "ticker": identity[0],
            "holder_identity_hash": identity[1],
            "previous_report_date": previous_date.isoformat(),
            "current_report_date": current_date.isoformat(),
            "previous_shares": before_shares,
            "current_shares": after_shares,
            "delta_shares": delta_shares,
            "previous_percentage": before_pct,
            "current_percentage": after_pct,
            "delta_percentage": delta_pct,
            "change_state": state,
            "source_verified": bool(source.get("source_verified")),
            "validation_state": "VALID",
            "derived_at": datetime.now(timezone.utc).isoformat(),
        })
    return changes


class SharedOwnershipEvidence:
    def __init__(
        self,
        client_id: str,
        *,
        backend: Any | None = None,
        coordinator: SharedEvidenceCoordinator | None = None,
        session: Any | None = None,
        api_key: str | None = None,
        workbook_reader: Callable[[bytes], Mapping[str, pd.DataFrame]] | None = None,
    ):
        self.client_id = _clean(client_id).upper() or "UNKNOWN"
        self.config = HubConfig.from_environment(client_id=self.client_id)
        self.backend = backend or (SupabaseEvidenceBackend(self.config) if self.config.ready else None)
        self.coordinator = coordinator or (
            SharedEvidenceCoordinator(self.backend, client_id=self.client_id) if self.backend is not None else None
        )
        self.session = session or requests.Session()
        self.api_key = _secret("ZAPI_KEY") if api_key is None else _clean(api_key)
        self.workbook_reader = workbook_reader or self._read_workbook

    @staticmethod
    def _read_workbook(content: bytes) -> Mapping[str, pd.DataFrame]:
        return pd.read_excel(BytesIO(content), sheet_name=None, header=None, dtype=object)

    @property
    def ready(self) -> bool:
        return self.backend is not None and self.coordinator is not None

    def _request(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        api: bool,
        stage: str,
    ) -> Any:
        if api and not self.api_key:
            raise RuntimeError(MissingReason.ENVIRONMENT_BLOCKED.value)
        stage = _clean(stage).upper() or ("ZAPI_INDEX" if api else "OFFICIAL_FILE")
        headers = {
            "Accept": "application/json" if api else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "User-Agent": (
                "Shared-IDX-Evidence-Hub/ownership-index"
                if api else "Shared-IDX-Evidence-Hub/ownership-file"
            ),
        }
        if api:
            headers["x-api-key"] = self.api_key
        try:
            response = self.session.request(
                "GET",
                url,
                params=dict(params or {}),
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
                allow_redirects=False,
            )
        except requests.Timeout as exc:
            raise RuntimeError(f"{stage}_TIMEOUT") from exc
        except requests.ConnectionError as exc:
            raise RuntimeError(f"{stage}_CONNECTION_ERROR") from exc
        status = int(getattr(response, "status_code", 0) or 0)
        if 300 <= status < 400:
            raise RuntimeError(f"{stage}_REDIRECT_BLOCKED")
        if status in {401, 403, 404, 429}:
            raise RuntimeError(f"{stage}_HTTP_{status}")
        if not 200 <= status < 300:
            raise RuntimeError(f"{stage}_HTTP_{status}")
        if not getattr(response, "content", b""):
            raise RuntimeError(MissingReason.EMPTY_RESPONSE.value)
        return response

    @staticmethod
    def _index_page(
        payload: Any, *, category: str, publication_date: date
    ) -> tuple[list[dict[str, Any]], int]:
        root = payload
        for _ in range(3):
            if isinstance(root, Mapping) and isinstance(root.get("data"), Mapping):
                root = root["data"]
            else:
                break
        values = root.get("data") if isinstance(root, Mapping) else None
        if not isinstance(values, list):
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        matches: list[dict[str, Any]] = []
        for item in values:
            if not isinstance(item, Mapping):
                continue
            item_category = _clean(item.get("category")).lower()
            item_date = _date(item.get("publishedAt"))
            url = _clean(item.get("url"))
            if item_category == category and item_date == publication_date and _official_url(url):
                matches.append({
                    "category": category,
                    "publication_date": publication_date,
                    "source_url": url,
                    "file_name": _clean(item.get("fileName")) or url.rsplit("/", 1)[-1],
                })
        try:
            total = max(len(values), int(root.get("total") or len(values)))
        except (TypeError, ValueError):
            total = len(values)
        return matches, total

    def get_publication(
        self, category: str, publication_date: date
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        category = _clean(category).lower()
        if category not in CATEGORIES:
            return [], {"state": MissingReason.CONTEXT_REJECTED.value, "api_calls": 0, "file_calls": 0}
        if not self.ready:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "api_calls": 0, "file_calls": 0}
        meta: dict[str, Any] = {
            "api_calls": 0,
            "file_calls": 0,
            "files": 0,
            "index_page_cap": MAX_INDEX_PAGES,
            "file_cap": MAX_FILES_PER_PUBLICATION,
        }
        pending_files: list[dict[str, Any]] = []
        pending_changes: list[dict[str, Any]] = []

        def read_current() -> list[dict[str, Any]]:
            files = self.backend.read_rows(
                "evidence_ownership_files",
                {"category": category, "publication_date": publication_date.isoformat(), "validation_state": "VALID"},
                limit=200,
            )
            rows: list[dict[str, Any]] = []
            for file_row in files:
                rows.extend(self.backend.read_rows(
                    "evidence_ownership_snapshots",
                    {"source_file_hash": file_row["source_file_hash"], "category": category, "validation_state": "VALID"},
                    limit=50000,
                ))
            return rows

        def fetch() -> list[dict[str, Any]]:
            if not self.api_key:
                raise RuntimeError(MissingReason.ENVIRONMENT_BLOCKED.value)
            entries: list[dict[str, Any]] = []
            for page in range(MAX_INDEX_PAGES):
                start = page * INDEX_PAGE_SIZE
                meta["api_calls"] += 1
                index_response = self._request(
                    OWNERSHIP_INDEX_URL,
                    params={"category": category, "length": INDEX_PAGE_SIZE, "start": start},
                    api=True,
                    stage="ZAPI_INDEX",
                )
                try:
                    page_entries, total = self._index_page(
                        index_response.json(), category=category, publication_date=publication_date
                    )
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(MissingReason.PARSE_FAILURE.value) from exc
                entries.extend(page_entries)
                if entries or start + INDEX_PAGE_SIZE >= total:
                    break
            if not entries:
                raise RuntimeError(MissingReason.NO_FILE.value)
            if len(entries) > MAX_FILES_PER_PUBLICATION:
                raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)

            all_rows: list[dict[str, Any]] = []
            existing = self.backend.read_rows("evidence_ownership_snapshots", {"category": category}, limit=50000)
            for entry in entries:
                if meta["file_calls"] >= MAX_FILES_PER_PUBLICATION:
                    raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
                meta["file_calls"] += 1
                response = self._request(
                    entry["source_url"], api=False, stage="OFFICIAL_FILE"
                )
                final_url = _clean(getattr(response, "url", entry["source_url"]))
                content = bytes(response.content)
                content_type = _clean((getattr(response, "headers", {}) or {}).get("Content-Type")).lower()
                if not _official_url(final_url) or len(content) > MAX_FILE_BYTES or not content.startswith(b"PK"):
                    raise RuntimeError(MissingReason.INVALID_CONTENT_TYPE.value)
                if content_type and not any(token in content_type for token in ("spreadsheet", "excel", "octet-stream")):
                    raise RuntimeError(MissingReason.INVALID_CONTENT_TYPE.value)
                source_hash = hashlib.sha256(content).hexdigest()
                try:
                    sheets = self.workbook_reader(content)
                except Exception as exc:
                    raise RuntimeError(MissingReason.PARSE_FAILURE.value) from exc
                rows = parse_ownership_workbook(
                    sheets,
                    category=category,
                    publication_date=publication_date,
                    source_url=final_url,
                    source_file_hash=source_hash,
                )
                report_dates = {_date(row.get("report_date")) for row in rows}
                if None in report_dates or len(report_dates) != 1:
                    raise RuntimeError(MissingReason.WRONG_PERIOD.value)
                current_report_date = next(iter(report_dates))
                if current_report_date > publication_date:
                    raise RuntimeError(MissingReason.WRONG_PERIOD.value)
                previous_dates = sorted({
                    value for value in (_date(row.get("report_date")) for row in existing)
                    if value is not None and value < current_report_date
                })
                previous = [
                    row for row in existing
                    if previous_dates and _date(row.get("report_date")) == previous_dates[-1]
                ]
                pending_files.append({
                    "source_file_hash": source_hash,
                    "category": category,
                    "publication_date": publication_date.isoformat(),
                    "report_date": current_report_date.isoformat(),
                    "source_url": final_url,
                    "file_name": entry["file_name"],
                    "source_verified": True,
                    "parser_version": PARSER_VERSION,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "validation_state": "VALID",
                })
                all_rows.extend(rows)
                pending_changes.extend(derive_ownership_changes(previous, rows))
            meta["files"] = len(pending_files)
            return all_rows

        def persist(rows: list[Mapping[str, Any]]) -> int:
            files_written = self.backend.upsert_rows(
                "evidence_ownership_files", pending_files, conflict=("source_file_hash", "category")
            )
            if len(files_written) != len(pending_files):
                raise RuntimeError(MissingReason.PERSIST_FAILURE.value)
            written = self.backend.upsert_rows(
                "evidence_ownership_snapshots", rows,
                conflict=("source_file_hash", "ticker", "holder_identity_hash"),
            )
            if pending_changes:
                self.backend.upsert_rows(
                    "evidence_ownership_changes", pending_changes,
                    conflict=("source_file_hash", "ticker", "holder_identity_hash", "change_state"),
                )
            return len(written)

        result = self.coordinator.get_or_refresh(
            EvidenceKey("ZAPI", "OWNERSHIP_FILES", category.upper(), publication_date),
            read_current=read_current,
            fetch=fetch,
            persist=persist,
            validate=lambda rows: validate_ownership_rows(rows, category=category),
            minimum_rows=1,
            lease_seconds=600,
        )
        rows = [dict(row) for row in result.rows]
        return rows, {
            "state": result.reason,
            "category": category,
            "publication_date": publication_date.isoformat(),
            "rows": len(rows),
            "cache_hit": result.cache_hit,
            "request_avoided": result.request_avoided,
            "lease_state": result.lease_state,
            **meta,
        }


__all__ = [
    "CATEGORIES",
    "OWNERSHIP_INDEX_URL",
    "PARSER_VERSION",
    "INDEX_PAGE_SIZE",
    "MAX_INDEX_PAGES",
    "MAX_FILES_PER_PUBLICATION",
    "REQUEST_TIMEOUT_SECONDS",
    "SharedOwnershipEvidence",
    "derive_ownership_changes",
    "parse_ownership_workbook",
    "validate_ownership_rows",
]
