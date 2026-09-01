from __future__ import annotations

"""Shared ZAPI discovery -> official IDX XBRL factual evidence producer.

This module ports the verified Pasticuan issuer/context/period protections into
a scanner-neutral boundary.  It persists reports and raw canonical facts only;
it does not calculate scanner scores, growth signals, ranks, or recommendations.
"""

from datetime import date, datetime, timezone
from io import BytesIO
import hashlib
import math
import os
import re
from typing import Any, Iterable, Mapping
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET
import zipfile

import requests

from shared_evidence_hub import (
    EvidenceKey,
    HubConfig,
    MissingReason,
    SharedEvidenceCoordinator,
    SupabaseEvidenceBackend,
)


ZAPI_FINANCIAL_REPORT_URL = "https://api.zpi.web.id/v1/finance:idx/financial-report"
IDX_ATTACHMENT_BASE = "https://www.idx.co.id/"
PARSER_VERSION = "phase5.6-idx-xbrl-v1"
MAX_ATTACHMENT_BYTES = 25_000_000
MAX_UNCOMPRESSED_BYTES = 60_000_000
PERIODS = {"tw1": ("Q1", 3, 31), "tw2": ("Q2", 6, 30), "tw3": ("Q3", 9, 30), "audit": ("FY", 12, 31)}

FACT_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue", "revenues", "totalrevenue", "sales", "netsales", "pendapatan", "pendapatanusaha"),
    "net_income": ("netincome", "profitloss", "profitlossattributabletoownersofparententity", "lababersih"),
    "assets": ("assets", "totalassets", "totalaset", "aset"),
    "liabilities": ("liabilities", "totalliabilities", "totalliabilitas", "liabilitas"),
    "equity": ("equity", "totalequity", "stockholdersequity", "shareholdersequity", "ekuitas"),
    "cash": ("cash", "cashandcashequivalents", "cashandbank", "kasdansetarakas"),
    "short_term_debt": ("shorttermdebt", "currentshorttermborrowings", "currentborrowings", "utangjangkapendek"),
    "long_term_debt": ("longtermdebt", "noncurrentborrowings", "longtermborrowings", "utangjangkapanjang"),
    "total_debt": ("totaldebt", "totalborrowings", "borrowings", "interestbearingdebt", "utangberbunga"),
    "operating_cash_flow": (
        "operatingcashflow", "cashflowfromoperations", "cashflowsfromusedinoperatingactivities",
        "netcashflowsfromusedinoperatingactivities", "netcashflowsreceivedfromusedinoperatingactivities",
        "netcashflowsprovidedbyusedinoperatingactivities", "ocf", "aruskasoperasi",
    ),
    "capex": (
        "capex", "capitalexpenditure", "capitalexpenditures", "paymentstoacquirepropertyplantandequipment",
        "paymentsforacquisitionofpropertyplantandequipment", "paymentsforacquisitionofpropertyandequipment",
        "paymentsforadvancesforpurchaseofpropertyplantandequipment", "paymentsforadvancesforpurchaseofpropertyandequipment",
        "paymentsforacquisitionofintangibleassets", "paymentsforacquisitionofoilandgasassets",
        "paymentsforacquisitionofminingproperties", "paymentsforacquisitionofinvestmentproperties",
        "paymentsforacquisitionofexplorationandevaluationassets", "paymentsforacquisitionofothernonfinancialassets",
        "belanjamodal",
    ),
    "free_cash_flow": ("freecashflow", "freecashflows", "aruskasbebas"),
}

DURATION_FACTS = frozenset({"revenue", "net_income", "operating_cash_flow", "capex", "free_cash_flow"})


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _secret(name: str) -> str:
    value = _clean(os.getenv(name, ""))
    if value:
        return value
    try:
        import streamlit as st

        return _clean(st.secrets.get(name, ""))
    except Exception:
        return ""


def _ticker(value: Any) -> str:
    return _clean(value).upper().removesuffix(".JK")


def _local_name(value: Any) -> str:
    text = _clean(value)
    if "}" in text:
        text = text.rsplit("}", 1)[-1]
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    return text


def _attribute(element: Any, name: str) -> str:
    wanted = name.lower()
    for key, value in getattr(element, "attrib", {}).items():
        if _local_name(key).lower() == wanted:
            return _clean(value)
    return ""


def _concept_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _clean(value).lower())


def _concept_score(concept: str, aliases: Iterable[str]) -> int:
    key = _concept_key(concept)
    best = 0
    for alias in aliases:
        candidate = _concept_key(alias)
        if key == candidate:
            best = max(best, 100)
        elif len(candidate) >= 10 and (key.endswith(candidate) or key.startswith(candidate)):
            best = max(best, 82)
    return best


def _number(value: Any) -> float | None:
    text = _clean(value)
    if not text or text.lower() in {"-", "—", "na", "n/a", "nan", "none"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace("Rp", "").replace("IDR", "").replace("\u00a0", "").replace(" ", "")
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".") if text.rfind(",") > text.rfind(".") else text.replace(",", "")
    elif "," in text:
        pieces = text.split(",")
        text = "".join(pieces) if all(len(piece) == 3 for piece in pieces[1:]) else text.replace(",", ".")
    elif text.count(".") > 1:
        text = text.replace(".", "")
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return -number if negative else number


def _fact_number(element: Any, *, inline: bool) -> float | None:
    if _attribute(element, "nil").lower() in {"true", "1"}:
        return None
    value = _number("".join(element.itertext()))
    if value is None:
        return None
    if inline:
        scale = _number(_attribute(element, "scale"))
        if scale is not None:
            value *= 10.0 ** int(scale)
        if _attribute(element, "sign") == "-":
            value = -abs(value)
    return value


def _official_idx_url(value: Any) -> bool:
    parsed = urlparse(_clean(value))
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and any(
        host == domain or host.endswith(f".{domain}") for domain in ("idx.co.id", "idx.id")
    )


def _attachment_url(value: Any) -> str:
    url = urljoin(IDX_ATTACHMENT_BASE, _clean(value))
    return url if _official_idx_url(url) else ""


def _parse_date(value: Any) -> date | None:
    text = _clean(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def period_contract(year: int, period_code: str) -> tuple[str, str, date]:
    code = _clean(period_code).lower()
    if code not in PERIODS:
        raise ValueError("unsupported financial-report period")
    period_type, month, day = PERIODS[code]
    target = date(int(year), month, day)
    return period_type, f"{int(year)}-{period_type}", target


def _documents(payload: bytes, filename: str) -> list[tuple[str, bytes]]:
    if not payload:
        raise RuntimeError(MissingReason.EMPTY_RESPONSE.value)
    if payload[:4] != b"PK\x03\x04":
        return [(filename or "filing.xbrl", payload)]
    documents: list[tuple[str, bytes]] = []
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            total = 0
            for member in archive.infolist():
                lower = member.filename.lower()
                if member.is_dir() or not lower.endswith((".xbrl", ".xml", ".xhtml", ".html", ".htm")):
                    continue
                if member.file_size < 0 or member.file_size > MAX_UNCOMPRESSED_BYTES:
                    continue
                total += member.file_size
                if total > MAX_UNCOMPRESSED_BYTES:
                    raise RuntimeError(MissingReason.INVALID_CONTENT_TYPE.value)
                documents.append((member.filename, archive.read(member)))
    except zipfile.BadZipFile as exc:
        raise RuntimeError(MissingReason.PARSE_FAILURE.value) from exc
    if not documents:
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    return documents


def parse_idx_xbrl_facts(
    payload: bytes,
    *,
    ticker: str,
    report_period: str,
    period_type: str,
    period_end: date,
    publication_date: date,
    source_url: str,
    filename: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse only requested-issuer facts from current-period IDX contexts."""

    code = _ticker(ticker)
    if not code or not _official_idx_url(source_url):
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    units: dict[tuple[str, str], str] = {}
    facts: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for member_name, content in _documents(bytes(payload), filename):
        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            parse_errors.append(member_name)
            continue
        for element in root.iter():
            local = _local_name(element.tag).lower()
            if local == "context":
                context_id = _attribute(element, "id")
                if not context_id:
                    continue
                start = end = instant = None
                dimensions = 0
                identifier = scheme = ""
                for child in element.iter():
                    child_name = _local_name(child.tag).lower()
                    if child_name == "startdate":
                        start = _parse_date(child.text)
                    elif child_name == "enddate":
                        end = _parse_date(child.text)
                    elif child_name == "instant":
                        instant = _parse_date(child.text)
                    elif child_name in {"explicitmember", "typedmember"}:
                        dimensions += 1
                    elif child_name == "identifier" and not identifier:
                        identifier = _clean(child.text)
                        scheme = _attribute(child, "scheme")
                issuer_code, separator, role = identifier.lower().rpartition("_")
                issuer_match = bool(
                    scheme.lower().rstrip("/") == "http://www.idx.co.id/xbrl"
                    and separator and issuer_code == code.lower() and role in {"maker", "approver"}
                )
                contexts[(member_name, context_id)] = {
                    "id": context_id, "start": start, "end": end, "instant": instant,
                    "dimensions": dimensions, "identifier": identifier, "scheme": scheme,
                    "issuer_match": issuer_match,
                }
            elif local == "unit":
                unit_id = _attribute(element, "id")
                measures = [_clean(child.text) for child in element.iter() if _local_name(child.tag).lower() == "measure"]
                if unit_id:
                    units[(member_name, unit_id)] = " ".join(value for value in measures if value)
        for element in root.iter():
            local = _local_name(element.tag).lower()
            inline = local in {"nonfraction", "fraction"}
            context_ref = _attribute(element, "contextref")
            if not context_ref:
                continue
            concept = _attribute(element, "name") if inline else _local_name(element.tag)
            value = _fact_number(element, inline=inline)
            if concept and value is not None:
                facts.append({
                    "concept": concept, "value": value,
                    "context": contexts.get((member_name, context_ref), {}),
                    "unit": units.get((member_name, _attribute(element, "unitref")), ""),
                })
    if not facts:
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    matched = [context for context in contexts.values() if context.get("issuer_match")]
    if not matched:
        if any(context.get("identifier") for context in contexts.values()):
            raise RuntimeError(MissingReason.ISSUER_MISMATCH.value)
        raise RuntimeError(MissingReason.ISSUER_IDENTITY_MISSING.value)

    selected: dict[str, tuple[float, float, str, str]] = {}
    capex_components: dict[str, tuple[float, float, str, str]] = {}
    for canonical, aliases in FACT_ALIASES.items():
        candidates: list[tuple[float, float, str, str]] = []
        for fact in facts:
            concept_score = _concept_score(fact["concept"], (canonical, *aliases))
            context = fact["context"]
            if concept_score <= 0 or not context.get("issuer_match"):
                continue
            context_end = context.get("instant") or context.get("end")
            if context_end is None or abs((period_end - context_end).days) > 7:
                continue
            score = float(concept_score + 100 - 10 * abs((period_end - context_end).days))
            context_id = _clean(context.get("id")).lower()
            score += 24 if int(context.get("dimensions", 0)) == 0 else 0
            score += 8 if "current" in context_id else 0
            score += 6 if "consolidat" in context_id else 0
            score -= 25 if any(token in context_id for token in ("prior", "previous", "comparative")) else 0
            if canonical in DURATION_FACTS:
                start = context.get("start")
                score += 30 if start and start.year == period_end.year and start.month == 1 and start.day <= 7 else -20
            elif context.get("instant"):
                score += 12
            candidates.append((score, float(fact["value"]), _clean(fact["unit"]), _clean(fact["concept"])))
        if canonical == "capex":
            for candidate in candidates:
                concept = _concept_key(candidate[3])
                if concept.startswith("paymentsforacquisitionof") or concept.startswith("paymentsforadvancesforpurchaseof"):
                    current = capex_components.get(concept)
                    if current is None or candidate[0] > current[0]:
                        capex_components[concept] = candidate
            if capex_components:
                first = max(capex_components.values(), key=lambda item: item[0])
                selected[canonical] = (first[0], sum(abs(item[1]) for item in capex_components.values()), first[2], "SUM_COMPONENTS")
                continue
        if candidates:
            selected[canonical] = max(candidates, key=lambda item: item[0])
    if len(selected) < 3:
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)

    issuer_identity = " | ".join(sorted({_clean(context["identifier"]) for context in matched if context.get("identifier")}))
    document_hash = hashlib.sha256(bytes(payload)).hexdigest()
    fetched_at = datetime.now(timezone.utc).isoformat()
    report = {
        "ticker": code,
        "report_period": report_period,
        "period_type": period_type,
        "report_date": period_end.isoformat(),
        "publication_date": publication_date.isoformat(),
        "source": "IDX_OFFICIAL_XBRL",
        "source_url": source_url,
        "issuer_identity": issuer_identity,
        "issuer_match": True,
        "context_state": "CURRENT_PERIOD_ISSUER_MATCHED",
        "parser_version": PARSER_VERSION,
        "source_document_hash": document_hash,
        "freshness_state": "CURRENT",
        "validation_state": "VALID",
        "fetched_at": fetched_at,
    }
    normalized: list[dict[str, Any]] = []
    for name, (_, value, unit, _) in sorted(selected.items()):
        upper_unit = unit.upper()
        currency = "USD" if "USD" in upper_unit and "IDR" not in upper_unit else "IDR" if "IDR" in upper_unit else None
        normalized.append({
            "ticker": code,
            "report_period": report_period,
            "fact_name": name,
            "fact_value": value,
            "currency": currency,
            "unit_scale": 1,
            "period_type": period_type,
            "report_date": period_end.isoformat(),
            "publication_date": publication_date.isoformat(),
            "source": "IDX_OFFICIAL_XBRL",
            "source_url": source_url,
            "issuer_identity": issuer_identity,
            "context_state": "CURRENT_PERIOD_ISSUER_MATCHED",
            "parser_version": PARSER_VERSION,
            "source_document_hash": document_hash,
            "freshness_state": "CURRENT",
            "validation_state": "VALID",
            "fetched_at": fetched_at,
        })
    return report, normalized


def validate_financial_facts(
    rows: Iterable[Mapping[str, Any]], *, ticker: str, report_period: str, minimum_facts: int = 3
) -> tuple[bool, str]:
    records = [dict(row) for row in rows]
    if len(records) < max(1, int(minimum_facts)):
        return False, MissingReason.INSUFFICIENT_HISTORY.value
    names: set[str] = set()
    for row in records:
        name = _clean(row.get("fact_name"))
        value = row.get("fact_value")
        if _ticker(row.get("ticker")) != _ticker(ticker) or row.get("report_period") != report_period:
            return False, MissingReason.WRONG_PERIOD.value
        if not name or name in names or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return False, MissingReason.PARSE_FAILURE.value
        if row.get("context_state") != "CURRENT_PERIOD_ISSUER_MATCHED" or row.get("source") != "IDX_OFFICIAL_XBRL":
            return False, MissingReason.CONTEXT_REJECTED.value
        if not _official_idx_url(row.get("source_url")):
            return False, MissingReason.CONTEXT_REJECTED.value
        names.add(name)
    return True, "VALID"


class SharedFinancialEvidence:
    def __init__(
        self,
        client_id: str,
        *,
        backend: Any | None = None,
        coordinator: SharedEvidenceCoordinator | None = None,
        session: Any | None = None,
        api_key: str | None = None,
    ):
        self.client_id = _clean(client_id).upper() or "UNKNOWN"
        self.config = HubConfig.from_environment(client_id=self.client_id)
        self.backend = backend or (SupabaseEvidenceBackend(self.config) if self.config.ready else None)
        self.coordinator = coordinator or (
            SharedEvidenceCoordinator(self.backend, client_id=self.client_id) if self.backend is not None else None
        )
        self.session = session or requests.Session()
        self.api_key = _secret("ZAPI_KEY") if api_key is None else _clean(api_key)

    @property
    def ready(self) -> bool:
        return self.backend is not None and self.coordinator is not None

    def _request(self, url: str, *, params: Mapping[str, Any] | None = None, api: bool) -> Any:
        if api and not self.api_key:
            raise RuntimeError(MissingReason.ENVIRONMENT_BLOCKED.value)
        headers = {"Accept": "application/json" if api else "application/zip,application/xml,text/xml,text/html,*/*"}
        if api:
            headers["x-api-key"] = self.api_key
        try:
            response = self.session.request("GET", url, params=dict(params or {}), headers=headers, timeout=30)
        except requests.Timeout as exc:
            raise RuntimeError(MissingReason.TIMEOUT.value) from exc
        except requests.ConnectionError as exc:
            raise RuntimeError(MissingReason.CONNECTION_ERROR.value) from exc
        status = int(getattr(response, "status_code", 0) or 0)
        if status in {401, 403, 404, 429}:
            raise RuntimeError(f"HTTP_{status}")
        if not 200 <= status < 300:
            raise RuntimeError(f"HTTP_{status}")
        if not getattr(response, "content", b""):
            raise RuntimeError(MissingReason.EMPTY_RESPONSE.value)
        return response

    @staticmethod
    def _manifest(payload: Any, *, ticker: str, year: int, period_code: str) -> dict[str, Any]:
        root = payload
        for _ in range(3):
            if isinstance(root, Mapping) and isinstance(root.get("data"), Mapping):
                root = root["data"]
            else:
                break
        rows = root.get("data") if isinstance(root, Mapping) else None
        if not isinstance(rows, list):
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        code = _ticker(ticker)
        expected_type, _, _ = period_contract(year, period_code)
        candidates: list[dict[str, Any]] = []
        issuer_seen = False
        matching_issuer_seen = False
        wrong_period_seen = False
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            row_code = _ticker(row.get("KodeEmiten") or row.get("kodeEmiten"))
            if row_code != code:
                if row_code:
                    issuer_seen = True
                continue
            matching_issuer_seen = True
            row_year = int(_number(row.get("Report_Year") or row.get("reportYear")) or 0)
            row_period = _clean(row.get("Report_Period") or row.get("reportPeriod")).upper()
            if row_year != int(year) or row_period not in {expected_type, period_code.upper(), "AUDIT" if expected_type == "FY" else expected_type}:
                wrong_period_seen = True
                continue
            for attachment in row.get("Attachments") or row.get("attachments") or []:
                if not isinstance(attachment, Mapping):
                    continue
                attachment_code = _ticker(attachment.get("Emiten_Code") or attachment.get("emitenCode") or code)
                if attachment_code != code:
                    raise RuntimeError(MissingReason.ISSUER_MISMATCH.value)
                filename = _clean(attachment.get("File_Name") or attachment.get("fileName"))
                url = _attachment_url(attachment.get("File_Path") or attachment.get("filePath"))
                lower = filename.lower()
                if not url or not lower.endswith((".zip", ".xbrl", ".xml", ".xhtml", ".html", ".htm")):
                    continue
                if not ("xbrl" in lower or "instance" in lower or lower.endswith((".xbrl", ".xml"))):
                    continue
                modified = _parse_date(attachment.get("File_Modified") or row.get("File_Modified"))
                if modified is None:
                    continue
                rank = 0 if "instance" in lower and lower.endswith(".zip") else 1 if "inlinexbrl" in lower else 2
                candidates.append({"url": url, "filename": filename, "publication_date": modified, "rank": rank})
        if not candidates:
            if issuer_seen and not matching_issuer_seen:
                raise RuntimeError(MissingReason.ISSUER_MISMATCH.value)
            if wrong_period_seen:
                raise RuntimeError(MissingReason.WRONG_PERIOD.value)
            raise RuntimeError(MissingReason.NO_REPORT.value)
        return min(candidates, key=lambda item: item["rank"])

    def get_period(
        self, ticker: str, year: int, period_code: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        code = _ticker(ticker)
        try:
            period_type, report_period, period_end = period_contract(year, period_code)
        except ValueError:
            return [], {"state": MissingReason.WRONG_PERIOD.value, "api_calls": 0, "attachment_calls": 0}
        if not self.ready or not code:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "api_calls": 0, "attachment_calls": 0}
        meta: dict[str, Any] = {"api_calls": 0, "attachment_calls": 0}
        pending_report: dict[str, Any] = {}

        def read_current() -> list[dict[str, Any]]:
            reports = self.backend.read_rows(
                "evidence_financial_reports",
                {"ticker": code, "report_period": report_period, "issuer_match": "true", "validation_state": "VALID"},
                limit=20,
            )
            rows: list[dict[str, Any]] = []
            for report in reports:
                rows.extend(self.backend.read_rows(
                    "evidence_financial_facts",
                    {"ticker": code, "report_period": report_period, "source_document_hash": report["source_document_hash"], "validation_state": "VALID"},
                    limit=100,
                ))
            return rows

        def fetch() -> list[dict[str, Any]]:
            if not self.api_key:
                raise RuntimeError(MissingReason.ENVIRONMENT_BLOCKED.value)
            meta["api_calls"] += 1
            response = self._request(
                ZAPI_FINANCIAL_REPORT_URL,
                params={"year": int(year), "period": period_code.lower(), "code": code, "length": 100, "start": 0},
                api=True,
            )
            try:
                manifest = self._manifest(response.json(), ticker=code, year=year, period_code=period_code)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(MissingReason.PARSE_FAILURE.value) from exc
            meta["attachment_calls"] += 1
            attachment = self._request(manifest["url"], api=False)
            final_url = _clean(getattr(attachment, "url", manifest["url"])) or manifest["url"]
            content = bytes(attachment.content)
            content_type = _clean((getattr(attachment, "headers", {}) or {}).get("Content-Type")).lower()
            if not _official_idx_url(final_url) or len(content) > MAX_ATTACHMENT_BYTES:
                raise RuntimeError(MissingReason.INVALID_CONTENT_TYPE.value)
            if content_type and not any(token in content_type for token in ("zip", "xml", "html", "octet-stream")):
                raise RuntimeError(MissingReason.INVALID_CONTENT_TYPE.value)
            report, facts = parse_idx_xbrl_facts(
                content,
                ticker=code,
                report_period=report_period,
                period_type=period_type,
                period_end=period_end,
                publication_date=manifest["publication_date"],
                source_url=final_url,
                filename=manifest["filename"],
            )
            pending_report.update(report)
            return facts

        def persist(rows: list[Mapping[str, Any]]) -> int:
            report_written = self.backend.upsert_rows(
                "evidence_financial_reports", [pending_report],
                conflict=("ticker", "report_period", "source_document_hash"),
            )
            if len(report_written) != 1:
                raise RuntimeError(MissingReason.PERSIST_FAILURE.value)
            written = self.backend.upsert_rows(
                "evidence_financial_facts", rows,
                conflict=("ticker", "report_period", "fact_name", "source_document_hash"),
            )
            return len(written)

        result = self.coordinator.get_or_refresh(
            EvidenceKey("IDX", "XBRL", code, period_end),
            read_current=read_current,
            fetch=fetch,
            persist=persist,
            validate=lambda rows: validate_financial_facts(rows, ticker=code, report_period=report_period),
            minimum_rows=3,
            lease_seconds=600,
        )
        rows = [dict(row) for row in result.rows]
        return rows, {
            "state": result.reason,
            "ticker": code,
            "report_period": report_period,
            "rows": len(rows),
            "cache_hit": result.cache_hit,
            "request_avoided": result.request_avoided,
            "lease_state": result.lease_state,
            **meta,
        }


__all__ = [
    "FACT_ALIASES",
    "PARSER_VERSION",
    "SharedFinancialEvidence",
    "ZAPI_FINANCIAL_REPORT_URL",
    "parse_idx_xbrl_facts",
    "period_contract",
    "validate_financial_facts",
]
