from __future__ import annotations

"""Shared factual persistence for official IDX Trade Detail participant rows."""

from datetime import date, datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from idx_trade_detail_discovery import DiscoveryAttempt, TradeDetailDiscoveryError
from shared_evidence_hub import (
    EvidenceKey,
    HubConfig,
    MissingReason,
    SharedEvidenceCoordinator,
    SupabaseEvidenceBackend,
)


SOURCE = "IDX_PUBLIC_TRADE_DETAIL_PUBLIK"
PROVENANCE = "VERIFIED_IDX_PUBLIC_TRADE_DETAIL_PARTICIPANT_FLOW_NOT_BENEFICIAL_OWNER"


def _json_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


class SharedParticipantEvidence:
    def __init__(self, client_id: str):
        self.client_id = str(client_id).strip().upper()
        self.config = HubConfig.from_environment(client_id=self.client_id)
        self.backend = SupabaseEvidenceBackend(self.config) if self.config.ready else None
        self.coordinator = (
            SharedEvidenceCoordinator(self.backend, client_id=self.client_id)
            if self.backend is not None else None
        )

    @property
    def ready(self) -> bool:
        return self.backend is not None and self.coordinator is not None

    def status(self) -> str:
        return self.config.status()["state"]

    @staticmethod
    def _to_rows(
        frame: pd.DataFrame,
        trade_date: date,
        source_url: str,
        source_file_hash: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        fetched_at = datetime.now(timezone.utc).isoformat()
        for _, row in frame.iterrows():
            ticker = str(row.get("ticker") or "").strip().upper().removesuffix(".JK")
            broker = str(row.get("broker_code") or "").strip().upper()
            if not ticker or not broker:
                continue
            rows.append({
                "source": SOURCE,
                "trade_date": trade_date.isoformat(),
                "ticker": ticker,
                "broker_code": broker,
                "buy_value": _json_number(row.get("buy_value")) or 0,
                "sell_value": _json_number(row.get("sell_value")) or 0,
                "buy_volume": _json_number(row.get("buy_volume")) or 0,
                "sell_volume": _json_number(row.get("sell_volume")) or 0,
                "net_value": _json_number(row.get("net_value")) or 0,
                "net_volume": _json_number(row.get("net_volume")) or 0,
                "buy_avg": _json_number(row.get("buy_avg")),
                "sell_avg": _json_number(row.get("sell_avg")),
                "source_url": source_url,
                "source_file_hash": source_file_hash,
                "source_verified": True,
                "provenance_state": PROVENANCE,
                "fetched_at": fetched_at,
                "validation_state": "VALID",
            })
        return rows

    @staticmethod
    def _to_frame(rows: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
        frame["gross_value"] = pd.to_numeric(frame["buy_value"], errors="coerce").fillna(0) + pd.to_numeric(frame["sell_value"], errors="coerce").fillna(0)
        return frame

    def get_day(
        self,
        trade_date: date,
        *,
        download: Callable[..., tuple[Any, str]],
        aggregate: Callable[..., pd.DataFrame],
        minimum_ticker_breadth: int = 100,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if not self.ready:
            return pd.DataFrame(), {"state": "HUB_MISSING", "diagnostics": []}
        diagnostics: list[DiscoveryAttempt] = []
        fetch_meta: dict[str, Any] = {}

        def read_current():
            return self.backend.read_rows(
                "evidence_participant_flow",
                {"source": SOURCE, "trade_date": trade_date.isoformat()},
                limit=50000,
            )

        def fetch():
            path: Path | None = None
            try:
                raw_path, source_url = download(trade_date, diagnostics=diagnostics)
                path = Path(raw_path)
                source_file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                frame = aggregate(raw_path, trade_date)
                fetch_meta.update({
                    "source_url": source_url,
                    "source_file_hash": source_file_hash,
                    "ticker_breadth": int(frame["ticker"].nunique()) if not frame.empty else 0,
                })
                return self._to_rows(frame, trade_date, source_url, source_file_hash)
            except TradeDetailDiscoveryError as exc:
                reason = exc.attempts[-1].result_state if exc.attempts else MissingReason.NO_FILE.value
                raise RuntimeError(reason) from exc
            except RuntimeError as exc:
                if "PARSE" in str(exc).upper():
                    raise RuntimeError(MissingReason.PARSE_FAILURE.value) from exc
                raise
            finally:
                if path is not None:
                    path.unlink(missing_ok=True)

        def persist(rows):
            written = self.backend.upsert_rows(
                "evidence_participant_flow",
                rows,
                conflict=("source", "trade_date", "ticker", "broker_code"),
            )
            return len(written)

        def validate(rows):
            breadth = len({str(row.get("ticker") or "") for row in rows})
            return (
                breadth >= max(1, int(minimum_ticker_breadth)),
                "VALID" if breadth >= max(1, int(minimum_ticker_breadth)) else MissingReason.INSUFFICIENT_HISTORY.value,
            )

        result = self.coordinator.get_or_refresh(
            EvidenceKey("IDX", "TRADE_DETAIL", "IDX_ALL", trade_date),
            read_current=read_current,
            fetch=fetch,
            persist=persist,
            validate=validate,
            minimum_rows=max(1, int(minimum_ticker_breadth)),
            lease_seconds=600,
        )
        meta = {
            "state": result.reason,
            "provider_called": result.provider_called,
            "request_avoided": result.request_avoided,
            "lease_state": result.lease_state,
            "rows": len(result.rows),
            "ticker_breadth": len({str(row.get("ticker") or "") for row in result.rows}),
            "diagnostics": [attempt.safe_dict() for attempt in diagnostics],
            **fetch_meta,
        }
        return self._to_frame(result.rows), meta

    def metrics(self) -> dict[str, int]:
        return self.coordinator.metrics() if self.coordinator is not None else {}


__all__ = ["PROVENANCE", "SOURCE", "SharedParticipantEvidence"]
