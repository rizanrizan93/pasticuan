from __future__ import annotations

"""Durable chunked scan orchestration.

The module deliberately keeps browser/session state out of the execution contract.
Supabase owns the job, per-ticker leases, retry counters, and completed payloads.
An in-process daemon worker is only an accelerator: if Streamlit is recycled, a
later session claims expired leases and continues from the last durable item.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Any, Callable, Mapping
import math
import os
import time
import uuid

import numpy as np
import pandas as pd

from scanner_database import ScannerDatabaseBridge

RESUMABLE_SCAN_VERSION = "9.4.4"
TERMINAL_JOB_STATES = {"COMPLETE", "COMPLETE_WITH_FAILURES", "FAILED", "CANCELLED"}
ACTIVE_JOB_STATES = {"PENDING", "RUNNING", "PAUSED", "FINALIZING"}
TERMINAL_ITEM_STATES = {"COMPLETE", "FAILED", "SKIPPED"}


def json_safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.isoformat()
    if isinstance(value, pd.Series):
        return {str(key): json_safe(item) for key, item in value.to_dict().items()}
    if isinstance(value, pd.DataFrame):
        return [{str(key): json_safe(item) for key, item in row.items()} for row in value.to_dict("records")]
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    try:
        if bool(pd.isna(value)):
            return None
    except Exception:
        pass
    return str(value)


def frame_from_records(records: Any) -> pd.DataFrame:
    if not isinstance(records, list):
        return pd.DataFrame()
    rows = [row for row in records if isinstance(row, Mapping)]
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ItemOutcome:
    success: bool
    payload: Mapping[str, Any] | None = None
    error: str = ""
    retry_delay_seconds: int = 15


@dataclass(frozen=True)
class WorkerStatus:
    job_id: str
    worker_id: str
    alive: bool
    started_at: float
    last_error: str = ""


class _WorkerHandle:
    def __init__(self, job_id: str, worker_id: str, target: Callable[[], None]) -> None:
        self.job_id = job_id
        self.worker_id = worker_id
        self.started_at = time.time()
        self.last_error = ""
        self.stop_event = Event()

        def wrapped() -> None:
            try:
                target()
            except Exception as exc:  # Final safety net; job loop handles normal failures.
                self.last_error = f"{type(exc).__name__}: {str(exc)[:800]}"

        self.thread = Thread(target=wrapped, name=f"idx-job-{job_id[:8]}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def status(self) -> WorkerStatus:
        return WorkerStatus(
            job_id=self.job_id,
            worker_id=self.worker_id,
            alive=self.thread.is_alive(),
            started_at=self.started_at,
            last_error=self.last_error,
        )


_WORKERS: dict[str, _WorkerHandle] = {}
_WORKER_LOCK = Lock()


def worker_status(job_id: str) -> WorkerStatus | None:
    with _WORKER_LOCK:
        handle = _WORKERS.get(job_id)
        if handle is None:
            return None
        status = handle.status()
        if not status.alive:
            _WORKERS.pop(job_id, None)
        return status


def start_worker(
    job_id: str,
    runner: Callable[[str], None],
    *,
    worker_id: str | None = None,
) -> WorkerStatus:
    """Start at most one local worker per job in this Python process."""
    with _WORKER_LOCK:
        existing = _WORKERS.get(job_id)
        if existing is not None and existing.thread.is_alive():
            return existing.status()
        identifier = worker_id or f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
        handle = _WorkerHandle(job_id, identifier, lambda: runner(identifier))
        _WORKERS[job_id] = handle
        handle.start()
        return handle.status()


class _LeaseHeartbeat:
    """Keep job and item leases alive while one provider chunk is running."""

    def __init__(self, bridge: ScannerDatabaseBridge, job_id: str, worker_id: str, lease_seconds: int) -> None:
        self.bridge = bridge
        self.job_id = job_id
        self.worker_id = worker_id
        self.lease_seconds = max(60, int(lease_seconds))
        self.stop_event = Event()
        interval = max(15.0, min(60.0, self.lease_seconds / 3.0))

        def beat() -> None:
            while not self.stop_event.wait(interval):
                try:
                    self.bridge.renew_scan_job_leases(
                        self.job_id, self.worker_id, lease_seconds=self.lease_seconds,
                    )
                except Exception:
                    # The main loop remains authoritative. A temporary heartbeat
                    # failure is tolerated; the next database operation will expose
                    # a durable repository outage if it persists.
                    pass

        self.thread = Thread(target=beat, name=f"idx-heartbeat-{job_id[:8]}", daemon=True)

    def __enter__(self) -> "_LeaseHeartbeat":
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)


def run_durable_job_loop(
    *,
    bridge_factory: Callable[[], ScannerDatabaseBridge],
    job_id: str,
    worker_id: str,
    process_chunk: Callable[[dict[str, Any], pd.DataFrame, str], Mapping[str, ItemOutcome]],
    finalize_job: Callable[[dict[str, Any], ScannerDatabaseBridge, str], Mapping[str, Any] | None],
    idle_sleep_seconds: float = 2.0,
    lease_seconds: int = 300,
    max_loop_seconds: int = 45 * 60,
) -> None:
    """Claim, process, checkpoint and finalize a persisted scan job.

    Each item is committed independently after its chunk finishes. A process death
    can therefore lose at most the currently leased chunk; expired leases are
    automatically returned to RETRY by the database RPC.
    """
    bridge = bridge_factory()
    loop_started = time.monotonic()
    consecutive_loop_errors = 0

    while time.monotonic() - loop_started < max(60, int(max_loop_seconds)):
        try:
            job = bridge.read_scan_job(job_id)
            if not job or str(job.get("status", "")).upper() in TERMINAL_JOB_STATES:
                return
            lease = bridge.claim_scan_job_lease(job_id, worker_id, lease_seconds=lease_seconds)
            if not lease:
                # Another server worker currently owns the job. It remains durable.
                return
            job = lease
            phase = str(job.get("phase") or "TECHNICAL")
            chunk_size = max(1, int(job.get("chunk_size", 20) or 20))
            if str(job.get("status", "")).upper() == "FINALIZING":
                try:
                    with _LeaseHeartbeat(bridge, job_id, worker_id, lease_seconds):
                        summary = finalize_job(job, bridge, worker_id) or {}
                    refreshed = bridge.refresh_scan_job_counters(job_id)
                    failed = int(refreshed.get("failed_items", 0) or 0)
                    final_status = "COMPLETE_WITH_FAILURES" if failed else "COMPLETE"
                    bridge.update_scan_job(
                        job_id, status=final_status, phase="COMPLETE",
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        lease_expires_at=None, active_worker=None,
                        result_summary=json_safe(summary), last_error=None,
                    )
                except Exception as exc:
                    bridge.update_scan_job(
                        job_id, status="PAUSED", active_worker=None, lease_expires_at=None,
                        last_error=f"FINALIZE_FAILED: {type(exc).__name__}: {str(exc)[:1000]}",
                    )
                return
            claimed = bridge.claim_scan_job_items(
                job_id,
                phase,
                limit=chunk_size,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
            )

            if not claimed.empty:
                try:
                    with _LeaseHeartbeat(bridge, job_id, worker_id, lease_seconds):
                        outcomes = process_chunk(job, claimed.copy(), worker_id)
                    outcomes = outcomes if isinstance(outcomes, Mapping) else {}
                except Exception as exc:
                    # Release the whole claimed chunk immediately. Waiting for a
                    # lease timeout would make reconnects appear frozen.
                    outcomes = {
                        str(row.get("item_key", "")): ItemOutcome(
                            False,
                            payload={"ticker": row.get("ticker")},
                            error=f"CHUNK_FAILED: {type(exc).__name__}: {str(exc)[:800]}",
                            retry_delay_seconds=30,
                        )
                        for row in claimed.to_dict("records")
                    }
                for _, item_row in claimed.iterrows():
                    item = item_row.to_dict()
                    key = str(item.get("item_key", ""))
                    outcome = outcomes.get(key)
                    if outcome is None:
                        outcome = ItemOutcome(False, error="PROCESSOR_RETURNED_NO_OUTCOME")
                    if outcome.success:
                        bridge.complete_scan_job_item(item, result_payload=json_safe(outcome.payload or {}))
                    else:
                        bridge.fail_scan_job_item(
                            item,
                            outcome.error or "ITEM_FAILED",
                            retry_delay_seconds=outcome.retry_delay_seconds,
                            result_payload=json_safe(outcome.payload or {}),
                        )
                counters = bridge.refresh_scan_job_counters(job_id)
                bridge.update_scan_job(
                    job_id,
                    status="RUNNING",
                    result_summary={
                        "completed_items": int(counters.get("completed_items", 0) or 0),
                        "failed_items": int(counters.get("failed_items", 0) or 0),
                        "retry_items": int(counters.get("retry_items", 0) or 0),
                        "last_chunk_size": int(len(claimed)),
                        "last_worker": worker_id,
                        "last_checkpoint_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                consecutive_loop_errors = 0
                continue

            items = bridge.read_scan_job_items(job_id, phase=phase, include_payload=False)
            if items.empty:
                bridge.update_scan_job(job_id, status="FAILED", last_error="JOB_HAS_NO_ITEMS", finished_at=datetime.now(timezone.utc).isoformat())
                return
            statuses = items.get("status", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
            terminal = statuses.isin(TERMINAL_ITEM_STATES)
            if bool(terminal.all()):
                bridge.update_scan_job(job_id, status="FINALIZING", active_worker=worker_id)
                latest = bridge.read_scan_job(job_id)
                try:
                    with _LeaseHeartbeat(bridge, job_id, worker_id, lease_seconds):
                        summary = finalize_job(latest, bridge, worker_id) or {}
                    refreshed = bridge.refresh_scan_job_counters(job_id)
                    failed = int(refreshed.get("failed_items", 0) or 0)
                    final_status = "COMPLETE_WITH_FAILURES" if failed else "COMPLETE"
                    bridge.update_scan_job(
                        job_id,
                        status=final_status,
                        phase="COMPLETE",
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        lease_expires_at=None,
                        active_worker=None,
                        result_summary=json_safe(summary),
                        last_error=None,
                    )
                except Exception as exc:
                    bridge.update_scan_job(
                        job_id,
                        status="PAUSED",
                        phase=phase,
                        active_worker=None,
                        lease_expires_at=None,
                        last_error=f"FINALIZE_FAILED: {type(exc).__name__}: {str(exc)[:1000]}",
                    )
                return

            # Items may be leased by another worker or waiting for retry backoff.
            bridge.update_scan_job(job_id, status="RUNNING", active_worker=worker_id)
            time.sleep(max(0.2, float(idle_sleep_seconds)))
            consecutive_loop_errors = 0
        except Exception as exc:
            consecutive_loop_errors += 1
            try:
                update_fields = {
                    "status": "PAUSED" if consecutive_loop_errors >= 3 else "RUNNING",
                    "last_error": f"WORKER_LOOP: {type(exc).__name__}: {str(exc)[:1000]}",
                    "active_worker": None if consecutive_loop_errors >= 3 else worker_id,
                }
                if consecutive_loop_errors >= 3:
                    update_fields["lease_expires_at"] = None
                bridge.update_scan_job(job_id, **update_fields)
            except Exception:
                pass
            if consecutive_loop_errors >= 3:
                return
            time.sleep(min(10.0, 1.5 * consecutive_loop_errors))

    try:
        bridge.update_scan_job(
            job_id,
            status="PAUSED",
            active_worker=None,
            lease_expires_at=None,
            last_error="WORKER_TIME_SLICE_EXPIRED; safe to resume from database",
        )
    except Exception:
        pass


__all__ = [
    "RESUMABLE_SCAN_VERSION",
    "TERMINAL_JOB_STATES",
    "ACTIVE_JOB_STATES",
    "TERMINAL_ITEM_STATES",
    "ItemOutcome",
    "WorkerStatus",
    "json_safe",
    "frame_from_records",
    "start_worker",
    "worker_status",
    "run_durable_job_loop",
]
