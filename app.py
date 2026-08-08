from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import streamlit as st


APP_VERSION = "9.8.0-guarded-real-money"
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(
    page_title="IDX Scanner v9 Macro-First",
    page_icon="📊",
    layout="wide",
)

REQUIRED_FILES = (
    "scanner.py",
    "scanner_database.py",
    "narrative_engine.py",
    "macro_engine.py",
    "simple_focus.py",
    "two_stage_pipeline.py",
    "free_data_providers.py",
    "ihsg_direction.py",
    "idx_trading_calendar.py",
    "incremental_store.py",
    "research_maintenance.py",
    "selector_engine.py",
    "database_first.py",
    "resumable_scan.py",
    "resumable_app_engine.py",
    "decision_overlay.py",
    "v9_dashboard.py",
)
missing = [name for name in REQUIRED_FILES if not (APP_ROOT / name).is_file()]
if missing:
    st.error("Deployment v9 tidak lengkap.")
    st.code("\n".join(missing), language="text")
    st.stop()

from scanner import parse_portfolio_csv, parse_ticker_csv  # noqa: E402
from scanner_database import ScannerDatabaseBridge  # noqa: E402
from macro_engine import MACRO_ENGINE_VERSION  # noqa: E402
from simple_focus import SIMPLE_FOCUS_VERSION  # noqa: E402
from decision_overlay import DECISION_OVERLAY_VERSION  # noqa: E402
from v9_dashboard import V9_DASHBOARD_VERSION, render_dashboard_html, select_top_candidates  # noqa: E402
from database_first import DATABASE_FIRST_VERSION  # noqa: E402
from resumable_scan import (  # noqa: E402
    frame_from_records,
    run_durable_job_loop,
    start_worker,
    worker_status,
)
from resumable_app_engine import (  # noqa: E402
    finalize_daily_scan_job,
    process_daily_scan_chunk,
)


st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; padding-bottom: 2.5rem;}
      [data-testid="stMetricValue"] {font-size: 1.35rem;}
      .v9-note {border:1px solid #334155; border-radius:10px; padding:12px 14px; background:#0f172a;}
      .small-muted {font-size:.84rem; color:#94a3b8;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _database_job_error_message(error: object, *, action: str = "membaca") -> str:
    """Explain database failures without misclassifying timeouts as permission errors."""
    text = str(error or "").strip()
    lowered = text.lower()
    if any(token in lowered for token in ("transport failure", "readtimeout", "read timed out", "connecttimeout", "connectionerror", "connection aborted", "temporarily unavailable")):
        return (
            f"Supabase timeout/gangguan jaringan saat {action} repository job. "
            "Scanner v9.8.0 memakai staged scan, bounded provider refresh, guarded real-money authorization, dan batch checkpoint. "
            "Ini bukan indikasi permission 403/42501. Coba muat ulang halaman; bila berulang, "
            "cek status project Supabase/koneksi Streamlit dan naikkan SCANNER_DATABASE_TIMEOUT. "
            f"Detail: {text[:500]}"
        )
    if any(token in lowered for token in ("http 401", "http 403", "42501", "permission denied", "row-level security")):
        return (
            f"Hak akses Supabase ditolak saat {action} repository job. Jalankan "
            "database/permissions_hotfix_v9_4_1.sql dan pastikan Streamlit memakai SUPABASE_SECRET_KEY "
            "atau SUPABASE_SERVICE_ROLE_KEY, bukan publishable/anon key. "
            f"Detail: {text[:500]}"
        )
    if any(token in lowered for token in ("pgrst205", "does not exist", "relation", "schema cache", "not found")):
        return (
            f"Schema resumable belum siap saat {action} repository job. Jalankan "
            "database/migration_v12_resumable_scan_jobs.sql lalu database/permissions_hotfix_v9_4_1.sql. "
            f"Detail: {text[:500]}"
        )
    return f"Repository job gagal saat {action}. Detail: {text[:600]}"








def _safe_display(frame: pd.DataFrame | None, columns: Iterable[str] | None = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if columns is not None:
        selected = [column for column in columns if column in out.columns]
        out = out.loc[:, selected]
    for column in out.select_dtypes(include=["object"]).columns:
        out[column] = out[column].map(lambda value: "" if value is None else str(value))
    return out.reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)


@st.cache_data(ttl=1800, show_spinner=False)




@st.cache_data(ttl=900, show_spinner=False)



@st.cache_data(ttl=900, show_spinner=False)
















def _runtime_tokens(itick_token: str = "", twelve_token: str = "") -> dict[str, str]:
    def secret(name: str) -> str:
        try:
            return str(st.secrets.get(name, "") or "").strip()
        except Exception:
            return ""

    # Tokens are read at runtime only and are never persisted in scan_jobs.
    return {
        "itick_api_token": str(itick_token or os.getenv("ITICK_API_TOKEN", "") or secret("ITICK_API_TOKEN")).strip(),
        "twelve_data_api_key": str(twelve_token or os.getenv("TWELVE_DATA_API_KEY", "") or secret("TWELVE_DATA_API_KEY")).strip(),
    }


def _start_resumable_worker(job: Mapping[str, object], runtime: Mapping[str, str]) -> object:
    job_id = str(job.get("job_id", ""))
    runtime_copy = dict(runtime)

    def runner(worker_id: str) -> None:
        process = lambda current_job, items, wid: process_daily_scan_chunk(
            current_job, items, wid, runtime=runtime_copy,
        )
        finalize = lambda current_job, bridge, wid: finalize_daily_scan_job(
            current_job, bridge, wid, runtime=runtime_copy,
        )
        run_durable_job_loop(
            bridge_factory=ScannerDatabaseBridge,
            job_id=job_id,
            worker_id=worker_id,
            process_chunk=process,
            finalize_job=finalize,
        )

    return start_worker(job_id, runner)


def _artifact_frame(artifacts: pd.DataFrame, artifact_type: str) -> pd.DataFrame:
    if artifacts is None or artifacts.empty or "artifact_type" not in artifacts.columns:
        return pd.DataFrame()
    subset = artifacts.loc[artifacts["artifact_type"].astype(str).eq(artifact_type)]
    if subset.empty:
        return pd.DataFrame()
    payload = subset.sort_values("chunk_number").iloc[-1].get("payload")
    return frame_from_records(payload)


def _job_result_from_artifacts(job: Mapping[str, object], artifacts: pd.DataFrame) -> dict[str, object]:
    artifact_types = set(artifacts.get("artifact_type", pd.Series(dtype=str)).astype(str).tolist()) if isinstance(artifacts, pd.DataFrame) else set()
    final_available = "FINAL_NEXT_LEADERS" in artifact_types or "FINAL_SWING_READY" in artifact_types
    leaders = _artifact_frame(artifacts, "FINAL_NEXT_LEADERS")
    swings = _artifact_frame(artifacts, "FINAL_SWING_READY")
    if not final_available:
        leaders = _artifact_frame(artifacts, "PROVISIONAL_NEXT_LEADERS")
        swings = _artifact_frame(artifacts, "PROVISIONAL_SWING_READY")
    leaders_all = _artifact_frame(artifacts, "FINAL_NEXT_LEADERS_ALL")
    swings_all = _artifact_frame(artifacts, "FINAL_SWING_READY_ALL")
    evidence_detail = _artifact_frame(artifacts, "FINAL_EVIDENCE_DETAIL")
    scoring_contract = _artifact_frame(artifacts, "FINAL_SCORING_CONTRACT")
    macro_snapshot = _artifact_frame(artifacts, "FINAL_MACRO_SNAPSHOT")
    macro_sector = _artifact_frame(artifacts, "FINAL_MACRO_SECTOR_MAP")
    portfolio = _artifact_frame(artifacts, "FINAL_PORTFOLIO")
    coverage = _artifact_frame(artifacts, "FINAL_COVERAGE")
    ohlcv_audit = _artifact_frame(artifacts, "FINAL_OHLCV_AUDIT")
    stage_timings = _artifact_frame(artifacts, "FINAL_STAGE_TIMINGS")
    database_sync = _artifact_frame(artifacts, "FINAL_DATABASE_SYNC_REPORT")
    fundamental_provider_audit = _artifact_frame(artifacts, "FINAL_FUNDAMENTAL_PROVIDER_AUDIT")
    audit = _artifact_frame(artifacts, "FINAL_JOB_AUDIT")
    if audit.empty:
        audit = _artifact_frame(artifacts, "PROVISIONAL_JOB_AUDIT")
    completed = int(job.get("completed_items", 0) or 0)
    universe = list(job.get("universe_payload") or [])
    completed_tickers: list[str] = []
    if not audit.empty:
        row = audit.iloc[0]
        raw_completed = row.get("ohlcv_ready_ticker_list") or row.get("completed_ticker_list")
        if isinstance(raw_completed, list):
            completed_tickers = [str(value) for value in raw_completed if str(value).strip()]
    if not completed_tickers:
        completed_tickers = [str(value) for value in universe[:completed]]
    elapsed = 0.0
    try:
        started = pd.Timestamp(job.get("started_at"))
        finished = pd.Timestamp(job.get("finished_at"))
        elapsed = max(0.0, float((finished - started).total_seconds()))
    except Exception:
        pass
    return {
        "mode": "resumable_chunked_daily_scan",
        "scanner_version": APP_VERSION,
        "prepared": {str(ticker): pd.DataFrame() for ticker in completed_tickers},
        "focus_screens": {
            "next_leaders": leaders,
            "swing_ready": swings,
            "next_leaders_all": leaders_all,
            "swing_ready_all": swings_all,
            "multibagger": leaders,
            "core_swing": swings,
            "production_evidence_detail": evidence_detail,
            "production_scoring_audit": scoring_contract,
        },
        "macro_snapshot": macro_snapshot,
        "macro_sector_map": macro_sector,
        "portfolio_analysis": portfolio,
        "database_coverage_after": coverage,
        "scan_coverage_summary": audit,
        "ohlcv_database_audit": ohlcv_audit,
        "stage_timings": stage_timings,
        "database_sync_report": database_sync,
        "fundamental_history_report": fundamental_provider_audit,
        "scan_elapsed_seconds": elapsed,
        "job_id": job.get("job_id"),
        "job_status": job.get("status"),
        "ranking_state": "FINAL" if final_available else "PROVISIONAL",
        "ranking_quality_state": (
            str(audit.iloc[0].get("ranking_state") or "UNKNOWN").upper()
            if isinstance(audit, pd.DataFrame) and not audit.empty else "UNKNOWN"
        ),
    }


def _render_resumable_job(job: Mapping[str, object], bridge: ScannerDatabaseBridge) -> None:
    total = max(0, int(job.get("total_items", 0) or 0))
    completed = max(0, int(job.get("completed_items", 0) or 0))
    failed = max(0, int(job.get("failed_items", 0) or 0))
    retries = max(0, int(job.get("retry_items", 0) or 0))
    done = min(total, completed + failed)
    item_progress_pct = 100.0 * done / total if total else 0.0
    status = str(job.get("status", "UNKNOWN")).upper()
    # Item processing may be 100% while ranking/execution finalisation is still
    # running.  Show actual finalizer sub-phases instead of appearing frozen at 95%.
    phase = str(job.get("phase", "") or "").upper()
    finalizer_progress = {
        "RANKING_READY": 96.0,
        "EVIDENCE_REFRESH": 97.0,
        "EXECUTION_VERIFY": 98.0,
        "DATABASE_SYNC": 99.0,
        "ARTIFACT_PUBLISH": 99.5,
    }
    progress_pct = finalizer_progress.get(phase, min(item_progress_pct, 95.0)) if status == "FINALIZING" else item_progress_pct

    terminal_items = pd.DataFrame()
    partial_count = 0
    if status in {"FINALIZING", "COMPLETE", "COMPLETE_WITH_FAILURES", "FAILED", "CANCELLED"}:
        try:
            terminal_items = bridge.read_scan_job_items(
                str(job.get("job_id")), include_payload=True, limit=5000,
            )
            if not terminal_items.empty and "result_payload" in terminal_items.columns:
                completion_states = terminal_items["result_payload"].map(
                    lambda value: str(value.get("completion_state", ""))
                    if isinstance(value, Mapping) else ""
                )
                partial_count = int(completion_states.isin({
                    "PARTIAL_SNAPSHOT", "PARTIAL_HISTORY", "MARKET_STATUS_ONLY", "NO_PUBLIC_EVIDENCE",
                    "TECHNICAL_UNAVAILABLE",
                }).sum())
        except Exception:
            terminal_items = pd.DataFrame()

    cols = st.columns(6)
    cols[0].metric("Status job", status)
    cols[1].metric("Selesai diproses", f"{completed}/{total}")
    cols[2].metric("Data parsial", partial_count)
    cols[3].metric("Gagal sistem", failed)
    cols[4].metric("Menunggu retry", retries)
    cols[5].metric("Progress", f"{progress_pct:.1f}%")
    progress_label = str(job.get("phase", "") or "")
    if status != "FINALIZING":
        progress_label = f"{progress_label} • chunk {job.get('chunk_size', 0)} ticker"
    st.progress(min(100, int(progress_pct)), text=progress_label)
    if status == "FINALIZING":
        finalizing_text = {
            "RANKING_READY": "Ranking provisional sudah siap; memilih kandidat untuk evidence enrichment.",
            "EVIDENCE_REFRESH": "Memperbarui fundamental/news/status hanya pada shortlist prioritas; bukan seluruh 400 ticker.",
            "EXECUTION_VERIFY": "Memvalidasi harga/EOD kandidat prioritas (maks. 12 ticker), bukan mengulang scan 400 ticker.",
            "DATABASE_SYNC": "Ranking final sudah dihitung; menyinkronkan tabel keputusan secara terbatas.",
            "ARTIFACT_PUBLISH": "Sinkronisasi utama selesai; menerbitkan artifact ranking final.",
        }.get(phase, "Pemrosesan ticker selesai; finalizer sedang membentuk ranking final.")
        st.info(f"{done}/{total} ticker terminal. {finalizing_text}")
    worker = worker_status(str(job.get("job_id", "")))
    if worker is not None and worker.alive:
        st.caption(f"Worker server aktif: {worker.worker_id}. Browser boleh terputus; checkpoint tersimpan di database. Jika host tidur/recycle, job pause lalu session berikutnya melanjutkan setelah lease kedaluwarsa.")
    else:
        st.caption("Tidak ada worker lokal aktif. Job tetap tersimpan dan dapat diklaim kembali oleh session berikutnya.")
    if job.get("last_error"):
        st.warning(str(job.get("last_error")))
    with st.expander("Status ticker / retry", expanded=status in {"PAUSED", "COMPLETE_WITH_FAILURES", "FAILED"}):
        try:
            items = terminal_items if not terminal_items.empty else bridge.read_scan_job_items(
                str(job.get("job_id")), include_payload=True, limit=5000,
            )
            if not items.empty and "result_payload" in items.columns:
                local = items.copy()
                local["completion_state"] = local["result_payload"].map(
                    lambda value: value.get("completion_state") if isinstance(value, Mapping) else None
                )
                local["snapshot_ready"] = local["result_payload"].map(
                    lambda value: value.get("snapshot_ready") if isinstance(value, Mapping) else None
                )
                local["history_periods"] = local["result_payload"].map(
                    lambda value: value.get("history_periods") if isinstance(value, Mapping) else None
                )
                local["database_sync_state"] = local["result_payload"].map(
                    lambda value: value.get("database_sync_state") if isinstance(value, Mapping) else None
                )
                local = local.drop(columns=["result_payload"], errors="ignore")
                items = local
        except Exception as exc:
            items = pd.DataFrame([{"status": "READ_FAIL", "last_error": str(exc)}])
        st.dataframe(_safe_display(items), width="stretch", hide_index=True)


st.title("IDX Super Scanner v9.8.0 — Macro-First Guarded Real Money")
st.caption(
    f"{APP_VERSION} • database-first {DATABASE_FIRST_VERSION} • macro {MACRO_ENGINE_VERSION} • decision {SIMPLE_FOCUS_VERSION} • inventory {DECISION_OVERLAY_VERSION} • dashboard {V9_DASHBOARD_VERSION}"
)
st.markdown(
    """
    <div class="v9-note">
      <b>Single Scan • Database-First • Resumable</b><br>
      Satu tombol menjalankan seluruh engine. Data database yang masih current dipakai ulang, evidence missing/stale diperbarui otomatis, lalu hasil dan delta evidence disimpan kembali untuk scan berikutnya.
      Output utama: <b>Top 3 Report Card</b>, <b>The Next Leader</b>, dan <b>Swing Ready</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Input")
    ticker_file = st.file_uploader("Universe ticker CSV", type=["csv"])
    portfolio_file = st.file_uploader("Portfolio CSV (opsional)", type=["csv"])
    account_size = float(st.number_input("Nilai akun (Rp)", min_value=0, value=5_000_000, step=500_000))
    cash_on_hand = float(st.number_input("Cash tersedia (Rp)", min_value=0, value=1_000_000, step=100_000))
    # Chunking/retry are engine responsibilities, not trading decisions. A fixed
    # 40-name durable chunk halves orchestration round trips versus the old 20-name
    # default while keeping each checkpoint small enough for Streamlit/Supabase.
    job_chunk_size = 40
    max_attempts = 2
    with st.expander("Pengaturan scan", expanded=False):
        period = st.selectbox("Riwayat OHLCV", ["3y", "5y", "10y"], index=1)
        risk_per_trade_pct = st.slider("Risiko per transaksi", 0.25, 2.00, 0.75, 0.25) / 100.0
        itick_token = st.text_input("iTick API token", value=os.getenv("ITICK_API_TOKEN", ""), type="password")
        twelve_token = st.text_input("Twelve Data API key", value=os.getenv("TWELVE_DATA_API_KEY", ""), type="password")
        st.caption("Database-first otomatis: data current dipakai ulang; hanya evidence missing/stale yang di-refresh dan seluruh delta disimpan kembali.")
    run_scan = st.button("SCAN", type="primary", width="stretch")


# -----------------------------------------------------------------------------
# One-button durable workflow. Database maintenance is part of every scan.
# -----------------------------------------------------------------------------
bridge = ScannerDatabaseBridge()
job_type = "DAILY_SCAN"
runtime = _runtime_tokens(itick_token, twelve_token)
active_job: dict[str, object] = {}
job_error = ""
try:
    requested_job_id = str(st.session_state.get("v94_job_id", "") or "")
    if requested_job_id:
        candidate_job = bridge.read_scan_job(requested_job_id)
        if str(candidate_job.get("job_type", "")).upper() == job_type:
            active_job = candidate_job
    if not active_job:
        active_job = bridge.read_latest_scan_job(
            job_type=job_type,
            statuses=["PENDING", "RUNNING", "PAUSED", "FINALIZING", "COMPLETE", "COMPLETE_WITH_FAILURES"],
        )
except Exception as exc:
    job_error = f"{type(exc).__name__}: {str(exc)[:500]}"

if run_scan:
    portfolio = pd.DataFrame()
    if portfolio_file is not None:
        try:
            portfolio = parse_portfolio_csv(portfolio_file)
        except Exception as exc:
            st.error(f"Portfolio CSV tidak valid: {exc}")
            st.stop()

    if ticker_file is not None:
        try:
            tickers = parse_ticker_csv(ticker_file, max_tickers=400, strict_limit=True)
        except Exception as exc:
            st.error(f"CSV ticker tidak valid: {exc}")
            st.stop()
        portfolio_tickers = portfolio["ticker"].dropna().astype(str).drop_duplicates().tolist() if not portfolio.empty and "ticker" in portfolio else []
        all_tickers = list(dict.fromkeys(portfolio_tickers + tickers))[:400]
        safe_config = {
            "period": period,
            "account_size_idr": account_size,
            "cash_on_hand_idr": cash_on_hand,
            "risk_per_trade_pct": risk_per_trade_pct,
            "allow_partial_database": True,
            "portfolio_records": portfolio.to_dict("records") if not portfolio.empty else [],
            "provider_batch_size": int(job_chunk_size),
            # One-button scan, staged like the faster Emir pipeline: all names
            # receive technical discovery first; expensive evidence refresh is
            # job-global and bounded after provisional ranking. Never bind deep
            # provider budgets to chunk_size.
            "evidence_refresh_cap": 16,
            "evidence_fundamental_cap": 16,
            "evidence_official_cap": 8,
            "evidence_snapshot_cap": 12,
            "evidence_news_cap": 12,
            "execution_verification_cap": 10,
        }
        try:
            active_job = bridge.create_or_resume_scan_job(
                job_type=job_type,
                tickers=all_tickers,
                config_payload=safe_config,
                phase="TECHNICAL",
                chunk_size=int(job_chunk_size),
                max_attempts=int(max_attempts),
                model_version=APP_VERSION,
            )
            st.session_state["v94_job_id"] = str(active_job.get("job_id", ""))
        except Exception as exc:
            st.error(_database_job_error_message(f"{type(exc).__name__}: {exc}", action="membuat/melanjutkan"))
            st.stop()
    elif not active_job or str(active_job.get("job_type", "")).upper() != job_type:
        st.error("Upload CSV untuk membuat job baru. Job aktif sebelumnya akan otomatis dilanjutkan bila masih tersedia.")
        st.stop()

    _start_resumable_worker(active_job, runtime)
    try:
        refreshed = bridge.read_scan_job(str(active_job.get("job_id")))
        if refreshed:
            active_job = refreshed
    except Exception as exc:
        # Keep the durable job object we already have; a transient read timeout
        # must not blank the UI or terminate a worker that has already started.
        st.warning(_database_job_error_message(f"{type(exc).__name__}: {exc}", action="menyegarkan"))

if job_error and not active_job:
    st.error(_database_job_error_message(job_error, action="membaca"))

if active_job and str(active_job.get("job_type", "")).upper() == job_type:
    status = str(active_job.get("status", "")).upper()
    pause_resume_key = f"v94_auto_resumed_{active_job.get('job_id', '')}"
    auto_resume_paused = status == "PAUSED" and not bool(st.session_state.get(pause_resume_key, False))
    if auto_resume_paused:
        st.session_state[pause_resume_key] = True
    should_start_worker = status in {"PENDING", "RUNNING", "FINALIZING"} or status == "PAUSED" and (run_scan or auto_resume_paused)
    if should_start_worker:
        _start_resumable_worker(active_job, runtime)
        try:
            active_job = bridge.read_scan_job(str(active_job.get("job_id")))
        except Exception:
            pass
    _render_resumable_job(active_job, bridge)

    status = str(active_job.get("status", "")).upper()
    if status in {"COMPLETE_WITH_FAILURES", "FAILED"}:
        try:
            failed_rows = bridge.read_scan_job_items(
                str(active_job.get("job_id")), include_payload=False, limit=5000,
            )
            failed_rows = failed_rows.loc[
                failed_rows.get("status", pd.Series(dtype=str)).astype(str).str.upper().eq("FAILED")
            ] if not failed_rows.empty else pd.DataFrame()
        except Exception:
            failed_rows = pd.DataFrame()
        if not failed_rows.empty:
            st.error(
                "Job lama selesai dengan ticker gagal. v9.8.0 mengulang ticker tersebut di job yang sama, "
                "sehingga checkpoint ticker COMPLETE dan basis ranking sebelumnya tetap digabungkan."
            )
            if st.button("Ulangi hanya ticker gagal dalam job yang sama", type="primary", key=f"retry_failed_{active_job.get('job_id')}"):
                requeued = bridge.requeue_failed_scan_job_items(str(active_job.get("job_id")))
                st.session_state.pop("v9_scan_result", None)
                refreshed_job = bridge.read_scan_job(str(active_job.get("job_id")))
                _start_resumable_worker(refreshed_job, runtime)
                st.success(f"{requeued} ticker dikembalikan ke antrean tanpa menghapus hasil ticker yang sudah COMPLETE.")
                st.rerun()

    if status in {"COMPLETE", "COMPLETE_WITH_FAILURES"}:
        artifacts = bridge.read_scan_job_artifacts(str(active_job.get("job_id")))
        st.session_state["v9_scan_result"] = _job_result_from_artifacts(active_job, artifacts)
    elif status in {"PENDING", "RUNNING", "FINALIZING"}:
        provisional_loaded = False
        if status == "FINALIZING":
            try:
                artifacts = bridge.read_scan_job_artifacts(str(active_job.get("job_id")))
                provisional = _job_result_from_artifacts(active_job, artifacts)
                artifact_types = set(artifacts.get("artifact_type", pd.Series(dtype=str)).astype(str).tolist()) if isinstance(artifacts, pd.DataFrame) else set()
                has_ranking_artifact = "PROVISIONAL_NEXT_LEADERS" in artifact_types or "PROVISIONAL_SWING_READY" in artifact_types
                if has_ranking_artifact:
                    st.session_state["v9_scan_result"] = provisional
                    provisional_loaded = True
                    st.warning(
                        "Ranking sementara sudah tersedia. Finalizer masih memverifikasi harga/entry dan menyimpan hasil final ke database."
                    )
            except Exception as exc:
                st.caption(f"Artifact ranking sementara belum dapat dibaca: {type(exc).__name__}")
        if not provisional_loaded:
            st.info("Job berjalan di server. Ponsel/browser boleh ditutup; checkpoint dan data yang sudah selesai tetap tersimpan di database.")
        time.sleep(2.0)
        st.rerun()
    elif status == "PAUSED":
        st.warning("Job dipause secara aman. Session baru mencoba auto-resume satu kali; tekan Mulai / Lanjutkan bila provider/database sudah pulih.")
        st.stop()
    elif status in {"FAILED", "CANCELLED"}:
        st.error("Job berhenti terminal. Audit ticker di atas menunjukkan item yang gagal.")
        st.stop()


result = st.session_state.get("v9_scan_result")
if not result:
    st.info("Upload universe ticker lalu jalankan scanner. Maksimum 400 ticker.")
    st.stop()

focus = result.get("focus_screens", {})
leaders = focus.get("next_leaders", pd.DataFrame()).copy()
swings = focus.get("swing_ready", pd.DataFrame()).copy()
macro_snapshot = result.get("macro_snapshot", pd.DataFrame())
elapsed = _finite(result.get("scan_elapsed_seconds"), 0.0)
ranking_state = str(result.get("ranking_state", "FINAL")).upper()
coverage_summary = result.get("scan_coverage_summary", pd.DataFrame())
if ranking_state == "PROVISIONAL":
    basis = ""
    if isinstance(coverage_summary, pd.DataFrame) and not coverage_summary.empty:
        row = coverage_summary.iloc[0]
        requested = int(_finite(row.get("requested_tickers"), 0))
        ready = int(_finite(row.get("technical_ready_tickers", row.get("completed_tickers")), 0))
        unavailable = int(_finite(row.get("technical_unavailable_tickers"), 0))
        basis = f" Basis saat ini {ready}/{requested} ticker technical-ready; {unavailable} belum memiliki OHLCV valid."
    st.warning(
        "Ranking yang tampil masih PROVISIONAL karena job sedang FINALIZING. "
        "Entry/SL/TP dan execution readiness dapat berubah setelah verifikasi harga selesai." + basis
    )

coverage_row = coverage_summary.iloc[0] if isinstance(coverage_summary, pd.DataFrame) and not coverage_summary.empty else pd.Series(dtype=object)
requested_metric = int(_finite(coverage_row.get("requested_tickers"), len(result.get("prepared", {}))))
ohlcv_ready_metric = int(_finite(coverage_row.get("ohlcv_ready_tickers"), len(result.get("prepared", {}))))
leader_valid_metric = int(_finite(coverage_row.get("next_leader_final_score_valid"), len(leaders)))
swing_valid_metric = int(_finite(coverage_row.get("swing_final_score_valid"), len(swings)))
metrics = st.columns(6)
metrics[0].metric("Universe", requested_metric)
metrics[1].metric("OHLCV ready", f"{ohlcv_ready_metric}/{requested_metric}" if requested_metric else ohlcv_ready_metric)
metrics[2].metric("Leader scored", leader_valid_metric)
metrics[3].metric("Swing scored", swing_valid_metric)
metrics[4].metric("Macro regime", str(macro_snapshot.iloc[0].get("macro_regime", "DATA_PENDING")) if not macro_snapshot.empty else "DATA_PENDING")
metrics[5].metric("Waktu", f"{elapsed:.1f} dtk")
ranking_quality_state = str(result.get("ranking_quality_state", coverage_row.get("ranking_state", "UNKNOWN"))).upper()
if ranking_quality_state not in {"VALID", "UNKNOWN"}:
    st.error(
        f"Ranking quality: {ranking_quality_state}. Hasil hanya mencakup {ohlcv_ready_metric}/{requested_metric} ticker OHLCV-ready dan belum layak menjadi dasar order penuh."
    )
if isinstance(coverage_summary, pd.DataFrame) and not coverage_summary.empty:
    requested_count = int(_finite(coverage_row.get("requested_tickers"), 0))
    fundamental_count = int(_finite(coverage_row.get("fundamental_ready"), 0))
    if requested_count > 0 and fundamental_count < requested_count:
        st.warning(
            f"Evidence fundamental lengkap tersedia untuk {fundamental_count}/{requested_count} ticker. "
            "Ticker lain tetap dipindai secara teknikal tetapi berstatus DATA_PENDING sampai cache/backfill terisi."
        )

if not leaders.empty:
    leaders["Final Score"] = pd.to_numeric(leaders.get("final_score", leaders.get("v9_next_leader_score")), errors="coerce")
if not swings.empty:
    swings["Final Score"] = pd.to_numeric(swings.get("final_score", swings.get("v9_swing_score")), errors="coerce")

dashboard_tab, market_tab, leader_tab, swing_tab, portfolio_tab = st.tabs([
    "Top 3 Dashboard", "Market Map", "The Next Leader", "Swing Ready", "Portfolio & Audit",
])

with dashboard_tab:
    market_regime = str(macro_snapshot.iloc[0].get("macro_regime", "DATA_PENDING")) if not macro_snapshot.empty else "DATA_PENDING"
    scan_id = str(active_job.get("job_id", "") or result.get("scan_id", "")) if isinstance(active_job, Mapping) else str(result.get("scan_id", ""))
    as_of = result.get("scan_finished_at", result.get("scan_started_at", ""))
    top_leaders = select_top_candidates(leaders, model="NEXT_LEADER", limit=3)
    top_swings = select_top_candidates(swings, model="SWING_READY", limit=3)
    top_leader_tab, top_swing_tab = st.tabs(["Top 3 Next Leader", "Top 3 Swing"])
    with top_leader_tab:
        st.markdown(render_dashboard_html(top_leaders, model="NEXT_LEADER", scan_id=scan_id, as_of=as_of, market_regime=market_regime), unsafe_allow_html=True)
        if not top_leaders.empty:
            st.download_button("Download Top 3 Next Leader CSV", top_leaders.to_csv(index=False).encode("utf-8-sig"), "v9_top3_next_leader.csv", "text/csv")
    with top_swing_tab:
        st.markdown(render_dashboard_html(top_swings, model="SWING_READY", scan_id=scan_id, as_of=as_of, market_regime=market_regime), unsafe_allow_html=True)
        if not top_swings.empty:
            st.download_button("Download Top 3 Swing CSV", top_swings.to_csv(index=False).encode("utf-8-sig"), "v9_top3_swing.csv", "text/csv")

with market_tab:
    if not macro_snapshot.empty:
        row = macro_snapshot.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Market context", f"{_finite(row.get('market_context_score', row.get('macro_regime_score'))):.1f}")
        cols[1].metric("Coverage", f"{_finite(row.get('macro_data_coverage_pct')):.0f}%")
        cols[2].metric("Breadth > EMA50", f"{_finite(row.get('breadth_above_ema50_pct')):.1f}%")
        cols[3].metric("IHSG 20D", f"{100.0 * _finite(row.get('ihsg_return_20d')):.2f}%")
        factor_cols = [column for column in macro_snapshot.columns if column.startswith("factor_")]
        factor_view = pd.DataFrame({
            "factor": [column.replace("factor_", "").replace("_score", "") for column in factor_cols],
            "score": [row.get(column) for column in factor_cols],
        })
        st.subheader("Macro factors")
        st.dataframe(_safe_display(factor_view), width="stretch", hide_index=True)
    st.subheader("Sector opportunity map")
    st.dataframe(_safe_display(result.get("macro_sector_map", pd.DataFrame())), width="stretch", hide_index=True)
    ihsg = result.get("ihsg_direction") or {}
    if isinstance(ihsg, Mapping):
        st.caption(
            f"IHSG model: {ihsg.get('consensus_direction', 'NO_EDGE')} • "
            f"confidence {_finite(ihsg.get('consensus_confidence')):.1f}% • "
            f"risk multiplier {_finite(ihsg.get('risk_budget_multiplier'), 0.5):.2f}x"
        )
    with st.expander("Macro source audit"):
        st.dataframe(_safe_display(result.get("macro_source_report", pd.DataFrame())), width="stretch", hide_index=True)

with leader_tab:
    columns = [
        "rank", "ticker", "sector", "candidate_type", "status",
        "Final Score", "score_coverage_pct",
        "fundamental_freshness_state", "fundamental_data_quality_score", "fundamental_conviction_cap",
        "fundamental_official_verified", "fundamental_official_source_coverage_pct", "fundamental_cashflow_state",
        "market_regime", "market_context_score", "real_money_authorization_state", "real_money_authorization_blockers", "real_money_manual_checks",
        "sector_source", "sector_confidence_pct",
        "business_quality_score", "future_fundamental_score", "valuation_mos_score",
        "management_capital_score", "issuer_macro_alignment_score",
        "narrative_flow_score", "technical_readiness_score",
        "silent_accumulation_score", "inventory_multi_horizon_score", "inventory_lifecycle",
        "distribution_risk_score", "anti_chase_gate", "decision_overlay_state", "retail_adoption_stage",
        "entry_low", "entry_high", "trigger", "stop_loss", "tp1", "tp2", "rr1",
        "recommended_allocation_idr", "recommended_lots", "selected_reason", "primary_risk",
    ]
    view = _safe_display(leaders.head(50), columns)
    if view.empty:
        st.warning("Belum ada kandidat The Next Leader.")
    else:
        st.dataframe(view, width="stretch", hide_index=True)
        st.download_button("Download The Next Leader CSV", view.to_csv(index=False).encode("utf-8-sig"), "v9_the_next_leader.csv", "text/csv")

with swing_tab:
    columns = [
        "rank", "ticker", "sector", "status", "Final Score", "score_coverage_pct",
        "real_money_authorization_state", "real_money_authorization_blockers", "real_money_manual_checks",
        "fundamental_data_quality_score", "fundamental_official_verified", "market_regime", "market_context_score",
        "production_gate_reason", "technical_execution_score", "issuer_macro_alignment_score", "narrative_flow_score",
        "silent_accumulation_score", "inventory_multi_horizon_score", "inventory_lifecycle",
        "distribution_risk_score", "anti_chase_gate", "decision_overlay_state",
        "business_quality_score", "risk_data_score", "next_leader_score", "strategy",
        "entry_low", "entry_high", "trigger_price", "stop_loss", "tp1", "tp2", "rr1", "rr2",
        "stockbit_order_lots", "next_action", "selected_reason", "primary_risk",
    ]
    view = _safe_display(swings.head(50), columns)
    if view.empty:
        st.warning("Belum ada setup Swing Ready.")
    else:
        st.dataframe(view, width="stretch", hide_index=True)
        st.download_button("Download Swing Ready CSV", view.to_csv(index=False).encode("utf-8-sig"), "v9_swing_ready.csv", "text/csv")

with portfolio_tab:
    portfolio_analysis = result.get("portfolio_analysis", pd.DataFrame())
    portfolio_summary = result.get("portfolio_summary", {})
    if isinstance(portfolio_summary, Mapping) and portfolio_summary:
        cols = st.columns(4)
        cols[0].metric("Nilai pasar", f"Rp {_finite(portfolio_summary.get('market_value_idr')):,.0f}")
        cols[1].metric("Unrealized P/L", f"Rp {_finite(portfolio_summary.get('unrealized_pnl_idr')):,.0f}")
        cols[2].metric("Open risk", f"Rp {_finite(portfolio_summary.get('open_risk_idr')):,.0f}")
        cols[3].metric("Estimasi equity", f"Rp {_finite(portfolio_summary.get('estimated_equity_idr')):,.0f}")
    if isinstance(portfolio_analysis, pd.DataFrame) and not portfolio_analysis.empty:
        st.dataframe(_safe_display(portfolio_analysis), width="stretch", hide_index=True)
    else:
        st.info("Portfolio CSV belum diunggah.")

    with st.expander("Ticker belum masuk ranking"):
        leader_all = focus.get("next_leaders_all", pd.DataFrame())
        swing_all = focus.get("swing_ready_all", pd.DataFrame())
        pending_frames = []
        if isinstance(leader_all, pd.DataFrame) and not leader_all.empty:
            leader_mask = leader_all["rank_eligible"].fillna(False).astype(bool) if "rank_eligible" in leader_all.columns else pd.Series(False, index=leader_all.index)
            local = leader_all.loc[~leader_mask].copy()
            if not local.empty:
                local["model"] = "THE_NEXT_LEADER"
                pending_frames.append(local)
        if isinstance(swing_all, pd.DataFrame) and not swing_all.empty:
            swing_mask = swing_all["rank_eligible"].fillna(False).astype(bool) if "rank_eligible" in swing_all.columns else pd.Series(False, index=swing_all.index)
            local = swing_all.loc[~swing_mask].copy()
            if not local.empty:
                local["model"] = "SWING_READY"
                pending_frames.append(local)
        if pending_frames:
            concat_frames = [frame.dropna(axis=1, how="all") for frame in pending_frames]
            pending = pd.concat(concat_frames, ignore_index=True, sort=False)
        else:
            pending = pd.DataFrame()
        st.dataframe(_safe_display(pending), width="stretch", hide_index=True)
    with st.expander("Scoring contract"):
        st.dataframe(_safe_display(focus.get("production_scoring_audit", pd.DataFrame())), width="stretch", hide_index=True)
    with st.expander("Evidence per ticker"):
        evidence = _safe_display(focus.get("production_evidence_detail", pd.DataFrame()))
        st.dataframe(evidence, width="stretch", hide_index=True)
        if not evidence.empty:
            st.download_button("Download Evidence CSV", evidence.to_csv(index=False).encode("utf-8-sig"), "v9_evidence.csv", "text/csv")
    with st.expander("Data coverage dan pipeline"):
        st.dataframe(_safe_display(result.get("scan_coverage_summary", pd.DataFrame())), width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("scanner_data_contract_audit", pd.DataFrame())), width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("stage_timings", pd.DataFrame())), width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("two_stage_coverage_audit", pd.DataFrame())), width="stretch", hide_index=True)
    with st.expander("OHLCV database dan provider audit"):
        ohlcv_audit = _safe_display(result.get("ohlcv_database_audit", pd.DataFrame()))
        st.dataframe(ohlcv_audit, width="stretch", hide_index=True)
        if not ohlcv_audit.empty:
            st.download_button("Download OHLCV Audit CSV", ohlcv_audit.to_csv(index=False).encode("utf-8-sig"), "v9_ohlcv_audit.csv", "text/csv")
    with st.expander("Fundamental dan database audit"):
        st.dataframe(_safe_display(result.get("fundamental_history_report", pd.DataFrame())), width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("database_sync_report", pd.DataFrame())), width="stretch", hide_index=True)
