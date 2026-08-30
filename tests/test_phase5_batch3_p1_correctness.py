from pathlib import Path

import numpy as np
import pandas as pd

from scanner_database import (
    DatabaseSettings,
    FEATURE_CACHE_SCHEMA_VERSION,
    ScannerDatabaseBridge,
    _semantic_hash,
)
from fast_scan_engine import reconcile_requested_universe
from simple_focus import _weighted_final
from two_stage_pipeline import (
    ShortlistConfig,
    build_enrichment_recall_diagnostics,
    build_enrichment_shortlist,
)
from decision_snapshot import build_final_decision_snapshot, reload_final_decision_snapshot


ROOT = Path(__file__).resolve().parents[1]


def _tickers(count: int) -> list[str]:
    return [f"T{index:03d}.JK" for index in range(count)]


def test_core_universe_is_invariant_to_portfolio_overlay():
    core = _tickers(400)
    cases = (
        [],
        ["OUTSIDE.JK"],
        [f"OUT{index:02d}.JK" for index in range(15)],
        ["T010.JK", "T010.JK"],
        ["OUTSIDE.JK", "T010.JK"],
        ["T010.JK", "OUTSIDE.JK"],
    )

    resolved = []
    for portfolio in cases:
        universe, outside = reconcile_requested_universe(core, portfolio, max_tickers=400)
        resolved.append(universe)
        assert set(universe) == set(core)
        assert len(universe) == 400
        if "OUTSIDE.JK" in portfolio:
            assert "OUTSIDE.JK" in outside
        if "T010.JK" in portfolio:
            assert "T010.JK" not in outside

    assert all(candidate == core for candidate in resolved)


def test_feature_cache_rejects_payload_without_semantic_input_identity():
    class FakeBridge(ScannerDatabaseBridge):
        def __init__(self):
            super().__init__(DatabaseSettings(
                enabled=True, mode="SUPABASE_REST",
                supabase_url="https://example.test", supabase_key="secret",
                supabase_key_type="SECRET",
            ))

        def _get_rows(self, table, params):
            if table == "scanner_feature_cache" and "payload" not in str(params.get("select")):
                return [{
                    "ticker": "AAA.JK", "last_bar_date": "2026-08-28",
                    "feature_state": "CURRENT", "source_tier": "IDX",
                    "scanner_version": "scanner-v1",
                    "feature_schema_version": "ALL_ELIGIBLE_LITE_V1",
                    "content_hash": "legacy", "updated_at": "2026-08-28T12:00:00Z",
                }]
            if table == "scanner_feature_cache":
                return [{
                    "ticker": "AAA.JK",
                    "payload": {
                        "ticker": "AAA.JK", "technical_ready": True,
                        "completion_state": "TECHNICAL_READY",
                        "ohlcv_last_bar_date": "2026-08-28",
                        "signal": {"quality_score": 70.0},
                    },
                }]
            return []

    hits, audit = FakeBridge().read_feature_cache(
        ["AAA.JK"], expected_session="2026-08-28", scanner_version="scanner-v1",
    )

    assert hits == {}
    assert "STALE_OR_INCOMPATIBLE" in set(audit["status"])


def test_feature_cache_semantic_hit_and_material_miss_parity():
    technical_fields = {
        "quality_score": 71.0, "entry": 100.0, "stop_loss": 95.0,
        "tp1": 108.0, "tp2": 115.0, "rr1": 1.6, "rr2": 3.0,
        "status": "ENTRY_PLAN_READY", "coverage": 100.0,
    }
    identity = {
        "symbol": "AAA.JK",
        "feature_schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "scanner_version": "scanner-v2",
        "completed_session": "2026-08-28",
        "latest_ohlcv_timestamp": "2026-08-28",
        "bar_count": 300,
        "ohlcv_fingerprint": "ohlcv-hash-v1",
        "corporate_action_series_identity": "ohlcv-hash-v1",
        "indicator_config_version": "config-v1",
        "benchmark_context_identity": "benchmark-hash-v1",
    }
    payload = {
        "ticker": "AAA.JK", "technical_ready": True,
        "completion_state": "TECHNICAL_READY",
        "ohlcv_last_bar_date": "2026-08-28",
        "feature_input_identity": identity,
        "signal": technical_fields,
    }

    class FakeBridge(ScannerDatabaseBridge):
        def __init__(self, *, ohlcv_hash="ohlcv-hash-v1", benchmark_hash="benchmark-hash-v1", local_payload=None):
            super().__init__(DatabaseSettings(
                enabled=True, mode="SUPABASE_REST",
                supabase_url="https://example.test", supabase_key="secret",
                supabase_key_type="SECRET",
            ))
            self.ohlcv_hash = ohlcv_hash
            self.benchmark_hash = benchmark_hash
            self.local_payload = dict(local_payload or payload)

        def _get_rows(self, table, params):
            selected = str(params.get("select"))
            if table == "scanner_feature_cache" and selected == "ticker,payload":
                return [{"ticker": "AAA.JK", "payload": self.local_payload}]
            if table == "scanner_feature_cache":
                return [{
                    "ticker": "AAA.JK", "last_bar_date": "2026-08-28",
                    "feature_state": "CURRENT", "source_tier": "IDX",
                    "scanner_version": "scanner-v2",
                    "feature_schema_version": FEATURE_CACHE_SCHEMA_VERSION,
                    "content_hash": _semantic_hash(self.local_payload),
                    "updated_at": "2026-08-28T12:00:00Z",
                }]
            if table == "ohlcv_daily_cache":
                return [
                    {"ticker": "AAA.JK", "last_bar_date": "2026-08-28", "bar_count": 300, "content_hash": self.ohlcv_hash, "refresh_state": "CURRENT"},
                    {"ticker": "^JKSE", "last_bar_date": "2026-08-28", "bar_count": 300, "content_hash": self.benchmark_hash, "refresh_state": "CURRENT"},
                ]
            return []

    hits, _ = FakeBridge().read_feature_cache(
        ["AAA.JK"], expected_session="2026-08-28", scanner_version="scanner-v2",
        indicator_config_version="config-v1",
    )
    assert hits["AAA.JK"]["signal"] == technical_fields

    material_misses = (
        (FakeBridge(ohlcv_hash="revised-bars"), "config-v1", "2026-08-28"),
        (FakeBridge(benchmark_hash="revised-benchmark"), "config-v1", "2026-08-28"),
        (FakeBridge(), "config-v2", "2026-08-28"),
        (FakeBridge(), "config-v1", "2026-08-29"),
    )
    for bridge, config_version, session in material_misses:
        misses, _ = bridge.read_feature_cache(
            ["AAA.JK"], expected_session=session, scanner_version="scanner-v2",
            indicator_config_version=config_version,
        )
        assert misses == {}

    for changed_field, changed_value in (
        ("symbol", "BBB.JK"),
        ("feature_schema_version", "OLD_SCHEMA"),
        ("completed_session", "2026-08-27"),
        ("latest_ohlcv_timestamp", "2026-08-27"),
        ("bar_count", 299),
        ("ohlcv_fingerprint", "other-series"),
        ("corporate_action_series_identity", "pre-split-series"),
        ("indicator_config_version", "other-config"),
        ("benchmark_context_identity", "other-benchmark"),
    ):
        changed = dict(payload)
        changed["feature_input_identity"] = {**identity, changed_field: changed_value}
        misses, _ = FakeBridge(local_payload=changed).read_feature_cache(
            ["AAA.JK"], expected_session="2026-08-28", scanner_version="scanner-v2",
            indicator_config_version="config-v1",
        )
        assert misses == {}


def test_shortlist_reserves_a_deterministic_exploration_cohort():
    universe = _tickers(10)
    preliminary = pd.DataFrame({
        "ticker": universe,
        "preliminary_multibagger_score": [90, 89, 88, 87, 86, 85, 84, 83, 82, 81],
        "score_coverage_pct": [100, 100, 100, 100, 100, 100, 100, 20, 10, 100],
    })

    selected, audit = build_enrichment_shortlist(
        universe,
        preliminary_focus={"multibagger": preliminary},
        config=ShortlistConfig(max_tickers=5, multibagger_quota=5, core_quota=0, technical_rescue_quota=0),
    )

    assert len(selected) == 5
    assert "enrichment_cohort" in audit.columns
    assert (audit["enrichment_cohort"] == "EXPLORATION").sum() >= 1
    assert set(audit.loc[audit["enrichment_cohort"] == "EXPLORATION", "ticker"]) & {"T007.JK", "T008.JK"}

    final_focus = {"next_leaders_all": pd.DataFrame([
        {"ticker": selected[-1], "ranking_score": 99.0, "rank_eligible": True},
        {"ticker": selected[0], "ranking_score": 98.0, "rank_eligible": True},
        {"ticker": selected[1], "ranking_score": 97.0, "rank_eligible": True},
    ])}
    diagnostics = build_enrichment_recall_diagnostics(audit, final_focus, top_n=3)
    leader = diagnostics.loc[diagnostics["lane"].eq("NEXT_LEADER")].iloc[0]
    assert leader["provisional_shortlist_size"] == 4
    assert leader["exploration_cohort_size"] == 1
    assert leader["promoted_after_enrichment"] == 1
    assert leader["final_top_n_recall_from_provisional_pct"] == 66.67


def test_weighted_final_separates_unknown_stale_failure_zero_and_not_applicable():
    weights = {"observed": 0.5, "other": 0.5}

    missing_score, missing_coverage = _weighted_final(
        {"observed": (80.0, 100.0, "CURRENT"), "other": (np.nan, 0.0, "MISSING")},
        weights,
        min_coverage=0.0,
    )
    stale_score, stale_coverage = _weighted_final(
        {"observed": (80.0, 100.0, "CURRENT"), "other": (100.0, 100.0, "STALE")},
        weights,
        min_coverage=0.0,
    )
    failure_score, failure_coverage = _weighted_final(
        {"observed": (80.0, 100.0, "CURRENT"), "other": (100.0, 100.0, "PROVIDER_FAILURE")},
        weights,
        min_coverage=0.0,
    )
    zero_score, zero_coverage = _weighted_final(
        {"observed": (80.0, 100.0, "CURRENT"), "other": (0.0, 100.0, "CURRENT")},
        weights,
        min_coverage=0.0,
    )
    na_score, na_coverage = _weighted_final(
        {"observed": (80.0, 100.0, "CURRENT"), "other": (np.nan, 0.0, "NOT_APPLICABLE")},
        weights,
        min_coverage=0.0,
    )

    assert (missing_score, missing_coverage) == (80.0, 50.0)
    assert (stale_score, stale_coverage) == (80.0, 50.0)
    assert (failure_score, failure_coverage) == (80.0, 50.0)
    assert (zero_score, zero_coverage) == (40.0, 100.0)
    assert (na_score, na_coverage) == (80.0, 100.0)


def test_existing_json_artifact_path_carries_final_decision_snapshot():
    bridge = ScannerDatabaseBridge(DatabaseSettings())
    snapshot = {
        "snapshot_schema_version": "FINAL_DECISION_SNAPSHOT_V1",
        "snapshot_id": "snapshot-1",
        "scan_id": "scan-1",
        "decisions": [{"ticker": "AAA.JK", "rank": 1, "ranking_score": 77.0}],
    }
    payloads = bridge.build_payloads({
        "scan_id": "scan-1", "scanner_version": "scanner-v1",
        "final_decision_snapshot": snapshot,
        "focus_screens": {},
    })

    assert payloads["scan_job_artifacts"][0]["artifact_type"] == "FINAL_DECISION_SNAPSHOT_V1"
    assert payloads["scan_job_artifacts"][0]["payload"] == snapshot


def test_final_decision_snapshot_live_persisted_reloaded_parity():
    result = {
        "scan_id": "scan-parity", "scanner_version": "scanner-v2",
        "focus_screens": {"next_leaders_all": pd.DataFrame([{
            "ticker": "AAA.JK", "rank": 1, "ranking_score": 77.0,
            "v9_next_leader_score": 76.0, "score_coverage_pct": 82.0,
            "thesis_confidence_pct": 79.0, "ranking_guardrail_state": "CLEAN",
            "real_money_authorization_state": "BLOCKED_MANUAL_CHECK",
            "production_rank_eligible": True, "entry": 100.0, "stop_loss": 95.0,
            "tp1": 110.0, "tp2": 120.0, "rr2": 4.0,
            "fundamental_freshness_state": "CURRENT",
            "fundamental_official_state": "VERIFIED",
        }])},
    }
    live = build_final_decision_snapshot(
        result, universe=["AAA.JK", "BBB.JK"], completed_session="2026-08-28",
    )
    bridge = ScannerDatabaseBridge(DatabaseSettings())
    payloads = bridge.build_payloads({**result, "final_decision_snapshot": live})
    persisted = payloads["scan_job_artifacts"][0]["payload"]
    reloaded = reload_final_decision_snapshot(payloads["scan_job_artifacts"])

    assert live == persisted == reloaded
    assert reloaded["decisions"][0]["entry"] == 100.0
    assert reloaded["decisions"][0]["stop_loss"] == 95.0
    assert reloaded["decisions"][0]["tp1"] == 110.0
    assert reloaded["decisions"][0]["tp2"] == 120.0
    assert reloaded["decisions"][0]["rr"] == 4.0
