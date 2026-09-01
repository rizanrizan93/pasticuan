from __future__ import annotations

"""Executable witness index for every Phase 5.6 Task 38 adversarial case."""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


CONTRACT_VERSION = "1.0.0-phase5.6-task38"


@dataclass(frozen=True)
class Witness:
    test_file: str
    test_name: str
    alternative_names: tuple[str, ...] = ()


def _w(test_file: str, test_name: str, *alternative_names: str) -> Witness:
    return Witness(test_file, test_name, alternative_names)


ADVERSARIAL_WITNESSES: Mapping[str, Mapping[str, Witness]] = {
    "WORKFLOW": {
        "first_untracked_cache": _w("test_workflow_cache_detection.py", "test_first_untracked_cache_is_detected"),
        "tracked_cache_modification": _w("test_workflow_cache_detection.py", "test_tracked_cache_modification_is_detected"),
        "unchanged_cache": _w("test_workflow_cache_detection.py", "test_unchanged_cache_is_not_detected"),
        "unrelated_untracked_isolation": _w("test_workflow_cache_detection.py", "test_unrelated_untracked_and_runtime_artifacts_are_ignored"),
    },
    "SHARED_CACHE": {
        "cache_hit_avoids_provider_call": _w("test_shared_evidence_hub.py", "test_cache_hit_avoids_provider_call"),
        "stale_cache_triggers_refresh": _w("test_shared_evidence_hub.py", "test_stale_cache_triggers_one_refresh"),
        "second_scanner_reuses_first": _w("test_shared_evidence_hub.py", "test_second_scanner_reuses_first_scanner_facts"),
        "concurrent_duplicate_prevention": _w("test_shared_evidence_hub.py", "test_concurrent_refresh_allows_only_one_provider_call"),
        "expired_lease_recovery": _w("test_shared_evidence_hub.py", "test_expired_refresh_lease_is_recovered"),
    },
    "ZAPI": {
        "missing_key": _w("test_shared_stock_summary_evidence.py", "test_missing_key_on_cache_miss_makes_no_http_request"),
        "http_401": _w("test_shared_stock_summary_evidence.py", "test_http_and_quota_failures_are_explicit"),
        "http_403": _w("test_shared_stock_summary_evidence.py", "test_http_and_quota_failures_are_explicit"),
        "http_404": _w("test_shared_stock_summary_evidence.py", "test_http_and_quota_failures_are_explicit"),
        "http_429": _w("test_shared_stock_summary_evidence.py", "test_http_and_quota_failures_are_explicit"),
        "timeout": _w("test_shared_stock_summary_evidence.py", "test_network_failures_are_explicit"),
        "empty_success_payload": _w("test_shared_stock_summary_evidence.py", "test_empty_malformed_and_wrong_date_payloads_fail_closed"),
        "malformed_payload": _w("test_shared_stock_summary_evidence.py", "test_empty_malformed_and_wrong_date_payloads_fail_closed"),
        "stale_evidence": _w("test_shared_stock_summary_evidence.py", "test_stale_validation_state_refreshes_once"),
        "quota_exhaustion": _w("test_shared_stock_summary_evidence.py", "test_http_and_quota_failures_are_explicit"),
    },
    "STOCK_SUMMARY": {
        "bulk_response": _w("test_shared_stock_summary_evidence.py", "test_bulk_request_persists_filtered_universe_in_one_call"),
        "pagination_or_limit": _w("test_shared_stock_summary_evidence.py", "test_bulk_request_persists_filtered_universe_in_one_call"),
        "duplicate_ticker_date": _w("test_shared_stock_summary_evidence.py", "test_upsert_readback_prevents_duplicate_daily_rows"),
        "malformed_rows": _w("test_shared_stock_summary_evidence.py", "test_validation_rejects_negative_fact_and_insufficient_breadth"),
        "missing_rows": _w("test_shared_stock_summary_evidence.py", "test_validation_rejects_negative_fact_and_insufficient_breadth"),
        "shared_reuse": _w("test_shared_stock_summary_evidence.py", "test_second_scanner_reuses_same_daily_key"),
    },
    "OWNERSHIP": {
        "new_holder": _w("test_shared_ownership_evidence.py", "test_comparable_snapshots_derive_increase_new_and_exit_only"),
        "increased_holding": _w("test_shared_ownership_evidence.py", "test_comparable_snapshots_derive_increase_new_and_exit_only"),
        "reduced_holding": _w("test_phase5_6_adversarial_contract.py", "test_ownership_reduction_threshold_and_no_change_states"),
        "threshold_crossing": _w("test_phase5_6_adversarial_contract.py", "test_ownership_reduction_threshold_and_no_change_states"),
        "holder_exit": _w("test_shared_ownership_evidence.py", "test_comparable_snapshots_derive_increase_new_and_exit_only"),
        "no_change": _w("test_phase5_6_adversarial_contract.py", "test_ownership_reduction_threshold_and_no_change_states"),
        "invalid_or_missing_identity": _w("test_shared_ownership_evidence.py", "test_parser_rejects_nonofficial_source_duplicate_and_empty_workbook"),
        "incomparable_snapshot": _w("test_shared_ownership_evidence.py", "test_noncomparable_or_first_snapshot_never_invents_changes"),
        "reported_holder_semantics": _w("test_shared_ownership_evidence.py", "test_factual_rows_do_not_claim_beneficial_owner_broker_or_bandar"),
    },
    "FINANCIAL": {
        "issuer_mismatch": _w("test_shared_financial_evidence.py", "test_parser_fails_closed_for_wrong_issuer_period_and_malformed_xml"),
        "wrong_issuer_context": _w("test_shared_financial_evidence.py", "test_parser_fails_closed_for_wrong_issuer_period_and_malformed_xml"),
        "wrong_reporting_period": _w("test_shared_financial_evidence.py", "test_parser_fails_closed_for_wrong_issuer_period_and_malformed_xml"),
        "ytd_quarter_semantics": _w("test_shared_financial_evidence.py", "test_period_contract_is_deterministic"),
        "source_verification": _w("test_shared_financial_evidence.py", "test_attachment_requires_official_final_url_and_xbrl_content_type"),
        "ocf_capex_compatibility": _w("test_evidence_coverage.py", "test_fcf_requires_period_compatible_ocf_and_capex"),
        "malformed_xbrl": _w("test_shared_financial_evidence.py", "test_parser_fails_closed_for_wrong_issuer_period_and_malformed_xml"),
        "missing_identity": _w("test_shared_financial_evidence.py", "test_parser_fails_closed_for_wrong_issuer_period_and_malformed_xml"),
        "report_discovery_deduplication": _w("test_shared_financial_evidence.py", "test_pipeline_discovers_one_period_downloads_once_and_persists_reports_before_facts"),
    },
    "ANNOUNCEMENTS": {
        "event_deduplication": _w("test_shared_announcement_evidence.py", "test_deduplication_is_deterministic_and_conflicts_fail_closed"),
        "source_document_identity": _w("test_shared_announcement_evidence.py", "test_validation_rejects_document_confirmation_without_document_hash"),
        "publication_event_date_distinction": _w("test_shared_announcement_evidence.py", "test_explicit_event_date_is_separate_from_publication_and_created_dates"),
        "title_only_insufficient": _w("test_shared_announcement_evidence.py", "test_announcement_metadata_never_confirms_title_only_material_event"),
        "point_in_time_ordering": _w("test_evidence_coverage.py", "test_future_published_catalyst_is_not_visible"),
    },
    "CAPITAL_ACTIONS": {
        "share_count_delta": _w("test_shared_capital_action_evidence.py", "test_pre_and_post_are_explicit_and_delta_is_arithmetic"),
        "rights": _w("test_shared_capital_action_evidence.py", "test_rights_ratio_is_preserved_without_inventing_share_counts"),
        "warrants_or_conversion": _w("test_shared_capital_action_evidence.py", "test_additional_listing_uses_explicit_additional_shares"),
        "split_reverse_split": _w("test_shared_capital_action_evidence.py", "test_reverse_split_requires_explicit_action_not_title"),
        "duplicate_event": _w("test_shared_capital_action_evidence.py", "test_deduplication_is_deterministic_and_provider_id_conflicts_fail"),
    },
    "PARTICIPANT": {
        "exact_http_reason": _w("test_idx_trade_detail_discovery.py", "test_http_reason_is_preserved_for_every_candidate"),
        "parse_failure": _w("test_shared_participant_evidence.py", "test_parse_failure_is_not_collapsed_to_url_not_found"),
        "valid_aggregation": _w("test_public_idx_broker_flow.py", "test_aggregate_trade_detail_reconstructs_participant_flow", "test_emir_owned_provider_reconstructs_participant_flow"),
        "participant_not_beneficial_owner": _w("test_shared_participant_evidence.py", "test_one_official_file_serves_both_scanners"),
        "no_broker_direct_escalation": _w("test_shared_participant_evidence.py", "test_one_official_file_serves_both_scanners"),
    },
    "TEMPORAL": {
        "idx_holiday": _w("test_shared_evidence_contracts.py", "test_trade_date_requires_real_completed_idx_session"),
        "weekend": _w("test_phase5_batch9_shared_p2.py", "test_idx_calendar_weekend_holiday_timezone_and_unknown_coverage", "test_calendar_and_timezone_contract"),
        "unfinished_session": _w("test_shared_evidence_contracts.py", "test_trade_date_requires_real_completed_idx_session"),
        "future_publication": _w("test_shared_evidence_contracts.py", "test_report_date_never_substitutes_for_publication_date"),
    },
    "DB": {
        "insert": _w("test_shared_evidence_validation.py", "test_bounded_cohort_roundtrip_reuses_one_fetch"),
        "upsert_idempotency": _w("test_shared_stock_summary_evidence.py", "test_upsert_readback_prevents_duplicate_daily_rows"),
        "readback": _w("test_shared_evidence_validation.py", "test_bounded_cohort_roundtrip_reuses_one_fetch"),
        "duplicate_prevention": _w("test_shared_evidence_schema_contract.py", "test_every_table_has_expected_deterministic_primary_identity"),
        "rls_access_control": _w("test_shared_evidence_schema_contract.py", "test_rls_grants_and_functions_follow_least_privilege"),
        "persist_failure": _w("test_shared_evidence_hub.py", "test_partial_persist_is_classified_persist_failure"),
        "readback_failure": _w("test_shared_evidence_hub.py", "test_missing_readback_is_classified_readback_failure"),
    },
    "SECURITY": {
        "no_secret_output": _w("test_idx_trade_detail_discovery.py", "test_diagnostics_never_contain_request_headers_or_credentials"),
        "no_idx_flow_backend": _w("test_shared_evidence_hub.py", "test_configuration_never_falls_back_to_idx_flow_or_exposes_secret"),
    },
}


def audit_adversarial_witnesses(test_root: Path) -> dict[str, object]:
    root = Path(test_root)
    missing_files: list[str] = []
    missing_tests: list[str] = []
    parsed: dict[str, set[str]] = {}
    for family, cases in ADVERSARIAL_WITNESSES.items():
        if not cases:
            raise ValueError(f"ADVERSARIAL_FAMILY_EMPTY:{family}")
        for case, witness in cases.items():
            path = root / witness.test_file
            if not path.is_file():
                missing_files.append(witness.test_file)
                continue
            if witness.test_file not in parsed:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                parsed[witness.test_file] = {
                    node.name for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
                }
            names = (witness.test_name, *witness.alternative_names)
            if not any(name in parsed[witness.test_file] for name in names):
                missing_tests.append(f"{family}.{case}:{witness.test_file}::{witness.test_name}")
    if missing_files:
        raise ValueError("ADVERSARIAL_WITNESS_FILE_MISSING:" + ",".join(sorted(set(missing_files))))
    if missing_tests:
        raise ValueError("ADVERSARIAL_WITNESS_TEST_MISSING:" + ",".join(sorted(missing_tests)))
    return {
        "contract_version": CONTRACT_VERSION,
        "families": len(ADVERSARIAL_WITNESSES),
        "cases": sum(len(cases) for cases in ADVERSARIAL_WITNESSES.values()),
        "witness_files": sorted(parsed),
        "fixture_only": True,
        "status": "COMPLETE",
    }


__all__ = ["ADVERSARIAL_WITNESSES", "CONTRACT_VERSION", "Witness", "audit_adversarial_witnesses"]
