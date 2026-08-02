from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from emir_method_engine import build_emir_method_profile
from narrative_engine import build_narrative_intelligence
from scanner import ScanConfig
from scanner_focus import allocate_multibagger_capital


def make_frame(periods: int = 620) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=periods)
    base = np.linspace(100.0, 160.0, periods)
    # Preserve a stable long history, then produce a visible 20-day flow lead.
    base[-21:] = np.linspace(145.0, 165.0, 21)
    volume = np.full(periods, 1_000_000.0)
    volume[-10:] = 1_800_000.0
    return pd.DataFrame(
        {
            "Open": base * 0.995,
            "High": base * 1.015,
            "Low": base * 0.985,
            "Close": base,
            "Volume": volume,
        },
        index=index,
    )


def strong_silent(*, include_broker: bool = True) -> dict[str, object]:
    profile: dict[str, object] = {
        "effective_silent_accumulation_score": 84.0,
        "silent_accumulation_score": 84.0,
        "silent_accumulation_confidence": 92.0,
        "persistent_bid_score": 82.0,
        "accumulation_persistence_score": 86.0,
        "weighted_close_location20": 0.78,
        "absorption_confirmed_days20": 3,
        "churning_support_days20": 1,
        "failed_absorption_days20": 0,
        "distribution_days20": 0,
        "silent_accumulation_state": "SILENT_ACCUMULATION_CONFIRMED",
        "liquidity_bucket": "LIQUID",
        "adtv20_idr": 50_000_000_000,
    }
    if include_broker:
        profile.update(
            {
                "broksum_days": 10,
                "broksum_net_ratio": 0.15,
                "broksum_signal": "ACCUMULATION_PROXY",
            }
        )
    return profile


def event_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "detected_at": frame.index[-1] + timedelta(days=1),
                "materiality_score": 82.0,
                "financial_bridge_score": 76.0,
            }
        ]
    )


class EmirMethodEngineTests(unittest.TestCase):
    def test_early_flow_story_convergence_can_be_production_eligible(self) -> None:
        frame = make_frame()
        result = build_emir_method_profile(
            ticker="TEST.JK",
            frame=frame,
            active_events=event_frame(frame),
            outcomes={"narrative_conversion_resolved_20d": 10},
            fundamental={
                "fundamental_history_years": 3,
                "fundamental_history_quarters": 8,
                "fundamental_source_count": 3,
            },
            silent_profile=strong_silent(include_broker=True),
            narrative_effective_score=78.0,
            narrative_evidence_coverage_pct=88.0,
            narrative_evidence_mode="EVENT_PLUS_STRUCTURED_FINANCIAL",
            alignment_effective_score=72.0,
            alignment_coverage_pct=80.0,
            adoption_stage="EARLY_DISCOVERY",
            crowding_risk_score=18.0,
            hard_block=False,
        )
        self.assertTrue(result["flow_preceded_narrative"])
        self.assertEqual(
            result["smart_money_flow_evidence_mode"],
            "BROKER_SUMMARY_OBSERVED_UNVERIFIED",
        )
        self.assertFalse(result["broker_summary_direct_verified"])
        self.assertIn(
            result["narrative_lifecycle_state"],
            {"FLOW_LED_STORY_CONFIRMED", "EARLY_NARRATIVE_FLOW_CONVERGENCE"},
        )
        self.assertTrue(result["emir_method_production_eligible"])
        self.assertEqual(result["emir_method_state"], "EMIR_FRAMEWORK_READY")
        self.assertGreater(result["emir_method_score"], 70.0)
        self.assertGreater(result["emir_swing_rank_adjustment"], 0.0)
        self.assertGreater(result["emir_position_cap_pct"], 0.0)

    def test_distribution_is_rejected_and_position_cap_zero(self) -> None:
        frame = make_frame()
        silent = strong_silent(include_broker=True)
        silent.update(
            {
                "distribution_days20": 6,
                "failed_absorption_days20": 3,
                "silent_accumulation_state": "DISTRIBUTION_RISK",
                "broksum_net_ratio": -0.20,
                "broksum_signal": "DISTRIBUTION_PROXY",
            }
        )
        result = build_emir_method_profile(
            ticker="DIST.JK",
            frame=frame,
            active_events=event_frame(frame),
            outcomes={"narrative_conversion_resolved_20d": 10},
            fundamental={
                "fundamental_history_years": 3,
                "fundamental_history_quarters": 8,
                "fundamental_source_count": 3,
            },
            silent_profile=silent,
            narrative_effective_score=82.0,
            narrative_evidence_coverage_pct=90.0,
            narrative_evidence_mode="SOURCE_EVENT",
            alignment_effective_score=75.0,
            alignment_coverage_pct=85.0,
            adoption_stage="EXHAUSTION_OR_DISTRIBUTION",
            crowding_risk_score=80.0,
            hard_block=False,
        )
        self.assertEqual(result["emir_method_state"], "EMIR_FRAMEWORK_REJECT")
        self.assertFalse(result["emir_method_production_eligible"])
        self.assertEqual(result["emir_position_cap_pct"], 0.0)
        self.assertIn("DISTRIBUTION_RISK", result["emir_risk_flags"])
        self.assertLessEqual(result["emir_swing_rank_adjustment"], -7.0)

    def test_broker_distribution_alone_is_a_reject_signal(self) -> None:
        frame = make_frame()
        silent = strong_silent(include_broker=True)
        silent.update(
            {
                "broksum_net_ratio": -0.20,
                "broksum_signal": "DISTRIBUTION_PROXY",
            }
        )
        result = build_emir_method_profile(
            ticker="BROKERDIST.JK",
            frame=frame,
            active_events=event_frame(frame),
            outcomes={"narrative_conversion_resolved_20d": 10},
            fundamental={
                "fundamental_history_years": 3,
                "fundamental_history_quarters": 8,
                "fundamental_source_count": 3,
            },
            silent_profile=silent,
            narrative_effective_score=78.0,
            narrative_evidence_coverage_pct=88.0,
            narrative_evidence_mode="SOURCE_EVENT",
            alignment_effective_score=72.0,
            alignment_coverage_pct=80.0,
            adoption_stage="EARLY_DISCOVERY",
            crowding_risk_score=18.0,
            hard_block=False,
        )
        self.assertEqual(result["emir_method_state"], "EMIR_FRAMEWORK_REJECT")
        self.assertIn("DISTRIBUTION_RISK", result["emir_risk_flags"])
        self.assertEqual(result["emir_position_cap_pct"], 0.0)

    def test_ohlcv_proxy_is_never_presented_as_broker_evidence(self) -> None:
        frame = make_frame()
        common = dict(
            ticker="PROXY.JK",
            frame=frame,
            active_events=event_frame(frame),
            outcomes={"narrative_conversion_resolved_20d": 10},
            fundamental={
                "fundamental_history_years": 3,
                "fundamental_history_quarters": 8,
                "fundamental_source_count": 3,
            },
            narrative_effective_score=78.0,
            narrative_evidence_coverage_pct=88.0,
            narrative_evidence_mode="SOURCE_EVENT",
            alignment_effective_score=72.0,
            alignment_coverage_pct=80.0,
            adoption_stage="EARLY_DISCOVERY",
            crowding_risk_score=18.0,
            hard_block=False,
        )
        proxy = build_emir_method_profile(
            **common,
            silent_profile=strong_silent(include_broker=False),
        )
        direct = build_emir_method_profile(
            **common,
            silent_profile=strong_silent(include_broker=True),
        )
        self.assertEqual(proxy["smart_money_flow_evidence_mode"], "OHLCV_PRICE_VOLUME_PROXY_ONLY")
        self.assertTrue(direct["smart_money_flow_evidence_mode"].startswith("BROKER_SUMMARY"))
        self.assertLessEqual(proxy["emir_position_cap_pct"], 10.0)
        self.assertGreater(direct["broker_summary_coverage_pct"], 0.0)

    def test_multibagger_allocation_obeys_emir_position_cap(self) -> None:
        candidate = {
            "ticker": "CAP.JK",
            "multibagger_status": "MULTIBAGGER_A_CANDIDATE",
            "compounding_state": "ACCUMULATE_NOW",
            "growth_score": 20.0,
            "profitability_score": 16.0,
            "earnings_quality_score": 17.0,
            "balance_sheet_score": 11.0,
            "valuation_score": 6.0,
            "momentum_score": 9.0,
            "accumulation_score": 9.0,
            "fundamental_data_grade": "A",
            "fundamental_reliability": "HIGH",
            "fundamental_consensus_score": 92.0,
            "fundamental_history_coverage": 90.0,
            "fundamental_official_verified": True,
            "fundamental_official_reference": True,
            "fundamental_source_count": 3,
            "solvency_coverage": 100.0,
            "fundamental_model": "GENERAL",
            "technical_entry_state": "READY_FOR_STOCKBIT_VERIFY",
            "fundamental_conflicts": "",
            "red_flags": "",
            "entry": 1000.0,
            "last_price": 995.0,
            "multibagger_score": 90.0,
            "emir_method_state": "EMIR_FRAMEWORK_READY",
            "emir_method_production_eligible": True,
            "emir_position_cap_pct": 8.0,
        }
        cfg = ScanConfig().replace(
            multibagger_capital_budget_idr=10_000_000.0,
            multibagger_core_cap_pct=0.35,
        )
        result = allocate_multibagger_capital(pd.DataFrame([candidate]), cfg)
        self.assertTrue(bool(result.loc[0, "allocation_eligible"]))
        self.assertLessEqual(float(result.loc[0, "allocation_cap_pct"]), 8.0)
        self.assertLessEqual(float(result.loc[0, "strategic_target_weight_pct"]), 8.0)

    def test_multibagger_allocation_waits_when_emir_evidence_pending(self) -> None:
        candidate = {
            "ticker": "WAIT.JK",
            "multibagger_status": "MULTIBAGGER_A_CANDIDATE",
            "compounding_state": "ACCUMULATE_NOW",
            "growth_score": 20.0,
            "profitability_score": 16.0,
            "earnings_quality_score": 17.0,
            "balance_sheet_score": 11.0,
            "valuation_score": 6.0,
            "momentum_score": 9.0,
            "accumulation_score": 9.0,
            "fundamental_data_grade": "A",
            "fundamental_reliability": "HIGH",
            "fundamental_consensus_score": 92.0,
            "fundamental_history_coverage": 90.0,
            "fundamental_official_verified": True,
            "fundamental_official_reference": True,
            "fundamental_source_count": 3,
            "solvency_coverage": 100.0,
            "fundamental_model": "GENERAL",
            "technical_entry_state": "READY_FOR_STOCKBIT_VERIFY",
            "fundamental_conflicts": "",
            "red_flags": "",
            "entry": 1000.0,
            "last_price": 995.0,
            "multibagger_score": 90.0,
            "emir_method_state": "EMIR_FRAMEWORK_EVIDENCE_PENDING",
            "emir_method_production_eligible": False,
            "emir_position_cap_pct": 0.0,
        }
        result = allocate_multibagger_capital(
            pd.DataFrame([candidate]),
            ScanConfig().replace(multibagger_capital_budget_idr=10_000_000.0),
        )
        self.assertFalse(bool(result.loc[0, "allocation_eligible"]))
        self.assertEqual(float(result.loc[0, "strategic_target_weight_pct"]), 0.0)

    def test_disabled_engine_keeps_stable_emir_schema(self) -> None:
        config = SimpleNamespace(narrative_enabled=False)
        result = build_narrative_intelligence(
            prepared={"TEST.JK": make_frame(260)},
            scan_config=config,
        )
        profile = result["profiles"].iloc[0]
        self.assertEqual(profile["emir_method_state"], "DISABLED")
        self.assertFalse(bool(profile["emir_method_production_eligible"]))
        self.assertEqual(float(profile["emir_position_cap_pct"]), 0.0)


if __name__ == "__main__":
    unittest.main()
