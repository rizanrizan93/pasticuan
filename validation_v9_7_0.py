from __future__ import annotations

import ast
import importlib
import pathlib
import time

import numpy as np
import pandas as pd

from decision_overlay import apply_methodology_guardrails, inventory_lifecycle_profile
from v9_dashboard import render_dashboard_html, select_top_candidates


ROOT = pathlib.Path(__file__).resolve().parent


def synthetic_frame(seed: int = 1, bars: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.00025, 0.014, bars)
    close = 100.0 * np.cumprod(1.0 + returns)
    spread = np.clip(np.abs(rng.normal(0.025, 0.009, bars)), 0.006, 0.08)
    volume = rng.lognormal(13.7, 0.5, bars)
    return pd.DataFrame({
        "Open": close * (1.0 + rng.normal(0.0, 0.002, bars)),
        "High": close * (1.0 + spread / 2.0),
        "Low": close * (1.0 - spread / 2.0),
        "Close": close,
        "Volume": volume,
    })


def check_compile() -> None:
    for path in ROOT.glob("*.py"):
        compile(path.read_text(), str(path), "exec")


def check_imports() -> None:
    modules = [p.stem for p in ROOT.glob("*.py") if p.name not in {"app.py", pathlib.Path(__file__).name}]
    for module in sorted(modules):
        importlib.import_module(module)


def check_no_duplicate_top_level_symbols() -> None:
    for name in ("scanner.py", "app.py", "simple_focus.py", "decision_overlay.py", "v9_dashboard.py"):
        tree = ast.parse((ROOT / name).read_text())
        symbols = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        assert len(symbols) == len(set(symbols)), f"duplicate top-level symbol in {name}"


def check_inventory_contract() -> None:
    profile = inventory_lifecycle_profile(synthetic_frame(seed=9, bars=800))
    for key in (
        "inventory_multi_horizon_score",
        "inventory_multi_horizon_coverage_pct",
        "distribution_risk_score",
        "inventory_lifecycle",
        "anti_chase_gate",
        "markup_extension_pct",
        "reaccumulation_quality_score",
    ):
        assert key in profile, key
    assert "inventory_score_756d" in profile, "756D horizon must be available with 800 bars"
    assert 0.0 <= float(profile["inventory_multi_horizon_score"]) <= 100.0
    assert 0.0 <= float(profile["distribution_risk_score"]) <= 100.0


def check_guardrails() -> None:
    source = pd.DataFrame([
        {
            "ticker": "CHASE.JK", "status": "BUY_ZONE", "v9_next_leader_score": 88.0,
            "score_coverage_pct": 90.0, "production_gate_pass": True,
            "inventory_lifecycle": "MARKUP", "distribution_risk_score": 20.0,
            "anti_chase_gate": True, "recommended_allocation_idr": 1_000_000.0,
            "recommended_lots": 10, "multibagger_status": "MULTIBAGGER_A_CANDIDATE",
            "research_recommendation_status": "BUY_ZONE", "multibagger_rank_eligible": True,
            "research_eligible": True,
        },
        {
            "ticker": "DIST.JK", "status": "BUY_ZONE", "v9_next_leader_score": 92.0,
            "score_coverage_pct": 93.0, "production_gate_pass": True,
            "inventory_lifecycle": "DISTRIBUTION", "distribution_risk_score": 82.0,
            "anti_chase_gate": False, "recommended_allocation_idr": 1_000_000.0,
            "recommended_lots": 10, "multibagger_status": "MULTIBAGGER_A_CANDIDATE",
            "research_recommendation_status": "BUY_ZONE", "multibagger_rank_eligible": True,
            "research_eligible": True,
        },
    ])
    out = apply_methodology_guardrails(source, model="NEXT_LEADER")
    chase = out.loc[out["ticker"].eq("CHASE.JK")].iloc[0]
    dist = out.loc[out["ticker"].eq("DIST.JK")].iloc[0]
    assert chase["status"] == "WAIT"
    assert int(chase["recommended_lots"]) == 0
    assert chase["decision_overlay_state"] == "V9_WAIT_REACCUMULATION"
    assert dist["status"] == "RESEARCH_ONLY"
    assert not bool(dist["methodology_gate_pass"])
    assert not bool(dist["multibagger_rank_eligible"])


def check_legacy_compatibility() -> None:
    # Older durable jobs can lack the new inventory fields.  Guardrails must fail-open
    # to neutral lifecycle rather than crash or invent a distribution block.
    old = pd.DataFrame([{"ticker": "OLD.JK", "status": "WATCH", "v9_next_leader_score": 70.0}])
    out = apply_methodology_guardrails(old, model="NEXT_LEADER")
    assert len(out) == 1
    assert bool(out.iloc[0]["methodology_gate_pass"])


def check_dashboard() -> None:
    frame = pd.DataFrame([
        {
            "ticker": "AAA.JK", "sector": "ENERGY", "status": "BUY_ZONE", "rank_eligible": True,
            "v9_next_leader_score": 82.0, "final_score": 82.0, "score_coverage_pct": 91.0,
            "methodology_priority": 1, "business_quality_score": 80.0, "future_fundamental_score": 76.0,
            "valuation_mos_score": 72.0, "management_capital_score": 70.0,
            "issuer_macro_alignment_score": 74.0, "narrative_flow_score": 77.0,
            "silent_accumulation_score": 73.0, "technical_readiness_score": 68.0,
            "inventory_multi_horizon_score": 75.0, "inventory_lifecycle": "INVENTORY_COLLECTION",
            "distribution_risk_score": 18.0, "anti_chase_gate": False,
            "reaccumulation_quality_score": 65.0, "accumulation_dominance_pct": 75.0,
            "last_price": 100.0, "entry_low": 96.0, "entry_high": 100.0, "trigger": 102.0,
            "stop_loss": 90.0, "tp1": 115.0, "tp2": 130.0, "rr1": 1.5, "rr2": 2.5,
            "recommended_lots": 5, "selected_reason": "Synthetic validation", "primary_risk": "Synthetic risk",
        }
    ])
    top = select_top_candidates(frame, model="NEXT_LEADER", limit=3)
    html = render_dashboard_html(top, model="NEXT_LEADER", scan_id="validation", as_of="2026-08-08", market_regime="RISK_ON")
    assert "TOP 3" in html
    assert "AAA" in html
    assert "Valuation / MOS" in html
    assert "Inventory" in html


def benchmark_overlay() -> float:
    frames = [synthetic_frame(seed=i + 100, bars=800) for i in range(100)]
    started = time.perf_counter()
    for frame in frames:
        inventory_lifecycle_profile(frame)
    return time.perf_counter() - started


def main() -> None:
    checks = [
        check_compile,
        check_imports,
        check_no_duplicate_top_level_symbols,
        check_inventory_contract,
        check_guardrails,
        check_legacy_compatibility,
        check_dashboard,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    elapsed = benchmark_overlay()
    print(f"BENCH overlay_100x800={elapsed:.3f}s estimated_400={elapsed * 4.0:.3f}s")
    print("VALIDATION_V9_7_0=PASS")


if __name__ == "__main__":
    main()
