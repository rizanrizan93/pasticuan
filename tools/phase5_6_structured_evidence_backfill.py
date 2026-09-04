from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import json

from shared_company_evidence import SharedCompanyEvidence
from shared_evidence_hub import HubConfig, SupabaseEvidenceBackend
from shared_fundamental_rate_limit_patch import install as install_rate_limit_patch
from shared_fundamental_runtime import SharedFundamentalRuntime
from shared_public_fundamental_projection import refresh_public_fundamental_projection
from shared_structured_fundamental_evidence import SharedStructuredFundamentalEvidence
from shared_structured_ownership_evidence import SharedStructuredOwnershipEvidence


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ownership-limit", type=int, default=800)
    p.add_argument("--fundamental-gap-limit", type=int, default=250)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--skip-bridges", action="store_true")
    p.add_argument("--skip-public-projection-refresh", action="store_true")
    return p.parse_args()


def _ticker(value: object) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def main() -> int:
    args = _args()
    install_rate_limit_patch()
    config = HubConfig.from_environment(client_id="PASTICUAN")
    backend = SupabaseEvidenceBackend(config)
    exact_fundamentals = SharedStructuredFundamentalEvidence("PASTICUAN", config=config, backend=backend)
    structured = SharedFundamentalRuntime("PASTICUAN", config=config, backend=backend)

    if args.skip_bridges:
        exact_bridge_meta = {"state": "SKIPPED_FOR_FOCUSED_PROVIDER_PROOF", "rows": 0}
        bridge_meta = {"state": "SKIPPED_FOR_FOCUSED_PROVIDER_PROOF", "rows": 0}
    else:
        _, exact_bridge_meta = exact_fundamentals.bridge_operational_financial_facts()
        _, bridge_meta = structured.bridge_operational_snapshots(limit=5000)

    source_rows = backend.read_rows(
        "latest_fundamental_snapshots",
        {},
        select=(
            "ticker,fundamental_source_families,revenue_growth,earnings_growth,roe,"
            "net_margin,operating_cash_flow,debt_equity,as_of"
        ),
        limit=5000,
    )
    tickers = sorted({_ticker(row.get("ticker")) for row in source_rows if _ticker(row.get("ticker"))})
    gap_tickers: list[str] = []
    for row in source_rows:
        code = _ticker(row.get("ticker"))
        if not code:
            continue
        keys = ("revenue_growth", "earnings_growth", "roe", "net_margin", "operating_cash_flow", "debt_equity")
        present = sum(row.get(key) not in (None, "") for key in keys)
        if not str(row.get("fundamental_source_families") or "").strip() or present < 4:
            gap_tickers.append(code)
    gap_tickers = list(dict.fromkeys(gap_tickers))[: max(0, args.fundamental_gap_limit)]

    fundamental_results: dict[str, object] = {
        "candidate_gaps": len(gap_tickers),
        "attempted": 0,
        "refreshed": 0,
        "failed": 0,
        "deferred": 0,
        "rows": 0,
        "provider_success": {},
        "failure_states": {},
        "failure_samples": [],
        "rate_limit_state": "NONE",
    }
    provider_success: Counter[str] = Counter()
    failure_states: Counter[str] = Counter()

    # Fundamental refresh is deliberately sequential. Each ticker may require
    # three Pluang calls or four Yahoo calls, and all ZAPI endpoints share one
    # account-level rate window. shared_fundamental_rate_limit_patch adds an
    # additional inter-request pace and prevents same-window fallback on 429.
    for code in gap_tickers:
        fundamental_results["attempted"] = int(fundamental_results["attempted"]) + 1
        try:
            rows, meta = structured.refresh_structured(code)
            if rows:
                fundamental_results["refreshed"] = int(fundamental_results["refreshed"]) + 1
                fundamental_results["rows"] = int(fundamental_results["rows"]) + len(rows)
                provider_success[str(meta.get("provider") or "UNKNOWN")] += 1
            else:
                fundamental_results["failed"] = int(fundamental_results["failed"]) + 1
                attempts = meta.get("attempts") if isinstance(meta.get("attempts"), list) else []
                for attempt in attempts:
                    if isinstance(attempt, dict):
                        failure_states[f"{attempt.get('provider','UNKNOWN')}:{attempt.get('state','UNKNOWN')}"] += 1
                samples = fundamental_results["failure_samples"]
                if isinstance(samples, list) and len(samples) < 12:
                    samples.append({"ticker": code, "state": meta.get("state"), "attempts": attempts, "rate_limit": meta.get("rate_limit")})
                state = str(meta.get("state") or "")
                if state.startswith("ZAPI_RATE_LIMIT_"):
                    fundamental_results["rate_limit_state"] = state
                    break
        except Exception as exc:
            fundamental_results["failed"] = int(fundamental_results["failed"]) + 1
            failure_states[f"UNCAUGHT:{type(exc).__name__}"] += 1
            samples = fundamental_results["failure_samples"]
            if isinstance(samples, list) and len(samples) < 12:
                samples.append({"ticker": code, "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
    fundamental_results["deferred"] = max(0, len(gap_tickers) - int(fundamental_results["attempted"]))
    fundamental_results["provider_success"] = dict(provider_success)
    fundamental_results["failure_states"] = dict(failure_states)

    ownership_tickers = tickers[: max(0, args.ownership_limit)]
    today = date.today()

    def one_ownership(code: str) -> dict[str, object]:
        company = SharedCompanyEvidence("PASTICUAN")
        owner = SharedStructuredOwnershipEvidence("PASTICUAN")
        try:
            profile_rows, meta = company.get_profile(code, today)
            if profile_rows:
                rows, _ = owner.persist_idx_profile(profile_rows[0])
                if rows:
                    return {"ticker": code, "state": "IDX_PROFILE", "rows": len(rows), "api_calls": int(meta.get("api_calls") or 0)}
            rows, own_meta = owner.refresh_pluang(code, observed_on=today)
            if rows:
                return {"ticker": code, "state": "PLUANG_FALLBACK", "rows": len(rows), "api_calls": int(meta.get("api_calls") or 0) + 1}
            return {"ticker": code, "state": str(own_meta.get("state") or "NO_ROWS"), "rows": 0, "api_calls": int(meta.get("api_calls") or 0) + int(own_meta.get("api_calls") or 0)}
        except Exception as exc:
            return {"ticker": code, "state": type(exc).__name__, "rows": 0, "api_calls": 0}

    ownership_results = {"attempted": 0, "idx_profile": 0, "pluang_fallback": 0, "failed": 0, "rows": 0}
    if ownership_tickers:
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 2))) as executor:
            futures = {executor.submit(one_ownership, code): code for code in ownership_tickers}
            for future in as_completed(futures):
                result = future.result()
                ownership_results["attempted"] += 1
                ownership_results["rows"] += int(result.get("rows") or 0)
                state = str(result.get("state") or "")
                if state == "IDX_PROFILE":
                    ownership_results["idx_profile"] += 1
                elif state == "PLUANG_FALLBACK":
                    ownership_results["pluang_fallback"] += 1
                else:
                    ownership_results["failed"] += 1

    projection_meta = (
        {"state": "SKIPPED", "projection_rows": 0, "persisted_rows": 0}
        if args.skip_public_projection_refresh
        else refresh_public_fundamental_projection(backend)
    )

    summary = {
        "exact_operational_financial_bridge": exact_bridge_meta,
        "structured_fundamental_bridge": bridge_meta,
        "structured_provider_gap_fill": fundamental_results,
        "structured_ownership": ownership_results,
        "public_fundamental_projection": projection_meta,
        "universe_tickers": len(tickers),
        "ownership_limit": len(ownership_tickers),
        "policy": "FACTS_ONLY_NO_SCORING_OR_GATE_CHANGE",
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
