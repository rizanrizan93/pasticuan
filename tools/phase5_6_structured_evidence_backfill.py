from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
import json

from shared_company_evidence import SharedCompanyEvidence
from shared_evidence_hub import HubConfig, SupabaseEvidenceBackend
from shared_structured_fundamental_evidence import SharedStructuredFundamentalEvidence
from shared_structured_ownership_evidence import SharedStructuredOwnershipEvidence


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ownership-limit", type=int, default=800)
    p.add_argument("--fundamental-gap-limit", type=int, default=250)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def _ticker(value: object) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def main() -> int:
    args = _args()
    config = HubConfig.from_environment(client_id="PASTICUAN")
    backend = SupabaseEvidenceBackend(config)
    fundamentals = SharedStructuredFundamentalEvidence(
        "PASTICUAN", config=config, backend=backend
    )

    _, bridge_meta = fundamentals.bridge_operational(limit=5000)

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

    fundamental_results = {"attempted": 0, "refreshed": 0, "failed": 0, "rows": 0}
    for code in gap_tickers:
        fundamental_results["attempted"] += 1
        try:
            rows, meta = fundamentals.refresh_pluang(code, observed_at=datetime.now(timezone.utc))
            if rows:
                fundamental_results["refreshed"] += 1
                fundamental_results["rows"] += len(rows)
            else:
                fundamental_results["failed"] += 1
        except Exception:
            fundamental_results["failed"] += 1

    ownership_tickers = tickers[: max(0, args.ownership_limit)]
    today = date.today()

    def one(code: str) -> dict[str, object]:
        company = SharedCompanyEvidence("PASTICUAN")
        owner = SharedStructuredOwnershipEvidence("PASTICUAN")
        try:
            profile_rows, meta = company.get_profile(code, today)
            if profile_rows:
                rows, own_meta = owner.persist_idx_profile(profile_rows[0])
                if rows:
                    return {"ticker": code, "state": "IDX_PROFILE", "rows": len(rows), "api_calls": int(meta.get("api_calls") or 0)}
            rows, own_meta = owner.refresh_pluang(code, observed_on=today)
            if rows:
                return {"ticker": code, "state": "PLUANG_FALLBACK", "rows": len(rows), "api_calls": int(meta.get("api_calls") or 0) + 1}
            return {"ticker": code, "state": str(own_meta.get("state") or "NO_ROWS"), "rows": 0, "api_calls": int(meta.get("api_calls") or 0) + int(own_meta.get("api_calls") or 0)}
        except Exception as exc:
            return {"ticker": code, "state": type(exc).__name__, "rows": 0, "api_calls": 0}

    ownership_results = {"attempted": 0, "idx_profile": 0, "pluang_fallback": 0, "failed": 0, "rows": 0}
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as executor:
        futures = {executor.submit(one, code): code for code in ownership_tickers}
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

    summary = {
        "structured_fundamental_bridge": bridge_meta,
        "pluang_fundamental_gap_fill": fundamental_results,
        "structured_ownership": ownership_results,
        "universe_tickers": len(tickers),
        "ownership_limit": len(ownership_tickers),
        "policy": "FACTS_ONLY_NO_SCORING_OR_GATE_CHANGE",
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
