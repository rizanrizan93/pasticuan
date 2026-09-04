from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time

from shared_evidence_hub import HubConfig, SupabaseEvidenceBackend
from shared_yahoo_ownership_evidence import YahooOwnershipEvidence


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=706)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pause-seconds", type=float, default=0.20)
    return parser.parse_args()


def _ticker(value: object) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def main() -> int:
    args = _args()
    config = HubConfig.from_environment(client_id="PASTICUAN_YAHOO_OWNERSHIP")
    if not config.ready:
        raise SystemExit("SHARED_EVIDENCE_SUPABASE credentials are required")
    backend = SupabaseEvidenceBackend(config)
    source_rows = backend.read_rows(
        "latest_fundamental_snapshots", {}, select="ticker", limit=5000
    )
    tickers = sorted({_ticker(row.get("ticker")) for row in source_rows if _ticker(row.get("ticker"))})
    tickers = tickers[: max(0, int(args.limit))]

    states: Counter[str] = Counter()
    summary = {
        "universe_candidates": len(tickers),
        "attempted": 0,
        "refreshed": 0,
        "failed": 0,
        "concentration_rows": 0,
        "named_rows": 0,
        "institutional_rows": 0,
        "mutual_fund_rows": 0,
        "failure_samples": [],
        "policy": "FACTS_ONLY_NO_FREE_FLOAT_OR_KSEI_INFERENCE",
    }

    def one(code: str) -> dict[str, object]:
        try:
            result = YahooOwnershipEvidence(config=config, backend=backend).refresh(code)
            time.sleep(max(0.0, float(args.pause_seconds)))
            return result
        except Exception as exc:
            time.sleep(max(0.0, float(args.pause_seconds)))
            return {"ticker": code, "state": type(exc).__name__, "error": str(exc)[:180], "rows": 0}

    with ThreadPoolExecutor(max_workers=max(1, min(int(args.workers), 3))) as executor:
        futures = {executor.submit(one, code): code for code in tickers}
        for future in as_completed(futures):
            result = future.result()
            summary["attempted"] += 1
            state = str(result.get("state") or "UNKNOWN")
            states[state] += 1
            if state == "REFRESHED":
                summary["refreshed"] += 1
                summary["concentration_rows"] += int(result.get("concentration_rows") or 0)
                summary["named_rows"] += int(result.get("named_rows") or 0)
                summary["institutional_rows"] += int(result.get("institutional_rows") or 0)
                summary["mutual_fund_rows"] += int(result.get("mutual_fund_rows") or 0)
            else:
                summary["failed"] += 1
                samples = summary["failure_samples"]
                if isinstance(samples, list) and len(samples) < 20:
                    samples.append({
                        "ticker": result.get("ticker"),
                        "state": state,
                        "error": result.get("error"),
                    })

    summary["states"] = dict(states)
    summary["coverage_pct"] = round(100.0 * summary["refreshed"] / max(1, summary["attempted"]), 2)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["refreshed"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
