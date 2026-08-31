from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evidence_coverage import CoveragePolicy, build_evidence_coverage


def _universe(path: Path) -> list[str]:
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#"):
            continue
        values.extend(part.strip() for part in line.split(",") if part.strip())
    result = list(dict.fromkeys(values))
    if len(result) != 400:
        raise ValueError(f"COVERAGE_UNIVERSE_MUST_BE_EXACTLY_400:{len(result)}")
    return result


def _csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if path else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description="Scoring-independent evidence coverage report")
    parser.add_argument("--universe", default=str(ROOT / "tools/idx_400_production_universe_2026-07.txt"))
    parser.add_argument("--ohlcv", default="")
    parser.add_argument("--fundamentals", default="")
    parser.add_argument("--forward", default="")
    parser.add_argument("--foreign", default="")
    parser.add_argument("--broker", default="")
    parser.add_argument("--as-of", default="")
    parser.add_argument("--broker-required", action="store_true")
    parser.add_argument("--output-dir", default="evidence_coverage_output")
    args = parser.parse_args()
    detail, summary = build_evidence_coverage(
        _universe(Path(args.universe)),
        ohlcv=_csv(args.ohlcv), fundamentals=_csv(args.fundamentals),
        forward=_csv(args.forward), foreign=_csv(args.foreign), broker=_csv(args.broker),
        as_of=args.as_of or None,
        policy=CoveragePolicy(require_broker=args.broker_required),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "coverage_by_ticker.csv", index=False)
    (output / "coverage_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
