from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evidence_independence_audit import audit_evidence_independence


def _universe(path: str) -> list[str] | None:
    if not path:
        return None
    values: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#"):
            continue
        values.extend(part.strip() for part in line.split(",") if part.strip())
    return list(dict.fromkeys(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Score-independent evidence overlap audit")
    parser.add_argument("--detail", required=True, help="Ticker-level factual availability CSV")
    parser.add_argument("--universe", default="", help="Optional newline/comma-separated ticker universe")
    parser.add_argument("--require-universe-size", type=int, default=0)
    parser.add_argument("--correlation-threshold", type=float, default=0.90)
    parser.add_argument("--minimum-paired-rows", type=int, default=20)
    parser.add_argument("--output-dir", default="evidence_independence_output")
    args = parser.parse_args()
    universe = _universe(args.universe)
    if args.require_universe_size and len(universe or []) != args.require_universe_size:
        raise ValueError(f"AUDIT_UNIVERSE_SIZE_MISMATCH:{len(universe or [])}")
    matrix, summary = audit_evidence_independence(
        pd.read_csv(args.detail), universe=universe,
        duplicate_correlation_threshold=args.correlation_threshold,
        minimum_paired_rows=args.minimum_paired_rows,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output / "component_matrix.csv", index=False)
    (output / "independence_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
