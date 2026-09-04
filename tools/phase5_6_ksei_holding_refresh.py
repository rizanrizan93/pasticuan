from __future__ import annotations

import json

from shared_ksei_holding_composition import SharedKseiHoldingComposition


def main() -> int:
    producer = SharedKseiHoldingComposition("PASTICUAN")
    result = producer.refresh()
    print(json.dumps(result, sort_keys=True, default=str))
    state = str(result.get("state") or "")
    # Network/publication absence is observable but fail-soft.  Persistence
    # partial/error states must fail the producer proof.
    if state in {"REFRESHED", "OFFICIAL_ARCHIVE_NOT_FOUND", "ENVIRONMENT_BLOCKED"}:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
