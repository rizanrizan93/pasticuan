from __future__ import annotations

"""One-shot shape-only diagnostic for the ZAPI IDX market-time reference payload.

This intentionally emits no provider values, labels, raw payloads, headers, URLs with
credentials, or secrets. It exists only to identify the response container contract
needed by the Phase 5.6 factual reference parser.
"""

import json
from typing import Any, Mapping

from shared_company_evidence import REFERENCE_URL, SharedCompanyEvidence


def _shape(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {
            "type": "mapping",
            "keys": sorted(str(key) for key in value.keys()),
        }
    if isinstance(value, list):
        item_types = sorted({type(item).__name__ for item in value})
        mapping_keys = sorted(
            {
                str(key)
                for item in value
                if isinstance(item, Mapping)
                for key in item.keys()
            }
        )
        return {
            "type": "list",
            "count": len(value),
            "item_types": item_types,
            "mapping_keys": mapping_keys,
        }
    return {"type": type(value).__name__}


def main() -> int:
    evidence = SharedCompanyEvidence("PASTICUAN")
    payload = evidence._request(REFERENCE_URL, {"set": "market-time"})
    if not isinstance(payload, Mapping):
        print(json.dumps({"root": _shape(payload)}, sort_keys=True))
        return 0

    data = payload.get("data")
    report = {
        "root": _shape(payload),
        "items": _shape(payload.get("items")),
        "data": _shape(data),
        "data_value": _shape(data.get("value")) if isinstance(data, Mapping) else {"type": "unavailable"},
        "content": _shape(payload.get("content")),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
