from __future__ import annotations

import json

from shared_evidence_hub import HubConfig, SupabaseEvidenceBackend
from shared_public_ownership_projection import refresh_public_ownership_projection


def main() -> int:
    config = HubConfig.from_environment(client_id="PASTICUAN_PUBLIC_OWNERSHIP_PROJECTION")
    if not config.ready:
        raise SystemExit("SHARED_EVIDENCE_SUPABASE credentials are required")
    result = refresh_public_ownership_projection(SupabaseEvidenceBackend(config))
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("state") == "REFRESHED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
