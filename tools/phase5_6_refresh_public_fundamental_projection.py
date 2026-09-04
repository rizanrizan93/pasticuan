from __future__ import annotations

import json

from shared_evidence_hub import HubConfig, SupabaseEvidenceBackend
from shared_public_fundamental_projection import refresh_public_fundamental_projection


def main() -> int:
    config = HubConfig.from_environment(client_id="PASTICUAN_PUBLIC_PROJECTION")
    if not config.ready:
        raise SystemExit("SHARED_EVIDENCE_SUPABASE credentials are required")
    backend = SupabaseEvidenceBackend(config)
    meta = refresh_public_fundamental_projection(backend)
    print(json.dumps(meta, sort_keys=True))
    return 0 if meta.get("state") == "REFRESHED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
