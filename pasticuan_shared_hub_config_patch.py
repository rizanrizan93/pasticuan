from __future__ import annotations

"""Reuse PASTICUAN's existing backend Supabase credentials for its Shared Hub.

The Shared Evidence Hub is hosted in the same PASTICUAN Supabase project. This
adapter only aliases credentials when the configured operational project ref is
exactly the expected shared-hub ref and the key is backend-safe.
"""

import os
from urllib.parse import urlparse
from typing import Any


PATCH_VERSION = "1.0.0-phase5.6-config"
PASTICUAN_SHARED_HUB_PROJECT_REF = "mbtsvflwszcgdtijdgas"


def _project_ref(url: Any) -> str:
    host = str(urlparse(str(url or "")).hostname or "").lower()
    return host.split(".", 1)[0] if host.endswith(".supabase.co") else ""


def apply_settings(settings: Any) -> str:
    if str(os.getenv("SHARED_EVIDENCE_SUPABASE_URL", "")).strip():
        return "EXPLICIT_SHARED_CONFIG_PRESERVED"
    url = str(getattr(settings, "supabase_url", "") or "").strip().rstrip("/")
    key = str(getattr(settings, "supabase_key", "") or "").strip()
    key_type = str(getattr(settings, "supabase_key_type", "") or "").strip().upper()
    mode = str(getattr(settings, "mode", "") or "").strip().upper()
    if _project_ref(url) != PASTICUAN_SHARED_HUB_PROJECT_REF:
        return "PROJECT_REF_REJECTED"
    if mode != "SUPABASE_REST" or key_type not in {"SECRET", "SERVICE_ROLE"} or not key:
        return "BACKEND_CREDENTIALS_UNAVAILABLE"
    os.environ["SHARED_EVIDENCE_SUPABASE_URL"] = url
    if key_type == "SECRET":
        os.environ["SHARED_EVIDENCE_SUPABASE_SECRET_KEY"] = key
    else:
        os.environ["SHARED_EVIDENCE_SUPABASE_SERVICE_ROLE_KEY"] = key
    return "PASTICUAN_OPERATIONAL_CONFIG_ALIASED"


def install() -> None:
    try:
        from scanner_database import DatabaseSettings
        apply_settings(DatabaseSettings.from_env())
    except Exception:
        # Shared-hub access remains fail-soft; ordinary scanner DB operation is
        # never made dependent on this convenience alias.
        return


__all__ = ["PATCH_VERSION", "PASTICUAN_SHARED_HUB_PROJECT_REF", "apply_settings", "install"]
