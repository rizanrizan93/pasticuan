from __future__ import annotations

"""Bridge current technical decision fields into the legacy snapshot schema.

The production decision frame emits ``strategy`` and ``setup_status`` while the
bounded database serializer still asks for the older audit names
``active_setup`` and ``technical_entry_state``.  The mismatch caused both audit
columns to persist as NULL even though the setup had been calculated.

This patch is persistence-only.  It does not change setup detection, scoring,
ranking, gates, or execution geometry.
"""

from functools import wraps
from typing import Any, Iterable

import pandas as pd


PATCH_VERSION = "1.0.0-current-technical-persistence-aliases"


def _with_current_aliases(frame: pd.DataFrame | None, columns: Iterable[str]) -> pd.DataFrame | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame
    requested = set(columns)
    if "active_setup" not in requested and "technical_entry_state" not in requested:
        return frame

    local = frame.copy()
    if "active_setup" in requested and "active_setup" not in local.columns:
        for source in ("strategy", "setup"):
            if source in local.columns:
                local["active_setup"] = local[source]
                break
    if "technical_entry_state" in requested and "technical_entry_state" not in local.columns:
        for source in ("setup_status", "status", "decision_state"):
            if source in local.columns:
                local["technical_entry_state"] = local[source]
                break
    return local


def install() -> dict[str, str]:
    import scanner_database

    current = scanner_database._frame_records
    if bool(getattr(current, "_pasticuan_technical_semantics_patch", False)):
        return {"patch_version": PATCH_VERSION, "state": "ALREADY_INSTALLED"}

    @wraps(current)
    def wrapped(frame: pd.DataFrame | None, columns: Iterable[str]) -> list[dict[str, Any]]:
        requested = tuple(columns)
        return current(_with_current_aliases(frame, requested), requested)

    setattr(wrapped, "_pasticuan_technical_semantics_patch", True)
    setattr(wrapped, "_pasticuan_technical_semantics_patch_version", PATCH_VERSION)
    scanner_database._frame_records = wrapped
    return {"patch_version": PATCH_VERSION, "state": "INSTALLED"}


__all__ = ["PATCH_VERSION", "install"]
