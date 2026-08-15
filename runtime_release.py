"""Keep Streamlit hot-reloads on one release contract.

Streamlit re-executes ``app.py`` in a long-lived interpreter. Normal Python
imports can therefore leave already-imported scanner modules on the previous
release while the UI itself shows the new release. This helper reloads the
small dependency chain only when a release marker proves that it is stale and
then installs release-local integrity hooks.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping, Sequence


def _install_integrity_patch(expected: str) -> None:
    # Unit tests exercise evidence_governance directly. Avoid global monkey
    # patching inside pytest; the independent Streamlit smoke launches a clean
    # interpreter and validates the production hook.
    if "pytest" in sys.modules:
        return
    patch = importlib.import_module("runtime_integrity_patch")
    patch.install(expected)
    # Dashboard modules can be reloaded immediately before the integrity patch;
    # reinstall the presentation-only placement hook on every Streamlit rerun.
    try:
        live_forward = importlib.import_module("live_forward_evidence")
        installer = getattr(live_forward, "install_dashboard_cost_integrity", None)
        if callable(installer):
            installer()
    except Exception:
        pass


def refresh_release_runtime(
    *,
    reload_order: Sequence[str],
    version_markers: Mapping[str, str],
) -> tuple[str, tuple[str, ...]]:
    """Return the on-disk release and reload any stale loaded dependency chain."""
    importlib.invalidate_caches()
    contract = importlib.import_module("release_contract")
    contract = importlib.reload(contract)
    expected = str(contract.SCANNER_RELEASE_VERSION)

    stale = any(
        module_name in sys.modules
        and str(getattr(sys.modules[module_name], attribute, "")) != expected
        for module_name, attribute in version_markers.items()
    )
    reloaded: list[str] = []
    if stale:
        for module_name in reload_order:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            importlib.reload(module)
            reloaded.append(module_name)

    _install_integrity_patch(expected)
    return expected, tuple(reloaded)


__all__ = ["refresh_release_runtime"]
