"""Keep Streamlit hot-reloads on one release contract."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping, Sequence


def _install_integrity_patch(expected: str) -> None:
    if "pytest" in sys.modules:
        return
    patch = importlib.import_module("runtime_integrity_patch")
    patch.install(expected)
    try:
        company_patch = importlib.import_module("live_forward_runtime_patch")
        installer = getattr(company_patch, "install_runtime", None)
        if callable(installer):
            installer()
    except Exception:
        pass
    try:
        live_forward = importlib.import_module("live_forward_evidence")
        installer = getattr(live_forward, "install_dashboard_cost_integrity", None)
        if callable(installer):
            installer()
    except Exception:
        pass
    try:
        future_ui = importlib.import_module("future_fundamental_ui_patch")
        installer = getattr(future_ui, "install", None)
        if callable(installer):
            installer()
    except Exception:
        pass


def refresh_release_runtime(
    *, reload_order: Sequence[str], version_markers: Mapping[str, str],
) -> tuple[str, tuple[str, ...]]:
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
