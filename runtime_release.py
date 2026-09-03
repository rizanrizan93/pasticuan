"""Keep Streamlit hot-reloads on one release contract."""

from __future__ import annotations

import importlib
import sys
import warnings
from collections.abc import Mapping, Sequence


_LAST_PATCH_STATUS: dict[str, dict[str, str]] = {}


def _try_optional_patch(module_name: str, installer_name: str, *args: object) -> None:
    key = f"{module_name}.{installer_name}"
    try:
        module = importlib.import_module(module_name)
        installer = getattr(module, installer_name, None)
        if not callable(installer):
            _LAST_PATCH_STATUS[key] = {"state": "NOT_CALLABLE", "detail": ""}
            warnings.warn(f"Optional runtime patch {key} is not callable", RuntimeWarning, stacklevel=2)
            return
        installer(*args)
        _LAST_PATCH_STATUS[key] = {"state": "INSTALLED", "detail": ""}
    except Exception as exc:
        detail = f"{type(exc).__name__}: {str(exc)[:240]}"
        _LAST_PATCH_STATUS[key] = {"state": "FAILED", "detail": detail}
        warnings.warn(f"Optional runtime patch {key} failed: {detail}", RuntimeWarning, stacklevel=2)


def runtime_patch_status() -> dict[str, dict[str, str]]:
    return {key: dict(value) for key, value in _LAST_PATCH_STATUS.items()}


def _install_integrity_patch(expected: str) -> None:
    if "pytest" in sys.modules:
        return
    _LAST_PATCH_STATUS.clear()
    _try_optional_patch("zapi_runtime_patch", "install")
    _try_optional_patch("broker_runtime_patch", "install")

    patch = importlib.import_module("runtime_integrity_patch")
    patch.install(expected)
    _LAST_PATCH_STATUS["runtime_integrity_patch.install"] = {"state": "INSTALLED", "detail": ""}

    _try_optional_patch("live_forward_runtime_patch", "install_runtime")
    _try_optional_patch("live_forward_evidence", "install_dashboard_cost_integrity")
    _try_optional_patch("future_fundamental_ui_patch", "install")
    _try_optional_patch("pasticuan_shared_hub_config_patch", "install")
    _try_optional_patch("shared_fundamental_runtime_patch", "install")


def refresh_release_runtime(
    *,
    reload_order: Sequence[str], version_markers: Mapping[str, str],
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


__all__ = ["refresh_release_runtime", "runtime_patch_status"]
