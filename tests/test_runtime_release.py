import types

import runtime_release


def test_stale_loaded_modules_are_reloaded_in_dependency_order(monkeypatch):
    research = types.SimpleNamespace(SCANNER_VERSION="old")
    core = types.SimpleNamespace(SCANNER_VERSION="old")
    database = types.SimpleNamespace(SCANNER_VERSION="old")
    official_guard = types.SimpleNamespace(SCANNER_VERSION="old")
    focus = types.SimpleNamespace(SIMPLE_FOCUS_VERSION="old")
    downstream = types.SimpleNamespace(ENGINE_VERSION="old")
    fast = types.SimpleNamespace(FAST_SCAN_VERSION="old")
    modules = {
        "research_maintenance": research,
        "scanner": core,
        "scanner_database": database,
        "official_evidence_guard": official_guard,
        "simple_focus": focus,
        "resumable_app_engine": downstream,
        "fast_scan_engine": fast,
    }
    for name, module in modules.items():
        monkeypatch.setitem(runtime_release.sys.modules, name, module)

    calls: list[str] = []
    real_reload = runtime_release.importlib.reload

    def fake_reload(module):
        if getattr(module, "__name__", "") == "release_contract":
            return real_reload(module)
        name = next(name for name, candidate in modules.items() if candidate is module)
        calls.append(name)
        return module

    monkeypatch.setattr(runtime_release.importlib, "reload", fake_reload)
    expected, reloaded = runtime_release.refresh_release_runtime(
        reload_order=tuple(modules),
        version_markers={
            "research_maintenance": "SCANNER_VERSION",
            "scanner": "SCANNER_VERSION",
            "scanner_database": "SCANNER_VERSION",
            "official_evidence_guard": "SCANNER_VERSION",
            "simple_focus": "SIMPLE_FOCUS_VERSION",
            "resumable_app_engine": "ENGINE_VERSION",
            "fast_scan_engine": "FAST_SCAN_VERSION",
        },
    )

    assert expected.startswith("9.8.16-")
    assert calls == list(modules)
    assert reloaded == tuple(calls)
