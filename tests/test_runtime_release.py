import types

import runtime_release


def test_stale_loaded_modules_are_reloaded_in_dependency_order(monkeypatch):
    stale = types.SimpleNamespace(SIMPLE_FOCUS_VERSION="old")
    downstream = types.SimpleNamespace(ENGINE_VERSION="old")
    monkeypatch.setitem(runtime_release.sys.modules, "simple_focus", stale)
    monkeypatch.setitem(runtime_release.sys.modules, "resumable_app_engine", downstream)

    calls: list[str] = []
    real_reload = runtime_release.importlib.reload

    def fake_reload(module):
        if getattr(module, "__name__", "") == "release_contract":
            return real_reload(module)
        name = "simple_focus" if module is stale else "resumable_app_engine"
        calls.append(name)
        return module

    monkeypatch.setattr(runtime_release.importlib, "reload", fake_reload)
    expected, reloaded = runtime_release.refresh_release_runtime(
        reload_order=("simple_focus", "resumable_app_engine"),
        version_markers={
            "simple_focus": "SIMPLE_FOCUS_VERSION",
            "resumable_app_engine": "ENGINE_VERSION",
        },
    )

    assert expected.startswith("9.8.13-")
    assert calls == ["simple_focus", "resumable_app_engine"]
    assert reloaded == tuple(calls)
