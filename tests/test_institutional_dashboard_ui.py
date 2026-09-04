from __future__ import annotations

import tomllib
from pathlib import Path

from v9_dashboard import V9_DASHBOARD_VERSION, _INSTITUTIONAL_CSS


def test_institutional_dashboard_skin_is_presentation_only_contract() -> None:
    assert V9_DASHBOARD_VERSION == "1.6.0-institutional-ui"
    assert ".v9-card:before" in _INSTITUTIONAL_CSS
    assert ".v9-auth" in _INSTITUTIONAL_CSS
    assert ".v9-gates" in _INSTITUTIONAL_CSS
    assert "@media(max-width:640px)" in _INSTITUTIONAL_CSS
    assert "#071019" in _INSTITUTIONAL_CSS


def test_streamlit_theme_matches_dashboard_shell() -> None:
    config = tomllib.loads(Path(".streamlit/config.toml").read_text(encoding="utf-8"))
    theme = config["theme"]
    sidebar = theme["sidebar"]
    assert theme["base"] == "dark"
    assert theme["backgroundColor"] == "#071019"
    assert theme["primaryColor"] == "#0F9D8A"
    assert theme["showWidgetBorder"] is True
    assert sidebar["backgroundColor"] == "#08131E"
