from __future__ import annotations

from pathlib import Path

import scanner


def test_idx_flow_remote_ohlcv_seed_loader_is_removed():
    assert not hasattr(scanner, "_load_canonical_remote_ohlcv_seed")


def test_ohlcv_runtime_has_no_cross_scanner_remote_seed():
    source = Path(scanner.__file__).read_text(encoding="utf-8").lower()
    assert "idx_400_ohlcv_1y.csv.gz" not in source
    assert "rizanrizan93/idx-flow-scanner" not in source
