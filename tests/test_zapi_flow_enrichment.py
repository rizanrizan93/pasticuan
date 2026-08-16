from __future__ import annotations

import numpy as np
import pandas as pd

import zapi_flow_enrichment as zapi


def _history() -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None).normalize(), periods=20)
    rows = []
    for ticker, direction in (("POS1", 1.0), ("POS2", 0.6), ("NEG1", -0.6), ("NEG2", -1.0)):
        for i, day in enumerate(dates):
            volume = 10_000_000.0
            net = direction * volume * (0.006 + 0.0001 * (i % 3))
            buy = 1_100_000.0 + max(net, 0.0)
            sell = 1_100_000.0 + max(-net, 0.0)
            rows.append(
                {
                    "ticker": ticker,
                    "trade_date": day,
                    "foreign_buy_shares": buy,
                    "foreign_sell_shares": sell,
                    "foreign_net_shares": net,
                    "volume": volume,
                    "source": "SYNTHETIC_ZAPI",
                    "flow_unit": "SHARES",
                }
            )
    return pd.DataFrame(rows)


def test_foreign_flow_score_separates_accumulation_and_distribution() -> None:
    scored = zapi.score_foreign_history(_history())
    pos = scored.set_index("ticker").loc["POS1"]
    neg = scored.set_index("ticker").loc["NEG2"]
    assert float(pos["zapi_foreign_flow_score"]) > float(neg["zapi_foreign_flow_score"])
    assert pos["zapi_foreign_state"] == "NET_ACCUMULATION"
    assert neg["zapi_foreign_state"] == "NET_DISTRIBUTION"
    assert float(pos["zapi_foreign_flow_coverage_pct"]) >= 95.0


def test_super_blend_is_bounded_and_coverage_aware(monkeypatch) -> None:
    features = pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "zapi_foreign_flow_score": 95.0,
                "zapi_foreign_flow_coverage_pct": 100.0,
                "zapi_accumulation_confirmation_score": 90.0,
                "zapi_smart_money_confirmation_score": 92.0,
                "zapi_smc_flow_confirmation_score": 94.0,
                "zapi_foreign_state": "NET_ACCUMULATION",
            }
        ]
    )
    monkeypatch.setattr(zapi, "get_zapi_features", lambda tickers: (features.copy(), {"state": "TEST"}))
    universe = pd.DataFrame(
        [
            {
                "ticker": "TEST.JK",
                "flow_silent_accumulation_score": 60.0,
                "flow_silent_accumulation_confidence": 80.0,
            }
        ]
    )
    out = zapi.enrich_super_universe(universe).iloc[0]
    assert 60.0 < float(out["flow_silent_accumulation_score"]) <= 71.0
    assert np.isclose(float(out["zapi_confirmation_weight_pct"]), 35.0)
    assert 80.0 < float(out["flow_silent_accumulation_confidence"]) <= 90.0
