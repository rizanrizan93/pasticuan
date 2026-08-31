from __future__ import annotations

import numpy as np
import pandas as pd

import zapi_flow_enrichment as zapi


def _history() -> pd.DataFrame:
    dates = list(reversed(zapi._expected_idx_sessions(count=20)))
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


def test_foreign_coverage_uses_idx_sessions_and_does_not_fill_gaps() -> None:
    as_of = pd.Timestamp("2026-08-31 17:00", tz="Asia/Jakarta")
    expected = list(reversed(zapi._expected_idx_sessions(as_of, 20)))
    rows = [
        {"ticker": "TEST", "trade_date": day, "foreign_net_shares": 1.0,
         "foreign_buy_shares": 2.0, "foreign_sell_shares": 1.0, "volume": 10.0,
         "source": "ZAPI_IDX_FOREIGN_FLOW"}
        for day in expected[1:]
    ]
    rows.append({"ticker": "TEST", "trade_date": "2026-08-25", "foreign_net_shares": 99.0,
                 "foreign_buy_shares": 100.0, "foreign_sell_shares": 1.0, "volume": 100.0,
                 "source": "ZAPI_IDX_FOREIGN_FLOW"})

    row = zapi.score_foreign_history(pd.DataFrame(rows), ["TEST"], as_of=as_of).iloc[0]

    assert int(row["foreign_expected_sessions"]) == 20
    assert int(row["foreign_observed_sessions"]) == 19
    assert float(row["foreign_coverage_ratio"]) == 0.95
    assert row["foreign_freshness_state"] == "FRESH"
    assert row["foreign_window_state"] == "PARTIAL"
    assert float(row["zapi_foreign_net_shares_20d"]) == 19.0


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
