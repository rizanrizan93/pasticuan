from __future__ import annotations

import numpy as np
import pandas as pd

import public_idx_broker_flow as broker


def test_aggregate_trade_detail_reconstructs_participant_flow(tmp_path):
    path = tmp_path / "trade.csv"
    frame = pd.DataFrame([
        {"asset": "TEST", "participant_buy": "AB", "participant_sell": "CD", "volume": 1000, "value": 1000000, "tradingdate": "2026-08-14"},
        {"asset": "TEST", "participant_buy": "AB", "participant_sell": "EF", "volume": 500, "value": 600000, "tradingdate": "2026-08-14"},
        {"asset": "TEST", "participant_buy": "CD", "participant_sell": "AB", "volume": 200, "value": 220000, "tradingdate": "2026-08-14"},
    ])
    frame.to_csv(path, sep="|", index=False)
    out = broker.aggregate_trade_detail(str(path), pd.Timestamp("2026-08-14").date(), ["TEST"])
    ab = out.loc[out["broker_code"].eq("AB")].iloc[0]
    assert float(ab["buy_value"]) == 1600000
    assert float(ab["sell_value"]) == 220000
    assert float(ab["net_value"]) == 1380000
    assert float(ab["buy_avg"]) == 1600000 / 1500


def test_daily_trim_keeps_top_buyers_and_sellers():
    rows = []
    for i in range(12):
        rows.append({
            "trade_date": pd.Timestamp("2026-08-14"), "ticker": "TEST", "broker_code": f"B{i:02d}",
            "buy_value": (100-i) * 1000, "sell_value": 0, "buy_volume": (100-i), "sell_volume": 0,
            "buy_avg": 1000, "sell_avg": np.nan, "net_value": (100-i) * 1000,
            "net_volume": (100-i), "gross_value": (100-i) * 1000,
            "source": broker.SOURCE_NAME, "source_verified": True, "provenance_state": "TEST",
        })
    out = broker.trim_daily_top_flow(pd.DataFrame(rows), top_n=10)
    assert len(out) == 10
    assert out["side"].eq("TOP_NET_BUYER").all()


def test_score_broker_history_has_bounded_accumulation_score():
    rows = []
    for day in pd.bdate_range("2026-07-20", periods=20):
        for rank, value in enumerate((100, 80, 60)):
            rows.append({
                "trade_date": day, "ticker": "TEST", "broker_code": ["AB", "CD", "EF"][rank],
                "net_value": value * 1000000, "net_volume": value * 1000, "side": "TOP_NET_BUYER",
                "net_rank": rank + 1, "buy_value": value * 1000000, "sell_value": 0,
                "buy_volume": value * 1000, "sell_volume": 0, "buy_avg": 1000, "sell_avg": np.nan,
                "gross_value": value * 1000000,
            })
    scored = broker.score_broker_history(pd.DataFrame(rows), ["TEST"])
    row = scored.iloc[0]
    assert 0 <= float(row["broker_accumulation_score"]) <= 100
    assert row["broker_top_buyer_code"] == "AB"
    assert row["broker_accumulation_state"] == "PARTICIPANT_ACCUMULATION"


def test_missing_cache_is_safe_at_normalized_consumer_boundary(monkeypatch):
    monkeypatch.setattr(broker, "load_public_cache", lambda: pd.DataFrame())
    empty_schema = pd.DataFrame(columns=["trade_date", "ticker", "broker_code", "side", "net_value", "net_rank"])
    scored = broker.score_broker_history(empty_schema, ["TEST"])
    assert list(scored["ticker"]) == ["TEST"]
