from __future__ import annotations

import pytest

from scanner import (
    _stable_unique_delimited,
    combine_fundamental_history,
    parse_idx_xbrl_attachment,
)


def _instance(
    *,
    period_end: str,
    revenue: float,
    net_income: float,
    ocf: float,
    ppe_capex: float,
    intangible_capex: float,
) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
             xmlns:id="http://www.idx.co.id/xbrl/taxonomy/2020">
  <xbrli:context id="CurrentYearDuration">
    <xbrli:entity><xbrli:identifier scheme="TEST">TEST</xbrli:identifier></xbrli:entity>
    <xbrli:period><xbrli:startDate>2026-01-01</xbrli:startDate><xbrli:endDate>{period_end}</xbrli:endDate></xbrli:period>
  </xbrli:context>
  <xbrli:context id="CurrentYearInstant">
    <xbrli:entity><xbrli:identifier scheme="TEST">TEST</xbrli:identifier></xbrli:entity>
    <xbrli:period><xbrli:instant>{period_end}</xbrli:instant></xbrli:period>
  </xbrli:context>
  <xbrli:unit id="IDR"><xbrli:measure>iso4217:IDR</xbrli:measure></xbrli:unit>
  <id:Revenues contextRef="CurrentYearDuration" unitRef="IDR">{revenue}</id:Revenues>
  <id:ProfitLoss contextRef="CurrentYearDuration" unitRef="IDR">{net_income}</id:ProfitLoss>
  <id:NetCashFlowsReceivedFromUsedInOperatingActivities contextRef="CurrentYearDuration" unitRef="IDR">{ocf}</id:NetCashFlowsReceivedFromUsedInOperatingActivities>
  <id:PaymentsForAcquisitionOfPropertyAndEquipment contextRef="CurrentYearDuration" unitRef="IDR">{ppe_capex}</id:PaymentsForAcquisitionOfPropertyAndEquipment>
  <id:PaymentsForAcquisitionOfIntangibleAssets contextRef="CurrentYearDuration" unitRef="IDR">{intangible_capex}</id:PaymentsForAcquisitionOfIntangibleAssets>
  <id:Assets contextRef="CurrentYearInstant" unitRef="IDR">1000</id:Assets>
  <id:Liabilities contextRef="CurrentYearInstant" unitRef="IDR">400</id:Liabilities>
  <id:Equity contextRef="CurrentYearInstant" unitRef="IDR">600</id:Equity>
</xbrli:xbrl>""".encode()


def _parse(period_end: str, period_type: str, **values: float):
    return parse_idx_xbrl_attachment(
        _instance(period_end=period_end, **values),
        ticker="TEST.JK",
        period_end=period_end,
        period_type=period_type,
        source_url="https://www.idx.co.id/official/TEST/instance.zip",
        defer_ytd_conversion=True,
    )


def test_idx_taxonomy_property_and_equipment_capex_components_are_summed() -> None:
    row = _parse(
        "2026-03-31",
        "Q1",
        revenue=100,
        net_income=10,
        ocf=20,
        ppe_capex=7,
        intangible_capex=3,
    ).iloc[0]

    assert row["capex"] == pytest.approx(10.0)


def test_ytd_conversion_is_deferred_until_all_official_periods_are_combined() -> None:
    q1 = _parse(
        "2026-03-31",
        "Q1",
        revenue=100,
        net_income=10,
        ocf=20,
        ppe_capex=4,
        intangible_capex=1,
    )
    q2 = _parse(
        "2026-06-30",
        "Q2",
        revenue=250,
        net_income=25,
        ocf=50,
        ppe_capex=12,
        intangible_capex=3,
    )

    combined = combine_fundamental_history(q1, q2)
    latest = combined.loc[combined["period_type"].eq("Q2")].iloc[0]

    assert latest["statement_basis"] == "STANDALONE_QUARTER_FROM_YTD"
    assert latest["revenue"] == pytest.approx(150.0)
    assert latest["net_income"] == pytest.approx(15.0)
    assert latest["operating_cash_flow"] == pytest.approx(30.0)
    assert latest["capex"] == pytest.approx(10.0)
    assert "YTD_PREDECESSOR_MISSING" not in str(latest["validation_flags"])


def test_cached_provider_and_flag_labels_do_not_accumulate_recursively() -> None:
    providers = _stable_unique_delimited(
        [
            "Yahoo Fundamentals Timeseries Direct + IDX_OFFICIAL_XBRL • YAHOO + IDX_OFFICIAL_XBRL • YAHOO",
            "IDX_OFFICIAL_XBRL • YAHOO",
        ],
        r"\s+\+\s+",
    )
    flags = _stable_unique_delimited(
        ["OCF sering negatif • Konflik data historis • OCF sering negatif", "Konflik data historis"],
        r"\s*•\s*",
    )

    assert providers == ["Yahoo Fundamentals Timeseries Direct", "IDX_OFFICIAL_XBRL • YAHOO"]
    assert flags == ["OCF sering negatif", "Konflik data historis"]
