from simple_focus import _business_component, _future_component


def _row(revenue_growth: float, earnings_growth: float) -> dict[str, float]:
    return {
        "fund_revenue_growth": revenue_growth,
        "fund_earnings_growth": earnings_growth,
        "fund_forward_financial_capacity_score": 72.0,
        "fund_forward_financial_capacity_coverage_pct": 90.0,
        "fund_reinvestment_runway_pillar": 68.0,
        "fund_reinvestment_runway_coverage_pct": 85.0,
        "fund_forward_growth_persistence_score": 74.0,
        "fund_forward_growth_persistence_coverage_pct": 88.0,
    }


def test_realised_growth_changes_business_but_is_not_double_counted_in_future_pillar():
    weak = _row(-0.20, -0.30)
    strong = _row(0.35, 0.50)

    assert _business_component(strong)[0] > _business_component(weak)[0]
    assert _future_component(strong)[:2] == _future_component(weak)[:2]
