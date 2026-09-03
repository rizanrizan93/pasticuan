from datetime import date

import pytest

from shared_company_evidence import normalize_reference_values


OBSERVED_ON = date(2026, 9, 3)
SOURCE_PERIOD = date(2026, 9, 1)


def _payload(*, set_name: str, **fields):
    return {
        "data": {
            "dataset": "reference",
            "provider": "idx",
            "set": set_name,
            **fields,
        }
    }


def _normalize(payload, set_name: str):
    return normalize_reference_values(
        payload,
        set_name=set_name,
        source_period=SOURCE_PERIOD,
        observed_on=OBSERVED_ON,
    )


def test_market_time_accepts_observed_scalar_value() -> None:
    rows = _normalize(_payload(set_name="market-time", value="observed-provider-value"), "market-time")
    assert len(rows) == 1
    assert rows[0]["set_name"] == "market-time"
    assert rows[0]["label"] == "observed-provider-value"
    assert rows[0]["provider"] == "IDX_REFERENCE_VIA_ZAPI"
    assert rows[0]["validation_state"] == "VALID"


def test_list_reference_sets_keep_existing_items_contract() -> None:
    rows = _normalize(_payload(set_name="boards", items=["Main", "Development"]), "boards")
    assert [row["label"] for row in rows] == ["Development", "Main"]


@pytest.mark.parametrize("value", [None, "", "   ", True, {}, []])
def test_market_time_rejects_missing_blank_bool_or_structured_value(value) -> None:
    with pytest.raises(RuntimeError, match="PARSE_FAILURE"):
        _normalize(_payload(set_name="market-time", value=value), "market-time")


def test_market_time_does_not_fall_back_to_items() -> None:
    with pytest.raises(RuntimeError, match="PARSE_FAILURE"):
        _normalize(_payload(set_name="market-time", items=["legacy-shape"]), "market-time")


def test_other_reference_sets_do_not_accept_scalar_value() -> None:
    with pytest.raises(RuntimeError, match="PARSE_FAILURE"):
        _normalize(_payload(set_name="sectors", value="scalar-wrong-shape"), "sectors")
