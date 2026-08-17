from __future__ import annotations

"""Post-ZAPI execution safety and shadow-calibration utilities for Super Scanner.

ZAPI remains confirmation evidence. These helpers may delay or de-authorize a
swing entry when foreign selling is severe, but they never promote READY and
never rewrite the underlying business/future-fundamental thesis.
"""

from typing import Any

import numpy as np
import pandas as pd


POST_CALIBRATION_VERSION = "1.0.0-super-zapi-post-calibration"


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value or "").strip().upper() in {"1", "TRUE", "YES", "Y", "PASS", "READY", "VERIFIED"}


def _tokens(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text or text.upper() in {"NONE", "NAN"}:
        return []
    return list(dict.fromkeys(part.strip() for part in text.split("|") if part.strip() and part.strip().upper() != "NONE"))


def _join_tokens(values: list[str]) -> str:
    clean = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    return " | ".join(clean) if clean else "NONE"


def enrich_super_shadow(frame: pd.DataFrame) -> pd.DataFrame:
    """Persist auditable pre/post ZAPI fields for later 5D/20D/60D OOS review."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    out = frame.copy()
    pre = pd.to_numeric(out.get("zapi_super_original_silent_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    if "zapi_super_post_silent_score" in out.columns:
        post = pd.to_numeric(out["zapi_super_post_silent_score"], errors="coerce")
    elif "flow_silent_accumulation_score" in out.columns:
        post = pd.to_numeric(out["flow_silent_accumulation_score"], errors="coerce")
    else:
        post = pd.to_numeric(out.get("silent_accumulation_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    coverage = pd.to_numeric(out.get("zapi_foreign_flow_coverage_pct", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)

    out["zapi_shadow_pre_silent_score"] = pre.round(3)
    out["zapi_shadow_post_silent_score"] = post.round(3)
    out["zapi_shadow_silent_score_delta"] = (post - pre).round(3)
    out["zapi_shadow_calibration_state"] = np.where(coverage.gt(0.0), "PENDING_FORWARD_OUTCOME", "NO_ZAPI_EVIDENCE")
    out["zapi_shadow_forward_horizons"] = "5D|20D|60D"
    out["zapi_shadow_policy"] = "CAPTURE_PRE_POST_NOW_RECALIBRATE_ONLY_AFTER_FORWARD_OOS"
    out["zapi_post_calibration_version"] = POST_CALIBRATION_VERSION
    return out


def apply_super_foreign_shock_guard(frame: pd.DataFrame) -> pd.DataFrame:
    """Delay Swing actionability on severe foreign selling without hard-blocking thesis.

    Policy:
    - coverage <60%: fail-soft, existing authorization is unchanged;
    - 1D <= -4%: diagnostic caution only;
    - 1D <= -8%: reclaim/absorption required before Swing actionability;
    - 1D <= -15%: extreme one-day shock, same authorization downgrade;
    - severe/extreme plus negative 5D/20D or NET_DISTRIBUTION: wait for flow stabilization;
    - positive foreign flow is confirmation only and can never promote authorization.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    out = frame.copy()

    states: list[str] = []
    severities: list[float] = []
    actions: list[str] = []
    reasons: list[str] = []
    pre_auth_state: list[str] = []
    pre_auth_pass: list[bool] = []
    pre_actionable: list[bool] = []
    pre_order_builder: list[bool] = []

    for idx, row in out.iterrows():
        auth_state = str(row.get("real_money_authorization_state") or "")
        auth_pass = _truthy(row.get("real_money_authorization_pass"))
        actionable = _truthy(row.get("actionable_rank_eligible"))
        order_builder = _truthy(row.get("order_builder_eligible"))
        pre_auth_state.append(auth_state)
        pre_auth_pass.append(auth_pass)
        pre_actionable.append(actionable)
        pre_order_builder.append(order_builder)

        coverage = _finite(row.get("zapi_foreign_flow_coverage_pct"), np.nan)
        one_day = _finite(row.get("zapi_foreign_net_participation_1d"), np.nan)
        five_day = _finite(row.get("zapi_foreign_net_participation_5d"), np.nan)
        twenty_day = _finite(row.get("zapi_foreign_net_participation_20d"), np.nan)
        foreign_state = str(row.get("zapi_foreign_state") or "").upper()

        state = "NO_EXECUTION_OVERRIDE"
        severity = 0.0
        action = "KEEP_EXISTING_AUTHORIZATION"
        reason = "ZAPI confirmation does not alter Super authorization."

        if not np.isfinite(coverage) or coverage < 60.0 or not np.isfinite(one_day):
            state = "ZAPI_INSUFFICIENT_OR_STALE_FOR_EXECUTION_GUARD"
            reason = "Foreign-flow coverage below execution-guard threshold; fail-soft and keep existing authorization."
        else:
            if one_day <= -0.15:
                shock_label = "EXTREME"
                severity = 100.0
            elif one_day <= -0.08:
                shock_label = "SEVERE"
                severity = 75.0
            elif one_day <= -0.04:
                shock_label = "MODERATE"
                severity = 45.0
            else:
                shock_label = "NONE"

            persistent_distribution = bool(
                shock_label in {"SEVERE", "EXTREME"}
                and (
                    (np.isfinite(five_day) and five_day <= -0.01)
                    or (np.isfinite(twenty_day) and twenty_day <= -0.01)
                    or foreign_state == "NET_DISTRIBUTION"
                )
            )

            if persistent_distribution:
                state = "PERSISTENT_FOREIGN_DISTRIBUTION_WAIT"
                action = "WAIT_FLOW_STABILIZATION_AND_RECLAIM"
                reason = "Severe foreign selling persists into 5D/20D or NET_DISTRIBUTION; research thesis remains intact but Swing entry is not actionable."
                manual = _tokens(row.get("real_money_manual_checks"))
                manual.append("ZAPI_FOREIGN_DISTRIBUTION_WAIT_STABILIZATION")
                out.at[idx, "real_money_manual_checks"] = _join_tokens(manual)
                if auth_state.upper() != "REAL_MONEY_BLOCKED":
                    out.at[idx, "real_money_authorization_state"] = "REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED"
                out.at[idx, "real_money_authorization_pass"] = False
                if "order_builder_eligible" in out.columns:
                    out.at[idx, "order_builder_eligible"] = False
                if "order_ready" in out.columns:
                    out.at[idx, "order_ready"] = False
                if "actionable_rank_eligible" in out.columns:
                    out.at[idx, "actionable_rank_eligible"] = False
                if "execution_gate_state" in out.columns:
                    out.at[idx, "execution_gate_state"] = "BLOCKED"
            elif shock_label in {"SEVERE", "EXTREME"}:
                state = f"{shock_label}_ONE_DAY_FOREIGN_SELL_SHOCK_RECLAIM_REQUIRED"
                action = "REQUIRE_ABSORPTION_OR_RECLAIM_BEFORE_ENTRY"
                reason = "Large 1D foreign sell shock with non-persistent medium-window flow; require absorption/reclaim before Swing authorization."
                manual = _tokens(row.get("real_money_manual_checks"))
                manual.append("ZAPI_FOREIGN_SHOCK_REQUIRE_ABSORPTION_RECLAIM")
                out.at[idx, "real_money_manual_checks"] = _join_tokens(manual)
                if auth_state.upper() != "REAL_MONEY_BLOCKED":
                    out.at[idx, "real_money_authorization_state"] = "REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED"
                out.at[idx, "real_money_authorization_pass"] = False
                if "order_builder_eligible" in out.columns:
                    out.at[idx, "order_builder_eligible"] = False
                if "order_ready" in out.columns:
                    out.at[idx, "order_ready"] = False
                if "actionable_rank_eligible" in out.columns:
                    out.at[idx, "actionable_rank_eligible"] = False
                if "execution_gate_state" in out.columns:
                    out.at[idx, "execution_gate_state"] = "BLOCKED"
            elif shock_label == "MODERATE":
                state = "MODERATE_ONE_DAY_FOREIGN_SELL_CAUTION"
                action = "MONITOR_ABSORPTION_NO_AUTHORIZATION_CHANGE"
                reason = "Moderate 1D foreign selling is diagnostic only; existing Super authorization remains controlling."
            elif one_day >= 0.08:
                state = "STRONG_ONE_DAY_FOREIGN_ACCUMULATION_CONFIRMATION_ONLY"
                action = "CONFIRMATION_ONLY_NO_READY_PROMOTION"
                reason = "Strong foreign accumulation confirms flow but can never promote READY by itself."
            else:
                state = "FOREIGN_FLOW_NO_SHOCK"
                reason = "No material one-day foreign-flow shock; existing Super authorization remains controlling."

        states.append(state)
        severities.append(severity)
        actions.append(action)
        reasons.append(reason)

    out["zapi_pre_guard_real_money_authorization_state"] = pre_auth_state
    out["zapi_pre_guard_real_money_authorization_pass"] = pre_auth_pass
    out["zapi_pre_guard_actionable_rank_eligible"] = pre_actionable
    out["zapi_pre_guard_order_builder_eligible"] = pre_order_builder
    out["zapi_foreign_shock_state"] = states
    out["zapi_foreign_shock_severity"] = severities
    out["zapi_execution_flow_guard_state"] = actions
    out["zapi_execution_flow_guard_reason"] = reasons
    out["zapi_execution_guard_policy"] = "ZAPI_CAN_ONLY_DELAY_OR_DEAUTHORIZE_SWING_ENTRY_NEVER_PROMOTE_READY"
    out["zapi_post_calibration_version"] = POST_CALIBRATION_VERSION
    return out


__all__ = [
    "POST_CALIBRATION_VERSION",
    "enrich_super_shadow",
    "apply_super_foreign_shock_guard",
]
