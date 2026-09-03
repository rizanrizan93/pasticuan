-- Phase 5.6 v28: expand the factual calculation_state contract.
-- This replaces only the CHECK constraint; it does not delete or rewrite rows.

do $$
begin
  alter table public.evidence_capital_actions
    drop constraint if exists evidence_capital_actions_calculation_check;

  alter table public.evidence_capital_actions
    add constraint evidence_capital_actions_calculation_check
    check (
      calculation_state = any (
        array[
          'NO_SHARE_FACTS',
        'EXPLICIT_DELTA_ONLY',
        'EXPLICIT_PRE_POST',
        'EXPLICIT_DELTA_POST_DERIVED_PRE',
        'EXPLICIT_PRE_DELTA_DERIVED_POST',
        'EXPLICIT_EVENT_SHARES_POST_NO_DELTA',
        'EXPLICIT_EVENT_SHARES_ONLY',
        'EXPLICIT_POST_ONLY',
        'EXPLICIT_PRE_ONLY',
        'REPORTED_POST_NEGATIVE_EXPLICIT_PRE_DELTA',
        'REPORTED_POST_NEGATIVE_EXPLICIT_DELTA',
        'REPORTED_POST_NEGATIVE_EXPLICIT_PRE',
        'REPORTED_POST_NEGATIVE_EVENT_SHARES_ONLY',
        'REPORTED_POST_NEGATIVE_NO_USABLE_TOTAL'
        ]::text[]
      )
    ) not valid;

  alter table public.evidence_capital_actions
    validate constraint evidence_capital_actions_calculation_check;
end;
$$;
