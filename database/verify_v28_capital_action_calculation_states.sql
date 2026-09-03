-- Read-only verification for Phase 5.6 capital-action calculation-state v28.

do $$
declare
  definition text;
begin
  select pg_get_constraintdef(oid)
    into definition
  from pg_constraint
  where conrelid='public.evidence_capital_actions'::regclass
    and conname='evidence_capital_actions_calculation_check';

  if definition is null then
    raise exception 'calculation-state constraint missing';
  end if;

  if position('EXPLICIT_EVENT_SHARES_POST_NO_DELTA' in definition) = 0
     or position('REPORTED_POST_NEGATIVE_EVENT_SHARES_ONLY' in definition) = 0 then
    raise exception 'calculation-state constraint is stale';
  end if;
end;
$$;
