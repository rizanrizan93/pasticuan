-- Read-only verification for Phase 5.6 capital-action date-span v25.

do $$
begin
  if not exists (
    select 1 from information_schema.columns
    where table_schema='public' and table_name='evidence_capital_actions'
      and column_name='event_date_kind' and is_nullable='NO'
  ) then
    raise exception 'event_date_kind missing or nullable';
  end if;

  if not exists (
    select 1 from information_schema.columns
    where table_schema='public' and table_name='evidence_capital_actions'
      and column_name='event_start_date' and data_type='date'
  ) then
    raise exception 'event_start_date missing';
  end if;

  if not exists (
    select 1 from information_schema.columns
    where table_schema='public' and table_name='evidence_capital_actions'
      and column_name='event_end_date' and data_type='date'
  ) then
    raise exception 'event_end_date missing';
  end if;

  if not exists (
    select 1 from pg_constraint
    where conrelid='public.evidence_capital_actions'::regclass
      and conname='evidence_capital_actions_event_date_semantics_check'
  ) then
    raise exception 'date semantics constraint missing';
  end if;

  if exists (
    select 1 from public.evidence_capital_actions
    where not (
      (
        event_date_kind='POINT'
        and event_start_date is null
        and event_end_date is null
      )
      or
      (
        event_date_kind='RANGE_END'
        and event_start_date is not null
        and event_end_date is not null
        and event_start_date <= event_end_date
        and event_date = event_end_date
      )
    )
  ) then
    raise exception 'capital-action date semantics violation';
  end if;
end;
$$;
