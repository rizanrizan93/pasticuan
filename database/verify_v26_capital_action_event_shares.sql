-- Read-only verification for Phase 5.6 capital-action event_shares v26.

do $$
begin
  if not exists (
    select 1
    from information_schema.columns
    where table_schema='public'
      and table_name='evidence_capital_actions'
      and column_name='event_shares'
      and data_type='numeric'
      and is_nullable='YES'
  ) then
    raise exception 'event_shares missing, wrong type, or unexpectedly non-nullable';
  end if;
end;
$$;
