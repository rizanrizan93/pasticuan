-- Read-only verification for Phase 5.6 capital-action reported_post_shares v27.

do $$
begin
  if not exists (
    select 1
    from information_schema.columns
    where table_schema='public'
      and table_name='evidence_capital_actions'
      and column_name='reported_post_shares'
      and data_type='numeric'
      and is_nullable='YES'
  ) then
    raise exception 'reported_post_shares missing, wrong type, or unexpectedly non-nullable';
  end if;
end;
$$;
