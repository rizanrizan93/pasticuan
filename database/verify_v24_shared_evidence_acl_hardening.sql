-- Verify the Phase 5.6 ACL-only hardening without changing database state.

do $$
declare
  table_name text;
  relation_oid regclass;
  privilege_name text;
  service_role_acl text[];
begin
  foreach table_name in array array[
    'evidence_provider_state', 'evidence_refresh_leases', 'evidence_ingestion_runs',
    'evidence_failures', 'evidence_raw_payloads', 'evidence_market_daily',
    'evidence_foreign_flow', 'evidence_participant_flow', 'evidence_ownership_files',
    'evidence_ownership_snapshots', 'evidence_ownership_changes', 'evidence_financial_reports',
    'evidence_financial_facts', 'evidence_announcements', 'evidence_capital_actions',
    'evidence_companies', 'evidence_reference_values', 'evidence_brokers',
    'evidence_broker_market_daily', 'evidence_risk_events',
    'evidence_trading_calendar'
  ] loop
    relation_oid := to_regclass(format('public.%I', table_name));
    if relation_oid is null then
      raise exception 'missing shared evidence table: %', table_name;
    end if;

    if not (
      has_table_privilege('service_role', relation_oid, 'SELECT')
      and has_table_privilege('service_role', relation_oid, 'INSERT')
      and has_table_privilege('service_role', relation_oid, 'UPDATE')
    ) then
      raise exception 'missing intended service_role privilege: %', table_name;
    end if;

    if has_table_privilege('service_role', relation_oid, 'DELETE')
       or has_table_privilege('service_role', relation_oid, 'TRUNCATE')
       or has_table_privilege('service_role', relation_oid, 'REFERENCES')
       or has_table_privilege('service_role', relation_oid, 'TRIGGER')
       or has_table_privilege('service_role', relation_oid, 'MAINTAIN') then
      raise exception 'unexpected service_role privilege: %', table_name;
    end if;

    select coalesce(array_agg(acl.privilege_type order by acl.privilege_type), array[]::text[])
      into service_role_acl
      from pg_class relation
      cross join lateral aclexplode(coalesce(relation.relacl, acldefault('r', relation.relowner))) acl
      where relation.oid = relation_oid
        and acl.grantee = (select oid from pg_roles where rolname = 'service_role');
    if service_role_acl <> array['INSERT', 'SELECT', 'UPDATE']::text[] then
      raise exception 'service_role raw ACL is not exact for %: %', table_name, service_role_acl;
    end if;

    if exists (
      select 1
      from pg_class relation
      cross join lateral aclexplode(coalesce(relation.relacl, acldefault('r', relation.relowner))) acl
      where relation.oid = relation_oid and acl.grantee = 0
    ) then
      raise exception 'PUBLIC retains table privilege: %', table_name;
    end if;

    foreach privilege_name in array array[
      'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE',
      'REFERENCES', 'TRIGGER', 'MAINTAIN'
    ] loop
      if has_table_privilege('anon', relation_oid, privilege_name) then
        raise exception 'anon retains % on %', privilege_name, table_name;
      end if;
      if has_table_privilege('authenticated', relation_oid, privilege_name) then
        raise exception 'authenticated retains % on %', privilege_name, table_name;
      end if;
    end loop;

    if not (select relrowsecurity from pg_class where oid = relation_oid) then
      raise exception 'RLS disabled on %', table_name;
    end if;
  end loop;

  if (
    select count(*)
    from pg_proc
    where pronamespace = 'public'::regnamespace
      and proname in (
        'evidence_acquire_refresh_lease',
        'evidence_complete_refresh_lease',
        'evidence_fail_refresh_lease'
      )
  ) <> 3 then
    raise exception 'shared refresh lease function inventory changed';
  end if;

  if exists (
    select 1
    from pg_proc
    where pronamespace = 'public'::regnamespace
      and proname in (
        'evidence_acquire_refresh_lease',
        'evidence_complete_refresh_lease',
        'evidence_fail_refresh_lease'
      )
      and (
        prosecdef
        or array_to_string(proconfig, ',') not in ('search_path=', 'search_path=""')
        or not has_function_privilege('service_role', oid, 'EXECUTE')
        or has_function_privilege('anon', oid, 'EXECUTE')
        or has_function_privilege('authenticated', oid, 'EXECUTE')
        or exists (
          select 1
          from aclexplode(coalesce(proacl, acldefault('f', proowner))) acl
          where acl.grantee = 0 and acl.privilege_type = 'EXECUTE'
        )
      )
  ) then
    raise exception 'shared refresh lease function security changed';
  end if;
end;
$$;

select
  c.relname as table_name,
  c.relrowsecurity as rls_enabled,
  has_table_privilege('service_role', c.oid, 'SELECT') as service_role_select,
  has_table_privilege('service_role', c.oid, 'INSERT') as service_role_insert,
  has_table_privilege('service_role', c.oid, 'UPDATE') as service_role_update,
  not has_table_privilege('service_role', c.oid, 'DELETE') as service_role_delete_denied,
  not has_table_privilege('service_role', c.oid, 'TRUNCATE') as service_role_truncate_denied,
  not has_table_privilege('service_role', c.oid, 'REFERENCES') as service_role_references_denied,
  not has_table_privilege('service_role', c.oid, 'TRIGGER') as service_role_trigger_denied,
  not has_table_privilege('service_role', c.oid, 'MAINTAIN') as service_role_maintain_denied,
  not has_table_privilege('anon', c.oid, 'SELECT') as anon_select_denied,
  not has_table_privilege('authenticated', c.oid, 'SELECT') as authenticated_select_denied
from pg_class c
where c.relnamespace = 'public'::regnamespace
  and c.relkind = 'r'
  and c.relname = any(array[
    'evidence_provider_state', 'evidence_refresh_leases', 'evidence_ingestion_runs',
    'evidence_failures', 'evidence_raw_payloads', 'evidence_market_daily',
    'evidence_foreign_flow', 'evidence_participant_flow', 'evidence_ownership_files',
    'evidence_ownership_snapshots', 'evidence_ownership_changes', 'evidence_financial_reports',
    'evidence_financial_facts', 'evidence_announcements', 'evidence_capital_actions',
    'evidence_companies', 'evidence_reference_values', 'evidence_brokers',
    'evidence_broker_market_daily', 'evidence_risk_events',
    'evidence_trading_calendar'
  ])
order by c.relname;
