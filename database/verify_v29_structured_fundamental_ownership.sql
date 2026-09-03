-- Phase 5.6 structured evidence verification.
do $$
declare
  missing_tables integer;
  bad_rls integer;
  bad_public_acl integer;
  bad_service_acl integer;
begin
  select count(*) into missing_tables
  from (values
    ('evidence_fundamental_metrics'),
    ('evidence_shareholder_profiles')
  ) as expected(table_name)
  where not exists (
    select 1 from information_schema.tables t
    where t.table_schema='public' and t.table_name=expected.table_name
  );
  if missing_tables <> 0 then
    raise exception 'structured evidence tables missing: %', missing_tables;
  end if;

  select count(*) into bad_rls
  from pg_class c
  join pg_namespace n on n.oid=c.relnamespace
  where n.nspname='public'
    and c.relname in ('evidence_fundamental_metrics','evidence_shareholder_profiles')
    and not c.relrowsecurity;
  if bad_rls <> 0 then
    raise exception 'structured evidence RLS mismatch: %', bad_rls;
  end if;

  select count(*) into bad_public_acl
  from information_schema.role_table_grants
  where table_schema='public'
    and table_name in ('evidence_fundamental_metrics','evidence_shareholder_profiles')
    and grantee in ('PUBLIC','anon','authenticated');
  if bad_public_acl <> 0 then
    raise exception 'structured evidence public ACL present: %', bad_public_acl;
  end if;

  select count(*) into bad_service_acl
  from (
    select table_name, array_agg(privilege_type order by privilege_type) as privileges
    from information_schema.role_table_grants
    where table_schema='public'
      and table_name in ('evidence_fundamental_metrics','evidence_shareholder_profiles')
      and grantee='service_role'
    group by table_name
  ) grants
  where privileges <> array['INSERT','SELECT','UPDATE'];
  if bad_service_acl <> 0 then
    raise exception 'structured evidence service_role ACL mismatch: %', bad_service_acl;
  end if;
end
$$;
