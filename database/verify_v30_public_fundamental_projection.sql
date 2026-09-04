-- Verification for Phase 5.6 v30 public factual projection.

select
    c.relrowsecurity as rls_enabled,
    exists (
        select 1 from pg_policies p
        where p.schemaname = 'public'
          and p.tablename = 'phase56_public_fundamental_snapshots'
          and p.policyname = 'phase56_public_fundamental_select'
          and p.cmd = 'SELECT'
          and p.roles @> array['anon','authenticated']::name[]
    ) as read_policy_present
from pg_class c
join pg_namespace n on n.oid = c.relnamespace
where n.nspname = 'public'
  and c.relname = 'phase56_public_fundamental_snapshots';

select grantee, privilege_type
from information_schema.role_table_grants
where table_schema = 'public'
  and table_name = 'phase56_public_fundamental_snapshots'
  and grantee in ('anon','authenticated','service_role')
order by grantee, privilege_type;

select
    count(*) as rows_total,
    count(*) filter (where jsonb_typeof(proxy_metrics) = 'object') as proxy_object_rows,
    count(*) filter (where jsonb_typeof(official_metrics) = 'object') as official_object_rows,
    count(*) filter (where proxy_period_end is not null) as period_anchored_rows,
    count(*) filter (where official_period_end is not null) as official_period_rows
from public.phase56_public_fundamental_snapshots;
