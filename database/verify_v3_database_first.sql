-- Run after migration_v3_database_first_fixed.sql.
select
    to_regclass('public.fundamental_cache') as fundamental_cache,
    to_regclass('public.fundamental_history_cache') as fundamental_history_cache,
    to_regclass('public.forward_quality_cache') as forward_quality_cache,
    to_regclass('public.refresh_state') as refresh_state,
    to_regclass('public.scan_checkpoints') as scan_checkpoints,
    to_regclass('public.latest_fundamental_snapshots') as latest_fundamental_snapshots;

select
    table_schema,
    table_name,
    column_name,
    data_type
from information_schema.columns
where table_schema = 'public'
  and table_name = 'fundamental_snapshots'
  and column_name = 'fundamental_reliability';

select tablename, rowsecurity
from pg_tables
where schemaname = 'public'
  and tablename in (
    'fundamental_cache','fundamental_history_cache','forward_quality_cache',
    'refresh_state','scan_checkpoints'
  )
order by tablename;
