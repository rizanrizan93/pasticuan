-- Read-only verification for v9.8.2 Hotfix 7.
select
    to_regclass('public.ohlcv_daily_cache') is not null as ohlcv_table_exists,
    to_regclass('public.scanner_feature_cache') is not null as feature_table_exists,
    has_table_privilege('service_role', 'public.ohlcv_daily_cache', 'SELECT,INSERT,UPDATE,DELETE') as ohlcv_service_role_rw,
    has_table_privilege('service_role', 'public.scanner_feature_cache', 'SELECT,INSERT,UPDATE,DELETE') as feature_service_role_rw;

select
    (select count(*) from public.fundamental_cache
      where upper(btrim(coalesce(statement_date, ''))) = 'NAT') as nat_statement_dates,
    (select count(*) from public.fundamental_history_cache
      where upper(btrim(coalesce(latest_period, ''))) = 'NAT') as nat_history_periods;

select
    p.proname,
    coalesce(array_to_string(p.proconfig, ','), '') as function_config,
    has_function_privilege('anon', p.oid, 'EXECUTE') as anon_execute,
    has_function_privilege('authenticated', p.oid, 'EXECUTE') as authenticated_execute
from pg_proc p
join pg_namespace n on n.oid = p.pronamespace
where n.nspname = 'public'
  and p.proname in ('set_scanner_updated_at', 'set_refresh_state_updated_at', 'rls_auto_enable')
order by p.proname;

select event_object_table, trigger_name, action_statement
from information_schema.triggers
where trigger_schema = 'public'
  and trigger_name in ('trg_refresh_state_updated_at', 'trg_ohlcv_daily_cache_updated_at', 'trg_scanner_feature_cache_updated_at')
order by event_object_table;
