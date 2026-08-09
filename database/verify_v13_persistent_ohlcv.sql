select
    to_regclass('public.ohlcv_daily_cache') is not null as table_exists,
    has_table_privilege('service_role','public.ohlcv_daily_cache','SELECT,INSERT,UPDATE,DELETE') as service_role_dml,
    not has_table_privilege('anon','public.ohlcv_daily_cache','SELECT') as anon_blocked,
    not has_table_privilege('authenticated','public.ohlcv_daily_cache','SELECT') as authenticated_blocked;

select column_name, data_type, is_nullable
from information_schema.columns
where table_schema='public' and table_name='ohlcv_daily_cache'
order by ordinal_position;
