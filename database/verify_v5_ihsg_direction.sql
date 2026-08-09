-- Run after migration_v5_ihsg_direction.sql.

select
    to_regclass('public.ihsg_direction_snapshots') is not null
        as ihsg_direction_table_ready,
    exists (
        select 1
        from information_schema.columns
        where table_schema = 'public'
          and table_name = 'research_outcomes'
          and column_name = 'forward_return_1d'
    ) as one_day_outcome_ready;

select
    column_name,
    data_type,
    is_nullable
from information_schema.columns
where table_schema = 'public'
  and table_name = 'ihsg_direction_snapshots'
order by ordinal_position;

select
    schemaname,
    tablename,
    rowsecurity
from pg_tables
where schemaname = 'public'
  and tablename in ('ihsg_direction_snapshots', 'research_outcomes');

select
    indexname,
    indexdef
from pg_indexes
where schemaname = 'public'
  and tablename = 'ihsg_direction_snapshots'
order by indexname;
