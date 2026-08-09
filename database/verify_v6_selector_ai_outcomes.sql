-- Run after migration_v6_selector_ai_outcomes.sql.

select
    to_regclass('public.ai_execution_outcomes') is not null
        as ai_execution_outcomes_ready,
    to_regclass('public.selector_snapshots') is not null
        as selector_snapshots_ready,
    to_regclass('public.selector_outcomes') is not null
        as selector_outcomes_ready,
    to_regclass('public.selector_model_evaluations') is not null
        as selector_model_evaluations_ready;

select
    table_name,
    column_name,
    data_type,
    is_nullable
from information_schema.columns
where table_schema = 'public'
  and table_name in (
      'ai_execution_outcomes',
      'selector_snapshots',
      'selector_outcomes',
      'selector_model_evaluations'
  )
order by table_name, ordinal_position;

select
    schemaname,
    tablename,
    rowsecurity
from pg_tables
where schemaname = 'public'
  and tablename in (
      'ai_execution_outcomes',
      'selector_snapshots',
      'selector_outcomes',
      'selector_model_evaluations'
  )
order by tablename;

select
    tablename,
    indexname,
    indexdef
from pg_indexes
where schemaname = 'public'
  and tablename in (
      'ai_execution_outcomes',
      'selector_snapshots',
      'selector_outcomes',
      'selector_model_evaluations'
  )
order by tablename, indexname;
