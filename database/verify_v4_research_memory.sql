select c.table_name,
       coalesce(pc.relrowsecurity, false) as rowsecurity
from information_schema.tables c
left join pg_class pc on pc.relname = c.table_name
left join pg_namespace pn on pn.oid = pc.relnamespace and pn.nspname = c.table_schema
where c.table_schema = 'public'
  and c.table_name in (
    'model_registry','source_events','research_outcomes','backfill_state','idx_trading_calendar'
  )
order by c.table_name;

select table_name, column_name, data_type
from information_schema.columns
where table_schema = 'public'
  and (
    (table_name in ('fundamental_cache','fundamental_history_cache','forward_quality_cache')
     and column_name in ('parser_version','event_fingerprint','next_check_at','refresh_reason'))
    or (table_name = 'refresh_state' and column_name in ('parser_version','event_fingerprint','refresh_reason'))
  )
order by table_name, column_name;

select table_name, column_name, data_type
from information_schema.columns
where table_schema='public' and ((table_name='multibagger_snapshots' and column_name in ('silent_accumulation_raw_score','silent_accumulation_liquidity_adjustment','silent_accumulation_liquidity_min_confirmation','silent_accumulation_calibration_policy','liquidity_bucket')) or (table_name='eoff_predictions' and column_name in ('best_buy_raw_date','best_buy_calendar_state','best_buy_calendar_verified','best_buy_date_adjustment_days','eoff_fib_unique_anchor_count','eoff_fib_unique_anchor_ratio','eoff_fib_dominant_anchor_share','eoff_unique_anchor_gate','eoff_unique_anchor_signature')))
order by table_name,column_name;
