-- Run after migration_v9_narrative_intelligence.sql.

select table_name
from information_schema.tables
where table_schema = 'public'
  and table_name in (
      'narrative_events',
      'narrative_event_outcomes',
      'narrative_snapshots'
  )
order by table_name;

select table_name, column_name, data_type
from information_schema.columns
where table_schema = 'public'
  and (
      (table_name = 'narrative_events'
       and column_name in (
           'narrative_event_id', 'detected_at', 'event_type',
           'official_verified', 'novelty_score'
       ))
      or
      (table_name = 'narrative_event_outcomes'
       and column_name in (
           'narrative_outcome_id', 'signal_timestamp',
           'net_excess_return_5d_pct', 'net_excess_return_20d_pct',
           'net_excess_return_60d_pct'
       ))
      or
      (table_name = 'narrative_snapshots'
       and column_name in (
           'snapshot_id', 'issuer_alignment_state',
           'retail_adoption_stage',
           'narrative_flow_convergence_state',
           'narrative_hard_block'
       ))
  )
order by table_name, column_name;

select schemaname, tablename, rowsecurity
from pg_tables
where schemaname = 'public'
  and tablename in (
      'narrative_events',
      'narrative_event_outcomes',
      'narrative_snapshots'
  )
order by tablename;
