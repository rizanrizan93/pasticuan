-- Run after migration_v10_narrative_safety.sql.

select table_name, column_name, data_type
from information_schema.columns
where table_schema = 'public'
  and (
      (table_name = 'issuer_master' and column_name in (
          'official_domain', 'official_domain_verified',
          'official_domain_checked_at'
      ))
      or
      (table_name = 'narrative_events' and column_name in (
          'source_hostname', 'registered_official_domain', 'source_state',
          'source_present', 'official_claimed', 'entity_match_state', 'event_status',
          'requested_event_status', 'lifecycle_evidence_state',
          'resolved_at', 'supersedes_event_id', 'resolution_source_url'
      ))
      or
      (table_name = 'narrative_event_outcomes' and column_name in (
          'impact_sign', 'impact_direction', 'entry_policy',
          'directional_excess_return_5d_pct',
          'directional_excess_return_20d_pct',
          'directional_excess_return_60d_pct'
      ))
      or
      (table_name = 'narrative_snapshots' and column_name in (
          'narrative_missing_source_event_count',
          'narrative_inactive_lifecycle_event_count',
          'narrative_entity_unverified_event_count',
          'narrative_production_policy'
      ))
  )
order by table_name, column_name;

select schemaname, tablename, rowsecurity
from pg_tables
where schemaname = 'public'
  and tablename in (
      'issuer_master', 'narrative_events',
      'narrative_event_outcomes', 'narrative_snapshots'
  )
order by tablename;

select
    count(*) filter (where source_state = 'MISSING_SOURCE') as missing_source_rows,
    count(*) filter (where event_status in ('RESOLVED','SUPERSEDED','REVERSED'))
        as inactive_lifecycle_rows,
    count(*) filter (where entity_match_state = 'AMBIGUOUS_TICKER_UNVERIFIED')
        as ambiguous_entity_rows
from public.narrative_events;
