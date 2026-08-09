-- IDX Super Scanner v7.16.3: verify the most recent Supabase upload.
-- Read-only. No schema change and safe to run repeatedly.

with latest_scan as (
    select scan_id, as_of, started_at, finished_at, ticker_count,
           prepared_count, multibagger_count, core_swing_count,
           model_version, schema_version
    from public.scan_runs
    order by as_of desc
    limit 1
), table_counts as (
    select 'scan_runs'::text as table_name, count(*)::bigint as rows_found
      from public.scan_runs r join latest_scan l on r.scan_id = l.scan_id
    union all
    select 'fundamental_snapshots', count(*)
      from public.fundamental_snapshots r join latest_scan l on r.scan_id = l.scan_id
    union all
    select 'multibagger_snapshots', count(*)
      from public.multibagger_snapshots r join latest_scan l on r.scan_id = l.scan_id
    union all
    select 'technical_snapshots', count(*)
      from public.technical_snapshots r join latest_scan l on r.scan_id = l.scan_id
    union all
    select 'narrative_snapshots', count(*)
      from public.narrative_snapshots r join latest_scan l on r.scan_id = l.scan_id
    union all
    select 'project_events', count(*)
      from public.project_events r join latest_scan l on r.scan_id = l.scan_id
    union all
    select 'provider_health', count(*)
      from public.provider_health r join latest_scan l on r.scan_id = l.scan_id
)
select
    l.scan_id,
    l.as_of,
    l.started_at,
    l.finished_at,
    l.ticker_count,
    l.prepared_count,
    l.multibagger_count,
    l.core_swing_count,
    l.model_version,
    l.schema_version,
    c.table_name,
    c.rows_found,
    case
        when c.table_name = 'scan_runs' and c.rows_found = 1 then 'OK'
        when c.table_name <> 'scan_runs' and c.rows_found > 0 then 'OK'
        else 'MISSING_OR_EMPTY'
    end as verification_state
from latest_scan l
cross join table_counts c
order by case c.table_name
    when 'scan_runs' then 1
    when 'fundamental_snapshots' then 2
    when 'multibagger_snapshots' then 3
    when 'technical_snapshots' then 4
    when 'narrative_snapshots' then 5
    when 'project_events' then 6
    when 'provider_health' then 7
    else 99 end;

-- Optional detail query: uncomment after copying the scan_id from the result above.
-- select ticker, multibagger_status, confidence_adjusted_multibagger_score, as_of
-- from public.multibagger_snapshots
-- where scan_id = '<SCAN_ID>'
-- order by confidence_adjusted_multibagger_score desc nulls last;
