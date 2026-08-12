-- IDX Super Scanner v9.8.6 free-tier storage safety
-- Bounded retention for Supabase Free Plan. Idempotent.
begin;

create or replace function public.scanner_free_tier_housekeeping(
    p_keep_terminal_jobs integer default 2,
    p_keep_fundamental_snapshots integer default 4,
    p_keep_multibagger_snapshots integer default 3,
    p_keep_technical_snapshots integer default 3,
    p_keep_narrative_snapshots integer default 3,
    p_keep_narrative_events integer default 20,
    p_keep_source_events integer default 12
)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
    v_jobs integer := 0;
    v_fund integer := 0;
    v_multi integer := 0;
    v_tech integer := 0;
    v_narr_snap integer := 0;
    v_narr_event integer := 0;
    v_source integer := 0;
begin
    with ranked as (
        select job_id,
               row_number() over (order by coalesce(finished_at, updated_at, created_at) desc) as rn
        from public.scan_jobs
        where status in ('COMPLETE','COMPLETE_WITH_FAILURES','FAILED','CANCELLED')
    ), deleted as (
        delete from public.scan_jobs j
        using ranked r
        where j.job_id = r.job_id
          and r.rn > greatest(1, p_keep_terminal_jobs)
        returning 1
    ) select count(*) into v_jobs from deleted;

    with ranked as (
        select snapshot_id,
               row_number() over (partition by ticker order by as_of desc, updated_at desc) as rn
        from public.fundamental_snapshots
    ), deleted as (
        delete from public.fundamental_snapshots t
        using ranked r
        where t.snapshot_id=r.snapshot_id and r.rn > greatest(1,p_keep_fundamental_snapshots)
        returning 1
    ) select count(*) into v_fund from deleted;

    with ranked as (
        select snapshot_id,
               row_number() over (partition by ticker order by as_of desc, updated_at desc) as rn
        from public.multibagger_snapshots
    ), deleted as (
        delete from public.multibagger_snapshots t
        using ranked r
        where t.snapshot_id=r.snapshot_id and r.rn > greatest(1,p_keep_multibagger_snapshots)
        returning 1
    ) select count(*) into v_multi from deleted;

    with ranked as (
        select snapshot_id,
               row_number() over (partition by ticker order by as_of desc, updated_at desc) as rn
        from public.technical_snapshots
    ), deleted as (
        delete from public.technical_snapshots t
        using ranked r
        where t.snapshot_id=r.snapshot_id and r.rn > greatest(1,p_keep_technical_snapshots)
        returning 1
    ) select count(*) into v_tech from deleted;

    with ranked as (
        select snapshot_id,
               row_number() over (partition by ticker order by as_of desc, updated_at desc) as rn
        from public.narrative_snapshots
    ), deleted as (
        delete from public.narrative_snapshots t
        using ranked r
        where t.snapshot_id=r.snapshot_id and r.rn > greatest(1,p_keep_narrative_snapshots)
        returning 1
    ) select count(*) into v_narr_snap from deleted;

    with ranked as (
        select narrative_event_id,
               row_number() over (
                   partition by ticker
                   order by official_verified desc,
                            coalesce(materiality_score,0) desc,
                            detected_at desc,
                            updated_at desc
               ) as rn
        from public.narrative_events
    ), deleted as (
        delete from public.narrative_events t
        using ranked r
        where t.narrative_event_id=r.narrative_event_id and r.rn > greatest(5,p_keep_narrative_events)
        returning 1
    ) select count(*) into v_narr_event from deleted;

    with ranked as (
        select event_key,
               row_number() over (
                   partition by coalesce(ticker,'__MARKET__'), event_type
                   order by coalesce(event_date, detected_at::date) desc, updated_at desc
               ) as rn
        from public.source_events
    ), deleted as (
        delete from public.source_events t
        using ranked r
        where t.event_key=r.event_key and r.rn > greatest(3,p_keep_source_events)
        returning 1
    ) select count(*) into v_source from deleted;

    delete from public.provider_health where as_of < now() - interval '45 days';
    delete from public.scan_runs
     where scan_id not in (
        select scan_id from public.scan_runs order by as_of desc limit 5
     );

    return jsonb_build_object(
        'state','FREE_TIER_HOUSEKEEPING_COMPLETE',
        'jobs_deleted',v_jobs,
        'fundamental_snapshots_deleted',v_fund,
        'multibagger_snapshots_deleted',v_multi,
        'technical_snapshots_deleted',v_tech,
        'narrative_snapshots_deleted',v_narr_snap,
        'narrative_events_deleted',v_narr_event,
        'source_events_deleted',v_source
    );
end;
$$;

revoke all on function public.scanner_free_tier_housekeeping(integer,integer,integer,integer,integer,integer,integer) from public;
grant execute on function public.scanner_free_tier_housekeeping(integer,integer,integer,integer,integer,integer,integer) to service_role;

commit;
