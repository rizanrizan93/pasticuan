-- IDX Super Scanner v9.4.1: durable resumable chunked jobs + backend privileges.
-- Prerequisite: schema_v2.sql + migrations v3 through v11.
-- Idempotent.
begin;

create table if not exists public.scan_jobs (
    job_id text primary key,
    job_type text not null,
    job_key text not null,
    universe_hash text not null,
    config_hash text not null,
    universe_payload jsonb not null default '[]'::jsonb,
    config_payload jsonb not null default '{}'::jsonb,
    status text not null default 'PENDING',
    phase text not null default 'PENDING',
    total_items integer not null default 0,
    completed_items integer not null default 0,
    failed_items integer not null default 0,
    retry_items integer not null default 0,
    chunk_size integer not null default 20,
    max_attempts integer not null default 2,
    active_worker text,
    lease_expires_at timestamptz,
    result_summary jsonb not null default '{}'::jsonb,
    last_error text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    started_at timestamptz,
    finished_at timestamptz,
    model_version text not null,
    schema_version text not null
);

create unique index if not exists uq_scan_jobs_active_key
    on public.scan_jobs (job_key)
    where status in ('PENDING','RUNNING','PAUSED','FINALIZING');
create index if not exists idx_scan_jobs_resume
    on public.scan_jobs (job_type, universe_hash, config_hash, updated_at desc);
create index if not exists idx_scan_jobs_status
    on public.scan_jobs (status, updated_at desc);

create table if not exists public.scan_job_items (
    item_key text primary key,
    job_id text not null references public.scan_jobs(job_id) on delete cascade,
    ticker text not null,
    phase text not null,
    sequence_no integer not null default 0,
    status text not null default 'PENDING',
    attempt_count integer not null default 0,
    max_attempts integer not null default 2,
    next_attempt_at timestamptz,
    lease_owner text,
    lease_expires_at timestamptz,
    last_error text,
    result_payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    started_at timestamptz,
    completed_at timestamptz,
    model_version text not null,
    schema_version text not null
);

create index if not exists idx_scan_job_items_claim
    on public.scan_job_items (job_id, phase, status, next_attempt_at, sequence_no);
create index if not exists idx_scan_job_items_lease
    on public.scan_job_items (job_id, lease_expires_at)
    where status = 'RUNNING';
create index if not exists idx_scan_job_items_ticker
    on public.scan_job_items (job_id, ticker);

create table if not exists public.scan_job_artifacts (
    artifact_key text primary key,
    job_id text not null references public.scan_jobs(job_id) on delete cascade,
    artifact_type text not null,
    chunk_number integer not null default 0,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    model_version text not null,
    schema_version text not null
);

create index if not exists idx_scan_job_artifacts_job
    on public.scan_job_artifacts (job_id, artifact_type, chunk_number);


-- Backend-only privileges. Supabase service/secret keys still require explicit
-- table grants; RLS bypass alone does not grant SELECT/INSERT/UPDATE/DELETE.
alter table public.scan_jobs enable row level security;
alter table public.scan_job_items enable row level security;
alter table public.scan_job_artifacts enable row level security;

revoke all on table public.scan_jobs from public, anon, authenticated;
revoke all on table public.scan_job_items from public, anon, authenticated;
revoke all on table public.scan_job_artifacts from public, anon, authenticated;

grant usage on schema public to service_role;
grant select, insert, update, delete on table public.scan_jobs to service_role;
grant select, insert, update, delete on table public.scan_job_items to service_role;
grant select, insert, update, delete on table public.scan_job_artifacts to service_role;

-- Keep future postgres-owned scanner tables backend-readable without exposing
-- them to anon/authenticated roles.
alter default privileges for role postgres in schema public
    grant select, insert, update, delete on tables to service_role;

create or replace function public.claim_scan_job_items(
    p_job_id text,
    p_phase text,
    p_limit integer,
    p_worker_id text,
    p_lease_seconds integer default 300
)
returns setof public.scan_job_items
language plpgsql
security definer
set search_path = public
as $$
begin
    -- An interrupted worker never strands a ticker. Expired leases return to RETRY.
    update public.scan_job_items
       set status = case when attempt_count >= max_attempts then 'FAILED' else 'RETRY' end,
           lease_owner = null,
           lease_expires_at = null,
           next_attempt_at = case when attempt_count >= max_attempts then null else now() end,
           last_error = coalesce(last_error, 'LEASE_EXPIRED'),
           updated_at = now()
     where job_id = p_job_id
       and phase = p_phase
       and status = 'RUNNING'
       and lease_expires_at is not null
       and lease_expires_at < now();

    return query
    with candidates as (
        select item_key
          from public.scan_job_items
         where job_id = p_job_id
           and phase = p_phase
           and status in ('PENDING','RETRY')
           and attempt_count < max_attempts
           and (next_attempt_at is null or next_attempt_at <= now())
         order by sequence_no, ticker
         for update skip locked
         limit greatest(1, p_limit)
    )
    update public.scan_job_items i
       set status = 'RUNNING',
           attempt_count = i.attempt_count + 1,
           lease_owner = p_worker_id,
           lease_expires_at = now() + make_interval(secs => greatest(30, p_lease_seconds)),
           started_at = coalesce(i.started_at, now()),
           updated_at = now()
      from candidates c
     where i.item_key = c.item_key
    returning i.*;
end;
$$;

create or replace function public.refresh_scan_job_counters(p_job_id text)
returns public.scan_jobs
language plpgsql
security definer
set search_path = public
as $$
declare
    v_job public.scan_jobs;
begin
    update public.scan_jobs j
       set completed_items = x.completed_items,
           failed_items = x.failed_items,
           retry_items = x.retry_items,
           updated_at = now()
      from (
        select
            count(*) filter (where status = 'COMPLETE')::integer as completed_items,
            count(*) filter (where status = 'FAILED')::integer as failed_items,
            count(*) filter (where status = 'RETRY')::integer as retry_items
          from public.scan_job_items
         where job_id = p_job_id
      ) x
     where j.job_id = p_job_id
    returning j.* into v_job;
    return v_job;
end;
$$;

create or replace function public.claim_scan_job_lease(
    p_job_id text,
    p_worker_id text,
    p_lease_seconds integer default 300
)
returns public.scan_jobs
language plpgsql
security definer
set search_path = public
as $$
declare
    v_job public.scan_jobs;
begin
    update public.scan_jobs
       set active_worker = p_worker_id,
           lease_expires_at = now() + make_interval(secs => greatest(60, p_lease_seconds)),
           status = case when status in ('PENDING','PAUSED') then 'RUNNING' else status end,
           started_at = coalesce(started_at, now()),
           updated_at = now()
     where job_id = p_job_id
       and status in ('PENDING','RUNNING','PAUSED','FINALIZING')
       and (active_worker is null or active_worker = p_worker_id or lease_expires_at is null or lease_expires_at < now())
    returning * into v_job;
    return v_job;
end;
$$;

create or replace function public.renew_scan_job_leases(
    p_job_id text,
    p_worker_id text,
    p_lease_seconds integer default 300
)
returns public.scan_jobs
language plpgsql
security definer
set search_path = public
as $$
declare
    v_job public.scan_jobs;
begin
    update public.scan_job_items
       set lease_expires_at = now() + make_interval(secs => greatest(60, p_lease_seconds)),
           updated_at = now()
     where job_id = p_job_id
       and status = 'RUNNING'
       and lease_owner = p_worker_id;

    update public.scan_jobs
       set lease_expires_at = now() + make_interval(secs => greatest(60, p_lease_seconds)),
           updated_at = now()
     where job_id = p_job_id
       and status in ('RUNNING','FINALIZING')
       and active_worker = p_worker_id
    returning * into v_job;
    return v_job;
end;
$$;

revoke all on function public.claim_scan_job_items(text,text,integer,text,integer) from public;
revoke all on function public.refresh_scan_job_counters(text) from public;
revoke all on function public.claim_scan_job_lease(text,text,integer) from public;
revoke all on function public.renew_scan_job_leases(text,text,integer) from public;
grant execute on function public.claim_scan_job_items(text,text,integer,text,integer) to service_role;
grant execute on function public.refresh_scan_job_counters(text) to service_role;
grant execute on function public.claim_scan_job_lease(text,text,integer) to service_role;
grant execute on function public.renew_scan_job_leases(text,text,integer) to service_role;

commit;
