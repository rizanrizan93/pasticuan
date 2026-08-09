-- IDX Super Scanner v7.0.0 database-first migration R1 (Supabase/PostgreSQL)
-- Prerequisite: database/schema_v2.sql has already been applied.
-- Safe to run more than once.

begin;

-- Preserve the complete, normalised scanner row so future scans can read it
-- without downloading the same provider payload again.
create table if not exists public.fundamental_cache (
    ticker text primary key,
    payload jsonb not null default '{}'::jsonb,
    source_families text,
    data_grade text,
    coverage numeric,
    statement_date text,
    source_fetched_at timestamptz,
    source_checked_at timestamptz not null,
    content_hash text,
    refresh_state text not null default 'CURRENT',
    model_version text not null,
    schema_version text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.fundamental_history_cache (
    ticker text primary key,
    payload jsonb not null default '[]'::jsonb,
    latest_period text,
    period_count integer not null default 0,
    source_families text,
    source_checked_at timestamptz not null,
    content_hash text,
    refresh_state text not null default 'CURRENT',
    model_version text not null,
    schema_version text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.forward_quality_cache (
    ticker text primary key,
    payload jsonb not null default '[]'::jsonb,
    project_count integer not null default 0,
    source_families text,
    last_verified_at timestamptz,
    source_checked_at timestamptz not null,
    content_hash text,
    refresh_state text not null default 'CURRENT',
    model_version text not null,
    schema_version text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- A durable freshness ledger separates "checked" from "changed". This avoids
-- parsing or downloading an unchanged filing on every scanner execution.
create table if not exists public.refresh_state (
    entity_key text primary key,
    entity_type text not null,
    ticker text,
    source_family text,
    state text not null,
    last_checked_at timestamptz,
    last_changed_at timestamptz,
    valid_until timestamptz,
    content_hash text,
    detail text,
    payload_metadata jsonb not null default '{}'::jsonb,
    model_version text not null,
    schema_version text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- Foundation for resumable scans. v7 records phase checkpoints so future
-- batches can continue after a Streamlit reboot instead of starting from zero.
create table if not exists public.scan_checkpoints (
    checkpoint_id text primary key,
    scan_id text not null,
    phase text not null,
    batch_number integer not null default 0,
    last_ticker text,
    status text not null,
    rows_completed integer not null default 0,
    rows_total integer not null default 0,
    detail text,
    as_of timestamptz not null,
    model_version text not null,
    schema_version text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (scan_id, phase, batch_number)
);

-- Additional lineage fields for v7 snapshots. Existing rows remain valid.
alter table public.fundamental_snapshots
    add column if not exists fundamental_fetched_at timestamptz,
    add column if not exists database_source_state text,
    add column if not exists content_hash text;

-- v2 declared this categorical field as numeric. v7 corrects the type.
-- PostgreSQL does not allow ALTER TYPE while a view depends on the column,
-- so temporarily remove and recreate the latest-snapshot view.
drop view if exists public.latest_fundamental_snapshots;

do $$
declare
    current_data_type text;
begin
    select c.data_type
      into current_data_type
      from information_schema.columns c
     where c.table_schema = 'public'
       and c.table_name = 'fundamental_snapshots'
       and c.column_name = 'fundamental_reliability';

    if current_data_type is not null and current_data_type <> 'text' then
        execute 'alter table public.fundamental_snapshots '
             || 'alter column fundamental_reliability type text '
             || 'using fundamental_reliability::text';
    end if;
end $$;

create or replace view public.latest_fundamental_snapshots
with (security_invoker = true) as
select distinct on (ticker) *
from public.fundamental_snapshots
order by ticker, as_of desc;

revoke all on table public.latest_fundamental_snapshots from anon, authenticated;
grant select on table public.latest_fundamental_snapshots to service_role;

create index if not exists idx_fundamental_cache_checked
    on public.fundamental_cache (source_checked_at desc);
create index if not exists idx_fundamental_history_cache_checked
    on public.fundamental_history_cache (source_checked_at desc);
create index if not exists idx_forward_quality_cache_checked
    on public.forward_quality_cache (source_checked_at desc);
create index if not exists idx_refresh_state_ticker_type
    on public.refresh_state (ticker, entity_type, last_checked_at desc);
create index if not exists idx_scan_checkpoints_scan_phase
    on public.scan_checkpoints (scan_id, phase, batch_number);

-- Keep updated_at accurate during PostgREST upserts.
create or replace function public.set_scanner_updated_at()
returns trigger
language plpgsql
security invoker
set search_path = public
as $$
begin
    new.updated_at = now();
    if tg_table_name = 'refresh_state'
       and new.content_hash is not distinct from old.content_hash then
        new.last_changed_at = old.last_changed_at;
    end if;
    return new;
end;
$$;

do $$
declare
    table_name text;
begin
    foreach table_name in array array[
        'fundamental_cache','fundamental_history_cache','forward_quality_cache',
        'refresh_state','scan_checkpoints'
    ]
    loop
        execute format('drop trigger if exists %I on public.%I', 'trg_' || table_name || '_updated_at', table_name);
        execute format(
            'create trigger %I before update on public.%I for each row execute function public.set_scanner_updated_at()',
            'trg_' || table_name || '_updated_at', table_name
        );
        execute format('alter table public.%I enable row level security', table_name);
        execute format('revoke all on table public.%I from anon, authenticated', table_name);
        execute format('grant all on table public.%I to service_role', table_name);
    end loop;
end $$;

grant usage on schema public to service_role;
grant usage, select on all sequences in schema public to service_role;

commit;
