-- IDX Super Scanner v9.8.2 Hotfix 7: persistence/data-contract integrity.
-- Prerequisite: migrations through v12. This migration also repairs a missing
-- v13/v15 OHLCV/feature-cache deployment without deleting legacy data.
-- Idempotent and safe to run more than once.

begin;

create table if not exists public.ohlcv_daily_cache (
    ticker text primary key,
    payload jsonb not null default '[]'::jsonb,
    bar_count integer not null default 0,
    first_bar_date date,
    last_bar_date date,
    source_family text,
    source_tier text,
    source_checked_at timestamptz not null default now(),
    refresh_state text not null default 'MISSING',
    last_error text,
    content_hash text,
    model_version text not null,
    schema_version text not null,
    payload_compact text,
    payload_codec text,
    compact_bar_count integer,
    compact_hash text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

alter table public.ohlcv_daily_cache
    add column if not exists payload_compact text,
    add column if not exists payload_codec text,
    add column if not exists compact_bar_count integer,
    add column if not exists compact_hash text;

create table if not exists public.scanner_feature_cache (
    ticker text primary key,
    last_bar_date date,
    feature_state text not null default 'PARTIAL',
    source_tier text,
    scanner_version text not null,
    feature_schema_version text not null,
    payload jsonb not null default '{}'::jsonb,
    content_hash text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_ohlcv_daily_cache_last_bar
    on public.ohlcv_daily_cache (last_bar_date desc);
create index if not exists idx_ohlcv_daily_cache_refresh
    on public.ohlcv_daily_cache (refresh_state, source_checked_at desc);
create index if not exists idx_scanner_feature_cache_last_bar
    on public.scanner_feature_cache (last_bar_date desc);
create index if not exists idx_scanner_feature_cache_state
    on public.scanner_feature_cache (feature_state, updated_at desc);

-- Generic trigger: never dereference a column that is absent from the table.
create or replace function public.set_scanner_updated_at()
returns trigger
language plpgsql
security invoker
set search_path = pg_catalog
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

-- refresh_state has the additional semantic-change rule and therefore gets a
-- table-specific trigger function with a statically valid NEW row type.
create or replace function public.set_refresh_state_updated_at()
returns trigger
language plpgsql
security invoker
set search_path = pg_catalog
as $$
begin
    new.updated_at = now();
    if new.content_hash is not distinct from old.content_hash then
        new.last_changed_at = old.last_changed_at;
    end if;
    return new;
end;
$$;

drop trigger if exists trg_refresh_state_updated_at on public.refresh_state;
create trigger trg_refresh_state_updated_at
before update on public.refresh_state
for each row execute function public.set_refresh_state_updated_at();

drop trigger if exists trg_ohlcv_daily_cache_updated_at on public.ohlcv_daily_cache;
create trigger trg_ohlcv_daily_cache_updated_at
before update on public.ohlcv_daily_cache
for each row execute function public.set_scanner_updated_at();

drop trigger if exists trg_scanner_feature_cache_updated_at on public.scanner_feature_cache;
create trigger trg_scanner_feature_cache_updated_at
before update on public.scanner_feature_cache
for each row execute function public.set_scanner_updated_at();

alter table public.ohlcv_daily_cache enable row level security;
alter table public.scanner_feature_cache enable row level security;
revoke all on table public.ohlcv_daily_cache from public, anon, authenticated;
revoke all on table public.scanner_feature_cache from public, anon, authenticated;
grant select, insert, update, delete on table public.ohlcv_daily_cache to service_role;
grant select, insert, update, delete on table public.scanner_feature_cache to service_role;
grant usage on schema public to service_role;

-- Trigger functions do not need to be API-callable. The event-trigger helper
-- is privileged DDL and must also remain inaccessible to application roles.
revoke execute on function public.set_scanner_updated_at() from public, anon, authenticated;
revoke execute on function public.set_refresh_state_updated_at() from public, anon, authenticated;
do $$
begin
    if to_regprocedure('public.rls_auto_enable()') is not null then
        execute 'revoke execute on function public.rls_auto_enable() from public, anon, authenticated';
    end if;
end;
$$;

-- Repair legacy text cache metadata. JSON payload keys are set to JSON null so
-- old rows cannot reintroduce the literal string "NaT" on cache reads.
update public.fundamental_cache
set statement_date = null,
    payload = case
        when upper(btrim(coalesce(payload->>'latest_statement_date', ''))) in ('', '<NA>', 'NA', 'N/A', 'NAN', 'NAT', 'NONE', 'NULL')
        then jsonb_set(payload, '{latest_statement_date}', 'null'::jsonb, false)
        else payload
    end
where upper(btrim(coalesce(statement_date, ''))) in ('', '<NA>', 'NA', 'N/A', 'NAN', 'NAT', 'NONE', 'NULL');

update public.fundamental_history_cache
set latest_period = null
where upper(btrim(coalesce(latest_period, ''))) in ('', '<NA>', 'NA', 'N/A', 'NAN', 'NAT', 'NONE', 'NULL');

commit;
