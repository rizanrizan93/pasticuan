-- IDX Super Scanner v9.5.0 persistent OHLCV cache
-- Prerequisite: migration_v12_resumable_scan_jobs.sql
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
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_ohlcv_daily_cache_last_bar
    on public.ohlcv_daily_cache (last_bar_date desc);
create index if not exists idx_ohlcv_daily_cache_refresh
    on public.ohlcv_daily_cache (refresh_state, source_checked_at desc);

alter table public.ohlcv_daily_cache enable row level security;
revoke all on table public.ohlcv_daily_cache from public, anon, authenticated;
grant select, insert, update, delete on table public.ohlcv_daily_cache to service_role;
grant usage on schema public to service_role;

-- Self-contained updated_at trigger; safe even when an older migration was skipped.
create or replace function public.set_scanner_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists trg_ohlcv_daily_cache_updated_at on public.ohlcv_daily_cache;
create trigger trg_ohlcv_daily_cache_updated_at
before update on public.ohlcv_daily_cache
for each row execute function public.set_scanner_updated_at();

commit;
