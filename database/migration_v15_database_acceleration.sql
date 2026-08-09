-- IDX Super Scanner v9.8.2: ALL_ELIGIBLE_LITE database acceleration.
-- Prerequisite: migrations through v14.
-- Idempotent. Existing legacy OHLCV JSON remains readable during conversion.
begin;

alter table if exists public.ohlcv_daily_cache
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

create index if not exists idx_scanner_feature_cache_last_bar
    on public.scanner_feature_cache (last_bar_date desc);
create index if not exists idx_scanner_feature_cache_state
    on public.scanner_feature_cache (feature_state, updated_at desc);

alter table public.scanner_feature_cache enable row level security;
revoke all on table public.scanner_feature_cache from public, anon, authenticated;
grant select, insert, update, delete on table public.scanner_feature_cache to service_role;
grant usage on schema public to service_role;

drop trigger if exists trg_scanner_feature_cache_updated_at on public.scanner_feature_cache;
create trigger trg_scanner_feature_cache_updated_at
before update on public.scanner_feature_cache
for each row execute function public.set_scanner_updated_at();

commit;
