-- v9.8.18 evidence governance / OOS calibration / provider negative cache
-- Applied to production on 2026-08-15. Idempotent for reproducible environments.

alter table public.source_documents add column if not exists source_https_verified boolean not null default false;
alter table public.source_documents add column if not exists entity_match_verified boolean not null default false;
alter table public.source_documents add column if not exists entity_match_method text;
alter table public.source_documents add column if not exists source_tier text;

alter table public.project_events add column if not exists source_quorum_count integer not null default 0;
alter table public.project_events add column if not exists entity_match_verified boolean not null default false;
alter table public.project_events add column if not exists evidence_date date;

alter table public.management_roles add column if not exists source_quorum_count integer not null default 0;
alter table public.management_roles add column if not exists source_quorum_verified boolean not null default false;
alter table public.management_roles add column if not exists entity_match_verified boolean not null default false;
alter table public.management_roles add column if not exists source_family text;

alter table public.ownership_events add column if not exists source_quorum_count integer not null default 0;
alter table public.ownership_events add column if not exists source_quorum_verified boolean not null default false;
alter table public.ownership_events add column if not exists entity_match_verified boolean not null default false;
alter table public.ownership_events add column if not exists source_family text;

alter table public.corporate_events add column if not exists source_quorum_count integer not null default 0;
alter table public.corporate_events add column if not exists source_quorum_verified boolean not null default false;
alter table public.corporate_events add column if not exists entity_match_verified boolean not null default false;
alter table public.corporate_events add column if not exists source_family text;

create table if not exists public.provider_negative_cache (
  provider text not null,
  request_family text not null,
  cache_key text not null,
  failure_class text not null,
  http_status integer,
  retry_after timestamptz not null,
  hit_count integer not null default 1,
  last_error text,
  last_checked_at timestamptz not null default now(),
  last_success_at timestamptz,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key(provider, request_family, cache_key)
);
create index if not exists idx_provider_negative_cache_retry on public.provider_negative_cache(retry_after);
create index if not exists idx_provider_negative_cache_provider on public.provider_negative_cache(provider, request_family);

create table if not exists public.guardrail_calibrations (
  calibration_id text primary key,
  strategy text not null,
  model_version text not null,
  calibration_state text not null,
  trained_through date,
  evaluation_start date,
  evaluation_end date,
  sample_count integer not null default 0,
  distinct_signal_dates integer not null default 0,
  fold_count integer not null default 0,
  objective_name text not null default 'RETURN_DRAWDOWN_STABILITY',
  objective_value numeric,
  parameters jsonb not null default '{}'::jsonb,
  metrics jsonb not null default '{}'::jsonb,
  active boolean not null default false,
  produced_at timestamptz not null default now(),
  created_at timestamptz not null default now()
);
create index if not exists idx_guardrail_calibrations_active on public.guardrail_calibrations(strategy, active, produced_at desc);

alter table public.provider_negative_cache enable row level security;
alter table public.guardrail_calibrations enable row level security;
revoke all on public.provider_negative_cache from anon, authenticated;
revoke all on public.guardrail_calibrations from anon, authenticated;

do $$ begin
  if not exists (select 1 from pg_constraint where conname='source_documents_verified_https_ck') then
    alter table public.source_documents add constraint source_documents_verified_https_ck check (not source_https_verified or source_url like 'https://%');
  end if;
  if not exists (select 1 from pg_constraint where conname='project_events_quorum_count_ck') then
    alter table public.project_events add constraint project_events_quorum_count_ck check (not coalesce(project_source_quorum_verified,false) or source_quorum_count >= 2);
  end if;
end $$;
