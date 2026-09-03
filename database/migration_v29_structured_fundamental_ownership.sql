-- Phase 5.6 structured fundamental + issuer shareholder evidence.
-- Additive only. These tables are scanner-neutral factual/normalized evidence;
-- no score, rank, recommendation, entry, stop, or take-profit semantics are stored.

create table if not exists public.evidence_fundamental_metrics (
    provider text not null,
    ticker text not null,
    period_end date,
    statement_date date,
    metric_name text not null,
    metric_value numeric,
    metric_unit text,
    source_families text not null,
    official_verified boolean not null default false,
    source_record_hash text not null,
    lineage_state text not null,
    observed_at timestamptz not null,
    validation_state text not null,
    fetched_at timestamptz not null default now(),
    primary key (provider, ticker, metric_name, source_record_hash),
    constraint evidence_fundamental_metrics_ticker_check
      check (ticker = upper(ticker) and ticker <> ''),
    constraint evidence_fundamental_metrics_source_check
      check (source_families <> ''),
    constraint evidence_fundamental_metrics_validation_check
      check (validation_state in ('VALID', 'STALE'))
);

comment on table public.evidence_fundamental_metrics is
  'Scanner-neutral normalized fundamental metrics with explicit source families and observation time. No scanner score or recommendation semantics.';

create table if not exists public.evidence_shareholder_profiles (
    provider text not null,
    ticker text not null,
    source_period date not null,
    observed_on date not null,
    holder_identity_hash text not null,
    holder_name text not null,
    shares_held numeric,
    ownership_percentage numeric,
    holder_category text,
    source_profile_hash text not null,
    source_url text not null,
    source_verified boolean not null default false,
    validation_state text not null,
    fetched_at timestamptz not null default now(),
    primary key (provider, ticker, source_period, holder_identity_hash),
    constraint evidence_shareholder_profiles_ticker_check
      check (ticker = upper(ticker) and ticker <> ''),
    constraint evidence_shareholder_profiles_name_check
      check (holder_name <> ''),
    constraint evidence_shareholder_profiles_quantitative_check
      check (shares_held is not null or ownership_percentage is not null),
    constraint evidence_shareholder_profiles_shares_check
      check (shares_held is null or shares_held >= 0),
    constraint evidence_shareholder_profiles_percentage_check
      check (ownership_percentage is null or ownership_percentage between 0 and 100),
    constraint evidence_shareholder_profiles_validation_check
      check (validation_state in ('VALID', 'STALE'))
);

comment on table public.evidence_shareholder_profiles is
  'Issuer-reported/profile shareholder facts. This table does not establish beneficial ownership, broker identity, bandar identity, or KSEI >1%/>5% publication membership.';

create index if not exists evidence_fundamental_metrics_ticker_period_idx
  on public.evidence_fundamental_metrics (ticker, period_end desc, observed_at desc);

create index if not exists evidence_fundamental_metrics_metric_idx
  on public.evidence_fundamental_metrics (metric_name, ticker, observed_at desc);

create index if not exists evidence_shareholder_profiles_ticker_period_idx
  on public.evidence_shareholder_profiles (ticker, source_period desc, observed_on desc);

create index if not exists evidence_shareholder_profiles_holder_idx
  on public.evidence_shareholder_profiles (holder_identity_hash, ticker, observed_on desc);

do $$
declare
  table_name text;
begin
  foreach table_name in array array[
    'evidence_fundamental_metrics',
    'evidence_shareholder_profiles'
  ]
  loop
    execute format('alter table public.%I enable row level security', table_name);
    execute format('revoke all on table public.%I from public, anon, authenticated', table_name);
    execute format('revoke all on table public.%I from service_role', table_name);
    execute format('grant select, insert, update on table public.%I to service_role', table_name);
  end loop;
end
$$;
