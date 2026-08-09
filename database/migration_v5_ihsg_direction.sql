-- IDX Super Scanner v7.2.0 IHSG direction and point-in-time outcome migration
-- Prerequisite: schema_v2.sql + migration_v3 + migration_v4
-- Idempotent and safe to run more than once.

begin;

alter table public.research_outcomes
    add column if not exists forward_return_1d numeric;

create table if not exists public.ihsg_direction_snapshots (
    snapshot_id text primary key,
    scan_id text,
    ticker text not null default '^JKSE',
    as_of timestamptz not null,
    horizon text,
    horizon_bars integer not null,
    raw_direction text,
    prediction_state text,
    prob_up_pct numeric,
    prob_sideways_pct numeric,
    prob_down_pct numeric,
    confidence_pct numeric,
    expected_return_pct numeric,
    return_p25_pct numeric,
    return_p75_pct numeric,
    neutral_band_pct numeric,
    analogue_count integer,
    effective_analogue_count numeric,
    features_used integer,
    median_distance numeric,
    validation_state text,
    validation_predictions integer,
    directional_validation_predictions integer,
    directional_accuracy_pct numeric,
    directional_accuracy_ci_low_pct numeric,
    validation_coverage_pct numeric,
    brier_score numeric,
    baseline_brier_score numeric,
    brier_skill_pct numeric,
    actionable boolean not null default false,
    data_state text,
    eod_final boolean not null default false,
    benchmark_close numeric,
    regime text,
    regime_score numeric,
    consensus_direction text,
    consensus_confidence numeric,
    risk_budget_multiplier numeric,
    risk_action text,
    feature_coverage_pct numeric,
    breadth_member_count integer,
    breadth_ema50_pct numeric,
    feature_hash text,
    model_version text not null,
    schema_version text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint ihsg_direction_probability_bounds check (
        (prob_up_pct is null or prob_up_pct between 0 and 100)
        and (prob_sideways_pct is null or prob_sideways_pct between 0 and 100)
        and (prob_down_pct is null or prob_down_pct between 0 and 100)
    ),
    constraint ihsg_direction_risk_cap check (
        risk_budget_multiplier is null or risk_budget_multiplier between 0 and 1
    )
);

create index if not exists idx_ihsg_direction_asof
    on public.ihsg_direction_snapshots (as_of desc, horizon_bars);
create index if not exists idx_ihsg_direction_validation
    on public.ihsg_direction_snapshots (validation_state, prediction_state, as_of desc);
create index if not exists idx_ihsg_direction_model
    on public.ihsg_direction_snapshots (model_version, horizon_bars, as_of desc);

drop trigger if exists trg_ihsg_direction_snapshots_updated_at
    on public.ihsg_direction_snapshots;
create trigger trg_ihsg_direction_snapshots_updated_at
    before update on public.ihsg_direction_snapshots
    for each row execute function public.set_scanner_updated_at();

alter table public.ihsg_direction_snapshots enable row level security;
revoke all on table public.ihsg_direction_snapshots from anon, authenticated;
grant all on table public.ihsg_direction_snapshots to service_role;
grant usage on schema public to service_role;
grant usage, select on all sequences in schema public to service_role;

commit;
