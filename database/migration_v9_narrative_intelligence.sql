-- IDX Super Scanner v7.6.0 point-in-time Narrative Intelligence migration.
-- Prerequisite: schema_v2.sql + migrations v3 through v8.
-- Idempotent and safe to run more than once.

begin;

create table if not exists public.narrative_events (
    narrative_event_id text primary key,
    ticker text not null,
    event_date date,
    detected_at timestamptz not null,
    event_type text not null,
    event_family text,
    headline text not null,
    summary text,
    source_url text,
    source_family text,
    source_quality_score numeric,
    official_verified boolean not null default false,
    materiality_score numeric,
    novelty_score numeric,
    financial_bridge_score numeric,
    narrative_decay_weight numeric,
    catalyst_proximity_score numeric,
    event_strength_score numeric,
    signed_event_strength numeric,
    event_age_days numeric,
    impact_direction text,
    impact_sign integer,
    content_hash text,
    event_cluster_key text,
    event_evidence_state text,
    event_active boolean,
    future_detection_invalid boolean not null default false,
    detection_time_source text,
    narrative_engine_version text,
    model_version text not null,
    schema_version text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.narrative_event_outcomes (
    narrative_outcome_id text primary key,
    narrative_event_id text not null,
    ticker text not null,
    event_type text,
    event_family text,
    signal_timestamp timestamptz not null,
    signal_date date,
    anchor_date date,
    entry_reference numeric,
    roundtrip_cost_pct numeric,
    stock_return_5d_pct numeric,
    benchmark_return_5d_pct numeric,
    net_excess_return_5d_pct numeric,
    mfe_5d_pct numeric,
    mae_5d_pct numeric,
    converted_5d boolean,
    stock_return_20d_pct numeric,
    benchmark_return_20d_pct numeric,
    net_excess_return_20d_pct numeric,
    mfe_20d_pct numeric,
    mae_20d_pct numeric,
    converted_20d boolean,
    stock_return_60d_pct numeric,
    benchmark_return_60d_pct numeric,
    net_excess_return_60d_pct numeric,
    mfe_60d_pct numeric,
    mae_60d_pct numeric,
    converted_60d boolean,
    outcome_status text not null default 'OPEN_NO_FORWARD_BARS',
    resolved_at timestamptz,
    narrative_engine_version text,
    model_version text not null,
    schema_version text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.narrative_snapshots (
    snapshot_id text primary key,
    scan_id text,
    ticker text not null,
    as_of timestamptz not null,
    narrative_as_of timestamptz,
    narrative_state text,
    narrative_score numeric,
    narrative_effective_score numeric,
    narrative_evidence_coverage_pct numeric,
    narrative_event_count integer,
    narrative_active_event_count integer,
    narrative_event_cluster_count integer,
    narrative_corroborated_cluster_count integer,
    narrative_positive_event_count integer,
    narrative_negative_event_count integer,
    narrative_official_event_count integer,
    latest_narrative_event text,
    latest_narrative_event_type text,
    latest_narrative_event_date date,
    narrative_source_quality_score numeric,
    narrative_novelty_score numeric,
    narrative_financial_bridge_score numeric,
    issuer_alignment_score numeric,
    issuer_alignment_effective_score numeric,
    issuer_alignment_coverage_pct numeric,
    issuer_alignment_state text,
    issuer_alignment_positive_events integer,
    issuer_alignment_negative_events integer,
    retail_adoption_stage text,
    retail_adoption_proxy_score numeric,
    retail_adoption_proxy_coverage_pct numeric,
    retail_proxy_disclaimer text,
    narrative_crowding_risk_score numeric,
    narrative_flow_convergence_score numeric,
    narrative_flow_effective_score numeric,
    narrative_flow_convergence_coverage_pct numeric,
    narrative_flow_convergence_state text,
    narrative_silent_integration_state text,
    narrative_contradiction_count integer,
    narrative_hard_block boolean not null default false,
    narrative_primary_reason text,
    narrative_primary_risk text,
    narrative_overlay_reliability_pct numeric,
    narrative_swing_overlay_reliability_pct numeric,
    narrative_growth_rank_adjustment numeric,
    narrative_turnaround_rank_adjustment numeric,
    narrative_swing_rank_adjustment numeric,
    narrative_news_collection_state text,
    narrative_items_reviewed integer,
    narrative_flow_proxy_score numeric,
    narrative_conversion_rate_5d_pct numeric,
    narrative_conversion_effective_5d_score numeric,
    narrative_conversion_expectancy_5d_pct numeric,
    narrative_conversion_resolved_5d integer,
    narrative_conversion_state_5d text,
    narrative_conversion_rate_20d_pct numeric,
    narrative_conversion_effective_20d_score numeric,
    narrative_conversion_expectancy_20d_pct numeric,
    narrative_conversion_resolved_20d integer,
    narrative_conversion_state_20d text,
    narrative_conversion_rate_60d_pct numeric,
    narrative_conversion_effective_60d_score numeric,
    narrative_conversion_expectancy_60d_pct numeric,
    narrative_conversion_resolved_60d integer,
    narrative_conversion_state_60d text,
    narrative_engine_version text,
    model_version text not null,
    schema_version text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_narrative_events_ticker_detected
    on public.narrative_events (ticker, detected_at desc);
create index if not exists idx_narrative_events_type_detected
    on public.narrative_events (event_type, detected_at desc);
create index if not exists idx_narrative_events_official_risk
    on public.narrative_events
        (ticker, materiality_score desc, detected_at desc)
    where official_verified is true and impact_sign < 0;
create index if not exists idx_narrative_outcomes_ticker_signal
    on public.narrative_event_outcomes (ticker, signal_timestamp desc);
create index if not exists idx_narrative_outcomes_open
    on public.narrative_event_outcomes (outcome_status, signal_date);
create index if not exists idx_narrative_snapshots_ticker_asof
    on public.narrative_snapshots (ticker, as_of desc);
create index if not exists idx_narrative_snapshots_convergence
    on public.narrative_snapshots
        (narrative_flow_effective_score desc nulls last, as_of desc)
    where narrative_hard_block is false;

do $$
declare
    table_name text;
begin
    foreach table_name in array array[
        'narrative_events',
        'narrative_event_outcomes',
        'narrative_snapshots'
    ]
    loop
        execute format(
            'drop trigger if exists %I on public.%I',
            'trg_' || table_name || '_updated_at',
            table_name
        );
        execute format(
            'create trigger %I before update on public.%I '
            'for each row execute function public.set_scanner_updated_at()',
            'trg_' || table_name || '_updated_at',
            table_name
        );
        execute format(
            'alter table public.%I enable row level security',
            table_name
        );
        execute format(
            'revoke all on table public.%I from anon, authenticated',
            table_name
        );
    end loop;
end $$;

commit;
