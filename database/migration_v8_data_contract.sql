-- IDX Super Scanner v7.5.4 coverage-aware data contract migration
-- Prerequisite: schema_v2.sql + migrations v3 through v7.
-- Idempotent and safe to run more than once.

begin;

alter table public.multibagger_snapshots
    add column if not exists multibagger_scoring_state text,
    add column if not exists multibagger_metric_coverage_pct numeric,
    add column if not exists multibagger_metric_data_gate boolean,
    add column if not exists growth_pillar_coverage_pct numeric,
    add column if not exists profitability_pillar_coverage_pct numeric,
    add column if not exists cashflow_pillar_coverage_pct numeric,
    add column if not exists safety_pillar_coverage_pct numeric,
    add column if not exists runway_pillar_coverage_pct numeric,
    add column if not exists valuation_pillar_coverage_pct numeric;

alter table public.selector_snapshots
    add column if not exists production_selection_rank numeric,
    add column if not exists selector_rank_eligible boolean,
    add column if not exists selector_data_state text,
    add column if not exists technical_feature_coverage_pct numeric,
    add column if not exists selector_missing_feature_count integer,
    add column if not exists selector_missing_features text,
    add column if not exists effective_silent_accumulation_score numeric,
    add column if not exists silent_accumulation_confidence numeric;

create index if not exists idx_multibagger_metric_ready_rank
    on public.multibagger_snapshots
        (multibagger_metric_coverage_pct desc nulls last)
    where multibagger_metric_data_gate is true;

create index if not exists idx_selector_production_rank
    on public.selector_snapshots
        (horizon_bars, production_selection_rank asc nulls last)
    where selector_rank_eligible is true;

commit;
