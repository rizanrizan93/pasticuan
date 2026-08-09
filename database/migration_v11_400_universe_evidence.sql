-- IDX Super Scanner v7.11.0: 400-universe evidence comparability and calibrated flow.
-- Prerequisite: schema_v2.sql + migrations v3 through v10.
-- Idempotent.
begin;

alter table if exists public.multibagger_snapshots
    add column if not exists multibagger_evidence_class text,
    add column if not exists multibagger_rank_eligible boolean not null default false,
    add column if not exists multibagger_score_comparability_pct numeric,
    add column if not exists multibagger_production_rank numeric,
    add column if not exists narrative_evidence_coverage_pct numeric,
    add column if not exists issuer_alignment_coverage_pct numeric,
    add column if not exists emir_method_coverage_pct numeric,
    add column if not exists distribution_severity_score numeric,
    add column if not exists distribution_penalty_points numeric,
    add column if not exists distribution_evidence_state text;

alter table if exists public.selector_snapshots
    add column if not exists relative_overlay_weight_pct numeric,
    add column if not exists selector_universe_state text,
    add column if not exists score_inflation_guard_active boolean not null default false;

create index if not exists idx_multibagger_full_evidence_rank
    on public.multibagger_snapshots (multibagger_production_rank asc nulls last)
    where multibagger_rank_eligible is true;
create index if not exists idx_multibagger_evidence_class
    on public.multibagger_snapshots (multibagger_evidence_class, as_of desc);
create index if not exists idx_selector_universe_state
    on public.selector_snapshots (selector_universe_state, as_of desc);

commit;
