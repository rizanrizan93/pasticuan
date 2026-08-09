-- IDX Super Scanner v9.8.0: official reconciliation + guarded real-money authorization.
-- Prerequisite: schema_v2.sql and migrations through v13.
-- Idempotent.
begin;

alter table if exists public.fundamental_snapshots
    add column if not exists fundamental_official_source_coverage_pct numeric,
    add column if not exists fundamental_reconciliation_state text,
    add column if not exists fundamental_cashflow_statement_coverage_pct numeric,
    add column if not exists fundamental_consensus_score numeric;

alter table if exists public.multibagger_snapshots
    add column if not exists v9_next_leader_score numeric,
    add column if not exists final_score numeric,
    add column if not exists real_money_authorization_state text,
    add column if not exists real_money_authorization_pass boolean,
    add column if not exists real_money_authorization_blockers text,
    add column if not exists real_money_manual_checks text,
    add column if not exists real_money_risk_budget_cap_pct numeric,
    add column if not exists real_money_risk_budget_idr numeric,
    add column if not exists real_money_risk_per_share numeric,
    add column if not exists real_money_risk_lots_cap integer,
    add column if not exists fundamental_conviction_cap numeric,
    add column if not exists fundamental_score_cap_reason text,
    add column if not exists fundamental_data_quality_score numeric,
    add column if not exists fundamental_cashflow_state text,
    add column if not exists fundamental_leverage_risk_state text,
    add column if not exists fundamental_official_state text,
    add column if not exists fundamental_official_verified boolean,
    add column if not exists fundamental_official_source_coverage_pct numeric,
    add column if not exists fundamental_consensus_score numeric,
    add column if not exists fundamental_history_coverage_pct numeric,
    add column if not exists fundamental_cashflow_coverage_pct numeric,
    add column if not exists market_regime text,
    add column if not exists market_context_score numeric,
    add column if not exists market_context_coverage_pct numeric,
    add column if not exists market_context_provenance_state text,
    add column if not exists independent_price_verified boolean,
    add column if not exists inventory_multi_horizon_score numeric,
    add column if not exists distribution_risk_score numeric,
    add column if not exists reaccumulation_quality_score numeric,
    add column if not exists anti_chase_gate boolean;

create index if not exists idx_multibagger_real_money_state
    on public.multibagger_snapshots (real_money_authorization_state, as_of desc);
create index if not exists idx_multibagger_market_regime
    on public.multibagger_snapshots (market_regime, as_of desc);

commit;
