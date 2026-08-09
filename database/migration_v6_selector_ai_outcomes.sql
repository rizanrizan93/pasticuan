-- IDX Super Scanner v7.3.0 selector + execution outcome migration
-- Prerequisite: schema_v2.sql + migrations v3, v4, and v5.
-- Idempotent and safe to run more than once.

begin;

create table if not exists public.ai_execution_outcomes (
    signal_id text primary key,
    ticker text not null,
    strategy text,
    signal_date timestamptz not null,
    memory_state text not null,
    result text,
    fill_date timestamptz,
    exit_date timestamptz,
    resolved_at timestamptz,
    outcome_quality text,
    no_fill_reason text,
    entry numeric,
    trigger_price numeric,
    stop_loss numeric,
    tp1 numeric,
    tp2 numeric,
    fill_price numeric,
    exit_price numeric,
    fill_delay_bars integer,
    fill_slippage_pct numeric,
    filled boolean,
    tp1_hit boolean,
    tp1_before_sl boolean,
    tp2_hit boolean,
    outcome_ambiguous boolean,
    r_multiple numeric,
    expectancy_after_cost_r numeric,
    mfe_r numeric,
    mae_r numeric,
    mfe_pct numeric,
    mae_pct numeric,
    gross_return_pct numeric,
    net_return_pct numeric,
    roundtrip_cost_pct numeric,
    cost_r numeric,
    ai_version text,
    model_version text not null,
    schema_version text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.selector_snapshots (
    snapshot_id text primary key,
    ticker text not null,
    as_of timestamptz not null,
    horizon text,
    horizon_bars integer not null,
    selection_rank numeric,
    swing_selection_score numeric,
    multibagger_timing_selector_score numeric,
    technical_selection_score numeric,
    silent_accumulation_score numeric,
    relative_strength_score numeric,
    expected_excess_return_pct numeric,
    outperform_probability_pct numeric,
    selector_score numeric,
    ai_weight_pct numeric,
    model_state text,
    champion_model text,
    selected_reason text,
    selection_risks text,
    model_version text not null,
    schema_version text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint selector_probability_bounds check (
        outperform_probability_pct is null
        or outperform_probability_pct between 0 and 100
    ),
    constraint selector_ai_weight_bounds check (
        ai_weight_pct is null or ai_weight_pct between 0 and 100
    )
);

create table if not exists public.selector_outcomes (
    outcome_id text primary key,
    snapshot_id text,
    ticker text not null,
    signal_date timestamptz not null,
    horizon text,
    horizon_bars integer not null,
    predicted_excess_return_pct numeric,
    outperform_probability_pct numeric,
    selector_score numeric,
    model_state text,
    champion_model text,
    outcome_status text not null,
    resolved_at timestamptz,
    stock_return_pct numeric,
    benchmark_return_pct numeric,
    net_excess_return_pct numeric,
    outperformed_after_cost boolean,
    model_version text not null,
    schema_version text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint selector_outcome_probability_bounds check (
        outperform_probability_pct is null
        or outperform_probability_pct between 0 and 100
    )
);

create table if not exists public.selector_model_evaluations (
    evaluation_id text primary key,
    as_of timestamptz not null,
    horizon text,
    horizon_bars integer not null,
    model text not null,
    selector_version text,
    training_rows integer,
    model_fit_rows integer,
    calibration_rows integer,
    evaluation_rows integer,
    evaluation_dates integer,
    evaluation_tickers integer,
    brier_score numeric,
    baseline_brier_score numeric,
    brier_skill_pct numeric,
    net_excess_expectancy_pct numeric,
    net_absolute_expectancy_pct numeric,
    topk_hit_rate_pct numeric,
    spearman_ic numeric,
    max_drawdown_pct numeric,
    challenger_rank integer,
    promoted_model text,
    walkforward_best_model text,
    ai_promotion_state text,
    ai_can_influence boolean not null default false,
    model_version text not null,
    schema_version text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_ai_execution_ticker_date
    on public.ai_execution_outcomes (ticker, signal_date desc);
create index if not exists idx_ai_execution_state
    on public.ai_execution_outcomes (memory_state, strategy, signal_date desc);
create index if not exists idx_ai_execution_resolved
    on public.ai_execution_outcomes (resolved_at desc)
    where memory_state = 'RESOLVED';

create index if not exists idx_selector_snapshots_ticker_date
    on public.selector_snapshots (ticker, as_of desc, horizon_bars);
create index if not exists idx_selector_snapshots_model
    on public.selector_snapshots (model_state, champion_model, as_of desc);

create index if not exists idx_selector_outcomes_ticker_date
    on public.selector_outcomes (ticker, signal_date desc, horizon_bars);
create index if not exists idx_selector_outcomes_open
    on public.selector_outcomes (signal_date, horizon_bars)
    where outcome_status = 'OPEN';

create index if not exists idx_selector_evaluations_date
    on public.selector_model_evaluations (as_of desc, horizon_bars, challenger_rank);
create index if not exists idx_selector_evaluations_promotion
    on public.selector_model_evaluations (ai_promotion_state, model, as_of desc);

drop trigger if exists trg_ai_execution_outcomes_updated_at
    on public.ai_execution_outcomes;
create trigger trg_ai_execution_outcomes_updated_at
    before update on public.ai_execution_outcomes
    for each row execute function public.set_scanner_updated_at();

drop trigger if exists trg_selector_snapshots_updated_at
    on public.selector_snapshots;
create trigger trg_selector_snapshots_updated_at
    before update on public.selector_snapshots
    for each row execute function public.set_scanner_updated_at();

drop trigger if exists trg_selector_outcomes_updated_at
    on public.selector_outcomes;
create trigger trg_selector_outcomes_updated_at
    before update on public.selector_outcomes
    for each row execute function public.set_scanner_updated_at();

drop trigger if exists trg_selector_model_evaluations_updated_at
    on public.selector_model_evaluations;
create trigger trg_selector_model_evaluations_updated_at
    before update on public.selector_model_evaluations
    for each row execute function public.set_scanner_updated_at();

alter table public.ai_execution_outcomes enable row level security;
alter table public.selector_snapshots enable row level security;
alter table public.selector_outcomes enable row level security;
alter table public.selector_model_evaluations enable row level security;

revoke all on table public.ai_execution_outcomes from anon, authenticated;
revoke all on table public.selector_snapshots from anon, authenticated;
revoke all on table public.selector_outcomes from anon, authenticated;
revoke all on table public.selector_model_evaluations from anon, authenticated;

grant all on table public.ai_execution_outcomes to service_role;
grant all on table public.selector_snapshots to service_role;
grant all on table public.selector_outcomes to service_role;
grant all on table public.selector_model_evaluations to service_role;
grant usage on schema public to service_role;
grant usage, select on all sequences in schema public to service_role;

commit;
