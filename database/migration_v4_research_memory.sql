-- IDX Super Scanner v7.1.0 research-memory and event-aware migration
-- Prerequisite: schema_v2.sql + migration_v3_database_first.sql
-- Idempotent and safe to run more than once.

begin;

alter table public.fundamental_cache
    add column if not exists parser_version text,
    add column if not exists event_fingerprint text,
    add column if not exists next_check_at timestamptz,
    add column if not exists refresh_reason text;

alter table public.fundamental_history_cache
    add column if not exists parser_version text,
    add column if not exists event_fingerprint text,
    add column if not exists next_check_at timestamptz,
    add column if not exists refresh_reason text;

alter table public.forward_quality_cache
    add column if not exists parser_version text,
    add column if not exists event_fingerprint text,
    add column if not exists next_check_at timestamptz,
    add column if not exists refresh_reason text;

alter table public.refresh_state
    add column if not exists parser_version text,
    add column if not exists event_fingerprint text,
    add column if not exists refresh_reason text;

alter table public.multibagger_snapshots
    add column if not exists silent_accumulation_raw_score numeric,
    add column if not exists silent_accumulation_liquidity_adjustment numeric,
    add column if not exists silent_accumulation_liquidity_min_confirmation numeric,
    add column if not exists silent_accumulation_calibration_policy text,
    add column if not exists liquidity_bucket text;

alter table public.eoff_predictions
    add column if not exists best_buy_raw_date text,
    add column if not exists best_buy_calendar_state text,
    add column if not exists best_buy_calendar_verified boolean,
    add column if not exists best_buy_date_adjustment_days integer,
    add column if not exists eoff_fib_unique_anchor_count integer,
    add column if not exists eoff_fib_unique_anchor_ratio numeric,
    add column if not exists eoff_fib_dominant_anchor_share numeric,
    add column if not exists eoff_unique_anchor_gate boolean,
    add column if not exists eoff_unique_anchor_signature text;

create table if not exists public.model_registry (
    component text primary key,
    semantic_version text not null,
    is_active boolean not null default true,
    released_at timestamptz not null,
    config_hash text,
    metadata jsonb not null default '{}'::jsonb,
    model_version text not null,
    schema_version text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.source_events (
    event_key text primary key,
    ticker text,
    event_type text not null,
    event_date date,
    source_family text,
    content_hash text,
    event_fingerprint text,
    refresh_required boolean not null default false,
    detected_at timestamptz not null,
    last_seen_at timestamptz not null,
    resolved_at timestamptz,
    model_version text not null,
    schema_version text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.research_outcomes (
    outcome_id text primary key,
    ticker text not null,
    signal_family text not null,
    signal_timestamp timestamptz not null,
    signal_date date not null,
    anchor_id text,
    liquidity_bucket text,
    predicted_state text,
    predicted_direction text,
    signal_score numeric,
    signal_confidence numeric,
    prediction_window_start date,
    prediction_window_end date,
    entry_reference numeric,
    horizon_bars integer not null default 20,
    outcome_status text not null default 'OPEN',
    resolved_at timestamptz,
    actual_low_date date,
    actual_high_date date,
    forward_return_5d numeric,
    forward_return_10d numeric,
    forward_return_20d numeric,
    maximum_favourable_excursion numeric,
    maximum_adverse_excursion numeric,
    hit boolean,
    model_version text not null,
    schema_version text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.backfill_state (
    entity_key text primary key,
    ticker text not null,
    entity_type text not null,
    cohort integer not null default 0,
    active_cohort integer,
    priority integer not null default 99,
    status text not null,
    refresh_reason text,
    last_attempt_at timestamptz,
    last_success_at timestamptz,
    next_due_at timestamptz,
    failure_count integer not null default 0,
    model_version text not null,
    schema_version text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- Official IDX sessions can be loaded from an official exchange-calendar file.
-- Scanner falls back to weekday-only validation until this table is populated.
create table if not exists public.idx_trading_calendar (
    trade_date date primary key,
    is_open boolean not null,
    session_type text not null default 'REGULAR',
    source_family text not null default 'IDX_OFFICIAL_CALENDAR',
    source_url text,
    content_hash text,
    verified_at timestamptz,
    notes text,
    model_version text not null,
    schema_version text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_source_events_ticker_date
    on public.source_events (ticker, event_date desc);
create index if not exists idx_source_events_refresh
    on public.source_events (refresh_required, detected_at desc);
create index if not exists idx_research_outcomes_open
    on public.research_outcomes (outcome_status, signal_family, signal_date);
create index if not exists idx_research_outcomes_ticker
    on public.research_outcomes (ticker, signal_date desc);
create index if not exists idx_research_outcomes_liquidity
    on public.research_outcomes (signal_family, liquidity_bucket, outcome_status);
create index if not exists idx_backfill_due
    on public.backfill_state (entity_type, status, next_due_at);
create index if not exists idx_calendar_open
    on public.idx_trading_calendar (is_open, trade_date);

-- Reuse v3 updated_at trigger function.
do $$
declare
    table_name text;
begin
    foreach table_name in array array[
        'model_registry','source_events','research_outcomes','backfill_state','idx_trading_calendar'
    ]
    loop
        execute format('drop trigger if exists %I on public.%I', 'trg_' || table_name || '_updated_at', table_name);
        execute format(
            'create trigger %I before update on public.%I for each row execute function public.set_scanner_updated_at()',
            'trg_' || table_name || '_updated_at', table_name
        );
        execute format('alter table public.%I enable row level security', table_name);
        execute format('revoke all on table public.%I from anon, authenticated', table_name);
        execute format('grant all on table public.%I to service_role', table_name);
    end loop;
end $$;

grant usage on schema public to service_role;
grant usage, select on all sequences in schema public to service_role;

commit;
