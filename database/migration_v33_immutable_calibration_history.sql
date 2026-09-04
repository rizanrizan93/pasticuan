-- v33: Preserve scanner-specific point-in-time decisions for calibration.
-- Existing *_snapshots tables intentionally remain same-day deduplicated runtime stores.
-- This table is scanner-specific derived history and is NOT part of the Shared Evidence Hub.

create table if not exists public.pasticuan_calibration_snapshots (
    calibration_id text primary key,
    scan_id text not null,
    ticker text not null,
    scanner_version text,
    scan_as_of timestamptz,
    captured_at timestamptz not null default now(),
    last_captured_at timestamptz not null default now(),
    capture_version text not null default 'v33',

    has_multibagger boolean not null default false,
    has_technical boolean not null default false,
    has_fundamental boolean not null default false,
    multibagger_snapshot_id text,
    technical_snapshot_id text,
    fundamental_snapshot_id text,

    multibagger_status text,
    multibagger_lane text,
    research_recommendation_status text,
    research_score numeric,
    ranking_score numeric,
    final_score numeric,
    score_coverage_pct numeric,
    multibagger_production_rank numeric,
    multibagger_rank_eligible boolean,
    real_money_authorization_state text,
    real_money_authorization_pass boolean,
    real_money_authorization_blockers text,
    anti_chase_gate boolean,

    future_fundamental_score numeric,
    future_fundamental_coverage_pct numeric,
    market_context_score numeric,
    market_context_coverage_pct numeric,
    fundamental_official_verified boolean,
    fundamental_official_source_coverage_pct numeric,
    fundamental_cashflow_coverage_pct numeric,
    fundamental_score numeric,
    fundamental_coverage numeric,
    fundamental_period_end text,
    fundamental_statement_date text,

    technical_readiness_score numeric,
    technical_readiness_coverage_pct numeric,
    active_setup text,
    technical_entry_state text,
    last_price numeric,
    entry numeric,
    stop_loss numeric,
    tp1 numeric,
    tp2 numeric,
    execution_readiness_score numeric,
    silent_accumulation_score numeric,
    relative_strength60 numeric,
    adtv20_idr numeric,

    constraint pasticuan_calibration_snapshots_scan_ticker_key unique (scan_id, ticker)
);

create index if not exists idx_pasticuan_calibration_scan_asof
    on public.pasticuan_calibration_snapshots (scan_as_of desc, scan_id);
create index if not exists idx_pasticuan_calibration_ticker_asof
    on public.pasticuan_calibration_snapshots (ticker, scan_as_of desc);

create table if not exists public.pasticuan_calibration_outcomes (
    calibration_id text not null references public.pasticuan_calibration_snapshots(calibration_id),
    horizon_sessions smallint not null check (horizon_sessions in (1, 3, 5, 10, 20)),
    evaluation_session date,
    observed_at timestamptz not null default now(),
    close_price numeric,
    return_pct numeric,
    mfe_pct numeric,
    mae_pct numeric,
    tp1_hit boolean,
    tp2_hit boolean,
    stop_hit boolean,
    outcome_state text not null default 'PENDING',
    source text not null default 'OHLCV_DAILY',
    primary key (calibration_id, horizon_sessions)
);

create index if not exists idx_pasticuan_calibration_outcomes_session
    on public.pasticuan_calibration_outcomes (evaluation_session, horizon_sessions);

create or replace function public.upsert_pasticuan_calibration_snapshot(
    p_source_table text,
    p_payload jsonb
) returns void
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
    v_scan_id text := nullif(btrim(coalesce(p_payload->>'scan_id', '')), '');
    v_ticker text := nullif(upper(btrim(coalesce(p_payload->>'ticker', ''))), '');
    v_calibration_id text;
begin
    if v_scan_id is null or v_ticker is null then
        return;
    end if;
    if p_source_table not in ('multibagger_snapshots', 'technical_snapshots', 'fundamental_snapshots') then
        raise exception 'unsupported calibration source table: %', p_source_table;
    end if;

    v_calibration_id := v_scan_id || '|' || v_ticker;

    insert into public.pasticuan_calibration_snapshots (
        calibration_id, scan_id, ticker, scanner_version, scan_as_of, last_captured_at,
        has_multibagger, has_technical, has_fundamental,
        multibagger_snapshot_id, technical_snapshot_id, fundamental_snapshot_id,
        multibagger_status, multibagger_lane, research_recommendation_status,
        research_score, ranking_score, final_score, score_coverage_pct,
        multibagger_production_rank, multibagger_rank_eligible,
        real_money_authorization_state, real_money_authorization_pass,
        real_money_authorization_blockers, anti_chase_gate,
        future_fundamental_score, future_fundamental_coverage_pct,
        market_context_score, market_context_coverage_pct,
        fundamental_official_verified, fundamental_official_source_coverage_pct,
        fundamental_cashflow_coverage_pct, fundamental_score, fundamental_coverage,
        fundamental_period_end, fundamental_statement_date,
        technical_readiness_score, technical_readiness_coverage_pct,
        active_setup, technical_entry_state, last_price, entry, stop_loss, tp1, tp2,
        execution_readiness_score, silent_accumulation_score, relative_strength60, adtv20_idr
    ) values (
        v_calibration_id, v_scan_id, v_ticker,
        nullif(p_payload->>'model_version', ''),
        nullif(p_payload->>'as_of', '')::timestamptz,
        now(),
        p_source_table = 'multibagger_snapshots',
        p_source_table = 'technical_snapshots',
        p_source_table = 'fundamental_snapshots',
        case when p_source_table = 'multibagger_snapshots' then p_payload->>'snapshot_id' end,
        case when p_source_table = 'technical_snapshots' then p_payload->>'snapshot_id' end,
        case when p_source_table = 'fundamental_snapshots' then p_payload->>'snapshot_id' end,
        nullif(p_payload->>'multibagger_status', ''),
        nullif(p_payload->>'multibagger_lane', ''),
        nullif(p_payload->>'research_recommendation_status', ''),
        nullif(p_payload->>'research_score', '')::numeric,
        nullif(p_payload->>'ranking_score', '')::numeric,
        nullif(p_payload->>'final_score', '')::numeric,
        nullif(p_payload->>'score_coverage_pct', '')::numeric,
        nullif(p_payload->>'multibagger_production_rank', '')::numeric,
        nullif(p_payload->>'multibagger_rank_eligible', '')::boolean,
        nullif(p_payload->>'real_money_authorization_state', ''),
        nullif(p_payload->>'real_money_authorization_pass', '')::boolean,
        nullif(p_payload->>'real_money_authorization_blockers', ''),
        nullif(p_payload->>'anti_chase_gate', '')::boolean,
        nullif(p_payload->>'future_fundamental_score', '')::numeric,
        nullif(p_payload->>'future_fundamental_coverage_pct', '')::numeric,
        nullif(p_payload->>'market_context_score', '')::numeric,
        nullif(p_payload->>'market_context_coverage_pct', '')::numeric,
        nullif(p_payload->>'fundamental_official_verified', '')::boolean,
        nullif(p_payload->>'fundamental_official_source_coverage_pct', '')::numeric,
        nullif(p_payload->>'fundamental_cashflow_coverage_pct', '')::numeric,
        nullif(p_payload->>'fundamental_score', '')::numeric,
        nullif(p_payload->>'fundamental_coverage', '')::numeric,
        nullif(p_payload->>'period_end', ''),
        nullif(p_payload->>'statement_date', ''),
        nullif(p_payload->>'technical_readiness_score', '')::numeric,
        nullif(p_payload->>'technical_readiness_coverage_pct', '')::numeric,
        nullif(p_payload->>'active_setup', ''),
        nullif(p_payload->>'technical_entry_state', ''),
        nullif(p_payload->>'last_price', '')::numeric,
        nullif(p_payload->>'entry', '')::numeric,
        nullif(p_payload->>'stop_loss', '')::numeric,
        nullif(p_payload->>'tp1', '')::numeric,
        nullif(p_payload->>'tp2', '')::numeric,
        nullif(p_payload->>'execution_readiness_score', '')::numeric,
        nullif(p_payload->>'silent_accumulation_score', '')::numeric,
        nullif(p_payload->>'relative_strength60', '')::numeric,
        nullif(p_payload->>'adtv20_idr', '')::numeric
    )
    on conflict (scan_id, ticker) do update set
        scanner_version = coalesce(excluded.scanner_version, pasticuan_calibration_snapshots.scanner_version),
        scan_as_of = coalesce(excluded.scan_as_of, pasticuan_calibration_snapshots.scan_as_of),
        last_captured_at = now(),
        has_multibagger = pasticuan_calibration_snapshots.has_multibagger or excluded.has_multibagger,
        has_technical = pasticuan_calibration_snapshots.has_technical or excluded.has_technical,
        has_fundamental = pasticuan_calibration_snapshots.has_fundamental or excluded.has_fundamental,
        multibagger_snapshot_id = coalesce(excluded.multibagger_snapshot_id, pasticuan_calibration_snapshots.multibagger_snapshot_id),
        technical_snapshot_id = coalesce(excluded.technical_snapshot_id, pasticuan_calibration_snapshots.technical_snapshot_id),
        fundamental_snapshot_id = coalesce(excluded.fundamental_snapshot_id, pasticuan_calibration_snapshots.fundamental_snapshot_id),
        multibagger_status = coalesce(excluded.multibagger_status, pasticuan_calibration_snapshots.multibagger_status),
        multibagger_lane = coalesce(excluded.multibagger_lane, pasticuan_calibration_snapshots.multibagger_lane),
        research_recommendation_status = coalesce(excluded.research_recommendation_status, pasticuan_calibration_snapshots.research_recommendation_status),
        research_score = coalesce(excluded.research_score, pasticuan_calibration_snapshots.research_score),
        ranking_score = coalesce(excluded.ranking_score, pasticuan_calibration_snapshots.ranking_score),
        final_score = coalesce(excluded.final_score, pasticuan_calibration_snapshots.final_score),
        score_coverage_pct = coalesce(excluded.score_coverage_pct, pasticuan_calibration_snapshots.score_coverage_pct),
        multibagger_production_rank = coalesce(excluded.multibagger_production_rank, pasticuan_calibration_snapshots.multibagger_production_rank),
        multibagger_rank_eligible = coalesce(excluded.multibagger_rank_eligible, pasticuan_calibration_snapshots.multibagger_rank_eligible),
        real_money_authorization_state = coalesce(excluded.real_money_authorization_state, pasticuan_calibration_snapshots.real_money_authorization_state),
        real_money_authorization_pass = coalesce(excluded.real_money_authorization_pass, pasticuan_calibration_snapshots.real_money_authorization_pass),
        real_money_authorization_blockers = coalesce(excluded.real_money_authorization_blockers, pasticuan_calibration_snapshots.real_money_authorization_blockers),
        anti_chase_gate = coalesce(excluded.anti_chase_gate, pasticuan_calibration_snapshots.anti_chase_gate),
        future_fundamental_score = coalesce(excluded.future_fundamental_score, pasticuan_calibration_snapshots.future_fundamental_score),
        future_fundamental_coverage_pct = coalesce(excluded.future_fundamental_coverage_pct, pasticuan_calibration_snapshots.future_fundamental_coverage_pct),
        market_context_score = coalesce(excluded.market_context_score, pasticuan_calibration_snapshots.market_context_score),
        market_context_coverage_pct = coalesce(excluded.market_context_coverage_pct, pasticuan_calibration_snapshots.market_context_coverage_pct),
        fundamental_official_verified = coalesce(excluded.fundamental_official_verified, pasticuan_calibration_snapshots.fundamental_official_verified),
        fundamental_official_source_coverage_pct = coalesce(excluded.fundamental_official_source_coverage_pct, pasticuan_calibration_snapshots.fundamental_official_source_coverage_pct),
        fundamental_cashflow_coverage_pct = coalesce(excluded.fundamental_cashflow_coverage_pct, pasticuan_calibration_snapshots.fundamental_cashflow_coverage_pct),
        fundamental_score = coalesce(excluded.fundamental_score, pasticuan_calibration_snapshots.fundamental_score),
        fundamental_coverage = coalesce(excluded.fundamental_coverage, pasticuan_calibration_snapshots.fundamental_coverage),
        fundamental_period_end = coalesce(excluded.fundamental_period_end, pasticuan_calibration_snapshots.fundamental_period_end),
        fundamental_statement_date = coalesce(excluded.fundamental_statement_date, pasticuan_calibration_snapshots.fundamental_statement_date),
        technical_readiness_score = coalesce(excluded.technical_readiness_score, pasticuan_calibration_snapshots.technical_readiness_score),
        technical_readiness_coverage_pct = coalesce(excluded.technical_readiness_coverage_pct, pasticuan_calibration_snapshots.technical_readiness_coverage_pct),
        active_setup = coalesce(excluded.active_setup, pasticuan_calibration_snapshots.active_setup),
        technical_entry_state = coalesce(excluded.technical_entry_state, pasticuan_calibration_snapshots.technical_entry_state),
        last_price = coalesce(excluded.last_price, pasticuan_calibration_snapshots.last_price),
        entry = coalesce(excluded.entry, pasticuan_calibration_snapshots.entry),
        stop_loss = coalesce(excluded.stop_loss, pasticuan_calibration_snapshots.stop_loss),
        tp1 = coalesce(excluded.tp1, pasticuan_calibration_snapshots.tp1),
        tp2 = coalesce(excluded.tp2, pasticuan_calibration_snapshots.tp2),
        execution_readiness_score = coalesce(excluded.execution_readiness_score, pasticuan_calibration_snapshots.execution_readiness_score),
        silent_accumulation_score = coalesce(excluded.silent_accumulation_score, pasticuan_calibration_snapshots.silent_accumulation_score),
        relative_strength60 = coalesce(excluded.relative_strength60, pasticuan_calibration_snapshots.relative_strength60),
        adtv20_idr = coalesce(excluded.adtv20_idr, pasticuan_calibration_snapshots.adtv20_idr);
end;
$$;

create or replace function public.capture_pasticuan_calibration_trigger()
returns trigger
language plpgsql
security definer
set search_path = public, pg_temp
as $$
begin
    perform public.upsert_pasticuan_calibration_snapshot(TG_TABLE_NAME, to_jsonb(NEW));
    return NEW;
end;
$$;

drop trigger if exists trg_capture_pasticuan_calibration_multibagger on public.multibagger_snapshots;
create trigger trg_capture_pasticuan_calibration_multibagger
after insert or update on public.multibagger_snapshots
for each row execute function public.capture_pasticuan_calibration_trigger();

drop trigger if exists trg_capture_pasticuan_calibration_technical on public.technical_snapshots;
create trigger trg_capture_pasticuan_calibration_technical
after insert or update on public.technical_snapshots
for each row execute function public.capture_pasticuan_calibration_trigger();

drop trigger if exists trg_capture_pasticuan_calibration_fundamental on public.fundamental_snapshots;
create trigger trg_capture_pasticuan_calibration_fundamental
after insert or update on public.fundamental_snapshots
for each row execute function public.capture_pasticuan_calibration_trigger();

-- Seed only observations that still exist. Same-day observations already overwritten
-- before v33 cannot be reconstructed safely and are deliberately not invented.
do $$
declare r record;
begin
    for r in select to_jsonb(s) payload from public.multibagger_snapshots s where nullif(s.scan_id, '') is not null loop
        perform public.upsert_pasticuan_calibration_snapshot('multibagger_snapshots', r.payload);
    end loop;
    for r in select to_jsonb(s) payload from public.technical_snapshots s where nullif(s.scan_id, '') is not null loop
        perform public.upsert_pasticuan_calibration_snapshot('technical_snapshots', r.payload);
    end loop;
    for r in select to_jsonb(s) payload from public.fundamental_snapshots s where nullif(s.scan_id, '') is not null loop
        perform public.upsert_pasticuan_calibration_snapshot('fundamental_snapshots', r.payload);
    end loop;
end;
$$;

alter table public.pasticuan_calibration_snapshots enable row level security;
alter table public.pasticuan_calibration_outcomes enable row level security;

revoke all on public.pasticuan_calibration_snapshots from public, anon, authenticated;
revoke all on public.pasticuan_calibration_outcomes from public, anon, authenticated;
revoke all on public.pasticuan_calibration_snapshots from service_role;
revoke all on public.pasticuan_calibration_outcomes from service_role;
grant select on public.pasticuan_calibration_snapshots to service_role;
grant select, insert, update on public.pasticuan_calibration_outcomes to service_role;

revoke all on function public.upsert_pasticuan_calibration_snapshot(text, jsonb) from public, anon, authenticated;
revoke all on function public.capture_pasticuan_calibration_trigger() from public, anon, authenticated;

comment on table public.pasticuan_calibration_snapshots is
'PASTICUAN-only immutable per-run decision history. Scanner scores/ranks are not Shared Evidence Hub facts.';
comment on table public.pasticuan_calibration_outcomes is
'Forward outcome observations for PASTICUAN calibration snapshots; kept separate from decision-time evidence.';
