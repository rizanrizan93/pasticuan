-- Phase 5.6 scanner-neutral Shared IDX Evidence Hub.
-- Additive only. Scanner-private result/ranking tables are intentionally untouched.

create table if not exists public.evidence_provider_state (
    provider text not null,
    endpoint_family text not null,
    scope text not null,
    target_date date not null,
    last_attempt_at timestamptz,
    last_success_at timestamptz,
    latest_source_date date,
    response_state text not null default 'MISSING',
    http_status integer,
    quota_remaining bigint,
    rate_limit_state text,
    negative_cache_until timestamptz,
    payload_hash text,
    retry_after timestamptz,
    error_classification text,
    next_refresh_eligible_at timestamptz,
    updated_at timestamptz not null default now(),
    primary key (provider, endpoint_family, scope, target_date)
);

create table if not exists public.evidence_refresh_leases (
    provider text not null,
    evidence_family text not null,
    scope text not null,
    target_date date not null,
    lease_state text not null default 'AVAILABLE',
    lease_owner text,
    lease_expires_at timestamptz,
    attempt_count integer not null default 0,
    result_state text,
    last_error text,
    acquired_at timestamptz,
    completed_at timestamptz,
    updated_at timestamptz not null default now(),
    primary key (provider, evidence_family, scope, target_date),
    constraint evidence_refresh_leases_state_check
      check (lease_state in ('AVAILABLE', 'HELD', 'COMPLETED', 'FAILED'))
);

create table if not exists public.evidence_ingestion_runs (
    id bigint generated always as identity primary key,
    run_key text not null unique,
    provider text not null,
    evidence_family text not null,
    scope text not null,
    target_date date,
    producer_client text not null,
    started_at timestamptz not null default now(),
    completed_at timestamptz,
    result_state text not null default 'RUNNING',
    calls_attempted integer not null default 0,
    rows_received integer not null default 0,
    rows_valid integer not null default 0,
    rows_persisted integer not null default 0,
    cache_hits integer not null default 0,
    calls_avoided integer not null default 0,
    detail jsonb not null default '{}'::jsonb
);

create table if not exists public.evidence_failures (
    id bigint generated always as identity primary key,
    provider text not null,
    evidence_family text not null,
    scope text not null,
    target_date date,
    reason text not null,
    http_status integer,
    attempted_url text,
    discovery_method text,
    content_type text,
    content_length bigint,
    occurred_at timestamptz not null default now(),
    retry_after timestamptz,
    detail jsonb not null default '{}'::jsonb
);

create table if not exists public.evidence_raw_payloads (
    payload_hash text primary key,
    provider text not null,
    evidence_family text not null,
    source_url text,
    source_date date,
    fetched_at timestamptz not null default now(),
    content_type text,
    content_length bigint,
    validation_state text not null,
    payload jsonb,
    storage_reference text
);

create table if not exists public.evidence_market_daily (
    provider text not null,
    trade_date date not null,
    ticker text not null,
    open numeric,
    high numeric,
    low numeric,
    close numeric,
    previous numeric,
    volume numeric,
    value numeric,
    frequency numeric,
    bid numeric,
    offer numeric,
    bid_volume numeric,
    offer_volume numeric,
    listed_shares numeric,
    tradeable_shares numeric,
    foreign_buy numeric,
    foreign_sell numeric,
    non_regular_volume numeric,
    non_regular_value numeric,
    non_regular_frequency numeric,
    source_url text,
    payload_hash text,
    fetched_at timestamptz not null default now(),
    freshness_state text not null default 'CURRENT',
    validation_state text not null default 'VALID',
    primary key (provider, trade_date, ticker)
);

create table if not exists public.evidence_foreign_flow (
    provider text not null,
    trade_date date not null,
    ticker text not null,
    foreign_buy_shares numeric,
    foreign_sell_shares numeric,
    foreign_net_shares numeric,
    foreign_buy_value numeric,
    foreign_sell_value numeric,
    foreign_net_value numeric,
    volume numeric,
    value numeric,
    flow_unit text not null default 'SHARES',
    source_family text not null default 'ZAPI_IDX_FOREIGN_FLOW',
    source_url text,
    payload_hash text,
    fetched_at timestamptz not null default now(),
    freshness_state text not null default 'CURRENT',
    validation_state text not null default 'VALID',
    primary key (provider, trade_date, ticker)
);

create table if not exists public.evidence_participant_flow (
    source text not null,
    trade_date date not null,
    ticker text not null,
    broker_code text not null,
    buy_value numeric not null default 0,
    sell_value numeric not null default 0,
    buy_volume numeric not null default 0,
    sell_volume numeric not null default 0,
    net_value numeric not null default 0,
    net_volume numeric not null default 0,
    buy_avg numeric,
    sell_avg numeric,
    source_url text,
    source_file_hash text,
    source_verified boolean not null default false,
    provenance_state text not null,
    fetched_at timestamptz not null default now(),
    validation_state text not null default 'VALID',
    primary key (source, trade_date, ticker, broker_code)
);

comment on table public.evidence_participant_flow is
  'Official public IDX participant flow; participant is not a beneficial owner and is not automatic bandar evidence.';

create table if not exists public.evidence_ownership_files (
    source_file_hash text not null,
    category text not null,
    publication_date date,
    report_date date,
    source_url text not null,
    file_name text,
    source_verified boolean not null default false,
    parser_version text,
    fetched_at timestamptz not null default now(),
    validation_state text not null,
    primary key (source_file_hash, category)
);

create table if not exists public.evidence_ownership_snapshots (
    source_file_hash text not null,
    category text not null,
    ticker text not null,
    holder_identity_hash text not null,
    holder_name text,
    report_date date,
    publication_date date,
    shares_held numeric,
    ownership_percentage numeric,
    holder_classification text,
    holder_type text,
    local_foreign_state text,
    source_url text,
    source_verified boolean not null default false,
    validation_state text not null,
    fetched_at timestamptz not null default now(),
    primary key (source_file_hash, ticker, holder_identity_hash),
    foreign key (source_file_hash, category)
      references public.evidence_ownership_files (source_file_hash, category)
      deferrable initially deferred
);

comment on table public.evidence_ownership_snapshots is
  'Reported shareholder evidence only; it does not establish beneficial ownership, broker identity, or bandar identity.';

create table if not exists public.evidence_ownership_changes (
    source_file_hash text not null,
    previous_source_file_hash text,
    category text not null,
    ticker text not null,
    holder_identity_hash text not null,
    previous_report_date date not null,
    current_report_date date not null,
    previous_shares numeric,
    current_shares numeric,
    delta_shares numeric,
    previous_percentage numeric,
    current_percentage numeric,
    delta_percentage numeric,
    change_state text not null,
    source_verified boolean not null default false,
    validation_state text not null,
    derived_at timestamptz not null default now(),
    primary key (source_file_hash, ticker, holder_identity_hash, change_state),
    foreign key (source_file_hash, category)
      references public.evidence_ownership_files (source_file_hash, category)
      deferrable initially deferred,
    constraint evidence_ownership_changes_state_check check (change_state in (
      'NEW_1PCT_HOLDER', 'NEW_5PCT_HOLDER', 'INCREASED_REPORTED_HOLDING',
      'REDUCED_REPORTED_HOLDING', 'EXITED_REPORTED_HOLDER',
      'OWNERSHIP_CONCENTRATION_RISING', 'OWNERSHIP_CONCENTRATION_FALLING',
      'FOREIGN_OWNERSHIP_SHIFT'
    )),
    constraint evidence_ownership_changes_period_check
      check (current_report_date > previous_report_date)
);

comment on table public.evidence_ownership_changes is
  'Mathematical deltas between comparable reported-holder snapshots; not beneficial-owner, broker, or bandar identity.';

create table if not exists public.evidence_financial_reports (
    ticker text not null,
    report_period text not null,
    period_type text not null,
    report_date date,
    publication_date date not null,
    source text not null,
    source_url text not null,
    issuer_identity text,
    issuer_match boolean not null default false,
    context_state text not null,
    parser_version text,
    source_document_hash text not null,
    freshness_state text not null,
    validation_state text not null,
    fetched_at timestamptz not null default now(),
    primary key (ticker, report_period, source_document_hash)
);

create table if not exists public.evidence_financial_facts (
    ticker text not null,
    report_period text not null,
    fact_name text not null,
    fact_value numeric,
    currency text,
    unit_scale numeric,
    period_type text not null,
    report_date date,
    publication_date date not null,
    source text not null,
    source_url text not null,
    issuer_identity text,
    context_state text not null,
    parser_version text,
    source_document_hash text not null,
    freshness_state text not null,
    validation_state text not null,
    fetched_at timestamptz not null default now(),
    primary key (ticker, report_period, fact_name, source_document_hash),
    foreign key (ticker, report_period, source_document_hash)
      references public.evidence_financial_reports (ticker, report_period, source_document_hash)
      deferrable initially deferred
);

create table if not exists public.evidence_announcements (
    source_event_id text not null,
    ticker text,
    title text,
    subject text,
    summary text,
    event_date date,
    event_at timestamptz,
    publication_date date not null,
    published_at timestamptz,
    event_type text,
    event_confirmation_state text not null default 'METADATA_ONLY_NOT_DOCUMENT_CONFIRMED',
    announcement_no text,
    form_id text,
    attachment_count integer not null default 0,
    attachment_urls jsonb not null default '[]'::jsonb,
    source text not null,
    source_url text,
    source_document_hash text,
    payload_hash text,
    source_verified boolean not null default false,
    validation_state text not null,
    fetched_at timestamptz not null default now(),
    primary key (source_event_id),
    constraint evidence_announcements_attachment_count_check check (attachment_count >= 0),
    constraint evidence_announcements_confirmation_check check (event_confirmation_state in (
      'METADATA_ONLY_NOT_DOCUMENT_CONFIRMED', 'DOCUMENT_CONFIRMED'
    ))
);

comment on table public.evidence_announcements is
  'Official announcement metadata; material-event semantics require document confirmation and are never inferred from title alone.';

create table if not exists public.evidence_capital_actions (
    ticker text not null,
    event_type text not null,
    event_date date not null,
    publication_date date,
    pre_shares numeric,
    post_shares numeric,
    delta_shares numeric,
    delta_percent numeric,
    ratio_before numeric,
    ratio_after numeric,
    raw_action text,
    calculation_state text not null default 'NO_SHARE_FACTS',
    source text not null,
    source_feed text not null,
    source_period date not null,
    observed_on date not null,
    source_url text,
    source_id text not null,
    payload_hash text not null,
    source_verified boolean not null default false,
    validation_state text not null,
    fetched_at timestamptz not null default now(),
    primary key (ticker, event_type, event_date, source_id),
    constraint evidence_capital_actions_calculation_check check (calculation_state in (
      'NO_SHARE_FACTS', 'EXPLICIT_DELTA_ONLY', 'EXPLICIT_PRE_POST',
      'EXPLICIT_DELTA_POST_DERIVED_PRE', 'EXPLICIT_PRE_DELTA_DERIVED_POST'
    )),
    constraint evidence_capital_actions_ratio_before_check check (ratio_before is null or ratio_before > 0),
    constraint evidence_capital_actions_ratio_after_check check (ratio_after is null or ratio_after > 0)
);

alter table public.evidence_capital_actions
  add column if not exists ratio_before numeric,
  add column if not exists ratio_after numeric,
  add column if not exists raw_action text,
  add column if not exists calculation_state text not null default 'NO_SHARE_FACTS',
  add column if not exists source_feed text,
  add column if not exists source_period date,
  add column if not exists observed_on date,
  add column if not exists payload_hash text,
  add column if not exists source_verified boolean not null default false;

comment on table public.evidence_capital_actions is
  'Scanner-neutral issued-share and capital-action facts; ratios and titles never imply unreported share counts or dilution.';

create table if not exists public.evidence_companies (
    provider text not null,
    ticker text not null,
    company_name text,
    sector text,
    sub_sector text,
    industry text,
    sub_industry text,
    listing_board text,
    listing_date date,
    listed_shares numeric,
    main_business text,
    profile jsonb not null default '{}'::jsonb,
    profile_kind text not null default 'DIRECTORY',
    source_period date,
    observed_on date,
    change_state text not null default 'NEW',
    source_url text,
    payload_hash text,
    source_verified boolean not null default false,
    validation_state text not null,
    fetched_at timestamptz not null default now(),
    primary key (provider, ticker),
    constraint evidence_companies_listed_shares_check check (listed_shares is null or listed_shares >= 0),
    constraint evidence_companies_change_state_check check (change_state in ('NEW', 'CHANGED', 'UNCHANGED')),
    constraint evidence_companies_profile_kind_check check (profile_kind in ('DIRECTORY', 'DETAILED_PROFILE'))
);

alter table public.evidence_companies
  add column if not exists sub_sector text,
  add column if not exists sub_industry text,
  add column if not exists listed_shares numeric,
  add column if not exists main_business text,
  add column if not exists profile_kind text not null default 'DIRECTORY',
  add column if not exists source_period date,
  add column if not exists observed_on date,
  add column if not exists change_state text not null default 'NEW';

create table if not exists public.evidence_reference_values (
    provider text not null,
    set_name text not null,
    value_key text not null,
    label text not null,
    source_period date not null,
    observed_on date not null,
    source_url text not null,
    payload_hash text not null,
    source_verified boolean not null default false,
    validation_state text not null,
    fetched_at timestamptz not null default now(),
    primary key (provider, set_name, value_key),
    constraint evidence_reference_values_set_check check (set_name in ('sectors', 'boards', 'market-time'))
);

comment on table public.evidence_companies is
  'Slow-moving factual company directory/profile evidence; names and roles carry no subjective governance conclusion.';

create table if not exists public.evidence_brokers (
    provider text not null,
    broker_code text not null,
    broker_name text,
    member_status text,
    ownership_category text,
    foreign_ownership_percentage numeric,
    profile jsonb not null default '{}'::jsonb,
    source_url text,
    payload_hash text,
    source_verified boolean not null default false,
    validation_state text not null,
    fetched_at timestamptz not null default now(),
    primary key (provider, broker_code)
);

alter table public.evidence_brokers
  add column if not exists paid_up_capital numeric,
  add column if not exists mkbd numeric,
  add column if not exists branch_count integer,
  add column if not exists profile_kind text not null default 'EXCHANGE_MEMBER',
  add column if not exists evidence_scope text not null default 'MARKET_WIDE',
  add column if not exists source_period date,
  add column if not exists observed_on date,
  add column if not exists change_state text not null default 'NEW';

do $$
begin
  if not exists (
    select 1 from pg_constraint
    where conname = 'evidence_brokers_reference_check'
      and conrelid = 'public.evidence_brokers'::regclass
  ) then
    alter table public.evidence_brokers
      add constraint evidence_brokers_reference_check check (
        (foreign_ownership_percentage is null or foreign_ownership_percentage between 0 and 100)
        and (paid_up_capital is null or paid_up_capital >= 0)
        and (mkbd is null or mkbd >= 0)
        and (branch_count is null or branch_count >= 0)
        and profile_kind = 'EXCHANGE_MEMBER'
        and evidence_scope = 'MARKET_WIDE'
        and change_state in ('NEW', 'CHANGED', 'UNCHANGED')
      );
  end if;
end;
$$;

comment on table public.evidence_brokers is
  'Exchange-member reference metadata only; foreign/local fields are provider-reported and do not establish ticker-level activity.';

create table if not exists public.evidence_broker_market_daily (
    provider text not null,
    activity_date date not null,
    broker_code text not null,
    broker_name text,
    traded_value numeric,
    traded_volume numeric,
    frequency bigint,
    source_event_id text,
    evidence_scope text not null default 'MARKET_WIDE',
    source_url text not null,
    payload_hash text not null,
    source_verified boolean not null default false,
    validation_state text not null,
    fetched_at timestamptz not null default now(),
    primary key (provider, activity_date, broker_code),
    constraint evidence_broker_market_scope_check check (evidence_scope = 'MARKET_WIDE'),
    constraint evidence_broker_market_nonnegative_check check (
      (traded_value is null or traded_value >= 0)
      and (traded_volume is null or traded_volume >= 0)
      and (frequency is null or frequency >= 0)
    )
);

comment on table public.evidence_broker_market_daily is
  'Market-wide daily broker totals without ticker identity; never ticker-level broker evidence.';

create table if not exists public.evidence_risk_events (
    provider text not null,
    event_type text not null,
    event_date date not null,
    ticker text not null default 'IDX_ALL',
    source_id text not null,
    publication_date date,
    active_state text,
    source text,
    source_feed text,
    source_period date,
    window_end_date date,
    observed_on date,
    date_semantics text,
    title text,
    details jsonb not null default '{}'::jsonb,
    source_url text,
    payload_hash text,
    source_verified boolean not null default false,
    validation_state text not null,
    fetched_at timestamptz not null default now(),
    primary key (provider, event_type, event_date, ticker, source_id),
    constraint evidence_risk_events_active_state_check check (active_state is null or active_state in (
      'UMA_ACTIVE_OR_RECENT', 'SUSPENSION_ACTIVE_OR_RECENT', 'RECENT_DILUTION_EVENT',
      'MARGIN_ELIGIBLE', 'LENDABLE'
    )),
    constraint evidence_risk_events_window_check check (
      source_period is null or window_end_date is null or window_end_date >= source_period
    )
);

alter table public.evidence_risk_events
  add column if not exists source text,
  add column if not exists source_feed text,
  add column if not exists source_period date,
  add column if not exists window_end_date date,
  add column if not exists observed_on date,
  add column if not exists date_semantics text,
  add column if not exists title text,
  add column if not exists details jsonb not null default '{}'::jsonb,
  add column if not exists payload_hash text;

comment on table public.evidence_risk_events is
  'Factual UMA, suspension, margin, lendable, and explicit dilution context; rows do not create scanner rejection gates.';

create table if not exists public.evidence_trading_calendar (
    trade_date date primary key,
    is_session boolean not null,
    session_state text not null,
    source text not null,
    source_url text,
    source_verified boolean not null default false,
    validation_state text not null,
    fetched_at timestamptz not null default now()
);

create index if not exists evidence_provider_state_refresh_idx
  on public.evidence_provider_state (provider, endpoint_family, next_refresh_eligible_at);
create index if not exists evidence_refresh_leases_expiry_idx
  on public.evidence_refresh_leases (lease_state, lease_expires_at);
create index if not exists evidence_failures_lookup_idx
  on public.evidence_failures (provider, evidence_family, target_date, occurred_at desc);
create index if not exists evidence_market_ticker_date_idx
  on public.evidence_market_daily (ticker, trade_date desc);
create index if not exists evidence_foreign_ticker_date_idx
  on public.evidence_foreign_flow (ticker, trade_date desc);
create index if not exists evidence_participant_ticker_date_idx
  on public.evidence_participant_flow (ticker, trade_date desc, broker_code);
create index if not exists evidence_ownership_ticker_date_idx
  on public.evidence_ownership_snapshots (ticker, publication_date desc);
create index if not exists evidence_ownership_changes_ticker_date_idx
  on public.evidence_ownership_changes (ticker, current_report_date desc, change_state);
create index if not exists evidence_financial_ticker_publication_idx
  on public.evidence_financial_facts (ticker, publication_date desc, fact_name);
create index if not exists evidence_financial_document_readback_idx
  on public.evidence_financial_facts (ticker, report_period, source_document_hash);
create index if not exists evidence_announcements_ticker_date_idx
  on public.evidence_announcements (ticker, publication_date desc);
create index if not exists evidence_announcements_feed_date_idx
  on public.evidence_announcements (source, publication_date desc, validation_state);
create index if not exists evidence_capital_actions_ticker_date_idx
  on public.evidence_capital_actions (ticker, event_date desc);
create index if not exists evidence_capital_actions_feed_period_idx
  on public.evidence_capital_actions (source, source_period desc, observed_on desc, validation_state);
create index if not exists evidence_companies_provider_freshness_idx
  on public.evidence_companies (provider, validation_state, fetched_at desc);
create index if not exists evidence_reference_values_lookup_idx
  on public.evidence_reference_values (provider, set_name, validation_state, fetched_at desc);
create index if not exists evidence_brokers_reference_freshness_idx
  on public.evidence_brokers (provider, validation_state, fetched_at desc);
create index if not exists evidence_broker_market_daily_lookup_idx
  on public.evidence_broker_market_daily (activity_date desc, broker_code, validation_state);
create index if not exists evidence_risk_events_ticker_date_idx
  on public.evidence_risk_events (ticker, event_date desc, event_type);
create index if not exists evidence_risk_events_feed_window_idx
  on public.evidence_risk_events (source, source_period desc, window_end_date desc, validation_state);

create or replace function public.evidence_acquire_refresh_lease(
    p_provider text,
    p_family text,
    p_scope text,
    p_target_date date,
    p_holder text,
    p_lease_seconds integer default 300
)
returns table (acquired boolean, lease_state text, expires_at timestamptz, current_holder text)
language plpgsql
security invoker
set search_path = ''
as $$
begin
  return query
  with claimed as (
    insert into public.evidence_refresh_leases as current_lease (
      provider, evidence_family, scope, target_date, lease_state,
      lease_owner, lease_expires_at, attempt_count, acquired_at, updated_at
    ) values (
      upper(p_provider), upper(p_family), upper(p_scope), p_target_date, 'HELD',
      p_holder, now() + make_interval(secs => greatest(30, least(p_lease_seconds, 3600))),
      1, now(), now()
    )
    on conflict (provider, evidence_family, scope, target_date) do update
      set lease_state = 'HELD',
          lease_owner = excluded.lease_owner,
          lease_expires_at = excluded.lease_expires_at,
          attempt_count = current_lease.attempt_count + 1,
          acquired_at = now(),
          completed_at = null,
          last_error = case
            when current_lease.lease_state = 'HELD' and current_lease.lease_expires_at <= now()
              then 'REFRESH_LEASE_EXPIRED'
            else null
          end,
          updated_at = now()
      where current_lease.lease_state <> 'HELD'
         or current_lease.lease_expires_at <= now()
         or current_lease.lease_owner = excluded.lease_owner
    returning true, current_lease.lease_state,
      current_lease.lease_expires_at, current_lease.lease_owner
  )
  select * from claimed;

  if found then
    return;
  end if;

  return query
    select false, current_lease.lease_state,
      current_lease.lease_expires_at, current_lease.lease_owner
    from public.evidence_refresh_leases as current_lease
    where current_lease.provider = upper(p_provider)
      and current_lease.evidence_family = upper(p_family)
      and current_lease.scope = upper(p_scope)
      and current_lease.target_date = p_target_date;
end;
$$;

create or replace function public.evidence_complete_refresh_lease(
    p_provider text,
    p_family text,
    p_scope text,
    p_target_date date,
    p_holder text,
    p_result_state text default 'COMPLETED'
)
returns boolean
language sql
security invoker
set search_path = ''
as $$
  with changed as (
    update public.evidence_refresh_leases
       set lease_state = 'COMPLETED',
           result_state = upper(p_result_state),
           lease_expires_at = null,
           completed_at = now(),
           updated_at = now()
     where provider = upper(p_provider)
       and evidence_family = upper(p_family)
       and scope = upper(p_scope)
       and target_date = p_target_date
       and lease_owner = p_holder
       and lease_state = 'HELD'
     returning 1
  ) select exists(select 1 from changed);
$$;

create or replace function public.evidence_fail_refresh_lease(
    p_provider text,
    p_family text,
    p_scope text,
    p_target_date date,
    p_holder text,
    p_reason text
)
returns boolean
language sql
security invoker
set search_path = ''
as $$
  with changed as (
    update public.evidence_refresh_leases
       set lease_state = 'FAILED',
           result_state = 'ERROR',
           last_error = left(p_reason, 160),
           lease_expires_at = null,
           completed_at = now(),
           updated_at = now()
     where provider = upper(p_provider)
       and evidence_family = upper(p_family)
       and scope = upper(p_scope)
       and target_date = p_target_date
       and lease_owner = p_holder
       and lease_state = 'HELD'
     returning 1
  ) select exists(select 1 from changed);
$$;

do $$
declare
  table_name text;
begin
  foreach table_name in array array[
    'evidence_provider_state', 'evidence_refresh_leases', 'evidence_ingestion_runs',
    'evidence_failures', 'evidence_raw_payloads', 'evidence_market_daily',
    'evidence_foreign_flow', 'evidence_participant_flow', 'evidence_ownership_files',
    'evidence_ownership_snapshots', 'evidence_ownership_changes', 'evidence_financial_reports',
    'evidence_financial_facts', 'evidence_announcements', 'evidence_capital_actions',
    'evidence_companies', 'evidence_reference_values', 'evidence_brokers',
    'evidence_broker_market_daily', 'evidence_risk_events',
    'evidence_trading_calendar'
  ] loop
    execute format('alter table public.%I enable row level security', table_name);
    execute format('revoke all on table public.%I from public, anon, authenticated', table_name);
    execute format('grant select, insert, update on table public.%I to service_role', table_name);
  end loop;
end;
$$;

grant usage, select on sequence public.evidence_ingestion_runs_id_seq to service_role;
grant usage, select on sequence public.evidence_failures_id_seq to service_role;
revoke all on function public.evidence_acquire_refresh_lease(text, text, text, date, text, integer)
  from public, anon, authenticated;
revoke all on function public.evidence_complete_refresh_lease(text, text, text, date, text, text)
  from public, anon, authenticated;
revoke all on function public.evidence_fail_refresh_lease(text, text, text, date, text, text)
  from public, anon, authenticated;
grant execute on function public.evidence_acquire_refresh_lease(text, text, text, date, text, integer)
  to service_role;
grant execute on function public.evidence_complete_refresh_lease(text, text, text, date, text, text)
  to service_role;
grant execute on function public.evidence_fail_refresh_lease(text, text, text, date, text, text)
  to service_role;
