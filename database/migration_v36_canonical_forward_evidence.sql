-- v36: Canonical scanner-neutral forward factual evidence.
-- One factual event identity is shared by EMIR and PASTICUAN; downstream
-- analytical outputs remain scanner-private.

create table if not exists public.evidence_forward_events (
    canonical_event_id text primary key,
    ticker text not null,
    event_category text not null,
    evidence_type text not null,
    evidence_date date not null,
    observed_at timestamptz,
    title text not null,
    value_numeric numeric,
    unit text,
    horizon text,
    primary_source_url text not null,
    corroboration_urls jsonb not null default '[]'::jsonb,
    source_families jsonb not null default '[]'::jsonb,
    source_quorum_count integer not null default 0,
    source_quorum_verified boolean not null default false,
    entity_match_verified boolean not null default false,
    source_verified boolean not null default false,
    evidence_confidence numeric,
    payload jsonb not null default '{}'::jsonb,
    producer_clients jsonb not null default '[]'::jsonb,
    producer_records jsonb not null default '{}'::jsonb,
    evidence_tier text not null default 'DIRECT_VERIFIED',
    validation_state text not null default 'VALID',
    contract_version text not null default 'v1',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint evidence_forward_events_category_check check (event_category in (
      'CONTRACT_BACKLOG', 'CAPEX_EXPANSION', 'GUIDANCE', 'PRODUCT_LAUNCH',
      'JV_MA', 'ADVERSE_FORWARD', 'OTHER_FORWARD'
    )),
    constraint evidence_forward_events_tier_check check (evidence_tier in (
      'DIRECT_VERIFIED', 'RESEARCH_ONLY'
    )),
    constraint evidence_forward_events_quorum_count_check check (source_quorum_count >= 0),
    constraint evidence_forward_events_confidence_check check (
      evidence_confidence is null or evidence_confidence between 0 and 1
    ),
    constraint evidence_forward_events_https_check check (
      primary_source_url ~* '^https://[^[:space:]]+$'
    )
);

comment on table public.evidence_forward_events is
'Canonical scanner-neutral forward factual evidence shared by EMIR and PASTICUAN. Analytical interpretation stays outside this table.';
comment on column public.evidence_forward_events.evidence_date is
'Primary factual event/publication date. Later corroboration publication dates belong in provenance/payload and must not replace the primary event date.';
comment on column public.evidence_forward_events.event_category is
'Neutral event taxonomy only; it carries no scanner interpretation.';

create index if not exists evidence_forward_events_ticker_date_idx
  on public.evidence_forward_events (ticker, evidence_date desc, event_category);
create index if not exists evidence_forward_events_strict_idx
  on public.evidence_forward_events (
    source_verified, source_quorum_verified, entity_match_verified,
    evidence_date desc
  );

alter table public.evidence_forward_events enable row level security;
revoke all on table public.evidence_forward_events from public, anon, authenticated, service_role;
grant select, insert, update on table public.evidence_forward_events to service_role;
