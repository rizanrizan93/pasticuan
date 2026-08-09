-- IDX Super Scanner v7.6.1 Narrative Safety & Ranking Separation migration.
-- Prerequisite: schema_v2.sql + migrations v3 through v9.
-- Idempotent and safe to run more than once.

begin;

alter table if exists public.issuer_master
    add column if not exists official_domain text,
    add column if not exists official_domain_verified boolean not null default false,
    add column if not exists official_domain_checked_at timestamptz;

alter table if exists public.narrative_events
    add column if not exists source_hostname text,
    add column if not exists registered_official_domain text,
    add column if not exists source_state text,
    add column if not exists source_present boolean not null default false,
    add column if not exists official_claimed boolean not null default false,
    add column if not exists entity_match_state text,
    add column if not exists event_status text not null default 'ACTIVE',
    add column if not exists requested_event_status text,
    add column if not exists lifecycle_evidence_state text,
    add column if not exists resolved_at timestamptz,
    add column if not exists supersedes_event_id text,
    add column if not exists resolution_source_url text;

alter table if exists public.narrative_event_outcomes
    add column if not exists impact_sign integer,
    add column if not exists impact_direction text,
    add column if not exists entry_policy text,
    add column if not exists directional_excess_return_5d_pct numeric,
    add column if not exists directional_excess_return_20d_pct numeric,
    add column if not exists directional_excess_return_60d_pct numeric;

alter table if exists public.narrative_snapshots
    add column if not exists narrative_missing_source_event_count integer,
    add column if not exists narrative_inactive_lifecycle_event_count integer,
    add column if not exists narrative_entity_unverified_event_count integer,
    add column if not exists narrative_production_policy text;

update public.narrative_events
set source_present = case
        when source_url ~* '^https://[^/]+(?:/|$)' then true
        else false
    end,
    source_state = coalesce(
        source_state,
        case
            when source_url ~* '^https://[^/]+(?:/|$)' then 'SOURCE_IDENTIFIED'
            else 'MISSING_SOURCE'
        end
    ),
    official_claimed = case
        when official_verified is true then true
        else coalesce(official_claimed, false)
    end,
    event_status = coalesce(nullif(upper(event_status), ''), 'ACTIVE'),
    requested_event_status = coalesce(
        nullif(upper(requested_event_status), ''),
        coalesce(nullif(upper(event_status), ''), 'ACTIVE')
    ),
    lifecycle_evidence_state = coalesce(
        lifecycle_evidence_state,
        case
            when upper(coalesce(event_status, 'ACTIVE')) = 'ACTIVE'
                then 'ACTIVE_EVENT'
            when resolution_source_url ~* '^https://[^/]+(?:/|$)'
                then 'RESOLUTION_SOURCE_IDENTIFIED'
            else 'RESOLUTION_SOURCE_MISSING_KEPT_ACTIVE'
        end
    )
where source_state is null
   or requested_event_status is null
   or lifecycle_evidence_state is null
   or (official_verified is true and official_claimed is false);


update public.narrative_event_outcomes
set impact_sign = coalesce(
        impact_sign,
        case upper(coalesce(impact_direction, ''))
            when 'POSITIVE' then 1
            when 'NEGATIVE' then -1
            else null
        end
    ),
    entry_policy = coalesce(
        entry_policy,
        'LEGACY_SAME_OR_NEXT_SESSION_REQUIRES_RECOMPUTE'
    ),
    directional_excess_return_5d_pct = coalesce(
        directional_excess_return_5d_pct,
        coalesce(
            impact_sign,
            case upper(coalesce(impact_direction, ''))
                when 'POSITIVE' then 1
                when 'NEGATIVE' then -1
                else null
            end
        ) * net_excess_return_5d_pct
    ),
    directional_excess_return_20d_pct = coalesce(
        directional_excess_return_20d_pct,
        coalesce(
            impact_sign,
            case upper(coalesce(impact_direction, ''))
                when 'POSITIVE' then 1
                when 'NEGATIVE' then -1
                else null
            end
        ) * net_excess_return_20d_pct
    ),
    directional_excess_return_60d_pct = coalesce(
        directional_excess_return_60d_pct,
        coalesce(
            impact_sign,
            case upper(coalesce(impact_direction, ''))
                when 'POSITIVE' then 1
                when 'NEGATIVE' then -1
                else null
            end
        ) * net_excess_return_60d_pct
    );

create index if not exists idx_narrative_events_lifecycle
    on public.narrative_events (ticker, event_status, detected_at desc);
create index if not exists idx_narrative_events_source_state
    on public.narrative_events (source_state, detected_at desc);
create index if not exists idx_narrative_events_entity_state
    on public.narrative_events (entity_match_state, detected_at desc);
create index if not exists idx_narrative_outcomes_directional_20d
    on public.narrative_event_outcomes
        (event_family, directional_excess_return_20d_pct desc nulls last);
create index if not exists idx_issuer_master_official_domain
    on public.issuer_master (official_domain)
    where official_domain is not null;

commit;
