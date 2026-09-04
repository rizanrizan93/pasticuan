-- Phase 5.6 v31: public-provider ownership concentration facts.
-- Separate from KSEI >1%/>5% ownership snapshots and issuer shareholder profiles.
-- No free-float, beneficial-owner, broker/bandar, score, rank, or gate inference.

create table if not exists public.evidence_ownership_concentration_metrics (
    provider text not null,
    ticker text not null,
    source_period date not null,
    observed_on date not null,
    metric_name text not null,
    metric_value numeric not null,
    metric_unit text not null,
    source_record_hash text not null,
    source_url text not null,
    source_verified boolean not null default false,
    official_verified boolean not null default false,
    source_authority text not null,
    lineage_state text not null,
    validation_state text not null,
    fetched_at timestamptz not null default now(),
    primary key (provider, ticker, source_period, metric_name),
    constraint evidence_ownership_concentration_ticker_check check (ticker ~ '^[A-Z][A-Z0-9]{3,5}$'),
    constraint evidence_ownership_concentration_metric_check check (
        metric_name in (
            'insiders_held_pct',
            'institutions_held_pct',
            'institutions_float_held_pct',
            'institutions_count'
        )
    ),
    constraint evidence_ownership_concentration_unit_check check (metric_unit in ('PERCENT','COUNT')),
    constraint evidence_ownership_concentration_value_check check (
        (metric_unit = 'PERCENT' and metric_value >= 0 and metric_value <= 100)
        or (metric_unit = 'COUNT' and metric_value >= 0)
    ),
    constraint evidence_ownership_concentration_authority_check check (
        source_authority in ('PUBLIC_PROVIDER','OFFICIAL','ISSUER')
    ),
    constraint evidence_ownership_concentration_validation_check check (validation_state = 'VALID')
);

comment on table public.evidence_ownership_concentration_metrics is
'Scanner-neutral ownership concentration facts. Yahoo/public-provider metrics are not KSEI, issuer shareholder registers, free float, beneficial ownership, broker/bandar identity, or decision scores.';

alter table public.evidence_ownership_concentration_metrics enable row level security;
revoke all on table public.evidence_ownership_concentration_metrics from public;
revoke all on table public.evidence_ownership_concentration_metrics from anon;
revoke all on table public.evidence_ownership_concentration_metrics from authenticated;
revoke all on table public.evidence_ownership_concentration_metrics from service_role;
grant select, insert, update on table public.evidence_ownership_concentration_metrics to service_role;
