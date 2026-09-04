-- Phase 5.6 v32: scanner-neutral public ownership-context projection.
-- Source facts are Yahoo/public-provider ownership concentration only.
-- This surface is NOT KSEI, issuer register, free float, beneficial ownership,
-- broker/bandar identity, score, rank, recommendation, or production gate.

create table if not exists public.phase56_public_ownership_snapshots (
    ticker text primary key,
    source_period date,
    observed_on date,
    insiders_held_pct numeric,
    institutions_held_pct numeric,
    institutions_float_held_pct numeric,
    institutions_count numeric,
    coverage_pct numeric not null default 0,
    source_authority text not null default 'PUBLIC_PROVIDER',
    official_verified boolean not null default false,
    provenance_state text not null default 'PUBLIC_PROVIDER_YAHOO_CONCENTRATION_NOT_IDX_KSEI',
    source_state text not null default 'PHASE5_6_PUBLIC_OWNERSHIP_CONTEXT',
    refreshed_at timestamptz not null default now(),
    constraint phase56_public_ownership_ticker_check check (ticker ~ '^[A-Z][A-Z0-9]{3,5}$'),
    constraint phase56_public_ownership_insiders_check check (insiders_held_pct is null or (insiders_held_pct >= 0 and insiders_held_pct <= 100)),
    constraint phase56_public_ownership_institutions_check check (institutions_held_pct is null or (institutions_held_pct >= 0 and institutions_held_pct <= 100)),
    constraint phase56_public_ownership_institutions_float_check check (institutions_float_held_pct is null or (institutions_float_held_pct >= 0 and institutions_float_held_pct <= 100)),
    constraint phase56_public_ownership_count_check check (institutions_count is null or institutions_count >= 0),
    constraint phase56_public_ownership_coverage_check check (coverage_pct >= 0 and coverage_pct <= 100),
    constraint phase56_public_ownership_authority_check check (source_authority = 'PUBLIC_PROVIDER'),
    constraint phase56_public_ownership_official_check check (official_verified = false)
);

comment on table public.phase56_public_ownership_snapshots is
'Read-only public Phase 5.6 ownership concentration context. Not KSEI, issuer shareholder register, free float, beneficial ownership, broker/bandar identity, score, rank, recommendation, or gate.';

alter table public.phase56_public_ownership_snapshots enable row level security;

revoke all on table public.phase56_public_ownership_snapshots from public;
revoke all on table public.phase56_public_ownership_snapshots from anon;
revoke all on table public.phase56_public_ownership_snapshots from authenticated;
revoke all on table public.phase56_public_ownership_snapshots from service_role;

grant select on table public.phase56_public_ownership_snapshots to anon;
grant select on table public.phase56_public_ownership_snapshots to authenticated;
grant select, insert, update on table public.phase56_public_ownership_snapshots to service_role;

drop policy if exists phase56_public_ownership_select on public.phase56_public_ownership_snapshots;
create policy phase56_public_ownership_select
on public.phase56_public_ownership_snapshots
for select
to anon, authenticated
using (true);
