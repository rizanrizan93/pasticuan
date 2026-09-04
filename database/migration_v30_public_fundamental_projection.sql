-- Phase 5.6 v30: scanner-neutral public factual fundamental projection.
-- This is a compact read-only consumer surface. Raw evidence remains private.
-- No score, rank, recommendation, gate, entry, stop, target, or Future Fundamental.

create table if not exists public.phase56_public_fundamental_snapshots (
    ticker text primary key,
    proxy_period_end date,
    proxy_observed_at timestamptz,
    official_period_end date,
    official_observed_at timestamptz,
    proxy_metrics jsonb not null default '{}'::jsonb,
    official_metrics jsonb not null default '{}'::jsonb,
    source_families jsonb not null default '[]'::jsonb,
    official_coverage_pct numeric not null default 0,
    source_state text not null default 'PHASE5_6_PUBLIC_FACTUAL_PROJECTION',
    refreshed_at timestamptz not null default now()
);

comment on table public.phase56_public_fundamental_snapshots is
'Read-only public projection of scanner-neutral Phase 5.6 factual fundamentals. No score, rank, gate, recommendation, entry, stop, target, or Future Fundamental.';

alter table public.phase56_public_fundamental_snapshots enable row level security;

revoke all on table public.phase56_public_fundamental_snapshots from public;
revoke all on table public.phase56_public_fundamental_snapshots from anon;
revoke all on table public.phase56_public_fundamental_snapshots from authenticated;
revoke all on table public.phase56_public_fundamental_snapshots from service_role;

grant select on table public.phase56_public_fundamental_snapshots to anon;
grant select on table public.phase56_public_fundamental_snapshots to authenticated;
grant select, insert, update on table public.phase56_public_fundamental_snapshots to service_role;

drop policy if exists phase56_public_fundamental_select on public.phase56_public_fundamental_snapshots;
create policy phase56_public_fundamental_select
on public.phase56_public_fundamental_snapshots
for select
to anon, authenticated
using (true);
