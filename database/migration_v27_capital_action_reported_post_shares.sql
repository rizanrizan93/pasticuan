-- Phase 5.6 v27: preserve the source-reported issued-history sharesAfter
-- even when a legacy source row contains an impossible negative total.
-- Additive only. Normalized post_shares remains the usable non-negative total.

alter table public.evidence_capital_actions
  add column if not exists reported_post_shares numeric;

comment on column public.evidence_capital_actions.reported_post_shares is
  'Exact source-reported sharesAfter/postShares value when present. May preserve historical source anomalies; post_shares remains the normalized usable non-negative total.';
