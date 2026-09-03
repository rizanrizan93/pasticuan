-- Phase 5.6 v26: preserve the source-reported issued-history share quantity
-- without forcing generic `shares` into delta semantics.
-- Additive only. Existing rows remain valid with event_shares NULL.

alter table public.evidence_capital_actions
  add column if not exists event_shares numeric;

comment on column public.evidence_capital_actions.event_shares is
  'Explicit source-reported signed share quantity associated with the capital-action event. It is not assumed to equal delta_shares unless the source explicitly provides delta semantics.';
