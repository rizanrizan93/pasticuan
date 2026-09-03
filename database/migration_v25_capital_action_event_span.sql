-- Phase 5.6 v25: preserve explicit date spans for aggregate capital-action rows.
-- Additive only. Existing point events remain POINT with no synthetic span.

alter table public.evidence_capital_actions
  add column if not exists event_date_kind text not null default 'POINT',
  add column if not exists event_start_date date,
  add column if not exists event_end_date date;

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conrelid = 'public.evidence_capital_actions'::regclass
      and conname = 'evidence_capital_actions_event_date_semantics_check'
  ) then
    alter table public.evidence_capital_actions
      add constraint evidence_capital_actions_event_date_semantics_check
      check (
        (
          event_date_kind = 'POINT'
          and event_start_date is null
          and event_end_date is null
        )
        or
        (
          event_date_kind = 'RANGE_END'
          and event_start_date is not null
          and event_end_date is not null
          and event_start_date <= event_end_date
          and event_date = event_end_date
        )
      );
  end if;
end;
$$;

comment on column public.evidence_capital_actions.event_date_kind is
  'Factual date semantics: POINT for point events, RANGE_END when event_date is the explicit end of a source-reported date span.';
comment on column public.evidence_capital_actions.event_start_date is
  'Explicit source-reported start of an aggregate capital-action date span; never inferred.';
comment on column public.evidence_capital_actions.event_end_date is
  'Explicit source-reported end of an aggregate capital-action date span; equals event_date when event_date_kind=RANGE_END.';
