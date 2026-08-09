-- IDX Super Scanner v7.5.1 Multibagger lane migration
-- Prerequisite: schema_v2.sql + migrations v3 through v6.
-- Idempotent and safe to run more than once.

begin;

alter table public.multibagger_snapshots
    add column if not exists multibagger_lane text,
    add column if not exists research_eligible boolean,
    add column if not exists research_eligibility_reason text,
    add column if not exists portfolio_allocation_eligible boolean,
    add column if not exists growth_compounder_score numeric,
    add column if not exists growth_compounder_base_score numeric,
    add column if not exists growth_compounder_selection_score numeric,
    add column if not exists turnaround_recovery_score numeric,
    add column if not exists confidence_adjusted_turnaround_score numeric,
    add column if not exists turnaround_selection_score numeric,
    add column if not exists turnaround_research_state text,
    add column if not exists turnaround_recovery_signals integer,
    add column if not exists turnaround_gate_reasons text,
    add column if not exists critical_research_flags boolean,
    add column if not exists operational_recovery_flags boolean,
    add column if not exists effective_silent_accumulation_score numeric;

create index if not exists idx_multibagger_growth_lane_rank
    on public.multibagger_snapshots
        (growth_compounder_selection_score desc nulls last)
    where research_eligible is true
      and multibagger_lane = 'GROWTH_COMPOUNDER';

create index if not exists idx_multibagger_turnaround_lane_rank
    on public.multibagger_snapshots
        (turnaround_selection_score desc nulls last)
    where research_eligible is true
      and multibagger_lane = 'TURNAROUND_CYCLICAL';

commit;
