-- Run after migration_v7_multibagger_lanes.sql.

select
    count(*) = 16 as multibagger_lane_columns_ready
from information_schema.columns
where table_schema = 'public'
  and table_name = 'multibagger_snapshots'
  and column_name in (
      'multibagger_lane',
      'research_eligible',
      'research_eligibility_reason',
      'portfolio_allocation_eligible',
      'growth_compounder_score',
      'growth_compounder_base_score',
      'growth_compounder_selection_score',
      'turnaround_recovery_score',
      'confidence_adjusted_turnaround_score',
      'turnaround_selection_score',
      'turnaround_research_state',
      'turnaround_recovery_signals',
      'turnaround_gate_reasons',
      'critical_research_flags',
      'operational_recovery_flags',
      'effective_silent_accumulation_score'
  );

select
    column_name,
    data_type,
    is_nullable
from information_schema.columns
where table_schema = 'public'
  and table_name = 'multibagger_snapshots'
  and column_name in (
      'multibagger_lane',
      'research_eligible',
      'portfolio_allocation_eligible',
      'growth_compounder_selection_score',
      'turnaround_selection_score',
      'turnaround_research_state',
      'effective_silent_accumulation_score'
  )
order by ordinal_position;

select
    indexname,
    indexdef
from pg_indexes
where schemaname = 'public'
  and tablename = 'multibagger_snapshots'
  and indexname in (
      'idx_multibagger_growth_lane_rank',
      'idx_multibagger_turnaround_lane_rank'
  )
order by indexname;
