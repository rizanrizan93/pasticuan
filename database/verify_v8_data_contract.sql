-- Run after migration_v8_data_contract.sql.

select
    count(*) = 9 as multibagger_data_contract_ready
from information_schema.columns
where table_schema = 'public'
  and table_name = 'multibagger_snapshots'
  and column_name in (
      'multibagger_scoring_state',
      'multibagger_metric_coverage_pct',
      'multibagger_metric_data_gate',
      'growth_pillar_coverage_pct',
      'profitability_pillar_coverage_pct',
      'cashflow_pillar_coverage_pct',
      'safety_pillar_coverage_pct',
      'runway_pillar_coverage_pct',
      'valuation_pillar_coverage_pct'
  );

select
    count(*) = 8 as selector_data_contract_ready
from information_schema.columns
where table_schema = 'public'
  and table_name = 'selector_snapshots'
  and column_name in (
      'production_selection_rank',
      'selector_rank_eligible',
      'selector_data_state',
      'technical_feature_coverage_pct',
      'selector_missing_feature_count',
      'selector_missing_features',
      'effective_silent_accumulation_score',
      'silent_accumulation_confidence'
  );

select
    indexname,
    indexdef
from pg_indexes
where schemaname = 'public'
  and indexname in (
      'idx_multibagger_metric_ready_rank',
      'idx_selector_production_rank'
  )
order by indexname;
