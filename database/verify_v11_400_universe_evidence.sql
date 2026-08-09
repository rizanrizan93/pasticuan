-- Verify IDX Super Scanner database schema v11.
-- Expected: 13 column rows, followed by 3 index rows.

select table_name, column_name, data_type, is_nullable
from information_schema.columns
where table_schema = 'public'
  and (
    (table_name = 'multibagger_snapshots' and column_name in (
      'multibagger_evidence_class','multibagger_rank_eligible',
      'multibagger_score_comparability_pct','multibagger_production_rank',
      'narrative_evidence_coverage_pct','issuer_alignment_coverage_pct',
      'emir_method_coverage_pct','distribution_severity_score',
      'distribution_penalty_points','distribution_evidence_state'
    ))
    or
    (table_name = 'selector_snapshots' and column_name in (
      'relative_overlay_weight_pct','selector_universe_state',
      'score_inflation_guard_active'
    ))
  )
order by table_name, column_name;

select schemaname, tablename, indexname
from pg_indexes
where schemaname = 'public'
  and indexname in (
    'idx_multibagger_full_evidence_rank',
    'idx_multibagger_evidence_class',
    'idx_selector_universe_state'
  )
order by indexname;
