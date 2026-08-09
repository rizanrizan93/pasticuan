select table_name, column_name, data_type
from information_schema.columns
where table_schema = 'public'
  and table_name in ('fundamental_snapshots','multibagger_snapshots')
  and column_name in (
    'fundamental_official_source_coverage_pct','fundamental_reconciliation_state',
    'real_money_authorization_state','real_money_authorization_pass',
    'fundamental_conviction_cap','market_context_score','independent_price_verified'
  )
order by table_name, column_name;
