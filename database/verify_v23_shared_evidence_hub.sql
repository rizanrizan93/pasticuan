-- Run only after migration_v23_shared_evidence_hub.sql is independently verified and applied.
select
  to_regclass('public.evidence_provider_state') is not null as provider_state_exists,
  to_regclass('public.evidence_refresh_leases') is not null as refresh_leases_exists,
  to_regclass('public.evidence_market_daily') is not null as market_exists,
  to_regclass('public.evidence_foreign_flow') is not null as foreign_exists,
  to_regclass('public.evidence_participant_flow') is not null as participant_exists,
  to_regclass('public.evidence_financial_reports') is not null as financial_reports_exists,
  to_regclass('public.evidence_financial_facts') is not null as financial_facts_exists,
  to_regclass('public.evidence_announcements') is not null as announcements_exists,
  to_regclass('public.evidence_capital_actions') is not null as capital_actions_exists,
  to_regclass('public.evidence_companies') is not null as companies_exists,
  to_regclass('public.evidence_reference_values') is not null as reference_values_exists,
  to_regclass('public.evidence_brokers') is not null as brokers_exists,
  to_regclass('public.evidence_broker_market_daily') is not null as broker_market_daily_exists,
  to_regclass('public.evidence_risk_events') is not null as risk_events_exists,
  to_regclass('public.evidence_ownership_snapshots') is not null as ownership_exists,
  to_regclass('public.evidence_ownership_changes') is not null as ownership_changes_exists;

select c.relname, c.relrowsecurity
from pg_class c
join pg_namespace n on n.oid = c.relnamespace
where n.nspname = 'public' and c.relname like 'evidence_%'
order by c.relname;

select
  has_table_privilege('service_role', 'public.evidence_provider_state', 'SELECT,INSERT,UPDATE') as provider_state_service_role,
  has_table_privilege('service_role', 'public.evidence_refresh_leases', 'SELECT,INSERT,UPDATE') as leases_service_role,
  has_table_privilege('service_role', 'public.evidence_capital_actions', 'SELECT,INSERT,UPDATE') as capital_actions_service_role,
  has_table_privilege('service_role', 'public.evidence_companies', 'SELECT,INSERT,UPDATE') as companies_service_role,
  has_table_privilege('service_role', 'public.evidence_reference_values', 'SELECT,INSERT,UPDATE') as reference_values_service_role,
  has_table_privilege('service_role', 'public.evidence_brokers', 'SELECT,INSERT,UPDATE') as brokers_service_role,
  has_table_privilege('service_role', 'public.evidence_broker_market_daily', 'SELECT,INSERT,UPDATE') as broker_market_daily_service_role,
  has_table_privilege('service_role', 'public.evidence_risk_events', 'SELECT,INSERT,UPDATE') as risk_events_service_role,
  not has_table_privilege('anon', 'public.evidence_foreign_flow', 'SELECT') as foreign_denied_anon,
  not has_table_privilege('anon', 'public.evidence_capital_actions', 'SELECT') as capital_actions_denied_anon,
  not has_table_privilege('anon', 'public.evidence_companies', 'SELECT') as companies_denied_anon,
  not has_table_privilege('anon', 'public.evidence_broker_market_daily', 'SELECT') as broker_market_daily_denied_anon,
  not has_table_privilege('anon', 'public.evidence_risk_events', 'SELECT') as risk_events_denied_anon,
  not has_table_privilege('authenticated', 'public.evidence_participant_flow', 'SELECT') as participant_denied_authenticated;

select
  has_function_privilege('service_role', 'public.evidence_acquire_refresh_lease(text,text,text,date,text,integer)', 'EXECUTE') as acquire_service_role,
  not has_function_privilege('anon', 'public.evidence_acquire_refresh_lease(text,text,text,date,text,integer)', 'EXECUTE') as acquire_denied_anon,
  not has_function_privilege('authenticated', 'public.evidence_complete_refresh_lease(text,text,text,date,text,text)', 'EXECUTE') as complete_denied_authenticated;
