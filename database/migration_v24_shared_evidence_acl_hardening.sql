-- Phase 5.6 ACL-only hardening for the existing shared evidence tables.
-- The explicit table inventory keeps this remediation scanner-neutral and bounded.

do $$
declare
  table_name text;
begin
  foreach table_name in array array[
    'evidence_provider_state', 'evidence_refresh_leases', 'evidence_ingestion_runs',
    'evidence_failures', 'evidence_raw_payloads', 'evidence_market_daily',
    'evidence_foreign_flow', 'evidence_participant_flow', 'evidence_ownership_files',
    'evidence_ownership_snapshots', 'evidence_ownership_changes', 'evidence_financial_reports',
    'evidence_financial_facts', 'evidence_announcements', 'evidence_capital_actions',
    'evidence_companies', 'evidence_reference_values', 'evidence_brokers',
    'evidence_broker_market_daily', 'evidence_risk_events',
    'evidence_trading_calendar'
  ] loop
    execute format(
      'revoke all privileges on table public.%I from service_role',
      table_name
    );
    execute format(
      'grant select, insert, update on table public.%I to service_role',
      table_name
    );
  end loop;
end;
$$;
