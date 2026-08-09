-- IDX Super Scanner v9.4.1 permission hotfix for migration v12.
-- Run once in Supabase SQL Editor as the project owner/postgres role.
-- Idempotent and safe to rerun.
begin;

grant usage on schema public to service_role;

alter table if exists public.scan_jobs enable row level security;
alter table if exists public.scan_job_items enable row level security;
alter table if exists public.scan_job_artifacts enable row level security;

revoke all on table public.scan_jobs from public, anon, authenticated;
revoke all on table public.scan_job_items from public, anon, authenticated;
revoke all on table public.scan_job_artifacts from public, anon, authenticated;

grant select, insert, update, delete on table public.scan_jobs to service_role;
grant select, insert, update, delete on table public.scan_job_items to service_role;
grant select, insert, update, delete on table public.scan_job_artifacts to service_role;

grant execute on function public.claim_scan_job_items(text,text,integer,text,integer) to service_role;
grant execute on function public.refresh_scan_job_counters(text) to service_role;
grant execute on function public.claim_scan_job_lease(text,text,integer) to service_role;
grant execute on function public.renew_scan_job_leases(text,text,integer) to service_role;

alter default privileges for role postgres in schema public
    grant select, insert, update, delete on tables to service_role;

commit;
