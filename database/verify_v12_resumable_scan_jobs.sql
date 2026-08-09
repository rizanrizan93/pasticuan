-- IDX Super Scanner v9.4.1 resumable repository verification.
select to_regclass('public.scan_jobs') as scan_jobs,
       to_regclass('public.scan_job_items') as scan_job_items,
       to_regclass('public.scan_job_artifacts') as scan_job_artifacts;

select
    has_schema_privilege('service_role', 'public', 'USAGE') as service_role_schema_usage,
    has_table_privilege('service_role', 'public.scan_jobs', 'SELECT,INSERT,UPDATE,DELETE') as scan_jobs_dml,
    has_table_privilege('service_role', 'public.scan_job_items', 'SELECT,INSERT,UPDATE,DELETE') as scan_job_items_dml,
    has_table_privilege('service_role', 'public.scan_job_artifacts', 'SELECT,INSERT,UPDATE,DELETE') as scan_job_artifacts_dml;

select
    has_table_privilege('anon', 'public.scan_jobs', 'SELECT') as anon_scan_jobs_select,
    has_table_privilege('authenticated', 'public.scan_jobs', 'SELECT') as authenticated_scan_jobs_select;

select proname,
       has_function_privilege('service_role', oid, 'EXECUTE') as service_role_execute
from pg_proc
where proname in (
    'claim_scan_job_items',
    'refresh_scan_job_counters',
    'claim_scan_job_lease',
    'renew_scan_job_leases'
)
order by proname;

select relname, relrowsecurity
from pg_class
where oid in (
    'public.scan_jobs'::regclass,
    'public.scan_job_items'::regclass,
    'public.scan_job_artifacts'::regclass
)
order by relname;

select job_id, job_type, status, phase, total_items, completed_items,
       failed_items, retry_items, active_worker, lease_expires_at, updated_at
from public.scan_jobs
order by updated_at desc
limit 10;
