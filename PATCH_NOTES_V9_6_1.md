# Patch Notes v9.6.1 — Supabase Network Resilience

## Fixed

1. **False permission diagnosis on timeout**
   - `ReadTimeout`/connection failures are no longer shown as if they were HTTP 403/42501 permission failures.
   - The UI now separates transport, permission, and missing-schema errors.

2. **Supabase repository read resilience**
   - GET requests retry up to 3 times with exponential backoff.
   - Read timeout defaults to 18 seconds and is clamped to a 15-second safety floor, so an old `SCANNER_DATABASE_TIMEOUT = "8"` secret does not recreate the production failure.
   - Connect timeout is separately configurable.

3. **Safe idempotent write retry**
   - UPSERT and PATCH operations retry transient network/HTTP 408/425/429/5xx failures.
   - Safe RPCs (`refresh_scan_job_counters`, `renew_scan_job_leases`, `claim_scan_job_lease`) may retry.
   - `claim_scan_job_items` is deliberately **not** auto-retried because a timed-out first response may already have claimed a chunk.

4. **UI continuity after worker start**
   - If the immediate readback after starting a worker times out, the app keeps the already-known durable job object instead of blanking the job UI or crashing that rerun.

5. **Pandas warning cleanup**
   - Removed the repeated blank-string `DataFrame.replace` FutureWarning path in `resumable_app_engine.py`.
   - Hardened pending-frame concatenation against all-NA frame warnings.

## Database migration

No new SQL migration is required. Keep `migration_v12_resumable_scan_jobs.sql`, `permissions_hotfix_v9_4_1.sql`, and `migration_v13_persistent_ohlcv.sql` already installed.
