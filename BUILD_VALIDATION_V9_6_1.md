# Build Validation v9.6.1

- All top-level Python modules compile with Python 3.12 syntax: PASS.
- Transport retry tests: PASS.
- Permission-error non-retry test: PASS.
- Unsafe RPC (`claim_scan_job_items`) non-retry test: PASS.
- Idempotent UPSERT retry test: PASS.
- Legacy 8-second timeout safety-floor test: PASS.
- No database schema change: PASS (v12/v13 compatible).
- Ranking/scoring model logic is unchanged from v9.6.0.
