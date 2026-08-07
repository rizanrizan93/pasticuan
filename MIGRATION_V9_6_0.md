# Migration v9.6.0

No new SQL migration is required.

Prerequisites remain:
- `database/migration_v12_resumable_scan_jobs.sql`
- `database/permissions_hotfix_v9_4_1.sql`
- `database/migration_v13_persistent_ohlcv.sql`

Run the existing v12/v13 verification scripts if the deployment reports database permission or OHLCV-cache errors.
