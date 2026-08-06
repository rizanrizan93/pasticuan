# Migration v9.5.0

## Prerequisite

- v12 resumable job tables and permission hotfix already applied.
- Streamlit uses a Supabase service-role/secret key.

## SQL

Run:

```sql
-- paste/run database/migration_v13_persistent_ohlcv.sql
```

Then verify:

```sql
-- paste/run database/verify_v13_persistent_ohlcv.sql
```

Expected verification values:

- `table_exists = true`
- `service_role_dml = true`
- `anon_blocked = true`
- `authenticated_blocked = true`

No existing fundamental, narrative, job, or ranking tables are deleted.
