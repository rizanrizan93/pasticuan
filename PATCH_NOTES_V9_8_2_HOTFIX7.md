# IDX Super Scanner v9.8.2 Hotfix 7 — Persistence Integrity

## Root causes repaired

1. The shared `set_scanner_updated_at()` trigger referenced
   `NEW.content_hash` even on tables without that column. Ordinary upserts could
   fail with `record "new" has no field "content_hash"`.
2. Production had migration v15's feature table but was missing the v13
   `ohlcv_daily_cache` relation, disabling the intended database-first path.
3. Pandas `NaT` could be serialised as the literal text `"NaT"` before a typed
   date write or inside fundamental cache metadata.

## Fix

- Migration v16 installs a generic updated-at trigger and a separate
  refresh-state content-hash trigger.
- Migration v16 idempotently creates/repairs compact OHLCV and feature-cache
  relations without deleting legacy data.
- Temporal values are normalised to ISO date/timestamp or SQL `NULL`.
- Trigger/API helper execution is revoked from `anon` and `authenticated`.

No scoring weight, ranking gate, or feature-cache compatibility version changed.
