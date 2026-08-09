> Production runtime audit: v9.8.2 Hotfix 9 (universe sector integrity).

# IDX Super Scanner v9.8.2 — ALL_ELIGIBLE_LITE / Database Acceleration

Production workflow:

`Upload ticker CSV -> SCAN -> ALL_ELIGIBLE_LITE discovery -> bounded evidence enrichment -> final ranking -> fail-soft persistence`

## What changed

v9.8.2 freezes the analytical methodology and changes only runtime/database architecture.

- Every ticker in the uploaded universe remains eligible for discovery/ranking.
- A current `scanner_feature_cache` row is reused directly; the scanner does not re-open 800–900 OHLCV bars for that ticker.
- Missing/stale feature rows alone go through the full technical engine.
- Persistent OHLCV uses compact zlib/base64 CSV transport (`payload_compact`) instead of selecting the large legacy JSONB payload on normal reads.
- Legacy OHLCV rows are converted lazily in one bounded batch after they are read.
- Existing incremental Yahoo/IDX refresh remains active: current caches generate zero provider calls; stale caches request a short overlap window.
- Deep fundamental/news/official IDX work remains bounded to the promoted shortlist plus a small maintenance lane.
- Supabase remains fail-soft. A database outage cannot block ranking production.

## Runtime policy

- Universe: max 400 tickers.
- `ALL_ELIGIBLE_LITE`: all tickers compete; current feature-cache hits skip technical recomputation.
- Evidence enrichment: max 8 names/job by default.
- Two enrichment slots are reserved for rotating stale/missing database maintenance when possible.
- Official IDX/XBRL refresh: max 4 names/job.
- Execution verification: max 6 names/job.
- Individual yfinance retry remains disabled by default.
- Real-money authorization, valuation, future fundamental, inventory lifecycle, anti-chase, distribution guard, and SMC/ICT logic are unchanged.

## Database migration

Run `database/migration_v15_database_acceleration.sql` once.

The scanner remains fail-soft without v15, but it falls back to legacy v9.8.1 database behavior and therefore does **not** receive the main feature-cache/compact-OHLCV acceleration.

v15 adds:

- `scanner_feature_cache`
- `ohlcv_daily_cache.payload_compact`
- `ohlcv_daily_cache.payload_codec`
- `ohlcv_daily_cache.compact_bar_count`
- `ohlcv_daily_cache.compact_hash`

Legacy `ohlcv_daily_cache.payload` data is not destructively deleted.

## Freeze policy

After v9.8.2, do not add indicators or scoring modules before live calibration. Changes should be limited to proven bugs, data-integrity failures, or provider/runtime defects. The next phase is scanner-vs-independent-analysis calibration and market-outcome tracking.

## v9.8.2 Hotfix 3 — Calibration
Hotfix 3 freezes scoring weights and adds calendar-aware fundamental freshness, latest-history-over-proxy precedence, thesis archetypes, bounded refresh promotion and clearer research-vs-execution labelling. No new SQL migration is required; migration v15 remains current.

## v9.8.2 Hotfix 4 — Runtime Stability
Hotfix 4 fixes the production finalizer crash caused by `fundamental_map` being read before initialization and hardens optional provider branches so research ranking survives missing external evidence. Analytical weights/methodology are unchanged; migration v15 remains current.

## v9.8.2 Hotfix 5 — 400 Universe Audit
Hotfix 5 keeps explicit provider budgets exact and rejects malformed feature-cache rows before they can suppress ranking. No scoring or schema change.

## v9.8.2 Hotfix 6 — Data Contract Integrity
Hotfix 6 preserves uploaded IDX-IC sector/rank/role metadata, prevents metadata-only fundamental rows from crashing a cold scan, and separates OHLCV acquisition state from database cache/write state. All analytical weights remain frozen and migration v15 remains current.

## v9.8.2 Hotfix 8 — Fundamental Period Integrity

Hotfix 8 prevents a missing official XBRL fact from masking a valid
same-period proxy fact, and replaces fixed four-row growth lags with explicit
same-calendar-quarter comparisons. This repairs false annual fallbacks and
wrong-period YoY values when quarterly histories contain gaps. The fundamental
feature model lineage is advanced to `7.6.0`; scoring weights are unchanged.

## v9.8.2 Hotfix 9 — Universe Sector Integrity

An explicit IDX-IC sector from an uploaded universe now replaces the
missing-evidence sentinel `UNKNOWN` in a cached fundamental row. The complete
classification bundle is replaced atomically, so macro-sector coverage cannot
remain at 0% while the upload contains valid sector evidence. Scoring weights
remain unchanged.

## v9.8.2 Hotfix 7 — Persistence Integrity
Hotfix 7 separates the generic `updated_at` trigger from the `refresh_state`
content-hash rule, repairs missing v13/v15 OHLCV tables through idempotent
migration v16, and prevents pandas/string `NaT` values from crossing typed date
contracts. Scoring weights and the feature-cache compatibility version remain
unchanged.
