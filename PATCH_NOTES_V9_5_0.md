# Patch Notes v9.5.0

## Fixed

1. OHLCV was effectively session-dependent; 291/300 tickers could become `TECHNICAL_UNAVAILABLE` when a live provider failed.
2. `COMPLETE` was visually confused with data readiness.
3. Rows with empty final scores could still appear with ranks.
4. Macro breadth could look confident on a very small technical sample.
5. Provider failures were not sufficiently visible in durable audit output.
6. Repeated Pandas concat/fillna FutureWarnings in active resumable paths.

## Added

- Supabase OHLCV cache and v13 migration.
- Database-first delta refresh and stale-cache fallback.
- Benchmark (`^JKSE`) persistence.
- Per-ticker OHLCV status, bar count, last date, session lag, source tier, and acquisition error.
- Production ranking eligibility gate.
- Durable ranked and unranked artifacts.
- Macro breadth coverage guard.
- Warm-cache scan tests proving no public OHLCV call is made when stored data is current.
