# IDX Super Scanner v9.5.0 — Persistent Database-First

v9.5.0 restores the intended operating contract:

`public data → normalise → persist → read database → refresh only MISSING/STALE → calculate finite Final Score → rank`

## Main changes

- Persistent daily OHLCV cache in Supabase (`ohlcv_daily_cache`).
- Daily scan reads OHLCV from the database before calling public providers.
- Current cache skips public OHLCV calls.
- Missing/stale cache is refreshed and merged incrementally, then written back.
- Usable stale history remains available when a public provider temporarily fails.
- Fundamental, history, market status, and news remain database-first with bounded delta refresh.
- Only rows with a finite Final Score and valid status receive a rank.
- `DATA_PENDING` and `REJECT` rows are retained in audit tables but are not ranked.
- Macro regime becomes provisional when breadth coverage is insufficient.
- Resumable job status is separated from evidence quality.
- Durable artifacts include ranked outputs, unranked audit, evidence detail, scoring contract, macro, coverage, and OHLCV provider audit.

## Required SQL

Run in Supabase SQL Editor after the existing v12 resumable-job migration:

1. `database/migration_v13_persistent_ohlcv.sql`
2. `database/verify_v13_persistent_ohlcv.sql`

The Streamlit backend must use `SUPABASE_SECRET_KEY` or `SUPABASE_SERVICE_ROLE_KEY`.

## Daily scan behavior

For each chunk, scanner:

1. loads stored fundamentals, history, market status, news, and OHLCV;
2. identifies missing or stale evidence by data family;
3. fetches only the required public delta;
4. merges and persists the delta;
5. computes technical, macro, narrative-flow, and fundamental scores;
6. checkpoints each ticker;
7. finalizes a global finite-score ranking.

A temporary provider failure no longer removes a ticker when sufficient cached OHLCV exists.

## Output

- Market Map
- The Next Leader
- Swing Ready
- Portfolio & Audit

The UI displays a human-readable `Final Score`, requested universe, OHLCV-ready coverage, finite-score counts, ranking quality state, and per-ticker OHLCV audit.
