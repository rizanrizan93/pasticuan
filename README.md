# IDX Super Scanner v9.6.0 — Quality Integrity

v9.6.0 keeps the v9.5 persistent database-first architecture and hardens the decision layer after the 2026-08-07 production scan exposed sector, freshness, and ranking-quality problems.

## Production contract

1. Read persistent OHLCV/fundamental/narrative evidence first.
2. Refresh only MISSING/STALE evidence from public providers.
3. Persist the delta back to the database.
4. Recompute issuer sector, freshness, business quality, future fundamental, valuation, management, macro alignment, narrative-flow, and technical readiness.
5. A ticker receives a production Final Score/rank only after the production evidence gates pass.
6. Partial evidence remains visible as research/audit evidence but cannot occupy a production rank.

## v9.6 integrity changes

- Canonical issuer sector classifier with explicit-provider priority, conservative text inference, and small audited fallback coverage for previously UNKNOWN names.
- No synthetic UNKNOWN macro-sector bucket.
- Persistent database snapshot is primary over local/session cache; older cache can only fill missing fields.
- Fundamental eligibility is recomputed each run; stale old booleans cannot keep a ticker production-ready.
- Statement freshness: CURRENT <=210 days, ACCEPTABLE_STALE <=300 days, otherwise STALE.
- The Next Leader production gate: total score coverage >=70%, business coverage >=55%, future coverage >=40%, resolved sector/macro, fresh-enough fundamental evidence, and quality floors.
- Sparse valuation/future evidence is capped so one optimistic metric cannot create a near-100 pillar score.
- Narrative/flow cannot promote a fundamentally weak issuer through the production gate.
- Swing Ready requires score coverage >=65%, technical score >=40, technical coverage >=50%, and macro coverage >=50%.
- Non-sharia issuers are not universally hard-blocked; sharia filtering only applies when `ScanConfig.sharia_only=True`.

## Database

No new SQL migration is required for v9.6.0. Keep v12 resumable jobs and v13 persistent OHLCV installed. Sector/freshness metadata is persisted inside the existing fundamental JSON payload.

## Validation

See `TEST_REPORT_V9_6_0.md` and `BUILD_VALIDATION_V9_6_0.md`.
