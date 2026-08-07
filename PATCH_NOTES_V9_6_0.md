# Patch Notes v9.6.0

## Fixed

- UNKNOWN sector values no longer receive a generic macro score.
- Added canonical issuer classification and audited fallback for the UNKNOWN names observed in the production scan.
- Fixed database-first precedence: persistent database fundamental rows are authoritative over older local cache rows, while missing fields can still be coalesced from fallback evidence.
- Fundamental production eligibility is recomputed from score, coverage, source count, history periods, statement age, and sector state.
- Refresh queue prioritizes unresolved sectors, missing/ineligible fundamentals, insufficient history, and stale statements.
- Business-quality scoring now gives explicit weight to latest growth, acceleration, inflection, margins, cash generation, conversion, and balance-sheet safety.
- Future-fundamental and valuation scores are capped when evidence coverage is sparse.
- The Next Leader now has production quality/coverage/freshness gates; DATA_PENDING/RESEARCH_ONLY names cannot receive a production rank.
- Swing Ready now requires a real technical floor and sufficient technical/macro evidence.
- Removed the implicit all-universe sharia hard block; it is now an explicit `sharia_only` policy.

## Unchanged

- v12 resumable jobs.
- v13 persistent OHLCV.
- Final production model weights: Next Leader 25/20/15/10/10/15/5 and Swing 40/15/25/10/10.
