# IDX Super Scanner v9.8.2 Hotfix 9 — Universe Sector Integrity

## Proven defect

The uploaded 300-ticker contract retained `sector_idx_ic`, but an existing
fundamental-cache row normalized to the sentinel `UNKNOWN` won the field-level
merge. The explicit uploaded sector was therefore present in metadata yet did
not reach macro issuer mapping. This suppressed valid macro-sector evidence and
reduced ranking coverage.

## Fix

- Treat `UNKNOWN`, `MISSING`, `UNCLASSIFIED`, and null-like text as absent only
  for the cached sector-classification bundle.
- When the uploaded universe contains a valid explicit sector, replace
  `sector`, raw/source/confidence/version fields, `idx_sector`, and
  `sector_idx_ic` atomically.
- Preserve every valid primary sector and all unrelated fundamental fields.

No score weight, threshold, or ranking formula changed.
