# IDX Super Scanner v9.8.2 Hotfix 8 — Fundamental Period Integrity

## Root causes repaired

1. Period preference was applied to a whole provider row. If a verified IDX
   XBRL row existed but omitted a fact, a valid same-period proxy fact was
   discarded.
2. Quarterly YoY used a fixed four-row offset. Missing periods meant the row
   four positions earlier was not necessarily the same quarter last year.
3. When current quarterly facts were masked, the feature builder silently fell
   back to annual growth while still exposing the value as history growth.

## Fix

- Select the best available source independently for each financial fact.
- Match growth periods by calendar year and quarter and fail closed when a
  comparable period is absent.
- Match prior-year balance-sheet rows by period rather than row position.
- Label mixed-source periods as `OFFICIAL_PARTIAL_PROXY_FILL` instead of
  presenting an incomplete official row as fully confirmed.
- Advance the fundamental feature lineage from `7.5.2` to `7.6.0` so persisted
  outputs identify the repaired calculation.

No ranking weight, eligibility threshold, or technical feature formula changed.
