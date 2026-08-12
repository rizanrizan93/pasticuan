# IDX Super Scanner v9.8.6 — Free Tier Storage Safety

Storage-only production hardening. No scoring, feature-cache version, actionability gate, SMC/ICT, inventory, macro, or valuation change.

Live connected Supabase cleanup:
- before: ~282 MB database
- after bounded pruning + VACUUM FULL: ~142 MB
- terminal resumable jobs retained: 2 newest

Migrations:
- v18 adds bounded retention RPC
- v19 runs retention automatically after `scan_runs` persistence, fail-soft

Retention defaults keep current caches plus limited recent history rather than unbounded scan/event snapshots.
