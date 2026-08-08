# Patch Notes — v9.7.0

## Decision architecture

- Added `decision_overlay.py` as a pure, testable methodology boundary.
- Added multi-horizon inventory proxy: 20/60/120/252/504/756D.
- Added lifecycle, distribution gate, reaccumulation quality and anti-chase logic.
- Core Next Leader weights and valuation/fundamental computation remain unchanged.
- Distribution cannot remain rank-eligible.
- Markup anti-chase removes allocation/order eligibility until reaccumulation/entry geometry improves.

## Dashboard

- Added `v9_dashboard.py`.
- Added Top 3 Next Leader and Top 3 Swing report-card dashboard inspired by the Emir layout.
- Added lifecycle, inventory and anti-chase columns to detailed tables.

## Simplification / stability

- Removed 18 unused `scanner.py` imports from `app.py`.
- Removed unused `two_stage_pipeline` imports from `app.py`.
- UI now depends on a smaller public surface.
- No aggressive extraction from `scanner.py` in this release; the stable core is intentionally frozen while new logic is isolated around it.

## Runtime

Synthetic benchmark on this build: multi-horizon overlay ~7.2 seconds estimated for 400 tickers × 800 bars on the validation container. Live scan time remains dominated by network/provider/database I/O.

## Database

No migration required.
