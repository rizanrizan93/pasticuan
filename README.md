# IDX Super Scanner v9.7.0 — Modular Inventory & Emir-Style Dashboard

v9.7.0 keeps the v9.6.1 database-first/network-resilient backbone and changes the **decision/UI architecture**, not the fundamental valuation contract.

## What changed

### 1. Lower coupling

- `decision_overlay.py` is a pure OHLCV/decision module. It does **not** import Streamlit, Supabase, scanner providers, or the fundamental engine.
- `v9_dashboard.py` is presentation-only. It accepts DataFrames and renders the Top 3 report card without importing scanner/database/network code.
- `app.py` no longer imports 18 unused symbols from `scanner.py`; all unused `two_stage_pipeline` UI imports were removed.
- `scanner.py` remains unchanged in this release. This is intentional: a large refactor of the stable 11k-line core during a dashboard/methodology change would increase regression risk.

### 2. Multi-horizon smart-money inventory overlay

Every technical-ready ticker now receives an OHLCV proxy across:

- 20D
- 60D
- 120D
- 252D
- 504D
- 756D

The overlay emits:

- `inventory_multi_horizon_score`
- `inventory_multi_horizon_coverage_pct`
- `distribution_risk_score`
- `inventory_lifecycle`
- `anti_chase_gate`
- `markup_extension_pct`
- `reaccumulation_quality_score`
- `accumulation_dominance_pct`

The 756D horizon is supported by raising the bounded per-ticker technical history from 750 to 800 bars.

This is explicitly an **OHLCV proxy**. It never claims to identify a specific broker or shareholder.

### 3. Lifecycle and anti-chasing guardrails

Lifecycle states:

`INVENTORY_COLLECTION → EARLY_CONVERGENCE → MARKUP → REACCUMULATION → DISTRIBUTION`

Guardrails are applied **after** the core v9 score is calculated:

- advanced MARKUP + anti-chase condition: a Next Leader `BUY_ZONE` is downgraded to `WAIT`; Swing execution states are downgraded to `WATCHLIST` and `WAIT_REACCUMULATION`.
- DISTRIBUTION: methodology gate blocks production ranking/order eligibility.
- core business/future fundamental/valuation score math is not rewritten by the lifecycle overlay.

### 4. Emir-style Top 3 dashboard

New first tab: **Top 3 Dashboard** with two sub-tabs:

- Top 3 Next Leader
- Top 3 Swing

Each card displays:

- final score and coverage
- recommendation state
- entry / trigger / stop / TP1 / TP2 / RR
- factor bars
- accumulation gauge
- silent accumulation, multi-horizon inventory, distribution risk and reaccumulation quality
- lifecycle / anti-chase badges
- report-card stars
- thesis summary and primary risk

The old Market Map, The Next Leader, Swing Ready and Portfolio/Audit tables remain available.

## Core production contract retained

1. Read persistent OHLCV/fundamental/narrative evidence first.
2. Refresh only missing/stale evidence.
3. Persist deltas to the database.
4. Compute business quality, future fundamental, valuation/MOS, management/capital allocation, macro/sector, narrative-flow and technical readiness.
5. Apply production evidence gates.
6. Apply lifecycle/anti-chase guardrails as a separate final decision layer.
7. Rank only production-eligible candidates.

## Database

**No new SQL migration is required.** New lifecycle fields are stored inside the existing resumable item/result JSON payload and scan artifacts.

Keep all v9.6.1 database migrations/hotfixes, especially v12 resumable jobs, v13 persistent OHLCV and `permissions_hotfix_v9_4_1.sql`.

## Validation

See `TEST_REPORT_V9_7_0.md` and run:

```bash
python validation_v9_7_0.py
```
