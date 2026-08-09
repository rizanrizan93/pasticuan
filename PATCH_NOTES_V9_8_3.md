# IDX Super Scanner v9.8.3 — Evidence Integrity

- Loads durable forward/project evidence in both resumable chunk and finalizer paths.
- Retains chunk-time valuation evidence and persists it to `fundamental_cache`.
- Adds clearly labelled financial-outcome proxies for future capacity, reinvestment, management execution, and capital allocation. These proxies never claim to be direct guidance, management biography, or governance evidence.
- Uses current verified KSEI total shares for point-in-time valuation without a duplicate split adjustment.
- Supports a separately labelled latest-full-year fallback when a true TTM denominator is unavailable.
- Treats HTTPS `idx.id`/`idx.co.id` regulator URLs as official while retaining stricter issuer-domain verification.
- Adds JSONB numeric-tolerant semantic hash V2 plus database-side read-back verification.
- Keeps direct project evidence mandatory for project-led classification; ordinary compounders use a financial-capacity lane.

Validation: `validation_v9_8_3_evidence_integrity.py` and the existing Hotfix 4/9/performance suites.
