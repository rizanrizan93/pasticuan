# Deployment Checklist v9.6.0

1. Replace the repository files with the v9.6.0 deploy-only package.
2. Ensure `issuer_classification.py` is present in repository root.
3. Keep existing Streamlit Secrets unchanged.
4. Do not delete Supabase data, scan jobs, or OHLCV cache.
5. No new SQL migration is required if v12 + permissions hotfix + v13 were already installed.
6. Reboot Streamlit Cloud.
7. Confirm header contains `9.6.0-quality-integrity`.
8. Run a 20-ticker Daily Scan first and inspect OHLCV ready, production coverage, sector source, freshness, Final Score and rank.
9. Only production-ranked rows should have a finite rank; DATA_PENDING/RESEARCH_ONLY rows stay in audit.
