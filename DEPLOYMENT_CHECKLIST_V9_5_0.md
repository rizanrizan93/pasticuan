# Deployment Checklist v9.5.0

1. Replace repository files with the deploy-only package.
2. Preserve Streamlit Secrets.
3. Run `database/migration_v13_persistent_ohlcv.sql`.
4. Run `database/verify_v13_persistent_ohlcv.sql`.
5. Reboot Streamlit.
6. Confirm header `9.5.0-persistent-database-first`.
7. Run 5–20 tickers once and inspect `OHLCV database dan provider audit`.
8. Repeat the same universe; current OHLCV should show a database-current source and avoid public refetch.
9. Confirm only finite `Final Score` rows receive ranks.
10. Expand to the intended 300–400 ticker universe.
