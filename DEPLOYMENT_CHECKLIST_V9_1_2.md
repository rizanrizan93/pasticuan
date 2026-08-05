# Deployment checklist v9.1.2

Replace these files in the repository root:

- `app.py`
- `database_first.py`
- `free_data_providers.py`
- `scanner.py`
- `scanner_database.py`

For a clean deployment, deploying the complete deploy-only package is preferred.

No new SQL migration is required if `database/migration_v4_research_memory.sql` was previously applied, because market-status and news-review persistence use the existing `source_events` table.

After deployment:

1. Reboot Streamlit.
2. Run `Isi Database` with 20 tickers once.
3. Confirm Queue Summary, Provider Audit, and Database Sync Audit.
4. Confirm at least one of the ready metrics gains more than zero or inspect `BACKFILL_STALLED` details.
5. Increase to 80 snapshot/history, 400 market-status, and 80 news only after the small test succeeds.
