# Deployment Checklist v9.6.1

1. Replace the repository files with the contents of this package.
2. Keep existing Supabase migrations; no new migration is required for v9.6.1.
3. In Streamlit Secrets use a backend secret/service-role key, not anon/publishable.
4. Recommended transport settings:
   - `SCANNER_DATABASE_TIMEOUT = "18"`
   - `SCANNER_DATABASE_CONNECT_TIMEOUT = "5"`
   - `SCANNER_DATABASE_READ_ATTEMPTS = "3"`
   - `SCANNER_DATABASE_RETRY_BACKOFF = "0.8"`
5. Reboot/redeploy the Streamlit app.
6. Confirm the header shows `9.6.1-network-resilience`.
7. Press **Segarkan status job** once. A temporary network problem should now report a timeout/network message, not a permissions message.
8. Start/resume the existing job. COMPLETE ticker checkpoints remain in Supabase and are not intentionally reset by this patch.
