# Deployment Checklist — v9.7.0

1. Keep existing v9.6.1 Supabase schema/migrations.
2. Commit the four replaced production files and both new modules listed in `FILES_TO_REPLACE_V9_7_0.md`.
3. Confirm `decision_overlay.py` and `v9_dashboard.py` exist at repository root.
4. Keep the existing `requirements.txt` and Streamlit secrets.
5. Deploy/reboot Streamlit.
6. Confirm header shows `9.7.0-modular-inventory-dashboard`.
7. Run a small 20-ticker scan first.
8. Check Top 3 Dashboard, The Next Leader, Swing Ready and Portfolio/Audit tabs.
9. Confirm anti-chase candidates display `V9_WAIT_REACCUMULATION` and have no suggested allocation/order eligibility.
10. Confirm distribution candidates are not production-ranked.
11. Then run the normal 400-ticker universe.
