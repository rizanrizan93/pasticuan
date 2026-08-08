# Migration v9.6.1 → v9.7.0

No SQL migration is required.

## Replace / add

Replace:

- `app.py`
- `simple_focus.py`
- `resumable_app_engine.py`
- `resumable_scan.py`
- `README.md`

Add:

- `decision_overlay.py`
- `v9_dashboard.py`
- `validation_v9_7_0.py`

All existing database SQL files remain valid.

## Important

`app.py` treats `decision_overlay.py` and `v9_dashboard.py` as required deployment files. Commit both new files together with the updated `app.py`.
