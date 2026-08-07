# Test Report v9.6.0

Final automated suite: 77 passed, 0 failed.

The suite covers:
- persistent OHLCV cold/warm behavior;
- stale-database fallback when public OHLCV fails;
- resumable job/checkpoint/retry/finalizer paths;
- signal checkpoint integrity;
- production ranking finite-score contract;
- macro breadth guard;
- UNKNOWN sector recovery and removal of fake UNKNOWN sector-map score;
- stale fundamental eligibility reset;
- database-over-local fundamental precedence with field-level fallback;
- Next Leader production coverage/quality/freshness gates;
- narrative cannot rescue weak business quality;
- Swing Ready technical floor;
- optional sharia-only policy;
- controlled 20-name quality ranking separation.

`pytest -q -W error::FutureWarning`: 77 passed, 0 FutureWarning.

All 17 production Python modules compile. Sixteen non-UI modules import successfully in the build environment. `app.py` compiles; runtime import/headless Streamlit is not executed here because Streamlit is not installed in the build container.

Live Supabase/Yahoo/iTick/Twelve Data connectivity is not claimed by this report because production credentials/network are not available in the build environment.
