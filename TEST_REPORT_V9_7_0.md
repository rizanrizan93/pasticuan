# Test Report — IDX Super Scanner v9.7.0

Validation date: 2026-08-08

## Automated validation

Command:

```bash
python validation_v9_7_0.py
```

Result:

- PASS compile all root Python modules
- PASS import all non-Streamlit production modules
- PASS no duplicate top-level symbols in key modules
- PASS 20/60/120/252/504/756D inventory contract
- PASS anti-chase downgrade behavior
- PASS distribution production block
- PASS old durable-job compatibility when new lifecycle fields are absent
- PASS Top 3 dashboard selection/render contract
- PASS synthetic overlay benchmark

Benchmark from validation run:

- 100 tickers × 800 bars: ~1.79s
- estimated 400 tickers × 800 bars: ~7.16s

The benchmark measures the new local inventory overlay only. It does not include network provider, Supabase, news, KSEI/IDX or fundamental retrieval latency.

## Static import boundary

`app.py` scanner imports reduced from 36 to 18; all retained scanner imports are referenced.

`scanner.py` was intentionally not structurally refactored in this release to avoid mixing a high-risk core refactor with a methodology/UI release.

## Environment limitation

The validation container did not have the `streamlit` CLI installed globally, so an interactive `streamlit run app.py` server startup was not executed here. The deployment `requirements.txt` still pins Streamlit and `app.py` compiles successfully.
