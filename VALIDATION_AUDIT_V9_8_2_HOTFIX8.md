# Validation Audit — v9.8.2 Hotfix 8

- DMAS-shaped missing-period regression: PASS.
- Empty official fact / valid same-period proxy fallback: PASS.
- Same-quarter comparison with a non-contiguous history: PASS.
- All current v9.8.2 validators: PASS.

The regression reproduces the production period pattern that previously
reported annual declines in place of Q2 YoY growth.
