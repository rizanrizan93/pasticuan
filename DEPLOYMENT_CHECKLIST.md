# Deployment Checklist v4.3.1

- [ ] Upload all files from this folder to the GitHub repository root.
- [ ] Main file is `app.py`.
- [ ] Python runtime is 3.12.
- [ ] `requirements.txt` is present.
- [ ] Do not upload a real secrets file or API token to GitHub.
- [ ] Optional `ITICK_API_TOKEN` is stored only in Streamlit Secrets.
- [ ] Optional `TWELVE_DATA_API_KEY` is stored only in Streamlit Secrets.
- [ ] Reboot the Streamlit app after replacement.
- [ ] Upload a small 10–20 ticker CSV for the first smoke test.
- [ ] Confirm Audit Universe displays Yahoo, IDX patch, iTick, fresh-cache, stale-cache, and unavailable tiers.
- [ ] Confirm stale `CACHE_FALLBACK` candidates do not receive `BUY_LIMIT`.
- [ ] Run full universe only after the smoke test succeeds.
- [ ] For Multibagger research, enable full-universe fundamentals outside market hours.
