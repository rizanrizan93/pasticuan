# IDX Dual Tab Scanner — Modularized

Files:
- app.py
- data_engine.py
- technical_analyst.py
- fundamental_analyst.py
- catalyst_nlp.py

Run:
```bash
streamlit run app.py
```


## Deployment

For Streamlit Community Cloud, keep `app.py` at the repo root alongside `requirements.txt`.


## Profit-first mode
- Default mode is now conservative.
- Long entries are only allowed when the market regime is BULL and the setup passes strict quality gates.
- Trend lag is no longer a decision input; it is not used to rank trades.
