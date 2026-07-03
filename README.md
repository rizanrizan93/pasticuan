# Predictive Stock Scanner

Streamlit scanner probabilistik untuk screening ticker, entry zone, stoploss, takeprofit, dan horizon terbaik.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input

- Paste ticker satu baris atau dipisah koma
- Atau upload CSV berisi kolom ticker/symbol

## Output

- Top 20 setup
- Entry zone
- Stoploss
- TP1 / TP2
- Best horizon per emiten
- CSV export
