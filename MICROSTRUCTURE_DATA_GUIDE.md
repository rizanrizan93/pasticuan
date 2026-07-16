# Microstructure Data Guide

## Automatic mode

No additional upload is required. The scanner uses intraday OHLCV to estimate
buy pressure and ARA-lock persistence. These fields are explicitly marked as
proxies and must not be interpreted as actual broker identity or actual queue
lots.

## Broker Summary CSV

Required columns:

```csv
ticker,date,broker_code,buy_value,sell_value
ANTM,2026-07-15,YP,15000000000,5000000000
```

`buy_volume` and `sell_volume` may replace value columns.

## Order Book snapshot CSV

Required columns:

```csv
ticker,timestamp,level,bid_price,bid_lots,offer_price,offer_lots
ANTM,2026-07-15 15:55:00,1,3200,250000,3210,50000
```

Use one timestamp for all captured levels. A snapshot older than 36 hours is
shown but not used to upgrade a signal.
