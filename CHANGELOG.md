# Changelog

## v4.3.1 — Free Multi-Source OHLCV

- Added cache-first daily OHLCV orchestration.
- Added expected completed-EOD date logic to avoid treating an intraday candle as final daily data.
- Added merge-on-refresh so provider failures never erase verified historical bars.
- Added official IDX Stock Summary latest-bar patch for stale cached histories.
- Added optional iTick free-tier daily and intraday OHLCV adapter.
- Added persistent internal iTick rate guard capped at 4 calls per minute.
- Added secondary-provider price-basis alignment against overlapping cached closes.
- Added `CACHE_FRESH_VERIFIED`, `LIVE_IDX_EOD_PATCH`, and `LIVE_ITICK_FREE_FALLBACK` source tiers.
- Reduced Streamlit daily market cache TTL to 10 minutes; cache-first logic prevents unnecessary Yahoo requests.
- Added free-data setup documentation and Streamlit secrets example.
- Added four multi-source regression tests; total 103 tests.

## v4.3.0 — TradingView & Stockbit Integration

- Added complete Streamlit deployment manifest.
- Added Pine Script v6 confirmation indicator.
- Added Stockbit screener presets.
- Added normalized TradingView bridge CSV and in-app download tab.
- Increased automatic independent-price verification shortlist from 24 to 40 candidates.
- Relabeled OOS probability as setup-level chronological holdout.
