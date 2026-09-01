# Phase 5.6 ZAPI IDX endpoint inventory

Verified against the current ZAPI IDX documentation on 2026-09-01. The catalog exposes 47 endpoints. ZAPI is a transport/cache layer over official IDX data; persisted rows must retain the original IDX URL or attachment URL whenever the endpoint supplies one. A ZAPI response alone does not turn a market-wide statistic into ticker-specific participant or beneficial-owner evidence.

Classes: **A** required bulk daily, **B** required event-driven, **C** required slow-moving, **D** useful conditional, **E** redundant, **F** semantically unsafe for scanner inference, **G** not useful for the 400-equity evidence hub.

| Endpoint | Class | Family / upstream | Shape and 400-universe use | Natural refresh and quota contract | Persistence / temporal meaning / missing semantics |
|---|---:|---|---|---|---|
| `/stock-summary` | A | IDX TradingSummary | Bulk; all equities in paged call | Once per completed IDX session; never per ticker | `evidence_market_daily`; `TRADE_DATE`; empty=`PROVIDER_NO_DATA` |
| `/index-summary` | E | IDX TradingSummary | Bulk indices; stock scanner can derive market context elsewhere | Daily only if index evidence is explicitly requested | Raw/audit only; `TRADE_DATE`; empty=`PROVIDER_NO_DATA` |
| `/top-movers` | E | IDX TradingSummary | Bulk shortlist already derivable from stock summary | Do not spend quota in normal ingestion | No durable destination; empty=`PROVIDER_NO_DATA` |
| `/broker-summary` | F | IDX ExchangeMember summary | Bulk market-wide broker activity, not ticker-specific flow | At most daily for market/member diagnostics | `evidence_brokers` audit metadata only; never `BROKER_DIRECT`; empty=`PROVIDER_NO_DATA` |
| `/companies` | C | IDX ListedCompany | Bulk company list | 30-day TTL or listing event | `evidence_companies`; current reference state; empty=`PROVIDER_NO_DATA` |
| `/securities` | C | IDX ListedCompany | Bulk listed securities and issued shares | Weekly plus listing/capital-action delta | `evidence_companies`; effective/listing dates; empty=`PROVIDER_NO_DATA` |
| `/financial-report` | B | IDX ListedCompany attachments | Bulk by year/period, optional ticker | Quarterly/event-driven; request only missing report periods | `evidence_financial_reports`; `REPORT_DATE` + `PUBLICATION_DATE`; empty=`NO_REPORT` |
| `/market-activity` | E | IDX market activity | Bulk legacy suspend/relisting/UMA view | Specialized `/uma` and `/suspension` preferred | Raw/audit fallback; event/publication dates; empty=`NO_FILE` |
| `/news` | E | IDX news | Global paged feed overlaps announcements/press releases | Do not fetch when canonical feeds cover cursor | Raw/audit only; `PUBLICATION_DATE`; empty=`PROVIDER_NO_DATA` |
| `/sitemap` | D | ZAPI/IDX discovery | Global endpoint discovery | Only after documented route change | `evidence_provider_state`; fetch time; error=`PROVIDER_NO_DATA` |
| `/raw` | F | Arbitrary allowed IDX primary path | Bulk or filtered; semantics depend on caller path | Only for a proven documented gap and allowlisted path | `evidence_raw_payloads`; never scored directly; malformed=`PARSE_FAILURE` |
| `/announcements` | B | IDX NewsAnnouncement | Global incremental feed, page/cursor based | Frequent delta; stop at known event id/hash | `evidence_announcements`; `PUBLICATION_DATE`; empty page is valid end-of-feed |
| `/company-announcements` | D | IDX NewsAnnouncement | Per ticker | Finalist/backfill only; never 400 daily | `evidence_announcements`; `PUBLICATION_DATE`; empty=`NO_REPORT` |
| `/foreign-flow` | A | IDX TradingSummary | Bulk paged daily foreign flow | Once per missing completed session | `evidence_foreign_flow`; `TRADE_DATE`; empty=`PROVIDER_NO_DATA`, never forward-fill |
| `/margin-summary` | D | IDX margin-period summary | Bulk, period-based rather than daily | Refresh only when source period advances | `evidence_risk_events`; source period date; empty=`NO_FILE` |
| `/ownership-files` | B | IDX/KSEI-linked official files | Bulk file index for 1%, 5%, classification, type | Delta by new URL/hash, slow/event-driven | `evidence_ownership_files` then snapshots; report/publication dates; empty=`NO_FILE` |
| `/brokers` | C | IDX ExchangeMember | One bulk list; `full` performs many upstream enrichments | List monthly; full only on changed/new member | `evidence_brokers`; current reference state; empty=`PROVIDER_NO_DATA` |
| `/stock-history` | D | IDX TradingSummary | Per ticker multi-session OHLCV/foreign flow | Only repair missing history; never 400 routine calls | Market/foreign tables; `TRADE_DATE`; empty=`PROVIDER_NO_DATA` |
| `/company-profile` | C | IDX ListedCompany | Per ticker slow-moving profile | Long TTL; changed/new issuers or finalists | `evidence_companies`; reported roles/shareholders remain reported facts; empty=`NO_MATCH` |
| `/calendar` | A | IDX calendar | Bulk market calendar/session evidence | Daily delta with long retention | `evidence_trading_calendar`; actual session date; empty=`NO_FILE` |
| `/ipo` | B | IDX ListingActivity | Bulk listing pipeline | Weekly/event delta | `evidence_capital_actions`; event/publication dates; empty is valid no-event |
| `/trading-info-daily` | D | IDX StockData | Per ticker latest trading detail | Finalist diagnostic only | Market audit facts; `TRADE_DATE`; empty=`NO_MATCH` |
| `/lendable-stock` | D | IDX SLB | Bulk list; optional full boards/files | Weekly or when source file changes | `evidence_risk_events`; effective/source date; empty=`NO_FILE` |
| `/uma` | B | IDX market surveillance | Global paged events | Incremental daily/event-driven | `evidence_risk_events`; event/publication dates; empty is valid no-event |
| `/derivatives` | G | IDX DerivativesData | Derivative contracts, not required equity facts | No normal ingestion | No destination |
| `/index-constituent` | D | IDX index membership | Bulk by index/group | On rebalance or monthly | `evidence_companies` profile metadata; effective date; empty=`NO_FILE` |
| `/suspension` | B | IDX market surveillance | Global paged events | Incremental daily/event-driven | `evidence_risk_events`; event/publication dates; empty is valid no-event |
| `/bonds` | G | IDX BondSukuk | Debt instruments, outside current equity scope | No normal ingestion | No destination |
| `/bond-tickers` | G | IDX BondSukuk | Debt autocomplete | No normal ingestion | No destination |
| `/bond-daily` | G | IDX BondSukuk | Debt daily summary | No normal ingestion | No destination |
| `/bond-trades` | G | IDX BondSukuk | Debt transactions | No normal ingestion | No destination |
| `/bond-repo` | G | IDX BondSukuk | Debt repo data | No normal ingestion | No destination |
| `/bond-index` | G | IDX BondSukuk | Bond index data | No normal ingestion | No destination |
| `/bond-issuers` | G | IDX BondSukuk | Debt issuer reference | No normal ingestion | No destination |
| `/participants` | C | IDX participant reference | Bulk market participant list | Monthly/long TTL | `evidence_brokers`/reference facts; current state; empty=`PROVIDER_NO_DATA` |
| `/primary-dealers` | G | IDX government debt reference | SBN dealers, outside equity scope | No normal ingestion | No destination |
| `/issued-history` | B | IDX ListingActivity | Bulk pageable, optional ticker/action | Event delta by source id/date | `evidence_capital_actions`; listing/event date; empty is valid no-event |
| `/reference` | C | IDX reference lists | Small bulk sector/board/market-time lists | Monthly/long TTL | Company/calendar reference; current/effective state; empty=`PROVIDER_NO_DATA` |
| `/warrant-providers` | G | IDX StructuredWarrant | Structured warrants, outside common-share evidence | No normal ingestion | No destination |
| `/press-release` | B | IDX official press release | Global paged feed | Incremental cursor/hash delta | `evidence_announcements`; `PUBLICATION_DATE`; empty page is valid end-of-feed |
| `/futures-contracts` | G | IDX DerivativesData | Futures codes, outside current equity scope | No normal ingestion | No destination |
| `/additional-listings` | B | IDX ListingActivity | Bulk monthly events | Current and previous month delta | `evidence_capital_actions`; event/publication date; empty is valid no-event |
| `/delistings` | B | IDX ListingActivity | Bulk monthly events | Current and previous month delta | `evidence_capital_actions`/company state; event date; empty is valid no-event |
| `/dividends` | B | IDX ListingActivity | Bulk monthly events | Current and previous month delta | `evidence_capital_actions`; cum/ex/record/payment dates kept distinct; empty valid |
| `/new-listings` | B | IDX ListingActivity | Bulk monthly events | Current and previous month delta | `evidence_capital_actions`/companies; listing/publication dates; empty valid |
| `/rights-offerings` | B | IDX ListingActivity | Bulk monthly events | Current and previous month delta | `evidence_capital_actions`; ex/event/publication dates; empty valid |
| `/stock-splits` | B | IDX ListingActivity | Bulk monthly split/reverse-split events | Current and previous month delta | `evidence_capital_actions`; event/publication dates; empty valid |

## Quota and provenance rules

- Bulk first: stock summary, foreign flow, calendar, announcements, ownership indexes, surveillance, and capital-action feeds are fetched once per evidence key, not once per scanner or ticker.
- Delta only: a valid uniqueness key or payload/document hash makes an already persisted row a cache hit.
- Slow data refreshes slowly: company, broker, participant, and reference endpoints use long TTLs.
- Conditional endpoints are called only for a specific missing gap or finalist cohort.
- HTTP 401/403/404/429, timeout, connection error, empty payload, malformed payload, rate-limit headers, retry-after, and provider-no-data remain distinct provider states.
- Quota values are persisted only when the provider actually supplies them. Absence is `null`, never an invented allowance.
- Official source URLs, attachment URLs, source dates, payload/document hashes, fetch time, parser version, validation state, and freshness state are retained with normalized facts.

Source: https://zpi.web.id/api/finance/idx
