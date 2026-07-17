"""Specialty and intraday scanners for IDX Super Scanner.

Contains Sniper Entry, BPJS, BSJP, Multibagger, PRE-ARA and ARA continuation
logic. It depends on the stable primitives exposed by :mod:`scanner`.
"""
from __future__ import annotations
from scanner import Any, BinaryIO, DownloadReport, IDX_DAILY_FINAL_HOUR, IDX_DAILY_FINAL_MINUTE, IDX_REGULAR_DECISION_START_HOUR, IDX_REGULAR_DECISION_START_MINUTE, Iterable, Mapping, MarketContext, ScanConfig, ThreadPoolExecutor, _clean_ohlcv, _extract_batch, _finite, _jakarta_timestamp, _pipe_parts, _price_structure_target_pair, _read_csv, _safe_text, _silent_accumulation_metrics, _truthy, as_completed, cmf, fetch_itick_ohlcv, idx_ara_pct, idx_daily_price_band, idx_regular_decision_window, idx_tick_size, is_valid_idx_price, math, near_upper_auto_rejection, normalize_idx_ticker, np, obv, pd, round_idx_price, size_stockbit_order

def download_intraday_ohlcv(tickers: Iterable[str], period: str='5d', interval: str='5m', batch_size: int=30, itick_api_token: str='') -> tuple[dict[str, pd.DataFrame], DownloadReport]:
    """Download bounded intraday OHLCV for specialty screens.

    BPJS uses 5-minute bars so the opening-range confirmation can become
    measurable shortly after 09:15 WIB. A per-ticker 15-minute fallback is
    retained when Yahoo does not return 5-minute data. Provider failure returns
    an audit report and never fabricates an intraday signal from daily data.
    """
    import yfinance as yf
    requested = list(dict.fromkeys(tickers))
    histories: dict[str, pd.DataFrame] = {}
    failed: dict[str, str] = {}
    warnings: dict[str, str] = {}
    source_tiers: dict[str, str] = {}
    fallback_interval = '15m' if interval == '5m' else None

    def mark_interval(frame: pd.DataFrame, value: str) -> pd.DataFrame:
        if frame is not None and (not frame.empty):
            frame.attrs['source_interval'] = value
            try:
                frame.attrs['interval_minutes'] = float(value[:-1]) if value.endswith('m') else np.nan
            except Exception:
                frame.attrs['interval_minutes'] = np.nan
        return frame
    for start_idx in range(0, len(requested), max(1, int(batch_size))):
        batch = requested[start_idx:start_idx + max(1, int(batch_size))]
        try:
            raw = yf.download(batch, period=period, interval=interval, group_by='ticker', auto_adjust=True, repair=False, actions=False, threads=True, progress=False, timeout=20, prepost=False)
            for ticker in batch:
                frame = _extract_batch(raw, ticker, len(batch))
                frame = mark_interval(_clean_ohlcv(frame, strict=True), interval)
                if not frame.empty:
                    histories[ticker] = frame
                    source_tiers[ticker] = f'LIVE_YAHOO_INTRADAY_{interval.upper()}'
                else:
                    failed[ticker] = f'Intraday batch {interval} kosong'
        except Exception as exc:
            for ticker in batch:
                failed[ticker] = f'Intraday batch {interval} gagal: {type(exc).__name__}'
    missing = [ticker for ticker in requested if ticker not in histories]

    def retry_one(ticker: str) -> tuple[str, pd.DataFrame, str | None, str | None]:
        primary_error: str | None = None
        try:
            frame = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True, repair=False, actions=False, timeout=15, prepost=False)
            clean = mark_interval(_clean_ohlcv(frame, strict=True), interval)
            if not clean.empty:
                return (ticker, clean, None, None)
            primary_error = f'Intraday individual {interval} kosong'
        except Exception as exc:
            primary_error = f'{type(exc).__name__}: {str(exc)[:100]}'
        if fallback_interval:
            try:
                frame = yf.Ticker(ticker).history(period=period, interval=fallback_interval, auto_adjust=True, repair=False, actions=False, timeout=15, prepost=False)
                clean = mark_interval(_clean_ohlcv(frame, strict=True), fallback_interval)
                if not clean.empty:
                    warning = f'{interval} tidak tersedia; memakai fallback {fallback_interval}'
                    return (ticker, clean, None, warning)
            except Exception as exc:
                fallback_error = f'{type(exc).__name__}: {str(exc)[:100]}'
                primary_error = f'{primary_error}; fallback {fallback_interval}: {fallback_error}'
        return (ticker, pd.DataFrame(), primary_error or 'Intraday tidak tersedia', None)
    if missing:
        with ThreadPoolExecutor(max_workers=min(3, len(missing))) as pool:
            futures = [pool.submit(retry_one, ticker) for ticker in missing]
            for future in as_completed(futures):
                ticker, frame, error, warning = future.result()
                if not frame.empty:
                    histories[ticker] = frame
                    source_interval = str(frame.attrs.get('source_interval', interval)).upper()
                    source_tiers[ticker] = f'LIVE_YAHOO_INTRADAY_{source_interval}'
                    failed.pop(ticker, None)
                    if warning:
                        warnings[ticker] = warning
                else:
                    failed[ticker] = error or 'Intraday tidak tersedia'
    missing = [ticker for ticker in requested if ticker not in histories]
    if missing and str(itick_api_token or '').strip():
        secondary, secondary_report = fetch_itick_ohlcv(missing, api_token=itick_api_token, period=period, interval=interval, max_tickers=len(missing))
        for ticker, frame in secondary.items():
            frame.attrs['source_interval'] = interval
            try:
                frame.attrs['interval_minutes'] = float(interval[:-1]) if interval.endswith('m') else np.nan
            except Exception:
                frame.attrs['interval_minutes'] = np.nan
            histories[ticker] = frame
            source_tiers[ticker] = f'LIVE_ITICK_FREE_INTRADAY_{interval.upper()}'
            failed.pop(ticker, None)
            warnings[ticker] = 'Yahoo intraday gagal; memakai fallback iTick free'
        if not secondary_report.empty:
            for _, row in secondary_report.loc[secondary_report['status'].ne('OK')].iterrows():
                failed.setdefault(str(row['ticker']), _safe_text(row['error']) or _safe_text(row['status']))
    for ticker in requested:
        source_tiers.setdefault(ticker, 'UNAVAILABLE' if ticker not in histories else f'LIVE_YAHOO_INTRADAY_{interval.upper()}')
    report = DownloadReport(requested=requested, downloaded=sorted(histories), failed=failed, provider=f'Free intraday: Yahoo {interval} → iTick optional', adjusted_prices=True, downloaded_at=pd.Timestamp.now(tz='Asia/Jakarta').isoformat(), warnings=warnings, source_tiers=source_tiers)
    return (histories, report)

def _intraday_session(frame: pd.DataFrame, asof: Any | None=None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    attrs = dict(getattr(frame, 'attrs', {}) or {})
    out = _clean_ohlcv(frame, strict=True)
    if out.empty:
        return out
    out.attrs.update(attrs)
    if asof is not None:
        interval_minutes = _finite(attrs.get('interval_minutes'), np.nan)
        if not np.isfinite(interval_minutes) or interval_minutes <= 0:
            try:
                diffs = pd.Series(pd.DatetimeIndex(out.index)[1:] - pd.DatetimeIndex(out.index)[:-1])
                diffs = diffs.dt.total_seconds().div(60.0)
                diffs = diffs[(diffs > 0) & (diffs <= 60)]
                interval_minutes = float(diffs.median()) if not diffs.empty else 5.0
            except Exception:
                interval_minutes = 5.0
        bar_delta = pd.to_timedelta(float(interval_minutes), unit='min')
        cutoff = _jakarta_timestamp(asof).tz_localize(None) - bar_delta
        out = out[pd.DatetimeIndex(out.index) <= cutoff].copy()
        out.attrs.update(attrs)
        if out.empty:
            return out
    last_date = pd.Timestamp(out.index[-1]).date()
    session = out[pd.Index(out.index).map(lambda x: pd.Timestamp(x).date() == last_date)].copy()
    session.attrs.update(attrs)
    return session

def _intraday_interval_minutes(frame: pd.DataFrame) -> float:
    attr_value = _finite(getattr(frame, 'attrs', {}).get('interval_minutes'), np.nan)
    if np.isfinite(attr_value) and attr_value > 0:
        return float(attr_value)
    if frame is None or len(frame.index) < 2:
        return np.nan
    try:
        index = pd.DatetimeIndex(frame.index)
        diffs = pd.Series(index[1:] - index[:-1]).dt.total_seconds().div(60.0)
        diffs = diffs[(diffs > 0) & (diffs <= 60)]
        return float(diffs.median()) if not diffs.empty else np.nan
    except Exception:
        return np.nan

def _intraday_metrics_v440(frame: pd.DataFrame, now: Any | None=None, max_stale_minutes: int=20) -> dict[str, Any]:
    reference = _jakarta_timestamp(now)
    session = _intraday_session(frame, asof=reference)
    if session.empty:
        return {'intraday_bars': 0.0, 'intraday_interval_minutes': np.nan, 'opening_range_bars': 0.0, 'post_orb_bars': 0.0, 'intraday_data_state': 'NO_DATA', 'session_vwap': np.nan, 'session_close_location': np.nan, 'late_volume_acceleration': np.nan, 'opening_volume_ratio': np.nan, 'orb_high': np.nan, 'orb_low': np.nan, 'intraday_last': np.nan, 'intraday_return': np.nan, 'intraday_session_date': None, 'intraday_last_bar_time': pd.NaT, 'intraday_age_minutes': np.nan, 'intraday_fresh': False}
    interval_minutes = _intraday_interval_minutes(session)
    if not np.isfinite(interval_minutes) or interval_minutes <= 0:
        interval_minutes = 5.0
    opening_range_bars = max(1, int(np.ceil(15.0 / interval_minutes)))
    orb_n = min(opening_range_bars, len(session))
    post_orb_bars = max(0, len(session) - opening_range_bars)
    typical = (session['High'] + session['Low'] + session['Close']) / 3.0
    total_volume = float(session['Volume'].sum())
    session_vwap = float((typical * session['Volume']).sum() / total_volume) if total_volume > 0 else np.nan
    day_high = float(session['High'].max())
    day_low = float(session['Low'].min())
    last_close = float(session['Close'].iloc[-1])
    location = (last_close - day_low) / (day_high - day_low) if day_high > day_low else 0.5
    late_acceleration = np.nan
    if len(session) >= 4:
        late_n = min(max(2, int(np.ceil(20.0 / interval_minutes))), max(2, len(session) // 3))
        late_n = min(late_n, len(session) - 1)
        late_mean = float(session['Volume'].tail(late_n).mean())
        prior = session['Volume'].iloc[:-late_n]
        prior_mean = float(prior.mean()) if len(prior) else np.nan
        late_acceleration = late_mean / prior_mean if np.isfinite(prior_mean) and prior_mean > 0 else np.nan
    opening_ratio = np.nan
    opening_slice = session['Volume'].head(orb_n)
    remainder = session['Volume'].iloc[opening_range_bars:]
    if len(opening_slice) and len(remainder):
        remainder_mean = float(remainder.mean())
        opening_mean = float(opening_slice.mean())
        opening_ratio = opening_mean / remainder_mean if remainder_mean > 0 else np.nan
    orb_high = float(session['High'].head(orb_n).max())
    orb_low = float(session['Low'].head(orb_n).min())
    first_open = float(session['Open'].iloc[0])
    intraday_return = last_close / first_open - 1 if first_open > 0 else np.nan
    last_bar_time = pd.Timestamp(session.index[-1])
    completed_at = last_bar_time + pd.to_timedelta(float(interval_minutes), unit='min')
    reference_naive = reference.tz_localize(None)
    age_minutes = max(0.0, (reference_naive - completed_at).total_seconds() / 60.0)
    same_session_date = last_bar_time.date() == reference_naive.date()
    stale_limit = max(float(max_stale_minutes), 2.0 * interval_minutes + 5.0)
    fresh = bool(same_session_date and age_minutes <= stale_limit)
    if not fresh:
        data_state = 'STALE_SESSION'
    elif post_orb_bars >= 1:
        data_state = 'LIVE_READY'
    else:
        data_state = 'OPENING_RANGE_FORMING'
    return {'intraday_bars': float(len(session)), 'intraday_interval_minutes': float(interval_minutes), 'opening_range_bars': float(opening_range_bars), 'post_orb_bars': float(post_orb_bars), 'intraday_data_state': data_state, 'session_vwap': session_vwap, 'session_close_location': location, 'late_volume_acceleration': late_acceleration, 'opening_volume_ratio': opening_ratio, 'orb_high': orb_high, 'orb_low': orb_low, 'intraday_last': last_close, 'intraday_return': intraday_return, 'intraday_session_date': last_bar_time.date().isoformat(), 'intraday_last_bar_time': last_bar_time, 'intraday_age_minutes': round(age_minutes, 1), 'intraday_fresh': fresh}

def _fundamental_records(fundamentals: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if fundamentals is None or fundamentals.empty or 'ticker' not in fundamentals:
        return {}
    return {str(row['ticker']): row.to_dict() for _, row in fundamentals.drop_duplicates('ticker', keep='last').iterrows()}

def _specialty_sizing(entry: float, stop: float, cfg: ScanConfig, position_cap: float, risk_cap: float) -> dict[str, Any]:
    """Conservative indicative sizing for manual high-risk specialty trades."""
    specialty_cfg = cfg.replace(max_position_pct=min(cfg.max_position_pct, position_cap), risk_per_trade_pct=min(cfg.risk_per_trade_pct, risk_cap))
    sized = size_stockbit_order(entry, stop, specialty_cfg)
    return {'suggested_lots': int(_finite(sized.get('suggested_lots'), 0)), 'capital_required_idr': _finite(sized.get('capital_required_idr'), 0), 'max_loss_idr': _finite(sized.get('max_loss_idr'), 0), 'specialty_position_cap_pct': 100.0 * position_cap, 'specialty_risk_cap_pct': 100.0 * risk_cap}

def scan_multibagger_candidates(prepared: Mapping[str, pd.DataFrame], fundamentals: pd.DataFrame | None, core_signals: pd.DataFrame | None=None, config: ScanConfig | None=None) -> pd.DataFrame:
    """Rank long-horizon growth/quality candidates; not a return guarantee."""
    cfg = config or ScanConfig()
    f_map = _fundamental_records(fundamentals)
    signal_map: dict[str, dict[str, Any]] = {}
    if core_signals is not None and (not core_signals.empty) and ('ticker' in core_signals):
        ranked = core_signals.copy()
        ranked['_q'] = pd.to_numeric(ranked.get('composite_score', ranked.get('quality_score')), errors='coerce').fillna(0)
        for ticker, group in ranked.sort_values('_q', ascending=False).groupby('ticker', sort=False):
            signal_map[str(ticker)] = group.iloc[0].to_dict()
    rows: list[dict[str, Any]] = []
    for ticker, frame in prepared.items():
        if frame is None or frame.empty or len(frame) < 220:
            continue
        fund = f_map.get(ticker, {})
        coverage = _finite(fund.get('fundamental_coverage'), 0.0)
        if coverage <= 0:
            continue
        row = frame.iloc[-1]
        close = _finite(row.get('Close'), 0.0)
        adtv = _finite(row.get('ADTV20'), 0.0)
        revenue_growth = _finite(fund.get('revenue_growth'), np.nan)
        earnings_growth = _finite(fund.get('earnings_growth'), np.nan)
        roe = _finite(fund.get('roe'), np.nan)
        roa = _finite(fund.get('roa'), np.nan)
        net_margin = _finite(fund.get('net_margin'), np.nan)
        operating_margin = _finite(fund.get('operating_margin'), np.nan)
        debt_equity = _finite(fund.get('debt_equity'), np.nan)
        current_ratio = _finite(fund.get('current_ratio'), np.nan)
        cash_to_debt = _finite(fund.get('cash_to_debt'), np.nan)
        ocf = _finite(fund.get('operating_cash_flow'), np.nan)
        fcf = _finite(fund.get('free_cash_flow'), np.nan)
        peg = _finite(fund.get('peg_ratio'), np.nan)
        fcf_yield = _finite(fund.get('fcf_yield'), np.nan)
        market_cap = _finite(fund.get('market_cap'), np.nan)
        fundamental_model = _safe_text(fund.get('fundamental_model')) or 'GENERAL'
        is_financial = fundamental_model == 'FINANCIAL'
        red_flags = _safe_text(fund.get('fundamental_red_flags'))
        roc60 = _finite(row.get('ROC60'), -1.0)
        roc120 = _finite(row.get('ROC120'), -1.0)
        rs60 = _finite(row.get('REL_STRENGTH60'), -1.0)
        dist_high = _finite(row.get('DIST_52W_HIGH'), -1.0)
        cmf_v = _finite(row.get('CMF20'), -1.0)
        obv_up = _finite(row.get('OBV_SLOPE10'), -1.0) > 0
        accumulation, up_down = _silent_accumulation_metrics(frame)
        growth_score = 0.0
        growth_score += 15.0 if revenue_growth >= 0.2 else 10.0 if revenue_growth >= 0.1 else 4.0 if revenue_growth >= 0 else 0.0
        growth_score += 15.0 if earnings_growth >= 0.25 else 10.0 if earnings_growth >= 0.12 else 4.0 if earnings_growth >= 0 else 0.0
        profitability_score = 0.0
        profitability_score += 8.0 if roe >= 0.2 else 5.0 if roe >= 0.12 else 2.0 if roe >= 0.08 else 0.0
        profitability_score += 4.0 if roa >= 0.08 else 2.0 if roa >= 0.04 else 0.0
        profitability_score += 4.0 if operating_margin >= 0.15 else 2.0 if operating_margin >= 0.08 else 0.0
        profitability_score += 4.0 if net_margin >= 0.12 else 2.0 if net_margin >= 0.06 else 0.0
        balance_score = 0.0
        if not is_financial:
            balance_score += 4.0 if np.isfinite(ocf) and ocf > 0 else 0.0
            balance_score += 3.0 if np.isfinite(fcf) and fcf > 0 else 0.0
            balance_score += 3.0 if np.isfinite(debt_equity) and debt_equity <= 0.8 else 1.0 if np.isfinite(debt_equity) and debt_equity <= 1.5 else 0.0
            balance_score += 3.0 if np.isfinite(current_ratio) and current_ratio >= 1.5 else 1.0 if np.isfinite(current_ratio) and current_ratio >= 1.0 else 0.0
            balance_score += 2.0 if np.isfinite(cash_to_debt) and cash_to_debt >= 0.5 else 1.0 if np.isfinite(cash_to_debt) and cash_to_debt >= 0.2 else 0.0
        solvency_fields = (debt_equity, current_ratio, cash_to_debt)
        solvency_coverage = 100.0 * sum((np.isfinite(value) for value in solvency_fields)) / len(solvency_fields)
        valuation_score = 0.0
        valuation_score += 6.0 if np.isfinite(peg) and 0 < peg <= 1.5 else 3.0 if np.isfinite(peg) and peg <= 2.5 else 0.0
        valuation_score += 4.0 if np.isfinite(fcf_yield) and fcf_yield >= 0.04 else 2.0 if np.isfinite(fcf_yield) and fcf_yield > 0 else 0.0
        momentum_score = 0.0
        momentum_score += 6.0 if roc60 >= 0.15 else 4.0 if roc60 >= 0.07 else 0.0
        momentum_score += 5.0 if roc120 >= 0.25 else 3.0 if roc120 >= 0.12 else 0.0
        momentum_score += 4.0 if rs60 > 0 else 0.0
        momentum_score += 3.0 if dist_high >= -0.15 else 1.0 if dist_high >= -0.3 else 0.0
        momentum_score += 2.0 if close > _finite(row.get('EMA200'), float('inf')) else 0.0
        accumulation_score = 0.0
        accumulation_score += 5.0 if accumulation >= 80 else 3.0 if accumulation >= 65 else 0.0
        accumulation_score += 3.0 if cmf_v > 0 and obv_up else 0.0
        accumulation_score += 2.0 if adtv >= 3000000000 else 1.0 if adtv >= 1000000000 else 0.0
        total = growth_score + profitability_score + balance_score + valuation_score + momentum_score + accumulation_score
        severe_flags = any((flag in red_flags for flag in ('Margin bersih negatif', 'OCF negatif', 'DER tinggi')))
        if severe_flags:
            total = min(total, 69.0)
        if np.isfinite(market_cap) and market_cap < 300000000000:
            total -= 4.0
        total = max(0.0, min(100.0, total))
        if total < 60:
            continue
        if total >= 82 and coverage >= 70 and (adtv >= 1500000000) and (not severe_flags) and (not is_financial) and (solvency_coverage >= 66.0):
            status = 'MULTIBAGGER_A_CANDIDATE'
        elif total >= 72 and coverage >= 60 and (not severe_flags):
            status = 'MULTIBAGGER_B_CANDIDATE'
        else:
            status = 'MULTIBAGGER_WATCHLIST'
        sig = signal_map.get(ticker, {})
        technical_entry_state = _safe_text(sig.get('status')) or 'NO_ACTIVE_ENTRY_SETUP'
        rows.append({'ticker': ticker, 'multibagger_status': status, 'multibagger_score': round(total, 1), 'growth_score': round(growth_score, 1), 'profitability_score': round(profitability_score, 1), 'balance_sheet_score': round(balance_score, 1), 'valuation_score': round(valuation_score, 1), 'momentum_score': round(momentum_score, 1), 'accumulation_score': round(accumulation_score, 1), 'fundamental_coverage': coverage, 'fundamental_score': fund.get('fundamental_score'), 'revenue_growth': revenue_growth, 'earnings_growth': earnings_growth, 'roe': roe, 'roa': roa, 'net_margin': net_margin, 'debt_equity': debt_equity, 'current_ratio': current_ratio, 'cash_to_debt': cash_to_debt, 'operating_cash_flow': ocf, 'free_cash_flow': fcf, 'solvency_coverage': round(solvency_coverage, 1), 'fundamental_model': fundamental_model, 'peg_ratio': peg, 'fcf_yield': fcf_yield, 'market_cap': market_cap, 'roc60': roc60, 'roc120': roc120, 'relative_strength60': rs60, 'distance_52w_high': dist_high, 'silent_accumulation_score': accumulation, 'up_down_value_ratio20': up_down, 'adtv20_idr': adtv, 'active_setup': sig.get('setup', ''), 'technical_entry_state': technical_entry_state, 'entry': sig.get('entry', np.nan), 'stop_loss': sig.get('stop_loss', np.nan), 'tp1': sig.get('tp1', np.nan), 'tp2': sig.get('tp2', np.nan), 'horizon': '12–36 months; quarterly review', 'red_flags': red_flags, 'note': 'Financial-sector solvency needs CAR/NPL/LDR verification; generic model caps A grade' if is_financial else 'Candidate ranking, not a forecast or guaranteed multiple'})
    result = pd.DataFrame(rows)
    if not result.empty:
        rank = {'MULTIBAGGER_A_CANDIDATE': 0, 'MULTIBAGGER_B_CANDIDATE': 1, 'MULTIBAGGER_WATCHLIST': 2}
        result['_rank'] = result['multibagger_status'].map(rank).fillna(9)
        result = result.sort_values(['_rank', 'multibagger_score', 'adtv20_idr'], ascending=[True, False, False]).drop(columns='_rank').reset_index(drop=True)
    return result

def build_specialty_screens(prepared: Mapping[str, pd.DataFrame], fundamentals: pd.DataFrame | None=None, core_signals: pd.DataFrame | None=None, market_context: MarketContext | None=None, intraday: Mapping[str, pd.DataFrame] | None=None, config: ScanConfig | None=None, now: Any | None=None) -> dict[str, pd.DataFrame]:
    """Create all specialty tables from the same prepared daily universe."""
    return {'sniper': scan_sniper_entries(core_signals, prepared, config=config), 'bsjp': scan_bsjp_candidates(prepared, intraday=intraday, core_signals=core_signals, config=config, now=now, market_context=market_context), 'bpjs': scan_bpjs_candidates(prepared, intraday=intraday, core_signals=core_signals, config=config, now=now, market_context=market_context), 'multibagger': scan_multibagger_candidates(prepared, fundamentals, core_signals=core_signals, config=config), 'ara_hunter': scan_ara_hunter_candidates(prepared, intraday=intraday, core_signals=core_signals, config=config, now=now, market_context=market_context)}

def _intraday_metrics(frame: pd.DataFrame, now: Any | None=None, max_stale_minutes: int=20) -> dict[str, Any]:
    """Add a completed-session flag without making old intraday bars 'live'.

    BPJS/BSJP still require fresh bars during their trading windows. ARA Hunter,
    however, may legitimately review the completed intraday session after close
    or during the next pre-market. v4.4 incorrectly treated that completed
    session as unusable merely because it was older than 20 minutes.
    """
    metrics = _intraday_metrics_v440(frame, now=now, max_stale_minutes=max_stale_minutes)
    reference = _jakarta_timestamp(now)
    session_date_text = metrics.get('intraday_session_date')
    complete = False
    if session_date_text:
        try:
            session_date = pd.Timestamp(session_date_text).date()
            today = reference.date()
            after_final = (reference.hour, reference.minute) >= (IDX_DAILY_FINAL_HOUR, IDX_DAILY_FINAL_MINUTE)
            premarket = (reference.hour, reference.minute) < (IDX_REGULAR_DECISION_START_HOUR, IDX_REGULAR_DECISION_START_MINUTE)
            complete = bool(session_date < today or (session_date == today and after_final) or (reference.weekday() >= 5 and session_date <= today) or (premarket and session_date < today))
        except Exception:
            complete = False
    metrics['intraday_session_complete'] = complete
    if complete and metrics.get('intraday_data_state') == 'STALE_SESSION':
        metrics['intraday_data_state'] = 'FINAL_SESSION'
    return metrics

def _specialty_hard_context_map(signals: pd.DataFrame | None) -> dict[str, str]:
    """Return only genuinely untradeable context for speculative screens."""
    result: dict[str, str] = {}
    if signals is None or signals.empty or 'ticker' not in signals:
        return result
    hard_tokens = ('SUSPENS', 'FCA', 'FULL CALL AUCTION', 'PENGHENTIAN SEMENTARA', 'DATA ABSOLUT', 'OHLCV FINAL/LIVE TIDAK MEMADAI', 'KONFLIK QUOTE', 'KONFLIK HARGA', 'UNAVAILABLE')
    for ticker, group in signals.groupby('ticker'):
        messages: list[str] = []
        for _, row in group.iterrows():
            candidates = [_safe_text(row.get('critical_blockers')), _safe_text(row.get('analyst_hard_blockers'))]
            if _truthy(row.get('market_status_critical_blocker', False)):
                candidates.append('Status perdagangan IDX negatif')
            if _truthy(row.get('quote_critical_blocker', False)):
                candidates.append('Konflik quote/OHLCV')
            for text in candidates:
                upper = text.upper()
                if text and any((token in upper for token in hard_tokens)):
                    messages.append(text)
        result[str(ticker)] = ' • '.join(dict.fromkeys(messages))
    return result

def _clip_score(value: Any) -> float:
    return float(max(0.0, min(100.0, _finite(value, 0.0))))

def _ara_intraday_microstructure(frame: pd.DataFrame, ara_price: float, now: Any | None=None) -> dict[str, Any]:
    """Estimate aggressive buying and ARA-lock quality from intraday OHLCV.

    These are transparent proxies.  They are not broker identity, queue lots,
    or exchange market depth.  The function intentionally labels them as proxy
    fields so the dashboard cannot misrepresent inferred data as observed data.
    """
    session = _intraday_session(frame, asof=_jakarta_timestamp(now))
    empty = {'proxy_intraday_bars': 0, 'proxy_signed_volume_imbalance': np.nan, 'proxy_clv_pressure': np.nan, 'proxy_vwap_hold_ratio': np.nan, 'proxy_late_buy_imbalance': np.nan, 'proxy_late_volume_share': np.nan, 'proxy_directional_efficiency': np.nan, 'proxy_ara_lock_ratio': np.nan, 'proxy_final_lock_bars': 0, 'proxy_relock_count': 0, 'proxy_max_unlock_pct': np.nan, 'proxy_range_compression': np.nan, 'orderflow_proxy_score': np.nan, 'queue_proxy_score': np.nan}
    if session is None or session.empty or len(session) < 2:
        return empty
    volume = pd.to_numeric(session['Volume'], errors='coerce').fillna(0.0).clip(lower=0.0)
    close = pd.to_numeric(session['Close'], errors='coerce')
    open_ = pd.to_numeric(session['Open'], errors='coerce')
    high = pd.to_numeric(session['High'], errors='coerce')
    low = pd.to_numeric(session['Low'], errors='coerce')
    total_volume = float(volume.sum())
    if total_volume <= 0 or close.dropna().empty:
        return empty
    delta = close.diff()
    fallback_direction = np.sign((close - open_).fillna(0.0))
    direction = pd.Series(np.where(delta > 0, 1.0, np.where(delta < 0, -1.0, fallback_direction)), index=session.index)
    signed_imbalance = float((direction * volume).sum() / total_volume)
    bar_range = (high - low).replace(0.0, np.nan)
    clv = ((2.0 * close - high - low) / bar_range).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    clv_pressure = float((clv * volume).sum() / total_volume)
    typical = (high + low + close) / 3.0
    cum_volume = volume.cumsum().replace(0.0, np.nan)
    cumulative_vwap = (typical * volume).cumsum() / cum_volume
    vwap_hold = float((close >= cumulative_vwap).fillna(False).mean())
    late_n = max(2, int(np.ceil(len(session) * 0.25)))
    late_volume = volume.tail(late_n)
    late_total = float(late_volume.sum())
    late_direction = direction.tail(late_n)
    late_buy_imbalance = float((late_direction * late_volume).sum() / late_total) if late_total > 0 else np.nan
    late_share = float(late_total / total_volume)
    close_path = close.dropna()
    gross_path = float(close_path.diff().abs().sum())
    net_path = float(close_path.iloc[-1] - close_path.iloc[0]) if len(close_path) >= 2 else 0.0
    directional_efficiency = float(max(-1.0, min(1.0, net_path / gross_path))) if gross_path > 0 else 0.0
    tick = idx_tick_size(max(_finite(ara_price, 0.0), _finite(close.iloc[-1], 0.0)))
    lock_tolerance = max(float(tick), _finite(ara_price, 0.0) * 0.0015)
    near_lock = close >= float(ara_price) - lock_tolerance if ara_price > 0 else pd.Series(False, index=session.index)
    tail_lock = near_lock.tail(max(3, int(np.ceil(len(session) * 0.35))))
    lock_ratio = float(tail_lock.mean()) if len(tail_lock) else 0.0
    final_lock_bars = 0
    for value in reversed(near_lock.fillna(False).tolist()):
        if value:
            final_lock_bars += 1
        else:
            break
    relock_count = int((~near_lock.shift(1, fill_value=False) & near_lock).sum())
    max_unlock_pct = np.nan
    if bool(near_lock.any()) and ara_price > 0:
        first_lock_pos = int(np.argmax(near_lock.to_numpy(dtype=bool)))
        after_first = close.iloc[first_lock_pos:]
        max_unlock_pct = float(after_first.min() / ara_price - 1.0)
    earlier_range = bar_range.iloc[:-late_n].dropna()
    final_range = bar_range.tail(late_n).dropna()
    range_compression = np.nan
    if not earlier_range.empty and (not final_range.empty) and (float(earlier_range.mean()) > 0):
        range_compression = float(final_range.mean() / earlier_range.mean())

    def positive_component(value: float, weak: float, strong: float) -> float:
        if not np.isfinite(value):
            return 45.0
        if value <= weak:
            return max(0.0, 50.0 * (value - -1.0) / max(weak + 1.0, 1e-09))
        if value >= strong:
            return 100.0
        return 50.0 + 50.0 * (value - weak) / max(strong - weak, 1e-09)
    flow_score = _clip_score(0.25 * positive_component(signed_imbalance, 0.0, 0.45) + 0.2 * positive_component(clv_pressure, 0.0, 0.65) + 0.2 * max(0.0, min(100.0, vwap_hold * 100.0)) + 0.2 * positive_component(_finite(late_buy_imbalance, 0.0), 0.0, 0.5) + 0.15 * positive_component(directional_efficiency, 0.0, 0.6))
    compression_component = 50.0
    if np.isfinite(range_compression):
        compression_component = 100.0 if range_compression <= 0.45 else 70.0 if range_compression <= 0.75 else 35.0
    unlock_component = 50.0
    if np.isfinite(max_unlock_pct):
        unlock_component = 100.0 if max_unlock_pct >= -0.005 else 75.0 if max_unlock_pct >= -0.015 else 35.0
    queue_score = _clip_score(35.0 * lock_ratio + 20.0 * min(1.0, final_lock_bars / 5.0) + 15.0 * min(1.0, relock_count / 3.0) + 15.0 * unlock_component / 100.0 + 15.0 * compression_component / 100.0)
    return {'proxy_intraday_bars': int(len(session)), 'proxy_signed_volume_imbalance': signed_imbalance, 'proxy_clv_pressure': clv_pressure, 'proxy_vwap_hold_ratio': vwap_hold, 'proxy_late_buy_imbalance': late_buy_imbalance, 'proxy_late_volume_share': late_share, 'proxy_directional_efficiency': directional_efficiency, 'proxy_ara_lock_ratio': lock_ratio, 'proxy_final_lock_bars': int(final_lock_bars), 'proxy_relock_count': relock_count, 'proxy_max_unlock_pct': max_unlock_pct, 'proxy_range_compression': range_compression, 'orderflow_proxy_score': round(flow_score, 1), 'queue_proxy_score': round(queue_score, 1)}

def _ara_history_profile(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute strictly historical (excluding latest bar) ARA base rates."""
    result = {'historical_ara_events': 0, 'historical_next_day_positive_rate': np.nan, 'historical_next_day_strong_rate': np.nan, 'historical_next_day_ara_rate': np.nan, 'historical_median_next_day_return': np.nan, 'historical_pre_ara_samples': 0, 'historical_pre_ara_hit_rate': np.nan}
    if frame is None or frame.empty or len(frame) < 65:
        return result
    data = frame.iloc[:-1].copy()
    close = pd.to_numeric(data.get('Close'), errors='coerce')
    open_ = pd.to_numeric(data.get('Open'), errors='coerce')
    high = pd.to_numeric(data.get('High'), errors='coerce')
    low = pd.to_numeric(data.get('Low'), errors='coerce')
    if close is None or close.dropna().size < 10:
        return result
    next_returns: list[float] = []
    next_ara_flags: list[bool] = []
    next_strong_flags: list[bool] = []
    pre_hits: list[bool] = []
    for i in range(1, len(data) - 1):
        prev_close = _finite(close.iloc[i - 1], 0.0)
        current_close = _finite(close.iloc[i], 0.0)
        next_close = _finite(close.iloc[i + 1], 0.0)
        if min(prev_close, current_close, next_close) <= 0:
            continue
        ret = current_close / prev_close - 1.0
        ara_threshold = 0.88 * idx_ara_pct(prev_close)
        next_ret = next_close / current_close - 1.0
        next_threshold = 0.88 * idx_ara_pct(current_close)
        if ret >= ara_threshold:
            next_returns.append(next_ret)
            next_ara_flags.append(next_ret >= next_threshold)
            next_strong_flags.append(next_ret >= max(0.04, 0.35 * idx_ara_pct(current_close)))
        vol_ratio = _finite(data.iloc[i].get('VOL_RATIO'), np.nan)
        close_location = _finite(data.iloc[i].get('CLOSE_LOCATION'), np.nan)
        body_atr = _finite(data.iloc[i].get('BODY_ATR'), np.nan)
        high20 = _finite(data.iloc[i].get('HIGH20_PREV'), np.nan)
        cmf = _finite(data.iloc[i].get('CMF20'), np.nan)
        obv = _finite(data.iloc[i].get('OBV_SLOPE10'), np.nan)
        candidate = bool(np.isfinite(vol_ratio) and vol_ratio >= 1.5 and np.isfinite(close_location) and (close_location >= 0.72) and np.isfinite(body_atr) and (body_atr >= 0.45) and (0.015 <= ret <= 0.25) and (not np.isfinite(high20) or current_close > high20 or ret >= 0.04) and (np.isfinite(cmf) and cmf > 0 or (np.isfinite(obv) and obv > 0)))
        if candidate:
            pre_hits.append(next_ret >= next_threshold)
    if next_returns:
        result.update({'historical_ara_events': len(next_returns), 'historical_next_day_positive_rate': float(np.mean(np.array(next_returns) > 0)), 'historical_next_day_strong_rate': float(np.mean(next_strong_flags)), 'historical_next_day_ara_rate': float(np.mean(next_ara_flags)), 'historical_median_next_day_return': float(np.median(next_returns))})
    if pre_hits:
        result.update({'historical_pre_ara_samples': len(pre_hits), 'historical_pre_ara_hit_rate': float(np.mean(pre_hits))})
    return result

def parse_orderbook_snapshot_csv(source: bytes | BinaryIO | pd.DataFrame) -> pd.DataFrame:
    """Parse a user-supplied 1-10 level order-book snapshot.

    Required logical fields: ticker, timestamp, bid_price, bid_lots,
    offer_price, offer_lots.  Common aliases are accepted.  This optional
    evidence is kept separate from the automatic queue proxy.
    """
    frame = _read_csv(source)
    frame.columns = [str(column).strip().lower().replace(' ', '_') for column in frame.columns]
    aliases = {'time': 'timestamp', 'datetime': 'timestamp', 'asof': 'timestamp', 'bid': 'bid_price', 'bidprice': 'bid_price', 'buy_price': 'bid_price', 'bid_volume': 'bid_lots', 'bid_vol': 'bid_lots', 'buy_lots': 'bid_lots', 'offer': 'offer_price', 'ask': 'offer_price', 'ask_price': 'offer_price', 'offerprice': 'offer_price', 'sell_price': 'offer_price', 'offer_volume': 'offer_lots', 'ask_lots': 'offer_lots', 'sell_lots': 'offer_lots'}
    frame = frame.rename(columns={column: aliases.get(column, column) for column in frame.columns})
    required = {'ticker', 'timestamp', 'bid_price', 'bid_lots', 'offer_price', 'offer_lots'}
    if not required.issubset(frame.columns):
        raise ValueError('Order-book CSV wajib memiliki ticker, timestamp, bid_price, bid_lots, offer_price, dan offer_lots')
    frame['ticker'] = frame['ticker'].map(normalize_idx_ticker)
    frame['timestamp'] = pd.to_datetime(frame['timestamp'], errors='coerce')
    for column in ('bid_price', 'bid_lots', 'offer_price', 'offer_lots'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.dropna(subset=['ticker', 'timestamp'])
    frame = frame[(frame['bid_lots'].fillna(0) >= 0) & (frame['offer_lots'].fillna(0) >= 0)]
    rows: list[dict[str, Any]] = []
    for ticker, group in frame.groupby('ticker', sort=False):
        latest = group['timestamp'].max()
        snapshot = group[group['timestamp'].eq(latest)].copy()
        snapshot = snapshot.sort_values('level') if 'level' in snapshot.columns else snapshot
        bid_lots = float(snapshot['bid_lots'].fillna(0).sum())
        offer_lots = float(snapshot['offer_lots'].fillna(0).sum())
        total = bid_lots + offer_lots
        imbalance = (bid_lots - offer_lots) / total if total > 0 else np.nan
        best_bid = float(snapshot['bid_price'].max()) if snapshot['bid_price'].notna().any() else np.nan
        best_offer = float(snapshot['offer_price'].min()) if snapshot['offer_price'].notna().any() else np.nan
        midpoint = (best_bid + best_offer) / 2.0 if np.isfinite(best_bid) and np.isfinite(best_offer) else np.nan
        spread_pct = (best_offer - best_bid) / midpoint if np.isfinite(midpoint) and midpoint > 0 else np.nan
        signal = 'STRONG_BUY_QUEUE' if np.isfinite(imbalance) and imbalance >= 0.35 else 'BUY_QUEUE' if np.isfinite(imbalance) and imbalance >= 0.12 else 'SELL_HEAVY' if np.isfinite(imbalance) and imbalance <= -0.25 else 'BALANCED'
        rows.append({'ticker': ticker, 'orderbook_asof': latest, 'orderbook_levels': int(len(snapshot)), 'orderbook_bid_lots': bid_lots, 'orderbook_offer_lots': offer_lots, 'orderbook_imbalance': imbalance, 'orderbook_best_bid': best_bid, 'orderbook_best_offer': best_offer, 'orderbook_spread_pct': spread_pct, 'orderbook_signal': signal, 'orderbook_note': 'Snapshot pengguna; bukan feed real-time otomatis'})
    return pd.DataFrame(rows)

def apply_ara_external_confirmation(ara: pd.DataFrame, broker_summary: pd.DataFrame | None=None, orderbook: pd.DataFrame | None=None, now: Any | None=None) -> pd.DataFrame:
    """Add optional observed broker/order-book evidence to ARA proxy signals."""
    if ara is None or ara.empty:
        return pd.DataFrame() if ara is None else ara.copy()
    out = ara.copy()
    broker_summary = broker_summary if broker_summary is not None else pd.DataFrame()
    orderbook = orderbook if orderbook is not None else pd.DataFrame()
    if not broker_summary.empty:
        keep = [c for c in ('ticker', 'broksum_asof', 'broksum_days', 'broksum_net_ratio', 'broksum_signal', 'top_net_buy_brokers', 'top_net_sell_brokers', 'broksum_note') if c in broker_summary]
        out = out.merge(broker_summary[keep].drop_duplicates('ticker'), on='ticker', how='left')
    if 'broksum_signal' not in out:
        out['broksum_signal'] = 'UNAVAILABLE'
    out['broksum_signal'] = out['broksum_signal'].fillna('UNAVAILABLE')
    if not orderbook.empty:
        out = out.merge(orderbook.drop_duplicates('ticker'), on='ticker', how='left')
    if 'orderbook_signal' not in out:
        out['orderbook_signal'] = 'UNAVAILABLE'
    out['orderbook_signal'] = out['orderbook_signal'].fillna('UNAVAILABLE')
    reference = _jakarta_timestamp(now)
    for idx, row in out.iterrows():
        bonus = 0.0
        evidence: list[str] = []
        broksum_signal = _safe_text(row.get('broksum_signal')).upper()
        if broksum_signal == 'ACCUMULATION_PROXY':
            bonus += 6.0
            evidence.append('Broker summary: akumulasi')
        elif broksum_signal == 'DISTRIBUTION_PROXY':
            bonus -= 9.0
            evidence.append('Broker summary: distribusi')
        orderbook_signal = _safe_text(row.get('orderbook_signal')).upper()
        orderbook_fresh = False
        orderbook_asof = pd.to_datetime(row.get('orderbook_asof'), errors='coerce')
        if pd.notna(orderbook_asof):
            if orderbook_asof.tzinfo is None:
                orderbook_asof = orderbook_asof.tz_localize('Asia/Jakarta')
            else:
                orderbook_asof = orderbook_asof.tz_convert('Asia/Jakarta')
            age_hours = (reference - orderbook_asof).total_seconds() / 3600.0
            orderbook_fresh = -1.0 <= age_hours <= 36.0
            out.at[idx, 'orderbook_age_hours'] = round(age_hours, 1)
        if orderbook_fresh and orderbook_signal == 'STRONG_BUY_QUEUE':
            bonus += 10.0
            evidence.append('Order book: antrean beli kuat')
        elif orderbook_fresh and orderbook_signal == 'BUY_QUEUE':
            bonus += 5.0
            evidence.append('Order book: bid dominan')
        elif orderbook_fresh and orderbook_signal == 'SELL_HEAVY':
            bonus -= 12.0
            evidence.append('Order book: offer dominan')
        elif orderbook_signal != 'UNAVAILABLE' and (not orderbook_fresh):
            evidence.append('Snapshot order book kedaluwarsa')
        base = _finite(row.get('ara_model_score'), row.get('ara_hunter_score'))
        final_score = _clip_score(base + bonus)
        out.at[idx, 'external_confirmation_bonus'] = round(bonus, 1)
        out.at[idx, 'ara_final_score'] = round(final_score, 1)
        out.at[idx, 'external_evidence'] = ' • '.join(evidence)
        status = _safe_text(row.get('ara_hunter_status'))
        if status == 'ARA_CONTINUATION_READY' and orderbook_fresh and (orderbook_signal == 'STRONG_BUY_QUEUE') and (broksum_signal == 'ACCUMULATION_PROXY'):
            out.at[idx, 'ara_hunter_status'] = 'ARA_CONTINUATION_VERIFIED_ORDERFLOW'
            out.at[idx, 'order_instruction'] = 'PLAN_NEXT_SESSION_LIMIT_NO_MARKET_CHASE'
        elif status == 'ARA_CONTINUATION_CANDIDATE' and final_score >= 74 and orderbook_fresh and (orderbook_signal in {'STRONG_BUY_QUEUE', 'BUY_QUEUE'}):
            out.at[idx, 'ara_hunter_status'] = 'ARA_CONTINUATION_READY'
        elif status == 'PRE_ARA_CANDIDATE' and final_score >= 76 and orderbook_fresh and (orderbook_signal in {'STRONG_BUY_QUEUE', 'BUY_QUEUE'}):
            out.at[idx, 'ara_hunter_status'] = 'PRE_ARA_READY'
            out.at[idx, 'order_instruction'] = 'BUY_LIMIT_USER_SIZE_AFTER_TRIGGER'
    return out

def scan_ara_hunter_candidates(prepared: Mapping[str, pd.DataFrame], intraday: Mapping[str, pd.DataFrame] | None=None, core_signals: pd.DataFrame | None=None, config: ScanConfig | None=None, now: Any | None=None, market_context: MarketContext | None=None) -> pd.DataFrame:
    """Two-model ARA engine: PRE-ARA and next-session ARA continuation.

    The model is signal-first.  Account risk never suppresses a valid setup.
    Actual broker summary and market depth are optional external confirmation;
    automatic scores use only auditable OHLCV-derived proxies.
    """
    cfg = config or ScanConfig()
    intraday = intraday or {}
    critical_map = _specialty_hard_context_map(core_signals)
    now_jkt = _jakarta_timestamp(now)
    market_regime = market_context.regime if market_context is not None else 'NOT_EVALUATED'
    live_window = idx_regular_decision_window(now_jkt)
    rows: list[dict[str, Any]] = []
    for ticker, frame in prepared.items():
        if frame is None or frame.empty or len(frame) < 60:
            continue
        row = frame.iloc[-1]
        prev = frame.iloc[-2]
        close = _finite(row.get('Close'), 0.0)
        prev_close = _finite(prev.get('Close'), 0.0)
        if close <= 0 or prev_close <= 0:
            continue
        ara_price = idx_daily_price_band(prev_close)[1]
        if not ara_price:
            continue
        daily_return = close / prev_close - 1.0
        room = ara_price / close - 1.0
        vol_ratio = _finite(row.get('VOL_RATIO'), 0.0)
        value_today = _finite(row.get('VALUE'), 0.0)
        adtv = _finite(row.get('ADTV20'), 0.0)
        close_location = _finite(row.get('CLOSE_LOCATION'), 0.0)
        body_atr = _finite(row.get('BODY_ATR'), 0.0)
        cmf_v = _finite(row.get('CMF20'), -1.0)
        obv_up = _finite(row.get('OBV_SLOPE10'), -1.0) > 0
        rsi_v = _finite(row.get('RSI14'), 50.0)
        mfi_v = _finite(row.get('MFI14'), 50.0)
        breakout20 = close > _finite(row.get('HIGH20_PREV'), float('inf'))
        opening_gap = _finite(row.get('Open'), close) / prev_close - 1.0
        accumulation, up_down = _silent_accumulation_metrics(frame)
        context_blocker = critical_map.get(ticker, '')
        im = _intraday_metrics(intraday.get(ticker, pd.DataFrame()), now_jkt, cfg.max_intraday_stale_minutes)
        micro = _ara_intraday_microstructure(intraday.get(ticker, pd.DataFrame()), ara_price, now_jkt)
        hist = _ara_history_profile(frame)
        intraday_usable = bool(im.get('intraday_bars', 0) >= 6 and (_truthy(im.get('intraday_fresh')) or _truthy(im.get('intraday_session_complete'))))
        near_or_locked = near_upper_auto_rejection(prev_close, close, _finite(row.get('High'), close))
        ara_confirmed_today = bool(near_or_locked and daily_return >= 0.88 * idx_ara_pct(prev_close))
        daily_score = 0.0
        daily_score += 20.0 if vol_ratio >= 3.0 else 16.0 if vol_ratio >= 2.0 else 10.0 if vol_ratio >= 1.4 else 0.0
        daily_score += 12.0 if value_today >= 20000000000 else 9.0 if value_today >= 8000000000 else 5.0 if value_today >= 1000000000 else 0.0
        daily_score += 14.0 if close_location >= 0.9 else 10.0 if close_location >= 0.78 else 5.0 if close_location >= 0.65 else 0.0
        daily_score += 11.0 if body_atr >= 1.0 else 8.0 if body_atr >= 0.65 else 4.0 if body_atr >= 0.45 else 0.0
        daily_score += 12.0 if cmf_v >= 0.1 and obv_up else 8.0 if cmf_v > 0 or obv_up else 0.0
        daily_score += 10.0 if breakout20 else 5.0 if daily_return >= 0.04 else 0.0
        daily_score += 9.0 if 0.03 <= daily_return <= 0.2 else 5.0 if 0.015 <= daily_return <= 0.25 else 0.0
        daily_score += 5.0 if 52 <= rsi_v <= 82 and 50 <= mfi_v <= 88 else 0.0
        daily_score += 7.0 if accumulation >= 70 else 4.0 if accumulation >= 55 else 0.0
        if opening_gap > 0.12:
            daily_score -= 8.0
        daily_score = _clip_score(daily_score)
        flow_proxy = _finite(micro.get('orderflow_proxy_score'), np.nan)
        if not np.isfinite(flow_proxy):
            flow_proxy = _clip_score(35.0 + 20.0 * max(-1.0, min(1.0, cmf_v)) + 15.0 * (1.0 if obv_up else 0.0) + 15.0 * max(0.0, min(1.0, close_location)) + 15.0 * max(0.0, min(1.0, _finite(up_down, 1.0) / 2.0)))
        queue_proxy = _finite(micro.get('queue_proxy_score'), np.nan)
        if not np.isfinite(queue_proxy):
            queue_proxy = _clip_score(25.0 + 45.0 * close_location + 15.0 * min(1.0, max(0.0, vol_ratio - 1.0) / 3.0))
        pre_hist = _finite(hist.get('historical_pre_ara_hit_rate'), np.nan)
        pre_hist_score = 50.0 if not np.isfinite(pre_hist) else min(100.0, 30.0 + 140.0 * pre_hist)
        pre_score = _clip_score(0.55 * daily_score + 0.3 * flow_proxy + 0.1 * queue_proxy + 0.05 * pre_hist_score)
        prior_5d_return = np.nan
        if len(frame) >= 6:
            base_close = _finite(frame.iloc[-6].get('Close'), 0.0)
            if base_close > 0:
                prior_5d_return = close / base_close - 1.0
        continuation_daily = _clip_score(20.0 * min(1.0, close_location) + 18.0 * min(1.0, max(0.0, vol_ratio) / 4.0) + 16.0 * (1.0 if breakout20 else 0.4) + 16.0 * min(1.0, accumulation / 80.0) + 15.0 * (1.0 if body_atr >= 0.65 else 0.55) + 15.0 * (1.0 if opening_gap <= 0.1 else 0.25))
        hist_strong = _finite(hist.get('historical_next_day_strong_rate'), np.nan)
        hist_positive = _finite(hist.get('historical_next_day_positive_rate'), np.nan)
        hist_score = 50.0
        if np.isfinite(hist_strong) or np.isfinite(hist_positive):
            hist_score = _clip_score(20.0 + 50.0 * _finite(hist_strong, 0.0) + 30.0 * _finite(hist_positive, 0.0))
        continuation_score = _clip_score(0.3 * continuation_daily + 0.3 * flow_proxy + 0.3 * queue_proxy + 0.1 * hist_score)
        if np.isfinite(prior_5d_return) and prior_5d_return >= 0.7:
            continuation_score = max(0.0, continuation_score - 10.0)
        if opening_gap > 0.15:
            continuation_score = max(0.0, continuation_score - 8.0)
        daily_valid = bool(value_today >= 750000000.0 and vol_ratio >= 1.25 and (close_location >= 0.65) and (body_atr >= 0.4) and (cmf_v > -0.03 or obv_up) and (breakout20 or daily_return >= 0.025) and (0.008 <= daily_return <= 0.28) and (opening_gap <= 0.15) and (not context_blocker))
        warnings: list[str] = []
        if market_regime in {'RISK_OFF', 'UNKNOWN'}:
            warnings.append(f'Regime {market_regime}; disclosure only')
        if not intraday_usable:
            warnings.append('Tanpa intraday usable; orderflow/queue memakai proxy daily berkonfidensi lebih rendah')
        warnings.append('Orderflow proxy bukan broker summary; queue proxy bukan antrean bid aktual')
        if ara_confirmed_today and (not context_blocker):
            ara_model = 'ARA_CONTINUATION'
            prediction_horizon = 'NEXT_SESSION'
            model_score = continuation_score
            if continuation_score >= 74.0 and (queue_proxy >= 58.0 or flow_proxy >= 72.0):
                status = 'ARA_CONTINUATION_READY'
                instruction = 'PLAN_NEXT_SESSION_LIMIT_NO_MARKET_CHASE'
                signal_valid = True
            elif continuation_score >= 62.0:
                status = 'ARA_CONTINUATION_CANDIDATE'
                instruction = 'VERIFY_OPENING_QUEUE_AND_BROKER_FLOW'
                signal_valid = True
            else:
                status = 'ARA_CONFIRMED_ONLY'
                instruction = 'NO_AUTOMATIC_CONTINUATION_ENTRY'
                signal_valid = False
            warnings.append('Sudah ARA/dekat ARA; tidak dapat diperlakukan sebagai entry pre-ARA')
        elif daily_valid and room >= 0.012:
            ara_model = 'PRE_ARA'
            prediction_horizon = 'SESSION_NOW' if live_window and _truthy(im.get('intraday_fresh')) else 'NEXT_SESSION'
            model_score = pre_score
            if pre_score >= 76.0 and flow_proxy >= 62.0:
                status = 'PRE_ARA_READY'
                instruction = 'BUY_LIMIT_USER_SIZE_AFTER_TRIGGER'
                signal_valid = True
            elif pre_score >= 64.0:
                status = 'PRE_ARA_CANDIDATE'
                instruction = 'WAIT_ORB_VWAP_VOLUME_TRIGGER'
                signal_valid = True
            elif pre_score >= 54.0:
                status = 'PRE_ARA_WATCHLIST'
                instruction = 'WATCH_ONLY'
                signal_valid = False
            else:
                continue
        else:
            continue
        atr_v = _finite(row.get('ATR14'), close * 0.03)
        entry = round_idx_price(close, 'up')
        stop = round_idx_price(entry - max(1.0 * atr_v, 3 * idx_tick_size(entry)), 'down')
        if stop >= entry:
            stop = round_idx_price(entry - 3 * idx_tick_size(entry), 'down')
        risk = max(entry - stop, idx_tick_size(entry))
        target_ceiling = ara_price if ara_model == 'PRE_ARA' else idx_daily_price_band(close)[1]
        recent_range = max(
            _finite(row.get('High'), close) - _finite(row.get('Low'), close),
            _finite(frame['High'].tail(min(10, len(frame))).max(), close) - _finite(frame['Low'].tail(min(10, len(frame))).min(), close),
            idx_tick_size(entry),
        )
        targets = _price_structure_target_pair(
            frame, entry, setup=ara_model,
            explicit_levels=[
                (_finite(row.get('HIGH20_PREV'), np.nan), 'PRIOR_20D_HIGH'),
                (_finite(row.get('HIGH55_PREV'), np.nan), 'PRIOR_55D_HIGH'),
                (_finite(row.get('HIGH252'), np.nan), 'PRIOR_52W_HIGH'),
                (target_ceiling, 'CURRENT_SESSION_ARA_LIMIT' if ara_model == 'PRE_ARA' else 'NEXT_SESSION_ARA_LIMIT'),
            ],
            projection_origin=max(entry, _finite(row.get('HIGH20_PREV'), entry)),
            projection_height=recent_range,
            price_ceiling=target_ceiling,
        )
        target_valid = bool(targets['target_structure_valid'])
        ara_tp1, ara_tp2 = targets['tp1'], targets['tp2']
        rr1 = round((ara_tp1 - entry) / risk, 2) if target_valid else np.nan
        rr2 = round((ara_tp2 - entry) / risk, 2) if target_valid else np.nan
        risk_pct = (entry - stop) / entry if entry > 0 else np.nan
        risk_grade = 'VERY_HIGH' if ara_model == 'ARA_CONTINUATION' or risk_pct >= 0.08 else 'HIGH'
        if not target_valid:
            warnings.append('Dua target struktur belum tersedia; ARA objective tetap batas harga, bukan RR sintetis')
        if ara_model == 'ARA_CONTINUATION':
            warnings.append('Target dan RR continuation wajib dihitung ulang terhadap harga pembukaan sesi berikutnya')
        if room < 0.02 and ara_model == 'PRE_ARA':
            warnings.append('Ruang ke ARA sempit; hindari market chase')
        payload = {'ticker': ticker, 'ara_hunter_status': status, 'ara_model': ara_model, 'prediction_horizon': prediction_horizon, 'signal_valid': signal_valid, 'ara_model_score': round(model_score, 1), 'ara_hunter_score': round(model_score, 1), 'daily_momentum_score': round(daily_score, 1), 'orderflow_proxy_score': round(flow_proxy, 1), 'queue_proxy_score': round(queue_proxy, 1), 'proxy_confidence': 'HIGH' if intraday_usable else 'MEDIUM_LOW', 'last_price': close, 'previous_close': prev_close, 'ara_price': ara_price, 'room_to_ara_pct': room, 'daily_return_pct': daily_return, 'opening_gap_pct': opening_gap, 'entry_reference': entry, 'hard_stop': stop, 'ara_tp1': ara_tp1, 'ara_tp2': ara_tp2, 'tp1_basis': targets['tp1_basis'], 'tp2_basis': targets['tp2_basis'], 'target_model': 'PRICE_STRUCTURE_ONLY', 'target_structure': targets['target_structure'], 'target_structure_valid': target_valid, 'target_recalc_required': ara_model == 'ARA_CONTINUATION', 'rr1': rr1, 'rr2': rr2, 'risk_pct': risk_pct, 'volume_ratio': vol_ratio, 'value_today_idr': value_today, 'adtv20_idr': adtv, 'close_location': close_location, 'body_atr': body_atr, 'cmf20': cmf_v, 'rsi14': rsi_v, 'mfi14': mfi_v, 'breakout20': breakout20, 'silent_accumulation_score': accumulation, 'up_down_value_ratio20': up_down, 'intraday_session_complete': _truthy(im.get('intraday_session_complete')), 'intraday_data_state': im.get('intraday_data_state'), 'intraday_session_date': im.get('intraday_session_date'), 'intraday_age_minutes': im.get('intraday_age_minutes'), 'market_regime': market_regime, 'late_volume_acceleration': im.get('late_volume_acceleration'), 'session_close_location': im.get('session_close_location'), 'order_instruction': instruction, 'account_risk_gate_applied': False, 'risk_class': risk_grade, 'warnings': ' • '.join(warnings), 'blockers': context_blocker, 'observed_broker_summary': False, 'observed_orderbook': False}
        payload.update(micro)
        payload['orderflow_proxy_score'] = round(flow_proxy, 1)
        payload['queue_proxy_score'] = round(queue_proxy, 1)
        payload.update(hist)
        rows.append(payload)
    result = pd.DataFrame(rows)
    if not result.empty:
        rank = {'ARA_CONTINUATION_VERIFIED_ORDERFLOW': 0, 'PRE_ARA_READY': 1, 'ARA_CONTINUATION_READY': 2, 'PRE_ARA_CANDIDATE': 3, 'ARA_CONTINUATION_CANDIDATE': 4, 'ARA_CONFIRMED_ONLY': 5, 'PRE_ARA_WATCHLIST': 6}
        result['_rank'] = result['ara_hunter_status'].map(rank).fillna(9)
        result = result.sort_values(['_rank', 'ara_model_score', 'value_today_idr'], ascending=[True, False, False]).drop(columns='_rank').reset_index(drop=True)
    return result

def _critical_context_map(signals: pd.DataFrame | None) -> dict[str, str]:
    """Return only non-negotiable trading/data blockers for specialty screens.

    v4.6.0 incorrectly propagated missing independent-price verification and
    other optional evidence from the core pipeline into BPJS/BSJP. Most
    specialty tickers were not part of the bounded automatic quote shortlist,
    so their scores were capped at 55 and READY became structurally unreachable.
    """
    result: dict[str, str] = {}
    if signals is None or signals.empty or 'ticker' not in signals:
        return result
    critical_tokens = ('SUSPENS', 'FCA', 'SPECIAL MONITOR', 'PEMANTAUAN KHUSUS', 'KONFLIK QUOTE', 'KONFLIK HARGA', 'OHLCV TIDAK TERSEDIA', 'CORPORATE ACTION BELUM DISESUAIKAN')
    for ticker, group in signals.groupby('ticker'):
        messages: list[str] = []
        for _, row in group.iterrows():
            source_tier = _safe_text(row.get('ohlcv_source_tier')).upper()
            if source_tier == 'UNAVAILABLE':
                messages.append('OHLCV daily tidak tersedia')
            if _truthy(row.get('market_status_critical_blocker', False)):
                messages.append('Suspensi/FCA/status perdagangan negatif')
            if _truthy(row.get('quote_critical_blocker', False)):
                messages.append('Konflik quote/OHLCV')
            for item in _pipe_parts(row.get('critical_blockers')):
                upper = item.upper()
                if any((token in upper for token in critical_tokens)):
                    messages.append(item)
        result[str(ticker)] = ' • '.join(dict.fromkeys(messages))
    return result

def _specialty_risk_warnings(*, market_regime: str, adtv: float, cfg: ScanConfig, stop_pct: float, conditions: Mapping[str, bool], room_to_ara: float=np.nan) -> list[str]:
    warnings: list[str] = []
    if market_regime in {'RISK_OFF', 'UNKNOWN'}:
        warnings.append(f'Regime IHSG {market_regime}')
    if adtv < cfg.min_adtv_idr:
        warnings.append(f'ADTV rendah Rp{adtv / 1000000000.0:.2f} miliar')
    if np.isfinite(stop_pct) and stop_pct > 0.035:
        warnings.append(f'Stop teknikal lebar {stop_pct:.1%}')
    if np.isfinite(room_to_ara) and room_to_ara < 0.04:
        warnings.append(f'Ruang ke ARA tinggal {room_to_ara:.1%}')
    for label, passed in conditions.items():
        if not passed:
            warnings.append(label)
    return list(dict.fromkeys(warnings))

def scan_bsjp_candidates(prepared: Mapping[str, pd.DataFrame], intraday: Mapping[str, pd.DataFrame] | None=None, core_signals: pd.DataFrame | None=None, config: ScanConfig | None=None, now: Any | None=None, market_context: MarketContext | None=None) -> pd.DataFrame:
    """Signal-first Beli Sore Jual Pagi scanner.

    READY is based on a valid late-session momentum structure. Regime, stop
    width, liquidity preference, account sizing, and missing optional evidence
    are disclosures rather than automatic rejection.
    """
    cfg = config or ScanConfig()
    intraday = intraday or {}
    critical_map = _critical_context_map(core_signals)
    now_jkt = _jakarta_timestamp(now)
    market_regime = market_context.regime if market_context is not None else 'NOT_EVALUATED'
    minute = now_jkt.hour * 60 + now_jkt.minute
    in_window = now_jkt.weekday() < 5 and 14 * 60 + 30 <= minute <= 15 * 60 + 49
    rows: list[dict[str, Any]] = []
    for ticker, frame in prepared.items():
        if frame is None or frame.empty or len(frame) < 60:
            continue
        row, prev = (frame.iloc[-1], frame.iloc[-2])
        close, prev_close = (_finite(row.get('Close'), 0.0), _finite(prev.get('Close'), 0.0))
        if close <= 0 or prev_close <= 0:
            continue
        daily_return = close / prev_close - 1.0
        adtv = _finite(row.get('ADTV20'), 0.0)
        vol_ratio = _finite(row.get('VOL_RATIO'), 0.0)
        cmf_v = _finite(row.get('CMF20'), -1.0)
        obv_up = _finite(row.get('OBV_SLOPE10'), -1.0) > 0
        close_location = _finite(row.get('CLOSE_LOCATION'), 0.0)
        rs_v = _finite(row.get('REL_STRENGTH60'), -1.0)
        rsi_v = _finite(row.get('RSI14'), 50.0)
        atr_v = _finite(row.get('ATR14'), 0.0)
        trend = close >= _finite(row.get('EMA20'), close) and _finite(row.get('EMA20'), 0.0) >= _finite(row.get('EMA50'), 0.0)
        # During the live session the last completed daily close is today's ARA reference.
        ara_price = idx_daily_price_band(close)[1]
        im = _intraday_metrics(intraday.get(ticker, pd.DataFrame()), now_jkt, cfg.max_intraday_stale_minutes)
        has_any_intraday = im['intraday_bars'] > 0
        has_intraday = im['intraday_bars'] >= 6 and _truthy(im['intraday_fresh'])
        intraday_last = _finite(im.get('intraday_last'), close)
        room_to_ara = ara_price / intraday_last - 1.0 if ara_price and intraday_last > 0 else np.nan
        above_vwap = has_intraday and intraday_last >= _finite(im.get('session_vwap'), float('inf'))
        late_volume = has_intraday and _finite(im.get('late_volume_acceleration'), 0.0) >= 1.05
        location_ok = has_intraday and _finite(im.get('session_close_location'), 0.0) >= 0.62
        positive_session = has_intraday and _finite(im.get('intraday_return'), -1.0) >= -0.003
        daily_components = {'Trend harian belum mendukung': trend, 'Volume harian belum menguat': vol_ratio >= 0.9, 'CMF masih lemah': cmf_v >= -0.03, 'OBV belum naik': obv_up, 'Close harian tidak dekat high': close_location >= 0.55, 'Relative strength lemah': rs_v >= -0.03, 'RSI di luar zona momentum': 42 <= rsi_v <= 82, 'Return harian terlalu ekstrem': -0.03 <= daily_return <= 0.12}
        session_components = {'Harga belum di atas VWAP sesi': above_vwap, 'Late volume belum akseleratif': late_volume, 'Close sesi belum dekat high': location_ok, 'Momentum sesi masih negatif': positive_session}
        daily_hits = sum((bool(v) for v in daily_components.values()))
        session_hits = sum((bool(v) for v in session_components.values()))
        score = 0.0
        score += 12.0 if trend else 5.0
        score += 12.0 if vol_ratio >= 1.5 else 8.0 if vol_ratio >= 1.1 else 4.0 if vol_ratio >= 0.9 else 0.0
        score += 12.0 if cmf_v >= 0.08 and obv_up else 8.0 if cmf_v >= -0.03 and obv_up else 3.0 if cmf_v >= -0.03 else 0.0
        score += 9.0 if close_location >= 0.75 else 6.0 if close_location >= 0.55 else 0.0
        score += 8.0 if rs_v > 0 else 4.0 if rs_v >= -0.03 else 0.0
        score += 7.0 if 0.003 <= daily_return <= 0.07 else 4.0 if -0.03 <= daily_return <= 0.12 else 0.0
        score += 15.0 if above_vwap else 0.0
        score += 12.0 if late_volume else 5.0 if has_intraday else 0.0
        score += 10.0 if location_ok else 4.0 if has_intraday else 0.0
        score += 3.0 if positive_session else 0.0
        score = min(100.0, score)
        context_blocker = critical_map.get(ticker, '')
        if score < 50 and daily_hits < 4:
            continue
        entry = round_idx_price(intraday_last if has_intraday else close, 'nearest')
        session = _intraday_session(intraday.get(ticker, pd.DataFrame()), asof=now_jkt)
        if has_intraday and (not session.empty):
            recent_low = float(session['Low'].tail(min(5, len(session))).min())
            raw_stop = recent_low - idx_tick_size(recent_low)
        else:
            raw_stop = entry - max(0.035 * entry, 1.0 * atr_v, 3 * idx_tick_size(entry))
        stop = round_idx_price(raw_stop, 'down')
        if stop is None or stop >= entry:
            stop = round_idx_price(entry - max(3 * idx_tick_size(entry), 0.025 * entry), 'down')
        risk = max(entry - stop, idx_tick_size(entry))
        session_high = _finite(session['High'].max(), entry) if not session.empty else _finite(row.get('High'), entry)
        session_low = _finite(session['Low'].min(), stop) if not session.empty else _finite(row.get('Low'), stop)
        next_session_ara = idx_daily_price_band(max(intraday_last, close))[1]
        targets = _price_structure_target_pair(
            frame, entry, setup='BSJP',
            explicit_levels=[
                (session_high, 'CURRENT_SESSION_HIGH'),
                (_finite(row.get('High'), np.nan), 'LAST_COMPLETED_DAILY_HIGH'),
                (_finite(row.get('HIGH20_PREV'), np.nan), 'PRIOR_20D_HIGH'),
                (next_session_ara, 'NEXT_SESSION_ARA_LIMIT'),
            ],
            projection_origin=session_high,
            projection_height=max(session_high - session_low, idx_tick_size(entry)),
            price_ceiling=next_session_ara,
        )
        tp1, tp2 = targets['tp1'], targets['tp2']
        target_valid = bool(targets['target_structure_valid'])
        rr1 = round((tp1 - entry) / risk, 2) if target_valid else np.nan
        rr2 = round((tp2 - entry) / risk, 2) if target_valid else np.nan
        stop_pct = (entry - stop) / entry if entry > 0 else np.nan
        setup_valid = bool(in_window and has_intraday and above_vwap and (session_hits >= 2) and (daily_hits >= 4) and (score >= 68.0) and target_valid and (not context_blocker))
        if setup_valid:
            status, action = ('BSJP_READY', 'BUY_SORE_SELL_NEXT_MORNING')
        elif not has_any_intraday:
            status, action = ('BSJP_DATA_UNAVAILABLE', 'RETRY_INTRADAY_5M')
        elif not _truthy(im['intraday_fresh']):
            status, action = ('BSJP_STALE_INTRADAY', 'REFRESH_INTRADAY_DATA')
        elif not has_intraday:
            status, action = ('BSJP_WAIT_SESSION_BARS', 'RUN_AGAIN_AFTER_MORE_5M_BARS')
        elif not in_window:
            status, action = ('BSJP_WATCHLIST', 'RUN_AGAIN_14_30_15_49_WIB')
        else:
            status, action = ('BSJP_WATCHLIST', 'WAIT_LATE_SESSION_CONFIRMATION')
        warnings = _specialty_risk_warnings(market_regime=market_regime, adtv=adtv, cfg=cfg, stop_pct=stop_pct, room_to_ara=room_to_ara, conditions={**daily_components, **session_components})
        if not target_valid:
            warnings.append('Dua target struktur harga belum tersedia; RR tidak difabrikasi')
        rows.append({'ticker': ticker, 'bsjp_status': status, 'bsjp_score': round(score, 1), 'setup_valid': setup_valid, 'action': action, 'last_price': close, 'intraday_last': intraday_last, 'entry': entry, 'stop_loss': stop, 'morning_tp1': tp1, 'morning_tp2': tp2, 'tp1_basis': targets['tp1_basis'], 'tp2_basis': targets['tp2_basis'], 'target_model': 'PRICE_STRUCTURE_ONLY', 'target_structure': targets['target_structure'], 'target_structure_valid': target_valid, 'rr1': rr1, 'rr2': rr2, 'risk_pct': stop_pct, 'daily_return_pct': daily_return, 'volume_ratio': vol_ratio, 'adtv20_idr': adtv, 'cmf20': cmf_v, 'relative_strength60': rs_v, 'session_vwap': im.get('session_vwap'), 'session_close_location': im.get('session_close_location'), 'late_volume_acceleration': im.get('late_volume_acceleration'), 'room_to_ara_pct': room_to_ara, 'daily_confirmation_count': daily_hits, 'session_confirmation_count': session_hits, 'intraday_bars': int(im['intraday_bars']), 'intraday_interval_minutes': im['intraday_interval_minutes'], 'intraday_data_state': im['intraday_data_state'], 'intraday_session_date': im['intraday_session_date'], 'intraday_age_minutes': im['intraday_age_minutes'], 'intraday_fresh': im['intraday_fresh'], 'market_regime': market_regime, 'execution_window': '14:30–15:49 WIB', 'exit_window': '09:00–10:00 WIB next session', 'order_instruction': 'BUY_LIMIT_USER_SIZE' if setup_valid else 'WATCH_ONLY', 'account_risk_gate_applied': False, 'risk_class': 'HIGH_OVERNIGHT_GAP_RISK', 'warnings': ' • '.join(warnings), 'blockers': context_blocker, **_specialty_sizing(entry, stop, cfg, position_cap=1.0, risk_cap=1.0)})
    result = pd.DataFrame(rows)
    if not result.empty:
        rank = {'BSJP_READY': 0, 'BSJP_WATCHLIST': 1, 'BSJP_WAIT_SESSION_BARS': 2, 'BSJP_STALE_INTRADAY': 3, 'BSJP_DATA_UNAVAILABLE': 4}
        result['_rank'] = result['bsjp_status'].map(rank).fillna(9)
        result = result.sort_values(['_rank', 'bsjp_score', 'adtv20_idr'], ascending=[True, False, False]).drop(columns='_rank').reset_index(drop=True)
    return result

def scan_bpjs_candidates(prepared: Mapping[str, pd.DataFrame], intraday: Mapping[str, pd.DataFrame] | None=None, core_signals: pd.DataFrame | None=None, config: ScanConfig | None=None, now: Any | None=None, market_context: MarketContext | None=None) -> pd.DataFrame:
    """Signal-first Beli Pagi Jual Sore scanner with ORB/VWAP as hard setup logic."""
    cfg = config or ScanConfig()
    intraday = intraday or {}
    critical_map = _critical_context_map(core_signals)
    now_jkt = _jakarta_timestamp(now)
    market_regime = market_context.regime if market_context is not None else 'NOT_EVALUATED'
    minute = now_jkt.hour * 60 + now_jkt.minute
    in_window = now_jkt.weekday() < 5 and 9 * 60 + 20 <= minute <= 10 * 60 + 45
    rows: list[dict[str, Any]] = []
    for ticker, frame in prepared.items():
        if frame is None or frame.empty or len(frame) < 60:
            continue
        row, prev = (frame.iloc[-1], frame.iloc[-2])
        close, prev_close = (_finite(row.get('Close'), 0.0), _finite(prev.get('Close'), 0.0))
        if close <= 0 or prev_close <= 0:
            continue
        adtv = _finite(row.get('ADTV20'), 0.0)
        rs_v = _finite(row.get('REL_STRENGTH60'), -1.0)
        cmf_v = _finite(row.get('CMF20'), -1.0)
        trend = _finite(row.get('EMA20'), 0.0) >= _finite(row.get('EMA50'), float('inf')) and close >= _finite(row.get('EMA50'), float('inf'))
        im = _intraday_metrics(intraday.get(ticker, pd.DataFrame()), now_jkt, cfg.max_intraday_stale_minutes)
        has_any_intraday = im['intraday_bars'] > 0
        has_intraday = im['intraday_data_state'] == 'LIVE_READY' and im['post_orb_bars'] >= 1
        session = _intraday_session(intraday.get(ticker, pd.DataFrame()), asof=now_jkt)
        session_open = float(session['Open'].iloc[0]) if has_intraday and (not session.empty) else np.nan
        intraday_last = _finite(im.get('intraday_last'), close)
        gap = session_open / close - 1 if np.isfinite(session_open) and close > 0 else np.nan
        above_vwap = has_intraday and intraday_last >= _finite(im.get('session_vwap'), float('inf'))
        orb_break = has_intraday and intraday_last >= _finite(im.get('orb_high'), float('inf'))
        opening_volume = has_intraday and _finite(im.get('opening_volume_ratio'), 0.0) >= 1.05
        location_ok = has_intraday and _finite(im.get('session_close_location'), 0.0) >= 0.6
        gap_ok = has_intraday and np.isfinite(gap) and (-0.03 <= gap <= 0.07)
        conditions = {'Trend daily belum mendukung': trend, 'Relative strength lemah': rs_v >= -0.03, 'CMF masih negatif': cmf_v >= -0.05, 'Harga belum di atas VWAP': above_vwap, 'ORB belum ditembus': orb_break, 'Opening volume belum menguat': opening_volume, 'Close sesi belum dekat high': location_ok, 'Gap di luar rentang ideal': gap_ok}
        score = 0.0
        score += 12.0 if trend else 4.0
        score += 8.0 if rs_v > 0 else 4.0 if rs_v >= -0.03 else 0.0
        score += 8.0 if cmf_v >= 0.05 else 4.0 if cmf_v >= -0.05 else 0.0
        score += 20.0 if above_vwap else 0.0
        score += 22.0 if orb_break else 0.0
        score += 10.0 if opening_volume else 4.0 if has_intraday else 0.0
        score += 10.0 if location_ok else 4.0 if has_intraday else 0.0
        score += 5.0 if gap_ok else 0.0
        score += 5.0 if adtv >= cfg.min_adtv_idr else 2.0 if adtv >= 500000000 else 0.0
        score = min(100.0, score)
        if score < 45 and (not (trend or orb_break)):
            continue
        entry = round_idx_price(intraday_last if orb_break else _finite(im.get('orb_high'), intraday_last), 'up')
        if has_intraday and (not session.empty):
            recent_low = float(session['Low'].tail(min(4, len(session))).min())
            raw_stop = min(_finite(im.get('orb_low'), recent_low), recent_low) - idx_tick_size(recent_low)
        else:
            raw_stop = entry * 0.965
        stop = round_idx_price(raw_stop, 'down')
        if stop is None or stop >= entry:
            stop = round_idx_price(entry - max(3 * idx_tick_size(entry), 0.025 * entry), 'down')
        risk = max(entry - stop, idx_tick_size(entry))
        ara_price = idx_daily_price_band(close)[1]
        room_to_ara = ara_price / entry - 1.0 if ara_price and entry > 0 else np.nan
        session_high = _finite(session['High'].max(), entry) if not session.empty else entry
        session_low = _finite(session['Low'].min(), stop) if not session.empty else stop
        orb_high = _finite(im.get('orb_high'), entry)
        orb_low = _finite(im.get('orb_low'), session_low)
        targets = _price_structure_target_pair(
            frame, entry, setup='BPJS',
            explicit_levels=[
                (session_high, 'CURRENT_SESSION_HIGH'),
                (_finite(row.get('High'), np.nan), 'PRIOR_SESSION_HIGH'),
                (_finite(row.get('HIGH20_PREV'), np.nan), 'PRIOR_20D_HIGH'),
                (ara_price, 'CURRENT_SESSION_ARA_LIMIT'),
            ],
            projection_origin=orb_high,
            projection_height=max(orb_high - orb_low, session_high - session_low, idx_tick_size(entry)),
            price_ceiling=ara_price,
        )
        tp1, tp2 = targets['tp1'], targets['tp2']
        target_valid = bool(targets['target_structure_valid'])
        rr1 = round((tp1 - entry) / risk, 2) if target_valid else np.nan
        rr2 = round((tp2 - entry) / risk, 2) if target_valid else np.nan
        stop_pct = (entry - stop) / entry if entry > 0 else np.nan
        context_blocker = critical_map.get(ticker, '')
        setup_valid = bool(in_window and has_intraday and above_vwap and orb_break and (score >= 68.0) and target_valid and (not context_blocker))
        if setup_valid:
            status, action = ('BPJS_READY', 'BUY_OPENING_BREAKOUT_SELL_BEFORE_CLOSE')
        elif not has_any_intraday:
            status, action = ('BPJS_DATA_UNAVAILABLE', 'RETRY_INTRADAY_5M')
        elif not _truthy(im['intraday_fresh']):
            status, action = ('BPJS_STALE_INTRADAY', 'REFRESH_INTRADAY_DATA')
        elif not has_intraday:
            status, action = ('BPJS_WAIT_OPENING_BARS', 'RUN_AGAIN_AFTER_OPENING_RANGE')
        elif not in_window:
            status, action = ('BPJS_WATCHLIST', 'RUN_AGAIN_09_20_10_45_WIB')
        else:
            status, action = ('BPJS_WATCHLIST', 'WAIT_ORB_VWAP_CONFIRMATION')
        warnings = _specialty_risk_warnings(market_regime=market_regime, adtv=adtv, cfg=cfg, stop_pct=stop_pct, room_to_ara=room_to_ara, conditions=conditions)
        if not target_valid:
            warnings.append('Dua target struktur harga belum tersedia; RR tidak difabrikasi')
        rows.append({'ticker': ticker, 'bpjs_status': status, 'bpjs_score': round(score, 1), 'setup_valid': setup_valid, 'action': action, 'last_price': close, 'intraday_last': intraday_last, 'opening_gap_pct': gap, 'entry': entry, 'stop_loss': stop, 'day_tp1': tp1, 'day_tp2': tp2, 'tp1_basis': targets['tp1_basis'], 'tp2_basis': targets['tp2_basis'], 'target_model': 'PRICE_STRUCTURE_ONLY', 'target_structure': targets['target_structure'], 'target_structure_valid': target_valid, 'rr1': rr1, 'rr2': rr2, 'risk_pct': stop_pct, 'session_vwap': im.get('session_vwap'), 'orb_high': im.get('orb_high'), 'orb_low': im.get('orb_low'), 'opening_volume_ratio': im.get('opening_volume_ratio'), 'session_close_location': im.get('session_close_location'), 'adtv20_idr': adtv, 'relative_strength60': rs_v, 'room_to_ara_pct': room_to_ara, 'confirmation_count': sum((bool(v) for v in conditions.values())), 'intraday_bars': int(im['intraday_bars']), 'intraday_interval_minutes': im['intraday_interval_minutes'], 'opening_range_bars': int(im['opening_range_bars']), 'post_orb_bars': int(im['post_orb_bars']), 'intraday_data_state': im['intraday_data_state'], 'intraday_session_date': im['intraday_session_date'], 'intraday_age_minutes': im['intraday_age_minutes'], 'intraday_fresh': im['intraday_fresh'], 'market_regime': market_regime, 'execution_window': '09:20–10:45 WIB', 'mandatory_exit': 'Before regular-market close', 'order_instruction': 'BUY_LIMIT_USER_SIZE' if setup_valid else 'WATCH_ONLY', 'account_risk_gate_applied': False, 'risk_class': 'HIGH_INTRADAY_EXECUTION_RISK', 'warnings': ' • '.join(warnings), 'blockers': context_blocker, **_specialty_sizing(entry, stop, cfg, position_cap=1.0, risk_cap=1.0)})
    result = pd.DataFrame(rows)
    if not result.empty:
        rank = {'BPJS_READY': 0, 'BPJS_WATCHLIST': 1, 'BPJS_WAIT_OPENING_BARS': 2, 'BPJS_STALE_INTRADAY': 3, 'BPJS_DATA_UNAVAILABLE': 4}
        result['_rank'] = result['bpjs_status'].map(rank).fillna(9)
        result = result.sort_values(['_rank', 'bpjs_score', 'adtv20_idr'], ascending=[True, False, False]).drop(columns='_rank').reset_index(drop=True)
    return result

def specialty_intraday_shortlist(prepared: Mapping[str, pd.DataFrame], core_signals: pd.DataFrame | None=None, max_candidates: int=120) -> list[str]:
    """Balanced shortlist for BPJS, BSJP, ARA, and core confirmation.

    Earlier releases reserved no explicit capacity for BPJS/BSJP. A ticker could
    have a valid morning/late-session profile but never receive intraday data,
    making READY impossible before the specialty detector even ran.
    """
    rows: list[dict[str, Any]] = []
    core_tickers: set[str] = set()
    if core_signals is not None and (not core_signals.empty) and ('ticker' in core_signals):
        core_tickers = set(core_signals['ticker'].astype(str))
    for ticker, frame in prepared.items():
        if frame is None or frame.empty or len(frame) < 25:
            continue
        row, prev = (frame.iloc[-1], frame.iloc[-2])
        close, prev_close = (_finite(row.get('Close'), 0.0), _finite(prev.get('Close'), 0.0))
        if close <= 0 or prev_close <= 0:
            continue
        adtv = _finite(row.get('ADTV20'), 0.0)
        vol_ratio = _finite(row.get('VOL_RATIO'), 0.0)
        close_location = _finite(row.get('CLOSE_LOCATION'), 0.0)
        cmf_v = _finite(row.get('CMF20'), -1.0)
        obv_up = _finite(row.get('OBV_SLOPE10'), -1.0) > 0
        rs_v = _finite(row.get('REL_STRENGTH60'), -1.0)
        body_atr = _finite(row.get('BODY_ATR'), 0.0)
        daily_return = close / prev_close - 1.0
        ema20, ema50 = (_finite(row.get('EMA20'), 0.0), _finite(row.get('EMA50'), 0.0))
        trend = ema20 >= ema50 and close >= ema50
        breakout = close > _finite(row.get('HIGH20_PREV'), float('inf'))
        liquid = min(35.0, 35.0 * adtv / 5000000000.0) if adtv > 0 else 0.0
        core_score = liquid + min(20.0, 10.0 * max(0.0, vol_ratio - 0.5))
        core_score += 12.0 if trend else 0.0
        core_score += 10.0 if rs_v > 0 else 0.0
        core_score += 8.0 if ticker in core_tickers else 0.0
        bpjs_score = liquid * 0.65
        bpjs_score += 22.0 if trend else 5.0
        bpjs_score += 15.0 if rs_v > 0 else 7.0 if rs_v >= -0.03 else 0.0
        bpjs_score += 12.0 if cmf_v >= 0.05 else 6.0 if cmf_v >= -0.05 else 0.0
        bpjs_score += 8.0 if -0.03 <= daily_return <= 0.06 else 0.0
        bpjs_score += 5.0 if close_location >= 0.55 else 0.0
        bsjp_score = liquid * 0.55
        bsjp_score += 18.0 if vol_ratio >= 1.3 else 10.0 if vol_ratio >= 0.9 else 0.0
        bsjp_score += 15.0 if cmf_v >= 0.03 and obv_up else 8.0 if cmf_v >= -0.03 else 0.0
        bsjp_score += 15.0 if close_location >= 0.7 else 8.0 if close_location >= 0.55 else 0.0
        bsjp_score += 10.0 if rs_v > 0 else 4.0 if rs_v >= -0.03 else 0.0
        bsjp_score += 8.0 if -0.03 <= daily_return <= 0.1 else 0.0
        ara_score = min(30.0, max(0.0, (vol_ratio - 1.0) * 15.0))
        ara_score += 20.0 if close_location >= 0.85 else 12.0 if close_location >= 0.7 else 0.0
        ara_score += 15.0 if body_atr >= 0.8 else 8.0 if body_atr >= 0.5 else 0.0
        ara_score += 15.0 if breakout else 0.0
        ara_score += 15.0 if daily_return >= 0.05 else 9.0 if daily_return >= 0.02 else 0.0
        ara_score += 5.0 if cmf_v > 0 or rs_v > 0 else 0.0
        rows.append({'ticker': ticker, 'core_score': core_score, 'bpjs_score': bpjs_score, 'bsjp_score': bsjp_score, 'ara_score': ara_score, 'adtv': adtv})
    if not rows:
        return []
    ranked = pd.DataFrame(rows)
    total = max(1, min(int(max_candidates), len(ranked)))
    buckets = ['bpjs_score', 'bsjp_score', 'ara_score', 'core_score']
    slots = max(1, total // len(buckets))
    merged: list[str] = []
    for score_col in buckets:
        ordered = ranked.sort_values([score_col, 'adtv'], ascending=[False, False])['ticker'].tolist()
        merged.extend([ticker for ticker in ordered[:slots] if ticker not in merged])
    pointers = {name: 0 for name in buckets}
    ordered_map = {name: ranked.sort_values([name, 'adtv'], ascending=[False, False])['ticker'].tolist() for name in buckets}
    while len(merged) < total:
        added = False
        for name in buckets:
            items = ordered_map[name]
            while pointers[name] < len(items) and items[pointers[name]] in merged:
                pointers[name] += 1
            if pointers[name] < len(items):
                merged.append(items[pointers[name]])
                pointers[name] += 1
                added = True
                if len(merged) >= total:
                    break
        if not added:
            break
    return merged[:total]

def scan_sniper_entries(core_signals: pd.DataFrame | None, prepared: Mapping[str, pd.DataFrame], config: ScanConfig | None=None) -> pd.DataFrame:
    """Build an actionable ICT Sniper table from valid Unicorn structures.

    A Sniper setup is evaluated independently from the final core status. The
    earlier implementation required ``core_status == EXECUTION_READY`` and then
    re-applied a second set of stricter volume/flow/RR/stop gates. That circular
    dependency made a valid FVG retracement almost impossible to classify as
    ready. Current-bar volume is now interpreted by entry phase: contraction is
    acceptable on a limit retracement, while expansion is rewarded on a reclaim
    trigger. Risk metrics are warnings, not account blockers.
    """
    cfg = config or ScanConfig()
    columns = ['ticker', 'sniper_status', 'sniper_entry_mode', 'sniper_score', 'core_status', 'core_action', 'setup', 'last_price', 'entry_low', 'entry_high', 'sniper_entry', 'sniper_trigger', 'sniper_stop', 'sniper_tp1', 'sniper_tp2', 'rr1', 'rr2', 'stop_pct', 'distance_atr', 'volume_ratio', 'volume_context', 'silent_accumulation_score', 'structure_grade', 'trigger_state', 'valid_until', 'risk_warnings', 'primary_sniper_blocker', 'blockers', 'reason']
    if core_signals is None or core_signals.empty:
        return pd.DataFrame(columns=columns)
    setup_series = core_signals.get('setup', pd.Series(index=core_signals.index, dtype=object))
    candidates = core_signals[setup_series.eq('UNICORN_SNIPER_ICT')].copy()
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    candidates['_quality'] = pd.to_numeric(candidates.get('quality_score', 0.0), errors='coerce').fillna(0.0)
    candidates['_status_bonus'] = candidates.get('status', pd.Series(index=candidates.index, dtype=object)).astype(str).eq('EXECUTION_READY').astype(int)
    candidates = candidates.sort_values(['_status_bonus', '_quality'], ascending=False).drop_duplicates('ticker', keep='first').drop(columns=['_quality', '_status_bonus'], errors='ignore')
    rows: list[dict[str, Any]] = []
    for _, signal in candidates.iterrows():
        ticker = str(signal.get('ticker'))
        frame = prepared.get(ticker)
        if frame is None or frame.empty:
            continue
        row = frame.iloc[-1]
        previous = frame.iloc[-2] if len(frame) >= 2 else row
        evidence = _safe_text(signal.get('evidence'))
        blockers_text = _safe_text(signal.get('blockers'))
        action = _safe_text(signal.get('action'))
        core_status = _safe_text(signal.get('status'))
        quality = np.clip(_finite(signal.get('quality_score'), 0.0), 0.0, 100.0)
        distance = _finite(signal.get('distance_atr'), 99.0)
        volume_ratio = _finite(signal.get('volume_ratio'), _finite(row.get('VOL_RATIO'), 0.0))
        accumulation = np.clip(_finite(signal.get('silent_accumulation_score'), 50.0), 0.0, 100.0)
        rr1 = _finite(signal.get('rr1'), np.nan)
        rr2 = _finite(signal.get('rr2'), np.nan)
        stop_pct = _finite(signal.get('stop_pct'), np.nan)
        close = _finite(signal.get('last_price'), _finite(row.get('Close'), np.nan))
        open_v = _finite(row.get('Open'), close)
        prev_high = _finite(previous.get('High'), close)
        close_location = _finite(row.get('CLOSE_LOCATION'), 0.5)
        bull_rejection = _truthy(row.get('BULL_REJECTION', False))
        reclaim = bool(action == 'READY_TRIGGER' or core_status == 'EXECUTION_READY' or bull_rejection or (np.isfinite(close) and close > prev_high and (close > open_v) and (close_location >= 0.6)))
        sweep = 'Sell-side liquidity sweep' in evidence
        bos = 'Bullish BOS dengan displacement' in evidence
        fvg = 'Bullish FVG valid' in evidence
        ob_overlap = 'FVG overlap dengan order-block proxy' in evidence
        discount = 'Zona berada di discount dealing range' in evidence
        inferred_ready_structure = bool(core_status == 'EXECUTION_READY' and quality >= 82.0)
        core_structure = bool(sweep and bos and fvg or inferred_ready_structure)
        strict_confluence = bool(ob_overlap and discount and (quality > 74.0) or (inferred_ready_structure and quality >= 88.0))
        daily_source_tier = _safe_text(signal.get('ohlcv_source_tier')).upper()
        daily_source_ok = daily_source_tier not in {'CACHE_FALLBACK', 'UNAVAILABLE'}
        invalid = _truthy(signal.get('invalidated', False)) or not _truthy(signal.get('detected', True))
        invalid_action = action in {'NO_SETUP', 'TOO_EXTENDED_WAIT_NEW_BASE', 'WAIT_CHOCH'}
        level_values = [signal.get('entry'), signal.get('stop_loss'), signal.get('tp1'), signal.get('tp2')]
        levels_valid = all((is_valid_idx_price(_finite(value, np.nan)) for value in level_values))
        valid_until = signal.get('valid_until')
        expired = False
        if valid_until is not None and (not pd.isna(valid_until)):
            try:
                expired = pd.Timestamp(frame.index[-1]) > pd.Timestamp(valid_until)
            except Exception:
                expired = False
        hard: list[str] = []
        if not core_structure:
            hard.append('Sweep–BOS–FVG belum lengkap')
        if invalid or invalid_action:
            hard.append('Struktur Unicorn/Sniper tidak valid')
        if expired:
            hard.append('Zona FVG sudah kedaluwarsa')
        if not levels_valid:
            hard.append('Level entry/SL/TP tidak valid menurut fraksi IDX')
        if not daily_source_ok:
            hard.append('OHLCV daily bukan hasil live')
        if _truthy(signal.get('pending_close', False)):
            hard.append('Daily candle belum final')
        if _truthy(signal.get('market_status_critical_blocker', False)):
            hard.append('Suspensi/FCA/status perdagangan negatif')
        if _truthy(signal.get('quote_critical_blocker', False)):
            hard.append('Konflik quote/candle')
        in_trigger_zone = bool(np.isfinite(distance) and distance <= 0.35)
        in_limit_range = bool(np.isfinite(distance) and distance <= 1.0)
        if reclaim and in_trigger_zone:
            entry_mode = 'RECLAIM_TRIGGER'
            trigger_state = 'CONFIRMED'
            volume_context = 'RECLAIM_EXPANSION' if volume_ratio >= 1.05 else 'RECLAIM_PRICE_CONFIRMATION'
            volume_score = 12.0 if volume_ratio >= 1.2 else 9.0 if volume_ratio >= 1.05 else 7.0
            trigger_score = 18.0
        elif in_limit_range and action in {'WAIT_FVG_RETRACE', 'WAIT_STRICT_UNICORN_CONFLUENCE', 'READY_LIMIT'}:
            entry_mode = 'LIMIT_FVG_RETRACE'
            trigger_state = 'RESTING_LIMIT_VALID'
            volume_context = 'HEALTHY_RETRACE_CONTRACTION' if 0.45 <= volume_ratio <= 1.2 else 'RETRACE_VOLUME_WARNING'
            volume_score = 12.0 if 0.55 <= volume_ratio <= 1.05 else 8.0 if 0.35 <= volume_ratio <= 1.35 else 3.0
            trigger_score = 12.0
        elif core_structure and np.isfinite(distance) and (distance > 1.0):
            entry_mode = 'WAIT_FVG_RETRACE'
            trigger_state = 'PRICE_TOO_FAR'
            volume_context = 'NOT_APPLICABLE'
            volume_score = 4.0
            trigger_score = 3.0
        else:
            entry_mode = 'WATCH_STRUCTURE'
            trigger_state = 'INCOMPLETE'
            volume_context = 'UNRESOLVED'
            volume_score = 3.0
            trigger_score = 4.0
        structure_score = 30.0 if core_structure else 10.0
        confluence_score = (8.0 if ob_overlap else 2.0) + (7.0 if discount else 2.0) + (5.0 if strict_confluence else 2.0)
        location_score = (20.0 if distance <= 0.15 else 16.0 if distance <= 0.35 else 11.0 if distance <= 0.75 else 7.0 if distance <= 1.0 else 0.0) if np.isfinite(distance) else 0.0
        flow_score = 10.0 if accumulation >= 70 else 7.0 if accumulation >= 60 else 4.0 if accumulation >= 45 else 1.0
        geometry_score = 5.0 if np.isfinite(rr2) and rr2 >= 3.0 else 3.0 if np.isfinite(rr2) and rr2 >= 2.0 else 1.0
        quality_bonus = min(5.0, quality * 0.05)
        score = min(100.0, structure_score + confluence_score + location_score + trigger_score + volume_score + flow_score + geometry_score + quality_bonus)
        risk_warnings: list[str] = []
        if not ob_overlap:
            risk_warnings.append('OB×FVG overlap belum terkonfirmasi')
        if not discount:
            risk_warnings.append('Zona bukan discount ideal')
        if accumulation < 60:
            risk_warnings.append(f'Flow proxy moderat/lemah {accumulation:.0f}/100')
        if np.isfinite(stop_pct) and stop_pct > cfg.max_stop_pct:
            risk_warnings.append(f'SL lebar {stop_pct:.1%}')
        if np.isfinite(rr2) and rr2 < cfg.min_rr2:
            risk_warnings.append(f'RR2 rendah {rr2:.2f}')
        if volume_context in {'RETRACE_VOLUME_WARNING', 'RECLAIM_PRICE_CONFIRMATION'}:
            risk_warnings.append('Konfirmasi volume tidak ideal')
        if core_status != 'EXECUTION_READY':
            risk_warnings.append(f"Core status {core_status or 'UNKNOWN'}; Sniper dinilai independen")
        if blockers_text:
            risk_warnings.append('Core warning tersedia')
        ready = bool(not hard and entry_mode in {'RECLAIM_TRIGGER', 'LIMIT_FVG_RETRACE'} and (quality >= 68.0) and (score >= 70.0))
        if ready:
            status = 'SNIPER_READY'
        elif not hard and core_structure and np.isfinite(distance) and (distance <= 1.5):
            status = 'WAIT_SNIPER_RETRACE'
        elif hard:
            status = 'SNIPER_REJECT'
        else:
            status = 'SNIPER_WATCHLIST'
        structure_grade = 'STRICT' if strict_confluence else 'VALID' if core_structure else 'INCOMPLETE'
        rows.append({'ticker': ticker, 'sniper_status': status, 'sniper_entry_mode': entry_mode, 'sniper_score': round(float(score), 1), 'core_status': core_status, 'core_action': action, 'setup': 'ICT_SNIPER_CALIBRATED', 'last_price': close, 'entry_low': signal.get('entry_low'), 'entry_high': signal.get('entry_high'), 'sniper_entry': signal.get('entry'), 'sniper_trigger': signal.get('trigger'), 'sniper_stop': signal.get('stop_loss'), 'sniper_tp1': signal.get('tp1'), 'sniper_tp2': signal.get('tp2'), 'rr1': rr1, 'rr2': rr2, 'stop_pct': stop_pct, 'distance_atr': distance, 'volume_ratio': volume_ratio, 'volume_context': volume_context, 'silent_accumulation_score': accumulation, 'structure_grade': structure_grade, 'trigger_state': trigger_state, 'valid_until': valid_until, 'risk_warnings': ' • '.join(risk_warnings), 'primary_sniper_blocker': hard[0] if hard else 'NONE' if ready else 'ENTRY_NOT_READY', 'blockers': ' • '.join(hard), 'reason': 'Sweep–BOS–FVG valid; entry dinilai menurut fase retracement/reclaim. Risiko akun, stop, RR, dan flow lemah ditampilkan sebagai warning.'})
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        rank = {'SNIPER_READY': 0, 'WAIT_SNIPER_RETRACE': 1, 'SNIPER_WATCHLIST': 2, 'SNIPER_REJECT': 3}
        result['_rank'] = result['sniper_status'].map(rank).fillna(9)
        result = result.sort_values(['_rank', 'sniper_score', 'rr2'], ascending=[True, False, False]).drop(columns='_rank').reset_index(drop=True)
    return result
__all__ = ['download_intraday_ohlcv', 'specialty_intraday_shortlist', 'scan_sniper_entries', 'scan_bsjp_candidates', 'scan_bpjs_candidates', 'scan_multibagger_candidates', 'scan_ara_hunter_candidates', 'build_specialty_screens', 'parse_orderbook_snapshot_csv', 'apply_ara_external_confirmation']
