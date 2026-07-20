"""Specialty and intraday scanners for IDX Super Scanner.

Contains Sniper Entry, BPJS, BSJP, Multibagger, PRE-ARA and ARA continuation
logic. It depends on the stable primitives exposed by :mod:`scanner`.
"""
from __future__ import annotations
from scanner import Any, BinaryIO, DownloadReport, classify_provider_error, IDX_DAILY_FINAL_HOUR, IDX_DAILY_FINAL_MINUTE, IDX_REGULAR_DECISION_START_HOUR, IDX_REGULAR_DECISION_START_MINUTE, Iterable, Mapping, MarketContext, ScanConfig, ThreadPoolExecutor, clean_ohlcv, extract_download_batch, safe_number, jakarta_timestamp, pipe_parts, price_structure_target_pair, read_csv_input, safe_text, silent_accumulation_metrics, truthy, as_completed, cmf, fetch_itick_ohlcv, idx_ara_pct, idx_daily_price_band, idx_regular_decision_window, idx_tick_size, is_valid_idx_price, math, near_upper_auto_rejection, normalize_idx_ticker, np, obv, pd, round_idx_price, size_stockbit_order

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
                frame = extract_download_batch(raw, ticker, len(batch))
                frame = mark_interval(clean_ohlcv(frame, strict=True), interval)
                if not frame.empty:
                    histories[ticker] = frame
                    source_tiers[ticker] = f'LIVE_YAHOO_INTRADAY_{interval.upper()}'
                else:
                    failed[ticker] = f'Intraday batch {interval} kosong'
        except Exception as exc:
            for ticker in batch:
                failed[ticker] = f'{classify_provider_error(exc)}: Intraday batch {interval} gagal: {type(exc).__name__}'
    missing = [ticker for ticker in requested if ticker not in histories]

    def retry_one(ticker: str) -> tuple[str, pd.DataFrame, str | None, str | None]:
        primary_error: str | None = None
        try:
            frame = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True, repair=False, actions=False, timeout=15, prepost=False)
            clean = mark_interval(clean_ohlcv(frame, strict=True), interval)
            if not clean.empty:
                return (ticker, clean, None, None)
            primary_error = f'Intraday individual {interval} kosong'
        except Exception as exc:
            primary_error = f'{classify_provider_error(exc)}: {type(exc).__name__}: {str(exc)[:100]}'
        if fallback_interval:
            try:
                frame = yf.Ticker(ticker).history(period=period, interval=fallback_interval, auto_adjust=True, repair=False, actions=False, timeout=15, prepost=False)
                clean = mark_interval(clean_ohlcv(frame, strict=True), fallback_interval)
                if not clean.empty:
                    warning = f'{interval} tidak tersedia; memakai fallback {fallback_interval}'
                    return (ticker, clean, None, warning)
            except Exception as exc:
                fallback_error = f'{classify_provider_error(exc)}: {type(exc).__name__}: {str(exc)[:100]}'
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
                failed.setdefault(str(row['ticker']), safe_text(row['error']) or safe_text(row['status']))
    for ticker in requested:
        source_tiers.setdefault(ticker, 'UNAVAILABLE' if ticker not in histories else f'LIVE_YAHOO_INTRADAY_{interval.upper()}')
    report = DownloadReport(requested=requested, downloaded=sorted(histories), failed=failed, provider=f'Free intraday: Yahoo {interval} → iTick optional', adjusted_prices=True, downloaded_at=pd.Timestamp.now(tz='Asia/Jakarta').isoformat(), warnings=warnings, source_tiers=source_tiers)
    return (histories, report)

def _intraday_session(frame: pd.DataFrame, asof: Any | None=None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    attrs = dict(getattr(frame, 'attrs', {}) or {})
    out = clean_ohlcv(frame, strict=True)
    if out.empty:
        return out
    out.attrs.update(attrs)
    if asof is not None:
        interval_minutes = safe_number(attrs.get('interval_minutes'), np.nan)
        if not np.isfinite(interval_minutes) or interval_minutes <= 0:
            try:
                diffs = pd.Series(pd.DatetimeIndex(out.index)[1:] - pd.DatetimeIndex(out.index)[:-1])
                diffs = diffs.dt.total_seconds().div(60.0)
                diffs = diffs[(diffs > 0) & (diffs <= 60)]
                interval_minutes = float(diffs.median()) if not diffs.empty else 5.0
            except Exception:
                interval_minutes = 5.0
        bar_delta = pd.to_timedelta(float(interval_minutes), unit='min')
        cutoff = jakarta_timestamp(asof).tz_localize(None) - bar_delta
        out = out[pd.DatetimeIndex(out.index) <= cutoff].copy()
        out.attrs.update(attrs)
        if out.empty:
            return out
    last_date = pd.Timestamp(out.index[-1]).date()
    session = out[pd.Index(out.index).map(lambda x: pd.Timestamp(x).date() == last_date)].copy()
    session.attrs.update(attrs)
    return session

def _intraday_interval_minutes(frame: pd.DataFrame) -> float:
    attr_value = safe_number(getattr(frame, 'attrs', {}).get('interval_minutes'), np.nan)
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
    reference = jakarta_timestamp(now)
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
    return {'suggested_lots': int(safe_number(sized.get('suggested_lots'), 0)), 'capital_required_idr': safe_number(sized.get('capital_required_idr'), 0), 'max_loss_idr': safe_number(sized.get('max_loss_idr'), 0), 'specialty_position_cap_pct': 100.0 * position_cap, 'specialty_risk_cap_pct': 100.0 * risk_cap}


def _specialty_prebudget_gate(
    *,
    mode: str,
    signal_ready: bool,
    in_window: bool,
    intraday_fresh: bool,
    requires_intraday: bool,
    entry: float,
    stop: float,
    tp1: float,
    tp2: float,
    rr1: float,
    rr2: float,
    risk_pct: float,
    adtv: float,
    target_valid: bool,
    market_regime: str,
    context_blocker: str,
    cfg: ScanConfig,
    risk_fraction: float | None = None,
    position_fraction: float | None = None,
) -> dict[str, Any]:
    """Separate structural setup validity from optional account-order gating.

    SIGNAL_FIRST is the default: a technically valid setup remains visible even
    when regime, liquidity preference, RR, stop width, or account size are not
    ideal. ACCOUNT_GUARDED keeps the conservative shared-budget order workflow.
    """
    mode_key = str(mode).upper()
    policy = safe_text(getattr(cfg, 'execution_policy', 'SIGNAL_FIRST')).upper() or 'SIGNAL_FIRST'
    account_guarded = policy == 'ACCOUNT_GUARDED'
    limits = {
        'BPJS': (0.045, 1.20, 1.80),
        'BSJP': (0.040, 1.00, 1.50),
        'ARA': (0.060, 1.20, 1.80),
        'SNIPER': (min(cfg.max_stop_pct, 0.060), 1.50, 2.20),
    }
    max_stop, min_rr1, min_rr2 = limits.get(mode_key, (cfg.max_stop_pct, cfg.min_rr1, cfg.min_rr2))
    risk_cap = min(
        cfg.specialty_risk_per_trade_pct,
        float(risk_fraction) if risk_fraction is not None else cfg.specialty_risk_per_trade_pct,
    )
    position_cap = min(
        cfg.specialty_max_position_pct,
        float(position_fraction) if position_fraction is not None else cfg.specialty_max_position_pct,
    )
    sizing = _specialty_sizing(entry, stop, cfg, position_cap=position_cap, risk_cap=risk_cap)

    hard_blockers: list[str] = []
    risk_warnings: list[str] = []
    account_warnings: list[str] = []
    if not signal_ready:
        hard_blockers.append('SETUP_TRIGGER_NOT_READY')
    if not in_window:
        hard_blockers.append('OUTSIDE_EXECUTION_WINDOW')
    if requires_intraday and not intraday_fresh:
        hard_blockers.append('INTRADAY_NOT_FRESH')
    if not target_valid:
        hard_blockers.append('PRICE_STRUCTURE_TARGETS_INVALID')
    if context_blocker:
        hard_blockers.append('CRITICAL_MARKET_OR_DATA_CONTEXT')
    levels = (entry, stop, tp1, tp2)
    if not all(is_valid_idx_price(safe_number(value, np.nan)) for value in levels):
        hard_blockers.append('INVALID_IDX_ORDER_LEVELS')
    elif not (stop < entry < tp1 < tp2):
        hard_blockers.append('INVALID_RISK_TARGET_SEQUENCE')

    regime = str(market_regime or 'UNKNOWN').upper()
    if regime in {'RISK_OFF', 'UNKNOWN', 'NOT_EVALUATED', ''}:
        risk_warnings.append('MARKET_REGIME_NOT_RISK_ON')
    minimum_adtv = max(500_000_000.0, min(float(cfg.min_adtv_idr), 1_000_000_000.0))
    if not np.isfinite(adtv) or adtv < minimum_adtv:
        risk_warnings.append('LIQUIDITY_BELOW_SPECIALTY_PREFERENCE')
    if not np.isfinite(risk_pct) or risk_pct <= 0 or risk_pct > max_stop:
        risk_warnings.append('STOP_DISTANCE_EXCEEDS_MODE_PREFERENCE')
    if not np.isfinite(rr1) or rr1 < min_rr1:
        risk_warnings.append('RR1_BELOW_MODE_PREFERENCE')
    if not np.isfinite(rr2) or rr2 < min_rr2:
        risk_warnings.append('RR2_BELOW_MODE_PREFERENCE')
    if int(safe_number(sizing.get('suggested_lots'), 0)) < 1:
        account_warnings.append('ACCOUNT_SIZE_CANNOT_SUPPORT_ONE_LOT')

    hard_blockers = list(dict.fromkeys(hard_blockers))
    risk_warnings = list(dict.fromkeys(risk_warnings))
    account_warnings = list(dict.fromkeys(account_warnings))
    setup_ready = bool(signal_ready and not hard_blockers)
    guarded_eligible = bool(setup_ready and not risk_warnings and not account_warnings)
    prebudget_eligible = bool(account_guarded and guarded_eligible)
    setup_state = 'SETUP_READY' if setup_ready else 'DAILY_RADAR'
    order_state = 'PRE_BUDGET_READY' if prebudget_eligible else 'ACCOUNT_GUARD_WAIT' if account_guarded and setup_ready else 'USER_MANAGED' if setup_ready else 'DAILY_RADAR'
    execution_blockers = hard_blockers + (risk_warnings + account_warnings if account_guarded else [])

    return {
        **sizing,
        'signal_ready': bool(signal_ready),
        'setup_ready': setup_ready,
        'setup_state': setup_state,
        'execution_policy': policy,
        'specialty_prebudget_order_eligible': prebudget_eligible,
        'specialty_order_ready': False,
        'specialty_order_state': order_state,
        'specialty_execution_blockers': ' | '.join(execution_blockers),
        'specialty_hard_blockers': ' | '.join(hard_blockers),
        'specialty_risk_warnings': ' | '.join(risk_warnings),
        'specialty_account_warnings': ' | '.join(account_warnings),
        'account_risk_gate_applied': account_guarded,
        'sizing_is_informational': not account_guarded,
        'stockbit_order_template': 'BRACKET_ORDER_LIMIT',
        'stockbit_time_in_force': 'GFD',
        'broker_submission_mode': 'MANUAL_STOCKBIT',
        'requires_stockbit_price_check': True,
        'opening_gap_recheck_required': True,
        'stockbit_order_lots': 0,
    }

def _apply_specialty_portfolio_budget(
    screens: dict[str, pd.DataFrame],
    cfg: ScanConfig,
    *,
    current_positions: int = 0,
    current_open_risk_idr: float = 0.0,
    cash_on_hand_idr: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Reserve one shared cash/risk budget across all specialty strategies."""
    out = {name: frame.copy() for name, frame in screens.items()}
    policy = safe_text(getattr(cfg, 'execution_policy', 'SIGNAL_FIRST')).upper() or 'SIGNAL_FIRST'
    if policy != 'ACCOUNT_GUARDED':
        for frame in out.values():
            if frame.empty:
                continue
            frame['specialty_order_ready'] = False
            frame['account_risk_gate_applied'] = False
            frame['sizing_is_informational'] = True
            frame['stockbit_order_lots'] = 0
        return out
    available_cash = max(0.0, float(cfg.cash_on_hand_idr if cash_on_hand_idr is None else cash_on_hand_idr))
    overall_risk = max(0.0, cfg.account_size_idr * cfg.max_portfolio_risk_pct - max(0.0, float(current_open_risk_idr)))
    specialty_risk = max(0.0, cfg.account_size_idr * cfg.max_specialty_portfolio_risk_pct)
    available_risk = min(overall_risk, specialty_risk)
    available_slots = max(0, min(int(cfg.max_specialty_positions), int(cfg.max_positions) - max(0, int(current_positions))))
    score_columns = {'bpjs': 'bpjs_score', 'bsjp': 'bsjp_score', 'ara_hunter': 'ara_model_score', 'sniper': 'sniper_score'}
    status_columns = {'bpjs': 'bpjs_status', 'bsjp': 'bsjp_status', 'ara_hunter': 'ara_hunter_status', 'sniper': 'sniper_status'}
    candidates: list[tuple[float, str, object, str]] = []
    for name, score_column in score_columns.items():
        frame = out.get(name, pd.DataFrame())
        if frame.empty or 'specialty_prebudget_order_eligible' not in frame:
            continue
        for idx, row in frame[frame['specialty_prebudget_order_eligible'].map(truthy)].iterrows():
            score = safe_number(row.get(score_column), 0.0)
            rr2 = min(5.0, max(0.0, safe_number(row.get('rr2'), 0.0)))
            candidates.append((score + 2.0 * rr2, name, idx, str(row.get('ticker'))))
    selected_tickers: set[str] = set()
    for _, name, idx, ticker in sorted(candidates, reverse=True):
        frame = out[name]
        capital = max(0.0, safe_number(frame.at[idx, 'capital_required_idr'], np.inf))
        risk = max(0.0, safe_number(frame.at[idx, 'max_loss_idr'], np.inf))
        blockers: list[str] = []
        if ticker in selected_tickers:
            blockers.append('ALTERNATE_STRATEGY_SAME_TICKER')
        if available_slots <= 0:
            blockers.append('SPECIALTY_POSITION_SLOT_UNAVAILABLE')
        if not np.isfinite(capital) or capital > available_cash:
            blockers.append('SPECIALTY_CASH_BUDGET')
        if not np.isfinite(risk) or risk > available_risk:
            blockers.append('SPECIALTY_PORTFOLIO_HEAT')
        if blockers:
            frame.at[idx, 'specialty_order_state'] = 'SIGNAL_READY_BUDGET_WAIT'
            existing = [item for item in safe_text(frame.at[idx, 'specialty_execution_blockers']).split(' | ') if item]
            frame.at[idx, 'specialty_execution_blockers'] = ' | '.join(list(dict.fromkeys(existing + blockers)))
            continue
        frame.at[idx, 'specialty_order_ready'] = True
        frame.at[idx, 'specialty_order_state'] = 'ORDER_READY'
        frame.at[idx, 'stockbit_order_lots'] = int(safe_number(frame.at[idx, 'suggested_lots'], 0))
        frame.at[idx, 'order_instruction'] = 'STOCKBIT_BRACKET_GFD_RECHECK_PRICE'
        status_column = status_columns[name]
        if name == 'ara_hunter':
            model = safe_text(frame.at[idx, 'ara_model']).upper()
            frame.at[idx, status_column] = 'ARA_CONTINUATION_ORDER_READY' if model == 'ARA_CONTINUATION' else 'PRE_ARA_ORDER_READY'
        else:
            frame.at[idx, status_column] = f'{name.upper()}_ORDER_READY'
        selected_tickers.add(ticker)
        available_cash -= capital
        available_risk -= risk
        available_slots -= 1
    for name, frame in out.items():
        if frame.empty or 'specialty_order_state' not in frame:
            continue
        frame['specialty_remaining_cash_idr'] = available_cash
        frame['specialty_remaining_risk_idr'] = available_risk
        frame['specialty_remaining_slots'] = available_slots
    return out


def build_daily_opportunity_board(screens: Mapping[str, pd.DataFrame], per_strategy: int = 3) -> pd.DataFrame:
    """Return daily ranked candidates without pretending every row is a trade."""
    specs = {
        'BPJS': ('bpjs', 'bpjs_status', 'bpjs_score', 'entry', 'stop_loss', 'day_tp1', 'day_tp2'),
        'BSJP': ('bsjp', 'bsjp_status', 'bsjp_score', 'entry', 'stop_loss', 'morning_tp1', 'morning_tp2'),
        'ARA_HUNTER': ('ara_hunter', 'ara_hunter_status', 'ara_model_score', 'entry_reference', 'hard_stop', 'ara_tp1', 'ara_tp2'),
        'SNIPER': ('sniper', 'sniper_status', 'sniper_score', 'sniper_entry', 'sniper_stop', 'sniper_tp1', 'sniper_tp2'),
        'MULTIBAGGER': ('multibagger', 'multibagger_status', 'capital_conviction_score', 'entry', 'stop_loss', 'tp1', 'tp2'),
    }
    rows: list[dict[str, Any]] = []
    for strategy, (key, status_col, score_col, entry_col, stop_col, tp1_col, tp2_col) in specs.items():
        frame = screens.get(key, pd.DataFrame())
        if frame is None or frame.empty or 'ticker' not in frame:
            continue
        ranked = frame.sort_values(score_col, ascending=False, na_position='last').head(max(1, int(per_strategy)))
        for _, row in ranked.iterrows():
            decision = safe_text(row.get('specialty_order_state')) or safe_text(row.get('compounding_state')) or 'DAILY_RADAR'
            rows.append({
                'strategy': strategy,
                'ticker': row.get('ticker'),
                'decision_state': decision,
                'status': row.get(status_col),
                'score': safe_number(row.get(score_col), np.nan),
                'entry': row.get(entry_col, np.nan),
                'stop_loss': row.get(stop_col, np.nan),
                'tp1': row.get(tp1_col, np.nan),
                'tp2': row.get(tp2_col, np.nan),
                'rr1': row.get('rr1', np.nan),
                'rr2': row.get('rr2', np.nan),
                'order_ready': truthy(row.get('specialty_order_ready', False)),
                'stockbit_order_lots': int(safe_number(row.get('stockbit_order_lots'), 0)),
                'capital_destination': 'MULTIBAGGER_COMPOUNDING' if strategy == 'MULTIBAGGER' else 'DAILY_PROFIT_ENGINE',
                'capital_priority_rank': row.get('capital_priority_rank', np.nan),
                'strategic_target_weight_pct': row.get('strategic_target_weight_pct', np.nan),
                'recommended_allocation_idr': row.get('recommended_allocation_idr', np.nan),
                'next_action': row.get('order_instruction', row.get('review_action', row.get('action', 'WATCH_ONLY'))),
                'blockers': row.get('specialty_execution_blockers', row.get('red_flags', row.get('blockers', ''))),
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        rank = {'ORDER_READY': 0, 'ACCUMULATE_NOW': 1, 'PRE_BUDGET_READY': 2, 'SETUP_READY': 2, 'USER_MANAGED': 2, 'SIGNAL_READY': 3, 'ACCOUNT_GUARD_WAIT': 4, 'SIGNAL_READY_BUDGET_WAIT': 4, 'DAILY_RADAR': 5, 'RESEARCH_ONLY': 6}
        result['_rank'] = result['decision_state'].map(rank).fillna(9)
        result = result.sort_values(['_rank', 'score'], ascending=[True, False]).drop(columns='_rank').reset_index(drop=True)
    return result


def _profit_liquidity_score(adtv: Any) -> float:
    value = max(0.0, safe_number(adtv, 0.0))
    if value <= 0:
        return 0.0
    # Log scale: Rp250m≈20, Rp1b≈45, Rp3b≈70, Rp10b≈95.
    return float(max(0.0, min(100.0, 20.0 + 50.0 * np.log10(max(value, 250_000_000.0) / 250_000_000.0) / np.log10(40.0))))


def _profit_target_score(rr1: Any, rr2: Any, target_valid: Any=True) -> float:
    if not truthy(target_valid):
        return 0.0
    first = safe_number(rr1, np.nan)
    second = safe_number(rr2, np.nan)
    if not np.isfinite(first) or not np.isfinite(second):
        return 20.0
    return float(max(0.0, min(100.0, 18.0 + 24.0 * min(2.0, max(0.0, first)) + 17.0 * min(3.0, max(0.0, second)))))


def _profit_grade(score: float) -> str:
    if score >= 88.0:
        return 'A+'
    if score >= 80.0:
        return 'A'
    if score >= 72.0:
        return 'B+'
    if score >= 64.0:
        return 'B'
    return 'C'


def build_profit_order_builder(
    core_signals: pd.DataFrame | None,
    screens: Mapping[str, pd.DataFrame],
    config: ScanConfig | None=None,
) -> pd.DataFrame:
    """Create one cross-strategy conviction ranking for the manual Order Builder.

    Conviction is not a promise of profit. It is a comparable 0-100 ranking of
    structural quality, timing, demand/flow, liquidity, target geometry, data
    quality and—where available—historical/OOS evidence. Account size and
    portfolio heat do not affect the score in SIGNAL_FIRST mode.
    """
    cfg = config or ScanConfig()
    rows: list[dict[str, Any]] = []

    def append_row(
        *, strategy: str, row: Mapping[str, Any], status: str, decision_state: str,
        entry: Any, stop: Any, tp1: Any, tp2: Any, rr1: Any, rr2: Any,
        structure: float, timing: float, flow: float, liquidity: float,
        target: float, data: float, validation: float, weights: tuple[float, ...],
        next_action: str, warnings: str='', horizon: str='', trigger: Any=np.nan,
        target_valid: Any=True,
    ) -> None:
        components = [structure, timing, flow, liquidity, target, data, validation]
        score = float(sum(w * max(0.0, min(100.0, c)) for w, c in zip(weights, components)))
        if not truthy(target_valid):
            score -= 15.0
        if safe_number(rr2, np.nan) < 1.0:
            score -= 8.0
        score = round(max(0.0, min(100.0, score)), 1)
        rows.append({
            'profit_rank': np.nan,
            'ticker': row.get('ticker'),
            'strategy': strategy,
            'horizon': horizon,
            'decision_state': decision_state,
            'setup_status': status,
            'profit_conviction_score': score,
            'conviction_grade': _profit_grade(score),
            'entry': safe_number(entry, np.nan),
            'trigger_price': safe_number(trigger, np.nan),
            'stop_loss': safe_number(stop, np.nan),
            'tp1': safe_number(tp1, np.nan),
            'tp2': safe_number(tp2, np.nan),
            'rr1': safe_number(rr1, np.nan),
            'rr2': safe_number(rr2, np.nan),
            'structure_score': round(structure, 1),
            'timing_score': round(timing, 1),
            'flow_score': round(flow, 1),
            'liquidity_score': round(liquidity, 1),
            'target_quality_score': round(target, 1),
            'data_quality_score': round(data, 1),
            'validation_score': round(validation, 1),
            'order_ready': truthy(row.get('specialty_order_ready')) or truthy(row.get('autopilot_verified')),
            'stockbit_order_lots': int(safe_number(row.get('stockbit_order_lots'), 0.0)),
            'next_action': next_action,
            'warnings': warnings,
            'conviction_basis': (
                f'Structure {structure:.0f}; timing {timing:.0f}; flow {flow:.0f}; '
                f'liquidity {liquidity:.0f}; target {target:.0f}; data {data:.0f}; validation {validation:.0f}'
            ),
        })

    # Core profit engine: controlled pullback continuation only. Other core
    # patterns remain visible in their dedicated tabs and do not dilute the
    # strategy comparison requested by the user.
    if core_signals is not None and not core_signals.empty:
        core = core_signals[core_signals.get('setup', pd.Series(index=core_signals.index, dtype=object)).eq('PULLBACK_CONTINUATION')].copy()
        eligible_status = {'EXECUTION_READY', 'READY_FOR_STOCKBIT_VERIFY', 'SIGNAL_READY', 'ENTRY_PLAN_READY'}
        core = core[core.get('status', pd.Series(index=core.index, dtype=object)).isin(eligible_status)]
        for _, row in core.iterrows():
            status = safe_text(row.get('status'))
            action = safe_text(row.get('action'))
            quality = safe_number(row.get('quality_score'), 0.0)
            analyst = safe_number(row.get('analyst_fusion_score'), quality)
            structure = 0.55 * quality + 0.45 * analyst
            distance = safe_number(row.get('distance_atr'), 99.0)
            action_score = 100.0 if action == 'READY_TRIGGER' else 82.0 if action in {'READY_LIMIT', 'WAIT_PULLBACK_CONFIRMATION'} else 55.0
            status_score = {'EXECUTION_READY': 100.0, 'READY_FOR_STOCKBIT_VERIFY': 96.0, 'SIGNAL_READY': 84.0, 'ENTRY_PLAN_READY': 70.0}.get(status, 50.0)
            proximity = max(0.0, 100.0 - 35.0 * max(0.0, distance)) if np.isfinite(distance) else 30.0
            timing = 0.45 * action_score + 0.35 * status_score + 0.20 * proximity
            flow = safe_number(row.get('silent_accumulation_score'), 50.0)
            liquidity = _profit_liquidity_score(row.get('adtv20_idr'))
            target = _profit_target_score(row.get('rr1'), row.get('rr2'), row.get('target_structure_valid', True))
            completeness = safe_number(row.get('data_completeness_score'), 50.0)
            confidence = safe_number(row.get('execution_confidence_score'), completeness)
            data = 0.55 * completeness + 0.45 * confidence
            validation = safe_number(row.get('validation_gate_score'), 50.0)
            probability = safe_number(row.get('probability_estimate'), np.nan)
            if np.isfinite(probability):
                probability_pct = 100.0 * probability if 0.0 <= probability <= 1.0 else probability
                validation = 0.55 * validation + 0.45 * max(0.0, min(100.0, probability_pct))
            append_row(
                strategy='PULLBACK_CONTINUATION', row=row, status=status,
                decision_state='SETUP_READY' if status != 'ENTRY_PLAN_READY' else 'ENTRY_PLAN',
                entry=row.get('entry'), trigger=row.get('stockbit_trigger_price', row.get('trigger')),
                stop=row.get('stop_loss'), tp1=row.get('tp1'), tp2=row.get('tp2'),
                rr1=row.get('rr1'), rr2=row.get('rr2'), structure=structure,
                timing=timing, flow=flow, liquidity=liquidity, target=target,
                data=data, validation=validation,
                weights=(0.28, 0.20, 0.12, 0.10, 0.12, 0.10, 0.08),
                next_action=safe_text(row.get('order_instruction')) or action,
                warnings=safe_text(row.get('signal_risk_warnings')) or safe_text(row.get('evidence_warnings')),
                horizon='2–20 trading days', target_valid=row.get('target_structure_valid', True),
            )

    specs = [
        ('SNIPER', 'sniper', 'sniper_status', 'sniper_score', 'sniper_entry', 'sniper_trigger', 'sniper_stop', 'sniper_tp1', 'sniper_tp2'),
        ('BPJS', 'bpjs', 'bpjs_status', 'bpjs_score', 'entry', np.nan, 'stop_loss', 'day_tp1', 'day_tp2'),
        ('BSJP', 'bsjp', 'bsjp_status', 'bsjp_score', 'entry', np.nan, 'stop_loss', 'morning_tp1', 'morning_tp2'),
        ('ARA', 'ara_hunter', 'ara_hunter_status', 'ara_model_score', 'entry_reference', np.nan, 'hard_stop', 'ara_tp1', 'ara_tp2'),
    ]
    for strategy, key, status_col, score_col, entry_col, trigger_col, stop_col, tp1_col, tp2_col in specs:
        frame = screens.get(key, pd.DataFrame())
        if frame is None or frame.empty:
            continue
        for _, row in frame.iterrows():
            status = safe_text(row.get(status_col))
            if strategy == 'SNIPER':
                allowed = {'SNIPER_ORDER_READY', 'SNIPER_SIGNAL_READY', 'WAIT_SNIPER_RETRACE'}
            elif strategy == 'BPJS':
                allowed = {'BPJS_ORDER_READY', 'BPJS_SIGNAL_READY'}
            elif strategy == 'BSJP':
                allowed = {'BSJP_ORDER_READY', 'BSJP_SIGNAL_READY'}
            else:
                allowed = {'PRE_ARA_ORDER_READY', 'PRE_ARA_SIGNAL_READY', 'ARA_CONTINUATION_ORDER_READY', 'ARA_CONTINUATION_SIGNAL_READY', 'ARA_CONTINUATION_FLOW_VERIFIED_SIGNAL'}
            if status not in allowed:
                continue
            base = safe_number(row.get(score_col), 0.0)
            liquidity = _profit_liquidity_score(row.get('adtv20_idr'))
            target = _profit_target_score(row.get('rr1'), row.get('rr2'), row.get('target_structure_valid', True))
            validation = 50.0
            if strategy == 'SNIPER':
                structure = base
                trigger_state = safe_text(row.get('trigger_state'))
                timing = 100.0 if trigger_state == 'CONFIRMED' else 82.0 if trigger_state == 'RESTING_LIMIT_VALID' else 55.0
                flow = safe_number(row.get('silent_accumulation_score'), 50.0)
                data = 92.0 if safe_text(row.get('primary_sniper_blocker')) == 'NONE' else 70.0
                validation = 55.0
                weights = (0.30, 0.22, 0.12, 0.08, 0.12, 0.08, 0.08)
                decision = 'SETUP_READY' if 'SIGNAL_READY' in status or 'ORDER_READY' in status else 'ENTRY_PLAN'
                horizon = '1–15 trading days'
                warnings = safe_text(row.get('risk_warnings'))
            elif strategy == 'BPJS':
                structure = base
                timing = 0.35 * (100.0 if truthy(row.get('orb_breakout_confirmed')) else 35.0) + 0.25 * (100.0 if truthy(row.get('orb_hold_confirmed')) else 35.0) + 0.20 * max(0.0, min(100.0, 100.0 * safe_number(row.get('session_close_location'), 0.0))) + 0.20 * max(0.0, min(100.0, 50.0 + 10000.0 * safe_number(row.get('vwap_slope_pct'), 0.0)))
                flow = max(0.0, min(100.0, 45.0 + 20.0 * safe_number(row.get('breakout_volume_ratio'), 0.0) + 35.0 * max(0.0, safe_number(row.get('directional_efficiency'), 0.0))))
                data = 100.0 if truthy(row.get('intraday_fresh')) else 35.0
                weights = (0.25, 0.25, 0.15, 0.10, 0.10, 0.10, 0.05)
                decision = 'SETUP_READY'
                horizon = 'Intraday; exit before close'
                warnings = safe_text(row.get('warnings'))
            elif strategy == 'BSJP':
                structure = base
                timing = 0.30 * max(0.0, min(100.0, 100.0 * safe_number(row.get('session_close_location'), 0.0))) + 0.25 * max(0.0, min(100.0, 50.0 + 10000.0 * safe_number(row.get('vwap_slope_pct'), 0.0))) + 0.25 * max(0.0, min(100.0, 45.0 * safe_number(row.get('late_volume_acceleration'), 0.0))) + 0.20 * (100.0 if truthy(row.get('afternoon_breakout')) else 55.0)
                flow = max(0.0, min(100.0, 45.0 + 35.0 * max(0.0, safe_number(row.get('directional_efficiency'), 0.0)) + 35.0 * max(0.0, safe_number(row.get('late_volume_share'), 0.0))))
                data = 100.0 if truthy(row.get('intraday_fresh')) else 35.0
                weights = (0.25, 0.25, 0.15, 0.10, 0.10, 0.10, 0.05)
                decision = 'SETUP_READY'
                horizon = 'Overnight; exit next morning'
                warnings = safe_text(row.get('warnings'))
            else:
                model = safe_text(row.get('ara_model')) or 'PRE_ARA'
                strategy = model
                structure = base
                timing = 100.0 if 'ORDER_READY' in status or 'FLOW_VERIFIED' in status else 90.0
                flow = 0.50 * safe_number(row.get('orderflow_proxy_score'), 50.0) + 0.50 * safe_number(row.get('queue_proxy_score'), 50.0)
                data = 90.0 if safe_text(row.get('proxy_confidence')) == 'HIGH' else 60.0
                validation = max(25.0, min(100.0, 100.0 * safe_number(row.get('historical_shrunk_rate'), 0.25)))
                weights = (0.25, 0.20, 0.20, 0.10, 0.10, 0.10, 0.05)
                decision = 'SETUP_READY'
                horizon = 'Session now' if safe_text(row.get('prediction_horizon')) == 'SESSION_NOW' else 'Next session'
                warnings = safe_text(row.get('warnings'))
            append_row(
                strategy=strategy, row=row, status=status, decision_state=decision,
                entry=row.get(entry_col), trigger=row.get(trigger_col) if isinstance(trigger_col, str) else np.nan,
                stop=row.get(stop_col), tp1=row.get(tp1_col), tp2=row.get(tp2_col),
                rr1=row.get('rr1'), rr2=row.get('rr2'), structure=structure, timing=timing,
                flow=flow, liquidity=liquidity, target=target, data=data,
                validation=validation, weights=weights,
                next_action=safe_text(row.get('order_instruction')) or safe_text(row.get('action')),
                warnings=warnings, horizon=horizon, target_valid=row.get('target_structure_valid', True),
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    minimum = safe_number(getattr(cfg, 'profit_conviction_min_score', 68.0), 68.0)
    result = result[result['profit_conviction_score'] >= minimum].copy()
    if result.empty:
        return result
    readiness_rank = {'ORDER_READY': 0, 'SETUP_READY': 1, 'ENTRY_PLAN': 2}
    result['_readiness'] = result['decision_state'].map(readiness_rank).fillna(9)
    result = result.sort_values(['profit_conviction_score', '_readiness', 'target_quality_score', 'liquidity_score'], ascending=[False, True, False, False])
    # Only one primary order plan per ticker. Preserve alternate strategies for context.
    alternatives = result.groupby('ticker')['strategy'].apply(lambda s: ' • '.join(dict.fromkeys(map(str, s)))).to_dict()
    result = result.drop_duplicates('ticker', keep='first').copy()
    result['alternate_strategies'] = result['ticker'].map(alternatives)
    result = result.drop(columns='_readiness').head(int(getattr(cfg, 'profit_order_builder_limit', 20))).reset_index(drop=True)
    result['profit_rank'] = np.arange(1, len(result) + 1)
    return result

def _bounded_score(value: Any, maximum: float) -> float:
    """Normalize a non-negative component to a transparent 0-100 scale."""
    numeric = max(0.0, safe_number(value, 0.0))
    return float(max(0.0, min(100.0, 100.0 * numeric / max(1e-9, float(maximum)))))


def _multibagger_data_integrity_score(row: Mapping[str, Any]) -> float:
    grade = safe_text(row.get('fundamental_data_grade')).upper() or 'D'
    reliability = safe_text(row.get('fundamental_reliability')).upper() or 'UNKNOWN'
    grade_score = {'A': 100.0, 'B': 82.0, 'C': 60.0, 'D': 25.0}.get(grade, 20.0)
    reliability_score = {'HIGH': 100.0, 'MEDIUM': 72.0, 'LOW': 42.0, 'UNKNOWN': 20.0}.get(reliability, 20.0)
    consensus = max(0.0, min(100.0, safe_number(row.get('fundamental_consensus_score'), 0.0)))
    history = max(0.0, min(100.0, safe_number(row.get('fundamental_history_coverage'), 0.0)))
    official = 100.0 if truthy(row.get('fundamental_official_verified')) else 55.0 if truthy(row.get('fundamental_official_reference')) else 25.0
    source_count = max(0.0, safe_number(row.get('fundamental_source_count'), 0.0))
    source_score = 100.0 if source_count >= 3 else 80.0 if source_count >= 2 else 35.0 if source_count >= 1 else 0.0
    return round(
        0.25 * grade_score
        + 0.20 * reliability_score
        + 0.20 * consensus
        + 0.15 * history
        + 0.10 * official
        + 0.10 * source_score,
        1,
    )


def _multibagger_solvency_strength(row: Mapping[str, Any]) -> float:
    if safe_text(row.get('fundamental_model')).upper() == 'FINANCIAL':
        car = safe_number(row.get('car'), np.nan)
        npl = safe_number(row.get('npl_gross'), np.nan)
        ldr = safe_number(row.get('ldr'), np.nan)
        car_score = 100.0 if np.isfinite(car) and car >= 0.20 else 82.0 if np.isfinite(car) and car >= 0.15 else 65.0 if np.isfinite(car) and car >= 0.12 else 0.0
        npl_score = 100.0 if np.isfinite(npl) and npl <= 0.015 else 78.0 if np.isfinite(npl) and npl <= 0.03 else 40.0 if np.isfinite(npl) and npl <= 0.05 else 0.0
        ldr_score = 100.0 if np.isfinite(ldr) and 0.75 <= ldr <= 0.90 else 75.0 if np.isfinite(ldr) and 0.65 <= ldr <= 1.00 else 25.0 if np.isfinite(ldr) else 0.0
        return round(0.40 * car_score + 0.35 * npl_score + 0.25 * ldr_score, 1)
    balance = _bounded_score(row.get('balance_sheet_score'), 12.0)
    coverage = max(0.0, min(100.0, safe_number(row.get('solvency_coverage'), 0.0)))
    return round(0.78 * balance + 0.22 * coverage, 1)


def _multibagger_technical_timing_score(state: Any) -> float:
    value = safe_text(state).upper()
    if value in {'EXECUTION_READY', 'READY_FOR_STOCKBIT_VERIFY'}:
        return 100.0
    if value in {'ENTRY_PLAN_READY', 'READY_FOR_PRICE_VERIFY'}:
        return 82.0
    if value == 'SIGNAL_READY':
        return 68.0
    if value in {'WATCHLIST_ENTRY', 'PENDING_DATA', 'PENDING_CLOSE'}:
        return 42.0
    return 30.0


def _multibagger_capital_tier(row: Mapping[str, Any], conviction: float) -> tuple[str, float]:
    status = safe_text(row.get('multibagger_status')).upper()
    grade_a = status == 'MULTIBAGGER_A_CANDIDATE'
    grade_b = status == 'MULTIBAGGER_B_CANDIDATE'
    if grade_a and conviction >= 88.0:
        return ('CORE_COMPOUNDING', 1.35)
    if grade_a and conviction >= 80.0:
        return ('HIGH_CONVICTION', 1.10)
    if grade_b and conviction >= 78.0:
        return ('SATELLITE_GROWTH', 0.82)
    if grade_b and conviction >= 72.0:
        return ('STARTER_ONLY', 0.58)
    return ('WATCH_ONLY', 0.0)


def _multibagger_tier_cap_pct(tier: str, cfg: ScanConfig) -> float:
    values = {
        'CORE_COMPOUNDING': 100.0 * max(0.0, min(1.0, cfg.multibagger_core_cap_pct)),
        'HIGH_CONVICTION': 100.0 * max(0.0, min(1.0, cfg.multibagger_high_cap_pct)),
        'SATELLITE_GROWTH': 100.0 * max(0.0, min(1.0, cfg.multibagger_satellite_cap_pct)),
        'STARTER_ONLY': 100.0 * max(0.0, min(1.0, cfg.multibagger_starter_cap_pct)),
    }
    return float(values.get(tier, 0.0))


def _capped_weight_distribution(raw_weights: Mapping[object, float], caps: Mapping[object, float]) -> dict[object, float]:
    """Normalize weights with per-name concentration caps; unused weight stays cash."""
    remaining = 100.0
    active = {idx for idx, value in raw_weights.items() if value > 0 and caps.get(idx, 0.0) > 0}
    allocated = {idx: 0.0 for idx in raw_weights}
    while active and remaining > 1e-8:
        raw_total = sum(max(0.0, raw_weights[idx]) for idx in active)
        if raw_total <= 0:
            break
        provisional = {idx: remaining * max(0.0, raw_weights[idx]) / raw_total for idx in active}
        capped_now = [idx for idx in active if provisional[idx] >= caps[idx] - allocated[idx] - 1e-8]
        if not capped_now:
            for idx in active:
                allocated[idx] += provisional[idx]
            remaining = 0.0
            break
        for idx in capped_now:
            room = max(0.0, caps[idx] - allocated[idx])
            allocated[idx] += room
            remaining -= room
            active.remove(idx)
    return {idx: round(max(0.0, value), 2) for idx, value in allocated.items()}


def allocate_multibagger_capital(candidates: pd.DataFrame, config: ScanConfig | None=None) -> pd.DataFrame:
    """Rank where the Multibagger sleeve should place the most capital.

    Strategic target weights are conviction-based and capped per tier. Actual
    deployment is allowed only for ACCUMULATE_NOW/STARTER_NOW; weights assigned
    to candidates waiting for an entry zone remain explicit cash reserve.
    """
    cfg = config or ScanConfig()
    if candidates is None or candidates.empty:
        return candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    out = candidates.copy()
    defaults = {
        'capital_conviction_score': 0.0,
        'capital_tier': 'WATCH_ONLY',
        'capital_priority_score': 0.0,
        'capital_priority_rank': np.nan,
        'allocation_eligible': False,
        'allocation_cap_pct': 0.0,
        'strategic_target_weight_pct': 0.0,
        'deploy_now_weight_pct': 0.0,
        'strategic_target_amount_idr': 0.0,
        'recommended_allocation_idr': 0.0,
        'recommended_lots': 0,
        'estimated_order_value_idr': 0.0,
        'allocation_reference_price': np.nan,
        'allocation_action': 'NO_ALLOCATION',
        'allocation_reason': '',
        'multibagger_budget_idr': max(0.0, cfg.multibagger_capital_budget_idr),
        'multibagger_cash_reserve_idr': max(0.0, cfg.multibagger_capital_budget_idr),
    }
    for column, default in defaults.items():
        if column not in out:
            out[column] = default

    pillar_columns = {
        'Growth': ('growth_score', 22.0),
        'Profitability': ('profitability_score', 18.0),
        'Cash-flow quality': ('earnings_quality_score', 18.0),
        'Valuation': ('valuation_score', 8.0),
        'Momentum': ('momentum_score', 12.0),
        'Smart-money proxy': ('accumulation_score', 10.0),
    }
    raw_weights: dict[object, float] = {}
    caps: dict[object, float] = {}
    eligible_rows: list[tuple[float, object]] = []
    minimum = max(0.0, min(100.0, cfg.multibagger_min_capital_conviction))

    for idx, row in out.iterrows():
        normalized = {label: _bounded_score(row.get(column), maximum) for label, (column, maximum) in pillar_columns.items()}
        data_integrity = _multibagger_data_integrity_score(row)
        solvency = _multibagger_solvency_strength(row)
        timing = _multibagger_technical_timing_score(row.get('technical_entry_state'))
        conviction = round(
            0.18 * normalized['Growth']
            + 0.14 * normalized['Profitability']
            + 0.20 * normalized['Cash-flow quality']
            + 0.15 * solvency
            + 0.07 * normalized['Valuation']
            + 0.10 * data_integrity
            + 0.05 * normalized['Momentum']
            + 0.08 * normalized['Smart-money proxy']
            + 0.03 * timing,
            1,
        )
        tier, tier_multiplier = _multibagger_capital_tier(row, conviction)
        red_text = safe_text(row.get('red_flags')).upper()
        severe_tokens = ('OCF NEGATIF', 'MARGIN BERSIH NEGATIF', 'DER TINGGI', 'DILUSI TINGGI')
        severe = truthy(row.get('severe_fundamental_flags')) or bool(safe_text(row.get('fundamental_conflicts')).strip()) or any(token in red_text for token in severe_tokens)
        status = safe_text(row.get('multibagger_status')).upper()
        eligible = bool(
            tier != 'WATCH_ONLY'
            and conviction >= minimum
            and status in {'MULTIBAGGER_A_CANDIDATE', 'MULTIBAGGER_B_CANDIDATE'}
            and not severe
        )
        cap_pct = _multibagger_tier_cap_pct(tier, cfg) if eligible else 0.0
        priority = conviction + (6.0 if status == 'MULTIBAGGER_A_CANDIDATE' else 0.0) + (3.0 if safe_text(row.get('compounding_state')).upper() == 'ACCUMULATE_NOW' else 1.0 if safe_text(row.get('compounding_state')).upper() == 'STARTER_NOW' else 0.0) + (2.0 if truthy(row.get('fundamental_official_verified')) else 0.0)
        out.at[idx, 'capital_conviction_score'] = conviction
        out.at[idx, 'capital_tier'] = tier
        out.at[idx, 'capital_priority_score'] = round(priority, 1)
        out.at[idx, 'allocation_eligible'] = eligible
        out.at[idx, 'allocation_cap_pct'] = round(cap_pct, 1)
        out.at[idx, 'data_integrity_score'] = data_integrity
        out.at[idx, 'solvency_strength_score'] = solvency
        out.at[idx, 'technical_timing_score'] = timing
        strongest = sorted({**normalized, 'Data integrity': data_integrity, 'Solvency': solvency}.items(), key=lambda item: item[1], reverse=True)[:3]
        strongest_text = ', '.join(f'{label} {value:.0f}' for label, value in strongest)
        out.at[idx, 'allocation_reason'] = f'{tier}; conviction {conviction:.1f}/100; strongest pillars: {strongest_text}'
        if eligible:
            eligible_rows.append((priority, idx))

    max_names = max(1, int(cfg.multibagger_max_holdings))
    selected = {idx for _, idx in sorted(eligible_rows, reverse=True)[:max_names]}
    for idx in out.index:
        if idx not in selected:
            out.at[idx, 'allocation_eligible'] = False
            out.at[idx, 'allocation_cap_pct'] = 0.0
            continue
        conviction = safe_number(out.at[idx, 'capital_conviction_score'], 0.0)
        tier = safe_text(out.at[idx, 'capital_tier'])
        tier_multiplier = {'CORE_COMPOUNDING': 1.35, 'HIGH_CONVICTION': 1.10, 'SATELLITE_GROWTH': 0.82, 'STARTER_ONLY': 0.58}.get(tier, 0.0)
        raw_weights[idx] = max(1.0, conviction - 55.0) ** 2 * tier_multiplier
        caps[idx] = safe_number(out.at[idx, 'allocation_cap_pct'], 0.0)

    weights = _capped_weight_distribution(raw_weights, caps)
    ranked_indices = [idx for _, idx in sorted(((safe_number(out.at[idx, 'capital_priority_score']), idx) for idx in selected), reverse=True)]
    for rank, idx in enumerate(ranked_indices, start=1):
        target_weight = weights.get(idx, 0.0)
        state = safe_text(out.at[idx, 'compounding_state']).upper()
        deploy = target_weight if state in {'ACCUMULATE_NOW', 'STARTER_NOW'} else 0.0
        action = 'ALLOCATE_LARGEST' if rank == 1 and deploy > 0 else 'ALLOCATE_NOW' if deploy > 0 else 'WAIT_ENTRY_ZONE'
        out.at[idx, 'capital_priority_rank'] = rank
        out.at[idx, 'strategic_target_weight_pct'] = target_weight
        out.at[idx, 'deploy_now_weight_pct'] = round(deploy, 2)
        out.at[idx, 'allocation_action'] = action

    budget = max(0.0, safe_number(cfg.multibagger_capital_budget_idr, 0.0))
    deployed_value = 0.0
    for idx, row in out.iterrows():
        deploy_weight = safe_number(row.get('deploy_now_weight_pct'), 0.0)
        target_amount = budget * safe_number(row.get('strategic_target_weight_pct'), 0.0) / 100.0
        proposed = budget * deploy_weight / 100.0
        entry = safe_number(row.get('entry'), np.nan)
        last_price = safe_number(row.get('last_price'), np.nan)
        reference = entry if np.isfinite(entry) and entry > 0 else last_price
        lots = int(math.floor(proposed / (reference * 100.0))) if proposed > 0 and np.isfinite(reference) and reference > 0 else 0
        order_value = lots * reference * 100.0 if lots > 0 else 0.0
        out.at[idx, 'multibagger_budget_idr'] = budget
        out.at[idx, 'allocation_reference_price'] = reference
        out.at[idx, 'strategic_target_amount_idr'] = round(target_amount, 0)
        out.at[idx, 'recommended_allocation_idr'] = round(proposed, 0)
        out.at[idx, 'recommended_lots'] = lots
        out.at[idx, 'estimated_order_value_idr'] = round(order_value, 0)
        deployed_value += order_value
        if proposed > 0 and lots <= 0:
            out.at[idx, 'allocation_action'] = 'BUDGET_BELOW_ONE_LOT'
    reserve = max(0.0, budget - deployed_value)
    out['multibagger_cash_reserve_idr'] = round(reserve, 0)
    out['capital_rank_selected'] = out.index.to_series().isin(selected)
    return out


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
        coverage = safe_number(fund.get('fundamental_coverage'), 0.0)
        if coverage <= 0:
            continue
        row = frame.iloc[-1]
        close = safe_number(row.get('Close'), 0.0)
        adtv = safe_number(row.get('ADTV20'), 0.0)
        revenue_growth = safe_number(fund.get('revenue_growth'), np.nan)
        earnings_growth = safe_number(fund.get('earnings_growth'), np.nan)
        roe = safe_number(fund.get('roe'), np.nan)
        roa = safe_number(fund.get('roa'), np.nan)
        net_margin = safe_number(fund.get('net_margin'), np.nan)
        operating_margin = safe_number(fund.get('operating_margin'), np.nan)
        debt_equity = safe_number(fund.get('debt_equity'), np.nan)
        current_ratio = safe_number(fund.get('current_ratio'), np.nan)
        cash_to_debt = safe_number(fund.get('cash_to_debt'), np.nan)
        ocf = safe_number(fund.get('operating_cash_flow'), np.nan)
        fcf = safe_number(fund.get('free_cash_flow'), np.nan)
        peg = safe_number(fund.get('peg_ratio'), np.nan)
        fcf_yield = safe_number(fund.get('fcf_yield'), np.nan)
        market_cap = safe_number(fund.get('market_cap'), np.nan)
        fundamental_reliability = safe_text(fund.get('fundamental_reliability')).upper() or 'UNKNOWN'
        fundamental_data_grade = safe_text(fund.get('fundamental_data_grade')).upper() or 'D'
        fundamental_score_10 = safe_number(fund.get('fundamental_score_10'), safe_number(fund.get('fundamental_score'), np.nan) / 10.0)
        source_count = int(safe_number(fund.get('fundamental_source_count'), 0.0))
        source_families = safe_text(fund.get('fundamental_source_families'))
        history_quarters = int(safe_number(fund.get('fundamental_history_quarters'), 0.0))
        history_years = int(safe_number(fund.get('fundamental_history_years'), 0.0))
        history_coverage = safe_number(fund.get('fundamental_history_coverage'), 0.0)
        consensus_score = safe_number(fund.get('fundamental_consensus_score'), np.nan)
        fundamental_conflicts = safe_text(fund.get('fundamental_conflicts'))
        official_reference = truthy(fund.get('fundamental_official_reference', False))
        official_verified = truthy(fund.get('fundamental_official_verified', False))
        cash_conversion = safe_number(fund.get('history_cash_conversion'), np.nan)
        positive_ocf_ratio = safe_number(fund.get('history_positive_ocf_ratio'), np.nan)
        positive_earnings_ratio = safe_number(fund.get('history_positive_earnings_ratio'), np.nan)
        margin_stability = safe_number(fund.get('history_margin_stability'), np.nan)
        share_dilution = safe_number(fund.get('history_share_dilution_yoy'), np.nan)
        roic_proxy = safe_number(fund.get('history_roic_proxy'), np.nan)
        net_debt_ebitda = safe_number(fund.get('history_net_debt_ebitda'), np.nan)
        interest_coverage = safe_number(fund.get('history_interest_coverage'), np.nan)
        statement_age_days = safe_number(fund.get('statement_age_days'), np.nan)
        fundamental_model = safe_text(fund.get('fundamental_model')) or 'GENERAL'
        is_financial = fundamental_model == 'FINANCIAL'
        red_flags = safe_text(fund.get('fundamental_red_flags'))
        roc60 = safe_number(row.get('ROC60'), -1.0)
        roc120 = safe_number(row.get('ROC120'), -1.0)
        rs60 = safe_number(row.get('REL_STRENGTH60'), -1.0)
        dist_high = safe_number(row.get('DIST_52W_HIGH'), -1.0)
        cmf_v = safe_number(row.get('CMF20'), -1.0)
        obv_up = safe_number(row.get('OBV_SLOPE10'), -1.0) > 0
        accumulation, up_down = silent_accumulation_metrics(frame)
        growth_score = 0.0
        growth_score += 11.0 if revenue_growth >= 0.2 else 7.0 if revenue_growth >= 0.1 else 3.0 if revenue_growth >= 0 else 0.0
        growth_score += 11.0 if earnings_growth >= 0.25 else 7.0 if earnings_growth >= 0.12 else 3.0 if earnings_growth >= 0 else 0.0
        profitability_score = 0.0
        profitability_score += 7.0 if roe >= 0.2 else 4.5 if roe >= 0.12 else 2.0 if roe >= 0.08 else 0.0
        profitability_score += 3.0 if roa >= 0.08 else 1.5 if roa >= 0.04 else 0.0
        profitability_score += 4.0 if operating_margin >= 0.15 else 2.0 if operating_margin >= 0.08 else 0.0
        profitability_score += 4.0 if net_margin >= 0.12 else 2.0 if net_margin >= 0.06 else 0.0
        earnings_quality_score = 0.0
        earnings_quality_score += 5.0 if 0.8 <= cash_conversion <= 1.8 else 3.0 if cash_conversion >= 0.6 else 0.0
        earnings_quality_score += 4.0 if np.isfinite(fcf) and fcf > 0 else 0.0
        earnings_quality_score += 4.0 if positive_ocf_ratio >= 0.875 else 2.0 if positive_ocf_ratio >= 0.625 else 0.0
        earnings_quality_score += 3.0 if positive_earnings_ratio >= 0.875 else 1.5 if positive_earnings_ratio >= 0.625 else 0.0
        earnings_quality_score += 2.0 if np.isfinite(share_dilution) and share_dilution <= 0.02 else 1.0 if np.isfinite(share_dilution) and share_dilution <= 0.05 else 0.0
        balance_score = 0.0
        if not is_financial:
            balance_score += 4.0 if np.isfinite(debt_equity) and debt_equity <= 0.8 else 2.0 if np.isfinite(debt_equity) and debt_equity <= 1.5 else 0.0
            balance_score += 2.0 if np.isfinite(current_ratio) and current_ratio >= 1.5 else 1.0 if np.isfinite(current_ratio) and current_ratio >= 1.0 else 0.0
            balance_score += 2.0 if np.isfinite(cash_to_debt) and cash_to_debt >= 0.5 else 1.0 if np.isfinite(cash_to_debt) and cash_to_debt >= 0.2 else 0.0
            balance_score += 2.0 if np.isfinite(net_debt_ebitda) and net_debt_ebitda <= 1.5 else 1.0 if np.isfinite(net_debt_ebitda) and net_debt_ebitda <= 3.0 else 0.0
            balance_score += 2.0 if np.isfinite(interest_coverage) and interest_coverage >= 6.0 else 1.0 if np.isfinite(interest_coverage) and interest_coverage >= 3.0 else 0.0
        solvency_fields = (debt_equity, current_ratio, cash_to_debt)
        solvency_coverage = 100.0 * sum((np.isfinite(value) for value in solvency_fields)) / len(solvency_fields)
        valuation_score = 0.0
        valuation_score += 4.0 if np.isfinite(peg) and 0 < peg <= 1.5 else 2.0 if np.isfinite(peg) and 1.5 < peg <= 2.5 else 0.0
        valuation_score += 4.0 if np.isfinite(fcf_yield) and fcf_yield >= 0.04 else 2.0 if np.isfinite(fcf_yield) and fcf_yield > 0 else 0.0
        momentum_score = 0.0
        momentum_score += 4.0 if roc60 >= 0.15 else 2.5 if roc60 >= 0.07 else 0.0
        momentum_score += 3.0 if roc120 >= 0.25 else 2.0 if roc120 >= 0.12 else 0.0
        momentum_score += 2.0 if rs60 > 0 else 0.0
        momentum_score += 2.0 if dist_high >= -0.15 else 1.0 if dist_high >= -0.3 else 0.0
        momentum_score += 1.0 if close > safe_number(row.get('EMA200'), float('inf')) else 0.0
        accumulation_score = 0.0
        accumulation_score += 5.0 if accumulation >= 80 else 3.0 if accumulation >= 65 else 0.0
        accumulation_score += 3.0 if cmf_v > 0 and obv_up else 0.0
        accumulation_score += 2.0 if adtv >= 3000000000 else 1.0 if adtv >= 1000000000 else 0.0
        total = growth_score + profitability_score + earnings_quality_score + balance_score + valuation_score + momentum_score + accumulation_score
        severe_flags = any((flag in red_flags for flag in ('Margin bersih negatif', 'OCF negatif', 'DER tinggi')))
        severe_flags = bool(severe_flags or fundamental_conflicts or (np.isfinite(share_dilution) and share_dilution > 0.12))
        if severe_flags:
            total = min(total, 69.0)
        if np.isfinite(market_cap) and market_cap < 300000000000:
            total -= 4.0
        total = max(0.0, min(100.0, total))
        if total < 60:
            continue
        statement_current = bool(np.isfinite(statement_age_days) and 0 <= statement_age_days <= cfg.max_statement_age_days)
        car_raw = safe_number(fund.get('history_car'), np.nan)
        npl_raw = safe_number(fund.get('history_npl_gross'), np.nan)
        ldr_raw = safe_number(fund.get('history_ldr'), np.nan)
        car = car_raw / 100.0 if car_raw > 1.5 else car_raw
        npl_gross = npl_raw / 100.0 if npl_raw > 1.0 else npl_raw
        ldr = ldr_raw / 100.0 if ldr_raw > 2.0 else ldr_raw
        bank_prudential_gate = bool(
            is_financial and np.isfinite(car) and np.isfinite(npl_gross) and np.isfinite(ldr)
            and car >= 0.12 and npl_gross <= 0.03 and 0.65 <= ldr <= 1.00
        )
        general_solvency_gate = bool((not is_financial) and solvency_coverage >= 66.0)
        fundamental_data_gate = bool(
            coverage >= 70.0 and fundamental_data_grade in {'A', 'B'}
            and fundamental_reliability == 'HIGH' and statement_current
            and history_coverage >= 55.0
        )
        grade_a_gate = bool(
            fundamental_data_grade == 'A' and source_count >= 2
            and official_verified
            and (history_quarters >= 8 or history_years >= 3)
            and np.isfinite(consensus_score) and consensus_score >= 75.0
            and not fundamental_conflicts
        )
        if total >= 82 and fundamental_score_10 >= 8.0 and fundamental_data_gate and grade_a_gate and (adtv >= 1500000000) and (not severe_flags) and (general_solvency_gate or bank_prudential_gate):
            status = 'MULTIBAGGER_A_CANDIDATE'
        elif total >= 72 and fundamental_score_10 >= 7.0 and fundamental_data_grade in {'A', 'B', 'C'} and coverage >= 60 and (not severe_flags):
            status = 'MULTIBAGGER_B_CANDIDATE'
        else:
            status = 'MULTIBAGGER_WATCHLIST'
        sig = signal_map.get(ticker, {})
        technical_entry_state = safe_text(sig.get('status')) or 'NO_ACTIVE_ENTRY_SETUP'
        actionable_entry_states = {
            'EXECUTION_READY', 'READY_FOR_STOCKBIT_VERIFY',
            'ENTRY_PLAN_READY', 'READY_FOR_PRICE_VERIFY',
        }
        if status == 'MULTIBAGGER_A_CANDIDATE' and technical_entry_state in actionable_entry_states:
            compounding_state = 'ACCUMULATE_NOW'
            review_action = 'ADD_GRADUALLY_WITH_CORE_ENTRY_PLAN'
        elif status == 'MULTIBAGGER_A_CANDIDATE':
            compounding_state = 'WAIT_ACCUMULATION_ZONE'
            review_action = 'KEEP_REALIZED_PROFIT_AS_COMPOUNDING_CASH'
        elif status == 'MULTIBAGGER_B_CANDIDATE' and technical_entry_state in actionable_entry_states and total >= 78:
            compounding_state = 'STARTER_NOW'
            review_action = 'OPEN_SMALL_STARTER_ONLY; ADD AFTER QUARTERLY CONFIRMATION'
        elif status == 'MULTIBAGGER_B_CANDIDATE':
            compounding_state = 'RESEARCH_AND_WAIT'
            review_action = 'VERIFY_QUARTERLY_TREND_BEFORE_ADDING'
        else:
            compounding_state = 'RESEARCH_ONLY'
            review_action = 'NO_COMPOUNDING_ALLOCATION'
        max_allocation = min(cfg.max_position_pct, 0.20 if status == 'MULTIBAGGER_A_CANDIDATE' else 0.12 if status == 'MULTIBAGGER_B_CANDIDATE' else 0.0)
        rows.append({'ticker': ticker, 'multibagger_status': status, 'multibagger_score': round(total, 1), 'growth_score': round(growth_score, 1), 'profitability_score': round(profitability_score, 1), 'earnings_quality_score': round(earnings_quality_score, 1), 'balance_sheet_score': round(balance_score, 1), 'valuation_score': round(valuation_score, 1), 'momentum_score': round(momentum_score, 1), 'accumulation_score': round(accumulation_score, 1), 'fundamental_coverage': coverage, 'fundamental_score': fund.get('fundamental_score'), 'fundamental_score_10': fundamental_score_10, 'fundamental_reliability': fundamental_reliability, 'fundamental_data_grade': fundamental_data_grade, 'fundamental_source_count': source_count, 'fundamental_source_families': source_families, 'fundamental_history_quarters': history_quarters, 'fundamental_history_years': history_years, 'fundamental_history_coverage': history_coverage, 'fundamental_consensus_score': consensus_score, 'fundamental_conflicts': fundamental_conflicts, 'fundamental_official_reference': official_reference, 'fundamental_official_verified': official_verified, 'statement_age_days': statement_age_days, 'statement_current': statement_current, 'statement_age_state': 'CURRENT' if statement_current else 'UNKNOWN' if not np.isfinite(statement_age_days) else 'STALE', 'peg_valid_for_valuation': bool(np.isfinite(peg) and peg > 0), 'fundamental_data_gate': fundamental_data_gate, 'grade_a_gate': grade_a_gate, 'severe_fundamental_flags': severe_flags, 'revenue_growth': revenue_growth, 'earnings_growth': earnings_growth, 'roe': roe, 'roa': roa, 'net_margin': net_margin, 'debt_equity': debt_equity, 'current_ratio': current_ratio, 'cash_to_debt': cash_to_debt, 'operating_cash_flow': ocf, 'free_cash_flow': fcf, 'cash_conversion_ttm': cash_conversion, 'positive_ocf_ratio': positive_ocf_ratio, 'positive_earnings_ratio': positive_earnings_ratio, 'margin_stability': margin_stability, 'share_dilution_yoy': share_dilution, 'roic_proxy': roic_proxy, 'net_debt_ebitda': net_debt_ebitda, 'interest_coverage': interest_coverage, 'solvency_coverage': round(solvency_coverage, 1), 'fundamental_model': fundamental_model, 'car': car, 'npl_gross': npl_gross, 'ldr': ldr, 'bank_prudential_gate': bank_prudential_gate, 'peg_ratio': peg, 'fcf_yield': fcf_yield, 'market_cap': market_cap, 'last_price': close, 'roc60': roc60, 'roc120': roc120, 'relative_strength60': rs60, 'distance_52w_high': dist_high, 'silent_accumulation_score': accumulation, 'up_down_value_ratio20': up_down, 'adtv20_idr': adtv, 'active_setup': sig.get('setup', ''), 'technical_entry_state': technical_entry_state, 'entry': sig.get('entry', np.nan), 'stop_loss': sig.get('stop_loss', np.nan), 'tp1': sig.get('tp1', np.nan), 'tp2': sig.get('tp2', np.nan), 'compounding_state': compounding_state, 'review_action': review_action, 'profit_allocation_pct': 100.0 * cfg.multibagger_profit_allocation_pct, 'max_position_pct_equity': 100.0 * max_allocation, 'horizon': '12–36 months; quarterly review', 'red_flags': ' • '.join(part for part in (red_flags, fundamental_conflicts) if part), 'note': 'Bank grade A requires CAR/NPL/LDR history plus verified IDX/XBRL and multi-source consensus' if is_financial else 'Candidate ranking, not a forecast or guaranteed multiple'})
    result = pd.DataFrame(rows)
    if not result.empty:
        rank = {'MULTIBAGGER_A_CANDIDATE': 0, 'MULTIBAGGER_B_CANDIDATE': 1, 'MULTIBAGGER_WATCHLIST': 2}
        result['_rank'] = result['multibagger_status'].map(rank).fillna(9)
        result = result.sort_values(['_rank', 'multibagger_score', 'adtv20_idr'], ascending=[True, False, False]).drop(columns='_rank').reset_index(drop=True)
        result = allocate_multibagger_capital(result, cfg)
        result['_capital_rank_sort'] = pd.to_numeric(result.get('capital_priority_rank'), errors='coerce').fillna(9999)
        result = result.sort_values(['_capital_rank_sort', 'capital_conviction_score', 'multibagger_score'], ascending=[True, False, False]).drop(columns='_capital_rank_sort').reset_index(drop=True)
    return result

def build_specialty_screens(
    prepared: Mapping[str, pd.DataFrame],
    fundamentals: pd.DataFrame | None = None,
    core_signals: pd.DataFrame | None = None,
    market_context: MarketContext | None = None,
    intraday: Mapping[str, pd.DataFrame] | None = None,
    config: ScanConfig | None = None,
    now: Any | None = None,
    current_positions: int = 0,
    current_open_risk_idr: float = 0.0,
    cash_on_hand_idr: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Create specialty tables, then allocate one shared risk/cash budget."""
    cfg = config or ScanConfig()
    screens = {
        'sniper': scan_sniper_entries(core_signals, prepared, config=cfg),
        'bsjp': scan_bsjp_candidates(prepared, intraday=intraday, core_signals=core_signals, config=cfg, now=now, market_context=market_context),
        'bpjs': scan_bpjs_candidates(prepared, intraday=intraday, core_signals=core_signals, config=cfg, now=now, market_context=market_context),
        'multibagger': scan_multibagger_candidates(prepared, fundamentals, core_signals=core_signals, config=cfg),
        'ara_hunter': scan_ara_hunter_candidates(prepared, intraday=intraday, core_signals=core_signals, config=cfg, now=now, market_context=market_context),
    }
    screens = _apply_specialty_portfolio_budget(
        screens, cfg,
        current_positions=current_positions,
        current_open_risk_idr=current_open_risk_idr,
        cash_on_hand_idr=cash_on_hand_idr,
    )
    screens['daily_opportunities'] = build_daily_opportunity_board(screens)
    screens['profit_order_builder'] = build_profit_order_builder(core_signals, screens, cfg)
    return screens

def _intraday_metrics(frame: pd.DataFrame, now: Any | None=None, max_stale_minutes: int=20) -> dict[str, Any]:
    """Add a completed-session flag without making old intraday bars 'live'.

    BPJS/BSJP still require fresh bars during their trading windows. ARA Hunter,
    however, may legitimately review the completed intraday session after close
    or during the next pre-market. v4.4 incorrectly treated that completed
    session as unusable merely because it was older than 20 minutes.
    """
    metrics = _intraday_metrics_v440(frame, now=now, max_stale_minutes=max_stale_minutes)
    reference = jakarta_timestamp(now)
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
            candidates = [safe_text(row.get('critical_blockers')), safe_text(row.get('analyst_hard_blockers'))]
            if truthy(row.get('market_status_critical_blocker', False)):
                candidates.append('Status perdagangan IDX negatif')
            if truthy(row.get('quote_critical_blocker', False)):
                candidates.append('Konflik quote/OHLCV')
            for text in candidates:
                upper = text.upper()
                if text and any((token in upper for token in hard_tokens)):
                    messages.append(text)
        result[str(ticker)] = ' • '.join(dict.fromkeys(messages))
    return result

def _clip_score(value: Any) -> float:
    return float(max(0.0, min(100.0, safe_number(value, 0.0))))

def _ara_intraday_microstructure(frame: pd.DataFrame, ara_price: float, now: Any | None=None) -> dict[str, Any]:
    """Estimate aggressive buying and ARA-lock quality from intraday OHLCV.

    These are transparent proxies.  They are not broker identity, queue lots,
    or exchange market depth.  The function intentionally labels them as proxy
    fields so the dashboard cannot misrepresent inferred data as observed data.
    """
    session = _intraday_session(frame, asof=jakarta_timestamp(now))
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
    tick = idx_tick_size(max(safe_number(ara_price, 0.0), safe_number(close.iloc[-1], 0.0)))
    lock_tolerance = max(float(tick), safe_number(ara_price, 0.0) * 0.0015)
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
    flow_score = _clip_score(0.25 * positive_component(signed_imbalance, 0.0, 0.45) + 0.2 * positive_component(clv_pressure, 0.0, 0.65) + 0.2 * max(0.0, min(100.0, vwap_hold * 100.0)) + 0.2 * positive_component(safe_number(late_buy_imbalance, 0.0), 0.0, 0.5) + 0.15 * positive_component(directional_efficiency, 0.0, 0.6))
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
        prev_close = safe_number(close.iloc[i - 1], 0.0)
        current_close = safe_number(close.iloc[i], 0.0)
        next_close = safe_number(close.iloc[i + 1], 0.0)
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
        vol_ratio = safe_number(data.iloc[i].get('VOL_RATIO'), np.nan)
        close_location = safe_number(data.iloc[i].get('CLOSE_LOCATION'), np.nan)
        body_atr = safe_number(data.iloc[i].get('BODY_ATR'), np.nan)
        high20 = safe_number(data.iloc[i].get('HIGH20_PREV'), np.nan)
        cmf = safe_number(data.iloc[i].get('CMF20'), np.nan)
        obv = safe_number(data.iloc[i].get('OBV_SLOPE10'), np.nan)
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
    frame = read_csv_input(source)
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
    reference = jakarta_timestamp(now)
    for idx, row in out.iterrows():
        bonus = 0.0
        evidence: list[str] = []
        broksum_signal = safe_text(row.get('broksum_signal')).upper()
        if broksum_signal == 'ACCUMULATION_PROXY':
            bonus += 6.0
            evidence.append('Broker summary: akumulasi')
        elif broksum_signal == 'DISTRIBUTION_PROXY':
            bonus -= 9.0
            evidence.append('Broker summary: distribusi')
        orderbook_signal = safe_text(row.get('orderbook_signal')).upper()
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
        base = safe_number(row.get('ara_model_score'), row.get('ara_hunter_score'))
        final_score = _clip_score(base + bonus)
        out.at[idx, 'external_confirmation_bonus'] = round(bonus, 1)
        out.at[idx, 'ara_final_score'] = round(final_score, 1)
        out.at[idx, 'external_evidence'] = ' • '.join(evidence)
        status = safe_text(row.get('ara_hunter_status'))
        out.at[idx, 'external_confirmation_pass'] = False
        if status == 'ARA_CONTINUATION_SIGNAL_READY' and orderbook_fresh and (orderbook_signal == 'STRONG_BUY_QUEUE') and (broksum_signal == 'ACCUMULATION_PROXY'):
            out.at[idx, 'ara_hunter_status'] = 'ARA_CONTINUATION_FLOW_VERIFIED_SIGNAL'
            out.at[idx, 'external_confirmation_pass'] = True
            out.at[idx, 'order_instruction'] = 'RECHECK_NEXT_SESSION_PRICE_BEFORE_ANY_ORDER'
        elif status == 'ARA_CONTINUATION_CANDIDATE' and final_score >= 74 and orderbook_fresh and (orderbook_signal in {'STRONG_BUY_QUEUE', 'BUY_QUEUE'}):
            out.at[idx, 'ara_hunter_status'] = 'ARA_CONTINUATION_SIGNAL_READY'
            out.at[idx, 'signal_ready'] = True
            out.at[idx, 'specialty_order_state'] = 'SIGNAL_READY_RESCAN_REQUIRED'
        elif status == 'PRE_ARA_CANDIDATE' and final_score >= 76 and orderbook_fresh and (orderbook_signal in {'STRONG_BUY_QUEUE', 'BUY_QUEUE'}):
            out.at[idx, 'ara_hunter_status'] = 'PRE_ARA_SIGNAL_READY'
            out.at[idx, 'signal_ready'] = True
            out.at[idx, 'specialty_order_state'] = 'SIGNAL_READY_RESCAN_REQUIRED'
            out.at[idx, 'order_instruction'] = 'RERUN_SCANNER_FOR_SHARED_RISK_GATE'
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
    now_jkt = jakarta_timestamp(now)
    market_regime = market_context.regime if market_context is not None else 'NOT_EVALUATED'
    live_window = idx_regular_decision_window(now_jkt)
    rows: list[dict[str, Any]] = []
    for ticker, frame in prepared.items():
        if frame is None or frame.empty or len(frame) < 60:
            continue
        row = frame.iloc[-1]
        prev = frame.iloc[-2]
        completed_close = safe_number(row.get('Close'), 0.0)
        prior_close = safe_number(prev.get('Close'), 0.0)
        if completed_close <= 0 or prior_close <= 0:
            continue
        context_blocker = critical_map.get(ticker, '')
        im = _intraday_metrics(intraday.get(ticker, pd.DataFrame()), now_jkt, cfg.max_intraday_stale_minutes)
        session = _intraday_session(intraday.get(ticker, pd.DataFrame()), asof=now_jkt)
        live_session = bool(
            not session.empty
            and safe_text(im.get('intraday_session_date')) == now_jkt.date().isoformat()
            and now_jkt.weekday() < 5
        )
        # Daily frames intentionally exclude today's incomplete candle. During
        # a live Monday session the last daily close is therefore Friday and is
        # the correct ARA reference; the intraday price is today's observation.
        if live_session:
            reference_close = completed_close
            close = safe_number(im.get('intraday_last'), completed_close)
            session_open = safe_number(session.iloc[0].get('Open'), close)
            observed_high = safe_number(session['High'].max(), close)
            observed_ara_price = idx_daily_price_band(reference_close)[1]
            ara_price = observed_ara_price
            daily_return = close / reference_close - 1.0
            opening_gap = session_open / reference_close - 1.0
            session_volume = float(pd.to_numeric(session['Volume'], errors='coerce').fillna(0.0).sum())
            vol_ma20 = safe_number(row.get('VOL_MA20'), 0.0)
            vol_ratio = session_volume / vol_ma20 if vol_ma20 > 0 else 0.0
            typical = (session['High'] + session['Low'] + session['Close']) / 3.0
            value_today = float((typical * session['Volume']).sum())
            close_location = safe_number(im.get('session_close_location'), 0.0)
            atr_v_live = safe_number(row.get('ATR14'), completed_close * 0.03)
            body_atr = abs(close - session_open) / atr_v_live if atr_v_live > 0 else 0.0
            breakout20 = close > safe_number(row.get('HIGH20_PREV'), float('inf'))
            prediction_horizon_default = 'SESSION_NOW'
        else:
            reference_close = prior_close
            close = completed_close
            observed_high = safe_number(row.get('High'), close)
            observed_ara_price = idx_daily_price_band(reference_close)[1]
            # The candidate is for the next session, whose band references the
            # latest completed close—not the close from two sessions ago.
            ara_price = idx_daily_price_band(completed_close)[1]
            daily_return = close / reference_close - 1.0
            opening_gap = safe_number(row.get('Open'), close) / reference_close - 1.0
            vol_ratio = safe_number(row.get('VOL_RATIO'), 0.0)
            value_today = safe_number(row.get('VALUE'), 0.0)
            close_location = safe_number(row.get('CLOSE_LOCATION'), 0.0)
            body_atr = safe_number(row.get('BODY_ATR'), 0.0)
            breakout20 = close > safe_number(row.get('HIGH20_PREV'), float('inf'))
            prediction_horizon_default = 'NEXT_SESSION'
        if not ara_price or not observed_ara_price:
            continue
        room = ara_price / close - 1.0
        adtv = safe_number(row.get('ADTV20'), 0.0)
        cmf_v = safe_number(row.get('CMF20'), -1.0)
        obv_up = safe_number(row.get('OBV_SLOPE10'), -1.0) > 0
        rsi_v = safe_number(row.get('RSI14'), 50.0)
        mfi_v = safe_number(row.get('MFI14'), 50.0)
        accumulation, up_down = silent_accumulation_metrics(frame)
        micro = _ara_intraday_microstructure(intraday.get(ticker, pd.DataFrame()), observed_ara_price, now_jkt)
        hist = _ara_history_profile(frame)
        intraday_usable = bool(im.get('intraday_bars', 0) >= 6 and (truthy(im.get('intraday_fresh')) or truthy(im.get('intraday_session_complete'))))
        near_or_locked = near_upper_auto_rejection(reference_close, close, observed_high)
        ara_confirmed_today = bool(near_or_locked and daily_return >= 0.88 * idx_ara_pct(reference_close))
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
        flow_proxy = safe_number(micro.get('orderflow_proxy_score'), np.nan)
        if not np.isfinite(flow_proxy):
            flow_proxy = _clip_score(35.0 + 20.0 * max(-1.0, min(1.0, cmf_v)) + 15.0 * (1.0 if obv_up else 0.0) + 15.0 * max(0.0, min(1.0, close_location)) + 15.0 * max(0.0, min(1.0, safe_number(up_down, 1.0) / 2.0)))
        queue_proxy = safe_number(micro.get('queue_proxy_score'), np.nan)
        if not np.isfinite(queue_proxy):
            queue_proxy = _clip_score(25.0 + 45.0 * close_location + 15.0 * min(1.0, max(0.0, vol_ratio - 1.0) / 3.0))
        pre_hist = safe_number(hist.get('historical_pre_ara_hit_rate'), np.nan)
        pre_samples = int(safe_number(hist.get('historical_pre_ara_samples'), 0.0))
        # Beta shrinkage prevents one or two historical hits from dominating the score.
        pre_successes = int(round(max(0.0, min(1.0, safe_number(pre_hist, 0.0))) * pre_samples))
        pre_shrunk_rate = (pre_successes + 1.0) / (pre_samples + 5.0)
        pre_hist_score = _clip_score(25.0 + 150.0 * pre_shrunk_rate)
        session_efficiency = safe_number(micro.get('proxy_directional_efficiency'), np.nan)
        vwap_hold_ratio = safe_number(micro.get('proxy_vwap_hold_ratio'), np.nan)
        late_buy_imbalance = safe_number(micro.get('proxy_late_buy_imbalance'), np.nan)
        micro_quality = 50.0
        if intraday_usable:
            micro_quality = _clip_score(
                35.0 * max(0.0, min(1.0, safe_number(session_efficiency, 0.0)))
                + 35.0 * max(0.0, min(1.0, safe_number(vwap_hold_ratio, 0.0)))
                + 30.0 * max(0.0, min(1.0, (safe_number(late_buy_imbalance, -1.0) + 1.0) / 2.0))
            )
        pre_score = _clip_score(0.47 * daily_score + 0.25 * flow_proxy + 0.10 * queue_proxy + 0.10 * micro_quality + 0.08 * pre_hist_score)
        prior_5d_return = np.nan
        if len(frame) >= 6:
            base_close = safe_number(frame.iloc[-6].get('Close'), 0.0)
            if base_close > 0:
                prior_5d_return = close / base_close - 1.0
        continuation_daily = _clip_score(20.0 * min(1.0, close_location) + 18.0 * min(1.0, max(0.0, vol_ratio) / 4.0) + 16.0 * (1.0 if breakout20 else 0.4) + 16.0 * min(1.0, accumulation / 80.0) + 15.0 * (1.0 if body_atr >= 0.65 else 0.55) + 15.0 * (1.0 if opening_gap <= 0.1 else 0.25))
        hist_strong = safe_number(hist.get('historical_next_day_strong_rate'), np.nan)
        hist_positive = safe_number(hist.get('historical_next_day_positive_rate'), np.nan)
        hist_events = int(safe_number(hist.get('historical_ara_events'), 0.0))
        strong_successes = int(round(max(0.0, min(1.0, safe_number(hist_strong, 0.0))) * hist_events))
        positive_successes = int(round(max(0.0, min(1.0, safe_number(hist_positive, 0.0))) * hist_events))
        strong_shrunk = (strong_successes + 1.0) / (hist_events + 5.0)
        positive_shrunk = (positive_successes + 2.0) / (hist_events + 5.0)
        hist_score = _clip_score(20.0 + 55.0 * strong_shrunk + 25.0 * positive_shrunk)
        continuation_score = _clip_score(0.28 * continuation_daily + 0.27 * flow_proxy + 0.30 * queue_proxy + 0.15 * hist_score)
        exhaustion_penalty = 0.0
        if np.isfinite(prior_5d_return):
            exhaustion_penalty += 14.0 if prior_5d_return >= 0.70 else 8.0 if prior_5d_return >= 0.45 else 3.0 if prior_5d_return >= 0.30 else 0.0
        exhaustion_penalty += 10.0 if opening_gap > 0.15 else 5.0 if opening_gap > 0.10 else 0.0
        exhaustion_penalty += 6.0 if close_location < 0.78 else 0.0
        if intraday_usable and safe_number(micro.get('proxy_max_unlock_pct'), 0.0) < -0.02:
            exhaustion_penalty += 8.0
        pre_score = max(0.0, pre_score - 0.55 * exhaustion_penalty)
        continuation_score = max(0.0, continuation_score - exhaustion_penalty)
        daily_valid = bool(
            value_today >= 750000000.0 and vol_ratio >= 1.25 and close_location >= 0.65
            and body_atr >= 0.4 and (cmf_v > -0.03 or obv_up)
            and (breakout20 or daily_return >= 0.025) and 0.008 <= daily_return <= 0.28
            and opening_gap <= 0.15 and not context_blocker
            and (not intraday_usable or safe_number(micro.get('proxy_directional_efficiency'), 0.0) >= -0.05)
        )
        warnings: list[str] = []
        if market_regime in {'RISK_OFF', 'UNKNOWN', 'NOT_EVALUATED'}:
            warnings.append(f'Regime {market_regime}; ORDER_READY diblokir')
        if not intraday_usable:
            warnings.append('Tanpa intraday usable; orderflow/queue memakai proxy daily berkonfidensi lebih rendah')
        warnings.append('Orderflow proxy bukan broker summary; queue proxy bukan antrean bid aktual')
        if ara_confirmed_today and (not context_blocker):
            ara_model = 'ARA_CONTINUATION'
            prediction_horizon = 'NEXT_SESSION'
            model_score = continuation_score
            if continuation_score >= 74.0 and (queue_proxy >= 58.0 or flow_proxy >= 72.0):
                status = 'ARA_CONTINUATION_SIGNAL_READY'
                instruction = 'VERIFY_NEXT_SESSION_OPENING_QUEUE'
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
            prediction_horizon = prediction_horizon_default if live_window and truthy(im.get('intraday_fresh')) else 'NEXT_SESSION'
            model_score = pre_score
            if pre_score >= 76.0 and flow_proxy >= 62.0:
                status = 'PRE_ARA_SIGNAL_READY'
                instruction = 'WAIT_SHARED_RISK_AND_PRICE_GATE'
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
                status = 'PRE_ARA_DAILY_RADAR'
                instruction = 'WATCH_FOR_VOLUME_BREAKOUT_AND_VWAP'
                signal_valid = False
        else:
            ara_model = 'PRE_ARA'
            prediction_horizon = prediction_horizon_default
            model_score = pre_score
            status = 'PRE_ARA_DAILY_RADAR'
            instruction = 'WATCH_FOR_VOLUME_BREAKOUT_AND_VWAP'
            signal_valid = False
        atr_v = safe_number(row.get('ATR14'), close * 0.03)
        entry = round_idx_price(close, 'up')
        stop = round_idx_price(entry - max(1.0 * atr_v, 3 * idx_tick_size(entry)), 'down')
        if stop >= entry:
            stop = round_idx_price(entry - 3 * idx_tick_size(entry), 'down')
        risk = max(entry - stop, idx_tick_size(entry))
        target_ceiling = ara_price if ara_model == 'PRE_ARA' else idx_daily_price_band(close)[1]
        recent_range = max(
            safe_number(row.get('High'), close) - safe_number(row.get('Low'), close),
            safe_number(frame['High'].tail(min(10, len(frame))).max(), close) - safe_number(frame['Low'].tail(min(10, len(frame))).min(), close),
            idx_tick_size(entry),
        )
        targets = price_structure_target_pair(
            frame, entry, setup=ara_model,
            explicit_levels=[
                (safe_number(row.get('HIGH20_PREV'), np.nan), 'PRIOR_20D_HIGH'),
                (safe_number(row.get('HIGH55_PREV'), np.nan), 'PRIOR_55D_HIGH'),
                (safe_number(row.get('HIGH252'), np.nan), 'PRIOR_52W_HIGH'),
                (target_ceiling, 'CURRENT_SESSION_ARA_LIMIT' if ara_model == 'PRE_ARA' and prediction_horizon == 'SESSION_NOW' else 'NEXT_SESSION_ARA_LIMIT'),
            ],
            projection_origin=max(entry, safe_number(row.get('HIGH20_PREV'), entry)),
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
            if status in {'PRE_ARA_SIGNAL_READY', 'ARA_CONTINUATION_SIGNAL_READY'}:
                status = 'PRE_ARA_WATCHLIST' if ara_model == 'PRE_ARA' else 'ARA_CONTINUATION_CANDIDATE'
                instruction = 'WATCH_ONLY_TARGETS_INVALID'
                signal_valid = False
        if ara_model == 'ARA_CONTINUATION':
            warnings.append('Target dan RR continuation wajib dihitung ulang terhadap harga pembukaan sesi berikutnya')
        if room < 0.02 and ara_model == 'PRE_ARA':
            warnings.append('Ruang ke ARA sempit; hindari market chase')
        gate = _specialty_prebudget_gate(
            mode='ARA',
            signal_ready=bool(status == 'PRE_ARA_SIGNAL_READY' and signal_valid and target_valid),
            in_window=bool(ara_model == 'PRE_ARA' and prediction_horizon == 'SESSION_NOW' and live_window),
            intraday_fresh=truthy(im.get('intraday_fresh')),
            requires_intraday=True,
            entry=entry,
            stop=stop,
            tp1=ara_tp1,
            tp2=ara_tp2,
            rr1=rr1,
            rr2=rr2,
            risk_pct=risk_pct,
            adtv=adtv,
            target_valid=target_valid,
            market_regime=market_regime,
            context_blocker=context_blocker,
            cfg=cfg,
            risk_fraction=min(cfg.specialty_risk_per_trade_pct, 0.0025),
            position_fraction=min(cfg.specialty_max_position_pct, 0.15),
        )
        payload = {'ticker': ticker, 'ara_hunter_status': status, 'ara_model': ara_model, 'prediction_horizon': prediction_horizon, 'signal_valid': signal_valid, 'ara_model_score': round(model_score, 1), 'ara_hunter_score': round(model_score, 1), 'ara_edge_score': round(model_score, 1), 'daily_momentum_score': round(daily_score, 1), 'microstructure_quality_score': round(micro_quality, 1), 'historical_shrunk_rate': round(pre_shrunk_rate if ara_model == 'PRE_ARA' else strong_shrunk, 4), 'exhaustion_penalty': round(exhaustion_penalty, 1), 'orderflow_proxy_score': round(flow_proxy, 1), 'queue_proxy_score': round(queue_proxy, 1), 'proxy_confidence': 'HIGH' if intraday_usable else 'MEDIUM_LOW', 'last_price': close, 'last_completed_close': completed_close, 'previous_close': reference_close, 'price_band_reference': reference_close if live_session else completed_close, 'live_price_source': 'INTRADAY_COMPLETED_BAR' if live_session else 'FINAL_DAILY_BAR', 'ara_price': ara_price, 'room_to_ara_pct': room, 'daily_return_pct': daily_return, 'opening_gap_pct': opening_gap, 'entry_reference': entry, 'hard_stop': stop, 'ara_tp1': ara_tp1, 'ara_tp2': ara_tp2, 'tp1_basis': targets['tp1_basis'], 'tp2_basis': targets['tp2_basis'], 'target_model': 'PRICE_STRUCTURE_ONLY', 'target_structure': targets['target_structure'], 'target_structure_valid': target_valid, 'target_recalc_required': ara_model == 'ARA_CONTINUATION', 'rr1': rr1, 'rr2': rr2, 'risk_pct': risk_pct, 'volume_ratio': vol_ratio, 'value_today_idr': value_today, 'adtv20_idr': adtv, 'close_location': close_location, 'body_atr': body_atr, 'cmf20': cmf_v, 'rsi14': rsi_v, 'mfi14': mfi_v, 'breakout20': breakout20, 'silent_accumulation_score': accumulation, 'up_down_value_ratio20': up_down, 'intraday_session_complete': truthy(im.get('intraday_session_complete')), 'intraday_data_state': im.get('intraday_data_state'), 'intraday_session_date': im.get('intraday_session_date'), 'intraday_age_minutes': im.get('intraday_age_minutes'), 'market_regime': market_regime, 'late_volume_acceleration': im.get('late_volume_acceleration'), 'session_close_location': im.get('session_close_location'), 'order_instruction': instruction, 'risk_class': risk_grade, 'warnings': ' • '.join(warnings), 'blockers': context_blocker, 'observed_broker_summary': False, 'observed_orderbook': False, **gate}
        payload.update(micro)
        payload['orderflow_proxy_score'] = round(flow_proxy, 1)
        payload['queue_proxy_score'] = round(queue_proxy, 1)
        payload.update(hist)
        rows.append(payload)
    result = pd.DataFrame(rows)
    if not result.empty:
        rank = {'ARA_CONTINUATION_ORDER_READY': 0, 'PRE_ARA_ORDER_READY': 0, 'PRE_ARA_SIGNAL_READY': 1, 'ARA_CONTINUATION_SIGNAL_READY': 2, 'PRE_ARA_CANDIDATE': 3, 'ARA_CONTINUATION_CANDIDATE': 4, 'ARA_CONFIRMED_ONLY': 5, 'PRE_ARA_WATCHLIST': 6, 'PRE_ARA_DAILY_RADAR': 7}
        result['_rank'] = result['ara_hunter_status'].map(rank).fillna(9)
        ready_count = int(result['signal_ready'].map(truthy).sum()) if 'signal_ready' in result else 0
        limit = max(int(cfg.daily_radar_limit), ready_count)
        result = result.sort_values(['_rank', 'ara_model_score', 'value_today_idr'], ascending=[True, False, False]).head(limit).drop(columns='_rank').reset_index(drop=True)
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
            source_tier = safe_text(row.get('ohlcv_source_tier')).upper()
            if source_tier == 'UNAVAILABLE':
                messages.append('OHLCV daily tidak tersedia')
            if truthy(row.get('market_status_critical_blocker', False)):
                messages.append('Suspensi/FCA/status perdagangan negatif')
            if truthy(row.get('quote_critical_blocker', False)):
                messages.append('Konflik quote/OHLCV')
            for item in pipe_parts(row.get('critical_blockers')):
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


def _session_quality_features(session: pd.DataFrame, im: Mapping[str, Any]) -> dict[str, float | bool]:
    """Return transparent intraday continuation features from completed bars."""
    defaults: dict[str, float | bool] = {
        'vwap_slope_pct': np.nan,
        'directional_efficiency': np.nan,
        'breakout_volume_ratio': np.nan,
        'late_volume_share': np.nan,
        'u_shape_volume_ratio': np.nan,
        'upper_wick_ratio': np.nan,
        'afternoon_breakout': False,
        'orb_breakout_confirmed': False,
        'orb_hold_confirmed': False,
        'orb_extension_ratio': np.nan,
    }
    if session is None or session.empty or len(session) < 2:
        return defaults
    volume = pd.to_numeric(session['Volume'], errors='coerce').fillna(0.0).clip(lower=0.0)
    close = pd.to_numeric(session['Close'], errors='coerce')
    open_ = pd.to_numeric(session['Open'], errors='coerce')
    high = pd.to_numeric(session['High'], errors='coerce')
    low = pd.to_numeric(session['Low'], errors='coerce')
    typical = (high + low + close) / 3.0
    cum_volume = volume.cumsum().replace(0.0, np.nan)
    cumulative_vwap = (typical * volume).cumsum() / cum_volume
    vwap_slope_pct = np.nan
    if cumulative_vwap.dropna().size >= 3:
        base = safe_number(cumulative_vwap.dropna().iloc[-3], np.nan)
        last = safe_number(cumulative_vwap.dropna().iloc[-1], np.nan)
        if np.isfinite(base) and base > 0 and np.isfinite(last):
            vwap_slope_pct = last / base - 1.0
    gross = float(close.diff().abs().sum())
    net = float(close.iloc[-1] - close.iloc[0])
    efficiency = net / gross if gross > 0 else 0.0
    interval = max(1.0, safe_number(im.get('intraday_interval_minutes'), 5.0))
    orb_n = max(1, int(safe_number(im.get('opening_range_bars'), np.ceil(15.0 / interval))))
    orb_n = min(orb_n, len(session))
    orb_high = safe_number(im.get('orb_high'), safe_number(high.head(orb_n).max(), np.nan))
    orb_low = safe_number(im.get('orb_low'), safe_number(low.head(orb_n).min(), np.nan))
    post = session.iloc[orb_n:]
    breakout_confirmed = False
    breakout_volume_ratio = np.nan
    if not post.empty and np.isfinite(orb_high):
        break_mask = pd.to_numeric(post['Close'], errors='coerce') > orb_high
        breakout_confirmed = bool(break_mask.any())
        if breakout_confirmed:
            break_idx = break_mask[break_mask].index[0]
            break_volume = safe_number(post.loc[break_idx, 'Volume'], 0.0)
            baseline = safe_number(volume.head(orb_n).median(), 0.0)
            if baseline <= 0:
                baseline = safe_number(volume.median(), 0.0)
            breakout_volume_ratio = break_volume / baseline if baseline > 0 else np.nan
    orb_width = max(0.0, orb_high - orb_low) if np.isfinite(orb_high) and np.isfinite(orb_low) else np.nan
    extension = (safe_number(close.iloc[-1], np.nan) - orb_high) / orb_width if np.isfinite(orb_width) and orb_width > 0 else np.nan
    hold = bool(np.isfinite(orb_high) and safe_number(close.iloc[-1], -np.inf) >= orb_high - 0.10 * max(orb_width, idx_tick_size(orb_high)))
    late_n = max(2, int(np.ceil(len(session) * 0.25)))
    early_n = max(1, min(orb_n, late_n))
    late_share = float(volume.tail(late_n).sum() / volume.sum()) if volume.sum() > 0 else np.nan
    middle = volume.iloc[early_n:-late_n] if len(volume) > early_n + late_n else pd.Series(dtype=float)
    middle_mean = safe_number(middle.mean(), np.nan)
    edge_mean = safe_number(pd.concat([volume.head(early_n), volume.tail(late_n)]).mean(), np.nan)
    u_shape = edge_mean / middle_mean if np.isfinite(middle_mean) and middle_mean > 0 else np.nan
    afternoon_breakout = False
    if len(session) >= 7:
        prior_high = safe_number(high.iloc[-7:-2].max(), np.nan)
        afternoon_breakout = bool(np.isfinite(prior_high) and safe_number(close.iloc[-1], 0.0) > prior_high)
    candle_range = max(0.0, safe_number(high.iloc[-1], 0.0) - safe_number(low.iloc[-1], 0.0))
    upper_wick = max(0.0, safe_number(high.iloc[-1], 0.0) - max(safe_number(open_.iloc[-1], 0.0), safe_number(close.iloc[-1], 0.0)))
    upper_wick_ratio = upper_wick / candle_range if candle_range > 0 else 0.0
    defaults.update({
        'vwap_slope_pct': vwap_slope_pct,
        'directional_efficiency': float(max(-1.0, min(1.0, efficiency))),
        'breakout_volume_ratio': breakout_volume_ratio,
        'late_volume_share': late_share,
        'u_shape_volume_ratio': u_shape,
        'upper_wick_ratio': upper_wick_ratio,
        'afternoon_breakout': afternoon_breakout,
        'orb_breakout_confirmed': breakout_confirmed,
        'orb_hold_confirmed': hold,
        'orb_extension_ratio': extension,
    })
    return defaults

def scan_bsjp_candidates(prepared: Mapping[str, pd.DataFrame], intraday: Mapping[str, pd.DataFrame] | None=None, core_signals: pd.DataFrame | None=None, config: ScanConfig | None=None, now: Any | None=None, market_context: MarketContext | None=None) -> pd.DataFrame:
    """Rank late-session candidates for an overnight momentum hold.

    The edge model requires price acceptance above VWAP, constructive closing
    location and late demand.  Daily trend/regime/risk remain disclosures in
    SIGNAL_FIRST; stale data and invalid price structure remain hard blockers.
    """
    cfg = config or ScanConfig()
    intraday = intraday or {}
    critical_map = _critical_context_map(core_signals)
    now_jkt = jakarta_timestamp(now)
    market_regime = market_context.regime if market_context is not None else 'NOT_EVALUATED'
    minute = now_jkt.hour * 60 + now_jkt.minute
    in_window = now_jkt.weekday() < 5 and 14 * 60 + 30 <= minute <= 15 * 60 + 49
    rows: list[dict[str, Any]] = []
    for ticker, frame in prepared.items():
        if frame is None or frame.empty or len(frame) < 60:
            continue
        row, prev = frame.iloc[-1], frame.iloc[-2]
        close, prev_close = safe_number(row.get('Close'), 0.0), safe_number(prev.get('Close'), 0.0)
        if close <= 0 or prev_close <= 0:
            continue
        daily_return = close / prev_close - 1.0
        adtv = safe_number(row.get('ADTV20'), 0.0)
        vol_ratio = safe_number(row.get('VOL_RATIO'), 0.0)
        cmf_v = safe_number(row.get('CMF20'), -1.0)
        obv_up = safe_number(row.get('OBV_SLOPE10'), -1.0) > 0
        close_location = safe_number(row.get('CLOSE_LOCATION'), 0.0)
        rs_v = safe_number(row.get('REL_STRENGTH60'), -1.0)
        rsi_v = safe_number(row.get('RSI14'), 50.0)
        atr_v = safe_number(row.get('ATR14'), 0.0)
        trend = bool(close >= safe_number(row.get('EMA20'), close) and safe_number(row.get('EMA20'), 0.0) >= safe_number(row.get('EMA50'), 0.0))
        ara_price = idx_daily_price_band(close)[1]
        im = _intraday_metrics(intraday.get(ticker, pd.DataFrame()), now_jkt, cfg.max_intraday_stale_minutes)
        session = _intraday_session(intraday.get(ticker, pd.DataFrame()), asof=now_jkt)
        has_any_intraday = im['intraday_bars'] > 0
        has_intraday = im['intraday_bars'] >= 6 and truthy(im['intraday_fresh'])
        intraday_last = safe_number(im.get('intraday_last'), close)
        room_to_ara = ara_price / intraday_last - 1.0 if ara_price and intraday_last > 0 else np.nan
        above_vwap = bool(has_intraday and intraday_last >= safe_number(im.get('session_vwap'), float('inf')))
        late_accel = safe_number(im.get('late_volume_acceleration'), 0.0)
        location = safe_number(im.get('session_close_location'), 0.0)
        positive_session = safe_number(im.get('intraday_return'), -1.0) >= -0.002
        session_q = _session_quality_features(session, im)
        vwap_rising = safe_number(session_q.get('vwap_slope_pct'), -1.0) >= -0.0005
        efficiency = safe_number(session_q.get('directional_efficiency'), -1.0)
        afternoon_breakout = truthy(session_q.get('afternoon_breakout'))
        upper_wick_ok = safe_number(session_q.get('upper_wick_ratio'), 1.0) <= 0.35
        demand_pattern = bool(late_accel >= 1.10 or afternoon_breakout or safe_number(session_q.get('late_volume_share'), 0.0) >= 0.30)
        acceptance = bool(above_vwap and location >= 0.65 and positive_session and vwap_rising)

        daily_components = {
            'Trend harian belum mendukung': trend,
            'Volume harian belum menguat': vol_ratio >= 0.85,
            'CMF/OBV belum mendukung': cmf_v >= -0.03 or obv_up,
            'Close harian tidak dekat high': close_location >= 0.55,
            'Relative strength lemah': rs_v >= -0.03,
            'RSI di luar zona momentum': 42 <= rsi_v <= 82,
            'Return harian terlalu ekstrem': -0.03 <= daily_return <= 0.12,
        }
        session_components = {
            'Harga belum diterima di atas VWAP': above_vwap,
            'VWAP belum menanjak': vwap_rising,
            'Late demand belum muncul': demand_pattern,
            'Close sesi belum dekat high': location >= 0.65,
            'Directional efficiency rendah': efficiency >= 0.25,
            'Upper wick akhir terlalu besar': upper_wick_ok,
        }
        daily_hits = sum(bool(v) for v in daily_components.values())
        session_hits = sum(bool(v) for v in session_components.values())
        score = 0.0
        score += 10.0 if trend else 4.0
        score += 10.0 if vol_ratio >= 1.4 else 7.0 if vol_ratio >= 1.0 else 3.0 if vol_ratio >= 0.85 else 0.0
        score += 10.0 if cmf_v >= 0.08 and obv_up else 7.0 if cmf_v >= -0.03 or obv_up else 0.0
        score += 8.0 if rs_v > 0 else 4.0 if rs_v >= -0.03 else 0.0
        score += 7.0 if 0.003 <= daily_return <= 0.07 else 3.0 if -0.03 <= daily_return <= 0.12 else 0.0
        score += 15.0 if above_vwap else 0.0
        score += 8.0 if vwap_rising else 2.0
        score += 12.0 if late_accel >= 1.35 else 9.0 if demand_pattern else 0.0
        score += 10.0 if location >= 0.80 else 7.0 if location >= 0.65 else 0.0
        score += 7.0 if efficiency >= 0.55 else 4.0 if efficiency >= 0.25 else 0.0
        score += 3.0 if afternoon_breakout else 0.0
        score = min(100.0, score)

        context_blocker = critical_map.get(ticker, '')
        entry = round_idx_price(intraday_last if has_intraday else close, 'nearest')
        if has_intraday and not session.empty:
            last_n = min(6, len(session))
            recent_low = safe_number(session['Low'].tail(last_n).min(), entry)
            vwap_support = safe_number(im.get('session_vwap'), recent_low)
            structural_support = max(recent_low, vwap_support - 0.15 * max(atr_v, idx_tick_size(entry)))
            raw_stop = structural_support - idx_tick_size(structural_support)
        else:
            raw_stop = entry - max(0.035 * entry, 1.0 * atr_v, 3 * idx_tick_size(entry))
        stop = round_idx_price(raw_stop, 'down')
        if stop is None or stop >= entry:
            stop = round_idx_price(entry - max(3 * idx_tick_size(entry), 0.025 * entry), 'down')
        risk = max(entry - stop, idx_tick_size(entry))
        session_high = safe_number(session['High'].max(), entry) if not session.empty else safe_number(row.get('High'), entry)
        session_low = safe_number(session['Low'].min(), stop) if not session.empty else safe_number(row.get('Low'), stop)
        next_session_ara = idx_daily_price_band(max(intraday_last, close))[1]
        targets = price_structure_target_pair(
            frame, entry, setup='BSJP',
            explicit_levels=[
                (session_high, 'CURRENT_SESSION_HIGH'),
                (safe_number(row.get('High'), np.nan), 'LAST_COMPLETED_DAILY_HIGH'),
                (safe_number(row.get('HIGH20_PREV'), np.nan), 'PRIOR_20D_HIGH'),
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
        setup_valid = bool(
            in_window and has_intraday and acceptance and demand_pattern and session_hits >= 4
            and daily_hits >= 4 and score >= 70.0 and target_valid and not context_blocker
        )
        if setup_valid:
            status, action = 'BSJP_SIGNAL_READY', 'WAIT_SHARED_RISK_AND_PRICE_GATE'
        elif not in_window:
            status, action = 'BSJP_DAILY_RADAR', 'RUN_AGAIN_14_30_15_49_WIB'
        elif not has_any_intraday:
            status, action = 'BSJP_DATA_UNAVAILABLE', 'RETRY_INTRADAY_5M'
        elif not truthy(im['intraday_fresh']):
            status, action = 'BSJP_STALE_INTRADAY', 'REFRESH_INTRADAY_DATA'
        elif not has_intraday:
            status, action = 'BSJP_WAIT_SESSION_BARS', 'RUN_AGAIN_AFTER_MORE_5M_BARS'
        else:
            status, action = 'BSJP_WATCHLIST', 'WAIT_LATE_SESSION_ACCEPTANCE'
        warnings = _specialty_risk_warnings(market_regime=market_regime, adtv=adtv, cfg=cfg, stop_pct=stop_pct, room_to_ara=room_to_ara, conditions={**daily_components, **session_components})
        if not target_valid:
            warnings.append('Dua target struktur harga belum tersedia; RR tidak difabrikasi')
        gate = _specialty_prebudget_gate(
            mode='BSJP', signal_ready=setup_valid, in_window=in_window,
            intraday_fresh=truthy(im.get('intraday_fresh')), requires_intraday=True,
            entry=entry, stop=stop, tp1=tp1, tp2=tp2, rr1=rr1, rr2=rr2,
            risk_pct=stop_pct, adtv=adtv, target_valid=target_valid,
            market_regime=market_regime, context_blocker=context_blocker, cfg=cfg,
            risk_fraction=min(cfg.specialty_risk_per_trade_pct, 0.0030),
            position_fraction=min(cfg.specialty_max_position_pct, 0.18),
        )
        rows.append({
            'ticker': ticker, 'bsjp_status': status, 'bsjp_score': round(score, 1),
            'bsjp_edge_score': round(score, 1), 'setup_valid': setup_valid, 'action': action,
            'last_price': close, 'intraday_last': intraday_last, 'entry': entry, 'stop_loss': stop,
            'morning_tp1': tp1, 'morning_tp2': tp2, 'tp1_basis': targets['tp1_basis'],
            'tp2_basis': targets['tp2_basis'], 'target_model': 'PRICE_STRUCTURE_ONLY',
            'target_structure': targets['target_structure'], 'target_structure_valid': target_valid,
            'rr1': rr1, 'rr2': rr2, 'risk_pct': stop_pct, 'daily_return_pct': daily_return,
            'volume_ratio': vol_ratio, 'adtv20_idr': adtv, 'cmf20': cmf_v,
            'relative_strength60': rs_v, 'session_vwap': im.get('session_vwap'),
            'vwap_slope_pct': session_q.get('vwap_slope_pct'),
            'session_close_location': im.get('session_close_location'),
            'late_volume_acceleration': im.get('late_volume_acceleration'),
            'late_volume_share': session_q.get('late_volume_share'),
            'u_shape_volume_ratio': session_q.get('u_shape_volume_ratio'),
            'directional_efficiency': session_q.get('directional_efficiency'),
            'afternoon_breakout': session_q.get('afternoon_breakout'),
            'upper_wick_ratio': session_q.get('upper_wick_ratio'),
            'room_to_ara_pct': room_to_ara, 'daily_confirmation_count': daily_hits,
            'session_confirmation_count': session_hits, 'intraday_bars': int(im['intraday_bars']),
            'intraday_interval_minutes': im['intraday_interval_minutes'],
            'intraday_data_state': im['intraday_data_state'],
            'intraday_session_date': im['intraday_session_date'],
            'intraday_age_minutes': im['intraday_age_minutes'], 'intraday_fresh': im['intraday_fresh'],
            'market_regime': market_regime, 'execution_window': '14:30–15:49 WIB',
            'exit_window': '09:00–10:00 WIB next session',
            'order_instruction': 'WAIT_SHARED_RISK_AND_PRICE_GATE' if setup_valid else 'WATCH_ONLY',
            'risk_class': 'HIGH_OVERNIGHT_GAP_RISK', 'warnings': ' • '.join(warnings),
            'blockers': context_blocker, **gate,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        rank = {'BSJP_ORDER_READY': 0, 'BSJP_SIGNAL_READY': 1, 'BSJP_WATCHLIST': 2, 'BSJP_DAILY_RADAR': 3, 'BSJP_WAIT_SESSION_BARS': 4, 'BSJP_STALE_INTRADAY': 5, 'BSJP_DATA_UNAVAILABLE': 6}
        result['_rank'] = result['bsjp_status'].map(rank).fillna(9)
        ready_count = int(result['signal_ready'].map(truthy).sum()) if 'signal_ready' in result else 0
        limit = max(int(cfg.daily_radar_limit), ready_count)
        result = result.sort_values(['_rank', 'bsjp_score', 'adtv20_idr'], ascending=[True, False, False]).head(limit).drop(columns='_rank').reset_index(drop=True)
    return result

def scan_bpjs_candidates(prepared: Mapping[str, pd.DataFrame], intraday: Mapping[str, pd.DataFrame] | None=None, core_signals: pd.DataFrame | None=None, config: ScanConfig | None=None, now: Any | None=None, market_context: MarketContext | None=None) -> pd.DataFrame:
    """Rank opening-range continuation candidates using confirmation, not touch.

    A valid BPJS setup requires a completed close above the 15-minute opening
    range, price acceptance above VWAP, and no excessive extension.  Volume,
    daily trend and liquidity adjust conviction but do not fabricate a signal.
    """
    cfg = config or ScanConfig()
    intraday = intraday or {}
    critical_map = _critical_context_map(core_signals)
    now_jkt = jakarta_timestamp(now)
    market_regime = market_context.regime if market_context is not None else 'NOT_EVALUATED'
    minute = now_jkt.hour * 60 + now_jkt.minute
    in_window = now_jkt.weekday() < 5 and 9 * 60 + 20 <= minute <= 10 * 60 + 45
    rows: list[dict[str, Any]] = []
    for ticker, frame in prepared.items():
        if frame is None or frame.empty or len(frame) < 60:
            continue
        row, prev = frame.iloc[-1], frame.iloc[-2]
        close, prev_close = safe_number(row.get('Close'), 0.0), safe_number(prev.get('Close'), 0.0)
        if close <= 0 or prev_close <= 0:
            continue
        adtv = safe_number(row.get('ADTV20'), 0.0)
        rs_v = safe_number(row.get('REL_STRENGTH60'), -1.0)
        cmf_v = safe_number(row.get('CMF20'), -1.0)
        atr_v = safe_number(row.get('ATR14'), close * 0.03)
        trend = bool(safe_number(row.get('EMA20'), 0.0) >= safe_number(row.get('EMA50'), float('inf')) and close >= safe_number(row.get('EMA50'), float('inf')))
        im = _intraday_metrics(intraday.get(ticker, pd.DataFrame()), now_jkt, cfg.max_intraday_stale_minutes)
        has_any_intraday = im['intraday_bars'] > 0
        has_intraday = im['intraday_data_state'] == 'LIVE_READY' and im['post_orb_bars'] >= 1
        session = _intraday_session(intraday.get(ticker, pd.DataFrame()), asof=now_jkt)
        session_q = _session_quality_features(session, im)
        session_open = float(session['Open'].iloc[0]) if has_intraday and not session.empty else np.nan
        intraday_last = safe_number(im.get('intraday_last'), close)
        gap = session_open / close - 1.0 if np.isfinite(session_open) and close > 0 else np.nan
        vwap = safe_number(im.get('session_vwap'), np.nan)
        orb_high = safe_number(im.get('orb_high'), np.nan)
        orb_low = safe_number(im.get('orb_low'), np.nan)
        orb_width = max(0.0, orb_high - orb_low) if np.isfinite(orb_high) and np.isfinite(orb_low) else np.nan
        above_vwap = bool(has_intraday and np.isfinite(vwap) and intraday_last >= vwap)
        vwap_rising = safe_number(session_q.get('vwap_slope_pct'), -1.0) >= -0.0005
        orb_break = bool(has_intraday and truthy(session_q.get('orb_breakout_confirmed')))
        orb_hold = bool(has_intraday and truthy(session_q.get('orb_hold_confirmed')))
        opening_volume_ratio = safe_number(im.get('opening_volume_ratio'), 0.0)
        breakout_volume_ratio = safe_number(session_q.get('breakout_volume_ratio'), np.nan)
        location = safe_number(im.get('session_close_location'), 0.0)
        efficiency = safe_number(session_q.get('directional_efficiency'), -1.0)
        extension_ratio = safe_number(session_q.get('orb_extension_ratio'), np.nan)
        max_extension = max(1.25, 0.75 * atr_v / max(orb_width, idx_tick_size(close))) if np.isfinite(orb_width) and orb_width > 0 else 1.5
        not_extended = bool(not np.isfinite(extension_ratio) or extension_ratio <= max_extension)
        gap_ok = bool(has_intraday and np.isfinite(gap) and -0.03 <= gap <= 0.07)
        orb_width_pct = orb_width / close if np.isfinite(orb_width) and close > 0 else np.nan
        orb_width_ok = bool(np.isfinite(orb_width_pct) and 0.002 <= orb_width_pct <= 0.055)
        conditions = {
            'Trend daily belum mendukung': trend,
            'Relative strength lemah': rs_v >= -0.03,
            'CMF masih negatif': cmf_v >= -0.05,
            'Harga belum diterima di atas VWAP': above_vwap,
            'VWAP belum menanjak': vwap_rising,
            'ORB belum ditutup tembus': orb_break,
            'Breakout belum bertahan': orb_hold,
            'Opening range terlalu lebar/sempit': orb_width_ok,
            'Harga sudah terlalu extended dari ORB': not_extended,
            'Close sesi belum dekat high': location >= 0.60,
            'Directional efficiency rendah': efficiency >= 0.25,
            'Gap di luar rentang ideal': gap_ok,
        }
        score = 0.0
        score += 9.0 if trend else 4.0
        score += 7.0 if rs_v > 0 else 4.0 if rs_v >= -0.03 else 0.0
        score += 6.0 if cmf_v >= 0.05 else 3.0 if cmf_v >= -0.05 else 0.0
        score += 15.0 if above_vwap else 0.0
        score += 7.0 if vwap_rising else 2.0
        score += 18.0 if orb_break and orb_hold else 10.0 if orb_break else 0.0
        score += 8.0 if opening_volume_ratio >= 1.5 else 5.0 if opening_volume_ratio >= 1.05 else 2.0 if has_intraday else 0.0
        score += 7.0 if np.isfinite(breakout_volume_ratio) and breakout_volume_ratio >= 1.0 else 4.0 if np.isfinite(breakout_volume_ratio) and breakout_volume_ratio >= 0.5 else 2.0
        score += 8.0 if location >= 0.78 else 5.0 if location >= 0.60 else 0.0
        score += 7.0 if efficiency >= 0.55 else 4.0 if efficiency >= 0.25 else 0.0
        score += 5.0 if gap_ok else 1.0
        score += 4.0 if orb_width_ok else 0.0
        score += 4.0 if not_extended else 0.0
        score += 5.0 if adtv >= cfg.min_adtv_idr else 2.0 if adtv >= 500_000_000 else 0.0
        score = min(100.0, score)

        entry_reference = max(orb_high, intraday_last) if orb_break else orb_high
        entry = round_idx_price(entry_reference, 'up')
        if entry is None:
            entry = round_idx_price(intraday_last, 'up')
        if has_intraday and not session.empty:
            post_n = max(1, int(safe_number(im.get('opening_range_bars'), 1)))
            post = session.iloc[post_n:]
            recent_low = safe_number(post['Low'].tail(min(4, len(post))).min(), orb_high) if not post.empty else orb_high
            support_candidates = [value for value in (recent_low, orb_high - 0.10 * max(orb_width, idx_tick_size(entry)), vwap - 0.10 * max(orb_width, idx_tick_size(entry))) if np.isfinite(value) and value < entry]
            structural_support = max(support_candidates) if support_candidates else orb_low
            raw_stop = structural_support - idx_tick_size(structural_support)
        else:
            raw_stop = entry - max(0.035 * entry, 1.0 * atr_v, 3 * idx_tick_size(entry))
        stop = round_idx_price(raw_stop, 'down')
        if stop is None or stop >= entry:
            stop = round_idx_price(entry - max(3 * idx_tick_size(entry), 0.025 * entry), 'down')
        risk = max(entry - stop, idx_tick_size(entry))
        ara_price = idx_daily_price_band(close)[1]
        room_to_ara = ara_price / entry - 1.0 if ara_price and entry > 0 else np.nan
        session_high = safe_number(session['High'].max(), entry) if not session.empty else entry
        session_low = safe_number(session['Low'].min(), stop) if not session.empty else stop
        targets = price_structure_target_pair(
            frame, entry, setup='BPJS',
            explicit_levels=[
                (session_high, 'CURRENT_SESSION_HIGH'),
                (safe_number(row.get('High'), np.nan), 'PRIOR_SESSION_HIGH'),
                (safe_number(row.get('HIGH20_PREV'), np.nan), 'PRIOR_20D_HIGH'),
                (ara_price, 'CURRENT_SESSION_ARA_LIMIT'),
            ],
            projection_origin=orb_high,
            projection_height=max(orb_width if np.isfinite(orb_width) else 0.0, session_high - session_low, idx_tick_size(entry)),
            price_ceiling=ara_price,
        )
        tp1, tp2 = targets['tp1'], targets['tp2']
        target_valid = bool(targets['target_structure_valid'])
        rr1 = round((tp1 - entry) / risk, 2) if target_valid else np.nan
        rr2 = round((tp2 - entry) / risk, 2) if target_valid else np.nan
        stop_pct = (entry - stop) / entry if entry > 0 else np.nan
        context_blocker = critical_map.get(ticker, '')
        setup_valid = bool(
            in_window and has_intraday and above_vwap and vwap_rising and orb_break and orb_hold
            and not_extended and score >= 70.0 and target_valid and not context_blocker
        )
        if setup_valid:
            status, action = 'BPJS_SIGNAL_READY', 'WAIT_SHARED_RISK_AND_PRICE_GATE'
        elif not has_any_intraday and not in_window:
            status, action = 'BPJS_DAILY_RADAR', 'RUN_AGAIN_09_20_10_45_WIB'
        elif not has_any_intraday:
            status, action = 'BPJS_DATA_UNAVAILABLE', 'RETRY_INTRADAY_5M'
        elif not truthy(im['intraday_fresh']):
            status, action = 'BPJS_STALE_INTRADAY', 'REFRESH_INTRADAY_DATA'
        elif not has_intraday:
            status, action = 'BPJS_WAIT_OPENING_BARS', 'RUN_AGAIN_AFTER_OPENING_RANGE'
        elif not in_window:
            status, action = 'BPJS_DAILY_RADAR', 'RUN_AGAIN_09_20_10_45_WIB'
        else:
            status, action = 'BPJS_WATCHLIST', 'WAIT_ORB_CLOSE_VWAP_ACCEPTANCE'
        warnings = _specialty_risk_warnings(market_regime=market_regime, adtv=adtv, cfg=cfg, stop_pct=stop_pct, room_to_ara=room_to_ara, conditions=conditions)
        if not target_valid:
            warnings.append('Dua target struktur harga belum tersedia; RR tidak difabrikasi')
        gate = _specialty_prebudget_gate(
            mode='BPJS', signal_ready=setup_valid, in_window=in_window,
            intraday_fresh=truthy(im.get('intraday_fresh')), requires_intraday=True,
            entry=entry, stop=stop, tp1=tp1, tp2=tp2, rr1=rr1, rr2=rr2,
            risk_pct=stop_pct, adtv=adtv, target_valid=target_valid,
            market_regime=market_regime, context_blocker=context_blocker, cfg=cfg,
            risk_fraction=min(cfg.specialty_risk_per_trade_pct, 0.0035),
            position_fraction=min(cfg.specialty_max_position_pct, 0.20),
        )
        rows.append({
            'ticker': ticker, 'bpjs_status': status, 'bpjs_score': round(score, 1),
            'bpjs_edge_score': round(score, 1), 'setup_valid': setup_valid, 'action': action,
            'last_price': close, 'intraday_last': intraday_last, 'opening_gap_pct': gap,
            'entry': entry, 'stop_loss': stop, 'day_tp1': tp1, 'day_tp2': tp2,
            'tp1_basis': targets['tp1_basis'], 'tp2_basis': targets['tp2_basis'],
            'target_model': 'PRICE_STRUCTURE_ONLY', 'target_structure': targets['target_structure'],
            'target_structure_valid': target_valid, 'rr1': rr1, 'rr2': rr2,
            'risk_pct': stop_pct, 'session_vwap': im.get('session_vwap'),
            'vwap_slope_pct': session_q.get('vwap_slope_pct'), 'orb_high': orb_high,
            'orb_low': orb_low, 'orb_width_pct': orb_width_pct,
            'orb_extension_ratio': session_q.get('orb_extension_ratio'),
            'orb_breakout_confirmed': session_q.get('orb_breakout_confirmed'),
            'orb_hold_confirmed': session_q.get('orb_hold_confirmed'),
            'opening_volume_ratio': im.get('opening_volume_ratio'),
            'breakout_volume_ratio': session_q.get('breakout_volume_ratio'),
            'session_close_location': im.get('session_close_location'),
            'directional_efficiency': session_q.get('directional_efficiency'),
            'adtv20_idr': adtv, 'relative_strength60': rs_v,
            'room_to_ara_pct': room_to_ara, 'confirmation_count': sum(bool(v) for v in conditions.values()),
            'intraday_bars': int(im['intraday_bars']), 'intraday_interval_minutes': im['intraday_interval_minutes'],
            'opening_range_bars': int(im['opening_range_bars']), 'post_orb_bars': int(im['post_orb_bars']),
            'intraday_data_state': im['intraday_data_state'], 'intraday_session_date': im['intraday_session_date'],
            'intraday_age_minutes': im['intraday_age_minutes'], 'intraday_fresh': im['intraday_fresh'],
            'market_regime': market_regime, 'execution_window': '09:20–10:45 WIB',
            'mandatory_exit': 'Before regular-market close',
            'order_instruction': 'WAIT_SHARED_RISK_AND_PRICE_GATE' if setup_valid else 'WATCH_ONLY',
            'risk_class': 'HIGH_INTRADAY_EXECUTION_RISK', 'warnings': ' • '.join(warnings),
            'blockers': context_blocker, **gate,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        rank = {'BPJS_ORDER_READY': 0, 'BPJS_SIGNAL_READY': 1, 'BPJS_WATCHLIST': 2, 'BPJS_DAILY_RADAR': 3, 'BPJS_WAIT_OPENING_BARS': 4, 'BPJS_STALE_INTRADAY': 5, 'BPJS_DATA_UNAVAILABLE': 6}
        result['_rank'] = result['bpjs_status'].map(rank).fillna(9)
        ready_count = int(result['signal_ready'].map(truthy).sum()) if 'signal_ready' in result else 0
        limit = max(int(cfg.daily_radar_limit), ready_count)
        result = result.sort_values(['_rank', 'bpjs_score', 'adtv20_idr'], ascending=[True, False, False]).head(limit).drop(columns='_rank').reset_index(drop=True)
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
        close, prev_close = (safe_number(row.get('Close'), 0.0), safe_number(prev.get('Close'), 0.0))
        if close <= 0 or prev_close <= 0:
            continue
        adtv = safe_number(row.get('ADTV20'), 0.0)
        vol_ratio = safe_number(row.get('VOL_RATIO'), 0.0)
        close_location = safe_number(row.get('CLOSE_LOCATION'), 0.0)
        cmf_v = safe_number(row.get('CMF20'), -1.0)
        obv_up = safe_number(row.get('OBV_SLOPE10'), -1.0) > 0
        rs_v = safe_number(row.get('REL_STRENGTH60'), -1.0)
        body_atr = safe_number(row.get('BODY_ATR'), 0.0)
        daily_return = close / prev_close - 1.0
        ema20, ema50 = (safe_number(row.get('EMA20'), 0.0), safe_number(row.get('EMA50'), 0.0))
        trend = ema20 >= ema50 and close >= ema50
        breakout = close > safe_number(row.get('HIGH20_PREV'), float('inf'))
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
    """Rank ICT Sniper structures by sequence, displacement, freshness and entry phase.

    The scanner does not treat an evidence string as sufficient on its own. If
    timestamps are available it validates sweep -> displacement/BOS -> FVG in
    chronological order, measures FVG age/width, and distinguishes a healthy
    retracement from a confirmed reclaim.  Synthetic/legacy rows retain a
    conservative inference path for backward compatibility.
    """
    cfg = config or ScanConfig()
    columns = [
        'ticker', 'sniper_status', 'sniper_entry_mode', 'sniper_score', 'sniper_edge_score',
        'core_status', 'core_action', 'setup', 'last_price', 'entry_low', 'entry_high',
        'sniper_entry', 'sniper_trigger', 'sniper_stop', 'sniper_tp1', 'sniper_tp2',
        'rr1', 'rr2', 'stop_pct', 'distance_atr', 'volume_ratio', 'volume_context',
        'silent_accumulation_score', 'structure_grade', 'trigger_state', 'sequence_valid',
        'bos_displacement_atr', 'bos_volume_ratio', 'fvg_age_bars', 'fvg_width_atr',
        'mitigation_pct', 'valid_until', 'risk_warnings', 'primary_sniper_blocker',
        'blockers', 'reason',
    ]
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
        evidence = safe_text(signal.get('evidence'))
        blockers_text = safe_text(signal.get('blockers'))
        action = safe_text(signal.get('action'))
        core_status = safe_text(signal.get('status'))
        quality = np.clip(safe_number(signal.get('quality_score'), 0.0), 0.0, 100.0)
        distance = safe_number(signal.get('distance_atr'), 99.0)
        volume_ratio = safe_number(signal.get('volume_ratio'), safe_number(row.get('VOL_RATIO'), 0.0))
        accumulation = np.clip(safe_number(signal.get('silent_accumulation_score'), 50.0), 0.0, 100.0)
        rr1 = safe_number(signal.get('rr1'), np.nan)
        rr2 = safe_number(signal.get('rr2'), np.nan)
        stop_pct = safe_number(signal.get('stop_pct'), np.nan)
        close = safe_number(signal.get('last_price'), safe_number(row.get('Close'), np.nan))
        open_v = safe_number(row.get('Open'), close)
        prev_high = safe_number(previous.get('High'), close)
        close_location = safe_number(row.get('CLOSE_LOCATION'), 0.5)
        bull_rejection = truthy(row.get('BULL_REJECTION', False))
        reclaim = bool(action == 'READY_TRIGGER' or core_status == 'EXECUTION_READY' or bull_rejection or (np.isfinite(close) and close > prev_high and close > open_v and close_location >= 0.62))
        sweep = 'Sell-side liquidity sweep' in evidence
        bos = 'Bullish BOS dengan displacement' in evidence
        fvg = 'Bullish FVG valid' in evidence
        ob_overlap = 'FVG overlap dengan order-block proxy' in evidence
        discount = 'Zona berada di discount dealing range' in evidence
        inferred_ready_structure = bool(core_status == 'EXECUTION_READY' and quality >= 82.0)
        core_structure = bool((sweep and bos and fvg) or inferred_ready_structure)

        signal_date = pd.to_datetime(signal.get('signal_date'), errors='coerce')
        zone_date = pd.to_datetime(signal.get('zone_created_date'), errors='coerce')
        sequence_valid = core_structure
        bos_displacement = np.nan
        bos_volume = np.nan
        fvg_age = safe_number(signal.get('zone_age_bars'), np.nan)
        if pd.notna(signal_date) and pd.notna(zone_date):
            try:
                signal_pos = int(np.flatnonzero(pd.DatetimeIndex(frame.index) == pd.Timestamp(signal_date))[-1])
                zone_pos = int(np.flatnonzero(pd.DatetimeIndex(frame.index) == pd.Timestamp(zone_date))[-1])
                sequence_valid = bool(signal_pos <= zone_pos)
                segment = frame.iloc[max(signal_pos + 1, 0):min(len(frame), zone_pos + 1)]
                if not segment.empty:
                    bos_displacement = safe_number(pd.to_numeric(segment.get('BODY_ATR'), errors='coerce').max(), np.nan)
                    bos_volume = safe_number(pd.to_numeric(segment.get('VOL_RATIO'), errors='coerce').max(), np.nan)
                    sequence_valid = bool(sequence_valid and safe_number(bos_displacement, 0.0) >= 0.50)
                fvg_age = float(max(0, len(frame) - 1 - zone_pos))
            except Exception:
                sequence_valid = core_structure
        elif core_structure:
            bos_displacement = 0.75 if quality >= 82 else 0.55
            bos_volume = max(volume_ratio, 1.05 if quality >= 82 else 0.8)

        entry_low = safe_number(signal.get('entry_low'), np.nan)
        entry_high = safe_number(signal.get('entry_high'), np.nan)
        atr_v = safe_number(row.get('ATR14'), np.nan)
        fvg_width = entry_high - entry_low if np.isfinite(entry_high) and np.isfinite(entry_low) else np.nan
        fvg_width_atr = fvg_width / atr_v if np.isfinite(fvg_width) and np.isfinite(atr_v) and atr_v > 0 else np.nan
        mitigation_pct = np.nan
        if np.isfinite(fvg_width) and fvg_width > 0 and np.isfinite(close):
            if close >= entry_high:
                mitigation_pct = 0.0
            elif close <= entry_low:
                mitigation_pct = 1.0
            else:
                mitigation_pct = (entry_high - close) / fvg_width
        fvg_fresh = bool(not np.isfinite(fvg_age) or fvg_age <= 20)
        fvg_size_ok = bool(not np.isfinite(fvg_width_atr) or 0.08 <= fvg_width_atr <= 1.20)
        strict_confluence = bool((ob_overlap and discount and quality >= 72.0) or (inferred_ready_structure and quality >= 88.0))
        displacement_ok = bool(sequence_valid and safe_number(bos_displacement, 0.0) >= 0.50)

        daily_source_tier = safe_text(signal.get('ohlcv_source_tier')).upper()
        daily_source_ok = daily_source_tier not in {'CACHE_FALLBACK', 'UNAVAILABLE'}
        invalid = truthy(signal.get('invalidated', False)) or not truthy(signal.get('detected', True))
        invalid_action = action in {'NO_SETUP', 'TOO_EXTENDED_WAIT_NEW_BASE', 'WAIT_CHOCH'}
        level_values = [signal.get('entry'), signal.get('stop_loss'), signal.get('tp1'), signal.get('tp2')]
        levels_valid = all(is_valid_idx_price(safe_number(value, np.nan)) for value in level_values)
        valid_until = signal.get('valid_until')
        expired = False
        if valid_until is not None and not pd.isna(valid_until):
            try:
                expired = pd.Timestamp(frame.index[-1]) > pd.Timestamp(valid_until)
            except Exception:
                expired = False
        hard: list[str] = []
        if not core_structure:
            hard.append('Sweep–BOS–FVG belum lengkap')
        if not sequence_valid or not displacement_ok:
            hard.append('Urutan sweep–displacement–FVG tidak valid')
        if invalid or invalid_action:
            hard.append('Struktur Unicorn/Sniper tidak valid')
        if expired or not fvg_fresh:
            hard.append('Zona FVG sudah kedaluwarsa')
        if not fvg_size_ok:
            hard.append('Ukuran FVG tidak proporsional terhadap ATR')
        if not levels_valid:
            hard.append('Level entry/SL/TP tidak valid menurut fraksi IDX')
        if not daily_source_ok:
            hard.append('OHLCV daily bukan hasil live')
        if truthy(signal.get('pending_close', False)):
            hard.append('Daily candle belum final')
        if truthy(signal.get('market_status_critical_blocker', False)):
            hard.append('Suspensi/FCA/status perdagangan negatif')
        if truthy(signal.get('quote_critical_blocker', False)):
            hard.append('Konflik quote/candle')

        in_trigger_zone = bool(np.isfinite(distance) and distance <= 0.40)
        in_limit_range = bool(np.isfinite(distance) and distance <= 1.0)
        if reclaim and in_trigger_zone:
            entry_mode = 'RECLAIM_TRIGGER'
            trigger_state = 'CONFIRMED'
            volume_context = 'RECLAIM_EXPANSION' if volume_ratio >= 1.05 else 'RECLAIM_PRICE_CONFIRMATION'
            volume_score = 9.0 if volume_ratio >= 1.20 else 7.0 if volume_ratio >= 1.05 else 5.0
            trigger_score = 14.0
        elif in_limit_range and action in {'WAIT_FVG_RETRACE', 'WAIT_STRICT_UNICORN_CONFLUENCE', 'READY_LIMIT'}:
            entry_mode = 'LIMIT_FVG_RETRACE'
            trigger_state = 'RESTING_LIMIT_VALID'
            volume_context = 'HEALTHY_RETRACE_CONTRACTION' if 0.45 <= volume_ratio <= 1.20 else 'RETRACE_VOLUME_WARNING'
            volume_score = 9.0 if 0.55 <= volume_ratio <= 1.05 else 6.0 if 0.35 <= volume_ratio <= 1.35 else 2.0
            trigger_score = 10.0
        elif core_structure and np.isfinite(distance) and distance > 1.0:
            entry_mode = 'WAIT_FVG_RETRACE'
            trigger_state = 'PRICE_TOO_FAR'
            volume_context = 'NOT_APPLICABLE'
            volume_score = 2.0
            trigger_score = 2.0
        else:
            entry_mode = 'WATCH_STRUCTURE'
            trigger_state = 'INCOMPLETE'
            volume_context = 'UNRESOLVED'
            volume_score = 2.0
            trigger_score = 3.0

        structure_score = 26.0 if core_structure and sequence_valid else 10.0
        displacement_score = 12.0 if safe_number(bos_displacement, 0.0) >= 0.80 and safe_number(bos_volume, 0.0) >= 1.05 else 9.0 if displacement_ok else 3.0
        confluence_score = (6.0 if ob_overlap else 2.0) + (6.0 if discount else 2.0) + (4.0 if strict_confluence else 1.0)
        freshness_score = 8.0 if safe_number(fvg_age, 99.0) <= 8 else 5.0 if safe_number(fvg_age, 99.0) <= 20 else 0.0
        location_score = 13.0 if np.isfinite(distance) and distance <= 0.15 else 10.0 if np.isfinite(distance) and distance <= 0.40 else 7.0 if np.isfinite(distance) and distance <= 0.75 else 4.0 if np.isfinite(distance) and distance <= 1.0 else 0.0
        flow_score = 7.0 if accumulation >= 70 else 5.0 if accumulation >= 55 else 3.0 if accumulation >= 40 else 1.0
        geometry_score = 5.0 if np.isfinite(rr2) and rr2 >= 3.0 else 3.0 if np.isfinite(rr2) and rr2 >= 2.0 else 1.0
        quality_bonus = min(5.0, quality * 0.05)
        score = min(100.0, structure_score + displacement_score + confluence_score + freshness_score + location_score + trigger_score + volume_score + flow_score + geometry_score + quality_bonus)

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
        if np.isfinite(fvg_age) and fvg_age > 12:
            risk_warnings.append(f'FVG berumur {int(fvg_age)} bar')
        if core_status != 'EXECUTION_READY':
            risk_warnings.append(f"Core status {core_status or 'UNKNOWN'}; Sniper dinilai independen")
        if blockers_text:
            risk_warnings.append('Core warning tersedia')
        ready = bool(not hard and entry_mode in {'RECLAIM_TRIGGER', 'LIMIT_FVG_RETRACE'} and quality >= 68.0 and score >= 72.0)
        if ready:
            status = 'SNIPER_SIGNAL_READY'
        elif not hard and core_structure and np.isfinite(distance) and distance <= 1.5:
            status = 'WAIT_SNIPER_RETRACE'
        elif hard:
            status = 'SNIPER_REJECT'
        else:
            status = 'SNIPER_WATCHLIST'
        structure_grade = 'STRICT' if strict_confluence and displacement_ok and fvg_fresh else 'VALID' if core_structure and sequence_valid else 'INCOMPLETE'
        target_valid = bool(levels_valid and np.isfinite(rr1) and np.isfinite(rr2))
        gate = _specialty_prebudget_gate(
            mode='SNIPER', signal_ready=ready, in_window=False,
            intraday_fresh=False, requires_intraday=False,
            entry=safe_number(signal.get('entry'), np.nan), stop=safe_number(signal.get('stop_loss'), np.nan),
            tp1=safe_number(signal.get('tp1'), np.nan), tp2=safe_number(signal.get('tp2'), np.nan),
            rr1=rr1, rr2=rr2, risk_pct=stop_pct,
            adtv=safe_number(signal.get('adtv20_idr'), safe_number(row.get('ADTV20'), 0.0)),
            target_valid=target_valid, market_regime=safe_text(signal.get('market_regime')),
            context_blocker=' • '.join(hard), cfg=cfg,
            risk_fraction=min(cfg.specialty_risk_per_trade_pct, 0.0035),
            position_fraction=min(cfg.specialty_max_position_pct, 0.20),
        )
        rows.append({
            'ticker': ticker, 'sniper_status': status, 'sniper_entry_mode': entry_mode,
            'sniper_score': round(float(score), 1), 'sniper_edge_score': round(float(score), 1),
            'core_status': core_status, 'core_action': action, 'setup': 'ICT_SNIPER_SEQUENCE',
            'last_price': close, 'entry_low': signal.get('entry_low'), 'entry_high': signal.get('entry_high'),
            'sniper_entry': signal.get('entry'), 'sniper_trigger': signal.get('trigger'),
            'sniper_stop': signal.get('stop_loss'), 'sniper_tp1': signal.get('tp1'),
            'sniper_tp2': signal.get('tp2'), 'rr1': rr1, 'rr2': rr2, 'stop_pct': stop_pct,
            'distance_atr': distance, 'volume_ratio': volume_ratio, 'volume_context': volume_context,
            'silent_accumulation_score': accumulation, 'structure_grade': structure_grade,
            'trigger_state': trigger_state, 'sequence_valid': sequence_valid,
            'bos_displacement_atr': bos_displacement, 'bos_volume_ratio': bos_volume,
            'fvg_age_bars': fvg_age, 'fvg_width_atr': fvg_width_atr,
            'mitigation_pct': mitigation_pct, 'valid_until': valid_until,
            'risk_warnings': ' • '.join(risk_warnings),
            'primary_sniper_blocker': hard[0] if hard else 'NONE' if ready else 'ENTRY_NOT_READY',
            'blockers': ' • '.join(hard),
            'order_instruction': 'USE_CORE_ORDER_TICKET' if ready else 'WATCH_ONLY',
            'reason': 'ICT sequence is ranked by displacement, FVG freshness, location and trigger phase.',
            **gate,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=columns)
    rank = {'SNIPER_ORDER_READY': 0, 'SNIPER_SIGNAL_READY': 1, 'WAIT_SNIPER_RETRACE': 2, 'SNIPER_WATCHLIST': 3, 'SNIPER_REJECT': 4}
    result['_rank'] = result['sniper_status'].map(rank).fillna(9)
    result = result.sort_values(['_rank', 'sniper_score', 'rr2'], ascending=[True, False, False]).drop(columns='_rank').reset_index(drop=True)
    return result

__all__ = ['download_intraday_ohlcv', 'specialty_intraday_shortlist', 'scan_sniper_entries', 'scan_bsjp_candidates', 'scan_bpjs_candidates', 'scan_multibagger_candidates', 'allocate_multibagger_capital', 'scan_ara_hunter_candidates', 'build_specialty_screens', 'build_daily_opportunity_board', 'build_profit_order_builder', 'parse_orderbook_snapshot_csv', 'apply_ara_external_confirmation']
