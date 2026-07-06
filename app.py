import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


st.set_page_config(page_title='Predictive Stock Scanner', layout='wide')
st.title('Predictive Stock Scanner')
st.caption('Scanner probabilistik untuk arah harga, entry zone, stoploss, takeprofit, dan expectancy.')

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header('Universe')
source = st.sidebar.radio('Sumber input', ['Paste tickers', 'Upload CSV'], index=0)

paste_text = 'BMRI\nBBCA\nASII\nTLKM\nMBMA'
if source == 'Paste tickers':
    paste_text = st.sidebar.text_area('Ticker (satu baris / koma)', value=paste_text, height=140)
    uploaded = None
else:
    uploaded = st.sidebar.file_uploader('Upload CSV universe / OHLCV', type=['csv'])
    st.sidebar.caption('CSV dapat berisi ticker list atau OHLCV lengkap dengan kolom symbol/ticker, date, open, high, low, close, volume.')

st.sidebar.header('Scan Settings')
period_months = st.sidebar.slider('Data historis (bulan)', 6, 60, 24)
max_tickers = st.sidebar.slider('Maks ticker', 20, 200, 100, step=10)
min_price = st.sidebar.number_input('Min harga (Rp)', value=100.0, step=10.0)
max_price = st.sidebar.number_input('Max harga (Rp)', value=50000.0, step=500.0)
min_avg_vol = st.sidebar.number_input('Min avg volume 20D', value=150000, step=50000)
lookbacks = st.sidebar.multiselect('Horizon prediksi (hari bursa)', [5, 10, 15, 20], default=[5, 10, 20])
mode = st.sidebar.selectbox('Mode risiko', ['Conservative', 'Balanced', 'Aggressive'], index=1)
workers = st.sidebar.slider('Parallel workers', 2, 12, 6)
run = st.sidebar.button('Run predictive scan', type='primary')

MODE_CFG = {
    'Conservative': {'entry_pad': 0.16, 'stop_pad': 1.05, 'tp_mult': 1.18, 'min_rr': 2.0, 'min_prob': 0.58},
    'Balanced': {'entry_pad': 0.22, 'stop_pad': 0.95, 'tp_mult': 1.08, 'min_rr': 1.6, 'min_prob': 0.54},
    'Aggressive': {'entry_pad': 0.28, 'stop_pad': 0.85, 'tp_mult': 0.98, 'min_rr': 1.3, 'min_prob': 0.50},
}
CFG = MODE_CFG[mode]

if 'scan_bundle' not in st.session_state:
    st.session_state.scan_bundle = None
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = None


# -----------------------------
# Helpers
# -----------------------------
def normalize_ticker(x: str) -> str:
    s = re.sub(r'\s+', '', str(x or '').upper()).replace('/', '-')
    if not s:
        return ''
    if s.endswith('.JK') or s.startswith('^'):
        return s
    if s.isalpha() and len(s) <= 5:
        return f'{s}.JK'
    return s


def _lower_map(cols) -> Dict[str, str]:
    return {str(c).strip().lower(): str(c) for c in cols}


def detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = _lower_map(df.columns)
    for cand in candidates:
        key = cand.lower()
        if key in lower:
            return lower[key]
    return None


def parse_universe(text: Optional[str], csv_df: Optional[pd.DataFrame]) -> List[str]:
    tickers: List[str] = []
    if csv_df is not None and not csv_df.empty:
        cols = [str(c).strip() for c in csv_df.columns]
        candidate = None
        for c in ['Ticker', 'ticker', 'Symbol', 'symbol', 'Kode', 'code', 'Stock']:
            if c in cols:
                candidate = c
                break
        if candidate is None:
            candidate = cols[0]
        for raw in csv_df[candidate].dropna().astype(str).tolist():
            sym = normalize_ticker(raw)
            if sym and sym not in tickers:
                tickers.append(sym)
    if text:
        for raw in re.split(r'[\n,;\t ]+', text):
            if not raw.strip():
                continue
            sym = normalize_ticker(raw)
            if sym and sym not in tickers:
                tickers.append(sym)
    return tickers


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [' '.join([str(p) for p in c if str(p) != '']).strip() for c in out.columns]
    out.columns = [str(c).strip() for c in out.columns]
    if 'Date' in out.columns:
        out = out.set_index('Date')
    out.index = pd.to_datetime(out.index, errors='coerce')
    out = out[~out.index.isna()].copy()
    needed = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not set(needed).issubset(set(out.columns)):
        return pd.DataFrame()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep='last')].copy()
    for c in needed + (['Adj Close'] if 'Adj Close' in out.columns else []):
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.dropna(subset=needed)
    return out


def load_ohlcv_frames(csv_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if csv_df is None or csv_df.empty:
        return {}
    sym_col = detect_column(csv_df, ['symbol', 'ticker', 'kode', 'code', 'stock', 'emiten'])
    date_col = detect_column(csv_df, ['date', 'datetime', 'time', 'timestamp'])
    open_col = detect_column(csv_df, ['open'])
    high_col = detect_column(csv_df, ['high'])
    low_col = detect_column(csv_df, ['low'])
    close_col = detect_column(csv_df, ['close'])
    vol_col = detect_column(csv_df, ['volume', 'vol'])
    if not all([sym_col, date_col, open_col, high_col, low_col, close_col, vol_col]):
        return {}
    tmp = csv_df[[sym_col, date_col, open_col, high_col, low_col, close_col, vol_col]].copy()
    tmp.columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    tmp['Symbol'] = tmp['Symbol'].astype(str).map(normalize_ticker)
    tmp['Date'] = pd.to_datetime(tmp['Date'], errors='coerce')
    tmp = tmp.dropna(subset=['Symbol', 'Date'])
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        tmp[c] = pd.to_numeric(tmp[c], errors='coerce')
    tmp = tmp.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    frames: Dict[str, pd.DataFrame] = {}
    for sym, g in tmp.groupby('Symbol'):
        g = g.sort_values('Date').set_index('Date')
        g = g[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        g['Symbol'] = sym
        g = _standardize_ohlcv(g)
        if not g.empty:
            frames[sym] = g
    return frames


@st.cache_data(ttl=1800, show_spinner=False)
def download_one(symbol: str, period: str) -> Tuple[str, pd.DataFrame, str]:
    candidates = [symbol]
    if symbol.endswith('.JK'):
        candidates.append(symbol[:-3])
    elif not symbol.startswith('^'):
        candidates.append(f'{symbol}.JK')
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]
    last_err = ''
    for cand in candidates:
        try:
            df = yf.Ticker(cand).history(period=period, interval='1d', auto_adjust=False, actions=False)
            df = _standardize_ohlcv(df)
            if not df.empty:
                df = df.copy()
                df['Symbol'] = symbol
                return symbol, df, ''
            last_err = 'history_empty'
        except Exception as exc:
            last_err = f'history:{type(exc).__name__}: {exc}'
        try:
            df = yf.download(cand, period=period, interval='1d', auto_adjust=False, progress=False, threads=False, group_by='column')
            df = _standardize_ohlcv(df)
            if not df.empty:
                df = df.copy()
                df['Symbol'] = symbol
                return symbol, df, ''
            last_err = 'download_empty'
        except Exception as exc:
            last_err = f'download:{type(exc).__name__}: {exc}'
    return symbol, pd.DataFrame(), last_err or 'failed'


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    rs = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_s
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def recent_swing_low(df: pd.DataFrame, n: int = 20) -> float:
    return float(df['Low'].tail(n).min()) if len(df) else np.nan


def recent_swing_high(df: pd.DataFrame, n: int = 20) -> float:
    return float(df['High'].tail(n).max()) if len(df) else np.nan


def finite_min(values, default=np.nan):
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(min(vals)) if vals else default


def finite_max(values, default=np.nan):
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(max(vals)) if vals else default


def safe_slope(series: pd.Series, n: int) -> pd.Series:
    def _slope(x):
        if len(x) < 2 or np.any(~np.isfinite(x)):
            return np.nan
        idx = np.arange(len(x), dtype=float)
        try:
            return float(np.polyfit(idx, np.asarray(x, dtype=float), 1)[0])
        except Exception:
            return np.nan
    return series.rolling(n, min_periods=n).apply(_slope, raw=False)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['ret1'] = out['Close'].pct_change()
    out['ret3'] = out['Close'].pct_change(3)
    out['ret5'] = out['Close'].pct_change(5)
    out['vol20'] = out['Close'].pct_change().rolling(20, min_periods=20).std()
    out['ema20'] = ema(out['Close'], 20)
    out['ema50'] = ema(out['Close'], 50)
    out['ema200'] = ema(out['Close'], 200)
    out['rsi14'] = rsi(out['Close'], 14)
    out['atr14'] = atr(out, 14)
    out['atr_pct'] = out['atr14'] / out['Close']
    out['adx14'] = adx(out, 14)
    out['vol_sma20'] = sma(out['Volume'], 20)
    out['vol_ratio20'] = out['Volume'] / out['vol_sma20']
    out['dist_ema20'] = (out['Close'] - out['ema20']) / out['Close']
    out['dist_ema50'] = (out['Close'] - out['ema50']) / out['Close']
    out['close_pos_20'] = (out['Close'] - out['Low'].rolling(20, min_periods=20).min()) / (
        out['High'].rolling(20, min_periods=20).max() - out['Low'].rolling(20, min_periods=20).min()
    )
    out['range20'] = (out['High'].rolling(20, min_periods=20).max() - out['Low'].rolling(20, min_periods=20).min()) / out['Close']
    out['body_ratio'] = (out['Close'] - out['Open']).abs() / (out['High'] - out['Low']).replace(0, np.nan)
    out['upper_wick'] = (out['High'] - out[['Open', 'Close']].max(axis=1)) / out['Close']
    out['lower_wick'] = (out[['Open', 'Close']].min(axis=1) - out['Low']) / out['Close']
    out['hh20'] = out['High'].rolling(20, min_periods=20).max()
    out['ll20'] = out['Low'].rolling(20, min_periods=20).min()
    out['hh10'] = out['High'].rolling(10, min_periods=10).max()
    out['ll10'] = out['Low'].rolling(10, min_periods=10).min()
    out['trend_slope10'] = safe_slope(out['Close'], 10)
    out['trend_slope20'] = safe_slope(out['Close'], 20)
    return out


def feature_columns() -> List[str]:
    return [
        'ret1', 'ret3', 'ret5', 'vol20', 'ema20', 'ema50', 'ema200',
        'rsi14', 'atr14', 'atr_pct', 'adx14', 'vol_sma20', 'vol_ratio20',
        'dist_ema20', 'dist_ema50', 'close_pos_20', 'range20', 'body_ratio',
        'upper_wick', 'lower_wick', 'hh20', 'll20', 'hh10', 'll10',
        'trend_slope10', 'trend_slope20',
    ]


def future_extreme(arr: np.ndarray, horizon: int, kind: str) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    for i in range(0, n - horizon):
        window = arr[i + 1:i + horizon + 1]
        window = window[np.isfinite(window)]
        if window.size == 0:
            continue
        out[i] = np.nanmax(window) if kind == 'max' else np.nanmin(window)
    return out


def build_dataset(all_frames: Dict[str, pd.DataFrame], horizons: List[int]) -> pd.DataFrame:
    rows = []
    for sym, raw in all_frames.items():
        if raw is None or raw.empty:
            continue
        df = compute_features(raw)
        close = df['Close'].to_numpy(dtype=float)
        for h in horizons:
            df[f'fwd_high_{h}'] = future_extreme(df['High'].to_numpy(dtype=float), h, 'max') / close - 1.0
            df[f'fwd_low_{h}'] = future_extreme(df['Low'].to_numpy(dtype=float), h, 'min') / close - 1.0
            df[f'fwd_close_{h}'] = df['Close'].shift(-h) / df['Close'] - 1.0
        df['Symbol'] = sym
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, axis=0)


def classify_setup_row(last: pd.Series, prev3_high: float) -> Tuple[str, str]:
    close = float(last.get('Close', np.nan))
    ema20_v = float(last.get('ema20', np.nan))
    ema50_v = float(last.get('ema50', np.nan))
    rsi_v = float(last.get('rsi14', np.nan))
    atr_v = float(last.get('atr14', np.nan))
    ll20 = float(last.get('ll20', np.nan))
    hh20 = float(last.get('hh20', np.nan))
    rng20 = float(last.get('range20', np.nan))
    trend_up = bool(np.isfinite(close) and np.isfinite(ema20_v) and np.isfinite(ema50_v) and close > ema20_v > ema50_v)
    trend_down = bool(np.isfinite(close) and np.isfinite(ema20_v) and np.isfinite(ema50_v) and close < ema20_v < ema50_v)
    near_support = np.isfinite(ll20) and close <= ll20 * 1.08
    near_resistance = np.isfinite(hh20) and close >= hh20 * 0.92
    squeeze = np.isfinite(atr_v) and np.isfinite(rng20) and float(last.get('atr_pct', np.nan)) < 0.06 and rng20 < 0.18
    oversold = np.isfinite(rsi_v) and rsi_v < 40
    reclaimed = np.isfinite(prev3_high) and close > prev3_high
    if trend_up and near_support:
        return 'PULLBACK', 'Trend up, retrace to support'
    if reclaimed and oversold and not trend_down:
        return 'UNICORN', 'Sweep/reclaim / imbalance fill'
    if trend_up and squeeze and near_resistance:
        return 'SNIPER', 'Tight base near breakout'
    if trend_down and oversold:
        return 'REVERSAL', 'Downtrend exhaustion'
    if trend_up:
        return 'PULLBACK', 'Primary trend continuation'
    if reclaimed:
        return 'UNICORN', 'Structure reclaim'
    return 'REVERSAL', 'Mean reversion / reversal watch'


@dataclass
class HorizonPack:
    up_q50: Pipeline
    up_q90: Pipeline
    low_q10: Pipeline
    low_q50: Pipeline
    clf_up: Pipeline
    val_dir_acc: float
    val_mae_up: float
    val_mae_low: float
    hist_setup_stats: Dict[str, Dict[str, float]]


def make_regressor(q: float) -> Pipeline:
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('gbr', GradientBoostingRegressor(
            loss='quantile', alpha=q, n_estimators=110, learning_rate=0.05,
            max_depth=3, min_samples_leaf=18, subsample=0.8, random_state=42,
        )),
    ])


def make_classifier() -> Pipeline:
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('gbc', GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.05, max_depth=3,
            min_samples_leaf=18, subsample=0.8, random_state=42,
        )),
    ])


def train_models(train_df: pd.DataFrame, horizons: List[int]) -> Dict[int, HorizonPack]:
    feats = feature_columns()
    out: Dict[int, HorizonPack] = {}
    base = train_df.dropna(subset=feats).copy()
    if base.empty:
        return out

    # Time split for model quality estimation
    dates = pd.Index(sorted(pd.to_datetime(base.index.unique())))
    if len(dates) < 40:
        split_date = dates.max()
    else:
        split_date = dates[int(len(dates) * 0.8)]
    train_mask = pd.to_datetime(base.index) < split_date
    val_mask = ~train_mask
    train_base = base.loc[train_mask].copy()
    val_base = base.loc[val_mask].copy()
    if train_base.empty:
        train_base = base.copy()
        val_base = base.iloc[-max(50, len(base)//5):].copy()

    for h in horizons:
        d = base.dropna(subset=[f'fwd_high_{h}', f'fwd_low_{h}', f'fwd_close_{h}']).copy()
        if len(d) < 250:
            continue
        d_train = train_base.dropna(subset=[f'fwd_high_{h}', f'fwd_low_{h}', f'fwd_close_{h}']).copy()
        d_val = val_base.dropna(subset=[f'fwd_high_{h}', f'fwd_low_{h}', f'fwd_close_{h}']).copy()
        if len(d_train) < 120:
            d_train = d.iloc[: int(len(d) * 0.8)].copy()
            d_val = d.iloc[int(len(d) * 0.8):].copy()

        Xtr = d_train[feats]
        Xva = d_val[feats] if not d_val.empty else d_train[feats].iloc[: min(50, len(d_train))]
        up_tr = d_train[f'fwd_high_{h}']
        low_tr = d_train[f'fwd_low_{h}']
        close_tr = d_train[f'fwd_close_{h}']
        clf_y_tr = (close_tr > 0).astype(int)

        up50 = make_regressor(0.50).fit(Xtr, up_tr)
        up90 = make_regressor(0.90).fit(Xtr, up_tr)
        low10 = make_regressor(0.10).fit(Xtr, low_tr)
        low50 = make_regressor(0.50).fit(Xtr, low_tr)
        clf = make_classifier().fit(Xtr, clf_y_tr)

        # Validation metrics
        if not d_val.empty:
            up_pred = up50.predict(Xva)
            low_pred = low50.predict(Xva)
            dir_prob = clf.predict_proba(Xva)[:, 1]
            dir_true = (d_val[f'fwd_close_{h}'] > 0).astype(int).to_numpy()
            val_dir_acc = float(np.mean((dir_prob >= 0.5) == dir_true))
            val_mae_up = float(np.nanmean(np.abs(up_pred - d_val[f'fwd_high_{h}'].to_numpy())))
            val_mae_low = float(np.nanmean(np.abs(low_pred - d_val[f'fwd_low_{h}'].to_numpy())))
        else:
            val_dir_acc = 0.5
            val_mae_up = float(np.nanmean(np.abs(up50.predict(Xtr) - up_tr.to_numpy())))
            val_mae_low = float(np.nanmean(np.abs(low50.predict(Xtr) - low_tr.to_numpy())))

        # Setup-level historical bias from the same training slice.
        hist_setup_stats: Dict[str, Dict[str, float]] = {}
        setup_rows = []
        for idx, row in d_train.iterrows():
            prev3_high = float(d_train.loc[:idx].tail(4)['High'].iloc[:-1].max()) if len(d_train.loc[:idx]) >= 4 else np.nan
            setup, _ = classify_setup_row(row, prev3_high)
            setup_rows.append((setup, row[f'fwd_close_{h}'], row[f'fwd_high_{h}'], row[f'fwd_low_{h}'], row['Close'], row.get('atr14', np.nan)))
        setup_df = pd.DataFrame(setup_rows, columns=['setup', 'fwd_close', 'fwd_high', 'fwd_low', 'close', 'atr14'])
        for setup_name, g in setup_df.groupby('setup'):
            if g.empty:
                continue
            mean_close = float(np.nanmean(g['fwd_close']))
            win_rate = float(np.mean(g['fwd_close'] > 0))
            up_rate = float(np.mean(g['fwd_high'] > 0.03))
            down_rate = float(np.mean(g['fwd_low'] < -0.03))
            # Compact quality score [0..1]
            hist_quality = float(np.clip(0.35 + 0.30 * win_rate + 0.20 * up_rate - 0.15 * down_rate + 0.10 * np.tanh(mean_close * 5), 0.0, 1.0))
            hist_setup_stats[setup_name] = {
                'mean_fwd_close': mean_close,
                'win_rate': win_rate,
                'up_rate': up_rate,
                'down_rate': down_rate,
                'quality': hist_quality,
            }

        out[h] = HorizonPack(up50, up90, low10, low50, clf, val_dir_acc, val_mae_up, val_mae_low, hist_setup_stats)
    return out


def score_plan(df: pd.DataFrame, plan: Dict[str, float], setup: str, pack: HorizonPack, horizon: int) -> Dict[str, float]:
    last = df.iloc[-1]
    close = float(last['Close'])
    atr_pct = float(last.get('atr_pct', np.nan))
    rsi_v = float(last.get('rsi14', np.nan))
    adx_v = float(last.get('adx14', np.nan))
    vol_ratio = float(last.get('vol_ratio20', np.nan))
    trend_slope = float(last.get('trend_slope10', np.nan))
    rr1 = float(plan['rr1'])
    rr2 = float(plan['rr2'])
    prob_up = float(plan['prob_up'])
    prob_entry = float(plan['prob_entry'])
    prob_tp1 = float(plan['prob_tp1'])
    prob_tp2 = float(plan['prob_tp2'])
    hist_quality = float(pack.hist_setup_stats.get(setup, {}).get('quality', 0.5))

    trend_score = 0.0
    if np.isfinite(trend_slope):
        trend_score += np.clip((trend_slope / close) * 1000, -2, 2)
    if np.isfinite(adx_v):
        trend_score += np.clip((adx_v - 15) / 10, -1, 2)
    if np.isfinite(rsi_v):
        trend_score += np.clip((rsi_v - 45) / 12, -1, 1)
    vol_score = 0.0 if not np.isfinite(vol_ratio) else np.clip((vol_ratio - 1.0), -0.5, 2.0)
    structure_score = {'PULLBACK': 1.0, 'UNICORN': 1.25, 'SNIPER': 1.35, 'REVERSAL': 0.55}.get(setup, 0.5)
    regime_penalty = 0.0 if not np.isfinite(atr_pct) else np.clip((atr_pct - 0.04) * 10, 0.0, 1.8)
    model_quality = float(np.clip(0.5 * pack.val_dir_acc + 0.5 * (1.0 / (1.0 + pack.val_mae_up + pack.val_mae_low)), 0.0, 1.0))

    score = (
        18
        + 10 * trend_score
        + 9 * structure_score
        + 8 * vol_score
        + 6 * np.clip(rr1, 0, 4)
        + 5 * np.clip(rr2, 0, 6)
        + 10 * hist_quality
        + 8 * model_quality
        + 8 * (prob_up - 0.5)
        + 4 * (prob_tp1 - 0.5)
        + 3 * (prob_tp2 - 0.5)
        - 8 * regime_penalty
    )
    score = float(np.clip(score, 0, 100))

    expected_move_from_current = float((plan['entry_mid'] * (1 + prob_up * max(plan['up50'], 0.0))) - close)
    expectancy_r = float((prob_tp1 * rr1) - ((1 - prob_tp1) * 1.0))
    forecast_confidence = float(np.clip(0.35 * prob_up + 0.25 * hist_quality + 0.20 * model_quality + 0.20 * np.clip(rr1 / 2.5, 0, 1), 0.0, 1.0))

    return {
        'score': score,
        'prob_up': prob_up,
        'prob_entry': prob_entry,
        'prob_tp1': prob_tp1,
        'prob_tp2': prob_tp2,
        'hist_quality': hist_quality,
        'model_quality': model_quality,
        'forecast_confidence': forecast_confidence,
        'expected_move_from_current': expected_move_from_current,
        'expectancy_r': expectancy_r,
    }


def build_plan(df: pd.DataFrame, pred: Dict[str, float], setup: str, pack: HorizonPack, horizon: int) -> Dict[str, float]:
    last = df.iloc[-1]
    close = float(last['Close'])
    atr_v = float(last.get('atr14', np.nan))
    ema20_v = float(last.get('ema20', np.nan))
    ema50_v = float(last.get('ema50', np.nan))
    ll20 = float(last.get('ll20', np.nan))
    hh20 = float(last.get('hh20', np.nan))
    swing_low = recent_swing_low(df, 20)
    swing_high = recent_swing_high(df, 20)
    low_10 = recent_swing_low(df, 10)
    high_10 = recent_swing_high(df, 10)
    support_anchor = finite_min([ll20, ema20_v, ema50_v, swing_low, low_10], default=close * 0.97)
    resistance_anchor = finite_max([hh20, ema20_v, ema50_v, swing_high, high_10], default=close * 1.03)
    atr_safe = atr_v if np.isfinite(atr_v) and atr_v > 0 else close * 0.03
    if setup == 'PULLBACK':
        entry_low = finite_max([support_anchor, close - 0.65 * atr_safe], default=close * 0.97)
        entry_high = finite_min([close + 0.10 * atr_safe, ema20_v * 1.01 if np.isfinite(ema20_v) else np.nan], default=max(entry_low * 1.01, close))
    elif setup == 'UNICORN':
        entry_low = finite_max([support_anchor, close - 0.55 * atr_safe], default=close * 0.97)
        entry_high = finite_min([close + 0.12 * atr_safe, ema20_v * 1.012 if np.isfinite(ema20_v) else np.nan], default=max(entry_low * 1.01, close))
    elif setup == 'SNIPER':
        entry_low = finite_max([close - 0.30 * atr_safe, ema20_v if np.isfinite(ema20_v) else np.nan], default=close * 0.985)
        entry_high = finite_min([resistance_anchor * 1.006, close + 0.18 * atr_safe], default=max(entry_low * 1.01, close))
    else:
        entry_low = finite_max([low_10, close - 0.40 * atr_safe], default=close * 0.98)
        entry_high = finite_min([close + 0.12 * atr_safe, support_anchor * 1.02 if np.isfinite(support_anchor) else np.nan], default=max(entry_low * 1.01, close))
    if not np.isfinite(entry_low):
        entry_low = close * 0.995
    if not np.isfinite(entry_high) or entry_high <= entry_low:
        entry_high = max(entry_low * 1.01, close)
    # Make entry zone slightly more precise in conservative mode.
    zone_w = max(entry_high - entry_low, atr_safe * 0.12)
    entry_mid = float((entry_low + entry_high) / 2.0)

    up50 = float(pred['up50'])
    up90 = float(pred['up90'])
    low10 = float(pred['low10'])
    low50 = float(pred['low50'])
    prob_up = float(pred['prob_up'])

    tp1 = max(entry_mid * (1 + max(up50, 0.0)), resistance_anchor if np.isfinite(resistance_anchor) else entry_mid * 1.02)
    tp2 = max(entry_mid * (1 + max(up90, 0.0)), tp1 * CFG['tp_mult'])
    structural_stop = finite_min([support_anchor, low10 * 0.995 if np.isfinite(low10) else np.nan], default=close * 0.97)
    stop_price = min(entry_low - 0.35 * atr_safe * CFG['stop_pad'], structural_stop, entry_low * 0.998)
    if not np.isfinite(stop_price) or stop_price >= entry_low:
        stop_price = entry_low * 0.97
    risk = max(entry_mid - stop_price, 1e-9)
    rr1 = (tp1 - entry_mid) / risk
    rr2 = (tp2 - entry_mid) / risk
    stop_pct = (stop_price / entry_mid) - 1.0

    scenario = 'bullish_retrace_then_rebound' if prob_up >= 0.58 and entry_mid < close else ('bullish_breakout_continuation' if prob_up >= 0.58 else ('bearish_or_range' if prob_up <= 0.42 else 'range_consolidation'))

    return {
        'best_horizon': int(horizon),
        'scenario': scenario,
        'entry_low': float(entry_low),
        'entry_high': float(entry_high),
        'entry_mid': float(entry_mid),
        'entry_zone_width': float(zone_w),
        'stop_price': float(stop_price),
        'stop_pct': float(stop_pct),
        'tp1': float(tp1),
        'tp2': float(tp2),
        'risk': float(risk),
        'rr1': float(rr1),
        'rr2': float(rr2),
        'up50': up50,
        'up90': up90,
        'low10': low10,
        'low50': low50,
    }


def predict_for_ticker(df: pd.DataFrame, models: Dict[int, HorizonPack], horizons: List[int]) -> Dict:
    feat = compute_features(df)
    if feat.empty:
        return {}
    last = feat.iloc[-1]
    X = feat[feature_columns()].iloc[[-1]]
    pred_by_h: Dict[int, Dict[str, float]] = {}
    for h in horizons:
        pack = models.get(h)
        if pack is None:
            continue
        up50 = float(pack.up_q50.predict(X)[0])
        up90 = float(pack.up_q90.predict(X)[0])
        low10 = float(pack.low_q10.predict(X)[0])
        low50 = float(pack.low_q50.predict(X)[0])
        prob_up = float(pack.clf_up.predict_proba(X)[0, 1])
        pred_by_h[h] = {
            'up50': up50,
            'up90': up90,
            'low10': low10,
            'low50': low50,
            'prob_up': prob_up,
        }
    if not pred_by_h:
        return {}

    setup, setup_note = classify_setup_row(last, float(feat['High'].shift(1).rolling(3).max().iloc[-1]) if len(feat) > 4 else np.nan)
    best_h = sorted(pred_by_h.keys(), key=lambda h: pred_by_h[h]['prob_up'] * (pred_by_h[h]['up50'] - pred_by_h[h]['low50']), reverse=True)[0]
    pack = models[best_h]
    plan = build_plan(feat, pred_by_h[best_h], setup, pack, best_h)
    metrics = score_plan(feat, plan, setup, pack, best_h)

    # Hard quality gate: keeps the Top 20 from being filled with weak names.
    if metrics['prob_up'] < CFG['min_prob'] or plan['rr1'] < CFG['min_rr']:
        return {
            'symbol': str(last.get('Symbol', '')),
            'date': feat.index[-1],
            'close': float(last['Close']),
            'setup': setup,
            'setup_note': setup_note,
            'best_horizon': plan['best_horizon'],
            'entry_low': plan['entry_low'],
            'entry_high': plan['entry_high'],
            'entry_mid': plan['entry_mid'],
            'stop_price': plan['stop_price'],
            'stop_pct': plan['stop_pct'],
            'tp1': plan['tp1'],
            'tp2': plan['tp2'],
            'rr1': plan['rr1'],
            'rr2': plan['rr2'],
            'prob_up': metrics['prob_up'],
            'prob_entry': metrics['prob_entry'],
            'prob_tp1': metrics['prob_tp1'],
            'prob_tp2': metrics['prob_tp2'],
            'score': float(metrics['score'] * 0.72),
            'expectancy_r': metrics['expectancy_r'],
            'forecast_confidence': metrics['forecast_confidence'],
            'hist_quality': metrics['hist_quality'],
            'model_quality': metrics['model_quality'],
            'scenario': plan['scenario'],
            'flag': 'LOW_QUALITY',
            'ema20': float(last.get('ema20', np.nan)),
            'ema50': float(last.get('ema50', np.nan)),
            'ema200': float(last.get('ema200', np.nan)),
            'rsi14': float(last.get('rsi14', np.nan)),
            'adx14': float(last.get('adx14', np.nan)),
            'vol_ratio20': float(last.get('vol_ratio20', np.nan)),
            'atr_pct': float(last.get('atr_pct', np.nan)),
            'trend_slope10': float(last.get('trend_slope10', np.nan)),
        }

    return {
        'symbol': str(last.get('Symbol', '')),
        'date': feat.index[-1],
        'close': float(last['Close']),
        'setup': setup,
        'setup_note': setup_note,
        **plan,
        **metrics,
        'flag': 'OK',
        'ema20': float(last.get('ema20', np.nan)),
        'ema50': float(last.get('ema50', np.nan)),
        'ema200': float(last.get('ema200', np.nan)),
        'rsi14': float(last.get('rsi14', np.nan)),
        'adx14': float(last.get('adx14', np.nan)),
        'vol_ratio20': float(last.get('vol_ratio20', np.nan)),
        'atr_pct': float(last.get('atr_pct', np.nan)),
        'trend_slope10': float(last.get('trend_slope10', np.nan)),
    }


def chart_ticker(df: pd.DataFrame, row: Dict):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'
    ))
    for name, value, color in [
        ('Entry low', row['entry_low'], '#2ecc71'),
        ('Entry high', row['entry_high'], '#27ae60'),
        ('Stop', row['stop_price'], '#e74c3c'),
        ('TP1', row['tp1'], '#3498db'),
        ('TP2', row['tp2'], '#8e44ad'),
    ]:
        fig.add_hline(y=float(value), line_width=1, line_dash='dot', line_color=color, annotation_text=name)
    fig.update_layout(height=540, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


csv_df = None
if uploaded is not None:
    try:
        csv_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f'CSV tidak bisa dibaca: {exc}')
        st.stop()

local_frames = load_ohlcv_frames(csv_df) if csv_df is not None else {}
if source == 'Upload CSV' and local_frames:
    st.success(f'CSV OHLCV terdeteksi: {len(local_frames)} ticker siap diproses tanpa download internet.')

tickers = parse_universe(paste_text if source == 'Paste tickers' else None, csv_df if not local_frames else None)
if local_frames:
    if tickers:
        tickers = [t for t in tickers if t in local_frames or normalize_ticker(t) in local_frames]
        if not tickers:
            tickers = list(local_frames.keys())
    else:
        tickers = list(local_frames.keys())
tickers = tickers[:max_tickers]


def run_scan() -> dict | None:
    if source == 'Upload CSV' and not local_frames and (csv_df is None or csv_df.empty):
        st.error('CSV kosong atau tidak valid.')
        return None
    if not tickers:
        st.error('Tidak ada ticker valid.')
        return None

    st.write(f'Scanning {len(tickers)} ticker...')
    frames: Dict[str, pd.DataFrame] = {}
    errs: Dict[str, str] = {}

    if local_frames:
        for sym in tickers:
            key = sym if sym in local_frames else normalize_ticker(sym)
            if key in local_frames:
                frames[sym] = local_frames[key].tail(700).copy()
            else:
                errs[sym] = 'not_found_in_csv'
    else:
        period = f'{period_months}mo'
        prog = st.progress(0)
        total = len(tickers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(download_one, sym, period): sym for sym in tickers}
            done = 0
            for fut in as_completed(futures):
                sym, df, err = fut.result()
                if not df.empty:
                    frames[sym] = df.tail(700).copy()
                else:
                    errs[sym] = err
                done += 1
                prog.progress(done / total)
        prog.empty()

    if not frames:
        st.error('Tidak ada data yang berhasil diproses.')
        if errs:
            with st.expander('Detail ticker gagal'):
                st.write(pd.DataFrame([{'symbol': k, 'error': v} for k, v in errs.items()]))
        st.info('Solusi paling stabil: upload CSV OHLCV dari export broker/Stockbit.')
        return {'frames': {}, 'filtered_frames': {}, 'out': pd.DataFrame(), 'top20': pd.DataFrame(), 'errs': errs}

    filtered_frames: Dict[str, pd.DataFrame] = {}
    for sym, df in frames.items():
        feat = compute_features(df)
        last = feat.iloc[-1]
        close = float(last['Close'])
        avg_vol = float(feat['Volume'].tail(20).mean())
        if close < min_price or close > max_price:
            continue
        if np.isfinite(avg_vol) and avg_vol < min_avg_vol:
            continue
        filtered_frames[sym] = df

    if not filtered_frames:
        st.warning('Semua ticker tersaring oleh harga/volume filter.')
        return {'frames': frames, 'filtered_frames': {}, 'out': pd.DataFrame(), 'top20': pd.DataFrame(), 'errs': errs}

    pooled = build_dataset(filtered_frames, lookbacks)
    if pooled.empty:
        st.error('Dataset training kosong.')
        return {'frames': frames, 'filtered_frames': filtered_frames, 'out': pd.DataFrame(), 'top20': pd.DataFrame(), 'errs': errs}

    models = train_models(pooled, lookbacks)
    if not models:
        st.error('Tidak cukup data untuk melatih model.')
        return {'frames': frames, 'filtered_frames': filtered_frames, 'out': pd.DataFrame(), 'top20': pd.DataFrame(), 'errs': errs}

    rows = []
    for sym, df in filtered_frames.items():
        res = predict_for_ticker(df, models, lookbacks)
        if not res:
            continue
        rows.append(res)

    if not rows:
        st.warning('Tidak ada hasil prediksi.')
        return {'frames': frames, 'filtered_frames': filtered_frames, 'out': pd.DataFrame(), 'top20': pd.DataFrame(), 'errs': errs}

    out = pd.DataFrame(rows)
    # Keep all names but sort by quality. Better entries rise to top, weak ones stay but lower.
    out = out.sort_values(['score', 'forecast_confidence', 'prob_tp1', 'rr1'], ascending=False).reset_index(drop=True)
    top20 = out.head(20).copy()
    return {'frames': frames, 'filtered_frames': filtered_frames, 'out': out, 'top20': top20, 'errs': errs}


if run:
    bundle = run_scan()
    if bundle is not None:
        st.session_state.scan_bundle = bundle
        if not bundle['top20'].empty:
            st.session_state.selected_symbol = str(bundle['top20'].iloc[0]['symbol'])

bundle = st.session_state.scan_bundle
if bundle is None:
    st.info("Isi universe ticker lalu klik 'Run predictive scan'.")
    st.write('Jika source online gagal di deploy, gunakan Upload CSV OHLCV.')
    st.stop()

out = bundle.get('out', pd.DataFrame())
top20 = bundle.get('top20', pd.DataFrame())
filtered_frames = bundle.get('filtered_frames', {})
errs = bundle.get('errs', {})

if top20 is None or top20.empty:
    st.warning('Belum ada hasil scan tersimpan.')
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric('Scanned', len(filtered_frames))
c2.metric('Valid setups', len(out))
c3.metric('Top score', f"{top20.iloc[0]['score']:.1f}")
c4.metric('Median RR1', f"{out['rr1'].median():.2f}")

st.subheader('Top 20 predictive setups')
show_cols = ['symbol', 'setup', 'flag', 'score', 'close', 'entry_low', 'entry_high', 'stop_price', 'tp1', 'tp2', 'best_horizon', 'prob_up', 'prob_entry', 'prob_tp1', 'prob_tp2', 'rr1', 'rr2', 'hist_quality', 'model_quality', 'forecast_confidence', 'scenario', 'setup_note']
display_df = top20.copy()
display_df['stop_pct'] = (display_df['stop_pct'] * 100).round(2)
display_df['prob_up'] = (display_df['prob_up'] * 100).round(1)
display_df['prob_entry'] = (display_df['prob_entry'] * 100).round(1)
display_df['prob_tp1'] = (display_df['prob_tp1'] * 100).round(1)
display_df['prob_tp2'] = (display_df['prob_tp2'] * 100).round(1)
st.dataframe(display_df[['symbol', 'setup', 'flag', 'score', 'close', 'entry_low', 'entry_high', 'stop_price', 'tp1', 'tp2', 'best_horizon', 'prob_up', 'prob_entry', 'prob_tp1', 'prob_tp2', 'rr1', 'rr2', 'hist_quality', 'model_quality', 'forecast_confidence', 'scenario', 'setup_note']], use_container_width=True, hide_index=True)

symbols = top20['symbol'].tolist()
if st.session_state.selected_symbol not in symbols:
    st.session_state.selected_symbol = symbols[0]
pick = st.selectbox('Lihat detail ticker', symbols, key='selected_symbol')
row = top20[top20['symbol'] == pick].iloc[0].to_dict()

st.subheader(f'Detail {pick}')
st.write(
    f"Setup: **{row['setup']}** | Scenario: **{row['scenario']}** | Horizon terbaik: **{int(row['best_horizon'])} hari** | "
    f"Entry: **{row['entry_low']:.2f} - {row['entry_high']:.2f}** | Stop: **{row['stop_price']:.2f}** ({row['stop_pct']*100:.2f}%) | "
    f"TP1: **{row['tp1']:.2f}** | TP2: **{row['tp2']:.2f}** | Prob up: **{row['prob_up']*100:.1f}%**"
)

chart_df = filtered_frames.get(pick)
if chart_df is not None and not chart_df.empty:
    chart_ticker(chart_df.tail(180), row)

st.subheader('Ringkasan kualitas')
qc1, qc2, qc3, qc4 = st.columns(4)
qc1.metric('Forecast confidence', f"{row['forecast_confidence']*100:.1f}%")
qc2.metric('Historical quality', f"{row['hist_quality']*100:.1f}%")
qc3.metric('Model quality', f"{row['model_quality']*100:.1f}%")
qc4.metric('Expectancy (R)', f"{row['expectancy_r']:.2f}")

st.caption('Scanner ini memberi arah probabilistik dan entry zone yang lolos filter historis, bukan kepastian profit.')

if errs:
    with st.expander('Detail ticker gagal'):
        st.write(pd.DataFrame([{'symbol': k, 'error': v} for k, v in errs.items()]))
