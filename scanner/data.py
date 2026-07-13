from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import BinaryIO, Iterable

import pandas as pd

from .models import DownloadReport


TICKER_COLUMNS = ("ticker", "tickers", "symbol", "symbols", "kode", "code", "emiten", "stock")


def normalize_idx_ticker(value: object) -> str | None:
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "NULL", "TICKER"}:
        return None
    text = re.sub(r"\s+", "", text)
    text = text.replace(".IDX", "").replace("IDX:", "")
    if text.endswith(".JK"):
        base = text[:-3]
    else:
        base = text
    if not re.fullmatch(r"[A-Z0-9]{3,8}", base):
        return None
    return f"{base}.JK"


def parse_ticker_csv(source: bytes | BinaryIO | pd.DataFrame, max_tickers: int = 500) -> list[str]:
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        payload = BytesIO(source) if isinstance(source, bytes) else source
        try:
            frame = pd.read_csv(payload, sep=None, engine="python")
        except UnicodeDecodeError:
            if hasattr(payload, "seek"):
                payload.seek(0)
            frame = pd.read_csv(payload, encoding="latin-1", sep=None, engine="python")
    if frame.empty or len(frame.columns) == 0:
        return []
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    selected = next((lookup[name] for name in TICKER_COLUMNS if name in lookup), frame.columns[0])
    result: list[str] = []
    seen: set[str] = set()
    for value in frame[selected].tolist():
        ticker = normalize_idx_ticker(value)
        if ticker and ticker not in seen:
            result.append(ticker)
            seen.add(ticker)
        if len(result) >= max_tickers:
            break
    return result


def _clean_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out.columns = [str(c).title() for c in out.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in out.columns for c in required):
        return pd.DataFrame()
    out = out[required]
    out.index = pd.to_datetime(out.index, errors="coerce")
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out["Volume"] = out["Volume"].fillna(0.0)
    return out


def _extract_batch(raw: pd.DataFrame, ticker: str, total: int) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if not isinstance(raw.columns, pd.MultiIndex):
        return _clean_ohlcv(raw) if total == 1 else pd.DataFrame()
    level0 = set(map(str, raw.columns.get_level_values(0)))
    level1 = set(map(str, raw.columns.get_level_values(1)))
    try:
        if ticker in level0:
            return _clean_ohlcv(raw[ticker])
        if ticker in level1:
            return _clean_ohlcv(raw.xs(ticker, axis=1, level=1))
    except (KeyError, ValueError):
        return pd.DataFrame()
    return pd.DataFrame()


def download_ohlcv(
    tickers: Iterable[str], period: str = "3y", batch_size: int = 50
) -> tuple[dict[str, pd.DataFrame], DownloadReport]:
    import yfinance as yf

    requested = list(dict.fromkeys(tickers))
    histories: dict[str, pd.DataFrame] = {}
    failed: dict[str, str] = {}
    for start in range(0, len(requested), batch_size):
        batch = requested[start : start + batch_size]
        try:
            raw = yf.download(
                batch,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                repair=False,
                actions=False,
                threads=True,
                progress=False,
                timeout=25,
            )
            for ticker in batch:
                frame = _extract_batch(raw, ticker, len(batch))
                if not frame.empty:
                    histories[ticker] = frame
                else:
                    failed[ticker] = "Data batch kosong"
        except Exception as exc:  # network providers can fail per batch
            for ticker in batch:
                failed[ticker] = f"Batch gagal: {type(exc).__name__}"

    missing = [t for t in requested if t not in histories]

    def retry_one(ticker: str) -> tuple[str, pd.DataFrame, str | None]:
        try:
            frame = yf.Ticker(ticker).history(
                period=period, interval="1d", auto_adjust=True, repair=False, actions=False, timeout=20
            )
            clean = _clean_ohlcv(frame)
            return ticker, clean, None if not clean.empty else "Data individual kosong"
        except Exception as exc:
            return ticker, pd.DataFrame(), f"{type(exc).__name__}: {str(exc)[:100]}"

    if missing:
        with ThreadPoolExecutor(max_workers=min(6, len(missing))) as pool:
            futures = [pool.submit(retry_one, ticker) for ticker in missing]
            for future in as_completed(futures):
                ticker, frame, error = future.result()
                if not frame.empty:
                    histories[ticker] = frame
                    failed.pop(ticker, None)
                else:
                    failed[ticker] = error or "Tidak ada data"

    report = DownloadReport(requested, sorted(histories), failed)
    return histories, report


def download_benchmark(period: str = "3y") -> pd.DataFrame:
    import yfinance as yf

    try:
        frame = yf.Ticker("^JKSE").history(
            period=period, interval="1d", auto_adjust=True, repair=False, actions=False, timeout=20
        )
        return _clean_ohlcv(frame)
    except Exception:
        return pd.DataFrame()
