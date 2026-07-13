from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

from .models import SetupPlan
from .price_rules import idx_tick_size, round_idx_price


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _last_true_index(mask: pd.Series, lookback: int) -> object | None:
    recent = mask.fillna(False).iloc[-lookback:]
    hits = recent[recent]
    return hits.index[-1] if len(hits) else None


def _bars_since(df: pd.DataFrame, timestamp: object | None) -> int | None:
    if timestamp is None:
        return None
    locations = np.flatnonzero(df.index == timestamp)
    if len(locations) == 0:
        return None
    return int(len(df) - 1 - locations[-1])


def _distance_to_zone(close: float, zone_low: float, zone_high: float, atr: float) -> float:
    if atr <= 0:
        return float("inf")
    if zone_low <= close <= zone_high:
        return 0.0
    if close > zone_high:
        return (close - zone_high) / atr
    return (zone_low - close) / atr


def _plan_prices(
    plan: SetupPlan,
    atr_value: float,
    raw_entry: float,
    raw_stop: float,
    tp1_rr: float = 1.5,
    tp2_rr: float = 2.5,
) -> SetupPlan:
    plan.entry_low = round_idx_price(plan.entry_low, "down")
    plan.entry_high = round_idx_price(plan.entry_high, "up")
    plan.entry = round_idx_price(raw_entry, "up")
    plan.trigger = round_idx_price(plan.trigger if plan.trigger is not None else raw_entry, "up")
    plan.stop_loss = round_idx_price(raw_stop, "down")
    if plan.entry is None or plan.stop_loss is None or plan.stop_loss >= plan.entry:
        plan.invalidated = True
        plan.reason = "Struktur tidak menghasilkan risiko positif yang valid"
        return plan
    risk = plan.entry - plan.stop_loss
    plan.tp1 = round_idx_price(plan.entry + tp1_rr * risk, "down")
    plan.tp2 = round_idx_price(plan.entry + tp2_rr * risk, "down")
    if plan.tp1 is not None:
        plan.rr1 = round((plan.tp1 - plan.entry) / risk, 2)
    if plan.tp2 is not None:
        plan.rr2 = round((plan.tp2 - plan.entry) / risk, 2)
    return plan


def detect_pullback_continuation(df: pd.DataFrame, ticker: str) -> SetupPlan:
    name = "PULLBACK_CONTINUATION"
    plan = SetupPlan(ticker=ticker, setup=name, detected=False, setup_score=0.0)
    if len(df) < 205:
        plan.reason = "Data tren jangka panjang belum cukup"
        return plan
    row = df.iloc[-1]
    prev = df.iloc[-2]
    close = _finite(row["Close"])
    atr_v = _finite(row["ATR14"])
    if close <= 0 or atr_v <= 0:
        plan.reason = "ATR/harga tidak valid"
        return plan

    ema20, ema50, ema200 = (_finite(row[x]) for x in ("EMA20", "EMA50", "EMA200"))
    trend = ema20 > ema50 > ema200 and close > ema50 and ema20 > _finite(df["EMA20"].iloc[-11])
    momentum = _finite(row["ROC60"], -1) > 0.04 and _finite(row["DIST_52W_HIGH"], -1) > -0.18
    support = ema20 if close >= ema20 - 0.45 * atr_v else max(ema50, _finite(row["VWAP20"]))
    recent = df.iloc[-5:]
    touched = bool((recent["Low"] <= recent["EMA20"] + 0.35 * recent["ATR14"]).any())
    held = close >= ema50 - 0.25 * atr_v
    pullback = trend and touched and held
    confirmation = bool(row["BULL_REJECTION"]) or (
        close > _finite(prev["High"]) and close > _finite(row["Open"])
    )
    vol_contract = _finite(recent["Volume"].iloc[:-1].mean()) < 0.92 * _finite(df["VOL_MA20"].iloc[-1], 1)
    relative = _finite(row["REL_STRENGTH60"], 0) > 0

    score = 0.0
    score += 25 if trend else 0
    score += 15 if momentum else 0
    score += 18 if pullback else (8 if trend and held else 0)
    score += 10 if vol_contract else 4
    score += 12 if confirmation else 3
    score += 10 if _finite(row["CMF20"]) > -0.03 else 0
    score += 10 if relative else 4
    plan.setup_score = min(100.0, score)
    plan.detected = bool(trend and momentum and pullback)
    if not plan.detected:
        plan.reason = "Belum memenuhi kombinasi uptrend, momentum, dan pullback ke value area"
        return plan

    touch_mask = (df["Low"] <= df["EMA20"] + 0.35 * df["ATR14"]) & (df["Close"] >= df["EMA50"])
    created = _last_true_index(touch_mask, 10) or df.index[-1]
    zone_low = support - 0.35 * atr_v
    zone_high = support + 0.40 * atr_v
    recent_low = _finite(df["Low"].iloc[-7:].min())
    pivot_low = _finite(row["LAST_PIVOT_LOW"], recent_low)
    structural_low = min(recent_low, pivot_low) if pivot_low > close - 4 * atr_v else recent_low
    raw_stop = structural_low - 0.20 * atr_v
    if confirmation and _distance_to_zone(close, zone_low, zone_high, atr_v) <= 0.5:
        raw_entry = max(close, _finite(row["High"]) + idx_tick_size(close))
        plan.entry_type = "BUY_STOP_CONFIRMATION"
        plan.action = "READY_TRIGGER"
    else:
        raw_entry = (zone_low + zone_high) / 2
        plan.entry_type = "LIMIT_ON_PULLBACK_THEN_CONFIRM"
        plan.action = "WAIT_PULLBACK_CONFIRMATION"
    plan.signal_date = df.index[-1]
    plan.zone_created_date = created
    plan.zone_age_bars = _bars_since(df, created)
    plan.valid_until = pd.Timestamp(df.index[-1]) + pd.offsets.BDay(10)
    plan.entry_low, plan.entry_high = zone_low, zone_high
    plan.trigger = _finite(row["High"]) + idx_tick_size(close)
    plan.distance_atr = round(_distance_to_zone(close, zone_low, zone_high, atr_v), 2)
    plan.evidence = [
        "EMA20 > EMA50 > EMA200",
        "Momentum 3 bulan positif",
        "Pullback menyentuh value area",
    ]
    if vol_contract:
        plan.evidence.append("Volume mengecil saat pullback")
    if confirmation:
        plan.evidence.append("Ada reclaim/rejection bullish")
    plan.reason = "Kelanjutan tren setelah pullback terkontrol"
    return _plan_prices(plan, atr_v, raw_entry, raw_stop)


def detect_breakout_retest(df: pd.DataFrame, ticker: str) -> SetupPlan:
    name = "BREAKOUT_RETEST"
    plan = SetupPlan(ticker=ticker, setup=name, detected=False, setup_score=0.0)
    if len(df) < 205:
        plan.reason = "Data belum cukup"
        return plan
    row = df.iloc[-1]
    close, atr_v = _finite(row["Close"]), _finite(row["ATR14"])
    if close <= 0 or atr_v <= 0:
        plan.reason = "ATR/harga tidak valid"
        return plan
    breakout_mask = (
        (df["Close"] > df["HIGH55_PREV"] + 0.05 * df["ATR14"])
        & (df["VOL_RATIO"] >= 1.25)
        & (df["BODY_ATR"] >= 0.40)
        & (df["Close"] > df["Open"])
    )
    breakout_date = _last_true_index(breakout_mask, 18)
    if breakout_date is None:
        plan.reason = "Belum ada breakout 55-hari dengan volume dan displacement"
        return plan
    pos = int(np.flatnonzero(df.index == breakout_date)[-1])
    breakout_row = df.iloc[pos]
    resistance = _finite(breakout_row["HIGH55_PREV"])
    breakout_atr = _finite(breakout_row["ATR14"], atr_v)
    post = df.iloc[pos + 1 :] if pos + 1 < len(df) else df.iloc[0:0]
    retest_mask = (
        (post["Low"] <= resistance + 0.45 * post["ATR14"])
        & (post["Low"] >= resistance - 1.0 * post["ATR14"])
        & (post["Close"] >= resistance - 0.10 * post["ATR14"])
    )
    retest_date = _last_true_index(retest_mask, min(12, len(post))) if not post.empty else None
    invalidated = bool((post["Close"] < resistance - 1.15 * post["ATR14"]).any()) if not post.empty else False
    confirmation = False
    retest_low = resistance - 0.6 * atr_v
    if retest_date is not None:
        rpos = int(np.flatnonzero(df.index == retest_date)[-1])
        retest_low = _finite(df["Low"].iloc[max(pos + 1, rpos - 2) : rpos + 1].min(), retest_low)
        latest_retest = df.iloc[rpos]
        confirmation = bool(latest_retest["BULL_REJECTION"]) or _finite(latest_retest["Close"]) > resistance
    trend = _finite(row["EMA20"]) > _finite(row["EMA50"]) > _finite(row["EMA200"])
    relative = _finite(row["REL_STRENGTH60"], 0) > 0
    breakout_quality = min(1.0, _finite(breakout_row["VOL_RATIO"]) / 2.0)
    score = 20 * float(trend) + 25 * breakout_quality + 15 * min(1.0, _finite(breakout_row["BODY_ATR"]))
    score += 22 if retest_date is not None else 6
    score += 10 if confirmation else 2
    score += 8 if relative else 3
    plan.setup_score = round(min(100.0, score), 1)
    plan.detected = not invalidated
    plan.invalidated = invalidated
    if invalidated:
        plan.reason = "Breakout sudah gagal: penutupan menembus bawah level invalidasi"
        return plan

    zone_low = resistance - 0.35 * atr_v
    zone_high = resistance + 0.35 * atr_v
    in_retest_area = _distance_to_zone(close, zone_low, zone_high, atr_v) <= 0.45
    if retest_date is not None and confirmation and in_retest_area:
        raw_entry = max(close, _finite(row["High"]) + idx_tick_size(close))
        plan.entry_type = "BUY_STOP_AFTER_RETEST"
        plan.action = "READY_TRIGGER"
    else:
        raw_entry = resistance + 0.10 * atr_v
        plan.entry_type = "LIMIT_RETEST_WITH_RECLAIM"
        plan.action = "WAIT_RETEST"
    raw_stop = min(retest_low, resistance - 0.70 * atr_v) - 0.15 * atr_v
    plan.signal_date = breakout_date
    plan.zone_created_date = breakout_date
    plan.zone_age_bars = _bars_since(df, breakout_date)
    plan.valid_until = pd.Timestamp(breakout_date) + pd.offsets.BDay(25)
    plan.entry_low, plan.entry_high = zone_low, zone_high
    plan.trigger = max(resistance, _finite(row["High"])) + idx_tick_size(close)
    plan.distance_atr = round(_distance_to_zone(close, zone_low, zone_high, atr_v), 2)
    plan.evidence = [
        "Breakout high 55-hari",
        f"Volume breakout {_finite(breakout_row['VOL_RATIO']):.2f}x",
        "Displacement bullish",
    ]
    if retest_date is not None:
        plan.evidence.append("Retest level breakout terdeteksi")
    if confirmation:
        plan.evidence.append("Retest ditutup dengan reclaim")
    plan.reason = "Breakout tervalidasi; eksekusi hanya setelah retest/reclaim"
    return _plan_prices(plan, atr_v, raw_entry, raw_stop)


def detect_reversal_accumulation(df: pd.DataFrame, ticker: str) -> SetupPlan:
    name = "REVERSAL_ACCUMULATION"
    plan = SetupPlan(ticker=ticker, setup=name, detected=False, setup_score=0.0)
    if len(df) < 205:
        plan.reason = "Data belum cukup"
        return plan
    row = df.iloc[-1]
    close, atr_v = _finite(row["Close"]), _finite(row["ATR14"])
    if close <= 0 or atr_v <= 0:
        plan.reason = "ATR/harga tidak valid"
        return plan
    prior = df.iloc[-150:-30]
    base = df.iloc[-30:]
    prior_high = _finite(prior["High"].max(), close)
    base_low, base_high = _finite(base["Low"].min(), close), _finite(base["High"].max(), close)
    decline = base_low / prior_high - 1 if prior_high > 0 else 0
    base_width = (base_high - base_low) / close
    based = decline <= -0.12 and base_width <= 0.32
    contraction = _finite(row["RANGE_CONTRACTION20"], 2) <= 0.95
    accumulation = (
        _finite(base["CMF20"].iloc[-10:].mean()) > 0.02
        and _finite(row["OBV_SLOPE10"]) > 0
    )
    sweep_mask = (
        (df["Low"] < df["LOW20_PREV"])
        & (df["Close"] > df["LOW20_PREV"])
        & (df["CLOSE_LOCATION"] > 0.58)
    )
    sweep_date = _last_true_index(sweep_mask, 25)
    if sweep_date is not None:
        spos = int(np.flatnonzero(df.index == sweep_date)[-1])
        sweep_low = _finite(df.iloc[spos]["Low"])
    else:
        spos, sweep_low = len(df) - 30, base_low
    choch_mask = (
        (df["Close"] > df["LAST_PIVOT_HIGH"] + 0.05 * df["ATR14"])
        & (df["Close"] > df["EMA20"])
        & (df["VOL_RATIO"] >= 1.05)
    )
    post_choch = choch_mask.iloc[spos:] if sweep_date is not None else choch_mask.iloc[-15:]
    choch_date = _last_true_index(post_choch, len(post_choch)) if len(post_choch) else None
    choch = choch_date is not None
    invalidated = close < sweep_low - 0.20 * atr_v

    score = 0.0
    score += 18 if based else 5
    score += 12 if contraction else 3
    score += 20 if accumulation else (8 if _finite(row["CMF20"]) > 0 else 0)
    score += 20 if sweep_date is not None else 0
    score += 22 if choch else 4
    score += 8 if close > _finite(row["EMA50"]) else 2
    plan.setup_score = min(100.0, score)
    plan.detected = bool(based and accumulation and sweep_date is not None and not invalidated)
    plan.invalidated = invalidated
    if not plan.detected:
        plan.reason = "Belum ada rangkaian decline–base–akumulasi–liquidity sweep yang lengkap"
        return plan

    structure_level = _finite(row["LAST_PIVOT_HIGH"], base_high)
    if choch_date is not None:
        cpos = int(np.flatnonzero(df.index == choch_date)[-1])
        structure_level = _finite(df.iloc[cpos]["LAST_PIVOT_HIGH"], structure_level)
    zone_low = max(_finite(row["EMA20"]), structure_level - 0.45 * atr_v)
    zone_high = structure_level + 0.30 * atr_v
    in_zone = _distance_to_zone(close, zone_low, zone_high, atr_v) <= 0.45
    confirmation = bool(row["BULL_REJECTION"]) or close > _finite(df["High"].iloc[-2])
    if choch and in_zone and confirmation:
        raw_entry = max(close, _finite(row["High"]) + idx_tick_size(close))
        plan.entry_type = "BUY_STOP_AFTER_CHOCH"
        plan.action = "READY_TRIGGER"
    elif choch:
        raw_entry = (zone_low + zone_high) / 2
        plan.entry_type = "LIMIT_ON_CHOCH_RETEST"
        plan.action = "WAIT_RETEST"
    else:
        raw_entry = structure_level + idx_tick_size(structure_level)
        plan.entry_type = "BUY_STOP_AFTER_CHOCH"
        plan.action = "WAIT_CHOCH"
    raw_stop = sweep_low - 0.20 * atr_v
    plan.signal_date = sweep_date
    plan.zone_created_date = choch_date or sweep_date
    plan.zone_age_bars = _bars_since(df, plan.zone_created_date)
    plan.valid_until = pd.Timestamp(sweep_date) + pd.offsets.BDay(30)
    plan.entry_low, plan.entry_high = zone_low, zone_high
    plan.trigger = structure_level + idx_tick_size(structure_level)
    plan.distance_atr = round(_distance_to_zone(close, zone_low, zone_high, atr_v), 2)
    plan.evidence = ["Penurunan diikuti base", "Proxy CMF/OBV menguat", "Sell-side liquidity sweep"]
    if contraction:
        plan.evidence.append("Volatilitas berkontraksi")
    if choch:
        plan.evidence.append("CHOCH/BOS bullish terkonfirmasi")
    plan.reason = "Reversal hanya dapat dieksekusi setelah perubahan struktur bullish"
    return _plan_prices(plan, atr_v, raw_entry, raw_stop, 1.5, 2.7)


def detect_unicorn_sniper(df: pd.DataFrame, ticker: str) -> SetupPlan:
    name = "UNICORN_SNIPER_ICT"
    plan = SetupPlan(ticker=ticker, setup=name, detected=False, setup_score=0.0)
    if len(df) < 120:
        plan.reason = "Data struktur belum cukup"
        return plan
    row = df.iloc[-1]
    close, atr_v = _finite(row["Close"]), _finite(row["ATR14"])
    if close <= 0 or atr_v <= 0:
        plan.reason = "ATR/harga tidak valid"
        return plan
    sweep_mask = (
        (df["Low"] < df["LOW20_PREV"])
        & (df["Close"] > df["LOW20_PREV"])
        & (df["CLOSE_LOCATION"] >= 0.55)
    )
    sweep_date = _last_true_index(sweep_mask, 35)
    if sweep_date is None:
        plan.reason = "Belum ada sell-side liquidity sweep bullish"
        return plan
    spos = int(np.flatnonzero(df.index == sweep_date)[-1])
    sweep_low = _finite(df.iloc[spos]["Low"])
    bos_mask = (
        (df["Close"] > df["LAST_PIVOT_HIGH"] + 0.05 * df["ATR14"])
        & (df["BODY_ATR"] >= 0.55)
        & (df["Close"] > df["Open"])
    )
    bos_post = bos_mask.iloc[spos + 1 :]
    bos_date = _last_true_index(bos_post, min(20, len(bos_post))) if len(bos_post) else None
    if bos_date is None:
        plan.setup_score = 28.0
        plan.reason = "Liquidity sweep ada, tetapi displacement/BOS belum terkonfirmasi"
        return plan
    bpos = int(np.flatnonzero(df.index == bos_date)[-1])
    fvg_window = df.iloc[max(spos + 1, bpos - 2) : min(len(df), bpos + 6)]
    fvg_hits = fvg_window[fvg_window["BULL_FVG"].fillna(False)]
    if fvg_hits.empty:
        plan.setup_score = 50.0
        plan.reason = "Sweep dan BOS ada, tetapi FVG displacement tidak valid"
        return plan
    fvg_date = fvg_hits.index[-1]
    fpos = int(np.flatnonzero(df.index == fvg_date)[-1])
    fvg_low = _finite(df.loc[fvg_date, "FVG_LOW"])
    fvg_high = _finite(df.loc[fvg_date, "FVG_HIGH"])
    if fvg_high <= fvg_low:
        plan.reason = "FVG tidak valid"
        return plan

    # Last down candle before displacement is used as an objective bullish OB proxy.
    search_ob = df.iloc[spos : max(spos + 1, fpos)]
    bear = search_ob[search_ob["BEAR_CANDLE"].fillna(False)]
    ob_overlap = False
    if not bear.empty:
        ob_row = bear.iloc[-1]
        ob_low, ob_high = _finite(ob_row["Low"]), max(_finite(ob_row["Open"]), _finite(ob_row["Close"]))
        overlap_low, overlap_high = max(fvg_low, ob_low), min(fvg_high, ob_high)
        if overlap_high > overlap_low:
            zone_low, zone_high = overlap_low, overlap_high
            ob_overlap = True
        else:
            zone_low, zone_high = fvg_low, fvg_high
    else:
        zone_low, zone_high = fvg_low, fvg_high

    after_fvg = df.iloc[fpos + 1 :]
    invalidated = (
        bool(
            (
                (after_fvg["Close"] < fvg_low - 0.15 * after_fvg["ATR14"])
                | (after_fvg["Close"] < sweep_low)
            ).any()
        )
        if not after_fvg.empty
        else False
    )
    dealing_high = _finite(df["High"].iloc[spos : bpos + 1].max(), close)
    equilibrium = (sweep_low + dealing_high) / 2
    discount = (zone_low + zone_high) / 2 <= equilibrium
    volume_ok = _finite(df.loc[bos_date, "VOL_RATIO"], 0) >= 1.05
    confirmation = bool(row["BULL_REJECTION"]) or close > _finite(df["High"].iloc[-2])
    distance = _distance_to_zone(close, zone_low, zone_high, atr_v)

    score = 20 + 25 + 15 + 15
    score += 10 if ob_overlap else 3
    score += 10 if discount else 2
    score += 5 if volume_ok else 0
    plan.setup_score = min(100.0, float(score))
    plan.detected = not invalidated
    plan.invalidated = invalidated
    if invalidated:
        plan.reason = "FVG/low sweep sudah ditutup tembus; zona tidak lagi valid"
        return plan

    in_zone = distance <= 0.35
    if in_zone and confirmation:
        raw_entry = max(close, _finite(row["High"]) + idx_tick_size(close))
        plan.entry_type = "BUY_STOP_FVG_RECLAIM"
        plan.action = "READY_TRIGGER"
    else:
        raw_entry = (zone_low + zone_high) / 2
        plan.entry_type = "LIMIT_FVG_THEN_RECLAIM"
        plan.action = "WAIT_FVG_RETRACE"
    raw_stop = min(sweep_low, zone_low - 0.45 * atr_v) - 0.10 * atr_v
    plan.signal_date = sweep_date
    plan.zone_created_date = fvg_date
    plan.zone_age_bars = _bars_since(df, fvg_date)
    plan.valid_until = pd.Timestamp(fvg_date) + pd.offsets.BDay(30)
    plan.entry_low, plan.entry_high = zone_low, zone_high
    plan.trigger = max(zone_high, _finite(row["High"])) + idx_tick_size(close)
    plan.distance_atr = round(distance, 2)
    plan.evidence = ["Sell-side liquidity sweep", "Bullish BOS dengan displacement", "Bullish FVG valid"]
    if ob_overlap:
        plan.evidence.append("FVG overlap dengan order-block proxy")
    if discount:
        plan.evidence.append("Zona berada di discount dealing range")
    plan.reason = "SMC/ICT dipakai sebagai timing confluence, bukan bukti standalone"
    return _plan_prices(plan, atr_v, raw_entry, raw_stop, 1.5, 2.7)


SETUP_DETECTORS: tuple[Callable[[pd.DataFrame, str], SetupPlan], ...] = (
    detect_pullback_continuation,
    detect_breakout_retest,
    detect_reversal_accumulation,
    detect_unicorn_sniper,
)


def detect_all_setups(df: pd.DataFrame, ticker: str) -> list[SetupPlan]:
    return [detector(df, ticker) for detector in SETUP_DETECTORS]
