from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .config import ScanConfig
from .indicators import prepare_indicators
from .models import MarketContext, SetupPlan
from .price_rules import near_upper_auto_rejection
from .setups import detect_all_setups


STATUS_ORDER = {"EXECUTION_READY": 0, "WATCHLIST_ENTRY": 1, "REJECT": 2}


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


class ScanEngine:
    def __init__(self, config: ScanConfig | None = None):
        self.config = config or ScanConfig()

    def _market_context(
        self, prepared: dict[str, pd.DataFrame], benchmark: pd.DataFrame | None
    ) -> tuple[MarketContext, pd.DataFrame | None]:
        bench_ind: pd.DataFrame | None = None
        above50: list[bool] = []
        above200: list[bool] = []
        for frame in prepared.values():
            if not frame.empty:
                last = frame.iloc[-1]
                if pd.notna(last.get("EMA50")):
                    above50.append(bool(last["Close"] > last["EMA50"]))
                if pd.notna(last.get("EMA200")):
                    above200.append(bool(last["Close"] > last["EMA200"]))
        breadth50 = 100 * float(np.mean(above50)) if above50 else None
        breadth200 = 100 * float(np.mean(above200)) if above200 else None
        context = MarketContext(breadth_ema50=breadth50, breadth_ema200=breadth200)
        if benchmark is None or benchmark.empty or len(benchmark) < 205:
            context.reason = "Data IHSG tidak tersedia/cukup; sinyal tidak boleh langsung dieksekusi"
            return context, bench_ind
        today = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None).normalize()
        benchmark_age = max(0, (today - pd.Timestamp(benchmark.index[-1]).normalize()).days)
        if benchmark_age > self.config.max_absolute_data_age_days:
            context.reason = f"Data IHSG berumur {benchmark_age} hari; regime tidak dapat dipercaya"
            return context, bench_ind
        bench_ind = prepare_indicators(benchmark)
        last = bench_ind.iloc[-1]
        close = _number(last["Close"])
        ema50, ema200 = _number(last["EMA50"]), _number(last["EMA200"])
        roc20 = _number(last["ROC20"])
        context.benchmark_close = close
        context.benchmark_roc20 = roc20
        risk_on = close > ema50 > ema200 and roc20 > 0 and (breadth50 is None or breadth50 >= 52)
        risk_off = (close < ema200 and roc20 < 0) or (breadth50 is not None and breadth50 < 35)
        if risk_on:
            context.regime = "RISK_ON"
            context.reason = "IHSG di atas EMA50/200, momentum positif, breadth mendukung"
        elif risk_off:
            context.regime = "RISK_OFF"
            context.reason = "IHSG/breadth menunjukkan risiko pasar tinggi"
        else:
            context.regime = "NEUTRAL"
            context.reason = "Sinyal IHSG dan breadth belum seragam"
        return context, bench_ind

    def _tradeability(
        self, frame: pd.DataFrame, asof: pd.Timestamp
    ) -> tuple[list[str], dict[str, float | int | str]]:
        cfg = self.config
        blockers: list[str] = []
        last = frame.iloc[-1]
        previous = frame.iloc[-2] if len(frame) >= 2 else last
        close = _number(last["Close"], 0)
        atr_pct = _number(last.get("ATR_PCT"), 0)
        adtv = _number(last.get("ADTV20"), 0)
        zero_vol = _number(last.get("ZERO_VOL20"), 1)
        value_today = _number(last.get("VALUE"), 0)
        lag = max(0, (pd.Timestamp(asof).normalize() - pd.Timestamp(frame.index[-1]).normalize()).days)
        now_jakarta = pd.Timestamp.now(tz="Asia/Jakarta")
        today = now_jakarta.tz_localize(None).normalize()
        absolute_age = max(0, (today - pd.Timestamp(frame.index[-1]).normalize()).days)
        current_bar_incomplete = (
            pd.Timestamp(frame.index[-1]).date() == now_jakarta.date()
            and (now_jakarta.hour, now_jakarta.minute) < (16, 15)
        )
        if len(frame) < cfg.min_bars:
            blockers.append(f"Riwayat hanya {len(frame)} bar (<{cfg.min_bars})")
        if close < cfg.min_price:
            blockers.append(f"Harga Rp{close:,.0f} di bawah minimum")
        if adtv < cfg.min_adtv_idr:
            blockers.append(f"ADTV20 Rp{adtv/1e9:.2f} miliar di bawah gate")
        if zero_vol > cfg.max_zero_volume_ratio:
            blockers.append(f"Hari volume nol {zero_vol:.0%} terlalu tinggi")
        if atr_pct < cfg.min_atr_pct:
            blockers.append(f"ATR {atr_pct:.1%} terlalu rendah/stagnan")
        if atr_pct > cfg.max_atr_pct:
            blockers.append(f"ATR {atr_pct:.1%} terlalu ekstrem")
        if lag > cfg.max_data_lag_days:
            blockers.append(f"Data tertinggal {lag} hari dari universe")
        if absolute_age > cfg.max_absolute_data_age_days:
            blockers.append(f"Data absolut sudah berumur {absolute_age} hari")
        if current_bar_incomplete:
            blockers.append("Daily bar hari ini belum dianggap final")
        if adtv > 0 and value_today < 0.15 * adtv:
            blockers.append("Nilai transaksi bar terakhir sangat rendah")
        if len(frame) >= 2 and near_upper_auto_rejection(
            _number(previous["Close"]), close, _number(last["High"])
        ):
            blockers.append("Harga dekat/terkunci ARA; risiko mengejar harga")
        metrics: dict[str, float | int | str] = {
            "last_price": close,
            "last_date": pd.Timestamp(frame.index[-1]).date().isoformat(),
            "data_lag_days": lag,
            "absolute_data_age_days": absolute_age,
            "current_bar_incomplete": int(current_bar_incomplete),
            "adtv20_idr": adtv,
            "atr_pct": atr_pct,
            "zero_volume_ratio20": zero_vol,
            "volume_ratio": _number(last.get("VOL_RATIO")),
            "rsi14": _number(last.get("RSI14")),
            "adx14": _number(last.get("ADX14")),
            "cmf20": _number(last.get("CMF20")),
            "roc60": _number(last.get("ROC60")),
            "distance_52w_high": _number(last.get("DIST_52W_HIGH")),
            "relative_strength60": _number(last.get("REL_STRENGTH60")),
        }
        return blockers, metrics

    def _finalize(
        self,
        plan: SetupPlan,
        frame: pd.DataFrame,
        context: MarketContext,
        trade_blockers: list[str],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        cfg = self.config
        result = plan.to_dict()
        close = float(metrics["last_price"])
        atr_value = _number(frame.iloc[-1].get("ATR14"), 0)
        score_adjustment = {"RISK_ON": 3.0, "NEUTRAL": 0.0, "RISK_OFF": -8.0, "UNKNOWN": -5.0}
        quality_score = round(max(0.0, min(100.0, plan.setup_score + score_adjustment[context.regime])), 1)
        blockers = list(trade_blockers)
        status = "REJECT"

        if plan.detected and not plan.invalidated:
            if plan.zone_age_bars is not None and plan.zone_age_bars > cfg.max_zone_age_bars:
                blockers.append(f"Zona berumur {plan.zone_age_bars} bar; sudah kedaluwarsa")
            if plan.valid_until is not None and pd.Timestamp(frame.index[-1]) > pd.Timestamp(plan.valid_until):
                blockers.append("Masa berlaku setup sudah habis")
            if plan.distance_atr is not None and plan.distance_atr > cfg.watch_distance_atr:
                blockers.append(f"Harga {plan.distance_atr:.2f} ATR dari zona; terlalu jauh")
                plan.action = "TOO_EXTENDED_WAIT_NEW_BASE"
                result["action"] = plan.action
            if plan.entry_low is not None and close < plan.entry_low - 0.75 * atr_value:
                blockers.append("Harga sudah menutup jauh di bawah zona entry")
            if plan.entry and plan.stop_loss:
                stop_pct = (plan.entry - plan.stop_loss) / plan.entry
                result["stop_pct"] = stop_pct
                if stop_pct > cfg.max_stop_pct:
                    blockers.append(f"Jarak SL {stop_pct:.1%} melebihi batas")
            else:
                result["stop_pct"] = np.nan
                blockers.append("Level entry/SL tidak valid")
            if (plan.rr1 or 0) < cfg.min_rr1 or (plan.rr2 or 0) < cfg.min_rr2:
                blockers.append("Risk/reward di bawah minimum")
            if quality_score < cfg.min_score:
                blockers.append(f"Quality score {quality_score:.0f} di bawah {cfg.min_score:.0f}")
            if context.regime == "RISK_OFF":
                blockers.append("Regime IHSG RISK_OFF")
            elif context.regime == "UNKNOWN":
                blockers.append("Regime IHSG tidak dapat diverifikasi")

            ready_action = plan.action in {"READY_TRIGGER", "READY_LIMIT"}
            close_enough = plan.distance_atr is not None and plan.distance_atr <= cfg.ready_distance_atr
            if not ready_action:
                blockers.append("Retest/reclaim/entry trigger belum lengkap")
            elif quality_score < cfg.execution_score:
                blockers.append(
                    f"Quality score {quality_score:.0f} belum mencapai execution threshold {cfg.execution_score:.0f}"
                )
            if plan.distance_atr is None:
                blockers.append("Jarak ke zona tidak dapat dihitung")
            if (
                ready_action
                and close_enough
                and not blockers
                and quality_score >= cfg.execution_score
                and context.regime in {"RISK_ON", "NEUTRAL"}
            ):
                status = "EXECUTION_READY"
            else:
                status = "WATCHLIST_ENTRY"
        result["quality_score"] = quality_score
        result["status"] = status
        result["blockers"] = " • ".join(blockers)
        result["blocker_count"] = len(blockers)
        result["market_regime"] = context.regime
        result["market_reason"] = context.reason
        result["breadth_ema50"] = context.breadth_ema50
        result["breadth_ema200"] = context.breadth_ema200
        result.update(metrics)
        if quality_score >= 88 and not blockers:
            grade = "A"
        elif quality_score >= 78 and len(blockers) <= 1:
            grade = "B+"
        elif quality_score >= 70:
            grade = "B"
        else:
            grade = "C"
        result["grade"] = grade
        result["status_rank"] = STATUS_ORDER[status]
        return result

    def scan(
        self, histories: dict[str, pd.DataFrame], benchmark: pd.DataFrame | None = None
    ) -> dict[str, Any]:
        if not histories:
            return {
                "signals": pd.DataFrame(),
                "universe": pd.DataFrame(),
                "prepared": {},
                "market_context": MarketContext(),
            }
        prepared: dict[str, pd.DataFrame] = {}
        for ticker, frame in histories.items():
            if frame is not None and not frame.empty:
                prepared[ticker] = prepare_indicators(frame, benchmark)
        if not prepared:
            return {
                "signals": pd.DataFrame(),
                "universe": pd.DataFrame(),
                "prepared": {},
                "market_context": MarketContext(),
            }
        context, _ = self._market_context(prepared, benchmark)
        asof_candidates = [pd.Timestamp(frame.index[-1]) for frame in prepared.values()]
        if benchmark is not None and not benchmark.empty:
            asof_candidates.append(pd.Timestamp(benchmark.index[-1]))
        asof = max(asof_candidates)
        signal_rows: list[dict[str, Any]] = []
        universe_rows: list[dict[str, Any]] = []

        for ticker, frame in prepared.items():
            trade_blockers, metrics = self._tradeability(frame, asof)
            plans = detect_all_setups(frame, ticker)
            finalized = [self._finalize(plan, frame, context, trade_blockers, metrics) for plan in plans]
            detected = [row for row in finalized if row["detected"]]
            signal_rows.extend(detected)
            candidates = detected or finalized
            best = sorted(candidates, key=lambda x: (x["status_rank"], -x["quality_score"]))[0]
            universe_rows.append(
                {
                    "ticker": ticker,
                    "best_setup": best["setup"] if best["detected"] else "NO_SETUP",
                    "status": best["status"],
                    "quality_score": best["quality_score"],
                    "grade": best["grade"],
                    "reason": best["reason"],
                    "blockers": best["blockers"],
                    **metrics,
                }
            )

        signals = pd.DataFrame(signal_rows)
        universe = pd.DataFrame(universe_rows)
        if not signals.empty:
            signals = signals.sort_values(
                ["status_rank", "quality_score", "rr2", "adtv20_idr"],
                ascending=[True, False, False, False],
                na_position="last",
            ).reset_index(drop=True)
        if not universe.empty:
            universe["status_rank"] = universe["status"].map(STATUS_ORDER)
            universe = universe.sort_values(
                ["status_rank", "quality_score", "adtv20_idr"], ascending=[True, False, False]
            ).drop(columns="status_rank").reset_index(drop=True)
        return {
            "signals": signals,
            "universe": universe,
            "prepared": prepared,
            "market_context": context,
            "asof": asof,
            "config": asdict(self.config),
        }
