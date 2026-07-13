from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScanConfig:
    """Conservative defaults for daily-bar IDX scanning."""

    min_bars: int = 220
    min_price: float = 50.0
    min_adtv_idr: float = 2_000_000_000.0
    min_atr_pct: float = 0.008
    max_atr_pct: float = 0.12
    max_zero_volume_ratio: float = 0.10
    min_score: float = 70.0
    execution_score: float = 78.0
    min_rr1: float = 1.35
    min_rr2: float = 2.0
    max_stop_pct: float = 0.09
    ready_distance_atr: float = 0.40
    watch_distance_atr: float = 2.0
    max_zone_age_bars: int = 30
    max_data_lag_days: int = 5
    max_absolute_data_age_days: int = 10
    fee_roundtrip_pct: float = 0.0045
    slippage_roundtrip_pct: float = 0.0020
    backtest_horizon_bars: int = 20
    backtest_target_rr: float = 2.0
    backtest_min_gap_bars: int = 10
    beta_prior_wins: float = 8.0
    beta_prior_losses: float = 8.0
    fundamental_top_n: int = 20

    def replace(self, **changes: object) -> "ScanConfig":
        values = self.__dict__.copy()
        values.update(changes)
        return ScanConfig(**values)
