from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    max_risk_score: float = 0.65
    min_confidence_to_trade: float = 0.60
    max_position_pct: float = 0.10
    max_leverage: int = 3
    min_liquidity_usdt: float = 50_000_000.0


@dataclass(frozen=True)
class FeatureConfig:
    fast_ema: int = 12
    slow_ema: int = 26
    rsi_period: int = 14
    atr_period: int = 14
    volatility_window: int = 20
    volume_window: int = 20


@dataclass(frozen=True)
class AppConfig:
    risk: RiskConfig = RiskConfig()
    features: FeatureConfig = FeatureConfig()

