from __future__ import annotations

from datetime import timezone

import pandas as pd

from ai_tradebot.core.config import FeatureConfig
from ai_tradebot.core.schema import MarketSnapshot
from ai_tradebot.features.indicators import atr, ema, rsi


REQUIRED_COLUMNS = {"time", "open", "high", "low", "close", "volume"}


class FeatureBuildError(ValueError):
    pass


def validate_ohlcv(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise FeatureBuildError(f"missing OHLCV columns: {sorted(missing)}")
    if len(df) < 60:
        raise FeatureBuildError("need at least 60 candles to build stable features")


def build_snapshot(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: FeatureConfig | None = None,
) -> MarketSnapshot:
    config = config or FeatureConfig()
    validate_ohlcv(df)

    data = df.copy()
    data["return_1"] = data["close"].pct_change()
    data["return_3"] = data["close"].pct_change(3)
    data["return_12"] = data["close"].pct_change(12)
    data["ema_fast"] = ema(data["close"], config.fast_ema)
    data["ema_slow"] = ema(data["close"], config.slow_ema)
    data["rsi"] = rsi(data["close"], config.rsi_period)
    data["atr"] = atr(data, config.atr_period)
    data["volatility"] = data["return_1"].rolling(config.volatility_window).std()
    data["volume_mean"] = data["volume"].rolling(config.volume_window).mean()
    data["volume_spike"] = data["volume"] / data["volume_mean"]

    row = data.iloc[-1]
    close = float(row["close"])
    timestamp = pd.to_datetime(row["time"]).to_pydatetime()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    features = {
        "close": close,
        "return_1": _finite(row["return_1"]),
        "return_3": _finite(row["return_3"]),
        "return_12": _finite(row["return_12"]),
        "ema_fast_distance": _finite(row["ema_fast"] / close - 1),
        "ema_slow_distance": _finite(row["ema_slow"] / close - 1),
        "ema_trend": _finite(row["ema_fast"] / row["ema_slow"] - 1),
        "rsi": _finite(row["rsi"], default=50.0),
        "atr_pct": _finite(row["atr"] / close),
        "volatility": _finite(row["volatility"]),
        "volume_spike": _finite(row["volume_spike"], default=1.0),
    }
    return MarketSnapshot(
        symbol=symbol.upper(),
        timestamp=timestamp,
        timeframe=timeframe,
        features=features,
        metadata={"rows": len(data)},
    )


def _finite(value: float, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(value):
        return default
    return value

