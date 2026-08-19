from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ai_tradebot.features.indicators import atr, ema, rsi


LABEL_MAP = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}
PROCESSED_DIR = Path("data/processed")


def build_direction_dataset(
    candles: dict[tuple[str, str], pd.DataFrame],
    horizon: int = 4,
    neutral_threshold: float = 0.003,
) -> pd.DataFrame:
    frames = []
    for (symbol, timeframe), df in candles.items():
        try:
            frame = build_features_and_labels(df, horizon=horizon, neutral_threshold=neutral_threshold)
            frame["symbol"] = symbol
            frame["timeframe"] = timeframe
            frames.append(frame)
        except ValueError:
            continue
    if not frames:
        return pd.DataFrame()
    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return dataset


def build_features_and_labels(
    df: pd.DataFrame,
    horizon: int = 4,
    neutral_threshold: float = 0.003,
) -> pd.DataFrame:
    if len(df) < 80:
        raise ValueError("need at least 80 candles")
    data = df.sort_values("time").copy()
    close = data["close"]
    data["return_1"] = close.pct_change(1)
    data["return_3"] = close.pct_change(3)
    data["return_6"] = close.pct_change(6)
    data["return_12"] = close.pct_change(12)
    data["ema_9"] = ema(close, 9)
    data["ema_21"] = ema(close, 21)
    data["ema_50"] = ema(close, 50)
    data["ema_9_dist"] = close / data["ema_9"] - 1
    data["ema_21_dist"] = close / data["ema_21"] - 1
    data["ema_50_dist"] = close / data["ema_50"] - 1
    data["ema_9_21_trend"] = data["ema_9"] / data["ema_21"] - 1
    data["ema_21_50_trend"] = data["ema_21"] / data["ema_50"] - 1
    data["rsi_14"] = rsi(close, 14)
    data["atr_pct"] = atr(data, 14) / close
    data["volatility_20"] = close.pct_change().rolling(20).std()
    data["volume_spike_20"] = data["volume"] / data["volume"].rolling(20).mean()
    data["quote_volume_log"] = np.log1p(data.get("quote_asset_volume", data["volume"] * close))
    data["trades_log"] = np.log1p(data.get("number_of_trades", 0))

    future_close = close.shift(-horizon)
    data["future_return"] = future_close / close - 1
    data["label"] = np.where(
        data["future_return"] > neutral_threshold,
        1,
        np.where(data["future_return"] < -neutral_threshold, -1, 0),
    )
    data["label_name"] = data["label"].map(LABEL_MAP)
    keep = ["symbol", "timeframe", "time", "open", "high", "low", "close", "volume"] + feature_columns()
    keep += ["future_return", "label", "label_name"]
    return data[keep].iloc[60:-horizon].copy()


def feature_columns() -> list[str]:
    return [
        "return_1",
        "return_3",
        "return_6",
        "return_12",
        "ema_9_dist",
        "ema_21_dist",
        "ema_50_dist",
        "ema_9_21_trend",
        "ema_21_50_trend",
        "rsi_14",
        "atr_pct",
        "volatility_20",
        "volume_spike_20",
        "quote_volume_log",
        "trades_log",
    ]


def save_dataset(dataset: pd.DataFrame, name: str = "direction_dataset.csv") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    dataset.to_csv(path, index=False)
    return path


def load_dataset(name: str = "direction_dataset.csv") -> pd.DataFrame:
    path = PROCESSED_DIR / name
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df

