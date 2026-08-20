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


def build_pair_relationship_dataset(
    base_df: pd.DataFrame,
    target_df: pd.DataFrame,
    base_symbol: str = "BTCUSDT",
    target_symbol: str = "ZECUSDT",
    timeframe: str = "15m",
    horizon: int = 4,
    neutral_threshold: float = 0.003,
) -> pd.DataFrame:
    base = build_features_and_labels(base_df, horizon=horizon, neutral_threshold=neutral_threshold)
    target = build_features_and_labels(target_df, horizon=horizon, neutral_threshold=neutral_threshold)

    base = base.rename(columns={column: f"base_{column}" for column in feature_columns()})
    base = base.rename(
        columns={
            "open": "base_open",
            "high": "base_high",
            "low": "base_low",
            "close": "base_close",
            "volume": "base_volume",
            "future_return": "base_future_return",
            "label": "base_label",
            "label_name": "base_label_name",
        }
    )

    base_keep = [
        "time",
        "base_open",
        "base_high",
        "base_low",
        "base_close",
        "base_volume",
        "base_future_return",
    ] + [f"base_{column}" for column in feature_columns()]
    data = target.merge(base[base_keep], on="time", how="inner").sort_values("time")
    if data.empty:
        target_ordered = target.sort_values("time").reset_index(drop=True).reset_index(names="row_id")
        base_ordered = base[base_keep].sort_values("time").reset_index(drop=True).reset_index(names="row_id")
        data = target_ordered.merge(base_ordered.drop(columns=["time"]), on="row_id", how="inner")

    target_return = data["close"].pct_change()
    base_return = data["base_close"].pct_change()
    data["base_return_1"] = base_return
    data["base_return_3"] = data["base_close"].pct_change(3)
    data["base_return_6"] = data["base_close"].pct_change(6)
    data["target_vs_base_return_1"] = target_return - base_return
    data["target_vs_base_return_3"] = data["return_3"] - data["base_return_3"]
    data["target_base_return_ratio_3"] = data["return_3"] / data["base_return_3"].replace(0, np.nan)
    data["target_base_corr_20"] = target_return.rolling(20).corr(base_return)
    data["target_base_beta_20"] = target_return.rolling(20).cov(base_return) / base_return.rolling(20).var()
    data["base_lead_return_1"] = base_return.shift(1)
    data["base_lead_return_3"] = data["base_return_3"].shift(1)

    data["symbol"] = target_symbol.upper()
    data["base_symbol"] = base_symbol.upper()
    data["target_symbol"] = target_symbol.upper()
    data["timeframe"] = timeframe
    keep = [
        "symbol",
        "base_symbol",
        "target_symbol",
        "timeframe",
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "base_close",
        "base_volume",
    ] + feature_columns() + relationship_feature_columns() + ["future_return", "label", "label_name"]
    return data[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


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


def relationship_feature_columns() -> list[str]:
    return [
        "base_return_1",
        "base_return_3",
        "base_return_6",
        "target_vs_base_return_1",
        "target_vs_base_return_3",
        "target_base_return_ratio_3",
        "target_base_corr_20",
        "target_base_beta_20",
        "base_lead_return_1",
        "base_lead_return_3",
    ] + [f"base_{column}" for column in feature_columns()]


def model_feature_columns(dataset: pd.DataFrame | None = None) -> list[str]:
    columns = feature_columns() + relationship_feature_columns()
    if dataset is None:
        return columns
    return [column for column in columns if column in dataset.columns]


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
