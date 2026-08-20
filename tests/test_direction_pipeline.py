from __future__ import annotations

import pandas as pd

from ai_tradebot.data.demo import make_demo_ohlcv
from ai_tradebot.features.dataset import build_direction_dataset, build_pair_relationship_dataset
from ai_tradebot.training.direction_models import train_one


def test_direction_dataset_and_model_training():
    candles = {
        ("BTCUSDT", "1h"): make_demo_ohlcv(rows=420, seed=1).assign(
            symbol="BTCUSDT", timeframe="1h", quote_asset_volume=1_000_000, number_of_trades=1000
        ),
        ("ETHUSDT", "1h"): make_demo_ohlcv(rows=420, seed=2).assign(
            symbol="ETHUSDT", timeframe="1h", quote_asset_volume=1_000_000, number_of_trades=1000
        ),
    }
    dataset = build_direction_dataset(candles, horizon=4, neutral_threshold=0.003)
    assert {"UP", "DOWN", "NEUTRAL"} & set(dataset["label_name"].unique())
    assert len(dataset) > 500

    metrics, predictions = train_one(dataset, "logistic_regression", "1h", test_fraction=0.25)
    assert metrics["model"] == "logistic_regression"
    assert 0 <= metrics["accuracy"] <= 1
    assert isinstance(predictions, pd.DataFrame)
    assert {"prediction_name", "confidence", "correct"}.issubset(predictions.columns)


def test_pair_relationship_dataset_and_model_training():
    btc = make_demo_ohlcv(rows=520, seed=1).assign(
        symbol="BTCUSDT", timeframe="15m", quote_asset_volume=1_000_000, number_of_trades=1000
    )
    zec = make_demo_ohlcv(rows=520, seed=3).assign(
        symbol="ZECUSDT", timeframe="15m", quote_asset_volume=1_000_000, number_of_trades=1000
    )

    dataset = build_pair_relationship_dataset(
        btc,
        zec,
        base_symbol="BTCUSDT",
        target_symbol="ZECUSDT",
        timeframe="15m",
        horizon=4,
        neutral_threshold=0.003,
    )

    assert len(dataset) > 300
    assert {"base_return_1", "target_vs_base_return_1", "target_base_beta_20"}.issubset(dataset.columns)
    assert dataset["base_symbol"].unique().tolist() == ["BTCUSDT"]
    assert dataset["target_symbol"].unique().tolist() == ["ZECUSDT"]

    metrics, predictions = train_one(dataset, "logistic_regression", "15m", test_fraction=0.25)
    assert metrics["timeframe"] == "15m"
    assert 0 <= metrics["accuracy"] <= 1
    assert {"prediction_name", "confidence", "correct"}.issubset(predictions.columns)
