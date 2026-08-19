from __future__ import annotations

import pandas as pd

from ai_tradebot.data.demo import make_demo_ohlcv
from ai_tradebot.features.dataset import build_direction_dataset
from ai_tradebot.models.training import train_one


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

