from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def make_demo_ohlcv(rows: int = 160, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 100 + np.cumsum(rng.normal(0.05, 0.9, rows))
    high = base + rng.uniform(0.2, 1.2, rows)
    low = base - rng.uniform(0.2, 1.2, rows)
    open_ = base + rng.normal(0, 0.25, rows)
    close = base + rng.normal(0, 0.25, rows)
    volume = rng.normal(1_000_000, 130_000, rows).clip(100_000)
    volume[-1] *= 2.2
    start = datetime.now(timezone.utc) - timedelta(hours=rows)
    return pd.DataFrame(
        {
            "time": [start + timedelta(hours=i) for i in range(rows)],
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

