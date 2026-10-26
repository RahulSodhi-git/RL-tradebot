#!/usr/bin/env python3
"""Single-file Binance USD-M data pipeline and PPO training pilot.

This is a research simulator, not a live-trading program. It downloads official
Binance monthly 5-minute kline archives, creates leakage-safe multi-timeframe
features, trains PPO, and evaluates on the newest chronological holdout.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import hmac
import io
import json
import math
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
UNIVERSAL_RAW_DIR = ROOT / "data" / "raw_1h"
UNIVERSAL_PROCESSED_DIR = ROOT / "data" / "processed_1h"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
INITIAL_CAPITAL_USD = 5_000.0
POSITION_FRACTION = 0.10
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ZECUSDT"]
TIMEFRAMES = {"15m": 3, "1h": 12, "4h": 48, "1d": 288}
UNIVERSAL_TIMEFRAMES = {"1h": 1, "4h": 4, "1d": 24}
FEATURE_COLUMNS = [
    f"{tf}_{feature}"
    for tf in TIMEFRAMES
    for feature in ("return", "ema_vote", "ema_gap", "rsi", "rsi_vote", "macd_vote", "volatility", "volume_ratio")
] + [
    "btc_1h_return", "btc_4h_return", "btc_1d_return",
    "btc_4h_ema_gap", "btc_1d_ema_gap",
]
UNIVERSAL_FEATURE_COLUMNS = [
    f"{tf}_{feature}"
    for tf in UNIVERSAL_TIMEFRAMES
    for feature in (
        "return", "ema_vote", "ema_gap", "rsi", "rsi_vote", "macd_vote",
        "volatility", "volume_ratio", "liquidity",
    )
] + [
    "btc_1h_return", "btc_4h_return", "btc_1d_return",
    "btc_4h_ema_gap", "btc_1d_ema_gap", "relative_strength_1h",
]


def month_range(start: str, end: str) -> list[str]:
    first = pd.Period(start, freq="M")
    last = pd.Period(end, freq="M")
    if first > last:
        raise ValueError("start month must not be after end month")
    return [str(period) for period in pd.period_range(first, last, freq="M")]


def fetch_bytes(url: str, attempts: int = 3) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "rl-tradebot-research/1.0"})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 == attempts:
                break
    raise RuntimeError(f"download failed after {attempts} attempts: {url}: {last_error}")


def verify_checksum(payload: bytes, checksum_payload: bytes, filename: str) -> None:
    expected = checksum_payload.decode("utf-8").strip().split()[0].lower()
    actual = hashlib.sha256(payload).hexdigest()
    if expected != actual:
        raise RuntimeError(f"checksum mismatch for {filename}: expected {expected}, got {actual}")


def download_archives(symbols: list[str], start: str, end: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    months = month_range(start, end)
    completed = 0
    for symbol in symbols:
        symbol_dir = RAW_DIR / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        for month in months:
            filename = f"{symbol}-5m-{month}.zip"
            destination = symbol_dir / filename
            if destination.exists() and destination.stat().st_size > 0:
                completed += 1
                continue
            url = f"{BASE_URL}/{symbol}/5m/{filename}"
            print(f"Downloading {url}")
            try:
                payload = fetch_bytes(url)
            except RuntimeError as exc:
                if "404" in str(exc):
                    print(f"Skipping unavailable archive: {filename}")
                    continue
                raise
            try:
                checksum = fetch_bytes(f"{url}.CHECKSUM", attempts=2)
                verify_checksum(payload, checksum, filename)
            except RuntimeError as exc:
                raise RuntimeError(f"refusing unverified archive {filename}: {exc}") from exc
            destination.write_bytes(payload)
            completed += 1
    print(f"Archives ready: {completed}")


def exchange_universe(exchange_info_path: Path) -> list[str]:
    payload = json.loads(exchange_info_path.read_text(encoding="utf-8"))
    symbols = [
        item["symbol"]
        for item in payload.get("symbols", [])
        if item.get("status") == "TRADING"
        and item.get("contractType") == "PERPETUAL"
        and item.get("quoteAsset") == "USDT"
    ]
    if "BTCUSDT" not in symbols:
        raise RuntimeError("exchange metadata does not contain active BTCUSDT perpetual")
    return sorted(set(symbols))


def download_universe_archives(
    exchange_info_path: Path, start: str, end: str, workers: int = 24
) -> list[str]:
    """Download checksum-verified 1h archives for every active USDT perpetual."""
    symbols = exchange_universe(exchange_info_path)
    months = month_range(start, end)
    UNIVERSAL_RAW_DIR.mkdir(parents=True, exist_ok=True)

    def download_one(task: tuple[str, str]) -> tuple[str, str, str]:
        symbol, month = task
        symbol_dir = UNIVERSAL_RAW_DIR / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{symbol}-1h-{month}.zip"
        destination = symbol_dir / filename
        if destination.exists() and destination.stat().st_size > 0:
            return symbol, month, "cached"
        encoded_symbol = urllib.parse.quote(symbol, safe="")
        encoded_filename = urllib.parse.quote(filename, safe="")
        url = f"{BASE_URL}/{encoded_symbol}/1h/{encoded_filename}"
        try:
            payload = fetch_bytes(url, attempts=2)
            checksum = fetch_bytes(f"{url}.CHECKSUM", attempts=2)
            verify_checksum(payload, checksum, filename)
        except RuntimeError as exc:
            if "404" in str(exc):
                return symbol, month, "missing"
            return symbol, month, f"error:{exc}"
        destination.write_bytes(payload)
        return symbol, month, "downloaded"

    tasks = [(symbol, month) for symbol in symbols for month in months]
    counts: dict[str, int] = {}
    usable: set[str] = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for completed, (symbol, _month, status) in enumerate(executor.map(download_one, tasks), start=1):
            counts[status] = counts.get(status, 0) + 1
            if status in ("cached", "downloaded"):
                usable.add(symbol)
            if completed % 500 == 0 or completed == len(tasks):
                print(f"Archive progress {completed:,}/{len(tasks):,}: {counts}")
    errors = sum(value for key, value in counts.items() if key.startswith("error:"))
    if errors:
        print(f"Warning: {errors} archive requests failed for reasons other than missing history")
    print(f"Hourly universe archives ready for {len(usable):,}/{len(symbols):,} symbols")
    return sorted(usable)


def load_symbol(symbol: str) -> pd.DataFrame:
    files = sorted((RAW_DIR / symbol).glob(f"{symbol}-5m-*.zip"))
    files += sorted((RAW_DIR / symbol).glob(f"{symbol}-5m-*.csv"))
    if not files:
        raise FileNotFoundError(f"no ZIP or CSV files found for {symbol}")
    frames: list[pd.DataFrame] = []
    for path in files:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
                if len(csv_names) != 1:
                    raise RuntimeError(f"unexpected archive layout: {path}")
                frame = pd.read_csv(io.BytesIO(archive.read(csv_names[0])), header=None, names=KLINE_COLUMNS)
        else:
            frame = pd.read_csv(path)
            frame = frame.rename(columns={"count": "trades"})
            missing = set(KLINE_COLUMNS) - set(frame.columns)
            if missing:
                raise RuntimeError(f"missing columns in {path}: {sorted(missing)}")
            frame = frame[KLINE_COLUMNS]
        frame["open_time"] = pd.to_numeric(frame["open_time"], errors="coerce")
        frame = frame.dropna(subset=["open_time"])
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    numeric = KLINE_COLUMNS[1:]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["close", "volume", "close_time"])
    data = data.sort_values("open_time").drop_duplicates("open_time", keep="last")
    # Binance began using microsecond timestamps in some archives in 2025.
    divisor = np.where(data["open_time"].to_numpy() > 10**15, 1_000_000, 1_000)
    data["time"] = pd.to_datetime(data["open_time"].to_numpy() / divisor, unit="s", utc=True)
    return data.reset_index(drop=True)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def timeframe_features(data: pd.DataFrame, name: str, step: int) -> pd.DataFrame:
    sample_indices = np.arange(step - 1, len(data), step)
    sampled = data.iloc[sample_indices][["close", "volume"]].copy().reset_index(drop=True)
    close = sampled["close"]
    fast = close.ewm(span=9, adjust=False).mean()
    slow = close.ewm(span=21, adjust=False).mean()
    ema_gap = fast / slow - 1
    rsi_value = rsi(close)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    tf = pd.DataFrame(
        {
            f"{name}_return": close.pct_change(),
            f"{name}_ema_vote": np.sign(ema_gap),
            f"{name}_ema_gap": ema_gap,
            f"{name}_rsi": (rsi_value - 50) / 50,
            f"{name}_rsi_vote": np.where(rsi_value < 30, 1.0, np.where(rsi_value > 70, -1.0, 0.0)),
            f"{name}_macd_vote": np.sign(macd - macd_signal),
            f"{name}_volatility": close.pct_change().rolling(20).std(),
            f"{name}_volume_ratio": sampled["volume"] / sampled["volume"].rolling(20).mean() - 1,
        }
    )
    expanded = pd.DataFrame(index=data.index, columns=tf.columns, dtype=float)
    expanded.loc[sample_indices, :] = tf.to_numpy()
    # Values appear only when that timeframe candle has fully closed, then remain available.
    return expanded.ffill()


def engineer_symbol(symbol: str) -> pd.DataFrame:
    data = load_symbol(symbol)
    result = data[["time", "close", "volume"]].copy()
    result.insert(0, "symbol", symbol)
    result["next_return"] = result["close"].shift(-1) / result["close"] - 1
    for name, step in TIMEFRAMES.items():
        result = pd.concat([result, timeframe_features(data, name, step)], axis=1)
    return result


def load_universal_symbol(symbol: str) -> pd.DataFrame:
    files = sorted((UNIVERSAL_RAW_DIR / symbol).glob(f"{symbol}-1h-*.zip"))
    if not files:
        raise FileNotFoundError(f"no hourly ZIP files found for {symbol}")
    frames: list[pd.DataFrame] = []
    for path in files:
        with zipfile.ZipFile(path) as archive:
            csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
            if len(csv_names) != 1:
                raise RuntimeError(f"unexpected archive layout: {path}")
            frame = pd.read_csv(io.BytesIO(archive.read(csv_names[0])), header=None, names=KLINE_COLUMNS)
        frame["open_time"] = pd.to_numeric(frame["open_time"], errors="coerce")
        frame = frame.dropna(subset=["open_time"])
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data[KLINE_COLUMNS[1:]] = data[KLINE_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["close", "volume", "quote_volume", "close_time"])
    data = data.sort_values("open_time").drop_duplicates("open_time", keep="last")
    divisor = np.where(data["open_time"].to_numpy() > 10**15, 1_000_000, 1_000)
    data["time"] = pd.to_datetime(data["open_time"].to_numpy() / divisor, unit="s", utc=True)
    return data.reset_index(drop=True)


def universal_timeframe_features(data: pd.DataFrame, name: str, step: int) -> pd.DataFrame:
    sample_indices = np.arange(step - 1, len(data), step)
    close = data.iloc[sample_indices]["close"].reset_index(drop=True)
    volume = data["volume"].rolling(step).sum().iloc[sample_indices].reset_index(drop=True)
    quote_volume = data["quote_volume"].rolling(step).sum().iloc[sample_indices].reset_index(drop=True)
    fast = close.ewm(span=9, adjust=False).mean()
    slow = close.ewm(span=21, adjust=False).mean()
    ema_gap = fast / slow - 1
    rsi_value = rsi(close)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    tf = pd.DataFrame(
        {
            f"{name}_return": close.pct_change(),
            f"{name}_ema_vote": np.sign(ema_gap),
            f"{name}_ema_gap": ema_gap,
            f"{name}_rsi": (rsi_value - 50) / 50,
            f"{name}_rsi_vote": np.where(rsi_value < 30, 1.0, np.where(rsi_value > 70, -1.0, 0.0)),
            f"{name}_macd_vote": np.sign(macd - macd_signal),
            f"{name}_volatility": close.pct_change().rolling(20).std(),
            f"{name}_volume_ratio": volume / volume.rolling(20).mean() - 1,
            f"{name}_liquidity": np.log1p(quote_volume),
        }
    )
    expanded = pd.DataFrame(index=data.index, columns=tf.columns, dtype=float)
    expanded.loc[sample_indices, :] = tf.to_numpy()
    return expanded.ffill()


def engineer_universal_symbol(symbol: str) -> pd.DataFrame:
    data = load_universal_symbol(symbol)
    result = data[["time", "close", "volume", "quote_volume"]].copy()
    result.insert(0, "symbol", symbol)
    result["next_return"] = result["close"].shift(-1) / result["close"] - 1
    for name, step in UNIVERSAL_TIMEFRAMES.items():
        result = pd.concat([result, universal_timeframe_features(data, name, step)], axis=1)
    return result


def prepare_universal_data(
    symbols: list[str], train_fraction: float = 0.80, minimum_rows: int = 2_000
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Prepare a shared hourly dataset while skipping insufficient or broken markets."""
    UNIVERSAL_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if "BTCUSDT" not in symbols:
        symbols = ["BTCUSDT", *symbols]
    btc_frame = engineer_universal_symbol("BTCUSDT")
    btc = btc_frame[
        ["time", "1h_return", "4h_return", "1d_return", "4h_ema_gap", "1d_ema_gap"]
    ].rename(
        columns={
            "1h_return": "btc_1h_return",
            "4h_return": "btc_4h_return",
            "1d_return": "btc_1d_return",
            "4h_ema_gap": "btc_4h_ema_gap",
            "1d_ema_gap": "btc_1d_ema_gap",
        }
    )
    frames: dict[str, pd.DataFrame] = {}
    split_indices: dict[str, int] = {}
    for completed, symbol in enumerate(symbols, start=1):
        try:
            frame = btc_frame.copy() if symbol == "BTCUSDT" else engineer_universal_symbol(symbol)
            merged = frame.merge(btc, on="time", how="left").replace([np.inf, -np.inf], np.nan)
            merged["relative_strength_1h"] = merged["1h_return"] - merged["btc_1h_return"]
            merged["btc_regime"] = np.where(
                (merged["btc_4h_ema_gap"] > 0) & (merged["btc_1d_ema_gap"] > 0), 1.0,
                np.where(
                    (merged["btc_4h_ema_gap"] < 0) & (merged["btc_1d_ema_gap"] < 0), -1.0, 0.0
                ),
            )
            merged = merged.dropna(subset=UNIVERSAL_FEATURE_COLUMNS + ["next_return"])
            merged = merged.reset_index(drop=True)
            if len(merged) < minimum_rows or float(merged["volume"].gt(0).mean()) < 0.95:
                continue
            frames[symbol] = merged
            split_indices[symbol] = int(len(merged) * train_fraction)
        except (FileNotFoundError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
            print(f"Skipping {symbol}: {exc}")
        if completed % 50 == 0:
            print(f"Prepared raw features {completed:,}/{len(symbols):,}; eligible={len(frames):,}")
    if "BTCUSDT" not in frames:
        raise RuntimeError("BTCUSDT did not pass universal data validation")

    feature_sum = pd.Series(0.0, index=UNIVERSAL_FEATURE_COLUMNS)
    feature_square_sum = pd.Series(0.0, index=UNIVERSAL_FEATURE_COLUMNS)
    feature_count = 0
    for symbol, frame in frames.items():
        values = frame.iloc[: split_indices[symbol]][UNIVERSAL_FEATURE_COLUMNS].astype("float64")
        feature_sum += values.sum()
        feature_square_sum += values.pow(2).sum()
        feature_count += len(values)
    means = feature_sum / feature_count
    variances = (feature_square_sum / feature_count - means.pow(2)).clip(lower=1e-12)
    stds = np.sqrt(variances)
    for symbol, frame in frames.items():
        normalized = frame.copy()
        normalized[UNIVERSAL_FEATURE_COLUMNS] = (
            (normalized[UNIVERSAL_FEATURE_COLUMNS] - means) / stds
        ).clip(-10, 10).astype("float32")
        normalized.to_parquet(UNIVERSAL_PROCESSED_DIR / f"{symbol}_features.parquet", index=False)
        frames[symbol] = normalized
    metadata = {
        "symbols": sorted(frames),
        "features": UNIVERSAL_FEATURE_COLUMNS,
        "means": {key: float(value) for key, value in means.items()},
        "stds": {key: float(value) for key, value in stds.items()},
        "interval": "1h",
        "position_fraction": POSITION_FRACTION,
    }
    (UNIVERSAL_PROCESSED_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Universal dataset eligible symbols: {len(frames):,}")
    return frames, split_indices


def prepare_data(
    symbols: list[str], train_fraction: float = 0.80, per_symbol_normalization: bool = False
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames = {symbol: engineer_symbol(symbol) for symbol in symbols}
    if "BTCUSDT" not in frames:
        raise ValueError("BTCUSDT is required as market-regime context")
    btc = frames["BTCUSDT"][
        ["time", "1h_return", "4h_return", "1d_return", "4h_ema_gap", "1d_ema_gap"]
    ].rename(
        columns={
            "1h_return": "btc_1h_return",
            "4h_return": "btc_4h_return",
            "1d_return": "btc_1d_return",
            "4h_ema_gap": "btc_4h_ema_gap",
            "1d_ema_gap": "btc_1d_ema_gap",
        }
    )
    split_indices: dict[str, int] = {}
    merged_frames: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        merged = frame.merge(btc, on="time", how="left").replace([np.inf, -np.inf], np.nan)
        # A slow BTC regime exists only when completed 4h and 1d EMA trends agree.
        # This uses information available at the decision timestamp, never future candles.
        merged["btc_regime"] = np.where(
            (merged["btc_4h_ema_gap"] > 0) & (merged["btc_1d_ema_gap"] > 0),
            1.0,
            np.where(
                (merged["btc_4h_ema_gap"] < 0) & (merged["btc_1d_ema_gap"] < 0), -1.0, 0.0
            ),
        )
        merged = merged.dropna(subset=FEATURE_COLUMNS + ["next_return"]).reset_index(drop=True)
        if len(merged) < 10_000:
            raise RuntimeError(f"insufficient usable rows for {symbol}: {len(merged)}")
        split_indices[symbol] = int(len(merged) * train_fraction)
        merged_frames[symbol] = merged

    training_features = pd.concat(
        [frame.iloc[: split_indices[symbol]][FEATURE_COLUMNS] for symbol, frame in merged_frames.items()],
        ignore_index=True,
    )
    global_means = training_features.mean()
    global_stds = training_features.std().replace(0, 1).fillna(1)
    for symbol, frame in merged_frames.items():
        normalized = frame.copy()
        if per_symbol_normalization:
            own_training = frame.iloc[: split_indices[symbol]][FEATURE_COLUMNS]
            means = own_training.mean()
            stds = own_training.std().replace(0, 1).fillna(1)
        else:
            means, stds = global_means, global_stds
        normalized[FEATURE_COLUMNS] = ((normalized[FEATURE_COLUMNS] - means) / stds).clip(-10, 10)
        normalized.to_parquet(PROCESSED_DIR / f"{symbol}_features.parquet", index=False)
        merged_frames[symbol] = normalized
        print(f"Prepared {symbol}: {len(normalized):,} rows; train split {split_indices[symbol]:,}")
    return merged_frames, split_indices


@dataclass(frozen=True)
class RiskSettings:
    fee_rate: float = 0.0005
    max_drawdown: float = 0.50
    drawdown_penalty: float = 0.001
    leverage_penalty: float = 0.000002
    turnover_penalty: float = 0.00002
    opposite_request_penalty: float = 0.002
    neutral_request_penalty: float = 0.0002
    aligned_profit_bonus: float = 0.10
    take_profit_pct: float = 0.02


class CryptoTradingEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        random_start: bool,
        episode_steps: int = 4_096,
        seed: int = 7,
        symbol_universe: list[str] | None = None,
        decision_steps: int = 1,
        btc_regime_filter: bool = False,
        feature_columns: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.frames = frames
        self.symbols = sorted(frames)
        self.symbol_universe = sorted(symbol_universe or self.symbols)
        if not set(self.symbols).issubset(self.symbol_universe):
            raise ValueError("all environment symbols must exist in symbol_universe")
        self.random_start = random_start
        self.episode_steps = episode_steps
        if decision_steps < 1:
            raise ValueError("decision_steps must be positive")
        self.decision_steps = decision_steps
        self.btc_regime_filter = btc_regime_filter
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.rng = np.random.default_rng(seed)
        self.risk = RiskSettings()
        obs_size = len(self.feature_columns) + 5 + len(self.symbol_universe)
        self.observation_space = spaces.Box(-20.0, 20.0, shape=(obs_size,), dtype=np.float32)
        # One position maximum. PPO chooses only hold/flat/long/short. Every new
        # position uses 10% of current equity at fixed 1x leverage.
        self.action_space = spaces.Discrete(4)
        self.frame = next(iter(frames.values()))
        self.symbol = self.symbols[0]
        self.index = 0
        self.end_index = 1
        self.equity = INITIAL_CAPITAL_USD
        self.peak_equity = INITIAL_CAPITAL_USD
        self.worst_drawdown = 0.0
        self.position = 0.0
        self.trades = 0
        self.long_pnl_usd = 0.0
        self.short_pnl_usd = 0.0
        self.long_requests = 0
        self.short_requests = 0
        self.opposite_requests = 0
        self.neutral_blocks = 0
        self.forced_regime_exits = 0
        self.take_profit_exits = 0
        self.trade_log: list[dict[str, object]] = []
        self.active_trade: dict[str, object] | None = None

    def _observation(self) -> np.ndarray:
        row = self.frame.iloc[self.index]
        drawdown = 1 - self.equity / self.peak_equity
        equity_return = self.equity / INITIAL_CAPITAL_USD - 1
        regime = float(row["btc_regime"])
        portfolio = np.array(
            [self.position, equity_return, drawdown, abs(self.position), regime], dtype=np.float32
        )
        identity = np.zeros(len(self.symbol_universe), dtype=np.float32)
        identity[self.symbol_universe.index(self.symbol)] = 1.0
        features = row[self.feature_columns].to_numpy(dtype=np.float32)
        return np.concatenate([features, portfolio, identity]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.symbol = str(self.rng.choice(self.symbols)) if self.random_start else self.symbols[0]
        self.frame = self.frames[self.symbol]
        max_start = max(0, len(self.frame) - self.episode_steps - 2)
        self.index = int(self.rng.integers(0, max_start + 1)) if self.random_start and max_start else 0
        self.end_index = min(len(self.frame) - 1, self.index + self.episode_steps)
        self.equity = INITIAL_CAPITAL_USD
        self.peak_equity = INITIAL_CAPITAL_USD
        self.worst_drawdown = 0.0
        self.position = 0.0
        self.trades = 0
        self.long_pnl_usd = 0.0
        self.short_pnl_usd = 0.0
        self.long_requests = 0
        self.short_requests = 0
        self.opposite_requests = 0
        self.neutral_blocks = 0
        self.forced_regime_exits = 0
        self.take_profit_exits = 0
        self.trade_log = []
        self.active_trade = None
        return self._observation(), {"symbol": self.symbol}

    def _charge_fee(self, turnover: float) -> float:
        fee = self.equity * turnover * self.risk.fee_rate
        self.equity = max(1e-6, self.equity - fee)
        return fee

    def _open_trade(self, target: float, regime: float, balance_before: float, opening_fee: float) -> None:
        row = self.frame.iloc[self.index]
        self.active_trade = {
            "symbol": self.symbol,
            "entry_time": row["time"],
            "entry_price": float(row["close"]),
            "direction": "long" if target > 0 else "short",
            "btc_regime_entry": int(regime),
            "balance_before_usd": balance_before,
            "opening_fee_usd": opening_fee,
            "gross_pnl_usd": 0.0,
            "trade_return_factor": 1.0,
            "max_favorable_pct": 0.0,
            "max_adverse_pct": 0.0,
        }

    def _close_trade(self, reason: str) -> None:
        if self.active_trade is None or self.position == 0:
            return
        closing_fee = self._charge_fee(abs(self.position))
        row = self.frame.iloc[min(self.index, len(self.frame) - 1)]
        trade = self.active_trade
        entry_time = pd.Timestamp(trade["entry_time"])
        exit_time = pd.Timestamp(row["time"])
        balance_before = float(trade["balance_before_usd"])
        trade.update(
            {
                "exit_time": exit_time,
                "exit_price": float(row["close"]),
                "holding_hours": (exit_time - entry_time).total_seconds() / 3_600,
                "exit_reason": reason,
                "closing_fee_usd": closing_fee,
                "net_pnl_usd": self.equity - balance_before,
                "balance_after_usd": self.equity,
            }
        )
        trade.pop("trade_return_factor", None)
        self.trade_log.append(trade.copy())
        self.active_trade = None
        self.position = 0.0
        self.trades += 1

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        command = int(action)
        regime = float(self.frame.iloc[self.index]["btc_regime"])
        action_penalty = 0.0
        if command == 2:
            self.long_requests += 1
        elif command == 3:
            self.short_requests += 1
        if command == 0:
            target = self.position
            if self.btc_regime_filter and target != 0 and target * regime <= 0:
                target = 0.0
                self.forced_regime_exits += 1
        elif command == 1:
            target = 0.0
        else:
            direction = 1.0 if command == 2 else -1.0
            if self.btc_regime_filter and direction * regime <= 0:
                if regime == 0:
                    self.neutral_blocks += 1
                    action_penalty = self.risk.neutral_request_penalty
                    target = 0.0
                else:
                    self.opposite_requests += 1
                    action_penalty = self.risk.opposite_request_penalty
                    target = self.position if self.position * regime > 0 else 0.0
            else:
                target = direction * POSITION_FRACTION

        previous_equity = self.equity
        old_position = self.position
        turnover = abs(target - old_position)
        if target != old_position:
            if old_position != 0:
                if self.btc_regime_filter and regime == 0:
                    exit_reason = "btc_regime_neutral"
                elif self.btc_regime_filter and old_position * regime < 0:
                    exit_reason = "btc_regime_reversed"
                elif target == 0:
                    exit_reason = "model_close"
                else:
                    exit_reason = "model_reverse"
                self._close_trade(exit_reason)
            if target != 0:
                balance_before = self.equity
                opening_fee = self._charge_fee(abs(target))
                self.position = target
                self._open_trade(target, regime, balance_before, opening_fee)
                self.trades += 1

        terminated = False
        steps_available = min(self.decision_steps, self.end_index - self.index)
        aligned_profit_reward = 0.0
        for _ in range(steps_available):
            market_return = float(self.frame.iloc[self.index]["next_return"])
            step_regime = float(self.frame.iloc[self.index]["btc_regime"])
            equity_before_market = self.equity
            self.equity *= max(1e-6, 1 + self.position * market_return)
            directional_pnl = self.equity - equity_before_market
            if self.position > 0:
                self.long_pnl_usd += directional_pnl
            elif self.position < 0:
                self.short_pnl_usd += directional_pnl
            if directional_pnl > 0 and self.position * step_regime > 0:
                aligned_profit_reward += self.risk.aligned_profit_bonus * math.log(
                    max(self.equity / equity_before_market, 1e-8)
                )
            if self.active_trade is not None and self.position != 0:
                direction = 1.0 if self.position > 0 else -1.0
                factor = float(self.active_trade["trade_return_factor"])
                factor *= max(1e-6, 1 + direction * market_return)
                trade_return_pct = (factor - 1) * 100
                self.active_trade["trade_return_factor"] = factor
                self.active_trade["gross_pnl_usd"] = (
                    float(self.active_trade["gross_pnl_usd"]) + directional_pnl
                )
                self.active_trade["max_favorable_pct"] = max(
                    float(self.active_trade["max_favorable_pct"]), trade_return_pct
                )
                self.active_trade["max_adverse_pct"] = min(
                    float(self.active_trade["max_adverse_pct"]), trade_return_pct
                )
            self.peak_equity = max(self.peak_equity, self.equity)
            drawdown = 1 - self.equity / self.peak_equity
            self.worst_drawdown = max(self.worst_drawdown, drawdown)
            self.index += 1

            if self.active_trade is not None and self.position != 0:
                trade_profit = float(self.active_trade["trade_return_factor"]) - 1
                next_regime = float(self.frame.iloc[self.index]["btc_regime"])
                if trade_profit >= self.risk.take_profit_pct:
                    self._close_trade("take_profit_2pct")
                    self.take_profit_exits += 1
                elif self.btc_regime_filter and self.position * next_regime <= 0:
                    self._close_trade(
                        "btc_regime_neutral" if next_regime == 0 else "btc_regime_reversed"
                    )
                    self.forced_regime_exits += 1
            if drawdown >= self.risk.max_drawdown or self.equity <= INITIAL_CAPITAL_USD * 0.05:
                terminated = True
                break

        truncated = bool(self.index >= self.end_index)
        if (terminated or truncated) and self.position != 0:
            self._close_trade("risk_stop" if terminated else "end_of_data")
        drawdown = 1 - self.equity / self.peak_equity
        reward = math.log(max(self.equity / previous_equity, 1e-8))
        reward += aligned_profit_reward
        reward -= action_penalty
        reward -= self.risk.drawdown_penalty * drawdown * max(steps_available, 1)
        reward -= self.risk.leverage_penalty * abs(self.position) ** 2 * max(steps_available, 1)
        reward -= self.risk.turnover_penalty * turnover
        info = {
            "symbol": self.symbol,
            "equity": self.equity,
            "drawdown": drawdown,
            "max_drawdown": self.worst_drawdown,
            "position": self.position,
            "position_value_usd": abs(self.position) * self.equity,
            "trades": self.trades,
            "long_pnl_usd": self.long_pnl_usd,
            "short_pnl_usd": self.short_pnl_usd,
            "long_requests": self.long_requests,
            "short_requests": self.short_requests,
            "opposite_requests": self.opposite_requests,
            "neutral_blocks": self.neutral_blocks,
            "forced_regime_exits": self.forced_regime_exits,
            "take_profit_exits": self.take_profit_exits,
            "completed_trades": len(self.trade_log),
        }
        return self._observation(), float(reward), terminated, truncated, info


def split_frames(
    frames: dict[str, pd.DataFrame], split_indices: dict[str, int]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    train = {symbol: frame.iloc[: split_indices[symbol]].reset_index(drop=True) for symbol, frame in frames.items()}
    test = {symbol: frame.iloc[split_indices[symbol] :].reset_index(drop=True) for symbol, frame in frames.items()}
    return train, test


def evaluate(model: PPO, test_frames: dict[str, pd.DataFrame], decision_steps: int, label: str) -> None:
    print(f"\nUntouched chronological holdout — {label} policy:")
    universe = sorted(test_frames)
    for symbol, frame in test_frames.items():
        env = CryptoTradingEnv(
            {symbol: frame},
            random_start=False,
            episode_steps=len(frame) - 1,
            symbol_universe=universe,
            decision_steps=decision_steps,
        )
        observation, _ = env.reset()
        done = False
        final_info: dict = {}
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, final_info = env.step(action)
            done = terminated or truncated
        total_return = (float(final_info["equity"]) / INITIAL_CAPITAL_USD - 1) * 100
        buy_hold = (float(frame.iloc[-1]["close"]) / float(frame.iloc[0]["close"]) - 1) * 100
        print(
            f"  {symbol:10s} return={total_return:8.2f}%  "
            f"equity=${float(final_info['equity']):9.2f}  "
            f"max drawdown={float(final_info['max_drawdown']) * 100:6.2f}%  "
            f"changes={int(final_info['trades']):5d}  buy-hold={buy_hold:8.2f}%"
        )


def evaluate_specialist(
    model: PPO,
    symbol: str,
    frame: pd.DataFrame,
    decision_steps: int,
    label: str,
    symbol_universe: list[str] | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    env = CryptoTradingEnv(
        {symbol: frame},
        random_start=False,
        episode_steps=len(frame) - 1,
        symbol_universe=symbol_universe or [symbol],
        decision_steps=decision_steps,
        btc_regime_filter=True,
        feature_columns=feature_columns,
    )
    observation, _ = env.reset()
    done = False
    final_info: dict = {}
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, final_info = env.step(action)
        done = terminated or truncated
    total_return = (float(final_info["equity"]) / INITIAL_CAPITAL_USD - 1) * 100
    requests = int(final_info["long_requests"]) + int(final_info["short_requests"])
    wrong = int(final_info["opposite_requests"])
    neutral = int(final_info["neutral_blocks"])
    blocked_rate = 100 * (wrong + neutral) / max(requests, 1)
    buy_hold = (float(frame.iloc[-1]["close"]) / float(frame.iloc[0]["close"]) - 1) * 100
    trades = pd.DataFrame(env.trade_log)
    if not trades.empty:
        trades.insert(1, "timeframe", label)
        trades.insert(2, "starting_capital_usd", INITIAL_CAPITAL_USD)
        trades.insert(3, "allocation_pct", POSITION_FRACTION * 100)
    wins = int((trades["net_pnl_usd"] > 0).sum()) if not trades.empty else 0
    win_rate = 100 * wins / max(len(trades), 1)
    print(
        f"  {symbol:10s} return={total_return:7.2f}%  dd={float(final_info['max_drawdown']) * 100:5.2f}%  "
        f"changes={int(final_info['trades']):4d}  longPnL=${float(final_info['long_pnl_usd']):8.2f}  "
        f"shortPnL=${float(final_info['short_pnl_usd']):8.2f}  blocked={blocked_rate:5.1f}%  "
        f"wins={win_rate:5.1f}%  takeProfits={int(final_info['take_profit_exits']):3d}  "
        f"buyHold={buy_hold:7.2f}%"
    )
    return (
        {
            "symbol": symbol,
            "timeframe": label,
            "return_pct": total_return,
            "ending_balance_usd": float(final_info["equity"]),
            "max_drawdown_pct": float(final_info["max_drawdown"]) * 100,
            "completed_trades": len(trades),
            "win_rate_pct": win_rate,
            "long_pnl_usd": float(final_info["long_pnl_usd"]),
            "short_pnl_usd": float(final_info["short_pnl_usd"]),
            "blocked_request_pct": blocked_rate,
            "take_profit_exits": int(final_info["take_profit_exits"]),
            "forced_regime_exits": int(final_info["forced_regime_exits"]),
            "buy_hold_pct": buy_hold,
        },
        trades,
    )


def train(symbols: list[str], timesteps: int, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    frames, split_indices = prepare_data(symbols)
    train_frames, test_frames = split_frames(frames, split_indices)
    bare_env = CryptoTradingEnv(train_frames, random_start=True, seed=seed)
    check_env(bare_env, warn=True)
    env = Monitor(bare_env)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1_024,
        batch_size=256,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        seed=seed,
        device="cpu",
        verbose=1,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "ppo_crypto_pilot"
    model.save(model_path)
    print(f"Saved model: {model_path}.zip")
    evaluate(model, test_frames, decision_steps=1, label="5m legacy")


def train_multiple_policies(symbols: list[str], timesteps: int, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    frames, split_indices = prepare_data(symbols)
    train_frames, test_frames = split_frames(frames, split_indices)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    policies = (("15m", 3), ("1h", 12), ("4h", 48))
    for offset, (label, decision_steps) in enumerate(policies):
        print(f"\nTraining {label} policy for {timesteps:,} PPO steps")
        bare_env = CryptoTradingEnv(
            train_frames,
            random_start=True,
            episode_steps=8_192,
            seed=seed + offset,
            decision_steps=decision_steps,
        )
        check_env(bare_env, warn=True)
        env = Monitor(bare_env)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1_024,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [128, 128]},
            seed=seed + offset,
            device="cpu",
            verbose=1,
        )
        model.learn(total_timesteps=timesteps, progress_bar=False)
        model_path = MODEL_DIR / f"ppo_crypto_{label}"
        model.save(model_path)
        print(f"Saved model: {model_path}.zip")
        evaluate(model, test_frames, decision_steps=decision_steps, label=label)


def train_specialists(symbols: list[str], timesteps: int, seed: int, timeframes: list[str]) -> None:
    """Train independent coin policies with a leakage-safe BTC slow-regime shield."""
    random.seed(seed)
    np.random.seed(seed)
    frames, split_indices = prepare_data(symbols, per_symbol_normalization=True)
    train_frames, test_frames = split_frames(frames, split_indices)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for timeframe_index, label in enumerate(timeframes):
        decision_steps = TIMEFRAMES[label]
        print(f"\nTraining independent {label} specialists with BTC regime filter")
        results: list[dict[str, object]] = []
        trade_reports: list[pd.DataFrame] = []
        for symbol_index, symbol in enumerate(symbols):
            specialist_seed = seed + timeframe_index * 100 + symbol_index
            print(f"  Training {symbol} {label} for {timesteps:,} PPO steps")
            bare_env = CryptoTradingEnv(
                {symbol: train_frames[symbol]},
                random_start=True,
                episode_steps=8_192,
                seed=specialist_seed,
                symbol_universe=[symbol],
                decision_steps=decision_steps,
                btc_regime_filter=True,
            )
            check_env(bare_env, warn=True)
            model = PPO(
                "MlpPolicy",
                Monitor(bare_env),
                learning_rate=3e-4,
                n_steps=1_024,
                batch_size=256,
                n_epochs=10,
                gamma=0.995,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                policy_kwargs={"net_arch": [128, 128]},
                seed=specialist_seed,
                device="cpu",
                verbose=0,
            )
            model.learn(total_timesteps=timesteps, progress_bar=False)
            model_path = MODEL_DIR / f"ppo_{symbol.lower()}_{label}_btc_regime"
            model.save(model_path)
            print(f"  Saved {model_path.name}.zip")
            result, trades = evaluate_specialist(model, symbol, test_frames[symbol], decision_steps, label)
            results.append(result)
            if not trades.empty:
                trade_reports.append(trades)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = REPORT_DIR / f"specialist_summary_{label}.csv"
        trades_path = REPORT_DIR / f"specialist_trades_{label}.csv"
        pd.DataFrame(results).to_csv(summary_path, index=False)
        combined_trades = pd.concat(trade_reports, ignore_index=True) if trade_reports else pd.DataFrame()
        combined_trades.to_csv(trades_path, index=False)
        print(
            f"{label} specialist average: return={np.mean([float(r['return_pct']) for r in results]):.2f}%  "
            f"drawdown={np.mean([float(r['max_drawdown_pct']) for r in results]):.2f}%"
        )
        print(f"Simple reports: {summary_path} and {trades_path}")


def train_universal(
    exchange_info_path: Path, timesteps: int, seed: int, fine_tune_count: int = 20
) -> None:
    """Train one shared hourly policy, then fine-tune only the weakest coins."""
    random.seed(seed)
    np.random.seed(seed)
    listed = exchange_universe(exchange_info_path)
    downloaded = [symbol for symbol in listed if (UNIVERSAL_RAW_DIR / symbol).exists()]
    frames, split_indices = prepare_universal_data(downloaded)
    train_frames, test_frames = split_frames(frames, split_indices)
    universe = sorted(frames)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    bare_env = CryptoTradingEnv(
        train_frames,
        random_start=True,
        episode_steps=2_048,
        seed=seed,
        symbol_universe=universe,
        decision_steps=1,
        btc_regime_filter=True,
        feature_columns=UNIVERSAL_FEATURE_COLUMNS,
    )
    check_env(bare_env, warn=True)
    model = PPO(
        "MlpPolicy",
        Monitor(bare_env),
        learning_rate=2e-4,
        n_steps=2_048,
        batch_size=512,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [256, 256]},
        seed=seed,
        device="cpu",
        verbose=1,
    )
    print(f"Training shared universal model on {len(universe):,} coins for {timesteps:,} PPO steps")
    model.learn(total_timesteps=timesteps, progress_bar=False)
    base_path = MODEL_DIR / "ppo_universal_1h"
    model.save(base_path)
    print(f"Saved shared model: {base_path}.zip")

    results: list[dict[str, object]] = []
    trade_reports: list[pd.DataFrame] = []
    for completed, symbol in enumerate(universe, start=1):
        result, trades = evaluate_specialist(
            model,
            symbol,
            test_frames[symbol],
            decision_steps=1,
            label="1h_universal",
            symbol_universe=universe,
            feature_columns=UNIVERSAL_FEATURE_COLUMNS,
        )
        results.append(result)
        if not trades.empty:
            trade_reports.append(trades)
        if completed % 50 == 0:
            print(f"Universal evaluation {completed:,}/{len(universe):,}")
    summary = pd.DataFrame(results).sort_values("return_pct")
    summary.to_csv(REPORT_DIR / "universal_summary_1h.csv", index=False)
    if trade_reports:
        pd.concat(trade_reports, ignore_index=True).to_csv(
            REPORT_DIR / "universal_trades_1h.csv", index=False
        )

    weak_symbols = summary.loc[summary["return_pct"] < 0, "symbol"].head(fine_tune_count).tolist()
    fine_tune_steps = max(20_000, timesteps // 10)
    print(f"Fine-tuning {len(weak_symbols)} weakest coins for {fine_tune_steps:,} steps each")
    specialist_results: list[dict[str, object]] = []
    for offset, symbol in enumerate(weak_symbols):
        specialist_env = CryptoTradingEnv(
            {symbol: train_frames[symbol]},
            random_start=True,
            episode_steps=2_048,
            seed=seed + 1_000 + offset,
            symbol_universe=universe,
            decision_steps=1,
            btc_regime_filter=True,
            feature_columns=UNIVERSAL_FEATURE_COLUMNS,
        )
        specialist = PPO.load(base_path, env=Monitor(specialist_env), device="cpu")
        specialist.learn(total_timesteps=fine_tune_steps, reset_num_timesteps=False, progress_bar=False)
        specialist_path = MODEL_DIR / f"ppo_{symbol.lower()}_1h_specialist"
        specialist.save(specialist_path)
        result, _trades = evaluate_specialist(
            specialist,
            symbol,
            test_frames[symbol],
            decision_steps=1,
            label="1h_specialist",
            symbol_universe=universe,
            feature_columns=UNIVERSAL_FEATURE_COLUMNS,
        )
        specialist_results.append(result)
    pd.DataFrame(specialist_results).to_csv(
        REPORT_DIR / "universal_specialist_summary_1h.csv", index=False
    )
    print("Universal shared training and targeted specialist fine-tuning complete")


def binance_api_request(
    base_url: str,
    path: str,
    params: dict[str, object] | None = None,
    method: str = "GET",
    signed: bool = False,
) -> object:
    params = dict(params or {})
    api_key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
    secret = os.environ.get("BINANCE_TESTNET_SECRET", "")
    if signed:
        if base_url.rstrip('/') not in (
            'https://demo-fapi.binance.com', 'https://testnet.binancefuture.com'
        ):
            raise RuntimeError('Signed requests are restricted to Binance demo/testnet hosts')
        if not api_key or not secret:
            raise RuntimeError(
                "set BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_SECRET on the test server"
            )
        params.update({"timestamp": int(time.time() * 1_000), "recvWindow": 5_000})
    query = urllib.parse.urlencode(params)
    if signed:
        signature = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        query = f"{query}&signature={signature}"
    url = f"{base_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{query}"
    headers = {"User-Agent": "rl-tradebot-testnet/1.0"}
    if api_key:
        headers["X-MBX-APIKEY"] = api_key
    request = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Binance API {exc.code} for {path}: {body}") from exc


def live_feature_row(base_url: str, symbol: str, btc_raw: pd.Series | None = None) -> pd.Series:
    payload = binance_api_request(
        base_url, "/fapi/v1/klines", {"symbol": symbol, "interval": "1h", "limit": 1_000}
    )
    if not isinstance(payload, list) or len(payload) < 600:
        raise RuntimeError(f"insufficient live hourly candles for {symbol}")
    data = pd.DataFrame(payload, columns=KLINE_COLUMNS)
    data[KLINE_COLUMNS] = data[KLINE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    data["time"] = pd.to_datetime(data["open_time"], unit="ms", utc=True)
    result = data[["time", "close", "volume", "quote_volume"]].copy()
    for name, step in UNIVERSAL_TIMEFRAMES.items():
        result = pd.concat([result, universal_timeframe_features(data, name, step)], axis=1)
    row = result.iloc[-1].copy()
    if symbol == "BTCUSDT":
        row["btc_1h_return"] = row["1h_return"]
        row["btc_4h_return"] = row["4h_return"]
        row["btc_1d_return"] = row["1d_return"]
        row["btc_4h_ema_gap"] = row["4h_ema_gap"]
        row["btc_1d_ema_gap"] = row["1d_ema_gap"]
    elif btc_raw is not None:
        for key in (
            "btc_1h_return", "btc_4h_return", "btc_1d_return",
            "btc_4h_ema_gap", "btc_1d_ema_gap",
        ):
            row[key] = btc_raw[key]
    else:
        raise ValueError("BTC context is required for non-BTC symbols")
    row["relative_strength_1h"] = row["1h_return"] - row["btc_1h_return"]
    row["btc_regime"] = (
        1.0 if row["btc_4h_ema_gap"] > 0 and row["btc_1d_ema_gap"] > 0
        else -1.0 if row["btc_4h_ema_gap"] < 0 and row["btc_1d_ema_gap"] < 0
        else 0.0
    )
    return row


def scan_universal_testnet(
    base_url: str, min_confidence: float = 0.55
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    metadata_path = UNIVERSAL_PROCESSED_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("train the universal model before scanning testnet")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    universe = list(metadata["symbols"])
    means = pd.Series(metadata["means"])
    stds = pd.Series(metadata["stds"])
    btc_row = live_feature_row(base_url, "BTCUSDT")

    def fetch(symbol: str) -> tuple[str, pd.Series | None, str | None]:
        try:
            row = btc_row.copy() if symbol == "BTCUSDT" else live_feature_row(base_url, symbol, btc_row)
            return symbol, row, None
        except Exception as exc:
            return symbol, None, str(exc)

    rows: dict[str, pd.Series] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for symbol, row, error in executor.map(fetch, universe):
            if row is not None:
                rows[symbol] = row
            elif error:
                print(f"Live scan skipped {symbol}: {error}")
    if not rows:
        raise RuntimeError("no live symbols produced usable features")

    observations: list[np.ndarray] = []
    symbols: list[str] = []
    for symbol, row in rows.items():
        normalized = ((row[UNIVERSAL_FEATURE_COLUMNS].astype(float) - means) / stds).clip(-10, 10)
        portfolio = np.array([0.0, 0.0, 0.0, 0.0, float(row["btc_regime"])], dtype=np.float32)
        identity = np.zeros(len(universe), dtype=np.float32)
        identity[universe.index(symbol)] = 1.0
        observations.append(
            np.concatenate([normalized.to_numpy(dtype=np.float32), portfolio, identity])
        )
        symbols.append(symbol)
    batch = np.stack(observations)
    model = PPO.load(MODEL_DIR / "ppo_universal_1h", device="cpu")
    with torch.no_grad():
        tensor, _ = model.policy.obs_to_tensor(batch)
        probabilities = model.policy.get_distribution(tensor).distribution.probs.cpu().numpy()

    rankings: list[dict[str, object]] = []
    for index, symbol in enumerate(symbols):
        regime = int(rows[symbol]["btc_regime"])
        desired_action = 2 if regime > 0 else 3 if regime < 0 else 1
        confidence = float(probabilities[index, desired_action])
        rankings.append(
            {
                "symbol": symbol,
                "btc_regime": regime,
                "action": ("long" if desired_action == 2 else "short" if desired_action == 3 else "flat"),
                "confidence": confidence,
                "price": float(rows[symbol]["close"]),
            }
        )
    rankings.sort(key=lambda item: float(item["confidence"]), reverse=True)
    eligible = [item for item in rankings if item["action"] != "flat" and item["confidence"] >= min_confidence]
    return (eligible[0] if eligible else None), rankings


def _quantity_for_notional(exchange_symbol: dict, notional: float, price: float) -> float:
    lot = next(item for item in exchange_symbol["filters"] if item["filterType"] == "MARKET_LOT_SIZE")
    step = float(lot["stepSize"])
    minimum = float(lot["minQty"])
    maximum = float(lot["maxQty"])
    quantity = math.floor((notional / price) / step + 1e-12) * step
    if quantity < minimum:
        raise RuntimeError(f"calculated quantity {quantity} is below minimum {minimum}")
    return min(quantity, maximum)


def execute_testnet_selection(base_url: str, selection: dict[str, object] | None) -> None:
    exchange_info = binance_api_request(base_url, "/fapi/v1/exchangeInfo")
    symbol_map = {item["symbol"]: item for item in exchange_info["symbols"]}
    positions = binance_api_request(base_url, "/fapi/v2/positionRisk", signed=True)
    open_positions = [item for item in positions if abs(float(item["positionAmt"])) > 0]
    desired_symbol = str(selection["symbol"]) if selection else None
    desired_direction = str(selection["action"]) if selection else None
    for position in open_positions:
        amount = float(position["positionAmt"])
        same = (
            position["symbol"] == desired_symbol
            and ((amount > 0 and desired_direction == "long") or (amount < 0 and desired_direction == "short"))
        )
        if same:
            print(f"Keeping existing {desired_direction} position in {desired_symbol}")
            return
        binance_api_request(
            base_url,
            "/fapi/v1/order",
            {
                "symbol": position["symbol"],
                "side": "SELL" if amount > 0 else "BUY",
                "type": "MARKET",
                "quantity": abs(amount),
                "reduceOnly": "true",
            },
            method="POST",
            signed=True,
        )
        print(f"Closed previous testnet position in {position['symbol']}")
    if selection is None:
        print("No signal passed the confidence threshold; remaining flat")
        return
    balances = binance_api_request(base_url, "/fapi/v2/balance", signed=True)
    usdt = next(item for item in balances if item["asset"] == "USDT")
    available = float(usdt["availableBalance"])
    symbol = str(selection["symbol"])
    price = float(selection["price"])
    quantity = _quantity_for_notional(symbol_map[symbol], available * POSITION_FRACTION, price)
    binance_api_request(
        base_url, "/fapi/v1/leverage", {"symbol": symbol, "leverage": 1}, method="POST", signed=True
    )
    side = "BUY" if selection["action"] == "long" else "SELL"
    binance_api_request(
        base_url,
        "/fapi/v1/order",
        {"symbol": symbol, "side": side, "type": "MARKET", "quantity": quantity},
        method="POST",
        signed=True,
    )
    print(
        f"Opened testnet {selection['action']} {symbol}: quantity={quantity}, "
        f"allocation=${available * POSITION_FRACTION:.2f}, leverage=1x"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "download", "train", "train-multi", "train-specialists",
            "download-universe", "train-universal", "testnet-scan", "all",
        ),
        nargs="?", default="all"
    )
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--start", default="2024-01", help="first monthly archive, YYYY-MM")
    parser.add_argument("--end", default="2024-06", help="last monthly archive, YYYY-MM")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeframes", nargs="+", choices=tuple(TIMEFRAMES), default=["1h"])
    parser.add_argument("--exchange-info", type=Path, default=ROOT / "data" / "exchangeInfo.json")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--fine-tune-count", type=int, default=20)
    parser.add_argument("--base-url", default="https://demo-fapi.binance.com")
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--execute", action="store_true", help="place real testnet orders; default is dry-run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [symbol.upper() for symbol in args.symbols]
    if "BTCUSDT" not in symbols:
        symbols.insert(0, "BTCUSDT")
    if args.mode in ("download", "all"):
        download_archives(symbols, args.start, args.end)
    if args.mode in ("train", "all"):
        train(symbols, args.timesteps, args.seed)
    if args.mode == "train-multi":
        train_multiple_policies(symbols, args.timesteps, args.seed)
    if args.mode == "train-specialists":
        train_specialists(symbols, args.timesteps, args.seed, args.timeframes)
    if args.mode == "download-universe":
        download_universe_archives(args.exchange_info, args.start, args.end, args.workers)
    if args.mode == "train-universal":
        train_universal(args.exchange_info, args.timesteps, args.seed, args.fine_tune_count)
    if args.mode == "testnet-scan":
        selection, rankings = scan_universal_testnet(args.base_url, args.min_confidence)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rankings).to_csv(REPORT_DIR / "latest_testnet_scan.csv", index=False)
        print(f"Global selection: {selection}")
        if args.execute:
            execute_testnet_selection(args.base_url, selection)
        else:
            print("Dry-run only. Add --execute after reviewing the scan report.")


if __name__ == "__main__":
    main()
