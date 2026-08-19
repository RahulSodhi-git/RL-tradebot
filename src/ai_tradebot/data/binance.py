from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import hmac
from pathlib import Path
import time
from typing import Iterable
from urllib.parse import urlencode

import pandas as pd
import requests

from ai_tradebot.core.settings import binance_settings


FAPI = "https://fapi.binance.com"
FAPI_TESTNET = "https://testnet.binancefuture.com"
RAW_DIR = Path("data/raw")


@dataclass(frozen=True)
class CandleRequest:
    symbol: str
    timeframe: str
    limit: int = 1000


class BinanceDataError(RuntimeError):
    pass


def base_url(source: str = "mainnet") -> str:
    return FAPI_TESTNET if source == "testnet" else FAPI


def get_usdt_perp_symbols(
    limit: int | None = None,
    min_quote_volume: float = 0,
    source: str = "mainnet",
) -> list[str]:
    data = _get_json("/fapi/v1/ticker/24hr", {}, source=source)
    rows: list[tuple[str, float]] = []
    for item in data:
        symbol = str(item.get("symbol", ""))
        if not symbol.endswith("USDT"):
            continue
        try:
            quote_volume = float(item.get("quoteVolume", 0))
        except (TypeError, ValueError):
            continue
        if quote_volume >= min_quote_volume:
            rows.append((symbol, quote_volume))
    rows.sort(key=lambda row: row[1], reverse=True)
    symbols = [symbol for symbol, _ in rows]
    return symbols[:limit] if limit else symbols


def fetch_klines(symbol: str, timeframe: str, limit: int = 1000, source: str = "mainnet") -> pd.DataFrame:
    payload = _get_json(
        "/fapi/v1/klines",
        {"symbol": symbol.upper(), "interval": timeframe, "limit": int(limit)},
        source=source,
    )
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame(payload, columns=columns)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for column in ["open", "high", "low", "close", "volume", "quote_asset_volume", "number_of_trades"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["symbol"] = symbol.upper()
    df["timeframe"] = timeframe
    df["source"] = source
    return df[
        [
            "symbol",
            "timeframe",
            "source",
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
        ]
    ].dropna()


def fetch_many(
    requests_: Iterable[CandleRequest],
    max_workers: int = 12,
    save_raw: bool = True,
    source: str = "mainnet",
) -> tuple[dict[tuple[str, str], pd.DataFrame], list[str]]:
    results: dict[tuple[str, str], pd.DataFrame] = {}
    errors: list[str] = []
    jobs = list(requests_)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_klines, job.symbol, job.timeframe, job.limit, source): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                df = future.result()
                results[(job.symbol.upper(), job.timeframe)] = df
                if save_raw:
                    save_candles(df, job.symbol, job.timeframe)
            except Exception as exc:
                errors.append(f"{job.symbol} {job.timeframe}: {exc}")
    return results, errors


def save_candles(df: pd.DataFrame, symbol: str, timeframe: str, source: str | None = None) -> Path:
    source = source or str(df.get("source", pd.Series(["mainnet"])).iloc[0])
    path = RAW_DIR / source / symbol.upper() / f"{timeframe}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_candles(symbol: str, timeframe: str, source: str = "mainnet") -> pd.DataFrame:
    path = RAW_DIR / source / symbol.upper() / f"{timeframe}.csv"
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def inspect_market_data(source: str, symbol: str, timeframe: str, limit: int = 120) -> dict:
    symbols = get_usdt_perp_symbols(limit=20, source=source)
    candles = fetch_klines(symbol, timeframe, limit=limit, source=source)
    return {
        "source": source,
        "endpoint": base_url(source),
        "symbols_sample": symbols,
        "symbols_sample_count": len(symbols),
        "candles": candles,
        "rows": len(candles),
        "start": candles["time"].min() if not candles.empty else None,
        "end": candles["time"].max() if not candles.empty else None,
        "columns": list(candles.columns),
    }


def test_futures_testnet_account() -> dict:
    settings = binance_settings()
    if not settings.has_testnet_keys:
        return {
            "ok": False,
            "reason": "Missing BINANCE_FUTURES_TESTNET_KEY / BINANCE_FUTURES_TESTNET_SECRET in .env",
        }
    try:
        server_time = _get_json("/fapi/v1/time", {}, source="testnet")["serverTime"]
        account = _signed_get(
            "/fapi/v2/account",
            {"timestamp": server_time, "recvWindow": 5000},
            settings.futures_testnet_key,
            settings.futures_testnet_secret,
            source="testnet",
        )
        assets = [
            {
                "asset": item.get("asset"),
                "walletBalance": item.get("walletBalance"),
                "availableBalance": item.get("availableBalance"),
            }
            for item in account.get("assets", [])
            if float(item.get("walletBalance", 0) or 0) != 0
            or float(item.get("availableBalance", 0) or 0) != 0
        ]
        return {"ok": True, "assets": assets, "can_read_account": True}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def _get_json(path: str, params: dict, source: str = "mainnet") -> list | dict:
    response = requests.get(base_url(source) + path, params=params, timeout=20)
    if response.status_code != 200:
        raise BinanceDataError(f"Binance HTTP {response.status_code}: {response.text[:200]}")
    return response.json()


def _signed_get(path: str, params: dict, api_key: str, api_secret: str, source: str = "testnet") -> dict:
    payload = dict(params)
    query = urlencode(payload)
    signature = hmac.new(api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
    payload["signature"] = signature
    response = requests.get(
        base_url(source) + path,
        params=payload,
        headers={"X-MBX-APIKEY": api_key},
        timeout=20,
    )
    if response.status_code != 200:
        raise BinanceDataError(f"Binance signed HTTP {response.status_code}: {response.text[:300]}")
    return response.json()
