from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class BinanceSettings:
    futures_testnet_key: str = (
        os.getenv("BINANCE_FUTURES_TESTNET_KEY")
        or os.getenv("BINANCE_API_KEY_FUT_TEST")
        or os.getenv("BINANCE_API_KEY_TEST")
        or ""
    )
    futures_testnet_secret: str = (
        os.getenv("BINANCE_FUTURES_TESTNET_SECRET")
        or os.getenv("BINANCE_API_SECRET_FUT_TEST")
        or os.getenv("BINANCE_API_SECRET_TEST")
        or ""
    )
    use_testnet: bool = (
        os.getenv("BINANCE_USE_TESTNET")
        or os.getenv("USE_TESTNET")
        or "true"
    ).lower() == "true"

    @property
    def has_testnet_keys(self) -> bool:
        return bool(self.futures_testnet_key and self.futures_testnet_secret)


def binance_settings() -> BinanceSettings:
    return BinanceSettings()
