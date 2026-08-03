"""Fetch public Binance market data (no API key needed) — spot + USD-M futures.

Spot uses data-api.binance.vision (works from US/cloud). Futures uses fapi
(needed for futures-only coins like LABUSDT). Futures is geo-blocked from US
cloud hosts, so those coins load only when the app runs from a non-US location.
"""
import requests
import pandas as pd

SPOT_HOSTS = ["https://data-api.binance.vision", "https://api.binance.com"]
FUT_HOSTS = ["https://fapi.binance.com"]


def _get(hosts, path, params):
    last = None
    for h in hosts:
        try:
            r = requests.get(h + path, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code} from {h}"
        except Exception as e:
            last = str(e)
    raise RuntimeError(last or "request failed")


def _to_df(data):
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbav", "tqav", "ignore"]
    df = pd.DataFrame(data, columns=cols)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["time", "open", "high", "low", "close", "volume"]]


def get_ohlcv(symbol="BTCUSDT", interval="1d", limit=500, closed_only=False, fut_first=False):
    symbol = symbol.upper()
    p = {"symbol": symbol, "interval": interval, "limit": limit}
    order = [(FUT_HOSTS, "/fapi/v1/klines"), (SPOT_HOSTS, "/api/v3/klines")] if fut_first \
        else [(SPOT_HOSTS, "/api/v3/klines"), (FUT_HOSTS, "/fapi/v1/klines")]
    # fut_first=True for futures-only coins (special scan) skips the doomed spot attempt = 1 request
    try:
        df = _to_df(_get(order[0][0], order[0][1], p))
    except Exception:
        df = _to_df(_get(order[1][0], order[1][1], p))
    if closed_only and len(df) > 1:         # drop the still-forming last candle
        df = df.iloc[:-1].reset_index(drop=True)
    return df


def tradeable_universe(min_quote_volume=1_000_000, limit=0):
    """ALL USDT perpetual futures with real 24h volume, ranked most-liquid first.
    Fully dynamic — no fixed list. limit=0 means every liquid coin; limit=N caps to top N."""
    try:
        data = _get(FUT_HOSTS, "/fapi/v1/ticker/24hr", {})
    except Exception:
        return []
    rows = []
    for d in data:
        s = d.get("symbol", "")
        if not s.endswith("USDT"):
            continue
        try:
            vol = float(d["quoteVolume"])
        except (KeyError, ValueError, TypeError):
            continue
        if vol >= min_quote_volume:
            rows.append((s, vol))
    rows.sort(key=lambda x: x[1], reverse=True)          # most liquid first
    syms = [s for s, _ in rows]
    return syms[:limit] if limit else syms


def high_volatility_coins(n=15, min_quote_volume=5_000_000, exclude=()):
    """Most volatile USDT perpetual futures by 24h price move, but liquid enough to trade.
    Adds high-vol small caps to the bot's scan. Needs non-US access (futures endpoint)."""
    try:
        data = _get(FUT_HOSTS, "/fapi/v1/ticker/24hr", {})
    except Exception:
        return []
    exclude = set(exclude)
    rows = []
    for d in data:
        s = d.get("symbol", "")
        if not s.endswith("USDT") or s in exclude:
            continue
        try:
            change = abs(float(d["priceChangePercent"]))   # volatility proxy
            volume = float(d["quoteVolume"])                # liquidity filter (avoid dead coins)
        except (KeyError, ValueError, TypeError):
            continue
        if volume >= min_quote_volume:
            rows.append((s, change))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows[:n]]


def quote_volumes():
    """24h quote (USDT) volume per USDT-perp futures symbol -> {symbol: quoteVolume}.
    Used by the special-trade LIQUIDITY FILTER to skip thin coins where a stop would slip."""
    out = {}
    try:
        for d in _get(FUT_HOSTS, "/fapi/v1/ticker/24hr", {}):
            s = d.get("symbol", "")
            if s.endswith("USDT"):
                try:
                    out[s] = float(d["quoteVolume"])
                except (KeyError, ValueError, TypeError):
                    continue
    except Exception:
        pass
    return out


def open_interest_change(symbol, hours=4, period="1h"):
    """Fractional change in OPEN INTEREST (base contracts) over the last `hours`.
    Positive = new positions being opened (real, funded momentum) rather than a thin spot-only
    spike — this is the special trades' OI-surge gate. Returns None if unavailable (Binance only
    keeps ~30 days of OI history)."""
    try:
        need = int(hours) + 2
        d = _get(FUT_HOSTS, "/futures/data/openInterestHist",
                 {"symbol": symbol.upper(), "period": period, "limit": need})
    except Exception:
        return None
    if not d or len(d) < int(hours) + 1:
        return None
    try:
        now = float(d[-1]["sumOpenInterest"])
        past = float(d[-1 - int(hours)]["sumOpenInterest"])
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    return (now / past - 1) if past > 0 else None


def get_symbols(quote="USDT"):
    """All actively trading spot + futures perpetual pairs for a quote asset."""
    out = set()
    try:
        info = _get(SPOT_HOSTS, "/api/v3/exchangeInfo", {})
        out |= {s["symbol"] for s in info["symbols"]
                if s.get("status") == "TRADING" and s.get("quoteAsset") == quote}
    except Exception:
        pass
    try:                                    # futures (skipped silently if geo-blocked)
        finfo = _get(FUT_HOSTS, "/fapi/v1/exchangeInfo", {})
        out |= {s["symbol"] for s in finfo["symbols"]
                if s.get("status") == "TRADING" and s.get("quoteAsset") == quote
                and s.get("contractType") == "PERPETUAL"}
    except Exception:
        pass
    return sorted(out) or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
