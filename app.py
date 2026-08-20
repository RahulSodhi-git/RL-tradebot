from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


BINANCE_FUTURES_URL = "https://fapi.binance.com"
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]


st.set_page_config(page_title="AI Market Prediction Lab", layout="wide")


def main() -> None:
    st.title("AI Market Prediction Lab")
    st.caption("Fresh start. First goal: understand BTC and ZEC candles. No trading. No orders.")

    with st.sidebar:
        st.header("Pair")
        base_symbol = st.text_input("Base coin", "BTCUSDT").strip().upper()
        target_symbol = st.text_input("Target coin", "ZECUSDT").strip().upper()
        timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=1)
        limit = st.slider("Candles", 100, 1000, 300, 50)

    st.subheader("1. Load Binance Candles")
    st.write("This checks if Binance gives us clean BTC and ZEC candle data.")

    if st.button("Load BTC and ZEC", type="primary"):
        try:
            base = fetch_klines(base_symbol, timeframe, limit)
            target = fetch_klines(target_symbol, timeframe, limit)
        except Exception as exc:
            st.error(f"Could not load Binance data: {exc}")
            return

        joined = build_relationship_frame(base, target, base_symbol, target_symbol)
        st.session_state["base"] = base
        st.session_state["target"] = target
        st.session_state["joined"] = joined
        st.success(f"Loaded {len(base)} {base_symbol} candles and {len(target)} {target_symbol} candles.")

    base = st.session_state.get("base")
    target = st.session_state.get("target")
    joined = st.session_state.get("joined")

    if base is None or target is None or joined is None:
        st.info("Click the button above to load the first dataset.")
        return

    render_summary(joined, base_symbol, target_symbol)
    render_charts(base, target, joined, base_symbol, target_symbol, timeframe)
    render_rows(joined, base_symbol, target_symbol)


def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    response = requests.get(
        f"{BINANCE_FUTURES_URL}/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=20,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Binance HTTP {response.status_code}: {response.text[:200]}")

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
    data = pd.DataFrame(response.json(), columns=columns)
    data["time"] = pd.to_datetime(data["open_time"], unit="ms", utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["symbol"] = symbol
    return data[["symbol", "time", "open", "high", "low", "close", "volume"]].dropna()


def build_relationship_frame(base: pd.DataFrame, target: pd.DataFrame, base_symbol: str, target_symbol: str) -> pd.DataFrame:
    base_frame = base[["time", "close", "volume"]].rename(
        columns={"close": f"{base_symbol}_close", "volume": f"{base_symbol}_volume"}
    )
    target_frame = target[["time", "open", "high", "low", "close", "volume"]].rename(
        columns={
            "open": f"{target_symbol}_open",
            "high": f"{target_symbol}_high",
            "low": f"{target_symbol}_low",
            "close": f"{target_symbol}_close",
            "volume": f"{target_symbol}_volume",
        }
    )
    joined = target_frame.merge(base_frame, on="time", how="inner").sort_values("time")
    joined[f"{base_symbol}_return_1"] = joined[f"{base_symbol}_close"].pct_change()
    joined[f"{target_symbol}_return_1"] = joined[f"{target_symbol}_close"].pct_change()
    joined["target_minus_base_return"] = joined[f"{target_symbol}_return_1"] - joined[f"{base_symbol}_return_1"]
    joined["same_direction"] = joined[f"{base_symbol}_return_1"] * joined[f"{target_symbol}_return_1"] > 0
    joined["target_beta_30"] = (
        joined[f"{target_symbol}_return_1"].rolling(30).cov(joined[f"{base_symbol}_return_1"])
        / joined[f"{base_symbol}_return_1"].rolling(30).var()
    )
    joined["target_corr_30"] = joined[f"{target_symbol}_return_1"].rolling(30).corr(joined[f"{base_symbol}_return_1"])
    joined["loaded_at"] = datetime.now(timezone.utc).isoformat()
    return joined.dropna().reset_index(drop=True)


def render_summary(joined: pd.DataFrame, base_symbol: str, target_symbol: str) -> None:
    st.subheader("2. Simple Relationship Summary")
    a, b, c, d = st.columns(4)
    a.metric("Aligned candles", f"{len(joined):,}")
    b.metric("Same direction", f"{joined['same_direction'].mean():.1%}")
    c.metric("Avg beta", f"{joined['target_beta_30'].mean():.2f}")
    d.metric("Avg correlation", f"{joined['target_corr_30'].mean():.2f}")

    st.write(
        f"Simple meaning: if beta is above 1, {target_symbol} usually moves more than {base_symbol}. "
        "Correlation near 1 means they often move together."
    )


def render_charts(
    base: pd.DataFrame,
    target: pd.DataFrame,
    joined: pd.DataFrame,
    base_symbol: str,
    target_symbol: str,
    timeframe: str,
) -> None:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(candle_chart(target, target_symbol, timeframe), use_container_width=True)
    with right:
        st.plotly_chart(return_chart(joined, base_symbol, target_symbol), use_container_width=True)


def candle_chart(target: pd.DataFrame, symbol: str, timeframe: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=target["time"],
            open=target["open"],
            high=target["high"],
            low=target["low"],
            close=target["close"],
            name=symbol,
        )
    )
    fig.update_layout(
        title=f"{symbol} candles ({timeframe})",
        height=500,
        xaxis_rangeslider_visible=False,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return fig


def return_chart(joined: pd.DataFrame, base_symbol: str, target_symbol: str) -> go.Figure:
    chart = joined.copy()
    chart[f"{base_symbol}_move_pct"] = (chart[f"{base_symbol}_close"] / chart[f"{base_symbol}_close"].iloc[0] - 1) * 100
    chart[f"{target_symbol}_move_pct"] = (chart[f"{target_symbol}_close"] / chart[f"{target_symbol}_close"].iloc[0] - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart["time"], y=chart[f"{base_symbol}_move_pct"], mode="lines", name=base_symbol))
    fig.add_trace(go.Scatter(x=chart["time"], y=chart[f"{target_symbol}_move_pct"], mode="lines", name=target_symbol))
    fig.update_layout(
        title=f"{base_symbol} vs {target_symbol} percentage move",
        height=500,
        yaxis_title="Move %",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return fig


def render_rows(joined: pd.DataFrame, base_symbol: str, target_symbol: str) -> None:
    st.subheader("3. Raw Relationship Rows")
    columns = [
        "time",
        f"{base_symbol}_close",
        f"{target_symbol}_close",
        f"{base_symbol}_return_1",
        f"{target_symbol}_return_1",
        "target_minus_base_return",
        "same_direction",
        "target_beta_30",
        "target_corr_30",
    ]
    st.dataframe(joined[columns].tail(80).sort_values("time", ascending=False), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
