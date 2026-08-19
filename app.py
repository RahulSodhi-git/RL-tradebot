from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai_tradebot.data.binance import (
    CandleRequest,
    fetch_klines,
    get_usdt_perp_symbols,
    inspect_market_data,
    save_candles,
    test_futures_testnet_account,
)
from ai_tradebot.features.dataset import build_direction_dataset, load_dataset, save_dataset
from ai_tradebot.models.training import (
    available_model_names,
    confusion_for,
    grouped_metrics,
    train_one,
)


TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
DEFAULT_MODELS = ["logistic_regression", "random_forest", "hist_gradient_boosting"]


st.set_page_config(page_title="AI Tradebot Lab", page_icon=".", layout="wide")


def init_state() -> None:
    st.session_state.setdefault("candles", {})
    st.session_state.setdefault("dataset", pd.DataFrame())
    st.session_state.setdefault("metrics", pd.DataFrame())
    st.session_state.setdefault("predictions", {})
    st.session_state.setdefault("errors", [])


def main() -> None:
    init_state()
    st.title("AI Tradebot Lab")
    st.caption("Market-direction prediction only. No live trading, no Binance keys, no order execution.")

    with st.sidebar:
        st.header("Binance Source")
        data_source_label = st.radio(
            "Market data source",
            ["Mainnet public market data", "Futures testnet market data"],
            index=0,
            help="Mainnet public data is usually best for model training. Testnet is useful for account/execution checks.",
        )
        data_source = "testnet" if data_source_label.startswith("Futures") else "mainnet"

        st.header("Dataset")
        universe_limit = st.slider("Symbols from Binance universe", 5, 260, 40, 5)
        min_quote_volume = st.number_input("Minimum 24h quote volume", min_value=0, value=50_000_000, step=10_000_000)
        candle_limit = st.slider("Candles per symbol/timeframe", 200, 1500, 800, 100)
        selected_timeframes = st.multiselect("Timeframes", TIMEFRAMES, default=["15m", "1h", "4h"])
        horizon = st.slider("Future horizon in candles", 1, 24, 4)
        neutral_threshold = st.slider("Neutral threshold (%)", 0.05, 3.0, 0.30, 0.05) / 100
        data_workers = st.slider("Parallel data workers", 2, 24, 12)

        st.header("Models")
        selected_models = st.multiselect("Models", available_model_names(), default=DEFAULT_MODELS)
        test_fraction = st.slider("Test split from newest data (%)", 10, 40, 25, 5) / 100
        model_workers = st.slider("Parallel model workers", 1, 8, 3)

    tab_source, tab_data, tab_train, tab_eval, tab_candles = st.tabs(
        ["Data Source", "Data Builder", "Live Training", "Evaluation", "Candles + Predictions"]
    )

    with tab_source:
        render_data_source(data_source, selected_timeframes[0] if selected_timeframes else "1h")

    with tab_data:
        render_data_builder(
            data_source,
            universe_limit,
            min_quote_volume,
            candle_limit,
            selected_timeframes,
            horizon,
            neutral_threshold,
            data_workers,
        )

    with tab_train:
        render_training(selected_models, selected_timeframes, test_fraction, model_workers)

    with tab_eval:
        render_evaluation()

    with tab_candles:
        render_candle_view()


def render_data_builder(
    data_source: str,
    universe_limit: int,
    min_quote_volume: float,
    candle_limit: int,
    selected_timeframes: list[str],
    horizon: int,
    neutral_threshold: float,
    data_workers: int,
) -> None:
    st.subheader("Build Multi-Timeframe Prediction Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbols", universe_limit)
    c2.metric("Timeframes", len(selected_timeframes))
    c3.metric("Candles each", candle_limit)
    c4.metric("Jobs", universe_limit * len(selected_timeframes))
    st.info(
        "Training source: "
        + ("Binance Futures TESTNET market data" if data_source == "testnet" else "Binance MAINNET public market data")
    )

    if st.button("Download candles and build labels", type="primary", disabled=not selected_timeframes):
        st.session_state.errors = []
        symbols = get_usdt_perp_symbols(limit=universe_limit, min_quote_volume=min_quote_volume, source=data_source)
        jobs = [CandleRequest(symbol, timeframe, candle_limit) for symbol in symbols for timeframe in selected_timeframes]
        candles: dict[tuple[str, str], pd.DataFrame] = {}
        progress = st.progress(0)
        status = st.empty()
        errors = []
        done = 0

        with ThreadPoolExecutor(max_workers=data_workers) as executor:
            futures = {
                executor.submit(fetch_klines, job.symbol, job.timeframe, job.limit, data_source): job for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                done += 1
                try:
                    df = future.result()
                    candles[(job.symbol, job.timeframe)] = df
                    save_candles(df, job.symbol, job.timeframe, source=data_source)
                    status.info(f"Downloaded {job.symbol} {job.timeframe} ({done}/{len(jobs)})")
                except Exception as exc:
                    errors.append(f"{job.symbol} {job.timeframe}: {exc}")
                    status.warning(f"Skipped {job.symbol} {job.timeframe}: {exc}")
                progress.progress(done / max(len(jobs), 1))

        dataset = build_direction_dataset(candles, horizon=horizon, neutral_threshold=neutral_threshold)
        if dataset.empty:
            st.error("No dataset was built. Binance may be unreachable, or not enough candles were returned.")
            st.session_state.errors = errors
            return

        path = save_dataset(dataset)
        st.session_state.candles = candles
        st.session_state.dataset = dataset
        st.session_state.errors = errors
        st.success(f"Dataset built: {len(dataset):,} rows saved to {path}")

    dataset = get_dataset()
    if not dataset.empty:
        st.markdown("**Dataset Summary**")
        a, b, c = st.columns(3)
        a.metric("Rows", f"{len(dataset):,}")
        b.metric("Symbols", dataset["symbol"].nunique())
        c.metric("Timeframes", dataset["timeframe"].nunique())
        left, right = st.columns(2)
        left.dataframe(label_balance(dataset), use_container_width=True, hide_index=True)
        right.dataframe(dataset.groupby(["timeframe"]).size().reset_index(name="rows"), use_container_width=True, hide_index=True)

    if st.session_state.errors:
        with st.expander("Download/build errors"):
            st.write(st.session_state.errors[:200])


def render_data_source(data_source: str, default_timeframe: str) -> None:
    st.subheader("Inspect Binance Data Before Feature Engineering")
    st.write(
        "Use this page to see what Binance returns before we train models. "
        "Testnet account access is only used for read-only connectivity checks."
    )
    left, right, third = st.columns([1, 1, 1])
    symbol = left.text_input("Sample symbol", "BTCUSDT").upper()
    timeframe = right.selectbox("Sample timeframe", TIMEFRAMES, index=TIMEFRAMES.index(default_timeframe) if default_timeframe in TIMEFRAMES else 2)
    limit = third.slider("Sample candles", 80, 500, 160, 20)

    if st.button("Inspect selected source", type="primary"):
        try:
            info = inspect_market_data(data_source, symbol=symbol, timeframe=timeframe, limit=limit)
            st.success(f"Connected to {info['source']} endpoint: {info['endpoint']}")
            a, b, c = st.columns(3)
            a.metric("Rows returned", info["rows"])
            b.metric("Sample symbols", info["symbols_sample_count"])
            c.metric("Columns", len(info["columns"]))
            st.markdown("**Sample universe symbols**")
            st.write(info["symbols_sample"])
            st.markdown("**Raw candle columns**")
            st.write(info["columns"])
            st.dataframe(info["candles"].tail(20), use_container_width=True, hide_index=True)
            st.plotly_chart(raw_candle_chart(info["candles"], symbol, timeframe, data_source), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not inspect {data_source}: {exc}")

    st.divider()
    st.markdown("**Futures testnet account check**")
    st.caption("Requires `.env` with BINANCE_FUTURES_TESTNET_KEY and BINANCE_FUTURES_TESTNET_SECRET.")
    if st.button("Check testnet account connection"):
        result = test_futures_testnet_account()
        if result.get("ok"):
            st.success("Testnet account read works.")
            st.dataframe(pd.DataFrame(result.get("assets", [])), use_container_width=True, hide_index=True)
        else:
            st.warning(result.get("reason", "Testnet account check failed."))


def render_training(
    selected_models: list[str],
    selected_timeframes: list[str],
    test_fraction: float,
    model_workers: int,
) -> None:
    st.subheader("Live Parallel Training")
    dataset = get_dataset()
    if dataset.empty:
        st.info("Build a dataset first.")
        return
    if not selected_models or not selected_timeframes:
        st.info("Select at least one model and one timeframe.")
        return

    jobs = [(model, timeframe) for timeframe in selected_timeframes for model in selected_models]
    st.write(f"Ready to train `{len(jobs)}` model/timeframe runs.")

    if st.button("Train selected models", type="primary"):
        progress = st.progress(0)
        log_box = st.container(height=240)
        rows = []
        predictions = {}
        errors = []
        done = 0

        with ThreadPoolExecutor(max_workers=model_workers) as executor:
            futures = {
                executor.submit(train_one, dataset, model, timeframe, test_fraction): (model, timeframe)
                for model, timeframe in jobs
            }
            for future in as_completed(futures):
                model, timeframe = futures[future]
                done += 1
                key = f"{timeframe}/{model}"
                try:
                    metrics, pred = future.result()
                    rows.append(metrics)
                    predictions[key] = pred
                    with log_box:
                        st.success(
                            f"{key}: accuracy {metrics['accuracy']:.3f}, "
                            f"UP precision {metrics['up_precision']:.3f}, "
                            f"DOWN precision {metrics['down_precision']:.3f}"
                        )
                except Exception as exc:
                    errors.append(f"{key}: {exc}")
                    with log_box:
                        st.error(f"{key}: {exc}")
                progress.progress(done / max(len(jobs), 1))

        metrics_df = pd.DataFrame(rows)
        if not metrics_df.empty:
            metrics_df = metrics_df.sort_values(["timeframe", "accuracy"], ascending=[True, False])
            Path("reports").mkdir(exist_ok=True)
            metrics_df.to_csv("reports/direction_model_metrics.csv", index=False)
        st.session_state.metrics = metrics_df
        st.session_state.predictions = predictions
        st.session_state.errors = errors
        st.success("Training run complete.")

    if not st.session_state.metrics.empty:
        st.dataframe(st.session_state.metrics, use_container_width=True, hide_index=True)


def render_evaluation() -> None:
    st.subheader("Evaluation")
    metrics = st.session_state.metrics
    predictions = st.session_state.predictions
    if metrics.empty:
        maybe_metrics = Path("reports/direction_model_metrics.csv")
        if maybe_metrics.exists():
            metrics = pd.read_csv(maybe_metrics)
    if metrics.empty:
        st.info("Train models first.")
        return

    st.dataframe(metrics, use_container_width=True, hide_index=True)

    if predictions:
        by_symbol, by_label = grouped_metrics(predictions)
        left, right = st.columns(2)
        with left:
            st.markdown("**Best/Worst By Symbol**")
            st.dataframe(by_symbol, use_container_width=True, hide_index=True)
        with right:
            st.markdown("**Prediction Quality By Predicted Class**")
            st.dataframe(by_label, use_container_width=True, hide_index=True)

        key = st.selectbox("Confusion matrix run", sorted(predictions.keys()))
        st.dataframe(confusion_for(predictions[key]), use_container_width=True)


def render_candle_view() -> None:
    st.subheader("Actual Candles With Model Predictions")
    predictions = st.session_state.predictions
    if not predictions:
        st.info("Train models first. The chart needs model predictions.")
        return

    key = st.selectbox("Model run", sorted(predictions.keys()), key="chart_run")
    pred_df = predictions[key]
    symbols = sorted(pred_df["symbol"].unique())
    symbol = st.selectbox("Symbol", symbols, key="chart_symbol")
    rows = pred_df[pred_df["symbol"] == symbol].sort_values("time").tail(220)
    if rows.empty:
        st.info("No prediction rows for this symbol.")
        return

    st.plotly_chart(prediction_candle_chart(rows, symbol, key), use_container_width=True)
    st.dataframe(
        rows[["time", "close", "prediction_name", "confidence", "label_name", "future_return", "correct"]]
        .tail(80)
        .sort_values("time", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def prediction_candle_chart(rows: pd.DataFrame, symbol: str, run_key: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=rows["time"],
            open=rows["open"],
            high=rows["high"],
            low=rows["low"],
            close=rows["close"],
            name="actual candles",
        )
    )
    colors = {"UP": "#16a34a", "DOWN": "#dc2626", "NEUTRAL": "#64748b"}
    symbols = {"UP": "triangle-up", "DOWN": "triangle-down", "NEUTRAL": "circle"}
    for label in ["UP", "DOWN", "NEUTRAL"]:
        subset = rows[rows["prediction_name"] == label]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["time"],
                y=subset["close"],
                mode="markers",
                name=f"predicted {label}",
                marker={
                    "color": colors[label],
                    "symbol": symbols[label],
                    "size": 8 + subset["confidence"] * 8,
                    "line": {"color": "white", "width": 1},
                },
                customdata=subset[["label_name", "confidence", "future_return", "correct"]],
                hovertemplate=(
                    "prediction=%{fullData.name}<br>"
                    "actual=%{customdata[0]}<br>"
                    "confidence=%{customdata[1]:.2f}<br>"
                    "future return=%{customdata[2]:.2%}<br>"
                    "correct=%{customdata[3]}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=f"{symbol} {run_key}: actual candles and predictions",
        xaxis_rangeslider_visible=False,
        height=620,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend_orientation="h",
    )
    return fig


def raw_candle_chart(rows: pd.DataFrame, symbol: str, timeframe: str, source: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=rows["time"],
            open=rows["open"],
            high=rows["high"],
            low=rows["low"],
            close=rows["close"],
            name="raw Binance candles",
        )
    )
    fig.update_layout(
        title=f"{symbol} {timeframe} raw candles from {source}",
        xaxis_rangeslider_visible=False,
        height=480,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig


def get_dataset() -> pd.DataFrame:
    if not st.session_state.dataset.empty:
        return st.session_state.dataset
    path = Path("data/processed/direction_dataset.csv")
    if path.exists():
        try:
            st.session_state.dataset = load_dataset()
        except Exception:
            pass
    return st.session_state.dataset


def label_balance(dataset: pd.DataFrame) -> pd.DataFrame:
    return (
        dataset.groupby(["timeframe", "label_name"])
        .size()
        .reset_index(name="rows")
        .sort_values(["timeframe", "label_name"])
    )


if __name__ == "__main__":
    main()
