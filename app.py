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


st.set_page_config(page_title="AI Market Prediction Lab", page_icon="chart_with_upwards_trend", layout="wide")


def init_state() -> None:
    st.session_state.setdefault("candles", {})
    st.session_state.setdefault("dataset", pd.DataFrame())
    st.session_state.setdefault("metrics", pd.DataFrame())
    st.session_state.setdefault("predictions", {})
    st.session_state.setdefault("errors", [])


def main() -> None:
    init_state()
    st.title("AI Market Prediction Lab")
    st.caption("Simple first version: get Binance candles, train direction models, and check if UP/DOWN predictions were correct. No trading. No orders.")

    settings = render_settings()

    show_stage_header("1", "Check Binance Data", "First we verify what Binance gives us: columns, candles, and sample symbols.")
    render_data_check(settings)

    show_stage_header("2", "Build Prediction Dataset", "Then we turn candles into features and labels: did price go UP, DOWN, or stay NEUTRAL after the chosen horizon?")
    render_dataset_builder(settings)

    show_stage_header("3", "Train Models", "Next we train multiple models in parallel for each timeframe and measure prediction quality.")
    render_training(settings)

    show_stage_header("4", "View Candles + Predictions", "Finally we plot real candles with model predictions on top, so you can see where the model was right or wrong.")
    render_prediction_view()


def render_settings() -> dict:
    with st.sidebar:
        st.header("Simple Setup")
        source_label = st.radio(
            "Binance data",
            ["Mainnet public candles", "Futures testnet candles"],
            help="Use mainnet public candles for normal model training. Use testnet when you only want to test the sandbox connection.",
        )
        source = "testnet" if source_label.startswith("Futures") else "mainnet"

        st.subheader("Data")
        symbols_limit = st.slider("Coins to use", 5, 260, 25, 5)
        timeframes = st.multiselect("Candle timeframes", TIMEFRAMES, default=["15m", "1h", "4h"])
        candle_limit = st.slider("Candles per coin/timeframe", 200, 1500, 600, 100)
        min_quote_volume = st.number_input("Minimum 24h quote volume", min_value=0, value=50_000_000, step=10_000_000)

        st.subheader("Prediction Label")
        horizon = st.slider("Predict after N candles", 1, 24, 4)
        neutral_threshold = st.slider("Neutral zone (%)", 0.05, 3.0, 0.30, 0.05) / 100

        st.subheader("Training")
        models = st.multiselect("Models", available_model_names(), default=DEFAULT_MODELS)
        test_fraction = st.slider("Newest data for testing (%)", 10, 40, 25, 5) / 100
        data_workers = st.slider("Parallel data jobs", 2, 24, 10)
        model_workers = st.slider("Parallel model jobs", 1, 8, 3)

    return {
        "source": source,
        "symbols_limit": symbols_limit,
        "timeframes": timeframes,
        "candle_limit": candle_limit,
        "min_quote_volume": min_quote_volume,
        "horizon": horizon,
        "neutral_threshold": neutral_threshold,
        "models": models,
        "test_fraction": test_fraction,
        "data_workers": data_workers,
        "model_workers": model_workers,
    }


def show_stage_header(number: str, title: str, text: str) -> None:
    st.divider()
    st.subheader(f"{number}. {title}")
    st.write(text)


def render_data_check(settings: dict) -> None:
    left, middle, right = st.columns([1, 1, 1])
    symbol = left.text_input("Sample coin", "BTCUSDT").upper()
    timeframe = middle.selectbox("Sample timeframe", TIMEFRAMES, index=2)
    limit = right.slider("Sample candles", 80, 500, 160, 20)

    c1, c2 = st.columns([1, 1])
    if c1.button("Check candle data", type="primary", use_container_width=True):
        try:
            info = inspect_market_data(settings["source"], symbol=symbol, timeframe=timeframe, limit=limit)
            st.success(f"Binance connection works: {info['source']} endpoint")
            a, b, c = st.columns(3)
            a.metric("Rows returned", info["rows"])
            b.metric("Available sample symbols", info["symbols_sample_count"])
            c.metric("Columns", len(info["columns"]))
            st.plotly_chart(raw_candle_chart(info["candles"], symbol, timeframe, settings["source"]), use_container_width=True)
            with st.expander("Show raw Binance rows and columns"):
                st.write(info["columns"])
                st.dataframe(info["candles"].tail(25), use_container_width=True, hide_index=True)
                st.write(info["symbols_sample"])
        except Exception as exc:
            st.error(f"Could not read Binance candles: {exc}")

    if c2.button("Check testnet account keys", use_container_width=True):
        result = test_futures_testnet_account()
        if result.get("ok"):
            st.success("Testnet account read works. Keys are loaded.")
            st.dataframe(pd.DataFrame(result.get("assets", [])), use_container_width=True, hide_index=True)
        else:
            st.warning(result.get("reason", "Testnet account check failed."))


def render_dataset_builder(settings: dict) -> None:
    jobs_count = settings["symbols_limit"] * len(settings["timeframes"])
    a, b, c, d = st.columns(4)
    a.metric("Coins", settings["symbols_limit"])
    b.metric("Timeframes", len(settings["timeframes"]))
    c.metric("Candles each", settings["candle_limit"])
    d.metric("Download jobs", jobs_count)

    disabled = not settings["timeframes"]
    if st.button("Build dataset", type="primary", disabled=disabled, use_container_width=True):
        st.session_state.errors = []
        try:
            symbols = get_usdt_perp_symbols(
                limit=settings["symbols_limit"],
                min_quote_volume=settings["min_quote_volume"],
                source=settings["source"],
            )
        except Exception as exc:
            st.error(f"Could not load symbol list from Binance: {exc}")
            return

        jobs = [
            CandleRequest(symbol, timeframe, settings["candle_limit"])
            for symbol in symbols
            for timeframe in settings["timeframes"]
        ]
        candles, errors = download_candles(jobs, settings)
        dataset = build_direction_dataset(
            candles,
            horizon=settings["horizon"],
            neutral_threshold=settings["neutral_threshold"],
        )

        if dataset.empty:
            st.error("No training dataset was created. Try fewer filters, more candles, or mainnet public candles.")
            st.session_state.errors = errors
            return

        path = save_dataset(dataset)
        st.session_state.candles = candles
        st.session_state.dataset = dataset
        st.session_state.errors = errors
        st.success(f"Dataset ready: {len(dataset):,} rows saved to {path}")

    dataset = get_dataset()
    if not dataset.empty:
        a, b, c = st.columns(3)
        a.metric("Dataset rows", f"{len(dataset):,}")
        b.metric("Coins in dataset", dataset["symbol"].nunique())
        c.metric("Timeframes in dataset", dataset["timeframe"].nunique())
        left, right = st.columns(2)
        left.dataframe(label_balance(dataset), use_container_width=True, hide_index=True)
        right.dataframe(dataset.groupby("timeframe").size().reset_index(name="rows"), use_container_width=True, hide_index=True)

    if st.session_state.errors:
        with st.expander("Download or build errors"):
            st.write(st.session_state.errors[:200])


def download_candles(jobs: list[CandleRequest], settings: dict) -> tuple[dict[tuple[str, str], pd.DataFrame], list[str]]:
    candles: dict[tuple[str, str], pd.DataFrame] = {}
    errors: list[str] = []
    progress = st.progress(0)
    status = st.empty()
    done = 0

    with ThreadPoolExecutor(max_workers=settings["data_workers"]) as executor:
        futures = {
            executor.submit(fetch_klines, job.symbol, job.timeframe, job.limit, settings["source"]): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            done += 1
            try:
                df = future.result()
                candles[(job.symbol, job.timeframe)] = df
                save_candles(df, job.symbol, job.timeframe, source=settings["source"])
                status.info(f"Downloaded {job.symbol} {job.timeframe} ({done}/{len(jobs)})")
            except Exception as exc:
                errors.append(f"{job.symbol} {job.timeframe}: {exc}")
                status.warning(f"Skipped {job.symbol} {job.timeframe}: {exc}")
            progress.progress(done / max(len(jobs), 1))

    return candles, errors


def render_training(settings: dict) -> None:
    dataset = get_dataset()
    if dataset.empty:
        st.info("Build the dataset first.")
        return
    if not settings["models"] or not settings["timeframes"]:
        st.info("Choose at least one model and one timeframe.")
        return

    jobs = [(model, timeframe) for timeframe in settings["timeframes"] for model in settings["models"]]
    st.write(f"Ready to train {len(jobs)} model/timeframe runs.")

    if st.button("Train models", type="primary", use_container_width=True):
        progress = st.progress(0)
        log_box = st.container(height=240)
        rows = []
        predictions = {}
        errors = []
        done = 0

        with ThreadPoolExecutor(max_workers=settings["model_workers"]) as executor:
            futures = {
                executor.submit(train_one, dataset, model, timeframe, settings["test_fraction"]): (model, timeframe)
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

        metrics = pd.DataFrame(rows)
        if not metrics.empty:
            metrics = metrics.sort_values(["timeframe", "accuracy"], ascending=[True, False])
            Path("reports").mkdir(exist_ok=True)
            metrics.to_csv("reports/direction_model_metrics.csv", index=False)

        st.session_state.metrics = metrics
        st.session_state.predictions = predictions
        st.session_state.errors = errors
        st.success("Training finished.")

    metrics = get_metrics()
    if not metrics.empty:
        st.dataframe(metrics, use_container_width=True, hide_index=True)

    if st.session_state.predictions:
        by_symbol, by_label = grouped_metrics(st.session_state.predictions)
        left, right = st.columns(2)
        left.dataframe(by_symbol, use_container_width=True, hide_index=True)
        right.dataframe(by_label, use_container_width=True, hide_index=True)
        run_key = st.selectbox("Confusion matrix", sorted(st.session_state.predictions.keys()))
        st.dataframe(confusion_for(st.session_state.predictions[run_key]), use_container_width=True)


def render_prediction_view() -> None:
    predictions = st.session_state.predictions
    if not predictions:
        st.info("Train models first. Then this chart will show real candles with UP/DOWN/NEUTRAL predictions.")
        return

    run_key = st.selectbox("Model run", sorted(predictions.keys()), key="chart_run")
    pred_df = predictions[run_key]
    symbol = st.selectbox("Coin", sorted(pred_df["symbol"].unique()), key="chart_symbol")
    rows = pred_df[pred_df["symbol"] == symbol].sort_values("time").tail(220)
    if rows.empty:
        st.info("No prediction rows for this coin.")
        return

    st.plotly_chart(prediction_candle_chart(rows, symbol, run_key), use_container_width=True)
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

    colors = {"UP": "#16803c", "DOWN": "#c2410c", "NEUTRAL": "#475569"}
    markers = {"UP": "triangle-up", "DOWN": "triangle-down", "NEUTRAL": "circle"}
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
                    "symbol": markers[label],
                    "size": 8 + subset["confidence"] * 8,
                    "line": {"color": "white", "width": 1},
                },
                customdata=subset[["label_name", "confidence", "future_return", "correct"]],
                hovertemplate=(
                    "actual=%{customdata[0]}<br>"
                    "confidence=%{customdata[1]:.2f}<br>"
                    "future return=%{customdata[2]:.2%}<br>"
                    "correct=%{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"{symbol} {run_key}: real candles and model predictions",
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


def get_metrics() -> pd.DataFrame:
    if not st.session_state.metrics.empty:
        return st.session_state.metrics
    path = Path("reports/direction_model_metrics.csv")
    if path.exists():
        try:
            st.session_state.metrics = pd.read_csv(path)
        except Exception:
            pass
    return st.session_state.metrics


def label_balance(dataset: pd.DataFrame) -> pd.DataFrame:
    return (
        dataset.groupby(["timeframe", "label_name"])
        .size()
        .reset_index(name="rows")
        .sort_values(["timeframe", "label_name"])
    )


if __name__ == "__main__":
    main()
