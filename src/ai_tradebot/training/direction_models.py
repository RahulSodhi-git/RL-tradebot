from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_tradebot.features.dataset import feature_columns


MODEL_DIR = Path("models/direction")
REPORT_DIR = Path("reports")


@dataclass(frozen=True)
class ModelRun:
    name: str
    timeframe: str


def available_model_names() -> list[str]:
    return ["logistic_regression", "random_forest", "hist_gradient_boosting", "extra_trees"]


def make_model(name: str):
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=180, learning_rate=0.06, random_state=42)
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=180,
            max_depth=14,
            min_samples_leaf=6,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    raise ValueError(f"unknown model: {name}")


def train_many(
    dataset: pd.DataFrame,
    model_names: list[str],
    timeframes: list[str],
    test_fraction: float = 0.25,
    max_workers: int = 4,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    jobs = [ModelRun(model, timeframe) for timeframe in timeframes for model in model_names]
    rows = []
    predictions: dict[str, pd.DataFrame] = {}
    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(train_one, dataset, job.name, job.timeframe, test_fraction): job for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            key = f"{job.timeframe}/{job.name}"
            try:
                metrics, pred = future.result()
                rows.append(metrics)
                predictions[key] = pred
            except Exception as exc:
                errors.append(f"{key}: {exc}")
    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(["timeframe", "accuracy"], ascending=[True, False])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_DIR / "direction_model_metrics.csv", index=False)
    return report, predictions, errors


def train_one(
    dataset: pd.DataFrame,
    model_name: str,
    timeframe: str,
    test_fraction: float = 0.25,
) -> tuple[dict, pd.DataFrame]:
    data = dataset[dataset["timeframe"] == timeframe].sort_values("time").reset_index(drop=True)
    if len(data) < 300:
        raise ValueError(f"not enough rows for {timeframe}: {len(data)}")
    split = int(len(data) * (1 - test_fraction))
    train = data.iloc[:split]
    test = data.iloc[split:]
    features = feature_columns()
    model = make_model(model_name)
    model.fit(train[features], train["label"])
    pred = model.predict(test[features])
    proba = _predict_proba(model, test[features], pred)

    pred_df = test[
        ["symbol", "timeframe", "time", "open", "high", "low", "close", "future_return", "label", "label_name"]
    ].copy()
    pred_df["prediction"] = pred
    pred_df["prediction_name"] = pred_df["prediction"].map({-1: "DOWN", 0: "NEUTRAL", 1: "UP"})
    pred_df["confidence"] = proba
    pred_df["correct"] = pred_df["prediction"] == pred_df["label"]

    metrics = metric_row(pred_df, model_name, timeframe)
    save_model(model, model_name, timeframe)
    pred_path = REPORT_DIR / "predictions" / timeframe
    pred_path.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_path / f"{model_name}.csv", index=False)
    return metrics, pred_df


def metric_row(pred_df: pd.DataFrame, model_name: str, timeframe: str) -> dict:
    y_true = pred_df["label"]
    y_pred = pred_df["prediction"]
    high_conf = pred_df[pred_df["confidence"] >= 0.60]
    return {
        "model": model_name,
        "timeframe": timeframe,
        "rows_tested": len(pred_df),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "up_precision": round(float(precision_score(y_true, y_pred, labels=[1], average="micro", zero_division=0)), 4),
        "down_precision": round(float(precision_score(y_true, y_pred, labels=[-1], average="micro", zero_division=0)), 4),
        "neutral_precision": round(float(precision_score(y_true, y_pred, labels=[0], average="micro", zero_division=0)), 4),
        "high_conf_accuracy": round(float(high_conf["correct"].mean()) if len(high_conf) else 0.0, 4),
    }


def grouped_metrics(predictions: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for key, df in predictions.items():
        timeframe, model = key.split("/", 1)
        item = df.copy()
        item["model"] = model
        item["run_timeframe"] = timeframe
        frames.append(item)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    all_pred = pd.concat(frames, ignore_index=True)
    by_symbol = (
        all_pred.groupby(["model", "timeframe", "symbol"])
        .agg(rows=("correct", "size"), accuracy=("correct", "mean"), avg_confidence=("confidence", "mean"))
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )
    by_label = (
        all_pred.groupby(["model", "timeframe", "prediction_name"])
        .agg(rows=("correct", "size"), accuracy=("correct", "mean"), avg_confidence=("confidence", "mean"))
        .reset_index()
    )
    return by_symbol, by_label


def confusion_for(pred_df: pd.DataFrame) -> pd.DataFrame:
    labels = [-1, 0, 1]
    matrix = confusion_matrix(pred_df["label"], pred_df["prediction"], labels=labels)
    names = ["DOWN", "NEUTRAL", "UP"]
    return pd.DataFrame(matrix, index=[f"actual_{n}" for n in names], columns=[f"pred_{n}" for n in names])


def save_model(model, model_name: str, timeframe: str) -> Path:
    path = MODEL_DIR / timeframe
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"{model_name}.joblib"
    joblib.dump(model, file_path)
    return file_path


def _predict_proba(model, x: pd.DataFrame, pred) -> list[float]:
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probs = model.predict_proba(x)
        return [float(row[classes.index(label)]) if label in classes else 0.0 for row, label in zip(probs, pred)]
    return [0.5 for _ in range(len(x))]

