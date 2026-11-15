"""Train and compare supervised models for Part 3, Task 1.

Both tasks share one feature set (time features, cyclical hour/day-of-week/month
encodings, whole-day holiday flag and weather encodings):

* Regression: predict ``traffic_volume`` (LinearRegression baseline vs RandomForest).
* Classification: predict the proxy ``high_risk`` label (LogisticRegression
  baseline vs RandomForest).

``high_risk`` is defined from ``traffic_volume`` and ``weather_main``, so the
weather one-hot features make the classification partly circular. Its scores
describe how well the proxy rule can be reconstructed; they are not evidence of
real accident prediction. Models are evaluated on the last 20% of hours in time.
Each model and baseline is also logged as an MLflow run (see ``--mlflow-uri``).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_FRACTION = 0.20
WEATHER_PREFIX = "weather_main_"
REG_TARGET = "traffic_volume"
CLF_TARGET = "high_risk"
# Cut-off for the class decision; fixed, not tuned on the test set.
DECISION_THRESHOLD = 0.5
REQUIRED_COLUMNS = (
    "timestamp",
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    REG_TARGET,
    CLF_TARGET,
)
CONTINUOUS_FEATURES = (
    "hour",
    "day_of_week",
    "month",
    "temp_c",
    "rain_1h",
    "snow_1h",
    "clouds_all",
)


def _cyclical(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    """Sine/cosine encoding so that e.g. hour 23 is next to hour 0."""
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def _holiday_flag(df: pd.DataFrame) -> pd.Series:
    """1 for every hour of a holiday date; the source flags only the 00:00 row."""
    is_holiday = df["holiday"].notna() & (df["holiday"].astype(str).str.lower() != "none")
    dates = df["timestamp"].dt.normalize()
    flag = is_holiday.groupby(dates).transform("max").astype("int8")
    LOG.info(
        "Holiday flag: %d holiday dates spread over %d hours.",
        dates[is_holiday].nunique(),
        int(flag.sum()),
    )
    return flag


def build_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sorted data, shared feature matrix) built from the labelled CSV."""
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError("Input is missing required columns: " + ", ".join(missing))
    weather_cols = sorted(c for c in data.columns if c.startswith(WEATHER_PREFIX))
    if not weather_cols:
        raise ValueError(f"No '{WEATHER_PREFIX}*' columns found in the input.")

    df = data.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"]  # time-sorted: the split relies on row order

    feats = pd.DataFrame(index=df.index)
    feats["hour"] = ts.dt.hour
    feats["day_of_week"] = ts.dt.dayofweek
    feats["month"] = ts.dt.month
    feats["is_weekend"] = (feats["day_of_week"] >= 5).astype("int8")
    feats["hour_sin"], feats["hour_cos"] = _cyclical(feats["hour"], 24)
    feats["dow_sin"], feats["dow_cos"] = _cyclical(feats["day_of_week"], 7)
    feats["month_sin"], feats["month_cos"] = _cyclical(feats["month"], 12)

    feats["holiday_flag"] = _holiday_flag(df)

    feats["temp_c"] = df["temp"] - 273.15
    for col in ("rain_1h", "snow_1h", "clouds_all"):
        feats[col] = df[col]
    feats["is_raining"] = (df["rain_1h"] > 0).astype("int8")
    feats["is_snowing"] = (df["snow_1h"] > 0).astype("int8")
    for col in weather_cols:
        feats[col] = df[col].astype("int8")

    if feats.isna().any().any():
        bad = feats.columns[feats.isna().any()].tolist()
        raise ValueError("Feature matrix contains missing values in: " + ", ".join(bad))
    return df, feats


def time_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION) -> int:
    """Return the first test row index; rows must already be time-sorted."""
    cut = int(round(len(df) * (1 - test_fraction)))
    LOG.info(
        "Time split: train %d rows (%s to %s), test %d rows (%s to %s).",
        cut,
        df["timestamp"].iloc[0],
        df["timestamp"].iloc[cut - 1],
        len(df) - cut,
        df["timestamp"].iloc[cut],
        df["timestamp"].iloc[-1],
    )
    return cut


def _preprocessor() -> ColumnTransformer:
    """Scale numeric columns for the linear models; trees are scale-invariant."""
    return ColumnTransformer(
        [("num", StandardScaler(), list(CONTINUOUS_FEATURES))],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def classification_metrics(y_true, proba) -> dict[str, float]:
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, proba),
        "PR_AUC": average_precision_score(y_true, proba),
    }


def _regression_models() -> dict:
    return {
        "Linear Regression": Pipeline([("prep", _preprocessor()), ("model", LinearRegression())]),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }


def _classification_models() -> dict:
    return {
        "Logistic Regression": Pipeline(
            [
                ("prep", _preprocessor()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def _fit_and_score(task, models, X_train, y_train, score):
    """Fit each model; return metric rows and the fitted models keyed by (task, name)."""
    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = score(model)
        LOG.info("%s %s: %s.", task.capitalize(), name, ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        rows.append({"task": task, "model": name, **metrics})
        fitted[(task, name)] = model
    return rows, fitted


def run_regression(X_train, X_test, y_train, y_test):
    """Score two reference baselines and the regression models on the test set."""
    mean_by_hour = X_test["hour"].map(y_train.groupby(X_train["hour"]).mean()).fillna(y_train.mean())
    train_mean = np.full(len(y_test), y_train.mean())
    rows = [
        {"task": "regression", "model": "Baseline: train mean", **regression_metrics(y_test, train_mean)},
        {"task": "regression", "model": "Baseline: mean by hour", **regression_metrics(y_test, mean_by_hour)},
    ]
    model_rows, fitted = _fit_and_score(
        "regression",
        _regression_models(),
        X_train,
        y_train,
        lambda model: regression_metrics(y_test, model.predict(X_test)),
    )
    return rows + model_rows, fitted


def run_classification(X_train, X_test, y_train, y_test):
    """Score the classification models on the test set."""
    return _fit_and_score(
        "classification",
        _classification_models(),
        X_train,
        y_train,
        lambda model: classification_metrics(y_test, model.predict_proba(X_test)[:, 1]),
    )


def importance_table(fitted: dict, columns) -> pd.DataFrame:
    """Random Forest feature importances for both tasks, largest regression first."""
    table = pd.DataFrame(
        {
            f"rf_{task}_importance": fitted[(task, "Random Forest")].feature_importances_
            for task in ("regression", "classification")
        },
        index=columns,
    )
    return table.sort_values("rf_regression_importance", ascending=False).rename_axis("feature").reset_index()


def train_and_evaluate(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Return (metrics, feature importances, run info, fitted models)."""
    df, feats = build_features(data)
    LOG.info("Shared feature set: %d features.", feats.shape[1])

    cut = time_split(df)
    X_train, X_test = feats.iloc[:cut], feats.iloc[cut:]
    y_reg_train, y_reg_test = df[REG_TARGET].iloc[:cut], df[REG_TARGET].iloc[cut:]
    y_clf_train, y_clf_test = df[CLF_TARGET].iloc[:cut], df[CLF_TARGET].iloc[cut:]

    if y_clf_train.nunique() < 2 or y_clf_test.nunique() < 2:
        raise ValueError("Train and test sets must each contain both high_risk classes.")
    LOG.info(
        "high_risk positives: train %d (%.2f%%), test %d (%.2f%%).",
        y_clf_train.sum(),
        100 * y_clf_train.mean(),
        y_clf_test.sum(),
        100 * y_clf_test.mean(),
    )
    if y_clf_test.mean() < 0.03:
        LOG.warning("Test positive rate is low; classification metrics are unstable.")

    reg_rows, reg_models = run_regression(X_train, X_test, y_reg_train, y_reg_test)
    clf_rows, clf_models = run_classification(X_train, X_test, y_clf_train, y_clf_test)
    models = {**reg_models, **clf_models}

    info = {
        "n_train": cut,
        "n_test": len(df) - cut,
        "train_end": str(df["timestamp"].iloc[cut - 1]),
        "test_start": str(df["timestamp"].iloc[cut]),
        "test_end": str(df["timestamp"].iloc[-1]),
        "test_positive_rate": float(y_clf_test.mean()),
        "decision_threshold": DECISION_THRESHOLD,
        "n_features": feats.shape[1],
        "features": feats.columns.tolist(),
    }
    return pd.DataFrame(reg_rows + clf_rows), importance_table(models, feats.columns), info, models


def log_to_mlflow(metrics, models, info, output_dir, tracking_uri, artifact_root, experiment):
    """Log one MLflow run per model (and baseline), with params, metrics and model."""
    mlflow.set_tracking_uri(tracking_uri)
    # The artifact location is fixed when an experiment is created; otherwise MLflow
    # would write artifacts relative to the current working directory.
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=artifact_root)
    mlflow.set_experiment(experiment)

    common_params = {
        "split": f"time-based {1 - TEST_FRACTION:.0%}/{TEST_FRACTION:.0%}",
        "random_state": RANDOM_STATE,
        "label_note": "high_risk is a proxy label; weather_main is both a feature and a label input",
        **{k: info[k] for k in ("train_end", "test_start", "n_train", "n_test", "n_features")},
    }
    for row in metrics.to_dict("records"):
        task, name = row["task"], row["model"]
        model = models.get((task, name))  # None for the reference baselines
        scores = {k: float(v) for k, v in row.items() if k not in ("task", "model") and pd.notna(v)}
        with mlflow.start_run(run_name=f"{task} - {name}"):
            mlflow.set_tags({"task": task, "model": name, "is_baseline": str(model is None)})
            mlflow.log_params(common_params)
            mlflow.log_metrics(scores)
            if task == "classification":
                mlflow.log_param("decision_threshold", DECISION_THRESHOLD)
            if model is not None:
                estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
                mlflow.log_params({f"model__{k}": v for k, v in estimator.get_params().items()})
                # The models are produced by this script, so pickle is acceptable here.
                mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
            mlflow.log_artifact(str(output_dir / "model_metrics.csv"))
            mlflow.log_artifact(str(output_dir / "feature_importance.csv"))
    LOG.info("Logged %d MLflow runs to %s (experiment '%s').", len(metrics), tracking_uri, experiment)


def _save_outputs(output_dir: Path, metrics, importance, info) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "model_metrics.csv"
    importance_path = output_dir / "feature_importance.csv"
    info_path = output_dir / "run_info.json"
    metrics.round(4).to_csv(metrics_path, index=False, encoding="utf-8")
    LOG.info("Saved model metrics to %s.", metrics_path)
    importance.round(5).to_csv(importance_path, index=False, encoding="utf-8")
    LOG.info("Saved feature importances to %s.", importance_path)
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    LOG.info("Saved run information to %s.", info_path)


def _parse_args() -> argparse.Namespace:
    part_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=part_dir / "data" / "processed" / "Metro_Interstate_Traffic_Volume_ml.csv",
        help="Labelled CSV from prepare_labels.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=part_dir / "outputs",
        help="Directory for metrics and feature-importance files.",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=f"sqlite:///{(part_dir / 'mlflow.db').as_posix()}",
        help="MLflow tracking URI (default: SQLite database Part-3/mlflow.db).",
    )
    parser.add_argument(
        "--mlflow-artifact-root",
        default=(part_dir / "mlartifacts").as_uri(),
        help="Artifact location used when creating the experiment (default: Part-3/mlartifacts).",
    )
    parser.add_argument("--experiment", default="task1-supervised-models", help="MLflow experiment name.")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    parser.add_argument(
        "--log-file", type=Path, default=part_dir / "logs" / "train_models.log", help="Log file path."
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Minimum level written to the console and log file (default: INFO).",
    )
    return parser.parse_args()


def _configure_logging(log_path: Path, level: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
        force=True,
    )


def main() -> int:
    args = _parse_args()
    try:
        _configure_logging(args.log_file, getattr(logging, args.log_level))
    except OSError:
        logging.basicConfig(level=logging.ERROR, stream=sys.stderr, force=True)
        LOG.exception("Failed to configure logging.")
        return 1

    try:
        LOG.info("Reading %s.", args.input)
        data = pd.read_csv(args.input, encoding="utf-8-sig")
        LOG.info("Loaded %d rows, %d columns.", *data.shape)
        metrics, importance, info, models = train_and_evaluate(data)
        _save_outputs(args.output_dir, metrics, importance, info)
        if not args.no_mlflow:
            log_to_mlflow(
                metrics, models, info, args.output_dir, args.mlflow_uri, args.mlflow_artifact_root, args.experiment
            )
    except (OSError, UnicodeError, MlflowException, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        LOG.exception("Model training failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
