"""Deep learning with explainability for Part 3, Task 3.

An LSTM predicts the next hour's ``traffic_volume`` from the previous 24 hourly
readings (volume and weather) plus the calendar features of the hour being
predicted. An LSTM is hard to explain directly, so SHAP is applied to a
comparable Random Forest trained on the same targets, using volume lags and
last-hour weather. The LSTM uses all 24 hours of weather. The SHAP findings
describe the Random Forest, not the LSTM itself.

Windows are built only inside runs of consecutive hourly rows, so no window spans
a gap in the data. Evaluation uses the same chronological 80/20 split as Task 1:
every test window predicts an hour in the last 20% of the data. The target is
real traffic volume, not the proxy ``high_risk`` label.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import shap
from mlflow.exceptions import MlflowException
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from train_models import RANDOM_STATE, build_features, time_split

LOG = logging.getLogger(__name__)

WINDOW = 24  # hours of history per prediction
SEQ_FEATURES = ("traffic_volume", "temp_c", "rain_1h", "snow_1h", "clouds_all")
WEATHER_FEATURES = SEQ_FEATURES[1:]
CAL_FEATURES = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "holiday_flag")
# Readable calendar features for the tree model (trees do not need sin/cos).
TREE_CAL_FEATURES = ("hour", "day_of_week", "is_weekend", "holiday_flag")
VAL_FRACTION = 0.10  # last part of the training windows, used for early stopping
LSTM_UNITS = 64
DROPOUT = 0.2
BATCH_SIZE = 128
MAX_EPOCHS = 80
PATIENCE = 5
LEARNING_RATE = 1e-3
RF_TREES = 200
SHAP_SAMPLE = 500


def restore_and_impute_weather(data: pd.DataFrame, raw: pd.DataFrame, train_end: int):
    """Restore raw weather and fit monthly imputation on the LSTM training period."""
    required = ("date_time", "traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all")
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError("Raw input is missing columns: " + ", ".join(missing))
    source = raw.copy()
    source["timestamp"] = pd.to_datetime(source["date_time"], format="%d/%m/%Y %H:%M", errors="raise")
    source = source.drop_duplicates("timestamp", keep="first").set_index("timestamp")
    timestamps = pd.DatetimeIndex(data["timestamp"])
    if timestamps.has_duplicates or timestamps.hasnans:
        raise ValueError("Processed input must have unique, non-missing timestamps.")
    if not timestamps.isin(source.index).all():
        raise ValueError("Raw input does not contain every processed timestamp.")
    source = source.loc[timestamps]
    for col in ("traffic_volume", "snow_1h", "clouds_all", "temp", "rain_1h"):
        source[col] = pd.to_numeric(source[col], errors="raise")
        if not np.isfinite(source[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Raw {col} contains non-finite values.")
    for col in ("traffic_volume", "snow_1h", "clouds_all"):
        if not np.allclose(data[col].to_numpy(dtype=float), source[col].to_numpy(), rtol=0, atol=1e-8):
            raise ValueError(f"Raw and processed {col} do not match.")

    result = data.copy()
    months = timestamps.month
    medians = {}
    counts = {}
    for col in ("temp", "rain_1h"):
        values = source[col].to_numpy(dtype=float).copy()
        if col == "rain_1h" and (values < 0).any():
            raise ValueError("Raw rain_1h contains negative values.")
        invalid = values <= 0 if col == "temp" else values > 9000
        processed = data[col].to_numpy(dtype=float)
        if not np.allclose(processed[~invalid], values[~invalid], rtol=0, atol=1e-8):
            raise ValueError(f"Raw and processed {col} differ outside imputed rows.")
        medians[col] = {}
        for month in sorted(set(months)):
            eligible = (months[:train_end] == month) & ~invalid[:train_end]
            if not eligible.any():
                if ((months == month) & invalid).any():
                    raise ValueError(f"No valid training values for {col} in month {month}.")
                continue
            median = float(np.median(values[:train_end][eligible]))
            medians[col][str(month)] = median
            values[(months == month) & invalid] = median
        result[col] = values
        counts[col] = int(invalid.sum())
        if counts[col]:
            LOG.warning("Re-imputed %d %s values using training-period monthly medians.", counts[col], col)
    return result, medians, counts


def contiguous_runs(timestamps: pd.Series) -> list[tuple[int, int]]:
    """Return (start, stop) row ranges whose rows are exactly one hour apart."""
    run_id = (timestamps.diff() != pd.Timedelta(hours=1)).cumsum().to_numpy()
    bounds = np.flatnonzero(np.diff(run_id)) + 1
    starts = np.concatenate(([0], bounds))
    stops = np.concatenate((bounds, [len(run_id)]))
    return list(zip(starts.tolist(), stops.tolist()))


def window_targets(timestamps: pd.Series, window: int = WINDOW) -> np.ndarray:
    """Row indices that can be predicted: they have `window` consecutive hours before them."""
    targets = [np.arange(start + window, stop) for start, stop in contiguous_runs(timestamps)]
    return np.concatenate(targets) if targets else np.array([], dtype=int)


def build_windows(df: pd.DataFrame, feats: pd.DataFrame, train_end: int):
    """Return the scaled LSTM inputs and the flattened lag table for every window."""
    target_idx = window_targets(df["timestamp"])
    history = target_idx[:, None] + np.arange(-WINDOW, 0)[None, :]

    raw = pd.DataFrame({"traffic_volume": df["traffic_volume"].to_numpy()})
    for col in WEATHER_FEATURES:
        raw[col] = feats[col].to_numpy()
    scaler = StandardScaler().fit(raw.iloc[:train_end])
    seq_all = scaler.transform(raw).astype("float32")
    x_seq = seq_all[history]
    x_cal = feats.loc[target_idx, list(CAL_FEATURES)].to_numpy("float32")
    y = df["traffic_volume"].to_numpy("float32")[target_idx]
    y_scaled = ((y - scaler.mean_[0]) / scaler.scale_[0]).astype("float32")

    # Flattened twin of the LSTM input: lag_1 is the most recent hour before the target.
    values = raw.to_numpy()
    lags = {f"volume_lag_{k}": values[target_idx - k, 0] for k in range(1, WINDOW + 1)}
    lag_table = pd.DataFrame(lags)
    for j, col in enumerate(WEATHER_FEATURES, start=1):
        lag_table[f"{col}_lag_1"] = values[target_idx - 1, j]
    for col in TREE_CAL_FEATURES:
        lag_table[col] = feats[col].to_numpy()[target_idx]

    return {
        "target_idx": target_idx,
        "x_seq": x_seq,
        "x_cal": x_cal,
        "y": y,
        "y_scaled": y_scaled,
        "lag_table": lag_table,
        "volume_mean": float(scaler.mean_[0]),
        "volume_scale": float(scaler.scale_[0]),
        "scaler": scaler,
    }


def split_windows(win: dict, cut: int):
    """Chronological train / validation / test masks (by target row)."""
    is_test = win["target_idx"] >= cut
    train_pos = np.flatnonzero(~is_test)
    n_val = int(round(len(train_pos) * VAL_FRACTION))
    if n_val < 1 or n_val >= len(train_pos) or not is_test.any():
        raise ValueError("Insufficient windows for nonempty training, validation and test splits.")
    return train_pos[:-n_val], train_pos[-n_val:], np.flatnonzero(is_test)


def regression_scores(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_lstm(win: dict, train_pos, val_pos):
    """Fit the LSTM; two inputs: the 24-hour sequence and the target-hour calendar."""
    import keras
    from keras import layers

    keras.utils.set_random_seed(RANDOM_STATE)
    seq_in = keras.Input(shape=win["x_seq"].shape[1:], name="sequence")
    cal_in = keras.Input(shape=win["x_cal"].shape[1:], name="target_hour_calendar")
    h = layers.LSTM(LSTM_UNITS)(seq_in)
    h = layers.Dropout(DROPOUT)(h)
    h = layers.Concatenate()([h, cal_in])
    h = layers.Dense(32, activation="relu")(h)
    out = layers.Dense(1)(h)
    model = keras.Model([seq_in, cal_in], out)
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse")

    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    history = model.fit(
        [win["x_seq"][train_pos], win["x_cal"][train_pos]],
        win["y_scaled"][train_pos],
        validation_data=([win["x_seq"][val_pos], win["x_cal"][val_pos]], win["y_scaled"][val_pos]),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[stop],
        verbose=0,
    )
    LOG.info(
        "LSTM trained for %d epochs (best val loss %.4f).",
        len(history.history["loss"]),
        min(history.history["val_loss"]),
    )
    return model, history.history


def lstm_predict(model, win: dict, pos) -> np.ndarray:
    scaled = model.predict([win["x_seq"][pos], win["x_cal"][pos]], batch_size=512, verbose=0).ravel()
    return np.maximum(0, scaled * win["volume_scale"] + win["volume_mean"])


def predict_from_saved(model, sequence, calendar, preprocessing: dict) -> np.ndarray:
    """Predict from imputed, unscaled sequence and calendar arrays using saved metadata."""
    sequence = np.asarray(sequence, dtype=float)
    calendar = np.asarray(calendar, dtype=float)
    expected = (preprocessing["window_hours"], len(preprocessing["sequence_features"]))
    if sequence.ndim != 3 or sequence.shape[1:] != expected:
        raise ValueError(f"Sequence must have shape (n, {expected[0]}, {expected[1]}).")
    if calendar.shape != (len(sequence), len(preprocessing["calendar_features"])):
        raise ValueError("Calendar must have one row per sequence and the saved feature count.")
    if not np.isfinite(sequence).all() or not np.isfinite(calendar).all():
        raise ValueError("Prediction inputs must be finite.")
    mean = np.asarray(preprocessing["scaler_mean"])
    scale = np.asarray(preprocessing["scaler_scale"])
    scaled = ((sequence - mean) / scale).astype("float32")
    raw = model.predict([scaled, calendar.astype("float32")], verbose=0).ravel()
    return np.maximum(0, raw * scale[0] + mean[0])


class LstmForecaster(mlflow.pyfunc.PythonModel):
    """MLflow wrapper so the saved Keras LSTM and its preprocessing load as one model."""

    def load_context(self, context):
        import keras

        self.model = keras.saving.load_model(context.artifacts["model"])
        self.preprocessing = json.loads(Path(context.artifacts["preprocessing"]).read_text(encoding="utf-8"))

    def predict(self, context, model_input, params=None):
        return predict_from_saved(
            self.model, model_input["sequence"], model_input["calendar"], self.preprocessing
        )


def fit_forest(x_train: pd.DataFrame, y_train) -> RandomForestRegressor:
    forest = RandomForestRegressor(
        n_estimators=RF_TREES, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
    )
    return forest.fit(x_train, y_train)


def shap_table(forest, x_sample: pd.DataFrame):
    explainer = shap.TreeExplainer(forest)
    values = explainer.shap_values(x_sample)
    importance = (
        pd.DataFrame({"feature": x_sample.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance["share"] = importance["mean_abs_shap"] / importance["mean_abs_shap"].sum()
    return explainer, values, importance


def group_importance(importance: pd.DataFrame) -> pd.DataFrame:
    """Sum |SHAP| shares per feature family (recent lags, day-ago lags, weather, calendar)."""

    def family(name: str) -> str:
        if name.startswith("volume_lag_"):
            lag = int(name.rsplit("_", 1)[1])
            return "volume lag 1 (previous hour)" if lag == 1 else (
                "volume lags 2-6" if lag <= 6 else "volume lags 7-24"
            )
        if name.endswith("_lag_1"):
            return "weather"
        return "calendar"

    grouped = importance.assign(group=importance["feature"].map(family))
    return grouped.groupby("group", as_index=False)[["mean_abs_shap", "share"]].sum().sort_values(
        "mean_abs_shap", ascending=False
    )


def _save_fig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s.", path)


def make_plots(output_dir: Path, history, preds: pd.DataFrame, explainer, values, x_sample,
               x_sample_nolag, values_nolag, local_cases) -> list[Path]:
    paths = []

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(history["loss"], label="train")
    ax.plot(history["val_loss"], label="validation")
    ax.set(xlabel="epoch", ylabel="MSE (scaled)", title="LSTM training curve")
    ax.legend()
    paths.append(output_dir / "lstm_training_curve.png")
    _save_fig(fig, paths[-1])

    week = preds.iloc[: 7 * 24]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(week["timestamp"], week["actual"], label="actual", color="black")
    ax.plot(week["timestamp"], week["lstm"], label="LSTM", color="tab:red")
    ax.plot(week["timestamp"], week["persistence"], label="persistence", color="tab:gray", alpha=0.6)
    ax.set(ylabel="traffic volume (vehicles/hour)", title="First test week: next-hour prediction")
    ax.legend()
    fig.autofmt_xdate()
    paths.append(output_dir / "lstm_test_week.png")
    _save_fig(fig, paths[-1])

    for name, vals, sample in (("shap_summary_bar", values, x_sample), ("shap_beeswarm", values, x_sample)):
        shap.summary_plot(vals, sample, plot_type="bar" if name.endswith("bar") else "dot",
                          max_display=12, show=False)
        fig = plt.gcf()
        fig.set_size_inches(8, 5)
        paths.append(output_dir / f"{name}.png")
        _save_fig(fig, paths[-1])

    shap.summary_plot(values_nolag, x_sample_nolag, plot_type="bar", show=False)
    plt.title("SHAP without volume lags (calendar + weather only)")
    paths.append(output_dir / "shap_bar_no_lags.png")
    _save_fig(plt.gcf(), paths[-1])

    for label, row in local_cases.items():
        explanation = shap.Explanation(
            values=values[row],
            base_values=float(np.ravel(explainer.expected_value)[0]),
            data=x_sample.iloc[row].to_numpy(),
            feature_names=list(x_sample.columns),
        )
        shap.plots.waterfall(explanation, max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(8, 5)
        paths.append(output_dir / f"shap_local_{label}.png")
        _save_fig(fig, paths[-1])
    return paths


def pick_local_cases(x_sample: pd.DataFrame) -> dict[str, int]:
    """One weekday morning rush-hour hour and one night hour from the SHAP sample."""
    weekday = x_sample["is_weekend"] == 0
    cases = {}
    for label, hour in (("weekday_rush_hour", 7), ("night", 3)):
        match = np.flatnonzero((x_sample["hour"] == hour).to_numpy() & weekday.to_numpy())
        if len(match):
            cases[label] = int(match[0])
    return cases


def run_pipeline(data: pd.DataFrame, raw: pd.DataFrame, output_dir: Path):
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    data = data.sort_values("timestamp").reset_index(drop=True)
    cut = time_split(data)
    target_idx = window_targets(data["timestamp"])
    train_pos, val_pos, test_pos = split_windows({"target_idx": target_idx}, cut)
    train_end = int(target_idx[val_pos[0]])
    data, imputation_medians, imputation_counts = restore_and_impute_weather(data, raw, train_end)
    df, feats = build_features(data)
    win = build_windows(df, feats, train_end)
    n_runs = sum(1 for s, e in contiguous_runs(df["timestamp"]) if e - s > WINDOW)
    LOG.info(
        "Windows: %d total (%d train, %d validation, %d test) from %d usable runs; %d rows have no full 24h history.",
        len(win["y"]), len(train_pos), len(val_pos), len(test_pos), n_runs, len(df) - len(win["y"]),
    )

    y_test = win["y"][test_pos]
    lags = win["lag_table"]
    train_all = np.concatenate([train_pos, val_pos])  # the forest has no early stopping

    model, history = train_lstm(win, train_pos, val_pos)
    lstm_pred = lstm_predict(model, win, test_pos)

    forest = fit_forest(lags.iloc[train_all], win["y"][train_all])
    rf_pred = forest.predict(lags.iloc[test_pos])
    nolag_cols = [c for c in lags.columns if not c.startswith("volume_lag_")]
    forest_nolag = fit_forest(lags.iloc[train_all][nolag_cols], win["y"][train_all])
    nolag_pred = forest_nolag.predict(lags.iloc[test_pos][nolag_cols])

    persistence = lags["volume_lag_1"].to_numpy()[test_pos]
    yesterday = lags["volume_lag_24"].to_numpy()[test_pos]
    results = {
        "Baseline: persistence (previous hour)": persistence,
        "Baseline: same hour yesterday": yesterday,
        "Random Forest (lag features, SHAP twin)": rf_pred,
        "Random Forest (calendar + weather only)": nolag_pred,
        "LSTM": lstm_pred,
    }
    metrics = pd.DataFrame(
        [{"model": name, **regression_scores(y_test, pred)} for name, pred in results.items()]
    )
    LOG.info("Test metrics:\n%s", metrics.round(3).to_string(index=False))

    preds = pd.DataFrame({
        "timestamp": df["timestamp"].to_numpy()[win["target_idx"][test_pos]],
        "actual": y_test,
        "lstm": lstm_pred,
        "random_forest": rf_pred,
        "persistence": persistence,
    })

    rng = np.random.default_rng(RANDOM_STATE)
    sample_pos = rng.choice(len(test_pos), size=min(SHAP_SAMPLE, len(test_pos)), replace=False)
    sample_pos.sort()
    x_sample = lags.iloc[test_pos].iloc[sample_pos].reset_index(drop=True)
    explainer, values, importance = shap_table(forest, x_sample)
    _, values_nolag, importance_nolag = shap_table(forest_nolag, x_sample[nolag_cols])
    groups = group_importance(importance)
    LOG.info("SHAP top features:\n%s", importance.head(8).round(3).to_string(index=False))
    LOG.info("SHAP by family:\n%s", groups.round(3).to_string(index=False))
    LOG.info("SHAP without lags:\n%s", importance_nolag.head(6).round(3).to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    plots = make_plots(
        output_dir, history, preds, explainer, values, x_sample,
        x_sample[nolag_cols], values_nolag, pick_local_cases(x_sample),
    )

    info = {
        "window_hours": WINDOW,
        "n_windows": int(len(win["y"])),
        "n_train": int(len(train_pos)),
        "n_validation": int(len(val_pos)),
        "n_test": int(len(test_pos)),
        "n_forest_train": int(len(train_all)),
        "n_test_rows": int(len(df) - cut),
        "n_train_period_rows": int(cut),
        "train_end_exclusive": str(df["timestamp"].iloc[train_end]),
        "usable_runs": n_runs,
        "test_start": str(df["timestamp"].iloc[cut]),
        "epochs_run": len(history["loss"]),
        "lstm_params": {"units": LSTM_UNITS, "dropout": DROPOUT, "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE},
        "rf_trees": RF_TREES,
        "shap_sample": int(len(x_sample)),
        "shap_family_share": {r.group: round(float(r.share), 4) for r in groups.itertuples()},
        "imputed_counts": imputation_counts,
        "snow_lag_1_nonzero_train": int((lags["snow_1h_lag_1"].iloc[train_all] > 0).sum()),
        "snow_lag_1_nonzero_test": int((lags["snow_1h_lag_1"].iloc[test_pos] > 0).sum()),
    }
    preprocessing = {
        "window_hours": WINDOW,
        "sequence_features": list(SEQ_FEATURES),
        "calendar_features": list(CAL_FEATURES),
        "forest_features": lags.columns.tolist(),
        "scaler_mean": win["scaler"].mean_.tolist(),
        "scaler_scale": win["scaler"].scale_.tolist(),
        "imputation_monthly_medians": imputation_medians,
        "scaler_and_imputation_fit_end_exclusive": info["train_end_exclusive"],
        "prediction_policy": "inverse-scale then clip to zero",
    }
    return {
        "metrics": metrics, "importance": importance, "importance_nolag": importance_nolag,
        "groups": groups, "preds": preds, "info": info, "model": model, "plots": plots,
        "forest": forest, "forest_nolag": forest_nolag, "preprocessing": preprocessing,
    }


def save_outputs(output_dir: Path, res: dict) -> list[Path]:
    paths = {
        "lstm_metrics.csv": res["metrics"].round(4),
        "shap_importance.csv": res["importance"].round(5),
        "shap_importance_no_lags.csv": res["importance_nolag"].round(5),
        "shap_family_importance.csv": res["groups"].round(5),
        "lstm_test_predictions.csv": res["preds"].round({c: 2 for c in res["preds"].columns[1:]}),
    }
    saved = []
    for name, table in paths.items():
        table.to_csv(output_dir / name, index=False, encoding="utf-8")
        saved.append(output_dir / name)
    (output_dir / "lstm_run_info.json").write_text(json.dumps(res["info"], indent=2), encoding="utf-8")
    saved.append(output_dir / "lstm_run_info.json")
    res["model"].save(output_dir / "lstm_model.keras")
    saved.append(output_dir / "lstm_model.keras")
    (output_dir / "lstm_preprocessing.json").write_text(json.dumps(res["preprocessing"], indent=2), encoding="utf-8")
    saved.append(output_dir / "lstm_preprocessing.json")
    for name in ("forest", "forest_nolag"):
        path = output_dir / f"lstm_{name}.joblib"
        joblib.dump(res[name], path, compress=3)
        saved.append(path)
    LOG.info("Saved %d output files to %s.", len(saved) + len(res["plots"]), output_dir)
    return saved + res["plots"]


def log_to_mlflow(res: dict, artifacts: list[Path], tracking_uri, artifact_root, experiment) -> None:
    """One run per model (and baseline); the LSTM run also carries the files."""
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=artifact_root)
    mlflow.set_experiment(experiment)
    info = res["info"]
    common = {k: info[k] for k in ("window_hours", "n_test", "test_start")}
    common["split"] = "chronological 80/20 (same as Task 1), windows inside consecutive hourly runs"
    for row in res["metrics"].to_dict("records"):
        name = row["model"]
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"task": "next-hour regression", "model": name,
                             "is_baseline": str(name.startswith("Baseline"))})
            mlflow.log_params(common)
            train_count = info["n_forest_train"] if name.startswith("Random Forest") else (
                info["n_train"] if name == "LSTM" else 0
            )
            mlflow.log_param("n_train", train_count)
            mlflow.log_param("n_validation", info["n_validation"] if name == "LSTM" else 0)
            mlflow.log_metrics({k: float(v) for k, v in row.items() if k != "model"})
            if name == "LSTM":
                mlflow.log_params({f"lstm__{k}": v for k, v in info["lstm_params"].items()})
                for path in artifacts:
                    if path.suffix != ".joblib":
                        mlflow.log_artifact(str(path))
            elif name.startswith("Random Forest (lag"):
                mlflow.log_params({"rf__n_estimators": info["rf_trees"], "shap_sample": info["shap_sample"]})
                mlflow.log_artifact(str(next(p for p in artifacts if p.name == "lstm_forest.joblib")))
            elif name.startswith("Random Forest (calendar"):
                mlflow.log_artifact(str(next(p for p in artifacts if p.name == "lstm_forest_nolag.joblib")))
    LOG.info("Logged %d MLflow runs to %s (experiment '%s').", len(res["metrics"]), tracking_uri, experiment)


def _parse_args() -> argparse.Namespace:
    part_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=part_dir / "data" / "processed" / "Metro_Interstate_Traffic_Volume_ml.csv",
        help="Labelled CSV from prepare_labels.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=part_dir / "outputs", help="Output directory.")
    parser.add_argument(
        "--raw-input", type=Path, default=part_dir.parent / "Metro_Interstate_Traffic_Volume.csv",
        help="Matching original CSV, used to fit weather imputation on training data only.",
    )
    parser.add_argument(
        "--mlflow-uri", default=f"sqlite:///{(part_dir / 'mlflow.db').as_posix()}",
        help="MLflow tracking URI (default: SQLite database Part-3/mlflow.db).",
    )
    parser.add_argument(
        "--mlflow-artifact-root", default=(part_dir / "mlartifacts").as_uri(),
        help="Artifact location used when creating the experiment (default: Part-3/mlartifacts).",
    )
    parser.add_argument("--experiment", default="task3-deep-learning", help="MLflow experiment name.")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    parser.add_argument("--log-file", type=Path, default=part_dir / "logs" / "lstm_model.log", help="Log file path.")
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO",
        help="Minimum level written to the console and log file (default: INFO).",
    )
    return parser.parse_args()


def _configure_logging(log_path: Path, level: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stderr)],
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
        LOG.info("Reading raw weather from %s.", args.raw_input)
        raw = pd.read_csv(args.raw_input, encoding="utf-8-sig")
        res = run_pipeline(data, raw, args.output_dir)
        artifacts = save_outputs(args.output_dir, res)
        if not args.no_mlflow:
            log_to_mlflow(res, artifacts, args.mlflow_uri, args.mlflow_artifact_root, args.experiment)
    except (OSError, UnicodeError, MlflowException, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        LOG.exception("LSTM pipeline failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
