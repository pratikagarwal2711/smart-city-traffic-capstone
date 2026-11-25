"""Task 6.1 and 6.2: model versioning and experiment tracking with MLflow.

The registered model ``traffic-volume-regressor`` (same-hour traffic volume, Task 1
features and 80/20 time split) holds three versions:

* v1: the Task 1 Random Forest registered by ``mlflow_registry.py`` (Task 4);
* v2: a Gradient Boosting regressor;
* v3: a Random Forest whose ``min_samples_leaf`` and ``max_features`` are tuned by
  time-ordered cross-validation on the training period only.

v2 and v3 are trained here and logged as runs in the experiment
``task6-model-versions`` (parameters, metrics, model). Every version is then reloaded
from the registry and scored on the test period, and a version is compared with the
metric logged by its training run (the Task 4 promotion gate). The ``champion`` alias
moves only to a version that beats v1 on test MAE *and* passes the gate; otherwise v1
stays champion. Test MAE is used for that choice, as in Task 4, so the winning score
is slightly optimistic.

Re-running does not create duplicate versions: a version that is already registered
under the same ``version_label`` tag is reused, not retrained.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

import mlops_common as common
import train_models as tm
from mlflow_registry import GATE_TOLERANCE

LOG = logging.getLogger(__name__)

VERSIONS_EXPERIMENT = "task6-model-versions"
TASK1_EXPERIMENT = "task1-supervised-models"
LABEL_TAG = "version_label"
GRADIENT_BOOSTING = "gradient-boosting"
TUNED_FOREST = "tuned-random-forest"
# Hyperparameters below are choices made for this simulation, not values from the brief.
GB_PARAMS = {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 5, "subsample": 0.8}
RF_FIXED = {"n_estimators": 300}
RF_GRID = {"min_samples_leaf": [2, 5, 10, 20], "max_features": [0.33, 0.5, 0.75, 1.0]}
CV_SPLITS = 3


def choose_champion(table: pd.DataFrame, baseline_version: str) -> tuple[str | None, str]:
    """Pick the champion version: the best gate-passing version that beats the baseline.

    ``table`` needs the columns ``version``, ``MAE`` and ``gate_passed``. The baseline
    (v1) keeps the alias when nothing beats it; ``None`` means no alias can be granted.
    """
    base = table[table["version"] == baseline_version].iloc[0]
    better = table[table["gate_passed"] & (table["MAE"] < base["MAE"])].sort_values("MAE")
    if not better.empty:
        best = better.iloc[0]
        return best["version"], f"v{best['version']} beats v{baseline_version} (MAE {best['MAE']:.2f} vs {base['MAE']:.2f}) and passed the gate"
    if base["gate_passed"]:
        return baseline_version, f"no gate-passing version beats v{baseline_version} (MAE {base['MAE']:.2f}); it stays champion"
    return None, f"v{baseline_version} failed the gate and no version beats it; alias unchanged"


def find_baseline_version(client: mlflow.MlflowClient) -> str:
    """v1: the earliest version registered from a Task 1 run (see mlflow_registry.py)."""
    versions = client.search_model_versions(f"name='{common.REGISTERED_MODEL}'")
    task1 = [v for v in versions if v.tags.get("source_experiment") == TASK1_EXPERIMENT]
    if not task1:
        raise ValueError(f"'{common.REGISTERED_MODEL}' has no Task 1 version; run mlflow_registry.py first.")
    return str(min(int(v.version) for v in task1))


def find_labelled(client: mlflow.MlflowClient, label: str):
    versions = client.search_model_versions(f"name='{common.REGISTERED_MODEL}'")
    return next((v for v in versions if v.tags.get(LABEL_TAG) == label), None)


def tune_forest(X_train, y_train) -> tuple[RandomForestRegressor, pd.DataFrame]:
    """Grid search on the training period only, with time-ordered folds (no test data)."""
    search = GridSearchCV(
        RandomForestRegressor(**RF_FIXED, n_jobs=-1, random_state=tm.RANDOM_STATE),
        RF_GRID,
        cv=TimeSeriesSplit(n_splits=CV_SPLITS),
        scoring="neg_mean_absolute_error",
    )
    search.fit(X_train, y_train)
    cv = pd.DataFrame(search.cv_results_)
    cv["cv_MAE"] = -cv["mean_test_score"]
    cv = cv[["param_min_samples_leaf", "param_max_features", "cv_MAE", "rank_test_score"]].rename(
        columns=lambda c: c.replace("param_", "")
    )
    LOG.info("Tuned forest best params %s (CV MAE %.2f).", search.best_params_, -search.best_score_)
    return search.best_estimator_, cv.sort_values("rank_test_score")


def train_and_register(label, build, data, output_dir: Path, family: str):
    """Fit one candidate, log it to MLflow and register it as a new model version."""
    X_train, X_test, y_train, y_test, info = data
    with mlflow.start_run(run_name=f"{family} ({label})") as run:
        mlflow.set_tags({"task": "model versioning", "model": family, LABEL_TAG: label})
        mlflow.log_params({
            "split": f"time-based {1 - tm.TEST_FRACTION:.0%}/{tm.TEST_FRACTION:.0%}",
            "random_state": tm.RANDOM_STATE,
            **{k: info[k] for k in ("train_end", "test_start", "n_train", "n_test", "n_features")},
        })
        started = time.perf_counter()
        model, extra = build(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        mlflow.log_params({f"model__{k}": v for k, v in model.get_params().items()})
        pred = model.predict(X_test)
        scores = common.regression_scores(y_test, pred)
        mlflow.log_metrics({**scores, "train_MAE": float(common.regression_scores(y_train, model.predict(X_train))["MAE"]),
                            "fit_seconds": fit_seconds})
        if extra is not None:
            path = output_dir / "model_versions_rf_tuning.csv"
            extra.round(4).to_csv(path, index=False, encoding="utf-8")
            mlflow.log_artifact(str(path))
            LOG.info("Saved forest tuning results to %s.", path)
        # The model is produced by this script, so pickle is acceptable here.
        logged = mlflow.sklearn.log_model(
            model, name="model", serialization_format="cloudpickle",
            signature=infer_signature(X_test.head(5), pred[:5]), input_example=X_test.head(5),
        )
        version = mlflow.register_model(
            logged.model_uri, common.REGISTERED_MODEL,
            tags={"source_run_id": run.info.run_id, "source_experiment": VERSIONS_EXPERIMENT,
                  "source_run_name": run.info.run_name, LABEL_TAG: label},
        )
    LOG.info("Trained %s: test MAE %.2f; registered as version %s.", family, scores["MAE"], version.version)
    return version


def evaluate_versions(client, data, baseline_version: str) -> pd.DataFrame:
    """Reload every registered version and score it; compare with the metric its run logged."""
    X_train, X_test, y_train, y_test, _ = data
    rows = []
    for v in sorted(client.search_model_versions(f"name='{common.REGISTERED_MODEL}'"), key=lambda v: int(v.version)):
        model = mlflow.sklearn.load_model(f"models:/{common.REGISTERED_MODEL}/{v.version}")
        test = common.regression_scores(y_test, model.predict(X_test))
        train_mae = common.regression_scores(y_train, model.predict(X_train))["MAE"]
        source = client.get_run(v.tags["source_run_id"])
        logged = source.data.metrics.get("MAE")
        diff = abs(test["MAE"] - logged) if logged is not None else float("nan")
        params = model.get_params()
        rows.append({
            "version": str(v.version),
            "label": v.tags.get(LABEL_TAG, "random-forest-task1" if str(v.version) == baseline_version else ""),
            "model": type(model).__name__,
            "key_params": json.dumps({k: params[k] for k in params if k in (
                "n_estimators", "learning_rate", "max_depth", "subsample", "min_samples_leaf", "max_features")}),
            "train_MAE": train_mae,
            "MAE": test["MAE"], "RMSE": test["RMSE"], "R2": test["R2"],
            "logged_MAE": logged, "abs_difference": diff,
            "gate_passed": bool(diff <= GATE_TOLERANCE["MAE"]),
            "source_experiment": v.tags.get("source_experiment", TASK1_EXPERIMENT),
            "source_run_id": v.tags["source_run_id"],
        })
        LOG.info("Version %s (%s): test MAE %.2f, RMSE %.2f, R2 %.4f; gate %s.", v.version, rows[-1]["label"],
                 test["MAE"], test["RMSE"], test["R2"], "passed" if rows[-1]["gate_passed"] else "FAILED")
    table = pd.DataFrame(rows)
    table["mae_vs_v1_pct"] = 100 * (table["MAE"] / table.loc[table["version"] == baseline_version, "MAE"].item() - 1)
    return table


def apply_champion(client, table: pd.DataFrame, baseline_version: str) -> pd.DataFrame:
    """Set the champion alias by the promotion policy and record metrics on each version."""
    winner, reason = choose_champion(table, baseline_version)
    previous = None
    try:
        previous = str(client.get_model_version_by_alias(common.REGISTERED_MODEL, common.CHAMPION_ALIAS).version)
    except MlflowException:
        LOG.debug("No champion alias exists yet.")
    if winner is None:
        LOG.warning("Champion alias not changed: %s.", reason)
    else:
        client.set_registered_model_alias(common.REGISTERED_MODEL, common.CHAMPION_ALIAS, winner)
        LOG.info("Champion is version %s (was %s): %s.", winner, previous, reason)
        if previous != winner:
            LOG.warning("Champion alias moved from version %s to version %s.", previous, winner)
    for row in table.to_dict("records"):
        for key, value in (("test_MAE", row["MAE"]), ("test_RMSE", row["RMSE"]), ("test_R2", row["R2"])):
            client.set_model_version_tag(common.REGISTERED_MODEL, row["version"], key, f"{value:.4f}")
        client.set_model_version_tag(common.REGISTERED_MODEL, row["version"], "gate_passed", str(row["gate_passed"]))
    client.set_registered_model_tag(common.REGISTERED_MODEL, "task", "regression")
    table = table.copy()
    table["alias"] = ["champion" if row == winner else "" for row in table["version"]]
    table["promotion_note"] = reason
    return table


def plot_versions(table: pd.DataFrame, path: Path) -> None:
    labels = [f"v{v}\n{lab}" for v, lab in zip(table["version"], table["label"])]
    colors = ["#2e7d32" if a else "#78909c" for a in table["alias"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, metric, title in zip(axes, ("MAE", "RMSE", "R2"), ("Test MAE (vehicles/h)", "Test RMSE (vehicles/h)", "Test R2")):
        bars = ax.bar(labels, table[metric], color=colors)
        ax.bar_label(bars, fmt="%.3f" if metric == "R2" else "%.1f", fontsize=8)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("traffic-volume-regressor versions (green = champion)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    LOG.info("Saved version comparison chart to %s.", path)


def run_versions(args) -> pd.DataFrame:
    mlflow.set_tracking_uri(args.mlflow_uri)
    common.ensure_experiment(args.experiment, args.mlflow_artifact_root)
    client = mlflow.MlflowClient()
    baseline = find_baseline_version(client)

    df, feats, cut = common.load_task1_frame(args.input)
    data = (
        feats.iloc[:cut], feats.iloc[cut:], df[tm.REG_TARGET].iloc[:cut], df[tm.REG_TARGET].iloc[cut:],
        {"train_end": str(df["timestamp"].iloc[cut - 1]), "test_start": str(df["timestamp"].iloc[cut]),
         "n_train": cut, "n_test": len(df) - cut, "n_features": feats.shape[1]},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def build_gb(X, y):
        return GradientBoostingRegressor(**GB_PARAMS, random_state=tm.RANDOM_STATE).fit(X, y), None

    candidates = ((GRADIENT_BOOSTING, build_gb, "Gradient Boosting"), (TUNED_FOREST, tune_forest, "Random Forest (tuned)"))
    for label, build, family in candidates:
        if find_labelled(client, label) is not None:
            LOG.info("Version '%s' is already registered; not retrained.", label)
            continue
        train_and_register(label, build, data, args.output_dir, family)

    table = apply_champion(client, evaluate_versions(client, data, baseline), baseline)
    csv_path, png_path = args.output_dir / "model_versions.csv", args.output_dir / "model_versions.png"
    table.round(4).to_csv(csv_path, index=False, encoding="utf-8")
    LOG.info("Saved version comparison to %s.", csv_path)
    plot_versions(table, png_path)

    with mlflow.start_run(run_name="version comparison"):
        mlflow.set_tags({"task": "model versioning summary"})
        mlflow.log_params({"gate_tolerance_MAE": GATE_TOLERANCE["MAE"], "baseline_version": baseline,
                           "gb_params": json.dumps(GB_PARAMS), "rf_grid": json.dumps(RF_GRID), "cv_splits": CV_SPLITS})
        champion = table.loc[table["alias"] == "champion", "version"]
        mlflow.log_metrics({"versions_compared": len(table), "gates_passed": int(table["gate_passed"].sum()),
                            "champion_version": float(champion.iloc[0]) if len(champion) else -1.0,
                            "best_test_MAE": float(table["MAE"].min())})
        for path in (csv_path, png_path):
            mlflow.log_artifact(str(path))
    LOG.info("Version comparison:\n%s", table[["version", "label", "model", "train_MAE", "MAE", "RMSE", "R2",
                                               "gate_passed", "alias"]].round(3).to_string(index=False))
    return table


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=common.DEFAULT_INPUT, help="Labelled CSV from prepare_labels.py.")
    parser.add_argument("--output-dir", type=Path, default=common.DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--mlflow-uri", default=common.DEFAULT_MLFLOW_URI,
                        help="MLflow tracking URI (default: SQLite database Part-3/mlflow.db).")
    parser.add_argument("--mlflow-artifact-root", default=common.DEFAULT_ARTIFACT_ROOT,
                        help="Artifact location used when creating the experiment (default: Part-3/mlartifacts).")
    parser.add_argument("--experiment", default=VERSIONS_EXPERIMENT, help="MLflow experiment name.")
    parser.add_argument("--log-file", type=Path, default=common.PART_DIR / "logs" / "model_versions.log", help="Log file path.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO",
                        help="Minimum level written to the console and log file (default: INFO).")
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
        table = run_versions(args)
    except (OSError, UnicodeError, MlflowException, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        LOG.exception("Model versioning failed.")
        return 1
    return 0 if table["gate_passed"].all() else 2


if __name__ == "__main__":
    raise SystemExit(main())
