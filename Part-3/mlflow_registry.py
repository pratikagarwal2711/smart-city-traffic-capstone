"""Advanced technique for Part 3, Task 4: MLflow experiment audit and model registry.

Tasks 1-3 log every model to MLflow. This script uses that history to:

* collect all runs into one comparison table and measure rerun-to-rerun variation;
* pick the best model per problem from the latest runs (lowest MAE for regression,
  highest PR AUC for the proxy ``high_risk`` classifier);
* register the selected models in the MLflow Model Registry (Task 3 models are made
  registrable here: the forest as a scikit-learn model, the LSTM as a pyfunc model
  that bundles its preprocessing);
* apply a promotion gate: a registered version only gets the ``champion`` (or
  ``challenger``) alias if loading it from the registry reproduces the test metric
  logged by the training run.

Test metrics are used for selection, as in Tasks 1-3; there is no separate
selection set, so the winning scores are slightly optimistic. The classifier
inherits the proxy-label caveats from Task 1.
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.artifacts
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature, infer_signature
from mlflow.types import Schema, TensorSpec
from sklearn.metrics import average_precision_score, mean_absolute_error

import lstm_model as lm
import train_models as tm

LOG = logging.getLogger(__name__)

REGISTRY_EXPERIMENT = "task4-mlflow-registry"
TASK1_EXPERIMENT = "task1-supervised-models"
TASK3_EXPERIMENT = "task3-deep-learning"
LSTM_RUN = "LSTM"
FOREST_RUN = "Random Forest (lag features, SHAP twin)"
# Tolerances for the promotion gate (MAE in vehicles per hour; PR AUC is unitless).
GATE_TOLERANCE = {"MAE": 0.05, "PR_AUC": 1e-4}
# Reruns of the same model whose primary metric differs by more than this raise a warning.
RERUN_WARNING_SHARE = 0.01
RESULT_METRICS = ("MAE", "RMSE", "R2", "PR_AUC", "F1", "ROC_AUC")


def collect_runs(client: mlflow.MlflowClient) -> pd.DataFrame:
    """One row per run in the Task 1-3 experiments, newest first."""
    rows = []
    for exp in client.search_experiments():
        if exp.name in ("Default", REGISTRY_EXPERIMENT):
            continue
        for run in client.search_runs([exp.experiment_id], max_results=5000):
            tags, data = run.data.tags, run.data
            rows.append({
                "experiment": exp.name,
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "task": tags.get("task", ""),
                "model": tags.get("model", run.info.run_name),
                "is_baseline": tags.get("is_baseline") == "True",
                "status": run.info.status,
                "start_time": pd.to_datetime(run.info.start_time, unit="ms", utc=True),
                "git_commit": tags.get("mlflow.source.git.commit", "")[:8],
                **data.metrics,
            })
    if not rows:
        raise ValueError("No MLflow runs found; run the Task 1-3 scripts first.")
    return pd.DataFrame(rows).sort_values("start_time", ascending=False).reset_index(drop=True)


def latest_runs(runs: pd.DataFrame) -> pd.DataFrame:
    return runs.drop_duplicates(["experiment", "run_name"], keep="first").reset_index(drop=True)


def rerun_variability(runs: pd.DataFrame) -> pd.DataFrame:
    """Spread of the primary metric across repeated runs of the same experiment and model."""
    rows = []
    for (experiment, run_name), group in runs.groupby(["experiment", "run_name"]):
        if len(group) < 2:
            continue
        metric = "MAE" if group["MAE"].notna().any() else "PR_AUC"
        values = group[metric].dropna()
        spread = float(values.max() - values.min())
        share = spread / abs(float(values.mean())) if values.mean() else 0.0
        rows.append({
            "experiment": experiment, "run_name": run_name, "n_runs": len(group),
            "n_git_commits": group["git_commit"].nunique(), "metric": metric,
            "min": values.min(), "max": values.max(), "spread": spread, "spread_share": share,
        })
        if share > RERUN_WARNING_SHARE:
            LOG.warning(
                "%s / %s: %s differs by %.1f%% across %d runs (%.3f to %.3f); a single run is not a stable estimate.",
                experiment, run_name, metric, 100 * share, len(group), values.min(), values.max(),
            )
    return pd.DataFrame(rows)


def ranked(latest: pd.DataFrame, experiment: str, metric: str, higher_is_better=False,
           task=None, models=None) -> pd.DataFrame:
    """Non-baseline latest runs of an experiment, best first."""
    sel = latest[(latest["experiment"] == experiment) & ~latest["is_baseline"]]
    if task:
        sel = sel[sel["task"] == task]
    if models:
        sel = sel[sel["model"].isin(models)]
    sel = sel.dropna(subset=[metric]).sort_values(metric, ascending=not higher_is_better)
    if sel.empty:
        raise ValueError(f"No candidate runs for {experiment} ({task or 'all tasks'}).")
    return sel


def find_version(client: mlflow.MlflowClient, name: str, source_run_id: str):
    """Existing registered version created from this training run, so reruns do not duplicate."""
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except MlflowException:
        return None
    return next((v for v in versions if v.tags.get("source_run_id") == source_run_id), None)


def build_task1_eval(data: pd.DataFrame):
    df, feats = tm.build_features(data)
    cut = tm.time_split(df)
    return feats.iloc[cut:], df[tm.REG_TARGET].iloc[cut:], df[tm.CLF_TARGET].iloc[cut:]


def build_task3_eval(data: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """Rebuild the Task 3 test windows exactly as lstm_model.run_pipeline does."""
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    data = data.sort_values("timestamp").reset_index(drop=True)
    cut = tm.time_split(data)
    target_idx = lm.window_targets(data["timestamp"])
    _, val_pos, test_pos = lm.split_windows({"target_idx": target_idx}, cut)
    train_end = int(target_idx[val_pos[0]])
    data, _, _ = lm.restore_and_impute_weather(data, raw, train_end)
    df, feats = tm.build_features(data)
    win = lm.build_windows(df, feats, train_end)
    scaler = win["scaler"]
    return {
        "y": win["y"][test_pos],
        "lags": win["lag_table"].iloc[test_pos].astype("float64").reset_index(drop=True),
        # The pyfunc model expects raw (unscaled) inputs and scales them itself.
        "sequence": (win["x_seq"][test_pos].astype("float64") * scaler.scale_ + scaler.mean_),
        "calendar": win["x_cal"][test_pos].astype("float64"),
    }


def register_version(client, name, model_uri, source_run, tags):
    version = mlflow.register_model(model_uri, name, tags={"source_run_id": source_run.run_id, **tags})
    LOG.info("Registered %s version %s from run %s.", name, version.version, source_run.run_id)
    return version


def common_tags(source_run, metric: str, rule: str) -> dict:
    return {
        "source_experiment": source_run.experiment,
        "source_run_name": source_run.run_name,
        "source_git_commit": source_run.git_commit,
        "selection_metric": metric,
        "selection_value": f"{source_run[metric]:.4f}",
        "selection_rule": rule,
    }


def register_task1(client, latest) -> list[dict]:
    """Register the best Task 1 regressor and classifier; return model specs for the gate."""
    specs = []
    plans = (
        ("traffic-volume-regressor", "regression", "MAE", False, "lowest test MAE among latest non-baseline runs"),
        ("high-risk-classifier", "classification", "PR_AUC", True,
         "highest test PR AUC among latest non-baseline runs (proxy label, inflated by design)"),
    )
    for name, task, metric, higher, rule in plans:
        best = ranked(latest, TASK1_EXPERIMENT, metric, higher, task=task).iloc[0]
        version = find_version(client, name, best["run_id"]) or register_version(
            client, name, f"runs:/{best['run_id']}/model", best, common_tags(best, metric, rule)
        )
        specs.append({"name": name, "version": version.version, "source": best, "metric": metric,
                      "rule": rule, "alias": "champion", "task": task})
    return specs


def log_task3_models(client, latest, task3_eval, artifact_dir: Path) -> list[dict]:
    """Make the Task 3 LSTM and forest registrable models and register them."""
    candidates = ranked(latest, TASK3_EXPERIMENT, "MAE", models=(LSTM_RUN, FOREST_RUN))
    rule = "lowest test MAE among latest Task 3 forecasting models; the other model is the challenger"
    plans = {LSTM_RUN: "next-hour-forecaster-lstm", FOREST_RUN: "next-hour-forecaster-rf"}
    specs = []
    for rank, (_, best) in enumerate(candidates.iterrows()):
        name = plans[best["model"]]
        version = find_version(client, name, best["run_id"])
        if version is None:
            version = log_and_register_task3(client, best, name, task3_eval, artifact_dir, rule)
        specs.append({"name": name, "version": version.version, "source": best, "metric": "MAE", "rule": rule,
                      "alias": "champion" if rank == 0 else "challenger", "task": "next-hour regression"})
    return specs


def log_and_register_task3(client, source, name, task3_eval, artifact_dir: Path, rule: str):
    tags = common_tags(source, "MAE", rule)
    run_id = source["run_id"]
    if source["model"] == FOREST_RUN:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="lstm_forest.joblib", dst_path=str(artifact_dir))
        forest = joblib.load(path)
        sample = task3_eval["lags"].head(5)
        info = mlflow.sklearn.log_model(
            forest, name="model", serialization_format="cloudpickle",
            signature=infer_signature(sample, forest.predict(sample)), input_example=sample,
        )
    else:
        files = {
            key: mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=fname, dst_path=str(artifact_dir))
            for key, fname in (("model", "lstm_model.keras"), ("preprocessing", "lstm_preprocessing.json"))
        }
        seq_shape, cal_shape = task3_eval["sequence"].shape[1:], task3_eval["calendar"].shape[1:]
        signature = ModelSignature(
            inputs=Schema([
                TensorSpec(np.dtype("float64"), (-1, *seq_shape), "sequence"),
                TensorSpec(np.dtype("float64"), (-1, *cal_shape), "calendar"),
            ]),
            outputs=Schema([TensorSpec(np.dtype("float64"), (-1,))]),
        )
        info = mlflow.pyfunc.log_model(
            name="model", python_model=lm.LstmForecaster(), artifacts=files, signature=signature,
            input_example={"sequence": task3_eval["sequence"][:2], "calendar": task3_eval["calendar"][:2]},
            code_paths=[str(Path(lm.__file__)), str(Path(tm.__file__))],
        )
    return register_version(client, name, info.model_uri, source, tags)


def recompute_metric(spec: dict, task1_eval, task3_eval) -> float:
    """Load the registered version (as a serving layer would) and recompute its test metric."""
    uri = f"models:/{spec['name']}/{spec['version']}"
    name = spec["name"]
    if name == "traffic-volume-regressor":
        X_test, y_reg, _ = task1_eval
        return float(mean_absolute_error(y_reg, mlflow.sklearn.load_model(uri).predict(X_test)))
    if name == "high-risk-classifier":
        X_test, _, y_clf = task1_eval
        return float(average_precision_score(y_clf, mlflow.sklearn.load_model(uri).predict_proba(X_test)[:, 1]))
    model = mlflow.pyfunc.load_model(uri)
    if name.endswith("lstm"):
        pred = model.predict({"sequence": task3_eval["sequence"], "calendar": task3_eval["calendar"]})
    else:
        pred = model.predict(task3_eval["lags"])
    return float(mean_absolute_error(task3_eval["y"], pred))


def promote(client, specs, task1_eval, task3_eval) -> pd.DataFrame:
    """Set the alias only when the registered model reproduces the logged metric."""
    rows = []
    for spec in specs:
        logged = float(spec["source"][spec["metric"]])
        recomputed = recompute_metric(spec, task1_eval, task3_eval)
        diff = abs(recomputed - logged)
        passed = diff <= GATE_TOLERANCE[spec["metric"]]
        if passed:
            client.set_registered_model_alias(spec["name"], spec["alias"], spec["version"])
            LOG.info("%s v%s passed the gate (%s %.4f vs logged %.4f); alias '%s' set.",
                     spec["name"], spec["version"], spec["metric"], recomputed, logged, spec["alias"])
        else:
            LOG.warning("%s v%s FAILED the gate: %s %.4f vs logged %.4f; no alias set.",
                        spec["name"], spec["version"], spec["metric"], recomputed, logged)
        client.set_registered_model_tag(spec["name"], "task", spec["task"])
        rows.append({
            "registered_model": spec["name"], "version": spec["version"],
            "alias": spec["alias"] if passed else "", "source_run_id": spec["source"]["run_id"],
            "metric": spec["metric"], "logged_value": logged, "registry_value": recomputed,
            "abs_difference": diff, "gate_passed": passed, "selection_rule": spec["rule"],
        })
    return pd.DataFrame(rows)


def plot_comparison(runs: pd.DataFrame, path: Path) -> None:
    """Bar = latest run; dots = every run, so rerun variation is visible."""
    panels = (("task1-supervised-models", "Task 1: same-hour volume (MAE)", "regression"),
              (TASK3_EXPERIMENT, "Task 3: next-hour volume (MAE)", None))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (experiment, title, task) in zip(axes, panels):
        sub = runs[(runs["experiment"] == experiment) & runs["MAE"].notna()]
        if task:
            sub = sub[sub["task"] == task]
        order = sub.drop_duplicates("run_name")["run_name"].tolist()[::-1]
        for i, name in enumerate(order):
            values = sub.loc[sub["run_name"] == name, "MAE"]
            ax.barh(i, values.iloc[0], color="#4c78a8", alpha=0.7)
            ax.scatter(values, [i] * len(values), color="black", s=14, zorder=3)
        ax.set_yticks(range(len(order)), [o.replace("regression - ", "") for o in order], fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("MAE (vehicles per hour); dots = individual runs")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOG.info("Saved run comparison chart to %s.", path)


def save_table(table: pd.DataFrame, path: Path, what: str) -> Path:
    table.to_csv(path, index=False, encoding="utf-8")
    LOG.info("Saved %s to %s.", what, path)
    return path


def run_registry(args) -> pd.DataFrame:
    mlflow.set_tracking_uri(args.mlflow_uri)
    client = mlflow.MlflowClient()
    if mlflow.get_experiment_by_name(args.experiment) is None:
        mlflow.create_experiment(args.experiment, artifact_location=args.mlflow_artifact_root)
    mlflow.set_experiment(args.experiment)

    runs = collect_runs(client)
    latest = latest_runs(runs)
    LOG.info("Collected %d runs from %d experiments (%d distinct models).",
             len(runs), runs["experiment"].nunique(), len(latest))
    variability = rerun_variability(runs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        save_table(runs.drop(columns=["start_time"]).assign(start_time=runs["start_time"].dt.strftime("%Y-%m-%d %H:%M")),
                   args.output_dir / "mlflow_run_comparison.csv", "run comparison"),
        save_table(variability.round(5), args.output_dir / "mlflow_rerun_variability.csv", "rerun variability"),
    ]

    LOG.info("Reading %s.", args.input)
    data = pd.read_csv(args.input, encoding="utf-8-sig")
    raw = pd.read_csv(args.raw_input, encoding="utf-8-sig")
    task1_eval = build_task1_eval(data)
    task3_eval = build_task3_eval(data, raw)

    with mlflow.start_run(run_name="registry update"), tempfile.TemporaryDirectory() as tmp:
        mlflow.set_tags({"task": "model registry and promotion gate"})
        specs = register_task1(client, latest) + log_task3_models(client, latest, task3_eval, Path(tmp))
        summary = promote(client, specs, task1_eval, task3_eval)
        outputs.append(save_table(summary.round(5), args.output_dir / "mlflow_registry_summary.csv", "registry summary"))
        chart = args.output_dir / "mlflow_run_comparison.png"
        plot_comparison(runs, chart)
        outputs.append(chart)
        mlflow.log_params({f"gate_tolerance_{k}": v for k, v in GATE_TOLERANCE.items()})
        mlflow.log_metrics({
            "runs_audited": len(runs), "models_registered": len(summary),
            "gates_passed": int(summary["gate_passed"].sum()),
            "reruns_over_threshold": int((variability["spread_share"] > RERUN_WARNING_SHARE).sum()),
        })
        for path in outputs:
            mlflow.log_artifact(str(path))
    LOG.info("Registry summary:\n%s", summary[["registered_model", "version", "alias", "metric",
                                               "logged_value", "registry_value", "gate_passed"]].round(4).to_string(index=False))
    return summary


def _parse_args() -> argparse.Namespace:
    part_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=part_dir / "data" / "processed" / "Metro_Interstate_Traffic_Volume_ml.csv",
                        help="Labelled CSV from prepare_labels.py (used to re-test registered models).")
    parser.add_argument("--raw-input", type=Path, default=part_dir.parent / "Metro_Interstate_Traffic_Volume.csv",
                        help="Original CSV, needed to rebuild the Task 3 test windows.")
    parser.add_argument("--output-dir", type=Path, default=part_dir / "outputs", help="Output directory.")
    parser.add_argument("--mlflow-uri", default=f"sqlite:///{(part_dir / 'mlflow.db').as_posix()}",
                        help="MLflow tracking URI (default: SQLite database Part-3/mlflow.db).")
    parser.add_argument("--mlflow-artifact-root", default=(part_dir / "mlartifacts").as_uri(),
                        help="Artifact location used when creating the experiment (default: Part-3/mlartifacts).")
    parser.add_argument("--experiment", default=REGISTRY_EXPERIMENT, help="MLflow experiment for this script's run.")
    parser.add_argument("--log-file", type=Path, default=part_dir / "logs" / "mlflow_registry.log", help="Log file path.")
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
        summary = run_registry(args)
    except (OSError, UnicodeError, MlflowException, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        LOG.exception("MLflow registry update failed.")
        return 1
    return 0 if summary["gate_passed"].all() else 2


if __name__ == "__main__":
    raise SystemExit(main())
