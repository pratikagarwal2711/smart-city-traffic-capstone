"""Helpers shared by the Task 6 scripts (model versions, serving API, monitoring).

Everything here reuses the Task 1 feature builder and time split, so versions, the
API and the monitor all work on the same 28 features and the same train/test cut.
"""

import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import train_models as tm

LOG = logging.getLogger(__name__)

PART_DIR = Path(__file__).resolve().parent
REGISTERED_MODEL = "traffic-volume-regressor"
CHAMPION_ALIAS = "champion"
DEFAULT_INPUT = PART_DIR / "data" / "processed" / "Metro_Interstate_Traffic_Volume_ml.csv"
DEFAULT_OUTPUT_DIR = PART_DIR / "outputs"
DEFAULT_MLFLOW_URI = f"sqlite:///{(PART_DIR / 'mlflow.db').as_posix()}"
DEFAULT_ARTIFACT_ROOT = (PART_DIR / "mlartifacts").as_uri()


def load_task1_frame(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return (time-sorted data, Task 1 feature matrix, first test row index)."""
    LOG.info("Reading %s.", path)
    data = pd.read_csv(path, encoding="utf-8-sig")
    df, feats = tm.build_features(data)
    return df, feats, tm.time_split(df)


def regression_scores(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def ensure_experiment(name: str, artifact_root: str) -> None:
    """Create the experiment with an explicit artifact location, then select it."""
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=artifact_root)
    mlflow.set_experiment(name)


def load_registered_model(name: str = REGISTERED_MODEL, alias: str = CHAMPION_ALIAS):
    """Load a registered model by alias; return (model, version string).

    Registered models are cloudpickle files, so only load them from a trusted store.
    """
    client = mlflow.MlflowClient()
    try:
        version = client.get_model_version_by_alias(name, alias).version
    except MlflowException as exc:
        raise ValueError(f"No version of '{name}' has alias '{alias}'; run mlflow_registry.py first.") from exc
    model = mlflow.sklearn.load_model(f"models:/{name}/{version}")
    LOG.info("Loaded %s version %s (alias '%s').", name, version, alias)
    return model, str(version)
