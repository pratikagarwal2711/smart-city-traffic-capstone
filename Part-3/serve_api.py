"""Task 6.3: FastAPI deployment mock-up serving the registered traffic-volume model.

It loads ``models:/traffic-volume-regressor@champion`` from the MLflow registry at
start-up and predicts same-hour ``traffic_volume`` (vehicles/hour) from the calendar
and weather of that hour, using the same 28 features as Task 1.

Run from the repository root::

    python Part-3/serve_api.py            # then open http://127.0.0.1:8000/docs

The registered model is a pickle file: only serve models from a store you trust.
This is a local simulation: no authentication, TLS, rate limiting or scaling.
"""

import argparse
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, Field

import mlops_common as common
import train_models as tm

LOG = logging.getLogger(__name__)

DEFAULT_STATUS_FILE = common.DEFAULT_OUTPUT_DIR / "monitoring_status.json"
UNIT = "vehicles/hour"
CATEGORY_NOTE = (
    "Part 2 traffic_category: Low < 4,500; Medium 4,500-5,500; High > 5,500 vehicles/hour "
    "(the project's fixed thresholds, not the Part 3 quartile categories)."
)


@dataclass
class Settings:
    mlflow_uri: str = os.environ.get("TRAFFIC_MLFLOW_URI", common.DEFAULT_MLFLOW_URI)
    model_name: str = os.environ.get("TRAFFIC_MODEL_NAME", common.REGISTERED_MODEL)
    alias: str = os.environ.get("TRAFFIC_MODEL_ALIAS", common.CHAMPION_ALIAS)
    status_file: Path = Path(os.environ.get("TRAFFIC_STATUS_FILE", DEFAULT_STATUS_FILE))


@dataclass
class ModelBundle:
    model: object
    name: str
    version: str
    alias: str

    @property
    def feature_names(self) -> list[str]:
        return list(self.model.feature_names_in_)

    @property
    def weather_groups(self) -> list[str]:
        return [c[len(tm.WEATHER_PREFIX):] for c in self.feature_names if c.startswith(tm.WEATHER_PREFIX)]


class PredictionRequest(BaseModel):
    timestamp: datetime = Field(description="Hour to predict, local time as in the dataset (time zone is ignored).")
    temp_c: float = Field(ge=-60, le=60, description="Air temperature in degrees Celsius.")
    rain_1h: float = Field(ge=0, description="Rain in the hour (mm).")
    snow_1h: float = Field(ge=0, description="Snow in the hour (mm).")
    clouds_all: float = Field(ge=0, le=100, description="Cloud cover (%).")
    weather_main: str = Field(description="Weather group, e.g. Clear, Clouds, Rain, Snow (see GET /model).")
    is_holiday: bool = Field(default=False, description="True when the date is a holiday.")

    model_config = {"json_schema_extra": {"example": {
        "timestamp": "2018-06-14T17:00:00", "temp_c": 22.5, "rain_1h": 0.0, "snow_1h": 0.0,
        "clouds_all": 40, "weather_main": "Clouds", "is_holiday": False}}}


class PredictionResponse(BaseModel):
    predicted_traffic_volume: float
    unit: str = UNIT
    traffic_category: str
    is_congested: bool
    category_note: str = CATEGORY_NOTE
    model_name: str
    model_version: str
    model_alias: str


def traffic_category(volume: float) -> str:
    """Part 2 categories (same boundaries as feature_engineering.py)."""
    if volume < 4500:
        return "Low"
    return "Medium" if volume <= 5500 else "High"


def build_request_frame(req: PredictionRequest, feature_names: list[str], weather: str) -> pd.DataFrame:
    """One-row feature matrix equal to what tm.build_features gives for the same hour."""
    ts = pd.Timestamp(req.timestamp)
    hour, dow, month = ts.hour, ts.dayofweek, ts.month
    values = {
        "hour": hour, "day_of_week": dow, "month": month, "is_weekend": int(dow >= 5),
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * month / 12), "month_cos": np.cos(2 * np.pi * month / 12),
        "holiday_flag": int(req.is_holiday), "temp_c": req.temp_c, "rain_1h": req.rain_1h,
        "snow_1h": req.snow_1h, "clouds_all": req.clouds_all,
        "is_raining": int(req.rain_1h > 0), "is_snowing": int(req.snow_1h > 0),
    }
    row = {name: values.get(name, 0) for name in feature_names}
    row[tm.WEATHER_PREFIX + weather] = 1
    return pd.DataFrame([row], columns=feature_names)


def load_bundle(settings: Settings) -> ModelBundle:
    mlflow.set_tracking_uri(settings.mlflow_uri)
    model, version = common.load_registered_model(settings.model_name, settings.alias)
    return ModelBundle(model, settings.model_name, version, settings.alias)


def create_app(settings: Settings | None = None, loader=None) -> FastAPI:
    """Build the app; ``loader`` (settings -> ModelBundle) can be replaced, e.g. in tests."""
    settings = settings or Settings()
    loader = loader or load_bundle

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = None
        try:
            app.state.bundle = loader(settings)
        except (OSError, ValueError, MlflowException):
            LOG.exception("Model could not be loaded; /predict will return 503.")
        yield

    app = FastAPI(
        title="Metro Interstate traffic volume API (deployment simulation)",
        description="Serves the registered same-hour traffic volume regressor. Local mock-up for Part 3, Task 6.",
        lifespan=lifespan,
    )

    def bundle() -> ModelBundle:
        if app.state.bundle is None:
            raise HTTPException(status_code=503, detail="Model is not loaded; see the server log.")
        return app.state.bundle

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok" if app.state.bundle is not None else "model not loaded",
                "model_loaded": app.state.bundle is not None}

    @app.get("/model")
    def model_info() -> dict:
        b = bundle()
        return {"name": b.name, "version": b.version, "alias": b.alias, "n_features": len(b.feature_names),
                "weather_groups": b.weather_groups, "unit": UNIT}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(req: PredictionRequest) -> PredictionResponse:
        b = bundle()
        groups = {g.lower(): g for g in b.weather_groups}
        weather = groups.get(req.weather_main.strip().lower())
        if weather is None:
            raise HTTPException(status_code=422, detail=f"Unknown weather_main '{req.weather_main}'. "
                                                        f"Use one of: {', '.join(sorted(groups.values()))}.")
        frame = build_request_frame(req, b.feature_names, weather)
        volume = max(0.0, float(b.model.predict(frame)[0]))
        LOG.info("Predicted %.1f vehicles/hour for %s (%s) with %s v%s.", volume, req.timestamp, weather, b.name, b.version)
        return PredictionResponse(
            predicted_traffic_volume=round(volume, 1), traffic_category=traffic_category(volume),
            is_congested=volume > 5500, model_name=b.name, model_version=b.version, model_alias=b.alias,
        )

    @app.get("/monitor/status")
    def monitor_status() -> dict:
        """Latest PASS/ALERT status written by monitoring.py."""
        try:
            return json.loads(settings.status_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="No monitoring status yet; run monitoring.py first.") from None

    return app


app = create_app()


def _parse_args() -> argparse.Namespace:
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1, local only).")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000).")
    parser.add_argument("--mlflow-uri", default=defaults.mlflow_uri, help="MLflow tracking URI.")
    parser.add_argument("--model-name", default=defaults.model_name, help="Registered model name.")
    parser.add_argument("--alias", default=defaults.alias, help="Registered model alias to serve (default: champion).")
    parser.add_argument("--status-file", type=Path, default=defaults.status_file,
                        help="monitoring_status.json written by monitoring.py.")
    parser.add_argument("--log-file", type=Path, default=common.PART_DIR / "logs" / "serve_api.log", help="Log file path.")
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
    import uvicorn

    args = _parse_args()
    try:
        _configure_logging(args.log_file, getattr(logging, args.log_level))
    except OSError:
        logging.basicConfig(level=logging.ERROR, stream=sys.stderr, force=True)
        LOG.exception("Failed to configure logging.")
        return 1
    settings = Settings(args.mlflow_uri, args.model_name, args.alias, args.status_file)
    LOG.info("Starting API on http://%s:%d (model %s@%s).", args.host, args.port, args.model_name, args.alias)
    # log_config=None keeps the handlers configured above for uvicorn's own loggers.
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_config=None, log_level=args.log_level.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
