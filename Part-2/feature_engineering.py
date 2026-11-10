"""Create model-ready features from the cleaned Metro traffic dataset."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
)
LOG = logging.getLogger(__name__)


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return a feature-engineered copy of the cleaned traffic data."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "The cleaned dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    features = data.copy()
    parsed_dates = pd.to_datetime(
        features["date_time"],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    invalid_dates = parsed_dates.isna()
    if invalid_dates.any():
        examples = features.loc[invalid_dates, "date_time"].head(5).tolist()
        raise ValueError(f"Invalid date_time values found, for example: {examples}")

    numeric_columns = ("temp", "rain_1h", "snow_1h", "traffic_volume")
    for column in numeric_columns:
        features[column] = pd.to_numeric(features[column], errors="raise")
        if not np.isfinite(features[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} contains missing or non-finite values.")

    LOG.debug("Parsed date_time range: %s to %s.", parsed_dates.min(), parsed_dates.max())
    features["hour"] = parsed_dates.dt.hour.astype("int8")
    features["day_of_week"] = parsed_dates.dt.dayofweek.astype("int8")
    features["is_weekend"] = features["day_of_week"].isin((5, 6)).astype("int8")
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)

    features["is_raining"] = (features["rain_1h"] > 0).astype("int8")
    features["is_snowing"] = (features["snow_1h"] > 0).astype("int8")

    for column in ("temp", "rain_1h"):
        mean = float(features[column].mean())
        standard_deviation = float(features[column].std(ddof=0))
        LOG.debug(
            "Normalization statistics for %s: mean=%s, population_std=%s.",
            column,
            mean,
            standard_deviation,
        )
        if standard_deviation == 0:
            raise ValueError(
                f"Cannot z-score normalize {column}: its standard deviation is zero."
            )
        features[f"{column}_zscore"] = (features[column] - mean) / standard_deviation

    traffic_volume = features["traffic_volume"]
    features["traffic_category"] = pd.Categorical(
        np.select(
            [traffic_volume < 4500, traffic_volume <= 5500],
            ["Low", "Medium"],
            default="High",
        ),
        categories=("Low", "Medium", "High"),
        ordered=True,
    )
    features["is_congested"] = (traffic_volume > 5500).astype("int8")
    LOG.debug(
        "Traffic target thresholds: Low < 4500; Medium 4500-5500 inclusive; "
        "High > 5500; congested > 5500."
    )

    for column in ("weather_main", "weather_description"):
        features[column] = (
            features[column].astype("string").fillna("Unknown").str.strip()
        )
        features[column] = features[column].mask(features[column].eq(""), "Unknown")
    features = pd.get_dummies(
        features,
        columns=["weather_main", "weather_description"],
        prefix=["weather_main", "weather_description"],
        dtype="int8",
    )

    LOG.info(
        "Feature engineering complete: %d rows, %d columns.",
        features.shape[0],
        features.shape[1],
    )
    return features


def _parse_args() -> argparse.Namespace:
    task_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=task_dir
        / "data"
        / "processed"
        / "Metro_Interstate_Traffic_Volume_cleaned.csv",
        help="Cleaned CSV from Task 1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=task_dir
        / "data"
        / "processed"
        / "Metro_Interstate_Traffic_Volume_features.csv",
        help="Feature-engineered CSV output.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=task_dir / "logs" / "feature_engineering.log",
        help="Log file path.",
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
    except (OSError, ValueError):
        logging.basicConfig(
            level=logging.ERROR,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,
            force=True,
        )
        LOG.exception("Failed to configure feature-engineering logging.")
        return 1

    try:
        LOG.info("Starting feature engineering.")
        with args.input.open("r", encoding="utf-8-sig", newline="") as input_file:
            data = pd.read_csv(input_file)
        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in data.columns
        ]
        if missing_columns:
            raise ValueError(
                "The cleaned dataset is missing required columns: "
                + ", ".join(missing_columns)
            )
        LOG.info(
            "Dataset before feature engineering: %d rows, %d columns.",
            data.shape[0],
            data.shape[1],
        )
        features = engineer_features(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(args.output, index=False, encoding="utf-8")
        LOG.info(
            "Dataset after feature engineering: %d rows, %d columns.",
            features.shape[0],
            features.shape[1],
        )
        LOG.info("Saved feature-engineered dataset to %s.", args.output)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        LOG.exception("Feature engineering failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
