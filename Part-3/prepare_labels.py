"""Create the Part 3 congestion category and proxy accident-risk label.

No accident dataset is available, so ``high_risk`` is a proxy label: High/Severe
congestion together with severe or low-visibility weather. It demonstrates the
classification workflow and must not be read as a prediction of real accidents.

Both labels are derived from ``traffic_volume`` (and ``weather_main``), so these
columns must never be used as model features when predicting the labels.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

LOG = logging.getLogger(__name__)

SEVERE_WEATHER = ("Thunderstorm", "Snow", "Squall")
LOW_VISIBILITY_WEATHER = ("Fog", "Mist", "Haze", "Smoke")
WEATHER_PREFIX = "weather_main_"
REQUIRED_COLUMNS = ("traffic_volume", "date_time")


def restore_weather_main(df: pd.DataFrame) -> pd.Series:
    """Recover ``weather_main`` from the one-hot columns created in Part 2."""
    columns = [c for c in df.columns if c.startswith(WEATHER_PREFIX)]
    if not columns:
        raise ValueError(f"No '{WEATHER_PREFIX}*' columns found in the input.")
    if not (df[columns].sum(axis=1) == 1).all():
        raise ValueError("Each row must have exactly one weather_main category.")
    return df[columns].idxmax(axis=1).str.removeprefix(WEATHER_PREFIX)


def add_proxy_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with congestion_category and the proxy high_risk label."""
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError("Input is missing required columns: " + ", ".join(missing))

    df = data.copy()
    df["timestamp"] = pd.to_datetime(
        df["date_time"], format="%d/%m/%Y %H:%M", errors="raise"
    )
    if "weather_main" not in df.columns:
        df["weather_main"] = restore_weather_main(df)
    df["is_low_visibility"] = df["weather_main"].isin(LOW_VISIBILITY_WEATHER).astype("int8")

    # Data-driven quartiles of traffic_volume
    q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.5, 0.75]).values
    LOG.info("Traffic volume quartiles: q1=%.1f, q2=%.1f, q3=%.1f.", q1, q2, q3)

    def bucket(v: float) -> str:
        if v <= q1:
            return "Low"
        elif v <= q2:
            return "Medium"
        elif v <= q3:
            return "High"
        return "Severe"

    df["congestion_category"] = pd.Categorical(
        df["traffic_volume"].apply(bucket),
        categories=("Low", "Medium", "High", "Severe"),
        ordered=True,
    )

    # High risk = High/Severe congestion together with severe/low-visibility weather
    high_congestion = df["congestion_category"].isin(["High", "Severe"])
    risky_weather = df["weather_main"].isin(SEVERE_WEATHER) | (df["is_low_visibility"] == 1)
    df["high_risk"] = (high_congestion & risky_weather).astype("int8")

    LOG.info(
        "congestion_category counts: %s.",
        df["congestion_category"].value_counts().sort_index().to_dict(),
    )
    LOG.info(
        "high_risk: %d of %d rows (%.2f%%) are positive.",
        df["high_risk"].sum(),
        len(df),
        100 * df["high_risk"].mean(),
    )
    LOG.debug("SEVERE_WEATHER=%s; LOW_VISIBILITY_WEATHER=%s.", SEVERE_WEATHER, LOW_VISIBILITY_WEATHER)
    return df


def _parse_args() -> argparse.Namespace:
    part_dir = Path(__file__).resolve().parent
    processed = part_dir.parent / "Part-2" / "data" / "processed"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=processed / "Metro_Interstate_Traffic_Volume_features.csv",
        help="Feature CSV from Part 2.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=part_dir / "data" / "processed" / "Metro_Interstate_Traffic_Volume_ml.csv",
        help="Labelled CSV output.",
    )
    parser.add_argument(
        "--log-file", type=Path, default=part_dir / "logs" / "prepare_labels.log", help="Log file path."
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
        labelled = add_proxy_labels(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        labelled.to_csv(args.output, index=False, encoding="utf-8")
        LOG.info("Saved labelled dataset (%d rows, %d columns) to %s.", *labelled.shape, args.output)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        LOG.exception("Label preparation failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
