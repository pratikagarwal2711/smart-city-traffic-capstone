"""Create traffic-pattern visualizations from the cleaned Metro traffic data."""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("date_time", "traffic_volume", "weather_main")
LOG = logging.getLogger(__name__)


def _load_data(input_path: Path) -> pd.DataFrame:
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        data = pd.read_csv(input_file)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(
            "The cleaned dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    parsed_dates = pd.to_datetime(
        data["date_time"],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    if parsed_dates.isna().any():
        examples = data.loc[parsed_dates.isna(), "date_time"].head(5).tolist()
        raise ValueError(f"Invalid date_time values found, for example: {examples}")

    traffic_volume = pd.to_numeric(data["traffic_volume"], errors="raise")
    if not np.isfinite(traffic_volume.to_numpy(dtype=float)).all():
        raise ValueError("traffic_volume contains missing or non-finite values.")

    data = data.copy()
    data["date_time"] = parsed_dates
    data["traffic_volume"] = traffic_volume
    data["weather_main"] = (
        data["weather_main"].astype("string").fillna("Unknown").str.strip()
    )
    data["weather_main"] = data["weather_main"].mask(
        data["weather_main"].eq(""), "Unknown"
    )
    data["hour"] = parsed_dates.dt.hour
    data["day_of_week"] = parsed_dates.dt.dayofweek
    data["is_weekend"] = data["day_of_week"].ge(5)
    data["is_congested"] = traffic_volume.gt(5500)
    LOG.info("Loaded %d cleaned hourly records from %s.", len(data), input_path)
    return data


def _plot_hourly_demand(data: pd.DataFrame) -> plt.Figure:
    hourly_means = (
        data.groupby("hour")["traffic_volume"].mean().reindex(range(24))
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(hourly_means.index, hourly_means.values, marker="o", linewidth=2)
    axis.set(
        title="Average traffic demand by hour",
        xlabel="Hour of day",
        ylabel="Mean traffic volume (vehicles/hour)",
        xticks=range(24),
    )
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    return figure


def _plot_weekday_weekend(data: pd.DataFrame) -> plt.Figure:
    profiles = data.groupby(["is_weekend", "hour"])["traffic_volume"].mean()
    figure, axis = plt.subplots(figsize=(10, 5))
    for is_weekend, label, color in (
        (False, "Weekday (Mon-Fri)", "tab:blue"),
        (True, "Weekend (Sat-Sun)", "tab:orange"),
    ):
        profile = profiles.loc[is_weekend].reindex(range(24))
        axis.plot(
            profile.index,
            profile.values,
            marker="o",
            markersize=3,
            label=label,
            color=color,
        )
    axis.set(
        title="Hourly traffic profile: weekdays vs. weekends",
        xlabel="Hour of day",
        ylabel="Mean traffic volume (vehicles/hour)",
        xticks=range(24),
    )
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def _plot_weather_relationship(data: pd.DataFrame) -> plt.Figure:
    weather_summary = (
        data.groupby("weather_main")["traffic_volume"]
        .agg(mean_traffic="mean", observations="size")
        .sort_values("mean_traffic")
    )
    if weather_summary.empty:
        raise ValueError("No weather categories are available to visualize.")

    figure, axis = plt.subplots(figsize=(12, 6))
    bars = axis.bar(
        weather_summary.index,
        weather_summary["mean_traffic"],
        color="tab:blue",
    )
    axis.bar_label(
        bars,
        labels=[
            f"n={observations:,}"
            for observations in weather_summary["observations"]
        ],
        padding=3,
        fontsize=8,
    )
    axis.set(
        title="Mean traffic by weather category (sample counts shown)",
        xlabel="Weather category",
        ylabel="Mean traffic volume (vehicles/hour)",
    )
    axis.tick_params(axis="x", labelrotation=35)
    plt.setp(axis.get_xticklabels(), ha="right")
    axis.margins(y=0.15)
    axis.grid(axis="y", alpha=0.3)
    figure.text(
        0.5,
        0.01,
        "Sample counts vary; interpret means with small samples cautiously.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    LOG.info(
        "Weather chart includes all %d observed categories with sample counts.",
        len(weather_summary),
    )
    return figure


def _plot_monthly_congestion(data: pd.DataFrame) -> plt.Figure:
    monthly = (
        data.assign(month=data["date_time"].dt.to_period("M"))
        .groupby("month", sort=True)
        .agg(
            congestion_rate=("is_congested", "mean"),
            observed_hours=("traffic_volume", "size"),
        )
    )
    monthly = monthly.reindex(
        pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    )
    dates = monthly.index.to_timestamp()
    figure, (rate_axis, coverage_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": (2, 1)},
    )
    coverage_axis.bar(
        dates,
        monthly["observed_hours"],
        width=20,
        color="0.75",
        alpha=0.4,
        label="Observed hours",
        zorder=1,
    )
    rate_axis.plot(
        dates,
        monthly["congestion_rate"].mul(100),
        color="tab:red",
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Congestion rate",
        zorder=2,
    )
    outage_month = pd.Period("2016-07", freq="M")
    if outage_month in monthly.index:
        outage_start = pd.Timestamp("2016-07-01")
        outage_end = pd.Timestamp("2016-08-01")
        rate_axis.axvspan(
            outage_start,
            outage_end,
            color="tab:orange",
            alpha=0.2,
            label="Includes known outage (23-24 Jul 2016)",
            zorder=0,
        )
        coverage_axis.axvspan(
            outage_start,
            outage_end,
            color="tab:orange",
            alpha=0.2,
            zorder=0,
        )
    rate_axis.set(
        title="Monthly congestion rate and data coverage",
        ylabel="Congestion rate (%)",
    )
    rate_axis.set_ylim(bottom=0)
    coverage_axis.set_ylabel("Observed hourly records")
    coverage_axis.set_xlabel("Month")
    coverage_axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    coverage_axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    coverage_axis.tick_params(axis="x", labelrotation=30)
    rate_axis.grid(axis="y", alpha=0.3)
    rate_axis.legend(loc="upper left")
    figure.tight_layout()
    return figure


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    try:
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)
    LOG.info("Saved figure to %s.", output_path)


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
        "--output-dir",
        type=Path,
        default=task_dir / "outputs" / "visualizations",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=task_dir / "logs" / "visualizations.log",
        help="Log file path.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Minimum level written to the console and log file (default: INFO).",
    )
    return parser.parse_args()


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
        LOG.exception("Failed to configure visualization logging.")
        return 1

    try:
        LOG.info("Starting traffic visualization generation.")
        data = _load_data(args.input)
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        visualizations = (
            (_plot_hourly_demand, "traffic_by_hour.png"),
            (_plot_weekday_weekend, "weekday_vs_weekend_by_hour.png"),
            (_plot_weather_relationship, "traffic_by_weather.png"),
            (_plot_monthly_congestion, "monthly_congestion_rate.png"),
        )
        for create_figure, filename in visualizations:
            _save_figure(create_figure(data), output_dir / filename)
        LOG.info(
            "Created %d traffic visualizations in %s.",
            len(visualizations),
            output_dir,
        )
    except (
        OSError,
        UnicodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ValueError,
    ):
        LOG.exception("Traffic visualization generation failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
