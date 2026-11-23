"""Travel-timing recommendation engine for Part 3, Task 5.

The data covers one corridor, so the engine recommends *when* to travel, not
which route to take. From the historical hourly volumes it builds an expected
traffic profile for each day type (Weekday, Weekend, Holiday) and weather group,
finds the lowest-volume travel window and writes a plain-language
recommendation, for example::

    For a weekday journey, consider travelling between 10:00 AM and 11:00 AM,
    when historical traffic volumes are typically lower.

Method notes:

* The expected volume of an hour is the historical mean for the same day type,
  weather group and hour of day. Cells with few observations are shrunk towards
  the all-weather mean of that day type and hour (``SHRINKAGE_HOURS``); this
  reduces the influence of sparse cells but does not guarantee reliable estimates.
* Windows are searched inside the daytime travel range (06:00-19:00 by default);
  a quieter evening window is mentioned separately. This range is a configurable
  choice, not a conclusion about when people can travel.
* The profile is descriptive, not a live forecast, and volume is not a safety
  measure: there is no accident data in this project. ``run_backtest`` checks the
  advice on the held-out last 20% of hours.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prepare_labels import LOW_VISIBILITY_WEATHER, SEVERE_WEATHER

LOG = logging.getLogger(__name__)

DAY_TYPES = ("Weekday", "Weekend", "Holiday")
WEATHER_GROUPS = ("Clear", "Clouds", "Rain/Drizzle", "Low visibility", "Severe weather")
ANY_WEATHER = "Any"
CONGESTION_LEVELS = ("Low", "Medium", "High", "Severe")
# Pseudo-observations that anchor a sparse day-type/weather/hour cell to the
# all-weather mean (empirical-Bayes style shrinkage); a cell with this many
# hours is weighted 50/50.
SHRINKAGE_HOURS = 10
# Travel windows must start at or after the first hour and end by the second.
DAYTIME = (6, 19)
EVENING = (19, 22)
DEFAULT_DURATIONS = (1, 2, 3)
# An evening window is only mentioned when it is at least this much quieter.
LATER_OPTION_GAIN = 0.10
# Weather changes below this share of the all-weather volume are called negligible.
WEATHER_NOTE_THRESHOLD = 0.05
TEST_FRACTION = 0.20
REQUIRED_COLUMNS = ("timestamp", "holiday", "weather_main", "traffic_volume")

DAY_TYPE_PHRASES = {"Weekday": "a weekday", "Weekend": "a weekend", "Holiday": "a holiday or event day"}
WEATHER_PHRASES = {
    "Clear": "in clear weather",
    "Clouds": "in cloudy weather",
    "Rain/Drizzle": "in rainy or drizzly weather",
    "Low visibility": "in low-visibility weather (fog, mist, haze or smoke)",
    "Severe weather": "in severe weather (thunderstorm, snow or squall)",
}


def format_hour(hour: int) -> str:
    """12-hour clock label for a whole hour, e.g. 10 -> '10:00 AM', 24 -> '12:00 AM'."""
    hour = hour % 24
    suffix = "AM" if hour < 12 else "PM"
    return f"{hour % 12 or 12}:00 {suffix}"


def format_window(start: int, end: int) -> str:
    return f"{format_hour(start)} and {format_hour(end)}"


def weather_group(weather_main: pd.Series) -> pd.Series:
    """Weather bands shared with Task 2 (Clear, Clouds, Rain/Drizzle, Low visibility, Severe)."""
    known = ("Clear", "Clouds", "Rain", "Drizzle") + LOW_VISIBILITY_WEATHER + SEVERE_WEATHER
    unknown = weather_main[~weather_main.isin(known)]
    if not unknown.empty:
        raise ValueError(f"Unknown weather categories: {', '.join(sorted(unknown.astype(str).unique()))}.")
    groups = weather_main.where(weather_main.isin(("Clear", "Clouds")), "Rain/Drizzle")
    groups[weather_main.isin(LOW_VISIBILITY_WEATHER)] = "Low visibility"
    groups[weather_main.isin(SEVERE_WEATHER)] = "Severe weather"
    return groups


def prepare_history(data: pd.DataFrame) -> pd.DataFrame:
    """Return time-sorted rows with date, hour, day_type and weather_group columns."""
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError("Input is missing required columns: " + ", ".join(missing))
    df = data[list(REQUIRED_COLUMNS)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["traffic_volume"] = pd.to_numeric(df["traffic_volume"], errors="coerce")
    bad = df["timestamp"].isna() | df["traffic_volume"].isna() | df["weather_main"].isna()
    if bad.any():
        LOG.warning("Dropped %d rows with a missing timestamp, volume or weather.", int(bad.sum()))
        df = df[~bad]
    if df.empty:
        raise ValueError("No usable rows in the input.")
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["date"] = df["timestamp"].dt.normalize()
    df["hour"] = df["timestamp"].dt.hour
    # The source flags a holiday only on its 00:00 row, so spread it over the whole date.
    flagged = df["holiday"].notna() & (df["holiday"].astype(str).str.lower() != "none")
    holiday_dates = set(df.loc[flagged, "date"])
    df["day_type"] = np.where(df["timestamp"].dt.dayofweek >= 5, "Weekend", "Weekday")
    df.loc[df["date"].isin(holiday_dates), "day_type"] = "Holiday"
    df["weather_group"] = weather_group(df["weather_main"])
    LOG.info(
        "History: %d hours from %s to %s; %d holiday dates; day types %s.",
        len(df),
        df["timestamp"].min(),
        df["timestamp"].max(),
        len(holiday_dates),
        df.drop_duplicates("date")["day_type"].value_counts().to_dict(),
    )
    return df[["timestamp", "date", "hour", "day_type", "weather_group", "traffic_volume"]]


def build_profile(history: pd.DataFrame) -> pd.DataFrame:
    """Expected volume for every day type x weather group x hour.

    ``weather_group`` 'Any' is the all-weather mean; the other groups are shrunk
    towards it by their sample size. Columns: day_type, weather_group, hour,
    n_hours, mean_volume (raw cell mean) and expected_volume.
    """
    overall = (
        history.groupby(["day_type", "hour"])["traffic_volume"]
        .agg(n_hours="size", mean_volume="mean")
        .reset_index()
        .assign(weather_group=ANY_WEATHER)
    )
    overall["expected_volume"] = overall["mean_volume"]

    cells = history.groupby(["day_type", "weather_group", "hour"])["traffic_volume"].agg(
        n_hours="size", mean_volume="mean"
    )
    grid = pd.MultiIndex.from_product(
        [DAY_TYPES, WEATHER_GROUPS, range(24)], names=["day_type", "weather_group", "hour"]
    )
    cells = cells.reindex(grid).reset_index()
    cells["n_hours"] = cells["n_hours"].fillna(0).astype(int)
    prior = overall.set_index(["day_type", "hour"])["expected_volume"]
    cells["prior"] = prior.reindex(pd.MultiIndex.from_frame(cells[["day_type", "hour"]])).to_numpy()
    weight = cells["n_hours"] / (cells["n_hours"] + SHRINKAGE_HOURS)
    cells["expected_volume"] = weight * cells["mean_volume"].fillna(cells["prior"]) + (1 - weight) * cells["prior"]

    columns = ["day_type", "weather_group", "hour", "n_hours", "mean_volume", "expected_volume"]
    profile = pd.concat([overall[columns], cells[columns]], ignore_index=True)
    return profile.sort_values(["day_type", "weather_group", "hour"]).reset_index(drop=True)


def find_windows(curve: pd.Series, duration: int, earliest: int, latest: int) -> pd.DataFrame:
    """All windows of ``duration`` hours inside [earliest, latest], with their mean volume."""
    rows = []
    for start in range(earliest, latest - duration + 1):
        values = curve.reindex(range(start, start + duration))
        if values.notna().all():
            rows.append((start, start + duration, float(values.mean())))
    return pd.DataFrame(rows, columns=["start", "end", "expected_volume"])


def non_overlapping(windows: pd.DataFrame, count: int) -> pd.DataFrame:
    """Greedily keep the ``count`` quietest windows that do not overlap each other."""
    chosen = []
    for row in windows.sort_values(["expected_volume", "start"]).itertuples():
        if all(row.start >= c.end or row.end <= c.start for c in chosen):
            chosen.append(row)
        if len(chosen) == count:
            break
    return pd.DataFrame(
        [(c.start, c.end, c.expected_volume) for c in chosen], columns=["start", "end", "expected_volume"]
    )


@dataclass
class Window:
    start: int
    end: int
    expected_volume: float

    @property
    def label(self) -> str:
        return f"{format_hour(self.start)} to {format_hour(self.end)}"


@dataclass
class Recommendation:
    day_type: str
    weather: str
    duration: int
    earliest: int
    latest: int
    best: Window
    level: str
    alternatives: list = field(default_factory=list)
    peak: Window | None = None
    range_average: float = float("nan")
    later_option: Window | None = None
    weather_ratio: float | None = None
    min_weather_hours: int | None = None
    text: str = ""

    @property
    def reduction_vs_peak(self) -> float:
        return 1 - self.best.expected_volume / self.peak.expected_volume

    @property
    def limited_evidence(self) -> bool:
        return self.min_weather_hours is not None and self.min_weather_hours < SHRINKAGE_HOURS


class RecommendationEngine:
    """Expected-volume profile plus the rules that turn it into travel advice."""

    def __init__(self, history: pd.DataFrame):
        self.profile = build_profile(history)
        # Same quartile definition as the Part 3 congestion_category.
        self.thresholds = tuple(history["traffic_volume"].quantile([0.25, 0.5, 0.75]).to_numpy())

    def level(self, volume: float) -> str:
        for name, cutoff in zip(CONGESTION_LEVELS, self.thresholds):
            if volume <= cutoff:
                return name
        return CONGESTION_LEVELS[-1]

    def curve(self, day_type: str, weather: str) -> pd.DataFrame:
        part = self.profile[(self.profile["day_type"] == day_type) & (self.profile["weather_group"] == weather)]
        return part.set_index("hour")

    def recommend(
        self,
        day_type: str,
        weather: str = ANY_WEATHER,
        duration: int = 1,
        earliest: int = DAYTIME[0],
        latest: int = DAYTIME[1],
        top_n: int = 3,
    ) -> Recommendation:
        day_type = _canonical(day_type, DAY_TYPES, "day type")
        weather = _canonical(weather, (ANY_WEATHER,) + WEATHER_GROUPS, "weather")
        if not 1 <= duration <= 6:
            raise ValueError("duration must be between 1 and 6 hours.")
        if not 0 <= earliest < latest <= 24 or latest - earliest < duration:
            raise ValueError("The travel range must be inside 0-24 and at least as long as the duration.")

        data = self.curve(day_type, weather)
        curve = data["expected_volume"]
        windows = find_windows(curve, duration, earliest, latest)
        if windows.empty:
            raise ValueError(f"No history for a {duration}-hour {day_type} window between {earliest} and {latest}.")
        picks = [Window(*row) for row in non_overlapping(windows, max(top_n, 1)).itertuples(index=False)]
        peak_row = windows.sort_values(["expected_volume", "start"]).iloc[-1]
        best = picks[0]

        rec = Recommendation(
            day_type=day_type,
            weather=weather,
            duration=duration,
            earliest=earliest,
            latest=latest,
            best=best,
            level=self.level(best.expected_volume),
            alternatives=picks[1:],
            peak=Window(int(peak_row.start), int(peak_row.end), float(peak_row.expected_volume)),
            range_average=float(curve.reindex(range(earliest, latest)).mean()),
        )
        rec.later_option = self._later_option(curve, duration, best, latest)
        if weather != ANY_WEATHER:
            hours = range(best.start, best.end)
            baseline = self.curve(day_type, ANY_WEATHER)["expected_volume"].reindex(hours).mean()
            rec.weather_ratio = float(curve.reindex(hours).mean() / baseline)
            rec.min_weather_hours = int(data["n_hours"].reindex(hours).min())
            if rec.limited_evidence:
                LOG.warning(
                    "%s / %s %02d:00-%02d:00 rests on as few as %d historical hours.",
                    day_type, weather, best.start, best.end, rec.min_weather_hours,
                )
        rec.text = compose_text(rec)
        return rec

    @staticmethod
    def _later_option(curve: pd.Series, duration: int, best: Window, latest: int) -> Window | None:
        # This optional suggestion is explicitly outside the requested range.
        evening = find_windows(curve, duration, max(EVENING[0], latest), EVENING[1])
        if evening.empty:
            return None
        row = evening.sort_values(["expected_volume", "start"]).iloc[0]
        if row.expected_volume <= best.expected_volume * (1 - LATER_OPTION_GAIN):
            return Window(int(row.start), int(row.end), float(row.expected_volume))
        return None


def _canonical(value: str, allowed: tuple, name: str) -> str:
    lookup = {a.lower(): a for a in allowed}
    key = str(value).strip().lower()
    if key not in lookup:
        raise ValueError(f"Unknown {name} '{value}'. Choose one of: {', '.join(allowed)}.")
    return lookup[key]


def compose_text(rec: Recommendation) -> str:
    """Plain-language recommendation for one scenario."""
    who = DAY_TYPE_PHRASES[rec.day_type] + " journey"
    if rec.weather != ANY_WEATHER:
        who += " " + WEATHER_PHRASES[rec.weather]
    best = rec.best
    parts = [
        f"For {who}, consider travelling between {format_window(best.start, best.end)}, "
        f"when historical traffic volumes are typically lower "
        f"(about {best.expected_volume:,.0f} vehicles per hour, '{rec.level}' on this corridor's "
        f"four-level congestion scale)."
    ]
    if rec.level in CONGESTION_LEVELS[2:]:
        parts.append("This is the quietest option in the travel range, but volumes are still relatively high.")
    if rec.peak and rec.reduction_vs_peak >= 0.01:
        parts.append(
            f"That is about {rec.reduction_vs_peak:.0%} less than the busiest period, "
            f"{rec.peak.label} (about {rec.peak.expected_volume:,.0f} vehicles per hour)."
        )
    if rec.alternatives:
        options = "; ".join(f"{w.label} (about {w.expected_volume:,.0f})" for w in rec.alternatives)
        parts.append(f"Other good options: {options}.")
    if rec.later_option:
        parts.append(
            f"If you can travel later, outside your requested range, {rec.later_option.label} is lighter still "
            f"(about {rec.later_option.expected_volume:,.0f} vehicles per hour)."
        )
    parts.append(_weather_note(rec))
    parts.append(
        f"This is based on historical hourly volumes on this corridor for travel between "
        f"{format_hour(rec.earliest)} and {format_hour(rec.latest)}; it is not a live forecast."
    )
    return " ".join(p for p in parts if p)


def _weather_note(rec: Recommendation) -> str:
    if rec.weather == ANY_WEATHER:
        return ""
    if rec.limited_evidence:
        note = (
            f"Few {rec.weather.lower()} hours exist in the history for this window (as few as "
            f"{rec.min_weather_hours}), so the estimate leans on the all-weather pattern."
        )
    else:
        change = rec.weather_ratio - 1
        if abs(change) < WEATHER_NOTE_THRESHOLD:
            note = "The weather-adjusted historical estimate differs from the all-weather average by under 5% at this time."
        else:
            note = (
                f"The weather-adjusted historical estimate is about {abs(change):.0%} "
                f"{'higher' if change > 0 else 'lower'} than the all-weather average at this time."
            )
    if rec.weather in ("Low visibility", "Severe weather"):
        note += " Traffic volume is not a safety measure, so allow extra time and drive to the conditions."
    return note


def scenario_table(engine: RecommendationEngine, durations=DEFAULT_DURATIONS) -> pd.DataFrame:
    """One recommendation per day type x weather x duration, with text."""
    rows = []
    for day_type in DAY_TYPES:
        for weather in (ANY_WEATHER,) + WEATHER_GROUPS:
            for duration in durations:
                rec = engine.recommend(day_type, weather, duration)
                rows.append(
                    {
                        "day_type": day_type,
                        "weather": weather,
                        "duration_hours": duration,
                        "start_hour": rec.best.start,
                        "end_hour": rec.best.end,
                        "window": rec.best.label,
                        "expected_volume": round(rec.best.expected_volume),
                        "congestion_level": rec.level,
                        "peak_window": rec.peak.label,
                        "peak_volume": round(rec.peak.expected_volume),
                        "reduction_vs_peak_pct": round(100 * rec.reduction_vs_peak, 1),
                        "alternatives": "; ".join(w.label for w in rec.alternatives),
                        "later_option": rec.later_option.label if rec.later_option else "",
                        "weather_vs_all_pct": (
                            round(100 * (rec.weather_ratio - 1), 1) if rec.weather_ratio is not None else None
                        ),
                        "min_weather_hours": rec.min_weather_hours,
                        "limited_evidence": rec.limited_evidence,
                        "recommendation": rec.text,
                    }
                )
    return pd.DataFrame(rows)


def split_history(history: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Chronological split matching Tasks 1-3 (history is already time-sorted)."""
    cut = int(len(history) * (1 - test_fraction))
    return history.iloc[:cut], history.iloc[cut:]


def run_backtest(history: pd.DataFrame, durations=DEFAULT_DURATIONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on the first 80% of hours and check the advice on the last 20%.

    Returns (window_check, weather_check). window_check compares, for each fully
    observed test day, the actual volume in the recommended window with the
    busiest window and with the average daytime hour. weather_check compares the
    mean absolute error of the hourly profile with and without weather.
    """
    train, test = split_history(history)
    engine = RecommendationEngine(train)
    LOG.info("Backtest: train %d hours to %s, test %d hours from %s.",
             len(train), train["timestamp"].max(), len(test), test["timestamp"].min())

    daily = test.pivot_table(index="date", columns="hour", values="traffic_volume", aggfunc="mean")
    day_types = test.drop_duplicates("date").set_index("date")["day_type"]
    first, last = DAYTIME
    daytime_hours = list(range(first, last))

    rows = []
    for day_type in DAY_TYPES:
        days = daily[day_types.reindex(daily.index) == day_type]
        complete = days.dropna(subset=daytime_hours) if set(daytime_hours) <= set(days.columns) else days.iloc[0:0]
        if complete.empty:
            LOG.warning("Backtest: no fully observed %s days in the test period.", day_type)
            continue
        average = complete[daytime_hours].mean(axis=1)
        for duration in durations:
            rec = engine.recommend(day_type, ANY_WEATHER, duration)
            best = complete[list(range(rec.best.start, rec.best.end))].mean(axis=1)
            peak = complete[list(range(rec.peak.start, rec.peak.end))].mean(axis=1)
            rows.append(
                {
                    "day_type": day_type,
                    "duration_hours": duration,
                    "recommended_window": rec.best.label,
                    "peak_window": rec.peak.label,
                    "test_days": len(complete),
                    "recommended_actual_mean": round(best.mean()),
                    "peak_actual_mean": round(peak.mean()),
                    "daytime_actual_mean": round(average.mean()),
                    "reduction_vs_peak_pct": round(100 * (1 - best.mean() / peak.mean()), 1),
                    "reduction_vs_daytime_pct": round(100 * (1 - best.mean() / average.mean()), 1),
                    "days_below_daytime_average_pct": round(100 * (best < average).mean(), 1),
                    "days_below_peak_pct": round(100 * (best < peak).mean(), 1),
                }
            )
    window_check = pd.DataFrame(rows)

    lookup = engine.profile.set_index(["day_type", "weather_group", "hour"])["expected_volume"]
    keys = pd.MultiIndex.from_frame(test[["day_type", "hour"]].assign(w=ANY_WEATHER)[["day_type", "w", "hour"]])
    base = lookup.reindex(keys).to_numpy()
    keys = pd.MultiIndex.from_frame(test[["day_type", "weather_group", "hour"]])
    aware = lookup.reindex(keys).to_numpy()
    actual = test["traffic_volume"].to_numpy()
    scored = pd.DataFrame(
        {
            "weather_group": test["weather_group"].to_numpy(),
            "base_error": np.abs(actual - base),
            "weather_error": np.abs(actual - aware),
        }
    )
    grouped = scored.groupby("weather_group").agg(
        test_hours=("base_error", "size"), mae_without_weather=("base_error", "mean"), mae_with_weather=("weather_error", "mean")
    )
    overall = pd.DataFrame(
        {
            "test_hours": [len(scored)],
            "mae_without_weather": [scored["base_error"].mean()],
            "mae_with_weather": [scored["weather_error"].mean()],
        },
        index=["All"],
    )
    weather_check = pd.concat([overall, grouped]).rename_axis("weather_group").round(1).reset_index()
    return window_check, weather_check


def plot_profile(engine: RecommendationEngine, path: Path) -> None:
    """Expected hourly volume per day type, with the recommended 1-hour window shaded."""
    fig, axes = plt.subplots(1, len(DAY_TYPES), figsize=(16, 4.5), sharey=True)
    for ax, day_type in zip(axes, DAY_TYPES):
        for weather in WEATHER_GROUPS:
            curve = engine.curve(day_type, weather)["expected_volume"]
            ax.plot(curve.index, curve.values, linewidth=1, alpha=0.7, label=weather)
        overall = engine.curve(day_type, ANY_WEATHER)["expected_volume"]
        ax.plot(overall.index, overall.values, color="black", linewidth=2.5, label="All weather")
        rec = engine.recommend(day_type)
        ax.axvspan(rec.best.start, rec.best.end, color="green", alpha=0.25, label=f"Recommended {rec.best.label}")
        ax.axvspan(*DAYTIME, color="grey", alpha=0.06)
        ax.set_title(day_type)
        ax.set_xlabel("Hour of day (shaded grey: daytime travel range)")
        ax.set_xticks(range(0, 24, 3))
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Expected traffic volume (vehicles/hour)")
    fig.suptitle("Expected hourly traffic by day type and weather")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    part_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        type=Path,
        default=part_dir / "data" / "processed" / "Metro_Interstate_Traffic_Volume_ml.csv",
        help="Labelled CSV from prepare_labels.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=part_dir / "outputs", help="Output directory.")
    parser.add_argument(
        "--day-type",
        help="Print one recommendation for this day type (Weekday, Weekend, Holiday) instead of writing outputs.",
    )
    parser.add_argument("--weather", default=ANY_WEATHER, help="Weather group (Any, " + ", ".join(WEATHER_GROUPS) + ").")
    parser.add_argument("--duration", type=int, default=1, help="Journey length in hours (default: 1).")
    parser.add_argument("--earliest", type=int, default=DAYTIME[0], help="Earliest start hour (default: 6).")
    parser.add_argument("--latest", type=int, default=DAYTIME[1], help="Latest end hour (default: 19).")
    parser.add_argument("--top-n", type=int, default=3, help="Windows to list including the best (default: 3).")
    parser.add_argument(
        "--log-file", type=Path, default=part_dir / "logs" / "recommendation_engine.log", help="Log file path."
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


def _save_outputs(output_dir: Path, history: pd.DataFrame, engine: RecommendationEngine) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = scenario_table(engine)
    window_check, weather_check = run_backtest(history)
    profile = engine.profile.copy()
    profile["expected_volume"] = profile["expected_volume"].round(1)
    profile["mean_volume"] = profile["mean_volume"].round(1)
    files = {
        "recommendation_profile.csv": profile,
        "recommendation_windows.csv": windows,
        "recommendation_backtest.csv": window_check,
        "recommendation_weather_check.csv": weather_check,
    }
    for name, table in files.items():
        table.to_csv(output_dir / name, index=False, encoding="utf-8")
        LOG.info("Saved %s.", output_dir / name)
    plot_profile(engine, output_dir / "recommendation_profile.png")
    LOG.info("Saved %s.", output_dir / "recommendation_profile.png")

    info = {
        "rows": len(history),
        "first_timestamp": str(history["timestamp"].min()),
        "last_timestamp": str(history["timestamp"].max()),
        "congestion_thresholds": [float(t) for t in engine.thresholds],
        "shrinkage_hours": SHRINKAGE_HOURS,
        "daytime_range": list(DAYTIME),
        "evening_range": list(EVENING),
        "scenarios": len(windows),
        "scenarios_with_limited_weather_evidence": int(windows["limited_evidence"].sum()),
    }
    info_path = output_dir / "recommendation_run_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    LOG.info("Saved %s.", info_path)
    LOG.info("Backtest windows:\n%s", window_check.to_string(index=False))
    LOG.info("Backtest weather check:\n%s", weather_check.to_string(index=False))


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
        history = prepare_history(pd.read_csv(args.input, encoding="utf-8-sig"))
        engine = RecommendationEngine(history)
        if args.day_type:
            rec = engine.recommend(
                args.day_type, args.weather, args.duration, args.earliest, args.latest, args.top_n
            )
            print(rec.text)
        else:
            _save_outputs(args.output_dir, history, engine)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        LOG.exception("Recommendation engine failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
