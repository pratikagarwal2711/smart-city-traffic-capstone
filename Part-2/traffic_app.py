# Run from the repository root after creating the cleaned dataset:
#   python Part-2/pipeline.py
#   python Part-2/traffic_app.py --help

"""Command-line traffic analytics for the Metro Interstate dataset."""

import argparse
import logging
import re
import sys
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
TASK_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    TASK_DIR
    / "data"
    / "processed"
    / "Metro_Interstate_Traffic_Volume_cleaned.csv"
)
LOG_FILE = TASK_DIR / "logs" / "traffic_app.log"
REQUIRED_COLUMNS = (
    "date_time",
    "traffic_volume",
    "weather_main",
    "weather_description",
)
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
RECOMMENDATION_COUNT = 5
COMMAND_NAMES = ("date", "peak", "compare", "recommend", "weather")


class UserInputError(ValueError):
    """An expected command-line input error that should not show a traceback."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        command = self.prog.rsplit(" ", maxsplit=1)[-1]
        if command in COMMAND_NAMES:
            LOG.info("Command '%s' invoked with args: %s", command, message)
        LOG.error("Invalid command arguments: %s.", message)
        self.print_usage(sys.stderr)
        self.exit(2)


def _configure_logging(log_path: Path) -> None:
    """Write concise INFO and ERROR messages to the console and log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
        force=True,
    )


def _build_parser() -> _ArgumentParser:
    """Create the top-level parser and all command-specific help."""
    parser = _ArgumentParser(
        description=(
            "Query traffic, congestion, and weather from the cleaned hourly "
            "Metro Interstate dataset."
        ),
        epilog=(
            "Examples:\n"
            "  python Part-2/traffic_app.py date 2017-07-10 --hour 8\n"
            "  python Part-2/traffic_app.py peak --top 5\n"
            "  python Part-2/traffic_app.py compare\n"
            "  python Part-2/traffic_app.py recommend\n"
            "  python Part-2/traffic_app.py weather 2017-07-10"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Cleaned CSV path (default: Task 1 output).",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=_ArgumentParser,
    )

    date_parser = subparsers.add_parser(
        "date",
        help="Show hourly traffic for a date, optionally narrowed to one hour.",
        description="Show traffic volume for a date in YYYY-MM-DD format.",
        epilog="Example: python Part-2/traffic_app.py date 2017-07-10 --hour 8",
    )
    date_parser.add_argument("date", help="Date to query (YYYY-MM-DD).")
    date_parser.add_argument(
        "--hour",
        type=int,
        metavar="HOUR",
        help="Optional hour to query (integer from 0 to 23).",
    )

    peak_parser = subparsers.add_parser(
        "peak",
        help="Show the highest-traffic hourly periods.",
        description="List the top N hourly traffic records.",
        epilog="Example: python Part-2/traffic_app.py peak --top 5",
    )
    peak_parser.add_argument(
        "--top",
        type=int,
        default=5,
        metavar="N",
        help="Number of periods to show (default: 5).",
    )

    subparsers.add_parser(
        "compare",
        help="Compare average weekday and weekend traffic.",
        description="Compare weekday and weekend traffic volume.",
        epilog="Example: python Part-2/traffic_app.py compare",
    )
    subparsers.add_parser(
        "recommend",
        help="Show the five quietest hours by historical average traffic.",
        description=(
            "Recommend quiet travel hours using pooled average traffic "
            "across the available dataset."
        ),
        epilog="Example: python Part-2/traffic_app.py recommend",
    )

    weather_parser = subparsers.add_parser(
        "weather",
        help="Show hourly weather and traffic for a date.",
        description="Show observed weather and traffic for a date.",
        epilog="Example: python Part-2/traffic_app.py weather 2017-07-10",
    )
    weather_parser.add_argument("date", help="Date to query (YYYY-MM-DD).")
    return parser


def _load_data(input_path: Path) -> pd.DataFrame:
    """Load and validate the cleaned hourly traffic CSV once."""
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        data = pd.read_csv(input_file)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))
    if data.empty:
        raise ValueError("The data file contains no traffic records.")

    parsed_dates = pd.to_datetime(
        data["date_time"],
        format="%d/%m/%Y %H:%M",
        errors="raise",
    )
    if parsed_dates.duplicated().any():
        raise ValueError(
            "Duplicate timestamps found; use the cleaned Task 1 dataset."
        )

    traffic_volume = pd.to_numeric(data["traffic_volume"], errors="raise")
    if not np.isfinite(traffic_volume.to_numpy(dtype=float)).all():
        raise ValueError("traffic_volume contains missing or non-finite values.")
    if traffic_volume.lt(0).any():
        raise ValueError("traffic_volume contains negative values.")

    data = data.copy()
    data["date_time"] = parsed_dates
    data["date"] = parsed_dates.dt.date
    data["hour"] = parsed_dates.dt.hour
    data["day_of_week"] = parsed_dates.dt.dayofweek
    data["traffic_volume"] = traffic_volume
    for column in ("weather_main", "weather_description"):
        data[column] = (
            data[column].astype("string").fillna("Unknown").str.strip()
        )
        data[column] = data[column].mask(data[column].eq(""), "Unknown")

    LOG.info("Loaded %s hourly records from '%s'.", f"{len(data):,}", input_path)
    return data


def _validate_date(value: str) -> date:
    """Parse only real calendar dates written as YYYY-MM-DD."""
    if not DATE_PATTERN.fullmatch(value):
        raise UserInputError(
            f"Invalid date '{value}'. Expected format: YYYY-MM-DD"
        )
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise UserInputError(
            f"Invalid date '{value}'. Expected format: YYYY-MM-DD"
        ) from error


def _validate_hour(hour: int | None) -> int | None:
    """Reject hours outside the 24-hour clock."""
    if hour is not None and not 0 <= hour <= 23:
        raise UserInputError(
            f"Invalid hour '{hour}'. Expected an integer from 0 to 23."
        )
    return hour


def _validate_top(top: int) -> int:
    """Require a positive number of rows for ranked commands."""
    if top < 1:
        raise UserInputError(
            f"Invalid top value '{top}'. Expected an integer greater than 0."
        )
    return top


def _validate_date_range(query_date: date, data: pd.DataFrame) -> None:
    """Reject dates outside the loaded dataset's observed date range."""
    first_date = data["date"].min()
    last_date = data["date"].max()
    if query_date < first_date or query_date > last_date:
        raise UserInputError(
            f"Date '{query_date.isoformat()}' is outside dataset range "
            f"{first_date.isoformat()} to {last_date.isoformat()}."
        )


def _print_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    right_aligned: Sequence[int] = (),
) -> None:
    """Print a compact, aligned table for end users."""
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        if rows
        else len(headers[index])
        for index in range(len(headers))
    ]

    def format_row(values: Sequence[str]) -> str:
        cells = []
        for index, value in enumerate(values):
            cell = (
                value.rjust(widths[index])
                if index in right_aligned
                else value.ljust(widths[index])
            )
            cells.append(cell)
        return " | ".join(cells)

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def _print_no_data(query_date: date, hour: int | None = None) -> int:
    """Show a friendly result when a valid date has no matching records."""
    if hour is None:
        print(f"No data found for that date: {query_date.isoformat()}.")
    else:
        print(
            f"No data found for that date and hour: "
            f"{query_date.isoformat()} {hour:02d}:00."
        )
    return 0


def _run_date(data: pd.DataFrame, query_date: date, hour: int | None) -> int:
    """Show the hourly traffic observations for one date."""
    selected = data.loc[data["date"].eq(query_date)]
    if hour is not None:
        selected = selected.loc[selected["hour"].eq(hour)]
    selected = selected.sort_values("date_time")
    if selected.empty:
        return _print_no_data(query_date, hour)

    rows = [
        [
            row.date_time.strftime("%Y-%m-%d"),
            f"{row.hour:02d}:00",
            f"{row.traffic_volume:,.0f}",
        ]
        for row in selected.itertuples()
    ]
    _print_table(
        ("Date", "Hour", "Traffic (vehicles/hour)"),
        rows,
        right_aligned=(2,),
    )
    return 0


def _run_peak(data: pd.DataFrame, top: int) -> int:
    """Show the top N hourly traffic observations."""
    selected = data.sort_values(
        ["traffic_volume", "date_time"],
        ascending=[False, True],
    ).head(top)
    if selected.empty:
        print("No traffic records are available.")
        return 0

    rows = [
        [
            row.date_time.strftime("%Y-%m-%d"),
            row.date_time.strftime("%H:%M"),
            f"{row.traffic_volume:,.0f}",
        ]
        for row in selected.itertuples()
    ]
    _print_table(
        ("Date", "Hour", "Traffic (vehicles/hour)"),
        rows,
        right_aligned=(2,),
    )
    return 0


def _run_compare(data: pd.DataFrame) -> int:
    """Compare mean traffic on weekdays and weekends."""
    rows = []
    for label, mask in (
        ("Weekday (Mon-Fri)", data["day_of_week"].lt(5)),
        ("Weekend (Sat-Sun)", data["day_of_week"].ge(5)),
    ):
        group = data.loc[mask, "traffic_volume"]
        rows.append(
            [
                label,
                f"{group.mean():,.1f}" if not group.empty else "No data",
                f"{len(group):,}",
            ]
        )
    _print_table(
        ("Period", "Mean traffic (vehicles/hour)", "Observed hours"),
        rows,
        right_aligned=(1, 2),
    )
    return 0


def _run_recommend(data: pd.DataFrame) -> int:
    """Recommend the five hours with the lowest pooled historical averages."""
    summary = (
        data.groupby("hour")["traffic_volume"]
        .agg(mean_traffic="mean", observations="size")
        .sort_values(["mean_traffic"], ascending=True)
        .head(RECOMMENDATION_COUNT)
    )
    rows = [
        [
            f"{hour:02d}:00",
            f"{row.mean_traffic:,.1f}",
            f"{int(row.observations):,}",
        ]
        for hour, row in summary.iterrows()
    ]
    _print_table(
        ("Hour", "Mean traffic (vehicles/hour)", "Observed hours"),
        rows,
        right_aligned=(1, 2),
    )
    print("Based on pooled historical averages; actual traffic may vary.")
    return 0


def _run_weather(data: pd.DataFrame, query_date: date) -> int:
    """Show observed weather and traffic for each hour of one date."""
    selected = data.loc[data["date"].eq(query_date)].sort_values("date_time")
    if selected.empty:
        return _print_no_data(query_date)

    rows = [
        [
            row.date_time.strftime("%H:%M"),
            row.weather_main,
            row.weather_description,
            f"{row.traffic_volume:,.0f}",
        ]
        for row in selected.itertuples()
    ]
    _print_table(
        ("Hour", "Weather", "Description", "Traffic (vehicles/hour)"),
        rows,
        right_aligned=(3,),
    )
    return 0


def _log_invocation(args: argparse.Namespace) -> None:
    """Record the selected command and its effective arguments."""
    details = []
    if args.command in ("date", "weather"):
        details.append(f"date={args.date}")
    if args.command == "date":
        details.append(f"hour={args.hour}")
    if args.command == "peak":
        details.append(f"top={args.top}")
    details.append(f"input={args.input}")
    LOG.info(
        "Command '%s' invoked with args: %s",
        args.command,
        ", ".join(details),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Configure logging, parse one command, load data once, and dispatch."""
    try:
        _configure_logging(LOG_FILE)
    except OSError as error:
        sys.stderr.write(f"ERROR - Unable to configure logging: {error}\n")
        return 1

    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    _log_invocation(args)
    try:
        if args.command in ("date", "weather"):
            query_date = _validate_date(args.date)
        if args.command == "date":
            args.hour = _validate_hour(args.hour)
        if args.command == "peak":
            args.top = _validate_top(args.top)

        try:
            data = _load_data(args.input)
        except (
            OSError,
            UnicodeError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
            ValueError,
        ) as error:
            LOG.error("Unable to load data file '%s': %s", args.input, error)
            return 1
        if args.command in ("date", "weather"):
            _validate_date_range(query_date, data)
        if args.command == "date":
            return _run_date(data, query_date, args.hour)
        if args.command == "peak":
            return _run_peak(data, args.top)
        if args.command == "compare":
            return _run_compare(data)
        if args.command == "recommend":
            return _run_recommend(data)
        if args.command == "weather":
            return _run_weather(data, query_date)
    except UserInputError as error:
        LOG.error("%s", error)
        return 2
    except Exception as error:
        LOG.error("Unexpected error while processing the command: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
