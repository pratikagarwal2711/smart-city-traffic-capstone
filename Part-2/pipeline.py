"""Clean the Metro Interstate Traffic Volume dataset."""

import argparse
import csv
import logging
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

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
NUMERIC_COLUMNS = (
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "traffic_volume",
)
LOG = logging.getLogger(__name__)


def _validate_schema(fieldnames: list[str] | None) -> list[str]:
    if fieldnames is None:
        raise ValueError("The CSV is empty or does not contain a header.")
    if len(fieldnames) != len(set(fieldnames)):
        raise ValueError("The CSV header contains duplicate column names.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(missing_columns)
        )
    return fieldnames


def _load_rows(input_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = _validate_schema(reader.fieldnames)
        rows: list[dict[str, str]] = []

        for record_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV record {record_number} has an inconsistent number of fields.")
            rows.append(row)

    LOG.info("Loaded %d records from %s; required schema validated.", len(rows), input_path)
    return rows, fieldnames


def _standardize_categories(rows: list[dict[str, str]]) -> None:
    changed_cells = 0
    category_columns = ("holiday", "weather_main", "weather_description")

    for row in rows:
        for column in category_columns:
            original = row[column]
            normalized = original.strip()
            if column == "holiday":
                normalized = "None" if normalized.casefold() in ("", "none") else normalized.title()
            else:
                normalized = normalized.title()
            if normalized != original:
                row[column] = normalized
                changed_cells += 1

    LOG.info(
        "Cleaning step: standardized categorical values; changed %d cells.",
        changed_cells,
    )


def _parse_timestamps(rows: list[dict[str, str]]) -> None:
    for record_number, row in enumerate(rows, start=2):
        value = row["date_time"].strip()
        try:
            parsed = datetime.strptime(value, "%d/%m/%Y %H:%M")
        except ValueError as error:
            raise ValueError(
                f"Invalid day-first date_time at CSV record {record_number}: {value!r}"
            ) from error
        row["date_time"] = parsed.strftime("%d/%m/%Y %H:%M")

    LOG.info("Cleaning step: parsed and validated %d date_time values.", len(rows))


def _remove_duplicates(
    rows: list[dict[str, str]], fieldnames: list[str]
) -> list[dict[str, str]]:
    unique_rows: list[dict[str, str]] = []
    seen_rows: set[tuple[str, ...]] = set()

    for row in rows:
        signature = tuple(row[column] for column in fieldnames)
        if signature not in seen_rows:
            unique_rows.append(row)
            seen_rows.add(signature)

    exact_duplicates = len(rows) - len(unique_rows)
    LOG.info(
        "Cleaning step: removed %d exact duplicate rows; %d records remain.",
        exact_duplicates,
        len(unique_rows),
    )
    if exact_duplicates:
        LOG.warning("Removed %d exact duplicate rows.", exact_duplicates)

    timestamp_rows: list[dict[str, str]] = []
    seen_timestamps: set[str] = set()
    for row in unique_rows:
        timestamp = row["date_time"]
        if timestamp not in seen_timestamps:
            timestamp_rows.append(row)
            seen_timestamps.add(timestamp)

    duplicate_hours = len(unique_rows) - len(timestamp_rows)
    LOG.info(
        "Cleaning step: kept the first row for each date_time and removed %d "
        "duplicate-hour rows; %d records remain.",
        duplicate_hours,
        len(timestamp_rows),
    )
    if duplicate_hours:
        LOG.warning(
            "Dropped %d rows with duplicate date_time values, keeping the first row.",
            duplicate_hours,
        )
    return timestamp_rows


def _validated_numeric_value(row: dict[str, str], column: str, record_number: int) -> float:
    raw_value = row[column].strip()
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(
            f"Non-numeric {column} value at CSV record {record_number}: {raw_value!r}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite {column} value at CSV record {record_number}: {raw_value!r}"
        )
    return value


def _impute_monthly_medians(rows: list[dict[str, str]]) -> None:
    parsed_dates = [
        datetime.strptime(row["date_time"], "%d/%m/%Y %H:%M") for row in rows
    ]
    values_by_month: dict[int, dict[str, list[float]]] = {
        month: {column: [] for column in ("temp", "rain_1h")}
        for month in range(1, 13)
    }
    outliers: dict[str, list[tuple[int, int]]] = {"temp": [], "rain_1h": []}

    for index, row in enumerate(rows):
        record_number = index + 2
        numeric_values = {
            column: _validated_numeric_value(row, column, record_number)
            for column in NUMERIC_COLUMNS
        }

        if numeric_values["temp"] <= 0:
            outliers["temp"].append((index, parsed_dates[index].month))
            LOG.warning(
                "Invalid temperature %s K at cleaned record %d; will impute by month.",
                row["temp"],
                record_number,
            )
        else:
            values_by_month[parsed_dates[index].month]["temp"].append(
                numeric_values["temp"]
            )

        if numeric_values["rain_1h"] < 0:
            raise ValueError(
                f"Impossible negative rain_1h at CSV record {record_number}: "
                f"{numeric_values['rain_1h']}"
            )
        if numeric_values["rain_1h"] > 9000:
            outliers["rain_1h"].append((index, parsed_dates[index].month))
            LOG.warning(
                "Physically implausible rainfall %s mm at cleaned record %d; "
                "will impute by month.",
                row["rain_1h"],
                record_number,
            )
        else:
            values_by_month[parsed_dates[index].month]["rain_1h"].append(
                numeric_values["rain_1h"]
            )

        if numeric_values["snow_1h"] < 0:
            raise ValueError(
                f"Impossible negative snow_1h at CSV record {record_number}: "
                f"{numeric_values['snow_1h']}"
            )
        if not 0 <= numeric_values["clouds_all"] <= 100:
            raise ValueError(
                f"clouds_all must be between 0 and 100 at CSV record {record_number}."
            )
        if numeric_values["traffic_volume"] < 0:
            raise ValueError(
                f"traffic_volume cannot be negative at CSV record {record_number}."
            )

    medians: dict[int, dict[str, float]] = {}
    for month, values in values_by_month.items():
        medians[month] = {}
        for column, monthly_values in values.items():
            if monthly_values:
                medians[month][column] = statistics.median(monthly_values)
                LOG.debug(
                    "Month %02d %s median used for imputation: %s.",
                    month,
                    column,
                    medians[month][column],
                )

    for column, entries in outliers.items():
        imputed_by_month: dict[int, int] = {}
        for index, month in entries:
            if month not in medians or column not in medians[month]:
                raise ValueError(
                    f"Cannot impute {column} outlier in month {month}: "
                    "no valid values are available for that month."
                )
            rows[index][column] = str(medians[month][column])
            imputed_by_month[month] = imputed_by_month.get(month, 0) + 1
        LOG.info(
            "Cleaning step: processed %d %s outliers using monthly medians "
            "(by month: %s).",
            len(entries),
            column,
            imputed_by_month,
        )
        if entries:
            LOG.warning(
                "Imputed %d %s outlier values using monthly medians.",
                len(entries),
                column,
            )


def _warn_known_outage(rows: list[dict[str, str]]) -> None:
    affected_rows = 0
    for row in rows:
        timestamp = datetime.strptime(row["date_time"], "%d/%m/%Y %H:%M")
        if timestamp.date().isoformat() in ("2016-07-23", "2016-07-24"):
            affected_rows += 1
    if affected_rows:
        LOG.warning(
            "Known low-traffic outage on 23-24 July 2016 is preserved "
            "(%d records); review before modelling.",
            affected_rows,
        )
    else:
        LOG.info("Known July 2016 traffic outage dates are not present in the input.")


def _write_rows(
    rows: list[dict[str, str]], fieldnames: list[str], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    LOG.info("Wrote %d cleaned records to %s.", len(rows), output_path)


def run_pipeline(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different to preserve the raw CSV.")

    rows, fieldnames = _load_rows(input_path)
    _standardize_categories(rows)
    _parse_timestamps(rows)
    rows = _remove_duplicates(rows, fieldnames)
    _impute_monthly_medians(rows)
    _warn_known_outage(rows)
    _write_rows(rows, fieldnames, output_path)


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
    project_dir = task_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_dir / "Metro_Interstate_Traffic_Volume.csv",
        help="Raw CSV path (default: dataset at the repository root).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=task_dir
        / "data"
        / "processed"
        / "Metro_Interstate_Traffic_Volume_cleaned.csv",
        help="Cleaned CSV path.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=task_dir / "logs" / "pipeline.log",
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
        LOG.exception("Failed to configure pipeline logging.")
        return 1

    try:
        LOG.info("Starting traffic data pipeline.")
        LOG.debug("Input CSV: %s; cleaned output CSV: %s.", args.input, args.output)
        run_pipeline(args.input, args.output)
        LOG.info("Traffic data pipeline completed successfully.")
    except (OSError, UnicodeError, csv.Error, ValueError):
        LOG.exception("Traffic data pipeline failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
