# Part 2: Data Pipeline

Run the cleaning pipeline from the repository root:

```powershell
python Part-2/pipeline.py
```

By default, the pipeline reads the raw `Metro_Interstate_Traffic_Volume.csv`
from the repository root, writes the cleaned data to
`Part-2/data/processed/Metro_Interstate_Traffic_Volume_cleaned.csv`, and writes
logs to `Part-2/logs/pipeline.log`. The log format includes a timestamp, level,
module name, and message. Use `--log-level DEBUG` to include troubleshooting
details. Override paths with `--input`, `--output`, or `--log-file`.

The pipeline validates the required columns before cleaning, standardizes
categorical values and day-first timestamps, removes exact duplicates and
duplicate timestamps (keeping the first row), and imputes invalid temperatures
and rainfall above 9,000 mm with the corresponding month's median. The known
23-24 July 2016 low-traffic outage is preserved and reported as a warning.

## Feature engineering

Run feature engineering after the cleaning pipeline:

```powershell
python Part-2/feature_engineering.py
```

By default, it reads
`Part-2/data/processed/Metro_Interstate_Traffic_Volume_cleaned.csv`, writes
`Part-2/data/processed/Metro_Interstate_Traffic_Volume_features.csv`, and logs
to `Part-2/logs/feature_engineering.log`. Use `--input`, `--output`,
`--log-file`, or `--log-level DEBUG` to override these defaults.

The script adds hour, day of week, weekend, and sine/cosine hour features;
one-hot encodes `weather_main` and `weather_description`; adds rain/snow
indicators and z-score features for temperature and rainfall. The
`traffic_category` target uses the established thresholds: Low < 4,500,
Medium 4,500–5,500 inclusive, and High > 5,500 vehicles/hour.
`is_congested` is 1 when traffic volume is > 5,500, otherwise 0. Original
columns are retained. Before/after dataset shapes are logged at INFO; derived
normalization statistics and the target thresholds are logged at DEBUG.

## Task 3: Traffic visualizations

Run the visualization script after the cleaning pipeline:

```powershell
python Part-2/visualize.py
```

It reads the cleaned CSV and saves four PNG figures to
`Part-2/outputs/visualizations/`. Each successful save is logged at INFO,
including the full output path, in `Part-2/logs/visualizations.log`. Use
`--input`, `--output-dir`, `--log-file`, or `--log-level` to override defaults.
The weather comparison displays all categories and labels each sample count.
Interpret categories with small samples cautiously.

### Interpretations

1. **Traffic by hour** — Demand rises sharply through the morning and peaks
   around 16:00 (about 5,709 vehicles/hour); the overnight trough is around
   03:00 (about 373 vehicles/hour).
2. **Weekday vs. weekend hourly profile** — Weekdays average about 3,557
   vehicles/hour versus about 2,624 on weekends, showing a substantial
   workday/weekend difference in demand.
3. **Traffic by weather** — The observed mean is higher for Clouds (about
   3,617 vehicles/hour) than Clear (about 3,055). This is an association, not
   evidence that weather causes the difference; weather categories also differ
   in their time-of-day mix. Sample counts vary widely: Thunderstorm has 453
   observations, Fog 201, Smoke 15, and Squall 1, so those means are less
   reliable than categories with more observations.
4. **Monthly congestion rate** — The share of observed hours above 5,500
   vehicles/hour varies month to month. Separate panels show the rate and the
   observed record count; the line breaks where no month data exists. The
   monthly share is calculated only from observed hours, not imputed hours.
   July 2016 is marked because it includes the known low-traffic outage on
   23-24 July. Interpret changes cautiously given uneven coverage and the
   outage; do not treat them as a continuous trend.

## Task 4: Command-line traffic app

Run the app from the repository root after creating the cleaned dataset:

```powershell
python Part-2/traffic_app.py --help
```

By default, the app reads
`Part-2/data/processed/Metro_Interstate_Traffic_Volume_cleaned.csv` once per
run and writes logs to `Part-2/logs/traffic_app.log` and the console. The log
format is `%(asctime)s - %(levelname)s - %(message)s`. Override the dataset
with the global `--input` option before the command. The supplied 2023 date
example is outside this dataset's 2012-10-02 through 2018-09-30 range; use an
in-range date such as `2017-07-10`.

### Commands and sample output

```powershell
python Part-2/traffic_app.py date 2017-07-10 --hour 8
python Part-2/traffic_app.py peak --top 3
python Part-2/traffic_app.py compare
python Part-2/traffic_app.py recommend
python Part-2/traffic_app.py weather 2017-07-10
```

`date` without `--hour` lists the observed hourly traffic rows for that date.
`peak` ranks individual hourly records and defaults to five. `compare` shows
weekday/weekend means. `recommend` lists the five lowest pooled historical
hourly means; it is a historical summary, not a guarantee of future traffic.
`weather` shows observed hourly weather and traffic for a date.

Example `date` output:

```text
Date       | Hour  | Traffic (vehicles/hour)
-----------+-------+------------------------
2017-07-10 | 08:00 |                   5,827
```

Example `compare` output:

```text
Period            | Mean traffic (vehicles/hour) | Observed hours
------------------+------------------------------+---------------
Weekday (Mon-Fri) |                      3,557.4 |         28,979
Weekend (Sat-Sun) |                      2,623.9 |         11,596
```

Example `peak --top 3` output:

```text
Date       | Hour  | Traffic (vehicles/hour)
-----------+-------+------------------------
2017-03-09 | 16:00 |                   7,280
2016-04-21 | 07:00 |                   7,260
2016-04-22 | 07:00 |                   7,241
```

Example `recommend` output:

```text
Hour  | Mean traffic (vehicles/hour) | Observed hours
------+------------------------------+---------------
03:00 |                        373.2 |          1,669
02:00 |                        392.6 |          1,697
01:00 |                        521.3 |          1,710
04:00 |                        701.3 |          1,703
00:00 |                        844.9 |          1,723
```

Example `weather 2017-07-10` output excerpt:

```text
Hour  | Weather | Description | Traffic (vehicles/hour)
------+---------+-------------+------------------------
00:00 | Rain    | Light Rain  |                     649
01:00 | Rain    | Light Rain  |                     397
```

Run `--help` after any command for its arguments and example. Invalid input
and data-loading failures are logged at ERROR and exit nonzero without a raw
traceback. A valid in-range date with no records prints a friendly message and
exits successfully. Each recognized command invocation is logged at INFO,
including the parser message for missing or malformed command arguments.

### CLI checks

Run valid commands with `--help`, `date`, `peak`, `compare`, `recommend`, and
`weather` as above. The following cases should print an ERROR message and exit
nonzero, except the in-range no-data query, which should print its friendly
message and exit 0:

```powershell
python Part-2/traffic_app.py date 14-05-2023
python Part-2/traffic_app.py date 2017-07-10 --hour 25
python Part-2/traffic_app.py date 2023-05-14
python Part-2/traffic_app.py date
python Part-2/traffic_app.py unknown
python Part-2/traffic_app.py --input Part-2/missing.csv compare
python Part-2/traffic_app.py date 2014-12-01
```

## Part 2 methodology and findings report

The concise two-page [Methodology and Findings report](Methodology_and_Findings_Report.pdf)
summarizes the cleaning, feature engineering, visualizations, and CLI methods;
key descriptive results; and the dataset and modelling limitations.
