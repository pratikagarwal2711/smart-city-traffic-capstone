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
