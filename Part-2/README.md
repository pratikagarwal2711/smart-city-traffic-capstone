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
