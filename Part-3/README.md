# Part 3: Machine Learning and AI

## Setup: proxy labels

No accident dataset is available, so Part 3 classification uses a **proxy**
accident-risk label. It demonstrates the classification workflow only and is
not a prediction of real accidents.

```powershell
python Part-3/prepare_labels.py
```

By default it reads `Part-2/data/processed/Metro_Interstate_Traffic_Volume_features.csv`
(run the Part 2 pipeline and feature engineering first), writes
`Part-3/data/processed/Metro_Interstate_Traffic_Volume_ml.csv`, and logs to
`Part-3/logs/prepare_labels.log`. Use `--input`, `--output`, `--log-file`, or
`--log-level` to override defaults.

The script adds these columns and keeps all Part 2 columns:

- `timestamp` – parsed `date_time`, for time-based splits.
- `weather_main` – restored from the one-hot `weather_main_*` columns.
- `is_low_visibility` – 1 when `weather_main` is Fog, Mist, Haze or Smoke.
- `congestion_category` – quartiles of `traffic_volume` (Low ≤ q1, Medium ≤ q2,
  High ≤ q3, Severe above). On this data q1/q2/q3 = 1,248.5 / 3,427 / 4,952,
  matching the known project quartiles, giving four near-equal classes.
- `high_risk` – 1 when congestion is High/Severe **and** (`weather_main` is in
  `SEVERE_WEATHER` = Thunderstorm, Snow, Squall **or** `is_low_visibility` = 1).
  This gives 2,964 of 40,575 hours (7.30%) positive, so use PR-AUC/F1 and class
  weights rather than accuracy.

The weather groupings are assumptions defined as constants in
[`prepare_labels.py`](prepare_labels.py); change them there if the definition
should differ.

## Cautions

- **Leakage:** both labels are derived from `traffic_volume` and `weather_main`.
  Do not use `traffic_volume`, `traffic_category`, `is_congested`,
  `congestion_category`, `weather_main*`, or `is_low_visibility` as features
  when predicting `high_risk`; predicting from them would be circular.
- **Not the Part 2 target:** `congestion_category` (quartiles, four classes) is
  separate from Part 2's `traffic_category`/`is_congested` (fixed 4,500/5,500
  thresholds). Both columns are kept and must not be mixed.
- **Quartiles use all rows.** For model evaluation, consider computing them on
  the training period only to avoid test-set information in the label.
- Use a time-based split (`timestamp`); coverage is uneven across years and
  2018 is partially covered.
- Rare weather classes (Squall 1 h, Smoke 15 h, Fog 201 h) give weak evidence.
