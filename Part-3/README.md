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

## Task 1: supervised models

```powershell
python -m pip install scikit-learn mlflow
python Part-3/train_models.py
```

It reads the labelled CSV above, logs to `Part-3/logs/train_models.log`, and
writes `Part-3/outputs/model_metrics.csv`, `feature_importance.csv` and
`run_info.json`. Options: `--input`, `--output-dir`, `--log-file`,
`--log-level`, `--mlflow-uri`, `--mlflow-artifact-root`, `--experiment`,
`--no-mlflow`.

**MLflow:** each model and baseline is logged as a run in the experiment
`task1-supervised-models` (parameters, split details, metrics, the fitted model
and the output CSVs). Tracking data is in the SQLite database
`Part-3/mlflow.db` and model artifacts in `Part-3/mlartifacts/` (about 145 MB,
git-ignored, recreated by the script). Each script run adds six new runs.
View them with:

```powershell
python -m pip install mlflow
mlflow ui --backend-store-uri sqlite:///Part-3/mlflow.db
```

Models are stored in cloudpickle format (they are produced by this script);
only load models you trust. The older MLflow file store (`mlruns`) is not used
because MLflow 3.x disables it by default.

**Notebook:** [`notebooks/Task1_supervised_models.ipynb`](notebooks/Task1_supervised_models.ipynb)
reuses `train_models.py` and adds exploration, actual-vs-predicted, error by
hour, ROC/PR curves, confusion matrices, feature importances and the MLflow run
table. It is saved with outputs. To re-run it:

```powershell
python -m pip install jupyter matplotlib
python -m jupyter nbconvert --to notebook --execute --inplace Part-3/notebooks/Task1_supervised_models.ipynb
```

**Shared features (28):** hour, day of week, month, weekend flag; cyclical
sin/cos of hour, day of week and month; a holiday flag spread over the whole
date; temperature (°C), rain, snow, cloud cover, rain/snow indicators; and
one-hot `weather_main`.

**Split:** time-based, first 80% of hours train (2012-10-02 to 2017-10-26 17:00,
32,460 rows), last 20% test (2017-10-26 18:00 to 2018-09-30, 8,115 rows). The
test period includes partially covered 2018. Fixed `random_state` of 42; the
classification threshold is fixed at 0.5.

**Models:** regression predicts `traffic_volume` (Linear Regression vs Random
Forest, plus reference baselines: train mean and mean by hour). Classification
predicts `high_risk` (Logistic Regression vs Random Forest, both with class
weights).

| Task | Model | Results on test set |
|---|---|---|
| Regression | Baseline: train mean | MAE 1,724; R² −0.000 |
| Regression | Baseline: mean by hour | MAE 622; R² 0.783 |
| Regression | Linear Regression | MAE 801; R² 0.735 |
| Regression | Random Forest | MAE 239; R² 0.959 |
| Classification | Logistic Regression | Acc 0.957; Prec 0.689; Rec 0.993; F1 0.814; ROC AUC 0.995 |
| Classification | Random Forest | Acc 0.986; Prec 0.880; Rec 0.986; F1 0.930; ROC AUC 0.997 |

Interpretation: Random Forest is best on both tasks. The linear regression
does not beat the mean-by-hour baseline because traffic is non-linear in hour
(bimodal), which supports the project finding that time of day dominates.
Random Forest regression importance is concentrated in the hour features.

**Classification scores are inflated by design.** `high_risk` is defined from
`traffic_volume` and `weather_main`, and the shared feature set includes the
`weather_main` one-hot columns (as requested). The models are largely
re-learning the label rule (high-volume hours combined with listed weather),
so the near-perfect ROC AUC does not show real accident-prediction skill. The
test positive rate (9.4%) also differs from train (6.8%), and the label
quartiles were computed on all rows (see Cautions). Results are associations,
single time split, no hyperparameter tuning or cross-validation.

## Cautions

- **Leakage:** both labels are derived from `traffic_volume` and `weather_main`.
  `train_models.py` deliberately includes `weather_main` one-hot features, as
  the task requires weather encodings, and excludes the other columns below.
  Otherwise, do not use `traffic_volume`, `traffic_category`, `is_congested`,
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
