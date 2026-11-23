# Part 3 observations log

Running record of methods, results and caveats per task. It is the source
material for the final capstone report and the bias and fairness report. Add an
entry whenever a task finishes. Figures are from `outputs/` and the logs.

## Data source statement

No real accident dataset was sourced. Classification uses a documented proxy
label, `high_risk` (High/Severe congestion with Thunderstorm, Snow, Squall,
Fog, Mist, Haze or Smoke), created by `prepare_labels.py`. It is not an
accident prediction.

## Task 1: supervised models (`train_models.py`)

**Method**
- 40,575 hourly rows; 28 shared features (time, cyclical hour/day/month,
  whole-day holiday flag, weather one-hot and numeric weather).
- Time-based split: first 80% train (2012-10-02 to 2017-10-26), last 20% test
  (2017-10-26 to 2018-09-30; 2018 partially covered). Seed 42, threshold 0.5,
  no tuning or cross-validation.
- Regression target `traffic_volume`; classification target `high_risk` (class
  weights). Decisions confirmed by the user: full weather features despite
  circularity, labels and quartiles reused unchanged, scikit-learn only.

**Results (test set)**

| Task | Model | Results |
|---|---|---|
| Regression | Train-mean baseline | MAE 1,724; R² 0.000 |
| Regression | Mean-by-hour baseline | MAE 622; R² 0.783 |
| Regression | Linear Regression | MAE 801; R² 0.735 |
| Regression | Random Forest | MAE 239; R² 0.959 |
| Classification | Logistic Regression | Acc 0.957; Prec 0.689; Rec 0.993; F1 0.814; ROC AUC 0.995; PR AUC 0.923 |
| Classification | Random Forest | Acc 0.986; Prec 0.880; Rec 0.986; F1 0.930; ROC AUC 0.997; PR AUC 0.955 |

**Observations**
- Random Forest is best on both tasks.
- Linear Regression is worse than the mean-by-hour baseline: traffic is
  non-linear (bimodal) in hour.
- Hour features dominate Random Forest regression importance (hour_cos 0.67,
  hour 0.14, hour_sin 0.05); holiday flag 0.012 and temperature 0.011.
- Classification scores are inflated: `high_risk` is built from `weather_main`
  and `traffic_volume`, and weather_main one-hot is a feature (circularity).
- Test positive rate 9.4% vs train 6.8%; quartile cut-offs used all rows.
- Holiday flag covers 53 holiday dates (1,203 hours) after de-duplication.
- MLflow: 6 runs (4 models/baselines for regression and 2 classifiers) logged to
  SQLite `mlflow.db`; MLflow 3.x refuses the old file store and defaults to the
  skops format, which rejects Random Forest trees, so cloudpickle is used. The
  model artifacts are about 145 MB, so they are git-ignored. Re-running the
  script adds duplicate runs.
- The notebook reproduces the script metrics exactly. `high_risk` is 0% for
  Clear, Clouds, Drizzle, Rain (and the single Squall hour) and 33-59% for
  Smoke, Fog, Mist, Thunderstorm, Snow and Haze: weather is a direct label input.

**Bias and fairness notes (for the fairness report)**
- Uneven year coverage (only 2013, 2016, 2017 near-complete) means the models
  are better informed about some periods than others.
- Rare weather classes (Squall 1 h, Smoke 15 h, Fog 201 h) have very little
  evidence yet drive the proxy label's weather condition.
- The proxy label encodes assumptions about which weather is risky; subgroup
  metrics (weekday/weekend, holiday, weather, year) have not been computed yet.

## Task 2: unsupervised models (`unsupervised_models.py`)

**Method**
- K-means on standardised hour (sin/cos), weekend flag, weather severity (0 other,
  1 low visibility, 2 severe; Part 3 groups) and traffic_volume; k = 2..10 scored by
  inertia and silhouette (10,000-hour sample). Seed 42, n_init 10.
- Association rules: items = time of day (4 bins), Weekday/Weekend, weather group
  (5); consequent = quartile `congestion_category`; support >= 0.01, confidence
  >= 0.5, ranked by lift; exact counting of 1-3 item antecedents (pandas, no
  mlxtend). Redundancy flag `adds_information` = confidence at least 0.02 above
  every simpler rule (my addition, not in the brief; all rules still saved).
- User-confirmed choices: Part 3 weather groups, quartile congestion levels, 6-hour
  time bins, thresholds, pandas-only.

**K-means observations**
- Silhouette: k=2 0.314, 4 0.379, 5 0.392, 6 0.393, 7 0.412, 9 0.434 (max), 10 0.431.
  k = 6 chosen for interpretability (judgement), inertia 61,294.
- Clusters (hours, mean volume): 0 weekend night 5,120, 1,227; 1 weekday night
  10,260, 1,267; 2 severe weather any time 2,753, 2,998 (44.6% proxy high_risk,
  mixed congestion); 3 weekend daytime 5,738, 3,914 (68% High); 4 weekday
  afternoon/evening 8,907, 4,848 (53% Severe); 5 weekday morning-midday 7,797,
  5,174 (59% Severe).
- Only severe weather forms its own cluster; low-visibility hours (4-16% of each
  time-based cluster) are spread across clusters.
- Smaller k merges weekend with weekday traffic or hides the weather cluster.

**Rule observations (44 rules; 14 informative)**
- Top informative: Night & Weekend -> Low (conf 0.896, lift 3.59, 2,611 h); Night
  -> Low (0.853, 3.41); Afternoon & Weekend -> High (0.835, 3.34); Afternoon &
  Weekday -> Severe (0.749, 3.00); Morning & Weekday -> Severe (0.593, 2.37);
  Evening -> Medium (0.658, 2.63).
- Weather adds little: weather-specific informative rules gain 2-3 points, except
  Morning & Weekday & Low visibility -> Severe (0.694 vs 0.593, 740 h).
  Evening & Severe weather -> Medium (0.679 vs 0.658, 461 h).
- Rules by consequent: Low 15, Severe 14, Medium 12, High 3.
- Highest-lift raw rules are redundant refinements of "Night -> Low".

**Caveats / bias notes**
- Congestion level derives from volume, which is also a K-means feature (partly
  circular). Quartiles use all rows, so levels are relative to this dataset.
- k = 6 and the 2-point redundancy threshold are judgement calls. K-means is scale
  and k dependent.
- Rare weather (Squall 1 h, Smoke 15, Fog 201) cannot form rules at 1% support.
- Uneven year coverage means patterns are dominated by 2013, 2016, 2017.
- MLflow: one run `kmeans k=6 + association rules` in `task2-unsupervised-models`
  (silhouette 0.393, 44 rules).

## Task 3: deep learning with explainability (`lstm_model.py`)

**Choice and rationale (for the report)**
- Chose option 2 (LSTM) because the data is an hourly sequence and the brief
  recommends it. A plain NN would add little over the Task 1 Random Forest; no
  image data exists for a CNN.
- LSTM is hard to explain, so SHAP `TreeExplainer` is applied to a Random Forest
  trained on the same targets (allowed by the brief). SHAP describes the
  Random Forest, not the LSTM; similar accuracy does not establish that the
  models use the same signals.

**Method**
- Task: predict next-hour `traffic_volume` from the previous 24 hours of volume,
  temperature, rain, snow, clouds, plus the calendar of the target hour (hour and
  day of week sine/cosine, weekend, holiday). Real volume target, not `high_risk`.
- Data has 2,588 gaps over 1 h; windows are built only inside consecutive hourly
  runs: 28,871 windows from 173 runs (11,704 rows lack 24 h of history).
  Split by target time as in Task 1 (test from 2017-10-26 18:00): 19,067 train,
  2,118 validation (last 10% of pre-test windows, early stopping), 7,686 test.
  The remaining 429 test rows are excluded because they lack a full history.
  The forest trains on all 21,185 pre-test windows.
- Restore the original weather readings before fitting monthly imputation and
  standardization on rows strictly before the first validation target. This
  avoids the full-dataset imputation performed upstream. Ten temperature and one
  rainfall outliers were re-imputed.
- LSTM(64) -> dropout 0.2 -> concat target-hour calendar -> Dense(32) -> 1.
  Adam 1e-3, batch 128, patience 5; stopped after 60 epochs (best validation
  loss 0.0129); inverse-scale and clip predictions to zero.
- RF twin: 200 trees, min_samples_leaf 2, features volume_lag_1..24, last-hour
  weather, hour, day_of_week, is_weekend, holiday_flag. The LSTM uses 24 hours
  of weather, not just the last hour. SHAP on 500 random test windows. Extra RF
  without volume lags to show calendar/weather effects.

**Results (test, next hour)**

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Persistence (previous hour) | 585.3 | 811.8 | 0.830 |
| Same hour yesterday | 571.6 | 1030.8 | 0.726 |
| Random Forest (lag features) | 147.3 | 234.1 | 0.986 |
| Random Forest (calendar + weather) | 271.3 | 448.0 | 0.948 |
| LSTM | 155.2 | 235.2 | 0.986 |

**Observations**
- LSTM reduces MAE about 73% vs persistence; the RF is slightly better in this
  single evaluation. The models use different inputs and training-window counts.
- SHAP feature and family shares describe this fitted RF on the sampled test
  windows. They are not causal effects, and cannot be transferred to the LSTM.
- Mean absolute SHAP shares: previous-hour volume 60.7%, calendar 21.5%,
  older volume lags 17.5%, numerical weather 0.3%. In the separate no-volume-lag
  RF, hour contributes 71.3% of total absolute SHAP.
- `snow_1h_lag_1` does not vary in the forest's training windows, and
  `weather_main=Snow` is not a forest feature. A zero attribution does not show
  that snow has no effect on traffic.

**Caveats / bias notes**
- The forest uses last-hour weather; the LSTM uses all 24 observed weather
  hours. Neither uses weather forecasts. Both need recent traffic readings.
- Gaps reduce usable data; uneven year coverage as in Task 2. Single split,
  no hyperparameter tuning, one random seed; no evaluation immediately after
  outages. Target-hour holiday data must be available at prediction time.
- MLflow: five runs in `task3-deep-learning` (LSTM run holds plots, CSVs and
  `lstm_model.keras`; forest runs hold fitted models). Saved preprocessing metadata
  provides scalers, feature order and imputation medians. Logs:
  `logs/lstm_model.log`. The notebook reads saved artifacts without overwriting.

## Task 4: advanced technique, MLflow audit and model registry (`mlflow_registry.py`)

**Choice and rationale (for the report)**
- Chose MLflow (recommended in the brief): already used in Tasks 1-3, links to the
  MLOps/deployment tasks, no new infrastructure. Quantisation, a GAN and
  self-supervised anomaly detection were not needed to meet the brief.
- Went beyond tracking: audit of all runs, Model Registry, promotion gate.

**Method**
- 17 runs from 3 experiments (12 distinct models) collected; latest run per model
  ranked on test MAE (regression) or PR AUC (classifier); baselines and the
  calendar-only forest excluded.
- Registered `traffic-volume-regressor`, `high-risk-classifier`,
  `next-hour-forecaster-rf` and `next-hour-forecaster-lstm` (each version v1).
  Task 3 models were plain artifacts, so the forest was re-logged as a sklearn
  model and the LSTM as a pyfunc model (`LstmForecaster` in `lstm_model.py`) with
  tensor signature (sequence 24x5, calendar 6).
- Gate: load from the registry, recompute the test metric, compare with the logged
  value. All four matched to 4 decimals (MAE 238.87, PR AUC 0.9545, MAE 147.32,
  MAE 155.16). Aliases: champions for the Task 1 models and Task 3 forest,
  challenger for the LSTM. A rerun created no new versions.

**Observations**
- Task 3 LSTM reruns differ by 2.1% MAE (155.16 vs 158.38; 60 vs 40 epochs). The
  forests differ by under 0.01%. Both runs share a git commit, but the weather
  imputation changed between them, so the difference is confounded with the data
  preparation change; one seed is not enough to rank LSTM vs forest.
- Forest (147.3) beats LSTM (155.2) by 8 vehicles/h; the LSTM's own spread across
  runs is about 3 vehicles/h.
- Outputs: `outputs/mlflow_run_comparison.csv/.png`, `mlflow_rerun_variability.csv`,
  `mlflow_registry_summary.csv`; logs in `logs/mlflow_registry.log`; run in the
  experiment `task4-mlflow-registry`.

**Limitations (for the report)**
- Local SQLite store, no authentication or shared server; `mlartifacts/` is
  git-ignored, so registered models need Tasks 1-3 re-run on a fresh checkout.
- No data versioning; selection and gate use the test set (optimistic scores; the gate
  checks reproducibility, not quality); the proxy-label classifier is still
  circular; aliases are script-driven, not an approval process; pickled models only
  from trusted stores; no monitoring or retraining.

## Outstanding for the Part 3 deliverable

- [x] MLflow tracking of the Task 1 runs (`mlflow.db`, experiment `task1-supervised-models`; model artifacts in git-ignored `mlartifacts/`)
- [x] Notebook for Task 1 (`notebooks/Task1_supervised_models.ipynb`, executed)
- [x] Task 2 unsupervised models: script, notebook, MLflow run, observations above
- [x] Task 3 LSTM + SHAP: script, notebook, MLflow runs, plots, observations above
- [x] Task 4 advanced technique: MLflow audit, model registry and promotion gate (`mlflow_registry.py`)
- [x] Task 4 notebook: read-only run and registry review (`notebooks/Task4_mlflow_registry.ipynb`)
- [ ] Deployment simulation
- [ ] Recommendation engine
- [ ] Bias and fairness report (subgroup metrics)
- [ ] Final capstone report covering Parts 1-3
- [ ] README: project structure, run steps, proxy statement (proxy statement present)
