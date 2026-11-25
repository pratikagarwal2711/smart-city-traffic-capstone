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

## Task 5: travel-timing recommendation system (`recommendation_engine.py`)

**Design (for the report)**
- One corridor, so the system recommends *when* to travel, not which route.
- Expected volume = historical mean per day type (Weekday/Weekend/Holiday) x weather
  group (Task 2 groups or Any) x hour, with sparse cells shrunk to the all-weather
  mean (10 pseudo-hours, a heuristic). Best window = lowest mean inside 06:00-19:00
  (assumption, configurable), plus next-best windows, the peak to avoid, and an
  evening option outside the requested range (up to 22:00) when at least 10% lower.
  Output includes plain-language text per scenario (54 scenarios).

**Observations**
- Weekday: 10:00-11:00 AM, about 4,453 vehicles/h, 30% below the 4-5 PM peak
  (6,325); evening 9-10 PM is lighter still (2,720). This is "High" on the Part 3
  quartile scale: the quietest daytime option is still busy. Weekend: 6-7 AM (1,103)
  at the edge of the range; midday is the busiest. Holiday: 6-7 AM (2,758).
- For the one-hour scenarios, weather rarely changes the best window: for Weekday and Weekend all weather
  groups give the same window except Weekday + Severe weather (6-7 PM, 4,213, a
  near tie with 10 AM). Weather-adjusted estimates differ from all-weather means by under 8% on those day types;
  Holiday x weather estimates move by about 9-12% (Rain/Drizzle, Low visibility and
  Severe weather rest on 1-8 hours and are flagged `limited_evidence`).
- Backtest (fit on first 80%, test after 2017-10-26, fully observed days): weekday
  1-hour recommended window 4,473 vs peak 6,415 and daytime average 5,288, below
  the daytime average on 228/229 days; weekend 1,123 vs 4,452 peak (97 days, 100%);
  holiday 2,476 vs 4,172 (10 days, 70% below average).
- Adding observed weather to the hourly profile cuts test MAE only from 306.2 to 303.2
  (severe weather 505.1 to 489.3). Consistent with Tasks 1-3: time of day and day
  type dominate; weather adds little.
- Outputs: `outputs/recommendation_profile.csv/.png`, `recommendation_windows.csv`,
  `recommendation_backtest.csv`, `recommendation_weather_check.csv`,
  `recommendation_run_info.json`; log `logs/recommendation_engine.log`; notebook
  `notebooks/Task5_recommendation_system.ipynb`; tests `test_recommendation_engine.py`.

**Limitations (for the report)**
- Historical averages, not a live forecast; no incidents, events or weather feed;
  user supplies the weather. Volume is not safety (no accident data), so bad-weather
  advice says allow extra time.
- 06:00-19:00 range is an assumption; the weekend answer sits on its edge.
- Holiday cells mix different holidays, ~10 test days, thin weather samples; holiday
  dates missing a 00:00 row are not detected, and the source also flags State Fair.
  Window variability across days is not reported. The recommender uses historical
  data and Task 2 weather group definitions; it does not load the registered models.
  The Task 1 random forest achieved test MAE 238.9 versus 622.0 for its mean-by-hour
  baseline; this does not establish that the models and recommendation profile
  produce equivalent predictions.
- The 10 pseudo-hours and thresholds for sparse evidence (10 hours), evening
  alternatives (10%) and weather notes (5%) are heuristics, not model-derived.
  A post-hoc sensitivity check on the same held-out split gave hourly MAE 305.4
  without shrinkage (using the all-weather mean for missing cells), 303.2 at 10,
  and 302.9 at 20. This does not independently validate or tune the choice.
  Weather comparisons are observational, not causal. Final recommendations use
  all data, including the backtest period.

## Task 6: MLOps and deployment simulation (`model_versions.py`, `serve_api.py`, `monitoring.py`)

**Design (for the report)**
- Model: `traffic-volume-regressor` (same-hour volume, Task 1 features and 80/20 time split). It is
  real traffic volume, not the proxy label. Reused the Task 1 feature builder, the MLflow store and the
  Task 4 promotion gate (reload from registry, recompute test MAE, tolerance 0.05).
- 6.1/6.2: three versions of one registered model: v1 Random Forest (Task 1), v2 Gradient Boosting
  (settings chosen, not tuned), v3 Random Forest tuned by a 4x4 grid with 3-fold time-ordered CV on the
  training period only. Runs logged in `task6-model-versions`. Rule agreed with the user: `champion` moves
  only to a version that beats v1 test MAE and passes the gate.
- 6.3: FastAPI serving `@champion` (`/health`, `/model`, `/predict`, `/monitor/status`).
- 6.4/6.5: monthly replay of the test period against the training period; PSI for temp_c, rain_1h, snow_1h,
  clouds_all, hour and the weather_main mix against (a) the whole training period and (b) the same calendar
  month (the user chose "ALERT if either alerts"); error drift = month MAE vs the model's full test MAE.
  Thresholds are heuristics (PSI warning 0.1, alert 0.2; error alert above 1.25x), logged at each run. No
  minimum hours per month (user decision). Overall status = latest month. Static HTML dashboard + JSON.
  Exit code 2 = ALERT (consistent with Task 4), 1 = error.

**Observations**
- v1 238.87, v2 239.79, v3 236.75 test MAE (RMSE 400.1/399.7/399.2; R2 0.959 for all). All pass the gate;
  v3 becomes champion by 2.1 vehicles/h (0.9%). It differs from v1 only in `max_features` (0.5 vs 1.0), so
  this is not evidence of a real improvement (single split, test set used for the choice).
- Train MAE 184.0 / 211.5 / 193.7: boosting generalises with the smallest train-test gap but not a better
  test score.
- Monitoring replay: overall **ALERT**, 12 of 12 months ALERT. Error drift in 3 months (Dec 2017 1.47x,
  Jan 2018 1.32x, Apr 2018 1.30x of 236.75). Whole-period PSI flags temp_c in all 12 months (seasonality).
  Same-month PSI still alerts every month: weather_main 9 months, clouds_all 8, rain_1h 7, temp_c 7;
  hour and snow_1h never. Oct 2017 (126 h) has the largest PSI.
- `rain_1h` is exactly 0 for all of 2017 and non-zero in about 6% of 2018 hours (about 4-15% in 2013-2016),
  so the rain PSI reflects a data recording gap, not necessarily weather.
- API check: Thursday 17:00 cloudy -> 6,098 (High); Saturday 03:00 clear -> 448 (Low). Unknown weather
  returns 422. The API feature builder matches `train_models.build_features` (unit test).
- Outputs: `outputs/model_versions.csv/.png`, `model_versions_rf_tuning.csv`, `monitoring_report.csv`,
  `monitoring_feature_drift.csv`, `monitoring_weather_shares.csv`, `monitoring_status.json`,
  `monitoring_dashboard.html`; logs `logs/model_versions.log`, `monitoring.log`, `serve_api.log`; notebook
  `notebooks/Task6_mlops_deployment.ipynb`; tests `test_task6.py`.

**Limitations (for the report)**
- Simulation only: historical replay, no live data, retraining, paging or approval workflow; API has no
  authentication or scaling; models are pickles (trusted store only).
- Thresholds are conventions, not validated; the worse-of-two-references rule is deliberately sensitive, so
  every month alerts. Single-month PSI (about 700 hours; 126 for the first month) is noisy. The reference MAE
  comes from the same test period.
- Re-running `mlflow_registry.py` (Task 4) resets `champion` to v1; run Task 6 afterwards. `mlartifacts/` is
  git-ignored, so v2/v3 need `model_versions.py` on a fresh checkout. Task 4 ignores `task6-*` experiments.

## Task 7: responsible and sustainable AI

- Report: [BIAS_AND_FAIRNESS_REPORT.md](BIAS_AND_FAIRNESS_REPORT.md), drawing on
  Parts 1-3 and the saved outputs. It distinguishes condition/time comparisons
  from demographic fairness, which the available data cannot establish.
  A [PDF version](BIAS_AND_FAIRNESS_REPORT.pdf) is also available.
- Recorded evidence includes differing monthly model MAE, higher severe-weather
  recommendation-profile MAE, and limited holiday backtest coverage. A
  comprehensive subgroup error audit has not been completed.
- The proxy is not an accident outcome; governance recommendations are future
  safeguards, not claims that the simulation is approved for deployment.
  Training time and artifact storage are documented, but energy and emissions
  have not been measured.

## Outstanding for the Part 3 deliverable

- [x] MLflow tracking of the Task 1 runs (`mlflow.db`, experiment `task1-supervised-models`; model artifacts in git-ignored `mlartifacts/`)
- [x] Notebook for Task 1 (`notebooks/Task1_supervised_models.ipynb`, executed)
- [x] Task 2 unsupervised models: script, notebook, MLflow run, observations above
- [x] Task 3 LSTM + SHAP: script, notebook, MLflow runs, plots, observations above
- [x] Task 4 advanced technique: MLflow audit, model registry and promotion gate (`mlflow_registry.py`)
- [x] Task 4 notebook: read-only run and registry review (`notebooks/Task4_mlflow_registry.ipynb`)
- [x] Deployment simulation (Task 6: model versions, FastAPI service, drift monitoring and PASS/ALERT dashboard, notebook, tests)
- [x] Recommendation engine (`recommendation_engine.py`, notebook, backtest, tests)
- [x] Bias and fairness report (existing subgroup evidence and audit gaps)
- [ ] Final capstone report covering Parts 1-3
- [ ] README: project structure, run steps, proxy statement (proxy statement present)
