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

## Outstanding for the Part 3 deliverable

- [x] MLflow tracking of the Task 1 runs (`mlflow.db`, experiment `task1-supervised-models`; model artifacts in git-ignored `mlartifacts/`)
- [x] Notebook for Task 1 (`notebooks/Task1_supervised_models.ipynb`, executed)
- [ ] Deployment simulation
- [ ] Recommendation engine
- [ ] Bias and fairness report (subgroup metrics)
- [ ] Final capstone report covering Parts 1-3
- [ ] README: project structure, run steps, proxy statement (proxy statement present)
