# Smart-City Traffic Capstone

Further analysis of the Metro Interstate Traffic Volume dataset for a
smart-city mobility audience. Part 1 contains the completed database,
descriptive-analysis, report, and Power BI dashboard deliverables. Part 2
contains Python data-cleaning, feature-engineering, visualization, and
command-line analytics tools for hourly traffic and weather analysis. Part 3
contains supervised, unsupervised, and sequence models, explainability analysis,
MLflow experiment tracking and model registration, and travel-timing recommendations.

## Repository structure

```text
.
├── Metro_Interstate_Traffic_Volume.csv       # Raw source data
├── Part-1/
│   ├── Data Analytics Insights Report.pdf / .docx
│   ├── Task 1 - sql/
│   │   ├── Capstone Metro Interstate Traffic DB.db / .sqbpro
│   │   └── Capstone Task-1.sql
│   ├── Task 2 and 3 - statistics, probability and correlation/
│   │   ├── Task 2 and 3 - Descriptive Statistics and Correlation.pbix
│   │   └── Task 2 Notes / Task 3 Notes (.pdf / .docx)
│   └── Task 4 - powerbi/
│       ├── Traffic_Dashboard.pbix
│       └── Task 4 Notes.pdf / .docx
├── Part-2/
│   ├── pipeline.py                           # Raw-to-cleaned pipeline
│   ├── feature_engineering.py                # ML feature preparation
│   ├── visualize.py                          # Task 3 traffic visualizations
│   ├── traffic_app.py                        # Command-line traffic analytics
│   ├── Methodology_and_Findings_Report.pdf   # Part 2 methodology/results report
│   ├── data/processed/                       # Generated CSV datasets
│   ├── outputs/visualizations/               # Generated Task 3 figures
│   └── logs/                                 # Pipeline log files
└── Part-3/
    ├── prepare_labels.py                     # Proxy congestion/high-risk labels
    ├── train_models.py                       # Task 1 regression/classification models + MLflow logging
    ├── unsupervised_models.py                # Task 2 K-means clustering + association rules
    ├── lstm_model.py                         # Task 3 next-hour LSTM forecasting + SHAP comparison
    ├── mlflow_registry.py                    # Task 4 MLflow run audit, model registry, promotion gate
    ├── recommendation_engine.py              # Task 5 historical travel-timing recommendations
    ├── notebooks/                            # Task 1-5 notebooks
    ├── README.md                             # Part 3 guide
    ├── OBSERVATIONS.md                       # Running findings log for the final reports
    ├── mlflow.db                             # MLflow tracking database
    ├── mlartifacts/                          # MLflow model artifacts (git-ignored)
    ├── data/processed/                       # Generated labelled CSV
    ├── outputs/                              # Generated metrics, predictions, and plots
    └── logs/                                 # Part 3 log files
```

Generated datasets and logs are created when the scripts run and are not
required to be edited manually.

## Tools and technologies

- Python 3.10 or later
- pandas and NumPy for CSV processing and feature engineering
- Matplotlib for traffic visualizations
- scikit-learn for the Part 3 supervised models (`pip install scikit-learn`)
- MLflow for Part 3 experiment tracking (`pip install mlflow`); Jupyter for the notebook
- SHAP and TensorFlow for the Part 3 Task 3 LSTM and explainability workflow
  (`python -m pip install shap tensorflow`; see `Part-3/README.md`)
- Python's standard-library `logging` module
- SQLite-compatible database tools for the Part 1 database and SQL
- Microsoft Power BI Desktop for the `.pbix` dashboard

The Python scripts have been run with Python 3.12, pandas 3.0.5, and NumPy
2.5.2.

## Set up and run the Python code

Run commands from the repository root in PowerShell. Create and activate a
virtual environment, then install the Python dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install pandas numpy matplotlib scikit-learn mlflow jupyter
```

Run the Task 1 cleaning pipeline first:

```powershell
python Part-2/pipeline.py
```

It reads `Metro_Interstate_Traffic_Volume.csv` and writes
`Part-2/data/processed/Metro_Interstate_Traffic_Volume_cleaned.csv`.

Then create the Task 2 feature dataset:

```powershell
python Part-2/feature_engineering.py
```

It reads the cleaned CSV and writes
`Part-2/data/processed/Metro_Interstate_Traffic_Volume_features.csv`.
Both scripts accept `--input`, `--output`, `--log-file`, and `--log-level`
options. Run either command with `--help` to see its defaults.

Generate the Task 3 traffic visualizations from the cleaned dataset:

```powershell
python Part-2/visualize.py
```

The script saves four PNG figures in `Part-2/outputs/visualizations/` and
logs to `Part-2/logs/visualizations.log`. It accepts `--input`,
`--output-dir`, `--log-file`, and `--log-level` options. See
[`Part-2/README.md`](Part-2/README.md) for the figures and their
interpretations.

Run the Task 4 command-line analytics app after the cleaning pipeline:

```powershell
python Part-2/traffic_app.py --help
```

The app reads the cleaned CSV once per run and supports five commands:

1. `date` - Show traffic for a specific date, and optionally a specific hour.
   Example: `python Part-2/traffic_app.py date 2017-07-10 --hour 8`
2. `peak` - Identify the top N high-traffic periods (default N=5).
   Example: `python Part-2/traffic_app.py peak --top 5`
3. `compare` - Compare average weekday vs weekend traffic, printed as a small
   table. Example: `python Part-2/traffic_app.py compare`
4. `recommend` - Recommend the lowest-traffic periods to travel (the 5
   quietest hours of the day).
   Example: `python Part-2/traffic_app.py recommend`
5. `weather` - Show weather and traffic info for a given date.
   Example: `python Part-2/traffic_app.py weather 2017-07-10`

Dates must be written as `YYYY-MM-DD` and fall within the dataset's range,
2012-10-02 to 2018-09-30; hours must be integers from 0 to 23. For example,
`date 2023-05-14 --hour 8` is correctly formatted but outside the range, so the
app logs an error and exits with a nonzero status. If you run the commands from
inside the `Part-2/` folder, use `python traffic_app.py ...` instead.

See [`Part-2/README.md`](Part-2/README.md) for sample output, validation
behavior, and test commands. The [Part 2 methodology and findings report](Part-2/Methodology_and_Findings_Report.pdf)
summarizes the workflow, measured patterns, and limitations.

Create the Part 3 proxy labels after feature engineering:

```powershell
python Part-3/prepare_labels.py
```

It writes `Part-3/data/processed/Metro_Interstate_Traffic_Volume_ml.csv` and
logs to `Part-3/logs/prepare_labels.log`. See [`Part-3/README.md`](Part-3/README.md)
for the label definitions and cautions.

## Run the mini-application

The project's interactive dashboard is
`Part-1/Task 4 - powerbi/Traffic_Dashboard.pbix`. Open that file in Microsoft Power BI Desktop
to view and interact with the dashboard. If Power BI prompts for a data-source
connection when refreshing, configure the source for your local environment.
The Python traffic analytics app is a separate command-line tool, not a web
application.

## Machine-learning models

Part 2 prepares features and targets; its CSV is not a model artifact or a
prediction service. Models are trained in Part 3 (below).

The feature dataset includes hour, weekday, weekend, cyclical hour, encoded
weather, rain/snow indicators, and z-score features for temperature and
rainfall. The `traffic_category` target is defined as:

- **Low:** traffic volume < 4,500 vehicles/hour
- **Medium:** 4,500–5,500 vehicles/hour, inclusive
- **High:** traffic volume > 5,500 vehicles/hour

The `is_congested` target is 1 above 5,500 vehicles/hour and 0 otherwise.
These cutoffs reuse the established project definitions; they are not fitted
model thresholds.

Part 3 adds two further labels in `Part-3/prepare_labels.py`: a four-class
`congestion_category` based on quartiles of `traffic_volume`, and a **proxy**
`high_risk` label (High/Severe congestion with severe or low-visibility
weather). No accident data exists, so `high_risk` is not an accident
prediction.

Part 3 Task 1 (`Part-3/train_models.py`) trains Linear Regression and Random
Forest to predict `traffic_volume`, and Logistic Regression and Random Forest
to predict `high_risk`, on a shared feature set with a time-based 80/20 split.
Test results: regression MAE 801 / R² 0.735 (linear) vs MAE 239 / R² 0.959
(Random Forest); classification F1 0.814 (logistic) vs 0.930 (Random Forest).
Classification scores are inflated because `high_risk` is defined from
`weather_main`, which is also a feature. See [Part-3/README.md](Part-3/README.md)
for the full metrics and caveats. Runs are tracked with MLflow
(`Part-3/mlflow.db`) and reviewed in `Part-3/notebooks/`. No model is deployed.

Part 3 Task 2 (`Part-3/unsupervised_models.py`) clusters hours with K-means
(k = 6: weekend/weekday night, weekend daytime, weekday afternoon/evening,
weekday morning-midday, and a severe-weather group) and mines association rules
for the quartile congestion level. Time of day and day type drive congestion;
weather adds little. It describes traffic conditions, not accidents.

Part 3 Task 3 (`Part-3/lstm_model.py`) forecasts next-hour traffic volume from
24 hours of recent readings. On the time-based test set the LSTM has MAE 155
(R² 0.986); the Random Forest comparison model has MAE 147. The LSTM uses
consecutive hourly histories, so hours without a full 24-hour history are not
evaluated. SHAP explains the Random Forest comparison, not the LSTM itself.
See [Part-3/README.md](Part-3/README.md) and
[`Part-3/notebooks/Task3_deep_learning.ipynb`](Part-3/notebooks/Task3_deep_learning.ipynb).

Part 3 Task 4 (`Part-3/mlflow_registry.py`) applies MLflow beyond tracking: it
audits all Task 1-3 runs, registers the best models in the MLflow Model Registry
and sets `champion`/`challenger` aliases only after each registered model
reproduces its logged test metric. See [Part-3/README.md](Part-3/README.md).

Part 3 Task 5 uses the prepared historical traffic data to suggest travel
windows by day type and weather on the single corridor. It does not choose routes,
provide a live forecast, or load the registered models. After creating the Part 3
labelled CSV, run:

```powershell
python Part-3/recommendation_engine.py
python Part-3/recommendation_engine.py --day-type weekday --weather "rain/drizzle" --duration 2
```

See [Part-3/README.md](Part-3/README.md) for outputs, backtesting, assumptions,
and the [Task 5 notebook](Part-3/notebooks/Task5_recommendation_system.ipynb).

## Logging

Each Python script configures logging in its command-line entry point. Messages
are written to the console and to a file. The data pipeline, feature, and
visualization scripts include timestamp, severity, module name, and message.
The traffic CLI uses timestamp, severity, and message as required for its
user-facing command log. The default level is INFO.

- Cleaning pipeline: `Part-2/logs/pipeline.log`
- Feature engineering: `Part-2/logs/feature_engineering.log`
- Visualizations: `Part-2/logs/visualizations.log`
- Traffic CLI: `Part-2/logs/traffic_app.log`
- Part 3 proxy labels: `Part-3/logs/prepare_labels.log`
- Part 3 model training: `Part-3/logs/train_models.log`
- Part 3 unsupervised analysis: `Part-3/logs/unsupervised_models.log`
- Part 3 MLflow registry (Task 4): `Part-3/logs/mlflow_registry.log`
- Part 3 recommendations (Task 5): `Part-3/logs/recommendation_engine.log`

Log levels are used as follows: DEBUG records fine-grained troubleshooting
values; INFO records expected milestones, including saved file paths; WARNING
records unexpected but recoverable events; ERROR records failures that prevent
a step from completing as planned.

For example:

```powershell
python Part-2/feature_engineering.py --log-level DEBUG
```

Errors are logged at ERROR and cause a nonzero exit status. Logs are appended
on subsequent runs.

## Reproduce or review the analysis

1. Start with the included raw CSV and run the cleaning and feature-engineering
   commands above, in that order.
2. The cleaning pipeline keeps the first row for each timestamp and applies
   month-specific median imputation to invalid temperature and rainfall values.
   The feature script preserves its input and creates a separate feature CSV.
3. Review the Part 1 results in `Part-1/Data Analytics Insights Report.pdf`
   and its editable `.docx`; review the SQL in
   `Part-1/Task 1 - sql/Capstone Task-1.sql` and the included SQLite database
   in the same folder.
4. Open `Part-1/Task 4 - powerbi/Traffic_Dashboard.pbix` in Power BI Desktop
   to review the dashboard. The Power BI file may require a local data-source
   connection before refreshing.

The generated cleaned and feature datasets each contain 40,575 hourly rows.
Run the scripts to reproduce those CSVs and their log records; Part 1's
historical report and dashboard are supplied as reviewable deliverables, not
automatically regenerated by the Python scripts.

## Assumptions and limitations

- The source has 48,204 records and day-first `date_time` values. Collection
  coverage is uneven across years, so raw yearly totals can be misleading.
- Multiple source rows can have the same timestamp. The pipeline removes exact
  duplicate rows and keeps the first row for each remaining duplicate
  timestamp; the chosen weather label may therefore depend on source order.
- Zero-Kelvin temperatures and rainfall above 9,000 mm are treated as invalid
  and imputed using the median for the corresponding calendar month. The known
  low-traffic outage on 23–24 July 2016 is preserved and logged as a warning.
- Weather categories with very few examples are represented in the encoded
  features but provide limited evidence for general conclusions.
- Observed weather/traffic relationships are associations, not causal effects.
- The Part 3 models use one time-based split without tuning or
  cross-validation; none is validated for deployment. The test period includes
  partially covered 2018 data.
