# Task 7: Bias, fairness, governance and sustainability

## Scope and conclusion

This report draws on the project's Part 1 descriptive analysis, Part 2 data
pipeline and visualizations, and Part 3 models, recommendations and deployment
simulation. The data describes hourly traffic and weather on one interstate
corridor from October 2012 to September 2018. It contains no accident outcomes
or demographic fields. Consequently, the project cannot establish accident
prediction performance or fairness across demographic groups. Differences
between traffic conditions or periods are useful checks, but are not evidence
of demographic discrimination.

The results support exploration of historical traffic patterns, not autonomous
real-world traffic or safety decisions. The strongest evidence of uneven errors
comes from the historical monthly monitoring replay and the recommendation
profile's weather breakdown; neither demonstrates the effects on actual people.

## Sampling and coverage

- The source has 48,204 records; the Part 2 pipeline retains 40,575 hourly rows
  after removing duplicates and keeping the first row for each timestamp.
  Repeated hours can have different weather descriptions, so the selected
  weather can depend on source order. The pipeline replaces invalid
  temperatures and extreme rainfall values with monthly medians and retains the
  unusually low traffic recorded on 23-24 July 2016, with a warning. These
  choices affect what later analyses see; the cause of the low readings is not
  established.
- Coverage is uneven: only 2013, 2016 and 2017 are near-complete, and 2018 is
  partial. Part 1 noted gaps in 2014 and 2015, so raw yearly totals are not
  directly comparable. Observed-hour averages and patterns reflect the hours
  actually recorded; neither missing periods nor other corridors are represented
  by evidence in this project.
- Weather categories are very uneven: the cleaned data contains one Squall
  hour, 15 Smoke hours and 201 Fog hours. Estimates for rare conditions are
  uncertain; the 1% minimum support for Part 3 association rules also means
  these individual categories cannot generate rules. The recorded `rain_1h` is
  zero throughout 2017 and non-zero in roughly 6% of 2018 hours. This may reflect
  a recording gap; a drift alert alone does not establish its cause.
- The sequence models require 24 consecutive prior hours. Of 8,115 test hours,
  7,686 meet that requirement and 429 are excluded. Their scores therefore do
  not measure performance immediately after gaps. The same-hour and next-hour
  models address different prediction tasks.
- Holiday information is limited by the source: the holiday flag also includes
  State Fair, and a holiday without a midnight record is not detected by the
  project workflow. The recommendation backtest has only 10 fully observed
  holiday test days, and some holiday-by-weather cells have only 1-8 hours.

## Proxy labels and interpretation

There is **no real accident dataset** in this project. The Part 3 `high_risk`
classification target is a proxy: an hour is positive if its traffic volume
is in the High or Severe quartile category and its weather is Thunderstorm,
Snow, Squall, Fog, Mist, Haze or Smoke. These weather groupings are project
choices, not verified accident-risk categories. The label marks 2,964 of 40,575
hours (7.30%); the positive share is 6.8% in training and 9.4% in test.

This definition can omit real accidents in other conditions and flag hours
without accidents. No accident data is available to measure either error.
Weather is also an input to the classifier, so it can relearn part of the rule
used to create its own target. The Random Forest's test F1 of 0.930 and PR AUC
of 0.955 describe agreement with that proxy, **not** accident prediction.
Quartile cutoffs were computed from all rows, including the test period, which
also introduces test-period information into label definition.

The Part 3 quartile congestion categories are relative to this dataset, whereas
Part 2 uses fixed thresholds of 4,500 and 5,500 vehicles/hour. They are not
interchangeable. Clustering uses traffic volume as an input and congestion
categories are derived from volume, so some alignment between them is built in.
Associations between weather, time and traffic are not causal findings.
Neither the real-volume forecasts nor the travel recommendations measure
accident risk; a lower-volume window has not been shown to be safer.

## Uneven errors and evidence gaps

The following comparisons concern distinct outputs and should not be read as
one model's subgroup audit:

| Evaluation | Observed differences | Interpretation |
|---|---|---|
| Task 6 same-hour Random Forest, historical test replay | Full-test MAE was 236.75 vehicles/hour. Monthly MAE was 347.14 in December 2017, 311.70 in January 2018 and 308.61 in April 2018, versus 162.85 in June 2018. | Average error varies across recorded months. The first replay month has only 126 hours; comparisons are descriptive, not proof of cause or future performance. |
| Task 5 historical recommendation profile, hourly test comparison | MAE with observed weather was 489.3 vehicles/hour for severe weather (856 hours), versus 303.2 overall (8,115 hours). Without weather it was 505.1 and 306.2 respectively. | The profile was less accurate in the recorded severe-weather group. This is not a forecast-weather or registered-model evaluation. |
| Task 5 one-hour window backtest, fully observed test days | The recommended window was below that day's daytime average on 228/229 weekdays, 97/97 weekend days and 7/10 holidays. | The holiday result has a very small sample. Excluding incomplete days limits what the backtest says about those days. Final recommendations use the full history, including this backtest period. |

Task 6's replay flagged all 12 months for feature drift; three also exceeded
the heuristic monthly error threshold. Whole-period comparisons can flag
seasonal differences, while even same-month comparisons still alerted each
month. Thresholds and the worse-of-two-references rule are heuristic, the MAE
reference includes the months being evaluated, and the replay is not live
monitoring. Alerts warrant investigation rather than a conclusion that a model
has failed or a specific group has been harmed.

Overall model scores can conceal differences by hour, day type, holiday, weather
or missing-history status. The project has not established a comprehensive
subgroup error audit for its classifiers and forecasts, including false-positive
and false-negative rates across these conditions. Sparse groups make such
estimates especially uncertain. The Random Forest SHAP analysis describes a
sample of its own next-hour predictions, not the LSTM; the forest's snow input
does not vary in its training windows, so its attribution cannot establish
snow's effect. The historical recommender also does not describe day-to-day
window variability or establish that a suggested time is feasible for every
traveller.

## Governance before real-world use

The project already tracks experiments and model versions with MLflow. Its
registry gate reloads an artifact and checks that its test metric is
reproduced; Task 6 uses a specified promotion rule and exposes a local API and
monitoring dashboard. These are useful reproducibility and simulation steps,
not independent approval or proof of quality. Model selection and the gate use
the test set, making the reported selection results potentially optimistic;
Task 6's v3 improvement over v1 is less than 1% on one split. The API and
monitoring use historical data and do not provide a live operational system.

Before any real-world decision-making, the responsible operator should:

1. Define the intended decision, accountable human reviewer and limits of use;
   retain human authority to reject, override or suspend the system. Do not
   present the proxy classifier as accident prediction or a low-volume
   recommendation as safety advice.
2. Obtain relevant, independently evaluated data for the intended location and
   period. If accident risk is the intended task, obtain and validate actual
   accident outcomes rather than relying on this proxy.
3. Use a separate selection and final evaluation strategy, and examine error
   counts and rates by time, weather, holiday and data availability, reporting
   sample sizes and uncertainty. Investigate sparse groups and missing data
   rather than treating aggregate metrics as fairness evidence. Any assessment
   of impacts on people would require appropriate evidence not present here.
4. Review data quality, availability of required inputs and the effects of
   gaps; verify the system in a controlled trial before relying on its outputs.
   The next-hour models need fresh consecutive observations, while the
   recommender is a historical profile and the API predicts same-hour volume.
5. Establish a documented approval, monitoring and incident-response process,
   with responsible owners and a way to suspend or roll back a model. Validate
   alert thresholds against operational evidence, investigate drift and errors,
   and version both data and models. Protect the service and load serialized
   model artifacts only from a trusted store: the existing local API and MLflow
   store do not provide a production authentication or approval workflow.

## Sustainability and resource trade-offs

The project has not measured energy use, hardware power, emissions or a
production workload, so it cannot quantify an environmental footprint or claim
that one model is greener. The available resource observations are:

- The documented Task 3 training plus SHAP workflow takes about seven minutes
  on CPU; its saved LSTM run used 60 epochs. The next-hour Random Forest achieved
  MAE 147.3 versus the LSTM's 155.2 on the same eligible test targets, but they
  use different weather inputs and training-window counts. One evaluation and
  confounded rerun differences do not establish a general winner or a measured
  energy advantage.
- The Task 6 tuned forest improved same-hour test MAE from 238.87 to 236.75
  compared with v1, after a 4-by-4 grid with three time-ordered training folds.
  This small single-split gain should be weighed against the additional fitting
  work; its energy cost was not measured.
- Model artifacts can occupy several hundred MB across Tasks 1-4; repeated
  Task 1 runs add runs and artifacts. Tracking supports reuse and audit, but the
  git-ignored artifact store must be regenerated on a fresh checkout. The
  recommendation engine uses historical profiles and does not load a model.

For future use, compare accuracy and subgroup performance alongside measured
training, storage and serving resources; reuse validated artifacts where
appropriate and avoid unnecessary retraining. The project provides no measured
evidence that its recommendations reduce travel time, fuel use or emissions.

## Project evidence

- [Part 1 data analytics report](../Part-1/Data%20Analytics%20Insights%20Report.pdf)
  and [Part 2 methodology report](../Part-2/Methodology_and_Findings_Report.pdf).
- [Part 2 pipeline and visualization documentation](../Part-2/README.md).
- [Part 3 methods and caveats](README.md) and [task observations](OBSERVATIONS.md).
- Saved results: [model metrics](outputs/model_metrics.csv),
  [monthly monitoring](outputs/monitoring_report.csv),
  [weather comparison](outputs/recommendation_weather_check.csv),
  [recommendation backtest](outputs/recommendation_backtest.csv),
  [model versions](outputs/model_versions.csv) and
  [sequence run details](outputs/lstm_run_info.json).
