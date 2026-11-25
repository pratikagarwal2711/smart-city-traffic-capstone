"""Task 6.4 and 6.5: model monitoring simulation with a PASS / ALERT dashboard.

The champion ``traffic-volume-regressor`` is replayed over the held-out test period
(2017-10-26 to 2018-09-30) in consecutive calendar months, as if each month were
newly arriving production data. Two kinds of drift are checked per month:

* **Feature distribution drift:** the Population Stability Index (PSI) of ``temp_c``,
  ``rain_1h``, ``snow_1h``, ``clouds_all``, ``hour`` and the ``weather_main`` mix
  against the training period. Each month is compared with (a) the whole training
  period and (b) the same calendar month in the training period (seasonally matched),
  and the worse of the two drives the status.
* **Prediction error drift:** the month's MAE against the model's full test-period MAE.

A month is ``ALERT`` when any check breaches its alert threshold, otherwise ``PASS``
(PSI between the warning and alert thresholds is counted as a warning only). The overall
status is that of the latest month. Nothing is excluded for having few hours, so short
months (e.g. the first, 26-31 Oct 2017) give noisier PSI and MAE.

The thresholds are common rules of thumb, not values derived from this data, and are
logged at every run. This is a replay of historical data: there is no live traffic,
no automated retraining and no paging.
"""

import argparse
import base64
import html
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException

import mlops_common as common
import train_models as tm

LOG = logging.getLogger(__name__)

NUMERIC_FEATURES = ("temp_c", "rain_1h", "snow_1h", "clouds_all", "hour")
WEATHER_FEATURE = "weather_main"
MONITORED = NUMERIC_FEATURES + (WEATHER_FEATURE,)
MONITOR_EXPERIMENT = "task6-monitoring"
# Heuristic conventions (not derived from this data); logged at every run.
PSI_WARNING = 0.1
PSI_ALERT = 0.2
ERROR_RATIO_ALERT = 1.25
PSI_BINS = 10
PSI_EPSILON = 1e-4  # floor for empty bins so the logarithm is defined
EXIT_PASS, EXIT_FAILURE, EXIT_ALERT = 0, 1, 2

STATUS_LABELS = {"PASS": "PASS / Normal", "ALERT": "ALERT / Requires investigation"}


def _psi_from_shares(ref: np.ndarray, cur: np.ndarray) -> float:
    ref, cur = np.clip(ref, PSI_EPSILON, None), np.clip(cur, PSI_EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def numeric_psi(reference, current, bins: int = PSI_BINS) -> float:
    """PSI with bins from reference quantiles; right-closed, so zero-heavy columns split into 0 / >0."""
    reference, current = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1))[1:-1])
    ref = np.bincount(np.searchsorted(edges, reference, side="left"), minlength=len(edges) + 1) / len(reference)
    cur = np.bincount(np.searchsorted(edges, current, side="left"), minlength=len(edges) + 1) / len(current)
    return _psi_from_shares(ref, cur)


def category_shares(labels, categories) -> np.ndarray:
    labels = pd.Series(labels)
    if labels.empty:
        return np.full(len(categories), np.nan)
    return labels.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()


def categorical_psi(reference, current, categories) -> float:
    ref, cur = category_shares(reference, categories), category_shares(current, categories)
    if np.isnan(ref).any() or np.isnan(cur).any():
        return float("nan")
    return _psi_from_shares(ref, cur)


def psi_status(value: float) -> str:
    """PASS below the warning threshold, WARNING up to the alert threshold, then ALERT; NaN has no verdict."""
    if np.isnan(value):
        return "N/A"
    if value >= PSI_ALERT:
        return "ALERT"
    return "WARNING" if value >= PSI_WARNING else "PASS"


def error_status(mae: float, reference_mae: float) -> str:
    return "ALERT" if mae > ERROR_RATIO_ALERT * reference_mae else "PASS"


def weather_labels(feats: pd.DataFrame) -> pd.Series:
    cols = [c for c in feats.columns if c.startswith(tm.WEATHER_PREFIX)]
    return feats[cols].idxmax(axis=1).str[len(tm.WEATHER_PREFIX):]


def _worst(*statuses: str) -> str:
    for level in ("ALERT", "WARNING", "PASS"):
        if level in statuses:
            return level
    return "N/A"


def _max_or_nan(values) -> float:
    finite = [v for v in values if not np.isnan(v)]
    return max(finite) if finite else float("nan")


def evaluate_windows(feats: pd.DataFrame, timestamps: pd.Series, y: pd.Series, pred: np.ndarray, cut: int):
    """Compare each test-period calendar month with the training reference.

    Returns (window summary, per-feature drift table, weather share table, reference MAE).
    """
    ref_feats, cur_feats = feats.iloc[:cut], feats.iloc[cut:]
    ref_weather, cur_weather = weather_labels(ref_feats), weather_labels(cur_feats)
    ref_month = timestamps.iloc[:cut].dt.month.to_numpy()
    cur_period = timestamps.iloc[cut:].dt.to_period("M").astype(str).to_numpy()
    abs_err = np.abs(np.asarray(pred) - y.iloc[cut:].to_numpy())
    reference_mae = float(abs_err.mean())
    categories = sorted(set(ref_weather) | set(cur_weather))

    windows, drift, shares = [], [], []
    for period in sorted(set(cur_period)):
        mask = cur_period == period
        same_month = ref_month == int(period[-2:])
        if not same_month.any():
            LOG.warning("%s: no training hours in the same calendar month; seasonal PSI unavailable.", period)
        feat_rows = []
        for feature in MONITORED:
            if feature == WEATHER_FEATURE:
                whole = categorical_psi(ref_weather, cur_weather[mask], categories)
                seasonal = categorical_psi(ref_weather[same_month], cur_weather[mask], categories)
            else:
                whole = numeric_psi(ref_feats[feature], cur_feats.loc[mask, feature])
                seasonal = numeric_psi(ref_feats.loc[same_month, feature], cur_feats.loc[mask, feature])
            feat_rows.append({
                "window": period, "feature": feature, "n_hours": int(mask.sum()),
                "psi_whole_period": whole, "psi_same_month": seasonal,
                "status_whole_period": psi_status(whole), "status_same_month": psi_status(seasonal),
                "status": _worst(psi_status(whole), psi_status(seasonal)),
            })
        drift.extend(feat_rows)
        for group in categories:
            shares.append({
                "window": period, "weather_main": group,
                "share_training_period": float((ref_weather == group).mean()),
                "share_same_month_training": float((ref_weather[same_month] == group).mean()) if same_month.any() else np.nan,
                "share_window": float((cur_weather[mask] == group).mean()),
            })
        mae = float(abs_err[mask].mean())
        feature_status = "ALERT" if any(d["status"] == "ALERT" for d in feat_rows) else "PASS"
        err_status = error_status(mae, reference_mae)
        windows.append({
            "window": period, "n_hours": int(mask.sum()), "MAE": mae, "reference_MAE": reference_mae,
            "mae_ratio": mae / reference_mae, "error_status": err_status,
            "max_psi_whole_period": _max_or_nan([d["psi_whole_period"] for d in feat_rows]),
            "max_psi_same_month": _max_or_nan([d["psi_same_month"] for d in feat_rows]),
            "feature_alerts": ", ".join(d["feature"] for d in feat_rows if d["status"] == "ALERT"),
            "feature_warnings": ", ".join(d["feature"] for d in feat_rows if d["status"] == "WARNING"),
            "feature_status": feature_status,
            "status": "ALERT" if "ALERT" in (feature_status, err_status) else "PASS",
        })
    return pd.DataFrame(windows), pd.DataFrame(drift), pd.DataFrame(shares), reference_mae


def build_status(windows: pd.DataFrame, drift: pd.DataFrame, model_name: str, model_version: str, alias: str) -> dict:
    """Overall status = status of the latest window, with the reasons for any alert."""
    latest = windows.iloc[-1]
    reasons = []
    if latest["error_status"] == "ALERT":
        reasons.append(f"Prediction error: MAE {latest['MAE']:.1f} is {latest['mae_ratio']:.2f}x the reference "
                       f"{latest['reference_MAE']:.1f} (alert above {ERROR_RATIO_ALERT}x).")
    for row in drift[(drift["window"] == latest["window"]) & (drift["status"] == "ALERT")].itertuples():
        breached = " and ".join(name for name, s in (("whole training period", row.status_whole_period),
                                                     ("same training month", row.status_same_month)) if s == "ALERT")
        reasons.append(f"Feature drift in {row.feature}: PSI {row.psi_whole_period:.2f} vs the whole training period, "
                       f"{row.psi_same_month:.2f} vs the same training month (alert at {PSI_ALERT}; breached vs {breached}).")
    status = str(latest["status"])
    return {
        "overall_status": status,
        "label": STATUS_LABELS[status],
        "latest_window": str(latest["window"]),
        "reasons": reasons,
        "windows_evaluated": len(windows),
        "windows_alert": int((windows["status"] == "ALERT").sum()),
        "windows_pass": int((windows["status"] == "PASS").sum()),
        "model": {"name": model_name, "version": model_version, "alias": alias},
        "thresholds": {"psi_warning": PSI_WARNING, "psi_alert": PSI_ALERT, "error_ratio_alert": ERROR_RATIO_ALERT,
                       "psi_bins": PSI_BINS, "note": "heuristic conventions, not derived from this data"},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _png_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _charts(windows: pd.DataFrame) -> tuple[str, str]:
    x = windows["window"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(x, windows["MAE"], color=["#c62828" if s == "ALERT" else "#2e7d32" for s in windows["error_status"]])
    ax.axhline(windows["reference_MAE"].iloc[0], color="black", lw=1, ls="--", label="reference MAE (full test period)")
    ax.axhline(ERROR_RATIO_ALERT * windows["reference_MAE"].iloc[0], color="#c62828", lw=1, ls=":",
               label=f"alert: {ERROR_RATIO_ALERT}x reference")
    ax.set_ylabel("MAE (vehicles/h)")
    ax.set_title("Prediction error by month")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    err = _png_base64(fig)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(x, windows["max_psi_whole_period"], marker="o", label="max PSI vs whole training period")
    ax.plot(x, windows["max_psi_same_month"], marker="s", label="max PSI vs same training month")
    ax.axhline(PSI_WARNING, color="#f9a825", ls=":", label=f"warning {PSI_WARNING}")
    ax.axhline(PSI_ALERT, color="#c62828", ls=":", label=f"alert {PSI_ALERT}")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylabel("PSI (log scale above 0.1)")
    ax.set_title("Worst feature PSI by month")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    return err, _png_base64(fig)


def _badge(status: str) -> str:
    colour = {"PASS": "#2e7d32", "ALERT": "#c62828", "WARNING": "#f9a825", "N/A": "#78909c"}[status]
    return f'<span class="badge" style="background:{colour}">{html.escape(status)}</span>'


def _psi_cell(value: float) -> str:
    colour = {"PASS": "#c8e6c9", "WARNING": "#fff59d", "ALERT": "#ef9a9a", "N/A": "#eceff1"}[psi_status(value)]
    text = "n/a" if np.isnan(value) else f"{value:.2f}"
    return f'<td style="background:{colour}">{text}</td>'


def _psi_matrix(drift: pd.DataFrame, column: str) -> str:
    table = drift.pivot(index="window", columns="feature", values=column)[list(MONITORED)]
    head = "".join(f"<th>{html.escape(f)}</th>" for f in table.columns)
    body = "".join(
        f"<tr><th>{html.escape(w)}</th>{''.join(_psi_cell(v) for v in row)}</tr>" for w, row in table.iterrows()
    )
    return f"<table><tr><th>month</th>{head}</tr>{body}</table>"


def render_dashboard(status: dict, windows: pd.DataFrame, drift: pd.DataFrame) -> str:
    colour = "#2e7d32" if status["overall_status"] == "PASS" else "#c62828"
    err_chart, psi_chart = _charts(windows)
    reasons = "".join(f"<li>{html.escape(r)}</li>" for r in status["reasons"]) or "<li>No alert in the latest month.</li>"
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(r.window)}</td><td>{r.n_hours}</td><td>{r.MAE:.1f}</td><td>{r.mae_ratio:.2f}</td>"
        f"<td>{_badge(r.error_status)}</td><td>{_badge(r.feature_status)}</td>"
        f"<td>{html.escape(r.feature_alerts)}</td><td>{_badge(r.status)}</td></tr>"
        for r in windows.itertuples()
    )
    m, t = status["model"], status["thresholds"]
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Traffic model monitoring</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 2rem; color: #212121; max-width: 1100px; }}
.banner {{ background: {colour}; color: white; padding: 1rem 1.5rem; border-radius: 8px; }}
.banner h1 {{ margin: 0; font-size: 2rem; }}
table {{ border-collapse: collapse; margin: 0.5rem 0 1.5rem; font-size: 0.9rem; }}
th, td {{ border: 1px solid #cfd8dc; padding: 4px 10px; text-align: right; }}
th {{ background: #eceff1; }}
td:first-child, th:first-child {{ text-align: left; }}
.badge {{ color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8rem; }}
img {{ max-width: 100%; }}
.note {{ background: #f5f5f5; padding: 0.7rem 1rem; border-left: 4px solid #90a4ae; }}
</style></head><body>
<div class="banner"><h1>{html.escape(status['label'])}</h1>
<div>Latest month evaluated: {html.escape(status['latest_window'])} &middot; model {html.escape(m['name'])}
v{html.escape(str(m['version']))} (@{html.escape(m['alias'])}) &middot; generated {html.escape(status['generated_at'])}</div></div>
<h2>Why</h2><ul>{reasons}</ul>
<p>{status['windows_alert']} of {status['windows_evaluated']} months are ALERT; {status['windows_pass']} are PASS.</p>
<h2>Status by month</h2>
<table><tr><th>month</th><th>hours</th><th>MAE</th><th>MAE / reference</th><th>error drift</th><th>feature drift</th>
<th>features in ALERT</th><th>status</th></tr>{rows}</table>
<h2>Charts</h2><img src="data:image/png;base64,{err_chart}" alt="MAE by month"><br>
<img src="data:image/png;base64,{psi_chart}" alt="PSI by month">
<h2>Feature PSI vs the whole training period</h2>{_psi_matrix(drift, 'psi_whole_period')}
<h2>Feature PSI vs the same calendar month in the training period</h2>{_psi_matrix(drift, 'psi_same_month')}
<div class="note"><b>How to read this.</b> A month is ALERT if any feature has PSI &ge; {t['psi_alert']} against either reference,
or its MAE exceeds {t['error_ratio_alert']}&times; the model's full test-period MAE; PSI from {t['psi_warning']} is a warning only.
These thresholds are common rules of thumb, not derived from this data. Months are replayed from historical test data; the first
month is partial (26&ndash;31 Oct 2017) and short months give noisier PSI and MAE. The whole-period reference is affected by
seasonality (temperature differs by month), so use the same-month columns to judge it. Reasons and PSI describe distribution change,
not causes.</div>
</body></html>
"""


def save_outputs(output_dir: Path, status: dict, windows, drift, shares, page: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "monitoring_report.csv": ("monitoring report", lambda p: windows.round(4).to_csv(p, index=False, encoding="utf-8")),
        "monitoring_feature_drift.csv": ("feature drift table", lambda p: drift.round(4).to_csv(p, index=False, encoding="utf-8")),
        "monitoring_weather_shares.csv": ("weather share table", lambda p: shares.round(4).to_csv(p, index=False, encoding="utf-8")),
        "monitoring_status.json": ("status file", lambda p: p.write_text(json.dumps(status, indent=2), encoding="utf-8")),
        "monitoring_dashboard.html": ("dashboard", lambda p: p.write_text(page, encoding="utf-8")),
    }
    written = []
    for name, (what, write) in writers.items():
        path = output_dir / name
        write(path)
        LOG.info("Saved %s to %s.", what, path)
        written.append(path)
    return written


def run_monitoring(args) -> dict:
    LOG.info("Alert thresholds (heuristic conventions, not derived from this data): feature PSI warning %.2f, alert %.2f; "
             "error alert when month MAE > %.2fx the reference MAE; %d PSI bins from training-period quantiles.",
             PSI_WARNING, PSI_ALERT, ERROR_RATIO_ALERT, PSI_BINS)
    mlflow.set_tracking_uri(args.mlflow_uri)
    model, version = common.load_registered_model(args.model_name, args.alias)
    df, feats, cut = common.load_task1_frame(args.input)
    pred = model.predict(feats.iloc[cut:])
    windows, drift, shares, reference_mae = evaluate_windows(feats, df["timestamp"], df[tm.REG_TARGET], pred, cut)
    LOG.info("Reference MAE (model %s v%s, full test period): %.2f vehicles/h.", args.model_name, version, reference_mae)

    for row in windows.itertuples():
        LOG.info("%s: %d hours, MAE %.1f (%.2fx), max PSI %.2f whole / %.2f same month -> %s.", row.window, row.n_hours,
                 row.MAE, row.mae_ratio, row.max_psi_whole_period, row.max_psi_same_month, row.status)
        if row.status == "ALERT":
            reasons = []
            if row.error_status == "ALERT":
                reasons.append(f"error drift (MAE {row.MAE:.1f} vs reference {row.reference_MAE:.1f})")
            if row.feature_alerts:
                reasons.append(f"feature drift in {row.feature_alerts}")
            LOG.warning("Monitoring ALERT for %s: %s.", row.window, "; ".join(reasons))

    status = build_status(windows, drift, args.model_name, version, args.alias)
    outputs = save_outputs(args.output_dir, status, windows, drift, shares, render_dashboard(status, windows, drift))
    LOG.info("Overall status: %s (latest month %s; %d of %d months ALERT).", status["label"], status["latest_window"],
             status["windows_alert"], status["windows_evaluated"])
    if status["overall_status"] == "ALERT":
        LOG.warning("System status is ALERT: %s", " | ".join(status["reasons"]))

    if not args.no_mlflow:
        common.ensure_experiment(args.experiment, args.mlflow_artifact_root)
        with mlflow.start_run(run_name=f"monitoring replay ({args.model_name} v{version})"):
            mlflow.set_tags({"task": "monitoring", "overall_status": status["overall_status"]})
            mlflow.log_params({"model_name": args.model_name, "model_version": version, "alias": args.alias,
                               "psi_warning": PSI_WARNING, "psi_alert": PSI_ALERT, "error_ratio_alert": ERROR_RATIO_ALERT,
                               "psi_bins": PSI_BINS, "monitored_features": ",".join(MONITORED),
                               "window": "calendar month", "reference": "training period (whole and same calendar month)"})
            latest = windows.iloc[-1]
            mlflow.log_metrics({"reference_MAE": reference_mae, "windows_evaluated": len(windows),
                                "windows_alert": status["windows_alert"], "latest_MAE": float(latest["MAE"]),
                                "latest_mae_ratio": float(latest["mae_ratio"]),
                                "latest_max_psi_whole_period": float(latest["max_psi_whole_period"]),
                                "latest_max_psi_same_month": float(latest["max_psi_same_month"]),
                                "overall_alert": float(status["overall_status"] == "ALERT")})
            for path in outputs:
                mlflow.log_artifact(str(path))
        LOG.info("Logged monitoring run to %s (experiment '%s').", args.mlflow_uri, args.experiment)
    return status


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=common.DEFAULT_INPUT, help="Labelled CSV from prepare_labels.py.")
    parser.add_argument("--output-dir", type=Path, default=common.DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--model-name", default=common.REGISTERED_MODEL, help="Registered model to monitor.")
    parser.add_argument("--alias", default=common.CHAMPION_ALIAS, help="Registered model alias to monitor (default: champion).")
    parser.add_argument("--mlflow-uri", default=common.DEFAULT_MLFLOW_URI,
                        help="MLflow tracking URI (default: SQLite database Part-3/mlflow.db).")
    parser.add_argument("--mlflow-artifact-root", default=common.DEFAULT_ARTIFACT_ROOT,
                        help="Artifact location used when creating the experiment (default: Part-3/mlartifacts).")
    parser.add_argument("--experiment", default=MONITOR_EXPERIMENT, help="MLflow experiment name.")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    parser.add_argument("--log-file", type=Path, default=common.PART_DIR / "logs" / "monitoring.log", help="Log file path.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO",
                        help="Minimum level written to the console and log file (default: INFO).")
    return parser.parse_args()


def _configure_logging(log_path: Path, level: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stderr)],
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)  # drop chart-axis chatter


def main() -> int:
    args = _parse_args()
    try:
        _configure_logging(args.log_file, getattr(logging, args.log_level))
    except OSError:
        logging.basicConfig(level=logging.ERROR, stream=sys.stderr, force=True)
        LOG.exception("Failed to configure logging.")
        return EXIT_FAILURE

    try:
        status = run_monitoring(args)
    except (OSError, UnicodeError, MlflowException, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        LOG.exception("Monitoring failed.")
        return EXIT_FAILURE
    return EXIT_ALERT if status["overall_status"] == "ALERT" else EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
