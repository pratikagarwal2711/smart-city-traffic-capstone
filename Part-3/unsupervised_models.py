"""Unsupervised analysis for Part 3, Task 2: K-means clusters and association rules.

* K-means groups hours by traffic conditions (hour of day, weekend flag, weather
  severity and traffic volume) and profiles what each cluster represents.
* Association rules mine "time of day + day type + weather -> congestion level"
  patterns and rank them by lift.

No accident data exists, so this describes traffic conditions only. Congestion
level is the Part 3 quartile ``congestion_category``, which is derived from
``traffic_volume``; that volume is also a K-means feature, so clusters partly
reflect the congestion levels by construction. Results are associations.
"""

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from prepare_labels import LOW_VISIBILITY_WEATHER, SEVERE_WEATHER
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger(__name__)

RANDOM_STATE = 42
SILHOUETTE_SAMPLE = 10_000  # silhouette is quadratic in rows, so score a sample
# Silhouette keeps creeping up to k=9, but k=6 is where it plateaus and the clusters stay
# easy to describe (weekday/weekend and peak/off-peak, plus one severe-weather group).
DEFAULT_K = 6
KMEANS_FEATURES = ("hour_sin", "hour_cos", "is_weekend", "weather_severity", "traffic_volume")
WEATHER_SEVERITY_LABELS = {0: "other", 1: "low visibility", 2: "severe"}
CONGESTION_LEVELS = ("Low", "Medium", "High", "Severe")
TIME_OF_DAY_BINS = {"Night": (0, 5), "Morning": (6, 11), "Afternoon": (12, 17), "Evening": (18, 23)}
RULE_ATTRIBUTES = ("time_of_day", "day_type", "weather_group")
REQUIRED_COLUMNS = ("timestamp", "weather_main", "traffic_volume", "congestion_category", "high_risk")
# A rule is flagged as adding information only if it is at least this much more
# confident than every simpler rule (e.g. "Night & Clear -> Low" vs "Night -> Low").
MIN_CONFIDENCE_GAIN = 0.02


def add_calendar_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Return a time-sorted copy with hour, weekend flag and cyclical hour columns."""
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError("Input is missing required columns: " + ", ".join(missing))
    df = data.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype("int8")
    angle = 2 * np.pi * df["hour"] / 24
    df["hour_sin"], df["hour_cos"] = np.sin(angle), np.cos(angle)
    return df


def weather_severity(weather_main: pd.Series) -> pd.Series:
    """Ordinal weather severity: 0 other, 1 low visibility, 2 severe (Part 3 groups)."""
    severity = pd.Series(0, index=weather_main.index, dtype="int8")
    severity[weather_main.isin(LOW_VISIBILITY_WEATHER)] = 1
    severity[weather_main.isin(SEVERE_WEATHER)] = 2
    return severity


def select_k(features: pd.DataFrame, k_values) -> pd.DataFrame:
    """Inertia and silhouette score for each candidate number of clusters."""
    scaled = StandardScaler().fit_transform(features)
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(scaled)
        silhouette = silhouette_score(
            scaled, model.labels_, sample_size=SILHOUETTE_SAMPLE, random_state=RANDOM_STATE
        )
        LOG.info("k=%d: inertia=%.0f, silhouette=%.3f.", k, model.inertia_, silhouette)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette})
    return pd.DataFrame(rows)


def fit_kmeans(features: pd.DataFrame, k: int) -> tuple[Pipeline, pd.Series]:
    """Fit standardised K-means; clusters are numbered by ascending mean traffic volume."""
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)),
        ]
    ).fit(features)
    raw_labels = pd.Series(model.named_steps["kmeans"].labels_, index=features.index)
    order = features["traffic_volume"].groupby(raw_labels).mean().sort_values().index
    labels = raw_labels.map({old: new for new, old in enumerate(order)}).rename("cluster")
    return model, labels


def profile_clusters(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """One row per cluster describing size, volume, timing, weather and risk."""
    grouped = df.groupby(labels)
    profile = pd.DataFrame(
        {
            "hours": grouped.size(),
            "share": grouped.size() / len(df),
            "mean_volume": grouped["traffic_volume"].mean(),
            "most_common_hours": grouped["hour"].agg(
                lambda h: ", ".join(str(x) for x in sorted(h.value_counts().head(4).index))
            ),
            "weekend_share": grouped["is_weekend"].mean(),
            "low_visibility_share": grouped["weather_severity"].apply(lambda s: (s == 1).mean()),
            "severe_weather_share": grouped["weather_severity"].apply(lambda s: (s == 2).mean()),
            "high_risk_share": grouped["high_risk"].mean(),
        }
    )
    congestion = pd.crosstab(labels, df["congestion_category"], normalize="index")
    congestion = congestion.reindex(columns=list(CONGESTION_LEVELS), fill_value=0)
    return profile.join(congestion.add_prefix("share_")).round(3)


def time_of_day(hour: pd.Series) -> pd.Series:
    labels = pd.Series(index=hour.index, dtype=object)
    for name, (start, end) in TIME_OF_DAY_BINS.items():
        labels[hour.between(start, end)] = name
    return labels


def weather_group(weather_main: pd.Series) -> pd.Series:
    """Readable weather bands for the rules; rare types are covered by the Part 3 groups."""
    groups = weather_main.where(weather_main.isin(("Clear", "Clouds")), "Rain/Drizzle")
    groups[weather_main.isin(LOW_VISIBILITY_WEATHER)] = "Low visibility"
    groups[weather_main.isin(SEVERE_WEATHER)] = "Severe weather"
    return groups


def rule_items(df: pd.DataFrame) -> pd.DataFrame:
    """Discretised columns used as rule items plus the congestion level."""
    return pd.DataFrame(
        {
            "time_of_day": time_of_day(df["hour"]),
            "day_type": np.where(df["is_weekend"] == 1, "Weekend", "Weekday"),
            "weather_group": weather_group(df["weather_main"]),
            "congestion": df["congestion_category"].astype(str),
        }
    )


def mine_rules(items: pd.DataFrame, min_support: float, min_confidence: float) -> pd.DataFrame:
    """Rules {time of day, day type, weather} -> congestion level, ranked by lift.

    Every combination of one to three attributes is counted exactly, which is what
    Apriori would return for rules with a single congestion-level consequent.
    support    = share of all hours matching antecedent and consequent.
    confidence = share of antecedent hours that have the consequent.
    lift       = confidence divided by the overall share of the consequent.
    """
    n = len(items)
    base_rate = items["congestion"].value_counts() / n
    tables = []
    for size in range(1, len(RULE_ATTRIBUTES) + 1):
        for cols in itertools.combinations(RULE_ATTRIBUTES, size):
            joint = items.groupby([*cols, "congestion"]).size().rename("count").reset_index()
            antecedent_count = items.groupby(list(cols)).size().rename("antecedent_hours")
            joint = joint.join(antecedent_count, on=list(cols))
            joint["antecedent"] = joint[list(cols)].agg(" & ".join, axis=1)
            tables.append(joint[["antecedent", "congestion", "count", "antecedent_hours"]])

    rules = pd.concat(tables, ignore_index=True)
    rules["support"] = rules["count"] / n
    rules["confidence"] = rules["count"] / rules["antecedent_hours"]
    rules["lift"] = rules["confidence"] / rules["congestion"].map(base_rate)
    rules["confidence_gain"] = rules["confidence"] - _best_general_confidence(rules, base_rate)
    rules["adds_information"] = rules["confidence_gain"] >= MIN_CONFIDENCE_GAIN
    rules = rules[(rules["support"] >= min_support) & (rules["confidence"] >= min_confidence)]
    rules = rules.sort_values(["lift", "support"], ascending=False).reset_index(drop=True)
    rules["meaning"] = [explain_rule(r, base_rate[r.congestion]) for r in rules.itertuples()]
    LOG.info(
        "%d rules pass support>=%.3f and confidence>=%.2f; %d add information over simpler rules.",
        len(rules),
        min_support,
        min_confidence,
        rules["adds_information"].sum(),
    )
    return rules


def _best_general_confidence(rules: pd.DataFrame, base_rate: pd.Series) -> pd.Series:
    """Highest confidence among simpler rules (fewer antecedent items, same consequent).

    A single-item rule is compared with the overall share of its consequent.
    """
    confidence_of = {
        (frozenset(a.split(" & ")), c): conf
        for a, c, conf in zip(rules["antecedent"], rules["congestion"], rules["confidence"])
    }

    def best(rule) -> float:
        items = rule.antecedent.split(" & ")
        simpler = (
            frozenset(subset)
            for size in range(1, len(items))
            for subset in itertools.combinations(items, size)
        )
        return max((confidence_of[(s, rule.congestion)] for s in simpler), default=base_rate[rule.congestion])

    return pd.Series([best(r) for r in rules.itertuples()], index=rules.index)


def explain_rule(rule, base_rate: float) -> str:
    """Plain-language sentence for one rule row."""
    return (
        f"For {rule.antecedent} hours, {rule.confidence:.0%} have {rule.congestion} congestion, "
        f"versus {base_rate:.0%} of all hours ({rule.lift:.1f}x as likely); {rule.count:,} hours."
    )


def log_to_mlflow(model, k_table, rules, info, output_dir, tracking_uri, artifact_root, experiment):
    """Log the chosen K-means model and the rule-mining summary as one MLflow run."""
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=artifact_root)
    mlflow.set_experiment(experiment)
    chosen = k_table[k_table["k"] == info["k"]].iloc[0]
    with mlflow.start_run(run_name=f"kmeans k={info['k']} + association rules"):
        mlflow.set_tags({"task": "unsupervised"})
        mlflow.log_params(
            {
                "k": info["k"],
                "kmeans_features": ",".join(KMEANS_FEATURES),
                "weather_severity": str(WEATHER_SEVERITY_LABELS),
                "min_support": info["min_support"],
                "min_confidence": info["min_confidence"],
                "random_state": RANDOM_STATE,
                "label_note": "congestion_category is derived from traffic_volume, a K-means feature",
            }
        )
        mlflow.log_metrics(
            {"inertia": chosen["inertia"], "silhouette": chosen["silhouette"], "n_rules": len(rules)}
        )
        # The model is produced by this script, so pickle is acceptable here.
        mlflow.sklearn.log_model(model, name="kmeans", serialization_format="cloudpickle")
        for name in ("kmeans_k_selection.csv", "kmeans_cluster_profile.csv", "association_rules.csv"):
            mlflow.log_artifact(str(output_dir / name))
    LOG.info("Logged MLflow run to %s (experiment '%s').", tracking_uri, experiment)


def run_analysis(data: pd.DataFrame, k: int, min_support: float, min_confidence: float):
    """Return (k table, fitted model, cluster profile, labelled data, rules, info)."""
    df = add_calendar_columns(data)
    df["weather_severity"] = weather_severity(df["weather_main"])
    features = df[list(KMEANS_FEATURES)]

    k_table = select_k(features, range(2, 11))
    model, labels = fit_kmeans(features, k)
    df["cluster"] = labels
    profile = profile_clusters(df, labels)

    rules = mine_rules(rule_items(df), min_support, min_confidence)
    info = {"k": k, "min_support": min_support, "min_confidence": min_confidence, "n_rows": len(df)}
    return k_table, model, profile, df, rules, info


def _parse_args() -> argparse.Namespace:
    part_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=part_dir / "data" / "processed" / "Metro_Interstate_Traffic_Volume_ml.csv",
        help="Labelled CSV from prepare_labels.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=part_dir / "outputs", help="Output directory.")
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K, help=f"Number of clusters (default: {DEFAULT_K}, see DEFAULT_K)."
    )
    parser.add_argument("--min-support", type=float, default=0.01, help="Minimum rule support.")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum rule confidence.")
    parser.add_argument(
        "--mlflow-uri",
        default=f"sqlite:///{(part_dir / 'mlflow.db').as_posix()}",
        help="MLflow tracking URI (default: SQLite database Part-3/mlflow.db).",
    )
    parser.add_argument(
        "--mlflow-artifact-root",
        default=(part_dir / "mlartifacts").as_uri(),
        help="Artifact location used when creating the experiment (default: Part-3/mlartifacts).",
    )
    parser.add_argument("--experiment", default="task2-unsupervised-models", help="MLflow experiment name.")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    parser.add_argument(
        "--log-file", type=Path, default=part_dir / "logs" / "unsupervised_models.log", help="Log file path."
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Minimum level written to the console and log file (default: INFO).",
    )
    return parser.parse_args()


def _configure_logging(log_path: Path, level: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
        force=True,
    )


def _save_outputs(output_dir: Path, k_table, profile, df, rules, info) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "kmeans_k_selection.csv": k_table.round(4),
        "kmeans_cluster_profile.csv": profile.rename_axis("cluster").reset_index(),
        "kmeans_cluster_assignments.csv": df[["timestamp", "cluster"]],
        "association_rules.csv": rules.round(4),
    }
    for name, table in files.items():
        table.to_csv(output_dir / name, index=False, encoding="utf-8")
        LOG.info("Saved %s.", output_dir / name)
    info_path = output_dir / "unsupervised_run_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    LOG.info("Saved %s.", info_path)


def main() -> int:
    args = _parse_args()
    try:
        _configure_logging(args.log_file, getattr(logging, args.log_level))
    except OSError:
        logging.basicConfig(level=logging.ERROR, stream=sys.stderr, force=True)
        LOG.exception("Failed to configure logging.")
        return 1

    try:
        LOG.info("Reading %s.", args.input)
        data = pd.read_csv(args.input, encoding="utf-8-sig")
        LOG.info("Loaded %d rows, %d columns.", *data.shape)
        k_table, model, profile, df, rules, info = run_analysis(
            data, args.k, args.min_support, args.min_confidence
        )
        _save_outputs(args.output_dir, k_table, profile, df, rules, info)
        if not args.no_mlflow:
            log_to_mlflow(
                model, k_table, rules, info, args.output_dir,
                args.mlflow_uri, args.mlflow_artifact_root, args.experiment,
            )
    except (OSError, UnicodeError, MlflowException, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        LOG.exception("Unsupervised analysis failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
