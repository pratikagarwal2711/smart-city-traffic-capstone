"""Focused tests for the Task 4 MLflow run audit and model selection."""

import unittest

import pandas as pd

import mlflow_registry as reg


def _runs():
    rows = [
        ("exp", "LSTM", "LSTM", False, 155.0, "2026-11-23", "aaa"),
        ("exp", "LSTM", "LSTM", False, 160.0, "2026-11-18", "aaa"),
        ("exp", "Forest", "Forest", False, 147.0, "2026-11-23", "aaa"),
        ("exp", "Baseline", "Baseline", True, 100.0, "2026-11-23", "aaa"),
    ]
    frame = pd.DataFrame(rows, columns=["experiment", "run_name", "model", "is_baseline", "MAE", "start_time", "git_commit"])
    frame["task"] = "t"
    frame["start_time"] = pd.to_datetime(frame["start_time"], utc=True)
    return frame.sort_values("start_time", ascending=False).reset_index(drop=True)


class RegistrySelectionTests(unittest.TestCase):
    def test_latest_runs_keeps_newest_per_model(self):
        latest = reg.latest_runs(_runs())
        self.assertEqual(len(latest), 3)
        self.assertEqual(latest.loc[latest["run_name"] == "LSTM", "MAE"].item(), 155.0)

    def test_ranked_excludes_baselines_and_orders_by_metric(self):
        best = reg.ranked(reg.latest_runs(_runs()), "exp", "MAE")
        self.assertEqual(best["run_name"].tolist(), ["Forest", "LSTM"])
        best_high = reg.ranked(reg.latest_runs(_runs()), "exp", "MAE", higher_is_better=True)
        self.assertEqual(best_high["run_name"].iloc[0], "LSTM")

    def test_ranked_without_candidates_raises(self):
        with self.assertRaises(ValueError):
            reg.ranked(reg.latest_runs(_runs()), "other", "MAE")

    def test_rerun_variability_reports_only_repeated_models(self):
        with self.assertLogs(reg.LOG, level="WARNING"):
            table = reg.rerun_variability(_runs())
        self.assertEqual(table["run_name"].tolist(), ["LSTM"])
        self.assertAlmostEqual(table["spread"].item(), 5.0)


if __name__ == "__main__":
    unittest.main()
