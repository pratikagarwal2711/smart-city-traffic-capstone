"""Focused tests for Task 6: version promotion, serving API and drift monitoring."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import model_versions as mv
import monitoring as mon
import serve_api as api
import train_models as tm


def _raw_frame(timestamp="2018-06-14 17:00:00", weather="Clouds", holiday="None"):
    """One-row input in the layout of the labelled CSV, with two weather groups."""
    row = {
        "timestamp": timestamp, "holiday": holiday, "temp": 295.65, "rain_1h": 0.0, "snow_1h": 0.0,
        "clouds_all": 40, "traffic_volume": 4000, "high_risk": 0,
        "weather_main_Clear": int(weather == "Clear"), "weather_main_Clouds": int(weather == "Clouds"),
    }
    return pd.DataFrame([row])


class StubModel:
    """Stands in for the registered model; records the frame it was asked to score."""

    feature_names_in_ = np.array(tm.build_features(_raw_frame())[1].columns)

    def __init__(self, value=5600.0):
        self.value, self.last_frame = value, None

    def predict(self, frame):
        self.last_frame = frame
        return np.full(len(frame), self.value)


class ChampionPolicyTests(unittest.TestCase):
    def _table(self, maes, gates=(True, True, True)):
        return pd.DataFrame({"version": ["1", "2", "3"], "MAE": maes, "gate_passed": list(gates)})

    def test_best_version_that_beats_v1_is_promoted(self):
        winner, _ = mv.choose_champion(self._table([240.0, 238.0, 236.0]), "1")
        self.assertEqual(winner, "3")

    def test_v1_stays_champion_when_nothing_beats_it(self):
        winner, _ = mv.choose_champion(self._table([240.0, 241.0, 240.0]), "1")
        self.assertEqual(winner, "1")

    def test_version_that_fails_the_gate_is_not_promoted(self):
        winner, _ = mv.choose_champion(self._table([240.0, 238.0, 200.0], gates=(True, True, False)), "1")
        self.assertEqual(winner, "2")

    def test_no_alias_when_v1_fails_gate_and_is_not_beaten(self):
        winner, _ = mv.choose_champion(self._table([240.0, 245.0, 250.0], gates=(False, True, True)), "1")
        self.assertIsNone(winner)


class PsiTests(unittest.TestCase):
    def test_identical_distributions_have_zero_psi(self):
        x = np.random.default_rng(0).normal(size=2000)
        self.assertAlmostEqual(mon.numeric_psi(x, x), 0.0, places=6)

    def test_shifted_distribution_alerts(self):
        rng = np.random.default_rng(0)
        value = mon.numeric_psi(rng.normal(size=5000), rng.normal(loc=2.0, size=5000))
        self.assertGreaterEqual(value, mon.PSI_ALERT)

    def test_zero_heavy_column_splits_into_zero_and_nonzero(self):
        reference = np.r_[np.zeros(950), np.ones(50)]
        self.assertGreaterEqual(mon.numeric_psi(reference, np.zeros(500)), 0.0)
        self.assertGreater(mon.numeric_psi(reference, np.ones(500)), mon.PSI_ALERT)

    def test_categorical_psi_detects_mix_change(self):
        ref = ["Clear"] * 50 + ["Rain"] * 50
        self.assertAlmostEqual(mon.categorical_psi(ref, ref, ["Clear", "Rain"]), 0.0, places=6)
        self.assertGreater(mon.categorical_psi(ref, ["Rain"] * 100, ["Clear", "Rain"]), mon.PSI_ALERT)

    def test_status_thresholds(self):
        self.assertEqual(mon.psi_status(0.05), "PASS")
        self.assertEqual(mon.psi_status(mon.PSI_WARNING), "WARNING")
        self.assertEqual(mon.psi_status(mon.PSI_ALERT), "ALERT")
        self.assertEqual(mon.psi_status(float("nan")), "N/A")
        self.assertEqual(mon.error_status(1.25 * 200, 200), "PASS")
        self.assertEqual(mon.error_status(1.26 * 200, 200), "ALERT")


class WindowTests(unittest.TestCase):
    """Training Jan-Apr 2017; test months March (stable) and April (drifted, worse errors) 2018."""

    @staticmethod
    def _frames(drifted_temp):
        rng = np.random.default_rng(1)
        stamps = pd.concat([
            pd.Series(pd.date_range("2017-01-01", "2017-04-30 23:00", freq="h")),
            pd.Series(pd.date_range("2018-03-01", "2018-04-30 23:00", freq="h")),
        ], ignore_index=True)
        cut = int((stamps < "2018-03-01").sum())
        weather = rng.choice(["Clear", "Clouds"], len(stamps))
        feats = pd.DataFrame({
            "temp_c": rng.normal(0, 3, len(stamps)), "rain_1h": 0.0, "snow_1h": 0.0,
            "clouds_all": rng.integers(0, 100, len(stamps)), "hour": stamps.dt.hour,
            "weather_main_Clear": (weather == "Clear").astype(int), "weather_main_Clouds": (weather == "Clouds").astype(int),
        })
        april = (stamps >= "2018-04-01").to_numpy()
        if drifted_temp:
            feats.loc[april, "temp_c"] += 30
        y = pd.Series(rng.normal(3000, 100, len(stamps)))
        pred = y.to_numpy() + np.where(april, 400.0, 20.0)
        return feats, stamps, y, pred, cut

    def test_stable_month_passes_and_drifted_month_alerts(self):
        feats, stamps, y, pred, cut = self._frames(drifted_temp=True)
        windows, drift, shares, ref_mae = mon.evaluate_windows(feats, stamps, y, pred[cut:], cut)
        self.assertEqual(windows["window"].tolist(), ["2018-03", "2018-04"])
        march, april = windows.iloc[0], windows.iloc[1]
        self.assertEqual(march["status"], "PASS")
        self.assertEqual(april["status"], "ALERT")
        self.assertEqual(april["error_status"], "ALERT")
        self.assertIn("temp_c", april["feature_alerts"])
        self.assertAlmostEqual(ref_mae, windows["MAE"].mul(windows["n_hours"]).sum() / windows["n_hours"].sum())
        self.assertEqual(set(shares["weather_main"]), {"Clear", "Clouds"})

    def test_overall_status_is_latest_window_and_short_windows_are_kept(self):
        feats, stamps, y, pred, cut = self._frames(drifted_temp=True)
        # Keep only 5 hours of April: it is still evaluated, not dropped.
        keep = ((stamps < "2018-04-01") | (stamps <= "2018-04-01 04:00")).to_numpy()
        windows, drift, _, _ = mon.evaluate_windows(feats[keep].reset_index(drop=True), stamps[keep].reset_index(drop=True),
                                                    y[keep].reset_index(drop=True), pred[keep][cut:], cut)
        self.assertEqual(windows.iloc[-1]["n_hours"], 5)
        status = mon.build_status(windows, drift, "m", "1", "champion")
        self.assertEqual(status["latest_window"], "2018-04")
        self.assertEqual(status["overall_status"], windows.iloc[-1]["status"])

    def test_seasonal_reference_separates_seasonality_from_drift(self):
        rng = np.random.default_rng(2)
        stamps = pd.concat([
            pd.Series(pd.date_range("2017-01-01", "2017-01-31 23:00", freq="h")),
            pd.Series(pd.date_range("2017-07-01", "2017-07-31 23:00", freq="h")),
            pd.Series(pd.date_range("2018-01-01", "2018-01-31 23:00", freq="h")),
        ], ignore_index=True)
        cut = int((stamps < "2018-01-01").sum())
        january = (stamps.dt.month == 1).to_numpy()
        feats = pd.DataFrame({
            "temp_c": np.where(january, rng.normal(-10, 2, len(stamps)), rng.normal(20, 2, len(stamps))),
            "rain_1h": 0.0, "snow_1h": 0.0, "clouds_all": 50, "hour": stamps.dt.hour,
            "weather_main_Clear": 1, "weather_main_Clouds": 0,
        })
        y = pd.Series(np.full(len(stamps), 3000.0))
        windows, drift, _, _ = mon.evaluate_windows(feats, stamps, y, y.to_numpy()[cut:] + 10.0, cut)
        temp = drift[drift["feature"] == "temp_c"].iloc[0]
        self.assertGreaterEqual(temp["psi_whole_period"], mon.PSI_ALERT)
        self.assertLess(temp["psi_same_month"], mon.PSI_WARNING)
        # The worse reference drives the status, so seasonality alone still raises ALERT.
        self.assertEqual(windows.iloc[0]["status"], "ALERT")

    def test_dashboard_shows_label(self):
        feats, stamps, y, pred, cut = self._frames(drifted_temp=True)
        windows, drift, _, _ = mon.evaluate_windows(feats, stamps, y, pred[cut:], cut)
        status = mon.build_status(windows, drift, "m", "3", "champion")
        page = mon.render_dashboard(status, windows, drift)
        self.assertIn("ALERT / Requires investigation", page)
        self.assertNotIn("PASS / Normal", page.split("<h2>Why</h2>")[0])


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.model = StubModel()
        self.tmp = tempfile.TemporaryDirectory()
        self.status_file = Path(self.tmp.name) / "monitoring_status.json"
        settings = api.Settings(status_file=self.status_file)
        loader = lambda s: api.ModelBundle(self.model, "traffic-volume-regressor", "3", "champion")
        self.client = TestClient(api.create_app(settings, loader))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.addCleanup(self.tmp.cleanup)
        self.body = {"timestamp": "2018-06-14T17:00:00", "temp_c": 22.5, "rain_1h": 0, "snow_1h": 0,
                     "clouds_all": 40, "weather_main": "clouds"}

    def test_predict_returns_prediction_and_category(self):
        resp = self.client.post("/predict", json=self.body)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["predicted_traffic_volume"], 5600.0)
        self.assertEqual(data["traffic_category"], "High")
        self.assertTrue(data["is_congested"])
        self.assertEqual(data["model_version"], "3")

    def test_request_features_match_training_feature_builder(self):
        self.client.post("/predict", json={**self.body, "temp_c": 295.65 - 273.15})
        expected = tm.build_features(_raw_frame())[1]
        pd.testing.assert_frame_equal(self.model.last_frame.astype(float), expected.astype(float),
                                      check_exact=False, check_names=False)

    def test_invalid_input_is_rejected(self):
        self.assertEqual(self.client.post("/predict", json={**self.body, "clouds_all": 150}).status_code, 422)
        self.assertEqual(self.client.post("/predict", json={**self.body, "weather_main": "Hail"}).status_code, 422)
        incomplete = {k: v for k, v in self.body.items() if k != "temp_c"}
        self.assertEqual(self.client.post("/predict", json=incomplete).status_code, 422)

    def test_category_boundaries_follow_part_2(self):
        self.assertEqual([api.traffic_category(v) for v in (4499.9, 4500, 5500, 5500.1)],
                         ["Low", "Medium", "Medium", "High"])

    def test_monitor_status_requires_a_status_file(self):
        self.assertEqual(self.client.get("/monitor/status").status_code, 404)
        self.status_file.write_text('{"overall_status": "PASS"}', encoding="utf-8")
        self.assertEqual(self.client.get("/monitor/status").json()["overall_status"], "PASS")

    def test_predict_is_unavailable_when_model_cannot_load(self):
        def failing(settings):
            raise ValueError("no champion")

        with self.assertLogs(api.LOG, level="ERROR"), TestClient(api.create_app(api.Settings(), failing)) as client:
            self.assertEqual(client.get("/health").json()["model_loaded"], False)
            self.assertEqual(client.post("/predict", json=self.body).status_code, 503)


if __name__ == "__main__":
    unittest.main()
