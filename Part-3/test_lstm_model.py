"""Focused tests for Task 3 data preparation and inference."""

import unittest

import numpy as np
import pandas as pd

import lstm_model as lm


class Task3Tests(unittest.TestCase):
    def setUp(self):
        timestamps = pd.date_range("2020-01-01", periods=80, freq="h")
        self.data = pd.DataFrame({
            "timestamp": timestamps,
            "traffic_volume": np.arange(80, dtype=float),
            "temp": np.full(80, 280.0),
            "rain_1h": np.zeros(80),
            "snow_1h": np.zeros(80),
            "clouds_all": np.zeros(80),
        })
        self.raw = self.data.drop(columns="timestamp").copy()
        self.raw["date_time"] = timestamps.strftime("%d/%m/%Y %H:%M")

    def test_weather_imputation_uses_only_training_rows(self):
        self.raw.loc[5, "temp"] = 0
        self.data.loc[5, "temp"] = 999
        self.raw.loc[50:, "temp"] = 310
        self.data.loc[50:, "temp"] = 310
        result, medians, counts = lm.restore_and_impute_weather(self.data, self.raw, 50)
        self.assertEqual(result.loc[5, "temp"], 280)
        self.assertEqual(medians["temp"]["1"], 280)
        self.assertEqual(counts["temp"], 1)

    def test_mismatched_raw_data_is_rejected(self):
        self.raw.loc[1, "traffic_volume"] = 999
        with self.assertRaisesRegex(ValueError, "traffic_volume do not match"):
            lm.restore_and_impute_weather(self.data, self.raw, 50)

    def test_scaler_excludes_validation(self):
        feats = pd.DataFrame({
            "temp_c": self.data["temp"] - 273.15,
            "rain_1h": self.data["rain_1h"],
            "snow_1h": self.data["snow_1h"],
            "clouds_all": self.data["clouds_all"],
            **{feature: np.zeros(80) for feature in lm.CAL_FEATURES},
            **{feature: np.zeros(80) for feature in lm.TREE_CAL_FEATURES},
        })
        windows = lm.build_windows(self.data, feats, 50)
        self.assertAlmostEqual(windows["volume_mean"], self.data["traffic_volume"].iloc[:50].mean())

    def test_windows_do_not_cross_gaps(self):
        timestamps = pd.Series(pd.date_range("2020-01-01", periods=30, freq="h").tolist()
                               + pd.date_range("2020-01-03", periods=26, freq="h").tolist())
        np.testing.assert_array_equal(lm.window_targets(timestamps), [24, 25, 26, 27, 28, 29, 54, 55])

    def test_empty_validation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Insufficient windows"):
            lm.split_windows({"target_idx": np.array([0, 1, 2])}, 2)

    def test_saved_inference_scales_and_clips(self):
        class FakeModel:
            def predict(self, inputs, verbose):
                np.testing.assert_array_equal(inputs[0], np.zeros((1, 2, 1)))
                return np.array([[-2.0]])

        metadata = {
            "window_hours": 2, "sequence_features": ["traffic_volume"],
            "calendar_features": ["hour"], "scaler_mean": [10.0], "scaler_scale": [10.0],
        }
        prediction = lm.predict_from_saved(FakeModel(), [[[10], [10]]], [[0]], metadata)
        np.testing.assert_array_equal(prediction, [0])
        with self.assertRaisesRegex(ValueError, "Calendar"):
            lm.predict_from_saved(FakeModel(), [[[10], [10]]], [[0, 1]], metadata)


if __name__ == "__main__":
    unittest.main()
