"""Focused tests for the Task 5 travel-timing recommendation engine."""

import unittest

import pandas as pd

import recommendation_engine as rec

# Weekday-like shape: night low, morning and afternoon peaks, midday dip at 10:00.
HOURLY = [300] * 5 + [2500, 6000, 5500, 5000, 4800, 4000, 4600, 4800, 4800, 5200, 5600, 6300, 5800, 4500, 3300, 2900, 2700, 2100, 1400]


def _history(extra=None, holiday_date=None):
    rows = []
    for day in pd.date_range("2018-01-01", periods=14):  # Monday 2018-01-01 to Sunday 2018-01-14
        for hour in range(24):
            volume = HOURLY[hour] if day.dayofweek < 5 else HOURLY[hour] / 2
            holiday = "Test Day" if holiday_date == day and hour == 0 else None
            rows.append((day + pd.Timedelta(hours=hour), holiday, "Clear", volume))
    frame = pd.DataFrame(rows, columns=["timestamp", "holiday", "weather_main", "traffic_volume"])
    if extra is not None:
        frame = pd.concat([frame, extra], ignore_index=True)
    return rec.prepare_history(frame)


class WindowTests(unittest.TestCase):
    def test_hour_labels_use_the_12_hour_clock(self):
        self.assertEqual([rec.format_hour(h) for h in (0, 10, 12, 15, 24)],
                         ["12:00 AM", "10:00 AM", "12:00 PM", "3:00 PM", "12:00 AM"])

    def test_non_overlapping_skips_windows_that_share_hours(self):
        windows = pd.DataFrame({"start": [10, 11, 15], "end": [12, 13, 17], "expected_volume": [1.0, 2.0, 3.0]})
        self.assertEqual(rec.non_overlapping(windows, 3)["start"].tolist(), [10, 15])


class HistoryTests(unittest.TestCase):
    def test_holiday_flag_is_spread_over_the_whole_date_and_beats_weekend(self):
        history = _history(holiday_date=pd.Timestamp("2018-01-06"))  # a Saturday
        by_date = history.groupby("date")["day_type"].agg(lambda s: set(s))
        self.assertEqual(by_date[pd.Timestamp("2018-01-06")], {"Holiday"})
        self.assertEqual(by_date[pd.Timestamp("2018-01-07")], {"Weekend"})
        self.assertEqual(by_date[pd.Timestamp("2018-01-02")], {"Weekday"})

    def test_missing_columns_are_rejected(self):
        with self.assertRaises(ValueError):
            rec.prepare_history(pd.DataFrame({"timestamp": []}))

    def test_unknown_weather_is_not_treated_as_rain(self):
        extra = pd.DataFrame({
            "timestamp": [pd.Timestamp("2018-02-05 10:00")],
            "holiday": [None],
            "weather_main": ["Tornado"],
            "traffic_volume": [100],
        })
        with self.assertRaisesRegex(ValueError, "Unknown weather categories: Tornado"):
            _history(extra)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = rec.RecommendationEngine(_history())

    def test_best_daytime_window_is_the_midday_dip_not_the_night(self):
        result = self.engine.recommend("weekday")
        self.assertEqual((result.best.start, result.best.end), (10, 11))
        self.assertEqual((result.peak.start, result.peak.end), (16, 17))
        self.assertIn("between 10:00 AM and 11:00 AM", result.text)
        self.assertIn("weekday journey", result.text)

    def test_evening_option_is_offered_only_when_clearly_quieter(self):
        result = self.engine.recommend("weekday")
        self.assertEqual((result.later_option.start, result.later_option.end), (21, 22))
        self.assertIn("lighter still", result.text)
        weekend = self.engine.recommend("weekend", earliest=6, latest=22)
        self.assertIsNone(weekend.later_option)

    def test_sparse_weather_cell_is_shrunk_towards_all_weather(self):
        extra = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2018-02-05 10:00")],  # a Monday
                "holiday": [None],
                "weather_main": ["Snow"],
                "traffic_volume": [100],
            }
        )
        engine = rec.RecommendationEngine(_history(extra))
        snow = engine.curve("Weekday", "Severe weather").loc[10, "expected_volume"]
        overall = engine.curve("Weekday", "Any").loc[10, "expected_volume"]
        self.assertLess(snow, overall)
        self.assertGreater(snow, 0.8 * overall)  # one hour among 10 pseudo-observations barely moves it

    def test_weather_with_few_hours_is_flagged_as_limited_evidence(self):
        result = self.engine.recommend("weekday", "Severe weather")  # no such hours in the history
        self.assertTrue(result.limited_evidence)
        self.assertIn("leans on the all-weather pattern", result.text)
        self.assertIn("Traffic volume is not a safety measure", result.text)

    def test_optional_later_window_starts_after_requested_range(self):
        result = self.engine.recommend("weekday", earliest=19, latest=20)
        self.assertGreaterEqual(result.later_option.start, 20)
        self.assertIn("outside your requested range", result.text)

    def test_invalid_inputs_raise_value_error(self):
        with self.assertRaises(ValueError):
            self.engine.recommend("monday")
        with self.assertRaises(ValueError):
            self.engine.recommend("weekday", "hail")
        with self.assertRaises(ValueError):
            self.engine.recommend("weekday", duration=3, earliest=6, latest=8)


if __name__ == "__main__":
    unittest.main()
