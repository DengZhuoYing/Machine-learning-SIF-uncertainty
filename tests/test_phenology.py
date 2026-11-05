from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from calculate_phenology import extract_phenology, extract_supported_pos


class PhenologyTests(unittest.TestCase):
    def setUp(self):
        self.doy = np.arange(1, 366)
        self.fitted = (
            np.exp(-((self.doy - 200) / 25) ** 2)
            + 1.6 * np.exp(-((self.doy - 345) / 20) ** 2)
        )
        self.observed = np.where(
            (self.doy >= 120) & (self.doy <= 280), self.fitted, np.nan
        )

    def test_winter_extrapolation_is_replaced_by_supported_summer_peak(self):
        self.assertEqual(self.doy[np.argmax(self.fitted)], 345)
        self.assertEqual(extract_supported_pos(self.fitted, self.doy, self.observed), 200)
        self.assertEqual(extract_phenology(self.fitted, [0.2], observed=self.observed)[1], [200])

    def test_real_late_season_peak_is_retained(self):
        fitted = np.exp(-((self.doy - 260) / 25) ** 2)
        self.assertEqual(extract_phenology(fitted, [0.2])[1], [260])

    def test_observation_distance_limit_is_configurable(self):
        observed = np.full(365, np.nan)
        observed[194] = 1.0
        self.assertEqual(extract_phenology(
            self.fitted, [0.2], observed=observed, max_gap_days=5
        )[1], [200])
        self.assertTrue(np.isnan(extract_phenology(
            self.fitted, [0.2], observed=observed, max_gap_days=4
        )[1][0]))

    def test_sos_eos_and_threshold_independence(self):
        for thresholds in [[0.1], [0.2], [0.3], [0.1, 0.2, 0.3]]:
            original = extract_phenology(self.fitted, thresholds)
            corrected = extract_phenology(self.fitted, thresholds, observed=self.observed)
            self.assertEqual(corrected[0], original[0])
            self.assertEqual(corrected[2], original[2])
            self.assertEqual(corrected[1], [200])

    def test_actual_leap_year_and_annual_endpoint_dates(self):
        dates = np.arange(np.datetime64('2020-01-01'), np.datetime64('2022-01-01'))
        fitted = np.concatenate([np.arange(366, dtype=float), np.arange(365, 0, -1.0)])
        self.assertEqual(extract_phenology(fitted, [0.2], dates=dates)[1], [366, 1])

    def test_sparse_sampling_uses_days_instead_of_sample_counts(self):
        for step in [4, 8]:
            offsets = np.arange(0, 365, step)
            dates = np.datetime64('2021-01-01') + offsets.astype('timedelta64[D]')
            fitted = np.zeros(len(offsets))
            fitted[10] = 2
            fitted[20] = 1
            observed = np.full(len(offsets), np.nan)
            observed[8] = observed[20] = 1
            expected = int(offsets[10] + 1) if step == 4 else int(offsets[20] + 1)
            self.assertEqual(extract_phenology(
                fitted, [0.2], dates=dates, observed=observed, max_gap_days=10
            )[1], [expected])

    def test_years_have_separate_observation_support(self):
        dates = np.arange(np.datetime64('2019-01-01'), np.datetime64('2021-01-01'))
        doy = (dates - dates.astype('datetime64[Y]')).astype(int) + 1
        fitted = np.exp(-((doy - 200) / 25) ** 2)
        observed = fitted.copy()
        observed[dates < np.datetime64('2020-01-01')] = np.nan
        pos = extract_phenology(fitted, [0.2], dates=dates, observed=observed)[1]
        self.assertTrue(np.isnan(pos[0]))
        self.assertEqual(pos[1], 200)

    def test_missing_and_flat_curves_have_no_pos(self):
        for fitted, observed in [
            (np.full(365, np.nan), self.observed),
            (self.fitted, np.full(365, np.nan)),
            (np.zeros(365), np.zeros(365)),
        ]:
            self.assertTrue(np.isnan(extract_phenology(fitted, [0.2], observed=observed)[1][0]))


if __name__ == '__main__':
    unittest.main()
