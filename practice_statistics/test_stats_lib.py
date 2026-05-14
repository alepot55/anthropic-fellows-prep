"""Test suite for stats_lib. Run with:

    python3 -m unittest test_stats_lib -v
"""

import unittest

import numpy as np

from stats_lib import (
    Histogram,
    RunningStats,
    bootstrap_mean,
    confidence_interval,
    normalize_distribution,
    percentile,
    safe_mean,
    sample_without_replacement,
    trimmed_mean,
    weighted_choice,
)


class TestSafeMean(unittest.TestCase):
    def test_no_nan_returns_plain_mean(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(safe_mean(data), 3.0, places=10)

    def test_mixed_finite_and_nan_ignores_nan(self):
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        # Mean of finite entries: (1 + 2 + 4 + 5) / 4 = 3.0
        self.assertAlmostEqual(safe_mean(data), 3.0, places=10)


class TestPercentile(unittest.TestCase):
    def test_median_is_p50(self):
        data = np.arange(1, 102, dtype=float)  # 1..101
        # Median of 1..101 is 51
        self.assertAlmostEqual(percentile(data, 50), 51.0, places=6)

    def test_quartiles(self):
        data = np.arange(1, 101, dtype=float)  # 1..100
        # 25th percentile of 1..100 (linear) is 25.75; 75th is 75.25
        self.assertAlmostEqual(percentile(data, 25), 25.75, places=6)
        self.assertAlmostEqual(percentile(data, 75), 75.25, places=6)

    def test_extremes(self):
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertAlmostEqual(percentile(data, 0), 10.0, places=6)
        self.assertAlmostEqual(percentile(data, 100), 50.0, places=6)


class TestTrimmedMean(unittest.TestCase):
    def test_ten_percent_trim_drops_extremes(self):
        data = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0]
        )
        # Trim 10% from each end -> drop [1.0] and [1000.0]
        # Mean of [2..9] = 44 / 8 = 5.5
        self.assertAlmostEqual(trimmed_mean(data, 0.1), 5.5, places=10)


class TestWeightedChoice(unittest.TestCase):
    def test_normalized_weights_runs_and_respects_seed(self):
        values = np.array([1, 2, 3, 4])
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        out_a = weighted_choice(values, weights, n=200, seed=7)
        out_b = weighted_choice(values, weights, n=200, seed=7)
        self.assertEqual(len(out_a), 200)
        self.assertTrue(np.array_equal(out_a, out_b))
        self.assertTrue(np.all(np.isin(out_a, values)))

    def test_unnormalized_weights_are_normalized_internally(self):
        values = np.array([10, 20, 30, 40])
        weights = np.array([1.0, 2.0, 3.0, 4.0])  # sums to 10, not 1
        out = weighted_choice(values, weights, n=500, seed=42)
        self.assertEqual(len(out), 500)
        self.assertTrue(np.all(np.isin(out, values)))


class TestSampleWithoutReplacement(unittest.TestCase):
    def test_returns_distinct_elements(self):
        data = np.arange(50)
        out = sample_without_replacement(data, n=20, seed=1)
        self.assertEqual(len(out), 20)
        self.assertEqual(len(set(out.tolist())), 20)
        self.assertTrue(np.all(np.isin(out, data)))


class TestBootstrapMean(unittest.TestCase):
    def test_same_seed_gives_same_result(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        m1, se1 = bootstrap_mean(data, n_resamples=200, seed=123)
        m2, se2 = bootstrap_mean(data, n_resamples=200, seed=123)
        self.assertAlmostEqual(m1, m2, places=12)
        self.assertAlmostEqual(se1, se2, places=12)

    def test_standard_error_matches_sample_std_of_resample_means(self):
        data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        n_resamples = 200
        seed = 2024
        _, se = bootstrap_mean(data, n_resamples=n_resamples, seed=seed)
        # Independently replay the bootstrap and compute the *sample*
        # standard deviation (ddof=1) of the resample means.
        rs = np.random.RandomState(seed)
        means = np.empty(n_resamples)
        for i in range(n_resamples):
            sample = rs.choice(data, size=data.size, replace=True)
            means[i] = np.mean(sample)
        expected = float(np.std(means, ddof=1))
        self.assertAlmostEqual(se, expected, places=10)


class TestConfidenceInterval(unittest.TestCase):
    def test_lower_strictly_less_than_upper(self):
        data = np.arange(1, 101, dtype=float)
        lower, upper = confidence_interval(data, confidence=0.95)
        self.assertLess(
            lower, upper, "lower bound must be less than upper bound"
        )

    def test_exact_values_for_uniform_grid(self):
        data = np.arange(0, 101, dtype=float)  # 0..100
        lower, upper = confidence_interval(data, confidence=0.90)
        # 5th percentile = 5.0, 95th percentile = 95.0
        self.assertAlmostEqual(lower, 5.0, places=6)
        self.assertAlmostEqual(upper, 95.0, places=6)


class TestNormalizeDistribution(unittest.TestCase):
    def test_normalizes_to_sum_one(self):
        weights = np.array([1.0, 3.0, 6.0])
        dist = normalize_distribution(weights)
        self.assertAlmostEqual(dist.sum(), 1.0, places=12)
        self.assertTrue(
            np.allclose(dist, np.array([0.1, 0.3, 0.6]), atol=1e-12)
        )


class TestRunningStats(unittest.TestCase):
    def test_basic_mean_and_variance(self):
        rs = RunningStats()
        for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            rs.add(x)
        # Mean = 40 / 8 = 5.0
        # Sample variance = sum((x - 5)^2) / 7 = 32 / 7
        self.assertAlmostEqual(rs.mean(), 5.0, places=10)
        self.assertAlmostEqual(rs.variance(), 32.0 / 7.0, places=8)

    def test_empty_and_single_value(self):
        rs = RunningStats()
        self.assertEqual(rs.mean(), 0.0)
        self.assertEqual(rs.variance(), 0.0)
        rs.add(42.0)
        self.assertAlmostEqual(rs.mean(), 42.0, places=10)
        # Fewer than two values -> variance is 0.0
        self.assertEqual(rs.variance(), 0.0)

    def test_two_instances_do_not_share_state(self):
        rs1 = RunningStats()
        rs1.add(10.0)
        rs1.add(20.0)
        rs1.add(30.0)
        rs2 = RunningStats()
        # rs2 has had nothing added yet
        self.assertEqual(rs2.mean(), 0.0)
        self.assertEqual(rs2.variance(), 0.0)
        rs2.add(100.0)
        self.assertAlmostEqual(rs2.mean(), 100.0, places=10)
        # rs1 should still report its own mean
        self.assertAlmostEqual(rs1.mean(), 20.0, places=10)


class TestHistogram(unittest.TestCase):
    def test_basic_assignment_to_correct_bins(self):
        h = Histogram(bins=4, low=0.0, high=4.0)
        for v in [0.5, 1.5, 2.5, 3.5]:
            h.add(v)
        self.assertTrue(
            np.array_equal(h.counts(), np.array([1, 1, 1, 1]))
        )

    def test_value_equal_to_high_goes_in_last_bin(self):
        h = Histogram(bins=4, low=0.0, high=4.0)
        h.add(4.0)  # exactly at the upper boundary
        counts = h.counts()
        self.assertEqual(int(counts.sum()), 1)
        self.assertEqual(int(counts[-1]), 1, "value == high belongs in the last bin")

    def test_below_low_is_clipped_to_first_bin(self):
        h = Histogram(bins=4, low=0.0, high=4.0)
        h.add(-100.0)
        h.add(0.0)
        counts = h.counts()
        self.assertEqual(int(counts[0]), 2)
        self.assertEqual(int(counts.sum()), 2)


class TestInstanceIndependence(unittest.TestCase):
    """State must not leak between independent objects."""

    def test_running_stats_instances_independent(self):
        a = RunningStats()
        a.add(1.0)
        a.add(2.0)
        a.add(3.0)
        b = RunningStats()
        b.add(100.0)
        b.add(200.0)
        # Each instance must report only its own values.
        self.assertAlmostEqual(a.mean(), 2.0, places=10)
        self.assertAlmostEqual(b.mean(), 150.0, places=10)


if __name__ == "__main__":
    unittest.main()
