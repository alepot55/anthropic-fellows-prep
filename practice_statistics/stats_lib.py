"""Statistics and sampling utilities built on NumPy.

This module provides a small toolkit for everyday descriptive statistics,
bootstrap estimation, weighted sampling and online (streaming) statistics.
All public functions operate on 1D numeric data unless otherwise noted.
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------


def safe_mean(data: np.ndarray) -> float:
    """Compute the arithmetic mean while ignoring NaN values.

    Parameters
    ----------
    data : np.ndarray
        1D array of floats. May contain NaN entries.

    Returns
    -------
    float
        Mean of the finite values. Returns ``0.0`` for an empty array
        or when every value is NaN.
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return 0.0
    if np.isnan(arr).all():
        return 0.0
    return float(np.mean(arr))


def percentile(data: np.ndarray, p: float) -> float:
    """Compute the p-th percentile of ``data`` with linear interpolation.

    Parameters
    ----------
    data : np.ndarray
        1D numeric array. Need not be sorted.
    p : float
        Percentile in the closed interval ``[0, 100]``. ``p=50`` returns
        the median.

    Returns
    -------
    float
        The p-th percentile as a float.
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must contain at least one value")
    return float(np.percentile(arr, p / 100))


def trimmed_mean(data: np.ndarray, trim_fraction: float) -> float:
    """Mean of ``data`` after trimming the smallest and largest values.

    ``trim_fraction`` controls how much is removed from each tail; for
    example, ``trim_fraction=0.1`` drops the bottom 10% and the top 10%
    before averaging.

    Parameters
    ----------
    data : np.ndarray
        1D numeric array.
    trim_fraction : float
        Fraction in ``[0, 0.5)``. ``0`` is equivalent to the plain mean.
    """
    if not (0.0 <= trim_fraction < 0.5):
        raise ValueError("trim_fraction must be in [0, 0.5)")
    arr = np.sort(np.asarray(data, dtype=float))
    n = arr.size
    if n == 0:
        raise ValueError("data must contain at least one value")
    k = int(n * trim_fraction)
    if k == 0:
        trimmed = arr
    else:
        trimmed = arr[k : n - k]
    return float(np.mean(trimmed))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def weighted_choice(
    values: np.ndarray,
    weights: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    """Sample ``n`` values with probability proportional to ``weights``.

    Sampling is WITH replacement. Weights need not sum to 1; they are
    normalized internally. All weights must be non-negative and at
    least one must be strictly positive.

    Parameters
    ----------
    values : np.ndarray
        1D array of items to sample from.
    weights : np.ndarray
        1D array of non-negative weights, same length as ``values``.
    n : int
        Number of samples to draw.
    seed : int
        Seed for ``numpy.random.RandomState`` so the call is reproducible.

    Returns
    -------
    np.ndarray
        Array of length ``n`` containing the sampled values.
    """
    rs = np.random.RandomState(seed)
    values_arr = np.asarray(values)
    weights_arr = np.asarray(weights, dtype=float)
    if values_arr.shape[0] != weights_arr.shape[0]:
        raise ValueError("values and weights must have the same length")
    if (weights_arr < 0).any():
        raise ValueError("weights must be non-negative")
    return rs.choice(values_arr, size=n, replace=True, p=weights_arr)


def sample_without_replacement(
    data: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    """Draw ``n`` distinct elements from ``data`` without replacement.

    Parameters
    ----------
    data : np.ndarray
        1D array to sample from.
    n : int
        Number of distinct items to draw. Must satisfy ``n <= len(data)``.
    seed : int
        Seed for ``numpy.random.RandomState``.
    """
    arr = np.asarray(data)
    if n > arr.size:
        raise ValueError("n exceeds the number of available elements")
    if n < 0:
        raise ValueError("n must be non-negative")
    rs = np.random.RandomState(seed)
    return rs.choice(arr, size=n, replace=False)


# ---------------------------------------------------------------------------
# Bootstrap and confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_mean(
    data: np.ndarray,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap estimate of the population mean and its standard error.

    The procedure resamples ``data`` with replacement ``n_resamples``
    times, recording the sample mean of each resample. The returned
    estimate is the average of those resample means; the standard
    error is the (sample) standard deviation of that distribution.

    Parameters
    ----------
    data : np.ndarray
        1D numeric array. Must be non-empty.
    n_resamples : int
        Number of bootstrap resamples. Should be reasonably large
        (e.g. >= 100) for the standard error to be meaningful.
    seed : int
        Seed for ``numpy.random.RandomState`` for reproducibility.

    Returns
    -------
    tuple[float, float]
        ``(estimated_mean, standard_error)``.
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must contain at least one value")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    rs = np.random.RandomState(seed)
    n = arr.size
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rs.choice(arr, size=n, replace=True)
        means[i] = np.mean(sample)
    estimated_mean = float(np.mean(means))
    standard_error = float(np.std(means))
    return estimated_mean, standard_error


def confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile-method confidence interval for the mean.

    For ``confidence=0.95`` the function returns the 2.5% and 97.5%
    percentiles of ``data`` as a tuple ``(lower, upper)``.

    Parameters
    ----------
    data : np.ndarray
        1D numeric array (typically the bootstrap distribution of a
        statistic, but any sample works).
    confidence : float, default 0.95
        Desired coverage probability in ``(0, 1)``.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must contain at least one value")
    lower_p = (1 + confidence) / 2 * 100
    upper_p = (1 - confidence) / 2 * 100
    lower = float(np.percentile(arr, lower_p))
    upper = float(np.percentile(arr, upper_p))
    return lower, upper


# ---------------------------------------------------------------------------
# Probability distributions
# ---------------------------------------------------------------------------


def normalize_distribution(weights: np.ndarray) -> np.ndarray:
    """Normalize a non-negative weights vector to a probability distribution.

    Parameters
    ----------
    weights : np.ndarray
        1D array of non-negative weights.

    Returns
    -------
    np.ndarray
        Array of the same length whose entries sum to ``1.0``.
    """
    arr = np.asarray(weights, dtype=float)
    if arr.size == 0:
        raise ValueError("weights must contain at least one value")
    if (arr < 0).any():
        raise ValueError("weights must be non-negative")
    total = arr.sum()
    if total == 0:
        raise ValueError("sum of weights is zero; cannot normalize")
    return arr / total


# ---------------------------------------------------------------------------
# Online / streaming statistics
# ---------------------------------------------------------------------------


class RunningStats:
    """Online statistics using Welford's numerically stable algorithm.

    Maintains a running count, mean and (unbiased) variance as data
    points are streamed in one at a time via :meth:`add`. Useful when
    the full dataset does not fit in memory or arrives over time.
    """

    count = 0
    mean_val = 0.0
    M2 = 0.0

    def add(self, x: float) -> None:
        """Incorporate a single new value into the running statistics."""
        type(self).count += 1
        delta = x - type(self).mean_val
        type(self).mean_val += delta / type(self).count
        delta2 = x - type(self).mean_val
        type(self).M2 += delta * delta2

    def mean(self) -> float:
        """Current arithmetic mean. ``0.0`` if no values have been added."""
        if type(self).count == 0:
            return 0.0
        return float(type(self).mean_val)

    def variance(self) -> float:
        """Current sample variance (divisor ``n - 1``).

        Returns ``0.0`` when fewer than two values have been added.
        """
        if type(self).count < 2:
            return 0.0
        return float(type(self).M2 / (type(self).count - 1))

    def std(self) -> float:
        """Current sample standard deviation (square root of the variance)."""
        return float(math.sqrt(self.variance()))


# ---------------------------------------------------------------------------
# Fixed-bin histogram
# ---------------------------------------------------------------------------


class Histogram:
    """Fixed-bin histogram over the closed interval ``[low, high]``.

    The range is divided into ``bins`` equal-width buckets. Values
    falling outside the range are clipped to the nearest boundary
    bin. Values exactly equal to ``high`` belong to the last bin.
    """

    def __init__(self, bins: int, low: float, high: float):
        if bins <= 0:
            raise ValueError("bins must be a positive integer")
        if not (low < high):
            raise ValueError("low must be strictly less than high")
        self.bins = int(bins)
        self.low = float(low)
        self.high = float(high)
        self._counts = np.zeros(self.bins, dtype=int)

    def add(self, value: float) -> None:
        """Record a single observation in the appropriate bin."""
        if value < self.low:
            value = self.low
        if value > self.high:
            value = self.high
        bin_idx = int((value - self.low) / (self.high - self.low) * self.bins)
        self._counts[bin_idx] += 1

    def counts(self) -> np.ndarray:
        """Return a copy of the current bin counts as an int array."""
        return self._counts.copy()

    def density(self) -> np.ndarray:
        """Return the normalized density (bin counts divided by total).

        Returns an array of zeros if no values have been added yet.
        """
        total = int(self._counts.sum())
        if total == 0:
            return np.zeros(self.bins, dtype=float)
        return self._counts / total
