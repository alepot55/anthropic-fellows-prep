# BUGS.md — Statistics & Sampling practice

**DO NOT READ DURING A TIMED RUN.**

This file is the post-mortem reference. It lists every deliberately
planted bug, the symptom, the test that catches it, the expected fix,
and the trap that makes each one easy to miss.

Run the suite for context:

```bash
python3 -m unittest test_stats_lib -v
```

Expected initial state: **13 failing / 8 passing** out of 21 tests.

---

## Bug 1 — `safe_mean` propagates NaN (easy)

**File / line:** `stats_lib.py`, `safe_mean`, the final `return` statement.

**Symptom:** `safe_mean` returns `nan` whenever the input contains any
NaN, even though the docstring promises NaN values are ignored.

**Test that catches it:** `TestSafeMean.test_mixed_finite_and_nan_ignores_nan`
fails with `nan != 3.0`.

**Buggy code:**

```python
return float(np.mean(arr))
```

**Fix:**

```python
return float(np.nanmean(arr))
```

**Why a candidate would fall for it:** the early-return for the
all-NaN case (`if np.isnan(arr).all(): return 0.0`) makes the function
*look* NaN-aware. The eye stops reading once that branch is there.
`np.mean` vs `np.nanmean` is a single character of difference.

---

## Bug 2 — `percentile` divides `p` by 100 (easy)

**File / line:** `stats_lib.py`, `percentile`, the `return` statement.

**Symptom:** the function treats its `p` argument as if `np.percentile`
expected a fraction in `[0, 1]`. So `percentile(data, 50)` returns the
0.5th percentile, which is near the minimum, not the median.

**Tests that catch it:**

- `TestPercentile.test_median_is_p50` (`1.5 != 51.0`)
- `TestPercentile.test_quartiles` (`1.2475 != 25.75`)
- `TestPercentile.test_extremes` (passes at `p=0`, fails at `p=100`)

**Buggy code:**

```python
return float(np.percentile(arr, p / 100))
```

**Fix:**

```python
return float(np.percentile(arr, p))
```

**Why a candidate would fall for it:** mental confusion between APIs
that take a fraction (e.g. SciPy's `scoreatpercentile` historically,
or quantile-style libraries) and NumPy's `percentile` which already
takes a 0–100 number. The `/ 100` "looks like a unit conversion".

---

## Bug 3 — `weighted_choice` does not normalize weights (medium)

**File / line:** `stats_lib.py`, `weighted_choice`, the `rs.choice` call.

**Symptom:** when caller passes weights that do not sum to 1,
`rs.choice` raises `ValueError: probabilities do not sum to 1`. The
docstring explicitly promises that the function normalizes internally.

**Test that catches it:** `TestWeightedChoice.test_unnormalized_weights_are_normalized_internally`
errors out (ValueError) instead of returning a sample.

**Buggy code:**

```python
return rs.choice(values_arr, size=n, replace=True, p=weights_arr)
```

**Fix:**

```python
weights_arr = weights_arr / weights_arr.sum()
return rs.choice(values_arr, size=n, replace=True, p=weights_arr)
```

(Or call `normalize_distribution` — that helper already exists in the
same module, which is another hint that normalization was *meant* to
happen here.)

**Why a candidate would fall for it:** the function does some defensive
checks (shape match, non-negative weights), which makes the rest look
"already handled". The fact that an existing `normalize_distribution`
function lives a few lines below is the giveaway.

---

## Bug 4 — `bootstrap_mean` standard error uses biased `std` (medium)

**File / line:** `stats_lib.py`, `bootstrap_mean`, the line
`standard_error = float(np.std(means))`.

**Symptom:** the reported standard error is the population (`ddof=0`)
standard deviation of the resample means instead of the sample
(`ddof=1`) one. For 200 resamples, the difference is roughly
`sqrt(200/199) ≈ 1.0025`, i.e. about 0.25%. Small but consistent.

**Test that catches it:** `TestBootstrapMean.test_standard_error_matches_sample_std_of_resample_means`
fails with a tight `places=10` tolerance.

**Buggy code:**

```python
standard_error = float(np.std(means))
```

**Fix:**

```python
standard_error = float(np.std(means, ddof=1))
```

**Why a candidate would fall for it:** `np.std` defaults to `ddof=0`,
which most people never notice because the difference is tiny. The
test ALMOST passes — only a strict tolerance catches it. Treat any
`np.std` / `np.var` on a sample as suspect: check the `ddof`.

---

## Bug 5 — `confidence_interval` swaps lower and upper percentiles (medium)

**File / line:** `stats_lib.py`, `confidence_interval`, the two
`lower_p` / `upper_p` lines.

**Symptom:** for `confidence=0.95` the function returns the 97.5th
percentile as `lower` and the 2.5th percentile as `upper`. The order
is wrong, so `lower > upper` on any non-degenerate data.

**Tests that catch it:**

- `TestConfidenceInterval.test_lower_strictly_less_than_upper`
  (`assertLess` fails: 97.5 not less than 2.5)
- `TestConfidenceInterval.test_exact_values_for_uniform_grid`
  (`lower=95.0` instead of `5.0`)

**Buggy code:**

```python
lower_p = (1 + confidence) / 2 * 100
upper_p = (1 - confidence) / 2 * 100
```

**Fix:**

```python
lower_p = (1 - confidence) / 2 * 100
upper_p = (1 + confidence) / 2 * 100
```

(Equivalently: just swap the two return values.)

**Why a candidate would fall for it:** the algebraic forms look
plausible — both expressions are "half of confidence-something". A
sanity check on a known sample (e.g. "for 95% on `arange(101)`, the
bounds are 2.5 and 97.5") catches it instantly; without that check
you can stare at the formula and nod.

---

## Bug 6 — `RunningStats` uses class attributes instead of instance state (hard)

**File / line:** `stats_lib.py`, `RunningStats` class definition and
every method body.

**Symptom:** `count`, `mean_val`, `M2` are defined at the class level
and every write inside `add` goes through `type(self).X`, which
mutates the class attribute. There is no `__init__`. As soon as two
`RunningStats` instances are created in the same process, they share
all of their internal state. A fresh instance also inherits the
running totals from previous instances.

**Tests that catch it (four of them):**

- `TestRunningStats.test_basic_mean_and_variance`
- `TestRunningStats.test_empty_and_single_value`
- `TestRunningStats.test_two_instances_do_not_share_state`
- `TestInstanceIndependence.test_running_stats_instances_independent`

(Whichever runs first in alphabetical order — `TestBootstrapMean`,
`TestConfidenceInterval`, … — does not pollute state, so by the time
`TestInstanceIndependence` and `TestRunningStats` run, every read
sees leaked totals from earlier tests.)

**Buggy code (skeleton):**

```python
class RunningStats:
    count = 0
    mean_val = 0.0
    M2 = 0.0

    def add(self, x):
        type(self).count += 1
        delta = x - type(self).mean_val
        type(self).mean_val += delta / type(self).count
        delta2 = x - type(self).mean_val
        type(self).M2 += delta * delta2

    def mean(self):
        if type(self).count == 0:
            return 0.0
        return float(type(self).mean_val)
    ...
```

**Fix:** introduce a real `__init__` and store state on `self`:

```python
class RunningStats:
    def __init__(self):
        self.count = 0
        self.mean_val = 0.0
        self.M2 = 0.0

    def add(self, x):
        self.count += 1
        delta = x - self.mean_val
        self.mean_val += delta / self.count
        delta2 = x - self.mean_val
        self.M2 += delta * delta2

    def mean(self):
        if self.count == 0:
            return 0.0
        return float(self.mean_val)
    ...
```

**Why a candidate would fall for it:** with only one instance the
class "works perfectly" — Welford's math is correct, the test for a
single fresh object passes. The bug is invisible until two instances
coexist or the test order causes leakage. The use of `type(self).X`
inside the methods *looks* like Pythonic class-level access but is a
load-bearing mistake. Watch for: missing `__init__`, mutable
class-level attributes, and `type(self).X = ...` / `Cls.X = ...`
inside instance methods.

**Sub-optimal fix to flag in debrief:** if the candidate adds
`__init__` but leaves `type(self).X` in `add`, the state still leaks.
Both pieces must move to `self`.

---

## Bug 7 — `Histogram.add` IndexError when `value == high` (hard)

**File / line:** `stats_lib.py`, `Histogram.add`, the `bin_idx`
computation.

**Symptom:** for any value exactly equal to `self.high`, the computed
`bin_idx` is `self.bins` (one past the last valid index), so
`self._counts[bin_idx] += 1` raises `IndexError`. The docstring
explicitly states that values equal to `high` belong to the last bin.

**Test that catches it:** `TestHistogram.test_value_equal_to_high_goes_in_last_bin`
errors out with `IndexError: index 4 is out of bounds for axis 0 with size 4`.

**Buggy code:**

```python
def add(self, value):
    if value < self.low:
        value = self.low
    if value > self.high:
        value = self.high
    bin_idx = int((value - self.low) / (self.high - self.low) * self.bins)
    self._counts[bin_idx] += 1
```

**Fix:** clamp `bin_idx` to the last valid bin index.

```python
bin_idx = int((value - self.low) / (self.high - self.low) * self.bins)
if bin_idx == self.bins:
    bin_idx = self.bins - 1
self._counts[bin_idx] += 1
```

(Or `bin_idx = min(bin_idx, self.bins - 1)`.)

**Why a candidate would fall for it:** the function already does two
explicit clipping steps for `value < low` and `value > high`, so the
boundary "looks handled". The trap is that the clip puts `value` at
exactly `high`, and the multiplicative index formula then *produces*
the out-of-range index. The fix belongs after the index calculation,
not before.

**Sub-optimal fix to flag in debrief:** replacing `>` with `>=` in the
upper clip is wrong — it does not move the value to the previous bin,
it just sets it to `high` again. The clamp must happen on the index.

---

## Test-by-test expected outcome

| Test | Expected | Catches |
| --- | --- | --- |
| `TestBootstrapMean.test_same_seed_gives_same_result` | pass | — |
| `TestBootstrapMean.test_standard_error_matches_sample_std_of_resample_means` | FAIL | Bug 4 |
| `TestConfidenceInterval.test_exact_values_for_uniform_grid` | FAIL | Bug 5 |
| `TestConfidenceInterval.test_lower_strictly_less_than_upper` | FAIL | Bug 5 |
| `TestHistogram.test_basic_assignment_to_correct_bins` | pass | — |
| `TestHistogram.test_below_low_is_clipped_to_first_bin` | pass | — |
| `TestHistogram.test_value_equal_to_high_goes_in_last_bin` | ERROR | Bug 7 |
| `TestInstanceIndependence.test_running_stats_instances_independent` | FAIL | Bug 6 |
| `TestNormalizeDistribution.test_normalizes_to_sum_one` | pass | — |
| `TestPercentile.test_extremes` | FAIL | Bug 2 |
| `TestPercentile.test_median_is_p50` | FAIL | Bug 2 |
| `TestPercentile.test_quartiles` | FAIL | Bug 2 |
| `TestRunningStats.test_basic_mean_and_variance` | FAIL | Bug 6 (state leak) |
| `TestRunningStats.test_empty_and_single_value` | FAIL | Bug 6 |
| `TestRunningStats.test_two_instances_do_not_share_state` | FAIL | Bug 6 |
| `TestSafeMean.test_mixed_finite_and_nan_ignores_nan` | FAIL | Bug 1 |
| `TestSafeMean.test_no_nan_returns_plain_mean` | pass | — |
| `TestSampleWithoutReplacement.test_returns_distinct_elements` | pass | — |
| `TestTrimmedMean.test_ten_percent_trim_drops_extremes` | pass | — |
| `TestWeightedChoice.test_normalized_weights_runs_and_respects_seed` | pass | — |
| `TestWeightedChoice.test_unnormalized_weights_are_normalized_internally` | ERROR | Bug 3 |

Totals: **8 pass, 11 fail, 2 error → 13 of 21 tests are red.**

---

## Difficulty summary

- **Easy (2):** Bug 1 (`np.mean` vs `np.nanmean`), Bug 2 (`p / 100`).
- **Medium (3):** Bug 3 (no normalization), Bug 4 (`ddof=0`), Bug 5 (swapped bounds).
- **Hard (2):** Bug 6 (class-level state in `RunningStats`), Bug 7 (boundary IndexError in `Histogram`).
