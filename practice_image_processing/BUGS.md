**DO NOT READ THIS FILE DURING TIMED RUN.**

This document lists every deliberately-introduced bug in `image_lib.py`,
the test that catches it, and the expected fix. Use it for debrief only.

---

## Bug 1 — `to_grayscale` collapses to a scalar

- **Difficulty:** easy
- **File / line:** `image_lib.py`, inside `to_grayscale` (~line 25)
- **Symptom:** the function returns a 0-D scalar (sum of *all* weighted
  pixels) instead of an (H, W) grayscale image.
- **Expected behavior:** apply the per-channel weights and reduce along
  the channel axis, producing shape `(H, W)`.
- **Tests that catch it:**
  - `TestToGrayscale.test_output_shape` (fails on `result.shape`)
  - `TestToGrayscale.test_uniform_image`
  - `TestToGrayscale.test_known_pixel_values`
- **Buggy line:**
  ```python
  return np.sum(weighted)
  ```
- **Fix:**
  ```python
  return np.sum(weighted, axis=-1)
  ```
- **Why a candidate falls for it:** the weighted multiplication is
  correct, and "sum" matches the docstring's "weighted sum" wording.
  The missing axis is the only thing wrong, and it's a one-token slip.

---

## Bug 2 — `threshold` uses `>=` instead of `>`

- **Difficulty:** easy
- **File / line:** `image_lib.py`, inside `threshold` (~line 47)
- **Symptom:** pixels whose value equals the threshold are classified
  as 1 instead of 0.
- **Expected behavior:** the docstring says "strictly greater" — only
  pixels with `value > threshold` should become 1.
- **Tests that catch it:**
  - `TestThreshold.test_boundary_strictly_greater`
  - (`test_basic_split` passes because no pixel sits exactly at 100.)
- **Buggy line:**
  ```python
  return (image >= value).astype(np.uint8)
  ```
- **Fix:**
  ```python
  return (image > value).astype(np.uint8)
  ```
- **Why a candidate falls for it:** "greater than" and "greater than or
  equal" feel interchangeable until a boundary test forces the choice.
  Only one test exercises the boundary; the other passes, masking the
  bug from anyone who only skims results.

---

## Bug 3 — `crop` returns a view, not a copy

- **Difficulty:** medium
- **File / line:** `image_lib.py`, inside `crop` (~line 73)
- **Symptom:** writes to the returned array mutate the original image
  because numpy slicing returns a view by default.
- **Expected behavior:** the docstring explicitly says the returned
  array must be independent of the original.
- **Tests that catch it:**
  - `TestCrop.test_modifying_crop_does_not_affect_original`
  - (`test_correct_region` passes — the slice contents are correct.)
- **Buggy line:**
  ```python
  return image[top:top + height, left:left + width]
  ```
- **Fix:**
  ```python
  return image[top:top + height, left:left + width].copy()
  ```
- **Why a candidate falls for it:** the slice *looks* like a fresh
  array. View-vs-copy is a classic NumPy footgun. A candidate without
  a test for mutation independence might never notice.

---

## Bug 4 — `histogram` drops trailing zero buckets

- **Difficulty:** medium
- **File / line:** `image_lib.py`, inside `histogram` (~line 88)
- **Symptom:** `np.bincount(arr)` returns an array of length
  `arr.max() + 1`. For an image with max value 100, the result has
  length 101 instead of the requested 256.
- **Expected behavior:** result length must equal `bins`.
- **Tests that catch it:**
  - `TestHistogram.test_partial_range_values`
  - (`test_full_range_values` passes because the image contains 255,
    so bincount returns 256 buckets anyway.)
- **Buggy line:**
  ```python
  return np.bincount(image.ravel())
  ```
- **Fix:**
  ```python
  return np.bincount(image.ravel(), minlength=bins)
  ```
- **Why a candidate falls for it:** `np.bincount` is the natural tool
  here, and tests that include the value 255 hide the bug. The
  `bins` parameter is technically unused in the buggy version, which
  a careful reader would spot, but it's not the kind of thing pytest
  output highlights.

---

## Bug 5 — `brightest_pixels` returns flat indices

- **Difficulty:** medium
- **File / line:** `image_lib.py`, inside `brightest_pixels` (~line 107)
- **Symptom:** returns a list of flat (1-D) indices like
  `[24, 23, 22]` instead of `[(4, 4), (4, 3), (4, 2)]`.
- **Expected behavior:** `(row, col)` integer tuples, brightest first.
- **Tests that catch it:**
  - `TestBrightestPixels.test_simple_5x5`
  - `TestBrightestPixels.test_tie_breaker_prefers_smaller_row_then_col`
- **Buggy line:**
  ```python
  return list(top)
  ```
- **Fix:**
  ```python
  rows, cols = np.unravel_index(top, image.shape)
  return list(zip(rows.tolist(), cols.tolist()))
  ```
- **Why a candidate falls for it:** the sort logic is correct, the
  tie-breaker reasoning in the comment is correct, and `argsort`
  naturally produces flat indices. The missing step (`unravel_index`)
  is exactly what someone reaching for `argsort` would forget.

---

## Bug 6 — `add_noise` ignores the seed

- **Difficulty:** hard
- **File / line:** `image_lib.py`, inside `add_noise` (~line 141)
- **Symptom:** the function calls `np.random.normal(...)` (the global
  RNG) instead of a `RandomState` seeded with `seed`. Two calls with
  the same seed produce different outputs.
- **Expected behavior:** results must be deterministic per `seed`.
- **Tests that catch it:**
  - `TestAddNoise.test_deterministic_same_seed`
  - (`test_output_shape_and_range` passes — shape and clip are fine.)
- **Buggy line:**
  ```python
  noise = np.random.normal(0.0, std, size=image.shape)
  ```
- **Fix:**
  ```python
  rs = np.random.RandomState(seed)
  noise = rs.normal(0.0, std, size=image.shape)
  ```
  (`np.random.default_rng(seed).normal(...)` is also fine.)
- **Why a candidate falls for it:** the `seed` parameter sits in the
  signature, looks used (because the function "uses randomness"), and
  the rest of the function works. Only a determinism test surfaces it.
  The fix is short but easy to miss without reading the signature
  carefully.

---

## Bug 7 — `compute_brightness_stats` consumes the generator four times

- **Difficulty:** hard
- **File / line:** `image_lib.py`, inside `compute_brightness_stats`
  (~line 153)
- **Symptom:** the generator expression `brightnesses` is consumed by
  the first list comprehension; the next three see an empty iterator.
  `np.min([])` raises
  `ValueError: zero-size array to reduction operation minimum`.
- **Expected behavior:** all four stats (`mean`, `min`, `max`, `std`)
  must be computed from the per-image mean brightnesses.
- **Tests that catch it:**
  - `TestComputeBrightnessStats.test_three_images`
  - `TestComputeBrightnessStats.test_single_image`
- **Buggy block:**
  ```python
  brightnesses = (img.mean() for img in images)
  return {
      "mean": float(np.mean([b for b in brightnesses])),
      "min": float(np.min([b for b in brightnesses])),
      ...
  }
  ```
- **Fix:** materialize once.
  ```python
  brightnesses = np.array([img.mean() for img in images])
  return {
      "mean": float(brightnesses.mean()),
      "min": float(brightnesses.min()),
      "max": float(brightnesses.max()),
      "std": float(brightnesses.std()),
  }
  ```
- **Why a candidate falls for it:** the code looks symmetric and
  reasonable — one source of brightness values, four stats over it.
  Generator exhaustion is exactly the kind of bug that confuses people
  who don't trace iterator state carefully. The traceback (`zero-size
  array`) is the strongest hint, but only if you read it.

---

## Initial test state

When all bugs are present:

- **Failing/erroring:** 11 tests
  - 3 in `TestToGrayscale` (bug 1)
  - 1 in `TestThreshold` (bug 2)
  - 1 in `TestCrop` (bug 3)
  - 1 in `TestHistogram` (bug 4)
  - 2 in `TestBrightestPixels` (bug 5)
  - 1 in `TestAddNoise` (bug 6)
  - 2 in `TestComputeBrightnessStats` (bug 7)
- **Passing (guardrails):** 10 tests across `normalize`, `invert`,
  `flip_horizontal`, `apply_mask`, `downsample`, plus
  `TestThreshold.test_basic_split`, `TestCrop.test_correct_region`,
  `TestHistogram.test_full_range_values`,
  `TestAddNoise.test_output_shape_and_range`.

After all fixes: 21/21 passing.
