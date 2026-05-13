"""Unit tests for image_lib.

Some tests are expected to fail until the bugs in image_lib.py are fixed.
Other tests act as guardrails: they currently pass, and they should keep
passing after your fixes.
"""

import unittest

import numpy as np

from image_lib import (
    add_noise,
    apply_mask,
    brightest_pixels,
    compute_brightness_stats,
    crop,
    downsample,
    flip_horizontal,
    histogram,
    invert,
    normalize,
    threshold,
    to_grayscale,
)


class TestToGrayscale(unittest.TestCase):
    def test_output_shape(self):
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        result = to_grayscale(image)
        self.assertEqual(result.shape, (10, 20))

    def test_uniform_image(self):
        image = np.full((5, 5, 3), 200, dtype=np.uint8)
        result = to_grayscale(image)
        self.assertEqual(result.shape, (5, 5))
        np.testing.assert_array_almost_equal(
            result, np.full((5, 5), 200.0), decimal=3
        )

    def test_known_pixel_values(self):
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        image[1, 1] = [100, 200, 50]
        result = to_grayscale(image)
        self.assertEqual(result.shape, (3, 3))
        expected = 0.299 * 100 + 0.587 * 200 + 0.114 * 50
        self.assertAlmostEqual(float(result[1, 1]), expected, places=3)
        self.assertAlmostEqual(float(result[0, 0]), 0.0, places=3)


class TestNormalize(unittest.TestCase):
    def test_uint8_input(self):
        image = np.array([[0, 128, 255]], dtype=np.uint8)
        result = normalize(image)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_array_almost_equal(
            result, [[0.0, 128 / 255, 1.0]], decimal=6
        )

    def test_float_input_rescales(self):
        image = np.array([[2.0, 4.0, 6.0]], dtype=np.float64)
        result = normalize(image)
        np.testing.assert_array_almost_equal(result, [[0.0, 0.5, 1.0]])


class TestThreshold(unittest.TestCase):
    def test_basic_split(self):
        image = np.array([[10, 200], [50, 240]], dtype=np.uint8)
        result = threshold(image, 100)
        np.testing.assert_array_equal(result, [[0, 1], [0, 1]])
        self.assertEqual(result.dtype, np.uint8)

    def test_boundary_strictly_greater(self):
        # Pixels equal to the threshold must NOT be selected.
        image = np.array([[100, 100, 101]], dtype=np.uint8)
        result = threshold(image, 100)
        np.testing.assert_array_equal(result, [[0, 0, 1]])


class TestInvert(unittest.TestCase):
    def test_uint8_invert(self):
        image = np.array([[0, 100, 255]], dtype=np.uint8)
        result = invert(image)
        np.testing.assert_array_equal(result, [[255, 155, 0]])
        self.assertEqual(result.dtype, np.uint8)


class TestCrop(unittest.TestCase):
    def test_correct_region(self):
        image = np.arange(25, dtype=np.uint8).reshape(5, 5)
        result = crop(image, top=1, left=1, height=2, width=3)
        np.testing.assert_array_equal(result, [[6, 7, 8], [11, 12, 13]])

    def test_modifying_crop_does_not_affect_original(self):
        image = np.arange(25, dtype=np.uint8).reshape(5, 5)
        original = image.copy()
        cropped = crop(image, top=1, left=1, height=2, width=3)
        cropped[0, 0] = 99
        np.testing.assert_array_equal(image, original)


class TestFlipHorizontal(unittest.TestCase):
    def test_basic_flip(self):
        image = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        result = flip_horizontal(image)
        np.testing.assert_array_equal(result, [[3, 2, 1], [6, 5, 4]])


class TestHistogram(unittest.TestCase):
    def test_full_range_values(self):
        image = np.array([0, 100, 255, 128, 255, 0], dtype=np.uint8)
        result = histogram(image)
        self.assertEqual(len(result), 256)
        self.assertEqual(int(result[0]), 2)
        self.assertEqual(int(result[100]), 1)
        self.assertEqual(int(result[255]), 2)

    def test_partial_range_values(self):
        # Image values stay well below 255; histogram must still have
        # length == bins (256), with zeros in the unused buckets.
        image = np.array([0, 50, 100, 50, 100, 100], dtype=np.uint8)
        result = histogram(image)
        self.assertEqual(len(result), 256)
        self.assertEqual(int(result[100]), 3)
        self.assertEqual(int(result[200]), 0)


class TestBrightestPixels(unittest.TestCase):
    def test_simple_5x5(self):
        image = np.array(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ],
            dtype=np.uint8,
        )
        result = brightest_pixels(image, 3)
        self.assertEqual(result, [(4, 4), (4, 3), (4, 2)])

    def test_tie_breaker_prefers_smaller_row_then_col(self):
        # Three pixels share the maximum value 9.
        # Expected order: (0, 1), (1, 0), (1, 1) — smaller row first,
        # then smaller column for ties.
        image = np.array([[5, 9], [9, 9]], dtype=np.uint8)
        result = brightest_pixels(image, 3)
        self.assertEqual(result, [(0, 1), (1, 0), (1, 1)])


class TestApplyMask(unittest.TestCase):
    def test_basic_mask(self):
        image = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        mask = np.array([[True, False], [False, True]])
        result = apply_mask(image, mask, fill=0)
        np.testing.assert_array_equal(result, [[10, 0], [0, 40]])


class TestDownsample(unittest.TestCase):
    def test_factor_two(self):
        image = np.arange(16, dtype=np.uint8).reshape(4, 4)
        result = downsample(image, 2)
        np.testing.assert_array_equal(result, [[0, 2], [8, 10]])


class TestAddNoise(unittest.TestCase):
    def test_output_shape_and_range(self):
        image = np.full((10, 10), 0.5, dtype=np.float64)
        result = add_noise(image, std=0.05, seed=0)
        self.assertEqual(result.shape, (10, 10))
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_deterministic_same_seed(self):
        image = np.full((8, 8), 0.5, dtype=np.float64)
        a = add_noise(image, std=0.1, seed=42)
        b = add_noise(image, std=0.1, seed=42)
        np.testing.assert_array_equal(a, b)


class TestComputeBrightnessStats(unittest.TestCase):
    def test_three_images(self):
        img1 = np.full((4, 4), 0.2, dtype=np.float64)
        img2 = np.full((4, 4), 0.5, dtype=np.float64)
        img3 = np.full((4, 4), 0.8, dtype=np.float64)
        stats = compute_brightness_stats([img1, img2, img3])
        self.assertAlmostEqual(stats["mean"], 0.5, places=6)
        self.assertAlmostEqual(stats["min"], 0.2, places=6)
        self.assertAlmostEqual(stats["max"], 0.8, places=6)
        # Population std over [0.2, 0.5, 0.8] = sqrt(0.06) ~= 0.244949
        self.assertAlmostEqual(stats["std"], np.sqrt(0.06), places=6)

    def test_single_image(self):
        img = np.full((3, 3), 0.4, dtype=np.float64)
        stats = compute_brightness_stats([img])
        self.assertAlmostEqual(stats["mean"], 0.4, places=6)
        self.assertAlmostEqual(stats["min"], 0.4, places=6)
        self.assertAlmostEqual(stats["max"], 0.4, places=6)
        self.assertAlmostEqual(stats["std"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
