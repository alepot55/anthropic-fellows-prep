"""Image processing utilities built on NumPy.

Images are represented as numpy arrays:
- Grayscale: 2D array, shape (H, W).
- RGB: 3D array, shape (H, W, 3).
- dtype is typically uint8 (values 0-255) or float64 (values in [0, 1]).
"""

import numpy as np


_RGB_TO_GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114])


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image (H, W, 3) to grayscale (H, W).

    Uses the standard luminance weighting: R=0.299, G=0.587, B=0.114.
    Output dtype follows from the multiplication (typically float64).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("to_grayscale expects an (H, W, 3) RGB image")
    weighted = image * _RGB_TO_GRAY_WEIGHTS
    return np.sum(weighted)


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize image values to [0, 1] as float64.

    For uint8 inputs, divide by 255. For float inputs, perform a
    min-max rescale so the output spans [0, 1] (a constant image
    becomes all zeros).
    """
    if image.dtype == np.uint8:
        return image.astype(np.float64) / 255.0
    arr = image.astype(np.float64)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def threshold(image: np.ndarray, value: float) -> np.ndarray:
    """Binary threshold. Pixels strictly greater than `value` become 1,
    others become 0. Return a uint8 array of the same shape.
    """
    return (image >= value).astype(np.uint8)


def invert(image: np.ndarray) -> np.ndarray:
    """Invert pixel values.

    uint8 -> 255 - x; float in [0, 1] -> 1.0 - x.
    """
    if image.dtype == np.uint8:
        return (255 - image).astype(np.uint8)
    return 1.0 - image


def crop(
    image: np.ndarray, top: int, left: int, height: int, width: int
) -> np.ndarray:
    """Return the rectangular sub-image starting at (top, left).

    The returned array is independent: modifying it must not affect
    the original image.
    """
    if top < 0 or left < 0 or height <= 0 or width <= 0:
        raise ValueError("crop bounds must be non-negative and size > 0")
    if top + height > image.shape[0] or left + width > image.shape[1]:
        raise ValueError("crop region exceeds image bounds")
    return image[top:top + height, left:left + width]


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    """Flip the image left-right."""
    return image[:, ::-1].copy()


def histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """Compute the histogram of a uint8 grayscale image.

    For the default (bins=256) the result has length 256, with index
    i counting the number of pixels equal to i.
    """
    if image.dtype != np.uint8:
        raise TypeError("histogram expects a uint8 image")
    return np.bincount(image.ravel())


def brightest_pixels(image: np.ndarray, n: int) -> list[tuple[int, int]]:
    """Coordinates of the `n` brightest pixels in a grayscale image.

    Returned as (row, col) tuples, ordered from brightest to dimmest.
    For ties, the pixel with the smaller row wins; on equal rows, the
    smaller column wins.
    """
    if image.ndim != 2:
        raise ValueError("brightest_pixels expects a 2D image")
    if n <= 0:
        return []
    flat = image.ravel()
    # Argsort ascending on negated values gives descending brightness
    # with a stable tie-breaker that follows original (row-major) order.
    order = np.argsort(-flat.astype(np.float64), kind="stable")
    top = order[:n]
    return list(top)


def apply_mask(
    image: np.ndarray, mask: np.ndarray, fill: int = 0
) -> np.ndarray:
    """Mask an image: keep pixels where `mask` is True, replace others
    with `fill`. Always returns a new array.
    """
    if mask.shape != image.shape[:2]:
        raise ValueError("mask shape must match image spatial dimensions")
    result = image.copy()
    if image.ndim == 3:
        result[~mask] = fill
    else:
        result[~mask] = fill
    return result


def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by taking every `factor`-th pixel in both spatial dims."""
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if image.ndim == 2:
        return image[::factor, ::factor].copy()
    return image[::factor, ::factor, :].copy()


def add_noise(image: np.ndarray, std: float, seed: int) -> np.ndarray:
    """Add zero-mean Gaussian noise (standard deviation `std`) to a
    float image in [0, 1]. The result is clipped back into [0, 1].

    `seed` controls a local RandomState so calls are reproducible.
    """
    noise = np.random.normal(0.0, std, size=image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0.0, 1.0)


def compute_brightness_stats(images: list) -> dict:
    """Aggregate brightness statistics across a list of grayscale images.

    Returns a dict with keys 'mean', 'min', 'max', 'std', each computed
    over the per-image mean brightness.
    """
    brightnesses = (img.mean() for img in images)
    return {
        "mean": float(np.mean([b for b in brightnesses])),
        "min": float(np.min([b for b in brightnesses])),
        "max": float(np.max([b for b in brightnesses])),
        "std": float(np.std([b for b in brightnesses])),
    }
