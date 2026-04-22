"""Shared subpixel peak-refinement helpers.

All helpers operate on image-like score arrays indexed as `[row, col] = [y, x]`
and return detector displacements in `(dx_px, dy_px)` order.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def _axis_offset_from_three_samples(
    samples: np.ndarray,
    *,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> float:
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if values.size != 3:
        return 0.0
    if transform is not None:
        values = transform(values)
    denom = values[0] - 2.0 * values[1] + values[2]
    if abs(float(denom)) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (values[0] - values[2]) / denom, -1.0, 1.0))


def parabolic_subpixel_offset(samples: np.ndarray) -> float:
    """Refine a 1D peak offset from three samples using a quadratic fit."""
    return _axis_offset_from_three_samples(samples)


def gaussian_subpixel_offset(samples: np.ndarray) -> float:
    """Refine a 1D peak offset in log-space, suitable for PSF-like peaks."""
    return _axis_offset_from_three_samples(
        samples,
        transform=lambda vals: np.log(np.clip(vals, 1e-12, None)),
    )


def subpixel_peak_offset_2d(
    scores: np.ndarray,
    peak_y: int,
    peak_x: int,
    *,
    method: str = "parabolic",
) -> Tuple[float, float]:
    """Refine a 2D peak and return `(dx_px, dy_px)` offsets around the integer peak."""
    arr = np.asarray(scores, dtype=np.float64)
    h, w = arr.shape
    if method == "parabolic":
        refine = parabolic_subpixel_offset
    elif method == "gaussian":
        refine = gaussian_subpixel_offset
    else:
        raise ValueError(f"unknown subpixel method {method!r}")

    if 1 <= peak_x < w - 1:
        dx = refine(arr[peak_y, peak_x - 1 : peak_x + 2])
    else:
        dx = 0.0
    if 1 <= peak_y < h - 1:
        dy = refine(arr[peak_y - 1 : peak_y + 2, peak_x])
    else:
        dy = 0.0
    return dx, dy

