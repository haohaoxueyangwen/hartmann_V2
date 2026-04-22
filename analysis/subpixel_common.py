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


def quadratic_surface_peak_offset_2d(
    scores: np.ndarray,
    peak_y: int,
    peak_x: int,
    *,
    radius: int = 2,
) -> Tuple[float, float]:
    """Fit a 2D quadratic surface around a peak and return `(dx_px, dy_px)`.

    The fitted model is:

        z = a x^2 + b y^2 + c x y + d x + e y + f

    where `(x, y) = (0, 0)` corresponds to the integer peak location.
    """
    arr = np.asarray(scores, dtype=np.float64)
    h, w = arr.shape
    fit_radius = max(1, int(radius))

    y0 = max(0, int(peak_y) - fit_radius)
    y1 = min(h, int(peak_y) + fit_radius + 1)
    x0 = max(0, int(peak_x) - fit_radius)
    x1 = min(w, int(peak_x) + fit_radius + 1)
    if (y1 - y0) < 3 or (x1 - x0) < 3:
        return 0.0, 0.0

    yy, xx = np.mgrid[y0:y1, x0:x1]
    x_rel = (xx - int(peak_x)).astype(np.float64).ravel()
    y_rel = (yy - int(peak_y)).astype(np.float64).ravel()
    z = arr[y0:y1, x0:x1].ravel()
    if z.size < 6:
        return 0.0, 0.0

    design = np.column_stack((
        x_rel * x_rel,
        y_rel * y_rel,
        x_rel * y_rel,
        x_rel,
        y_rel,
        np.ones_like(x_rel),
    ))

    try:
        coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0

    a, b, c, d, e, _ = coeffs
    hessian = np.array([[2.0 * a, c], [c, 2.0 * b]], dtype=np.float64)
    grad = np.array([d, e], dtype=np.float64)

    try:
        if abs(float(np.linalg.det(hessian))) < 1e-12:
            return 0.0, 0.0
        vertex = -np.linalg.solve(hessian, grad)
    except np.linalg.LinAlgError:
        return 0.0, 0.0

    dx = float(np.clip(vertex[0], -fit_radius, fit_radius))
    dy = float(np.clip(vertex[1], -fit_radius, fit_radius))
    if not np.isfinite(dx) or not np.isfinite(dy):
        return 0.0, 0.0
    return dx, dy


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
    elif method in {"quadratic2d", "quadratic_2d"}:
        return quadratic_surface_peak_offset_2d(arr, peak_y, peak_x, radius=2)
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
