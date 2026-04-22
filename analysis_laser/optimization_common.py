from __future__ import annotations
"""Shared geometry helpers for optimization and simple spot simulation.

Coordinate contract:
- physical coordinates use `(x_m, y_m)`
- detector continuous pixel coordinates use `(x_px, y_px)`
- image indexing remains `[row_px, col_px] = [y_px, x_px]`
- displacement arrays use `(dx, dy)` order in either pixel or meter units
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class GridGeometry:
    """Explicit detector-grid geometry.

    Attributes
    ----------
    centers_xy_px : np.ndarray
        Spot centers in detector continuous pixel coordinates `(x_px, y_px)`.
    grid_indices_rowcol : np.ndarray
        Integer `(row, col)` grid slot for each spot.
    grid_rows, grid_cols : int
        Dense grid dimensions.
    """

    centers_xy_px: np.ndarray
    grid_indices_rowcol: np.ndarray
    grid_rows: int
    grid_cols: int


def assert_xy_points(points: np.ndarray, *, name: str = "points") -> np.ndarray:
    """Validate and normalize an `(N, 2)` array of detector points in `(x, y)` order."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2) in (x, y) order, got {arr.shape}")
    return arr


def coerce_global_offsets_xy(
    global_offsets_xy: Union[np.ndarray, Tuple[float, float]],
    *,
    n_frames: Optional[int] = None,
    name: str = "global_offsets_xy",
) -> np.ndarray:
    """Normalize global offsets to an `(n_frames, 2)` array in `(dx, dy)` order."""
    arr = np.asarray(global_offsets_xy, dtype=np.float64)
    if arr.ndim == 1 and arr.shape == (2,):
        if n_frames is None:
            raise ValueError(f"n_frames is required when providing constant step {name}=(dx, dy)")
        k = np.arange(int(n_frames), dtype=np.float64)
        arr = np.column_stack([k * arr[0], k * arr[1]])
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_frames, 2) in (dx, dy) order, got {arr.shape}")
    if n_frames is not None and arr.shape[0] != int(n_frames):
        raise ValueError(f"{name} length {arr.shape[0]} does not match n_frames={n_frames}")
    return arr


def build_neighbor_graph(centers_xy: np.ndarray, spacing: float) -> List[List[int]]:
    """Build a symmetric neighbor graph for spot centers in `(x, y)` order."""
    centers = assert_xy_points(centers_xy, name="centers_xy")
    n_spots = len(centers)
    neighbors: List[List[int]] = [[] for _ in range(n_spots)]
    threshold = 1.5 * float(spacing)

    for i in range(n_spots):
        for j in range(i + 1, n_spots):
            dist = float(np.linalg.norm(centers[i] - centers[j]))
            if dist < threshold:
                neighbors[i].append(j)
                neighbors[j].append(i)

    return neighbors


def assert_row_col_indices(indices: np.ndarray, *, name: str = "indices") -> np.ndarray:
    """Validate and normalize an `(N, 2)` array of `(row, col)` grid indices."""
    arr = np.asarray(indices, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2) in (row, col) order, got {arr.shape}")
    return arr


def build_grid_centers_from_axes(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build `(x, y)` centers and `(row, col)` indices from axis coordinates."""
    x_arr = np.asarray(x_coords, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y_coords, dtype=np.float64).reshape(-1)
    centers = []
    indices = []
    for row, cy in enumerate(y_arr):
        for col, cx in enumerate(x_arr):
            centers.append([cx, cy])
            indices.append([row, col])
    return (
        assert_xy_points(np.asarray(centers, dtype=np.float64), name="centers_xy"),
        assert_row_col_indices(np.asarray(indices, dtype=np.int64), name="grid_indices"),
    )


def build_grid_geometry_from_axes(
    x_coords_px: np.ndarray,
    y_coords_px: np.ndarray,
) -> GridGeometry:
    """Build a `GridGeometry` instance from detector-axis coordinates."""
    centers_xy_px, grid_indices_rowcol = build_grid_centers_from_axes(x_coords_px, y_coords_px)
    return GridGeometry(
        centers_xy_px=centers_xy_px,
        grid_indices_rowcol=grid_indices_rowcol,
        grid_rows=int(len(np.asarray(y_coords_px).reshape(-1))),
        grid_cols=int(len(np.asarray(x_coords_px).reshape(-1))),
    )


def gaussian_radius_for_sigma(sigma: float) -> int:
    """Return the standard ROI radius used for Gaussian spot rendering."""
    return int(3.0 * float(sigma) + 1.0)


def render_gaussian_spots_xy(
    shape: Tuple[int, int],
    centers_xy: np.ndarray,
    sigma: float,
    y_grid: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    """Render Gaussian spots at detector centers given in `(x_px, y_px)` order."""
    centers = assert_xy_points(centers_xy, name="centers_xy")
    h, w = shape
    image = np.zeros(shape, dtype=np.float64)
    sigma_val = float(sigma)
    two_sigma_sq = 2.0 * sigma_val * sigma_val
    radius = gaussian_radius_for_sigma(sigma_val)

    for center_x_px, center_y_px in centers:
        y_min = max(0, int(center_y_px) - radius)
        y_max = min(h, int(center_y_px) + radius + 1)
        x_min = max(0, int(center_x_px) - radius)
        x_max = min(w, int(center_x_px) + radius + 1)
        if y_min >= y_max or x_min >= x_max:
            continue

        y_roi = y_grid[y_min:y_max, x_min:x_max]
        x_roi = x_grid[y_min:y_max, x_min:x_max]
        gaussian_roi = np.exp(
            -((y_roi - center_y_px) ** 2 + (x_roi - center_x_px) ** 2) / two_sigma_sq
        )
        image[y_min:y_max, x_min:x_max] += gaussian_roi

    return image


def gaussian_shift_gradients_xy(
    diff_image: np.ndarray,
    centers_xy: np.ndarray,
    sigma: float,
    y_grid: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    """Compute analytical `(dx, dy)` gradients for Gaussian spot shifts.

    Parameters
    ----------
    diff_image : np.ndarray
        Residual image `predicted - observed`.
    centers_xy : np.ndarray
        Spot centers in `(x_px, y_px)` order.
    sigma : float
        Gaussian sigma in pixels.
    y_grid, x_grid : np.ndarray
        Dense detector coordinate grids as returned by `np.indices(shape)`.
    """
    diff = np.asarray(diff_image, dtype=np.float64)
    centers = assert_xy_points(centers_xy, name="centers_xy")
    grads = np.zeros((len(centers), 2), dtype=np.float64)
    sigma_sq = float(sigma) ** 2
    radius = gaussian_radius_for_sigma(float(sigma))
    h, w = diff.shape

    for idx, (center_x_px, center_y_px) in enumerate(centers):
        y_min = max(0, int(center_y_px) - radius)
        y_max = min(h, int(center_y_px) + radius + 1)
        x_min = max(0, int(center_x_px) - radius)
        x_max = min(w, int(center_x_px) + radius + 1)
        if y_min >= y_max or x_min >= x_max:
            continue

        y_roi = y_grid[y_min:y_max, x_min:x_max]
        x_roi = x_grid[y_min:y_max, x_min:x_max]
        diff_roi = diff[y_min:y_max, x_min:x_max]
        gaussian_roi = np.exp(
            -((y_roi - center_y_px) ** 2 + (x_roi - center_x_px) ** 2) / (2.0 * sigma_sq)
        )
        common_term = 2.0 * diff_roi * gaussian_roi
        grads[idx, 0] = np.sum(common_term * (x_roi - center_x_px)) / sigma_sq
        grads[idx, 1] = np.sum(common_term * (y_roi - center_y_px)) / sigma_sq

    return grads


def build_frame_centers_sequence_xy(
    ideal_centers_xy: np.ndarray,
    global_offsets_xy: np.ndarray,
    local_shifts_xy: np.ndarray,
) -> np.ndarray:
    """Build per-frame detector centers in `(x_px, y_px)` order.

    Parameters
    ----------
    ideal_centers_xy : np.ndarray
        Base spot centers with shape `(n_spots, 2)`.
    global_offsets_xy : np.ndarray
        Per-frame global offsets with shape `(n_frames, 2)`.
    local_shifts_xy : np.ndarray
        Per-frame local shifts with shape `(n_frames, n_spots, 2)`.
    """
    centers = assert_xy_points(ideal_centers_xy, name="ideal_centers_xy")
    offsets = assert_xy_points(global_offsets_xy, name="global_offsets_xy")
    local = np.asarray(local_shifts_xy, dtype=np.float64)
    if local.ndim != 3 or local.shape[2] != 2:
        raise ValueError(
            f"local_shifts_xy must have shape (n_frames, n_spots, 2) in (dx, dy) order, got {local.shape}"
        )
    if local.shape[0] != offsets.shape[0] or local.shape[1] != centers.shape[0]:
        raise ValueError(
            "local_shifts_xy shape must match global_offsets_xy and ideal_centers_xy "
            f"got local={local.shape}, offsets={offsets.shape}, centers={centers.shape}"
        )
    return centers[np.newaxis, :, :] + offsets[:, np.newaxis, :] + local


def render_gaussian_sequence_xy(
    shape: Tuple[int, int],
    ideal_centers_xy: np.ndarray,
    global_offsets_xy: np.ndarray,
    local_shifts_xy: np.ndarray,
    sigma: float,
    y_grid: np.ndarray,
    x_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render an accumulated Gaussian image and return per-frame centers."""
    frame_centers = build_frame_centers_sequence_xy(
        ideal_centers_xy=ideal_centers_xy,
        global_offsets_xy=global_offsets_xy,
        local_shifts_xy=local_shifts_xy,
    )
    accumulated = np.zeros(shape, dtype=np.float64)
    for centers_xy in frame_centers:
        accumulated += render_gaussian_spots_xy(
            shape=shape,
            centers_xy=centers_xy,
            sigma=sigma,
            y_grid=y_grid,
            x_grid=x_grid,
    )
    return accumulated, frame_centers


def estimate_spot_fwhm_sigma_from_image(
    image: np.ndarray,
    spacing: float,
    *,
    percentile: float = 80.0,
    smooth_sigma: float = 1.0,
    max_samples: int = 20,
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate median spot FWHM/sigma from an image containing separated spots."""
    arr = np.asarray(image, dtype=np.float64)
    spacing_px = float(spacing)
    if arr.ndim != 2:
        raise ValueError(f"image must be 2-D, got shape {arr.shape}")
    if spacing_px <= 0:
        raise ValueError(f"spacing must be positive, got {spacing}")

    smooth = ndi.gaussian_filter(arr, sigma=float(smooth_sigma))
    threshold = np.percentile(smooth, float(percentile))
    neighborhood = max(1, int(spacing_px * 0.5))
    local_max = ndi.maximum_filter(smooth, size=neighborhood)
    spot_mask = (smooth == local_max) & (smooth > threshold)
    positions = np.array(np.where(spot_mask)).T

    if len(positions) < 3:
        return None, None

    h, w = arr.shape
    roi_radius = max(1, int(spacing_px * 0.4))
    fwhm_values: list[float] = []

    for spot_row_px, spot_col_px in positions[: min(int(max_samples), len(positions))]:
        row_min = max(0, int(spot_row_px) - roi_radius)
        row_max = min(h, int(spot_row_px) + roi_radius + 1)
        col_min = max(0, int(spot_col_px) - roi_radius)
        col_max = min(w, int(spot_col_px) + roi_radius + 1)

        roi = arr[row_min:row_max, col_min:col_max]
        if roi.size == 0:
            continue

        profile = roi[roi.shape[0] // 2, :]
        if len(profile) < 3:
            continue

        peak_val = float(np.max(profile))
        half_max = (peak_val + float(np.min(profile))) / 2.0
        above_half = profile > half_max
        if not np.any(above_half):
            continue

        indices = np.where(above_half)[0]
        fwhm_px = float(indices[-1] - indices[0])
        if 1.0 < fwhm_px < spacing_px * 0.5:
            fwhm_values.append(fwhm_px)

    if not fwhm_values:
        return None, None

    median_fwhm_px = float(np.median(np.asarray(fwhm_values, dtype=np.float64)))
    sigma_px = median_fwhm_px / 2.355
    return median_fwhm_px, sigma_px


def extract_centered_patch_xy(
    image: np.ndarray,
    center_xy: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Extract a square patch centered on a possibly non-integer `(x_px, y_px)` point."""
    arr = np.asarray(image, dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    center_x_px = float(center[0])
    center_y_px = float(center[1])
    r = int(radius)
    patch = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.float64)

    center_x_i = int(round(center_x_px))
    center_y_i = int(round(center_y_px))
    x0 = center_x_i - r
    x1 = center_x_i + r + 1
    y0 = center_y_i - r
    y1 = center_y_i + r + 1

    xs0 = max(0, x0)
    xs1 = min(arr.shape[1], x1)
    ys0 = max(0, y0)
    ys1 = min(arr.shape[0], y1)
    if xs0 >= xs1 or ys0 >= ys1:
        return patch

    px0 = xs0 - x0
    px1 = px0 + (xs1 - xs0)
    py0 = ys0 - y0
    py1 = py0 + (ys1 - ys0)
    patch[py0:py1, px0:px1] = arr[ys0:ys1, xs0:xs1]

    frac_dx_px = center_x_px - center_x_i
    frac_dy_px = center_y_px - center_y_i
    if abs(frac_dx_px) > 1e-9 or abs(frac_dy_px) > 1e-9:
        patch = ndi.shift(
            patch,
            shift=(-frac_dy_px, -frac_dx_px),
            order=1,
            mode='constant',
            cval=0.0,
            prefilter=False,
        )
    return patch


def compute_patch_roi_slices_xy(
    image_shape: Tuple[int, int],
    center_xy: np.ndarray,
    radius: int,
) -> Optional[Tuple[int, int, int, int, int, int, int, int]]:
    """Return image/patch ROI bounds for a patch centered at `(x_px, y_px)`."""
    h, w = image_shape
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    center_x_i = int(round(float(center[0])))
    center_y_i = int(round(float(center[1])))
    r = int(radius)

    x0 = center_x_i - r
    x1 = center_x_i + r + 1
    y0 = center_y_i - r
    y1 = center_y_i + r + 1

    xs0 = max(0, x0)
    xs1 = min(w, x1)
    ys0 = max(0, y0)
    ys1 = min(h, y1)
    if xs0 >= xs1 or ys0 >= ys1:
        return None

    px0 = xs0 - x0
    px1 = px0 + (xs1 - xs0)
    py0 = ys0 - y0
    py1 = py0 + (ys1 - ys0)
    return ys0, ys1, xs0, xs1, py0, py1, px0, px1


def add_shifted_template_patch_xy(
    image: np.ndarray,
    template_patch: np.ndarray,
    base_center_xy: np.ndarray,
    shift_xy: np.ndarray,
    radius: int,
) -> Optional[Tuple[int, int, int, int, int, int, int, int]]:
    """Shift a template patch by `(dx_px, dy_px)` and accumulate it into an image."""
    center = np.asarray(base_center_xy, dtype=np.float64).reshape(2)
    shift = np.asarray(shift_xy, dtype=np.float64).reshape(2)
    roi = compute_patch_roi_slices_xy(
        image_shape=image.shape,
        center_xy=center,
        radius=radius,
    )
    if roi is None:
        return None
    ys0, ys1, xs0, xs1, py0, py1, px0, px1 = roi

    shifted_patch = ndi.shift(
        np.asarray(template_patch, dtype=np.float64),
        shift=(float(shift[1]), float(shift[0])),
        order=1,
        mode='constant',
        cval=0.0,
        prefilter=False,
    )
    image[ys0:ys1, xs0:xs1] += shifted_patch[py0:py1, px0:px1]
    return roi


def shifted_template_patch_gradient_xy(
    diff_image: np.ndarray,
    grad_x_patch: np.ndarray,
    grad_y_patch: np.ndarray,
    base_center_xy: np.ndarray,
    shift_xy: np.ndarray,
    radius: int,
    roi: Optional[Tuple[int, int, int, int, int, int, int, int]] = None,
) -> np.ndarray:
    """Compute `(dx, dy)` gradient contribution for one shifted template patch."""
    roi_bounds = roi
    if roi_bounds is None:
        roi_bounds = compute_patch_roi_slices_xy(
            image_shape=np.asarray(diff_image).shape,
            center_xy=base_center_xy,
            radius=radius,
        )
    if roi_bounds is None:
        return np.zeros(2, dtype=np.float64)

    ys0, ys1, xs0, xs1, py0, py1, px0, px1 = roi_bounds
    shift = np.asarray(shift_xy, dtype=np.float64).reshape(2)
    diff_roi = np.asarray(diff_image, dtype=np.float64)[ys0:ys1, xs0:xs1]

    grad_x_shifted = -ndi.shift(
        np.asarray(grad_x_patch, dtype=np.float64),
        shift=(float(shift[1]), float(shift[0])),
        order=1,
        mode='constant',
        cval=0.0,
        prefilter=False,
    )
    grad_y_shifted = -ndi.shift(
        np.asarray(grad_y_patch, dtype=np.float64),
        shift=(float(shift[1]), float(shift[0])),
        order=1,
        mode='constant',
        cval=0.0,
        prefilter=False,
    )
    return np.array(
        [
            2.0 * np.sum(diff_roi * grad_x_shifted[py0:py1, px0:px1]),
            2.0 * np.sum(diff_roi * grad_y_shifted[py0:py1, px0:px1]),
        ],
        dtype=np.float64,
    )
