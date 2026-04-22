from __future__ import annotations
"""
Displacement Measurement Methods

Provides two classic methods for estimating sub-beam displacements
from Hartmann/Shack-Hartmann sensor images:

1. Centroid method — center-of-mass shift
2. Cross-correlation — FFT-based correlation peak

Also contains helper functions for beamlet detection, patch extraction,
and subpixel peak refinement.

Origin: Extracted from Dev_1 analysis/hartmann_analysis.py
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import maximum_filter, gaussian_filter, median_filter
from scipy import sparse
from scipy.sparse.linalg import lsqr as sparse_lsqr
from typing import Dict, List, Tuple, Optional

from analysis.subpixel_common import (
    quadratic_surface_peak_offset_2d,
    subpixel_peak_offset_2d,
)


# =============================================================================
# Beamlet Detection
# =============================================================================

def detect_beamlets(image: np.ndarray, rows: int, cols: int):
    """
    Detect beamlet positions in a Hartmann image.

    Returns ROI boxes plus centroid positions in `(x_px, y_px)` order.
    """
    expected = rows * cols
    smooth = gaussian_filter(image.astype(float), sigma=1.0)
    window = max(7, int(min(image.shape) / (3 * max(rows, cols))))
    if window % 2 == 0:
        window += 1
    local_max = maximum_filter(smooth, size=window) == smooth
    threshold = smooth.mean() + 0.3 * smooth.std()
    candidates = np.argwhere(local_max & (smooth >= threshold))

    if candidates.shape[0] < expected:
        boxes = compute_cells(image.shape, rows, cols)
        return boxes, compute_centroids(image, boxes, rows, cols)

    intensities = smooth[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(intensities)[::-1][:expected]
    selected = candidates[order]

    if selected.shape[0] < expected:
        boxes = compute_cells(image.shape, rows, cols)
        return boxes, compute_centroids(image, boxes, rows, cols)

    selected = selected[np.argsort(selected[:, 0])]
    # patch 半径：确保覆盖完整艾里斑 + 位移余量
    # 可通过 config_recover.XCORR_PATCH_HALF 覆盖
    try:
        from config_recover import XCORR_PATCH_HALF
        patch_half = XCORR_PATCH_HALF
    except (ImportError, AttributeError):
        patch_half = max(15, int(min(image.shape) / (3 * max(rows, cols))))
    max_patch_size = max(1, min(int(image.shape[0]), int(image.shape[1])))
    patch_size = min(patch_half * 2 + 1, max_patch_size)
    if patch_size % 2 == 0 and patch_size > 1:
        patch_size -= 1
    patch_half = patch_size // 2

    boxes: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    centers = np.zeros((rows, cols, 2), dtype=float)

    for r in range(rows):
        row_slice = selected[r * cols:(r + 1) * cols]
        row_slice = row_slice[np.argsort(row_slice[:, 1])]
        if row_slice.shape[0] < cols:
            fallback = compute_cells(image.shape, rows, cols)
            return fallback, compute_centroids(image, fallback, rows, cols)
        for c in range(cols):
            y, x = row_slice[c]
            max_y0 = max(0, image.shape[0] - patch_size)
            max_x0 = max(0, image.shape[1] - patch_size)
            y0 = int(np.clip(y - patch_half, 0, max_y0))
            x0 = int(np.clip(x - patch_half, 0, max_x0))
            y1 = y0 + patch_size
            x1 = x0 + patch_size
            boxes.append(((y0, y1), (x0, x1)))
            centers[r, c, 0] = x
            centers[r, c, 1] = y

    centroids = compute_centroids(image, boxes, rows, cols)
    return boxes, centroids


def compute_cells(shape: Tuple[int, int], rows: int, cols: int):
    """Compute uniform grid cells as fallback."""
    height, width = shape
    cell_h = height // rows
    cell_w = width // cols
    boxes = []
    for r in range(rows):
        y0 = r * cell_h
        y1 = (r + 1) * cell_h if r < rows - 1 else height
        for c in range(cols):
            x0 = c * cell_w
            x1 = (c + 1) * cell_w if c < cols - 1 else width
            boxes.append(((y0, y1), (x0, x1)))
    return boxes


def compute_centroids(image: np.ndarray, boxes, rows: int, cols: int) -> np.ndarray:
    """Compute center-of-mass centroids for each beamlet."""
    centroids = np.zeros((rows, cols, 2), dtype=float)
    for idx, box in enumerate(boxes):
        (y0, y1), (x0, x1) = box
        patch = image[y0:y1, x0:x1]
        yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
        total = float(patch.sum())
        if total <= 0:
            cy = (patch.shape[0] - 1) / 2.0
            cx = (patch.shape[1] - 1) / 2.0
        else:
            cy = float((patch * yy).sum() / total)
            cx = float((patch * xx).sum() / total)
        row = idx // cols
        col = idx % cols
        centroids[row, col, 0] = x0 + cx
        centroids[row, col, 1] = y0 + cy
    return centroids


def extract_patches(image: np.ndarray, boxes, rows: int, cols: int) -> List[List[np.ndarray]]:
    """Extract image patches for each beamlet."""
    patches: List[List[Optional[np.ndarray]]] = [[None for _ in range(cols)] for _ in range(rows)]
    for idx, box in enumerate(boxes):
        (y0, y1), (x0, x1) = box
        row = idx // cols
        col = idx % cols
        patches[row][col] = image[y0:y1, x0:x1]
    return patches


# =============================================================================
# Method 1: Centroid Shifts
# =============================================================================

def centroid_shifts(ref_centroids: np.ndarray, tgt_centroids: np.ndarray) -> np.ndarray:
    """
    Compute displacements from centroid differences.

    Parameters
    ----------
    ref_centroids, tgt_centroids : ndarray of shape (rows, cols, 2)

    Returns
    -------
    shifts : ndarray of shape (rows, cols, 2)
        Returned in `(dx_px, dy_px)` order.
    """
    return tgt_centroids - ref_centroids


# =============================================================================
# Method 2: Cross-Correlation
# =============================================================================

def _estimate_patch_background(patch: np.ndarray, *, percentile: float) -> float:
    """Estimate a robust local background level from a patch."""
    arr = np.asarray(patch, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, float(percentile)))


def _prepare_guided_xcorr_patch(
    patch: np.ndarray,
    *,
    bg_percentile: float,
    gamma: float,
) -> np.ndarray:
    """Suppress local background while preserving the native spot morphology."""
    arr = np.asarray(patch, dtype=float).copy()
    arr -= _estimate_patch_background(arr, percentile=bg_percentile)
    arr[arr < 0.0] = 0.0
    if gamma != 1.0:
        arr = np.power(arr, gamma, where=arr > 0.0, out=np.zeros_like(arr))
    return arr


def _build_template_weight(template_patch: np.ndarray, *, weight_alpha: float) -> np.ndarray:
    """Build a smooth, template-driven weight map for local ZNCC."""
    arr = np.asarray(template_patch, dtype=float)
    vmax = float(np.max(arr)) if arr.size else 0.0
    if vmax <= 1e-12:
        return np.ones_like(arr, dtype=float)

    norm = np.clip(arr / vmax, 0.0, None)
    weight = 0.15 + 0.85 * np.power(norm, float(weight_alpha))
    return np.clip(weight, 1e-6, None)


def _score_map_peak_metrics(
    score_map: np.ndarray,
    peak_rc: Tuple[int, int],
) -> Tuple[float, float]:
    """Measure peak isolation and local sharpness for a correlation score map."""
    scores = np.asarray(score_map, dtype=float)
    py, px = int(peak_rc[0]), int(peak_rc[1])
    peak_score = float(scores[py, px])

    y0 = max(0, py - 1)
    y1 = min(scores.shape[0], py + 2)
    x0 = max(0, px - 1)
    x1 = min(scores.shape[1], px + 2)

    local = scores[y0:y1, x0:x1].copy()
    local[py - y0, px - x0] = -np.inf
    local_neighbor = float(np.max(local)) if np.isfinite(local).any() else peak_score
    peak_sharpness = peak_score - local_neighbor

    competitors = scores.copy()
    competitors[y0:y1, x0:x1] = -np.inf
    second_peak = float(np.max(competitors)) if np.isfinite(competitors).any() else local_neighbor
    peak_margin = peak_score - second_peak
    return float(peak_margin), float(peak_sharpness)


def _xcorr_confidence_from_metrics(
    scores: np.ndarray,
    peak_margins: np.ndarray,
    peak_sharpness: np.ndarray,
    *,
    min_score: float,
    min_peak_margin: float,
    min_peak_sharpness: float,
) -> np.ndarray:
    """Map guided-correlation peak metrics to a stable 0..1 confidence field."""
    scores = np.asarray(scores, dtype=float)
    peak_margins = np.asarray(peak_margins, dtype=float)
    peak_sharpness = np.asarray(peak_sharpness, dtype=float)

    score_hi = max(min_score + 1e-6, min(0.95, max(0.30, min_score * 2.5)))
    margin_hi = max(min_peak_margin + 1e-6, min_peak_margin * 4.0)
    sharp_hi = max(min_peak_sharpness + 1e-6, min_peak_sharpness * 4.0)

    score_norm = np.clip((scores - min_score) / (score_hi - min_score), 0.0, 1.0)
    margin_norm = np.clip((peak_margins - min_peak_margin) / (margin_hi - min_peak_margin), 0.0, 1.0)
    sharp_norm = np.clip(
        (peak_sharpness - min_peak_sharpness) / (sharp_hi - min_peak_sharpness),
        0.0,
        1.0,
    )

    confidence = np.power(
        np.clip(score_norm * margin_norm * sharp_norm, 0.0, 1.0),
        1.0 / 3.0,
    )
    confidence[~np.isfinite(confidence)] = 0.0
    return confidence


def _guided_local_zncc_peak_shift(
    *,
    ref_box: Tuple[Tuple[int, int], Tuple[int, int]],
    ref_patch: np.ndarray,
    tgt_patch: np.ndarray,
    tgt_image: np.ndarray,
    search_radius: int,
    bg_percentile: float,
    gamma: float,
    weight_alpha: float,
    subpixel_method: str,
    subpixel_radius: int,
) -> Tuple[float, float, float, float, float]:
    """Search a constrained local ROI with weighted ZNCC."""
    (y0, y1), (x0, x1) = ref_box
    ny, nx = tgt_image.shape
    available_radius = int(min(
        max(0, int(search_radius)),
        int(y0),
        int(x0),
        int(ny - y1),
        int(nx - x1),
    ))
    if available_radius < 1:
        shift_x_px, shift_y_px = _xcorr_peak_shift(ref_patch, tgt_patch)
        return float(shift_x_px), float(shift_y_px), -1.0, 0.0, 0.0

    ys0 = int(y0 - available_radius)
    ys1 = int(y1 + available_radius)
    xs0 = int(x0 - available_radius)
    xs1 = int(x1 + available_radius)

    ref_proc = _prepare_guided_xcorr_patch(
        ref_patch,
        bg_percentile=bg_percentile,
        gamma=gamma,
    )
    search_roi = _prepare_guided_xcorr_patch(
        tgt_image[ys0:ys1, xs0:xs1],
        bg_percentile=bg_percentile,
        gamma=gamma,
    )
    weight = _build_template_weight(ref_proc, weight_alpha=weight_alpha)

    w = np.asarray(weight, dtype=float)
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        shift_x_px, shift_y_px = _xcorr_peak_shift(ref_patch, tgt_patch)
        return float(shift_x_px), float(shift_y_px), -1.0, 0.0, 0.0

    ref_mean = float(np.sum(w * ref_proc) / w_sum)
    ref_zero = ref_proc - ref_mean
    ref_weighted = ref_zero * w
    ref_var = float(np.sum(w * ref_zero * ref_zero))
    if ref_var <= 1e-12:
        shift_x_px, shift_y_px = _xcorr_peak_shift(ref_patch, tgt_patch)
        return float(shift_x_px), float(shift_y_px), -1.0, 0.0, 0.0

    numer = fftconvolve(search_roi, ref_weighted[::-1, ::-1], mode="valid")
    local_sum = fftconvolve(search_roi, w[::-1, ::-1], mode="valid")
    local_sum_sq = fftconvolve(search_roi * search_roi, w[::-1, ::-1], mode="valid")
    local_var = np.maximum(local_sum_sq - (local_sum * local_sum) / w_sum, 0.0)
    denom = np.sqrt(ref_var * local_var)

    scores = np.full_like(numer, -1.0, dtype=float)
    valid = denom > 1e-12
    scores[valid] = numer[valid] / denom[valid]

    peak = np.unravel_index(np.argmax(scores), scores.shape)
    best_score = float(scores[peak])
    best_dy = int(peak[0] - available_radius)
    best_dx = int(peak[1] - available_radius)
    peak_margin, peak_sharpness = _score_map_peak_metrics(scores, peak)

    if str(subpixel_method).lower() in {"quadratic2d", "quadratic_2d"}:
        sub_x, sub_y = quadratic_surface_peak_offset_2d(
            scores,
            peak[0],
            peak[1],
            radius=max(1, int(subpixel_radius)),
        )
    else:
        sub_x, sub_y = subpixel_peak_offset_2d(
            scores,
            peak[0],
            peak[1],
            method=subpixel_method,
        )

    return (
        float(best_dx + sub_x),
        float(best_dy + sub_y),
        float(best_score),
        float(peak_margin),
        float(peak_sharpness),
    )


def cross_correlation_shifts(
    ref_patches,
    tgt_patches,
    correction_factor: float = 1.0,
    *,
    ref_boxes=None,
    tgt_image: np.ndarray | None = None,
    search_radius: int | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | Dict[str, np.ndarray]:
    """
    Estimate shifts using constrained local correlation with subpixel refinement.

    Parameters
    ----------
    ref_patches, tgt_patches : List[List[ndarray]]
        Patches for each (row, col) beamlet
    correction_factor : float
        Multiplicative factor applied to the final shifts.
    ref_boxes : optional list of ROI boxes
        When provided together with ``tgt_image``, enables guided local ZNCC
        within ``search_radius`` instead of unconstrained same-box FFT xcorr.
    tgt_image : ndarray or None
        Full target image used by guided local search.
    search_radius : int or None
        Maximum integer displacement to search around the reference ROI.

    Returns
    -------
    shifts : ndarray of shape (rows, cols, 2)
        Returned in `(dx_px, dy_px)` order.
    """
    rows = len(ref_patches)
    cols = len(ref_patches[0])
    shifts = np.zeros((rows, cols, 2), dtype=float)
    scores = np.full((rows, cols), -1.0, dtype=float)
    peak_margins = np.zeros((rows, cols), dtype=float)
    peak_sharpness = np.zeros((rows, cols), dtype=float)
    repair_mask_any = np.zeros((rows, cols), dtype=bool)

    try:
        import config_recover as recover_config
    except ImportError:  # pragma: no cover
        recover_config = None

    use_guided = bool(getattr(recover_config, 'XCORR_USE_GUIDED', False))
    guided_ready = (
        use_guided and
        ref_boxes is not None and
        tgt_image is not None and
        len(ref_boxes) == rows * cols
    )

    if guided_ready:
        bg_percentile = float(getattr(recover_config, 'XCORR_BG_PERCENTILE', 10.0))
        gamma = float(getattr(recover_config, 'XCORR_GAMMA', 1.0))
        weight_alpha = float(getattr(recover_config, 'XCORR_WEIGHT_ALPHA', 0.75))
        subpixel_method = str(getattr(recover_config, 'XCORR_SUBPIXEL_METHOD', 'quadratic2d'))
        subpixel_radius = int(getattr(recover_config, 'XCORR_SUBPIXEL_RADIUS', 2))
        resolved_search_radius = int(
            getattr(recover_config, 'XCORR_SEARCH_RADIUS', 0)
            if search_radius is None
            else search_radius
        )

        for r in range(rows):
            for c in range(cols):
                ref = ref_patches[r][c]
                tgt = tgt_patches[r][c]
                if ref is None or tgt is None:
                    continue
                shift_x_px, shift_y_px, score, margin, sharpness = _guided_local_zncc_peak_shift(
                    ref_box=ref_boxes[r * cols + c],
                    ref_patch=ref,
                    tgt_patch=tgt,
                    tgt_image=tgt_image,
                    search_radius=resolved_search_radius,
                    bg_percentile=bg_percentile,
                    gamma=gamma,
                    weight_alpha=weight_alpha,
                    subpixel_method=subpixel_method,
                    subpixel_radius=subpixel_radius,
                )
                shifts[r, c, 0] = shift_x_px
                shifts[r, c, 1] = shift_y_px
                scores[r, c] = score
                peak_margins[r, c] = margin
                peak_sharpness[r, c] = sharpness

        enable_repair = bool(getattr(recover_config, 'XCORR_ENABLE_SPATIAL_REPAIR', True))
        min_score = float(getattr(recover_config, 'XCORR_MIN_SCORE', 0.10))
        min_peak_margin = float(getattr(recover_config, 'XCORR_MIN_PEAK_MARGIN', 0.015))
        min_peak_sharpness = float(getattr(recover_config, 'XCORR_MIN_PEAK_SHARPNESS', 0.010))
        outlier_thresh = float(getattr(recover_config, 'XCORR_SPATIAL_OUTLIER_PX', 2.0))
        if enable_repair and rows >= 3 and cols >= 3:
            low_conf_mask = (
                (~np.isfinite(scores)) |
                (scores < min_score) |
                (peak_margins < min_peak_margin) |
                (peak_sharpness < min_peak_sharpness)
            )
            for axis in range(2):
                med = median_filter(shifts[:, :, axis], size=3, mode='nearest')
                deviation = np.abs(shifts[:, :, axis] - med)
                repair_mask = low_conf_mask | (deviation > outlier_thresh)
                shifts[:, :, axis] = np.where(repair_mask, med, shifts[:, :, axis])
                repair_mask_any |= repair_mask

        confidence = _xcorr_confidence_from_metrics(
            scores,
            peak_margins,
            peak_sharpness,
            min_score=min_score,
            min_peak_margin=min_peak_margin,
            min_peak_sharpness=min_peak_sharpness,
        )
        if rows >= 3 and cols >= 3:
            deviation_norm = np.zeros((rows, cols), dtype=float)
            for axis in range(2):
                med = median_filter(shifts[:, :, axis], size=3, mode='nearest')
                deviation = np.abs(shifts[:, :, axis] - med)
                deviation_norm = np.maximum(
                    deviation_norm,
                    deviation / max(outlier_thresh, 1e-6),
                )
            spatial_consistency = np.clip(1.0 - deviation_norm, 0.0, 1.0)
            confidence *= (0.45 + 0.55 * spatial_consistency)
        if np.any(repair_mask_any):
            confidence = np.where(repair_mask_any, np.minimum(confidence, 0.20), confidence)

        shifts *= float(correction_factor)
        if return_diagnostics:
            return {
                "shifts": shifts,
                "score": scores,
                "peak_margin": peak_margins,
                "peak_sharpness": peak_sharpness,
                "confidence": confidence,
                "repair_mask": repair_mask_any.astype(np.uint8),
            }
        return shifts

    for r in range(rows):
        for c in range(cols):
            ref = ref_patches[r][c]
            tgt = tgt_patches[r][c]
            if ref is None or tgt is None:
                continue
            shift_x_px, shift_y_px = _xcorr_peak_shift(ref, tgt)
            shifts[r, c, 0] = shift_x_px * correction_factor
            shifts[r, c, 1] = shift_y_px * correction_factor
    if return_diagnostics:
        return {
            "shifts": shifts,
            "score": np.ones((rows, cols), dtype=float),
            "peak_margin": np.ones((rows, cols), dtype=float),
            "peak_sharpness": np.ones((rows, cols), dtype=float),
            "confidence": np.ones((rows, cols), dtype=float),
            "repair_mask": np.zeros((rows, cols), dtype=np.uint8),
        }
    return shifts


def phase_correlation_shifts(ref_patches, tgt_patches) -> np.ndarray:
    """
    Estimate shifts using phase correlation with subpixel refinement.

    Parameters
    ----------
    ref_patches, tgt_patches : List[List[ndarray]]
        Patches for each (row, col) beamlet

    Returns
    -------
    shifts : ndarray of shape (rows, cols, 2)
        Returned in `(dx_px, dy_px)` order.
    """
    rows = len(ref_patches)
    cols = len(ref_patches[0])
    shifts = np.zeros((rows, cols, 2), dtype=float)
    for r in range(rows):
        for c in range(cols):
            ref = ref_patches[r][c]
            tgt = tgt_patches[r][c]
            if ref is None or tgt is None:
                continue
            shift_x_px, shift_y_px = _phase_correlation_peak(ref, tgt)
            shifts[r, c, 0] = shift_x_px
            shifts[r, c, 1] = shift_y_px
    return shifts


# =============================================================================
# Subpixel Peak Refinement
# =============================================================================

def _xcorr_peak_shift(ref: np.ndarray, tgt: np.ndarray) -> Tuple[float, float]:
    """Compute subpixel shift from cross-correlation peak as `(dx_px, dy_px)`.

    This is now kept as a legacy fallback. The main X-ray recovery path uses
    guided local ZNCC in ``cross_correlation_shifts`` to decouple patch size
    from search range and avoid Hann-window scale bias.
    """
    ref = ref.astype(float) - np.mean(ref)
    tgt = tgt.astype(float) - np.mean(tgt)
    # Hanning 窗: 抑制方形截断谱泄漏，并产生完美的类高斯互相关峰（由于亚像素是以高斯拟合为基础的）
    win = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
    ref = ref * win
    tgt = tgt * win
    corr = fftconvolve(tgt, ref[::-1, ::-1], mode="same")
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    sub_x, sub_y = subpixel_peak_offset_2d(corr, peak[0], peak[1], method="gaussian")
    center_y = (corr.shape[0] - 1) / 2.0
    center_x = (corr.shape[1] - 1) / 2.0
    shift_y_px = peak[0] + sub_y - center_y
    shift_x_px = peak[1] + sub_x - center_x
    return shift_x_px, shift_y_px


def _phase_correlation_peak(ref: np.ndarray, tgt: np.ndarray) -> Tuple[float, float]:
    """Compute subpixel shift from phase-correlation peak as `(dx_px, dy_px)`."""
    ref = ref.astype(float) - np.mean(ref)
    tgt = tgt.astype(float) - np.mean(tgt)
    f_ref = np.fft.fft2(ref)
    f_tgt = np.fft.fft2(tgt)
    cross_power = f_tgt * np.conj(f_ref)
    denom = np.abs(cross_power)
    cross_power /= (denom + 1e-12)
    corr = np.fft.ifft2(cross_power)
    corr = np.fft.fftshift(np.abs(corr))
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    sub_x, sub_y = subpixel_peak_offset_2d(corr, peak[0], peak[1], method="parabolic")
    center_y = (corr.shape[0] - 1) / 2.0
    center_x = (corr.shape[1] - 1) / 2.0
    shift_y_px = peak[0] + sub_y - center_y
    shift_x_px = peak[1] + sub_x - center_x
    return shift_x_px, shift_y_px


# =============================================================================
# Wavefront Recovery from Shifts
# =============================================================================

def recover_from_shifts(
    shifts_px: np.ndarray,
    *,
    pixel_size: float,
    z_prop: float,
    spacing: float,
    wavelength: float,
    integration_method: str = 'southwell',
    measurement_confidence: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """
    Convert pixel shifts to physical displacements and phase.

    Parameters
    ----------
    shifts_px : ndarray of shape (rows, cols, 2)
    pixel_size : float (meters)
    z_prop : float (meters)
    spacing : float (meters)
    wavelength : float (meters)
    integration_method : str — 'southwell' or 'trapezoidal'

    Returns
    -------
    dict with `displacement_x_m`, `displacement_y_m`, `phase_rad`
    """
    displacement_x_m = shifts_px[:, :, 0] * pixel_size
    displacement_y_m = shifts_px[:, :, 1] * pixel_size
    slope_x = displacement_x_m / z_prop
    slope_y = displacement_y_m / z_prop
    phase_opd = integrate_slopes(
        slope_x,
        slope_y,
        spacing,
        method=integration_method,
        measurement_confidence=measurement_confidence,
    )
    phase_rad = (2.0 * np.pi / wavelength) * phase_opd
    return {
        "displacement_x_m": displacement_x_m,
        "displacement_y_m": displacement_y_m,
        "phase_rad": phase_rad,
    }


def integrate_slopes(slope_x: np.ndarray, slope_y: np.ndarray,
                     spacing: float,
                     method: str = 'southwell',
                     measurement_confidence: np.ndarray | None = None) -> np.ndarray:
    """Integrate slope fields to wavefront OPD.

    Parameters
    ----------
    slope_x, slope_y : ndarray of shape (rows, cols)
    spacing : float — sub-aperture pitch (meters)
    method : str — 'southwell' (least-squares, recommended) or 'trapezoidal'

    Returns
    -------
    phase : ndarray of shape (rows, cols) — wavefront OPD (meters)
    """
    if method == 'southwell':
        return _integrate_slopes_southwell(
            slope_x,
            slope_y,
            spacing,
            measurement_confidence=measurement_confidence,
        )
    elif method == 'trapezoidal':
        return _integrate_slopes_trapezoidal(slope_x, slope_y, spacing)
    else:
        raise ValueError(f"Unknown integration method: {method}. "
                         f"Use 'southwell' or 'trapezoidal'.")


def _integrate_slopes_trapezoidal(slope_x: np.ndarray, slope_y: np.ndarray,
                                  spacing: float) -> np.ndarray:
    """Original trapezoidal (prefix-sum) integration."""
    rows, cols = slope_x.shape
    phase_from_x = np.zeros((rows, cols), dtype=float)
    for c in range(1, cols):
        phase_from_x[:, c] = phase_from_x[:, c - 1] + slope_x[:, c - 1] * spacing

    phase_from_y = np.zeros((rows, cols), dtype=float)
    for r in range(1, rows):
        phase_from_y[r, :] = phase_from_y[r - 1, :] + slope_y[r - 1, :] * spacing

    phase = 0.5 * (phase_from_x + phase_from_y)
    phase -= phase.mean()
    return phase


def _integrate_slopes_southwell(slope_x: np.ndarray, slope_y: np.ndarray,
                                spacing: float,
                                measurement_confidence: np.ndarray | None = None) -> np.ndarray:
    """Southwell least-squares wavefront reconstruction.

    Builds a sparse linear system from slope measurements between all
    adjacent grid points and solves for the wavefront via least-squares.

    For each pair of horizontal neighbors (i,j) and (i,j+1):
        φ(i,j+1) - φ(i,j) = 0.5 * (sx(i,j) + sx(i,j+1)) * spacing

    For each pair of vertical neighbors (i,j) and (i+1,j):
        φ(i+1,j) - φ(i,j) = 0.5 * (sy(i,j) + sy(i+1,j)) * spacing

    This uses ALL available slope data simultaneously, distributing
    measurement noise optimally instead of accumulating it along a path.
    """
    rows, cols = slope_x.shape
    n = rows * cols

    confidence = None
    if measurement_confidence is not None:
        confidence = np.asarray(measurement_confidence, dtype=float)
        if confidence.shape != slope_x.shape:
            raise ValueError(
                f"measurement_confidence must match slope shape {slope_x.shape}, got {confidence.shape}"
            )
        confidence = np.clip(confidence, 0.0, 1.0)

    try:
        import config_recover as recover_config
        confidence_floor = float(getattr(recover_config, 'SOUTHWELL_CONFIDENCE_FLOOR', 0.05))
    except ImportError:  # pragma: no cover
        confidence_floor = 0.05

    # Pre-allocate COO arrays
    # Number of equations: (cols-1)*rows horizontal + (rows-1)*cols vertical
    n_eq = (cols - 1) * rows + (rows - 1) * cols
    row_idx = np.zeros(n_eq * 2, dtype=np.int32)
    col_idx = np.zeros(n_eq * 2, dtype=np.int32)
    val = np.zeros(n_eq * 2, dtype=float)
    b = np.zeros(n_eq, dtype=float)

    eq = 0
    ptr = 0

    # Horizontal equations: φ(i,j+1) - φ(i,j) = average_slope_x * spacing
    for i in range(rows):
        for j in range(cols - 1):
            idx_left = i * cols + j
            idx_right = i * cols + (j + 1)
            row_scale = 1.0
            if confidence is not None:
                edge_conf = 0.5 * (confidence[i, j] + confidence[i, j + 1])
                row_scale = np.sqrt(max(confidence_floor, float(edge_conf)))
            row_idx[ptr] = eq;     col_idx[ptr] = idx_right;  val[ptr] = row_scale;   ptr += 1
            row_idx[ptr] = eq;     col_idx[ptr] = idx_left;   val[ptr] = -row_scale;  ptr += 1
            b[eq] = row_scale * 0.5 * (slope_x[i, j] + slope_x[i, j + 1]) * spacing
            eq += 1

    # Vertical equations: φ(i+1,j) - φ(i,j) = average_slope_y * spacing
    for i in range(rows - 1):
        for j in range(cols):
            idx_bot = i * cols + j
            idx_top = (i + 1) * cols + j
            row_scale = 1.0
            if confidence is not None:
                edge_conf = 0.5 * (confidence[i, j] + confidence[i + 1, j])
                row_scale = np.sqrt(max(confidence_floor, float(edge_conf)))
            row_idx[ptr] = eq;     col_idx[ptr] = idx_top;    val[ptr] = row_scale;   ptr += 1
            row_idx[ptr] = eq;     col_idx[ptr] = idx_bot;    val[ptr] = -row_scale;  ptr += 1
            b[eq] = row_scale * 0.5 * (slope_y[i, j] + slope_y[i + 1, j]) * spacing
            eq += 1

    A = sparse.csr_matrix((val[:ptr], (row_idx[:ptr], col_idx[:ptr])),
                          shape=(eq, n))
    result = sparse_lsqr(A, b[:eq])
    phase = result[0].reshape(rows, cols)
    phase -= phase.mean()
    return phase


def remove_plane(data: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Remove best-fit plane from 2D data (tip/tilt removal).

    Handles NaN/Inf values by fitting the plane only to valid points,
    then subtracting it from the entire grid.
    """
    flat_data = data.ravel().astype(np.float64)
    flat_x = grid_x.ravel().astype(np.float64)
    flat_y = grid_y.ravel().astype(np.float64)

    valid = np.isfinite(flat_data)
    if np.sum(valid) < 3:
        # Not enough valid points to fit a plane -- just remove mean of valid
        mean_val = np.nanmean(flat_data) if np.any(valid) else 0.0
        result = data.copy().astype(np.float64)
        result[np.isfinite(result)] -= mean_val
        return result

    A_valid = np.column_stack([flat_x[valid], flat_y[valid], np.ones(np.sum(valid))])
    coeffs, _, _, _ = np.linalg.lstsq(A_valid, flat_data[valid], rcond=None)

    A_full = np.column_stack([flat_x, flat_y, np.ones(flat_x.size)])
    plane = (A_full @ coeffs).reshape(data.shape)
    return data - plane
