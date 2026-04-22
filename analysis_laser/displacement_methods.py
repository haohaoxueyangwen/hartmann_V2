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

import os
import multiprocessing as _mp
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import maximum_filter, gaussian_filter, map_coordinates
from scipy import sparse
from scipy.sparse.linalg import lsqr as sparse_lsqr
from typing import Dict, List, Tuple, Optional

from analysis_laser.subpixel_common import subpixel_peak_offset_2d


# =============================================================================
# Multiprocessing shared state (fork-based, copy-on-write)
# =============================================================================
_MP_REF_IMAGE: Optional[np.ndarray] = None
_MP_TGT_IMAGE: Optional[np.ndarray] = None


def _get_n_workers() -> int:
    """Get number of parallel workers from config."""
    try:
        from config_recover_laser import PARALLEL_WORKERS
        n = int(PARALLEL_WORKERS)
    except (ImportError, AttributeError):
        n = 0
    if n <= 0:
        n = max(1, (os.cpu_count() or 2) - 1)
    return n


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
        from config_recover_laser import XCORR_PATCH_HALF
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


def build_guided_boxes_from_centers(
    image_shape: Tuple[int, int],
    centers_xy_px: np.ndarray,
    rows: int,
    cols: int,
    *,
    patch_half: Optional[int] = None,
):
    """
    Build fixed ROIs centered on known approximate beamlet centers.

    This is the preferred path for laser mode because ideal grid centers are
    already known from simulation outputs, and ringed spots are easier to
    localize with guided template matching than with blind peak finding.
    """
    centers = np.asarray(centers_xy_px, dtype=float)
    if centers.shape != (rows * cols, 2):
        raise ValueError(
            f"centers_xy_px must have shape ({rows * cols}, 2), got {centers.shape}"
        )

    if patch_half is None:
        try:
            from config_recover_laser import XCORR_PATCH_HALF
            patch_half = int(XCORR_PATCH_HALF)
        except (ImportError, AttributeError):
            patch_half = 45
    patch_half = max(1, int(patch_half))
    patch_size = patch_half * 2 + 1

    height, width = int(image_shape[0]), int(image_shape[1])
    max_y0 = max(0, height - patch_size)
    max_x0 = max(0, width - patch_size)

    boxes: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    centroids = np.zeros((rows, cols, 2), dtype=float)
    for idx, (cx, cy) in enumerate(centers):
        x0 = int(np.clip(np.round(cx) - patch_half, 0, max_x0))
        y0 = int(np.clip(np.round(cy) - patch_half, 0, max_y0))
        x1 = x0 + patch_size
        y1 = y0 + patch_size
        boxes.append(((y0, y1), (x0, x1)))
        row = idx // cols
        col = idx % cols
        centroids[row, col, 0] = float(cx)
        centroids[row, col, 1] = float(cy)
    return boxes, centroids


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

def cross_correlation_shifts(ref_patches, tgt_patches,
                             correction_factor: float = 2.0) -> np.ndarray:
    """
    Estimate shifts using FFT cross-correlation with subpixel refinement.

    Parameters
    ----------
    ref_patches, tgt_patches : List[List[ndarray]]
        Patches for each (row, col) beamlet
    correction_factor : float
        Multiplicative factor to compensate systematic bias (default: 2.0)

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
            shift_x_px, shift_y_px = _xcorr_peak_shift(ref, tgt)
            shifts[r, c, 0] = shift_x_px * correction_factor
            shifts[r, c, 1] = shift_y_px * correction_factor
    return shifts


def guided_samebox_xcorr_shifts(
    ref_image: np.ndarray,
    tgt_image: np.ndarray,
    centers_xy_px: np.ndarray,
    rows: int,
    cols: int,
    *,
    patch_half: Optional[int] = None,
    correction_factor: float = 1.0,
) -> np.ndarray:
    """
    Legacy FFT xcorr, but using guided ROIs instead of blind beamlet detection.

    This keeps the old correlation kernel while removing one major error source:
    beamlet re-detection drift on ringed laser spots.
    """
    ref_boxes, _ = build_guided_boxes_from_centers(
        ref_image.shape,
        centers_xy_px,
        rows,
        cols,
        patch_half=patch_half,
    )
    ref_patches = extract_patches(ref_image, ref_boxes, rows, cols)
    tgt_patches = extract_patches(tgt_image, ref_boxes, rows, cols)
    return cross_correlation_shifts(
        ref_patches,
        tgt_patches,
        correction_factor=correction_factor,
    )


def guided_template_xcorr_shifts(
    ref_image: np.ndarray,
    tgt_image: np.ndarray,
    centers_xy_px: np.ndarray,
    rows: int,
    cols: int,
    *,
    patch_half: Optional[int] = None,
    search_radius: Optional[int] = None,
    correction_factor: float = 1.0,
) -> np.ndarray:
    """
    Estimate shifts with guided, template-based ZNCC around known spot centers.

    Compared with the legacy blind xcorr path, this is much more robust for
    ringed laser spots because it:
      1. uses known ideal centers instead of re-detecting peaks,
      2. preserves the ring structure as the correlation template,
      3. applies background suppression and dynamic-range compression.
    """
    shifts, _, _, _ = _guided_template_xcorr_full(
        ref_image,
        tgt_image,
        centers_xy_px,
        rows,
        cols,
        patch_half=patch_half,
        search_radius=search_radius,
        correction_factor=correction_factor,
        fallback_low_confidence=True,
    )
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
def _xcorr_peak_shift(ref: np.ndarray, tgt: np.ndarray) -> Tuple[float, float]:
    """Compute subpixel shift from cross-correlation peak as `(dx_px, dy_px)`.

    Hanning window is applied to BOTH patches. This introduces a systematic attenuation
    for spots that interacts with the Gaussian subpixel peak fit, which must be 
    compensated by XCORR_CORRECTION_FACTOR in config.
    """
    ref = ref.astype(float) - np.mean(ref)
    tgt = tgt.astype(float) - np.mean(tgt)
    # Hanning 窗: 抑制截断谱泄漏，保证相关峰的高斯分布以供拟合
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


def _estimate_patch_background(patch: np.ndarray, percentile: float = 20.0) -> float:
    """Estimate local background from the patch border."""
    arr = np.asarray(patch, dtype=float)
    if arr.size == 0:
        return 0.0
    border = max(1, int(round(min(arr.shape) * 0.1)))
    border_mask = np.ones_like(arr, dtype=bool)
    border_mask[border:-border, border:-border] = False
    samples = arr[border_mask]
    if samples.size == 0:
        samples = arr.ravel()
    return float(np.percentile(samples, percentile))


def _estimate_patch_fwhm(patch: np.ndarray) -> float:
    """Estimate the FWHM of a spot in a background-subtracted patch.

    Uses a horizontal line profile through the peak to measure
    the full-width at half-maximum.
    """
    arr = np.asarray(patch, dtype=float)
    if arr.size == 0:
        return max(arr.shape) * 0.3
    peak_idx = np.unravel_index(np.argmax(arr), arr.shape)
    profile = arr[peak_idx[0], :]
    peak_val = float(np.max(profile))
    if peak_val <= 0:
        return max(arr.shape) * 0.3
    half_max = peak_val * 0.5
    above = profile > half_max
    if not np.any(above):
        return max(arr.shape) * 0.3
    indices = np.where(above)[0]
    fwhm = float(indices[-1] - indices[0])
    return max(fwhm, 3.0)


def _build_core_gaussian_mask(shape, *, sigma_px: float) -> np.ndarray:
    """Build a 2D Gaussian soft-aperture mask centered on the patch.

    The mask has value 1.0 at the center and decays as
    exp(-r^2/(2*sigma^2)), suppressing diffuse tails from neighbors.
    """
    ny, nx = shape
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    yy, xx = np.mgrid[:ny, :nx]
    r_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    return np.exp(-r_sq / (2.0 * sigma_px * sigma_px))


def _prepare_laser_xcorr_patch(
    patch: np.ndarray,
    *,
    bg_percentile: float,
    gamma: float,
    core_mask_enabled: bool = True,
    core_sigma_scale: float = 1.2,
) -> np.ndarray:
    """Background-suppress, apply core mask, and compress dynamic range.

    The core Gaussian mask isolates the central lobe of each laser spot,
    suppressing diffuse tails that would otherwise contaminate the
    cross-correlation with signal from neighboring sub-apertures.
    """
    arr = np.asarray(patch, dtype=float).copy()
    arr = arr - _estimate_patch_background(arr, percentile=bg_percentile)
    arr[arr < 0.0] = 0.0
    if gamma != 1.0:
        arr = np.power(arr, gamma, where=arr > 0.0, out=np.zeros_like(arr))

    # Core Gaussian soft-aperture: suppress diffuse neighbor contamination
    if core_mask_enabled:
        fwhm = _estimate_patch_fwhm(arr)
        sigma_px = core_sigma_scale * fwhm / 2.355
        core_mask = _build_core_gaussian_mask(arr.shape, sigma_px=sigma_px)
        arr = arr * core_mask

    win = np.outer(np.hanning(arr.shape[0]), np.hanning(arr.shape[1]))
    return arr * win


def _build_template_weight(template_patch: np.ndarray, *, weight_alpha: float) -> np.ndarray:
    """
    Build a smooth weight map from the template itself.

    The template-derived weight naturally emphasizes the central lobe and bright
    diffraction rings without forcing a Gaussian prior.
    """
    arr = np.asarray(template_patch, dtype=float)
    vmax = float(np.max(arr)) if arr.size else 0.0
    if vmax <= 1e-12:
        return np.ones_like(arr, dtype=float)

    norm = arr / vmax
    ny, nx = arr.shape
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    yy, xx = np.mgrid[:ny, :nx]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    max_radius = int(max(2, np.floor(rr.max())))
    radial_mean = np.zeros(max_radius + 1, dtype=float)
    for r in range(max_radius + 1):
        mask = (rr >= r) & (rr < r + 1)
        if np.any(mask):
            radial_mean[r] = float(np.mean(norm[mask]))

    # Prefer the first bright ring outside the central lobe.
    ring_search = radial_mean[2:] if radial_mean.size > 2 else radial_mean
    if ring_search.size > 0:
        ring_radius = float(np.argmax(ring_search) + 2)
    else:
        ring_radius = 0.0

    center_sigma = max(1.5, 0.35 * max(ring_radius, 3.0))
    ring_sigma = max(1.5, 0.20 * max(ring_radius, 5.0))
    center_weight = np.exp(-0.5 * (rr / center_sigma) ** 2)
    if ring_radius > 1.0:
        ring_weight = np.exp(-0.5 * ((rr - ring_radius) / ring_sigma) ** 2)
    else:
        ring_weight = np.zeros_like(center_weight)

    ring_aware = 0.55 * center_weight + 0.90 * ring_weight
    win = np.outer(np.hanning(arr.shape[0]), np.hanning(arr.shape[1]))
    intensity_weight = np.power(np.clip(norm, 0.0, None), weight_alpha)
    weight = 0.20 * win + 0.45 * intensity_weight + 0.35 * ring_aware
    weight = np.clip(weight, 1e-6, None)
    return weight


def _weighted_zncc(ref: np.ndarray, tgt: np.ndarray, weight: np.ndarray) -> float:
    """Weighted zero-mean normalized cross-correlation for already aligned patches."""
    ref_arr = np.asarray(ref, dtype=float)
    tgt_arr = np.asarray(tgt, dtype=float)
    w = np.asarray(weight, dtype=float)
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        return -1.0
    w = w / w_sum
    ref_mean = float(np.sum(w * ref_arr))
    tgt_mean = float(np.sum(w * tgt_arr))
    ref_zero = ref_arr - ref_mean
    tgt_zero = tgt_arr - tgt_mean
    ref_norm = float(np.sqrt(np.sum(w * ref_zero * ref_zero)))
    tgt_norm = float(np.sqrt(np.sum(w * tgt_zero * tgt_zero)))
    if ref_norm <= 1e-12 or tgt_norm <= 1e-12:
        return -1.0
    return float(np.sum(w * ref_zero * tgt_zero) / (ref_norm * tgt_norm))


def _score_map_peak_metrics(
    score_map: np.ndarray,
    peak_rc: Tuple[int, int],
) -> Tuple[float, float]:
    """
    Measure how isolated and sharp a correlation peak is.

    Returns
    -------
    peak_margin, peak_sharpness : float
        `peak_margin` compares the best peak against the strongest competing
        peak outside the 3x3 neighborhood. `peak_sharpness` compares the peak
        against its immediate neighbors.
    """
    scores = np.asarray(score_map, dtype=float)
    py, px = int(peak_rc[0]), int(peak_rc[1])
    peak_score = float(scores[py, px])

    y0 = max(0, py - 1)
    y1 = min(scores.shape[0], py + 2)
    x0 = max(0, px - 1)
    x1 = min(scores.shape[1], px + 2)

    local = scores[y0:y1, x0:x1].copy()
    local_peak_y = py - y0
    local_peak_x = px - x0
    local[local_peak_y, local_peak_x] = -np.inf
    local_neighbor = float(np.max(local)) if np.isfinite(local).any() else peak_score
    peak_sharpness = peak_score - local_neighbor

    global_competitors = scores.copy()
    global_competitors[y0:y1, x0:x1] = -np.inf
    second_peak = (
        float(np.max(global_competitors))
        if np.isfinite(global_competitors).any()
        else local_neighbor
    )
    peak_margin = peak_score - second_peak
    return float(peak_margin), float(peak_sharpness)


def _guided_confidence_value(
    base_score,
    peak_margin,
    peak_sharpness,
    *,
    margin_weight: float,
    sharpness_weight: float,
) -> np.ndarray:
    """Combine template similarity with correlation-peak isolation metrics."""
    base = np.asarray(base_score, dtype=float)
    margin = np.asarray(peak_margin, dtype=float)
    sharpness = np.asarray(peak_sharpness, dtype=float)
    return (
        base
        + margin_weight * np.maximum(0.0, margin)
        + sharpness_weight * np.maximum(0.0, sharpness)
    )


def _extract_shifted_patch(
    image: np.ndarray,
    ref_box: Tuple[Tuple[int, int], Tuple[int, int]],
    *,
    shift_x_px: float,
    shift_y_px: float,
) -> np.ndarray:
    """Sample a patch from `image` at subpixel shift relative to a reference ROI."""
    (y0, y1), (x0, x1) = ref_box
    patch_h = int(y1 - y0)
    patch_w = int(x1 - x0)
    yy, xx = np.mgrid[0:patch_h, 0:patch_w]
    sample_y = y0 + yy + float(shift_y_px)
    sample_x = x0 + xx + float(shift_x_px)
    return map_coordinates(
        np.asarray(image, dtype=float),
        [sample_y, sample_x],
        order=1,
        mode='constant',
        cval=0.0,
    )


def _guided_template_peak_shift(
    *,
    ref_image: np.ndarray,
    tgt_image: np.ndarray,
    ref_box: Tuple[Tuple[int, int], Tuple[int, int]],
    ref_patch: np.ndarray,
    fallback_patch: np.ndarray,
    search_radius: int,
    bg_percentile: float,
    gamma: float,
    weight_alpha: float,
    core_mask_enabled: bool = True,
    core_sigma_scale: float = 1.2,
) -> Tuple[float, float, float, float, float]:
    """Search for the best weighted-ZNCC shift around a guided center."""
    (y0, y1), (x0, x1) = ref_box
    patch_h = int(y1 - y0)
    patch_w = int(x1 - x0)
    ny, nx = tgt_image.shape
    if patch_h <= 0 or patch_w <= 0:
        return 0.0, 0.0, -1.0, 0.0, 0.0

    ys0 = y0 - search_radius
    ys1 = y1 + search_radius
    xs0 = x0 - search_radius
    xs1 = x1 + search_radius
    if ys0 < 0 or xs0 < 0 or ys1 > ny or xs1 > nx:
        shift_x_px, shift_y_px = _xcorr_peak_shift(ref_patch, fallback_patch)
        return float(shift_x_px), float(shift_y_px), -1.0, 0.0, 0.0

    ref_proc = _prepare_laser_xcorr_patch(
        ref_patch,
        bg_percentile=bg_percentile,
        gamma=gamma,
        core_mask_enabled=core_mask_enabled,
        core_sigma_scale=core_sigma_scale,
    )
    weight = _build_template_weight(ref_proc, weight_alpha=weight_alpha)

    search_roi = _prepare_laser_xcorr_patch(
        tgt_image[ys0:ys1, xs0:xs1],
        bg_percentile=bg_percentile,
        gamma=gamma,
        core_mask_enabled=core_mask_enabled,
        core_sigma_scale=core_sigma_scale,
    )

    w = np.asarray(weight, dtype=float)
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        shift_x_px, shift_y_px = _xcorr_peak_shift(ref_patch, fallback_patch)
        return float(shift_x_px), float(shift_y_px), -1.0, 0.0, 0.0

    ref_mean = float(np.sum(w * ref_proc) / w_sum)
    ref_zero = ref_proc - ref_mean
    ref_weighted = ref_zero * w
    ref_var = float(np.sum(w * ref_zero * ref_zero))
    if ref_var <= 1e-12:
        shift_x_px, shift_y_px = _xcorr_peak_shift(ref_patch, fallback_patch)
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
    best_dy = int(peak[0] - search_radius)
    best_dx = int(peak[1] - search_radius)

    if not np.isfinite(best_score) or best_score <= -0.5:
        shift_x_px, shift_y_px = _xcorr_peak_shift(ref_patch, fallback_patch)
        return float(shift_x_px), float(shift_y_px), float(best_score), 0.0, 0.0

    peak_margin, peak_sharpness = _score_map_peak_metrics(scores, peak)

    sub_x, sub_y = subpixel_peak_offset_2d(
        scores,
        peak[0],
        peak[1],
        method="parabolic",
    )
    return (
        float(best_dx + sub_x),
        float(best_dy + sub_y),
        float(best_score),
        float(peak_margin),
        float(peak_sharpness),
    )


# =============================================================================
# Multiprocessing worker & dispatch
# =============================================================================

def _guided_spot_worker(task):
    """Process a single spot for guided template xcorr (multiprocessing worker).

    Reads ref_image / tgt_image from module-level globals that are shared
    via fork copy-on-write semantics.
    """
    (r, c, ref_box, ref_patch, tgt_patch,
     search_radius, bg_percentile, gamma, weight_alpha,
     core_mask_enabled, core_sigma_scale,
     min_score, min_peak_margin, min_peak_sharpness,
     min_confidence, margin_weight, sharpness_weight,
     correction_factor, fallback_low_confidence) = task

    if ref_patch is None or tgt_patch is None:
        return (r, c, 0.0, 0.0, -1.0, 0.0, 0.0)

    shift_x_px, shift_y_px, score, peak_margin, sharpness = _guided_template_peak_shift(
        ref_image=_MP_REF_IMAGE,
        tgt_image=_MP_TGT_IMAGE,
        ref_box=ref_box,
        ref_patch=ref_patch,
        fallback_patch=tgt_patch,
        search_radius=search_radius,
        bg_percentile=bg_percentile,
        gamma=gamma,
        weight_alpha=weight_alpha,
        core_mask_enabled=core_mask_enabled,
        core_sigma_scale=core_sigma_scale,
    )

    confidence = _guided_confidence_value(
        float(score), float(peak_margin), float(sharpness),
        margin_weight=margin_weight, sharpness_weight=sharpness_weight,
    )
    low_conf = (
        (not np.isfinite(score))
        or (score < min_score)
        or (confidence < min_confidence)
        or ((peak_margin < min_peak_margin) and (sharpness < min_peak_sharpness))
    )
    if fallback_low_confidence and low_conf:
        shift_x_px, shift_y_px = _xcorr_peak_shift(ref_patch, tgt_patch)
        score = float(min_score)
        peak_margin = 0.0
        sharpness = 0.0

    return (r, c,
            float(shift_x_px) * correction_factor,
            float(shift_y_px) * correction_factor,
            float(score), float(peak_margin), float(sharpness))


def _init_worker_images(ref_image, tgt_image):
    """Initialize worker process with shared image data."""
    global _MP_REF_IMAGE, _MP_TGT_IMAGE
    _MP_REF_IMAGE = ref_image
    _MP_TGT_IMAGE = tgt_image


def _parallel_guided_dispatch(tasks, ref_image, tgt_image, n_workers):
    """Dispatch guided xcorr tasks to a process pool.

    Uses default multiprocessing context (spawn on macOS Python 3.12+).
    Images are passed once per worker via initializer to avoid fork deadlocks
    and per-task pickle overhead.
    """
    try:
        with _mp.Pool(
            n_workers,
            initializer=_init_worker_images,
            initargs=(ref_image, tgt_image),
        ) as pool:
            results = pool.map(_guided_spot_worker, tasks)
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Fallback to sequential
        global _MP_REF_IMAGE, _MP_TGT_IMAGE
        _MP_REF_IMAGE = ref_image
        _MP_TGT_IMAGE = tgt_image
        return [_guided_spot_worker(t) for t in tasks]


def _guided_template_xcorr_full(
    ref_image: np.ndarray,
    tgt_image: np.ndarray,
    centers_xy_px: np.ndarray,
    rows: int,
    cols: int,
    *,
    patch_half: Optional[int] = None,
    search_radius: Optional[int] = None,
    correction_factor: float = 1.0,
    fallback_low_confidence: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full guided-template xcorr result with per-spot peak diagnostics.

    Returns
    -------
    shifts, scores, peak_margins, peak_sharpness : ndarray
    """
    try:
        import config_recover_laser as laser_config
    except ImportError:  # pragma: no cover
        laser_config = None

    if patch_half is None:
        patch_half = int(getattr(laser_config, "XCORR_PATCH_HALF", 45))
    if search_radius is None:
        search_radius = int(getattr(laser_config, "XCORR_SEARCH_RADIUS", 12))

    bg_percentile = float(getattr(laser_config, "LASER_XCORR_BG_PERCENTILE", 20.0))
    gamma = float(getattr(laser_config, "LASER_XCORR_GAMMA", 1.0))
    weight_alpha = float(getattr(laser_config, "LASER_XCORR_WEIGHT_ALPHA", 0.8))
    core_mask_enabled = bool(getattr(laser_config, "LASER_XCORR_CORE_MASK_ENABLED", True))
    core_sigma_scale = float(getattr(laser_config, "LASER_XCORR_CORE_SIGMA_SCALE", 1.2))
    min_score = float(getattr(laser_config, "LASER_XCORR_MIN_SCORE", 0.10))
    min_peak_margin = float(getattr(laser_config, "LASER_XCORR_MIN_PEAK_MARGIN", 0.015))
    min_peak_sharpness = float(getattr(laser_config, "LASER_XCORR_MIN_PEAK_SHARPNESS", 0.010))
    min_confidence = float(getattr(laser_config, "LASER_XCORR_MIN_CONFIDENCE", 0.13))
    margin_weight = float(getattr(laser_config, "LASER_XCORR_PEAK_MARGIN_WEIGHT", 0.80))
    sharpness_weight = float(getattr(laser_config, "LASER_XCORR_PEAK_SHARPNESS_WEIGHT", 0.60))

    ref_boxes, _ = build_guided_boxes_from_centers(
        ref_image.shape,
        centers_xy_px,
        rows,
        cols,
        patch_half=patch_half,
    )
    ref_patches = extract_patches(ref_image, ref_boxes, rows, cols)
    tgt_patches = extract_patches(tgt_image, ref_boxes, rows, cols)

    # Build per-spot task arguments
    ref_image_f = np.asarray(ref_image, dtype=float)
    tgt_image_f = np.asarray(tgt_image, dtype=float)
    spot_config = (
        int(search_radius), bg_percentile, gamma, weight_alpha,
        core_mask_enabled, core_sigma_scale,
        min_score, min_peak_margin, min_peak_sharpness,
        min_confidence, margin_weight, sharpness_weight,
        correction_factor, fallback_low_confidence,
    )
    tasks = []
    for r in range(rows):
        for c in range(cols):
            ref_patch = ref_patches[r][c]
            tgt_patch = tgt_patches[r][c]
            rp = np.asarray(ref_patch, dtype=float) if ref_patch is not None else None
            tp = np.asarray(tgt_patch, dtype=float) if tgt_patch is not None else None
            tasks.append((r, c, ref_boxes[r * cols + c], rp, tp) + spot_config)

    # Dispatch: parallel or sequential
    n_workers = _get_n_workers()
    if n_workers > 1 and len(tasks) > 1:
        results = _parallel_guided_dispatch(tasks, ref_image_f, tgt_image_f, n_workers)
    else:
        global _MP_REF_IMAGE, _MP_TGT_IMAGE
        _MP_REF_IMAGE = ref_image_f
        _MP_TGT_IMAGE = tgt_image_f
        results = [_guided_spot_worker(t) for t in tasks]

    # Collect results
    shifts = np.zeros((rows, cols, 2), dtype=float)
    scores = np.full((rows, cols), -1.0, dtype=float)
    peak_margins = np.zeros((rows, cols), dtype=float)
    peak_sharpness_arr = np.zeros((rows, cols), dtype=float)
    for res in results:
        r, c, sx, sy, sc, pm, ps = res
        shifts[r, c, 0] = sx
        shifts[r, c, 1] = sy
        scores[r, c] = sc
        peak_margins[r, c] = pm
        peak_sharpness_arr[r, c] = ps

    # ---- Spatial outlier repair ----
    # Detect spots whose displacement deviates from local median by more than
    # a threshold and replace them with the median of valid neighbours.
    enable_repair = bool(getattr(laser_config, "LASER_XCORR_ENABLE_SPATIAL_REPAIR", True))
    outlier_thresh = float(getattr(laser_config, "LASER_XCORR_SPATIAL_OUTLIER_PX", 2.5))
    if enable_repair and rows >= 3 and cols >= 3:
        from scipy.ndimage import median_filter as _medfilt
        for axis in range(2):
            med = _medfilt(shifts[:, :, axis], size=3)
            deviation = np.abs(shifts[:, :, axis] - med)
            outlier_mask = deviation > outlier_thresh
            if np.any(outlier_mask):
                shifts[:, :, axis] = np.where(outlier_mask, med, shifts[:, :, axis])

    return shifts, scores, peak_margins, peak_sharpness_arr


def guided_template_shift_scores(
    ref_image: np.ndarray,
    tgt_image: np.ndarray,
    centers_xy_px: np.ndarray,
    shifts_xy_px: np.ndarray,
    rows: int,
    cols: int,
    *,
    patch_half: Optional[int] = None,
) -> np.ndarray:
    """
    Score externally provided shift hypotheses with the same laser ring-aware metric.

    This is used to compare guided-template and legacy xcorr candidates under a
    common criterion before selecting the final per-spot displacement.
    """
    try:
        import config_recover_laser as laser_config
    except ImportError:  # pragma: no cover
        laser_config = None

    bg_percentile = float(getattr(laser_config, "LASER_XCORR_BG_PERCENTILE", 20.0))
    gamma = float(getattr(laser_config, "LASER_XCORR_GAMMA", 1.0))
    weight_alpha = float(getattr(laser_config, "LASER_XCORR_WEIGHT_ALPHA", 0.8))

    boxes, _ = build_guided_boxes_from_centers(
        ref_image.shape,
        centers_xy_px,
        rows,
        cols,
        patch_half=patch_half,
    )
    ref_patches = extract_patches(ref_image, boxes, rows, cols)
    shifts = np.asarray(shifts_xy_px, dtype=float)
    scores = np.full((rows, cols), -1.0, dtype=float)
    for r in range(rows):
        for c in range(cols):
            ref_patch = ref_patches[r][c]
            if ref_patch is None:
                continue
            shift_x = float(shifts[r, c, 0])
            shift_y = float(shifts[r, c, 1])
            tgt_patch = _extract_shifted_patch(
                tgt_image,
                boxes[r * cols + c],
                shift_x_px=shift_x,
                shift_y_px=shift_y,
            )
            core_mask_on = bool(getattr(laser_config, "LASER_XCORR_CORE_MASK_ENABLED", True))
            core_sigma_s = float(getattr(laser_config, "LASER_XCORR_CORE_SIGMA_SCALE", 1.2))
            ref_proc = _prepare_laser_xcorr_patch(
                ref_patch,
                bg_percentile=bg_percentile,
                gamma=gamma,
                core_mask_enabled=core_mask_on,
                core_sigma_scale=core_sigma_s,
            )
            tgt_proc = _prepare_laser_xcorr_patch(
                tgt_patch,
                bg_percentile=bg_percentile,
                gamma=gamma,
                core_mask_enabled=core_mask_on,
                core_sigma_scale=core_sigma_s,
            )
            weight = _build_template_weight(ref_proc, weight_alpha=weight_alpha)
            scores[r, c] = _weighted_zncc(ref_proc, tgt_proc, weight)
    return scores


def _neighbor_median_shifts(shifts_xy_px: np.ndarray) -> np.ndarray:
    """Predict each spot shift from the median of its local 3x3 neighborhood."""
    shifts = np.asarray(shifts_xy_px, dtype=float)
    rows, cols, _ = shifts.shape
    pred = np.zeros_like(shifts)
    for r in range(rows):
        for c in range(cols):
            r0 = max(0, r - 1)
            r1 = min(rows, r + 2)
            c0 = max(0, c - 1)
            c1 = min(cols, c + 2)
            neigh = shifts[r0:r1, c0:c1, :].reshape(-1, 2)
            if neigh.shape[0] <= 1:
                pred[r, c, :] = shifts[r, c, :]
                continue
            pred[r, c, :] = np.median(neigh, axis=0)
    return pred


def hybrid_guided_xcorr_shifts(
    ref_image: np.ndarray,
    tgt_image: np.ndarray,
    centers_xy_px: np.ndarray,
    rows: int,
    cols: int,
    *,
    patch_half: Optional[int] = None,
    search_radius: Optional[int] = None,
    correction_factor: float = 1.0,
) -> np.ndarray:
    """
    Laser-specific hybrid xcorr:
      1. guided ring-aware template xcorr
      2. guided legacy same-box xcorr
      3. per-spot score selection
      4. optional low-confidence spatial repair
    """
    try:
        import config_recover_laser as laser_config
    except ImportError:  # pragma: no cover
        laser_config = None

    if patch_half is None:
        patch_half = int(getattr(laser_config, "XCORR_PATCH_HALF", 45))
    if search_radius is None:
        search_radius = int(getattr(laser_config, "XCORR_SEARCH_RADIUS", 8))

    min_score = float(getattr(laser_config, "LASER_XCORR_MIN_SCORE", 0.10))
    min_confidence = float(getattr(laser_config, "LASER_XCORR_MIN_CONFIDENCE", 0.13))
    margin_weight = float(getattr(laser_config, "LASER_XCORR_PEAK_MARGIN_WEIGHT", 0.80))
    sharpness_weight = float(getattr(laser_config, "LASER_XCORR_PEAK_SHARPNESS_WEIGHT", 0.60))

    guided_shifts, _, guided_peak_margins, guided_peak_sharpness = _guided_template_xcorr_full(
        ref_image,
        tgt_image,
        centers_xy_px,
        rows,
        cols,
        patch_half=patch_half,
        search_radius=search_radius,
        correction_factor=correction_factor,
        fallback_low_confidence=False,
    )
    legacy_shifts = guided_samebox_xcorr_shifts(
        ref_image,
        tgt_image,
        centers_xy_px,
        rows,
        cols,
        patch_half=patch_half,
        correction_factor=correction_factor,
    )

    guided_scores = guided_template_shift_scores(
        ref_image,
        tgt_image,
        centers_xy_px,
        guided_shifts,
        rows,
        cols,
        patch_half=patch_half,
    )
    legacy_scores = guided_template_shift_scores(
        ref_image,
        tgt_image,
        centers_xy_px,
        legacy_shifts,
        rows,
        cols,
        patch_half=patch_half,
    )

    guided_confidence = _guided_confidence_value(
        guided_scores,
        guided_peak_margins,
        guided_peak_sharpness,
        margin_weight=margin_weight,
        sharpness_weight=sharpness_weight,
    )
    guided_bonus = np.maximum(0.0, guided_confidence - guided_scores)
    guided_confident = guided_confidence >= min_confidence
    guided_select_score = guided_scores + 0.25 * guided_bonus
    use_guided = (guided_scores >= legacy_scores) | (
        guided_confident & (guided_select_score >= legacy_scores)
    )
    shifts = np.where(use_guided[..., None], guided_shifts, legacy_shifts)
    scores = np.where(use_guided, guided_scores, legacy_scores)
    confidence = np.where(use_guided, guided_confidence, legacy_scores)

    if bool(getattr(laser_config, "LASER_XCORR_ENABLE_SPATIAL_REPAIR", True)):
        outlier_px = float(getattr(laser_config, "LASER_XCORR_SPATIAL_OUTLIER_PX", 2.5))
        repair_tol = float(getattr(laser_config, "LASER_XCORR_REPAIR_SCORE_TOL", 0.03))

        median_shifts = _neighbor_median_shifts(shifts)
        median_scores = guided_template_shift_scores(
            ref_image,
            tgt_image,
            centers_xy_px,
            median_shifts,
            rows,
            cols,
            patch_half=patch_half,
        )

        residual = np.sqrt(np.sum((shifts - median_shifts) ** 2, axis=-1))
        repair_mask = (
            (scores < min_score)
            | (residual > outlier_px)
            | (use_guided & (guided_confidence < min_confidence))
        )
        # For extreme outliers (> 3× threshold), repair unconditionally
        extreme_outlier = residual > (outlier_px * 3.0)
        adopt_median = extreme_outlier | (repair_mask & (median_scores >= (scores - repair_tol)))
        shifts = np.where(adopt_median[..., None], median_shifts, shifts)

    return shifts




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
    phase_opd = integrate_slopes(slope_x, slope_y, spacing, method=integration_method)
    phase_rad = (2.0 * np.pi / wavelength) * phase_opd
    return {
        "displacement_x_m": displacement_x_m,
        "displacement_y_m": displacement_y_m,
        "phase_rad": phase_rad,
    }


def integrate_slopes(slope_x: np.ndarray, slope_y: np.ndarray,
                     spacing: float,
                     method: str = 'southwell') -> np.ndarray:
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
        return _integrate_slopes_southwell(slope_x, slope_y, spacing)
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
                                spacing: float) -> np.ndarray:
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
            row_idx[ptr] = eq;     col_idx[ptr] = idx_right;  val[ptr] = 1.0;   ptr += 1
            row_idx[ptr] = eq;     col_idx[ptr] = idx_left;   val[ptr] = -1.0;  ptr += 1
            b[eq] = 0.5 * (slope_x[i, j] + slope_x[i, j + 1]) * spacing
            eq += 1

    # Vertical equations: φ(i+1,j) - φ(i,j) = average_slope_y * spacing
    for i in range(rows - 1):
        for j in range(cols):
            idx_bot = i * cols + j
            idx_top = (i + 1) * cols + j
            row_idx[ptr] = eq;     col_idx[ptr] = idx_top;    val[ptr] = 1.0;   ptr += 1
            row_idx[ptr] = eq;     col_idx[ptr] = idx_bot;    val[ptr] = -1.0;  ptr += 1
            b[eq] = 0.5 * (slope_y[i, j] + slope_y[i + 1, j]) * spacing
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
