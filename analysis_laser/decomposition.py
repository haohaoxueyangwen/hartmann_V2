from __future__ import annotations
"""
Accumulated Frame Decomposition Module

When multiple pulses with different phase aberrations are accumulated,
this module provides methods to decompose the accumulated frame back
into individual pulse contributions, enabling wavefront reconstruction
for each pulse.

Methods:
1. Template-based decomposition using known mask positions
2. Iterative least-squares optimization
3. Weighted extraction based on known PSF shifts
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.ndimage import shift as ndshift, map_coordinates
from scipy.optimize import minimize
from dataclasses import dataclass

from analysis_laser.displacement_methods import integrate_slopes
from analysis_laser.subpixel_common import subpixel_peak_offset_2d


@dataclass
class DecomposedPulse:
    """Result of decomposing a single pulse from accumulated data."""
    index: int
    extracted_image: np.ndarray
    estimated_shifts: np.ndarray  # (rows, cols, 2) shift in pixels as (dx_px, dy_px)
    confidence: float  # Quality metric for extraction
    global_shift_xy_px: Tuple[float, float] = (0.0, 0.0)  # stored as (dx_px, dy_px)


@dataclass
class MovementContext:
    """运动模式上下文，传入 decomposition 用于正确生成预测模板。

    当 mode='rotation' 时，_build_trajectory_templates 会用坐标旋转
    （而非像素平移）来生成预测帧，避免混合模糊。
    """
    mode: str = 'translation'  # 'translation' | 'rotation' | 'none'
    rotation_angles_rad: Optional[List[float]] = None
    rotation_center_xy_px: Optional[Tuple[float, float]] = None


def _extract_pulse_global_offset_m(pulse_record: Dict) -> Tuple[float, float]:
    """Extract a pulse global offset in `(dx_m, dy_m)` order from a pulse record."""
    if "global_shift_xy_m" in pulse_record:
        arr = np.asarray(pulse_record["global_shift_xy_m"], dtype=np.float64).reshape(-1)
        if arr.size != 2:
            raise ValueError(
                f"global_shift_xy_m must contain exactly 2 values, got shape {np.asarray(pulse_record['global_shift_xy_m']).shape}"
            )
        return float(arr[0]), float(arr[1])
    if "shift_xy_m" in pulse_record:
        arr = np.asarray(pulse_record["shift_xy_m"], dtype=np.float64).reshape(-1)
        if arr.size != 2:
            raise ValueError(
                f"shift_xy_m must contain exactly 2 values, got shape {np.asarray(pulse_record['shift_xy_m']).shape}"
            )
        return float(arr[0]), float(arr[1])
    return float(pulse_record.get("shift_x_m", 0.0)), float(pulse_record.get("shift_y_m", 0.0))


def decompose_accumulated_frame(
    accumulated_image: np.ndarray,
    ideal_accumulated_image: np.ndarray,
    pulse_ideal_images: List[np.ndarray],
    pulse_shifts_m: List[Tuple[float, float]],
    pixel_size_m: float,
    method: str = "weighted",
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
    movement: Optional[MovementContext] = None,
) -> List[DecomposedPulse]:
    """
    Decompose an accumulated frame into individual pulse contributions.
    
    Parameters:
    -----------
    accumulated_image : np.ndarray
        The accumulated detector image (sum of all pulses with aberrations)
    ideal_accumulated_image : np.ndarray
        The accumulated ideal image (sum of all pulses without aberrations)
    pulse_ideal_images : List[np.ndarray]
        List of ideal images for each pulse (different mask positions)
    pulse_shifts_m : List[Tuple[float, float]]
        Known `(dx_m, dy_m)` global shift for each pulse
    pixel_size_m : float
        Detector pixel size in meters
    method : str
        Decomposition method: "weighted", "iterative", or "template"
    grid_rows, grid_cols : int or None
        Explicit Hartmann grid shape. These must be provided to avoid silent
        grid-shape mismatches.
        
    Returns:
    --------
    List[DecomposedPulse] : Extracted pulse data
    """
    n_pulses = len(pulse_ideal_images)
    if len(pulse_shifts_m) != n_pulses:
        raise ValueError(
            f"pulse_shifts_m length {len(pulse_shifts_m)} does not match n_pulses={n_pulses}"
        )
    if float(pixel_size_m) <= 0:
        raise ValueError(f"pixel_size_m must be positive, got {pixel_size_m}")
    shifts_px = _coerce_shifts_px(pulse_shifts_m, float(pixel_size_m), n_pulses)
    
    if method == "weighted":
        return _weighted_decomposition(
            accumulated_image, ideal_accumulated_image,
            pulse_ideal_images, shifts_px, pixel_size_m,
            grid_rows=grid_rows, grid_cols=grid_cols,
            movement=movement,
        )
    elif method == "iterative":
        return _iterative_decomposition(
            accumulated_image, ideal_accumulated_image,
            pulse_ideal_images, shifts_px, pixel_size_m,
            grid_rows=grid_rows, grid_cols=grid_cols,
            movement=movement,
        )
    elif method == "template":
        return _template_decomposition(
            accumulated_image, ideal_accumulated_image,
            pulse_ideal_images, shifts_px, pixel_size_m,
            grid_rows=grid_rows, grid_cols=grid_cols,
            movement=movement,
        )
    else:
        raise ValueError(f"Unknown decomposition method: {method}")


def _weighted_decomposition(
    accumulated: np.ndarray,
    ideal_accumulated: np.ndarray,
    pulse_ideals: List[np.ndarray],
    shifts_px: np.ndarray,
    pixel_size_m: float,
    *,
    grid_rows: Optional[int],
    grid_cols: Optional[int],
    movement: Optional[MovementContext] = None,
) -> List[DecomposedPulse]:
    """
    Weighted decomposition based on ideal image contributions.
    
    For each pixel, weight the accumulated signal by the relative
    contribution of each pulse's ideal pattern.
    
    This assumes: accumulated ≈ sum_i (ideal_i * (1 + aberration_effect_i))
    """
    n_pulses = len(pulse_ideals)
    results = []
    effective_templates = _build_trajectory_templates(pulse_ideals, shifts_px, movement=movement)
    
    # Stack ideal images
    ideal_stack = np.stack(effective_templates, axis=0)  # (n_pulses, H, W)
    
    # Compute weights: each pulse's contribution at each pixel
    # Weight = ideal_i / sum(ideal_j) with smoothing to avoid division by zero
    total_ideal = np.sum(ideal_stack, axis=0) + 1e-10
    weights = ideal_stack / total_ideal[np.newaxis, :, :]  # (n_pulses, H, W)
    
    # Extract each pulse's contribution
    for i in range(n_pulses):
        # Weighted extraction
        extracted = accumulated * weights[i]
        
        # Estimate shifts by comparing with ideal
        # This is a simplified version - full implementation would use
        # cross-correlation per spot
        shifts = _estimate_local_shifts(
            extracted,
            effective_templates[i],
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )
        
        # Confidence based on correlation with ideal pattern
        correlation = _compute_correlation(extracted, effective_templates[i])
        
        results.append(DecomposedPulse(
            index=i,
            extracted_image=extracted,
            estimated_shifts=shifts,
            confidence=correlation,
            global_shift_xy_px=(float(shifts_px[i, 0]), float(shifts_px[i, 1])),
        ))
    
    return results


def _iterative_decomposition(
    accumulated: np.ndarray,
    ideal_accumulated: np.ndarray,
    pulse_ideals: List[np.ndarray],
    shifts_px: np.ndarray,
    pixel_size_m: float,
    n_iterations: int = 10,
    *,
    grid_rows: Optional[int],
    grid_cols: Optional[int],
    movement: Optional[MovementContext] = None,
) -> List[DecomposedPulse]:
    """
    Iterative refinement decomposition.
    
    Start with weighted estimate, then iteratively refine each pulse's
    contribution by subtracting other pulses' contributions.
    """
    n_pulses = len(pulse_ideals)
    
    # Initialize with weighted decomposition
    initial = _weighted_decomposition(
        accumulated, ideal_accumulated, pulse_ideals, shifts_px, pixel_size_m,
        grid_rows=grid_rows, grid_cols=grid_cols,
        movement=movement,
    )
    effective_templates = _build_trajectory_templates(pulse_ideals, shifts_px, movement=movement)
    
    # Current estimates
    estimates = [d.extracted_image.copy() for d in initial]
    
    for iteration in range(n_iterations):
        for i in range(n_pulses):
            # Compute residual: accumulated minus all other pulses
            other_contribution = sum(
                estimates[j] for j in range(n_pulses) if j != i
            )
            residual = accumulated - other_contribution
            
            # Constrain residual to be non-negative and match ideal pattern
            # Use ideal as a mask to constrain the solution
            mask = effective_templates[i] > effective_templates[i].max() * 0.01
            estimates[i] = np.where(mask, np.maximum(0, residual), 0)
    
    # Build results
    results = []
    for i in range(n_pulses):
        shifts = _estimate_local_shifts(
            estimates[i],
            effective_templates[i],
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )
        correlation = _compute_correlation(estimates[i], effective_templates[i])
        
        results.append(DecomposedPulse(
            index=i,
            extracted_image=estimates[i],
            estimated_shifts=shifts,
            confidence=correlation,
            global_shift_xy_px=(float(shifts_px[i, 0]), float(shifts_px[i, 1])),
        ))
    
    return results


def _template_decomposition(
    accumulated: np.ndarray,
    ideal_accumulated: np.ndarray,
    pulse_ideals: List[np.ndarray],
    shifts_px: np.ndarray,
    pixel_size_m: float,
    *,
    grid_rows: Optional[int],
    grid_cols: Optional[int],
    movement: Optional[MovementContext] = None,
) -> List[DecomposedPulse]:
    """
    Template-based decomposition using optimization.
    
    Model: accumulated = sum_i (alpha_i * shift(ideal_i, delta_i))
    
    Optimize for alpha_i (intensity) and delta_i (local shifts) for each pulse.
    """
    n_pulses = len(pulse_ideals)
    H, W = accumulated.shape
    effective_templates = _build_trajectory_templates(pulse_ideals, shifts_px, movement=movement)
    
    # Simplified version: assume each pulse contributes equally
    # and estimate local shifts from correlation
    
    results = []
    remaining = accumulated.copy()
    
    for i in range(n_pulses):
        # Estimate this pulse's contribution
        # Use correlation to find the best match with scaled ideal
        
        # Find optimal scaling
        mask = effective_templates[i] > effective_templates[i].max() * 0.01
        if mask.sum() > 0:
            # Linear regression: remaining ≈ alpha * ideal
            alpha = np.sum(remaining * effective_templates[i] * mask) / (np.sum(effective_templates[i]**2 * mask) + 1e-10)
            alpha = max(0, min(alpha, n_pulses))  # Constrain to reasonable range
        else:
            alpha = 1.0
        
        # Extract this pulse's contribution
        extracted = effective_templates[i] * alpha
        
        # Estimate local shifts
        shifts = _estimate_local_shifts(
            remaining,
            effective_templates[i],
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )
        
        # Update remaining for next iteration
        remaining = np.maximum(0, remaining - extracted)
        
        # Confidence
        correlation = _compute_correlation(extracted, effective_templates[i])
        
        results.append(DecomposedPulse(
            index=i,
            extracted_image=extracted,
            estimated_shifts=shifts,
            confidence=correlation,
            global_shift_xy_px=(float(shifts_px[i, 0]), float(shifts_px[i, 1])),
        ))
    
    return results


def _coerce_shifts_px(
    pulse_shifts_m: List[Tuple[float, float]],
    pixel_size_m: float,
    n_pulses: int,
) -> np.ndarray:
    shifts_px = np.asarray(pulse_shifts_m, dtype=np.float64)
    if shifts_px.ndim != 2 or shifts_px.shape != (n_pulses, 2):
        raise ValueError(
            f"pulse_shifts_m must have shape ({n_pulses}, 2), got {shifts_px.shape}"
        )
    return shifts_px / float(pixel_size_m)


def _rotate_image_coords(
    image: np.ndarray,
    angle_rad: float,
    center_xy_px: Tuple[float, float],
) -> np.ndarray:
    """对图像做坐标级旋转。

    使用反向映射 + 双线性插值。旋转角极小（~0.008°/帧）时
    插值误差在 1e-8 量级，远小于物理噪声。

    Parameters
    ----------
    image : (H, W) array
    angle_rad : 旋转角（弧度），正 = 逆时针
    center_xy_px : 旋转中心的 (x_px, y_px) 像素坐标
    """
    if abs(angle_rad) < 1e-15:
        return image.copy()
    H, W = image.shape
    cx, cy = float(center_xy_px[0]), float(center_xy_px[1])
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dx = xx - cx
    dy = yy - cy
    # 反向旋转：找源像素坐标
    src_x = cx + dx * cos_a + dy * sin_a
    src_y = cy - dx * sin_a + dy * cos_a

    return map_coordinates(
        image, [src_y, src_x], order=1, mode='constant', cval=0.0,
    )


def _build_trajectory_templates(
    pulse_ideals: List[np.ndarray],
    shifts_px: np.ndarray,
    *,
    movement: Optional[MovementContext] = None,
) -> List[np.ndarray]:
    """Blend measured per-pulse ideals with a template predicted along the known trajectory.

    When *movement* is provided and its mode is 'rotation', coordinate-level
    rotation (instead of pixel-level linear shift) is used to generate the
    predicted template.  This avoids the sub-pixel blur that arises from
    mixing a rotated image with a linearly-shifted one.
    """
    n_pulses = len(pulse_ideals)
    if n_pulses == 0:
        return []

    ref_template = np.asarray(pulse_ideals[0], dtype=np.float64)
    ref_shift = np.asarray(shifts_px[0], dtype=np.float64)
    blended = []

    use_rotation = (
        movement is not None
        and movement.mode == 'rotation'
        and movement.rotation_angles_rad is not None
        and movement.rotation_center_xy_px is not None
        and len(movement.rotation_angles_rad) >= n_pulses
    )

    # When the rotation center is extremely far from the image (R >> image size),
    # the arc is physically indistinguishable from a straight line.  In this regime
    # map_coordinates suffers from floating-point precision loss due to the huge
    # coordinate offsets, while ndshift remains numerically stable.
    # Threshold: fall back to ndshift when center distance > 10× image diagonal.
    if use_rotation:
        H, W = ref_template.shape
        cx, cy = movement.rotation_center_xy_px
        center_dist = float(np.hypot(cx - W / 2.0, cy - H / 2.0))
        image_diag = float(np.hypot(H, W))
        if center_dist > 10.0 * image_diag:
            use_rotation = False

    for i in range(n_pulses):
        current = np.asarray(pulse_ideals[i], dtype=np.float64)

        if use_rotation:
            # 旋转模式：用坐标旋转生成预测
            delta_angle = (
                float(movement.rotation_angles_rad[i])
                - float(movement.rotation_angles_rad[0])
            )
            predicted = _rotate_image_coords(
                ref_template, delta_angle, movement.rotation_center_xy_px,
            )
        else:
            # 平移模式：原始 ndshift 逻辑
            delta_xy = np.asarray(shifts_px[i], dtype=np.float64) - ref_shift
            predicted = ndshift(
                ref_template,
                shift=(float(delta_xy[1]), float(delta_xy[0])),
                order=1,
                mode='constant',
                cval=0.0,
                prefilter=False,
            )
        # Physical improvement: The 'current' frame from pulse_ideals is the
        # exact physical ground truth (with all coordinates/diffraction accurately simulated).
        # Averaging it with 'predicted' (a purely geometric shift/rotation) introduces
        # a slight sub-pixel blur due to numerical interpolation. 
        # By trusting the physical image 100%, we eliminate this artificial blurring error.
        template = current
        blended.append(np.asarray(template, dtype=np.float64))
    return blended


def _estimate_local_shifts(
    image: np.ndarray,
    reference: np.ndarray,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
) -> np.ndarray:
    """
    Estimate local shifts by cross-correlation in grid cells.
    
    Returns shifts in pixels with shape `(grid_rows, grid_cols, 2)` as `(dx_px, dy_px)`.
    """
    if grid_rows is None or grid_cols is None:
        raise ValueError(
            "grid_rows and grid_cols must be provided explicitly for decomposition; "
            "legacy 8x8 fallback has been removed to avoid silent grid-shape errors."
        )

    resolved_grid_rows = int(grid_rows)
    resolved_grid_cols = int(grid_cols)
    if resolved_grid_rows <= 0 or resolved_grid_cols <= 0:
        raise ValueError(
            f"grid_rows and grid_cols must be positive, got {resolved_grid_rows}x{resolved_grid_cols}"
        )

    H, W = image.shape
    cell_h = H // resolved_grid_rows
    cell_w = W // resolved_grid_cols
    if cell_h <= 0 or cell_w <= 0:
        raise ValueError(
            f"image shape {image.shape} is too small for grid {resolved_grid_rows}x{resolved_grid_cols}"
        )
    
    shifts = np.zeros((resolved_grid_rows, resolved_grid_cols, 2))
    
    for r in range(resolved_grid_rows):
        for c in range(resolved_grid_cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            
            img_patch = image[y0:y1, x0:x1]
            ref_patch = reference[y0:y1, x0:x1]
            
            # Cross-correlation
            shift_x_px, shift_y_px = _subpixel_shift(ref_patch, img_patch)
            shifts[r, c, 0] = shift_x_px
            shifts[r, c, 1] = shift_y_px

    # Smooth displacement field: real wavefronts are low-frequency (Zernike),
    # so high-freq pixel-to-pixel noise is unphysical and can be suppressed.
    from scipy.ndimage import gaussian_filter as _gf
    _smooth_sigma = 0.7  # mild: preserves structure, removes pixel noise
    shifts[:, :, 0] = _gf(shifts[:, :, 0], sigma=_smooth_sigma)
    shifts[:, :, 1] = _gf(shifts[:, :, 1], sigma=_smooth_sigma)
    
    return shifts


def _subpixel_shift(reference: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """Estimate subpixel shift between two patches using cross-correlation.

    Returns `(dx_px, dy_px)`.
    """
    from scipy.signal import fftconvolve
    
    # Normalize
    ref_norm = reference - reference.mean()
    tgt_norm = target - target.mean()
    
    if ref_norm.std() < 1e-10 or tgt_norm.std() < 1e-10:
        return 0.0, 0.0
    
    ref_norm = ref_norm / ref_norm.std()
    tgt_norm = tgt_norm / tgt_norm.std()
    
    # Cross-correlate using FFT (mathematically identical to correlate2d, ~100x faster)
    corr = fftconvolve(tgt_norm, ref_norm[::-1, ::-1], mode='same')
    
    # Find peak
    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    center = (corr.shape[0] // 2, corr.shape[1] // 2)
    
    y, x = peak_idx
    dx, dy = subpixel_peak_offset_2d(corr, y, x, method="parabolic")
    
    shift_y_px = (y + dy) - center[0]
    shift_x_px = (x + dx) - center[1]

    return shift_x_px, shift_y_px


def _compute_correlation(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute normalized correlation between two images."""
    i1 = image1.ravel()
    i2 = image2.ravel()
    
    if i1.std() < 1e-10 or i2.std() < 1e-10:
        return 0.0
    
    return float(np.corrcoef(i1, i2)[0, 1])


def extract_per_pulse_wavefront(
    accumulated_image: np.ndarray,
    ideal_accumulated_image: np.ndarray,
    pulse_data: List[Dict],
    pixel_size_m: float,
    z_prop_m: float,
    wavelength_m: float,
    spacing_m: float,
    method: str = "weighted",
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
) -> List[Dict]:
    """
    High-level function to extract per-pulse wavefront from accumulated data.
    
    Parameters:
    -----------
    accumulated_image : np.ndarray
        Accumulated detector image
    ideal_accumulated_image : np.ndarray
        Accumulated ideal image
    pulse_data : List[Dict]
        List of pulse information including ideal_image and shifts
        pixel_size_m, z_prop_m, wavelength_m, spacing_m : float
        Physical parameters
    method : str
        Decomposition method
    grid_rows, grid_cols : int or None
        Explicit Hartmann grid shape for local shift estimation.
        
    Returns:
    --------
    List[Dict] : Per-pulse wavefront reconstruction results
    """
    # Extract ideal images and shifts
    pulse_ideals = [p['ideal_image'] for p in pulse_data]
    pulse_global_offsets_m = [_extract_pulse_global_offset_m(pulse_record) for pulse_record in pulse_data]
    
    # Decompose
    decomposed = decompose_accumulated_frame(
        accumulated_image,
        ideal_accumulated_image,
        pulse_ideals,
        pulse_global_offsets_m,
        pixel_size_m,
        method=method,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )
    
    # Convert shifts to wavefront
    results = []
    for d in decomposed:
        # Convert pixel shifts `(dx_px, dy_px)` to physical displacements.
        displacement_x_m = d.estimated_shifts[:, :, 0] * pixel_size_m
        displacement_y_m = d.estimated_shifts[:, :, 1] * pixel_size_m
        
        # Convert to slopes
        slope_x = displacement_x_m / z_prop_m
        slope_y = displacement_y_m / z_prop_m
        
        # Integrate to phase
        phase_opd = integrate_slopes(slope_x, slope_y, spacing_m, method="trapezoidal")
        phase_rad = (2.0 * np.pi / wavelength_m) * phase_opd
        
        results.append({
            'pulse_index': d.index,
            'extracted_image': d.extracted_image,
            'displacement_x_m': displacement_x_m,
            'displacement_y_m': displacement_y_m,
            'slope_x_rad': slope_x,
            'slope_y_rad': slope_y,
            'phase_rad': phase_rad,
            'confidence': d.confidence
        })
    
    return results
