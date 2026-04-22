#!/usr/bin/env python3
from __future__ import annotations
"""
Step 2 (Laser Fork): 相位恢复 (Phase Recovery)

从 Step 1 (simulate.py) 的输出加载探测器图像，
执行位移检测或优化求解进行波前重建。

流程 A (位移检测法):
    python recover_laser.py result/laser/001 --method centroid
    python recover_laser.py result/laser/001 --method cross_correlation

流程 B (优化求解法):
    python recover_laser.py result/laser/001 --method optimization

运行所有方法:
    python recover_laser.py result/laser/001 --method all
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from plot_style import apply_plot_font_bump

apply_plot_font_bump()

from analysis_laser.optimization_common import (
    GridGeometry,
    assert_row_col_indices,
    assert_xy_points,
    build_grid_geometry_from_axes,
)
import config_recover_laser as config

from analysis_laser.displacement_methods import (
    detect_beamlets,
    compute_centroids,
    extract_patches,
    centroid_shifts,
    cross_correlation_shifts,
    guided_samebox_xcorr_shifts,
    recover_from_shifts,
    integrate_slopes,
    remove_plane,
)
from analysis_laser.movement_metadata import (
    MovementInfo,
    get_global_offsets_m,
    get_global_offsets_px_with_pixelsize,
    parse_movement_info as _parse_movement_info_impl,
)


# =============================================================================
# Load Step 1 outputs
# =============================================================================

def _legacy_movement_metadata_dict(info: MovementInfo, *, pixel_size_m: float | None) -> dict:
    """Convert shared movement metadata to the compatibility Step 2 dict shape."""
    offsets_xy_m = get_global_offsets_m(info).tolist()
    offsets_xy_px = None
    if pixel_size_m not in (None, 0, 0.0):
        offsets_xy_px = get_global_offsets_px_with_pixelsize(info, float(pixel_size_m)).tolist()

    relative_offsets_xy_px = None
    if info.pulse_offsets_xy_px is not None:
        relative_offsets_xy_px = info.pulse_offsets_xy_px.tolist()

    relative_offsets_xy_m = None
    if info.pulse_offsets_xy_m is not None:
        relative_offsets_xy_m = [list(offset_xy_m) for offset_xy_m in info.pulse_offsets_xy_m]

    return {
        "mode": info.mode,
        "is_uniform": info.is_uniform,
        "n_pulses": info.n_pulses,
        "initial_offset_xy_m": list(info.initial_offset_xy_m),
        "uniform_step_m": info.uniform_step_m,
        "uniform_angle_deg": info.uniform_angle_deg,
        "pulse_offsets_xy_m": offsets_xy_m,
        "pulse_offsets_xy_px": offsets_xy_px,
        "pulse_offsets_relative_xy_m": relative_offsets_xy_m,
        "pulse_offsets_relative_xy_px": relative_offsets_xy_px,
        "generated_from": info.generated_from,
    }


def _movement_info_from_params(params: dict) -> MovementInfo:
    """Parse raw simulation params into the shared canonical MovementInfo object."""
    return _parse_movement_info_impl(params)


def _movement_metadata_from_params(params: dict) -> dict:
    """Build normalized movement metadata from raw simulation params."""
    info = _movement_info_from_params(params)
    return _legacy_movement_metadata_dict(info, pixel_size_m=params.get('pixel_size_m'))


def _parse_movement_info(params: dict) -> dict:
    """Compatibility-only wrapper around normalized movement metadata parsing.

    Internal code should prefer `_resolve_movement_metadata(...)` or
    `_movement_metadata_from_params(...)`.
    """
    return _movement_metadata_from_params(params)


def _resolve_per_frame_displacement_stacks(payload: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return canonical per-frame displacement stacks without mutating the payload."""
    displacement_x_stack = payload.get('all_displacement_x_m')
    displacement_y_stack = payload.get('all_displacement_y_m')
    legacy_x_stack = payload.get('all_disp_x_m')
    legacy_y_stack = payload.get('all_disp_y_m')

    if displacement_x_stack is None:
        displacement_x_stack = legacy_x_stack
    if displacement_y_stack is None:
        displacement_y_stack = legacy_y_stack
    return displacement_x_stack, displacement_y_stack


def _movement_array_field(movement: dict, field_name: str) -> np.ndarray | None:
    """Return a movement metadata array field as `float64`, or `None` when absent."""
    field_value = movement.get(field_name)
    if field_value is None:
        return None
    return np.asarray(field_value, dtype=np.float64)


def _movement_optional_string_field(movement: dict, field_name: str) -> str | None:
    """Return an optional non-empty string movement field."""
    field_value = movement.get(field_name)
    if field_value in (None, ''):
        return None
    return str(field_value)


def _movement_optional_scalar_field(movement: dict, field_name: str) -> float | None:
    """Return an optional scalar movement field when present and non-zero."""
    field_value = movement.get(field_name)
    if field_value in (None, 0, 0.0):
        return None
    return float(field_value)


def _movement_global_offsets_px(movement: dict) -> np.ndarray | None:
    """Return absolute global pulse offsets in pixels from loaded movement metadata."""
    return _movement_array_field(movement, 'pulse_offsets_xy_px')


def _movement_global_offsets_m(movement: dict) -> np.ndarray | None:
    """Return absolute global pulse offsets in meters from loaded movement metadata."""
    return _movement_array_field(movement, 'pulse_offsets_xy_m')


def _movement_relative_offsets_px(movement: dict) -> np.ndarray | None:
    """Return relative pulse offsets in pixels when available."""
    return _movement_array_field(movement, 'pulse_offsets_relative_xy_px')


def _movement_relative_offsets_m(movement: dict) -> np.ndarray | None:
    """Return relative pulse offsets in meters when available."""
    return _movement_array_field(movement, 'pulse_offsets_relative_xy_m')


def _movement_initial_offset_m(movement: dict) -> np.ndarray:
    """Return the absolute pulse-0 mechanical offset in meters."""
    initial_offset_xy_m = movement.get('initial_offset_xy_m')
    if initial_offset_xy_m is None:
        return np.zeros(2, dtype=np.float64)
    initial_offset_xy_m = np.asarray(initial_offset_xy_m, dtype=np.float64)
    if initial_offset_xy_m.shape != (2,):
        raise ValueError(f"movement initial_offset_xy_m must have shape (2,), got {initial_offset_xy_m.shape}")
    return initial_offset_xy_m


def _movement_has_global_offsets_px(movement: dict) -> bool:
    """Return whether movement metadata includes absolute pulse offsets in pixels."""
    return _movement_global_offsets_px(movement) is not None


def _movement_mode(movement: dict) -> str:
    """Return the configured movement mode with a stable default."""
    return _movement_optional_string_field(movement, 'mode') or 'none'


def _movement_is_uniform(movement: dict) -> bool:
    """Return whether the current movement metadata represents a uniform scan."""
    return bool(movement.get('is_uniform', False))


def _movement_uniform_step_m(movement: dict) -> float | None:
    """Return the configured uniform step length in meters when available."""
    return _movement_optional_scalar_field(movement, 'uniform_step_m')


def _movement_generated_from(movement: dict) -> str | None:
    """Return the movement-metadata source marker when available."""
    return _movement_optional_string_field(movement, 'generated_from')


def _normalize_loaded_movement_metadata(movement: dict, *, pixel_size_m: float | None) -> dict:
    """Normalize legacy loaded movement dicts to the current absolute+relative contract."""
    normalized = dict(movement)
    initial_offset_xy_m = _movement_initial_offset_m(normalized)

    relative_offsets_xy_m = _movement_relative_offsets_m(normalized)
    absolute_offsets_xy_m = _movement_global_offsets_m(normalized)
    if absolute_offsets_xy_m is None and relative_offsets_xy_m is not None:
        absolute_offsets_xy_m = relative_offsets_xy_m + initial_offset_xy_m[np.newaxis, :]
        normalized['pulse_offsets_xy_m'] = absolute_offsets_xy_m.tolist()
    elif relative_offsets_xy_m is None and absolute_offsets_xy_m is not None:
        relative_offsets_xy_m = absolute_offsets_xy_m - initial_offset_xy_m[np.newaxis, :]
        normalized['pulse_offsets_relative_xy_m'] = relative_offsets_xy_m.tolist()

    relative_offsets_xy_px = _movement_relative_offsets_px(normalized)
    absolute_offsets_xy_px = _movement_global_offsets_px(normalized)
    if pixel_size_m not in (None, 0, 0.0):
        pixel_size_m = float(pixel_size_m)
        if absolute_offsets_xy_px is None and absolute_offsets_xy_m is not None:
            absolute_offsets_xy_px = absolute_offsets_xy_m / pixel_size_m
            normalized['pulse_offsets_xy_px'] = absolute_offsets_xy_px.tolist()
        if relative_offsets_xy_px is None and relative_offsets_xy_m is not None:
            relative_offsets_xy_px = relative_offsets_xy_m / pixel_size_m
            normalized['pulse_offsets_relative_xy_px'] = relative_offsets_xy_px.tolist()

    return normalized


def _resolve_movement_metadata(*, data: dict | None = None, params: dict | None = None) -> dict:
    """Return normalized movement metadata from loaded data or raw params."""
    if data is not None:
        loaded_movement = data.get('movement')
        if loaded_movement is not None:
            loaded_params = params or data.get('params', {})
            return _normalize_loaded_movement_metadata(
                loaded_movement,
                pixel_size_m=loaded_params.get('pixel_size_m'),
            )
        if params is None:
            params = data.get('params', {})
    return _movement_metadata_from_params(params or {})


def _load_pulse_ideal_images(sim_dir: str, n_pulses: int):
    """Load per-pulse ideal detector images if the simulation saved them."""
    if int(n_pulses) <= 0:
        return None
    frames_dir = os.path.join(sim_dir, "frames")
    if not os.path.isdir(frames_dir):
        return None

    pulse_ideal_images = []
    for i in range(int(n_pulses)):
        ideal_path = os.path.join(frames_dir, f"frame_{i:03d}_ideal.npy")
        if not os.path.exists(ideal_path):
            return None
        pulse_ideal_images.append(np.load(ideal_path))
    return pulse_ideal_images


def _load_reference_ideal_image(sim_dir: str):
    """Load a single ideal reference image when available."""
    reference_path = os.path.join(sim_dir, "reference_ideal.npy")
    if os.path.exists(reference_path):
        return np.load(reference_path)
    fallback_path = os.path.join(sim_dir, "frames", "frame_000_ideal.npy")
    if os.path.exists(fallback_path):
        return np.load(fallback_path)
    return None


def _load_simulation_params(sim_dir: str) -> dict:
    """Load `simulation_params.json` from a Step 1 output directory."""
    params_path = os.path.join(sim_dir, "simulation_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"simulation_params.json not found in {sim_dir}.\n"
            f"Did you run: python simulate.py --output-dir {sim_dir} ?"
        )
    with open(params_path, "r") as fp:
        return json.load(fp)


def _load_required_accumulated_image(sim_dir: str) -> np.ndarray:
    """Load the required accumulated detector image from a Step 1 output directory."""
    image_path = os.path.join(sim_dir, "accumulated.npy")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"accumulated.npy not found in {sim_dir}")
    return np.load(image_path)


def _load_wfe_truth_assets(sim_dir: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load legacy 2D WFE truth and optional per-frame phase-screen stack."""
    wfe_path = os.path.join(sim_dir, "wfe_phase_map.npy")
    legacy_wfe_truth = np.load(wfe_path) if os.path.exists(wfe_path) else None

    phase_screens_path = os.path.join(sim_dir, "phase_screens.npy")
    wfe_truth_stack = np.load(phase_screens_path, mmap_mode='r') if os.path.exists(phase_screens_path) else None
    return legacy_wfe_truth, wfe_truth_stack


def _select_representative_wfe_truth(
    *,
    params: dict,
    legacy_wfe_truth: np.ndarray | None,
    wfe_truth_stack: np.ndarray | None,
) -> tuple[np.ndarray | None, str | None]:
    """Return a 2D WFE truth map compatible with existing recovery metrics.

    Averaging independent per-frame phase screens would not be a physically
    meaningful ground truth, so a representative 2D map is only exposed when
    that choice is unambiguous.
    """
    if legacy_wfe_truth is not None:
        return np.asarray(legacy_wfe_truth, dtype=np.float64), "wfe_phase_map.npy"

    if wfe_truth_stack is None:
        return None, None

    stack = np.asarray(wfe_truth_stack)
    if stack.ndim == 2:
        return np.asarray(stack, dtype=np.float64), "phase_screens.npy"
    if stack.ndim != 3 or stack.shape[0] == 0:
        raise ValueError(f"Unsupported phase_screens.npy shape: {stack.shape}")

    aberr_mode = str(params.get('phase_aberration_mode', 'none') or 'none')
    if stack.shape[0] == 1 or aberr_mode == 'per_pulse':
        return np.asarray(stack[0], dtype=np.float64), "phase_screens.npy[frame0]"
    return None, "phase_screens.npy stack only"


def _load_optional_simulation_outputs(sim_dir: str, *, params: dict) -> dict:
    """Load optional Step 1 artifacts used by recovery and evaluation."""
    legacy_wfe_truth, wfe_truth_stack = _load_wfe_truth_assets(sim_dir)
    wfe_truth, wfe_truth_source = _select_representative_wfe_truth(
        params=params,
        legacy_wfe_truth=legacy_wfe_truth,
        wfe_truth_stack=wfe_truth_stack,
    )

    ideal_path = os.path.join(sim_dir, "ideal_accumulated.npy")
    ideal_image = np.load(ideal_path) if os.path.exists(ideal_path) else None

    centers_path = os.path.join(sim_dir, "ideal_centers.npy")
    ideal_centers = np.load(centers_path) if os.path.exists(centers_path) else None
    if ideal_centers is not None:
        ideal_centers = assert_xy_points(ideal_centers, name="ideal_centers")

    grid_indices_path = os.path.join(sim_dir, "spot_grid_indices.npy")
    spot_grid_indices = np.load(grid_indices_path) if os.path.exists(grid_indices_path) else None
    if spot_grid_indices is not None:
        spot_grid_indices = assert_row_col_indices(spot_grid_indices, name="spot_grid_indices")

    gt_shifts_path = os.path.join(sim_dir, "gt_shifts.npy")
    gt_shifts = np.load(gt_shifts_path) if os.path.exists(gt_shifts_path) else None

    gt_disp_path = os.path.join(sim_dir, "gt_displacements.npz")
    gt_displacements = dict(np.load(gt_disp_path, allow_pickle=True)) if os.path.exists(gt_disp_path) else None

    return {
        "wfe_truth": wfe_truth,
        "wfe_truth_stack": wfe_truth_stack,
        "wfe_truth_source": wfe_truth_source,
        "ideal_image": ideal_image,
        "ideal_centers": ideal_centers,
        "spot_grid_indices": spot_grid_indices,
        "gt_shifts": gt_shifts,
        "gt_displacements": gt_displacements,
    }


def _print_loaded_simulation_summary(
    *,
    sim_dir: str,
    params: dict,
    image: np.ndarray,
    optional_outputs: dict,
    pulse_ideal_images,
    reference_ideal_image,
    movement: dict,
) -> None:
    """Print a compact summary of the loaded Step 1 dataset."""
    print(f"Loaded simulation data from: {sim_dir}")
    print(f"  Mode: {params.get('mode', 'full')}")
    print(f"  Image shape: {image.shape}")
    wfe_truth_source = optional_outputs.get('wfe_truth_source')
    if optional_outputs.get('wfe_truth') is not None:
        print(f"  WFE ground truth: Yes ({wfe_truth_source})")
    elif optional_outputs.get('wfe_truth_stack') is not None:
        print(f"  WFE ground truth: Stack only ({wfe_truth_source})")
    else:
        print("  WFE ground truth: No")
    print(f"  Ideal image: {'Yes' if optional_outputs.get('ideal_image') is not None else 'No'}")
    print(f"  Ideal centers: {'Yes' if optional_outputs.get('ideal_centers') is not None else 'No'}")
    print(f"  Spot grid indices: {'Yes' if optional_outputs.get('spot_grid_indices') is not None else 'No'}")
    print(f"  Pulse ideal images: {'Yes' if pulse_ideal_images is not None else 'No'}")
    print(f"  Reference ideal image: {'Yes' if reference_ideal_image is not None else 'No'}")
    print(f"  GT shifts: {'Yes' if optional_outputs.get('gt_shifts') is not None else 'No'}")
    print(f"  GT displacements: {'Yes' if optional_outputs.get('gt_displacements') is not None else 'No'}")
    print(f"  Movement: {_movement_mode(movement)} | uniform={_movement_is_uniform(movement)}")
    uniform_step_m = _movement_uniform_step_m(movement)
    if uniform_step_m is not None:
        print(f"  Uniform step: {uniform_step_m*1e6:.3f} um")
    generated_from = _movement_generated_from(movement)
    if generated_from is not None:
        print(f"  Movement source: {generated_from}")
    offsets_m = _movement_global_offsets_m(movement)
    if offsets_m is not None:
        first_xy = offsets_m[0]
        last_xy = offsets_m[-1]
        print(
            "  Global offsets (absolute): "
            f"first=({first_xy[0]*1e6:.3f}, {first_xy[1]*1e6:.3f}) um, "
            f"last=({last_xy[0]*1e6:.3f}, {last_xy[1]*1e6:.3f}) um"
        )
        if offsets_m.shape[0] >= 2:
            step_lengths_um = np.linalg.norm(np.diff(offsets_m, axis=0), axis=1) * 1e6
            print(
                "  Step length range: "
                f"{np.min(step_lengths_um):.3f} .. {np.max(step_lengths_um):.3f} um"
            )


def load_simulation_data(sim_dir: str) -> dict:
    """
    加载 Step 1 的输出数据。
    
    Returns dict with:
        image         : ndarray — 叠加的探测器图像
        params        : dict   — 仿真参数
        wfe_truth     : ndarray or None — WFE ground truth
    """
    # Check directory exists
    if not os.path.isdir(sim_dir):
        raise FileNotFoundError(f"Simulation directory not found: {sim_dir}")

    params = _load_simulation_params(sim_dir)
    image = _load_required_accumulated_image(sim_dir)
    optional_outputs = _load_optional_simulation_outputs(sim_dir, params=params)
    pulse_ideal_images = _load_pulse_ideal_images(sim_dir, params.get('n_pulses', 1))
    reference_ideal_image = _load_reference_ideal_image(sim_dir)
    movement = _resolve_movement_metadata(params=params)
    _print_loaded_simulation_summary(
        sim_dir=sim_dir,
        params=params,
        image=image,
        optional_outputs=optional_outputs,
        pulse_ideal_images=pulse_ideal_images,
        reference_ideal_image=reference_ideal_image,
        movement=movement,
    )

    return {
        "image": image,
        "ideal_image": optional_outputs["ideal_image"],
        "params": params,
        "wfe_truth": optional_outputs["wfe_truth"],
        "wfe_truth_stack": optional_outputs["wfe_truth_stack"],
        "wfe_truth_source": optional_outputs["wfe_truth_source"],
        "ideal_centers": optional_outputs["ideal_centers"],
        "spot_grid_indices": optional_outputs["spot_grid_indices"],
        "gt_shifts": optional_outputs["gt_shifts"],
        "gt_displacements": optional_outputs["gt_displacements"],
        "pulse_ideal_images": pulse_ideal_images,
        "reference_ideal_image": reference_ideal_image,
        "movement": movement,
        "sim_dir": sim_dir,
    }


# =============================================================================
# Core Recovery Pipeline
# =============================================================================

def estimate_grid_size(image_shape, pitch_m, pixel_size_m, window_size_m=None):
    """估算子孔径网格大小 (rows, cols).
    
    使用与 GT 侧 (_compute_gt_displacements) 完全一致的算法:
      1. grid_count = round(window_size / spacing)
      2. 生成物理坐标
      3. 过滤到探测器视场内
    
    如果 window_size_m 未提供，则回退到简单除法估算。
    """
    if window_size_m is not None:
        # === GT-consistent logic ===
        spacing = pitch_m
        grid_count = max(1, int(round(window_size_m / spacing)))
        coords = (np.arange(grid_count) - (grid_count - 1) / 2.0) * spacing
        det_w = image_shape[1] * pixel_size_m
        det_h = image_shape[0] * pixel_size_m
        x_coords = coords[np.abs(coords) < det_w / 2.0]
        y_coords = coords[np.abs(coords) < det_h / 2.0]
        rows = max(2, len(y_coords))
        cols = max(2, len(x_coords))
    else:
        # Fallback (e.g., simple mode without window_size)
        pitch_px = pitch_m / pixel_size_m
        rows = max(2, int(image_shape[0] / pitch_px))
        cols = max(2, int(image_shape[1] / pitch_px))
    return rows, cols


def _resolve_guided_centers_for_pulse(data: dict, movement: dict, pulse_index: int | None) -> np.ndarray | None:
    """Return ideal spot centers shifted to the requested pulse when available."""
    ideal_centers = data.get('ideal_centers')
    if ideal_centers is None:
        return None

    centers = np.asarray(ideal_centers, dtype=np.float64)
    offsets_px = _movement_global_offsets_px(movement)
    if pulse_index is None or offsets_px is None:
        return centers
    if pulse_index < 0 or pulse_index >= int(offsets_px.shape[0]):
        return centers
    return centers + np.asarray(offsets_px[pulse_index], dtype=np.float64)[None, :]


def run_displacement_method(
    ref_image,
    tgt_image,
    rows,
    cols,
    method_name,
    *,
    guided_centers_px: np.ndarray | None = None,
):
    """
    用指定方法计算位移。
    
    Parameters
    ----------
    ref_image : 参考图（理想或首帧）
    tgt_image : 目标图（叠加图）
    rows, cols: 子孔径网格
    method_name: 'centroid', 'cross_correlation'
    
    Returns
    -------
    shifts_px : ndarray of shape (rows, cols, 2)
    """
    if method_name == 'centroid':
        # Detect beamlets in reference image
        ref_boxes, ref_centroids = detect_beamlets(ref_image, rows, cols)
        # Centroid: 需要分别检测两幅图中的质心位置，然后做差
        tgt_boxes, tgt_centroids = detect_beamlets(tgt_image, rows, cols)
        shifts = centroid_shifts(ref_centroids, tgt_centroids)
    elif method_name == 'cross_correlation':
        if (
            guided_centers_px is not None
            and bool(getattr(config, 'LASER_XCORR_USE_GUIDED_CENTERS', True))
        ):
            shifts = guided_samebox_xcorr_shifts(
                ref_image,
                tgt_image,
                guided_centers_px,
                rows,
                cols,
                patch_half=getattr(config, 'XCORR_PATCH_HALF', None),
                correction_factor=config.XCORR_CORRECTION_FACTOR,
            )
        else:
            # Legacy blind patch correlation fallback.
            ref_boxes, ref_centroids = detect_beamlets(ref_image, rows, cols)
            ref_patches = extract_patches(ref_image, ref_boxes, rows, cols)
            tgt_patches = extract_patches(tgt_image, ref_boxes, rows, cols)
            shifts = cross_correlation_shifts(
                ref_patches, tgt_patches,
                correction_factor=config.XCORR_CORRECTION_FACTOR
            )
    else:
        raise ValueError(
            f"Unknown method: {method_name}. Use 'centroid' or 'cross_correlation'."
        )

    return shifts


def _shift_reference_image(
    image: np.ndarray,
    delta_xy_px,
    *,
    order: int | None = None,
) -> np.ndarray:
    """Shift a reference image by (dx, dy) in pixels."""
    arr = np.asarray(image, dtype=np.float64)
    dx, dy = float(delta_xy_px[0]), float(delta_xy_px[1])
    interp_order = int(
        getattr(config, 'REFERENCE_SHIFT_ORDER', 1) if order is None else order
    )
    return ndi.shift(
        arr,
        shift=(dy, dx),
        order=interp_order,
        mode='constant',
        cval=0.0,
        prefilter=interp_order > 1,
    )


def _estimate_base_reference_from_ideal_accumulated(
    ideal_accumulated: np.ndarray,
    global_offsets_px: np.ndarray,
    *,
    reference_pulse_index: int = 0,
    reg: float | None = None,
) -> np.ndarray:
    """Estimate a single base ideal reference from ideal_accumulated and known trajectory.

    Assumes:
        ideal_accumulated ~= sum_k shift(reference_image, global_offsets[k] - global_offsets[ref])
    """
    arr = np.asarray(ideal_accumulated, dtype=np.float64)
    offsets = np.asarray(global_offsets_px, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"ideal_accumulated must be 2D, got {arr.shape}")
    if offsets.ndim != 2 or offsets.shape[1] != 2:
        raise ValueError(f"global_offsets_px must have shape (n_pulses, 2), got {offsets.shape}")
    if offsets.shape[0] == 0:
        raise ValueError("global_offsets_px is empty")
    if not (0 <= int(reference_pulse_index) < offsets.shape[0]):
        raise ValueError(
            f"reference_pulse_index={reference_pulse_index} out of range for n_pulses={offsets.shape[0]}"
        )

    rel_offsets = offsets - offsets[int(reference_pulse_index)]
    h, w = arr.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    kernel_fft = np.zeros((h, w), dtype=np.complex128)
    for dx, dy in rel_offsets:
        kernel_fft += np.exp(-2j * np.pi * (fx * float(dx) + fy * float(dy)))

    obs_fft = np.fft.fft2(arr)
    power = np.abs(kernel_fft) ** 2
    reg_strength = float(
        getattr(config, 'REFERENCE_DECONV_REG', 1e-3) if reg is None else reg
    )
    reg_term = reg_strength * max(float(power.max()), 1.0)
    base_fft = obs_fft * np.conj(kernel_fft) / (power + reg_term)
    base = np.real(np.fft.ifft2(base_fft))
    return np.maximum(base, 0.0)


def _build_reference_sequence_from_single(
    reference_image: np.ndarray,
    global_offsets_px: np.ndarray,
    *,
    reference_pulse_index: int = 0,
) -> list[np.ndarray]:
    """Generate per-pulse reference images by shifting a single reference image."""
    offsets = np.asarray(global_offsets_px, dtype=np.float64)
    rel_offsets = offsets - offsets[int(reference_pulse_index)]
    return [
        _shift_reference_image(reference_image, delta_xy_px)
        for delta_xy_px in rel_offsets
    ]


def _build_pulse_reference_images(data: dict, movement: dict) -> tuple[list[np.ndarray] | None, str | None]:
    """Build per-pulse ideal references from the best available prior source."""
    n_pulses = int(data['params'].get('n_pulses', 1))
    mode = str(getattr(config, 'REFERENCE_SEQUENCE_MODE', 'auto') or 'auto').lower()
    global_offsets_px = _movement_global_offsets_px(movement)

    pulse_ideal_images = data.get('pulse_ideal_images')
    if mode in ('auto', 'per_pulse'):
        if pulse_ideal_images is not None and len(pulse_ideal_images) == n_pulses:
            return [np.asarray(img, dtype=np.float64) for img in pulse_ideal_images], 'per_pulse_saved'
        if mode == 'per_pulse':
            return None, None

    reference_ideal_image = data.get('reference_ideal_image')
    if mode in ('auto', 'single_reference'):
        if reference_ideal_image is not None and global_offsets_px is not None and global_offsets_px.shape[0] == n_pulses:
            refs = _build_reference_sequence_from_single(
                np.asarray(reference_ideal_image, dtype=np.float64),
                global_offsets_px,
                reference_pulse_index=0,
            )
            return refs, 'single_reference_shifted'
        if mode == 'single_reference':
            return None, None

    ideal_accumulated = data.get('ideal_image')
    if mode in ('auto', 'deconvolve'):
        if ideal_accumulated is not None and global_offsets_px is not None and global_offsets_px.shape[0] == n_pulses:
            base_reference = _estimate_base_reference_from_ideal_accumulated(
                np.asarray(ideal_accumulated, dtype=np.float64),
                global_offsets_px,
                reference_pulse_index=0,
            )
            refs = _build_reference_sequence_from_single(
                base_reference,
                global_offsets_px,
                reference_pulse_index=0,
            )
            return refs, 'deconvolved_ideal_accumulated'
        if mode == 'deconvolve':
            return None, None

    return None, None


def _build_movement_context(movement: dict, params: dict):
    """Build a MovementContext for the decomposition algorithm.

    For translation / none modes, returns a default context (decomposition
    falls back to ndshift, identical to previous behaviour).

    For rotation mode, reads rotation metadata from *params* (the raw
    simulation_params.json contents) and converts the rotation center from
    physical (metre) to detector pixel coordinates.
    """
    from analysis_laser.decomposition import MovementContext

    mode = _movement_mode(movement)
    if mode != 'rotation':
        return MovementContext(mode=mode)

    # Read rotation metadata written by simulate.py
    R_m = float(params.get('rotation_radius_m', 50e-3))
    tilt_deg = float(params.get('rotation_grid_tilt_deg', 0.0))
    angles = params.get('rotation_pulse_angles_rad', None)
    pixel_size_m = float(params.get('pixel_size_m', 0.65e-6))

    # Rotation center in physical coordinates (metres, beam centre = origin)
    normal_rad = np.deg2rad(tilt_deg + 90.0)
    cx_m = R_m * float(np.cos(normal_rad))
    cy_m = R_m * float(np.sin(normal_rad))

    # Convert to detector pixel coordinates (origin = top-left corner)
    det_nx = int(params.get('detector_nx', 1024))
    det_ny = int(params.get('detector_ny', det_nx))
    cx_px = det_nx / 2.0 + cx_m / pixel_size_m
    cy_px = det_ny / 2.0 - cy_m / pixel_size_m  # Y axis flipped

    return MovementContext(
        mode='rotation',
        rotation_angles_rad=angles,
        rotation_center_xy_px=(cx_px, cy_px),
    )


def _recover_accumulated_with_decomposition(
    data: dict,
    movement: dict,
    rows: int,
    cols: int,
    method_name: str,
) -> dict | None:
    """Trajectory-aware accumulated recovery using known per-pulse ideal images."""
    if not bool(getattr(config, 'ENABLE_ACCUMULATED_DECOMPOSITION', True)):
        return None

    ideal_accumulated = data.get('ideal_image')
    pulse_offsets_xy_m = _movement_global_offsets_m(movement)
    n_pulses = int(data['params'].get('n_pulses', 1))
    pulse_reference_images, reference_source = _build_pulse_reference_images(data, movement)
    if (
        ideal_accumulated is None or
        pulse_reference_images is None or
        pulse_offsets_xy_m is None or
        len(pulse_reference_images) != n_pulses or
        len(pulse_offsets_xy_m) != n_pulses or
        n_pulses <= 1
    ):
        return None

    from analysis_laser.decomposition import decompose_accumulated_frame

    decomp_method = str(
        getattr(config, 'ACCUMULATED_DECOMPOSITION_METHOD', 'weighted') or 'weighted'
    ).lower()
    print(
        f"  Mode: accumulated decomposition recovery ({n_pulses} pulses, "
        f"method={decomp_method}, references={reference_source})"
    )

    movement_ctx = _build_movement_context(movement, data['params'])

    decomposed = decompose_accumulated_frame(
        accumulated_image=np.asarray(data['image'], dtype=np.float64),
        ideal_accumulated_image=np.asarray(ideal_accumulated, dtype=np.float64),
        pulse_ideal_images=[np.asarray(img, dtype=np.float64) for img in pulse_reference_images],
        pulse_shifts_m=[tuple(float(v) for v in xy) for xy in pulse_offsets_xy_m],
        pixel_size_m=float(data['params']['pixel_size_m']),
        method=decomp_method,
        grid_rows=int(rows),
        grid_cols=int(cols),
        movement=movement_ctx,
    )
    decomposed = sorted(decomposed, key=lambda item: int(item.index))

    all_shifts = []
    decomposition_images = []
    decomposition_confidences = []
    for pulse in decomposed:
        ideal_pulse = pulse_reference_images[pulse.index]
        shifts = run_displacement_method(
            ideal_pulse,
            pulse.extracted_image,
            rows,
            cols,
            method_name,
            guided_centers_px=_resolve_guided_centers_for_pulse(data, movement, pulse.index),
        )
        all_shifts.append(shifts)
        decomposition_images.append(np.asarray(pulse.extracted_image, dtype=np.float64))
        decomposition_confidences.append(float(pulse.confidence))
        print(f"    Pulse {pulse.index}: decomposition confidence={pulse.confidence:.4f}")

    if not all_shifts:
        raise RuntimeError("Accumulated decomposition produced no per-pulse recoveries.")

    return {
        "all_shifts_arr": np.asarray(all_shifts, dtype=np.float64),
        "n_recovered": len(all_shifts),
        "decomposition_method": decomp_method,
        "reference_sequence_source": reference_source,
        "decomposition_images": np.asarray(decomposition_images, dtype=np.float64),
        "decomposition_confidences": decomposition_confidences,
    }


def _recover_per_frame_sequence(
    *,
    data: dict,
    rows: int,
    cols: int,
    method_name: str,
    movement: dict,
) -> dict:
    """Recover per-frame shifts from saved aberrated/ideal frame pairs.

    This path is intended for simulation-time evaluation only.
    """
    sim_dir = data['sim_dir']
    params = data['params']
    n_pulses = int(params.get('n_pulses', 1))
    frames_dir = os.path.join(sim_dir, "frames")
    if not os.path.isdir(frames_dir):
        raise RuntimeError("frames/ directory not found for per-frame evaluation")

    all_shifts = []
    for i in range(n_pulses):
        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.npy")
        ideal_path = os.path.join(frames_dir, f"frame_{i:03d}_ideal.npy")
        if not os.path.exists(frame_path) or not os.path.exists(ideal_path):
            print(f"    Warning: missing frame {i}, stopping at {i} frames")
            break

        frame = np.load(frame_path)
        ideal = np.load(ideal_path)
        shifts = run_displacement_method(
            ideal,
            frame,
            rows,
            cols,
            method_name,
            guided_centers_px=_resolve_guided_centers_for_pulse(data, movement, i),
        )
        all_shifts.append(shifts)

    if not all_shifts:
        raise RuntimeError("No valid per-frame pairs were recovered from frames/ directory.")

    all_shifts_arr = np.asarray(all_shifts, dtype=np.float64)
    return {
        "all_shifts_arr": all_shifts_arr,
        "n_recovered": int(all_shifts_arr.shape[0]),
    }


def _select_representative_shifts(all_shifts_arr: np.ndarray, aberr_mode: str) -> tuple[np.ndarray, str]:
    """Reduce a shift sequence to a representative map for legacy aggregate outputs."""
    if str(aberr_mode or 'none') == 'per_pulse':
        print("  [per_pulse mode] keep per-pulse sequence; frame 0 is used only for aggregate maps")
        return np.asarray(all_shifts_arr[0], dtype=np.float64), "frame0"

    return np.mean(np.asarray(all_shifts_arr, dtype=np.float64), axis=0), "mean"


def _recover_sequence_phase_products(
    all_shifts_arr: np.ndarray,
    *,
    pixel_size: float,
    z_prop: float,
    pitch: float,
    wavelength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover per-frame phase/displacement products from a shift sequence."""
    all_phase_list = []
    all_dx_list = []
    all_dy_list = []
    for shifts in np.asarray(all_shifts_arr, dtype=np.float64):
        frame_recovery = recover_from_shifts(
            shifts,
            pixel_size=pixel_size,
            z_prop=z_prop,
            spacing=pitch,
            wavelength=wavelength,
            integration_method=config.INTEGRATION_METHOD,
        )
        all_phase_list.append(frame_recovery['phase_rad'])
        all_dx_list.append(frame_recovery['displacement_x_m'])
        all_dy_list.append(frame_recovery['displacement_y_m'])

    return (
        np.asarray(all_phase_list, dtype=np.float64),
        np.asarray(all_dx_list, dtype=np.float64),
        np.asarray(all_dy_list, dtype=np.float64),
    )


def _resolve_recovery_grid(
    *,
    image_shape,
    gt_disp: dict | None,
    pitch: float,
    pixel_size: float,
    window_size: float | None,
) -> tuple[int, int]:
    """Resolve Hartmann grid size for recovery outputs."""
    if gt_disp is not None and 'grid_rows' in gt_disp and 'grid_cols' in gt_disp:
        return int(gt_disp['grid_rows']), int(gt_disp['grid_cols'])
    return estimate_grid_size(image_shape, pitch, pixel_size, window_size_m=window_size)


def _print_recovery_context(
    *,
    params: dict,
    method_name: str,
    rows: int,
    cols: int,
    movement: dict,
    global_offsets_px: np.ndarray | None,
) -> None:
    """Print the recovery configuration and movement summary."""
    print(f"\n  Grid: {rows} × {cols} sub-apertures")
    print(f"  Method: {method_name}")
    if params.get('mode') == 'full' and _movement_mode(movement) == 'trajectory' and not _movement_is_uniform(movement):
        print("  Movement: non-uniform trajectory metadata detected")
    if global_offsets_px is not None and global_offsets_px.shape[0] >= 2:
        step_vectors_px = np.diff(global_offsets_px, axis=0)
        step_lengths_px = np.linalg.norm(step_vectors_px, axis=1)
        if _movement_is_uniform(movement):
            avg_step = np.mean(step_vectors_px, axis=0)
            print(f"  Global trajectory step: ({avg_step[0]:.3f}, {avg_step[1]:.3f}) px")
        else:
            print(
                "  Global trajectory step lengths: "
                f"{np.min(step_lengths_px):.3f} .. {np.max(step_lengths_px):.3f} px"
            )


def _detect_saved_per_frame_pairs(sim_dir: str) -> bool:
    """Return True when saved aberrated/ideal per-frame pairs are available."""
    frames_dir = os.path.join(sim_dir, "frames")
    return (
        os.path.isdir(frames_dir) and
        os.path.exists(os.path.join(frames_dir, "frame_000.npy")) and
        os.path.exists(os.path.join(frames_dir, "frame_000_ideal.npy"))
    )


def _resolve_primary_recovery_mode() -> str:
    """Validate and return the configured primary recovery mode."""
    primary_recovery_mode = str(getattr(config, 'PRIMARY_RECOVERY_MODE', 'accumulated') or 'accumulated').lower()
    if primary_recovery_mode not in {'accumulated', 'per_frame_eval', 'auto'}:
        raise ValueError(
            f"Unsupported PRIMARY_RECOVERY_MODE={primary_recovery_mode!r}; "
            "expected 'accumulated', 'per_frame_eval', or 'auto'."
        )
    return primary_recovery_mode


def _run_per_frame_primary_recovery(
    *,
    data: dict,
    rows: int,
    cols: int,
    method_name: str,
    aberr_mode: str,
    movement: dict,
) -> dict:
    """Run the saved per-frame oracle recovery path."""
    print(f"  Mode: per-frame evaluation recovery ({data['params'].get('n_pulses', 1)} frames)")
    per_frame_result = _recover_per_frame_sequence(
        data=data,
        rows=rows,
        cols=cols,
        method_name=method_name,
        movement=movement,
    )
    all_shifts_arr = per_frame_result["all_shifts_arr"]
    n_recovered = int(per_frame_result["n_recovered"])
    print(f"  Recovered {n_recovered} frames")
    shifts_px, _ = _select_representative_shifts(all_shifts_arr, aberr_mode)
    return {
        "shifts_px": shifts_px,
        "all_shifts_arr": all_shifts_arr,
        "n_recovered": n_recovered,
        "recovery_mode": "per_frame_eval",
        "decomposition_method": None,
        "reference_sequence_source": None,
        "decomposition_confidences": None,
        "decomposition_images": None,
    }


def _run_accumulated_decomposition_primary_recovery(
    *,
    data: dict,
    movement: dict,
    rows: int,
    cols: int,
    method_name: str,
    aberr_mode: str,
) -> dict | None:
    """Run trajectory-aware accumulated decomposition recovery when available."""
    try:
        decomp_result = _recover_accumulated_with_decomposition(
            data=data,
            movement=movement,
            rows=rows,
            cols=cols,
            method_name=method_name,
        )
    except Exception as exc:
        print(f"  Warning: accumulated decomposition failed: {exc}")
        return None

    if decomp_result is None:
        return None

    all_shifts_arr = decomp_result["all_shifts_arr"]
    shifts_px, _ = _select_representative_shifts(all_shifts_arr, aberr_mode)
    return {
        "shifts_px": shifts_px,
        "all_shifts_arr": all_shifts_arr,
        "n_recovered": int(decomp_result["n_recovered"]),
        "recovery_mode": "accumulated_decomposition",
        "decomposition_method": decomp_result["decomposition_method"],
        "reference_sequence_source": decomp_result.get("reference_sequence_source"),
        "decomposition_confidences": decomp_result["decomposition_confidences"],
        "decomposition_images": decomp_result["decomposition_images"],
    }


def _run_single_accumulated_primary_recovery(
    *,
    data: dict,
    rows: int,
    cols: int,
    method_name: str,
    params: dict,
    movement: dict,
    ideal_image: np.ndarray | None,
) -> dict:
    """Run the legacy single-image fallback for uniform or static recovery."""
    print(f"  Mode: accumulated image recovery")
    if params.get('mode') == 'full' and _movement_mode(movement) == 'trajectory' and not _movement_is_uniform(movement):
        raise RuntimeError(
            "Non-uniform trajectory recovery requires trajectory-aware decomposition "
            "or optimization priors. Legacy centroid/xcorr accumulated fallback is "
            "not valid for non-uniform Hartmann movement."
        )
    if ideal_image is not None:
        ref_image = ideal_image
        print(f"  Reference: ideal (no aberration)")
    else:
        ref_image = data['image']
        print(f"  Reference: self (no ideal image available)")

    guided_centers_px = None
    if int(params.get('n_pulses', 1)) == 1:
        guided_centers_px = _resolve_guided_centers_for_pulse(data, movement, 0)
    shifts_px = run_displacement_method(
        ref_image,
        data['image'],
        rows,
        cols,
        method_name,
        guided_centers_px=guided_centers_px,
    )
    return {
        "shifts_px": shifts_px,
        "all_shifts_arr": None,
        "n_recovered": 1,
        "recovery_mode": "accumulated_single",
        "decomposition_method": None,
        "reference_sequence_source": None,
        "decomposition_confidences": None,
        "decomposition_images": None,
    }


def _run_primary_recovery(
    *,
    data: dict,
    rows: int,
    cols: int,
    method_name: str,
    params: dict,
    movement: dict,
    ideal_image: np.ndarray | None,
    has_per_frame: bool,
    primary_recovery_mode: str,
) -> dict:
    """Run the primary recovery path and return a normalized result payload."""
    aberr_mode = params.get('phase_aberration_mode', 'none')

    if has_per_frame and primary_recovery_mode != 'per_frame_eval':
        print("  Per-frame aberrated frames detected: reserved for evaluation only, not used for primary recovery")

    if primary_recovery_mode == 'per_frame_eval':
        return _run_per_frame_primary_recovery(
            data=data,
            rows=rows,
            cols=cols,
            method_name=method_name,
            aberr_mode=aberr_mode,
            movement=movement,
        )

    decomp_result = _run_accumulated_decomposition_primary_recovery(
        data=data,
        movement=movement,
        rows=rows,
        cols=cols,
        method_name=method_name,
        aberr_mode=aberr_mode,
    )
    if decomp_result is not None:
        return decomp_result

    return _run_single_accumulated_primary_recovery(
        data=data,
        rows=rows,
        cols=cols,
        method_name=method_name,
        params=params,
        movement=movement,
        ideal_image=ideal_image,
    )


def _build_evaluation_baseline(
    *,
    data: dict,
    rows: int,
    cols: int,
    method_name: str,
    has_per_frame: bool,
    primary_recovery_mode: str,
    movement: dict,
    pixel_size: float,
    z_prop: float,
    pitch: float,
    wavelength: float,
) -> dict | None:
    """Build an oracle per-frame baseline for evaluation-only comparisons."""
    if (
        not has_per_frame or
        primary_recovery_mode == 'per_frame_eval' or
        not getattr(config, 'SAVE_PER_FRAME_EVAL_BASELINE', True)
    ):
        return None

    try:
        per_frame_eval = _recover_per_frame_sequence(
            data=data,
            rows=rows,
            cols=cols,
            method_name=method_name,
            movement=movement,
        )
        eval_all_shifts_arr = per_frame_eval["all_shifts_arr"]
        eval_all_phase_rad, eval_all_dx_m, eval_all_dy_m = _recover_sequence_phase_products(
            eval_all_shifts_arr,
            pixel_size=pixel_size,
            z_prop=z_prop,
            pitch=pitch,
            wavelength=wavelength,
        )
        print("  Evaluation baseline: built per-frame oracle sequence from saved aberrated frames")
        return {
            "mode": "per_frame_oracle",
            "all_shifts_px": eval_all_shifts_arr,
            "all_phase_rad": eval_all_phase_rad,
            "all_displacement_x_m": eval_all_dx_m,
            "all_displacement_y_m": eval_all_dy_m,
            "n_frames_recovered": int(per_frame_eval["n_recovered"]),
        }
    except Exception as exc:
        print(f"  Warning: per-frame evaluation baseline unavailable: {exc}")
        return None


def _recover_primary_phase_products(
    *,
    shifts_px: np.ndarray,
    all_shifts_arr: np.ndarray | None,
    pixel_size: float,
    z_prop: float,
    pitch: float,
    wavelength: float,
) -> dict:
    """Recover aggregate and per-frame phase/displacement products from shift fields."""
    recovery = recover_from_shifts(
        shifts_px,
        pixel_size=pixel_size,
        z_prop=z_prop,
        spacing=pitch,
        wavelength=wavelength,
        integration_method=config.INTEGRATION_METHOD,
    )

    if config.REMOVE_TIP_TILT:
        nr, nc = recovery['phase_rad'].shape
        gx = np.arange(nc)
        gy = np.arange(nr)
        gx, gy = np.meshgrid(gx, gy)
        phase_no_tilt = remove_plane(recovery['phase_rad'], gx, gy)
    else:
        phase_no_tilt = recovery['phase_rad'].copy()

    all_phase_rad = None
    all_displacement_x_m = None
    all_displacement_y_m = None
    if all_shifts_arr is not None:
        all_phase_rad, all_displacement_x_m, all_displacement_y_m = _recover_sequence_phase_products(
            np.asarray(all_shifts_arr, dtype=np.float64),
            pixel_size=pixel_size,
            z_prop=z_prop,
            pitch=pitch,
            wavelength=wavelength,
        )
        print(f"  Per-frame phase maps: {all_phase_rad.shape}")

    return {
        "recovery": recovery,
        "phase_no_tilt_rad": phase_no_tilt,
        "all_phase_rad": all_phase_rad,
        "all_displacement_x_m": all_displacement_x_m,
        "all_displacement_y_m": all_displacement_y_m,
    }


def _build_recovery_result_payload(
    *,
    method_name: str,
    rows: int,
    cols: int,
    params: dict,
    primary_recovery_mode: str,
    has_per_frame: bool,
    primary_recovery: dict,
    phase_products: dict,
    movement: dict,
    global_offsets_px: np.ndarray | None,
    evaluation_baseline: dict | None,
) -> dict:
    """Assemble the canonical Step 2 recovery result payload."""
    recovery = phase_products["recovery"]
    return {
        "method": method_name,
        "rows": rows,
        "cols": cols,
        "shifts_px": primary_recovery["shifts_px"],
        "aberr_mode": params.get('phase_aberration_mode', 'none'),
        "primary_recovery_mode": primary_recovery_mode,
        "has_per_frame_aberrated_data": has_per_frame,
        "used_per_frame_aberrated_for_recovery": primary_recovery["recovery_mode"] == "per_frame_eval",
        "all_shifts_px": primary_recovery["all_shifts_arr"],
        "all_phase_rad": phase_products["all_phase_rad"],
        "all_displacement_x_m": phase_products["all_displacement_x_m"],
        "all_displacement_y_m": phase_products["all_displacement_y_m"],
        "n_frames_recovered": int(primary_recovery["n_recovered"]),
        "recovery_mode": primary_recovery["recovery_mode"],
        "decomposition_method": primary_recovery["decomposition_method"],
        "reference_sequence_source": primary_recovery["reference_sequence_source"],
        "decomposition_confidences": primary_recovery["decomposition_confidences"],
        "decomposition_images": primary_recovery["decomposition_images"],
        "movement": movement,
        "global_offsets_px": global_offsets_px,
        "displacement_x_m": recovery['displacement_x_m'],
        "displacement_y_m": recovery['displacement_y_m'],
        "phase_rad": recovery['phase_rad'],
        "phase_no_tilt_rad": phase_products["phase_no_tilt_rad"],
        "evaluation_baseline": evaluation_baseline,
    }


def _attach_recovery_metrics(
    *,
    result: dict,
    data: dict,
    pitch: float,
    pixel_size: float,
) -> None:
    """Attach wavefront and displacement metrics to a recovery result in-place."""
    wfe_truth = data.get('wfe_truth')
    if wfe_truth is not None:
        result['metrics'] = compute_metrics(result['phase_no_tilt_rad'], wfe_truth, pitch, pixel_size)

    gt_disp = data.get('gt_displacements')
    if gt_disp is None:
        return

    all_shifts_arr = result.get('all_shifts_px')
    if all_shifts_arr is not None:
        disp_metrics = compute_displacement_metrics_perframe(all_shifts_arr, gt_disp)
    else:
        disp_metrics = compute_displacement_metrics(result['shifts_px'], gt_disp)
    result['disp_metrics'] = disp_metrics
    result['gt_displacements'] = gt_disp

    evaluation_baseline = result.get('evaluation_baseline')
    if evaluation_baseline is not None and evaluation_baseline.get('all_shifts_px') is not None:
        evaluation_baseline['disp_metrics'] = compute_displacement_metrics_perframe(
            evaluation_baseline['all_shifts_px'],
            gt_disp,
        )


def recover_phase(data: dict, method_name: str = 'cross_correlation') -> dict:
    """
    执行完整的相位恢复流程。

    真实主路径默认只使用单张含像差 accumulated 图和无像差参考先验。
    若仿真目录中存在逐帧含像差图，则可额外生成 evaluation-only 的
    per-frame oracle baseline，但它不参与主恢复。
    """
    params = data['params']
    image = data['image']
    ideal_image = data.get('ideal_image')
    movement = _resolve_movement_metadata(data=data, params=params)
    global_offsets_px = _movement_global_offsets_px(movement)

    pixel_size = params['pixel_size_m']
    z_prop = params['z_prop_m']
    pitch = params['pitch_m']
    wavelength = params['wavelength_m']

    window_size = params.get('window_size_m')  # full 模式有此字段

    gt_disp = data.get('gt_displacements')
    rows, cols = _resolve_recovery_grid(
        image_shape=image.shape,
        gt_disp=gt_disp,
        pitch=pitch,
        pixel_size=pixel_size,
        window_size=window_size,
    )
    _print_recovery_context(
        params=params,
        method_name=method_name,
        rows=rows,
        cols=cols,
        movement=movement,
        global_offsets_px=global_offsets_px,
    )

    has_per_frame = _detect_saved_per_frame_pairs(data['sim_dir'])
    primary_recovery_mode = _resolve_primary_recovery_mode()
    primary_recovery = _run_primary_recovery(
        data=data,
        rows=rows,
        cols=cols,
        method_name=method_name,
        params=params,
        movement=movement,
        ideal_image=ideal_image,
        has_per_frame=has_per_frame,
        primary_recovery_mode=primary_recovery_mode,
    )
    shifts_px = primary_recovery["shifts_px"]
    all_shifts_arr = primary_recovery["all_shifts_arr"]

    evaluation_baseline = _build_evaluation_baseline(
        data=data,
        rows=rows,
        cols=cols,
        method_name=method_name,
        has_per_frame=has_per_frame,
        primary_recovery_mode=primary_recovery_mode,
        movement=movement,
        pixel_size=pixel_size,
        z_prop=z_prop,
        pitch=pitch,
        wavelength=wavelength,
    )
    phase_products = _recover_primary_phase_products(
        shifts_px=shifts_px,
        all_shifts_arr=all_shifts_arr,
        pixel_size=pixel_size,
        z_prop=z_prop,
        pitch=pitch,
        wavelength=wavelength,
    )
    result = _build_recovery_result_payload(
        method_name=method_name,
        rows=rows,
        cols=cols,
        params=params,
        primary_recovery_mode=primary_recovery_mode,
        has_per_frame=has_per_frame,
        primary_recovery=primary_recovery,
        phase_products=phase_products,
        movement=movement,
        global_offsets_px=global_offsets_px,
        evaluation_baseline=evaluation_baseline,
    )
    _attach_recovery_metrics(
        result=result,
        data=data,
        pitch=pitch,
        pixel_size=pixel_size,
    )

    return result


def compute_metrics(phase_recovered, wfe_truth, pitch_m, pixel_size_m):
    """计算恢复相位与 ground truth 之间的误差指标。"""
    # Downsample truth to match recovered grid
    pitch_px = pitch_m / pixel_size_m
    nr, nc = phase_recovered.shape
    
    # Sample truth at sub-aperture centers
    cy = np.linspace(pitch_px / 2, wfe_truth.shape[0] - pitch_px / 2, nr).astype(int)
    cx = np.linspace(pitch_px / 2, wfe_truth.shape[1] - pitch_px / 2, nc).astype(int)
    cy = np.clip(cy, 0, wfe_truth.shape[0] - 1)
    cx = np.clip(cx, 0, wfe_truth.shape[1] - 1)
    ix, iy = np.meshgrid(cx, cy)
    truth_sampled = wfe_truth[iy, ix]

    # Remove tip/tilt from both
    gx = np.arange(nc)
    gy = np.arange(nr)
    gx, gy = np.meshgrid(gx, gy)
    truth_no_tilt = remove_plane(truth_sampled, gx, gy)
    recovered_no_tilt = remove_plane(phase_recovered, gx, gy)

    # Scale: normalize both to same range for correlation
    diff = recovered_no_tilt - truth_no_tilt
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    pv = float(np.ptp(diff))
    
    # Correlation
    t_flat = truth_no_tilt.ravel()
    r_flat = recovered_no_tilt.ravel()
    if np.std(t_flat) > 1e-12 and np.std(r_flat) > 1e-12:
        corr = float(np.corrcoef(t_flat, r_flat)[0, 1])
    else:
        corr = 0.0

    metrics = {
        "rmse_rad": rmse,
        "pv_rad": pv,
        "correlation": corr,
    }
    print(f"\n  Metrics vs ground truth:")
    print(f"    RMSE: {rmse:.4f} rad")
    print(f"    PV:   {pv:.4f} rad")
    print(f"    Corr: {corr:.4f}")

    return metrics


def compute_displacement_metrics(shifts_px, gt_disp):
    """
    比较恢复的位移与 ground truth 位移。
    
    shifts_px: (rows, cols, 2) — 恢复的位移 [dx, dy]
    gt_disp: dict from gt_displacements.npz
    """
    # GT 位移: 对所有帧取平均 (因为累积图是所有帧平均的结果)
    truth_displacement_x_px = np.mean(gt_disp['displacement_x_px'], axis=0)  # (rows, cols)
    truth_displacement_y_px = np.mean(gt_disp['displacement_y_px'], axis=0)  # (rows, cols)
    gt_rows, gt_cols = truth_displacement_x_px.shape
    
    # 恢复的位移
    rec_rows, rec_cols, _ = shifts_px.shape
    
    # 网格大小可能不完全一致，取交集
    min_rows = min(rec_rows, gt_rows)
    min_cols = min(rec_cols, gt_cols)
    
    recovered_displacement_x_px = shifts_px[:min_rows, :min_cols, 0]
    recovered_displacement_y_px = shifts_px[:min_rows, :min_cols, 1]
    truth_displacement_x_crop_px = truth_displacement_x_px[:min_rows, :min_cols]
    truth_displacement_y_crop_px = truth_displacement_y_px[:min_rows, :min_cols]
    
    frame_stats = _compute_displacement_frame_stats(
        recovered_displacement_x_px=recovered_displacement_x_px,
        recovered_displacement_y_px=recovered_displacement_y_px,
        truth_displacement_x_px=truth_displacement_x_crop_px,
        truth_displacement_y_px=truth_displacement_y_crop_px,
    )
    displacement_error_total_px = frame_stats["displacement_error_total_px"]
    rmse_dx = frame_stats["rmse_dx"]
    rmse_dy = frame_stats["rmse_dy"]
    rmse_total = frame_stats["rmse_total"]

    corr_dx = _displacement_component_correlation(
        frame_stats["recovered_displacement_x_px"],
        frame_stats["truth_displacement_x_px"],
    )
    corr_dy = _displacement_component_correlation(
        frame_stats["recovered_displacement_y_px"],
        frame_stats["truth_displacement_y_px"],
    )
    
    metrics = {
        "rmse_dx_px": rmse_dx,
        "rmse_dy_px": rmse_dy,
        "rmse_total_px": rmse_total,
        "mean_err_px": float(np.mean(displacement_error_total_px)),
        "max_err_px": float(np.max(displacement_error_total_px)),
        "corr_dx": corr_dx,
        "corr_dy": corr_dy,
        "grid_used": f"{min_rows}x{min_cols}",
    }
    
    print(f"\n  Displacement metrics vs GT:")
    print(f"    Grid: {min_rows}×{min_cols} (rec={rec_rows}×{rec_cols}, gt={gt_rows}×{gt_cols})")
    print(f"    RMSE dX: {rmse_dx:.4f} px")
    print(f"    RMSE dY: {rmse_dy:.4f} px")
    print(f"    RMSE total: {rmse_total:.4f} px")
    print(f"    Mean error: {np.mean(displacement_error_total_px):.4f} px")
    print(f"    Max error:  {np.max(displacement_error_total_px):.4f} px")
    print(f"    Corr dX: {corr_dx:.4f}, Corr dY: {corr_dy:.4f}")
    
    return metrics


def _compute_displacement_frame_stats(
    *,
    recovered_displacement_x_px: np.ndarray,
    recovered_displacement_y_px: np.ndarray,
    truth_displacement_x_px: np.ndarray,
    truth_displacement_y_px: np.ndarray,
) -> dict:
    """Compute per-frame displacement errors and RMSE on already aligned crops."""
    displacement_error_x_px = recovered_displacement_x_px - truth_displacement_x_px
    displacement_error_y_px = recovered_displacement_y_px - truth_displacement_y_px
    displacement_error_total_px = np.sqrt(displacement_error_x_px**2 + displacement_error_y_px**2)
    return {
        "recovered_displacement_x_px": recovered_displacement_x_px,
        "recovered_displacement_y_px": recovered_displacement_y_px,
        "truth_displacement_x_px": truth_displacement_x_px,
        "truth_displacement_y_px": truth_displacement_y_px,
        "displacement_error_x_px": displacement_error_x_px,
        "displacement_error_y_px": displacement_error_y_px,
        "displacement_error_total_px": displacement_error_total_px,
        "rmse_dx": float(np.sqrt(np.mean(displacement_error_x_px**2))),
        "rmse_dy": float(np.sqrt(np.mean(displacement_error_y_px**2))),
        "rmse_total": float(np.sqrt(np.mean(displacement_error_total_px**2))),
    }


def _displacement_component_correlation(
    recovered_displacement_px: np.ndarray,
    truth_displacement_px: np.ndarray,
) -> float:
    """Return correlation for one displacement component, guarding degenerate truth fields."""
    if np.std(truth_displacement_px) <= 1e-12:
        return 0.0
    return float(np.corrcoef(recovered_displacement_px.ravel(), truth_displacement_px.ravel())[0, 1])


def _build_per_frame_displacement_metric(
    *,
    frame_stats: dict,
    corr_dx: float,
    corr_dy: float,
) -> dict:
    """Package one frame's displacement RMSE/correlation summary."""
    return {
        'rmse_dx_px': frame_stats["rmse_dx"],
        'rmse_dy_px': frame_stats["rmse_dy"],
        'rmse_total_px': frame_stats["rmse_total"],
        'corr_dx': corr_dx,
        'corr_dy': corr_dy,
    }


def compute_displacement_metrics_perframe(all_shifts_px, gt_disp):
    """
    Per-frame 位移对比。
    
    all_shifts_px: (n_frames, rows, cols, 2) — 每帧恢复的位移
    gt_disp: dict from gt_displacements.npz, displacement_x_px shape (n_gt, rows, cols)
    """
    truth_displacement_x_px = gt_disp['displacement_x_px']  # (n_gt, rows, cols)
    truth_displacement_y_px = gt_disp['displacement_y_px']
    
    n_rec = all_shifts_px.shape[0]
    n_gt = truth_displacement_x_px.shape[0]
    n_compare = min(n_rec, n_gt)
    
    rec_rows, rec_cols = all_shifts_px.shape[1], all_shifts_px.shape[2]
    gt_rows, gt_cols = truth_displacement_x_px.shape[1], truth_displacement_x_px.shape[2]
    min_r = min(rec_rows, gt_rows)
    min_c = min(rec_cols, gt_cols)

    frame_stats_list, _, _, per_frame_rmse = _collect_per_frame_displacement_rmse_series(
        all_shifts=all_shifts_px,
        truth_displacement_x_all_px=truth_displacement_x_px,
        truth_displacement_y_all_px=truth_displacement_y_px,
        n_compare=n_compare,
        min_rows=min_r,
        min_cols=min_c,
    )

    all_displacement_error_x_px = np.array(
        [frame_stats["displacement_error_x_px"] for frame_stats in frame_stats_list]
    )
    all_displacement_error_y_px = np.array(
        [frame_stats["displacement_error_y_px"] for frame_stats in frame_stats_list]
    )
    
    all_displacement_error_total_px = np.sqrt(
        all_displacement_error_x_px**2 + all_displacement_error_y_px**2
    )

    rmse_dx = float(np.sqrt(np.mean(all_displacement_error_x_px**2)))
    rmse_dy = float(np.sqrt(np.mean(all_displacement_error_y_px**2)))
    rmse_total = float(np.sqrt(np.mean(all_displacement_error_total_px**2)))
    
    # Per-frame correlation and per-frame metric packaging
    per_frame_metrics = []
    corr_dx_list = []
    corr_dy_list = []
    for frame_stats in frame_stats_list:
        c_dx = _displacement_component_correlation(
            frame_stats["recovered_displacement_x_px"],
            frame_stats["truth_displacement_x_px"],
        )
        c_dy = _displacement_component_correlation(
            frame_stats["recovered_displacement_y_px"],
            frame_stats["truth_displacement_y_px"],
        )

        if np.std(frame_stats["truth_displacement_x_px"]) > 1e-12:
            corr_dx_list.append(c_dx)
        if np.std(frame_stats["truth_displacement_y_px"]) > 1e-12:
            corr_dy_list.append(c_dy)
        per_frame_metrics.append(
            _build_per_frame_displacement_metric(
                frame_stats=frame_stats,
                corr_dx=c_dx,
                corr_dy=c_dy,
            )
        )
    
    mean_corr_dx = float(np.mean(corr_dx_list)) if corr_dx_list else 0.0
    mean_corr_dy = float(np.mean(corr_dy_list)) if corr_dy_list else 0.0
    
    metrics = {
        "mode": "per_frame",
        "n_frames": n_compare,
        "rmse_dx_px": rmse_dx,
        "rmse_dy_px": rmse_dy,
        "rmse_total_px": rmse_total,
        "mean_err_px": float(np.mean(all_displacement_error_total_px)),
        "max_err_px": float(np.max(all_displacement_error_total_px)),
        "per_frame_rmse": per_frame_rmse,
        "mean_corr_dx": mean_corr_dx,
        "mean_corr_dy": mean_corr_dy,
        "per_frame_metrics": per_frame_metrics,
        "grid_used": f"{min_r}x{min_c}",
    }
    
    print(f"\n  Per-frame displacement metrics ({n_compare} frames):")
    print(f"    Grid: {min_r}×{min_c}")
    print(f"    RMSE dX: {rmse_dx:.4f} px")
    print(f"    RMSE dY: {rmse_dy:.4f} px")
    print(f"    RMSE total: {rmse_total:.4f} px")
    print(f"    Mean error: {np.mean(all_displacement_error_total_px):.4f} px")
    print(f"    Max error:  {np.max(all_displacement_error_total_px):.4f} px")
    print(f"    Mean Corr dX: {mean_corr_dx:.4f}, dY: {mean_corr_dy:.4f}")
    print(f"    Per-frame RMSE: {[f'{r:.4f}' for r in per_frame_rmse]}")
    
    return metrics


# =============================================================================
# Visualization
# =============================================================================

def _build_full_mode_grid_geometry(params: dict, image_shape, gt_disp=None) -> GridGeometry:
    """Build full-mode ideal centers and explicit detector-grid geometry."""
    h, w = image_shape
    pixel_size = float(params['pixel_size_m'])
    pitch = float(params['pitch_m'])

    centers_x_m = None
    centers_y_m = None
    if gt_disp is not None:
        centers_x_m = gt_disp.get('centers_x_m')
        centers_y_m = gt_disp.get('centers_y_m')

    if centers_x_m is not None and centers_y_m is not None:
        x_coords_px = np.asarray(centers_x_m, dtype=float) / pixel_size + w / 2.0
        y_coords_px = np.asarray(centers_y_m, dtype=float) / pixel_size + h / 2.0
    else:
        window_size = params.get('window_size_m')
        if window_size is not None:
            spacing = pitch
            grid_count = max(1, int(round(float(window_size) / spacing)))
            coords = (np.arange(grid_count) - (grid_count - 1) / 2.0) * spacing
            det_w = w * pixel_size
            det_h = h * pixel_size
            x_coords_m = coords[np.abs(coords) < det_w / 2.0]
            y_coords_m = coords[np.abs(coords) < det_h / 2.0]
            x_coords_px = x_coords_m / pixel_size + w / 2.0
            y_coords_px = y_coords_m / pixel_size + h / 2.0
        else:
            rows, cols = estimate_grid_size(image_shape, pitch, pixel_size, window_size_m=None)
            spacing_px = pitch / pixel_size
            x_coords_px = (np.arange(cols) - (cols - 1) / 2.0) * spacing_px + w / 2.0
            y_coords_px = (np.arange(rows) - (rows - 1) / 2.0) * spacing_px + h / 2.0

    grid_geometry = build_grid_geometry_from_axes(x_coords_px, y_coords_px)
    if grid_geometry.centers_xy_px.size == 0:
        raise ValueError("Failed to build full-mode ideal centers")
    return grid_geometry


def _build_full_mode_ideal_centers(params: dict, image_shape, gt_disp=None) -> np.ndarray:
    """Backward-compatible wrapper returning only `(x, y)` ideal centers."""
    grid_geometry = _build_full_mode_grid_geometry(
        params,
        image_shape,
        gt_disp=gt_disp,
    )
    return grid_geometry.centers_xy_px


def _save_convergence_plot(history, output_path: str, *, title: str):
    """Save optimization convergence history when available."""
    if not history:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(np.arange(len(history)), history, color='tab:blue', lw=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def _save_as_png(npy_path, data, *, allow_preview: bool = True):
    """将数组保存为 PNG 预览图。

    `allow_preview=False` 用于显式跳过非图像数据。
    """
    if not allow_preview:
        return
    png_path = npy_path.replace('.npy', '.png').replace('.npz', '.png')
    try:
        arr = np.squeeze(np.asarray(data, dtype=float))
        if arr.ndim == 2:
            plt.imsave(png_path, arr, cmap='magma')
        elif arr.ndim == 3:
            plt.imsave(png_path, arr[0], cmap='magma')
        elif arr.ndim == 4:
            plt.imsave(png_path, arr[0, ..., 0], cmap='magma')
    except Exception as e:
        print(f"    Warning: PNG 保存失败 {png_path}: {e}")


def _get_per_frame_displacement_stacks(result: dict):
    """Return per-frame displacement stacks with backward-compatible aliases."""
    return _resolve_per_frame_displacement_stacks(result)


def _save_named_array(output_dir: str, filename: str, array, *, allow_preview: bool = True) -> str:
    """Save a numpy array and optionally its PNG preview."""
    path = os.path.join(output_dir, filename)
    np.save(path, array)
    _save_as_png(path, array, allow_preview=allow_preview)
    return path


def _per_frame_displacement_filenames(method: str) -> dict[str, str]:
    """Return canonical and legacy filenames for per-frame displacement stacks."""
    return {
        "canonical_x": f"all_displacement_x_{method}.npy",
        "canonical_y": f"all_displacement_y_{method}.npy",
        "legacy_x": f"all_disp_x_{method}.npy",
        "legacy_y": f"all_disp_y_{method}.npy",
    }


def _save_per_frame_displacement_arrays(
    output_dir: str,
    method: str,
    displacement_x_stack_m: np.ndarray,
    displacement_y_stack_m: np.ndarray,
) -> None:
    """Persist per-frame displacement stacks under canonical and legacy filenames."""
    filenames = _per_frame_displacement_filenames(method)
    _save_named_array(output_dir, filenames["canonical_x"], displacement_x_stack_m)
    _save_named_array(output_dir, filenames["canonical_y"], displacement_y_stack_m)
    _save_legacy_per_frame_displacement_aliases(
        output_dir,
        method,
        displacement_x_stack_m,
        displacement_y_stack_m,
    )


def _save_legacy_per_frame_displacement_aliases(
    output_dir: str,
    method: str,
    displacement_x_stack_m: np.ndarray,
    displacement_y_stack_m: np.ndarray,
) -> None:
    """Persist only the remaining legacy per-frame displacement filename aliases."""
    filenames = _per_frame_displacement_filenames(method)
    _save_named_array(output_dir, filenames["legacy_x"], displacement_x_stack_m)
    _save_named_array(output_dir, filenames["legacy_y"], displacement_y_stack_m)


def _save_recovery_arrays(result: dict, output_dir: str) -> None:
    """Persist recovered arrays and sequence products to disk."""
    method = result['method']
    _save_named_array(output_dir, f"displacement_x_{method}.npy", result['displacement_x_m'])
    _save_named_array(output_dir, f"displacement_y_{method}.npy", result['displacement_y_m'])
    _save_named_array(output_dir, f"phase_recovered_{method}.npy", result['phase_no_tilt_rad'])

    if result.get('all_shifts_px') is not None:
        _save_named_array(output_dir, f"all_shifts_{method}.npy", result['all_shifts_px'])
        print(f"  Per-frame shifts: {result['all_shifts_px'].shape}")
    if result.get('all_phase_rad') is not None:
        _save_named_array(output_dir, f"all_phase_{method}.npy", result['all_phase_rad'])
        print(f"  Per-frame phase: {result['all_phase_rad'].shape}")

    all_displacement_x_m, all_displacement_y_m = _get_per_frame_displacement_stacks(result)
    if all_displacement_x_m is not None:
        _save_per_frame_displacement_arrays(
            output_dir,
            method,
            all_displacement_x_m,
            all_displacement_y_m,
        )
        print(f"  Per-frame displacements: {all_displacement_x_m.shape}")

    if result.get('decomposition_images') is not None:
        _save_named_array(output_dir, f"decomposition_images_{method}.npy", result['decomposition_images'])
        print(f"  Decomposition images: {result['decomposition_images'].shape}")

    if result.get('decomposition_confidences') is not None:
        with open(os.path.join(output_dir, f"decomposition_{method}.json"), "w") as fp:
            json.dump(
                {
                    "recovery_mode": result.get("recovery_mode"),
                    "decomposition_method": result.get("decomposition_method"),
                    "reference_sequence_source": result.get("reference_sequence_source"),
                    "decomposition_confidences": [float(x) for x in result['decomposition_confidences']],
                },
                fp,
                indent=2,
            )


def _save_evaluation_baseline_artifacts(output_dir: str, method: str, eval_baseline: dict | None) -> None:
    """Persist evaluation-only oracle baseline outputs when available."""
    if eval_baseline is None:
        return

    if eval_baseline.get('all_shifts_px') is not None:
        _save_named_array(output_dir, f"eval_baseline_all_shifts_{method}.npy", eval_baseline['all_shifts_px'])
    if eval_baseline.get('all_phase_rad') is not None:
        _save_named_array(output_dir, f"eval_baseline_all_phase_{method}.npy", eval_baseline['all_phase_rad'])
    with open(os.path.join(output_dir, f"eval_baseline_{method}.json"), "w") as fp:
        json.dump(
            {
                "mode": eval_baseline.get("mode"),
                "n_frames_recovered": int(eval_baseline.get("n_frames_recovered", 0)),
                "disp_metrics": eval_baseline.get("disp_metrics"),
            },
            fp,
            indent=2,
        )


def _build_metrics_payload(result: dict, eval_baseline: dict | None) -> dict | None:
    """Build the metrics JSON payload for a recovery result."""
    if 'metrics' not in result:
        return None

    metrics_payload = dict(result['metrics'])
    metrics_payload.update(
        {
            "recovery_mode": result.get('recovery_mode'),
            "primary_recovery_mode": result.get('primary_recovery_mode'),
            "used_per_frame_aberrated_for_recovery": bool(result.get('used_per_frame_aberrated_for_recovery')),
            "reference_sequence_source": result.get('reference_sequence_source'),
            "decomposition_method": result.get('decomposition_method'),
        }
    )
    if result.get('disp_metrics') is not None:
        metrics_payload["displacement_metrics"] = result['disp_metrics']
    if eval_baseline is not None:
        metrics_payload["evaluation_baseline"] = {
            "mode": eval_baseline.get("mode"),
            "n_frames_recovered": int(eval_baseline.get("n_frames_recovered", 0)),
            "disp_metrics": eval_baseline.get("disp_metrics"),
        }
    return metrics_payload


def _save_metrics_payload(output_dir: str, method: str, metrics_payload: dict | None) -> None:
    """Write metrics JSON when metrics are available."""
    if metrics_payload is None:
        return
    with open(os.path.join(output_dir, f"metrics_{method}.json"), "w") as fp:
        json.dump(metrics_payload, fp, indent=2)


def _save_recovery_summary_figure(result: dict, output_dir: str, data: dict) -> bool:
    """Save the main recovered-phase summary figure. Returns whether GT displacements exist."""
    method = result['method']
    has_gt_disp = 'gt_displacements' in result and result['gt_displacements'] is not None
    n_cols = 4 if has_gt_disp else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))

    ax = axes[0]
    im0 = ax.imshow(result['phase_no_tilt_rad'], cmap='RdBu_r', origin='lower')
    ax.set_title(f"Recovered Phase ({method})")
    plt.colorbar(im0, ax=ax, label='rad')

    ax = axes[1]
    wfe_truth = data.get('wfe_truth')
    if wfe_truth is not None:
        nr, nc = result['phase_no_tilt_rad'].shape
        pitch_px = data['params']['pitch_m'] / data['params']['pixel_size_m']
        cy = np.linspace(pitch_px / 2, wfe_truth.shape[0] - pitch_px / 2, nr).astype(int)
        cx = np.linspace(pitch_px / 2, wfe_truth.shape[1] - pitch_px / 2, nc).astype(int)
        cy = np.clip(cy, 0, wfe_truth.shape[0] - 1)
        cx = np.clip(cx, 0, wfe_truth.shape[1] - 1)
        ix, iy = np.meshgrid(cx, cy)
        truth_sampled = wfe_truth[iy, ix]
        gx = np.arange(nc)
        gy = np.arange(nr)
        gx, gy = np.meshgrid(gx, gy)
        truth_no_tilt = remove_plane(truth_sampled, gx, gy)
        im1 = ax.imshow(truth_no_tilt, cmap='RdBu_r', origin='lower')
        ax.set_title('Ground Truth Phase')
        plt.colorbar(im1, ax=ax, label='rad')
    else:
        ax.text(0.5, 0.5, 'No WFE ground truth', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title('Ground Truth Phase')

    ax = axes[2]
    im2 = ax.imshow(data['image'], cmap='magma', origin='lower')
    ax.set_title(f"Detector Image ({data['params']['n_pulses']} pulses)")
    plt.colorbar(im2, ax=ax, label='Counts')

    if has_gt_disp:
        ax = axes[3]
        gt_disp = result['gt_displacements']
        aberr_mode_vis = data['params'].get('phase_aberration_mode', 'none')
        if aberr_mode_vis == 'per_pulse':
            gt_dx_repr = gt_disp['displacement_x_px'][0]
            gt_dy_repr = gt_disp['displacement_y_px'][0]
        else:
            gt_dx_repr = np.mean(gt_disp['displacement_x_px'], axis=0)
            gt_dy_repr = np.mean(gt_disp['displacement_y_px'], axis=0)
        gt_total = np.sqrt(gt_dx_repr**2 + gt_dy_repr**2)

        shifts = result['shifts_px']
        rec_total = np.sqrt(shifts[:, :, 0]**2 + shifts[:, :, 1]**2)
        min_r = min(rec_total.shape[0], gt_total.shape[0])
        min_c = min(rec_total.shape[1], gt_total.shape[1])

        err = np.abs(rec_total[:min_r, :min_c] - gt_total[:min_r, :min_c])
        im3 = ax.imshow(err, cmap='hot', origin='lower')
        dm = result.get('disp_metrics', {})
        ax.set_title(f"Displacement Error\nRMSE={dm.get('rmse_total_px', 0):.4f} px")
        plt.colorbar(im3, ax=ax, label='Error (px)')

    plt.suptitle(f"Phase Recovery — {method}")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"recovery_{method}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n  Figure saved: {fig_path}")
    return has_gt_disp


def _representative_displacement_fields(result: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """Return representative GT displacement fields for summary visualization."""
    gt_disp = result['gt_displacements']
    aberr_mode_vis = result.get('aberr_mode', 'none')
    if aberr_mode_vis == 'per_pulse':
        return gt_disp['displacement_x_px'][0], gt_disp['displacement_y_px'][0], ' (frame 0 only, NOT averaged)'
    return (
        np.mean(gt_disp['displacement_x_px'], axis=0),
        np.mean(gt_disp['displacement_y_px'], axis=0),
        ' (mean)',
    )


def _crop_displacement_field_pair(
    *,
    recovered_displacement_x_px: np.ndarray,
    recovered_displacement_y_px: np.ndarray,
    truth_displacement_x_px: np.ndarray,
    truth_displacement_y_px: np.ndarray,
) -> dict:
    """Crop recovered and truth displacement fields to their shared grid extent."""
    min_rows = min(recovered_displacement_x_px.shape[0], truth_displacement_x_px.shape[0])
    min_cols = min(recovered_displacement_x_px.shape[1], truth_displacement_x_px.shape[1])
    recovered_displacement_x_crop_px = recovered_displacement_x_px[:min_rows, :min_cols]
    recovered_displacement_y_crop_px = recovered_displacement_y_px[:min_rows, :min_cols]
    truth_displacement_x_crop_px = truth_displacement_x_px[:min_rows, :min_cols]
    truth_displacement_y_crop_px = truth_displacement_y_px[:min_rows, :min_cols]
    frame_stats = _compute_displacement_frame_stats(
        recovered_displacement_x_px=recovered_displacement_x_crop_px,
        recovered_displacement_y_px=recovered_displacement_y_crop_px,
        truth_displacement_x_px=truth_displacement_x_crop_px,
        truth_displacement_y_px=truth_displacement_y_crop_px,
    )
    frame_stats["min_rows"] = min_rows
    frame_stats["min_cols"] = min_cols
    return frame_stats


def _displacement_display_limits(*arrays: np.ndarray) -> float:
    """Return a symmetric display range large enough for all provided displacement arrays."""
    if not arrays:
        return 1e-6
    return max(max(float(np.abs(arr).max()) for arr in arrays), 1e-6)


def _collect_per_frame_displacement_rmse_series(
    *,
    all_shifts: np.ndarray,
    truth_displacement_x_all_px: np.ndarray,
    truth_displacement_y_all_px: np.ndarray,
    n_compare: int,
    min_rows: int,
    min_cols: int,
) -> tuple[list[dict], list[float], list[float], list[float]]:
    """Compute per-frame displacement stats and RMSE series for visualization."""
    frame_stats_list: list[dict] = []
    per_frame_rmse_dx: list[float] = []
    per_frame_rmse_dy: list[float] = []
    per_frame_rmse_total: list[float] = []

    for frame_index in range(n_compare):
        frame_stats = _compute_perframe_displacement_frame_stats(
            all_shifts=all_shifts,
            truth_displacement_x_all_px=truth_displacement_x_all_px,
            truth_displacement_y_all_px=truth_displacement_y_all_px,
            frame_index=frame_index,
            min_rows=min_rows,
            min_cols=min_cols,
        )
        frame_stats_list.append(frame_stats)
        per_frame_rmse_dx.append(frame_stats["rmse_dx"])
        per_frame_rmse_dy.append(frame_stats["rmse_dy"])
        per_frame_rmse_total.append(frame_stats["rmse_total"])

    return frame_stats_list, per_frame_rmse_dx, per_frame_rmse_dy, per_frame_rmse_total


def save_results(result: dict, output_dir: str, data: dict):
    """保存恢复结果到 analysis/ 目录。"""
    os.makedirs(output_dir, exist_ok=True)
    method = result['method']
    _save_recovery_arrays(result, output_dir)

    eval_baseline = result.get('evaluation_baseline')
    _save_evaluation_baseline_artifacts(output_dir, method, eval_baseline)
    _save_metrics_payload(output_dir, method, _build_metrics_payload(result, eval_baseline))
    has_gt_disp = _save_recovery_summary_figure(result, output_dir, data)

    # Additional: detailed displacement comparison plot
    if has_gt_disp:
        _save_displacement_comparison(result, output_dir)

    # === 逐帧相位恢复可视化 ===
    if result.get('all_phase_rad') is not None:
        _save_perframe_phase(result, output_dir)

    # === 逐帧位移 vs GT 对比 ===
    if result.get('all_shifts_px') is not None and has_gt_disp:
        _save_perframe_displacement_comparison(result, output_dir)
        _save_metrics_statistics(result, output_dir)
        _save_all_frames_phase_summary(result, output_dir)
        _save_all_frames_disp_summary(result, output_dir)


def _plot_displacement_comparison_row(
    *,
    axes_row,
    truth_displacement_px: np.ndarray,
    recovered_displacement_px: np.ndarray,
    displacement_error_px: np.ndarray,
    component_label: str,
    gt_title_suffix: str,
    method: str,
    rmse_value: float,
) -> None:
    """Render one GT/recovered/error/scatter row for dX or dY comparison."""
    vmax_signal = _displacement_display_limits(truth_displacement_px, recovered_displacement_px)
    vmax_error = _displacement_display_limits(displacement_error_px)

    ax = axes_row[0]
    im = ax.imshow(truth_displacement_px, cmap='RdBu_r', vmin=-vmax_signal, vmax=vmax_signal, origin='lower')
    ax.set_title(f'GT {component_label}{gt_title_suffix}')
    plt.colorbar(im, ax=ax, label='px')

    ax = axes_row[1]
    im = ax.imshow(recovered_displacement_px, cmap='RdBu_r', vmin=-vmax_signal, vmax=vmax_signal, origin='lower')
    ax.set_title(f'Recovered {component_label} ({method})')
    plt.colorbar(im, ax=ax, label='px')

    ax = axes_row[2]
    im = ax.imshow(displacement_error_px, cmap='RdBu_r', vmin=-vmax_error, vmax=vmax_error, origin='lower')
    ax.set_title(f'Error {component_label} (RMSE={rmse_value:.4f})')
    plt.colorbar(im, ax=ax, label='px')

    ax = axes_row[3]
    ax.scatter(truth_displacement_px.ravel(), recovered_displacement_px.ravel(), alpha=0.5, s=10)
    scatter_limit = vmax_signal * 1.1
    ax.plot([-scatter_limit, scatter_limit], [-scatter_limit, scatter_limit], 'r--', lw=1)
    ax.set_xlim(-scatter_limit, scatter_limit)
    ax.set_ylim(-scatter_limit, scatter_limit)
    ax.set_xlabel(f'GT {component_label} (px)')
    ax.set_ylabel(f'Rec {component_label} (px)')
    ax.set_title(f'Scatter {component_label}')
    ax.set_aspect('equal')


def _save_displacement_comparison(result, output_dir):
    """保存详细的位移对比图: GT vs Recovered (dx/dy 分别显示)。 """
    method = result['method']
    shifts = result['shifts_px']
    truth_displacement_x_px, truth_displacement_y_px, label_suffix = _representative_displacement_fields(result)
    frame_stats = _crop_displacement_field_pair(
        recovered_displacement_x_px=shifts[:, :, 0],
        recovered_displacement_y_px=shifts[:, :, 1],
        truth_displacement_x_px=truth_displacement_x_px,
        truth_displacement_y_px=truth_displacement_y_px,
    )
    recovered_displacement_x_px = frame_stats["recovered_displacement_x_px"]
    recovered_displacement_y_px = frame_stats["recovered_displacement_y_px"]
    truth_displacement_x_px = frame_stats["truth_displacement_x_px"]
    truth_displacement_y_px = frame_stats["truth_displacement_y_px"]
    displacement_error_x_px = frame_stats["displacement_error_x_px"]
    displacement_error_y_px = frame_stats["displacement_error_y_px"]
    rmse_dx = frame_stats["rmse_dx"]
    rmse_dy = frame_stats["rmse_dy"]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    _plot_displacement_comparison_row(
        axes_row=axes[0],
        truth_displacement_px=truth_displacement_x_px,
        recovered_displacement_px=recovered_displacement_x_px,
        displacement_error_px=displacement_error_x_px,
        component_label='dX',
        gt_title_suffix=label_suffix,
        method=method,
        rmse_value=rmse_dx,
    )
    _plot_displacement_comparison_row(
        axes_row=axes[1],
        truth_displacement_px=truth_displacement_y_px,
        recovered_displacement_px=recovered_displacement_y_px,
        displacement_error_px=displacement_error_y_px,
        component_label='dY',
        gt_title_suffix=label_suffix,
        method=method,
        rmse_value=rmse_dy,
    )
    
    dm = result.get('disp_metrics', {})
    fig.suptitle(
        f"Displacement Comparison — {method}\n"
        f"RMSE total={dm.get('rmse_total_px', 0):.4f} px, "
        f"Corr dX={dm.get('corr_dx', 0):.4f}, dY={dm.get('corr_dy', 0):.4f}",
        fontsize=12
    )
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"displacement_comparison_{method}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Displacement comparison saved: {fig_path}")


def _save_perframe_phase(result, output_dir):
    """逐帧恢复相位可视化：每帧单独一张 + 总览图。"""
    method = result['method']
    all_phase = result['all_phase_rad']  # (n_frames, rows, cols)
    n_frames = all_phase.shape[0]

    phase_dir = os.path.join(output_dir, "per_frame_phase")
    os.makedirs(phase_dir, exist_ok=True)

    # 全局 colorbar 范围 (统一所有帧的颜色尺度)
    vmax = max(np.abs(all_phase).max(), 1e-6)

    # --- 逐帧保存 ---
    for i in range(n_frames):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        im = ax.imshow(all_phase[i], cmap='RdBu_r', origin='lower',
                        vmin=-vmax, vmax=vmax)
        ax.set_title(f"Recovered Phase — Frame {i} ({method})")
        plt.colorbar(im, ax=ax, label='rad')
        plt.tight_layout()
        fig_path = os.path.join(phase_dir, f"phase_frame_{i:03d}.png")
        plt.savefig(fig_path, dpi=120)
        plt.close()

    # --- 总览图 (所有帧的子图 grid) ---
    ncols_grid = min(5, n_frames)
    nrows_grid = int(np.ceil(n_frames / ncols_grid))
    fig, axes = plt.subplots(nrows_grid, ncols_grid,
                              figsize=(4 * ncols_grid, 3.5 * nrows_grid))
    if nrows_grid == 1 and ncols_grid == 1:
        axes = np.array([[axes]])
    elif nrows_grid == 1:
        axes = axes[np.newaxis, :]
    elif ncols_grid == 1:
        axes = axes[:, np.newaxis]

    for i in range(n_frames):
        r, c = divmod(i, ncols_grid)
        ax = axes[r, c]
        im = ax.imshow(all_phase[i], cmap='RdBu_r', origin='lower',
                        vmin=-vmax, vmax=vmax)
        ax.set_title(f"Frame {i}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
    # 隐藏多余子图
    for i in range(n_frames, nrows_grid * ncols_grid):
        r, c = divmod(i, ncols_grid)
        axes[r, c].set_visible(False)

    fig.suptitle(f"Per-Frame Recovered Phase — {method}", fontsize=13)
    plt.tight_layout()
    overview_path = os.path.join(phase_dir,
                                  f"phase_all_frames_{method}.png")
    plt.savefig(overview_path, dpi=120)
    plt.close()
    print(f"  Per-frame phase images saved: {phase_dir}/ ({n_frames} frames + overview)")


def _save_perframe_displacement_comparison(result, output_dir):
    """
    逐帧位移对比可视化 (互相关恢复 vs GT)。
    每帧: GT/Rec dx/dy 的 colormap + quiver 箭头叠加对比 + 误差图。
    最后: 一张 RMSE 随帧变化的曲线图。
    """
    method = result['method']
    all_shifts = result['all_shifts_px']        # (n_frames, rows, cols, 2)
    gt_disp = result['gt_displacements']
    truth_displacement_x_all_px = gt_disp['displacement_x_px']    # (n_gt, rows, cols)
    truth_displacement_y_all_px = gt_disp['displacement_y_px']

    n_rec = all_shifts.shape[0]
    n_gt = truth_displacement_x_all_px.shape[0]
    n_compare = min(n_rec, n_gt)
    min_r = min(all_shifts.shape[1], truth_displacement_x_all_px.shape[1])
    min_c = min(all_shifts.shape[2], truth_displacement_x_all_px.shape[2])

    disp_dir = os.path.join(output_dir, "per_frame_displacement")
    os.makedirs(disp_dir, exist_ok=True)

    frame_stats_list, per_frame_rmse_dx, per_frame_rmse_dy, per_frame_rmse_total = _collect_per_frame_displacement_rmse_series(
        all_shifts=all_shifts,
        truth_displacement_x_all_px=truth_displacement_x_all_px,
        truth_displacement_y_all_px=truth_displacement_y_all_px,
        n_compare=n_compare,
        min_rows=min_r,
        min_cols=min_c,
    )

    for i, frame_stats in enumerate(frame_stats_list):
        _save_single_frame_displacement_panels(
            disp_dir=disp_dir,
            method=method,
            frame_index=i,
            truth_displacement_x_px=frame_stats["truth_displacement_x_px"],
            truth_displacement_y_px=frame_stats["truth_displacement_y_px"],
            recovered_displacement_x_px=frame_stats["recovered_displacement_x_px"],
            recovered_displacement_y_px=frame_stats["recovered_displacement_y_px"],
            displacement_error_x_px=frame_stats["displacement_error_x_px"],
            displacement_error_y_px=frame_stats["displacement_error_y_px"],
            rmse_dx=frame_stats["rmse_dx"],
            rmse_dy=frame_stats["rmse_dy"],
            rmse_total=frame_stats["rmse_total"],
        )

    _save_displacement_quiver_overview(
        disp_dir=disp_dir,
        method=method,
        all_shifts=all_shifts,
        truth_displacement_x_all_px=truth_displacement_x_all_px,
        truth_displacement_y_all_px=truth_displacement_y_all_px,
        n_compare=n_compare,
        min_rows=min_r,
        min_cols=min_c,
        per_frame_rmse_total=per_frame_rmse_total,
    )
    _save_per_frame_rmse_curve(
        disp_dir=disp_dir,
        method=method,
        n_compare=n_compare,
        per_frame_rmse_dx=per_frame_rmse_dx,
        per_frame_rmse_dy=per_frame_rmse_dy,
        per_frame_rmse_total=per_frame_rmse_total,
    )

    print(f"  Per-frame displacement comparison saved: {disp_dir}/ "
          f"({n_compare} frames + quiver + RMSE curve)")


def _compute_perframe_displacement_frame_stats(
    *,
    all_shifts: np.ndarray,
    truth_displacement_x_all_px: np.ndarray,
    truth_displacement_y_all_px: np.ndarray,
    frame_index: int,
    min_rows: int,
    min_cols: int,
) -> dict:
    recovered_displacement_x_px = all_shifts[frame_index, :min_rows, :min_cols, 0]
    recovered_displacement_y_px = all_shifts[frame_index, :min_rows, :min_cols, 1]
    truth_displacement_x_px = truth_displacement_x_all_px[frame_index, :min_rows, :min_cols]
    truth_displacement_y_px = truth_displacement_y_all_px[frame_index, :min_rows, :min_cols]
    return _compute_displacement_frame_stats(
        recovered_displacement_x_px=recovered_displacement_x_px,
        recovered_displacement_y_px=recovered_displacement_y_px,
        truth_displacement_x_px=truth_displacement_x_px,
        truth_displacement_y_px=truth_displacement_y_px,
    )


def _save_single_frame_displacement_panels(
    *,
    disp_dir: str,
    method: str,
    frame_index: int,
    truth_displacement_x_px: np.ndarray,
    truth_displacement_y_px: np.ndarray,
    recovered_displacement_x_px: np.ndarray,
    recovered_displacement_y_px: np.ndarray,
    displacement_error_x_px: np.ndarray,
    displacement_error_y_px: np.ndarray,
    rmse_dx: float,
    rmse_dy: float,
    rmse_total: float,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    vmax_dx = _displacement_display_limits(truth_displacement_x_px, recovered_displacement_x_px)
    vmax_dy = _displacement_display_limits(truth_displacement_y_px, recovered_displacement_y_px)
    vmax_err_dx = _displacement_display_limits(displacement_error_x_px)
    vmax_err_dy = _displacement_display_limits(displacement_error_y_px)

    im = axes[0, 0].imshow(truth_displacement_x_px, cmap='RdBu_r', vmin=-vmax_dx, vmax=vmax_dx, origin='lower')
    axes[0, 0].set_title(f'GT dX (frame {frame_index})')
    plt.colorbar(im, ax=axes[0, 0], label='px')

    im = axes[0, 1].imshow(recovered_displacement_x_px, cmap='RdBu_r', vmin=-vmax_dx, vmax=vmax_dx, origin='lower')
    axes[0, 1].set_title(f'Rec dX ({method})')
    plt.colorbar(im, ax=axes[0, 1], label='px')

    im = axes[0, 2].imshow(displacement_error_x_px, cmap='RdBu_r', vmin=-vmax_err_dx, vmax=vmax_err_dx, origin='lower')
    axes[0, 2].set_title(f'Error dX (RMSE={rmse_dx:.4f})')
    plt.colorbar(im, ax=axes[0, 2], label='px')

    im = axes[1, 0].imshow(truth_displacement_y_px, cmap='RdBu_r', vmin=-vmax_dy, vmax=vmax_dy, origin='lower')
    axes[1, 0].set_title(f'GT dY (frame {frame_index})')
    plt.colorbar(im, ax=axes[1, 0], label='px')

    im = axes[1, 1].imshow(recovered_displacement_y_px, cmap='RdBu_r', vmin=-vmax_dy, vmax=vmax_dy, origin='lower')
    axes[1, 1].set_title(f'Rec dY ({method})')
    plt.colorbar(im, ax=axes[1, 1], label='px')

    im = axes[1, 2].imshow(displacement_error_y_px, cmap='RdBu_r', vmin=-vmax_err_dy, vmax=vmax_err_dy, origin='lower')
    axes[1, 2].set_title(f'Error dY (RMSE={rmse_dy:.4f})')
    plt.colorbar(im, ax=axes[1, 2], label='px')

    fig.suptitle(
        f"Frame {frame_index} Displacement — {method}\n"
        f"RMSE: dX={rmse_dx:.4f}, dY={rmse_dy:.4f}, total={rmse_total:.4f} px",
        fontsize=12,
    )
    plt.tight_layout()
    fig_path = os.path.join(disp_dir, f"displacement_frame_{frame_index:03d}.png")
    plt.savefig(fig_path, dpi=120)
    plt.close()


def _save_displacement_quiver_overview(
    *,
    disp_dir: str,
    method: str,
    all_shifts: np.ndarray,
    truth_displacement_x_all_px: np.ndarray,
    truth_displacement_y_all_px: np.ndarray,
    n_compare: int,
    min_rows: int,
    min_cols: int,
    per_frame_rmse_total: list[float],
) -> None:
    sample_indices = sorted(set([0, n_compare // 2, n_compare - 1]))
    fig, axes = plt.subplots(1, len(sample_indices), figsize=(6 * len(sample_indices), 5))
    if len(sample_indices) == 1:
        axes = [axes]
    yy, xx = np.mgrid[:min_rows, :min_cols]
    for ax, idx in zip(axes, sample_indices):
        truth_displacement_x_px = truth_displacement_x_all_px[idx, :min_rows, :min_cols]
        truth_displacement_y_px = truth_displacement_y_all_px[idx, :min_rows, :min_cols]
        recovered_displacement_x_px = all_shifts[idx, :min_rows, :min_cols, 0]
        recovered_displacement_y_px = all_shifts[idx, :min_rows, :min_cols, 1]
        ax.quiver(xx, yy, truth_displacement_x_px, truth_displacement_y_px, color='blue', alpha=0.6, scale=None, label='GT')
        ax.quiver(xx, yy, recovered_displacement_x_px, recovered_displacement_y_px, color='red', alpha=0.6, scale=None, label='Rec')
        ax.set_title(f"Frame {idx}\nRMSE={per_frame_rmse_total[idx]:.4f} px")
        ax.set_aspect('equal')
        ax.legend(fontsize=8, loc='upper right')
        ax.invert_yaxis()
    fig.suptitle(f"Displacement Quiver — {method} (blue=GT, red=Rec)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(disp_dir, f"quiver_overview_{method}.png"), dpi=120)
    plt.close()


def _save_per_frame_rmse_curve(
    *,
    disp_dir: str,
    method: str,
    n_compare: int,
    per_frame_rmse_dx: list[float],
    per_frame_rmse_dy: list[float],
    per_frame_rmse_total: list[float],
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    frames_x = np.arange(n_compare)
    ax.plot(frames_x, per_frame_rmse_dx, 'o-', label='RMSE dX', color='tab:blue')
    ax.plot(frames_x, per_frame_rmse_dy, 's-', label='RMSE dY', color='tab:orange')
    ax.plot(frames_x, per_frame_rmse_total, '^-', label='RMSE total', color='tab:red', lw=2)
    ax.set_xlabel('Frame index')
    ax.set_ylabel('RMSE (px)')
    ax.set_title(f'Per-Frame Displacement RMSE — {method}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(frames_x)
    plt.tight_layout()
    plt.savefig(os.path.join(disp_dir, f"per_frame_rmse_{method}.png"), dpi=120)
    plt.close()


def _extract_per_frame_metric_series(metrics: list[dict]) -> dict:
    """Collect per-frame RMSE/correlation series for plotting."""
    return {
        "rmse_dx": [m['rmse_dx_px'] for m in metrics],
        "rmse_dy": [m['rmse_dy_px'] for m in metrics],
        "rmse_total": [m['rmse_total_px'] for m in metrics],
        "corr_dx": [m['corr_dx'] for m in metrics],
        "corr_dy": [m['corr_dy'] for m in metrics],
    }


def _build_metric_statistics_summary(metric_series: dict) -> dict:
    """Prepare boxplot/bar-chart inputs for per-frame metric summary plots."""
    labels = ['RMSE Total', 'Corr dX', 'Corr dY']
    value_series = [
        metric_series["rmse_total"],
        metric_series["corr_dx"],
        metric_series["corr_dy"],
    ]
    means = [float(np.mean(values)) for values in value_series]
    stds = [float(np.std(values)) for values in value_series]
    return {
        "labels": labels,
        "value_series": value_series,
        "means": means,
        "stds": stds,
    }


def _correlation_series_ylim(corr_dx: list[float], corr_dy: list[float]) -> tuple[float, float]:
    """Return a stable y-axis range for correlation time-series plots."""
    lower = min(0.0, min(min(corr_dx), min(corr_dy)) - 0.1)
    return lower, 1.05


def _plot_metric_series_panels(axes, frames_x: np.ndarray, metric_series: dict) -> None:
    """Render RMSE and correlation time-series panels."""
    rmse_dx = metric_series["rmse_dx"]
    rmse_dy = metric_series["rmse_dy"]
    rmse_total = metric_series["rmse_total"]
    corr_dx = metric_series["corr_dx"]
    corr_dy = metric_series["corr_dy"]

    axes[0].plot(frames_x, rmse_dx, 'o-', label='RMSE dX', color='tab:blue')
    axes[0].plot(frames_x, rmse_dy, 's-', label='RMSE dY', color='tab:orange')
    axes[0].plot(frames_x, rmse_total, '^-', label='RMSE total', color='tab:red', lw=2)
    axes[0].set_xlabel('Pulse Index')
    axes[0].set_ylabel('RMSE (pixels)')
    axes[0].set_title('RMSE vs Pulse Index')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frames_x, corr_dx, 'o-', label='Corr dX', color='tab:blue')
    axes[1].plot(frames_x, corr_dy, 's-', label='Corr dY', color='tab:orange')
    axes[1].set_xlabel('Pulse Index')
    axes[1].set_ylabel('Correlation Coefficient')
    axes[1].set_title('Correlation vs Pulse Index')
    axes[1].set_ylim(*_correlation_series_ylim(corr_dx, corr_dy))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)


def _plot_metric_distribution_panels(axes, metric_summary: dict) -> None:
    """Render metric boxplot and mean/std summary panels."""
    axes[0].boxplot(metric_summary["value_series"], tick_labels=metric_summary["labels"])
    axes[0].set_ylabel('Value')
    axes[0].set_title('Metric Distribution (Boxplot)')
    axes[0].grid(True, alpha=0.3, axis='y')

    x_pos = np.arange(len(metric_summary["labels"]))
    axes[1].bar(
        x_pos,
        metric_summary["means"],
        yerr=metric_summary["stds"],
        capsize=5,
        color=['tab:red', 'tab:blue', 'tab:orange'],
        alpha=0.7,
    )
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(metric_summary["labels"])
    axes[1].set_ylabel('Value (Mean ± Std)')
    axes[1].set_title('Metric Statistics (Bar chart)')
    axes[1].grid(True, alpha=0.3, axis='y')


def _save_metrics_statistics(result, output_dir):
    """保存随帧变化的统计折线图和箱线图（RMSE 和相关系数）。"""
    method = result['method']
    metrics = result.get('disp_metrics', {}).get('per_frame_metrics', [])
    if not metrics:
        return
        
    n_frames = len(metrics)
    frames_x = np.arange(n_frames)
    metric_series = _extract_per_frame_metric_series(metrics)
    metric_summary = _build_metric_statistics_summary(metric_series)
    
    stat_dir = os.path.join(output_dir, "statistics")
    os.makedirs(stat_dir, exist_ok=True)
    
    # 1. 随帧折线图 (Line plot)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    _plot_metric_series_panels(axes, frames_x, metric_series)
    
    plt.suptitle(f"Reconstruction Metrics Across All Pulses — {method}", fontsize=14)
    plt.tight_layout()
    line_path = os.path.join(stat_dir, f"metrics_all_pulses_{method}.png")
    plt.savefig(line_path, dpi=120)
    plt.close()
    
    # 2. 箱线图 (Boxplot) + 柱状图 (Bar chart with standard deviation)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    _plot_metric_distribution_panels(axes, metric_summary)
    
    plt.suptitle(f"Statistical Summary of Errors — {method}", fontsize=14)
    plt.tight_layout()
    stat_path = os.path.join(stat_dir, f"error_statistics_{method}.png")
    plt.savefig(stat_path, dpi=120)
    plt.close()
    
    print(f"  Statistical plots saved: {stat_dir}/")


def _prepare_phase_frame_stats(
    *,
    truth_phase_rad: np.ndarray,
    recovered_phase_rad: np.ndarray,
) -> dict:
    """Crop phase maps to overlap, remove piston, and compute error statistics."""
    min_rows = min(truth_phase_rad.shape[0], recovered_phase_rad.shape[0])
    min_cols = min(truth_phase_rad.shape[1], recovered_phase_rad.shape[1])
    truth_phase_crop_rad = truth_phase_rad[:min_rows, :min_cols]
    recovered_phase_crop_rad = recovered_phase_rad[:min_rows, :min_cols]

    truth_phase_zero_mean_rad = truth_phase_crop_rad - np.mean(truth_phase_crop_rad)
    recovered_phase_zero_mean_rad = recovered_phase_crop_rad - np.mean(recovered_phase_crop_rad)
    phase_error_rad = recovered_phase_zero_mean_rad - truth_phase_zero_mean_rad
    rmse_rad = float(np.sqrt(np.mean(phase_error_rad**2)))
    vmax_phase = max(
        np.max(np.abs(truth_phase_zero_mean_rad)),
        np.max(np.abs(recovered_phase_zero_mean_rad)),
        1e-6,
    )
    vmax_error = max(np.max(np.abs(phase_error_rad)), 1e-6)
    return {
        "truth_phase_rad": truth_phase_zero_mean_rad,
        "recovered_phase_rad": recovered_phase_zero_mean_rad,
        "phase_error_rad": phase_error_rad,
        "rmse_rad": rmse_rad,
        "vmax_phase": vmax_phase,
        "vmax_error": vmax_error,
    }


def _plot_phase_summary_row(axes_row, fig, frame_stats: dict, frame_index: int) -> None:
    """Render one row of the all-frames phase summary figure."""
    truth_phase_rad = frame_stats["truth_phase_rad"]
    recovered_phase_rad = frame_stats["recovered_phase_rad"]
    phase_error_rad = frame_stats["phase_error_rad"]
    rmse_val = frame_stats["rmse_rad"]
    vmax_phase = frame_stats["vmax_phase"]
    vmax_err = frame_stats["vmax_error"]

    im0 = axes_row[0].imshow(truth_phase_rad, cmap='RdBu_r', origin='lower', vmin=-vmax_phase, vmax=vmax_phase)
    axes_row[0].set_title(f"Frame {frame_index} GT Phase (rad)" if frame_index == 0 else f"Frame {frame_index} GT Phase")
    fig.colorbar(im0, ax=axes_row[0], fraction=0.046, pad=0.04)

    im1 = axes_row[1].imshow(recovered_phase_rad, cmap='RdBu_r', origin='lower', vmin=-vmax_phase, vmax=vmax_phase)
    axes_row[1].set_title(f"Frame {frame_index} Rec Phase" if frame_index == 0 else "")
    fig.colorbar(im1, ax=axes_row[1], fraction=0.046, pad=0.04)

    im2 = axes_row[2].imshow(phase_error_rad, cmap='hot', origin='lower', vmin=0, vmax=vmax_err)
    axes_row[2].set_title(
        f"Frame {frame_index} Error\nRMSE={rmse_val:.3f} rad"
        if frame_index == 0
        else f"RMSE={rmse_val:.3f} rad"
    )
    fig.colorbar(im2, ax=axes_row[2], fraction=0.046, pad=0.04)

    for ax in axes_row:
        ax.axis('off')


def _plot_displacement_summary_row(axes_row, fig, frame_stats: dict, frame_index: int) -> None:
    """Render one row of the all-frames displacement summary figure."""
    truth_displacement_x_px = frame_stats["truth_displacement_x_px"]
    recovered_displacement_x_px = frame_stats["recovered_displacement_x_px"]
    truth_displacement_y_px = frame_stats["truth_displacement_y_px"]
    recovered_displacement_y_px = frame_stats["recovered_displacement_y_px"]
    displacement_error_x_px = frame_stats["displacement_error_x_px"]
    displacement_error_y_px = frame_stats["displacement_error_y_px"]
    rmse_dx = frame_stats["rmse_dx"]
    rmse_dy = frame_stats["rmse_dy"]

    vmax_dx = _displacement_display_limits(truth_displacement_x_px, recovered_displacement_x_px)
    vmax_dy = _displacement_display_limits(truth_displacement_y_px, recovered_displacement_y_px)
    vmax_err_dx = _displacement_display_limits(displacement_error_x_px)
    vmax_err_dy = _displacement_display_limits(displacement_error_y_px)

    im0 = axes_row[0].imshow(truth_displacement_x_px, cmap='RdBu_r', origin='lower', vmin=-vmax_dx, vmax=vmax_dx)
    axes_row[0].set_title(f"Frame {frame_index} GT dX (px)" if frame_index == 0 else "")
    fig.colorbar(im0, ax=axes_row[0], fraction=0.046, pad=0.04)

    im1 = axes_row[1].imshow(recovered_displacement_x_px, cmap='RdBu_r', origin='lower', vmin=-vmax_dx, vmax=vmax_dx)
    axes_row[1].set_title("Rec dX" if frame_index == 0 else "")
    fig.colorbar(im1, ax=axes_row[1], fraction=0.046, pad=0.04)

    im2 = axes_row[2].imshow(displacement_error_x_px, cmap='RdBu_r', origin='lower', vmin=-vmax_err_dx, vmax=vmax_err_dx)
    axes_row[2].set_title(f"Err dX (RMSE={rmse_dx:.3f})" if frame_index == 0 else f"Err dX RMSE={rmse_dx:.3f}")
    fig.colorbar(im2, ax=axes_row[2], fraction=0.046, pad=0.04)

    im3 = axes_row[3].imshow(truth_displacement_y_px, cmap='RdBu_r', origin='lower', vmin=-vmax_dy, vmax=vmax_dy)
    axes_row[3].set_title(f"Frame {frame_index} GT dY (px)" if frame_index == 0 else "")
    fig.colorbar(im3, ax=axes_row[3], fraction=0.046, pad=0.04)

    im4 = axes_row[4].imshow(recovered_displacement_y_px, cmap='RdBu_r', origin='lower', vmin=-vmax_dy, vmax=vmax_dy)
    axes_row[4].set_title("Rec dY" if frame_index == 0 else "")
    fig.colorbar(im4, ax=axes_row[4], fraction=0.046, pad=0.04)

    im5 = axes_row[5].imshow(displacement_error_y_px, cmap='RdBu_r', origin='lower', vmin=-vmax_err_dy, vmax=vmax_err_dy)
    axes_row[5].set_title(f"Err dY (RMSE={rmse_dy:.3f})" if frame_index == 0 else f"Err dY RMSE={rmse_dy:.3f}")
    fig.colorbar(im5, ax=axes_row[5], fraction=0.046, pad=0.04)

    for ax in axes_row:
        ax.axis('off')


def _save_all_frames_phase_summary(result, output_dir):
    """保存所有帧的 GT Phase, Rec Phase, Phase Error 汇总大图。"""
    method = result['method']
    all_recovered_phase_rad = result.get('all_phase_rad')
    gt_disp = result.get('gt_displacements')
    
    if all_recovered_phase_rad is None or gt_disp is None or 'phase_rad' not in gt_disp:
        return

    truth_phase_all_rad = gt_disp['phase_rad']  # (F, R, C)
    n_frames = min(all_recovered_phase_rad.shape[0], truth_phase_all_rad.shape[0])
    n_frames = min(n_frames, 20)  # limit max frames to avoid giant images

    fig, axes = plt.subplots(n_frames, 3, figsize=(12, 3 * n_frames))
    if n_frames == 1:
        axes = np.expand_dims(axes, 0)
        
    for f in range(n_frames):
        frame_stats = _prepare_phase_frame_stats(
            truth_phase_rad=truth_phase_all_rad[f],
            recovered_phase_rad=all_recovered_phase_rad[f],
        )
        _plot_phase_summary_row(axes[f], fig, frame_stats, f)
        
        for ax in axes[f]:
            ax.axis('off')

    plt.suptitle(f"Phase Recovery Summary — {method}", fontsize=16, y=0.99)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"all_frames_phase_summary_{method}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Master phase summary saved: {save_path}")


def _save_all_frames_disp_summary(result, output_dir):
    """保存所有帧的 GT dX/dY, Rec dX/dY, Error dX/dY 汇总六列大图。"""
    method = result['method']
    all_shifts_rec = result.get('all_shifts_px')
    gt_disp = result.get('gt_displacements')
    
    if all_shifts_rec is None or gt_disp is None or 'displacement_x_px' not in gt_disp:
        return

    truth_displacement_x_all_px = gt_disp['displacement_x_px']
    truth_displacement_y_all_px = gt_disp['displacement_y_px']
    
    n_frames = min(all_shifts_rec.shape[0], truth_displacement_x_all_px.shape[0])
    n_frames = min(n_frames, 20)  # limit max frames to avoid giant images
    min_rows = min(all_shifts_rec.shape[1], truth_displacement_x_all_px.shape[1])
    min_cols = min(all_shifts_rec.shape[2], truth_displacement_x_all_px.shape[2])
    frame_stats_list, _, _, _ = _collect_per_frame_displacement_rmse_series(
        all_shifts=all_shifts_rec,
        truth_displacement_x_all_px=truth_displacement_x_all_px,
        truth_displacement_y_all_px=truth_displacement_y_all_px,
        n_compare=n_frames,
        min_rows=min_rows,
        min_cols=min_cols,
    )

    fig, axes = plt.subplots(n_frames, 6, figsize=(24, 3 * n_frames))
    if n_frames == 1:
        axes = np.expand_dims(axes, 0)
        
    for f, frame_stats in enumerate(frame_stats_list):
        _plot_displacement_summary_row(axes[f], fig, frame_stats, f)
        
        for ax in axes[f]:
            ax.axis('off')

    plt.suptitle(f"Displacement Recovery Summary (dX and dY) — {method}", fontsize=16, y=0.99)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"all_frames_disp_summary_{method}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Master displacement summary saved: {save_path}")


def save_comparison(all_results: list, output_dir: str):
    """保存多方法对比图。"""
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, all_results):
        im = ax.imshow(result['phase_no_tilt_rad'], cmap='RdBu_r', origin='lower')
        title = result['method']
        if 'metrics' in result:
            title += f"\nRMSE={result['metrics']['rmse_rad']:.3f} rad"
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='rad')

    plt.suptitle("Method Comparison")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "method_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Comparison figure saved: {fig_path}")


# =============================================================================
# 流程 B: 优化求解法 (Optimization-based Recovery)
# =============================================================================


def _prepare_full_optimization_inputs(data: dict, params: dict) -> dict:
    """Resolve full-mode optimization inputs from simulation outputs and metadata."""
    image = np.asarray(data['image'], dtype=np.float64)
    ideal_accumulated = data.get('ideal_image')
    movement = _resolve_movement_metadata(data=data, params=params)
    global_offsets_px = _movement_global_offsets_px(movement)
    if global_offsets_px is None:
        raise ValueError("Full-mode optimization requires movement pulse offsets in pixels.")

    pulse_reference_images, reference_sequence_source = _build_pulse_reference_images(data, movement)
    if ideal_accumulated is None:
        if pulse_reference_images is None:
            raise ValueError(
                "Full-mode optimization requires ideal_accumulated.npy or a recoverable ideal reference sequence."
            )
        ideal_accumulated = np.sum(np.stack(pulse_reference_images, axis=0), axis=0)
    ideal_accumulated = np.asarray(ideal_accumulated, dtype=np.float64)

    gt_disp = data.get('gt_displacements')
    ideal_centers = data.get('ideal_centers')
    spot_grid_indices = data.get('spot_grid_indices')
    grid_rows = int(params.get('spot_grid_rows') or 0)
    grid_cols = int(params.get('spot_grid_cols') or 0)
    if ideal_centers is None:
        grid_geometry = _build_full_mode_grid_geometry(params, image.shape, gt_disp=gt_disp)
        ideal_centers = grid_geometry.centers_xy_px
        spot_grid_indices = grid_geometry.grid_indices_rowcol
        grid_rows = grid_geometry.grid_rows
        grid_cols = grid_geometry.grid_cols
    else:
        ideal_centers = assert_xy_points(ideal_centers, name="ideal_centers")
        if spot_grid_indices is not None:
            spot_grid_indices = assert_row_col_indices(spot_grid_indices, name="spot_grid_indices")
        elif grid_rows > 0 and grid_cols > 0:
            grid_geometry = _build_full_mode_grid_geometry(params, image.shape, gt_disp=gt_disp)
            spot_grid_indices = grid_geometry.grid_indices_rowcol

    pixel_size = float(params['pixel_size_m'])
    pitch = float(params['pitch_m'])
    return {
        "image": image,
        "ideal_accumulated": ideal_accumulated,
        "movement": movement,
        "global_offsets_px": global_offsets_px,
        "pulse_reference_images": pulse_reference_images,
        "reference_sequence_source": reference_sequence_source,
        "gt_disp": gt_disp,
        "ideal_centers": ideal_centers,
        "spot_grid_indices": spot_grid_indices,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "n_frames": int(params['n_pulses']),
        "pixel_size": pixel_size,
        "pitch": pitch,
        "z_prop": float(params['z_prop_m']),
        "wavelength": float(params['wavelength_m']),
        "spacing_px": pitch / pixel_size,
    }


def _resolve_full_optimization_grid_shape(full_opt: dict, image_shape: tuple[int, int], window_size_m) -> tuple[int, int]:
    """Choose the grid shape used for full-mode optimization initialization/reshape."""
    gt_disp = full_opt["gt_disp"]
    if gt_disp is not None and 'grid_rows' in gt_disp and 'grid_cols' in gt_disp:
        return int(gt_disp['grid_rows']), int(gt_disp['grid_cols'])
    grid_rows = int(full_opt["grid_rows"])
    grid_cols = int(full_opt["grid_cols"])
    if grid_rows > 0 and grid_cols > 0:
        return grid_rows, grid_cols
    return estimate_grid_size(
        image_shape,
        full_opt["pitch"],
        full_opt["pixel_size"],
        window_size_m=window_size_m,
    )


def _resolve_full_optimization_initializer(
    *,
    data: dict,
    movement: dict,
    n_frames: int,
    init_rows: int,
    init_cols: int,
) -> tuple[np.ndarray | None, str | None]:
    """Build an optional decomposition-based initializer for full-mode optimization."""
    if not bool(getattr(config, 'OPTIMIZATION_USE_DECOMPOSITION_INIT', True)):
        return None, None
    try:
        init_result = _recover_accumulated_with_decomposition(
            data=data,
            movement=movement,
            rows=init_rows,
            cols=init_cols,
            method_name='cross_correlation',
        )
    except Exception as exc:
        print(f"  Warning: optimization initializer unavailable: {exc}")
        return None, None
    if init_result is None or init_result.get('all_shifts_arr') is None:
        return None, None
    initial_shifts = np.asarray(init_result['all_shifts_arr'], dtype=np.float64).reshape(n_frames, -1, 2)
    optimization_init_source = f"decomposition_{init_result.get('decomposition_method', 'unknown')}"
    print(f"  Optimization initializer: {optimization_init_source}")
    return initial_shifts, optimization_init_source


def _run_full_optimization_solver(full_opt: dict, *, initial_shifts: np.ndarray | None):
    """Execute the full-mode accumulated-image optimizer."""
    return run_optimization_recovery(
        accumulated_image=full_opt["image"],
        ideal_accumulated_image=full_opt["ideal_accumulated"],
        pulse_ideal_images=full_opt["pulse_reference_images"] or [],
        ideal_centers=full_opt["ideal_centers"],
        n_frames=full_opt["n_frames"],
        global_offsets_px=full_opt["global_offsets_px"],
        spacing=full_opt["spacing_px"],
        pixel_size_m=full_opt["pixel_size"],
        z_prop_m=full_opt["z_prop"],
        wavelength_m=full_opt["wavelength"],
        method=config.OPTIMIZATION_SOLVER,
        max_shift=config.OPTIMIZATION_MAX_SHIFT,
        lambda_spatial=config.OPTIMIZATION_LAMBDA_SPATIAL,
        lambda_temporal=config.OPTIMIZATION_LAMBDA_TEMPORAL,
        n_zernike_modes=config.OPTIMIZATION_N_ZERNIKE,
        spot_sigma_scale=config.SPOT_SIGMA_SCALE,
        spot_model=config.OPTIMIZATION_SPOT_MODEL,
        initial_shifts=initial_shifts,
        ftol=config.OPTIMIZATION_FTOL,
        gtol=config.OPTIMIZATION_GTOL,
        verbose=True,
    )


def _reshape_full_optimization_shifts(full_opt: dict, opt_result) -> np.ndarray:
    """Project flat optimization shifts back to the Hartmann detector grid."""
    spot_grid_indices = full_opt["spot_grid_indices"]
    gt_disp = full_opt["gt_disp"]
    grid_rows = int(full_opt["grid_rows"])
    grid_cols = int(full_opt["grid_cols"])

    valid_spot_grid_indices = None
    if spot_grid_indices is not None:
        if getattr(opt_result, 'valid_center_indices', None) is not None:
            valid_spot_grid_indices = spot_grid_indices[np.asarray(opt_result.valid_center_indices, dtype=np.int64)]
        elif len(spot_grid_indices) == len(opt_result.valid_centers):
            valid_spot_grid_indices = spot_grid_indices

    return reshape_shifts_to_grid(
        opt_result.shifts,
        opt_result.valid_centers,
        int(gt_disp['grid_rows']) if gt_disp is not None and 'grid_rows' in gt_disp else grid_rows,
        int(gt_disp['grid_cols']) if gt_disp is not None and 'grid_cols' in gt_disp else grid_cols,
        spot_grid_indices=valid_spot_grid_indices,
    )


def _run_simple_optimization_recovery(data: dict, params: dict) -> dict:
    """Run legacy simple-mode Gaussian optimization recovery."""
    from simulation.simple_spots import (
        choose_solver_method,
        solve_subbeam_positions_fast,
        solve_zernike_constrained,
    )

    image = data['image']
    ideal_centers = data.get('ideal_centers')
    gt_shifts = data.get('gt_shifts')

    if ideal_centers is None:
        raise ValueError(
            "Optimization method requires ideal_centers.npy.\n"
            "This is only available for simple mode data (simulate.py --simple)."
        )

    n_frames = params['n_pulses']
    spot_size = params['spot_size']
    spacing = params['pitch_m']  # simple mode: pitch_m = spacing in px
    shift_per_frame_dx = params['shift_per_frame_dx']
    shift_per_frame_dy = params['shift_per_frame_dy']
    shift_per_frame = (shift_per_frame_dx, shift_per_frame_dy)

    print(f"\n  === Optimization Recovery (Simple Mode) ===")
    print(f"  Frames: {n_frames}, Spots: {len(ideal_centers)}")
    print(f"  Spot size: {spot_size}, Spacing: {spacing}")
    print(f"  Shift/frame: dx={shift_per_frame_dx:.2f}, dy={shift_per_frame_dy:.2f}")

    solver = config.OPTIMIZATION_SOLVER
    shift_mag = np.sqrt(shift_per_frame_dy**2 + shift_per_frame_dx**2)
    if solver == 'auto':
        solver = choose_solver_method(shift_mag, spot_size, spacing)
        print(f"  Auto-selected solver: {solver}")
    else:
        print(f"  Solver: {solver}")

    if solver == 'zernike':
        recovered_shifts, final_loss = solve_zernike_constrained(
            image.astype(np.float64), image.shape, ideal_centers,
            n_frames, shift_per_frame, spot_size,
            n_modes=config.OPTIMIZATION_N_ZERNIKE,
            max_coeff=config.OPTIMIZATION_MAX_SHIFT,
            lambda_temporal=config.OPTIMIZATION_LAMBDA_TEMPORAL,
            lambda_coeff=config.OPTIMIZATION_LAMBDA_COEFF,
        )
    else:
        recovered_shifts, final_loss = solve_subbeam_positions_fast(
            image.astype(np.float64), image.shape, ideal_centers,
            n_frames, shift_per_frame, spot_size,
            max_shift=config.OPTIMIZATION_MAX_SHIFT,
            spacing=spacing,
            lambda_spatial=config.OPTIMIZATION_LAMBDA_SPATIAL,
            lambda_temporal=config.OPTIMIZATION_LAMBDA_TEMPORAL,
        )

    print(f"  Final loss: {final_loss:.4e}")

    result = {
        "method": f"optimization_{solver}",
        "solver": solver,
        "recovered_shifts": recovered_shifts,
        "ideal_centers": ideal_centers,
        "final_loss": final_loss,
        "n_frames": n_frames,
    }

    if gt_shifts is not None:
        errors = np.sqrt(np.sum((gt_shifts - recovered_shifts)**2, axis=2)).flatten()
        result['errors'] = errors
        result['gt_shifts'] = gt_shifts
        print(f"\n  === Shift Estimation Results ===")
        print(f"  Mean error:   {np.mean(errors):.4f} px")
        print(f"  Median error: {np.median(errors):.4f} px")
        print(f"  Max error:    {np.max(errors):.4f} px")
        print(f"  Std error:    {np.std(errors):.4f} px")

    return result


def _select_representative_full_optimization_shifts(grid_shifts: np.ndarray, aberr_mode: str) -> np.ndarray:
    """Select the aggregate shift field used for summary displacement/phase maps."""
    if aberr_mode == 'per_pulse':
        return grid_shifts[0]
    return np.mean(grid_shifts, axis=0)


def _recover_shift_grid_products(
    grid_shifts: np.ndarray,
    *,
    pixel_size: float,
    z_prop: float,
    pitch: float,
    wavelength: float,
    aberr_mode: str,
) -> dict:
    """Recover aggregate and per-frame displacement/phase products from shift grids."""
    shifts_px = _select_representative_full_optimization_shifts(grid_shifts, aberr_mode)
    aggregate_recovery = recover_from_shifts(
        shifts_px,
        pixel_size=pixel_size,
        z_prop=z_prop,
        spacing=pitch,
        wavelength=wavelength,
        integration_method=config.INTEGRATION_METHOD,
    )

    if config.REMOVE_TIP_TILT:
        nr, nc = aggregate_recovery['phase_rad'].shape
        gx = np.arange(nc)
        gy = np.arange(nr)
        gx, gy = np.meshgrid(gx, gy)
        phase_no_tilt = remove_plane(aggregate_recovery['phase_rad'], gx, gy)
    else:
        phase_no_tilt = aggregate_recovery['phase_rad'].copy()

    phase_sequence = []
    displacement_x_sequence = []
    displacement_y_sequence = []
    for frame_shifts in grid_shifts:
        frame_recovery = recover_from_shifts(
            frame_shifts,
            pixel_size=pixel_size,
            z_prop=z_prop,
            spacing=pitch,
            wavelength=wavelength,
            integration_method=config.INTEGRATION_METHOD,
        )
        phase_sequence.append(frame_recovery['phase_rad'])
        displacement_x_sequence.append(frame_recovery['displacement_x_m'])
        displacement_y_sequence.append(frame_recovery['displacement_y_m'])

    return {
        "shifts_px": shifts_px,
        "aggregate_recovery": aggregate_recovery,
        "phase_no_tilt_rad": phase_no_tilt,
        "all_phase_rad": np.asarray(phase_sequence, dtype=np.float64),
        "all_displacement_x_m": np.asarray(displacement_x_sequence, dtype=np.float64),
        "all_displacement_y_m": np.asarray(displacement_y_sequence, dtype=np.float64),
    }


def _build_full_optimization_result(
    *,
    data: dict,
    full_opt: dict,
    opt_result,
    grid_shifts: np.ndarray,
    optimization_init_source: str,
) -> dict:
    """Build the full-mode optimization recovery result payload."""
    movement = full_opt["movement"]
    global_offsets_px = full_opt["global_offsets_px"]
    reference_sequence_source = full_opt["reference_sequence_source"]
    gt_disp = full_opt["gt_disp"]
    ideal_centers = full_opt["ideal_centers"]
    n_frames = full_opt["n_frames"]
    pixel_size = full_opt["pixel_size"]
    pitch = full_opt["pitch"]
    z_prop = full_opt["z_prop"]
    wavelength = full_opt["wavelength"]
    aberr_mode = data['params'].get('phase_aberration_mode', 'none')

    recovered_products = _recover_shift_grid_products(
        grid_shifts,
        pixel_size=pixel_size,
        z_prop=z_prop,
        pitch=pitch,
        wavelength=wavelength,
        aberr_mode=aberr_mode,
    )

    has_per_frame_aberrated_data = os.path.exists(os.path.join(data['sim_dir'], "frames", "frame_000.npy"))
    aggregate_recovery = recovered_products["aggregate_recovery"]
    result = {
        "method": f"optimization_{opt_result.solver_method}",
        "solver": opt_result.solver_method,
        "rows": int(grid_shifts.shape[1]),
        "cols": int(grid_shifts.shape[2]),
        "shifts_px": recovered_products["shifts_px"],
        "aberr_mode": aberr_mode,
        "all_shifts_px": grid_shifts,
        "all_phase_rad": recovered_products["all_phase_rad"],
        "all_displacement_x_m": recovered_products["all_displacement_x_m"],
        "all_displacement_y_m": recovered_products["all_displacement_y_m"],
        "n_frames_recovered": int(grid_shifts.shape[0]),
        "recovery_mode": "optimization_accumulated",
        "primary_recovery_mode": "accumulated",
        "has_per_frame_aberrated_data": has_per_frame_aberrated_data,
        "used_per_frame_aberrated_for_recovery": False,
        "movement": movement,
        "global_offsets_px": global_offsets_px,
        "reference_sequence_source": reference_sequence_source,
        "optimization_init_source": optimization_init_source,
        "displacement_x_m": aggregate_recovery['displacement_x_m'],
        "displacement_y_m": aggregate_recovery['displacement_y_m'],
        "phase_rad": aggregate_recovery['phase_rad'],
        "phase_no_tilt_rad": recovered_products["phase_no_tilt_rad"],
        "ideal_centers": ideal_centers,
        "valid_centers": opt_result.valid_centers,
        "recovered_shifts": opt_result.shifts,
        "final_loss": float(opt_result.final_loss),
        "convergence_history": [float(x) for x in opt_result.convergence_history],
        "n_iterations": int(opt_result.n_iterations),
        "elapsed_time": float(opt_result.elapsed_time),
        "n_frames": n_frames,
    }

    wfe_truth = data.get('wfe_truth')
    if wfe_truth is not None:
        result['metrics'] = compute_metrics(recovered_products["phase_no_tilt_rad"], wfe_truth, pitch, pixel_size)

    if gt_disp is not None:
        result['disp_metrics'] = compute_displacement_metrics_perframe(grid_shifts, gt_disp)
        result['gt_displacements'] = gt_disp

    return result


def recover_optimization(data: dict) -> dict:
    """
    使用优化求解法 (L-BFGS-B) 恢复位移。
    
    simple 模式沿用原始 Gaussian-spot 优化；
    full 模式使用 trajectory-aware accumulated inversion。
    """
    params = data['params']
    mode = params.get('mode', 'full')

    if mode == 'simple':
        return _run_simple_optimization_recovery(data, params)

    from analysis_laser.shift_optimization import (
        run_optimization_recovery,
        reshape_shifts_to_grid,
    )

    full_opt = _prepare_full_optimization_inputs(data, params)
    image = full_opt["image"]
    movement = full_opt["movement"]
    global_offsets_px = full_opt["global_offsets_px"]
    reference_sequence_source = full_opt["reference_sequence_source"]
    ideal_centers = full_opt["ideal_centers"]
    n_frames = full_opt["n_frames"]

    print(f"\n  === Optimization Recovery (Full Mode) ===")
    print(f"  Frames: {n_frames}, Spots: {len(ideal_centers)}")
    print(f"  Solver request: {config.OPTIMIZATION_SOLVER}")
    print(f"  Global offsets available: {global_offsets_px.shape}")
    if reference_sequence_source is not None:
        print(f"  Reference sequence source: {reference_sequence_source}")

    init_rows, init_cols = _resolve_full_optimization_grid_shape(
        full_opt,
        image.shape,
        params.get('window_size_m'),
    )
    initial_shifts, optimization_init_source = _resolve_full_optimization_initializer(
        data=data,
        movement=movement,
        n_frames=n_frames,
        init_rows=init_rows,
        init_cols=init_cols,
    )

    opt_result = _run_full_optimization_solver(full_opt, initial_shifts=initial_shifts)
    grid_shifts = _reshape_full_optimization_shifts(full_opt, opt_result)

    aberr_mode = params.get('phase_aberration_mode', 'none')
    if aberr_mode == 'per_pulse':
        print("  [per_pulse mode] optimization keeps per-pulse sequence; frame 0 used only for aggregate maps")
    return _build_full_optimization_result(
        data=data,
        full_opt=full_opt,
        opt_result=opt_result,
        grid_shifts=grid_shifts,
        optimization_init_source=optimization_init_source,
    )


def save_optimization_results(result: dict, output_dir: str, data: dict):
    """保存优化求解结果。"""
    if _is_full_optimization_result(result):
        _save_full_optimization_results(result, output_dir, data)
        return

    _save_simple_optimization_results(result, output_dir, data)


def _is_full_optimization_result(result: dict) -> bool:
    """Return whether an optimization result came from full-mode recovery."""
    return 'phase_no_tilt_rad' in result


def _write_json(path: str, payload: dict) -> None:
    """Write a JSON payload with stable indentation."""
    with open(path, "w") as fp:
        json.dump(payload, fp, indent=2)


def _build_full_optimization_summary(result: dict) -> dict:
    """Build the summary JSON payload for full-mode optimization recovery."""
    valid_centers = result.get('valid_centers')
    n_valid_centers = int(len(valid_centers)) if valid_centers is not None else 0
    summary = {
        "method": result['method'],
        "solver": result.get('solver'),
        "recovery_mode": result.get('recovery_mode'),
        "primary_recovery_mode": result.get('primary_recovery_mode'),
        "used_per_frame_aberrated_for_recovery": bool(result.get('used_per_frame_aberrated_for_recovery')),
        "reference_sequence_source": result.get('reference_sequence_source'),
        "optimization_init_source": result.get('optimization_init_source'),
        "final_loss": float(result.get('final_loss', float('nan'))),
        "n_iterations": int(result.get('n_iterations', 0)),
        "elapsed_time_s": float(result.get('elapsed_time', 0.0)),
        "n_frames": int(result.get('n_frames', result.get('n_frames_recovered', 0))),
        "valid_centers": n_valid_centers,
    }
    disp_metrics = result.get('disp_metrics')
    if disp_metrics:
        summary["rmse_total_px"] = float(disp_metrics.get('rmse_total_px', float('nan')))
        summary["mean_err_px"] = float(disp_metrics.get('mean_err_px', float('nan')))
    return summary


def _save_full_optimization_results(result: dict, output_dir: str, data: dict) -> None:
    """Save full-mode optimization outputs and metadata."""
    save_results(result, output_dir, data)
    _save_full_optimization_artifacts(result, output_dir)

def _save_full_optimization_artifacts(result: dict, output_dir: str) -> None:
    """Save full-mode optimization-specific arrays and summary artifacts."""
    method = result['method']
    os.makedirs(output_dir, exist_ok=True)

    if result.get('recovered_shifts') is not None:
        _save_named_array(output_dir, f"optimization_flat_shifts_{method}.npy", result['recovered_shifts'])
    if result.get('valid_centers') is not None:
        np.save(os.path.join(output_dir, f"optimization_valid_centers_{method}.npy"), result['valid_centers'])

    _save_optimization_convergence_artifact(
        history=result.get('convergence_history') or [],
        output_path=os.path.join(output_dir, f"optimization_convergence_{method}.png"),
        title=f"Optimization Convergence — {method}",
    )
    _write_json(
        os.path.join(output_dir, f"optimization_summary_{method}.json"),
        _build_full_optimization_summary(result),
    )


def _save_optimization_convergence_artifact(*, history: list[float], output_path: str, title: str) -> None:
    """Save convergence history plot when optimization iterations are available."""
    if not history:
        return
    _save_convergence_plot(history, output_path, title=title)


def _render_simple_optimization_reconstruction(result: dict, data: dict) -> np.ndarray:
    """Render the recovered accumulated image for simple-mode optimization."""
    from simulation.simple_spots import fast_render_gaussian_spots

    params = data['params']
    rec_shifts = result['recovered_shifts']
    ideal_centers = result['ideal_centers']
    n_frames = result['n_frames']
    spot_size = params['spot_size']
    shift_per_frame_dx = params['shift_per_frame_dx']
    shift_per_frame_dy = params['shift_per_frame_dy']
    image = data['image']
    shape = image.shape

    y_grid, x_grid = np.indices(shape)
    recovered_image = np.zeros(shape)
    for t in range(n_frames):
        gx = t * shift_per_frame_dx
        gy = t * shift_per_frame_dy
        centers_t = ideal_centers + np.array([gx, gy]) + rec_shifts[t]
        recovered_image += fast_render_gaussian_spots(shape, centers_t, spot_size, y_grid, x_grid)
    return recovered_image


def _build_simple_optimization_summary(result: dict) -> dict:
    """Build the metrics JSON payload for simple-mode optimization recovery."""
    summary = {
        "method": result['method'],
        "solver": result['solver'],
        "final_loss": float(result['final_loss']),
        "n_frames": int(result['n_frames']),
        "n_spots": int(len(result['ideal_centers'])),
    }
    if 'errors' in result:
        summary["mean_error_px"] = float(np.mean(result['errors']))
        summary["median_error_px"] = float(np.median(result['errors']))
        summary["max_error_px"] = float(np.max(result['errors']))
        summary["std_error_px"] = float(np.std(result['errors']))
    return summary


def _save_simple_optimization_results(result: dict, output_dir: str, data: dict) -> None:
    """Save simple-mode optimization outputs and diagnostic plots."""
    os.makedirs(output_dir, exist_ok=True)
    method = result['method']
    _save_named_array(output_dir, f"recovered_shifts_{method}.npy", result['recovered_shifts'])
    _save_simple_optimization_figure(result, output_dir, data)
    _write_json(
        os.path.join(output_dir, f"metrics_{method}.json"),
        _build_simple_optimization_summary(result),
    )


def _save_simple_optimization_figure(result: dict, output_dir: str, data: dict) -> None:
    """Render and save the simple-mode optimization comparison figure."""
    solver = result['solver']
    image = data['image']
    recovered_image = _render_simple_optimization_reconstruction(result, data)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].imshow(image, cmap='magma', origin='lower')
    axes[0].set_title(f"Observed ({result['n_frames']} frames)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(recovered_image, cmap='magma', origin='lower')
    axes[1].set_title(f"Recovered ({solver})")
    plt.colorbar(im1, ax=axes[1])

    if 'errors' in result:
        errors = result['errors']
        axes[2].hist(errors, bins=40, alpha=0.7, color='steelblue', edgecolor='white')
        axes[2].axvline(np.mean(errors), color='red', linestyle='--', lw=2, label=f'Mean: {np.mean(errors):.4f} px')
        axes[2].axvline(np.median(errors), color='orange', linestyle='--', lw=2, label=f'Median: {np.median(errors):.4f} px')
        axes[2].set_xlabel('Shift Error (px)')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Error Distribution')
        axes[2].legend()
    else:
        residual = image - recovered_image
        im2 = axes[2].imshow(residual, cmap='RdBu_r', origin='lower')
        axes[2].set_title('Residual (obs - rec)')
        plt.colorbar(im2, ax=axes[2])

    plt.suptitle(f"Optimization Recovery — {solver.upper()} Solver", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"recovery_{method}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Figure saved: {fig_path}")


def _can_run_optimization(data: dict) -> bool:
    """Return whether optimization recovery has enough priors to run."""
    params = data.get('params', {})
    if params.get('mode') == 'simple':
        return data.get('ideal_centers') is not None

    movement = _resolve_movement_metadata(data=data, params=params)
    has_offsets = _movement_has_global_offsets_px(movement)
    has_ideal_accumulated = data.get('ideal_image') is not None
    can_build_refs = _build_pulse_reference_images(data, movement)[0] is not None
    return bool(has_offsets and (has_ideal_accumulated or can_build_refs))


def _resolve_post_recovery_analysis_mode() -> str:
    """Validate and return the configured Step 3 auto-analysis mode."""
    mode = str(getattr(config, 'POST_RECOVERY_ANALYSIS_MODE', 'match') or 'match').lower()
    if mode not in {'match', 'both', 'none'}:
        raise ValueError(
            f"Unsupported POST_RECOVERY_ANALYSIS_MODE={mode!r}; expected 'match', 'both', or 'none'."
        )
    return mode


def _plan_post_recovery_analysis_branches(
    *,
    requested_method: str,
    optimization_available: bool,
    mode: str,
) -> list[tuple[str, bool]]:
    """Resolve the Step 3 analysis branches to run after Step 2 recovery."""
    if mode == 'none':
        return []
    if mode == 'both':
        branches: list[tuple[str, bool]] = [("step3_decomposition", True)]
        if optimization_available:
            branches.append(("step3_optimization", False))
        return branches

    if requested_method == 'optimization':
        return [("step3_optimization", False)] if optimization_available else []
    if requested_method == 'all':
        branches = [("step3_decomposition", True)]
        if optimization_available:
            branches.append(("step3_optimization", False))
        return branches
    return [("step3_decomposition", True)]


def _run_single_post_recovery_analysis(
    *,
    sim_dir: str,
    analysis_subdir_name: str,
    use_decomposition: bool,
) -> None:
    """Run one Step 3 analysis branch and persist its report."""
    from analysis_laser.hartmann_analysis import run_analysis_report_from_dir

    report_subdir = os.path.join("analysis", analysis_subdir_name)
    branch_name = "decomposition" if use_decomposition else "optimization"
    try:
        print(f"\n{'—' * 50}")
        print(f"  Running Step 3 analysis: {branch_name}")
        print(f"{'—' * 50}")
        run_analysis_report_from_dir(
            sim_dir,
            use_decomposition=use_decomposition,
            analysis_subdir=report_subdir,
        )
        print(f"  Step 3 report saved to: {os.path.join(sim_dir, report_subdir)}")
    except Exception as exc:
        print(f"  Warning: Step 3 {branch_name} analysis failed: {exc}")


def _run_post_recovery_analysis(sim_dir: str, *, requested_method: str, optimization_available: bool) -> None:
    """Run Step 3 analysis automatically after Step 2 recovery.

    Analysis reports are written into subdirectories under ``analysis/`` so they
    do not overwrite Step 2 recovery outputs.
    """
    if not bool(getattr(config, 'RUN_POST_RECOVERY_ANALYSIS', True)):
        return
    mode = _resolve_post_recovery_analysis_mode()
    branches = _plan_post_recovery_analysis_branches(
        requested_method=requested_method,
        optimization_available=optimization_available,
        mode=mode,
    )
    for analysis_subdir_name, use_decomposition in branches:
        _run_single_post_recovery_analysis(
            sim_dir=sim_dir,
            analysis_subdir_name=analysis_subdir_name,
            use_decomposition=use_decomposition,
        )


def _build_recover_arg_parser() -> argparse.ArgumentParser:
    """Create the Step 2 CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Step 2: Phase Recovery from Hartmann Sensor Data"
    )
    parser.add_argument(
        'sim_dir', type=str,
        help='Path to Step 1 output directory (e.g. result/001)'
    )
    parser.add_argument(
        '--method', type=str, default=config.DISPLACEMENT_METHOD,
        choices=ALL_METHODS + ['all'],
        help=f'Recovery method (default: {config.DISPLACEMENT_METHOD})'
    )
    parser.add_argument(
        '--patch-half', type=int, default=None,
        help='Cross-correlation patch half-width (overrides config)'
    )
    return parser


def _print_step2_banner() -> None:
    """Print the Step 2 banner."""
    print("=" * 70)
    print("  STEP 2: PHASE RECOVERY")
    print("=" * 70)


def _print_method_comparison_summary(all_results: list[dict]) -> None:
    """Print the per-method RMSE/correlation summary table."""
    print(f"\n{'=' * 60}")
    print(f"  Method Comparison Summary")
    print(f"{'=' * 60}")
    print(f"  {'Method':<25s} {'RMSE (rad)':<12s} {'Corr':<10s}")
    print(f"  {'-' * 47}")
    for result in all_results:
        metrics = result.get('metrics', {})
        rmse = f"{metrics.get('rmse_rad', float('nan')):.4f}" if metrics else 'N/A'
        corr = f"{metrics.get('correlation', float('nan')):.4f}" if metrics else 'N/A'
        print(f"  {result['method']:<25s} {rmse:<12s} {corr:<10s}")


def _run_requested_recovery_methods(
    *,
    data: dict,
    analysis_dir: str,
    requested_method: str,
    optimization_available: bool,
) -> None:
    """Dispatch the requested Step 2 recovery workflow."""
    if requested_method == 'optimization':
        result = recover_optimization(data)
        save_optimization_results(result, analysis_dir, data)
        return

    if requested_method == 'all':
        all_results = []
        for method in DISPLACEMENT_METHODS:
            print(f"\n{'—' * 50}")
            print(f"  Running: {method}")
            print(f"{'—' * 50}")
            result = recover_phase(data, method)
            save_results(result, analysis_dir, data)
            all_results.append(result)

        if optimization_available:
            print(f"\n{'—' * 50}")
            print(f"  Running: optimization")
            print(f"{'—' * 50}")
            opt_result = recover_optimization(data)
            save_optimization_results(opt_result, analysis_dir, data)

        if len(all_results) > 1:
            save_comparison(all_results, analysis_dir)
        _print_method_comparison_summary(all_results)
        return

    result = recover_phase(data, requested_method)
    save_results(result, analysis_dir, data)


# =============================================================================
# Main Entry Point
# =============================================================================

DISPLACEMENT_METHODS = ['centroid', 'cross_correlation']
ALL_METHODS = DISPLACEMENT_METHODS + ['optimization']


def main():
    parser = _build_recover_arg_parser()
    args = parser.parse_args()

    if args.patch_half is not None:
        config.XCORR_PATCH_HALF = args.patch_half

    _print_step2_banner()

    # Load data
    data = load_simulation_data(args.sim_dir)

    # Output directory
    analysis_dir = os.path.join(args.sim_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # --- Zernike 阶数一致性校验 ---
    sim_zernike_order = int(data.get('params', {}).get(
        'phase_aberration_max_order', 0))
    opt_n = int(getattr(config, 'OPTIMIZATION_N_ZERNIKE', 15))
    rec_n = int(getattr(config, 'RECONSTRUCTION_N_ZERNIKE', 15))
    if sim_zernike_order > 0:
        if opt_n < sim_zernike_order:
            print(f"  ⚠ WARNING: Simulation used {sim_zernike_order} Zernike "
                  f"modes, but OPTIMIZATION_N_ZERNIKE = {opt_n}. "
                  f"High-freq aberrations may be under-fitted.")
        if rec_n < sim_zernike_order:
            print(f"  ⚠ WARNING: Simulation used {sim_zernike_order} Zernike "
                  f"modes, but RECONSTRUCTION_N_ZERNIKE = {rec_n}. "
                  f"High-freq modes will be lost in reconstruction.")

    optimization_available = _can_run_optimization(data)
    _run_requested_recovery_methods(
        data=data,
        analysis_dir=analysis_dir,
        requested_method=args.method,
        optimization_available=optimization_available,
    )

    _run_post_recovery_analysis(
        args.sim_dir,
        requested_method=args.method,
        optimization_available=optimization_available,
    )

    print(f"\n✓ Step 2 complete. Results saved to: {analysis_dir}")


if __name__ == "__main__":
    main()
