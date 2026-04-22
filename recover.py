#!/usr/bin/env python3
from __future__ import annotations
"""
Step 2: 相位恢复 (Phase Recovery)

从 Step 1 (simulate.py) 的输出加载探测器图像，
执行位移检测或优化求解进行波前重建。

流程 A (位移检测法):
    python recover.py result/001 --method centroid
    python recover.py result/001 --method cross_correlation

流程 B (优化求解法):
    python recover.py result/002 --method optimization

运行所有方法:
    python recover.py result/001 --method all
"""

import os
import sys
import json
import csv
import argparse
import warnings
import numpy as np
MPLCONFIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib")
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy import ndimage as ndi
from plot_style import apply_plot_font_bump

apply_plot_font_bump()

from analysis.optimization_common import (
    GridGeometry,
    assert_row_col_indices,
    assert_xy_points,
    build_grid_geometry_from_axes,
)
import config_recover as config

from analysis.displacement_methods import (
    detect_beamlets,
    compute_centroids,
    extract_patches,
    centroid_shifts,
    cross_correlation_shifts,
    recover_from_shifts,
    integrate_slopes,
    remove_plane,
)
from analysis.movement_metadata import (
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

    `wfe_phase_map.npy` remains the preferred explicit 2D truth. When only
    `phase_screens.npy` exists, we only expose a 2D representative when that
    choice is unambiguous. Averaging independent per-frame phase screens would
    not be a physically meaningful simulation truth.
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
        print(f"  WFE simulation truth: Yes ({wfe_truth_source})")
    elif optional_outputs.get('wfe_truth_stack') is not None:
        print(f"  WFE simulation truth: Stack only ({wfe_truth_source})")
    else:
        print("  WFE simulation truth: No")
    print(f"  Ideal image: {'Yes' if optional_outputs.get('ideal_image') is not None else 'No'}")
    print(f"  Ideal centers: {'Yes' if optional_outputs.get('ideal_centers') is not None else 'No'}")
    print(f"  Spot grid indices: {'Yes' if optional_outputs.get('spot_grid_indices') is not None else 'No'}")
    print(f"  Pulse ideal images: {'Yes' if pulse_ideal_images is not None else 'No'}")
    print(f"  Reference ideal image: {'Yes' if reference_ideal_image is not None else 'No'}")
    print(f"  Truth shifts: {'Yes' if optional_outputs.get('gt_shifts') is not None else 'No'}")
    print(f"  Truth displacements: {'Yes' if optional_outputs.get('gt_displacements') is not None else 'No'}")
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
        wfe_truth     : ndarray or None — WFE simulation truth
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
    
    使用与 truth 侧 (_compute_gt_displacements) 完全一致的算法:
      1. grid_count = round(window_size / spacing)
      2. 生成物理坐标
      3. 过滤到探测器视场内
    
    如果 window_size_m 未提供，则回退到简单除法估算。
    """
    if window_size_m is not None:
        # === truth-consistent logic ===
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


def run_displacement_method(
    ref_image,
    tgt_image,
    rows,
    cols,
    method_name,
    *,
    correction_factor_override: float | None = None,
    return_diagnostics: bool = False,
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
    correction_factor = float(
        getattr(config, 'XCORR_CORRECTION_FACTOR', 1.0)
        if correction_factor_override is None
        else correction_factor_override
    )

    # Detect beamlets in reference image
    ref_boxes, ref_centroids = detect_beamlets(ref_image, rows, cols)

    if method_name == 'centroid':
        # Centroid: 需要分别检测两幅图中的质心位置，然后做差
        tgt_boxes, tgt_centroids = detect_beamlets(tgt_image, rows, cols)
        shifts = centroid_shifts(ref_centroids, tgt_centroids)
        diagnostics = None
    elif method_name == 'cross_correlation':
        # Cross-correlation: 必须用相同的 boxes 提取两幅图的 patch，
        # 这样位移才会体现在互相关峰的偏移上。
        # 如果分别检测 boxes，位移已被检测步骤吸收，互相关只能看到残差噪声。
        ref_patches = extract_patches(ref_image, ref_boxes, rows, cols)
        tgt_patches = extract_patches(tgt_image, ref_boxes, rows, cols)  # 关键: 用 ref_boxes!
        xcorr_result = cross_correlation_shifts(
            ref_patches,
            tgt_patches,
            correction_factor=correction_factor,
            ref_boxes=ref_boxes,
            tgt_image=tgt_image,
            search_radius=getattr(config, 'XCORR_SEARCH_RADIUS', 0),
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            shifts = np.asarray(xcorr_result["shifts"], dtype=np.float64)
            diagnostics = {
                key: np.asarray(value)
                for key, value in xcorr_result.items()
                if key != "shifts"
            }
        else:
            shifts = xcorr_result
            diagnostics = None
    else:
        raise ValueError(f"Unknown method: {method_name}. Use 'centroid' or 'cross_correlation'.")

    if return_diagnostics:
        return shifts, diagnostics
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
    from analysis.decomposition import MovementContext

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


def _resolve_accumulated_recovery_method() -> str:
    """Return the configured accumulated-image recovery algorithm."""
    recovery_method = str(
        getattr(config, 'ACCUMULATED_RECOVERY_METHOD', 'decomposition') or 'decomposition'
    ).lower()
    if recovery_method not in {'linearized_residual', 'decomposition'}:
        raise ValueError(
            f"Unsupported ACCUMULATED_RECOVERY_METHOD={recovery_method!r}; "
            "expected 'linearized_residual' or 'decomposition'."
        )
    return recovery_method


def _recover_accumulated_with_linearized_residual(
    data: dict,
    movement: dict,
    rows: int,
    cols: int,
    method_name: str,
) -> dict | None:
    """Recover per-pulse shifts directly from the accumulated residual image."""
    if not bool(getattr(config, 'ENABLE_ACCUMULATED_DECOMPOSITION', True)):
        return None
    if method_name != 'cross_correlation':
        return None

    n_pulses = int(data['params'].get('n_pulses', 1))
    pulse_reference_images, reference_source = _build_pulse_reference_images(data, movement)
    if pulse_reference_images is None or len(pulse_reference_images) != n_pulses or n_pulses <= 1:
        return None

    from analysis.accumulated_linearized import recover_linearized_accumulated_shifts

    print(
        f"  Mode: accumulated linearized residual recovery ({n_pulses} pulses, "
        f"references={reference_source})"
    )
    linearized_result = recover_linearized_accumulated_shifts(
        accumulated_image=np.asarray(data['image'], dtype=np.float64),
        ideal_accumulated_image=(
            None if data.get('ideal_image') is None else np.asarray(data['ideal_image'], dtype=np.float64)
        ),
        pulse_ideal_images=[np.asarray(img, dtype=np.float64) for img in pulse_reference_images],
        rows=int(rows),
        cols=int(cols),
        roi_expand_px=int(getattr(config, 'ACCUMULATED_LINEARIZED_ROI_EXPAND_PX', 10)),
        ridge_relative=float(getattr(config, 'ACCUMULATED_LINEARIZED_RIDGE_RELATIVE', 1e-2)),
        weight_power=float(getattr(config, 'ACCUMULATED_LINEARIZED_WEIGHT_POWER', 0.5)),
        min_weight=float(getattr(config, 'ACCUMULATED_LINEARIZED_MIN_WEIGHT', 0.02)),
        signal_quantile=float(getattr(config, 'ACCUMULATED_LINEARIZED_SIGNAL_QUANTILE', 0.20)),
        smooth_sigma=float(getattr(config, 'ACCUMULATED_LINEARIZED_SMOOTH_SIGMA', 0.0)),
        confidence_condition_scale=float(
            getattr(config, 'ACCUMULATED_LINEARIZED_CONFIDENCE_CONDITION_SCALE', 20.0)
        ),
        spatial_prior_relative=float(
            getattr(config, 'ACCUMULATED_LINEARIZED_SPATIAL_PRIOR_RELATIVE', 0.0)
        ),
        temporal_prior_relative=float(
            getattr(config, 'ACCUMULATED_LINEARIZED_TEMPORAL_PRIOR_RELATIVE', 0.0)
        ),
        confidence_data_power=float(
            getattr(config, 'ACCUMULATED_LINEARIZED_CONFIDENCE_DATA_POWER', 1.5)
        ),
        confidence_data_floor=float(
            getattr(config, 'ACCUMULATED_LINEARIZED_CONFIDENCE_DATA_FLOOR', 0.15)
        ),
    )
    print(
        "    Linearized solver: "
        f"median condition={np.median(linearized_result.condition_numbers):.3f}, "
        f"median residual ratio={np.median(linearized_result.residual_ratios):.3f}, "
        f"spatial prior={linearized_result.spatial_prior_weight:.3e}, "
        f"temporal prior={linearized_result.temporal_prior_weight:.3e}"
    )

    return {
        "all_shifts_arr": np.asarray(linearized_result.all_shifts_px, dtype=np.float64),
        "n_recovered": int(n_pulses),
        "decomposition_method": "linearized_residual",
        "accumulated_recovery_method": "linearized_residual",
        "reference_sequence_source": reference_source,
        "decomposition_images": None,
        "decomposition_confidences": None,
        "all_shift_confidence_arr": np.asarray(linearized_result.all_shift_confidence, dtype=np.float64),
        "linearized_condition_numbers": np.asarray(linearized_result.condition_numbers, dtype=np.float64),
        "linearized_residual_ratios": np.asarray(linearized_result.residual_ratios, dtype=np.float64),
        "linearized_median_data_scale": float(linearized_result.median_data_scale),
        "linearized_spatial_prior_weight": float(linearized_result.spatial_prior_weight),
        "linearized_temporal_prior_weight": float(linearized_result.temporal_prior_weight),
    }


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

    from analysis.decomposition import decompose_accumulated_frame

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
    all_shift_confidence = []
    decomposition_images = []
    decomposition_confidences = []
    decomp_correction_factor = float(
        getattr(config, 'XCORR_DECOMPOSITION_CORRECTION_FACTOR', getattr(config, 'XCORR_CORRECTION_FACTOR', 1.0))
    )
    for pulse in decomposed:
        ideal_pulse = pulse_reference_images[pulse.index]
        shifts, diagnostics = run_displacement_method(
            ideal_pulse,
            pulse.extracted_image,
            rows,
            cols,
            method_name,
            correction_factor_override=decomp_correction_factor if method_name == 'cross_correlation' else None,
            return_diagnostics=(method_name == 'cross_correlation'),
        )
        all_shifts.append(shifts)
        if diagnostics is not None and diagnostics.get('confidence') is not None:
            all_shift_confidence.append(np.asarray(diagnostics['confidence'], dtype=np.float64))
        decomposition_images.append(np.asarray(pulse.extracted_image, dtype=np.float64))
        decomposition_confidences.append(float(pulse.confidence))
        print(f"    Pulse {pulse.index}: decomposition confidence={pulse.confidence:.4f}")

    if not all_shifts:
        raise RuntimeError("Accumulated decomposition produced no per-pulse recoveries.")

    return {
        "all_shifts_arr": np.asarray(all_shifts, dtype=np.float64),
        "n_recovered": len(all_shifts),
        "decomposition_method": decomp_method,
        "accumulated_recovery_method": "decomposition",
        "reference_sequence_source": reference_source,
        "decomposition_images": np.asarray(decomposition_images, dtype=np.float64),
        "decomposition_confidences": decomposition_confidences,
        "all_shift_confidence_arr": (
            np.asarray(all_shift_confidence, dtype=np.float64)
            if len(all_shift_confidence) == len(all_shifts)
            else None
        ),
        "linearized_condition_numbers": None,
        "linearized_residual_ratios": None,
    }


def _recover_per_frame_sequence(
    *,
    data: dict,
    rows: int,
    cols: int,
    method_name: str,
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
    all_shift_confidence = []
    for i in range(n_pulses):
        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.npy")
        ideal_path = os.path.join(frames_dir, f"frame_{i:03d}_ideal.npy")
        if not os.path.exists(frame_path) or not os.path.exists(ideal_path):
            print(f"    Warning: missing frame {i}, stopping at {i} frames")
            break

        frame = np.load(frame_path)
        ideal = np.load(ideal_path)
        shifts, diagnostics = run_displacement_method(
            ideal,
            frame,
            rows,
            cols,
            method_name,
            return_diagnostics=(method_name == 'cross_correlation'),
        )
        all_shifts.append(shifts)
        if diagnostics is not None and diagnostics.get('confidence') is not None:
            all_shift_confidence.append(np.asarray(diagnostics['confidence'], dtype=np.float64))

    if not all_shifts:
        raise RuntimeError("No valid per-frame pairs were recovered from frames/ directory.")

    all_shifts_arr = np.asarray(all_shifts, dtype=np.float64)
    return {
        "all_shifts_arr": all_shifts_arr,
        "n_recovered": int(all_shifts_arr.shape[0]),
        "all_shift_confidence_arr": (
            np.asarray(all_shift_confidence, dtype=np.float64)
            if len(all_shift_confidence) == len(all_shifts)
            else None
        ),
    }


def _select_representative_shifts(all_shifts_arr: np.ndarray, aberr_mode: str) -> tuple[np.ndarray, str]:
    """Reduce a shift sequence to a representative map for legacy aggregate outputs."""
    if str(aberr_mode or 'none') == 'per_pulse':
        print("  [per_pulse mode] keep per-pulse sequence; frame 0 is used only for aggregate maps")
        return np.asarray(all_shifts_arr[0], dtype=np.float64), "frame0"

    return np.mean(np.asarray(all_shifts_arr, dtype=np.float64), axis=0), "mean"


def _select_representative_confidence(
    all_confidence_arr: np.ndarray | None,
    aberr_mode: str,
) -> np.ndarray | None:
    """Reduce a confidence sequence to the aggregate map aligned with representative shifts."""
    if all_confidence_arr is None:
        return None
    if str(aberr_mode or 'none') == 'per_pulse':
        return np.asarray(all_confidence_arr[0], dtype=np.float64)
    return np.mean(np.asarray(all_confidence_arr, dtype=np.float64), axis=0)


def _recover_sequence_phase_products(
    all_shifts_arr: np.ndarray,
    *,
    pixel_size: float,
    z_prop: float,
    pitch: float,
    wavelength: float,
    all_confidence_arr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover per-frame phase/displacement products from a shift sequence."""
    all_phase_list = []
    all_dx_list = []
    all_dy_list = []
    shifts_seq = np.asarray(all_shifts_arr, dtype=np.float64)
    confidence_seq = None if all_confidence_arr is None else np.asarray(all_confidence_arr, dtype=np.float64)
    for frame_idx, shifts in enumerate(shifts_seq):
        frame_confidence = None if confidence_seq is None else confidence_seq[frame_idx]
        frame_recovery = recover_from_shifts(
            shifts,
            pixel_size=pixel_size,
            z_prop=z_prop,
            spacing=pitch,
            wavelength=wavelength,
            integration_method=config.INTEGRATION_METHOD,
            measurement_confidence=frame_confidence,
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


def _is_dpc_only_output() -> bool:
    """Return whether outputs should be limited to DPC displacement products."""
    return bool(getattr(config, 'OUTPUT_DPC_ONLY', False))


def _run_per_frame_primary_recovery(
    *,
    data: dict,
    rows: int,
    cols: int,
    method_name: str,
    aberr_mode: str,
) -> dict:
    """Run the saved per-frame oracle recovery path."""
    print(f"  Mode: per-frame evaluation recovery ({data['params'].get('n_pulses', 1)} frames)")
    per_frame_result = _recover_per_frame_sequence(
        data=data,
        rows=rows,
        cols=cols,
        method_name=method_name,
    )
    all_shifts_arr = per_frame_result["all_shifts_arr"]
    n_recovered = int(per_frame_result["n_recovered"])
    print(f"  Recovered {n_recovered} frames")
    shifts_px, _ = _select_representative_shifts(all_shifts_arr, aberr_mode)
    return {
        "shifts_px": shifts_px,
        "all_shifts_arr": all_shifts_arr,
        "shift_confidence": _select_representative_confidence(
            per_frame_result.get("all_shift_confidence_arr"),
            aberr_mode,
        ),
        "all_shift_confidence_arr": per_frame_result.get("all_shift_confidence_arr"),
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
    accumulated_recovery_method = _resolve_accumulated_recovery_method()
    if accumulated_recovery_method == 'linearized_residual':
        try:
            linearized_result = _recover_accumulated_with_linearized_residual(
                data=data,
                movement=movement,
                rows=rows,
                cols=cols,
                method_name=method_name,
            )
        except Exception as exc:
            print(f"  Warning: accumulated linearized residual recovery failed: {exc}")
            linearized_result = None

        if linearized_result is not None:
            all_shifts_arr = linearized_result["all_shifts_arr"]
            shifts_px, _ = _select_representative_shifts(all_shifts_arr, aberr_mode)
            return {
                "shifts_px": shifts_px,
                "all_shifts_arr": all_shifts_arr,
                "shift_confidence": _select_representative_confidence(
                    linearized_result.get("all_shift_confidence_arr"),
                    aberr_mode,
                ),
                "all_shift_confidence_arr": linearized_result.get("all_shift_confidence_arr"),
                "n_recovered": int(linearized_result["n_recovered"]),
                "recovery_mode": "accumulated_linearized_residual",
                "decomposition_method": linearized_result["decomposition_method"],
                "accumulated_recovery_method": linearized_result.get("accumulated_recovery_method"),
                "reference_sequence_source": linearized_result.get("reference_sequence_source"),
                "decomposition_confidences": None,
                "decomposition_images": None,
                "linearized_condition_numbers": linearized_result.get("linearized_condition_numbers"),
                "linearized_residual_ratios": linearized_result.get("linearized_residual_ratios"),
                "linearized_median_data_scale": linearized_result.get("linearized_median_data_scale"),
                "linearized_spatial_prior_weight": linearized_result.get("linearized_spatial_prior_weight"),
                "linearized_temporal_prior_weight": linearized_result.get("linearized_temporal_prior_weight"),
            }

        print("  Falling back to legacy accumulated decomposition recovery")

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
        "shift_confidence": _select_representative_confidence(
            decomp_result.get("all_shift_confidence_arr"),
            aberr_mode,
        ),
        "all_shift_confidence_arr": decomp_result.get("all_shift_confidence_arr"),
        "n_recovered": int(decomp_result["n_recovered"]),
        "recovery_mode": "accumulated_decomposition",
        "decomposition_method": decomp_result["decomposition_method"],
        "accumulated_recovery_method": decomp_result.get("accumulated_recovery_method"),
        "reference_sequence_source": decomp_result.get("reference_sequence_source"),
        "decomposition_confidences": decomp_result["decomposition_confidences"],
        "decomposition_images": decomp_result["decomposition_images"],
        "linearized_condition_numbers": decomp_result.get("linearized_condition_numbers"),
        "linearized_residual_ratios": decomp_result.get("linearized_residual_ratios"),
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

    shifts_px, diagnostics = run_displacement_method(
        ref_image,
        data['image'],
        rows,
        cols,
        method_name,
        return_diagnostics=(method_name == 'cross_correlation'),
    )
    return {
        "shifts_px": shifts_px,
        "all_shifts_arr": None,
        "shift_confidence": None if diagnostics is None else diagnostics.get("confidence"),
        "all_shift_confidence_arr": None,
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
        )
        eval_all_shifts_arr = per_frame_eval["all_shifts_arr"]
        baseline = {
            "mode": "per_frame_oracle",
            "all_shifts_px": eval_all_shifts_arr,
            "all_shift_confidence": per_frame_eval.get("all_shift_confidence_arr"),
            "all_phase_rad": None,
            "all_displacement_x_m": eval_all_shifts_arr[..., 0] * float(pixel_size),
            "all_displacement_y_m": eval_all_shifts_arr[..., 1] * float(pixel_size),
            "n_frames_recovered": int(per_frame_eval["n_recovered"]),
        }
        if not _is_dpc_only_output():
            eval_all_phase_rad, eval_all_dx_m, eval_all_dy_m = _recover_sequence_phase_products(
                eval_all_shifts_arr,
                pixel_size=pixel_size,
                z_prop=z_prop,
                pitch=pitch,
                wavelength=wavelength,
                all_confidence_arr=per_frame_eval.get("all_shift_confidence_arr"),
            )
            baseline["all_phase_rad"] = eval_all_phase_rad
            baseline["all_displacement_x_m"] = eval_all_dx_m
            baseline["all_displacement_y_m"] = eval_all_dy_m
        print("  Evaluation baseline: built per-frame oracle DPC sequence from saved aberrated frames")
        return baseline
    except Exception as exc:
        print(f"  Warning: per-frame evaluation baseline unavailable: {exc}")
        return None


def _recover_primary_phase_products(
    *,
    shifts_px: np.ndarray,
    all_shifts_arr: np.ndarray | None,
    shift_confidence: np.ndarray | None,
    all_shift_confidence_arr: np.ndarray | None,
    pixel_size: float,
    z_prop: float,
    pitch: float,
    wavelength: float,
) -> dict:
    """Recover aggregate and per-frame phase/displacement products from shift fields."""
    if _is_dpc_only_output():
        shifts_arr = np.asarray(shifts_px, dtype=np.float64)
        all_shifts_seq = None if all_shifts_arr is None else np.asarray(all_shifts_arr, dtype=np.float64)
        all_displacement_x_m = None
        all_displacement_y_m = None
        if all_shifts_seq is not None:
            all_displacement_x_m = all_shifts_seq[..., 0] * float(pixel_size)
            all_displacement_y_m = all_shifts_seq[..., 1] * float(pixel_size)
            print(f"  Per-frame DPC displacement maps: {all_displacement_x_m.shape}")

        return {
            "recovery": {
                "displacement_x_m": shifts_arr[..., 0] * float(pixel_size),
                "displacement_y_m": shifts_arr[..., 1] * float(pixel_size),
                "phase_rad": None,
            },
            "phase_no_tilt_rad": None,
            "all_phase_rad": None,
            "all_displacement_x_m": all_displacement_x_m,
            "all_displacement_y_m": all_displacement_y_m,
            "shift_confidence": shift_confidence,
            "all_shift_confidence_arr": all_shift_confidence_arr,
        }

    recovery = recover_from_shifts(
        shifts_px,
        pixel_size=pixel_size,
        z_prop=z_prop,
        spacing=pitch,
        wavelength=wavelength,
        integration_method=config.INTEGRATION_METHOD,
        measurement_confidence=shift_confidence,
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
            all_confidence_arr=all_shift_confidence_arr,
        )
        print(f"  Per-frame phase maps: {all_phase_rad.shape}")

    return {
        "recovery": recovery,
        "phase_no_tilt_rad": phase_no_tilt,
        "all_phase_rad": all_phase_rad,
        "all_displacement_x_m": all_displacement_x_m,
        "all_displacement_y_m": all_displacement_y_m,
        "shift_confidence": shift_confidence,
        "all_shift_confidence_arr": all_shift_confidence_arr,
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
        "shift_confidence": phase_products["shift_confidence"],
        "all_shift_confidence": phase_products["all_shift_confidence_arr"],
        "all_phase_rad": phase_products["all_phase_rad"],
        "all_displacement_x_m": phase_products["all_displacement_x_m"],
        "all_displacement_y_m": phase_products["all_displacement_y_m"],
        "n_frames_recovered": int(primary_recovery["n_recovered"]),
        "recovery_mode": primary_recovery["recovery_mode"],
        "decomposition_method": primary_recovery.get("decomposition_method"),
        "accumulated_recovery_method": primary_recovery.get("accumulated_recovery_method"),
        "reference_sequence_source": primary_recovery.get("reference_sequence_source"),
        "decomposition_confidences": primary_recovery.get("decomposition_confidences"),
        "decomposition_images": primary_recovery.get("decomposition_images"),
        "linearized_condition_numbers": primary_recovery.get("linearized_condition_numbers"),
        "linearized_residual_ratios": primary_recovery.get("linearized_residual_ratios"),
        "linearized_median_data_scale": primary_recovery.get("linearized_median_data_scale"),
        "linearized_spatial_prior_weight": primary_recovery.get("linearized_spatial_prior_weight"),
        "linearized_temporal_prior_weight": primary_recovery.get("linearized_temporal_prior_weight"),
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
    gt_disp = data.get('gt_displacements')
    if (not _is_dpc_only_output()) and gt_disp is not None and 'phase_rad' in gt_disp:
        phase_metrics = compute_phase_metrics_from_gt_displacements(result, gt_disp)
        if phase_metrics is not None:
            result['phase_metrics_from_gt_displacements'] = phase_metrics
            representative_metrics = phase_metrics.get('representative_metrics')
            if representative_metrics is not None:
                result['metrics'] = representative_metrics
                result['metrics_source'] = 'gt_displacements.phase_rad'

    if (not _is_dpc_only_output()) and 'metrics' not in result:
        wfe_truth = data.get('wfe_truth')
        if wfe_truth is not None:
            result['metrics'] = compute_metrics(result['phase_no_tilt_rad'], wfe_truth, pitch, pixel_size)
            result['metrics_source'] = data.get('wfe_truth_source') or 'wfe_truth'

    if gt_disp is None:
        return

    all_shifts_arr = result.get('all_shifts_px')
    if all_shifts_arr is not None:
        disp_metrics = compute_displacement_metrics_perframe(all_shifts_arr, gt_disp)
    else:
        disp_metrics = compute_displacement_metrics(result['shifts_px'], gt_disp)
    result['disp_metrics'] = disp_metrics
    result['gt_displacements'] = gt_disp

    dpc_spot_products = compute_dpc_per_spot_products(
        result,
        gt_disp,
        params=data.get('params'),
        wfe_truth_stack=data.get('wfe_truth_stack'),
    )
    if dpc_spot_products is not None:
        result['dpc_spot_products'] = dpc_spot_products
        result['dpc_spot_metrics'] = dpc_spot_products['summary']
        summary = dpc_spot_products['summary']
        print("\n  Per-spot DPC diagnostics:")
        print(f"    Median sample abs err: {(summary.get('median_sample_abs_error_px') or 0.0):.4f} px")
        print(f"    P95 sample abs err:    {(summary.get('p95_sample_abs_error_px') or 0.0):.4f} px")
        print(
            f"    Median |parallel| err: {(summary.get('median_sample_abs_parallel_error_px') or 0.0):.4f} px"
        )
        print(
            f"    Median |perp| err:     {(summary.get('median_sample_abs_perpendicular_error_px') or 0.0):.4f} px"
        )
        print(
            "    Aux median rel err: "
            f"{(summary.get('median_mean_relative_error_pct') or 0.0):.2f}% "
            f"(true displacement >= {(summary.get('relative_error_truth_threshold_px') or 0.0):.2f} px)"
        )
        print(f"    Median scale(vec): {(summary.get('median_scale_vec') or 0.0):.4f}")
        print(f"    Median angle err:  {(summary.get('median_mean_angle_deg') or 0.0):.3f} deg")

    evaluation_baseline = result.get('evaluation_baseline')
    if evaluation_baseline is not None and evaluation_baseline.get('all_shifts_px') is not None:
        evaluation_baseline['disp_metrics'] = compute_displacement_metrics_perframe(
            evaluation_baseline['all_shifts_px'],
            gt_disp,
        )
        if (
            (not _is_dpc_only_output())
            and evaluation_baseline.get('all_phase_rad') is not None
            and 'phase_rad' in gt_disp
        ):
            evaluation_baseline['phase_metrics'] = compute_phase_metrics_perframe(
                evaluation_baseline['all_phase_rad'],
                gt_disp['phase_rad'],
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
        pixel_size=pixel_size,
        z_prop=z_prop,
        pitch=pitch,
        wavelength=wavelength,
    )
    phase_products = _recover_primary_phase_products(
        shifts_px=shifts_px,
        all_shifts_arr=all_shifts_arr,
        shift_confidence=primary_recovery.get("shift_confidence"),
        all_shift_confidence_arr=primary_recovery.get("all_shift_confidence_arr"),
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
    """计算恢复相位与 simulation truth 之间的误差指标。"""
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
    print(f"\n  Metrics vs simulation truth:")
    print(f"    RMSE: {rmse:.4f} rad")
    print(f"    PV:   {pv:.4f} rad")
    print(f"    Corr: {corr:.4f}")

    return metrics


def compute_phase_grid_metrics(phase_recovered_rad, truth_phase_rad) -> dict:
    """Compare recovered and truth phase maps that already live on the Hartmann grid."""
    recovered = np.asarray(phase_recovered_rad, dtype=np.float64)
    truth = np.asarray(truth_phase_rad, dtype=np.float64)
    min_rows = min(recovered.shape[0], truth.shape[0])
    min_cols = min(recovered.shape[1], truth.shape[1])
    recovered = recovered[:min_rows, :min_cols]
    truth = truth[:min_rows, :min_cols]

    grid_x, grid_y = np.meshgrid(np.arange(min_cols), np.arange(min_rows))
    recovered_no_tilt = remove_plane(recovered, grid_x, grid_y)
    truth_no_tilt = remove_plane(truth, grid_x, grid_y)
    error = recovered_no_tilt - truth_no_tilt

    recovered_flat = recovered_no_tilt.ravel()
    truth_flat = truth_no_tilt.ravel()
    if np.std(recovered_flat) > 1e-12 and np.std(truth_flat) > 1e-12:
        corr = float(np.corrcoef(recovered_flat, truth_flat)[0, 1])
    else:
        corr = 0.0

    return {
        "rmse_rad": float(np.sqrt(np.mean(error**2))),
        "pv_rad": float(np.ptp(error)),
        "correlation": corr,
        "grid_used": f"{min_rows}x{min_cols}",
    }


def compute_phase_metrics_perframe(all_phase_rad, truth_phase_all_rad) -> dict:
    """Compute per-frame phase metrics against grid-sampled truth phase."""
    recovered = np.asarray(all_phase_rad, dtype=np.float64)
    truth = np.asarray(truth_phase_all_rad, dtype=np.float64)
    n_frames = min(recovered.shape[0], truth.shape[0])
    per_frame_metrics = [
        compute_phase_grid_metrics(recovered[frame_idx], truth[frame_idx])
        for frame_idx in range(n_frames)
    ]
    rmse_values = [item["rmse_rad"] for item in per_frame_metrics]
    corr_values = [item["correlation"] for item in per_frame_metrics]
    return {
        "mode": "per_frame",
        "n_frames": int(n_frames),
        "mean_rmse_rad": float(np.mean(rmse_values)) if rmse_values else None,
        "max_rmse_rad": float(np.max(rmse_values)) if rmse_values else None,
        "mean_correlation": float(np.mean(corr_values)) if corr_values else None,
        "per_frame_metrics": per_frame_metrics,
    }


def compute_phase_metrics_from_gt_displacements(result: dict, gt_disp: dict) -> dict | None:
    """Build phase metrics using `gt_displacements.phase_rad` as grid truth."""
    if gt_disp is None or 'phase_rad' not in gt_disp:
        return None

    truth_phase_all = np.asarray(gt_disp['phase_rad'], dtype=np.float64)
    if truth_phase_all.ndim == 2:
        representative = compute_phase_grid_metrics(result['phase_no_tilt_rad'], truth_phase_all)
        print("\n  Phase metrics vs truth displacement phase:")
        print(f"    RMSE: {representative['rmse_rad']:.4f} rad")
        print(f"    PV:   {representative['pv_rad']:.4f} rad")
        print(f"    Corr: {representative['correlation']:.4f}")
        return {
            "mode": "single",
            "representative_metrics": representative,
        }

    if truth_phase_all.ndim != 3:
        return None

    all_phase_rad = result.get('all_phase_rad')
    if all_phase_rad is not None:
        per_frame = compute_phase_metrics_perframe(all_phase_rad, truth_phase_all)
        representative_index = 0 if result.get('aberr_mode') == 'per_pulse' else None
        if representative_index is None:
            truth_representative = np.mean(truth_phase_all[:per_frame["n_frames"]], axis=0)
            representative = compute_phase_grid_metrics(result['phase_no_tilt_rad'], truth_representative)
        else:
            representative = per_frame["per_frame_metrics"][representative_index]
        per_frame["representative_metrics"] = representative
        print("\n  Phase metrics vs truth displacement phase:")
        print(f"    Representative RMSE: {representative['rmse_rad']:.4f} rad")
        print(f"    Mean per-frame RMSE: {per_frame['mean_rmse_rad']:.4f} rad")
        print(f"    Mean per-frame Corr: {per_frame['mean_correlation']:.4f}")
        return per_frame

    representative_truth = truth_phase_all[0] if result.get('aberr_mode') == 'per_pulse' else np.mean(truth_phase_all, axis=0)
    representative = compute_phase_grid_metrics(result['phase_no_tilt_rad'], representative_truth)
    print("\n  Phase metrics vs truth displacement phase:")
    print(f"    RMSE: {representative['rmse_rad']:.4f} rad")
    print(f"    PV:   {representative['pv_rad']:.4f} rad")
    print(f"    Corr: {representative['correlation']:.4f}")
    return {
        "mode": "representative",
        "representative_metrics": representative,
    }


def compute_displacement_metrics(shifts_px, truth_disp):
    """
    比较恢复的位移与 simulation truth 位移。
    
    shifts_px: (rows, cols, 2) — 恢复的位移 [dx, dy]
    truth_disp: dict from gt_displacements.npz
    """
    # True displacement: average over frames because the accumulated image encodes all frames.
    truth_displacement_x_px = np.mean(truth_disp['displacement_x_px'], axis=0)  # (rows, cols)
    truth_displacement_y_px = np.mean(truth_disp['displacement_y_px'], axis=0)  # (rows, cols)
    truth_rows, truth_cols = truth_displacement_x_px.shape
    
    # 恢复的位移
    rec_rows, rec_cols, _ = shifts_px.shape
    
    # 网格大小可能不完全一致，取交集
    min_rows = min(rec_rows, truth_rows)
    min_cols = min(rec_cols, truth_cols)
    
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
    
    print(f"\n  Displacement metrics vs simulation truth:")
    print(f"    Grid: {min_rows}×{min_cols} (rec={rec_rows}×{rec_cols}, truth={truth_rows}×{truth_cols})")
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
    frame_index: int,
    frame_stats: dict,
    corr_dx: float,
    corr_dy: float,
) -> dict:
    """Package one frame's displacement RMSE/correlation summary."""
    error_total_px = np.asarray(frame_stats["displacement_error_total_px"], dtype=np.float64)
    mean_err_px = float(np.mean(error_total_px))
    median_err_px = float(np.median(error_total_px))
    p90_err_px = float(np.percentile(error_total_px, 90))
    p95_err_px = float(np.percentile(error_total_px, 95))
    max_err_px = float(np.max(error_total_px))
    bias_dx_px = float(np.mean(frame_stats["displacement_error_x_px"]))
    bias_dy_px = float(np.mean(frame_stats["displacement_error_y_px"]))

    truth_dx = np.asarray(frame_stats["truth_displacement_x_px"], dtype=np.float64)
    truth_dy = np.asarray(frame_stats["truth_displacement_y_px"], dtype=np.float64)
    recovered_dx = np.asarray(frame_stats["recovered_displacement_x_px"], dtype=np.float64)
    recovered_dy = np.asarray(frame_stats["recovered_displacement_y_px"], dtype=np.float64)
    truth_mag = np.sqrt(truth_dx**2 + truth_dy**2)
    recovered_mag = np.sqrt(recovered_dx**2 + recovered_dy**2)
    truth_energy = float(np.sum(truth_mag**2))
    scale_vec = (
        float(np.sum(truth_mag * recovered_mag) / truth_energy)
        if truth_energy > 1e-12
        else None
    )

    valid = (truth_mag > 1e-4) & (recovered_mag > 1e-4)
    if np.any(valid):
        cosine = (
            (truth_dx[valid] * recovered_dx[valid] + truth_dy[valid] * recovered_dy[valid]) /
            (truth_mag[valid] * recovered_mag[valid])
        )
        cosine = np.clip(cosine, -1.0, 1.0)
        mean_cosine = float(np.mean(cosine))
        mean_angle_deg = float(np.mean(np.degrees(np.arccos(cosine))))
    else:
        mean_cosine = None
        mean_angle_deg = None

    return {
        'frame_index': int(frame_index),
        'rmse_dx_px': frame_stats["rmse_dx"],
        'rmse_dy_px': frame_stats["rmse_dy"],
        'rmse_total_px': frame_stats["rmse_total"],
        'mean_err_px': mean_err_px,
        'median_err_px': median_err_px,
        'p90_err_px': p90_err_px,
        'p95_err_px': p95_err_px,
        'max_err_px': max_err_px,
        'bias_dx_px': bias_dx_px,
        'bias_dy_px': bias_dy_px,
        'scale_vec': scale_vec,
        'mean_cosine': mean_cosine,
        'mean_angle_deg': mean_angle_deg,
        'corr_dx': corr_dx,
        'corr_dy': corr_dy,
    }


def compute_displacement_metrics_perframe(all_shifts_px, truth_disp):
    """
    Per-frame 位移对比。
    
    all_shifts_px: (n_frames, rows, cols, 2) — 每帧恢复的位移
    truth_disp: dict from gt_displacements.npz, displacement_x_px shape (n_truth, rows, cols)
    """
    truth_displacement_x_px = truth_disp['displacement_x_px']  # (n_truth, rows, cols)
    truth_displacement_y_px = truth_disp['displacement_y_px']
    
    n_rec = all_shifts_px.shape[0]
    n_truth = truth_displacement_x_px.shape[0]
    n_compare = min(n_rec, n_truth)
    
    rec_rows, rec_cols = all_shifts_px.shape[1], all_shifts_px.shape[2]
    truth_rows, truth_cols = truth_displacement_x_px.shape[1], truth_displacement_x_px.shape[2]
    min_r = min(rec_rows, truth_rows)
    min_c = min(rec_cols, truth_cols)

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
                frame_index=len(per_frame_metrics),
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
        "median_frame_rmse_px": float(np.median(per_frame_rmse)) if per_frame_rmse else None,
        "p95_frame_rmse_px": float(np.percentile(per_frame_rmse, 95)) if per_frame_rmse else None,
        "best_frame_index": int(np.argmin(per_frame_rmse)) if per_frame_rmse else None,
        "best_frame_rmse_px": float(np.min(per_frame_rmse)) if per_frame_rmse else None,
        "worst_frame_index": int(np.argmax(per_frame_rmse)) if per_frame_rmse else None,
        "worst_frame_rmse_px": float(np.max(per_frame_rmse)) if per_frame_rmse else None,
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


def _safe_finite_stat(array, reducer) -> float | None:
    """Return a scalar stat over finite values, or None when no finite values exist."""
    arr = np.asarray(array, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(reducer(finite))


def _nanmean_over_frames(stack: np.ndarray) -> np.ndarray:
    """Compute a NaN-safe mean across the leading frame axis."""
    arr = np.asarray(stack, dtype=np.float64)
    valid = np.isfinite(arr)
    counts = np.sum(valid, axis=0)
    summed = np.nansum(arr, axis=0)
    out = np.full(arr.shape[1:], np.nan, dtype=np.float64)
    mask = counts > 0
    out[mask] = summed[mask] / counts[mask]
    return out


def _nanpercentile_over_frames(stack: np.ndarray, q: float) -> np.ndarray:
    """Compute a NaN-safe percentile across the leading frame axis."""
    arr = np.asarray(stack, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(all='ignore'):
            return np.nanpercentile(arr, q, axis=0)


def _nanmedian_over_frames(stack: np.ndarray) -> np.ndarray:
    """Compute a NaN-safe median across the leading frame axis."""
    return _nanpercentile_over_frames(stack, 50.0)


def _first_finite_float(*values) -> float | None:
    """Return the first finite float from candidates, else None."""
    for value in values:
        if value is None:
            continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(out):
            return out
    return None


def _compute_truth_phase_context_maps(
    *,
    truth_disp: dict,
    params: dict | None,
    wfe_truth_stack: np.ndarray | None,
    n_frames: int,
    min_rows: int,
    min_cols: int,
    mean_truth_mag_px: np.ndarray,
) -> dict:
    """Derive per-spot truth-side slope / local phase context maps."""
    context: dict[str, np.ndarray | None] = {
        "truth_equivalent_slope_mrad": None,
        "truth_phase_gradient_mag_rad_per_m": None,
        "truth_phase_local_rms_rad": None,
        "truth_phase_local_p95_rms_rad": None,
    }

    z_prop_m = _first_finite_float(
        truth_disp.get("z_prop_m"),
        None if params is None else params.get("z_prop_m"),
    )
    pixel_size_m = _first_finite_float(
        truth_disp.get("pixel_size_m"),
        None if params is None else params.get("pixel_size_m"),
    )
    wavelength_m = _first_finite_float(
        truth_disp.get("wavelength_m"),
        None if params is None else params.get("wavelength_m"),
    )

    if z_prop_m not in (None, 0.0) and pixel_size_m not in (None, 0.0):
        truth_equivalent_slope_rad = np.asarray(mean_truth_mag_px, dtype=np.float64) * float(pixel_size_m) / float(z_prop_m)
        context["truth_equivalent_slope_mrad"] = 1e3 * truth_equivalent_slope_rad
        if wavelength_m not in (None, 0.0):
            opd_scale = float(wavelength_m) / (2.0 * np.pi)
            if opd_scale > 0.0:
                context["truth_phase_gradient_mag_rad_per_m"] = (
                    np.asarray(mean_truth_mag_px, dtype=np.float64) * float(pixel_size_m) / (opd_scale * float(z_prop_m))
                )

    if wfe_truth_stack is None or params is None:
        return context

    window_size_m = _first_finite_float(params.get("window_size_m"))
    grid_points = params.get("grid_points")
    hole_diameter_m = _first_finite_float(params.get("hole_diameter_m"))
    centers_x_m = truth_disp.get("centers_x_m")
    centers_y_m = truth_disp.get("centers_y_m")
    if (
        window_size_m in (None, 0.0)
        or hole_diameter_m in (None, 0.0)
        or grid_points in (None, 0)
        or centers_x_m is None
        or centers_y_m is None
    ):
        return context

    stack = np.asarray(wfe_truth_stack)
    if stack.ndim == 2:
        stack = np.asarray(stack[np.newaxis, ...], dtype=np.float64)
    if stack.ndim != 3 or stack.shape[0] <= 0:
        return context

    n_use = min(int(n_frames), int(stack.shape[0]))
    if n_use <= 0:
        return context

    dx = float(window_size_m) / int(grid_points)
    x0_m = -0.5 * float(window_size_m)
    y0_m = -0.5 * float(window_size_m)
    hole_radius_px = max(1, int(np.ceil(float(hole_diameter_m) / (2.0 * dx))))
    centers_x = np.asarray(centers_x_m, dtype=np.float64)[:min_cols]
    centers_y = np.asarray(centers_y_m, dtype=np.float64)[:min_rows]

    spot_specs: list[tuple[int, int, int, int, int, int, np.ndarray]] = []
    for row_idx, yc in enumerate(centers_y):
        for col_idx, xc in enumerate(centers_x):
            cx_idx = int(np.clip(np.round((float(xc) - x0_m) / dx), 0, stack.shape[2] - 1))
            cy_idx = int(np.clip(np.round((float(yc) - y0_m) / dx), 0, stack.shape[1] - 1))
            py0 = max(0, cy_idx - hole_radius_px)
            py1 = min(stack.shape[1], cy_idx + hole_radius_px + 1)
            px0 = max(0, cx_idx - hole_radius_px)
            px1 = min(stack.shape[2], cx_idx + hole_radius_px + 1)
            yy, xx = np.ogrid[py0:py1, px0:px1]
            aperture_mask = ((yy - cy_idx) ** 2 + (xx - cx_idx) ** 2) <= (hole_radius_px ** 2)
            spot_specs.append((row_idx, col_idx, py0, py1, px0, px1, aperture_mask))

    phase_rms_stack = np.full((n_use, min_rows, min_cols), np.nan, dtype=np.float64)
    for frame_idx in range(n_use):
        phase_frame = np.asarray(stack[frame_idx], dtype=np.float64)
        for row_idx, col_idx, py0, py1, px0, px1, aperture_mask in spot_specs:
            patch = phase_frame[py0:py1, px0:px1]
            if patch.shape != aperture_mask.shape:
                valid_rows = min(patch.shape[0], aperture_mask.shape[0])
                valid_cols = min(patch.shape[1], aperture_mask.shape[1])
                patch = patch[:valid_rows, :valid_cols]
                mask = aperture_mask[:valid_rows, :valid_cols]
            else:
                mask = aperture_mask
            vals = patch[mask]
            if vals.size == 0:
                continue
            vals = np.asarray(vals, dtype=np.float64)
            phase_rms_stack[frame_idx, row_idx, col_idx] = float(np.sqrt(np.mean((vals - np.mean(vals)) ** 2)))

    context["truth_phase_local_rms_rad"] = _nanmedian_over_frames(phase_rms_stack)
    context["truth_phase_local_p95_rms_rad"] = _nanpercentile_over_frames(phase_rms_stack, 95.0)
    return context


def _build_spot_trust_products(spot_products: dict) -> dict:
    """Build per-spot trust maps and summary statistics from sample-level DPC errors."""
    truth_mag_px = np.asarray(spot_products["truth_mag_px"], dtype=np.float64)
    error_mag_px = np.asarray(spot_products["error_mag_px"], dtype=np.float64)
    angle_stack_deg = np.asarray(spot_products["angle_stack_deg"], dtype=np.float64)
    abs_parallel_error_px = np.abs(np.asarray(spot_products["parallel_error_px"], dtype=np.float64))
    abs_perpendicular_error_px = np.abs(np.asarray(spot_products["perpendicular_error_px"], dtype=np.float64))

    median_truth_mag_px = _nanmedian_over_frames(truth_mag_px)
    median_abs_magnitude_error_px = _nanmedian_over_frames(error_mag_px)
    p95_abs_magnitude_error_px = _nanpercentile_over_frames(error_mag_px, 95.0)
    median_abs_direction_error_deg = _nanmedian_over_frames(angle_stack_deg)
    p95_abs_direction_error_deg = _nanpercentile_over_frames(angle_stack_deg, 95.0)
    median_abs_parallel_error_px = _nanmedian_over_frames(abs_parallel_error_px)
    median_abs_perpendicular_error_px = _nanmedian_over_frames(abs_perpendicular_error_px)

    low_signal_threshold_px = float(getattr(config, "DPC_DIRECTION_LOW_SIGNAL_TRUTH_PX", 0.05))
    mag_good_px = float(getattr(config, "DPC_TRUST_MAG_ERROR_GOOD_PX", 0.01))
    mag_warn_px = float(getattr(config, "DPC_TRUST_MAG_ERROR_WARN_PX", 0.02))
    dir_good_deg = float(getattr(config, "DPC_TRUST_DIRECTION_GOOD_DEG", 5.0))
    dir_warn_deg = float(getattr(config, "DPC_TRUST_DIRECTION_WARN_DEG", 10.0))

    low_signal_mask = np.asarray(median_truth_mag_px < low_signal_threshold_px, dtype=bool)
    trust_code = np.zeros(median_truth_mag_px.shape, dtype=np.int8)
    trust_code[low_signal_mask] = -1

    valid_signal = (~low_signal_mask) & np.isfinite(median_abs_magnitude_error_px) & np.isfinite(median_abs_direction_error_deg)
    trusted_mask = (
        valid_signal
        & (median_abs_magnitude_error_px <= mag_good_px)
        & (median_abs_direction_error_deg <= dir_good_deg)
    )
    review_mask = (
        valid_signal
        & (~trusted_mask)
        & (median_abs_magnitude_error_px <= mag_warn_px)
        & (median_abs_direction_error_deg <= dir_warn_deg)
    )
    suspicious_mask = valid_signal & (~trusted_mask) & (~review_mask)

    trust_code[trusted_mask] = 2
    trust_code[review_mask] = 1
    trust_code[suspicious_mask] = 0

    trust_counts = {
        "n_low_signal": int(np.count_nonzero(low_signal_mask)),
        "n_trusted": int(np.count_nonzero(trusted_mask)),
        "n_review": int(np.count_nonzero(review_mask)),
        "n_suspicious": int(np.count_nonzero(suspicious_mask)),
    }
    total_spots = int(median_truth_mag_px.size)
    trust_summary = {
        "low_signal_truth_threshold_px": low_signal_threshold_px,
        "mag_error_good_px": mag_good_px,
        "mag_error_warn_px": mag_warn_px,
        "direction_error_good_deg": dir_good_deg,
        "direction_error_warn_deg": dir_warn_deg,
        "total_spots": total_spots,
        **trust_counts,
        "fraction_low_signal": trust_counts["n_low_signal"] / total_spots if total_spots > 0 else None,
        "fraction_trusted": trust_counts["n_trusted"] / total_spots if total_spots > 0 else None,
        "fraction_review": trust_counts["n_review"] / total_spots if total_spots > 0 else None,
        "fraction_suspicious": trust_counts["n_suspicious"] / total_spots if total_spots > 0 else None,
    }

    return {
        "median_truth_mag_px": median_truth_mag_px,
        "median_abs_magnitude_error_px": median_abs_magnitude_error_px,
        "p95_abs_magnitude_error_px": p95_abs_magnitude_error_px,
        "median_abs_direction_error_deg": median_abs_direction_error_deg,
        "p95_abs_direction_error_deg": p95_abs_direction_error_deg,
        "median_abs_parallel_error_px": median_abs_parallel_error_px,
        "median_abs_perpendicular_error_px": median_abs_perpendicular_error_px,
        "low_signal_mask": low_signal_mask,
        "trust_code": trust_code,
        "summary": trust_summary,
    }


def _spot_scale_through_origin(
    recovered_stack_px: np.ndarray,
    truth_stack_px: np.ndarray,
    *,
    min_truth_energy: float = 1e-8,
) -> np.ndarray:
    """Estimate a per-spot multiplicative scale using a zero-intercept fit over frames."""
    recovered = np.asarray(recovered_stack_px, dtype=np.float64)
    truth = np.asarray(truth_stack_px, dtype=np.float64)
    numerator = np.sum(recovered * truth, axis=0)
    denominator = np.sum(truth**2, axis=0)
    scale = np.full(denominator.shape, np.nan, dtype=np.float64)
    valid = denominator > float(min_truth_energy)
    scale[valid] = numerator[valid] / denominator[valid]
    return scale


def _spot_correlation_over_frames(
    recovered_stack_px: np.ndarray,
    truth_stack_px: np.ndarray,
) -> np.ndarray:
    """Estimate per-spot temporal correlation between recovered and truth DPC."""
    recovered = np.asarray(recovered_stack_px, dtype=np.float64)
    truth = np.asarray(truth_stack_px, dtype=np.float64)
    _, rows, cols = recovered.shape
    corr = np.full((rows, cols), np.nan, dtype=np.float64)
    for row_idx in range(rows):
        for col_idx in range(cols):
            rec_series = recovered[:, row_idx, col_idx]
            truth_series = truth[:, row_idx, col_idx]
            if np.std(rec_series) <= 1e-12 or np.std(truth_series) <= 1e-12:
                continue
            corr[row_idx, col_idx] = float(np.corrcoef(truth_series, rec_series)[0, 1])
    return corr


def _spot_vector_direction_metrics(
    recovered_x_px: np.ndarray,
    recovered_y_px: np.ndarray,
    truth_x_px: np.ndarray,
    truth_y_px: np.ndarray,
    *,
    min_vector_norm: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-spot mean cosine similarity and mean angular error across frames."""
    rec_x = np.asarray(recovered_x_px, dtype=np.float64)
    rec_y = np.asarray(recovered_y_px, dtype=np.float64)
    truth_x = np.asarray(truth_x_px, dtype=np.float64)
    truth_y = np.asarray(truth_y_px, dtype=np.float64)

    rec_norm = np.sqrt(rec_x**2 + rec_y**2)
    truth_norm = np.sqrt(truth_x**2 + truth_y**2)
    valid = (rec_norm > float(min_vector_norm)) & (truth_norm > float(min_vector_norm))

    cosine = np.full(rec_norm.shape, np.nan, dtype=np.float64)
    cosine[valid] = (
        (rec_x[valid] * truth_x[valid] + rec_y[valid] * truth_y[valid]) /
        (rec_norm[valid] * truth_norm[valid])
    )
    cosine = np.clip(cosine, -1.0, 1.0, out=cosine, where=np.isfinite(cosine))

    angle_deg = np.full(rec_norm.shape, np.nan, dtype=np.float64)
    angle_deg[np.isfinite(cosine)] = np.degrees(np.arccos(cosine[np.isfinite(cosine)]))
    return _nanmean_over_frames(cosine), _nanmean_over_frames(angle_deg)


def _spot_parallel_perpendicular_metrics(
    error_x_px: np.ndarray,
    error_y_px: np.ndarray,
    truth_x_px: np.ndarray,
    truth_y_px: np.ndarray,
    *,
    min_truth_norm: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Project vector errors onto truth-aligned parallel/perpendicular directions."""
    err_x = np.asarray(error_x_px, dtype=np.float64)
    err_y = np.asarray(error_y_px, dtype=np.float64)
    truth_x = np.asarray(truth_x_px, dtype=np.float64)
    truth_y = np.asarray(truth_y_px, dtype=np.float64)
    truth_norm = np.sqrt(truth_x**2 + truth_y**2)

    parallel_error_px = np.full(truth_norm.shape, np.nan, dtype=np.float64)
    perpendicular_error_px = np.full(truth_norm.shape, np.nan, dtype=np.float64)
    valid = truth_norm > float(min_truth_norm)
    parallel_error_px[valid] = (
        err_x[valid] * truth_x[valid] + err_y[valid] * truth_y[valid]
    ) / truth_norm[valid]
    perpendicular_error_px[valid] = (
        -err_x[valid] * truth_y[valid] + err_y[valid] * truth_x[valid]
    ) / truth_norm[valid]

    abs_parallel_error_px = np.abs(parallel_error_px)
    abs_perpendicular_error_px = np.abs(perpendicular_error_px)
    return {
        "parallel_error_px": parallel_error_px,
        "perpendicular_error_px": perpendicular_error_px,
        "mean_parallel_error_px": _nanmean_over_frames(parallel_error_px),
        "mean_abs_parallel_error_px": _nanmean_over_frames(abs_parallel_error_px),
        "p95_abs_parallel_error_px": _nanpercentile_over_frames(abs_parallel_error_px, 95.0),
        "mean_perpendicular_error_px": _nanmean_over_frames(perpendicular_error_px),
        "mean_abs_perpendicular_error_px": _nanmean_over_frames(abs_perpendicular_error_px),
        "p95_abs_perpendicular_error_px": _nanpercentile_over_frames(abs_perpendicular_error_px, 95.0),
    }


def _build_dpc_error_vs_truth_bins(
    *,
    truth_mag_px: np.ndarray,
    error_mag_px: np.ndarray,
    parallel_error_px: np.ndarray,
    perpendicular_error_px: np.ndarray,
    angle_deg: np.ndarray,
    bin_edges_px: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, np.inf),
) -> list[dict]:
    """Build frame-spot sample statistics binned by true displacement magnitude."""
    truth = np.asarray(truth_mag_px, dtype=np.float64).ravel()
    abs_error = np.asarray(error_mag_px, dtype=np.float64).ravel()
    abs_parallel = np.abs(np.asarray(parallel_error_px, dtype=np.float64).ravel())
    abs_perpendicular = np.abs(np.asarray(perpendicular_error_px, dtype=np.float64).ravel())
    angle = np.asarray(angle_deg, dtype=np.float64).ravel()

    finite_truth = np.isfinite(truth)
    total_count = int(np.count_nonzero(finite_truth))
    rows: list[dict] = []
    for lo, hi in zip(bin_edges_px[:-1], bin_edges_px[1:]):
        in_bin = finite_truth & (truth >= float(lo))
        label_hi = "inf"
        if np.isfinite(hi):
            in_bin &= truth < float(hi)
            label_hi = f"{float(hi):.2f}"
        count = int(np.count_nonzero(in_bin))
        truth_vals = truth[in_bin]
        err_vals = abs_error[in_bin & np.isfinite(abs_error)]
        par_vals = abs_parallel[in_bin & np.isfinite(abs_parallel)]
        perp_vals = abs_perpendicular[in_bin & np.isfinite(abs_perpendicular)]
        angle_vals = angle[in_bin & np.isfinite(angle)]
        rows.append(
            {
                "bin_label": f"[{float(lo):.2f}, {label_hi}) px",
                "bin_lo_px": float(lo),
                "bin_hi_px": None if not np.isfinite(hi) else float(hi),
                "count": count,
                "fraction_pct": None if total_count == 0 else 100.0 * count / total_count,
                "median_truth_mag_px": _safe_finite_stat(truth_vals, np.median),
                "median_abs_error_px": _safe_finite_stat(err_vals, np.median),
                "p95_abs_error_px": _safe_finite_stat(err_vals, lambda x: np.percentile(x, 95)),
                "median_abs_parallel_error_px": _safe_finite_stat(par_vals, np.median),
                "p95_abs_parallel_error_px": _safe_finite_stat(par_vals, lambda x: np.percentile(x, 95)),
                "median_abs_perpendicular_error_px": _safe_finite_stat(perp_vals, np.median),
                "p95_abs_perpendicular_error_px": _safe_finite_stat(perp_vals, lambda x: np.percentile(x, 95)),
                "median_angle_deg": _safe_finite_stat(angle_vals, np.median),
                "p95_angle_deg": _safe_finite_stat(angle_vals, lambda x: np.percentile(x, 95)),
            }
        )
    return rows


def _crop_optional_spot_map(map_2d, min_rows: int, min_cols: int) -> np.ndarray | None:
    """Crop an optional 2D per-spot diagnostic map to the comparison grid."""
    if map_2d is None:
        return None
    arr = np.asarray(map_2d, dtype=np.float64)
    if arr.ndim != 2:
        return None
    return arr[:min_rows, :min_cols]


def _array_value_or_none(array_2d, row_idx: int, col_idx: int) -> float | None:
    """Return a finite scalar from a per-spot map, or None when unavailable."""
    if array_2d is None:
        return None
    value = float(array_2d[row_idx, col_idx])
    return value if np.isfinite(value) else None


def compute_dpc_per_spot_products(
    result: dict,
    truth_disp: dict,
    *,
    params: dict | None = None,
    wfe_truth_stack: np.ndarray | None = None,
) -> dict | None:
    """Build per-spot DPC error, bias, scale, and direction diagnostics."""
    all_shifts_rec = result.get('all_shifts_px')
    if all_shifts_rec is None or truth_disp is None:
        return None
    if 'displacement_x_px' not in truth_disp or 'displacement_y_px' not in truth_disp:
        return None

    recovered = np.asarray(all_shifts_rec, dtype=np.float64)
    truth_x_all = np.asarray(truth_disp['displacement_x_px'], dtype=np.float64)
    truth_y_all = np.asarray(truth_disp['displacement_y_px'], dtype=np.float64)

    if recovered.ndim != 4 or recovered.shape[-1] != 2:
        return None
    if truth_x_all.ndim != 3 or truth_y_all.ndim != 3:
        return None

    n_frames = min(recovered.shape[0], truth_x_all.shape[0], truth_y_all.shape[0])
    min_rows = min(recovered.shape[1], truth_x_all.shape[1], truth_y_all.shape[1])
    min_cols = min(recovered.shape[2], truth_x_all.shape[2], truth_y_all.shape[2])
    if n_frames <= 0 or min_rows <= 0 or min_cols <= 0:
        return None

    recovered_x_px = recovered[:n_frames, :min_rows, :min_cols, 0]
    recovered_y_px = recovered[:n_frames, :min_rows, :min_cols, 1]
    truth_x_px = truth_x_all[:n_frames, :min_rows, :min_cols]
    truth_y_px = truth_y_all[:n_frames, :min_rows, :min_cols]

    error_x_px = recovered_x_px - truth_x_px
    error_y_px = recovered_y_px - truth_y_px
    error_mag_px = np.sqrt(error_x_px**2 + error_y_px**2)
    truth_mag_px = np.sqrt(truth_x_px**2 + truth_y_px**2)

    bias_x_px = np.mean(error_x_px, axis=0)
    bias_y_px = np.mean(error_y_px, axis=0)
    mean_abs_error_mag_px = np.mean(error_mag_px, axis=0)
    max_error_mag_px = np.max(error_mag_px, axis=0)
    rmse_x_px = np.sqrt(np.mean(error_x_px**2, axis=0))
    rmse_y_px = np.sqrt(np.mean(error_y_px**2, axis=0))
    rmse_mag_px = np.sqrt(np.mean(error_mag_px**2, axis=0))
    mean_truth_mag_px = np.mean(truth_mag_px, axis=0)

    scale_x = _spot_scale_through_origin(recovered_x_px, truth_x_px)
    scale_y = _spot_scale_through_origin(recovered_y_px, truth_y_px)
    scale_vec = _spot_scale_through_origin(
        np.sqrt(recovered_x_px**2 + recovered_y_px**2),
        np.sqrt(truth_x_px**2 + truth_y_px**2),
    )
    corr_x = _spot_correlation_over_frames(recovered_x_px, truth_x_px)
    corr_y = _spot_correlation_over_frames(recovered_y_px, truth_y_px)
    mean_cosine, mean_angle_deg = _spot_vector_direction_metrics(
        recovered_x_px,
        recovered_y_px,
        truth_x_px,
        truth_y_px,
    )
    parallel_perpendicular = _spot_parallel_perpendicular_metrics(
        error_x_px,
        error_y_px,
        truth_x_px,
        truth_y_px,
    )

    relative_truth_threshold_px = float(getattr(config, 'DPC_RELATIVE_ERROR_MIN_TRUTH_PX', 0.2))
    relative_error = np.full(error_mag_px.shape, np.nan, dtype=np.float64)
    valid_relative = truth_mag_px >= relative_truth_threshold_px
    relative_error[valid_relative] = error_mag_px[valid_relative] / truth_mag_px[valid_relative]
    mean_relative_error_pct = 100.0 * _nanmean_over_frames(relative_error)
    p95_relative_error_pct = 100.0 * _nanpercentile_over_frames(relative_error, 95.0)

    rec_norm = np.sqrt(recovered_x_px**2 + recovered_y_px**2)
    angle_stack_deg = np.full(rec_norm.shape, np.nan, dtype=np.float64)
    truth_norm_valid = truth_mag_px > 1e-8
    rec_norm_valid = rec_norm > 1e-8
    valid_angle = truth_norm_valid & rec_norm_valid
    if np.any(valid_angle):
        cosine = (
            (recovered_x_px[valid_angle] * truth_x_px[valid_angle]) +
            (recovered_y_px[valid_angle] * truth_y_px[valid_angle])
        ) / (rec_norm[valid_angle] * truth_mag_px[valid_angle])
        cosine = np.clip(cosine, -1.0, 1.0)
        angle_stack_deg[valid_angle] = np.degrees(np.arccos(cosine))
    truth_bins = _build_dpc_error_vs_truth_bins(
        truth_mag_px=truth_mag_px,
        error_mag_px=error_mag_px,
        parallel_error_px=parallel_perpendicular['parallel_error_px'],
        perpendicular_error_px=parallel_perpendicular['perpendicular_error_px'],
        angle_deg=angle_stack_deg,
    )

    confidence_map = None
    all_shift_confidence = result.get('all_shift_confidence')
    if all_shift_confidence is not None:
        conf_arr = np.asarray(all_shift_confidence, dtype=np.float64)
        if conf_arr.ndim == 3:
            confidence_map = np.mean(conf_arr[:n_frames, :min_rows, :min_cols], axis=0)
        elif conf_arr.ndim == 2:
            confidence_map = conf_arr[:min_rows, :min_cols]

    condition_numbers = _crop_optional_spot_map(
        result.get('linearized_condition_numbers'),
        min_rows,
        min_cols,
    )
    residual_ratios = _crop_optional_spot_map(
        result.get('linearized_residual_ratios'),
        min_rows,
        min_cols,
    )

    valid_rmse = np.isfinite(rmse_mag_px)
    worst_spots: list[dict] = []
    if np.any(valid_rmse):
        worst_coords = np.argwhere(valid_rmse)
        worst_values = rmse_mag_px[valid_rmse]
        worst_order = np.argsort(worst_values)[::-1][:10]
        for order_idx in worst_order:
            row_idx, col_idx = worst_coords[order_idx]
            worst_spots.append(
                {
                    "row": int(row_idx),
                    "col": int(col_idx),
                    "rmse_mag_px": float(rmse_mag_px[row_idx, col_idx]),
                    "bias_x_px": float(bias_x_px[row_idx, col_idx]),
                    "bias_y_px": float(bias_y_px[row_idx, col_idx]),
                    "mean_truth_mag_px": float(mean_truth_mag_px[row_idx, col_idx]),
                    "mean_abs_error_mag_px": float(mean_abs_error_mag_px[row_idx, col_idx]),
                    "mean_abs_parallel_error_px": _array_value_or_none(
                        parallel_perpendicular['mean_abs_parallel_error_px'],
                        row_idx,
                        col_idx,
                    ),
                    "mean_abs_perpendicular_error_px": _array_value_or_none(
                        parallel_perpendicular['mean_abs_perpendicular_error_px'],
                        row_idx,
                        col_idx,
                    ),
                    "mean_relative_error_pct": _array_value_or_none(
                        mean_relative_error_pct,
                        row_idx,
                        col_idx,
                    ),
                    "scale_vec": _array_value_or_none(scale_vec, row_idx, col_idx),
                    "mean_angle_deg": _array_value_or_none(mean_angle_deg, row_idx, col_idx),
                    "mean_confidence": _array_value_or_none(confidence_map, row_idx, col_idx),
                    "condition_number": _array_value_or_none(condition_numbers, row_idx, col_idx),
                    "residual_ratio": _array_value_or_none(residual_ratios, row_idx, col_idx),
                }
            )

    summary = {
        "mode": "per_spot_over_frames",
        "n_frames": int(n_frames),
        "grid_used": f"{min_rows}x{min_cols}",
        "median_spot_rmse_px": _safe_finite_stat(rmse_mag_px, np.median),
        "p95_spot_rmse_px": _safe_finite_stat(rmse_mag_px, lambda x: np.percentile(x, 95)),
        "max_spot_rmse_px": _safe_finite_stat(rmse_mag_px, np.max),
        "median_mean_abs_error_px": _safe_finite_stat(mean_abs_error_mag_px, np.median),
        "p95_mean_abs_error_px": _safe_finite_stat(mean_abs_error_mag_px, lambda x: np.percentile(x, 95)),
        "median_sample_abs_error_px": _safe_finite_stat(error_mag_px, np.median),
        "p95_sample_abs_error_px": _safe_finite_stat(error_mag_px, lambda x: np.percentile(x, 95)),
        "median_mean_abs_parallel_error_px": _safe_finite_stat(
            parallel_perpendicular['mean_abs_parallel_error_px'],
            np.median,
        ),
        "p95_mean_abs_parallel_error_px": _safe_finite_stat(
            parallel_perpendicular['mean_abs_parallel_error_px'],
            lambda x: np.percentile(x, 95),
        ),
        "median_sample_abs_parallel_error_px": _safe_finite_stat(
            np.abs(parallel_perpendicular['parallel_error_px']),
            np.median,
        ),
        "p95_sample_abs_parallel_error_px": _safe_finite_stat(
            np.abs(parallel_perpendicular['parallel_error_px']),
            lambda x: np.percentile(x, 95),
        ),
        "median_mean_abs_perpendicular_error_px": _safe_finite_stat(
            parallel_perpendicular['mean_abs_perpendicular_error_px'],
            np.median,
        ),
        "p95_mean_abs_perpendicular_error_px": _safe_finite_stat(
            parallel_perpendicular['mean_abs_perpendicular_error_px'],
            lambda x: np.percentile(x, 95),
        ),
        "median_sample_abs_perpendicular_error_px": _safe_finite_stat(
            np.abs(parallel_perpendicular['perpendicular_error_px']),
            np.median,
        ),
        "p95_sample_abs_perpendicular_error_px": _safe_finite_stat(
            np.abs(parallel_perpendicular['perpendicular_error_px']),
            lambda x: np.percentile(x, 95),
        ),
        "median_abs_bias_x_px": _safe_finite_stat(np.abs(bias_x_px), np.median),
        "median_abs_bias_y_px": _safe_finite_stat(np.abs(bias_y_px), np.median),
        "relative_error_truth_threshold_px": relative_truth_threshold_px,
        "median_truth_mag_px": _safe_finite_stat(mean_truth_mag_px, np.median),
        "p95_truth_mag_px": _safe_finite_stat(truth_mag_px, lambda x: np.percentile(x, 95)),
        "max_truth_mag_px": _safe_finite_stat(truth_mag_px, np.max),
        "median_mean_relative_error_pct": _safe_finite_stat(mean_relative_error_pct, np.median),
        "p95_mean_relative_error_pct": _safe_finite_stat(
            mean_relative_error_pct,
            lambda x: np.percentile(x, 95),
        ),
        "max_sample_abs_error_px": _safe_finite_stat(error_mag_px, np.max),
        "fraction_sample_truth_lt_0p05_px": _safe_finite_stat(
            truth_mag_px,
            lambda x: np.mean(x < 0.05),
        ),
        "fraction_sample_truth_lt_0p10_px": _safe_finite_stat(
            truth_mag_px,
            lambda x: np.mean(x < 0.10),
        ),
        "fraction_sample_truth_lt_0p20_px": _safe_finite_stat(
            truth_mag_px,
            lambda x: np.mean(x < 0.20),
        ),
        "fraction_sample_abs_error_lt_0p01_px": _safe_finite_stat(
            error_mag_px,
            lambda x: np.mean(x < 0.01),
        ),
        "fraction_sample_abs_error_lt_0p02_px": _safe_finite_stat(
            error_mag_px,
            lambda x: np.mean(x < 0.02),
        ),
        "fraction_sample_abs_error_lt_0p05_px": _safe_finite_stat(
            error_mag_px,
            lambda x: np.mean(x < 0.05),
        ),
        "median_signal_to_error_ratio": (
            None
            if _safe_finite_stat(truth_mag_px, np.median) in (None, 0.0) or
            _safe_finite_stat(error_mag_px, np.median) in (None, 0.0)
            else _safe_finite_stat(truth_mag_px, np.median) / _safe_finite_stat(error_mag_px, np.median)
        ),
        "median_scale_x": _safe_finite_stat(scale_x, np.median),
        "median_scale_y": _safe_finite_stat(scale_y, np.median),
        "median_scale_vec": _safe_finite_stat(scale_vec, np.median),
        "p05_scale_vec": _safe_finite_stat(scale_vec, lambda x: np.percentile(x, 5)),
        "p95_scale_vec": _safe_finite_stat(scale_vec, lambda x: np.percentile(x, 95)),
        "median_mean_angle_deg": _safe_finite_stat(mean_angle_deg, np.median),
        "p95_mean_angle_deg": _safe_finite_stat(mean_angle_deg, lambda x: np.percentile(x, 95)),
        "median_sample_angle_deg": _safe_finite_stat(angle_stack_deg, np.median),
        "p95_sample_angle_deg": _safe_finite_stat(angle_stack_deg, lambda x: np.percentile(x, 95)),
        "fraction_sample_angle_lt_5_deg": _safe_finite_stat(angle_stack_deg, lambda x: np.mean(x < 5.0)),
        "fraction_sample_angle_lt_10_deg": _safe_finite_stat(angle_stack_deg, lambda x: np.mean(x < 10.0)),
        "fraction_sample_angle_lt_15_deg": _safe_finite_stat(angle_stack_deg, lambda x: np.mean(x < 15.0)),
        "median_mean_cosine": _safe_finite_stat(mean_cosine, np.median),
        "median_confidence": _safe_finite_stat(confidence_map, np.median),
        "median_condition_number": _safe_finite_stat(condition_numbers, np.median),
        "median_residual_ratio": _safe_finite_stat(residual_ratios, np.median),
        "worst_spots": worst_spots,
    }

    truth_phase_context = _compute_truth_phase_context_maps(
        truth_disp=truth_disp,
        params=params,
        wfe_truth_stack=wfe_truth_stack,
        n_frames=n_frames,
        min_rows=min_rows,
        min_cols=min_cols,
        mean_truth_mag_px=mean_truth_mag_px,
    )
    trust_products = _build_spot_trust_products(
        {
            "truth_mag_px": truth_mag_px,
            "error_mag_px": error_mag_px,
            "angle_stack_deg": angle_stack_deg,
            "parallel_error_px": parallel_perpendicular["parallel_error_px"],
            "perpendicular_error_px": parallel_perpendicular["perpendicular_error_px"],
        }
    )
    summary.update(
        {
            "median_abs_magnitude_error_px": _safe_finite_stat(
                trust_products["median_abs_magnitude_error_px"],
                np.median,
            ),
            "p95_abs_magnitude_error_px": _safe_finite_stat(
                trust_products["p95_abs_magnitude_error_px"],
                lambda x: np.percentile(x, 95),
            ) if np.isfinite(np.asarray(trust_products["p95_abs_magnitude_error_px"], dtype=np.float64)).any() else None,
            "median_abs_direction_error_deg": _safe_finite_stat(
                trust_products["median_abs_direction_error_deg"],
                np.median,
            ),
            "p95_abs_direction_error_deg": _safe_finite_stat(
                trust_products["p95_abs_direction_error_deg"],
                lambda x: np.percentile(x, 95),
            ) if np.isfinite(np.asarray(trust_products["p95_abs_direction_error_deg"], dtype=np.float64)).any() else None,
            "n_low_signal_spots": trust_products["summary"]["n_low_signal"],
            "n_trusted_spots": trust_products["summary"]["n_trusted"],
            "n_review_spots": trust_products["summary"]["n_review"],
            "n_suspicious_spots": trust_products["summary"]["n_suspicious"],
        }
    )
    if truth_phase_context.get("truth_phase_local_rms_rad") is not None:
        summary["median_truth_phase_local_rms_rad"] = _safe_finite_stat(
            truth_phase_context["truth_phase_local_rms_rad"],
            np.median,
        )
    if truth_phase_context.get("truth_equivalent_slope_mrad") is not None:
        summary["median_truth_equivalent_slope_mrad"] = _safe_finite_stat(
            truth_phase_context["truth_equivalent_slope_mrad"],
            np.median,
        )

    return {
        "recovered_x_px": recovered_x_px,
        "recovered_y_px": recovered_y_px,
        "truth_x_px": truth_x_px,
        "truth_y_px": truth_y_px,
        "error_x_px": error_x_px,
        "error_y_px": error_y_px,
        "error_mag_px": error_mag_px,
        "truth_mag_px": truth_mag_px,
        "bias_x_px": bias_x_px,
        "bias_y_px": bias_y_px,
        "mean_abs_error_mag_px": mean_abs_error_mag_px,
        "max_error_mag_px": max_error_mag_px,
        "mean_truth_mag_px": mean_truth_mag_px,
        "relative_error_pct": 100.0 * relative_error,
        "mean_relative_error_pct": mean_relative_error_pct,
        "p95_relative_error_pct": p95_relative_error_pct,
        "parallel_error_px": parallel_perpendicular['parallel_error_px'],
        "perpendicular_error_px": parallel_perpendicular['perpendicular_error_px'],
        "mean_parallel_error_px": parallel_perpendicular['mean_parallel_error_px'],
        "mean_abs_parallel_error_px": parallel_perpendicular['mean_abs_parallel_error_px'],
        "p95_abs_parallel_error_px": parallel_perpendicular['p95_abs_parallel_error_px'],
        "mean_perpendicular_error_px": parallel_perpendicular['mean_perpendicular_error_px'],
        "mean_abs_perpendicular_error_px": parallel_perpendicular['mean_abs_perpendicular_error_px'],
        "p95_abs_perpendicular_error_px": parallel_perpendicular['p95_abs_perpendicular_error_px'],
        "angle_stack_deg": angle_stack_deg,
        "rmse_x_px": rmse_x_px,
        "rmse_y_px": rmse_y_px,
        "rmse_mag_px": rmse_mag_px,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "scale_vec": scale_vec,
        "corr_x": corr_x,
        "corr_y": corr_y,
        "mean_cosine": mean_cosine,
        "mean_angle_deg": mean_angle_deg,
        "mean_confidence": confidence_map,
        "condition_numbers": condition_numbers,
        "residual_ratios": residual_ratios,
        "truth_bins": truth_bins,
        "truth_phase_context": truth_phase_context,
        "trust_products": trust_products,
        "summary": summary,
    }


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
    """Return canonical filenames for per-frame displacement stacks."""
    return {
        "canonical_x": f"all_displacement_x_{method}.npy",
        "canonical_y": f"all_displacement_y_{method}.npy",
    }


def _save_per_frame_displacement_arrays(
    output_dir: str,
    method: str,
    displacement_x_stack_m: np.ndarray,
    displacement_y_stack_m: np.ndarray,
) -> None:
    """Persist per-frame displacement stacks under canonical filenames."""
    filenames = _per_frame_displacement_filenames(method)
    _save_named_array(output_dir, filenames["canonical_x"], displacement_x_stack_m)
    _save_named_array(output_dir, filenames["canonical_y"], displacement_y_stack_m)


def _save_recovery_arrays(result: dict, output_dir: str) -> None:
    """Persist recovered arrays and sequence products to disk."""
    method = result['method']
    _save_named_array(output_dir, f"displacement_x_{method}.npy", result['displacement_x_m'])
    _save_named_array(output_dir, f"displacement_y_{method}.npy", result['displacement_y_m'])
    if result.get('phase_no_tilt_rad') is not None:
        _save_named_array(output_dir, f"phase_recovered_{method}.npy", result['phase_no_tilt_rad'])
    if result.get('shift_confidence') is not None:
        _save_named_array(output_dir, f"shift_confidence_{method}.npy", result['shift_confidence'])

    if result.get('all_shifts_px') is not None:
        _save_named_array(output_dir, f"all_shifts_{method}.npy", result['all_shifts_px'])
        print(f"  Per-frame shifts: {result['all_shifts_px'].shape}")
    if result.get('all_shift_confidence') is not None:
        _save_named_array(output_dir, f"all_shift_confidence_{method}.npy", result['all_shift_confidence'])
    if result.get('all_phase_rad') is not None:
        _save_named_array(output_dir, f"all_phase_{method}.npy", result['all_phase_rad'])
        print(f"  Per-frame phase: {result['all_phase_rad'].shape}")

    if result.get('linearized_condition_numbers') is not None:
        _save_named_array(
            output_dir,
            f"linearized_condition_numbers_{method}.npy",
            result['linearized_condition_numbers'],
        )
    if result.get('linearized_residual_ratios') is not None:
        _save_named_array(
            output_dir,
            f"linearized_residual_ratios_{method}.npy",
            result['linearized_residual_ratios'],
        )

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

    if result.get('accumulated_recovery_method') == 'linearized_residual':
        with open(os.path.join(output_dir, f"accumulated_recovery_{method}.json"), "w") as fp:
            json.dump(
                {
                    "recovery_mode": result.get("recovery_mode"),
                    "accumulated_recovery_method": result.get("accumulated_recovery_method"),
                    "reference_sequence_source": result.get("reference_sequence_source"),
                    "median_condition_number": (
                        None if result.get('linearized_condition_numbers') is None
                        else float(np.median(result['linearized_condition_numbers']))
                    ),
                    "median_residual_ratio": (
                        None if result.get('linearized_residual_ratios') is None
                        else float(np.median(result['linearized_residual_ratios']))
                    ),
                    "median_data_scale": result.get('linearized_median_data_scale'),
                    "spatial_prior_weight": result.get('linearized_spatial_prior_weight'),
                    "temporal_prior_weight": result.get('linearized_temporal_prior_weight'),
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
    if eval_baseline.get('all_shift_confidence') is not None:
        _save_named_array(
            output_dir,
            f"eval_baseline_all_shift_confidence_{method}.npy",
            eval_baseline['all_shift_confidence'],
        )
    if eval_baseline.get('all_phase_rad') is not None:
        _save_named_array(output_dir, f"eval_baseline_all_phase_{method}.npy", eval_baseline['all_phase_rad'])
    with open(os.path.join(output_dir, f"eval_baseline_{method}.json"), "w") as fp:
        json.dump(
            {
                "mode": eval_baseline.get("mode"),
                "n_frames_recovered": int(eval_baseline.get("n_frames_recovered", 0)),
                "disp_metrics": eval_baseline.get("disp_metrics"),
                "phase_metrics": eval_baseline.get("phase_metrics"),
            },
            fp,
            indent=2,
        )


def _build_metrics_payload(result: dict, eval_baseline: dict | None) -> dict | None:
    """Build the metrics JSON payload for a recovery result."""
    metrics_payload = dict(result.get('metrics') or {})
    metrics_payload.update(
        {
            "recovery_mode": result.get('recovery_mode'),
            "primary_recovery_mode": result.get('primary_recovery_mode'),
            "used_per_frame_aberrated_for_recovery": bool(result.get('used_per_frame_aberrated_for_recovery')),
            "reference_sequence_source": result.get('reference_sequence_source'),
            "decomposition_method": result.get('decomposition_method'),
            "accumulated_recovery_method": result.get('accumulated_recovery_method'),
            "metrics_source": result.get('metrics_source') or "dpc_displacement",
            "output_mode": "dpc_only" if _is_dpc_only_output() else "phase_and_dpc",
        }
    )
    if result.get('phase_metrics_from_gt_displacements') is not None:
        metrics_payload["phase_metrics_from_gt_displacements"] = result['phase_metrics_from_gt_displacements']
    if result.get('linearized_condition_numbers') is not None:
        metrics_payload["linearized_median_condition_number"] = float(
            np.median(result['linearized_condition_numbers'])
        )
    if result.get('linearized_residual_ratios') is not None:
        metrics_payload["linearized_median_residual_ratio"] = float(
            np.median(result['linearized_residual_ratios'])
        )
    if result.get('linearized_median_data_scale') is not None:
        metrics_payload["linearized_median_data_scale"] = float(result['linearized_median_data_scale'])
    if result.get('linearized_spatial_prior_weight') is not None:
        metrics_payload["linearized_spatial_prior_weight"] = float(result['linearized_spatial_prior_weight'])
    if result.get('linearized_temporal_prior_weight') is not None:
        metrics_payload["linearized_temporal_prior_weight"] = float(result['linearized_temporal_prior_weight'])
    if result.get('disp_metrics') is not None:
        metrics_payload["displacement_metrics"] = result['disp_metrics']
    if result.get('dpc_spot_metrics') is not None:
        metrics_payload["dpc_spot_metrics"] = result['dpc_spot_metrics']
    if eval_baseline is not None:
        metrics_payload["evaluation_baseline"] = {
            "mode": eval_baseline.get("mode"),
            "n_frames_recovered": int(eval_baseline.get("n_frames_recovered", 0)),
            "disp_metrics": eval_baseline.get("disp_metrics"),
            "phase_metrics": eval_baseline.get("phase_metrics"),
        }
    return metrics_payload


def _save_metrics_payload(output_dir: str, method: str, metrics_payload: dict | None) -> None:
    """Write metrics JSON when metrics are available."""
    if metrics_payload is None:
        return
    with open(os.path.join(output_dir, f"metrics_{method}.json"), "w") as fp:
        json.dump(metrics_payload, fp, indent=2)


def _spot_map_display_limit(array_2d, *, symmetric: bool = False) -> tuple[float | None, float | None]:
    """Return robust display limits for a 2D diagnostic map."""
    if array_2d is None:
        return None, None
    arr = np.asarray(array_2d, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    if symmetric:
        vmax = max(float(np.percentile(np.abs(finite), 98)), 1e-6)
        return -vmax, vmax
    vmin = float(np.percentile(finite, 2))
    vmax = float(np.percentile(finite, 98))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return vmin, vmax


def _render_spot_metric_map(
    *,
    ax,
    data_2d,
    title: str,
    cmap: str,
    symmetric: bool = False,
    colorbar_label: str = '',
) -> None:
    """Render one per-spot diagnostic map or a placeholder when unavailable."""
    if data_2d is None:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, color='gray')
        ax.set_title(title)
        ax.axis('off')
        return

    arr = np.asarray(data_2d, dtype=np.float64)
    vmin, vmax = _spot_map_display_limit(arr, symmetric=symmetric)
    if vmin is None or vmax is None:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, color='gray')
        ax.set_title(title)
        ax.axis('off')
        return

    im = ax.imshow(arr, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)


def _save_dpc_spot_csv(csv_path: str, spot_products: dict) -> None:
    """Write one row per Hartmann spot, sorted by worst RMSE first."""
    rmse_mag_px = np.asarray(spot_products['rmse_mag_px'], dtype=np.float64)
    rows, cols = rmse_mag_px.shape
    coords = [(row_idx, col_idx) for row_idx in range(rows) for col_idx in range(cols)]
    coords.sort(
        key=lambda rc: (
            -np.inf
            if not np.isfinite(rmse_mag_px[rc[0], rc[1]])
            else float(rmse_mag_px[rc[0], rc[1]])
        ),
        reverse=True,
    )

    fieldnames = [
        'row',
        'col',
        'mean_truth_mag_px',
        'mean_abs_error_mag_px',
        'rmse_mag_px',
        'rmse_x_px',
        'rmse_y_px',
        'bias_x_px',
        'bias_y_px',
        'max_error_mag_px',
        'mean_parallel_error_px',
        'mean_abs_parallel_error_px',
        'p95_abs_parallel_error_px',
        'mean_perpendicular_error_px',
        'mean_abs_perpendicular_error_px',
        'p95_abs_perpendicular_error_px',
        'mean_relative_error_pct',
        'p95_relative_error_pct',
        'scale_x',
        'scale_y',
        'scale_vec',
        'corr_x',
        'corr_y',
        'mean_cosine',
        'mean_angle_deg',
        'mean_confidence',
        'condition_number',
        'residual_ratio',
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, col_idx in coords:
            writer.writerow(
                {
                    'row': int(row_idx),
                    'col': int(col_idx),
                    'mean_truth_mag_px': _array_value_or_none(spot_products['mean_truth_mag_px'], row_idx, col_idx),
                    'mean_abs_error_mag_px': _array_value_or_none(spot_products['mean_abs_error_mag_px'], row_idx, col_idx),
                    'rmse_mag_px': _array_value_or_none(spot_products['rmse_mag_px'], row_idx, col_idx),
                    'rmse_x_px': _array_value_or_none(spot_products['rmse_x_px'], row_idx, col_idx),
                    'rmse_y_px': _array_value_or_none(spot_products['rmse_y_px'], row_idx, col_idx),
                    'bias_x_px': _array_value_or_none(spot_products['bias_x_px'], row_idx, col_idx),
                    'bias_y_px': _array_value_or_none(spot_products['bias_y_px'], row_idx, col_idx),
                    'max_error_mag_px': _array_value_or_none(spot_products['max_error_mag_px'], row_idx, col_idx),
                    'mean_parallel_error_px': _array_value_or_none(
                        spot_products['mean_parallel_error_px'],
                        row_idx,
                        col_idx,
                    ),
                    'mean_abs_parallel_error_px': _array_value_or_none(
                        spot_products['mean_abs_parallel_error_px'],
                        row_idx,
                        col_idx,
                    ),
                    'p95_abs_parallel_error_px': _array_value_or_none(
                        spot_products['p95_abs_parallel_error_px'],
                        row_idx,
                        col_idx,
                    ),
                    'mean_perpendicular_error_px': _array_value_or_none(
                        spot_products['mean_perpendicular_error_px'],
                        row_idx,
                        col_idx,
                    ),
                    'mean_abs_perpendicular_error_px': _array_value_or_none(
                        spot_products['mean_abs_perpendicular_error_px'],
                        row_idx,
                        col_idx,
                    ),
                    'p95_abs_perpendicular_error_px': _array_value_or_none(
                        spot_products['p95_abs_perpendicular_error_px'],
                        row_idx,
                        col_idx,
                    ),
                    'mean_relative_error_pct': _array_value_or_none(
                        spot_products['mean_relative_error_pct'],
                        row_idx,
                        col_idx,
                    ),
                    'p95_relative_error_pct': _array_value_or_none(
                        spot_products['p95_relative_error_pct'],
                        row_idx,
                        col_idx,
                    ),
                    'scale_x': _array_value_or_none(spot_products['scale_x'], row_idx, col_idx),
                    'scale_y': _array_value_or_none(spot_products['scale_y'], row_idx, col_idx),
                    'scale_vec': _array_value_or_none(spot_products['scale_vec'], row_idx, col_idx),
                    'corr_x': _array_value_or_none(spot_products['corr_x'], row_idx, col_idx),
                    'corr_y': _array_value_or_none(spot_products['corr_y'], row_idx, col_idx),
                    'mean_cosine': _array_value_or_none(spot_products['mean_cosine'], row_idx, col_idx),
                    'mean_angle_deg': _array_value_or_none(spot_products['mean_angle_deg'], row_idx, col_idx),
                    'mean_confidence': _array_value_or_none(spot_products['mean_confidence'], row_idx, col_idx),
                    'condition_number': _array_value_or_none(spot_products['condition_numbers'], row_idx, col_idx),
                    'residual_ratio': _array_value_or_none(spot_products['residual_ratios'], row_idx, col_idx),
                }
            )


def _trust_code_label(code: int) -> str:
    """Convert an integer trust code to a readable label."""
    return {
        -1: "low_signal",
        0: "suspicious",
        1: "review",
        2: "trusted",
    }.get(int(code), "unknown")


def _render_trust_code_map(*, ax, trust_code: np.ndarray, title: str) -> None:
    """Render a categorical trust-level map."""
    arr = np.asarray(trust_code, dtype=np.int8)
    cmap = mcolors.ListedColormap(["lightgray", "tomato", "gold", "forestgreen"])
    norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    im = ax.imshow(arr, cmap=cmap, norm=norm, origin='lower')
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([-1, 0, 1, 2])
    cbar.set_ticklabels(["low", "bad", "review", "good"])


def _overlay_low_signal_mask(ax, low_signal_mask: np.ndarray | None) -> None:
    """Overlay a subtle gray mask for low-signal spots."""
    if low_signal_mask is None:
        return
    mask = np.asarray(low_signal_mask, dtype=bool)
    if not np.any(mask):
        return
    overlay = np.where(mask, 1.0, np.nan)
    ax.imshow(overlay, cmap='Greys', origin='lower', alpha=0.22, interpolation='nearest')


def _save_dpc_spot_trust_csv(csv_path: str, spot_products: dict) -> None:
    """Write one row per Hartmann spot with trust-oriented summary metrics."""
    trust_products = spot_products.get("trust_products")
    if trust_products is None:
        return

    truth_phase_context = spot_products.get("truth_phase_context") or {}
    median_abs_mag = np.asarray(trust_products["median_abs_magnitude_error_px"], dtype=np.float64)
    rows, cols = median_abs_mag.shape
    coords = [(row_idx, col_idx) for row_idx in range(rows) for col_idx in range(cols)]
    priority = {-1: 3, 0: 0, 1: 1, 2: 2}
    coords.sort(
        key=lambda rc: (
            priority.get(int(trust_products["trust_code"][rc[0], rc[1]]), 9),
            -float(median_abs_mag[rc[0], rc[1]]) if np.isfinite(median_abs_mag[rc[0], rc[1]]) else -np.inf,
        )
    )

    fieldnames = [
        "row",
        "col",
        "median_truth_mag_px",
        "median_abs_magnitude_error_px",
        "p95_abs_magnitude_error_px",
        "median_abs_direction_error_deg",
        "p95_abs_direction_error_deg",
        "median_abs_parallel_error_px",
        "median_abs_perpendicular_error_px",
        "mean_confidence",
        "condition_number",
        "residual_ratio",
        "truth_equivalent_slope_mrad",
        "truth_phase_local_rms_rad",
        "low_signal",
        "trust_code",
        "trust_label",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, col_idx in coords:
            code = int(trust_products["trust_code"][row_idx, col_idx])
            writer.writerow(
                {
                    "row": int(row_idx),
                    "col": int(col_idx),
                    "median_truth_mag_px": _array_value_or_none(trust_products["median_truth_mag_px"], row_idx, col_idx),
                    "median_abs_magnitude_error_px": _array_value_or_none(trust_products["median_abs_magnitude_error_px"], row_idx, col_idx),
                    "p95_abs_magnitude_error_px": _array_value_or_none(trust_products["p95_abs_magnitude_error_px"], row_idx, col_idx),
                    "median_abs_direction_error_deg": _array_value_or_none(trust_products["median_abs_direction_error_deg"], row_idx, col_idx),
                    "p95_abs_direction_error_deg": _array_value_or_none(trust_products["p95_abs_direction_error_deg"], row_idx, col_idx),
                    "median_abs_parallel_error_px": _array_value_or_none(trust_products["median_abs_parallel_error_px"], row_idx, col_idx),
                    "median_abs_perpendicular_error_px": _array_value_or_none(trust_products["median_abs_perpendicular_error_px"], row_idx, col_idx),
                    "mean_confidence": _array_value_or_none(spot_products.get("mean_confidence"), row_idx, col_idx),
                    "condition_number": _array_value_or_none(spot_products.get("condition_numbers"), row_idx, col_idx),
                    "residual_ratio": _array_value_or_none(spot_products.get("residual_ratios"), row_idx, col_idx),
                    "truth_equivalent_slope_mrad": _array_value_or_none(truth_phase_context.get("truth_equivalent_slope_mrad"), row_idx, col_idx),
                    "truth_phase_local_rms_rad": _array_value_or_none(truth_phase_context.get("truth_phase_local_rms_rad"), row_idx, col_idx),
                    "low_signal": bool(trust_products["low_signal_mask"][row_idx, col_idx]),
                    "trust_code": code,
                    "trust_label": _trust_code_label(code),
                }
            )


def _save_dpc_spot_trust_maps(
    *,
    output_dir: str,
    method: str,
    spot_products: dict,
) -> None:
    """Save per-spot trust-oriented heatmaps and supporting arrays."""
    trust_products = spot_products.get("trust_products")
    if trust_products is None:
        return

    _save_named_array(
        output_dir,
        f"dpc_median_abs_magnitude_error_px_per_spot_{method}.npy",
        trust_products["median_abs_magnitude_error_px"],
    )
    _save_named_array(
        output_dir,
        f"dpc_median_abs_direction_error_deg_per_spot_{method}.npy",
        trust_products["median_abs_direction_error_deg"],
    )
    _save_named_array(
        output_dir,
        f"dpc_low_signal_mask_per_spot_{method}.npy",
        trust_products["low_signal_mask"].astype(np.int8),
    )
    _save_named_array(
        output_dir,
        f"dpc_trust_code_per_spot_{method}.npy",
        trust_products["trust_code"],
        allow_preview=False,
    )
    truth_phase_context = spot_products.get("truth_phase_context") or {}
    if truth_phase_context.get("truth_equivalent_slope_mrad") is not None:
        _save_named_array(
            output_dir,
            f"dpc_truth_equivalent_slope_mrad_per_spot_{method}.npy",
            truth_phase_context["truth_equivalent_slope_mrad"],
        )
    if truth_phase_context.get("truth_phase_local_rms_rad") is not None:
        _save_named_array(
            output_dir,
            f"dpc_truth_phase_local_rms_rad_per_spot_{method}.npy",
            truth_phase_context["truth_phase_local_rms_rad"],
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    _render_spot_metric_map(
        ax=axes[0, 0],
        data_2d=trust_products["median_truth_mag_px"],
        title="Spot Median Truth Magnitude",
        cmap="magma",
        colorbar_label="px",
    )
    _render_spot_metric_map(
        ax=axes[0, 1],
        data_2d=trust_products["median_abs_magnitude_error_px"],
        title="Spot Median |Magnitude Error|",
        cmap="magma",
        colorbar_label="px",
    )
    direction_display = np.asarray(trust_products["median_abs_direction_error_deg"], dtype=np.float64).copy()
    direction_display[np.asarray(trust_products["low_signal_mask"], dtype=bool)] = np.nan
    _render_spot_metric_map(
        ax=axes[1, 0],
        data_2d=direction_display,
        title="Spot Median |Direction Error|",
        cmap="viridis",
        colorbar_label="deg",
    )
    _overlay_low_signal_mask(axes[1, 0], trust_products["low_signal_mask"])
    _render_trust_code_map(
        ax=axes[1, 1],
        trust_code=trust_products["trust_code"],
        title="Spot Trust Mask",
    )

    trust_summary = trust_products.get("summary", {})
    plt.suptitle(
        "Per-Spot Trust Maps\n"
        f"low-signal<{trust_summary.get('low_signal_truth_threshold_px', 0.0):.3f} px, "
        f"trusted={trust_summary.get('n_trusted', 0)}, "
        f"review={trust_summary.get('n_review', 0)}, "
        f"suspicious={trust_summary.get('n_suspicious', 0)}",
        fontsize=13,
    )
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"dpc_spot_trust_maps_{method}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()


def _save_dpc_spot_trust_statistics(
    *,
    stat_dir: str,
    method: str,
    spot_products: dict | None,
) -> None:
    """Save signal-vs-reliability scatter plots at per-spot level."""
    if spot_products is None or spot_products.get("trust_products") is None:
        return

    trust_products = spot_products["trust_products"]
    truth = np.asarray(trust_products["median_truth_mag_px"], dtype=np.float64).ravel()
    mag_err = np.asarray(trust_products["median_abs_magnitude_error_px"], dtype=np.float64).ravel()
    dir_err = np.asarray(trust_products["median_abs_direction_error_deg"], dtype=np.float64).ravel()
    trust_code = np.asarray(trust_products["trust_code"], dtype=np.int8).ravel()
    low_signal_threshold_px = float(trust_products["summary"].get("low_signal_truth_threshold_px", 0.05))
    mag_good_px = float(trust_products["summary"].get("mag_error_good_px", 0.01))
    mag_warn_px = float(trust_products["summary"].get("mag_error_warn_px", 0.02))
    dir_good_deg = float(trust_products["summary"].get("direction_error_good_deg", 5.0))
    dir_warn_deg = float(trust_products["summary"].get("direction_error_warn_deg", 10.0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    color_map = {-1: "lightgray", 0: "tomato", 1: "goldenrod", 2: "forestgreen"}
    label_map = {-1: "Low-signal", 0: "Suspicious", 1: "Review", 2: "Trusted"}
    for code in [-1, 0, 1, 2]:
        mask_mag = (trust_code == code) & np.isfinite(truth) & np.isfinite(mag_err)
        if np.any(mask_mag):
            axes[0].scatter(
                truth[mask_mag],
                mag_err[mask_mag],
                s=28,
                alpha=0.82,
                color=color_map[code],
                edgecolor="none",
                label=label_map[code],
            )
        mask_dir = (trust_code == code) & np.isfinite(truth) & np.isfinite(dir_err)
        if np.any(mask_dir):
            axes[1].scatter(
                truth[mask_dir],
                dir_err[mask_dir],
                s=28,
                alpha=0.82,
                color=color_map[code],
                edgecolor="none",
                label=label_map[code] if code == -1 else None,
            )

    axes[0].axvline(low_signal_threshold_px, color="gray", linestyle=":", lw=1.6, alpha=0.8)
    axes[0].axhline(mag_good_px, color="forestgreen", linestyle="--", lw=1.2, alpha=0.8)
    axes[0].axhline(mag_warn_px, color="goldenrod", linestyle=":", lw=1.2, alpha=0.8)
    axes[0].set_xlabel("Spot median truth displacement (px)")
    axes[0].set_ylabel("Spot median |magnitude error| (px)")
    axes[0].set_title("Per-Spot Magnitude Error vs Signal")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].axvline(low_signal_threshold_px, color="gray", linestyle=":", lw=1.6, alpha=0.8)
    axes[1].axhline(dir_good_deg, color="forestgreen", linestyle="--", lw=1.2, alpha=0.8)
    axes[1].axhline(dir_warn_deg, color="goldenrod", linestyle=":", lw=1.2, alpha=0.8)
    axes[1].set_xlabel("Spot median truth displacement (px)")
    axes[1].set_ylabel("Spot median |direction error| (deg)")
    axes[1].set_title("Per-Spot Direction Error vs Signal")
    axes[1].grid(True, alpha=0.25)

    plt.suptitle(
        "Per-Spot Trust Statistics\n"
        "Direction error is only interpretable above the low-signal threshold",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(stat_dir, f"dpc_spot_trust_scatter_{method}.png"), dpi=140)
    plt.close()


def _save_truth_phase_context_statistics(
    *,
    stat_dir: str,
    method: str,
    spot_products: dict | None,
) -> None:
    """Save truth-side slope / local phase context maps and scatter plots."""
    if spot_products is None:
        return

    trust_products = spot_products.get("trust_products")
    truth_phase_context = spot_products.get("truth_phase_context") or {}
    if trust_products is None:
        return

    truth_mag = np.asarray(trust_products["median_truth_mag_px"], dtype=np.float64)
    slope_mrad = truth_phase_context.get("truth_equivalent_slope_mrad")
    phase_rms_rad = truth_phase_context.get("truth_phase_local_rms_rad")
    if slope_mrad is None and phase_rms_rad is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
    _render_spot_metric_map(
        ax=axes[0, 0],
        data_2d=slope_mrad,
        title="Truth Equivalent Slope",
        cmap="magma",
        colorbar_label="mrad",
    )
    _render_spot_metric_map(
        ax=axes[0, 1],
        data_2d=phase_rms_rad,
        title="Truth Local Phase RMS (within hole)",
        cmap="magma",
        colorbar_label="rad",
    )

    truth_flat = truth_mag.ravel()
    if slope_mrad is not None:
        slope_flat = np.asarray(slope_mrad, dtype=np.float64).ravel()
        valid = np.isfinite(truth_flat) & np.isfinite(slope_flat)
        axes[1, 0].scatter(truth_flat[valid], slope_flat[valid], s=28, alpha=0.8, color="tab:blue", edgecolor="none")
        axes[1, 0].set_ylabel("Equivalent slope (mrad)")
    else:
        axes[1, 0].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[1, 0].transAxes, color="gray")
    axes[1, 0].set_xlabel("Spot median truth displacement (px)")
    axes[1, 0].set_title("Truth Displacement vs Equivalent Slope")
    axes[1, 0].grid(True, alpha=0.25)

    if phase_rms_rad is not None:
        phase_rms_flat = np.asarray(phase_rms_rad, dtype=np.float64).ravel()
        trust_code_flat = np.asarray(trust_products["trust_code"], dtype=np.int8).ravel()
        for code, color in [(-1, "lightgray"), (0, "tomato"), (1, "goldenrod"), (2, "forestgreen")]:
            mask = (trust_code_flat == code) & np.isfinite(truth_flat) & np.isfinite(phase_rms_flat)
            if np.any(mask):
                axes[1, 1].scatter(
                    truth_flat[mask],
                    phase_rms_flat[mask],
                    s=28,
                    alpha=0.82,
                    color=color,
                    edgecolor="none",
                    label=_trust_code_label(code),
                )
        axes[1, 1].legend(loc="best")
        axes[1, 1].set_ylabel("Local phase RMS (rad)")
    else:
        axes[1, 1].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[1, 1].transAxes, color="gray")
    axes[1, 1].set_xlabel("Spot median truth displacement (px)")
    axes[1, 1].set_title("Truth Displacement vs Local Phase RMS")
    axes[1, 1].grid(True, alpha=0.25)

    plt.suptitle(
        "Truth Wavefront Context\n"
        "Low signal tracks small slope more directly than small phase RMS",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(stat_dir, f"dpc_truth_wavefront_context_{method}.png"), dpi=140)
    plt.close()


def _save_dpc_spot_products(result: dict, output_dir: str) -> None:
    """Persist per-spot DPC diagnostics and a summary figure."""
    spot_products = result.get('dpc_spot_products')
    if spot_products is None:
        return

    method = result['method']
    raw_stack_names = {
        f"dpc_recovered_x_stack_px_{method}.npy": spot_products['recovered_x_px'],
        f"dpc_recovered_y_stack_px_{method}.npy": spot_products['recovered_y_px'],
        f"dpc_truth_x_stack_px_{method}.npy": spot_products['truth_x_px'],
        f"dpc_truth_y_stack_px_{method}.npy": spot_products['truth_y_px'],
        f"dpc_truth_mag_stack_px_{method}.npy": spot_products['truth_mag_px'],
        f"dpc_error_x_stack_px_{method}.npy": spot_products['error_x_px'],
        f"dpc_error_y_stack_px_{method}.npy": spot_products['error_y_px'],
        f"dpc_error_mag_stack_px_{method}.npy": spot_products['error_mag_px'],
        f"dpc_parallel_error_stack_px_{method}.npy": spot_products['parallel_error_px'],
        f"dpc_perpendicular_error_stack_px_{method}.npy": spot_products['perpendicular_error_px'],
        f"dpc_relative_error_pct_stack_{method}.npy": spot_products['relative_error_pct'],
    }
    for filename, array in raw_stack_names.items():
        _save_named_array(output_dir, filename, array, allow_preview=False)

    spot_map_names = {
        f"dpc_bias_x_per_spot_px_{method}.npy": spot_products['bias_x_px'],
        f"dpc_bias_y_per_spot_px_{method}.npy": spot_products['bias_y_px'],
        f"dpc_truth_mag_per_spot_px_{method}.npy": spot_products['mean_truth_mag_px'],
        f"dpc_rmse_x_per_spot_px_{method}.npy": spot_products['rmse_x_px'],
        f"dpc_rmse_y_per_spot_px_{method}.npy": spot_products['rmse_y_px'],
        f"dpc_rmse_mag_per_spot_px_{method}.npy": spot_products['rmse_mag_px'],
        f"dpc_mean_parallel_error_px_per_spot_{method}.npy": spot_products['mean_parallel_error_px'],
        f"dpc_mean_abs_parallel_error_px_per_spot_{method}.npy": spot_products['mean_abs_parallel_error_px'],
        f"dpc_mean_perpendicular_error_px_per_spot_{method}.npy": spot_products['mean_perpendicular_error_px'],
        f"dpc_mean_abs_perpendicular_error_px_per_spot_{method}.npy": spot_products['mean_abs_perpendicular_error_px'],
        f"dpc_mean_relative_error_pct_per_spot_{method}.npy": spot_products['mean_relative_error_pct'],
        f"dpc_p95_relative_error_pct_per_spot_{method}.npy": spot_products['p95_relative_error_pct'],
        f"dpc_scale_x_per_spot_{method}.npy": spot_products['scale_x'],
        f"dpc_scale_y_per_spot_{method}.npy": spot_products['scale_y'],
        f"dpc_scale_vec_per_spot_{method}.npy": spot_products['scale_vec'],
        f"dpc_mean_angle_deg_per_spot_{method}.npy": spot_products['mean_angle_deg'],
        f"dpc_mean_cosine_per_spot_{method}.npy": spot_products['mean_cosine'],
    }
    if spot_products.get('mean_confidence') is not None:
        spot_map_names[f"dpc_mean_confidence_per_spot_{method}.npy"] = spot_products['mean_confidence']
    if spot_products.get('condition_numbers') is not None:
        spot_map_names[f"dpc_condition_number_per_spot_{method}.npy"] = spot_products['condition_numbers']
    if spot_products.get('residual_ratios') is not None:
        spot_map_names[f"dpc_residual_ratio_per_spot_{method}.npy"] = spot_products['residual_ratios']
    for filename, array in spot_map_names.items():
        _save_named_array(output_dir, filename, array)

    csv_path = os.path.join(output_dir, f"dpc_spot_metrics_{method}.csv")
    _save_dpc_spot_csv(csv_path, spot_products)
    trust_csv_path = os.path.join(output_dir, f"dpc_spot_trust_summary_{method}.csv")
    _save_dpc_spot_trust_csv(trust_csv_path, spot_products)
    _save_dpc_spot_trust_maps(
        output_dir=output_dir,
        method=method,
        spot_products=spot_products,
    )

    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    _render_spot_metric_map(
        ax=axes[0, 0],
        data_2d=spot_products['mean_truth_mag_px'],
        title='Spot Mean Truth Magnitude',
        cmap='magma',
        colorbar_label='px',
    )
    _render_spot_metric_map(
        ax=axes[0, 1],
        data_2d=spot_products['mean_abs_error_mag_px'],
        title='Spot Mean Abs Error',
        cmap='magma',
        colorbar_label='px',
    )
    _render_spot_metric_map(
        ax=axes[0, 2],
        data_2d=spot_products['rmse_mag_px'],
        title='Spot RMSE Magnitude',
        cmap='magma',
        colorbar_label='px',
    )
    _render_spot_metric_map(
        ax=axes[0, 3],
        data_2d=spot_products['mean_abs_parallel_error_px'],
        title='Spot Mean |Parallel| Error',
        cmap='magma',
        colorbar_label='px',
    )
    _render_spot_metric_map(
        ax=axes[0, 4],
        data_2d=spot_products['mean_abs_perpendicular_error_px'],
        title='Spot Mean |Perp| Error',
        cmap='magma',
        colorbar_label='px',
    )
    _render_spot_metric_map(
        ax=axes[1, 0],
        data_2d=(
            None
            if spot_products.get('scale_vec') is None
            else np.asarray(spot_products['scale_vec'], dtype=np.float64) - 1.0
        ),
        title='Spot Vector Scale - 1',
        cmap='RdBu_r',
        symmetric=True,
        colorbar_label='ratio',
    )
    _render_spot_metric_map(
        ax=axes[1, 1],
        data_2d=spot_products['mean_angle_deg'],
        title='Spot Mean Angle Error',
        cmap='viridis',
        colorbar_label='deg',
    )
    _render_spot_metric_map(
        ax=axes[1, 2],
        data_2d=spot_products['bias_x_px'],
        title='Spot Mean Bias dX',
        cmap='RdBu_r',
        symmetric=True,
        colorbar_label='px',
    )
    _render_spot_metric_map(
        ax=axes[1, 3],
        data_2d=spot_products['bias_y_px'],
        title='Spot Mean Bias dY',
        cmap='RdBu_r',
        symmetric=True,
        colorbar_label='px',
    )
    _render_spot_metric_map(
        ax=axes[1, 4],
        data_2d=spot_products.get('residual_ratios'),
        title='Spot Residual Ratio',
        cmap='viridis',
        colorbar_label='ratio',
    )

    summary = spot_products.get('summary', {})
    fig.suptitle(
        "Per-Spot DPC Diagnostics\n"
        f"median sample |err|={summary.get('median_sample_abs_error_px') or 0.0:.4f} px, "
        f"p95 sample |err|={summary.get('p95_sample_abs_error_px') or 0.0:.4f} px, "
        f"median |parallel|={summary.get('median_sample_abs_parallel_error_px') or 0.0:.4f} px, "
        f"median |perp|={summary.get('median_sample_abs_perpendicular_error_px') or 0.0:.4f} px, "
        f"median scale={summary.get('median_scale_vec') or 0.0:.4f}",
        fontsize=13,
    )
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"dpc_spot_summary_{method}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Per-spot DPC diagnostics saved: {fig_path}")
    print(f"  Per-spot DPC CSV saved: {csv_path}")
    print(f"  Per-spot trust CSV saved: {trust_csv_path}")


def _save_recovery_summary_figure(result: dict, output_dir: str, data: dict) -> bool:
    """Save the main recovered-phase summary figure. Returns whether truth displacements exist."""
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
        ax.text(0.5, 0.5, 'No WFE simulation truth', ha='center', va='center',
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
    """Return representative true displacement fields for summary visualization."""
    gt_disp = result['gt_displacements']
    aberr_mode_vis = result.get('aberr_mode', 'none')
    if aberr_mode_vis == 'per_pulse':
        return gt_disp['displacement_x_px'][0], gt_disp['displacement_y_px'][0], ' (frame 0 only, simulation truth)'
    return (
        np.mean(gt_disp['displacement_x_px'], axis=0),
        np.mean(gt_disp['displacement_y_px'], axis=0),
        ' (mean, simulation truth)',
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
    _save_dpc_spot_products(result, output_dir)
    has_gt_disp = 'gt_displacements' in result and result['gt_displacements'] is not None
    if not _is_dpc_only_output():
        has_gt_disp = _save_recovery_summary_figure(result, output_dir, data)

    # DPC-only mode keeps only displacement/DPC comparisons.
    if has_gt_disp and bool(getattr(config, 'SAVE_DPC_FIGURES', True)):
        _save_displacement_comparison(result, output_dir)

    # === 逐帧相位恢复可视化 ===
    if (not _is_dpc_only_output()) and result.get('all_phase_rad') is not None:
        _save_perframe_phase(result, output_dir)

    # === 逐帧位移 vs truth 对比 ===
    if (
        result.get('all_shifts_px') is not None
        and has_gt_disp
        and bool(getattr(config, 'SAVE_DPC_FIGURES', True))
    ):
        _save_perframe_displacement_comparison(result, output_dir)
        _save_metrics_statistics(result, output_dir)
        _save_all_frames_disp_summary(result, output_dir)


def _plot_displacement_comparison_row(
    *,
    axes_row,
    truth_displacement_px: np.ndarray,
    recovered_displacement_px: np.ndarray,
    displacement_error_px: np.ndarray,
    component_label: str,
    truth_title_suffix: str,
    method: str,
    rmse_value: float,
) -> None:
    """Render one true/recovered/error/scatter row for dX or dY comparison."""
    vmax_signal = _displacement_display_limits(truth_displacement_px, recovered_displacement_px)
    vmax_error = _displacement_display_limits(displacement_error_px)

    ax = axes_row[0]
    im = ax.imshow(truth_displacement_px, cmap='RdBu_r', vmin=-vmax_signal, vmax=vmax_signal, origin='lower')
    ax.set_title(f'True {component_label}{truth_title_suffix}')
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
    ax.set_xlabel(f'True {component_label} (px)')
    ax.set_ylabel(f'Rec {component_label} (px)')
    ax.set_title(f'Scatter {component_label}')
    ax.set_aspect('equal')


def _save_displacement_comparison(result, output_dir):
    """保存详细的位移对比图: True vs Recovered (dx/dy 分别显示)。 """
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
        truth_title_suffix=label_suffix,
        method=method,
        rmse_value=rmse_dx,
    )
    _plot_displacement_comparison_row(
        axes_row=axes[1],
        truth_displacement_px=truth_displacement_y_px,
        recovered_displacement_px=recovered_displacement_y_px,
        displacement_error_px=displacement_error_y_px,
        component_label='dY',
        truth_title_suffix=label_suffix,
        method=method,
        rmse_value=rmse_dy,
    )
    
    dm = result.get('disp_metrics', {})
    corr_dx = dm.get('corr_dx', dm.get('mean_corr_dx', 0))
    corr_dy = dm.get('corr_dy', dm.get('mean_corr_dy', 0))
    fig.suptitle(
        f"Displacement Comparison — {method}\n"
        f"RMSE total={dm.get('rmse_total_px', 0):.4f} px, "
        f"Corr dX={corr_dx:.4f}, dY={corr_dy:.4f}",
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
    逐帧位移对比可视化 (互相关恢复 vs truth)。
    每帧: truth/recovered dx/dy 的 colormap + quiver 箭头叠加对比 + 误差图。
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
        frame_metric = result.get('disp_metrics', {}).get('per_frame_metrics', [])
        metric_row = frame_metric[i] if i < len(frame_metric) else {}
        _save_single_frame_quiver_plot(
            disp_dir=disp_dir,
            method=method,
            frame_index=i,
            truth_displacement_x_px=frame_stats["truth_displacement_x_px"],
            truth_displacement_y_px=frame_stats["truth_displacement_y_px"],
            recovered_displacement_x_px=frame_stats["recovered_displacement_x_px"],
            recovered_displacement_y_px=frame_stats["recovered_displacement_y_px"],
            rmse_total=frame_stats["rmse_total"],
            p95_err_px=float(metric_row.get('p95_err_px', frame_stats["rmse_total"])),
            scale_vec=metric_row.get('scale_vec'),
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
          f"({n_compare} frames + per-frame quiver + RMSE curve)")


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
    axes[0, 0].set_title(f'True dX (frame {frame_index})')
    plt.colorbar(im, ax=axes[0, 0], label='px')

    im = axes[0, 1].imshow(recovered_displacement_x_px, cmap='RdBu_r', vmin=-vmax_dx, vmax=vmax_dx, origin='lower')
    axes[0, 1].set_title(f'Rec dX ({method})')
    plt.colorbar(im, ax=axes[0, 1], label='px')

    im = axes[0, 2].imshow(displacement_error_x_px, cmap='RdBu_r', vmin=-vmax_err_dx, vmax=vmax_err_dx, origin='lower')
    axes[0, 2].set_title(f'Error dX (RMSE={rmse_dx:.4f})')
    plt.colorbar(im, ax=axes[0, 2], label='px')

    im = axes[1, 0].imshow(truth_displacement_y_px, cmap='RdBu_r', vmin=-vmax_dy, vmax=vmax_dy, origin='lower')
    axes[1, 0].set_title(f'True dY (frame {frame_index})')
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


def _save_single_frame_quiver_plot(
    *,
    disp_dir: str,
    method: str,
    frame_index: int,
    truth_displacement_x_px: np.ndarray,
    truth_displacement_y_px: np.ndarray,
    recovered_displacement_x_px: np.ndarray,
    recovered_displacement_y_px: np.ndarray,
    rmse_total: float,
    p95_err_px: float,
    scale_vec: float | None,
) -> None:
    """Save one true-vs-recovered quiver image per frame."""
    quiver_dir = os.path.join(disp_dir, "quiver_frames")
    os.makedirs(quiver_dir, exist_ok=True)

    rows, cols = truth_displacement_x_px.shape
    yy, xx = np.mgrid[:rows, :cols]
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5))
    ax.quiver(
        xx,
        yy,
        truth_displacement_x_px,
        truth_displacement_y_px,
        color='blue',
        alpha=0.60,
        scale=None,
        label='True',
    )
    ax.quiver(
        xx,
        yy,
        recovered_displacement_x_px,
        recovered_displacement_y_px,
        color='red',
        alpha=0.55,
        scale=None,
        label='Recovered',
    )
    scale_text = "n/a" if scale_vec is None else f"{scale_vec:.4f}"
    ax.set_title(
        f"Frame {frame_index}\n"
        f"RMSE={rmse_total:.4f} px, P95={p95_err_px:.4f} px, scale={scale_text}"
    )
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('Spot column')
    ax.set_ylabel('Spot row')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(quiver_dir, f"quiver_frame_{frame_index:03d}_{method}.png"), dpi=150)
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
        ax.quiver(xx, yy, truth_displacement_x_px, truth_displacement_y_px, color='blue', alpha=0.6, scale=None, label='True')
        ax.quiver(xx, yy, recovered_displacement_x_px, recovered_displacement_y_px, color='red', alpha=0.6, scale=None, label='Recovered')
        ax.set_title(f"Frame {idx}\nRMSE={per_frame_rmse_total[idx]:.4f} px")
        ax.set_aspect('equal')
        ax.legend(fontsize=8, loc='upper right')
        ax.invert_yaxis()
    fig.suptitle(f"Displacement Quiver — {method} (blue=True, red=Recovered)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(disp_dir, f"quiver_overview_{method}.png"), dpi=120)
    plt.close()


def _save_per_frame_metrics_tables(
    *,
    stat_dir: str,
    method: str,
    per_frame_metrics: list[dict],
    disp_metrics: dict | None,
) -> None:
    """Save rich per-frame metrics as CSV and JSON."""
    if not per_frame_metrics:
        return

    csv_path = os.path.join(stat_dir, f"per_frame_metrics_{method}.csv")
    fieldnames = list(per_frame_metrics[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_frame_metrics)

    summary_payload = {
        "summary": {
            "n_frames": None if disp_metrics is None else disp_metrics.get("n_frames"),
            "rmse_total_px": None if disp_metrics is None else disp_metrics.get("rmse_total_px"),
            "median_frame_rmse_px": None if disp_metrics is None else disp_metrics.get("median_frame_rmse_px"),
            "p95_frame_rmse_px": None if disp_metrics is None else disp_metrics.get("p95_frame_rmse_px"),
            "best_frame_index": None if disp_metrics is None else disp_metrics.get("best_frame_index"),
            "best_frame_rmse_px": None if disp_metrics is None else disp_metrics.get("best_frame_rmse_px"),
            "worst_frame_index": None if disp_metrics is None else disp_metrics.get("worst_frame_index"),
            "worst_frame_rmse_px": None if disp_metrics is None else disp_metrics.get("worst_frame_rmse_px"),
            "mean_corr_dx": None if disp_metrics is None else disp_metrics.get("mean_corr_dx"),
            "mean_corr_dy": None if disp_metrics is None else disp_metrics.get("mean_corr_dy"),
        },
        "per_frame_metrics": per_frame_metrics,
    }
    with open(os.path.join(stat_dir, f"per_frame_metrics_{method}.json"), "w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2)


def _save_per_frame_error_percentile_curve(
    *,
    stat_dir: str,
    method: str,
    per_frame_metrics: list[dict],
) -> None:
    """Plot per-frame error percentiles and vector diagnostics."""
    if not per_frame_metrics:
        return
    frames_x = np.arange(len(per_frame_metrics))
    median_err = [float(m['median_err_px']) for m in per_frame_metrics]
    p90_err = [float(m['p90_err_px']) for m in per_frame_metrics]
    p95_err = [float(m['p95_err_px']) for m in per_frame_metrics]
    max_err = [float(m['max_err_px']) for m in per_frame_metrics]
    scale_vec = [np.nan if m.get('scale_vec') is None else float(m['scale_vec']) for m in per_frame_metrics]
    angle_deg = [np.nan if m.get('mean_angle_deg') is None else float(m['mean_angle_deg']) for m in per_frame_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(frames_x, median_err, 'o-', label='Median err', color='tab:blue')
    axes[0].plot(frames_x, p90_err, 's-', label='P90 err', color='tab:orange')
    axes[0].plot(frames_x, p95_err, '^-', label='P95 err', color='tab:red')
    axes[0].plot(frames_x, max_err, 'd--', label='Max err', color='tab:purple')
    axes[0].set_xlabel('Frame index')
    axes[0].set_ylabel('Error (px)')
    axes[0].set_title('Per-Frame Error Percentiles')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(frames_x, scale_vec, 'o-', label='Vector scale', color='tab:green')
    axes[1].plot(frames_x, angle_deg, 's-', label='Mean angle err (deg)', color='tab:brown')
    axes[1].set_xlabel('Frame index')
    axes[1].set_title('Per-Frame Scale / Angle')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(stat_dir, f"per_frame_error_percentiles_{method}.png"), dpi=140)
    plt.close()


def _save_spot_metric_histograms(
    *,
    stat_dir: str,
    method: str,
    spot_products: dict | None,
) -> None:
    """Save histogram-style summaries for spot-level diagnostics."""
    if spot_products is None:
        return

    abs_err = np.asarray(spot_products.get('mean_abs_error_mag_px'), dtype=np.float64).ravel()
    abs_parallel = np.asarray(spot_products.get('mean_abs_parallel_error_px'), dtype=np.float64).ravel()
    abs_perpendicular = np.asarray(spot_products.get('mean_abs_perpendicular_error_px'), dtype=np.float64).ravel()
    scale = np.asarray(spot_products.get('scale_vec'), dtype=np.float64).ravel()
    angle = np.asarray(spot_products.get('mean_angle_deg'), dtype=np.float64).ravel()

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, values, title, color, xlabel in [
        (axes[0], abs_err[np.isfinite(abs_err)], 'Spot Mean |Error|', 'tab:red', 'Abs error (px)'),
        (axes[1], abs_parallel[np.isfinite(abs_parallel)], 'Spot Mean |Parallel Error|', 'tab:orange', '|Parallel| (px)'),
        (axes[2], abs_perpendicular[np.isfinite(abs_perpendicular)], 'Spot Mean |Perp Error|', 'tab:purple', '|Perp| (px)'),
        (axes[3], scale[np.isfinite(scale)], 'Spot Scale Distribution', 'tab:blue', 'Scale'),
        (axes[4], angle[np.isfinite(angle)], 'Spot Angle Error Distribution', 'tab:green', 'Angle (deg)'),
    ]:
        if values.size == 0:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, color='gray')
            ax.set_title(title)
            continue
        ax.hist(values, bins=30, color=color, alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(stat_dir, f"spot_metric_histograms_{method}.png"), dpi=140)
    plt.close()


def _save_truth_binned_dpc_statistics(
    *,
    stat_dir: str,
    method: str,
    spot_products: dict | None,
) -> None:
    """Save DPC error statistics binned by true displacement magnitude."""
    if spot_products is None:
        return
    truth_bins = spot_products.get('truth_bins')
    if not truth_bins:
        return

    csv_path = os.path.join(stat_dir, f"dpc_error_vs_truth_bins_{method}.csv")
    fieldnames = list(truth_bins[0].keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(truth_bins)

    labels = [row['bin_label'] for row in truth_bins]
    median_abs = [np.nan if row.get('median_abs_error_px') is None else float(row['median_abs_error_px']) for row in truth_bins]
    p95_abs = [np.nan if row.get('p95_abs_error_px') is None else float(row['p95_abs_error_px']) for row in truth_bins]
    median_par = [
        np.nan if row.get('median_abs_parallel_error_px') is None else float(row['median_abs_parallel_error_px'])
        for row in truth_bins
    ]
    median_perp = [
        np.nan if row.get('median_abs_perpendicular_error_px') is None else float(row['median_abs_perpendicular_error_px'])
        for row in truth_bins
    ]
    counts = [int(row.get('count') or 0) for row in truth_bins]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    x = np.arange(len(labels))
    axes[0].plot(x, median_abs, 'o-', color='tab:red', label='Median |error|')
    axes[0].plot(x, p95_abs, 's-', color='tab:orange', label='P95 |error|')
    axes[0].plot(x, median_par, '^-', color='tab:blue', label='Median |parallel|')
    axes[0].plot(x, median_perp, 'd-', color='tab:purple', label='Median |perp|')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha='right')
    axes[0].set_ylabel('Error (px)')
    axes[0].set_title('DPC Error vs True Displacement Magnitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].bar(x, counts, color='tab:gray', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha='right')
    axes[1].set_ylabel('Sample count')
    axes[1].set_title('Samples per True-Displacement Bin')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(stat_dir, f"dpc_error_vs_truth_bins_{method}.png"), dpi=140)
    plt.close()


def _save_angle_error_statistics(
    *,
    stat_dir: str,
    method: str,
    spot_products: dict | None,
) -> None:
    """Save slide-ready angle-error summaries with magnitude conditioning."""
    if spot_products is None:
        return

    angle_stack = np.asarray(spot_products.get('angle_stack_deg'), dtype=np.float64).ravel()
    truth_mag = np.asarray(spot_products.get('truth_mag_px'), dtype=np.float64).ravel()
    finite = np.isfinite(angle_stack) & np.isfinite(truth_mag)
    if not np.any(finite):
        return

    angle = angle_stack[finite]
    truth = truth_mag[finite]
    truth_bins = spot_products.get('truth_bins') or []

    def _summary_row(label: str, mask: np.ndarray) -> dict:
        count = int(np.count_nonzero(mask))
        if count == 0:
            return {
                "scope": label,
                "count": 0,
                "median_angle_deg": None,
                "p95_angle_deg": None,
                "fraction_lt_5_deg_pct": None,
                "fraction_lt_10_deg_pct": None,
                "fraction_lt_15_deg_pct": None,
            }
        vals = angle[mask]
        return {
            "scope": label,
            "count": count,
            "median_angle_deg": float(np.median(vals)),
            "p95_angle_deg": float(np.percentile(vals, 95)),
            "fraction_lt_5_deg_pct": 100.0 * float(np.mean(vals < 5.0)),
            "fraction_lt_10_deg_pct": 100.0 * float(np.mean(vals < 10.0)),
            "fraction_lt_15_deg_pct": 100.0 * float(np.mean(vals < 15.0)),
        }

    scopes = [
        _summary_row("all_finite_samples", np.ones_like(angle, dtype=bool)),
        _summary_row("truth_ge_0p05_px", truth >= 0.05),
        _summary_row("truth_ge_0p10_px", truth >= 0.10),
        _summary_row("truth_ge_0p20_px", truth >= 0.20),
    ]

    csv_path = os.path.join(stat_dir, f"dpc_angle_error_summary_{method}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(scopes[0].keys()))
        writer.writeheader()
        writer.writerows(scopes)

    angle_upper = float(np.percentile(angle, 99.5))
    angle_upper = max(angle_upper, 15.0)
    hist_bins = np.linspace(0.0, angle_upper, 40)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(angle, bins=hist_bins, color='tab:green', alpha=0.78)
    axes[0, 0].axvline(np.median(angle), color='darkgreen', linestyle='--', lw=1.8, label=f"Median={np.median(angle):.2f} deg")
    axes[0, 0].axvline(np.percentile(angle, 95), color='tab:olive', linestyle=':', lw=2.0, label=f"P95={np.percentile(angle, 95):.2f} deg")
    axes[0, 0].set_title('Angle Error Distribution')
    axes[0, 0].set_xlabel('Angle error (deg)')
    axes[0, 0].set_ylabel('Sample count')
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend()

    for mask, label, color in [
        (np.ones_like(angle, dtype=bool), 'All finite samples', 'tab:green'),
        (truth >= 0.05, 'True disp. >= 0.05 px', 'tab:blue'),
        (truth >= 0.10, 'True disp. >= 0.10 px', 'tab:red'),
    ]:
        vals = np.sort(angle[mask])
        if vals.size == 0:
            continue
        cdf = np.arange(1, vals.size + 1, dtype=np.float64) / vals.size
        axes[0, 1].plot(vals, cdf, lw=2.0, label=label, color=color)
    axes[0, 1].axvline(5.0, color='gray', linestyle=':', alpha=0.6)
    axes[0, 1].axvline(10.0, color='gray', linestyle='--', alpha=0.6)
    axes[0, 1].set_title('Angle Error CDF')
    axes[0, 1].set_xlabel('Angle error (deg)')
    axes[0, 1].set_ylabel('Cumulative fraction')
    axes[0, 1].set_ylim(0.0, 1.02)
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(loc='lower right')

    labels = [row['bin_label'] for row in truth_bins]
    x = np.arange(len(labels))
    median_angle = [np.nan if row.get('median_angle_deg') is None else float(row['median_angle_deg']) for row in truth_bins]
    p95_angle = [np.nan if row.get('p95_angle_deg') is None else float(row['p95_angle_deg']) for row in truth_bins]
    counts = [int(row.get('count') or 0) for row in truth_bins]
    axes[1, 0].plot(x, median_angle, 'o-', color='tab:green', label='Median angle')
    axes[1, 0].plot(x, p95_angle, 's-', color='tab:red', label='P95 angle')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=25, ha='right')
    axes[1, 0].set_ylabel('Angle error (deg)')
    axes[1, 0].set_title('Angle Error vs True Displacement')
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].axis('off')
    lines = []
    for row in scopes:
        if row["count"] <= 0:
            continue
        lines.append(
            f"{row['scope']}:"
            f"\n  median={row['median_angle_deg']:.2f} deg,"
            f" p95={row['p95_angle_deg']:.2f} deg"
            f"\n  <5 deg={row['fraction_lt_5_deg_pct']:.1f}%,"
            f" <10 deg={row['fraction_lt_10_deg_pct']:.1f}%,"
            f" <15 deg={row['fraction_lt_15_deg_pct']:.1f}%"
        )
    if counts:
        lines.append(f"sample counts by true-displacement bin: {counts}")
    axes[1, 1].text(
        0.02,
        0.98,
        "\n\n".join(lines),
        va='top',
        ha='left',
        fontsize=10.8,
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.6', fc='whitesmoke', ec='lightgray'),
    )
    axes[1, 1].set_title('Angle Error Summary')

    plt.suptitle(
        f"DPC Angle Error Summary — {method}\n"
        f"median angle={np.median(angle):.2f} deg, p95 angle={np.percentile(angle, 95):.2f} deg",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(stat_dir, f"dpc_angle_error_summary_{method}.png"), dpi=140)
    plt.close()


def _save_signal_vs_error_scale_statistics(
    *,
    stat_dir: str,
    method: str,
    spot_products: dict | None,
) -> None:
    """Save slide-ready plots comparing true displacement scale and recovery error scale."""
    if spot_products is None:
        return

    truth_mag = np.asarray(spot_products.get('truth_mag_px'), dtype=np.float64).ravel()
    error_mag = np.asarray(spot_products.get('error_mag_px'), dtype=np.float64).ravel()
    truth_mag = truth_mag[np.isfinite(truth_mag)]
    error_mag = error_mag[np.isfinite(error_mag)]
    if truth_mag.size == 0 or error_mag.size == 0:
        return

    median_truth = float(np.median(truth_mag))
    p95_truth = float(np.percentile(truth_mag, 95))
    max_truth = float(np.max(truth_mag))
    median_error = float(np.median(error_mag))
    p95_error = float(np.percentile(error_mag, 95))
    max_error = float(np.max(error_mag))
    median_signal_to_error = np.nan if median_error <= 0 else float(median_truth / median_error)

    fractions = {
        "truth_lt_0p05_px_pct": 100.0 * float(np.mean(truth_mag < 0.05)),
        "truth_lt_0p10_px_pct": 100.0 * float(np.mean(truth_mag < 0.10)),
        "truth_lt_0p20_px_pct": 100.0 * float(np.mean(truth_mag < 0.20)),
        "error_lt_0p01_px_pct": 100.0 * float(np.mean(error_mag < 0.01)),
        "error_lt_0p02_px_pct": 100.0 * float(np.mean(error_mag < 0.02)),
        "error_lt_0p05_px_pct": 100.0 * float(np.mean(error_mag < 0.05)),
    }

    summary_rows = [
        {"metric": "n_samples", "value": int(truth_mag.size)},
        {"metric": "median_truth_mag_px", "value": median_truth},
        {"metric": "p95_truth_mag_px", "value": p95_truth},
        {"metric": "max_truth_mag_px", "value": max_truth},
        {"metric": "median_error_mag_px", "value": median_error},
        {"metric": "p95_error_mag_px", "value": p95_error},
        {"metric": "max_error_mag_px", "value": max_error},
        {"metric": "median_signal_to_error_ratio", "value": median_signal_to_error},
    ]
    summary_rows.extend({"metric": key, "value": value} for key, value in fractions.items())
    csv_path = os.path.join(stat_dir, f"dpc_signal_vs_error_scale_{method}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    combined = np.concatenate([truth_mag, error_mag])
    upper = float(np.percentile(combined, 99.5))
    upper = max(upper, max_truth, max_error, 0.05)
    bins = np.linspace(0.0, upper, 40)

    truth_sorted = np.sort(truth_mag)
    error_sorted = np.sort(error_mag)
    truth_cdf = np.arange(1, truth_sorted.size + 1, dtype=np.float64) / truth_sorted.size
    error_cdf = np.arange(1, error_sorted.size + 1, dtype=np.float64) / error_sorted.size

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(truth_mag, bins=bins, color='tab:blue', alpha=0.78)
    axes[0, 0].axvline(median_truth, color='navy', linestyle='--', lw=1.8, label=f"Median={median_truth:.3f} px")
    axes[0, 0].axvline(p95_truth, color='tab:cyan', linestyle=':', lw=2.0, label=f"P95={p95_truth:.3f} px")
    axes[0, 0].set_title('True Displacement Magnitude Distribution')
    axes[0, 0].set_xlabel('True displacement magnitude (px)')
    axes[0, 0].set_ylabel('Sample count')
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].hist(error_mag, bins=bins, color='tab:red', alpha=0.78)
    axes[0, 1].axvline(median_error, color='darkred', linestyle='--', lw=1.8, label=f"Median={median_error:.3f} px")
    axes[0, 1].axvline(p95_error, color='tab:orange', linestyle=':', lw=2.0, label=f"P95={p95_error:.3f} px")
    axes[0, 1].set_title('Recovery Error Magnitude Distribution')
    axes[0, 1].set_xlabel('Recovery error magnitude (px)')
    axes[0, 1].set_ylabel('Sample count')
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].plot(truth_sorted, truth_cdf, color='tab:blue', lw=2.2, label='True displacement CDF')
    axes[1, 0].plot(error_sorted, error_cdf, color='tab:red', lw=2.2, label='Error magnitude CDF')
    axes[1, 0].axvline(0.01, color='tab:red', linestyle=':', alpha=0.6)
    axes[1, 0].axvline(0.10, color='tab:blue', linestyle=':', alpha=0.6)
    axes[1, 0].set_title('Signal Scale vs Error Scale (CDF)')
    axes[1, 0].set_xlabel('Magnitude (px)')
    axes[1, 0].set_ylabel('Cumulative fraction')
    axes[1, 0].set_ylim(0.0, 1.02)
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend(loc='lower right')

    axes[1, 1].axis('off')
    summary_lines = [
        f"Samples: {truth_mag.size}",
        f"Median true displacement = {median_truth:.4f} px",
        f"P95 true displacement = {p95_truth:.4f} px",
        f"Median |error| = {median_error:.4f} px",
        f"P95 |error| = {p95_error:.4f} px",
        f"Median signal / error = {median_signal_to_error:.1f}x" if np.isfinite(median_signal_to_error) else "Median signal / error = N/A",
        f"{fractions['truth_lt_0p10_px_pct']:.1f}% of true disp. < 0.10 px",
        f"{fractions['error_lt_0p01_px_pct']:.1f}% of error < 0.01 px",
        f"{fractions['error_lt_0p02_px_pct']:.1f}% of error < 0.02 px",
    ]
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va='top',
        ha='left',
        fontsize=11.5,
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.6', fc='whitesmoke', ec='lightgray'),
    )
    axes[1, 1].set_title('Key Scale Summary')

    plt.suptitle(
        f"DPC Signal Scale vs Recovery Error Scale — {method}\n"
        f"median true displacement={median_truth:.4f} px, median error={median_error:.4f} px",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(stat_dir, f"dpc_signal_vs_error_scale_{method}.png"), dpi=140)
    plt.close()


def _save_pointwise_direction_and_magnitude_error(
    *,
    stat_dir: str,
    method: str,
    spot_products: dict | None,
) -> None:
    """Save pointwise direction and magnitude error CSV/figure for every frame-spot sample."""
    if spot_products is None:
        return

    recovered_x = np.asarray(spot_products.get('recovered_x_px'), dtype=np.float64)
    recovered_y = np.asarray(spot_products.get('recovered_y_px'), dtype=np.float64)
    truth_x = np.asarray(spot_products.get('truth_x_px'), dtype=np.float64)
    truth_y = np.asarray(spot_products.get('truth_y_px'), dtype=np.float64)
    if recovered_x.shape != recovered_y.shape or recovered_x.shape != truth_x.shape or recovered_x.shape != truth_y.shape:
        return
    if recovered_x.ndim != 3:
        return

    rec_mag = np.sqrt(recovered_x**2 + recovered_y**2)
    truth_mag = np.sqrt(truth_x**2 + truth_y**2)
    mag_error = rec_mag - truth_mag
    abs_mag_error = np.abs(mag_error)

    rec_angle = np.degrees(np.arctan2(recovered_y, recovered_x))
    truth_angle = np.degrees(np.arctan2(truth_y, truth_x))
    signed_angle_error = ((rec_angle - truth_angle + 180.0) % 360.0) - 180.0
    abs_angle_error = np.abs(signed_angle_error)
    valid_angle = (rec_mag > 1e-8) & (truth_mag > 1e-8)
    signed_angle_error = np.where(valid_angle, signed_angle_error, np.nan)
    abs_angle_error = np.where(valid_angle, abs_angle_error, np.nan)

    csv_path = os.path.join(
        stat_dir,
        f"dpc_pointwise_direction_and_magnitude_error_{method}.csv",
    )
    fieldnames = [
        "frame",
        "row",
        "col",
        "truth_dx_px",
        "truth_dy_px",
        "recovered_dx_px",
        "recovered_dy_px",
        "truth_mag_px",
        "recovered_mag_px",
        "magnitude_error_px",
        "abs_magnitude_error_px",
        "direction_error_deg",
        "abs_direction_error_deg",
    ]
    n_frames, n_rows, n_cols = recovered_x.shape
    with open(csv_path, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for frame_idx in range(n_frames):
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    writer.writerow(
                        {
                            "frame": int(frame_idx),
                            "row": int(row_idx),
                            "col": int(col_idx),
                            "truth_dx_px": float(truth_x[frame_idx, row_idx, col_idx]),
                            "truth_dy_px": float(truth_y[frame_idx, row_idx, col_idx]),
                            "recovered_dx_px": float(recovered_x[frame_idx, row_idx, col_idx]),
                            "recovered_dy_px": float(recovered_y[frame_idx, row_idx, col_idx]),
                            "truth_mag_px": float(truth_mag[frame_idx, row_idx, col_idx]),
                            "recovered_mag_px": float(rec_mag[frame_idx, row_idx, col_idx]),
                            "magnitude_error_px": float(mag_error[frame_idx, row_idx, col_idx]),
                            "abs_magnitude_error_px": float(abs_mag_error[frame_idx, row_idx, col_idx]),
                            "direction_error_deg": (
                                None
                                if not np.isfinite(signed_angle_error[frame_idx, row_idx, col_idx])
                                else float(signed_angle_error[frame_idx, row_idx, col_idx])
                            ),
                            "abs_direction_error_deg": (
                                None
                                if not np.isfinite(abs_angle_error[frame_idx, row_idx, col_idx])
                                else float(abs_angle_error[frame_idx, row_idx, col_idx])
                            ),
                        }
                    )

    # One point per Hartmann spot: frame-wise median bias for direction and magnitude.
    per_spot_direction_deg = np.nanmedian(signed_angle_error, axis=0)
    per_spot_magnitude_px = np.nanmedian(mag_error, axis=0)
    median_abs_direction = _safe_finite_stat(abs_angle_error, np.median)
    p95_abs_direction = _safe_finite_stat(abs_angle_error, lambda x: np.percentile(x, 95))
    median_abs_magnitude = _safe_finite_stat(abs_mag_error, np.median)
    p95_abs_magnitude = _safe_finite_stat(abs_mag_error, lambda x: np.percentile(x, 95))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    _render_spot_metric_map(
        ax=axes[0],
        data_2d=per_spot_direction_deg,
        title='Pointwise Direction Error (median over frames)',
        cmap='RdBu_r',
        symmetric=True,
        colorbar_label='deg',
    )
    _render_spot_metric_map(
        ax=axes[1],
        data_2d=per_spot_magnitude_px,
        title='Pointwise Magnitude Error (median over frames)',
        cmap='RdBu_r',
        symmetric=True,
        colorbar_label='px',
    )
    plt.suptitle(
        "Pointwise Direction/Magnitude Error\n"
        f"median |dir|={0.0 if median_abs_direction is None else median_abs_direction:.2f} deg, "
        f"p95 |dir|={0.0 if p95_abs_direction is None else p95_abs_direction:.2f} deg, "
        f"median |mag|={0.0 if median_abs_magnitude is None else median_abs_magnitude:.4f} px, "
        f"p95 |mag|={0.0 if p95_abs_magnitude is None else p95_abs_magnitude:.4f} px",
        fontsize=13,
    )
    plt.tight_layout()
    fig_path = os.path.join(
        stat_dir,
        f"dpc_pointwise_direction_and_magnitude_error_{method}.png",
    )
    plt.savefig(fig_path, dpi=140)
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

    _save_per_frame_metrics_tables(
        stat_dir=stat_dir,
        method=method,
        per_frame_metrics=metrics,
        disp_metrics=result.get('disp_metrics'),
    )
    _save_per_frame_error_percentile_curve(
        stat_dir=stat_dir,
        method=method,
        per_frame_metrics=metrics,
    )
    _save_spot_metric_histograms(
        stat_dir=stat_dir,
        method=method,
        spot_products=result.get('dpc_spot_products'),
    )
    _save_truth_binned_dpc_statistics(
        stat_dir=stat_dir,
        method=method,
        spot_products=result.get('dpc_spot_products'),
    )
    _save_angle_error_statistics(
        stat_dir=stat_dir,
        method=method,
        spot_products=result.get('dpc_spot_products'),
    )
    _save_signal_vs_error_scale_statistics(
        stat_dir=stat_dir,
        method=method,
        spot_products=result.get('dpc_spot_products'),
    )
    _save_pointwise_direction_and_magnitude_error(
        stat_dir=stat_dir,
        method=method,
        spot_products=result.get('dpc_spot_products'),
    )
    _save_dpc_spot_trust_statistics(
        stat_dir=stat_dir,
        method=method,
        spot_products=result.get('dpc_spot_products'),
    )
    _save_truth_phase_context_statistics(
        stat_dir=stat_dir,
        method=method,
        spot_products=result.get('dpc_spot_products'),
    )
    
    print(f"  Statistical plots saved: {stat_dir}/")


def _prepare_phase_frame_stats(
    *,
    truth_phase_rad: np.ndarray,
    recovered_phase_rad: np.ndarray,
) -> dict:
    """Crop phase maps to overlap, remove piston/tip/tilt, and compute error statistics."""
    min_rows = min(truth_phase_rad.shape[0], recovered_phase_rad.shape[0])
    min_cols = min(truth_phase_rad.shape[1], recovered_phase_rad.shape[1])
    truth_phase_crop_rad = truth_phase_rad[:min_rows, :min_cols]
    recovered_phase_crop_rad = recovered_phase_rad[:min_rows, :min_cols]

    grid_x, grid_y = np.meshgrid(np.arange(min_cols), np.arange(min_rows))
    truth_phase_zero_mean_rad = remove_plane(truth_phase_crop_rad, grid_x, grid_y)
    recovered_phase_zero_mean_rad = remove_plane(recovered_phase_crop_rad, grid_x, grid_y)
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
    axes_row[0].set_title(f"Frame {frame_index} True Phase (rad)" if frame_index == 0 else f"Frame {frame_index} True Phase")
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
    axes_row[0].set_title(f"Frame {frame_index} True dX (px)" if frame_index == 0 else "")
    fig.colorbar(im0, ax=axes_row[0], fraction=0.046, pad=0.04)

    im1 = axes_row[1].imshow(recovered_displacement_x_px, cmap='RdBu_r', origin='lower', vmin=-vmax_dx, vmax=vmax_dx)
    axes_row[1].set_title("Rec dX" if frame_index == 0 else "")
    fig.colorbar(im1, ax=axes_row[1], fraction=0.046, pad=0.04)

    im2 = axes_row[2].imshow(displacement_error_x_px, cmap='RdBu_r', origin='lower', vmin=-vmax_err_dx, vmax=vmax_err_dx)
    axes_row[2].set_title(f"Err dX (RMSE={rmse_dx:.3f})" if frame_index == 0 else f"Err dX RMSE={rmse_dx:.3f}")
    fig.colorbar(im2, ax=axes_row[2], fraction=0.046, pad=0.04)

    im3 = axes_row[3].imshow(truth_displacement_y_px, cmap='RdBu_r', origin='lower', vmin=-vmax_dy, vmax=vmax_dy)
    axes_row[3].set_title(f"Frame {frame_index} True dY (px)" if frame_index == 0 else "")
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
    """保存所有帧的 True Phase, Recovered Phase, Phase Error 汇总大图。"""
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
    """保存所有帧的 True dX/dY, Recovered dX/dY, Error dX/dY 汇总六列大图。"""
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

    from analysis.shift_optimization import (
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
    from analysis.hartmann_analysis import run_analysis_report_from_dir

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
