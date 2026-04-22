#!/usr/bin/env python3
from __future__ import annotations
"""
Step 1: 光传输模拟 (Light Propagation Simulation)

模拟 Hartmann 光栅成像后在探测器上的叠加效果。
输出保存到 result/NNN/ 目录，供 Step 2 (recover.py) 使用。

Usage:
    python simulate.py                               # 默认输出到 result/xray/NNN
    python simulate.py --output-dir result/xray/001  # 指定输出目录
    python simulate.py --simple                      # 简化高斯光斑模式
"""

import os
import sys
import json
import glob
import argparse
from typing import List, Optional, Tuple
import numpy as np
MPLCONFIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib")
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import apply_plot_font_bump

apply_plot_font_bump()

from analysis.optimization_common import GridGeometry, assert_xy_points, build_grid_geometry_from_axes
import config_simulate as config


# =============================================================================
# Utility Functions
# =============================================================================

def _save_as_png(npy_path, data, *, allow_preview: bool = True):
    """将数组保存为 PNG 预览图。

    `allow_preview=False` 用于坐标表、标签表等非图像数据，避免生成无意义 PNG。
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

def get_next_run_dir(base_name: str = "result") -> str:
    """Get next available run directory (result/001, result/002, ...)."""
    os.makedirs(base_name, exist_ok=True)
    pattern = os.path.join(base_name, "[0-9][0-9][0-9]")
    existing = glob.glob(pattern)
    if not existing:
        return os.path.join(base_name, "001")
    numbers = []
    for d in existing:
        try:
            num = int(os.path.basename(d.rstrip(os.sep)))
            numbers.append(num)
        except ValueError:
            pass
    next_num = max(numbers) + 1 if numbers else 1
    return os.path.join(base_name, f"{next_num:03d}")


def _coerce_xy_pair(name: str, value) -> Tuple[float, float]:
    """校验并规范化 (dx, dy) 数值对。"""
    if not (hasattr(value, '__len__') and len(value) == 2):
        raise ValueError(f"{name} 必须是长度为 2 的 (dx, dy) 数值对")
    return (float(value[0]), float(value[1]))


def _build_offsets_from_step_intervals(cfg) -> Tuple[List[Tuple[float, float]], List[float], float, str, List[float], float, Optional[int]]:
    """根据总位移、固定角度和步间参数，生成逐 pulse 绝对位移轨迹。"""
    n_pulses = int(cfg.N_PULSES)
    expected = max(0, n_pulses - 1)
    raw_intervals = getattr(cfg, 'MOVEMENT_STEP_INTERVALS', None)
    if raw_intervals is None:
        raise ValueError("trajectory 模式下需要 MOVEMENT_STEP_INTERVALS")
    if len(raw_intervals) != expected:
        raise ValueError(
            f"MOVEMENT_STEP_INTERVALS 长度 ({len(raw_intervals)}) != N_PULSES-1 ({expected})")

    intervals = np.asarray(raw_intervals, dtype=float)
    if np.any(~np.isfinite(intervals)):
        raise ValueError("MOVEMENT_STEP_INTERVALS 只能包含有限数值")
    if np.any(intervals < 0):
        raise ValueError("MOVEMENT_STEP_INTERVALS 不能包含负数")

    interval_mode = str(getattr(cfg, 'MOVEMENT_STEP_INTERVALS_MODE', 'relative') or 'relative').lower()
    total_displacement_cfg = getattr(cfg, 'MOVEMENT_TOTAL_DISPLACEMENT_M', None)

    if interval_mode == 'relative':
        if total_displacement_cfg is None:
            raise ValueError("MOVEMENT_STEP_INTERVALS_MODE='relative' 需要 MOVEMENT_TOTAL_DISPLACEMENT_M")
        total_displacement = float(total_displacement_cfg)
        if total_displacement < 0:
            raise ValueError("MOVEMENT_TOTAL_DISPLACEMENT_M 不能为负数")
        if expected == 0 or total_displacement == 0.0:
            step_lengths = np.zeros(expected, dtype=float)
        else:
            interval_sum = float(np.sum(intervals))
            if interval_sum <= 0.0:
                raise ValueError("relative 模式下 MOVEMENT_STEP_INTERVALS 之和必须大于 0")
            step_lengths = intervals / interval_sum * total_displacement
    elif interval_mode == 'absolute':
        step_lengths = intervals.copy()
        total_displacement = float(np.sum(step_lengths))
        if total_displacement_cfg is not None:
            expected_total = float(total_displacement_cfg)
            if not np.isclose(total_displacement, expected_total, rtol=1e-9, atol=1e-15):
                raise ValueError(
                    "MOVEMENT_TOTAL_DISPLACEMENT_M 与 MOVEMENT_STEP_INTERVALS 的总和不一致")
    else:
        raise ValueError("MOVEMENT_STEP_INTERVALS_MODE 必须是 'relative' 或 'absolute'")

    base_angle_deg = float(cfg.ANGLE_DEG)
    jitter_std_deg = float(getattr(cfg, 'MOVEMENT_ANGLE_JITTER_STD_DEG', 0.0) or 0.0)
    if jitter_std_deg < 0.0:
        raise ValueError("MOVEMENT_ANGLE_JITTER_STD_DEG 不能为负数")
    jitter_seed_raw = getattr(cfg, 'MOVEMENT_ANGLE_JITTER_SEED', None)
    jitter_seed = None if jitter_seed_raw is None else int(jitter_seed_raw)
    rng = np.random.default_rng(jitter_seed)
    angle_jitter_deg = (
        rng.normal(loc=0.0, scale=jitter_std_deg, size=expected)
        if expected > 0 and jitter_std_deg > 0.0 else np.zeros(expected, dtype=float)
    )
    step_angles_deg = base_angle_deg + angle_jitter_deg
    step_angles_rad = np.deg2rad(step_angles_deg)
    step_dx = step_lengths * np.cos(step_angles_rad)
    step_dy = step_lengths * np.sin(step_angles_rad)
    cumulative_dx = np.concatenate(([0.0], np.cumsum(step_dx)))
    cumulative_dy = np.concatenate(([0.0], np.cumsum(step_dy)))
    offsets = [(float(dx), float(dy)) for dx, dy in zip(cumulative_dx, cumulative_dy)]
    total_path_length = float(np.sum(step_lengths))
    return (
        offsets,
        [float(x) for x in step_lengths],
        total_path_length,
        interval_mode,
        [float(base_angle_deg)] + [float(x) for x in step_angles_deg],
        float(jitter_std_deg),
        jitter_seed,
    )


def _build_base_movement_spec(
    *,
    mode: str,
    initial_offset_xy_m: Tuple[float, float],
) -> dict:
    """Create the shared movement-spec skeleton used by all full-mode paths."""
    return {
        "mode": mode,
        "initial_offset_xy_m": initial_offset_xy_m,
        "direction_deg": None,
        "translate_step_m": None,
        "pulse_offsets_xy_m": None,
        "metadata": {
            "movement_mode": mode,
            "movement_generated_from": mode,
            "movement_is_uniform": None,
            "movement_initial_offset_xy_m": list(initial_offset_xy_m),
            "movement_direction_deg": None,
            "movement_total_displacement_m": None,
            "movement_net_displacement_m": None,
            "movement_final_offset_xy_m": None,
            "movement_relative_final_offset_xy_m": None,
            "movement_uniform_step_m": None,
            "movement_uniform_angle_deg": None,
            "movement_step_intervals": None,
            "movement_step_intervals_mode": None,
            "movement_step_lengths_m": None,
            "movement_step_angles_deg": None,
            "movement_angle_jitter_std_deg": None,
            "movement_angle_jitter_seed": None,
            "movement_pulse_offsets_xy_m": None,
            "movement_pulse_absolute_offsets_xy_m": None,
        },
    }


def _relative_to_absolute_offsets(
    relative_offsets_xy_m: List[Tuple[float, float]],
    initial_offset_xy_m: np.ndarray,
) -> np.ndarray:
    """Convert relative per-pulse offsets to absolute offsets using the initial offset."""
    return np.asarray(relative_offsets_xy_m, dtype=float) + initial_offset_xy_m[np.newaxis, :]


def _attach_offsets_to_metadata(spec: dict, relative_offsets_xy_m: np.ndarray, absolute_offsets_xy_m: np.ndarray) -> None:
    """Populate shared relative/absolute offset fields in the movement metadata block."""
    spec["pulse_offsets_xy_m"] = [tuple(float(v) for v in row) for row in relative_offsets_xy_m]
    spec["metadata"]["movement_pulse_offsets_xy_m"] = relative_offsets_xy_m.tolist()
    spec["metadata"]["movement_pulse_absolute_offsets_xy_m"] = absolute_offsets_xy_m.tolist()
    spec["metadata"]["movement_final_offset_xy_m"] = [
        float(absolute_offsets_xy_m[-1, 0]),
        float(absolute_offsets_xy_m[-1, 1]),
    ]
    spec["metadata"]["movement_relative_final_offset_xy_m"] = [
        float(relative_offsets_xy_m[-1, 0]),
        float(relative_offsets_xy_m[-1, 1]),
    ]
    spec["metadata"]["movement_net_displacement_m"] = float(
        np.hypot(relative_offsets_xy_m[-1, 0], relative_offsets_xy_m[-1, 1])
    )


def _populate_none_movement_spec(spec: dict, *, cfg, n_pulses: int, initial_offset_xy_m: np.ndarray) -> dict:
    """Fill the full-movement spec for the stationary-mask case."""
    zero_offsets_xy_m = np.zeros((n_pulses, 2), dtype=float)
    absolute_offsets_xy_m = zero_offsets_xy_m + initial_offset_xy_m[np.newaxis, :]
    _attach_offsets_to_metadata(spec, zero_offsets_xy_m, absolute_offsets_xy_m)
    spec["metadata"]["movement_is_uniform"] = True
    spec["metadata"]["movement_total_displacement_m"] = 0.0
    spec["metadata"]["movement_uniform_step_m"] = 0.0
    spec["metadata"]["movement_uniform_angle_deg"] = float(getattr(cfg, 'ANGLE_DEG', 0.0))
    spec["metadata"]["movement_step_lengths_m"] = [0.0] * max(0, n_pulses - 1)
    spec["metadata"]["movement_step_angles_deg"] = [float(getattr(cfg, 'ANGLE_DEG', 0.0))] * n_pulses
    return spec


def _populate_translation_movement_spec(spec: dict, *, cfg, n_pulses: int, initial_offset_xy_m: np.ndarray) -> dict:
    """Fill the full-movement spec for constant-step translation."""
    spec["mode"] = 'translation'
    spec["direction_deg"] = float(cfg.ANGLE_DEG)
    spec["translate_step_m"] = float(cfg.STEP_M)
    ang_rad = np.deg2rad(float(cfg.ANGLE_DEG))
    step_dx = float(cfg.STEP_M) * float(np.cos(ang_rad))
    step_dy = float(cfg.STEP_M) * float(np.sin(ang_rad))
    pulse_indices = np.arange(n_pulses, dtype=float)
    relative_offsets_xy_m = np.column_stack([pulse_indices * step_dx, pulse_indices * step_dy])
    absolute_offsets_xy_m = relative_offsets_xy_m + initial_offset_xy_m[np.newaxis, :]
    _attach_offsets_to_metadata(spec, relative_offsets_xy_m, absolute_offsets_xy_m)
    spec["metadata"]["movement_mode"] = 'translation'
    spec["metadata"]["movement_generated_from"] = 'translation'
    spec["metadata"]["movement_is_uniform"] = True
    spec["metadata"]["movement_direction_deg"] = float(cfg.ANGLE_DEG)
    spec["metadata"]["movement_total_displacement_m"] = float(abs(cfg.STEP_M) * max(0, n_pulses - 1))
    spec["metadata"]["movement_uniform_step_m"] = float(cfg.STEP_M)
    spec["metadata"]["movement_uniform_angle_deg"] = float(cfg.ANGLE_DEG)
    spec["metadata"]["movement_step_lengths_m"] = [float(cfg.STEP_M)] * max(0, n_pulses - 1)
    spec["metadata"]["movement_step_angles_deg"] = [float(cfg.ANGLE_DEG)] * n_pulses
    return spec


def _populate_trajectory_movement_spec(spec: dict, *, cfg, initial_offset_xy_m: np.ndarray) -> dict:
    """Fill the full-movement spec for an explicit per-step trajectory."""
    raw_intervals = getattr(cfg, 'MOVEMENT_STEP_INTERVALS', None)
    if raw_intervals is None:
        raise ValueError("MOVEMENT_MODE='trajectory' 需要 MOVEMENT_STEP_INTERVALS")

    (
        relative_offsets,
        step_lengths,
        total_displacement,
        interval_mode,
        step_angles_deg,
        jitter_std_deg,
        jitter_seed,
    ) = _build_offsets_from_step_intervals(cfg)
    relative_offsets_xy_m = np.asarray(relative_offsets, dtype=float)
    absolute_offsets_xy_m = _relative_to_absolute_offsets(relative_offsets, initial_offset_xy_m)
    _attach_offsets_to_metadata(spec, relative_offsets_xy_m, absolute_offsets_xy_m)

    spec["metadata"]["movement_generated_from"] = (
        "step_intervals_with_angle_jitter" if jitter_std_deg > 0.0 else "step_intervals"
    )
    spec["metadata"]["movement_direction_deg"] = float(cfg.ANGLE_DEG)
    spec["metadata"]["movement_total_displacement_m"] = float(total_displacement)
    spec["metadata"]["movement_step_intervals"] = [float(x) for x in raw_intervals]
    spec["metadata"]["movement_step_intervals_mode"] = interval_mode
    spec["metadata"]["movement_step_lengths_m"] = step_lengths
    spec["metadata"]["movement_step_angles_deg"] = step_angles_deg
    spec["metadata"]["movement_angle_jitter_std_deg"] = float(jitter_std_deg)
    spec["metadata"]["movement_angle_jitter_seed"] = jitter_seed

    is_uniform = (
        len(step_lengths) == 0 or (
            np.allclose(step_lengths, step_lengths[0], rtol=1e-9, atol=1e-15) and
            np.allclose(
                step_angles_deg[1:],
                step_angles_deg[1] if len(step_angles_deg) > 1 else float(cfg.ANGLE_DEG),
                rtol=1e-9,
                atol=1e-12,
            )
        )
    )
    spec["metadata"]["movement_is_uniform"] = bool(is_uniform)
    if is_uniform:
        uniform_step = float(step_lengths[0]) if len(step_lengths) > 0 else 0.0
        uniform_angle = float(step_angles_deg[1]) if len(step_angles_deg) > 1 else float(cfg.ANGLE_DEG)
        spec["metadata"]["movement_uniform_step_m"] = uniform_step
        spec["metadata"]["movement_uniform_angle_deg"] = uniform_angle
    return spec



def _populate_rotation_movement_spec(spec: dict, *, cfg, n_pulses: int, initial_offset_xy_m: np.ndarray) -> dict:
    """Fill the full-movement spec for rotation (spinning disc) mode.

    The Hartmann mask is mounted on a rotating disc at radius R from the
    rotation center.  Each pulse sees the mask rotated by a small angle
    dtheta.  We compute the exact (dx, dy) that each grid hole undergoes
    and store them as equivalent linear offsets so that the recovery
    pipeline (recover.py) needs zero changes.

    Physical parameters are read from *cfg*:
        ROTATION_RADIUS_M        – disc radius (beam hit point), metres
        ROTATION_ANGULAR_VELOCITY – rad/s (None → auto from STEP_M / R)
        ROTATION_GRID_TILT_DEG   – angle between grid X axis and tangent
        ROTATION_DIRECTION       – +1 (CCW) or -1 (CW)
    """
    R = float(getattr(cfg, 'ROTATION_RADIUS_M', 50e-3))
    direction = int(getattr(cfg, 'ROTATION_DIRECTION', 1))
    grid_tilt_deg = float(getattr(cfg, 'ROTATION_GRID_TILT_DEG', 19.0))

    # Determine angular step per pulse
    omega = getattr(cfg, 'ROTATION_ANGULAR_VELOCITY', None)
    if omega is not None:
        omega = float(omega)
        rep_rate = 1.0 / float(getattr(cfg, 'EXPOSURE_TIME_S', 1.0)) * n_pulses  # rough
        dtheta = omega / max(rep_rate, 1.0)
    else:
        # Auto: derive from desired tangential step STEP_M
        step_m = float(getattr(cfg, 'STEP_M', 7e-6))
        dtheta = step_m / R  # exact: arc length = R * dtheta

    dtheta *= direction

    # Compute per-pulse rotation angles (relative to pulse 0)
    pulse_angles_rad = np.array([k * dtheta for k in range(n_pulses)], dtype=float)

    # Convert rotation to equivalent linear offsets in the beam frame.
    # The tangent direction in the beam frame is oriented at grid_tilt_deg
    # relative to the X-axis.
    tilt_rad = np.deg2rad(grid_tilt_deg)

    # Exact trigonometric conversion (NOT small-angle approximation):
    #   tangential displacement = R * sin(dtheta_k)
    #   radial displacement    = R * (1 - cos(dtheta_k))
    tangential = R * np.sin(pulse_angles_rad)  # along tangent
    radial = R * (1.0 - np.cos(pulse_angles_rad))  # perpendicular to tangent

    # Project tangential & radial into beam-frame X/Y
    dx_m = tangential * np.cos(tilt_rad) - radial * np.sin(tilt_rad)
    dy_m = tangential * np.sin(tilt_rad) + radial * np.cos(tilt_rad)

    relative_offsets_xy_m = np.column_stack([dx_m, dy_m])
    absolute_offsets_xy_m = relative_offsets_xy_m + initial_offset_xy_m[np.newaxis, :]

    # Populate spec
    spec["mode"] = 'rotation'
    _attach_offsets_to_metadata(spec, relative_offsets_xy_m, absolute_offsets_xy_m)

    # Step lengths between consecutive pulses
    step_lengths = [float(np.hypot(
        relative_offsets_xy_m[k+1, 0] - relative_offsets_xy_m[k, 0],
        relative_offsets_xy_m[k+1, 1] - relative_offsets_xy_m[k, 1],
    )) for k in range(n_pulses - 1)]
    total_disp = float(np.hypot(relative_offsets_xy_m[-1, 0], relative_offsets_xy_m[-1, 1]))

    spec["metadata"]["movement_mode"] = 'rotation'
    spec["metadata"]["movement_generated_from"] = 'rotation'
    spec["metadata"]["movement_is_uniform"] = True
    spec["metadata"]["movement_direction_deg"] = float(grid_tilt_deg)
    spec["metadata"]["movement_total_displacement_m"] = total_disp
    spec["metadata"]["movement_uniform_step_m"] = float(step_lengths[0]) if step_lengths else 0.0
    spec["metadata"]["movement_uniform_angle_deg"] = float(grid_tilt_deg)
    spec["metadata"]["movement_step_lengths_m"] = step_lengths
    step_angles_deg = [float(grid_tilt_deg)] * n_pulses
    spec["metadata"]["movement_step_angles_deg"] = step_angles_deg

    # Rotation-specific metadata
    spec["metadata"]["rotation_radius_m"] = float(R)
    spec["metadata"]["rotation_dtheta_rad"] = float(dtheta)
    spec["metadata"]["rotation_direction"] = int(direction)
    spec["metadata"]["rotation_grid_tilt_deg"] = float(grid_tilt_deg)
    spec["metadata"]["rotation_pulse_angles_rad"] = pulse_angles_rad.tolist()

    return spec


def _resolve_full_movement_spec(cfg) -> dict:
    """解析 full simulation 的运动配置，并生成底层 trajectory 所需的绝对 offsets。"""
    mv_mode = str(getattr(cfg, 'MOVEMENT_MODE', 'translation') or 'translation').lower()
    n_pulses = int(getattr(cfg, 'N_PULSES', 1))
    initial_offset_xy_m = _coerce_xy_pair(
        'MOVEMENT_INITIAL_OFFSET_XY_M',
        getattr(cfg, 'MOVEMENT_INITIAL_OFFSET_XY_M', (0.0, 0.0)),
    )
    initial_offset_arr = np.asarray(initial_offset_xy_m, dtype=float)
    spec = _build_base_movement_spec(
        mode=mv_mode,
        initial_offset_xy_m=initial_offset_xy_m,
    )

    if mv_mode == 'rotation':
        return _populate_rotation_movement_spec(
            spec,
            cfg=cfg,
            n_pulses=n_pulses,
            initial_offset_xy_m=initial_offset_arr,
        )

    if mv_mode == 'trajectory':
        return _populate_trajectory_movement_spec(
            spec,
            cfg=cfg,
            initial_offset_xy_m=initial_offset_arr,
        )

    if mv_mode == 'none':
        return _populate_none_movement_spec(
            spec,
            cfg=cfg,
            n_pulses=n_pulses,
            initial_offset_xy_m=initial_offset_arr,
        )

    return _populate_translation_movement_spec(
        spec,
        cfg=cfg,
        n_pulses=n_pulses,
        initial_offset_xy_m=initial_offset_arr,
    )


def _build_simple_frame_centers(
    ideal_centers: np.ndarray,
    n_frames: int,
    shift_per_frame: Tuple[float, float],
) -> np.ndarray:
    """Build per-frame ideal spot centers for simple mode using (x, y) convention."""
    centers = assert_xy_points(ideal_centers, name="ideal_centers")
    dx_pf, dy_pf = (float(shift_per_frame[0]), float(shift_per_frame[1]))
    frame_centers = np.zeros((int(n_frames), len(centers), 2), dtype=np.float64)
    for t in range(int(n_frames)):
        frame_centers[t] = centers + np.array([t * dx_pf, t * dy_pf], dtype=np.float64)
    return frame_centers


def _build_full_mode_grid_metadata(
    image_shape: Tuple[int, int],
    *,
    pixel_size_m: float,
    pitch_m: float,
    window_size_m: Optional[float],
    gt_disp: Optional[dict] = None,
) -> GridGeometry:
    """Build full-mode ideal centers and explicit detector-grid geometry."""
    h, w = image_shape

    centers_x_m = None
    centers_y_m = None
    if gt_disp is not None:
        centers_x_m = gt_disp.get("centers_x_m")
        centers_y_m = gt_disp.get("centers_y_m")

    if centers_x_m is not None and centers_y_m is not None:
            x_coords_px = np.asarray(centers_x_m, dtype=np.float64) / float(pixel_size_m) + w / 2.0
            y_coords_px = np.asarray(centers_y_m, dtype=np.float64) / float(pixel_size_m) + h / 2.0
    else:
        spacing = float(pitch_m)
        if window_size_m is not None:
            grid_count = max(1, int(round(float(window_size_m) / spacing)))
            coords = (np.arange(grid_count) - (grid_count - 1) / 2.0) * spacing
            det_w = w * float(pixel_size_m)
            det_h = h * float(pixel_size_m)
            x_coords_px = coords[np.abs(coords) < det_w / 2.0] / float(pixel_size_m) + w / 2.0
            y_coords_px = coords[np.abs(coords) < det_h / 2.0] / float(pixel_size_m) + h / 2.0
        else:
            spacing_px = float(pitch_m) / float(pixel_size_m)
            cols = max(1, int(round((w * float(pixel_size_m)) / float(pitch_m))))
            rows = max(1, int(round((h * float(pixel_size_m)) / float(pitch_m))))
            x_coords_px = (np.arange(cols) - (cols - 1) / 2.0) * spacing_px + w / 2.0
            y_coords_px = (np.arange(rows) - (rows - 1) / 2.0) * spacing_px + h / 2.0

    return build_grid_geometry_from_axes(x_coords_px, y_coords_px)


def _print_full_simulation_header(wavelength_m: float) -> float:
    """Print the standard full-simulation banner and return the Fresnel number."""
    print("=" * 70)
    print("  STEP 1: FULL WAVE-OPTICS SIMULATION")
    print(f"  Mask: {config.MASK_TYPE} | Propagation: {config.PROPAGATION_METHOD}")
    print("=" * 70)
    print(f"  Wavelength: {wavelength_m * 1e10:.3f} Å")

    aperture_radius_m = config.HARTMANN_HOLE_DIAMETER_M / 2.0
    propagation_distance_m = config.Z_PROP_M
    fresnel_num = (aperture_radius_m**2) / (propagation_distance_m * wavelength_m)
    regime = "Far-field (Fraunhofer)" if fresnel_num < 1.0 else "Near-field (Fresnel)"
    print(
        f"  Fresnel Number: {fresnel_num:.4f} "
        f"(a={aperture_radius_m*1e6:.1f}um, L={propagation_distance_m*1e3:.1f}mm) -> {regime}"
    )
    return float(fresnel_num)


def _build_full_simulation_components(wavelength_m: float):
    """Build simulator config objects and the resolved movement specification."""
    from simulation.hartmann_simulator import (
        PulseConfig,
        RotationConfig,
        OpticalConfig,
        GratingConfig,
        DetectorConfig,
        OutputConfig,
        MovementConfig,
    )

    pulse_cfg = PulseConfig(n_pulses=config.N_PULSES)
    rotation_cfg = RotationConfig(mode="velocity", step_angle_deg=0.0, angular_velocity_deg_s=0.0)
    optical_cfg = OpticalConfig(
        photon_energy_ev=config.PHOTON_ENERGY_EV,
        window_size=config.WINDOW_SIZE_M,
        grid_points=config.GRID_POINTS,
        z_propagation=config.Z_PROP_M,
        propagation_method=config.PROPAGATION_METHOD,
        source_mode=config.SOURCE_TYPE,
    )
    grating_cfg = GratingConfig(
        spacing=config.HARTMANN_PITCH_M,
        aperture=config.HARTMANN_HOLE_DIAMETER_M,
    )
    detector_cfg = DetectorConfig(
        pixel_size=config.DETECTOR_PIXEL_M,
        nx=config.DETECTOR_NX,
        ny=config.DETECTOR_NY,
        simulate_effects=config.ENABLE_DETECTOR_EFFECTS,
    )
    movement_spec = _resolve_full_movement_spec(config)
    # Build rotation-specific fields if applicable
    rot_meta = movement_spec.get("metadata", {})
    rot_radius = float(rot_meta.get("rotation_radius_m", 50e-3) or 50e-3)
    rot_dtheta = float(rot_meta.get("rotation_dtheta_rad", 0.0) or 0.0)
    rot_tilt = float(rot_meta.get("rotation_grid_tilt_deg", 0.0) or 0.0)
    rot_angles = rot_meta.get("rotation_pulse_angles_rad", None)

    movement_cfg = MovementConfig(
        mode=str(movement_spec["mode"]),
        plate1_direction_deg=float(movement_spec["direction_deg"] or 0.0),
        plate1_translate_step_m=float(movement_spec["translate_step_m"] or 0.0),
        plate1_initial_offset_m=movement_spec["initial_offset_xy_m"],
        plate1_pulse_offsets_xy_m=movement_spec["pulse_offsets_xy_m"],
        rotation_radius_m=rot_radius,
        rotation_dtheta_rad=rot_dtheta,
        rotation_grid_tilt_deg=rot_tilt,
        rotation_pulse_angles_rad=rot_angles,
    )
    output_cfg = OutputConfig(show_progress=True)
    return (
        pulse_cfg,
        rotation_cfg,
        optical_cfg,
        grating_cfg,
        detector_cfg,
        movement_spec,
        movement_cfg,
        output_cfg,
    )


def _apply_wfe_preset(optical_cfg) -> None:
    """Apply an optional WFE preset to the optical configuration."""
    if not config.WFE_PRESET:
        return
    try:
        from simulation.wfe_generator import get_wfe_preset

        wfe_params = get_wfe_preset(config.WFE_PRESET)
        optical_cfg.wfe_type = wfe_params.get('type')
        optical_cfg.wfe_params = wfe_params.get('params')
        optical_cfg.wfe_rms_rad = wfe_params.get('rms_rad')
        print(f"  WFE Preset: {config.WFE_PRESET}")
    except (ImportError, KeyError) as e:
        print(f"  Warning: WFE preset '{config.WFE_PRESET}' not available: {e}")


def _print_movement_spec_summary(movement_spec: dict) -> None:
    """Print the resolved full-mode movement summary."""
    if movement_spec["mode"] == 'trajectory':
        src = movement_spec["metadata"]["movement_generated_from"]
        print(f"  Movement: trajectory ({src})")
        if movement_spec["metadata"]["movement_direction_deg"] is not None:
            print(f"    Direction: {movement_spec['metadata']['movement_direction_deg']:.3f} deg")
        if movement_spec["metadata"]["movement_total_displacement_m"] is not None:
            print(f"    Total displacement: {movement_spec['metadata']['movement_total_displacement_m']*1e6:.3f} um")
        if movement_spec["metadata"]["movement_net_displacement_m"] is not None:
            print(f"    Net displacement: {movement_spec['metadata']['movement_net_displacement_m']*1e6:.3f} um")
        if movement_spec["metadata"]["movement_angle_jitter_std_deg"] not in (None, 0.0):
            print(f"    Angle jitter std: {movement_spec['metadata']['movement_angle_jitter_std_deg']:.6f} deg")
        print(f"    Uniform movement: {bool(movement_spec['metadata']['movement_is_uniform'])}")
    elif movement_spec["mode"] == 'rotation':
        meta = movement_spec["metadata"]
        R_mm = float(meta.get("rotation_radius_m", 0)) * 1e3
        dtheta_urad = float(meta.get("rotation_dtheta_rad", 0)) * 1e6
        tilt_deg = float(meta.get("rotation_grid_tilt_deg", 0))
        step_um = float(meta.get("rotation_radius_m", 0)) * abs(float(meta.get("rotation_dtheta_rad", 0))) * 1e6
        print(f"  Movement: rotation (R={R_mm:.1f} mm, dθ={dtheta_urad:.2f} µrad/pulse, tangential step={step_um:.3f} µm)")
        print(f"    Grid tilt: {tilt_deg:.1f} deg")
        if meta.get("movement_total_displacement_m") is not None:
            print(f"    Total displacement: {float(meta['movement_total_displacement_m'])*1e6:.3f} µm")
    elif movement_spec["mode"] == 'translation':
        print(
            "  Movement: translation "
            f"(angle={movement_spec['direction_deg']:.3f} deg, "
            f"step={movement_spec['translate_step_m']*1e6:.3f} um/pulse)"
        )
    else:
        print("  Movement: none")


def _configure_full_simulator_aberration(simulator) -> None:
    """Apply the configured phase-aberration mode to the simulator instance."""
    aberr_mode = str(config.PHASE_ABERRATION_MODE).lower()
    if aberr_mode in ('per_pulse', 'static'):
        simulator._aberration_mode = aberr_mode
        simulator._aberr_max_order = config.PHASE_ABERRATION_MAX_ORDER
        simulator._aberr_range = config.PHASE_ABERRATION_RANGE
        print(f"  Aberration mode: {aberr_mode} (max_order={config.PHASE_ABERRATION_MAX_ORDER})")
    else:
        print(f"  Aberration mode: none")


def _save_full_integrated_outputs(output_dir: str, result: dict):
    """Save accumulated / ideal / reference-ideal outputs for full-mode simulation."""
    integrated_sampled = result.get("integrated_sampled")
    if integrated_sampled is not None:
        _p = os.path.join(output_dir, "accumulated.npy")
        np.save(_p, integrated_sampled.astype(np.float32))
        _save_as_png(_p, integrated_sampled)
        if config.SAVE_ACCUMULATED_PNG:
            plt.figure(figsize=(6, 6))
            plt.imshow(integrated_sampled, cmap='magma', origin='lower')
            plt.title(f"Accumulated ({config.N_PULSES} pulses)")
            plt.colorbar(label='Counts')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "accumulated.png"), dpi=150)
            plt.close()
        print(f"  Accumulated image: {integrated_sampled.shape}")

    frames_ideal = result.get("frames_ideal")
    integrated_ideal = result.get("integrated_ideal")
    if integrated_ideal is not None:
        _p = os.path.join(output_dir, "ideal_accumulated.npy")
        np.save(_p, integrated_ideal.astype(np.float64))
        _save_as_png(_p, integrated_ideal)
        print(f"  Ideal accumulated: {integrated_ideal.shape}")

    if frames_ideal is not None and len(frames_ideal) > 0:
        _p = os.path.join(output_dir, "reference_ideal.npy")
        np.save(_p, np.asarray(frames_ideal[0], dtype=np.float64))
        _save_as_png(_p, frames_ideal[0])
        print(f"  Reference ideal: {np.asarray(frames_ideal[0]).shape} (pulse 0)")

    return integrated_sampled, integrated_ideal, frames_ideal


def _save_full_truth_and_grid_outputs(output_dir: str, result: dict, detector_shape: Tuple[int, int]):
    """Save WFE / phase screens / GT displacement outputs and explicit grid metadata."""
    wfe_map = result.get("wfe_phase_map_rad")
    if wfe_map is not None:
        _p = os.path.join(output_dir, "wfe_phase_map.npy")
        np.save(_p, wfe_map)
        _save_as_png(_p, wfe_map)
        print(f"  WFE phase map: {wfe_map.shape}")

    phase_screens = result.get("phase_screens")
    if phase_screens is not None:
        _p = os.path.join(output_dir, "phase_screens.npy")
        np.save(_p, phase_screens)
        _save_as_png(_p, phase_screens)
        print(f"  Phase screens: {phase_screens.shape}")

    gt_disp = result.get("gt_displacements")
    has_gt = False
    if gt_disp is not None:
        _p = os.path.join(output_dir, "gt_displacements.npz")
        np.savez(_p, **gt_disp)
        for _k, _v in gt_disp.items():
            if isinstance(_v, np.ndarray) and _v.ndim >= 2:
                _save_as_png(_p, _v)
                break
        has_gt = True
        n_gt_frames = gt_disp['displacement_x_px'].shape[0]
        print(f"  GT displacements: {gt_disp['grid_rows']}×{gt_disp['grid_cols']} sub-apertures, {n_gt_frames} frames")
        dx_px = gt_disp['displacement_x_px']
        dy_px = gt_disp['displacement_y_px']
        total_disp = np.sqrt(dx_px**2 + dy_px**2)
        print(f"  GT displacement stats: mean={np.mean(total_disp):.3f} px, max={np.max(total_disp):.3f} px")

    full_grid_geometry = _build_full_mode_grid_metadata(
        detector_shape,
        pixel_size_m=float(config.DETECTOR_PIXEL_M),
        pitch_m=float(config.HARTMANN_PITCH_M),
        window_size_m=float(config.WINDOW_SIZE_M),
        gt_disp=gt_disp,
    )
    _p = os.path.join(output_dir, "ideal_centers.npy")
    np.save(_p, full_grid_geometry.centers_xy_px.astype(np.float64))
    _save_as_png(_p, full_grid_geometry.centers_xy_px, allow_preview=False)
    _p = os.path.join(output_dir, "spot_grid_indices.npy")
    np.save(_p, full_grid_geometry.grid_indices_rowcol.astype(np.int32))
    print(
        "  Saved full-mode grid metadata: "
        f"{len(full_grid_geometry.centers_xy_px)} centers, "
        f"grid={full_grid_geometry.grid_rows}x{full_grid_geometry.grid_cols}"
    )
    return wfe_map, phase_screens, gt_disp, has_gt, full_grid_geometry


def _save_full_per_frame_outputs(output_dir: str, result: dict, frames_ideal) -> None:
    """Save optional aberrated / ideal per-frame full-mode outputs."""
    frames_sampled = result.get("frames_sampled")
    if config.SAVE_EACH_FRAME and frames_sampled is not None:
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        for i, frame in enumerate(frames_sampled):
            _p = os.path.join(frames_dir, f"frame_{i:03d}.npy")
            np.save(_p, frame)
            _save_as_png(_p, frame)
        if frames_ideal is not None:
            for i, frame in enumerate(frames_ideal):
                _p = os.path.join(frames_dir, f"frame_{i:03d}_ideal.npy")
                np.save(_p, frame)
                _save_as_png(_p, frame)
        print(f"  Saved {len(frames_sampled)} frames (aberrated + ideal)")


def _build_full_simulation_params(
    *,
    wavelength_m: float,
    fresnel_num: float,
    movement_spec: dict,
    wfe_map,
    has_gt: bool,
    full_grid_geometry: GridGeometry,
    has_reference_ideal: bool,
) -> dict:
    """Build the full-mode simulation_params payload consumed by Step 2."""
    sim_params = {
        "mode": "full",
        "photon_energy_ev": float(config.PHOTON_ENERGY_EV),
        "wavelength_m": float(wavelength_m),
        "z_prop_m": float(config.Z_PROP_M),
        "pixel_size_m": float(config.DETECTOR_PIXEL_M),
        "pitch_m": float(config.HARTMANN_PITCH_M),
        "hole_diameter_m": float(config.HARTMANN_HOLE_DIAMETER_M),
        "fresnel_number": float(fresnel_num),
        "n_pulses": config.N_PULSES,
        "detector_nx": config.DETECTOR_NX,
        "detector_ny": config.DETECTOR_NY,
        "window_size_m": float(config.WINDOW_SIZE_M),
        "grid_points": config.GRID_POINTS,
        "propagation_method": config.PROPAGATION_METHOD,
        "source_type": config.SOURCE_TYPE,
        "mask_type": config.MASK_TYPE,
        "angle_deg": float(config.ANGLE_DEG),
        "step_m": float(config.STEP_M),
        "wfe_preset": config.WFE_PRESET,
        "phase_aberration_mode": config.PHASE_ABERRATION_MODE,
        "has_wfe_ground_truth": wfe_map is not None,
        "has_gt_displacements": has_gt,
        "has_ideal_centers": True,
        "has_spot_grid_indices": True,
        "spot_grid_rows": int(full_grid_geometry.grid_rows),
        "spot_grid_cols": int(full_grid_geometry.grid_cols),
        "has_reference_ideal": has_reference_ideal,
        "ideal_reference_pulse_index": 0 if has_reference_ideal else None,
        **movement_spec["metadata"],
    }

    if bool(movement_spec["metadata"].get("movement_is_uniform")):
        if movement_spec["metadata"].get("movement_uniform_step_m") is not None:
            sim_params["step_m"] = float(movement_spec["metadata"]["movement_uniform_step_m"])
        if movement_spec["metadata"].get("movement_uniform_angle_deg") is not None:
            sim_params["angle_deg"] = float(movement_spec["metadata"]["movement_uniform_angle_deg"])
    else:
        sim_params["step_m"] = None
        sim_params["angle_deg"] = None
    return sim_params


# =============================================================================
# Step 1a: Simple Gaussian Spot Simulation (Original Method)
# =============================================================================

def run_simple_mode(output_dir: str):
    """
    简化高斯光斑模拟 — 仅生成数据，不做恢复。
    
    生成模拟的 Hartmann 光斑叠加图像和 ground truth 数据，
    保存到 output_dir，供 Step 2 (recover.py) 或优化求解器使用。
    """
    from simulation.simple_spots import (
        create_hartmann_grid_centers,
        compute_shift_per_frame,
        filter_valid_centers,
        generate_ground_truth_data,
        fast_render_gaussian_spots,
    )

    print("=" * 70)
    print("  STEP 1: SIMPLE GAUSSIAN SPOT SIMULATION")
    print("  (仅生成数据，恢复请运行 recover.py)")
    print("=" * 70)

    image_shape = config.SIMPLE_SPOT_IMAGE_SHAPE
    n_frames = config.N_PULSES
    spacing = config.SIMPLE_SPOT_SPACING
    spot_size = config.SIMPLE_SPOT_SIZE
    margin = config.SIMPLE_SPOT_MARGIN
    shift_mag = config.SIMPLE_SPOT_SHIFT_MAGNITUDE
    shift_angle = config.SIMPLE_SPOT_SHIFT_ANGLE_DEG
    wavefront_rms = config.SIMPLE_SPOT_WAVEFRONT_RMS
    noise_level = config.SIMPLE_SPOT_NOISE_LEVEL

    # 计算每帧全局位移
    shift_per_frame = compute_shift_per_frame(shift_mag, shift_angle)
    print(f"  Image: {image_shape}, Frames: {n_frames}")
    print(f"  Spot size: {spot_size}, Spacing: {spacing}")
    print(f"  Shift/frame: dx={shift_per_frame[0]:.2f}, dy={shift_per_frame[1]:.2f} px")

    # 生成光斑网格
    all_centers = create_hartmann_grid_centers(image_shape, spacing=spacing)
    ideal_centers, _ = filter_valid_centers(
        all_centers, image_shape, n_frames, shift_per_frame,
        margin=margin + spot_size
    )
    print(f"  Spots: {len(ideal_centers)} / {len(all_centers)} (after boundary filter)")

    # 生成带像差的叠加图 + ground truth
    gt_frames, accumulated, gt_shifts = generate_ground_truth_data(
        image_shape, ideal_centers, n_frames, shift_per_frame, spot_size,
        max_local_shift=config.SIMPLE_SPOT_MAX_SHIFT,
        wavefront_rms=wavefront_rms,
    )

    # 加噪声
    if noise_level > 0:
        accumulated_noisy = accumulated + noise_level * np.random.normal(size=accumulated.shape)
    else:
        accumulated_noisy = accumulated

    # 生成无像差理想叠加图
    Y_grid, X_grid = np.indices(image_shape)
    ideal_accumulated = np.zeros(image_shape)
    frame_centers = _build_simple_frame_centers(ideal_centers, n_frames, shift_per_frame)
    for centers_t in frame_centers:
        ideal_accumulated += fast_render_gaussian_spots(
            image_shape, centers_t, spot_size, Y_grid, X_grid
        )

    # =========================================================================
    # 保存输出 (与 full 模式格式一致，方便 recover.py 加载)
    # =========================================================================
    _p = os.path.join(output_dir, "accumulated.npy")
    np.save(_p, accumulated_noisy.astype(np.float64))
    _save_as_png(_p, accumulated_noisy)
    _p = os.path.join(output_dir, "ideal_accumulated.npy")
    np.save(_p, ideal_accumulated.astype(np.float32))
    _save_as_png(_p, ideal_accumulated)

    # WFE ground truth — 用 gt_shifts 的平均位移场近似
    # (简化模式没有真正的 WFE 相位图，但保存位移场供对比)
    _p = os.path.join(output_dir, "gt_shifts.npy")
    np.save(_p, gt_shifts)
    _save_as_png(_p, gt_shifts, allow_preview=False)

    # 保存参数 (与 full 模式格式一致)
    sim_params = {
        "mode": "simple",
        "wavelength_m": 1.0,  # 简化模式无真实波长
        "z_prop_m": 1.0,      # 简化模式无真实传播距离
        "pixel_size_m": 1.0,  # 简化模式归一化为像素
        "pitch_m": float(spacing),
        "n_pulses": n_frames,
        "detector_nx": image_shape[1],
        "detector_ny": image_shape[0],
        "spot_size": spot_size,
        "shift_magnitude": shift_mag,
        "shift_angle_deg": shift_angle,
        "shift_per_frame_dx": shift_per_frame[0],
        "shift_per_frame_dy": shift_per_frame[1],
        "wavefront_rms": wavefront_rms,
        "noise_level": noise_level,
        "n_spots": len(ideal_centers),
    }
    with open(os.path.join(output_dir, "simulation_params.json"), "w") as fp:
        json.dump(sim_params, fp, indent=2)

    # 保存光斑中心位置
    _p = os.path.join(output_dir, "ideal_centers.npy")
    np.save(_p, ideal_centers)
    _save_as_png(_p, ideal_centers, allow_preview=False)

    # 预览图
    if config.SAVE_ACCUMULATED_PNG:
        plt.figure(figsize=(6, 6))
        plt.imshow(accumulated_noisy, cmap='magma', origin='lower')
        plt.title(f"Simple Mode: {n_frames} frames, {len(ideal_centers)} spots")
        plt.colorbar(label='Intensity')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "accumulated.png"), dpi=150)
        plt.close()

    # 保存单帧 (可选)
    if config.SAVE_EACH_FRAME:
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        for i, frame in enumerate(gt_frames):
            _p = os.path.join(frames_dir, f"frame_{i:03d}.npy")
            np.save(_p, frame)
            _save_as_png(_p, frame)
        print(f"  Saved {n_frames} individual frames")

    print(f"\n✓ Simple mode complete. Results saved to: {output_dir}")
    print(f"  To run phase recovery: python recover.py {output_dir}")
    return


# =============================================================================
# Step 1b: Full Wave-Optics Simulation
# =============================================================================

def run_full_simulation(output_dir: str):
    """
    完整波动光学 Hartmann 仿真。
    
    输出文件:
      accumulated.npy           - 叠加探测器图像
      ideal_accumulated.npy     - 无像差理想叠加图
      wfe_phase_map.npy         - 输入 WFE 相位 (ground truth)
      simulation_params.json    - 完整仿真参数 (Step 2 需要)
      frames/frame_NNN.npy      - 单脉冲帧（可选）
    """
    from simulation.hartmann_simulator import PulseSuperpositionSimulator
    from simulation.wave_optics import calculate_wavelength

    wavelength = calculate_wavelength(config.PHOTON_ENERGY_EV)
    fresnel_num = _print_full_simulation_header(wavelength)
    (
        pulse_cfg,
        rotation_cfg,
        optical_cfg,
        grating_cfg,
        detector_cfg,
        movement_spec,
        movement_cfg,
        output_cfg,
    ) = _build_full_simulation_components(wavelength)
    _apply_wfe_preset(optical_cfg)
    _print_movement_spec_summary(movement_spec)

    # Run simulation
    simulator = PulseSuperpositionSimulator(
        pulse_config=pulse_cfg,
        rotation_config=rotation_cfg,
        optical_config=optical_cfg,
        grating_config=grating_cfg,
        detector_config=detector_cfg,
        output_config=output_cfg,
        movement_config=movement_cfg,
    )

    _configure_full_simulator_aberration(simulator)

    print(f"\nStarting simulation: {config.N_PULSES} pulses ...")
    result = simulator.simulate()

    # =========================================================================
    # Save outputs (数据接口 — Step 2 从这里读取)
    # =========================================================================

    integrated_sampled, _integrated_ideal, frames_ideal = _save_full_integrated_outputs(output_dir, result)
    wfe_map, _phase_screens, gt_disp, has_gt, full_grid_geometry = _save_full_truth_and_grid_outputs(
        output_dir,
        result,
        integrated_sampled.shape,
    )
    _save_full_per_frame_outputs(output_dir, result, frames_ideal)

    # 5. Simulation parameters (Step 2 needs these)
    sim_params = _build_full_simulation_params(
        wavelength_m=wavelength,
        fresnel_num=fresnel_num,
        movement_spec=movement_spec,
        wfe_map=wfe_map,
        has_gt=has_gt,
        full_grid_geometry=full_grid_geometry,
        has_reference_ideal=frames_ideal is not None and len(frames_ideal) > 0,
    )

    with open(os.path.join(output_dir, "simulation_params.json"), "w") as fp:
        json.dump(sim_params, fp, indent=2)

    print(f"\n✓ Step 1 complete. Results saved to: {output_dir}")
    print(f"  To run phase recovery: python recover.py {output_dir}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Hartmann Grating Light Propagation Simulation"
    )
    parser.add_argument(
        '--simple', action='store_true',
        help='Run simple Gaussian spot mode (for optimization testing)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: auto-increment result/xray/NNN)'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to a JSON file to override config_simulate.py parameters'
    )
    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = get_next_run_dir(os.path.join(config.OUTPUT_DIR, "xray"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if args.config:
        try:
            with open(args.config, 'r') as f:
                custom_cfg = json.load(f)
            
            # support both flat simulation_params format and hierarchical preset format
            updates = 0
            # 1. Look in 'config_simulate' block if it exists (e.g. presets/visible_527nm.json)
            if "config_simulate" in custom_cfg:
                for k, v in custom_cfg["config_simulate"].items():
                    if not k.startswith("_"):
                        setattr(config, k, v)
                        updates += 1
            else:
                # 2. Flat format JSON override
                mapping = {
                    "pitch_m": "HARTMANN_PITCH_M",
                    "hole_diameter_m": "HARTMANN_HOLE_DIAMETER_M",
                    "pixel_size_m": "DETECTOR_PIXEL_M",
                    "n_pulses": "N_PULSES",
                    "step_m": "STEP_M",
                    "angle_deg": "ANGLE_DEG",
                    "photon_energy_ev": "PHOTON_ENERGY_EV",
                    "z_prop_m": "Z_PROP_M",
                    "detector_nx": "DETECTOR_NX",
                    "detector_ny": "DETECTOR_NY",
                    "window_size_m": "WINDOW_SIZE_M",
                    "grid_points": "GRID_POINTS",
                    "propagation_method": "PROPAGATION_METHOD",
                    "source_type": "SOURCE_TYPE",
                    "mask_type": "MASK_TYPE",
                    "movement_mode": "MOVEMENT_MODE",
                    "wfe_preset": "WFE_PRESET",
                    "phase_aberration_mode": "PHASE_ABERRATION_MODE"
                }
                for k, v in custom_cfg.items():
                    if k in mapping:
                        setattr(config, mapping[k], v)
                        updates += 1
                    elif hasattr(config, k.upper()):
                        setattr(config, k.upper(), v)
                        updates += 1
            print(f"✅ Loaded {updates} parameter overrides from {args.config}")
        except Exception as e:
            print(f"Error loading config {args.config}: {e}")
            return

    if args.simple:
        run_simple_mode(output_dir)
    else:
        run_full_simulation(output_dir)


if __name__ == "__main__":
    main()
