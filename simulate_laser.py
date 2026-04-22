#!/usr/bin/env python3
from __future__ import annotations
"""
Step 1 (Laser): 激光 4f 中继 Hartmann 波前传感器仿真。

独立于现有 X 射线仿真链 (simulate.py)。
输出格式与 recover.py 兼容。

Usage:
    python simulate_laser.py                                        # 默认读取 config_simulate_laser.py
    python simulate_laser.py --output-dir result/laser/001          # 指定输出
    python simulate_laser.py --config other_laser_config.py         # 指定 Python 配置文件
"""

import os
import sys
import json
import glob
import argparse
import importlib.util
from typing import Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import apply_plot_font_bump

apply_plot_font_bump()

import config_simulate_laser as config

from simulation.laser_mode import (
    OpticalConfig, MaskConfig, DetectorConfig,
    AberrationConfig, MovementConfig, OutputConfig,
    Laser4fSimulator,
)
from simulation.laser_mode.mask import make_finite_hartmann_centers


# =============================================================================
# Utility
# =============================================================================

def _save_as_png(npy_path: str, data, *, allow_preview: bool = True):
    """将数组保存为 PNG 预览图。"""
    if not allow_preview:
        return
    png_path = npy_path.replace('.npy', '.png').replace('.npz', '.png')
    try:
        arr = np.squeeze(np.asarray(data, dtype=float))
        if arr.ndim == 2:
            plt.imsave(png_path, arr, cmap='magma')
        elif arr.ndim == 3:
            plt.imsave(png_path, arr[0], cmap='magma')
    except Exception as e:
        print(f"    Warning: PNG save failed {png_path}: {e}")


def get_next_run_dir(base_name: str = "result") -> str:
    """Get next available run directory."""
    os.makedirs(base_name, exist_ok=True)
    pattern = os.path.join(base_name, "[0-9][0-9][0-9]")
    existing = glob.glob(pattern)
    if not existing:
        return os.path.join(base_name, "001")
    numbers = []
    for d in existing:
        try:
            numbers.append(int(os.path.basename(d)))
        except ValueError:
            pass
    next_num = max(numbers) + 1 if numbers else 1
    return os.path.join(base_name, f"{next_num:03d}")


# =============================================================================
# Config loading
# =============================================================================

def _load_config_module(path: str):
    """从 Python 文件路径加载配置模块。"""
    spec = importlib.util.spec_from_file_location("laser_config_override", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载配置文件: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_configs_from_module(cfg):
    """从 config_simulate_laser 风格的 Python 模块构建 dataclass 配置。"""
    optical_cfg = OpticalConfig(
        wavelength_m=float(cfg.WAVELENGTH_M),
        sim_window_m=float(cfg.SIM_WINDOW_M),
        grid_points=int(cfg.GRID_POINTS),
        relay_z0_m=float(cfg.RELAY_Z0_M),
        relay_f1_m=float(cfg.RELAY_F1_M),
        relay_f2_m=float(cfg.RELAY_F2_M),
        relay_lens_aperture_m=cfg.RELAY_LENS_APERTURE_M,
        relay_fourier_stop_enable=bool(getattr(cfg, 'RELAY_FOURIER_STOP_ENABLE', False)),
        relay_fourier_stop_diameter_m=getattr(cfg, 'RELAY_FOURIER_STOP_DIAMETER_M', None),
        relay_fourier_stop_x_m=float(getattr(cfg, 'RELAY_FOURIER_STOP_X_M', 0.0)),
        relay_fourier_stop_y_m=float(getattr(cfg, 'RELAY_FOURIER_STOP_Y_M', 0.0)),
        relay_lens1_aberration_coeffs=cfg.RELAY_LENS1_ABERRATION_COEFFS,
        relay_lens2_aberration_coeffs=cfg.RELAY_LENS2_ABERRATION_COEFFS,
        source_type=str(cfg.SOURCE_TYPE),
        source_radius_m=float(cfg.SOURCE_RADIUS_M),
    )
    mask_cfg = MaskConfig(
        mask_window_m=float(cfg.MASK_WINDOW_M),
        pitch_m=float(cfg.HARTMANN_PITCH_M),
        hole_diameter_m=float(cfg.HARTMANN_HOLE_DIAMETER_M),
    )
    detector_cfg = DetectorConfig(
        pixel_size=float(cfg.DETECTOR_PIXEL_M),
        nx=int(cfg.DETECTOR_NX),
        ny=int(cfg.DETECTOR_NY),
        sampling_mode=str(cfg.DETECTOR_SAMPLING_MODE),
        upsample_factor=int(cfg.DETECTOR_UPSAMPLE),
        bin_size=cfg.DETECTOR_BIN_SIZE,
        interpolation=str(cfg.DETECTOR_INTERPOLATION),
    )
    aberration_cfg = AberrationConfig(
        mode=str(cfg.PHASE_ABERRATION_MODE),
        phase_plate_to_mask_m=float(getattr(cfg, 'PHASE_PLATE_TO_MASK_M', 0.0)),
        max_order=int(cfg.PHASE_ABERRATION_MAX_ORDER),
        amplitude_range=tuple(cfg.PHASE_ABERRATION_RANGE),
        pupil_radius_m=cfg.PHASE_ABERRATION_PUPIL_RADIUS_M,
    )
    movement_cfg = MovementConfig(
        mode=str(cfg.MOVEMENT_MODE),
        n_pulses=int(cfg.N_PULSES),
        step_m=float(cfg.STEP_M),
        angle_deg=float(cfg.ANGLE_DEG),
    )
    output_cfg = OutputConfig(
        show_progress=bool(getattr(cfg, 'SHOW_PROGRESS', True)),
        save_each_frame=bool(getattr(cfg, 'SAVE_EACH_FRAME', True)),
    )
    output_base_dir = str(getattr(cfg, 'OUTPUT_DIR', os.path.join("result", "laser")))
    return optical_cfg, mask_cfg, detector_cfg, aberration_cfg, movement_cfg, output_cfg, output_base_dir


# =============================================================================
# Header
# =============================================================================

def _print_header(optical_cfg: OpticalConfig, mask_cfg: MaskConfig,
                  detector_cfg: DetectorConfig, movement_cfg: MovementConfig):
    """打印仿真 banner。"""
    print("=" * 70)
    print("  LASER 4f RELAY HARTMANN SIMULATION")
    print("=" * 70)


# =============================================================================
# Save outputs
# =============================================================================

def _save_integrated_outputs(output_dir: str, result: dict):
    """保存累加图像。"""
    integrated = result.get("integrated_sampled")
    if integrated is not None:
        p = os.path.join(output_dir, "accumulated.npy")
        np.save(p, integrated.astype(np.float32))
        _save_as_png(p, integrated)
        print(f"  Accumulated image: {integrated.shape}")

    integrated_ideal = result.get("integrated_ideal")
    if integrated_ideal is not None:
        p = os.path.join(output_dir, "ideal_accumulated.npy")
        np.save(p, integrated_ideal.astype(np.float64))
        _save_as_png(p, integrated_ideal)
        print(f"  Ideal accumulated: {integrated_ideal.shape}")

    frames_ideal = result.get("frames_ideal")
    if frames_ideal is not None and len(frames_ideal) > 0:
        p = os.path.join(output_dir, "reference_ideal.npy")
        np.save(p, np.asarray(frames_ideal[0], dtype=np.float64))
        _save_as_png(p, frames_ideal[0])
        print(f"  Reference ideal: {np.asarray(frames_ideal[0]).shape} (pulse 0)")


def _save_truth_and_grid_outputs(output_dir: str, result: dict,
                                  optical_cfg: OpticalConfig,
                                  mask_cfg: MaskConfig,
                                  detector_cfg: DetectorConfig):
    """保存 GT、相位屏、网格元数据。"""
    wfe_map = result.get("wfe_phase_map_rad")
    if wfe_map is not None:
        p = os.path.join(output_dir, "wfe_phase_map.npy")
        np.save(p, np.asarray(wfe_map, dtype=np.float64))
        _save_as_png(p, wfe_map)
        print(f"  WFE phase map: {np.asarray(wfe_map).shape}")

    # 相位屏
    phase_screens = result.get("phase_screens")
    if phase_screens is not None:
        p = os.path.join(output_dir, "phase_screens.npy")
        np.save(p, phase_screens)
        _save_as_png(p, phase_screens)
        print(f"  Phase screens: {phase_screens.shape}")

    # GT 位移
    gt_disp = result.get("gt_displacements")
    has_gt = False
    if gt_disp is not None:
        p = os.path.join(output_dir, "gt_displacements.npz")
        np.savez(p, **gt_disp)
        has_gt = True
        n_gt_frames = gt_disp['displacement_x_px'].shape[0]
        print(f"  GT displacements: {gt_disp['grid_rows']}×{gt_disp['grid_cols']} "
              f"sub-apertures, {n_gt_frames} frames")
        dx_px = gt_disp['displacement_x_px']
        dy_px = gt_disp['displacement_y_px']
        total_disp = np.sqrt(dx_px**2 + dy_px**2)
        print(f"  GT displacement stats: mean={np.nanmean(total_disp):.3f} px, "
              f"max={np.nanmax(total_disp):.3f} px")

    # ideal_centers.npy — 探测器上的子孔径中心 (像素坐标)
    centers = make_finite_hartmann_centers(mask_cfg.mask_window_m, mask_cfg.pitch_m)
    if len(centers) > 0:
        M = optical_cfg.magnification_signed
        # 物方 → 像方 → 像素 (与 detector.py 的 j - N/2 + 0.5 约定一致)
        det_x_px = M * centers[:, 0] / detector_cfg.pixel_size + detector_cfg.nx / 2.0 - 0.5
        det_y_px = M * centers[:, 1] / detector_cfg.pixel_size + detector_cfg.ny / 2.0 - 0.5
        centers_xy_px = np.column_stack([det_x_px, det_y_px])
        p = os.path.join(output_dir, "ideal_centers.npy")
        np.save(p, centers_xy_px.astype(np.float64))
        _save_as_png(p, centers_xy_px, allow_preview=False)

        # spot_grid_indices.npy
        cx_unique = np.unique(centers[:, 0])
        cy_unique = np.unique(centers[:, 1])
        grid_indices = []
        for cx, cy in centers:
            col = int(np.searchsorted(cx_unique, cx))
            row = int(np.searchsorted(cy_unique, cy))
            grid_indices.append([row, col])
        grid_indices = np.array(grid_indices, dtype=np.int32)
        p = os.path.join(output_dir, "spot_grid_indices.npy")
        np.save(p, grid_indices)
        print(f"  Saved grid metadata: {len(centers)} centers, "
              f"grid={len(cy_unique)}x{len(cx_unique)}")

    return wfe_map, gt_disp, has_gt


def _save_per_frame_outputs(output_dir: str, result: dict):
    """保存逐帧输出。"""
    frames_sampled = result.get("frames_sampled")
    frames_ideal = result.get("frames_ideal")
    if frames_sampled is not None and len(frames_sampled) > 0:
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        for i, frame in enumerate(frames_sampled):
            p = os.path.join(frames_dir, f"frame_{i:03d}.npy")
            np.save(p, frame)
            _save_as_png(p, frame)
        if frames_ideal is not None:
            for i, frame in enumerate(frames_ideal):
                p = os.path.join(frames_dir, f"frame_{i:03d}_ideal.npy")
                np.save(p, frame)
                _save_as_png(p, frame)
        print(f"  Saved {len(frames_sampled)} frames (aberrated + ideal)")


def _build_simulation_params(optical_cfg: OpticalConfig,
                              mask_cfg: MaskConfig,
                              detector_cfg: DetectorConfig,
                              movement_cfg: MovementConfig,
                              aberration_cfg: AberrationConfig,
                              has_wfe_ground_truth: bool,
                              has_gt: bool,
                              n_centers: int) -> dict:
    """构建 simulation_params.json（兼容 recover.py）。

    movement metadata 与 X 射线 simulate.py 完全对齐。
    """
    M = optical_cfg.magnification_signed
    nf = (mask_cfg.hole_diameter_m / 2.0)**2 / (
        optical_cfg.relay_z0_m * optical_cfg.wavelength_m)

    n_pulses = movement_cfg.n_pulses
    angle_rad = np.deg2rad(movement_cfg.angle_deg)
    step = movement_cfg.step_m

    # 生成完整 movement 偏移量列表
    initial_offset_xy_m = [0.0, 0.0]
    relative_offsets = []
    absolute_offsets = []
    for i in range(n_pulses):
        dx_off = step * np.cos(angle_rad) * i
        dy_off = step * np.sin(angle_rad) * i
        relative_offsets.append([dx_off, dy_off])
        absolute_offsets.append([dx_off, dy_off])

    if n_pulses > 0:
        final_offset = absolute_offsets[-1]
        relative_final = relative_offsets[-1]
        total_displacement = float(np.hypot(relative_final[0], relative_final[1]))
        net_displacement = total_displacement
    else:
        final_offset = [0.0, 0.0]
        relative_final = [0.0, 0.0]
        total_displacement = 0.0
        net_displacement = 0.0

    step_lengths = [float(step)] * max(0, n_pulses - 1)
    step_angles = [float(movement_cfg.angle_deg)] * n_pulses

    return {
        "mode": "full",
        "propagation_method": "4f_relay",
        "wavelength_m": float(optical_cfg.wavelength_m),
        "z_prop_m": float(optical_cfg.z_eff_m),
        "pixel_size_m": float(detector_cfg.pixel_size),
        "pitch_m": float(mask_cfg.pitch_m),
        "hole_diameter_m": float(mask_cfg.hole_diameter_m),
        "fresnel_number": float(nf),
        "n_pulses": n_pulses,
        "detector_nx": detector_cfg.nx,
        "detector_ny": detector_cfg.ny,
        "window_size_m": float(mask_cfg.mask_window_m),
        "grid_points": optical_cfg.grid_points,
        "source_type": optical_cfg.source_type,
        "mask_type": "hartmann",
        "angle_deg": float(movement_cfg.angle_deg),
        "step_m": float(movement_cfg.step_m),
        "phase_aberration_mode": aberration_cfg.mode,
        "phase_aberration_max_order": int(aberration_cfg.max_order),
        "relay_fourier_stop_enable": bool(optical_cfg.relay_fourier_stop_enable),
        "relay_fourier_stop_diameter_m": (
            None if optical_cfg.relay_fourier_stop_diameter_m is None
            else float(optical_cfg.relay_fourier_stop_diameter_m)
        ),
        "has_wfe_ground_truth": bool(has_wfe_ground_truth),
        "has_gt_displacements": has_gt,
        "has_ideal_centers": True,
        "has_spot_grid_indices": True,
        "spot_grid_rows": int(mask_cfg.mask_window_m / mask_cfg.pitch_m),
        "spot_grid_cols": int(mask_cfg.mask_window_m / mask_cfg.pitch_m),
        "has_reference_ideal": True,
        "ideal_reference_pulse_index": 0,
        # --- movement metadata (与 X 射线 simulate.py 同构) ---
        "movement_mode": movement_cfg.mode,
        "movement_generated_from": "translation",
        "movement_is_uniform": True,
        "movement_initial_offset_xy_m": initial_offset_xy_m,
        "movement_direction_deg": float(movement_cfg.angle_deg),
        "movement_uniform_step_m": float(movement_cfg.step_m),
        "movement_uniform_angle_deg": float(movement_cfg.angle_deg),
        "movement_total_displacement_m": total_displacement,
        "movement_net_displacement_m": net_displacement,
        "movement_final_offset_xy_m": final_offset,
        "movement_relative_final_offset_xy_m": relative_final,
        "movement_pulse_offsets_xy_m": relative_offsets,
        "movement_pulse_absolute_offsets_xy_m": absolute_offsets,
        "movement_step_lengths_m": step_lengths,
        "movement_step_angles_deg": step_angles,
        "movement_step_intervals": None,
        "movement_step_intervals_mode": None,
        "movement_angle_jitter_std_deg": None,
        "movement_angle_jitter_seed": None,
    }


def _build_laser_metadata(optical_cfg: OpticalConfig,
                           mask_cfg: MaskConfig,
                           aberration_cfg: AberrationConfig) -> dict:
    """构建 laser_metadata.json（4f 专用参数 + 符号约定）。"""
    return {
        "relay_z0_m": float(optical_cfg.relay_z0_m),
        "relay_f1_m": float(optical_cfg.relay_f1_m),
        "relay_f2_m": float(optical_cfg.relay_f2_m),
        "relay_lens_aperture_m": float(optical_cfg.relay_lens_aperture_m),
        "relay_fourier_stop_enable": bool(optical_cfg.relay_fourier_stop_enable),
        "relay_fourier_stop_diameter_m": (
            None if optical_cfg.relay_fourier_stop_diameter_m is None
            else float(optical_cfg.relay_fourier_stop_diameter_m)
        ),
        "relay_fourier_stop_x_m": float(optical_cfg.relay_fourier_stop_x_m),
        "relay_fourier_stop_y_m": float(optical_cfg.relay_fourier_stop_y_m),
        "magnification_signed": float(optical_cfg.magnification_signed),
        "magnification_abs": float(optical_cfg.M_abs),
        "z_eff_m": float(optical_cfg.z_eff_m),
        "sim_window_m": float(optical_cfg.sim_window_m),
        "mask_window_m": float(mask_cfg.mask_window_m),
        "phase_plate_to_mask_m": float(aberration_cfg.phase_plate_to_mask_m),
        "image_inversion": True,
        "sign_convention": "magnification_signed = -f2/f1, negative means inverted image",
    }


# =============================================================================
# Main
# =============================================================================

def run_laser_simulation(output_dir: str,
                          optical_cfg: OpticalConfig,
                          mask_cfg: MaskConfig,
                          detector_cfg: DetectorConfig,
                          aberration_cfg: AberrationConfig,
                          movement_cfg: MovementConfig,
                          output_cfg: OutputConfig):
    """执行激光 4f 仿真并保存输出。"""
    _print_header(optical_cfg, mask_cfg, detector_cfg, movement_cfg)

    simulator = Laser4fSimulator(
        optical_config=optical_cfg,
        mask_config=mask_cfg,
        detector_config=detector_cfg,
        aberration_config=aberration_cfg,
        movement_config=movement_cfg,
        output_config=output_cfg,
    )

    result = simulator.simulate()

    # 保存输出
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving outputs to: {output_dir}")

    _save_integrated_outputs(output_dir, result)
    wfe_map, gt_disp, has_gt = _save_truth_and_grid_outputs(
        output_dir, result, optical_cfg, mask_cfg, detector_cfg)
    if output_cfg.save_each_frame:
        _save_per_frame_outputs(output_dir, result)

    # simulation_params.json
    centers = make_finite_hartmann_centers(mask_cfg.mask_window_m, mask_cfg.pitch_m)
    sim_params = _build_simulation_params(
        optical_cfg, mask_cfg, detector_cfg, movement_cfg, aberration_cfg,
        has_wfe_ground_truth=(wfe_map is not None),
        has_gt=has_gt, n_centers=len(centers))
    with open(os.path.join(output_dir, "simulation_params.json"), "w") as fp:
        json.dump(sim_params, fp, indent=2)

    # laser_metadata.json
    laser_meta = _build_laser_metadata(optical_cfg, mask_cfg, aberration_cfg)
    with open(os.path.join(output_dir, "laser_metadata.json"), "w") as fp:
        json.dump(laser_meta, fp, indent=2)

    print(f"\n✓ Laser simulation complete. Results saved to: {output_dir}")
    print(f"  To run phase recovery: python recover_laser.py {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Laser 4f relay Hartmann wavefront sensor simulation")
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--config', type=str, default=None,
                        help='Python config file (default: config_simulate_laser.py)')
    args = parser.parse_args()

    # 加载配置
    if args.config:
        try:
            cfg_module = _load_config_module(args.config)
            print(f"✅ Loaded config from {args.config}")
        except Exception as e:
            print(f"Error loading config {args.config}: {e}")
            return
    else:
        cfg_module = config
        print(f"✅ Using default config module: {config.__file__}")

    try:
        (optical_cfg, mask_cfg, detector_cfg,
         aberration_cfg, movement_cfg, output_cfg,
         output_base_dir) = _build_configs_from_module(cfg_module)
    except Exception as e:
        print(f"Error building configs: {e}")
        return

    # 输出目录
    output_dir = args.output_dir or get_next_run_dir(output_base_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    run_laser_simulation(
        output_dir, optical_cfg, mask_cfg, detector_cfg,
        aberration_cfg, movement_cfg, output_cfg)


if __name__ == '__main__':
    main()
