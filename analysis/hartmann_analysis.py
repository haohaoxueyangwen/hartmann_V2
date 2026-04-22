from __future__ import annotations
import os
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from plot_style import apply_plot_font_bump

apply_plot_font_bump()
import config_recover as recover_config
from analysis.movement_metadata import (
    MovementInfo,
    get_global_offsets_m,
    get_global_offsets_px_with_pixelsize,
    parse_movement_info,
)
from analysis.optimization_common import assert_xy_points

from analysis.displacement_methods import (
    detect_beamlets as _detect_beamlets,
    compute_centroids as _compute_centroids,
    compute_cells as _compute_cells,
    extract_patches as _extract_patches,
    centroid_shifts as _centroid_shifts,
    cross_correlation_shifts as _cross_correlation_shifts,
    phase_correlation_shifts as _phase_correlation_shifts,
    recover_from_shifts as _recover_from_shifts,
    integrate_slopes as _integrate_slopes,
    remove_plane as _remove_plane,
)


@dataclass
class MethodMetrics:
    name: str
    rmse_shift_px: float
    rmse_shift_m: float
    rmse_phase_rad: float
    corr_phase: float
    spatial_resolution_error: float


@dataclass
class ReconstructionOutputs:
    displacement_x_m: np.ndarray
    displacement_y_m: np.ndarray
    phase_rad: np.ndarray


@dataclass
class LoadedPulse:
    ideal_image: Optional[np.ndarray]
    ground_truth: Dict[str, np.ndarray]


@dataclass
class OptimizationAnalysisContext:
    n_frames: int
    grid_rows: int
    grid_cols: int
    pixel_size: float
    spacing: float
    z_prop: float
    wavelength: float
    ideal_centers: np.ndarray
    x_coords_px: np.ndarray
    y_coords_px: np.ndarray
    movement_info: MovementInfo
    global_offsets_px: np.ndarray
    spot_sigma_scale: float
    spot_model: str
    spacing_px: float
    opt_dir: str
    initial_shifts: Optional[np.ndarray]
    init_source: Optional[str]


@dataclass
class OptimizationSolverOutputs:
    summary_key: str
    actual_method: str
    result_summary: Dict[str, Any]
    recovered_frames: List[Dict[str, np.ndarray]]
    truth_frames: List[Dict[str, np.ndarray]]


def _resolve_grid_axes_from_centers(
    ideal_centers_xy_px: np.ndarray,
    *,
    grid_rows: int,
    grid_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve detector grid axes `(x_coords_px, y_coords_px)` from `(x, y)` centers."""
    centers = assert_xy_points(ideal_centers_xy_px, name="ideal_centers_xy_px")
    if centers.shape[0] != int(grid_rows) * int(grid_cols):
        raise ValueError(
            f"ideal_centers_xy_px must contain grid_rows*grid_cols={grid_rows * grid_cols} centers, "
            f"got {centers.shape[0]}"
        )

    rounded_x = np.round(centers[:, 0], decimals=9)
    rounded_y = np.round(centers[:, 1], decimals=9)
    unique_x = np.unique(rounded_x)
    unique_y = np.unique(rounded_y)
    if unique_x.size == int(grid_cols) and unique_y.size == int(grid_rows):
        return unique_x.astype(np.float64), unique_y.astype(np.float64)

    ordered = centers[np.lexsort((centers[:, 0], centers[:, 1]))]
    x_coords_px = ordered[: int(grid_cols), 0]
    y_coords_px = ordered[:: int(grid_cols), 1]
    return np.asarray(x_coords_px, dtype=np.float64), np.asarray(y_coords_px, dtype=np.float64)


def _resolve_optimization_grid_geometry(
    *,
    avg_truth: Dict[str, np.ndarray],
    accumulated_image: np.ndarray,
    sim_params: Optional[Dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve ideal centers and axis coordinates for optimization analysis."""
    grid_rows = int(avg_truth["grid_rows"])
    grid_cols = int(avg_truth["grid_cols"])
    pixel_size = float(avg_truth["detector_pixel_m"])
    spacing = float(avg_truth["pupil_spacing_m"])

    ideal_centers = None
    if sim_params is not None:
        ideal_centers = sim_params.get("_ideal_centers_xy_px")
    if ideal_centers is not None:
        ideal_centers = assert_xy_points(ideal_centers, name="ideal_centers")
        x_coords_px, y_coords_px = _resolve_grid_axes_from_centers(
            ideal_centers,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )
        return ideal_centers, x_coords_px, y_coords_px

    centers_x_m = avg_truth.get("centers_x_m")
    centers_y_m = avg_truth.get("centers_y_m")
    h, w = accumulated_image.shape
    if centers_x_m is None or centers_y_m is None:
        spacing_px = spacing / pixel_size
        x_coords_px = (np.arange(grid_cols, dtype=np.float64) - (grid_cols - 1) / 2.0) * spacing_px + w / 2.0
        y_coords_px = (np.arange(grid_rows, dtype=np.float64) - (grid_rows - 1) / 2.0) * spacing_px + h / 2.0
    else:
        x_coords_px = np.asarray(centers_x_m, dtype=np.float64) / pixel_size + w / 2.0
        y_coords_px = np.asarray(centers_y_m, dtype=np.float64) / pixel_size + h / 2.0

    ideal_centers = np.array([[cx, cy] for cy in y_coords_px for cx in x_coords_px], dtype=np.float64)
    return assert_xy_points(ideal_centers, name="ideal_centers"), x_coords_px, y_coords_px


def _load_simulation_params(output_dir: str) -> Dict[str, Any]:
    params_path = os.path.join(output_dir, "simulation_params.json")
    if not os.path.exists(params_path):
        return {}
    try:
        with open(params_path, "r") as fp:
            return json.load(fp)
    except Exception:
        return {}


def _load_optional_npy(path: str) -> Optional[np.ndarray]:
    return np.load(path) if os.path.exists(path) else None


def _shift_reference_image(reference_image: np.ndarray, shift_xy_px: np.ndarray) -> np.ndarray:
    dx = float(shift_xy_px[0])
    dy = float(shift_xy_px[1])
    return ndi.shift(
        np.asarray(reference_image, dtype=np.float64),
        shift=(dy, dx),
        order=1,
        mode='constant',
        cval=0.0,
        prefilter=False,
    )


def _load_pulse_ideal_sequence(output_dir: str, movement_info: Optional[MovementInfo], n_pulses: int) -> List[Optional[np.ndarray]]:
    frames_dir = os.path.join(output_dir, "frames")
    pulse_ideal_images: List[Optional[np.ndarray]] = []
    has_all_saved = True
    for i in range(n_pulses):
        frame_path = os.path.join(frames_dir, f"frame_{i:03d}_ideal.npy")
        if os.path.exists(frame_path):
            pulse_ideal_images.append(np.load(frame_path))
        else:
            has_all_saved = False
            break
    if has_all_saved and len(pulse_ideal_images) == n_pulses:
        return pulse_ideal_images

    reference_ideal = _load_optional_npy(os.path.join(output_dir, "reference_ideal.npy"))
    if reference_ideal is None or movement_info is None:
        return [None] * n_pulses

    sim_params = _load_simulation_params(output_dir)
    pixel_size_m = sim_params.get("pixel_size_m")
    if pixel_size_m in (None, 0, 0.0):
        return [None] * n_pulses
    global_offsets_px = get_global_offsets_px_with_pixelsize(movement_info, float(pixel_size_m))
    ref0 = np.asarray(global_offsets_px[0], dtype=np.float64)
    return [
        _shift_reference_image(reference_ideal, np.asarray(offset, dtype=np.float64) - ref0)
        for offset in global_offsets_px
    ]


def _load_analysis_inputs(output_dir: str) -> tuple[List[LoadedPulse], np.ndarray, np.ndarray, List[Optional[np.ndarray]], Dict[str, Any]]:
    sim_params = _load_simulation_params(output_dir)
    if not sim_params:
        raise FileNotFoundError(f"simulation_params.json not found in {output_dir}")

    accumulated = _load_optional_npy(os.path.join(output_dir, "accumulated.npy"))
    ideal_accumulated = _load_optional_npy(os.path.join(output_dir, "ideal_accumulated.npy"))
    if accumulated is None or ideal_accumulated is None:
        raise FileNotFoundError("accumulated.npy and ideal_accumulated.npy are required for Step 3 analysis")

    gt_path = os.path.join(output_dir, "gt_displacements.npz")
    if not os.path.exists(gt_path):
        raise FileNotFoundError("gt_displacements.npz is required for Step 3 analysis")
    gt = dict(np.load(gt_path, allow_pickle=True))

    n_pulses = int(sim_params.get("n_pulses", gt["displacement_x_m"].shape[0]))
    movement_info = parse_movement_info(sim_params) if sim_params else None
    pulse_ideal_images = _load_pulse_ideal_sequence(output_dir, movement_info, n_pulses)
    ideal_centers = _load_optional_npy(os.path.join(output_dir, "ideal_centers.npy"))
    if ideal_centers is not None:
        sim_params["_ideal_centers_xy_px"] = assert_xy_points(ideal_centers, name="ideal_centers")

    wavelength = float(gt.get("wavelength_m", sim_params.get("wavelength_m", 1.0)))
    z_prop = float(gt.get("z_prop_m", sim_params.get("z_prop_m", 1.0)))
    pixel_size = float(gt.get("pixel_size_m", sim_params.get("pixel_size_m", 1.0)))
    spacing = float(gt.get("spacing_m", sim_params.get("pitch_m", 1.0)))

    phase_rad = np.asarray(gt["phase_rad"], dtype=np.float64)
    truth_disp_x_m = np.asarray(gt["displacement_x_m"], dtype=np.float64)
    truth_disp_y_m = np.asarray(gt["displacement_y_m"], dtype=np.float64)
    phase_opd_m = phase_rad * wavelength / (2.0 * np.pi)
    truth_slope_x_rad = truth_disp_x_m / max(z_prop, 1e-12)
    truth_slope_y_rad = truth_disp_y_m / max(z_prop, 1e-12)

    pulses: List[LoadedPulse] = []
    for i in range(n_pulses):
        truth = {
            "grid_rows": int(gt["grid_rows"]),
            "grid_cols": int(gt["grid_cols"]),
            "pupil_spacing_m": spacing,
            "detector_pixel_m": pixel_size,
            "z_prop_m": z_prop,
            "wavelength_m": wavelength,
            "centers_x_m": np.asarray(gt["centers_x_m"], dtype=np.float64),
            "centers_y_m": np.asarray(gt["centers_y_m"], dtype=np.float64),
            "phase_rad": phase_rad[i],
            "phase_opd_m": phase_opd_m[i],
            "displacement_x_m": truth_disp_x_m[i],
            "displacement_y_m": truth_disp_y_m[i],
            "slope_x_rad": truth_slope_x_rad[i],
            "slope_y_rad": truth_slope_y_rad[i],
        }
        pulses.append(LoadedPulse(ideal_image=pulse_ideal_images[i], ground_truth=truth))

    return pulses, accumulated, ideal_accumulated, pulse_ideal_images, sim_params


def run_analysis_report(*, pulses, accumulated_image, ideal_accumulated_image, output_dir: str,
                        use_decomposition: bool = False, aberration_mode: str = "static",
                        ideal_images: Optional[List[np.ndarray]] = None,
                        analysis_subdir: str = "analysis") -> None:
    if not pulses:
        print("No pulses available for analysis.")
        return

    sim_params = _load_simulation_params(output_dir)
    movement_info: Optional[MovementInfo] = None
    if sim_params:
        movement_info = parse_movement_info(sim_params)
        print(
            f"Analysis movement metadata: mode={movement_info.mode}, "
            f"uniform={movement_info.is_uniform}, n_pulses={movement_info.n_pulses}"
        )

    analysis_dir = os.path.join(output_dir, analysis_subdir)
    os.makedirs(analysis_dir, exist_ok=True)

    methods = _build_methods()
    summary_per_method: Dict[str, List[MethodMetrics]] = {name: [] for name in methods}
    # Add optimization methods to summary
    summary_per_method["optimization_direct"] = []
    summary_per_method["optimization_zernike"] = []
    metrics_records: List[Dict[str, float]] = []

    # 不做单帧分析 - 真实场景中只有累积图
    # 所有方法都从累积图开始

    # Accumulated frame analysis
    if accumulated_image is not None and ideal_accumulated_image is not None:
        avg_truth, pulse_ideal_images = _prepare_accumulated_analysis_inputs(
            pulses=pulses,
            ideal_images=ideal_images,
        )
        _maybe_run_trajectory_design_analysis(
            movement_info=movement_info,
            avg_truth=avg_truth,
            accumulated_image=accumulated_image,
            ideal_accumulated_image=ideal_accumulated_image,
            analysis_dir=analysis_dir,
        )
        best_results = _run_analysis_branch(
            use_decomposition=use_decomposition,
            pulses=pulses,
            accumulated_image=accumulated_image,
            ideal_accumulated_image=ideal_accumulated_image,
            pulse_ideal_images=pulse_ideal_images,
            avg_truth=avg_truth,
            sim_params=sim_params,
            movement_info=movement_info,
            methods=methods,
            analysis_dir=analysis_dir,
            summary_per_method=summary_per_method,
            metrics_records=metrics_records,
        )
        
        # ===== 综合可视化输出 =====
        _create_visualizations(
            best_results=best_results,
            avg_truth=avg_truth,
            accumulated_image=accumulated_image,
            ideal_accumulated_image=ideal_accumulated_image,
            summary_per_method=summary_per_method,
            analysis_dir=analysis_dir,
        )

    _print_summary(summary_per_method)
    _write_metrics_json(metrics_records, analysis_dir)


def _prepare_accumulated_analysis_inputs(
    *,
    pulses,
    ideal_images: Optional[List[np.ndarray]],
) -> tuple[Dict[str, np.ndarray], List[np.ndarray]]:
    """Prepare shared averaged truth and pulse ideal images for accumulated analysis."""
    avg_truth = _build_average_truth(pulses)
    raw_pulse_ideal_images = ideal_images if ideal_images is not None else [p.ideal_image for p in pulses]
    pulse_ideal_images = (
        [np.asarray(img, dtype=np.float64) for img in raw_pulse_ideal_images]
        if raw_pulse_ideal_images is not None and len(raw_pulse_ideal_images) == len(pulses) and all(img is not None for img in raw_pulse_ideal_images)
        else []
    )
    return avg_truth, pulse_ideal_images


def _maybe_run_trajectory_design_analysis(
    *,
    movement_info: Optional[MovementInfo],
    avg_truth: Dict[str, np.ndarray],
    accumulated_image: np.ndarray,
    ideal_accumulated_image: np.ndarray,
    analysis_dir: str,
) -> None:
    """Run optional trajectory-design diagnostics when movement metadata is available."""
    if movement_info is None:
        return
    try:
        from .trajectory_design import run_trajectory_design_analysis

        design_dir = os.path.join(analysis_dir, "trajectory_design")
        design_result = run_trajectory_design_analysis(
            movement_info=movement_info,
            truth=avg_truth,
            detector_shape=accumulated_image.shape,
            ideal_accumulated_image=ideal_accumulated_image,
            output_dir=design_dir,
        )
        print(
            "Trajectory design analysis: "
            f"status={design_result['design_status']}, "
            f"identifiability={design_result['identifiability_score']:.3f}, "
            f"min cross-pulse sep={design_result['cross_pulse_separation_px']['min']:.3f} px"
        )
    except Exception as exc:
        print(f"Warning: trajectory design analysis failed: {exc}")


def _run_analysis_branch(
    *,
    use_decomposition: bool,
    pulses,
    accumulated_image: np.ndarray,
    ideal_accumulated_image: np.ndarray,
    pulse_ideal_images: List[np.ndarray],
    avg_truth: Dict[str, np.ndarray],
    sim_params: Optional[Dict[str, Any]],
    movement_info: Optional[MovementInfo],
    methods: Dict[str, Callable],
    analysis_dir: str,
    summary_per_method: Dict[str, List[MethodMetrics]],
    metrics_records: List[Dict[str, float]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Dispatch accumulated analysis to either decomposition or optimization branch."""
    if use_decomposition:
        print("\nAnalysis branch: decomposition")
        return _run_decomposition_analysis(
            pulses=pulses,
            accumulated_image=accumulated_image,
            ideal_accumulated_image=ideal_accumulated_image,
            avg_truth=avg_truth,
            movement_info=movement_info,
            methods=methods,
            analysis_dir=analysis_dir,
            summary_per_method=summary_per_method,
            metrics_records=metrics_records,
            include_in_summary=True,
        )

    print("\nAnalysis branch: optimization")
    return _run_optimization_analysis(
        pulses=pulses,
        accumulated_image=accumulated_image,
        ideal_accumulated_image=ideal_accumulated_image,
        pulse_ideal_images=pulse_ideal_images,
        avg_truth=avg_truth,
        sim_params=sim_params,
        analysis_dir=analysis_dir,
        summary_per_method=summary_per_method,
        metrics_records=metrics_records,
    )


def run_analysis_report_from_dir(
    output_dir: str,
    *,
    use_decomposition: bool = False,
    analysis_subdir: str = "analysis",
) -> None:
    pulses, accumulated, ideal_accumulated, pulse_ideal_images, sim_params = _load_analysis_inputs(output_dir)
    run_analysis_report(
        pulses=pulses,
        accumulated_image=accumulated,
        ideal_accumulated_image=ideal_accumulated,
        output_dir=output_dir,
        use_decomposition=bool(use_decomposition),
        aberration_mode=str(sim_params.get("phase_aberration_mode", "none")),
        ideal_images=pulse_ideal_images,
        analysis_subdir=analysis_subdir,
    )


def _build_reference_features(reference_image: np.ndarray, rows: int, cols: int) -> Dict[str, Any]:
    boxes, ref_centroids = _detect_beamlets(reference_image, rows, cols)
    ref_patches = _extract_patches(reference_image, boxes, rows, cols)
    return {
        "boxes": boxes,
        "rows": rows,
        "cols": cols,
        "centroids": ref_centroids,
        "patches": ref_patches,
    }


def _build_decomposition_initializer(
    *,
    pulses,
    accumulated_image: np.ndarray,
    ideal_accumulated_image: np.ndarray,
    pulse_ideal_images: List[np.ndarray],
    movement_info: Optional[MovementInfo],
    pixel_size: float,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Build an optimization initializer from decomposition + xcorr.

    This mirrors the main Step 2 recovery path: use accumulated-only inversion to
    get an initial per-pulse shift sequence, then refine with optimization.
    """
    if not pulse_ideal_images or len(pulse_ideal_images) != len(pulses):
        return None, None

    from .decomposition import decompose_accumulated_frame

    if movement_info is not None:
        pulse_shifts_arr = get_global_offsets_m(movement_info)
        if pulse_shifts_arr.shape[0] != len(pulses):
            return None, None
        pulse_shifts_m = [(float(dx), float(dy)) for dx, dy in pulse_shifts_arr]
    else:
        pulse_shifts_m = [(0.0, 0.0)] * len(pulses)

    decomp_method = str(
        getattr(recover_config, 'ACCUMULATED_DECOMPOSITION_METHOD', 'weighted') or 'weighted'
    ).lower()
    first_truth = pulses[0].ground_truth
    decomposed = decompose_accumulated_frame(
        accumulated_image=np.asarray(accumulated_image, dtype=np.float64),
        ideal_accumulated_image=np.asarray(ideal_accumulated_image, dtype=np.float64),
        pulse_ideal_images=[np.asarray(img, dtype=np.float64) for img in pulse_ideal_images],
        pulse_shifts_m=pulse_shifts_m,
        pixel_size_m=float(pixel_size),
        method=decomp_method,
        grid_rows=int(first_truth["grid_rows"]),
        grid_cols=int(first_truth["grid_cols"]),
    )
    decomposed = sorted(decomposed, key=lambda item: int(item.index))

    all_shifts = []
    for d in decomposed:
        truth = pulses[d.index].ground_truth
        rows = int(truth["grid_rows"])
        cols = int(truth["grid_cols"])
        ref_features = _build_reference_features(pulse_ideal_images[d.index], rows, cols)
        tgt_patches = _extract_patches(
            d.extracted_image,
            ref_features["boxes"],
            ref_features["rows"],
            ref_features["cols"],
        )
        shifts = _cross_correlation_shifts(
            ref_features["patches"],
            tgt_patches,
            correction_factor=float(getattr(recover_config, 'XCORR_CORRECTION_FACTOR', 1.0)),
            ref_boxes=ref_features["boxes"],
            tgt_image=d.extracted_image,
            search_radius=int(getattr(recover_config, 'XCORR_SEARCH_RADIUS', 0)),
        )
        all_shifts.append(shifts)

    if len(all_shifts) != len(pulses):
        return None, None

    return np.asarray(all_shifts, dtype=np.float64), f"decomposition_{decomp_method}"


def _run_optimization_analysis(
    *,
    pulses,
    accumulated_image: np.ndarray,
    ideal_accumulated_image: np.ndarray,
    pulse_ideal_images: List[np.ndarray],
    avg_truth: Dict[str, np.ndarray],
    sim_params: Optional[Dict[str, Any]],
    analysis_dir: str,
    summary_per_method: Dict[str, List[MethodMetrics]],
    metrics_records: List[Dict[str, float]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run optimization-based shift recovery from accumulated image.
    
    This recovers per-frame shifts by solving an inverse problem that
    minimizes the difference between rendered and observed accumulated images.
    """
    print("\n" + "="*70)
    print("=== Optimization-Based Shift Recovery from Accumulated Image ===")
    print("="*70)

    context = _prepare_optimization_analysis_context(
        pulses=pulses,
        accumulated_image=accumulated_image,
        ideal_accumulated_image=ideal_accumulated_image,
        pulse_ideal_images=pulse_ideal_images,
        avg_truth=avg_truth,
        sim_params=sim_params,
        analysis_dir=analysis_dir,
    )

    # Storage for direct shift comparison results
    best_method = None
    best_avg_rmse_px = None
    best_results: Dict[str, Any] = {"recovered": None, "truth": None, "avg_truth": avg_truth}
    
    # Template forward model currently supports direct solver only.
    solver_requests = ["direct"] if context.spot_model == "template" else ["direct", "zernike"]
    for requested_method in solver_requests:
        try:
            solver_outputs = _run_single_optimization_solver(
                requested_method=requested_method,
                context=context,
                pulses=pulses,
                accumulated_image=accumulated_image,
                ideal_accumulated_image=ideal_accumulated_image,
                pulse_ideal_images=pulse_ideal_images,
                summary_per_method=summary_per_method,
                metrics_records=metrics_records,
            )
            if (
                solver_outputs.recovered_frames
                and (
                    best_avg_rmse_px is None
                    or float(solver_outputs.result_summary["avg_rmse_shift_px"]) < best_avg_rmse_px
                )
            ):
                best_avg_rmse_px = float(solver_outputs.result_summary["avg_rmse_shift_px"])
                best_method = solver_outputs.summary_key
                best_results = {
                    "recovered": _average_reconstruction_outputs(solver_outputs.recovered_frames),
                    "truth": _average_reconstruction_outputs(solver_outputs.truth_frames),
                    "avg_truth": avg_truth,
                    "selected_method": best_method,
                }
            
        except Exception as e:
            print(f"  Error running {requested_method} solver: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    if best_method is not None:
        print(f"Selected optimization result for visualization: {best_method}")
    return best_results


def _prepare_optimization_analysis_context(
    *,
    pulses,
    accumulated_image: np.ndarray,
    ideal_accumulated_image: np.ndarray,
    pulse_ideal_images: List[np.ndarray],
    avg_truth: Dict[str, np.ndarray],
    sim_params: Optional[Dict[str, Any]],
    analysis_dir: str,
) -> OptimizationAnalysisContext:
    n_frames = len(pulses)
    grid_rows = int(avg_truth["grid_rows"])
    grid_cols = int(avg_truth["grid_cols"])
    pixel_size = float(avg_truth["detector_pixel_m"])
    spacing = float(avg_truth["pupil_spacing_m"])
    z_prop = float(avg_truth["z_prop_m"])
    wavelength = float(avg_truth["wavelength_m"])
    ideal_centers, x_coords_px, y_coords_px = _resolve_optimization_grid_geometry(
        avg_truth=avg_truth,
        accumulated_image=accumulated_image,
        sim_params=sim_params,
    )
    movement_info = parse_movement_info(sim_params or {})
    global_offsets_px = get_global_offsets_px_with_pixelsize(movement_info, pixel_size)
    if global_offsets_px.shape[0] != n_frames:
        raise ValueError(
            f"movement trajectory length {global_offsets_px.shape[0]} does not match n_frames={n_frames}"
        )
    print(
        "Using global offsets from simulation_params.json: "
        f"mode={movement_info.mode}, uniform={movement_info.is_uniform}"
    )
    if n_frames >= 2:
        step_vectors_px = np.diff(global_offsets_px, axis=0)
        if movement_info.is_uniform:
            avg_step_px = np.mean(step_vectors_px, axis=0)
            print(f"Average step in pixels: ({avg_step_px[0]:.2f}, {avg_step_px[1]:.2f})")
        else:
            step_lengths_px = np.linalg.norm(step_vectors_px, axis=1)
            print(
                "Trajectory step length range in pixels: "
                f"{np.min(step_lengths_px):.2f} .. {np.max(step_lengths_px):.2f}"
            )

    opt_dir = os.path.join(analysis_dir, "optimization")
    os.makedirs(opt_dir, exist_ok=True)

    initial_shifts = None
    init_source = None
    if bool(getattr(recover_config, 'OPTIMIZATION_USE_DECOMPOSITION_INIT', True)):
        try:
            initial_shifts, init_source = _build_decomposition_initializer(
                pulses=pulses,
                accumulated_image=accumulated_image,
                ideal_accumulated_image=ideal_accumulated_image,
                pulse_ideal_images=pulse_ideal_images,
                movement_info=movement_info,
                pixel_size=pixel_size,
            )
        except Exception as exc:
            print(f"Warning: optimization initializer unavailable in analysis: {exc}")
            initial_shifts, init_source = None, None
    if init_source is not None:
        print(f"Optimization initializer: {init_source}")

    return OptimizationAnalysisContext(
        n_frames=n_frames,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        pixel_size=pixel_size,
        spacing=spacing,
        z_prop=z_prop,
        wavelength=wavelength,
        ideal_centers=ideal_centers,
        x_coords_px=x_coords_px,
        y_coords_px=y_coords_px,
        movement_info=movement_info,
        global_offsets_px=global_offsets_px,
        spot_sigma_scale=float(getattr(recover_config, 'SPOT_SIGMA_SCALE', 1.0)),
        spot_model=str(getattr(recover_config, 'OPTIMIZATION_SPOT_MODEL', 'gaussian') or 'gaussian').lower(),
        spacing_px=spacing / pixel_size,
        opt_dir=opt_dir,
        initial_shifts=initial_shifts,
        init_source=init_source,
    )


def _map_valid_centers_to_grid_indices(
    valid_centers_xy_px: np.ndarray,
    *,
    x_coords_px: np.ndarray,
    y_coords_px: np.ndarray,
    grid_rows: int,
    grid_cols: int,
) -> np.ndarray:
    valid_centers = assert_xy_points(valid_centers_xy_px, name="valid_centers_xy_px")
    grid_indices = np.zeros((valid_centers.shape[0], 2), dtype=np.int64)
    for idx, (center_x_px, center_y_px) in enumerate(valid_centers):
        row = int(np.argmin(np.abs(y_coords_px - center_y_px)))
        col = int(np.argmin(np.abs(x_coords_px - center_x_px)))
        if row < 0 or row >= grid_rows or col < 0 or col >= grid_cols:
            raise ValueError(
                f"valid center ({center_x_px:.3f}, {center_y_px:.3f}) mapped outside grid "
                f"{grid_rows}x{grid_cols}"
            )
        grid_indices[idx, :] = (row, col)
    return grid_indices


def _compute_direct_shift_rmse_px(
    *,
    shifts_grid_px: np.ndarray,
    pulse_truth: Dict[str, np.ndarray],
    pixel_size: float,
) -> Optional[float]:
    truth_disp_x_m = np.array(pulse_truth["displacement_x_m"], dtype=np.float64)
    truth_disp_y_m = np.array(pulse_truth["displacement_y_m"], dtype=np.float64)
    opt_disp_x_m = np.asarray(shifts_grid_px[:, :, 0], dtype=np.float64) * pixel_size
    opt_disp_y_m = np.asarray(shifts_grid_px[:, :, 1], dtype=np.float64) * pixel_size
    valid_mask = (shifts_grid_px[:, :, 0] != 0) | (shifts_grid_px[:, :, 1] != 0)
    if not np.any(valid_mask):
        return None
    diff_x = (opt_disp_x_m - truth_disp_x_m)[valid_mask]
    diff_y = (opt_disp_y_m - truth_disp_y_m)[valid_mask]
    direct_rmse_m = float(np.sqrt(np.mean(diff_x**2 + diff_y**2)))
    return direct_rmse_m / pixel_size


def _build_optimization_result_summary(
    *,
    requested_method: str,
    actual_method: str,
    spot_model: str,
    init_source: Optional[str],
    n_frames: int,
    n_spots: int,
    final_loss: float,
    n_iterations: int,
    elapsed_time: float,
    method_metrics_list: List[MethodMetrics],
    direct_rmse_values_px: List[float],
) -> Dict[str, Any]:
    per_pulse_rmse_px = [float(m.rmse_shift_px) for m in method_metrics_list]
    return {
        "requested_solver": requested_method,
        "solver": actual_method,
        "spot_model": spot_model,
        "optimization_init_source": init_source,
        "n_frames": n_frames,
        "n_spots": n_spots,
        "final_loss": float(final_loss),
        "n_iterations": int(n_iterations),
        "elapsed_time": float(elapsed_time),
        "avg_rmse_shift_px": float(np.mean(per_pulse_rmse_px)) if per_pulse_rmse_px else float('nan'),
        "median_rmse_shift_px": float(np.median(per_pulse_rmse_px)) if per_pulse_rmse_px else float('nan'),
        "worst_pulse_rmse_px": float(np.max(per_pulse_rmse_px)) if per_pulse_rmse_px else float('nan'),
        "per_pulse_rmse_px": per_pulse_rmse_px,
        "avg_phase_corr": float(np.mean([m.corr_phase for m in method_metrics_list])) if method_metrics_list else float('nan'),
        "avg_direct_rmse_px": float(np.mean(direct_rmse_values_px)) if direct_rmse_values_px else 0.0,
    }


def _average_reconstruction_outputs(frames: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {
        "displacement_x_m": np.mean([frame["displacement_x_m"] for frame in frames], axis=0),
        "displacement_y_m": np.mean([frame["displacement_y_m"] for frame in frames], axis=0),
        "phase_rad": np.mean([frame["phase_rad"] for frame in frames], axis=0),
    }


def _reshape_optimization_shifts_to_grid(
    *,
    result,
    context: OptimizationAnalysisContext,
) -> np.ndarray:
    """Map valid optimization centers back onto the explicit detector grid."""
    from .shift_optimization import reshape_shifts_to_grid

    spot_grid_indices = _map_valid_centers_to_grid_indices(
        result.valid_centers,
        x_coords_px=context.x_coords_px,
        y_coords_px=context.y_coords_px,
        grid_rows=context.grid_rows,
        grid_cols=context.grid_cols,
    )
    return reshape_shifts_to_grid(
        result.shifts,
        result.valid_centers,
        context.grid_rows,
        context.grid_cols,
        spot_grid_indices=spot_grid_indices,
    )


def _evaluate_optimization_pulse(
    *,
    summary_key: str,
    actual_method: str,
    pulse_index: int,
    shifts_grid_px: np.ndarray,
    pulse_truth: Dict[str, np.ndarray],
    context: OptimizationAnalysisContext,
    metrics_records: List[Dict[str, float]],
    summary_per_method: Dict[str, List[MethodMetrics]],
) -> tuple[MethodMetrics, Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[float]]:
    """Evaluate one optimization-recovered pulse against its ground truth."""
    metrics, recovered, truth_adj = _compute_metrics(
        name=summary_key,
        shifts_px=shifts_grid_px,
        truth=pulse_truth,
        pixel_size=context.pixel_size,
        spacing=context.spacing,
        z_prop=context.z_prop,
        wavelength=context.wavelength,
    )

    record: Dict[str, float] = {
        "case": f"opt_{actual_method}_pulse_{pulse_index:03d}",
        "method": summary_key,
        "rmse_shift_px": metrics.rmse_shift_px,
        "rmse_shift_um": metrics.rmse_shift_m * 1e6,
        "rmse_phase_rad": metrics.rmse_phase_rad,
        "phase_corr": metrics.corr_phase,
        "spatial_error_percent": metrics.spatial_resolution_error * 100.0,
    }
    metrics_records.append(record)
    summary_per_method.setdefault(summary_key, []).append(metrics)

    direct_rmse_px = _compute_direct_shift_rmse_px(
        shifts_grid_px=shifts_grid_px,
        pulse_truth=pulse_truth,
        pixel_size=context.pixel_size,
    )
    return metrics, recovered, truth_adj, direct_rmse_px


def _save_optimization_pulse_outputs(
    *,
    actual_method: str,
    pulse_index: int,
    recovered: Dict[str, np.ndarray],
    truth_adj: Dict[str, np.ndarray],
    metrics: MethodMetrics,
    opt_dir: str,
) -> None:
    """Save selected per-pulse optimization artifacts used for inspection."""
    from visualization.visualization import create_summary_comparison

    base_path = os.path.join(opt_dir, f"{actual_method}_pulse_{pulse_index:03d}")
    _save_reconstruction_outputs(base_path, recovered, truth_adj, metrics)
    create_summary_comparison(
        recovered=recovered,
        truth=truth_adj,
        output_path=os.path.join(opt_dir, f"{actual_method}_frame_{pulse_index:03d}_summary.png"),
        title=f"Optimization ({actual_method.upper()}) - Frame {pulse_index}",
    )


def _write_optimization_summary(
    *,
    result_summary: Dict[str, Any],
    actual_method: str,
    opt_dir: str,
) -> None:
    """Persist and print the optimization-summary payload."""
    with open(os.path.join(opt_dir, f"{actual_method}_summary.json"), "w") as f:
        json.dump(result_summary, f, indent=2)

    print(f"  Final loss: {result_summary['final_loss']:.4e}")
    print(f"  Elapsed time: {result_summary['elapsed_time']:.2f}s")
    print(f"  Avg RMSE shift (indirect): {result_summary['avg_rmse_shift_px']:.4f} px")
    print(f"  Worst pulse RMSE: {result_summary['worst_pulse_rmse_px']:.4f} px")
    print(f"  Avg RMSE shift (direct vs truth): {result_summary['avg_direct_rmse_px']:.4f} px")
    print(f"  Avg phase corr: {result_summary['avg_phase_corr']:.4f}")


def _run_single_optimization_solver(
    *,
    requested_method: str,
    context: OptimizationAnalysisContext,
    pulses,
    accumulated_image: np.ndarray,
    ideal_accumulated_image: np.ndarray,
    pulse_ideal_images: List[np.ndarray],
    summary_per_method: Dict[str, List[MethodMetrics]],
    metrics_records: List[Dict[str, float]],
) -> OptimizationSolverOutputs:
    from .shift_optimization import run_optimization_recovery

    print(f"\n--- Running {requested_method.upper()} solver (spot_model={context.spot_model}) ---")

    result = run_optimization_recovery(
        accumulated_image=accumulated_image.astype(np.float64),
        ideal_accumulated_image=ideal_accumulated_image.astype(np.float64),
        pulse_ideal_images=pulse_ideal_images,
        ideal_centers=context.ideal_centers,
        n_frames=context.n_frames,
        global_offsets_px=context.global_offsets_px,
        spacing=context.spacing_px,
        pixel_size_m=context.pixel_size,
        z_prop_m=context.z_prop,
        wavelength_m=context.wavelength,
        method=requested_method,
        lambda_spatial=float(getattr(recover_config, 'OPTIMIZATION_LAMBDA_SPATIAL', 0.1)),
        lambda_temporal=float(getattr(recover_config, 'OPTIMIZATION_LAMBDA_TEMPORAL', 0.0)),
        n_zernike_modes=int(getattr(recover_config, 'OPTIMIZATION_N_ZERNIKE', 15)),
        spot_sigma_scale=context.spot_sigma_scale,
        spot_model=context.spot_model,
        initial_shifts=context.initial_shifts,
        ftol=float(getattr(recover_config, 'OPTIMIZATION_FTOL', 1e-9)),
        gtol=float(getattr(recover_config, 'OPTIMIZATION_GTOL', 1e-9)),
        verbose=True,
    )
    actual_method = str(getattr(result, 'solver_method', requested_method) or requested_method)
    summary_key = f"optimization_{actual_method}"
    _save_convergence_plot(result, context.opt_dir, actual_method)
    shifts_grid_all = _reshape_optimization_shifts_to_grid(result=result, context=context)

    method_recovered_frames: List[Dict[str, np.ndarray]] = []
    method_truth_frames: List[Dict[str, np.ndarray]] = []
    method_metrics_list: List[MethodMetrics] = []
    direct_rmse_values_px: List[float] = []

    for pulse_index in range(context.n_frames):
        shifts_grid = shifts_grid_all[pulse_index]
        pulse_truth = pulses[pulse_index].ground_truth
        metrics, recovered, truth_adj, direct_rmse_px = _evaluate_optimization_pulse(
            summary_key=summary_key,
            actual_method=actual_method,
            pulse_index=pulse_index,
            shifts_grid_px=shifts_grid,
            pulse_truth=pulse_truth,
            context=context,
            metrics_records=metrics_records,
            summary_per_method=summary_per_method,
        )
        if actual_method != requested_method:
            metrics_records[-1]["requested_solver"] = requested_method
        method_metrics_list.append(metrics)
        method_recovered_frames.append(recovered)
        method_truth_frames.append(truth_adj)

        if direct_rmse_px is not None:
            direct_rmse_values_px.append(float(direct_rmse_px))

        if pulse_index == 0 or pulse_index == context.n_frames - 1:
            _save_optimization_pulse_outputs(
                actual_method=actual_method,
                pulse_index=pulse_index,
                recovered=recovered,
                truth=truth_adj,
                metrics=metrics,
                opt_dir=context.opt_dir,
            )

    result_summary = _build_optimization_result_summary(
        requested_method=requested_method,
        actual_method=actual_method,
        spot_model=context.spot_model,
        init_source=context.init_source,
        n_frames=context.n_frames,
        n_spots=len(result.valid_centers),
        final_loss=result.final_loss,
        n_iterations=result.n_iterations,
        elapsed_time=result.elapsed_time,
        method_metrics_list=method_metrics_list,
        direct_rmse_values_px=direct_rmse_values_px,
    )
    _write_optimization_summary(
        result_summary=result_summary,
        actual_method=actual_method,
        opt_dir=context.opt_dir,
    )

    return OptimizationSolverOutputs(
        summary_key=summary_key,
        actual_method=actual_method,
        result_summary=result_summary,
        recovered_frames=method_recovered_frames,
        truth_frames=method_truth_frames,
    )


def _save_convergence_plot(result, output_dir: str, method: str) -> None:
    """Save convergence history plot."""
    if not result.convergence_history:
        return
    
    plt.figure(figsize=(8, 5))
    plt.semilogy(result.convergence_history, 'b-', linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.title(f"Optimization Convergence ({method.upper()} solver)")
    plt.grid(True, alpha=0.3)
    
    # Mark final value
    plt.axhline(y=result.final_loss, color='r', linestyle='--', alpha=0.5, label=f"Final: {result.final_loss:.2e}")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{method}_convergence.png"), dpi=150)
    plt.close()


def _resolve_decomposition_pulse_shifts_m(
    *,
    movement_info: Optional[MovementInfo],
    n_pulses: int,
) -> List[Tuple[float, float]]:
    """Resolve per-pulse global offsets in meters for decomposition analysis."""
    if movement_info is None:
        return [(0.0, 0.0)] * int(n_pulses)
    pulse_shifts_arr = get_global_offsets_m(movement_info)
    if pulse_shifts_arr.shape[0] != int(n_pulses):
        raise ValueError(
            f"movement trajectory length {pulse_shifts_arr.shape[0]} does not match pulse count {n_pulses}"
        )
    return [(float(dx), float(dy)) for dx, dy in pulse_shifts_arr]


def _process_single_decomposed_pulse(
    *,
    decomposed_pulse,
    pulses,
    pulse_ideals: List[np.ndarray],
    methods: Dict[str, Callable],
    analysis_dir: str,
    include_in_summary: bool,
    summary_per_method: Dict[str, List[MethodMetrics]],
    metrics_records: List[Dict[str, float]],
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """Run per-pulse analysis on one decomposition output and return the selected method result."""
    case_name = f"decomp_weighted_pulse_{decomposed_pulse.index:03d}"
    pulse_truth = pulses[decomposed_pulse.index].ground_truth
    xcorr_result = _process_case_with_return(
        case_name=case_name,
        reference_image=pulse_ideals[decomposed_pulse.index],
        target_image=decomposed_pulse.extracted_image,
        truth=pulse_truth,
        methods=methods,
        analysis_dir=analysis_dir,
        include_in_summary=include_in_summary,
        summary_per_method=summary_per_method,
        metrics_records=metrics_records,
        return_method="cross_correlation",
    )
    if xcorr_result:
        from visualization.visualization import create_summary_comparison

        frame_summary_path = os.path.join(analysis_dir, f"frame_{decomposed_pulse.index:03d}_summary.png")
        create_summary_comparison(
            recovered=xcorr_result["recovered"],
            truth=xcorr_result["truth"],
            output_path=frame_summary_path,
            title=f"Frame {decomposed_pulse.index} Wavefront Reconstruction"
        )
    print(f"    Pulse {decomposed_pulse.index}: confidence={decomposed_pulse.confidence:.4f}")
    return xcorr_result


def _combine_decomposition_results(
    *,
    all_recovered: List[Dict[str, np.ndarray]],
    all_truth: List[Dict[str, np.ndarray]],
    avg_truth: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Average per-pulse decomposition reconstructions for summary visualization."""
    if not all_recovered:
        return {"recovered": None, "truth": None, "avg_truth": avg_truth}

    avg_recovered = {
        "displacement_x_m": np.mean([r["displacement_x_m"] for r in all_recovered], axis=0),
        "displacement_y_m": np.mean([r["displacement_y_m"] for r in all_recovered], axis=0),
        "phase_rad": np.mean([r["phase_rad"] for r in all_recovered], axis=0),
    }
    avg_truth_adj = {
        "displacement_x_m": np.mean([t["displacement_x_m"] for t in all_truth], axis=0),
        "displacement_y_m": np.mean([t["displacement_y_m"] for t in all_truth], axis=0),
        "phase_rad": np.mean([t["phase_rad"] for t in all_truth], axis=0),
    }
    return {"recovered": avg_recovered, "truth": avg_truth_adj, "avg_truth": avg_truth}


def _run_decomposition_analysis(
    *,
    pulses,
    accumulated_image: np.ndarray,
    ideal_accumulated_image: np.ndarray,
    avg_truth: Dict[str, np.ndarray],
    movement_info: Optional[MovementInfo],
    methods: Dict[str, Callable],
    analysis_dir: str,
    summary_per_method: Dict[str, List[MethodMetrics]],
    metrics_records: List[Dict[str, float]],
    include_in_summary: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run analysis using decomposition of accumulated frame.
    
    Returns:
        Dict with 'recovered' and 'truth' containing the best (cross_correlation) results.
    """
    from .decomposition import decompose_accumulated_frame
    
    print("\n--- Running Accumulated Frame Decomposition ---")
    
    pulse_ideals = [p.ideal_image for p in pulses]
    pulse_shifts = _resolve_decomposition_pulse_shifts_m(
        movement_info=movement_info,
        n_pulses=len(pulses),
    )
    
    pixel_size = float(avg_truth["detector_pixel_m"])
    grid_rows = int(avg_truth["grid_rows"])
    grid_cols = int(avg_truth["grid_cols"])
    
    # Use weighted decomposition (best quality)
    decomp_method = "weighted"
    print(f"  Decomposition method: {decomp_method}")
    
    decomposed = decompose_accumulated_frame(
        accumulated_image=accumulated_image,
        ideal_accumulated_image=ideal_accumulated_image,
        pulse_ideal_images=pulse_ideals,
        pulse_shifts_m=pulse_shifts,
        pixel_size_m=pixel_size,
        method=decomp_method,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )
    
    all_recovered = []
    all_truth = []

    for d in decomposed:
        xcorr_result = _process_single_decomposed_pulse(
            decomposed_pulse=d,
            pulses=pulses,
            pulse_ideals=pulse_ideals,
            methods=methods,
            analysis_dir=analysis_dir,
            include_in_summary=include_in_summary,
            summary_per_method=summary_per_method,
            metrics_records=metrics_records,
        )
        if xcorr_result:
            all_recovered.append(xcorr_result["recovered"])
            all_truth.append(xcorr_result["truth"])

    return _combine_decomposition_results(
        all_recovered=all_recovered,
        all_truth=all_truth,
        avg_truth=avg_truth,
    )


def _process_case(
    *,
    case_name: str,
    reference_image: np.ndarray,
    target_image: np.ndarray,
    truth: Dict[str, np.ndarray],
    methods: Dict[str, Callable],
    analysis_dir: str,
    include_in_summary: bool,
    summary_per_method: Dict[str, List[MethodMetrics]],
    metrics_records: List[Dict[str, float]],
) -> None:
    rows = int(truth["grid_rows"])
    cols = int(truth["grid_cols"])

    boxes, ref_centroids = _detect_beamlets(reference_image, rows, cols)
    ref_patches = _extract_patches(reference_image, boxes, rows, cols)
    reference_features = {
        "boxes": boxes,
        "rows": rows,
        "cols": cols,
        "centroids": ref_centroids,
        "patches": ref_patches,
    }

    case_dir = os.path.join(analysis_dir, case_name)
    os.makedirs(case_dir, exist_ok=True)

    for method_name, estimator in methods.items():
        shifts = estimator(reference_features, target_image)
        metrics, recovered, truth_adj = _compute_metrics(
            name=method_name,
            shifts_px=shifts,
            truth=truth,
            pixel_size=float(truth["detector_pixel_m"]),
            spacing=float(truth["pupil_spacing_m"]),
            z_prop=float(truth["z_prop_m"]),
            wavelength=float(truth["wavelength_m"]),
        )

        if include_in_summary:
            summary_per_method[method_name].append(metrics)

        record = {
            "case": case_name,
            "method": method_name,
            "rmse_shift_px": metrics.rmse_shift_px,
            "rmse_shift_um": metrics.rmse_shift_m * 1e6,
            "rmse_phase_rad": metrics.rmse_phase_rad,
            "phase_corr": metrics.corr_phase,
            "spatial_error_percent": metrics.spatial_resolution_error * 100.0,
        }
        metrics_records.append(record)

        base_path = os.path.join(case_dir, method_name)
        _save_reconstruction_outputs(base_path, recovered, truth_adj, metrics)
        _save_visuals(base_path, recovered, truth_adj)


def _process_case_with_return(
    *,
    case_name: str,
    reference_image: np.ndarray,
    target_image: np.ndarray,
    truth: Dict[str, np.ndarray],
    methods: Dict[str, Callable],
    analysis_dir: str,
    include_in_summary: bool,
    summary_per_method: Dict[str, List[MethodMetrics]],
    metrics_records: List[Dict[str, float]],
    return_method: str = "cross_correlation",
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """Process case and return results for specified method."""
    rows = int(truth["grid_rows"])
    cols = int(truth["grid_cols"])

    boxes, ref_centroids = _detect_beamlets(reference_image, rows, cols)
    ref_patches = _extract_patches(reference_image, boxes, rows, cols)
    reference_features = {
        "boxes": boxes,
        "rows": rows,
        "cols": cols,
        "centroids": ref_centroids,
        "patches": ref_patches,
    }

    case_dir = os.path.join(analysis_dir, case_name)
    os.makedirs(case_dir, exist_ok=True)

    result = None
    for method_name, estimator in methods.items():
        shifts = estimator(reference_features, target_image)
        metrics, recovered, truth_adj = _compute_metrics(
            name=method_name,
            shifts_px=shifts,
            truth=truth,
            pixel_size=float(truth["detector_pixel_m"]),
            spacing=float(truth["pupil_spacing_m"]),
            z_prop=float(truth["z_prop_m"]),
            wavelength=float(truth["wavelength_m"]),
        )

        if include_in_summary:
            summary_per_method[method_name].append(metrics)

        record = {
            "case": case_name,
            "method": method_name,
            "rmse_shift_px": metrics.rmse_shift_px,
            "rmse_shift_um": metrics.rmse_shift_m * 1e6,
            "rmse_phase_rad": metrics.rmse_phase_rad,
            "phase_corr": metrics.corr_phase,
            "spatial_error_percent": metrics.spatial_resolution_error * 100.0,
        }
        metrics_records.append(record)

        base_path = os.path.join(case_dir, method_name)
        _save_reconstruction_outputs(base_path, recovered, truth_adj, metrics)
        # _save_visuals(base_path, recovered, truth_adj)  # Removed: too many scattered PNGs
        
        # Capture requested method result
        if method_name == return_method:
            result = {"recovered": recovered, "truth": truth_adj}
    
    return result


def _create_visualizations(
    *,
    best_results: Dict,
    avg_truth: Dict[str, np.ndarray],
    accumulated_image: np.ndarray,
    ideal_accumulated_image: np.ndarray,
    summary_per_method: Dict[str, List[MethodMetrics]],
    analysis_dir: str,
) -> None:
    """Create comprehensive visualization outputs."""
    from visualization.visualization import (
        create_summary_comparison,
        create_dpc_visualization,
        create_method_comparison,
        create_accumulated_visualization,
        create_error_statistics,
        create_sensor_image_visualization,
    )
    
    print("\n--- Creating Visualizations ---")
    
    # Create detailed sensor image visualization
    centroids_px = None
    if "centers_x_m" in avg_truth and "centers_y_m" in avg_truth:
        cx_m = avg_truth["centers_x_m"]
        cy_m = avg_truth["centers_y_m"]
        if cx_m is not None and cy_m is not None:
            pixel_size = float(avg_truth["detector_pixel_m"])
            h, w = accumulated_image.shape
            # Assuming centers are relative to optical axis at image center
            cx_px = cx_m / pixel_size + w / 2
            cy_px = cy_m / pixel_size + h / 2
            
            # Flatten to (N, 2)
            if cx_px.ndim > 1:
                cx_px = cx_px.ravel()
                cy_px = cy_px.ravel()
            
            centroids_px = np.column_stack((cx_px, cy_px))

    create_sensor_image_visualization(
        image=accumulated_image,
        output_path=os.path.join(analysis_dir, "sensor_image_detailed.png"),
        title="Hartmann Sensor Accumulated Image",
        centroids=centroids_px,
        pixel_size_um=float(avg_truth["detector_pixel_m"]) * 1e6,
    )
    
    # Accumulated image comparison
    create_accumulated_visualization(
        accumulated_image=accumulated_image,
        ideal_accumulated_image=ideal_accumulated_image,
        output_dir=analysis_dir,
    )
    
    # Method comparison bar chart
    create_method_comparison(
        metrics_per_method=summary_per_method,
        output_path=os.path.join(analysis_dir, "method_comparison.png"),
    )
    
    # If we have recovered results, create detailed visualizations
    if best_results.get("recovered") is not None:
        recovered = best_results["recovered"]
        truth = best_results["truth"]
        
        # Summary comparison (3x3 grid)
        create_summary_comparison(
            recovered=recovered,
            truth=truth,
            output_path=os.path.join(analysis_dir, "summary_comparison.png"),
            title="Wavefront Reconstruction: Cross-Correlation Method"
        )
        
        # DPC visualization
        create_dpc_visualization(
            recovered=recovered,
            truth=truth,
            output_dir=analysis_dir,
            z_prop=float(avg_truth["z_prop_m"]),
            wavelength=float(avg_truth["wavelength_m"]),
        )
        
        # Error statistics
        create_error_statistics(
            recovered=recovered,
            truth=truth,
            output_path=os.path.join(analysis_dir, "error_statistics.png"),
        )


def _build_methods() -> Dict[str, Callable]:
    """Build all displacement analysis methods."""
    def centroid(ref_features, target_image):
        rows, cols = ref_features["rows"], ref_features["cols"]
        tgt_boxes, tgt_centroids = _detect_beamlets(target_image, rows, cols)
        return _centroid_shifts(ref_features["centroids"], tgt_centroids)

    def cross_correlation(ref_features, target_image):
        tgt_patches = _extract_patches(
            target_image,
            ref_features["boxes"],
            ref_features["rows"],
            ref_features["cols"],
        )
        return _cross_correlation_shifts(
            ref_features["patches"],
            tgt_patches,
            correction_factor=float(getattr(recover_config, 'XCORR_CORRECTION_FACTOR', 1.0)),
            ref_boxes=ref_features["boxes"],
            tgt_image=target_image,
            search_radius=int(getattr(recover_config, 'XCORR_SEARCH_RADIUS', 0)),
        )

    def phase_correlation(ref_features, target_image):
        tgt_patches = _extract_patches(
            target_image,
            ref_features["boxes"],
            ref_features["rows"],
            ref_features["cols"],
        )
        return _phase_correlation_shifts(ref_features["patches"], tgt_patches)

    # All 3 classic methods
    return {
        "centroid": centroid,
        "cross_correlation": cross_correlation,
        "phase_correlation": phase_correlation,
    }


def _build_average_truth(pulses) -> Dict[str, np.ndarray]:
    template = pulses[0].ground_truth
    return {
        "grid_rows": template["grid_rows"],
        "grid_cols": template["grid_cols"],
        "pupil_spacing_m": template["pupil_spacing_m"],
        "detector_pixel_m": template["detector_pixel_m"],
        "z_prop_m": template["z_prop_m"],
        "wavelength_m": template["wavelength_m"],
        "centers_x_m": template.get("centers_x_m"),
        "centers_y_m": template.get("centers_y_m"),
        "phase_rad": np.mean([p.ground_truth["phase_rad"] for p in pulses], axis=0),
        "phase_opd_m": np.mean([p.ground_truth["phase_opd_m"] for p in pulses], axis=0),
        "displacement_x_m": np.mean([p.ground_truth["displacement_x_m"] for p in pulses], axis=0),
        "displacement_y_m": np.mean([p.ground_truth["displacement_y_m"] for p in pulses], axis=0),
        "slope_x_rad": np.mean([p.ground_truth["slope_x_rad"] for p in pulses], axis=0),
        "slope_y_rad": np.mean([p.ground_truth["slope_y_rad"] for p in pulses], axis=0),
    }


# NOTE: _detect_beamlets, _compute_cells, _compute_centroids, _extract_patches,
# _centroid_shifts, _cross_correlation_shifts, _phase_correlation_shifts,
# _recover_from_shifts, _integrate_slopes, _remove_plane
# are all imported from analysis.displacement_methods (see top of file)


def _compute_metrics(
    *,
    name: str,
    shifts_px: np.ndarray,
    truth: Dict[str, np.ndarray],
    pixel_size: float,
    spacing: float,
    z_prop: float,
    wavelength: float,
) -> Tuple[MethodMetrics, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    recovered = _recover_from_shifts(
        shifts_px,
        pixel_size=pixel_size,
        z_prop=z_prop,
        spacing=spacing,
        wavelength=wavelength,
    )

    recovered_disp_x_m = recovered["displacement_x_m"].copy()
    recovered_disp_y_m = recovered["displacement_y_m"].copy()
    recovered_phase_rad = recovered["phase_rad"].copy()

    truth_disp_x_m = np.array(truth["displacement_x_m"], dtype=float)
    truth_disp_y_m = np.array(truth["displacement_y_m"], dtype=float)
    truth_phase_rad = np.array(truth["phase_rad"], dtype=float)

    centers_x = truth.get("centers_x_m")
    centers_y = truth.get("centers_y_m")
    if centers_x is not None and centers_y is not None:
        grid_x, grid_y = np.meshgrid(centers_x, centers_y, indexing="xy")
        recovered_disp_x_m = _remove_plane(recovered_disp_x_m, grid_x, grid_y)
        recovered_disp_y_m = _remove_plane(recovered_disp_y_m, grid_x, grid_y)
        truth_disp_x_m = _remove_plane(truth_disp_x_m, grid_x, grid_y)
        truth_disp_y_m = _remove_plane(truth_disp_y_m, grid_x, grid_y)
        recovered_phase_rad = _remove_plane(recovered_phase_rad, grid_x, grid_y)
        truth_phase_rad = _remove_plane(truth_phase_rad, grid_x, grid_y)

    rmse_shift_m = float(
        np.sqrt(
            np.mean(
                (recovered_disp_x_m - truth_disp_x_m) ** 2 +
                (recovered_disp_y_m - truth_disp_y_m) ** 2
            )
        )
    )
    rmse_shift_px = float(rmse_shift_m / pixel_size) if pixel_size > 0 else 0.0

    diff_phase = recovered_phase_rad - truth_phase_rad
    rmse_phase_rad = float(np.sqrt(np.mean(diff_phase ** 2)))

    truth_flat = truth_phase_rad.ravel()
    approx_flat = recovered_phase_rad.ravel()
    if np.allclose(truth_flat.std(), 0.0) or np.allclose(approx_flat.std(), 0.0):
        corr_phase = 0.0
    else:
        corr_phase = float(np.corrcoef(truth_flat, approx_flat)[0, 1])

    spatial_err = _spatial_resolution_error(recovered_phase_rad, truth_phase_rad)

    metrics = MethodMetrics(
        name=name,
        rmse_shift_px=rmse_shift_px,
        rmse_shift_m=rmse_shift_m,
        rmse_phase_rad=rmse_phase_rad,
        corr_phase=corr_phase,
        spatial_resolution_error=spatial_err,
    )

    truth_adj = {
        "displacement_x_m": truth_disp_x_m,
        "displacement_y_m": truth_disp_y_m,
        "phase_rad": truth_phase_rad,
    }

    return metrics, {
        "displacement_x_m": recovered_disp_x_m,
        "displacement_y_m": recovered_disp_y_m,
        "phase_rad": recovered_phase_rad,
    }, truth_adj


def _spatial_resolution_error(estimated: np.ndarray, truth: np.ndarray) -> float:
    est_hf = _high_frequency_energy(estimated)
    truth_hf = _high_frequency_energy(truth)
    if truth_hf <= 1e-20:
        return 0.0
    return abs(est_hf - truth_hf) / truth_hf

def _high_frequency_energy(field: np.ndarray) -> float:
    centered = field - np.mean(field)
    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    ny, nx = field.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    radius = min(ny, nx) / 4.0
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) >= radius ** 2
    energy = float(np.sum((np.abs(spectrum) ** 2) * mask))
    return energy

def _save_reconstruction_outputs(base_path: str, recovered: Dict[str, np.ndarray], truth: Dict[str, np.ndarray], metrics: MethodMetrics) -> None:
    np.savez(
        f"{base_path}.npz",
        displacement_x_m=recovered["displacement_x_m"],
        displacement_y_m=recovered["displacement_y_m"],
        phase_rad=recovered["phase_rad"],
        truth_displacement_x_m=truth["displacement_x_m"],
        truth_displacement_y_m=truth["displacement_y_m"],
        truth_phase_rad=truth["phase_rad"],
        displacement_x_err_m=recovered["displacement_x_m"] - truth["displacement_x_m"],
        displacement_y_err_m=recovered["displacement_y_m"] - truth["displacement_y_m"],
        phase_err_rad=recovered["phase_rad"] - truth["phase_rad"],
        rmse_shift_px=metrics.rmse_shift_px,
        rmse_shift_m=metrics.rmse_shift_m,
        rmse_phase_rad=metrics.rmse_phase_rad,
        corr_phase=metrics.corr_phase,
        spatial_resolution_error=metrics.spatial_resolution_error,
    )

    metrics_path = f"{base_path}_metrics.json"
    with open(metrics_path, "w") as fp:
        json.dump({
            "rmse_shift_px": metrics.rmse_shift_px,
            "rmse_shift_um": metrics.rmse_shift_m * 1e6,
            "rmse_phase_rad": metrics.rmse_phase_rad,
            "phase_corr": metrics.corr_phase,
            "spatial_error_percent": metrics.spatial_resolution_error * 100.0,
        }, fp, indent=2)

def _save_visuals(base_path: str, recovered: Dict[str, np.ndarray], truth: Dict[str, np.ndarray]) -> None:
    recovered_phase_rad = recovered["phase_rad"]
    phase_err_rad = recovered_phase_rad - truth["phase_rad"]
    recovered_disp_x_um = recovered["displacement_x_m"] * 1e6
    recovered_disp_y_um = recovered["displacement_y_m"] * 1e6
    disp_x_err_um = recovered_disp_x_um - truth["displacement_x_m"] * 1e6
    disp_y_err_um = recovered_disp_y_um - truth["displacement_y_m"] * 1e6

    _save_field_plot(recovered_phase_rad, f"{base_path}_phase.png", "Recovered phase", "rad", symmetric=True)
    _save_field_plot(recovered_disp_x_um, f"{base_path}_disp_x_um.png", "Recovered displacement X", "um", symmetric=True)
    _save_field_plot(recovered_disp_y_um, f"{base_path}_disp_y_um.png", "Recovered displacement Y", "um", symmetric=True)
    _save_field_plot(phase_err_rad, f"{base_path}_phase_error.png", "Phase error", "rad", symmetric=True)
    _save_field_plot(disp_x_err_um, f"{base_path}_disp_x_error_um.png", "Displacement X error", "um", symmetric=True)
    _save_field_plot(disp_y_err_um, f"{base_path}_disp_y_error_um.png", "Displacement Y error", "um", symmetric=True)

def _save_field_plot(data: np.ndarray, path: str, title: str, unit: str, symmetric: bool = False) -> None:
    plt.figure(figsize=(5, 4))
    if symmetric:
        max_abs = np.nanmax(np.abs(data))
        vmax = max_abs if max_abs > 0 else None
        vmin = -vmax if vmax is not None else None
    else:
        vmin = None
        vmax = None
    im = plt.imshow(data, cmap="coolwarm", origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=unit)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def _write_metrics_json(metrics_records: List[Dict[str, float]], analysis_dir: str) -> None:
    report_path = os.path.join(analysis_dir, "metrics_summary.json")
    with open(report_path, "w") as fp:
        json.dump(metrics_records, fp, indent=2)

def _print_summary(summary: Dict[str, List[MethodMetrics]]) -> None:
    print("\nWavefront Reconstruction Metrics")
    print("Method               | RMSE shift (px) | RMSE shift (um) | RMSE phase (rad) | Phase corr | Spatial err (%)")
    print("-" * 102)
    for method_name, metrics_list in summary.items():
        if not metrics_list:
            continue
        rmse_px = np.array([m.rmse_shift_px for m in metrics_list])
        rmse_m = np.array([m.rmse_shift_m for m in metrics_list])
        rmse_phase = np.array([m.rmse_phase_rad for m in metrics_list])
        corr = np.array([m.corr_phase for m in metrics_list])
        spatial = np.array([m.spatial_resolution_error for m in metrics_list])
        print(
            f"{method_name:20s} | "
            f"{np.mean(rmse_px):12.4e} | "
            f"{np.mean(rmse_m) * 1e6:14.4f} | "
            f"{np.mean(rmse_phase):15.4e} | "
            f"{np.mean(corr):10.4f} | "
            f"{np.mean(spatial) * 100:14.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: Hartmann analysis from a Step 1 output directory"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to Step 1 output directory containing accumulated.npy and simulation_params.json",
    )
    parser.add_argument(
        "--use-decomposition",
        action="store_true",
        help="Use accumulated decomposition branch instead of optimization branch",
    )
    args = parser.parse_args()

    pulses, accumulated, ideal_accumulated, pulse_ideal_images, sim_params = _load_analysis_inputs(args.output_dir)
    run_analysis_report(
        pulses=pulses,
        accumulated_image=accumulated,
        ideal_accumulated_image=ideal_accumulated,
        output_dir=args.output_dir,
        use_decomposition=bool(args.use_decomposition),
        aberration_mode=str(sim_params.get("phase_aberration_mode", "none")),
        ideal_images=pulse_ideal_images,
        analysis_subdir="analysis",
    )


if __name__ == "__main__":
    main()
