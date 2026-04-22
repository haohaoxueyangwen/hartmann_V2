"""Trajectory-aware Hartmann movement design analysis.

This module evaluates whether a pulse trajectory is well separated and
well supported on the detector for accumulated-image inversion.

Unlike ``spacing_solver.py``, this analysis works with the full per-pulse
trajectory and is therefore suitable for non-uniform movement design.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from plot_style import apply_plot_font_bump

apply_plot_font_bump()

from .movement_metadata import MovementInfo, get_global_offsets_m, get_global_offsets_px_with_pixelsize
from .optimization_common import estimate_spot_fwhm_sigma_from_image
from .serialization import dump_json_serializable


def _build_ideal_centers_px(
    *,
    truth: Dict[str, Any],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Build ideal spot centers in detector pixel coordinates."""
    h, w = image_shape
    pixel_size = float(truth["detector_pixel_m"])
    spacing_m = float(truth["pupil_spacing_m"])
    grid_rows = int(truth["grid_rows"])
    grid_cols = int(truth["grid_cols"])

    centers_x_m = truth.get("centers_x_m")
    centers_y_m = truth.get("centers_y_m")
    if centers_x_m is not None and centers_y_m is not None:
        x_coords = np.asarray(centers_x_m, dtype=np.float64) / pixel_size + w / 2.0
        y_coords = np.asarray(centers_y_m, dtype=np.float64) / pixel_size + h / 2.0
    else:
        spacing_px = spacing_m / pixel_size
        x_coords = (np.arange(grid_cols, dtype=np.float64) - (grid_cols - 1) / 2.0) * spacing_px + w / 2.0
        y_coords = (np.arange(grid_rows, dtype=np.float64) - (grid_rows - 1) / 2.0) * spacing_px + h / 2.0

    return np.asarray([[cx, cy] for cy in y_coords for cx in x_coords], dtype=np.float64)


def _pairwise_offset_distances(offsets: np.ndarray) -> np.ndarray:
    if offsets.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    diffs = offsets[:, None, :] - offsets[None, :, :]
    dist = np.linalg.norm(diffs, axis=2)
    iu = np.triu_indices(offsets.shape[0], k=1)
    return dist[iu]


def _compute_visibility_mask(
    slot_positions_px: np.ndarray,
    *,
    detector_shape: tuple[int, int],
    margin_px: float,
) -> np.ndarray:
    h, w = detector_shape
    x = slot_positions_px[..., 0]
    y = slot_positions_px[..., 1]
    return (
        (x >= margin_px) &
        (x < (w - margin_px)) &
        (y >= margin_px) &
        (y < (h - margin_px))
    )


def _nearest_cross_pulse_distance(points: np.ndarray, pulse_ids: np.ndarray) -> np.ndarray:
    """Nearest-neighbor distance where the neighbor must come from another pulse."""
    n = int(points.shape[0])
    if n <= 1:
        return np.full(n, np.inf, dtype=np.float64)

    tree = cKDTree(points)
    k = min(max(8, int(np.unique(pulse_ids).size) + 2), n)
    result = np.full(n, np.inf, dtype=np.float64)

    while True:
        dists, idxs = tree.query(points, k=k)
        dists = np.atleast_2d(dists)
        idxs = np.atleast_2d(idxs)
        if dists.shape[0] != n:
            dists = dists.T
            idxs = idxs.T

        for i in range(n):
            for dist, j in zip(dists[i][1:], idxs[i][1:]):
                if int(j) == i:
                    continue
                if pulse_ids[int(j)] != pulse_ids[i]:
                    result[i] = float(dist)
                    break

        if np.all(np.isfinite(result)) or k >= n:
            return result
        k = min(n, max(k + 8, k * 2))


def _rasterize_support_map(
    valid_points: np.ndarray,
    *,
    detector_shape: tuple[int, int],
    support_radius_px: int,
) -> np.ndarray:
    """Rasterize a coarse occupancy / stack-depth map for slot supports."""
    h, w = detector_shape
    radius = max(1, int(support_radius_px))
    yy, xx = np.mgrid[-radius: radius + 1, -radius: radius + 1]
    disk = (xx ** 2 + yy ** 2) <= radius ** 2
    support = np.zeros((h, w), dtype=np.int32)

    for px, py in valid_points:
        cx = int(round(float(px)))
        cy = int(round(float(py)))
        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        mx0 = x0 - (cx - radius)
        mx1 = mx0 + (x1 - x0)
        my0 = y0 - (cy - radius)
        my1 = my0 + (y1 - y0)
        support[y0:y1, x0:x1] += disk[my0:my1, mx0:mx1].astype(np.int32)

    return support

def analyze_trajectory_design(
    *,
    movement_info: MovementInfo,
    truth: Dict[str, Any],
    detector_shape: tuple[int, int],
    ideal_accumulated_image: np.ndarray,
) -> Dict[str, Any]:
    """Analyze trajectory separability and detector support for accumulated inversion."""
    pixel_size_m = float(truth["detector_pixel_m"])
    spacing_m = float(truth["pupil_spacing_m"])
    spacing_px = float(spacing_m / pixel_size_m)

    ideal_centers_px = _build_ideal_centers_px(truth=truth, image_shape=detector_shape)
    global_offsets_px = get_global_offsets_px_with_pixelsize(movement_info, pixel_size_m)
    global_offsets_m = get_global_offsets_m(movement_info)
    slot_positions_px = ideal_centers_px[None, :, :] + global_offsets_px[:, None, :]

    _, sigma_px_est = estimate_spot_fwhm_sigma_from_image(
        np.asarray(ideal_accumulated_image, dtype=np.float64),
        spacing_px,
    )
    sigma_px = float(sigma_px_est) if sigma_px_est is not None else float(spacing_px) / 8.0
    margin_px = max(3.0 * sigma_px + 2.0, 2.0)
    support_radius_px = max(1, int(np.ceil(3.0 * sigma_px)))
    visibility = _compute_visibility_mask(
        slot_positions_px,
        detector_shape=detector_shape,
        margin_px=margin_px,
    )

    step_vectors_px = np.diff(global_offsets_px, axis=0) if global_offsets_px.shape[0] >= 2 else np.zeros((0, 2), dtype=np.float64)
    step_vectors_m = np.diff(global_offsets_m, axis=0) if global_offsets_m.shape[0] >= 2 else np.zeros((0, 2), dtype=np.float64)
    step_lengths_px = np.linalg.norm(step_vectors_px, axis=1)
    step_lengths_m = np.linalg.norm(step_vectors_m, axis=1)
    pairwise_offset_dist_px = _pairwise_offset_distances(global_offsets_px)
    pairwise_offset_dist_m = _pairwise_offset_distances(global_offsets_m)

    valid_positions = slot_positions_px[visibility]
    valid_pulse_ids = np.repeat(np.arange(slot_positions_px.shape[0], dtype=np.int32)[:, None], slot_positions_px.shape[1], axis=1)[visibility]
    nearest_cross_pulse_px = _nearest_cross_pulse_distance(valid_positions, valid_pulse_ids) if valid_positions.size else np.zeros(0, dtype=np.float64)

    occupancy_map = _rasterize_support_map(
        valid_positions,
        detector_shape=detector_shape,
        support_radius_px=support_radius_px,
    )
    occupied_pixels = int(np.count_nonzero(occupancy_map > 0))
    occupancy_ratio = float(occupied_pixels / occupancy_map.size) if occupancy_map.size else 0.0
    max_stack_depth = int(np.max(occupancy_map)) if occupancy_map.size else 0
    mean_stack_depth = float(np.mean(occupancy_map[occupancy_map > 0])) if occupied_pixels > 0 else 0.0

    visibility_per_pulse = np.mean(visibility, axis=1) if visibility.size else np.zeros(0, dtype=np.float64)
    visibility_per_aperture = np.mean(visibility, axis=0) if visibility.size else np.zeros(0, dtype=np.float64)

    critical_sep_px = 2.0 * sigma_px
    warning_sep_px = 4.0 * sigma_px
    critical_overlap_frac = float(np.mean(nearest_cross_pulse_px < critical_sep_px)) if nearest_cross_pulse_px.size else 0.0
    warning_overlap_frac = float(np.mean(nearest_cross_pulse_px < warning_sep_px)) if nearest_cross_pulse_px.size else 0.0
    min_cross_pulse_sep_px = float(np.min(nearest_cross_pulse_px)) if nearest_cross_pulse_px.size else float("inf")
    min_cross_pulse_sep_sigma = float(min_cross_pulse_sep_px / max(sigma_px, 1e-9)) if np.isfinite(min_cross_pulse_sep_px) else float("inf")

    visibility_ratio = float(np.mean(visibility)) if visibility.size else 0.0
    separation_score = float(np.clip(min_cross_pulse_sep_px / max(warning_sep_px, 1e-9), 0.0, 1.0)) if np.isfinite(min_cross_pulse_sep_px) else 1.0
    occupancy_score = float(max(0.0, 1.0 - critical_overlap_frac))
    identifiability_score = float(np.clip(min(visibility_ratio, separation_score, occupancy_score), 0.0, 1.0))
    if identifiability_score >= 0.8:
        design_status = "good"
    elif identifiability_score >= 0.5:
        design_status = "warning"
    else:
        design_status = "poor"

    return {
        "movement_mode": movement_info.mode,
        "is_uniform": bool(movement_info.is_uniform),
        "n_pulses": int(global_offsets_px.shape[0]),
        "n_apertures": int(ideal_centers_px.shape[0]),
        "n_slots_total": int(np.prod(slot_positions_px.shape[:2])),
        "spot_sigma_px": float(sigma_px),
        "support_radius_px": int(support_radius_px),
        "margin_px": float(margin_px),
        "net_displacement_px": np.asarray(global_offsets_px[-1] - global_offsets_px[0], dtype=np.float64),
        "net_displacement_m": np.asarray(global_offsets_m[-1] - global_offsets_m[0], dtype=np.float64),
        "step_length_px": {
            "min": float(np.min(step_lengths_px)) if step_lengths_px.size else 0.0,
            "max": float(np.max(step_lengths_px)) if step_lengths_px.size else 0.0,
            "mean": float(np.mean(step_lengths_px)) if step_lengths_px.size else 0.0,
            "std": float(np.std(step_lengths_px)) if step_lengths_px.size else 0.0,
            "values": np.asarray(step_lengths_px, dtype=np.float64),
        },
        "step_length_m": {
            "min": float(np.min(step_lengths_m)) if step_lengths_m.size else 0.0,
            "max": float(np.max(step_lengths_m)) if step_lengths_m.size else 0.0,
            "mean": float(np.mean(step_lengths_m)) if step_lengths_m.size else 0.0,
            "std": float(np.std(step_lengths_m)) if step_lengths_m.size else 0.0,
            "values": np.asarray(step_lengths_m, dtype=np.float64),
        },
        "pairwise_pulse_offset_separation_px": {
            "min": float(np.min(pairwise_offset_dist_px)) if pairwise_offset_dist_px.size else 0.0,
            "max": float(np.max(pairwise_offset_dist_px)) if pairwise_offset_dist_px.size else 0.0,
            "mean": float(np.mean(pairwise_offset_dist_px)) if pairwise_offset_dist_px.size else 0.0,
        },
        "pairwise_pulse_offset_separation_m": {
            "min": float(np.min(pairwise_offset_dist_m)) if pairwise_offset_dist_m.size else 0.0,
            "max": float(np.max(pairwise_offset_dist_m)) if pairwise_offset_dist_m.size else 0.0,
            "mean": float(np.mean(pairwise_offset_dist_m)) if pairwise_offset_dist_m.size else 0.0,
        },
        "visibility": {
            "overall_ratio": visibility_ratio,
            "per_pulse_ratio": np.asarray(visibility_per_pulse, dtype=np.float64),
            "per_aperture_ratio": np.asarray(visibility_per_aperture, dtype=np.float64),
        },
        "cross_pulse_separation_px": {
            "min": min_cross_pulse_sep_px,
            "mean": float(np.mean(nearest_cross_pulse_px)) if nearest_cross_pulse_px.size else float("inf"),
            "median": float(np.median(nearest_cross_pulse_px)) if nearest_cross_pulse_px.size else float("inf"),
            "min_over_sigma": min_cross_pulse_sep_sigma,
            "critical_threshold_px": float(critical_sep_px),
            "warning_threshold_px": float(warning_sep_px),
            "critical_overlap_fraction": critical_overlap_frac,
            "warning_overlap_fraction": warning_overlap_frac,
        },
        "detector_occupancy": {
            "occupied_pixel_ratio": occupancy_ratio,
            "max_stack_depth": max_stack_depth,
            "mean_stack_depth_on_support": mean_stack_depth,
        },
        "identifiability_score": identifiability_score,
        "design_status": design_status,
        "global_offsets_px": np.asarray(global_offsets_px, dtype=np.float64),
        "ideal_centers_px": np.asarray(ideal_centers_px, dtype=np.float64),
        "slot_positions_px": np.asarray(slot_positions_px, dtype=np.float64),
        "visibility_mask": np.asarray(visibility, dtype=bool),
        "occupancy_map": np.asarray(occupancy_map, dtype=np.int32),
        "nearest_cross_pulse_separation_px_values": np.asarray(nearest_cross_pulse_px, dtype=np.float64),
    }


def save_trajectory_design_report(result: Dict[str, Any], output_dir: str) -> None:
    """Save trajectory-design metrics and visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    dump_json_serializable(
        result,
        os.path.join(output_dir, "trajectory_design_summary.json"),
        stringify_dict_keys=True,
    )

    np.save(os.path.join(output_dir, "trajectory_design_occupancy_map.npy"), result["occupancy_map"])
    np.save(os.path.join(output_dir, "trajectory_design_slot_positions.npy"), result["slot_positions_px"])
    np.save(os.path.join(output_dir, "trajectory_design_visibility_mask.npy"), result["visibility_mask"])

    offsets = np.asarray(result["global_offsets_px"], dtype=np.float64)
    plt.figure(figsize=(6, 5))
    plt.plot(offsets[:, 0], offsets[:, 1], "-o", color="tab:blue", lw=1.5, ms=4)
    for i, (x, y) in enumerate(offsets):
        plt.text(x, y, str(i), fontsize=8, ha="left", va="bottom")
    plt.xlabel("Offset X (px)")
    plt.ylabel("Offset Y (px)")
    plt.title("Pulse Trajectory in Detector Pixels")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_path.png"), dpi=150)
    plt.close()

    slot_positions = np.asarray(result["slot_positions_px"], dtype=np.float64)
    visibility = np.asarray(result["visibility_mask"], dtype=bool)
    plt.figure(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for t in range(slot_positions.shape[0]):
        pts = slot_positions[t]
        vis = visibility[t]
        plt.scatter(pts[vis, 0], pts[vis, 1], s=8, alpha=0.6, color=cmap(t % 10), label=f"pulse {t}")
        if np.any(~vis):
            plt.scatter(pts[~vis, 0], pts[~vis, 1], s=8, alpha=0.35, color=cmap(t % 10), marker="x")
    plt.gca().invert_yaxis()
    plt.xlabel("Detector X (px)")
    plt.ylabel("Detector Y (px)")
    plt.title("All Slot Positions on Detector")
    if slot_positions.shape[0] <= 12:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_slot_positions.png"), dpi=150)
    plt.close()

    occupancy_map = np.asarray(result["occupancy_map"], dtype=np.float64)
    plt.figure(figsize=(7, 6))
    im = plt.imshow(occupancy_map, cmap="inferno", origin="lower")
    plt.colorbar(im, label="Support stack depth")
    plt.title("Trajectory Occupancy / Stack Depth")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_occupancy.png"), dpi=150)
    plt.close()

    sep_values = np.asarray(result["nearest_cross_pulse_separation_px_values"], dtype=np.float64)
    if sep_values.size:
        critical_sep_px = float(result["cross_pulse_separation_px"]["critical_threshold_px"])
        warning_sep_px = float(result["cross_pulse_separation_px"]["warning_threshold_px"])
        plt.figure(figsize=(7, 4))
        plt.hist(sep_values, bins=min(40, max(10, int(np.sqrt(sep_values.size)))), color="tab:blue", alpha=0.8, edgecolor="white")
        plt.axvline(critical_sep_px, color="tab:red", linestyle="--", lw=1.5, label=f"critical = {critical_sep_px:.2f}px")
        plt.axvline(warning_sep_px, color="tab:orange", linestyle="--", lw=1.5, label=f"warning = {warning_sep_px:.2f}px")
        plt.xlabel("Nearest cross-pulse separation (px)")
        plt.ylabel("Count")
        plt.title("Cross-Pulse Separation Histogram")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "trajectory_cross_pulse_separation_hist.png"), dpi=150)
        plt.close()


def run_trajectory_design_analysis(
    *,
    movement_info: MovementInfo,
    truth: Dict[str, Any],
    detector_shape: tuple[int, int],
    ideal_accumulated_image: np.ndarray,
    output_dir: str,
) -> Dict[str, Any]:
    """Compute and save a trajectory-design report."""
    result = analyze_trajectory_design(
        movement_info=movement_info,
        truth=truth,
        detector_shape=detector_shape,
        ideal_accumulated_image=ideal_accumulated_image,
    )
    save_trajectory_design_report(result, output_dir)
    return result
