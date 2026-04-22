from __future__ import annotations

"""Local linearized displacement recovery for accumulated Hartmann frames.

This module avoids the scale bias introduced by extracting per-pulse images
with static decomposition weights.  It solves the first-order image-shift
model directly on the accumulated residual:

    accumulated - ideal ~= sum_p (-Ix_p * dx_p - Iy_p * dy_p)

The templates are the real per-pulse ideal images, so the method does not
assume Gaussian spot shapes.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from analysis.displacement_methods import detect_beamlets


@dataclass
class LinearizedAccumulatedResult:
    """Recovered per-pulse displacement fields and solver diagnostics."""

    all_shifts_px: np.ndarray
    all_shift_confidence: np.ndarray
    condition_numbers: np.ndarray
    residual_ratios: np.ndarray
    median_data_scale: float
    spatial_prior_weight: float
    temporal_prior_weight: float


def _box_bounds(box) -> tuple[int, int, int, int]:
    """Return `(y0, y1, x0, x1)` for a detect_beamlets ROI box."""
    (y0, y1), (x0, x1) = box
    return int(y0), int(y1), int(x0), int(x1)


def _build_reference_geometry(
    pulse_ideal_images: Sequence[np.ndarray],
    rows: int,
    cols: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[list]]:
    """Precompute image gradients and spot boxes for each pulse reference."""
    gradients: list[tuple[np.ndarray, np.ndarray]] = []
    boxes_by_pulse: list[list] = []
    for ref_image in pulse_ideal_images:
        ref = np.asarray(ref_image, dtype=np.float64)
        grad_y, grad_x = np.gradient(ref)
        gradients.append((grad_x, grad_y))
        boxes, _ = detect_beamlets(ref, rows, cols)
        if len(boxes) != rows * cols:
            raise ValueError(
                f"detect_beamlets returned {len(boxes)} boxes for grid {rows}x{cols}"
            )
        boxes_by_pulse.append(boxes)
    return gradients, boxes_by_pulse


def _spot_union_roi(
    boxes_by_pulse: Sequence[Sequence],
    *,
    spot_index: int,
    image_shape: tuple[int, int],
    expand_px: int,
) -> tuple[int, int, int, int]:
    """Return an expanded ROI enclosing the same grid spot across all pulses."""
    height, width = image_shape
    y0 = height
    y1 = 0
    x0 = width
    x1 = 0
    for boxes in boxes_by_pulse:
        by0, by1, bx0, bx1 = _box_bounds(boxes[spot_index])
        y0 = min(y0, by0)
        y1 = max(y1, by1)
        x0 = min(x0, bx0)
        x1 = max(x1, bx1)

    pad = int(max(0, expand_px))
    return (
        max(0, y0 - pad),
        min(height, y1 + pad),
        max(0, x0 - pad),
        min(width, x1 + pad),
    )


def _weighted_solver_inputs(
    *,
    residual_patch: np.ndarray,
    ideal_patch: np.ndarray,
    design_matrix: np.ndarray,
    weight_power: float,
    min_weight: float,
    signal_quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply stable signal weighting and remove the lowest-signal pixels."""
    signal = np.maximum(np.asarray(ideal_patch, dtype=np.float64), 0.0)
    if signal.size == 0:
        raise ValueError("empty ROI patch")

    signal_max = float(np.max(signal))
    if signal_max > 1e-12:
        normalized_signal = signal / signal_max
        weights = np.power(normalized_signal, float(weight_power))
    else:
        normalized_signal = np.ones_like(signal)
        weights = np.ones_like(signal)

    weights = np.sqrt(np.clip(weights, float(min_weight), 1.0)).ravel()
    mask_threshold = float(np.quantile(weights, float(signal_quantile)))
    signal_mask = weights > mask_threshold
    if int(np.count_nonzero(signal_mask)) < design_matrix.shape[1] + 4:
        signal_mask = normalized_signal.ravel() > 0.0
    if int(np.count_nonzero(signal_mask)) < design_matrix.shape[1] + 4:
        signal_mask = np.ones_like(weights, dtype=bool)

    weighted_design = design_matrix[signal_mask] * weights[signal_mask, np.newaxis]
    weighted_residual = residual_patch.ravel()[signal_mask] * weights[signal_mask]
    return weighted_design, weighted_residual


def _solve_regularized_least_squares(
    weighted_design: np.ndarray,
    weighted_residual: np.ndarray,
    *,
    ridge_relative: float,
) -> tuple[np.ndarray, float, float]:
    """Solve a column-normalized ridge least-squares problem."""
    column_norms = np.sqrt(np.sum(weighted_design * weighted_design, axis=0)) + 1e-12
    normalized_design = weighted_design / column_norms
    normal_matrix = normalized_design.T @ normalized_design
    rhs = normalized_design.T @ weighted_residual

    ridge_scale = float(np.trace(normal_matrix) / max(normal_matrix.shape[0], 1))
    ridge = float(ridge_relative) * ridge_scale
    system = normal_matrix + ridge * np.eye(normal_matrix.shape[0])
    solution = np.linalg.solve(system, rhs) / column_norms

    predicted = weighted_design @ solution
    residual_norm = float(np.linalg.norm(weighted_residual - predicted))
    target_norm = float(np.linalg.norm(weighted_residual)) + 1e-12
    residual_ratio = residual_norm / target_norm
    condition_number = float(np.linalg.cond(system))
    return solution, condition_number, residual_ratio


def _confidence_from_solver(
    *,
    condition_number: float,
    residual_ratio: float,
    condition_scale: float,
) -> float:
    """Map local solver health to a conservative 0..1 confidence."""
    cond_confidence = 1.0 / (1.0 + max(float(condition_number) - 1.0, 0.0) / float(condition_scale))
    fit_confidence = np.exp(-max(float(residual_ratio), 0.0))
    return float(np.clip(cond_confidence * fit_confidence, 0.05, 1.0))


def _build_grid_laplacian(rows: int, cols: int) -> sparse.csr_matrix:
    """Return a 4-neighbour Laplacian over the Hartmann grid."""
    entries_row: list[int] = []
    entries_col: list[int] = []
    entries_val: list[float] = []

    def _index(row_idx: int, col_idx: int) -> int:
        return row_idx * cols + col_idx

    for row_idx in range(rows):
        for col_idx in range(cols):
            idx = _index(row_idx, col_idx)
            degree = 0
            for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nbr_row = row_idx + d_row
                nbr_col = col_idx + d_col
                if 0 <= nbr_row < rows and 0 <= nbr_col < cols:
                    degree += 1
                    entries_row.append(idx)
                    entries_col.append(_index(nbr_row, nbr_col))
                    entries_val.append(-1.0)
            entries_row.append(idx)
            entries_col.append(idx)
            entries_val.append(float(degree))

    size = rows * cols
    return sparse.coo_matrix((entries_val, (entries_row, entries_col)), shape=(size, size)).tocsr()


def _build_chain_laplacian(length: int) -> sparse.csr_matrix:
    """Return a 1D chain Laplacian."""
    if length <= 1:
        return sparse.csr_matrix((length, length), dtype=np.float64)
    main_diag = np.full(length, 2.0, dtype=np.float64)
    main_diag[0] = 1.0
    main_diag[-1] = 1.0
    off_diag = -np.ones(length - 1, dtype=np.float64)
    return sparse.diags(
        diagonals=[off_diag, main_diag, off_diag],
        offsets=[-1, 0, 1],
        shape=(length, length),
        format="csr",
    )


def _evaluate_solution_residual_ratio(
    weighted_design: np.ndarray,
    weighted_residual: np.ndarray,
    solution: np.ndarray,
) -> float:
    """Compute the post-fit residual ratio for one spot."""
    predicted = weighted_design @ solution
    residual_norm = float(np.linalg.norm(weighted_residual - predicted))
    target_norm = float(np.linalg.norm(weighted_residual)) + 1e-12
    return residual_norm / target_norm


def _refine_with_global_priors(
    *,
    initial_shifts: np.ndarray,
    local_confidence: np.ndarray,
    rows: int,
    cols: int,
    n_pulses: int,
    spatial_prior_relative: float,
    temporal_prior_relative: float,
    confidence_data_power: float,
    confidence_data_floor: float,
) -> np.ndarray | None:
    """Jointly refine all spots with confidence-weighted spatial and temporal smoothness."""
    n_spots = int(rows) * int(cols)

    data_floor = float(max(confidence_data_floor, 1e-6))
    data_power = float(max(confidence_data_power, 0.0))
    data_weights = np.clip(np.asarray(local_confidence, dtype=np.float64), data_floor, 1.0) ** data_power

    if (
        float(spatial_prior_relative) <= 0.0 and
        (float(temporal_prior_relative) <= 0.0 or int(n_pulses) <= 1)
    ):
        return None

    data_diag = sparse.diags(data_weights, offsets=0, shape=(n_spots, n_spots), format="csr")
    spatial_laplacian = _build_grid_laplacian(int(rows), int(cols))
    per_pulse_spot_system = data_diag + float(spatial_prior_relative) * spatial_laplacian
    system = sparse.kron(
        sparse.eye(int(n_pulses), format="csr"),
        per_pulse_spot_system,
        format="csr",
    )
    if float(temporal_prior_relative) > 0.0 and int(n_pulses) > 1:
        temporal_laplacian = _build_chain_laplacian(int(n_pulses))
        system = system + float(temporal_prior_relative) * sparse.kron(
            temporal_laplacian,
            sparse.eye(n_spots, format="csr"),
            format="csr",
        )
    system_csc = system.tocsc()

    initial = np.asarray(initial_shifts, dtype=np.float64)
    refined = np.zeros_like(initial)

    for component_idx in range(2):
        rhs = (initial[..., component_idx].reshape(int(n_pulses), n_spots) * data_weights[np.newaxis, :]).reshape(-1)
        refined_component = sparse_linalg.spsolve(system_csc, rhs)
        refined[..., component_idx] = np.asarray(refined_component, dtype=np.float64).reshape(
            int(n_pulses), int(rows), int(cols)
        )

    return refined.reshape(int(n_pulses), int(rows), int(cols), 2)


def recover_linearized_accumulated_shifts(
    *,
    accumulated_image: np.ndarray,
    pulse_ideal_images: Sequence[np.ndarray],
    rows: int,
    cols: int,
    ideal_accumulated_image: np.ndarray | None = None,
    roi_expand_px: int = 10,
    ridge_relative: float = 1e-2,
    weight_power: float = 0.5,
    min_weight: float = 0.02,
    signal_quantile: float = 0.20,
    smooth_sigma: float = 0.0,
    confidence_condition_scale: float = 20.0,
    spatial_prior_relative: float = 0.0,
    temporal_prior_relative: float = 0.0,
    confidence_data_power: float = 1.5,
    confidence_data_floor: float = 0.15,
) -> LinearizedAccumulatedResult:
    """Recover per-pulse shifts directly from an accumulated detector image.

    Parameters
    ----------
    accumulated_image:
        Aberrated accumulated detector image.
    pulse_ideal_images:
        Per-pulse ideal detector images at the known trajectory positions.
    rows, cols:
        Hartmann grid shape.
    ideal_accumulated_image:
        Optional ideal accumulated image.  When omitted, the sum of pulse
        references is used, keeping the residual consistent with the templates.
    """
    pulse_refs = [np.asarray(img, dtype=np.float64) for img in pulse_ideal_images]
    if not pulse_refs:
        raise ValueError("pulse_ideal_images is empty")

    accumulated = np.asarray(accumulated_image, dtype=np.float64)
    if accumulated.ndim != 2:
        raise ValueError(f"accumulated_image must be 2D, got {accumulated.shape}")
    if any(ref.shape != accumulated.shape for ref in pulse_refs):
        raise ValueError("all pulse ideal images must match accumulated_image shape")

    reference_sum = np.sum(np.stack(pulse_refs, axis=0), axis=0)
    if ideal_accumulated_image is None:
        ideal_accumulated = reference_sum
    else:
        ideal_accumulated = np.asarray(ideal_accumulated_image, dtype=np.float64)
        if ideal_accumulated.shape != accumulated.shape:
            ideal_accumulated = reference_sum

    residual_image = accumulated - ideal_accumulated
    gradients, boxes_by_pulse = _build_reference_geometry(pulse_refs, int(rows), int(cols))

    n_pulses = len(pulse_refs)
    shifts = np.zeros((n_pulses, int(rows), int(cols), 2), dtype=np.float64)
    confidence = np.zeros((n_pulses, int(rows), int(cols)), dtype=np.float64)
    condition_numbers = np.zeros((int(rows), int(cols)), dtype=np.float64)
    residual_ratios = np.zeros((int(rows), int(cols)), dtype=np.float64)
    local_confidence_scalar = np.zeros(int(rows) * int(cols), dtype=np.float64)

    for row in range(int(rows)):
        for col in range(int(cols)):
            spot_index = row * int(cols) + col
            y0, y1, x0, x1 = _spot_union_roi(
                boxes_by_pulse,
                spot_index=spot_index,
                image_shape=accumulated.shape,
                expand_px=int(roi_expand_px),
            )

            design_columns = []
            for grad_x, grad_y in gradients:
                design_columns.append((-grad_x[y0:y1, x0:x1]).ravel())
                design_columns.append((-grad_y[y0:y1, x0:x1]).ravel())
            design_matrix = np.stack(design_columns, axis=1)

            weighted_design, weighted_residual = _weighted_solver_inputs(
                residual_patch=residual_image[y0:y1, x0:x1],
                ideal_patch=reference_sum[y0:y1, x0:x1],
                design_matrix=design_matrix,
                weight_power=float(weight_power),
                min_weight=float(min_weight),
                signal_quantile=float(signal_quantile),
            )
            solution, condition_number, residual_ratio = _solve_regularized_least_squares(
                weighted_design,
                weighted_residual,
                ridge_relative=float(ridge_relative),
            )

            condition_numbers[row, col] = condition_number
            residual_ratios[row, col] = residual_ratio
            spot_confidence = _confidence_from_solver(
                condition_number=condition_number,
                residual_ratio=residual_ratio,
                condition_scale=float(confidence_condition_scale),
            )
            local_confidence_scalar[spot_index] = spot_confidence
            for pulse_index in range(n_pulses):
                shifts[pulse_index, row, col, 0] = solution[2 * pulse_index]
                shifts[pulse_index, row, col, 1] = solution[2 * pulse_index + 1]
                confidence[pulse_index, row, col] = spot_confidence

    median_data_scale = float(np.median(local_confidence_scalar)) if local_confidence_scalar.size else 0.0

    refined_solution = _refine_with_global_priors(
        initial_shifts=shifts,
        local_confidence=local_confidence_scalar,
        rows=int(rows),
        cols=int(cols),
        n_pulses=n_pulses,
        spatial_prior_relative=float(spatial_prior_relative),
        temporal_prior_relative=float(temporal_prior_relative),
        confidence_data_power=float(confidence_data_power),
        confidence_data_floor=float(confidence_data_floor),
    )
    if refined_solution is not None:
        refined_shifts = np.asarray(refined_solution, dtype=np.float64)
        refined_residual_ratios = np.zeros_like(residual_ratios)
        refined_confidence = np.zeros_like(confidence)
        for row in range(int(rows)):
            for col in range(int(cols)):
                spot_index = row * int(cols) + col
                spot_solution = np.zeros(2 * n_pulses, dtype=np.float64)
                for pulse_index in range(n_pulses):
                    spot_solution[2 * pulse_index] = refined_shifts[pulse_index, row, col, 0]
                    spot_solution[2 * pulse_index + 1] = refined_shifts[pulse_index, row, col, 1]

                y0, y1, x0, x1 = _spot_union_roi(
                    boxes_by_pulse,
                    spot_index=spot_index,
                    image_shape=accumulated.shape,
                    expand_px=int(roi_expand_px),
                )
                design_columns = []
                for grad_x, grad_y in gradients:
                    design_columns.append((-grad_x[y0:y1, x0:x1]).ravel())
                    design_columns.append((-grad_y[y0:y1, x0:x1]).ravel())
                design_matrix = np.stack(design_columns, axis=1)
                weighted_design, weighted_residual = _weighted_solver_inputs(
                    residual_patch=residual_image[y0:y1, x0:x1],
                    ideal_patch=reference_sum[y0:y1, x0:x1],
                    design_matrix=design_matrix,
                    weight_power=float(weight_power),
                    min_weight=float(min_weight),
                    signal_quantile=float(signal_quantile),
                )
                refined_residual_ratio = _evaluate_solution_residual_ratio(
                    weighted_design,
                    weighted_residual,
                    spot_solution,
                )
                refined_residual_ratios[row, col] = refined_residual_ratio
                spot_confidence = _confidence_from_solver(
                    condition_number=condition_numbers[row, col],
                    residual_ratio=refined_residual_ratio,
                    condition_scale=float(confidence_condition_scale),
                )
                refined_confidence[:, row, col] = spot_confidence

        shifts = refined_shifts
        residual_ratios = refined_residual_ratios
        confidence = refined_confidence

    if float(smooth_sigma) > 0.0:
        for pulse_index in range(n_pulses):
            shifts[pulse_index, :, :, 0] = gaussian_filter(
                shifts[pulse_index, :, :, 0],
                sigma=float(smooth_sigma),
                mode="nearest",
            )
            shifts[pulse_index, :, :, 1] = gaussian_filter(
                shifts[pulse_index, :, :, 1],
                sigma=float(smooth_sigma),
                mode="nearest",
            )

    return LinearizedAccumulatedResult(
        all_shifts_px=shifts,
        all_shift_confidence=confidence,
        condition_numbers=condition_numbers,
        residual_ratios=residual_ratios,
        median_data_scale=float(median_data_scale),
        spatial_prior_weight=float(spatial_prior_relative),
        temporal_prior_weight=float(temporal_prior_relative),
    )
