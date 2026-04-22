from __future__ import annotations
"""
Optimization-based shift recovery from accumulated Hartmann images.

This module recovers per-frame spot displacements from a single accumulated image
by solving an inverse problem:

    min_shifts ||render(ideal_centers + global_shifts + local_shifts) - I_obs||²
              + λ_spatial * spatial_smoothness
              + λ_temporal * temporal_smoothness

Two solver modes are supported:
1. direct: optimize per-spot shifts directly
2. zernike: parameterize shifts using Zernike polynomials (fewer parameters)
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Callable
import time

from analysis.optimization_common import (
    add_shifted_template_patch_xy,
    assert_row_col_indices,
    assert_xy_points,
    build_neighbor_graph,
    coerce_global_offsets_xy,
    estimate_spot_fwhm_sigma_from_image,
    extract_centered_patch_xy,
    gaussian_shift_gradients_xy,
    render_gaussian_sequence_xy,
    shifted_template_patch_gradient_xy,
)

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class OptimizationResult:
    """Result of optimization-based shift recovery."""
    shifts: np.ndarray  # [n_frames, n_valid_spots, 2]
    final_loss: float
    convergence_history: List[float] = field(default_factory=list)
    solver_method: str = "direct"
    n_iterations: int = 0
    elapsed_time: float = 0.0
    
    # Valid centers after boundary filtering
    valid_centers: Optional[np.ndarray] = None  # [n_valid_spots, 2] as (x, y)
    valid_center_indices: Optional[np.ndarray] = None  # indices into the original center list
    
    # Per-frame metrics (computed after recovery)
    per_frame_displacement_x_m: Optional[np.ndarray] = None  # [n_frames, n_spots]
    per_frame_displacement_y_m: Optional[np.ndarray] = None
    per_frame_phase_rad: Optional[np.ndarray] = None


def _template_patch_radius(sigma: float, spacing: float) -> int:
    """Choose a stable patch radius for template-based rendering."""
    radius_from_sigma = int(np.ceil(3.0 * float(sigma) + 2.0))
    radius_from_spacing = int(np.floor(max(float(spacing), 1.0) * 0.45))
    return max(4, min(radius_from_sigma, max(radius_from_spacing, 4)))


def build_template_bank(
    pulse_reference_images: List[np.ndarray],
    ideal_centers: np.ndarray,
    global_offsets_px: np.ndarray,
    spacing: float,
    sigma: float,
) -> Dict[str, np.ndarray]:
    """Extract per-pulse, per-spot template patches and their spatial gradients."""
    ideal_centers = assert_xy_points(ideal_centers, name="ideal_centers")
    n_frames = len(pulse_reference_images)
    n_spots = len(ideal_centers)
    offsets = coerce_global_offsets_xy(global_offsets_px, n_frames=n_frames, name="global_offsets_px")
    radius = _template_patch_radius(sigma, spacing)
    patch_shape = (2 * radius + 1, 2 * radius + 1)

    templates = np.zeros((n_frames, n_spots, *patch_shape), dtype=np.float64)
    grad_x = np.zeros_like(templates)
    grad_y = np.zeros_like(templates)
    centers = np.zeros((n_frames, n_spots, 2), dtype=np.float64)

    for t in range(n_frames):
        ref_img = np.asarray(pulse_reference_images[t], dtype=np.float64)
        for i in range(n_spots):
            base_center = np.asarray(ideal_centers[i], dtype=np.float64) + offsets[t]
            centers[t, i] = base_center
            patch = extract_centered_patch_xy(ref_img, base_center, radius)
            templates[t, i] = patch
            patch_grad_y, patch_grad_x = np.gradient(patch)
            grad_x[t, i] = patch_grad_x
            grad_y[t, i] = patch_grad_y

    return {
        "templates": templates,
        "grad_x": grad_x,
        "grad_y": grad_y,
        "centers": centers,
        "radius": radius,
    }


# =============================================================================
# Boundary Filtering
# =============================================================================

def filter_valid_centers(
    centers: np.ndarray, 
    shape: Tuple[int, int], 
    n_frames: int, 
    global_offsets_px: np.ndarray, 
    margin: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter centers to only include those that remain inside the image
    for all frames, with a safety margin.
    
    Parameters
    ----------
    centers : np.ndarray
        Shape (n_spots, 2) as (x, y)
    shape : tuple
        Image shape (H, W)
    n_frames : int
        Number of frames
    global_offsets_px : np.ndarray
        Absolute global offsets, shape (n_frames, 2), as (dx, dy) in pixels
    margin : float
        Safety margin from image edge
        
    Returns
    -------
    valid_centers : np.ndarray
        Filtered centers
    valid_mask : np.ndarray
        Boolean mask indicating which original centers are valid
    """
    centers = assert_xy_points(centers, name="centers")
    h, w = shape
    offsets = coerce_global_offsets_xy(global_offsets_px, n_frames=n_frames, name="global_offsets_px")
    min_global_dx_px = float(np.min(offsets[:, 0])) if len(offsets) else 0.0
    max_global_dx_px = float(np.max(offsets[:, 0])) if len(offsets) else 0.0
    min_global_dy_px = float(np.min(offsets[:, 1])) if len(offsets) else 0.0
    max_global_dy_px = float(np.max(offsets[:, 1])) if len(offsets) else 0.0

    valid_mask = (
        (centers[:, 1] + min_global_dy_px >= margin) &
        (centers[:, 1] + max_global_dy_px < h - margin) &
        (centers[:, 0] + min_global_dx_px >= margin) &
        (centers[:, 0] + max_global_dx_px < w - margin)
    )
    
    return centers[valid_mask], valid_mask


# =============================================================================
# Zernike Basis (from shared module)
# =============================================================================

from analysis.zernike_basis import (
    compute_zernike_basis_matrix as compute_zernike_basis,
    zernike_coeffs_to_shifts,
)


# =============================================================================
# Objective Functions with Analytical Gradients
# =============================================================================

class ConvergenceCallback:
    """Callback to record convergence history."""
    
    def __init__(self, verbose: bool = False):
        self.history = []
        self.verbose = verbose
        self.n_calls = 0
        self.pbar = None
    
    def __call__(self, xk):
        self.n_calls += 1
        if self.pbar is not None:
            self.pbar.update(1)


def _spatial_regularization_loss(
    local_shifts: np.ndarray,
    neighbors: Optional[List[List[int]]],
) -> float:
    """Compute the unweighted spatial smoothness penalty on local shifts."""
    if neighbors is None:
        return 0.0
    n_frames, n_spots, _ = local_shifts.shape
    reg_spatial = 0.0
    for t in range(n_frames):
        for i in range(n_spots):
            for j in neighbors[i]:
                if j > i:
                    diff_xy = local_shifts[t, i] - local_shifts[t, j]
                    reg_spatial += float(np.sum(diff_xy**2))
    return reg_spatial


def _temporal_regularization_loss(local_shifts: np.ndarray) -> float:
    """Compute the unweighted temporal smoothness penalty on local shifts."""
    if local_shifts.shape[0] < 2:
        return 0.0
    reg_temporal = 0.0
    for t in range(local_shifts.shape[0] - 1):
        diff_xy = local_shifts[t + 1] - local_shifts[t]
        reg_temporal += float(np.sum(diff_xy**2))
    return reg_temporal


def _apply_spatial_regularization_grad(
    grads: np.ndarray,
    *,
    local_shifts: np.ndarray,
    neighbors: Optional[List[List[int]]],
    lambda_spatial: float,
) -> None:
    """Add the weighted spatial regularization gradient in-place."""
    if lambda_spatial <= 0 or neighbors is None:
        return
    n_frames, n_spots, _ = local_shifts.shape
    for t in range(n_frames):
        for i in range(n_spots):
            for j in neighbors[i]:
                diff_xy = local_shifts[t, i] - local_shifts[t, j]
                grads[t, i] += 2.0 * lambda_spatial * diff_xy


def _apply_temporal_regularization_grad(
    grads: np.ndarray,
    *,
    local_shifts: np.ndarray,
    lambda_temporal: float,
) -> None:
    """Add the weighted temporal regularization gradient in-place."""
    if lambda_temporal <= 0:
        return
    n_frames = local_shifts.shape[0]
    for t in range(n_frames):
        if t > 0:
            grads[t] += 2.0 * lambda_temporal * (local_shifts[t] - local_shifts[t - 1])
        if t < n_frames - 1:
            grads[t] -= 2.0 * lambda_temporal * (local_shifts[t + 1] - local_shifts[t])


def _zernike_temporal_regularization_loss(coeffs: np.ndarray) -> float:
    """Compute the unweighted temporal smoothness penalty on Zernike coefficients."""
    if coeffs.shape[0] < 2:
        return 0.0
    reg_temporal = 0.0
    for t in range(coeffs.shape[0] - 1):
        diff = coeffs[t + 1] - coeffs[t]
        reg_temporal += float(np.sum(diff**2))
    return reg_temporal


def _apply_zernike_temporal_regularization_grad(
    grad_coeffs: np.ndarray,
    *,
    coeffs: np.ndarray,
    lambda_temporal: float,
) -> None:
    """Add the weighted temporal smoothness gradient on Zernike coefficients in-place."""
    if lambda_temporal <= 0:
        return
    n_frames = coeffs.shape[0]
    for t in range(n_frames):
        if t > 0:
            grad_coeffs[t] += 2.0 * lambda_temporal * (coeffs[t] - coeffs[t - 1])
        if t < n_frames - 1:
            grad_coeffs[t] -= 2.0 * lambda_temporal * (coeffs[t + 1] - coeffs[t])


def _coeff_l2_regularization_loss(coeffs: np.ndarray) -> float:
    """Compute the unweighted L2 penalty on coefficient arrays."""
    return float(np.sum(coeffs**2))


def _apply_coeff_l2_regularization_grad(
    grad_coeffs: np.ndarray,
    *,
    coeffs: np.ndarray,
    lambda_coeff: float,
) -> None:
    """Add the weighted L2 coefficient gradient in-place."""
    if lambda_coeff <= 0:
        return
    grad_coeffs += 2.0 * lambda_coeff * coeffs


def _run_lbfgsb_minimize(
    *,
    objective_fun: Callable,
    initial_guess: np.ndarray,
    objective_args: tuple,
    bounds,
    max_iter: int,
    ftol: float,
    gtol: float,
    verbose: bool,
    progress_desc: str,
) -> tuple[Any, List[float], ConvergenceCallback, float]:
    """Shared L-BFGS-B runner with optional tqdm progress handling."""
    history: List[float] = []
    callback = ConvergenceCallback(verbose=verbose)
    if HAS_TQDM and verbose:
        callback.pbar = tqdm(total=max_iter, desc=progress_desc)

    start_time = time.time()
    result = minimize(
        fun=objective_fun,
        x0=initial_guess,
        args=(*objective_args, history),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        callback=callback,
        options={'disp': verbose and not HAS_TQDM, 'maxiter': max_iter, 'ftol': float(ftol), 'gtol': float(gtol)},
    )
    elapsed = time.time() - start_time

    if callback.pbar is not None:
        callback.pbar.close()
    return result, history, callback, elapsed


def _accumulated_residual_and_data_loss(
    accumulated_pred: np.ndarray,
    Y_obs: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute the residual image and squared-error data loss."""
    diff = accumulated_pred - Y_obs
    data_loss = float(np.sum(diff**2))
    return diff, data_loss


def _record_loss_history(total_loss: float, history: Optional[List[float]]) -> None:
    """Append the current loss value when convergence history tracking is enabled."""
    if history is not None:
        history.append(float(total_loss))


def _gaussian_shift_gradient_stack(
    *,
    diff: np.ndarray,
    all_frame_centers: np.ndarray,
    sigma: float,
    Y_grid: np.ndarray,
    X_grid: np.ndarray,
) -> np.ndarray:
    """Backpropagate a shared residual image to per-frame Gaussian shift gradients."""
    n_frames, n_spots, _ = all_frame_centers.shape
    grads = np.zeros((n_frames, n_spots, 2), dtype=np.float64)
    for t in range(n_frames):
        grads[t] = gaussian_shift_gradients_xy(
            diff_image=diff,
            centers_xy=all_frame_centers[t],
            sigma=sigma,
            y_grid=Y_grid,
            x_grid=X_grid,
        )
    return grads


def _render_template_sequence_with_roi_bounds(
    *,
    shape: Tuple[int, int],
    templates: np.ndarray,
    base_centers: np.ndarray,
    local_shifts: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, list[list[Optional[Tuple[int, int, int, int, int, int, int, int]]]]]:
    """Render a shifted template sequence and retain the per-patch ROI tuples."""
    n_frames, n_spots = templates.shape[:2]
    accumulated_pred = np.zeros(shape, dtype=np.float64)
    roi_bounds: list[list[Optional[Tuple[int, int, int, int, int, int, int, int]]]] = [
        [None] * n_spots for _ in range(n_frames)
    ]

    for t in range(n_frames):
        for i in range(n_spots):
            roi_bounds[t][i] = add_shifted_template_patch_xy(
                image=accumulated_pred,
                template_patch=templates[t, i],
                base_center_xy=base_centers[t, i],
                shift_xy=local_shifts[t, i],
                radius=radius,
            )

    return accumulated_pred, roi_bounds


def _template_patch_gradient_stack(
    *,
    diff: np.ndarray,
    grad_x_templates: np.ndarray,
    grad_y_templates: np.ndarray,
    base_centers: np.ndarray,
    local_shifts: np.ndarray,
    radius: int,
    roi_bounds: list[list[Optional[Tuple[int, int, int, int, int, int, int, int]]]],
) -> np.ndarray:
    """Backpropagate the residual image through the template-patch forward model."""
    n_frames, n_spots = grad_x_templates.shape[:2]
    grads = np.zeros((n_frames, n_spots, 2), dtype=np.float64)

    for t in range(n_frames):
        for i in range(n_spots):
            roi = roi_bounds[t][i]
            if roi is None:
                continue
            grads[t, i] = shifted_template_patch_gradient_xy(
                diff_image=diff,
                grad_x_patch=grad_x_templates[t, i],
                grad_y_patch=grad_y_templates[t, i],
                base_center_xy=base_centers[t, i],
                shift_xy=local_shifts[t, i],
                radius=radius,
                roi=roi,
            )

    return grads


def _zernike_coeff_gradient_from_shift_gradients(
    *,
    grad_shifts: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Apply the linear Zernike chain rule from per-spot shift gradients to coefficients."""
    n_frames, _, _ = grad_shifts.shape
    n_modes = basis.shape[1]
    grad_coeffs = np.zeros((n_frames, 2, n_modes), dtype=np.float64)
    basis_t = basis.T
    for t in range(n_frames):
        grad_coeffs[t, 0, :] = basis_t @ grad_shifts[t, :, 0]
        grad_coeffs[t, 1, :] = basis_t @ grad_shifts[t, :, 1]
    return grad_coeffs


def _build_optimization_result(
    *,
    shifts: np.ndarray,
    final_loss: float,
    convergence_history: List[float],
    solver_method: str,
    n_iterations: int,
    elapsed_time: float,
) -> OptimizationResult:
    """Construct a normalized optimization result payload."""
    return OptimizationResult(
        shifts=shifts,
        final_loss=float(final_loss),
        convergence_history=convergence_history,
        solver_method=solver_method,
        n_iterations=int(n_iterations),
        elapsed_time=float(elapsed_time),
    )


def _print_solver_completion(
    *,
    verbose: bool,
    elapsed: float,
    final_loss: float,
) -> None:
    """Report the final optimization status when verbose logging is enabled."""
    if verbose:
        print(f"Optimization finished in {elapsed:.2f}s, final loss: {final_loss:.4e}")


def _finalize_solver_result(
    *,
    result,
    history: List[float],
    callback: ConvergenceCallback,
    elapsed: float,
    verbose: bool,
    shifts: np.ndarray,
    solver_method: str,
) -> OptimizationResult:
    """Convert an optimizer result into the normalized OptimizationResult payload."""
    _print_solver_completion(
        verbose=verbose,
        elapsed=elapsed,
        final_loss=float(result.fun),
    )
    return _build_optimization_result(
        shifts=shifts,
        final_loss=result.fun,
        convergence_history=history,
        solver_method=solver_method,
        n_iterations=callback.n_calls,
        elapsed_time=elapsed,
    )


def objective_direct_with_grad(
    params: np.ndarray,
    shape: Tuple[int, int],
    ideal_centers: np.ndarray,
    n_frames: int,
    global_offsets_px: np.ndarray,
    sigma: float,
    Y_obs: np.ndarray,
    Y_grid: np.ndarray,
    X_grid: np.ndarray,
    neighbors: Optional[List[List[int]]],
    lambda_spatial: float,
    lambda_temporal: float,
    history: Optional[List[float]] = None
) -> Tuple[float, np.ndarray]:
    """
    Objective function for direct solver with analytical gradient.
    """
    n_spots = len(ideal_centers)
    local_shifts = params.reshape((n_frames, n_spots, 2))
    offsets = coerce_global_offsets_xy(global_offsets_px, n_frames=n_frames, name="global_offsets_px")
    
    # Forward pass: render accumulated image
    accumulated_pred, all_frame_centers = render_gaussian_sequence_xy(
        shape=shape,
        ideal_centers_xy=ideal_centers,
        global_offsets_xy=offsets,
        local_shifts_xy=local_shifts,
        sigma=sigma,
        y_grid=Y_grid,
        x_grid=X_grid,
    )
    
    # Data loss
    diff, data_loss = _accumulated_residual_and_data_loss(accumulated_pred, Y_obs)
    
    reg_spatial = _spatial_regularization_loss(local_shifts, neighbors)
    reg_temporal = _temporal_regularization_loss(local_shifts)
    
    total_loss = data_loss + lambda_spatial * reg_spatial + lambda_temporal * reg_temporal
    
    # Record history
    _record_loss_history(total_loss, history)
    
    # Backward pass: compute gradients
    grads = _gaussian_shift_gradient_stack(
        diff=diff,
        all_frame_centers=all_frame_centers,
        sigma=sigma,
        Y_grid=Y_grid,
        X_grid=X_grid,
    )
    
    _apply_spatial_regularization_grad(
        grads,
        local_shifts=local_shifts,
        neighbors=neighbors,
        lambda_spatial=lambda_spatial,
    )
    _apply_temporal_regularization_grad(
        grads,
        local_shifts=local_shifts,
        lambda_temporal=lambda_temporal,
    )
    
    return total_loss, grads.flatten()


def objective_direct_template_with_grad(
    params: np.ndarray,
    shape: Tuple[int, int],
    ideal_centers: np.ndarray,
    n_frames: int,
    global_offsets_px: np.ndarray,
    template_bank: Dict[str, np.ndarray],
    Y_obs: np.ndarray,
    neighbors: Optional[List[List[int]]],
    lambda_spatial: float,
    lambda_temporal: float,
    history: Optional[List[float]] = None
) -> Tuple[float, np.ndarray]:
    """Template-based direct objective using per-pulse ideal reference patches."""
    n_spots = len(ideal_centers)
    local_shifts = params.reshape((n_frames, n_spots, 2))
    offsets = coerce_global_offsets_xy(global_offsets_px, n_frames=n_frames, name="global_offsets_px")
    templates = template_bank["templates"]
    grad_x_templates = template_bank["grad_x"]
    grad_y_templates = template_bank["grad_y"]
    base_centers = template_bank["centers"]
    radius = int(template_bank["radius"])
    accumulated_pred, roi_bounds = _render_template_sequence_with_roi_bounds(
        shape=shape,
        templates=templates,
        base_centers=base_centers,
        local_shifts=local_shifts,
        radius=radius,
    )

    diff, data_loss = _accumulated_residual_and_data_loss(accumulated_pred, Y_obs)

    reg_spatial = _spatial_regularization_loss(local_shifts, neighbors)
    reg_temporal = _temporal_regularization_loss(local_shifts)

    total_loss = data_loss + lambda_spatial * reg_spatial + lambda_temporal * reg_temporal
    _record_loss_history(total_loss, history)

    grads = _template_patch_gradient_stack(
        diff=diff,
        grad_x_templates=grad_x_templates,
        grad_y_templates=grad_y_templates,
        base_centers=base_centers,
        local_shifts=local_shifts,
        radius=radius,
        roi_bounds=roi_bounds,
    )

    _apply_spatial_regularization_grad(
        grads,
        local_shifts=local_shifts,
        neighbors=neighbors,
        lambda_spatial=lambda_spatial,
    )
    _apply_temporal_regularization_grad(
        grads,
        local_shifts=local_shifts,
        lambda_temporal=lambda_temporal,
    )

    return total_loss, grads.flatten()


def objective_zernike_with_grad(
    params: np.ndarray,
    shape: Tuple[int, int],
    ideal_centers: np.ndarray,
    n_frames: int,
    global_offsets_px: np.ndarray,
    sigma: float,
    Y_obs: np.ndarray,
    Y_grid: np.ndarray,
    X_grid: np.ndarray,
    basis: np.ndarray,
    lambda_temporal: float,
    lambda_coeff: float,
    history: Optional[List[float]] = None
) -> Tuple[float, np.ndarray]:
    """
    Objective function for Zernike-constrained solver with analytical gradient.
    """
    n_modes = basis.shape[1]
    n_spots = len(ideal_centers)
    
    coeffs = params.reshape((n_frames, 2, n_modes))
    local_shifts = zernike_coeffs_to_shifts(params, basis, n_frames, n_spots)
    offsets = coerce_global_offsets_xy(global_offsets_px, n_frames=n_frames, name="global_offsets_px")
    
    # Forward pass
    accumulated_pred, all_frame_centers = render_gaussian_sequence_xy(
        shape=shape,
        ideal_centers_xy=ideal_centers,
        global_offsets_xy=offsets,
        local_shifts_xy=local_shifts,
        sigma=sigma,
        y_grid=Y_grid,
        x_grid=X_grid,
    )
    
    # Losses
    diff, data_loss = _accumulated_residual_and_data_loss(accumulated_pred, Y_obs)
    
    reg_temporal = _zernike_temporal_regularization_loss(coeffs)
    reg_coeff = _coeff_l2_regularization_loss(coeffs)
    
    total_loss = data_loss + lambda_temporal * reg_temporal + lambda_coeff * reg_coeff
    
    _record_loss_history(total_loss, history)
    
    # Backward pass: gradient w.r.t. shifts first
    grad_shifts = _gaussian_shift_gradient_stack(
        diff=diff,
        all_frame_centers=all_frame_centers,
        sigma=sigma,
        Y_grid=Y_grid,
        X_grid=X_grid,
    )
    
    # Chain rule: gradient w.r.t. coefficients
    grad_coeffs = _zernike_coeff_gradient_from_shift_gradients(
        grad_shifts=grad_shifts,
        basis=basis,
    )
    
    # Regularization gradients
    _apply_zernike_temporal_regularization_grad(
        grad_coeffs,
        coeffs=coeffs,
        lambda_temporal=lambda_temporal,
    )
    _apply_coeff_l2_regularization_grad(
        grad_coeffs,
        coeffs=coeffs,
        lambda_coeff=lambda_coeff,
    )
    
    return total_loss, grad_coeffs.flatten()


# =============================================================================
# Solvers
# =============================================================================

def solve_direct(
    Y_obs: np.ndarray,
    ideal_centers: np.ndarray,
    n_frames: int,
    global_offsets_px: np.ndarray,
    sigma: float,
    spacing: float,
    template_bank: Optional[Dict[str, np.ndarray]] = None,
    initial_shifts: Optional[np.ndarray] = None,
    max_shift: float = 5.0,
    lambda_spatial: float = 0.1,
    lambda_temporal: float = 0.1,
    ftol: float = 1e-9,
    gtol: float = 1e-9,
    max_iter: int = 200,
    verbose: bool = True
) -> OptimizationResult:
    """
    Solve for per-spot shifts using direct parameterization.
    """
    shape = Y_obs.shape
    n_spots = len(ideal_centers)
    n_params = n_frames * n_spots * 2
    offsets = coerce_global_offsets_xy(global_offsets_px, n_frames=n_frames, name="global_offsets_px")
    
    Y_grid, X_grid = np.indices(shape, dtype=np.float64)
    neighbors = build_neighbor_graph(ideal_centers, spacing) if lambda_spatial > 0 else None
    _print_direct_solver_banner(
        verbose=verbose,
        n_params=n_params,
        n_frames=n_frames,
        n_spots=n_spots,
        lambda_spatial=lambda_spatial,
        lambda_temporal=lambda_temporal,
        template_bank=template_bank,
    )
    initial_guess, bounds = _build_direct_solver_parameters(
        initial_shifts=initial_shifts,
        n_frames=n_frames,
        n_spots=n_spots,
        max_shift=max_shift,
    )
    objective_fun, objective_args = _build_direct_objective_problem(
        shape=shape,
        ideal_centers=ideal_centers,
        n_frames=n_frames,
        offsets=offsets,
        sigma=sigma,
        Y_obs=Y_obs,
        Y_grid=Y_grid,
        X_grid=X_grid,
        template_bank=template_bank,
        neighbors=neighbors,
        lambda_spatial=lambda_spatial,
        lambda_temporal=lambda_temporal,
    )

    result, history, callback, elapsed = _run_lbfgsb_minimize(
        objective_fun=objective_fun,
        initial_guess=initial_guess,
        objective_args=objective_args,
        bounds=bounds,
        max_iter=max_iter,
        ftol=ftol,
        gtol=gtol,
        verbose=verbose,
        progress_desc="Optimizing",
    )
    
    shifts = result.x.reshape((n_frames, n_spots, 2))
    return _finalize_solver_result(
        result=result,
        history=history,
        callback=callback,
        elapsed=elapsed,
        verbose=verbose,
        shifts=shifts,
        solver_method="direct",
    )


def solve_zernike(
    Y_obs: np.ndarray,
    ideal_centers: np.ndarray,
    n_frames: int,
    global_offsets_px: np.ndarray,
    sigma: float,
    n_modes: int = 12,
    max_coeff: float = 10.0,
    lambda_temporal: float = 0.1,
    lambda_coeff: float = 0.01,
    ftol: float = 1e-9,
    gtol: float = 1e-9,
    max_iter: int = 500,
    verbose: bool = True
) -> OptimizationResult:
    """
    Solve for shifts using Zernike-constrained parameterization.
    """
    shape = Y_obs.shape
    n_spots = len(ideal_centers)
    offsets = coerce_global_offsets_xy(global_offsets_px, n_frames=n_frames, name="global_offsets_px")
    basis, actual_n_modes, n_params = _build_zernike_solver_basis(
        shape=shape,
        ideal_centers=ideal_centers,
        n_frames=n_frames,
        n_spots=n_spots,
        n_modes=n_modes,
    )
    Y_grid, X_grid = np.indices(shape, dtype=np.float64)
    _print_zernike_solver_banner(
        verbose=verbose,
        n_params=n_params,
        n_frames=n_frames,
        n_spots=n_spots,
        actual_n_modes=actual_n_modes,
        lambda_temporal=lambda_temporal,
        lambda_coeff=lambda_coeff,
    )
    initial_guess, bounds = _build_zernike_solver_parameters(
        n_params=n_params,
        max_coeff=max_coeff,
    )
    result, history, callback, elapsed = _run_lbfgsb_minimize(
        objective_fun=objective_zernike_with_grad,
        initial_guess=initial_guess,
        objective_args=(
            shape, ideal_centers, n_frames, offsets, sigma, Y_obs,
            Y_grid, X_grid, basis, lambda_temporal, lambda_coeff,
        ),
        bounds=bounds,
        max_iter=max_iter,
        ftol=ftol,
        gtol=gtol,
        verbose=verbose,
        progress_desc="Optimizing (Zernike)",
    )
    
    shifts = zernike_coeffs_to_shifts(result.x, basis, n_frames, n_spots)
    return _finalize_solver_result(
        result=result,
        history=history,
        callback=callback,
        elapsed=elapsed,
        verbose=verbose,
        shifts=shifts,
        solver_method="zernike",
    )


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_spot_sigma_from_image(image: np.ndarray, spacing: float) -> float:
    """
    Estimate Gaussian sigma from image by analyzing spot FWHM.
    """
    _, sigma_px = estimate_spot_fwhm_sigma_from_image(image, spacing)
    if sigma_px is None:
        return spacing / 8
    return float(sigma_px)


def choose_solver_method(
    n_frames: int,
    n_spots: int,
    spot_sigma: float,
    global_offsets_px: np.ndarray,
    n_zernike_modes: int = 12,
) -> str:
    """
    Automatically choose solver method based on problem characteristics.
    """
    offsets = coerce_global_offsets_xy(global_offsets_px, n_frames=n_frames, name="global_offsets_px")
    direct_params = int(2 * n_frames * n_spots)
    zernike_params = int(2 * n_frames * max(1, int(n_zernike_modes)))
    shift_magnitude = _mean_nonzero_step_magnitude(offsets)
    overlap_ratio = spot_sigma * 2 / shift_magnitude if shift_magnitude > 0 else float('inf')
    return _suggest_solver_method(
        direct_params=direct_params,
        zernike_params=zernike_params,
        overlap_ratio=overlap_ratio,
        total_spot_count=n_frames * n_spots,
    )


# =============================================================================
# Main Interface
# =============================================================================


def _mean_nonzero_step_magnitude(offsets: np.ndarray) -> float:
    """Estimate the characteristic non-zero step magnitude from an offset trajectory."""
    if len(offsets) < 2:
        return 0.0
    step_deltas = np.diff(offsets, axis=0)
    step_magnitudes = np.linalg.norm(step_deltas, axis=1)
    nonzero_steps = step_magnitudes[step_magnitudes > 0]
    return float(np.mean(nonzero_steps)) if len(nonzero_steps) else 0.0


def _suggest_solver_method(
    *,
    direct_params: int,
    zernike_params: int,
    overlap_ratio: float,
    total_spot_count: int,
) -> str:
    """Apply the direct-vs-Zernike heuristic once the problem descriptors are known."""
    if zernike_params >= direct_params:
        return 'direct'
    if overlap_ratio > 0.5 or total_spot_count > 1000:
        return 'zernike'
    return 'direct'


def _apply_template_solver_constraint(
    *,
    method: str,
    spot_model: str,
    verbose: bool,
) -> str:
    """Force template rendering onto the direct solver until template-Zernike is supported."""
    if spot_model == 'template' and method == 'zernike':
        if verbose:
            print("Template forward model is implemented for direct solver only; falling back from zernike to direct.")
        return 'direct'
    return method


def _print_direct_solver_banner(
    *,
    verbose: bool,
    n_params: int,
    n_frames: int,
    n_spots: int,
    lambda_spatial: float,
    lambda_temporal: float,
    template_bank: Optional[Dict[str, np.ndarray]],
) -> None:
    """Report direct-solver problem size and regularization settings."""
    if not verbose:
        return
    print(f"\n=== Direct Solver ===")
    print(f"Parameters: {n_params} (n_frames={n_frames}, n_spots={n_spots})")
    print(f"Regularization: λ_spatial={lambda_spatial}, λ_temporal={lambda_temporal}")
    print(f"Forward model: {'template' if template_bank is not None else 'gaussian'}")


def _build_direct_solver_parameters(
    *,
    initial_shifts: Optional[np.ndarray],
    n_frames: int,
    n_spots: int,
    max_shift: float,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Build the initial parameter vector and bounds for the direct solver."""
    n_params = n_frames * n_spots * 2
    if initial_shifts is not None:
        initial_arr = np.asarray(initial_shifts, dtype=np.float64)
        if initial_arr.shape != (n_frames, n_spots, 2):
            raise ValueError(
                f"initial_shifts must have shape ({n_frames}, {n_spots}, 2), got {initial_arr.shape}"
            )
        initial_guess = initial_arr.reshape(-1)
    else:
        initial_guess = np.zeros(n_params)
    bounds = [(-max_shift, max_shift)] * n_params
    return initial_guess, bounds


def _build_direct_objective_problem(
    *,
    shape: Tuple[int, int],
    ideal_centers: np.ndarray,
    n_frames: int,
    offsets: np.ndarray,
    sigma: float,
    Y_obs: np.ndarray,
    Y_grid: np.ndarray,
    X_grid: np.ndarray,
    template_bank: Optional[Dict[str, np.ndarray]],
    neighbors: Optional[List[List[int]]],
    lambda_spatial: float,
    lambda_temporal: float,
) -> tuple[Callable, tuple]:
    """Resolve the direct objective function and its shared argument tuple."""
    if template_bank is not None:
        return (
            objective_direct_template_with_grad,
            (
                shape, ideal_centers, n_frames, offsets, template_bank, Y_obs,
                neighbors, lambda_spatial, lambda_temporal,
            ),
        )
    return (
        objective_direct_with_grad,
        (
            shape, ideal_centers, n_frames, offsets, sigma, Y_obs,
            Y_grid, X_grid, neighbors, lambda_spatial, lambda_temporal,
        ),
    )


def _build_zernike_solver_basis(
    *,
    shape: Tuple[int, int],
    ideal_centers: np.ndarray,
    n_frames: int,
    n_spots: int,
    n_modes: int,
) -> tuple[np.ndarray, int, int]:
    """Build the Zernike basis and derived parameter counts."""
    basis = compute_zernike_basis(shape, ideal_centers, n_modes)
    actual_n_modes = basis.shape[1]
    n_params = n_frames * 2 * actual_n_modes
    return basis, actual_n_modes, n_params


def _print_zernike_solver_banner(
    *,
    verbose: bool,
    n_params: int,
    n_frames: int,
    n_spots: int,
    actual_n_modes: int,
    lambda_temporal: float,
    lambda_coeff: float,
) -> None:
    """Report Zernike-solver problem size and regularization settings."""
    if not verbose:
        return
    print(f"\n=== Zernike Solver ===")
    print(f"Parameters: {n_params} (n_frames={n_frames}, n_modes={actual_n_modes})")
    print(f"Reduction: {n_frames * n_spots * 2} -> {n_params} ({100*(1 - n_params/(n_frames*n_spots*2)):.1f}%)")
    print(f"Regularization: λ_temporal={lambda_temporal}, λ_coeff={lambda_coeff}")


def _build_zernike_solver_parameters(
    *,
    n_params: int,
    max_coeff: float,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Build the initial parameter vector and bounds for the Zernike solver."""
    initial_guess = np.zeros(n_params)
    bounds = [(-max_coeff, max_coeff)] * n_params
    return initial_guess, bounds


def _estimate_scaled_spot_sigma(
    *,
    ideal_accumulated_image: np.ndarray,
    spacing: float,
    spot_sigma_scale: float,
    verbose: bool,
) -> float:
    """Estimate and report the Gaussian spot sigma used by the optimization model."""
    sigma_estimated = estimate_spot_sigma_from_image(ideal_accumulated_image, spacing)
    sigma = float(sigma_estimated) * float(spot_sigma_scale)
    if verbose:
        if spot_sigma_scale != 1.0:
            print(
                f"Estimated spot sigma: {sigma_estimated:.2f} pixels -> "
                f"scaled to {sigma:.2f} pixels (scale={spot_sigma_scale:.2f})"
            )
        else:
            print(f"Estimated spot sigma: {sigma:.2f} pixels")
    return sigma


def _normalize_observation_image(
    accumulated_image: np.ndarray,
    *,
    verbose: bool,
) -> np.ndarray:
    """Normalize the observed accumulated image to a stable [0, 1] scale."""
    obs = np.asarray(accumulated_image, dtype=np.float64)
    obs_min = float(np.min(obs))
    obs_max = float(np.max(obs))
    if obs_max - obs_min > 0:
        normalized = (obs - obs_min) / (obs_max - obs_min)
    else:
        normalized = obs.copy()
    if verbose:
        print(f"Observation image range: [{obs_min:.2e}, {obs_max:.2e}] -> normalized to [0, 1]")
    return normalized


def _filter_optimization_centers(
    *,
    ideal_centers: np.ndarray,
    shape: Tuple[int, int],
    n_frames: int,
    global_offsets_px: np.ndarray,
    sigma: float,
    initial_shifts: Optional[np.ndarray],
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Filter centers that remain valid across all pulses and align the optional initializer."""
    valid_centers, valid_mask = filter_valid_centers(
        ideal_centers,
        shape,
        n_frames,
        global_offsets_px,
        margin=sigma * 3 + 5,
    )
    if verbose:
        print(f"Boundary filter: {len(valid_centers)} of {len(ideal_centers)} spots are valid")
    if len(valid_centers) < 4:
        raise ValueError(
            f"Too few valid centers ({len(valid_centers)}) after boundary filtering. "
            "Try reducing shift magnitude or increasing image size."
        )

    filtered_initial_shifts = initial_shifts
    if initial_shifts is not None:
        init_arr = np.asarray(initial_shifts, dtype=np.float64)
        if init_arr.ndim != 3 or init_arr.shape[0] != n_frames or init_arr.shape[2] != 2:
            raise ValueError(
                f"initial_shifts must have shape (n_frames, n_spots, 2), got {init_arr.shape}"
            )
        if init_arr.shape[1] == len(valid_mask):
            filtered_initial_shifts = init_arr[:, valid_mask, :]
        elif init_arr.shape[1] == len(valid_centers):
            filtered_initial_shifts = init_arr
        else:
            raise ValueError(
                f"initial_shifts second dimension {init_arr.shape[1]} does not match "
                f"original spots {len(valid_mask)} or valid spots {len(valid_centers)}"
            )
    return valid_centers, valid_mask, filtered_initial_shifts


def _prepare_template_bank(
    *,
    spot_model: str,
    pulse_ideal_images: List[np.ndarray],
    ideal_accumulated_image: np.ndarray,
    n_frames: int,
    ideal_centers: np.ndarray,
    global_offsets_px: np.ndarray,
    spacing: float,
    sigma: float,
    verbose: bool,
) -> tuple[str, Optional[Dict[str, np.ndarray]]]:
    """Build the template bank when template rendering is requested and available."""
    template_bank = None
    resolved_spot_model = str(spot_model or 'gaussian').lower()
    if resolved_spot_model == 'template':
        if pulse_ideal_images and len(pulse_ideal_images) == n_frames:
            ref_scale = max(float(np.max(np.asarray(ideal_accumulated_image, dtype=np.float64))), 1e-12)
            pulse_templates_norm = [
                np.asarray(img, dtype=np.float64) / ref_scale
                for img in pulse_ideal_images
            ]
            template_bank = build_template_bank(
                pulse_templates_norm,
                ideal_centers,
                global_offsets_px,
                spacing=spacing,
                sigma=sigma,
            )
            if verbose:
                print(
                    f"Template bank prepared: n_frames={n_frames}, n_spots={len(ideal_centers)}, "
                    f"radius={template_bank['radius']}"
                )
        else:
            if verbose:
                print("Template spot model requested, but no valid pulse reference sequence was provided; falling back to gaussian.")
            resolved_spot_model = 'gaussian'
    return resolved_spot_model, template_bank


def _resolve_solver_method(
    *,
    requested_method: str,
    n_frames: int,
    n_spots: int,
    sigma: float,
    global_offsets_px: np.ndarray,
    n_zernike_modes: int,
    spot_model: str,
    verbose: bool,
) -> str:
    """Resolve auto-selection and template-model constraints for the solver method."""
    method = str(requested_method)
    if method == 'auto':
        method = choose_solver_method(
            n_frames,
            n_spots,
            sigma,
            global_offsets_px,
            n_zernike_modes=n_zernike_modes,
        )
        if verbose:
            print(f"Auto-selected solver: {method}")
    return _apply_template_solver_constraint(
        method=method,
        spot_model=spot_model,
        verbose=verbose,
    )


def _build_zernike_solver_kwargs(
    *,
    Y_obs_normalized: np.ndarray,
    ideal_centers: np.ndarray,
    n_frames: int,
    global_offsets_px: np.ndarray,
    sigma: float,
    n_zernike_modes: int,
    max_shift: float,
    lambda_temporal: float,
    ftol: float,
    gtol: float,
    verbose: bool,
) -> Dict[str, Any]:
    """Assemble keyword arguments for the Zernike solver path."""
    return {
        "Y_obs": Y_obs_normalized,
        "ideal_centers": ideal_centers,
        "n_frames": n_frames,
        "global_offsets_px": global_offsets_px,
        "sigma": sigma,
        "n_modes": n_zernike_modes,
        "max_coeff": max_shift,
        "lambda_temporal": lambda_temporal,
        "lambda_coeff": 0.01,
        "ftol": ftol,
        "gtol": gtol,
        "verbose": verbose,
    }


def _build_direct_solver_kwargs(
    *,
    Y_obs_normalized: np.ndarray,
    ideal_centers: np.ndarray,
    n_frames: int,
    global_offsets_px: np.ndarray,
    sigma: float,
    spacing: float,
    template_bank: Optional[Dict[str, np.ndarray]],
    initial_shifts: Optional[np.ndarray],
    max_shift: float,
    lambda_spatial: float,
    lambda_temporal: float,
    ftol: float,
    gtol: float,
    verbose: bool,
) -> Dict[str, Any]:
    """Assemble keyword arguments for the direct solver path."""
    return {
        "Y_obs": Y_obs_normalized,
        "ideal_centers": ideal_centers,
        "n_frames": n_frames,
        "global_offsets_px": global_offsets_px,
        "sigma": sigma,
        "spacing": spacing,
        "template_bank": template_bank,
        "initial_shifts": initial_shifts,
        "max_shift": max_shift,
        "lambda_spatial": lambda_spatial,
        "lambda_temporal": lambda_temporal,
        "ftol": ftol,
        "gtol": gtol,
        "verbose": verbose,
    }


def _run_selected_solver(
    *,
    method: str,
    Y_obs_normalized: np.ndarray,
    ideal_centers: np.ndarray,
    n_frames: int,
    global_offsets_px: np.ndarray,
    sigma: float,
    spacing: float,
    template_bank: Optional[Dict[str, np.ndarray]],
    initial_shifts: Optional[np.ndarray],
    max_shift: float,
    lambda_spatial: float,
    lambda_temporal: float,
    n_zernike_modes: int,
    ftol: float,
    gtol: float,
    verbose: bool,
) -> OptimizationResult:
    """Dispatch to the selected optimization solver implementation."""
    if method == 'zernike':
        return solve_zernike(**_build_zernike_solver_kwargs(
            Y_obs_normalized=Y_obs_normalized,
            ideal_centers=ideal_centers,
            n_frames=n_frames,
            global_offsets_px=global_offsets_px,
            sigma=sigma,
            n_zernike_modes=n_zernike_modes,
            max_shift=max_shift,
            lambda_temporal=lambda_temporal,
            ftol=ftol,
            gtol=gtol,
            verbose=verbose,
        ))
    return solve_direct(**_build_direct_solver_kwargs(
        Y_obs_normalized=Y_obs_normalized,
        ideal_centers=ideal_centers,
        n_frames=n_frames,
        global_offsets_px=global_offsets_px,
        sigma=sigma,
        spacing=spacing,
        template_bank=template_bank,
        initial_shifts=initial_shifts,
        max_shift=max_shift,
        lambda_spatial=lambda_spatial,
        lambda_temporal=lambda_temporal,
        ftol=ftol,
        gtol=gtol,
        verbose=verbose,
    ))


def _finalize_optimization_result(
    *,
    result: OptimizationResult,
    pixel_size_m: float,
    valid_centers: np.ndarray,
    valid_mask: np.ndarray,
) -> OptimizationResult:
    """Attach physical-unit displacement products and valid-center metadata to the result."""
    disp_x_m = result.shifts[:, :, 0] * pixel_size_m
    disp_y_m = result.shifts[:, :, 1] * pixel_size_m
    result.per_frame_displacement_x_m = disp_x_m
    result.per_frame_displacement_y_m = disp_y_m
    result.valid_centers = valid_centers
    result.valid_center_indices = np.nonzero(valid_mask)[0].astype(np.int64)
    return result

def run_optimization_recovery(
    accumulated_image: np.ndarray,
    ideal_accumulated_image: np.ndarray,
    pulse_ideal_images: List[np.ndarray],
    ideal_centers: np.ndarray,
    n_frames: int,
    global_offsets_px: np.ndarray,
    spacing: float,
    pixel_size_m: float,
    z_prop_m: float,
    wavelength_m: float,
    method: str = 'auto',
    max_shift: float = 10.0,
    lambda_spatial: float = 0.1,
    lambda_temporal: float = 0.1,
    n_zernike_modes: int = 12,
    spot_sigma_scale: float = 1.0,
    spot_model: str = 'gaussian',
    initial_shifts: Optional[np.ndarray] = None,
    ftol: float = 1e-9,
    gtol: float = 1e-9,
    verbose: bool = True
) -> OptimizationResult:
    """
    Main interface: recover per-frame shifts from accumulated image.
    
    Parameters
    ----------
    accumulated_image : np.ndarray
        The observed accumulated image (sum of all frames with aberrations)
    ideal_accumulated_image : np.ndarray
        The ideal accumulated image (sum of all frames without aberrations)
    pulse_ideal_images : List[np.ndarray]
        List of ideal images for each pulse
    ideal_centers : np.ndarray
        Shape (n_spots, 2), ideal spot center positions (x, y) in pixels
    n_frames : int
        Number of frames accumulated
    global_offsets_px : np.ndarray
        Per-pulse absolute global offsets, shape (n_frames, 2), as (dx, dy) in pixels
    spacing : float
        Grid spacing in pixels
    pixel_size_m : float
        Detector pixel size in meters
    z_prop_m : float
        Propagation distance in meters
    wavelength_m : float
        Wavelength in meters
    method : str
        'auto', 'direct', or 'zernike'
    max_shift : float
        Maximum shift constraint in pixels
    lambda_spatial : float
        Spatial smoothness regularization weight
    lambda_temporal : float
        Temporal smoothness regularization weight
    n_zernike_modes : int
        Number of Zernike modes for zernike solver
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    OptimizationResult
        Contains recovered shifts and metrics
    """
    shape = accumulated_image.shape
    global_offsets_px = coerce_global_offsets_xy(global_offsets_px, n_frames=n_frames, name="global_offsets_px")
    spot_model = str(spot_model or 'gaussian').lower()

    sigma = _estimate_scaled_spot_sigma(
        ideal_accumulated_image=ideal_accumulated_image,
        spacing=spacing,
        spot_sigma_scale=spot_sigma_scale,
        verbose=verbose,
    )
    Y_obs_normalized = _normalize_observation_image(
        accumulated_image,
        verbose=verbose,
    )
    valid_centers, valid_mask, initial_shifts = _filter_optimization_centers(
        ideal_centers=ideal_centers,
        shape=shape,
        n_frames=n_frames,
        global_offsets_px=global_offsets_px,
        sigma=sigma,
        initial_shifts=initial_shifts,
        verbose=verbose,
    )
    n_spots = len(valid_centers)
    spot_model, template_bank = _prepare_template_bank(
        spot_model=spot_model,
        pulse_ideal_images=pulse_ideal_images,
        ideal_accumulated_image=ideal_accumulated_image,
        n_frames=n_frames,
        ideal_centers=valid_centers,
        global_offsets_px=global_offsets_px,
        spacing=spacing,
        sigma=sigma,
        verbose=verbose,
    )
    method = _resolve_solver_method(
        requested_method=method,
        n_frames=n_frames,
        n_spots=n_spots,
        sigma=sigma,
        global_offsets_px=global_offsets_px,
        n_zernike_modes=n_zernike_modes,
        spot_model=spot_model,
        verbose=verbose,
    )
    result = _run_selected_solver(
        method=method,
        Y_obs_normalized=Y_obs_normalized,
        ideal_centers=valid_centers,
        n_frames=n_frames,
        global_offsets_px=global_offsets_px,
        sigma=sigma,
        spacing=spacing,
        template_bank=template_bank,
        initial_shifts=initial_shifts,
        max_shift=max_shift,
        lambda_spatial=lambda_spatial,
        lambda_temporal=lambda_temporal,
        n_zernike_modes=n_zernike_modes,
        ftol=ftol,
        gtol=gtol,
        verbose=verbose,
    )
    return _finalize_optimization_result(
        result=result,
        pixel_size_m=pixel_size_m,
        valid_centers=valid_centers,
        valid_mask=valid_mask,
    )


def reshape_shifts_to_grid(
    shifts: np.ndarray,
    ideal_centers: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    spot_grid_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reshape flat shifts array to grid format.
    
    Parameters
    ----------
    shifts : np.ndarray
        Shape (n_frames, n_spots, 2)
    ideal_centers : np.ndarray
        Shape (n_spots, 2)
    grid_rows, grid_cols : int
        Grid dimensions
    spot_grid_indices : np.ndarray or None
        Optional `(row, col)` index array aligned with `ideal_centers`.
        
    Returns
    -------
    grid_shifts : np.ndarray
        Shape (n_frames, grid_rows, grid_cols, 2)
    """
    n_frames = shifts.shape[0]
    n_spots = shifts.shape[1]
    ideal_centers = assert_xy_points(ideal_centers, name="ideal_centers")
    if ideal_centers.shape[0] != n_spots:
        raise ValueError(
            f"ideal_centers length {ideal_centers.shape[0]} does not match shift spots {n_spots}"
        )

    if spot_grid_indices is not None:
        grid_indices = assert_row_col_indices(spot_grid_indices, name="spot_grid_indices")
        if grid_indices.shape[0] != n_spots:
            raise ValueError(
                f"spot_grid_indices length {grid_indices.shape[0]} does not match shift spots {n_spots}"
            )
        actual_rows = int(grid_rows) if int(grid_rows) > 0 else int(np.max(grid_indices[:, 0]) + 1)
        actual_cols = int(grid_cols) if int(grid_cols) > 0 else int(np.max(grid_indices[:, 1]) + 1)
        grid_shifts = np.zeros((n_frames, actual_rows, actual_cols, 2))
        for idx, (row, col) in enumerate(grid_indices):
            if 0 <= row < actual_rows and 0 <= col < actual_cols:
                grid_shifts[:, row, col, :] = shifts[:, idx, :]
        return grid_shifts

    unique_x = np.unique(ideal_centers[:, 0])
    unique_y = np.unique(ideal_centers[:, 1])
    y_to_row = {y: i for i, y in enumerate(unique_y)}
    x_to_col = {x: j for j, x in enumerate(unique_x)}
    actual_rows = len(unique_y)
    actual_cols = len(unique_x)
    grid_shifts = np.zeros((n_frames, actual_rows, actual_cols, 2))

    for idx, (cx, cy) in enumerate(ideal_centers):
        row = y_to_row.get(cy)
        col = x_to_col.get(cx)
        if row is not None and col is not None:
            grid_shifts[:, row, col, :] = shifts[:, idx, :]

    return grid_shifts
