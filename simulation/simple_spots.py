from __future__ import annotations
"""
Simple Gaussian Spot Simulation Module (from Original version)

This module provides simplified spot generation using Gaussian profiles,
useful for testing optimization methods under controlled conditions.
It serves as a baseline/reference implementation compared to the full
wave-optics simulation in hartmann_simulator.py.

Key functions:
- generate_wavefront_distortion: Zernike-based displacement field
- create_hartmann_grid_centers: Regular grid spot centers
- fast_render_gaussian_spots: Fast Gaussian spot rendering
- generate_ground_truth_data: Full ground truth simulation
- run_shift_recovery: End-to-end shift recovery pipeline (simplified)

Origin: Original version (opt_method_phiRecon.py)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from plot_style import apply_plot_font_bump

apply_plot_font_bump()

from analysis.optimization_common import (
    assert_xy_points,
    build_neighbor_graph,
    gaussian_shift_gradients_xy,
    render_gaussian_sequence_xy,
    render_gaussian_spots_xy,
)
from analysis.zernike_basis import (
    ZERNIKE_INDICES,
    zernike_basis_func,
    compute_zernike_basis_matrix,
    zernike_coeffs_to_shifts,
)


# =============================================================================
# 1. Data Generation
# =============================================================================

def generate_wavefront_distortion_xy(shape, centers, n_zernike_modes=15,
                                     rms_amplitude=1.0, seed=None):
    """
    Generate spatially correlated displacement field simulating wavefront-induced distortion.
    Uses Zernike-like polynomial basis for realistic low-frequency structure.

    Parameters
    ----------
    shape : tuple
        Image shape (H, W)
    centers : array
        Sub-beam center positions [(x, y), ...]
    n_zernike_modes : int
        Number of Zernike-like modes to use
    rms_amplitude : float
        RMS amplitude of the displacement field
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    displacement_x, displacement_y : arrays of shape (n_centers,)
        Returned in canonical `(dx_px, dy_px)` order. `centers` are interpreted
        as `(x_px, y_px)` detector coordinates.
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = shape
    centers = assert_xy_points(centers, name="centers")
    n_centers = len(centers)

    cx_norm = 2.0 * (centers[:, 0] - w / 2) / min(h, w)
    cy_norm = 2.0 * (centers[:, 1] - h / 2) / min(h, w)

    rho = np.sqrt(cx_norm**2 + cy_norm**2)
    theta = np.arctan2(cy_norm, cx_norm)

    n_modes_to_use = min(n_zernike_modes, len(ZERNIKE_INDICES))

    coeff_y = np.zeros(n_modes_to_use)
    coeff_x = np.zeros(n_modes_to_use)

    for i in range(1, n_modes_to_use):
        n, m = ZERNIKE_INDICES[i]
        weight = 1.0 / (n + 1)
        coeff_y[i] = np.random.randn() * weight
        coeff_x[i] = np.random.randn() * weight

    local_shift_y_px = np.zeros(n_centers)
    local_shift_x_px = np.zeros(n_centers)

    for i in range(1, n_modes_to_use):
        n, m = ZERNIKE_INDICES[i]
        basis = zernike_basis_func(n, m, rho, theta)
        local_shift_y_px += coeff_y[i] * basis
        local_shift_x_px += coeff_x[i] * basis

    current_rms_y = np.sqrt(np.mean(local_shift_y_px**2))
    current_rms_x = np.sqrt(np.mean(local_shift_x_px**2))

    if current_rms_y > 1e-10:
        local_shift_y_px = local_shift_y_px / current_rms_y * rms_amplitude
    if current_rms_x > 1e-10:
        local_shift_x_px = local_shift_x_px / current_rms_x * rms_amplitude

    return local_shift_x_px, local_shift_y_px


def generate_wavefront_distortion(shape, centers, n_zernike_modes=15,
                                  rms_amplitude=1.0, seed=None):
    """Backward-compatible wrapper returning legacy `(dy_px, dx_px)` order."""
    local_shift_x_px, local_shift_y_px = generate_wavefront_distortion_xy(
        shape,
        centers,
        n_zernike_modes=n_zernike_modes,
        rms_amplitude=rms_amplitude,
        seed=seed,
    )
    return local_shift_y_px, local_shift_x_px


def compute_shift_per_frame(shift_magnitude, shift_angle_deg):
    """Compute the (dx, dy) shift per frame from magnitude and angle."""
    angle_rad = np.deg2rad(shift_angle_deg)
    dx_per_frame = shift_magnitude * np.cos(angle_rad)
    dy_per_frame = shift_magnitude * np.sin(angle_rad)
    return (dx_per_frame, dy_per_frame)


def create_hartmann_grid_centers(shape, spacing=20):
    """Create a regular grid of spot centers in `(x, y)` order."""
    h, w = shape
    y_coords = np.arange(spacing // 2, h, spacing)
    x_coords = np.arange(spacing // 2, w, spacing)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    centers = []
    for y, x in zip(yy.flat, xx.flat):
        if 0 <= y < h and 0 <= x < w:
            centers.append([x, y])
    return np.array(centers, dtype=np.float64)


def filter_valid_centers(centers, shape, n_frames, shift_per_frame, margin=5.0):
    """
    Filter centers to only include those that remain inside the image
    for all frames, with a safety margin.
    """
    centers = assert_xy_points(centers, name="centers")
    h, w = shape
    dx_pf, dy_pf = shift_per_frame  # (dx, dy) convention
    frame_indices = np.arange(int(n_frames), dtype=np.float64)
    global_dx_px = frame_indices * float(dx_pf)
    global_dy_px = frame_indices * float(dy_pf)
    min_global_dx_px = float(np.min(global_dx_px)) if global_dx_px.size else 0.0
    max_global_dx_px = float(np.max(global_dx_px)) if global_dx_px.size else 0.0
    min_global_dy_px = float(np.min(global_dy_px)) if global_dy_px.size else 0.0
    max_global_dy_px = float(np.max(global_dy_px)) if global_dy_px.size else 0.0

    valid_mask = (
        (centers[:, 1] + min_global_dy_px >= margin) &
        (centers[:, 1] + max_global_dy_px < h - margin) &
        (centers[:, 0] + min_global_dx_px >= margin) &
        (centers[:, 0] + max_global_dx_px < w - margin)
    )
    return centers[valid_mask], valid_mask


def fast_render_gaussian_spots(shape, centers, spot_size, Y_grid, X_grid):
    """
    Optimized Gaussian spot renderer using ROI-based rendering.
    `centers` must be in `(x, y)` order.
    """
    return render_gaussian_spots_xy(
        shape=shape,
        centers_xy=centers,
        sigma=float(spot_size) / 2.0,
        y_grid=Y_grid,
        x_grid=X_grid,
    )


def generate_ground_truth_data(shape, centers, n_frames, shift_per_frame, spot_size,
                               max_local_shift=None, n_zernike_modes=12,
                               wavefront_rms=1.5):
    """
    Generate ground truth data with spatially correlated local shifts.

    Parameters
    ----------
    shape : tuple
        Image shape (H, W)
    centers : array
        Sub-beam center positions as (x, y)
    n_frames : int
        Number of frames
    shift_per_frame : tuple
        Global (dx, dy) shift per frame
    spot_size : float
        Gaussian spot size
    max_local_shift : float or None
        Maximum local shift magnitude (pixels). None = no clipping.
    n_zernike_modes : int
        Number of Zernike modes for wavefront distortion
    wavefront_rms : float
        RMS amplitude of displacement (pixels)

    Returns
    -------
    ground_truth_frames, accumulated_image, true_local_shifts
    """
    centers = assert_xy_points(centers, name="centers")
    ground_truth_frames = []
    accumulated_image = np.zeros(shape)
    true_local_shifts = []
    Y_grid, X_grid = np.indices(shape)

    dx_pf, dy_pf = shift_per_frame  # (dx, dy) convention

    for t in range(n_frames):
        local_shift_x_px, local_shift_y_px = generate_wavefront_distortion_xy(
            shape, centers,
            n_zernike_modes=n_zernike_modes,
            rms_amplitude=wavefront_rms,
            seed=t * 1000 + 42,
        )

        frame_centers = []
        frame_local_shifts = []
        global_dy_px = t * dy_pf
        global_dx_px = t * dx_pf

        for i, (cx, cy) in enumerate(centers):
            dx, dy = local_shift_x_px[i], local_shift_y_px[i]
            if max_local_shift is not None:
                magnitude = np.sqrt(dx**2 + dy**2)
                if magnitude > 1e-10:
                    scale_factor = min(1.0, max_local_shift / magnitude)
                    dx *= scale_factor
                    dy *= scale_factor
            frame_local_shifts.append([dx, dy])  # stored as (dx, dy)
            frame_centers.append([cx + global_dx_px + dx, cy + global_dy_px + dy])  # stored as (x, y)

        frame_image = fast_render_gaussian_spots(shape, frame_centers, spot_size, Y_grid, X_grid)
        ground_truth_frames.append(frame_image)
        accumulated_image += frame_image
        true_local_shifts.append(frame_local_shifts)

    return np.array(ground_truth_frames), accumulated_image, np.array(true_local_shifts)


# =============================================================================
# 2. Objective Functions with Analytical Gradients
# =============================================================================

def objective_function_with_grad(params, shape, ideal_centers, n_frames,
                                 shift_per_frame, spot_size, Y_obs,
                                 Y_grid, X_grid, neighbors=None,
                                 lambda_spatial=0.0, lambda_temporal=0.0):
    """
    Returns both the Loss (Scalar) and the Jacobian (Gradient Vector).

    Regularization:
    - lambda_spatial: penalizes shift differences between neighboring spots
    - lambda_temporal: penalizes shift differences between consecutive frames
    """
    n_spots = len(ideal_centers)
    local_shifts = params.reshape((n_frames, n_spots, 2))

    dx_pf, dy_pf = shift_per_frame  # (dx, dy) convention
    frame_indices = np.arange(int(n_frames), dtype=np.float64)
    global_offsets_xy = np.column_stack([frame_indices * dx_pf, frame_indices * dy_pf])
    accumulated_pred, all_frame_centers = render_gaussian_sequence_xy(
        shape=shape,
        ideal_centers_xy=ideal_centers,
        global_offsets_xy=global_offsets_xy,
        local_shifts_xy=local_shifts,
        sigma=spot_size / 2.0,
        y_grid=Y_grid,
        x_grid=X_grid,
    )

    diff = accumulated_pred - Y_obs
    data_loss = np.sum(diff**2)

    reg_loss_spatial = 0.0
    reg_loss_temporal = 0.0

    if lambda_spatial > 0 and neighbors is not None:
        for t in range(n_frames):
            for i in range(n_spots):
                for j in neighbors[i]:
                    if j > i:
                        diff_shift = local_shifts[t, i] - local_shifts[t, j]
                        reg_loss_spatial += np.sum(diff_shift**2)

    if lambda_temporal > 0:
        for t in range(n_frames - 1):
            diff_shift = local_shifts[t + 1] - local_shifts[t]
            reg_loss_temporal += np.sum(diff_shift**2)

    total_loss = data_loss + lambda_spatial * reg_loss_spatial + lambda_temporal * reg_loss_temporal

    # Backward pass: analytical gradient
    grads = np.zeros((n_frames, n_spots, 2))
    sigma = spot_size / 2.0

    for t in range(n_frames):
        grads[t] = gaussian_shift_gradients_xy(
            diff_image=diff,
            centers_xy=all_frame_centers[t],
            sigma=sigma,
            y_grid=Y_grid,
            x_grid=X_grid,
        )

    if lambda_spatial > 0 and neighbors is not None:
        for t in range(n_frames):
            for i in range(n_spots):
                for j in neighbors[i]:
                    diff_shift = local_shifts[t, i] - local_shifts[t, j]
                    grads[t, i] += 2 * lambda_spatial * diff_shift

    if lambda_temporal > 0:
        for t in range(n_frames):
            if t > 0:
                grads[t] += 2 * lambda_temporal * (local_shifts[t] - local_shifts[t - 1])
            if t < n_frames - 1:
                grads[t] -= 2 * lambda_temporal * (local_shifts[t + 1] - local_shifts[t])

    return total_loss, grads.flatten()


# =============================================================================
# 4. Direct Solver
# =============================================================================

def solve_subbeam_positions_fast(Y_obs, shape, ideal_centers, n_frames,
                                 shift_per_frame, spot_size, max_shift=4.0,
                                 spacing=30, lambda_spatial=0.0,
                                 lambda_temporal=0.0):
    """Solve for per-frame spot shifts using direct L-BFGS-B optimization."""
    n_spots = len(ideal_centers)
    n_params = n_frames * n_spots * 2

    initial_guess = np.zeros(n_params)
    bounds = [(-max_shift, max_shift)] * n_params
    Y_grid, X_grid = np.indices(shape)

    neighbors = build_neighbor_graph(ideal_centers, spacing) if lambda_spatial > 0 else None

    print(f"Starting Optimization... Params: {n_params}, Max Shift: {max_shift}")
    start_time = time.time()

    result = minimize(
        fun=objective_function_with_grad,
        x0=initial_guess,
        args=(shape, ideal_centers, n_frames, shift_per_frame, spot_size,
              Y_obs, Y_grid, X_grid, neighbors, lambda_spatial, lambda_temporal),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'disp': True, 'maxiter': 200, 'ftol': 1e-9}
    )

    print(f"Optimization finished in {time.time() - start_time:.2f} seconds.")
    return result.x.reshape((n_frames, n_spots, 2)), result.fun


# =============================================================================
# 5. Zernike-Constrained Solver
# =============================================================================

def objective_zernike_with_grad(params, shape, ideal_centers, n_frames,
                                shift_per_frame, spot_size, Y_obs,
                                Y_grid, X_grid, basis_matrix,
                                lambda_temporal=0.0, lambda_coeff=0.0):
    """Objective function for Zernike-constrained optimization."""
    n_modes = basis_matrix.shape[1]
    n_spots = len(ideal_centers)
    coeffs = params.reshape((n_frames, 2, n_modes))
    local_shifts = zernike_coeffs_to_shifts(params, basis_matrix, n_frames, n_spots)

    dx_pf, dy_pf = shift_per_frame  # (dx, dy) convention
    frame_indices = np.arange(int(n_frames), dtype=np.float64)
    global_offsets_xy = np.column_stack([frame_indices * dx_pf, frame_indices * dy_pf])
    accumulated_pred, all_frame_centers = render_gaussian_sequence_xy(
        shape=shape,
        ideal_centers_xy=ideal_centers,
        global_offsets_xy=global_offsets_xy,
        local_shifts_xy=local_shifts,
        sigma=spot_size / 2.0,
        y_grid=Y_grid,
        x_grid=X_grid,
    )

    diff = accumulated_pred - Y_obs
    data_loss = np.sum(diff**2)

    reg_temporal = 0.0
    if lambda_temporal > 0:
        for t in range(n_frames - 1):
            reg_temporal += np.sum((coeffs[t + 1] - coeffs[t])**2)

    reg_coeff = np.sum(params**2) if lambda_coeff > 0 else 0.0
    total_loss = data_loss + lambda_temporal * reg_temporal + lambda_coeff * reg_coeff

    sigma = spot_size / 2.0

    grad_shifts = np.zeros((n_frames, n_spots, 2))
    for t in range(n_frames):
        grad_shifts[t] = gaussian_shift_gradients_xy(
            diff_image=diff,
            centers_xy=all_frame_centers[t],
            sigma=sigma,
            y_grid=Y_grid,
            x_grid=X_grid,
        )

    grad_coeffs = np.zeros((n_frames, 2, n_modes))
    for t in range(n_frames):
        grad_coeffs[t, 0, :] = basis_matrix.T @ grad_shifts[t, :, 0]
        grad_coeffs[t, 1, :] = basis_matrix.T @ grad_shifts[t, :, 1]

    if lambda_temporal > 0:
        for t in range(n_frames):
            if t > 0:
                grad_coeffs[t] += 2 * lambda_temporal * (coeffs[t] - coeffs[t - 1])
            if t < n_frames - 1:
                grad_coeffs[t] -= 2 * lambda_temporal * (coeffs[t + 1] - coeffs[t])

    if lambda_coeff > 0:
        grad_coeffs += 2 * lambda_coeff * coeffs

    return total_loss, grad_coeffs.flatten()


def solve_zernike_constrained(Y_obs, shape, ideal_centers, n_frames,
                              shift_per_frame, spot_size, n_modes=12,
                              max_coeff=10.0, lambda_temporal=0.1,
                              lambda_coeff=0.01):
    """Solve using Zernike-constrained parameterization (fewer parameters)."""
    n_spots = len(ideal_centers)
    basis_matrix = compute_zernike_basis_matrix(shape, ideal_centers, n_modes)
    actual_n_modes = basis_matrix.shape[1]
    n_params = n_frames * 2 * actual_n_modes

    print(f"\n=== Zernike-Constrained Solver ===")
    print(f"Parameters: {n_params} (vs {n_frames * n_spots * 2} for direct)")

    initial_guess = np.zeros(n_params)
    bounds = [(-max_coeff, max_coeff)] * n_params
    Y_grid, X_grid = np.indices(shape)

    start_time = time.time()
    result = minimize(
        fun=objective_zernike_with_grad,
        x0=initial_guess,
        args=(shape, ideal_centers, n_frames, shift_per_frame, spot_size,
              Y_obs, Y_grid, X_grid, basis_matrix, lambda_temporal, lambda_coeff),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'disp': True, 'maxiter': 500, 'ftol': 1e-9}
    )
    print(f"Zernike optimization finished in {time.time() - start_time:.2f} seconds.")

    optimized_shifts = zernike_coeffs_to_shifts(result.x, basis_matrix, n_frames, n_spots)
    return optimized_shifts, result.fun


def choose_solver_method(shift_magnitude, spot_size, spacing):
    """Auto-choose solver: 'direct' vs 'zernike' based on overlap."""
    overlap_ratio = spot_size / shift_magnitude if shift_magnitude > 0 else float('inf')
    return 'zernike' if overlap_ratio > 0.5 else 'direct'


# =============================================================================
# 6. End-to-end Pipeline
# =============================================================================

def run_shift_recovery(
    image_shape=(1024, 1024),
    n_frames=10,
    spacing=40,
    spot_size=5,
    image_margin=20,
    shift_magnitude=3,
    shift_angle_deg=20.0,
    max_shift_constraint=5.0,
    solver_method='auto',
    zernike_solver_n_modes=12,
    zernike_lambda_temporal=0.1,
    zernike_lambda_coeff=0.01,
    gt_max_local_shift=3.0,
    n_zernike_modes=12,
    wavefront_rms=1.5,
    noise_level=0.05,
    accumulated_image=None,
    show_plots=False,
    verbose=True,
):
    """
    Run Hartmann shift recovery on simulated or real accumulated image.

    This is the Original method's end-to-end pipeline using simplified
    Gaussian spots. Useful for testing optimization approaches.
    """
    is_simulation = accumulated_image is None

    if is_simulation:
        H, W = image_shape
    else:
        H, W = accumulated_image.shape
        image_shape = (H, W)

    shift_step = compute_shift_per_frame(shift_magnitude, shift_angle_deg)
    if verbose:
        print(f"Shift per frame: dx={shift_step[0]:.2f}, dy={shift_step[1]:.2f} px")

    all_centers = create_hartmann_grid_centers((H, W), spacing=spacing)
    ideal_centers, valid_mask = filter_valid_centers(
        all_centers, (H, W), n_frames, shift_step, margin=image_margin + spot_size
    )
    if verbose:
        print(f"Using {len(ideal_centers)} of {len(all_centers)} centers")

    if is_simulation:
        gt_frames, y_obs, gt_shifts = generate_ground_truth_data(
            (H, W), ideal_centers, n_frames, shift_step, spot_size,
            max_local_shift=gt_max_local_shift,
            n_zernike_modes=n_zernike_modes,
            wavefront_rms=wavefront_rms,
        )
        y_obs_input = y_obs + noise_level * np.random.normal(size=y_obs.shape)
    else:
        y_obs_input = accumulated_image.astype(np.float64)
        gt_frames, gt_shifts = None, None
        y_obs = accumulated_image

    if solver_method == 'auto':
        actual_solver = choose_solver_method(shift_magnitude, spot_size, spacing)
        if verbose:
            print(f"Auto-selected solver: '{actual_solver}'")
    else:
        actual_solver = solver_method

    if actual_solver == 'zernike':
        rec_shifts, final_loss = solve_zernike_constrained(
            y_obs_input, (H, W), ideal_centers, n_frames, shift_step, spot_size,
            n_modes=zernike_solver_n_modes,
            max_coeff=max_shift_constraint,
            lambda_temporal=zernike_lambda_temporal,
            lambda_coeff=zernike_lambda_coeff,
        )
    else:
        rec_shifts, final_loss = solve_subbeam_positions_fast(
            y_obs_input, (H, W), ideal_centers, n_frames, shift_step, spot_size,
            max_shift=max_shift_constraint,
        )

    # Generate recovered image
    Y_grid, X_grid = np.indices((H, W))
    y_recovered = np.zeros((H, W))
    dx_ss, dy_ss = shift_step  # (dx, dy) convention
    frame_indices = np.arange(int(n_frames), dtype=np.float64)
    global_offsets_xy = np.column_stack([frame_indices * dx_ss, frame_indices * dy_ss])
    y_recovered, _ = render_gaussian_sequence_xy(
        shape=(H, W),
        ideal_centers_xy=ideal_centers,
        global_offsets_xy=global_offsets_xy,
        local_shifts_xy=rec_shifts,
        sigma=spot_size / 2.0,
        y_grid=Y_grid,
        x_grid=X_grid,
    )

    errors = None
    overlap_ratio = spot_size / shift_magnitude if shift_magnitude > 0 else float('inf')

    if is_simulation:
        errors = np.sqrt(np.sum((gt_shifts - rec_shifts)**2, axis=2)).flatten()
        if verbose:
            print(f"\n=== Shift Estimation Results ===")
            print(f"Solver method: {actual_solver}")
            print(f"Overlap ratio: {overlap_ratio:.2f}")
            print(f"Mean error: {np.mean(errors):.4f} px")
            print(f"Max error:  {np.max(errors):.4f} px")

    return {
        'recovered_shifts': rec_shifts,
        'ideal_centers': ideal_centers,
        'accumulated_image': y_obs_input,
        'recovered_image': y_recovered,
        'solver_method': actual_solver,
        'final_loss': final_loss,
        'shift_step': shift_step,
        'gt_shifts': gt_shifts,
        'gt_frames': gt_frames,
        'errors': errors,
        'overlap_ratio': overlap_ratio,
        'n_frames': n_frames,
        'spot_size': spot_size,
        'spacing': spacing,
        'is_simulation': is_simulation,
    }
