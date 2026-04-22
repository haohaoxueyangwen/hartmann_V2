from __future__ import annotations
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Union, Literal, Dict, Any
import logging
import json
from dataclasses import dataclass
from typing import Literal

# Import optimized FFT functions
try:
    from simulation.optimizations import (
        fft2 as opt_fft2,
        ifft2 as opt_ifft2,
        fftshift as opt_fftshift,
        ifftshift as opt_ifftshift,
        CachedFrequencyGrid,
        get_fft_backend,
    )
    _USE_OPTIMIZED_FFT = True
except ImportError:
    _USE_OPTIMIZED_FFT = False
    opt_fft2 = np.fft.fft2
    opt_ifft2 = np.fft.ifft2
    opt_fftshift = np.fft.fftshift
    opt_ifftshift = np.fft.ifftshift
    CachedFrequencyGrid = None
    def get_fft_backend(): return 'numpy'

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)



def validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_positive_array(arr: np.ndarray, name: str) -> None:
    arr = np.asarray(arr)
    if np.any(arr <= 0):
        raise ValueError(f"{name} elements must be positive")


def validate_range(value: float, name: str, min_val: float, max_val: float) -> None:
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be in [{min_val}, {max_val}], got {value}")


def validate_shape_match(arr1: np.ndarray, arr2: np.ndarray, name1: str, name2: str) -> None:
    if arr1.shape != arr2.shape:
        raise ValueError(f"{name1} shape {arr1.shape} != {name2} shape {arr2.shape}")


def validate_field(field: np.ndarray, name: str = "field") -> None:
    arr = np.asarray(field)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got ndim={arr.ndim}")
    if not np.iscomplexobj(arr):
        raise ValueError(f"{name} must be complex-valued")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")


def calculate_wavelength(photon_energy_ev: float) -> float:
    h = 6.62607015e-34
    c = 299792458.0
    e_charge = 1.602176634e-19
    photon_energy_joule = photon_energy_ev * e_charge
    if photon_energy_joule <= 0:
        raise ValueError("photon_energy_ev must be positive")
    return h * c / photon_energy_joule


def make_grid(window_size: float, grid_points: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    X = (np.arange(grid_points) - grid_points / 2) * window_size / grid_points
    Y = X.copy()
    X, Y = np.meshgrid(X, Y)
    dx = window_size / grid_points
    return X, Y, dx, dx
def fresnel_prop(field_in: np.ndarray, 
                 wavelength: float, 
                 dx: float, 
                 dy: float, 
                 propagation_distance: float,
                 config: Optional["PropagationConfig"] = None,
                 validate_input: bool = True) -> Union[Tuple[np.ndarray, float, float], Tuple[np.ndarray, float, float, dict]]:
    if config is None:
        config = PropagationConfig()
    
    if validate_input:
        validate_field(field_in, "field_in")
        validate_positive(wavelength, "wavelength")
        validate_positive(dx, "dx")
        validate_positive(dy, "dy")
        validate_positive(propagation_distance, "propagation_distance")
    
    ny_orig, nx_orig = field_in.shape
    field_size_x = nx_orig * dx
    field_size_y = ny_orig * dy
    wavenumber = 2.0 * np.pi / wavelength
    
    diagnostics = {
        'method': 'fresnel',
        'original_shape': (ny_orig, nx_orig),
        'original_sampling': (dx, dy),
        'wavelength': wavelength,
        'propagation_distance': propagation_distance,
        'padding_applied': False,
        'apodization_applied': False,
        'bandwidth_filter_applied': False,
        'energy_error': 0.0
    }
    
    field_work = field_in.copy()
    origin = None
    
    if config.padding_factor is not None and config.padding_factor > 1:
        field_work, origin = apply_zero_padding(field_work, config.padding_factor, mode='center')
        diagnostics['padding_applied'] = True
        diagnostics['padding_factor'] = config.padding_factor
        diagnostics['padded_shape'] = field_work.shape
        logger.debug(f"Applied zero padding: {(ny_orig, nx_orig)} → {field_work.shape}")
    elif config.padding_factor is None:
        estimated_factor = estimate_optimal_padding_factor(
            max(field_size_x, field_size_y),
            wavelength,
            propagation_distance,
            min_factor=2
        )
        if estimated_factor > 1:
            field_work, origin = apply_zero_padding(field_work, estimated_factor, mode='center')
            diagnostics['padding_applied'] = True
            diagnostics['padding_factor'] = estimated_factor
            diagnostics['padded_shape'] = field_work.shape
            logger.debug(f"Auto-padding: factor={estimated_factor}, shape {(ny_orig, nx_orig)} → {field_work.shape}")
    
    if config.apodization_type != 'none':
        window = create_apodization_window(
            field_work.shape,
            config.apodization_type,
            config.apodization_width
        )
        field_work = field_work * window
        diagnostics['apodization_applied'] = True
        diagnostics['apodization_type'] = config.apodization_type
        diagnostics['apodization_width'] = config.apodization_width
        logger.debug(f"Applied {config.apodization_type} window (width={config.apodization_width})")
    
    energy_before = np.sum(np.abs(field_work)**2) * dx * dy
    
    ny_work, nx_work = field_work.shape
    # Use cached frequency grids if available (optimized)
    if _USE_OPTIMIZED_FFT and CachedFrequencyGrid is not None:
        freq_grid_x, freq_grid_y = CachedFrequencyGrid.get((ny_work, nx_work), dx, dy)
    else:
        freq_x = np.fft.fftfreq(nx_work, d=dx)
        freq_y = np.fft.fftfreq(ny_work, d=dy)
        freq_grid_x, freq_grid_y = np.meshgrid(freq_x, freq_y)
    
    fft_field = opt_fft2(field_work)
    
    if config.enable_bandwidth_filter:
        field_before_filter = fft_field.copy()
        fft_field = apply_bandwidth_filter(
            fft_field,
            wavelength,
            dx,
            dy,
            cutoff_mode='safety_factor',
            cutoff_value=config.bandwidth_safety_factor,
            filter_type='hard'
        )
        diagnostics['bandwidth_filter_applied'] = True
        diagnostics['bandwidth_safety_factor'] = config.bandwidth_safety_factor
        filter_fraction = np.sum(np.abs(fft_field)**2) / (np.sum(np.abs(field_before_filter)**2) + 1e-30)
        diagnostics['bandwidth_filter_fraction'] = filter_fraction
        logger.debug(f"Bandwidth filter: safety_factor={config.bandwidth_safety_factor}, retained {filter_fraction*100:.2f}%")
    
    transfer_function = np.exp(1j * wavenumber * propagation_distance) * \
                       np.exp(-1j * np.pi * wavelength * propagation_distance * 
                             (freq_grid_x**2 + freq_grid_y**2))
    
    field_out = opt_ifft2(fft_field * transfer_function)
    
    if config.padding_factor is not None and config.padding_factor > 1 and origin is not None:
        field_out = extract_center(field_out, (ny_orig, nx_orig), origin)
        diagnostics['extracted_to_original'] = True
    
    energy_after = np.sum(np.abs(field_out)**2) * dx * dy
    energy_error = abs(energy_after - energy_before) / (energy_before + 1e-30)
    diagnostics['energy_before'] = energy_before
    diagnostics['energy_after'] = energy_after
    diagnostics['energy_error'] = energy_error
    
    if energy_error > config.energy_tolerance:
        logger.warning(f"Energy conservation violated: error={energy_error*100:.3f}% > tolerance={config.energy_tolerance*100:.3f}%")
    
    if config.return_diagnostics:
        return field_out, dx, dy, diagnostics
    else:
        return field_out, dx, dy

def fourier_prop(field_in: np.ndarray, 
                 wavelength: float, 
                 dx: float, 
                 dy: float, 
                 propagation_distance: float,
                 field_size: Optional[float] = None,
                 window_size: Optional[float] = None,
                 config: Optional["PropagationConfig"] = None,
                 validate_input: bool = True) -> Union[Tuple[np.ndarray, float, float], Tuple[np.ndarray, float, float, dict]]:
    if config is None:
        config = PropagationConfig()
    
    if validate_input:
        validate_field(field_in, "field_in")
        validate_positive(wavelength, "wavelength")
        validate_positive(dx, "dx")
        validate_positive(dy, "dy")
        validate_positive(propagation_distance, "propagation_distance")
    
    if window_size is not None and field_size is None:
        field_size = window_size
    
    ny_orig, nx_orig = field_in.shape
    
    if field_size is None:
        field_size = float(max(nx_orig * dx, ny_orig * dy))
    else:
        field_size = float(field_size)
    
    wavenumber = 2.0 * np.pi / wavelength
    
    diagnostics = {
        'method': 'fourier',
        'original_shape': (ny_orig, nx_orig),
        'original_sampling': (dx, dy),
        'field_size': field_size,
        'wavelength': wavelength,
        'propagation_distance': propagation_distance,
        'padding_applied': False,
        'apodization_applied': False,
        'energy_error': 0.0
    }
    
    field_work = field_in.copy()
    origin = None
    
    if config.padding_factor is not None and config.padding_factor > 1:
        field_work, origin = apply_zero_padding(field_work, config.padding_factor, mode='center')
        diagnostics['padding_applied'] = True
        diagnostics['padding_factor'] = config.padding_factor
        diagnostics['padded_shape'] = field_work.shape
        logger.debug(f"Applied zero padding: {(ny_orig, nx_orig)} → {field_work.shape}")
    
    if config.apodization_type != 'none':
        window = create_apodization_window(
            field_work.shape,
            config.apodization_type,
            config.apodization_width
        )
        field_work = field_work * window
        diagnostics['apodization_applied'] = True
        diagnostics['apodization_type'] = config.apodization_type
        diagnostics['apodization_width'] = config.apodization_width
        logger.debug(f"Applied {config.apodization_type} window (width={config.apodization_width})")
    
    energy_before = np.sum(np.abs(field_work)**2) * dx * dy
    
    ny_work, nx_work = field_work.shape
    fourier_field = (dx * dy) * opt_fftshift(opt_fft2(field_work))
    
    dx_out = wavelength * propagation_distance / field_size
    dy_out = wavelength * propagation_distance / field_size
    
    diagnostics['output_sampling'] = (dx_out, dy_out)
    
    x_coords = (np.arange(nx_work) - nx_work / 2) * dx_out
    y_coords = (np.arange(ny_work) - ny_work / 2) * dy_out
    grid_x_out, grid_y_out = np.meshgrid(x_coords, y_coords)
    
    phase_factor = np.exp(1j * wavenumber * propagation_distance) / \
                  (1j * wavelength * propagation_distance) * \
                  np.exp(1j * wavenumber * (grid_x_out**2 + grid_y_out**2) / 
                        (2.0 * propagation_distance))
    
    field_out = phase_factor * fourier_field
    
    if config.padding_factor is not None and config.padding_factor > 1 and origin is not None:
        field_out = extract_center(field_out, (ny_orig, nx_orig), origin)
        diagnostics['extracted_to_original'] = True
    
    energy_after = np.sum(np.abs(field_out)**2) * dx_out * dy_out
    energy_error = abs(energy_after - energy_before) / (energy_before + 1e-30)
    diagnostics['energy_before'] = energy_before
    diagnostics['energy_after'] = energy_after
    diagnostics['energy_error'] = energy_error
    
    if energy_error > config.energy_tolerance:
        logger.warning(f"Energy conservation violated: error={energy_error*100:.3f}% > tolerance={config.energy_tolerance*100:.3f}%")
    
    if config.return_diagnostics:
        return field_out, dx_out, dy_out, diagnostics
    else:
        return field_out, dx_out, dy_out


def make_plane_source(X: np.ndarray,
                      Y: np.ndarray,
                      wavelength: Optional[float] = None,
                      photon_energy_ev: Optional[float] = None,
                      spot_radius: Optional[float] = None,
                      energy: float = 1.0) -> np.ndarray:
    if wavelength is None and photon_energy_ev is not None:
        try:
            wavelength = calculate_wavelength(float(photon_energy_ev))
        except Exception:
            wavelength = None
    field = np.ones_like(X, dtype=np.complex64)
    if spot_radius is not None and spot_radius > 0:
        top_hat = ((X**2 + Y**2) <= (float(spot_radius)**2))
        field = field * top_hat.astype(field.dtype)
    try:
        dx = float(np.abs(X[0, 1] - X[0, 0])) if X.shape[1] > 1 else None
        dy = float(np.abs(Y[1, 0] - Y[0, 0])) if Y.shape[0] > 1 else None
    except Exception:
        dx = dy = None
    if dx is not None and dy is not None:
        norm = np.sqrt(np.sum(np.abs(field)**2) * dx * dy + 1e-30)
        field = field / norm
        field = field * np.sqrt(max(float(energy), 1e-30))
    return field


def make_gaussian_source(X: np.ndarray,
                         Y: np.ndarray,
                         sigma: float,
                         energy: float = 1.0) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 0:
        field = np.ones_like(X, dtype=np.complex64)
    else:
        r2 = X.astype(np.float64)**2 + Y.astype(np.float64)**2
        amp = np.exp(-r2 / (2.0 * sigma * sigma))
        field = amp.astype(np.complex64)
    try:
        dx = float(np.abs(X[0, 1] - X[0, 0])) if X.shape[1] > 1 else None
        dy = float(np.abs(Y[1, 0] - Y[0, 0])) if Y.shape[0] > 1 else None
    except Exception:
        dx = dy = None
    if dx is not None and dy is not None:
        norm = np.sqrt(np.sum(np.abs(field)**2) * dx * dy + 1e-30)
        field = field / norm
        field = field * np.sqrt(max(float(energy), 1e-30))
    return field


def ang_spec(field_in: np.ndarray, 
             wavelength: float, 
             dx: float, 
             dy: float, 
             propagation_distance: float,
             config: Optional['PropagationConfig'] = None,
             validate_input: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    if config is None:
        config = PropagationConfig()
    
    if validate_input:
        validate_field(field_in, "field_in")
        validate_positive(wavelength, "wavelength")
        validate_positive(dx, "dx")
        validate_positive(dy, "dy")
        validate_positive(propagation_distance, "propagation_distance")
    
    ny_orig, nx_orig = field_in.shape
    field_size_x = nx_orig * dx
    field_size_y = ny_orig * dy
    wavenumber = 2.0 * np.pi / wavelength
    
    diagnostics = {
        'method': 'angular_spectrum',
        'original_shape': (ny_orig, nx_orig),
        'original_sampling': (dx, dy),
        'wavelength': wavelength,
        'propagation_distance': propagation_distance,
        'padding_applied': False,
        'apodization_applied': False,
        'bandwidth_filter_applied': False,
        'sampling_adequate': True,
        'energy_error': 0.0
    }
    
    sampling_check = check_angular_spectrum_sampling(
        wavelength, dx, dy,
        safety_factor=config.bandwidth_safety_factor,
        numerical_aperture=config.numerical_aperture
    )
    
    diagnostics['sampling_check'] = sampling_check
    sampling_adequate = sampling_check['dx_ok'] and sampling_check['dy_ok']
    diagnostics['sampling_adequate'] = sampling_adequate
    diagnostics['recommended_method'] = sampling_check.get('recommended_method', 'angular_spectrum')
    
    if not sampling_adequate:
        logger.warning(
            f"Angular spectrum sampling inadequate! "
            f"dx={dx*1e9:.2f} nm (limit: {sampling_check['dx_nyquist']*1e9:.2f} nm), "
            f"dy={dy*1e9:.2f} nm (limit: {sampling_check['dy_nyquist']*1e9:.2f} nm). "
            f"Recommended method: {sampling_check.get('recommended_method', 'N/A')}"
        )
    
    field_work = field_in.copy()
    origin = None
    
    if config.padding_factor is not None and config.padding_factor > 1:
        field_work, origin = apply_zero_padding(field_work, config.padding_factor, mode='center')
        diagnostics['padding_applied'] = True
        diagnostics['padding_factor'] = config.padding_factor
        diagnostics['padded_shape'] = field_work.shape
        logger.debug(f"Applied zero padding: {(ny_orig, nx_orig)} → {field_work.shape}")
    elif config.padding_factor is None:
        estimated_factor = estimate_optimal_padding_factor(
            max(field_size_x, field_size_y),
            wavelength,
            propagation_distance,
            min_factor=2
        )
        if estimated_factor > 1:
            field_work, origin = apply_zero_padding(field_work, estimated_factor, mode='center')
            diagnostics['padding_applied'] = True
            diagnostics['padding_factor'] = estimated_factor
            diagnostics['padded_shape'] = field_work.shape
            logger.debug(f"Auto-padding: factor={estimated_factor}, shape {(ny_orig, nx_orig)} → {field_work.shape}")
    
    if config.apodization_type != 'none':
        window = create_apodization_window(
            field_work.shape,
            config.apodization_type,
            config.apodization_width
        )
        field_work = field_work * window
        diagnostics['apodization_applied'] = True
        diagnostics['apodization_type'] = config.apodization_type
        diagnostics['apodization_width'] = config.apodization_width
        logger.debug(f"Applied {config.apodization_type} window (width={config.apodization_width})")
    
    energy_before = np.sum(np.abs(field_work)**2) * dx * dy
    
    ny_work, nx_work = field_work.shape
    # Use cached frequency grids if available (optimized)
    if _USE_OPTIMIZED_FFT and CachedFrequencyGrid is not None:
        freq_grid_x, freq_grid_y = CachedFrequencyGrid.get((ny_work, nx_work), dx, dy)
    else:
        freq_x = np.fft.fftfreq(nx_work, d=dx)
        freq_y = np.fft.fftfreq(ny_work, d=dy)
        freq_grid_x, freq_grid_y = np.meshgrid(freq_x, freq_y)
    
    fft_field = opt_fft2(field_work)
    
    if config.enable_bandwidth_filter or config.numerical_aperture is not None:
        field_before_filter = fft_field.copy()
        
        if config.numerical_aperture is not None:
            cutoff_mode = 'numerical_aperture'
            cutoff_value = config.numerical_aperture
        else:
            cutoff_mode = 'safety_factor'
            cutoff_value = config.bandwidth_safety_factor
        
        fft_field = apply_bandwidth_filter(
            fft_field,
            wavelength,
            dx,
            dy,
            cutoff_mode=cutoff_mode,
            cutoff_value=cutoff_value,
            filter_type='hard'
        )
        diagnostics['bandwidth_filter_applied'] = True
        diagnostics['bandwidth_cutoff_mode'] = cutoff_mode
        diagnostics['bandwidth_cutoff_value'] = cutoff_value
        filter_fraction = np.sum(np.abs(fft_field)**2) / (np.sum(np.abs(field_before_filter)**2) + 1e-30)
        diagnostics['bandwidth_filter_fraction'] = filter_fraction
        logger.debug(f"Bandwidth filter: {cutoff_mode}={cutoff_value}, retained {filter_fraction*100:.2f}%")
    
    k_squared = (2.0 * np.pi * freq_grid_x)**2 + (2.0 * np.pi * freq_grid_y)**2
    sqrt_arg = wavenumber**2 - k_squared
    
    sqrt_arg = np.maximum(sqrt_arg, 0.0)
    
    transfer_function = np.exp(1j * propagation_distance * np.sqrt(sqrt_arg + 0.0j))
    
    field_out = opt_ifft2(fft_field * transfer_function)
    
    if config.padding_factor is not None and config.padding_factor > 1 and origin is not None:
        field_out = extract_center(field_out, (ny_orig, nx_orig), origin)
        diagnostics['extracted_to_original'] = True
    
    energy_after = np.sum(np.abs(field_out)**2) * dx * dy
    energy_error = abs(energy_after - energy_before) / (energy_before + 1e-30)
    diagnostics['energy_before'] = energy_before
    diagnostics['energy_after'] = energy_after
    diagnostics['energy_error'] = energy_error
    
    if energy_error > config.energy_tolerance:
        logger.warning(f"Energy conservation violated: error={energy_error*100:.3f}% > tolerance={config.energy_tolerance*100:.3f}%")
    
    if config.return_diagnostics:
        return field_out, diagnostics
    else:
        return field_out




def compute_rotation_step(rotation_mode: str,
                          step_angle_deg: float,
                          angular_velocity_deg_s: float,
                          pulse_rep_rate: float) -> float:
    if rotation_mode == 'omega':
        return np.deg2rad(angular_velocity_deg_s) / max(pulse_rep_rate, 1e-30)
    else:
        return np.deg2rad(step_angle_deg)













# classic_centroid, threshold_centroid, gaussian_fit_centroid, 
# peak_search_centroid, correlation_centroid, windowed_centroid,
# batch_centroid_computation, analyze_spot_resolution, batch_resolution_analysis


def compute_fresnel_number(aperture_size: float, 
                          wavelength: float, 
                          propagation_distance: float) -> float:
    a = float(aperture_size)
    lam = float(wavelength)
    z = float(propagation_distance)
    
    if z <= 0:
        raise ValueError(f"，: {z}")
    if lam <= 0:
        raise ValueError(f"，: {lam}")
    
    fresnel_num = (a ** 2) / (lam * z)
    return float(fresnel_num)


def classify_propagation_regime(fresnel_number: float) -> str:
    FN = float(fresnel_number)
    if FN > 10.0:
        return 'near'
    elif FN >= 0.1:
        return 'fresnel'
    else:
        return 'far'


def check_angular_spectrum_sampling(wavelength: float,
                                    dx: float,
                                    dy: float,
                                    safety_factor: float = 2.0,
                                    numerical_aperture: Optional[float] = None,
                                    theta_max: Optional[float] = None) -> dict:
    validate_positive(wavelength, "wavelength")
    validate_positive(dx, "dx")
    validate_positive(dy, "dy")
    
    lam = float(wavelength)
    dx_val = float(dx)
    dy_val = float(dy)
    sf = float(safety_factor)
    
    k = 2 * np.pi / lam
    k_x_max = np.pi / dx_val
    k_y_max = np.pi / dy_val
    
    dx_nyquist_basic = lam / (2.0 * sf)
    dy_nyquist_basic = lam / (2.0 * sf)
    
    if numerical_aperture is not None:
        theta_max = np.arcsin(min(numerical_aperture, 1.0))
    
    if theta_max is not None:
        k_lateral_max = k * np.sin(theta_max)
        dx_nyquist_na = np.pi / (k_lateral_max * sf)
        dy_nyquist_na = np.pi / (k_lateral_max * sf)
        
        dx_nyquist = min(dx_nyquist_basic, dx_nyquist_na)
        dy_nyquist = min(dy_nyquist_basic, dy_nyquist_na)
    else:
        dx_nyquist = dx_nyquist_basic
        dy_nyquist = dy_nyquist_basic
    
    dx_ok = dx_val <= dx_nyquist
    dy_ok = dy_val <= dy_nyquist
    
    margin_x = dx_nyquist / max(dx_val, 1e-30)
    margin_y = dy_nyquist / max(dy_val, 1e-30)
    min_margin = min(margin_x, margin_y)
    
    if min_margin > 2.0:
        recommended_method = 'ang_spec'
    elif min_margin > 0.5:
        recommended_method = 'fresnel'
    else:
        recommended_method = 'fourier'
    
    bandwidth_usage = min(k_x_max / k, k_y_max / k)
    
    return {
        'dx_nyquist': dx_nyquist,
        'dy_nyquist': dy_nyquist,
        'dx_ok': dx_ok,
        'dy_ok': dy_ok,
        'margin_x': float(margin_x),
        'margin_y': float(margin_y),
        'bandwidth_usage': float(bandwidth_usage),
        'recommended_method': recommended_method,
        'numerical_aperture': numerical_aperture,
        'theta_max_rad': theta_max
    }


def compute_required_zero_padding(field_size: float,
                                  propagation_distance: float,
                                  wavelength: float,
                                  min_padding_factor: float = 2.0) -> int:
    lam = float(wavelength)
    z = float(propagation_distance)
    L = float(field_size)
    
    spread_scale = np.sqrt(lam * z)
    
    required_size = L + 2.0 * spread_scale
    padding_factor = max(min_padding_factor, required_size / L)
    
    padding_pow2 = int(2 ** np.ceil(np.log2(padding_factor)))
    
    return max(2, padding_pow2)




# apply_scintillator_psf function has been migrated to sim_errors module


def compute_energy_conservation(field_before: np.ndarray,
                                field_after: np.ndarray,
                                dx_before: float,
                                dy_before: float,
                                dx_after: float,
                                dy_after: float) -> dict:
    E_before = np.sum(np.abs(field_before) ** 2) * dx_before * dy_before
    E_after = np.sum(np.abs(field_after) ** 2) * dx_after * dy_after
    
    rel_error = abs(E_after - E_before) / max(E_before, 1e-30)
    conserved = rel_error < 0.01
    
    return {
        'energy_before': float(E_before),
        'energy_after': float(E_after),
        'relative_error': float(rel_error),
        'conserved': bool(conserved)
    }


def simulate_partial_coherence_multimode(source_generator_func,
                                        propagation_func,
                                        num_modes: int = 5,
                                        mode_weights: Optional[np.ndarray] = None,
                                        **source_params) -> np.ndarray:
    if mode_weights is None:
        weights = np.ones(num_modes) / num_modes
    else:
        weights = np.asarray(mode_weights, dtype=float)
        weights = weights / np.sum(weights)
    
    intensity_total = None
    
    for mode_idx in range(num_modes):
        U0_mode = source_generator_func(mode_idx, **source_params)
        U1_mode = propagation_func(U0_mode)
        I_mode = np.abs(U1_mode) ** 2
        
        if intensity_total is None:
            intensity_total = weights[mode_idx] * I_mode
        else:
            intensity_total += weights[mode_idx] * I_mode
    
    return intensity_total


def print_simulation_diagnostics(wavelength: float,
                                 dx: float,
                                 dy: float,
                                 aperture_size: float,
                                 z1: float,
                                 z2: float,
                                 scintillator_thickness: Optional[float] = None) -> None:
    print("=" * 70)
    print("Rigorous Wave Optics Simulation Diagnostics")
    print("=" * 70)
    
    E_keV = 1.2398e-9 / wavelength / 1e3
    print(f"Wavelength:         {wavelength*1e10:.4f} Å  (Energy: {E_keV:.2f} keV)")
    print(f"Wave number k:      {2*np.pi/wavelength:.2e} m⁻¹")
    
    FN1 = compute_fresnel_number(aperture_size, wavelength, z1)
    FN2 = compute_fresnel_number(aperture_size, wavelength, z2)
    regime1 = classify_propagation_regime(FN1)
    regime2 = classify_propagation_regime(FN2)
    
    print(f"\nFresnel Numbers:")
    print(f"  Source → Hartmann (z={z1*1e3:.1f} mm): FN = {FN1:.3f} [{regime1}]")
    print(f"  Hartmann → Detector (z={z2*1e3:.1f} mm): FN = {FN2:.3f} [{regime2}]")
    
    check = check_angular_spectrum_sampling(wavelength, dx, dy)
    print(f"\nSampling Nyquist Check:")
    print(f"  Current Δx = {dx*1e9:.2f} nm, Δy = {dy*1e9:.2f} nm")
    print(f"  Recommended Δx ≤ {check['dx_nyquist']*1e9:.2f} nm")
    print(f"  Status: {'✓ Satisfied' if check['dx_ok'] else '✗ Not satisfied'} (Margin: {check['margin_x']:.2f}×)")
    
    pad1 = compute_required_zero_padding(aperture_size*10, z1, wavelength)
    pad2 = compute_required_zero_padding(aperture_size*10, z2, wavelength)
    print(f"\nRecommended Zero Padding Factors:")
    print(f"  First stage propagation: {pad1}×")
    print(f"  Second stage propagation: {pad2}×")
    
    if scintillator_thickness is not None:
        sigma1 = 0.03 * scintillator_thickness
        sigma2 = 0.18 * scintillator_thickness
        print(f"\nScintillator PSF (thickness t={scintillator_thickness*1e6:.1f} μm):")
        print(f"  Narrow Gaussian σ1 = {sigma1*1e6:.3f} μm")
        print(f"  Broad Gaussian σ2 = {sigma2*1e6:.3f} μm")
        print(f"  Weight a = 0.8")
    
    print("=" * 70)



def propagate_with_auto_padding(field: np.ndarray,
                                wavelength: float,
                                dx: float,
                                dy: float,
                                z: float,
                                method: str = 'auto',
                                enable_bandwidth_filter: bool = True,
                                safety_factor: float = 2.0) -> Tuple[np.ndarray, dict]:
    ny, nx = field.shape
    L_x = nx * dx
    L_y = ny * dy
    aperture_est = min(L_x, L_y) / 4.0
    
    FN = compute_fresnel_number(aperture_est, wavelength, z)
    regime = classify_propagation_regime(FN)
    
    if method == 'auto':
        if regime == 'far':
            method_actual = 'fourier'
        elif regime == 'near':
            method_actual = 'fresnel'
        else:
            method_actual = 'ang_spec'
    else:
        method_actual = method
    
    pad_factor = compute_required_zero_padding(max(L_x, L_y), z, wavelength)
    
    ny_pad = ny * pad_factor
    nx_pad = nx * pad_factor
    field_padded = np.zeros((ny_pad, nx_pad), dtype=complex)
    
    start_y = (ny_pad - ny) // 2
    start_x = (nx_pad - nx) // 2
    field_padded[start_y:start_y+ny, start_x:start_x+nx] = field
    
    if enable_bandwidth_filter:
        # Use cached frequency grids if available (optimized)
        if _USE_OPTIMIZED_FFT and CachedFrequencyGrid is not None:
            freq_grid_x, freq_grid_y = CachedFrequencyGrid.get((ny_pad, nx_pad), dx, dy)
            Kx = freq_grid_x * 2 * np.pi
            Ky = freq_grid_y * 2 * np.pi
        else:
            kx = np.fft.fftfreq(nx_pad, dx) * 2 * np.pi
            ky = np.fft.fftfreq(ny_pad, dy) * 2 * np.pi
            Kx, Ky = np.meshgrid(kx, ky)
        K_sq = Kx**2 + Ky**2
        k = 2 * np.pi / wavelength
        
        k_cutoff = k / safety_factor
        bandwidth_filter = (K_sq <= k_cutoff**2).astype(float)
        
        F_field = opt_fft2(field_padded)
        F_field_filtered = F_field * bandwidth_filter
        field_padded = opt_ifft2(F_field_filtered)
    
    E_before = np.sum(np.abs(field_padded)**2) * dx * dy
    
    if method_actual == 'ang_spec':
        result = ang_spec(field_padded, wavelength, dx, dy, z)
        field_prop_padded = result if isinstance(result, np.ndarray) else result[0]
    elif method_actual == 'fresnel':
        result = fresnel_prop(field_padded, wavelength, dx, dy, z)
        field_prop_padded = result[0]
    elif method_actual == 'fourier':
        result = fourier_prop(field_padded, wavelength, dx, dy, z, window_size=1.0)
        field_prop_padded = result[0]
    else:
        raise ValueError(f": {method_actual}")
    
    field_propagated = field_prop_padded[start_y:start_y+ny, start_x:start_x+nx]
    
    E_after = np.sum(np.abs(field_propagated)**2) * dx * dy
    
    info = {
        'method_used': method_actual,
        'fresnel_number': float(FN),
        'regime': regime,
        'padding_factor': int(pad_factor),
        'bandwidth_filtered': bool(enable_bandwidth_filter),
        'energy_before': float(E_before),
        'energy_after': float(E_after),
        'energy_error': float(abs(E_after - E_before) / max(E_before, 1e-30))
    }
    
    return field_propagated, info


def create_propagation_pipeline(source_field: np.ndarray,
                               wavelength: float,
                               dx: float,
                               dy: float,
                               z_stages: list,
                               masks: Optional[list] = None,
                               scintillator_thickness: Optional[float] = None,
                               pixel_size_detector: Optional[float] = None,
                               photon_flux: Optional[float] = None,
                               enable_diagnostics: bool = True) -> dict:
    field = source_field.copy()
    diagnostics = []
    
    for i, z in enumerate(z_stages):
        if enable_diagnostics:
            print(f"\n---  {i+1}/{len(z_stages)}: z = {z*1e3:.1f} mm ---")
        
        field, info = propagate_with_auto_padding(field, wavelength, dx, dy, z)
        
        if masks is not None and i < len(masks) and masks[i] is not None:
            field = field * masks[i]
            if enable_diagnostics:
                print(f" (: {np.sum(np.abs(masks[i])**2)/masks[i].size:.2%})")
        
        diagnostics.append(info)
        
        if enable_diagnostics:
            print(f": {info['method_used']}, FN={info['fresnel_number']:.2f}, "
                  f": {info['energy_error']*100:.3f}%")
    
    intensity = np.abs(field) ** 2
    
    try:
        from simulator.blur import apply_gaussian_blur as _apply_blur
        intensity_blurred = _apply_blur(intensity, sigma=2.0)
    except Exception:
        intensity_blurred = intensity.copy()
    
    detector_counts = intensity_blurred.copy()
    
    if enable_diagnostics:
        print(f"，2，")
    
    return {
        'field_final': field,
        'intensity_final': intensity_blurred,
        'detector_counts': detector_counts,
        'stage_diagnostics': diagnostics
    }

@dataclass
class PropagationConfig:
    method: Literal['auto', 'ang_spec', 'fresnel', 'fourier'] = 'ang_spec'
    enable_auto_padding: Optional[bool] = None
    padding_factor: Optional[int] = None
    apodization_type: Literal['none', 'tukey', 'hanning'] = 'none'
    apodization_width: float = 0.1
    enable_bandwidth_filter: bool = True
    bandwidth_safety_factor: float = 2.0
    numerical_aperture: Optional[float] = None
    energy_tolerance: float = 0.01
    return_diagnostics: bool = False

def estimate_optimal_padding_factor(field_size: float,
                                    wavelength: float,
                                    propagation_distance: float,
                                    min_factor: float = 2.0) -> int:
    """
    Estimate optimal zero-padding factor to avoid aliasing.
    """
    # Heuristic: ensure padding covers diffraction spread
    # For simplicity, return a safe factor
    return int(max(min_factor, 2))


def apply_zero_padding(field: np.ndarray, 
                       padding_factor: float, 
                       mode: str = 'center') -> Tuple[np.ndarray, Tuple[int, int]]:
    ny, nx = field.shape
    ny_pad = int(round(ny * padding_factor))
    nx_pad = int(round(nx * padding_factor))
    
    padded = np.zeros((ny_pad, nx_pad), dtype=field.dtype)
    
    start_y = (ny_pad - ny) // 2
    start_x = (nx_pad - nx) // 2
    
    padded[start_y:start_y+ny, start_x:start_x+nx] = field
    
    return padded, (start_y, start_x)


def extract_center(field: np.ndarray, 
                   original_shape: Tuple[int, int], 
                   origin: Tuple[int, int]) -> np.ndarray:
    ny, nx = original_shape
    start_y, start_x = origin
    return field[start_y:start_y+ny, start_x:start_x+nx]


def create_apodization_window(shape: Tuple[int, int], 
                              type: str, 
                              width: float) -> np.ndarray:
    ny, nx = shape
    if type == 'none':
        return np.ones(shape)
    
    if type == 'hanning':
        hy = np.hanning(ny)
        hx = np.hanning(nx)
        return np.outer(hy, hx)
        
    if type == 'tukey':
        alpha = width
        if alpha <= 0: return np.ones(shape)
        if alpha >= 1: 
            hy = np.hanning(ny)
            hx = np.hanning(nx)
            return np.outer(hy, hx)
            
        def _tukey(N, alpha):
            n = np.arange(N)
            w = np.ones(N)
            L = int(alpha * (N - 1) / 2)
            if L > 0:
                w[:L] = 0.5 * (1 + np.cos(np.pi * (2*n[:L]/(alpha*(N-1)) - 1)))
                w[-L:] = 0.5 * (1 + np.cos(np.pi * (2*n[-L:]/(alpha*(N-1)) - 2/alpha + 1)))
            return w
            
        wy = _tukey(ny, alpha)
        wx = _tukey(nx, alpha)
        return np.outer(wy, wx)

    return np.ones(shape)


def apply_bandwidth_filter(field_k: np.ndarray,
                           wavelength: float,
                           dx: float,
                           dy: float,
                           cutoff_mode: str = 'safety_factor',
                           cutoff_value: float = 2.0,
                           filter_type: str = 'hard') -> np.ndarray:
    ny, nx = field_k.shape
    # Use cached frequency grids if available (optimized)
    if _USE_OPTIMIZED_FFT and CachedFrequencyGrid is not None:
        freq_grid_x, freq_grid_y = CachedFrequencyGrid.get((ny, nx), dx, dy)
        KX = freq_grid_x * 2 * np.pi
        KY = freq_grid_y * 2 * np.pi
    else:
        kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
    K_sq = KX**2 + KY**2
    k0 = 2 * np.pi / wavelength
    
    if cutoff_mode == 'numerical_aperture':
        k_cutoff = k0 * min(cutoff_value, 1.0)
    else:
        k_cutoff = k0 / max(cutoff_value, 1.0)
        
    mask = (K_sq <= k_cutoff**2).astype(float)
    return field_k * mask
