from __future__ import annotations
"""
Hartmann Sensor Performance Analysis Module

Calculates key performance metrics:
- Sensitivity (minimum detectable slope)
- Angular resolution
- Dynamic range
"""

import numpy as np
from typing import Dict, Any

from .optimization_common import estimate_spot_fwhm_sigma_from_image
from .serialization import dump_json_serializable


def calculate_sensor_performance(
    pixel_size: float,
    z_prop: float,
    wavelength: float,
    spacing: float,
    spot_sigma_px: float = 2.5,
    subpixel_precision: float = 0.1,
) -> Dict[str, Any]:
    """
    Calculate Hartmann sensor performance metrics.
    
    Parameters:
    -----------
    pixel_size : float
        Detector pixel size in meters
    z_prop : float
        Propagation distance in meters
    wavelength : float
        X-ray wavelength in meters
    spacing : float
        Grating period in meters
    spot_sigma_px : float
        Spot size (sigma) in pixels
    subpixel_precision : float
        Achievable subpixel measurement precision (typically 0.01-0.1 pixel)
        
    Returns:
    --------
    Dict containing all performance metrics
    """
    # Minimum detectable displacement
    min_detectable_shift_m = subpixel_precision * pixel_size
    
    # Minimum detectable slope (angular resolution)
    min_slope = min_detectable_shift_m / z_prop
    angular_resolution_rad = min_slope
    angular_resolution_urad = min_slope * 1e6
    
    # Minimum detectable phase gradient
    min_phase_grad = min_slope * (2 * np.pi / wavelength)
    
    # Maximum measurable displacement (limited by grating period)
    max_shift = spacing / 2  # Half period before overlap
    max_slope = max_shift / z_prop
    
    # Dynamic range
    dynamic_range = max_slope / min_slope if min_slope > 0 else np.inf
    
    # Spatial resolution (determined by grating period)
    spatial_resolution = spacing
    
    return {
        # System parameters
        "pixel_size_m": pixel_size,
        "pixel_size_um": pixel_size * 1e6,
        "z_prop_m": z_prop,
        "wavelength_m": wavelength,
        "wavelength_nm": wavelength * 1e9,
        "spacing_m": spacing,
        "spacing_um": spacing * 1e6,
        "spot_sigma_px": spot_sigma_px,
        "spot_sigma_um": spot_sigma_px * pixel_size * 1e6,
        
        # Sensitivity
        "subpixel_precision_px": subpixel_precision,
        "min_detectable_shift_nm": min_detectable_shift_m * 1e9,
        "angular_resolution_urad": angular_resolution_urad,
        "angular_resolution_rad": angular_resolution_rad,
        "min_phase_gradient_rad_per_m": min_phase_grad,
        
        # Dynamic range
        "max_shift_um": max_shift * 1e6,
        "max_slope_mrad": max_slope * 1e3,
        "dynamic_range": dynamic_range,
        
        # Spatial resolution
        "spatial_resolution_um": spatial_resolution * 1e6,
    }


def print_sensor_performance(metrics: Dict[str, Any]) -> None:
    """Print sensor performance metrics in a formatted way."""
    print("\n" + "="*70)
    print("Hartmann Sensor Performance Analysis")
    print("="*70)
    
    print(f"\n【System Parameters】")
    print(f"  Wavelength λ = {metrics['wavelength_nm']:.4f} nm")
    print(f"  Propagation distance z = {metrics['z_prop_m']*100:.1f} cm")
    print(f"  Grating period = {metrics['spacing_um']:.0f} µm")
    print(f"  Pixel size = {metrics['pixel_size_um']:.3f} µm")
    print(f"  Spot σ = {metrics['spot_sigma_px']:.2f} px = {metrics['spot_sigma_um']:.1f} µm")
    
    print(f"\n【Sensitivity】")
    print(f"  Subpixel precision = {metrics['subpixel_precision_px']:.2f} px = {metrics['min_detectable_shift_nm']:.0f} nm")
    print(f"  Angular resolution = {metrics['angular_resolution_urad']:.2f} µrad")
    print(f"  Min phase gradient = {metrics['min_phase_gradient_rad_per_m']:.0f} rad/m")
    
    print(f"\n【Dynamic Range】")
    print(f"  Max measurable shift = {metrics['max_shift_um']:.0f} µm")
    print(f"  Max measurable slope = {metrics['max_slope_mrad']:.2f} mrad")
    print(f"  Dynamic range = {metrics['dynamic_range']:.0f}:1")
    
    print(f"\n【Spatial Resolution】")
    print(f"  Spatial resolution = {metrics['spatial_resolution_um']:.0f} µm (grating period)")
    print("="*70)



def calculate_spot_size_fresnel(
    hole_diameter: float,
    wavelength: float,
    z_prop: float,
    pixel_size: float = 1.0,  # If 1.0, returns meters. If pixel_size, returns pixels.
) -> float:
    """
    Calculate approximate spot size (diameter) in Fresnel region.
    
    Using Fresnel characteristic scale: D = 2 * sqrt(lambda * z)
    This represents the scale of the first Fresnel zone or the characteristic 
    diffraction scale in the near field.
    """
    # User requested formula: 2 * sqrt(lambda * z)
    spot_diameter_m = 2.0 * np.sqrt(wavelength * z_prop)
    
    if pixel_size == 1.0:
        return spot_diameter_m
    else:
        return spot_diameter_m / pixel_size


def calculate_expected_displacement(
    aberration_coeff_rad: float,
    aperture_m: float,
    wavelength_m: float,
    z_prop_m: float,
) -> Dict[str, float]:
    """
    Calculate expected displacement from a given aberration coefficient.
    
    For Zernike-like aberrations, the phase gradient is approximately:
    ∇φ ≈ coeff / (aperture/2)
    
    Displacement = (λ/2π) × ∇φ × z
    """
    phase_gradient = aberration_coeff_rad / (aperture_m / 2)
    displacement_m = (wavelength_m / (2 * np.pi)) * phase_gradient * z_prop_m
    
    return {
        "phase_gradient_rad_per_m": phase_gradient,
        "displacement_m": displacement_m,
        "displacement_um": displacement_m * 1e6,
        "displacement_nm": displacement_m * 1e9,
    }


def save_performance_json(metrics: Dict[str, Any], output_path: str) -> None:
    """Save performance metrics to JSON file."""
    dump_json_serializable(metrics, output_path)


def analyze_spot_size(
    hole_diameter: float,
    wavelength: float,
    z_prop: float,
    pixel_size: float,
    ideal_image: np.ndarray = None,
    spacing_px: float = None
) -> Dict[str, Any]:
    """
    Comprehensive spot size analysis.
    
    Parameters
    ----------
    hole_diameter : float
        Hole diameter in meters
    wavelength : float
        Wavelength in meters
    z_prop : float
        Propagation distance in meters
    pixel_size : float
        Pixel size in meters
    ideal_image : np.ndarray, optional
        Ideal accumulated image for FWHM measurement
    spacing_px : float, optional
        Spacing in pixels (required if ideal_image is provided)
        
    Returns
    -------
    Dict with spot size metrics
    """
    # 1. Theoretical spot size (Fresnel characteristic scale)
    fresnel_diameter_m = 2.0 * np.sqrt(wavelength * z_prop)
    fresnel_diameter_px = fresnel_diameter_m / pixel_size
    
    # 2. Fresnel number
    fresnel_number = (hole_diameter / 2)**2 / (wavelength * z_prop)
    
    # 3. Measured FWHM from ideal image (if provided)
    measured_fwhm_px = None
    measured_fwhm_m = None
    measured_sigma_px = None
    
    if ideal_image is not None and spacing_px is not None:
        measured_fwhm_px, measured_sigma_px = estimate_spot_fwhm_sigma_from_image(
            np.asarray(ideal_image, dtype=np.float64),
            float(spacing_px),
        )
        if measured_fwhm_px is not None:
            measured_fwhm_m = measured_fwhm_px * pixel_size

    return {
        "fresnel_diameter_m": fresnel_diameter_m,
        "fresnel_diameter_um": fresnel_diameter_m * 1e6,
        "fresnel_diameter_px": fresnel_diameter_px,
        "fresnel_number": fresnel_number,
        "measured_fwhm_px": measured_fwhm_px,
        "measured_fwhm_m": measured_fwhm_m,
        "measured_fwhm_um": measured_fwhm_m * 1e6 if measured_fwhm_m else None,
        "measured_sigma_px": measured_sigma_px,
        "hole_diameter_m": hole_diameter,
        "hole_diameter_um": hole_diameter * 1e6,
    }


def print_spot_size_analysis(spot_metrics: Dict[str, Any]) -> None:
    """Print spot size analysis in formatted way."""
    print("\n" + "="*70)
    print("Spot Size Analysis")
    print("="*70)
    
    print(f"\n【Aperture】")
    print(f"  Hole diameter = {spot_metrics['hole_diameter_um']:.2f} µm")
    
    print(f"\n【Theoretical (Fresnel)】")
    print(f"  Fresnel number N_F = {spot_metrics['fresnel_number']:.3f}")
    if spot_metrics['fresnel_number'] < 1:
        regime = "Far-field (Fraunhofer)"
    elif spot_metrics['fresnel_number'] < 10:
        regime = "Near-field (Fresnel)"
    else:
        regime = "Very near-field"
    print(f"  Diffraction regime: {regime}")
    print(f"  Fresnel characteristic scale = {spot_metrics['fresnel_diameter_um']:.2f} µm = {spot_metrics['fresnel_diameter_px']:.2f} px")
    
    if spot_metrics['measured_fwhm_px'] is not None:
        print(f"\n【Measured from Ideal Image】")
        print(f"  FWHM = {spot_metrics['measured_fwhm_px']:.2f} px = {spot_metrics['measured_fwhm_um']:.2f} µm")
        print(f"  σ (Gaussian) = {spot_metrics['measured_sigma_px']:.2f} px")
        
        ratio = spot_metrics['measured_fwhm_px'] / spot_metrics['fresnel_diameter_px']
        print(f"  Measured/Theoretical ratio = {ratio:.2f}")
    
    print("="*70)
