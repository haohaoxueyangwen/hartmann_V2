from __future__ import annotations
"""
Microlens Array Module

Creates a microlens array mask that applies a parabolic phase profile to each lenslet,
focusing light to spots on the detector.

Based on mla.py implementation.
"""

import numpy as np
from typing import Tuple


def create_microlens_array(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    pitch: float,
    focal_length: float,
    wavelength: float,
    fill_factor: float = 1.0,
    lens_shape: str = 'circle',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a microlens array mask.
    
    Unlike a Hartmann mask (binary aperture), a microlens array applies a 
    parabolic phase shift to each lenslet, focusing light to spots.
    
    Parameters:
    -----------
    X, Y : np.ndarray
        Coordinate grids (in meters)
    pitch : float
        Period between lenslets (in meters)
    focal_length : float
        Focal length of each lenslet (in meters)
    wavelength : float
        Wavelength of light (in meters)
    fill_factor : float
        Linear fill factor (0-1). 1.0 means lens diameter = pitch (no gap)
    lens_shape : str
        Lens shape: 'circle' or 'square'
        
    Returns:
    --------
    amplitude_mask : np.ndarray
        Binary mask (1 inside lens, 0 outside)
    phase_mask : np.ndarray
        Phase profile from lenslet focusing (in radians)
    """
    # Wrap coordinates to unit cell (-pitch/2, pitch/2)
    # This maps each point to its local lenslet center
    x_local = np.mod(X + pitch/2, pitch) - pitch/2
    y_local = np.mod(Y + pitch/2, pitch) - pitch/2
    
    # Effective lens radius based on linear fill factor
    effective_radius = (pitch * fill_factor) / 2
    r_local = np.sqrt(x_local**2 + y_local**2)
    
    # Determine inside region based on lens shape
    if lens_shape.lower() == 'circle':
        inside = r_local <= effective_radius
    else:  # square
        half_size = effective_radius
        inside = (np.abs(x_local) <= half_size) & (np.abs(y_local) <= half_size)
    
    amplitude_mask = inside.astype(float)
    
    # Thin lens phase: φ = -k/(2f) * r²
    k = 2 * np.pi / wavelength
    phase_mask = -k / (2 * focal_length) * (x_local**2 + y_local**2)
    
    # Zero phase outside aperture
    phase_mask = np.where(inside, phase_mask, 0.0)
    
    return amplitude_mask, phase_mask


def get_lenslet_positions(
    window_size: float,
    pitch: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the positions of all lenslet centers within the window.
    
    Returns:
    --------
    x_centers, y_centers : np.ndarray
        Arrays of lenslet center coordinates
    """
    n_lenslets = int(np.ceil(window_size / pitch)) + 1
    half_n = n_lenslets // 2
    
    # Create grid of centers
    indices = np.arange(-half_n, half_n + 1)
    x_centers = indices * pitch
    y_centers = indices * pitch
    
    # Filter to those within the window
    x_centers = x_centers[np.abs(x_centers) <= window_size/2]
    y_centers = y_centers[np.abs(y_centers) <= window_size/2]
    
    return x_centers, y_centers


def calculate_microlens_spot_size(
    focal_length: float,
    wavelength: float,
    lens_diameter: float,
) -> float:
    """
    Calculate the expected spot size (Airy disk radius) at the focus.
    
    Parameters:
    -----------
    focal_length : float
        Focal length of lenslet (m)
    wavelength : float
        Wavelength (m)
    lens_diameter : float
        Diameter of lenslet (m)
        
    Returns:
    --------
    spot_radius : float
        First Airy disk radius (m)
    """
    # Airy disk radius = 1.22 * lambda * f/D
    return 1.22 * wavelength * focal_length / lens_diameter


def calculate_mla_na(
    pitch: float,
    focal_length: float,
) -> float:
    """
    Calculate MLA numerical aperture (NA).
    
    NA = D/(2f) (small angle approximation)
    
    Returns:
    --------
    NA : float
        Numerical aperture
    """
    return pitch / (2 * focal_length)
