from __future__ import annotations
import numpy as np
from math import factorial

def zernike_polynomial(n, m, rho, phi):
    """
    Calculate Zernike polynomial (n, m) at coordinates (rho, phi).
    rho: radial coordinate (normalized 0..1)
    phi: angular coordinate
    """
    if (n - m) % 2 != 0:
        return np.zeros_like(rho)
    
    R = np.zeros_like(rho)
    for k in range((n - abs(m)) // 2 + 1):
        c = (-1)**k * factorial(n - k)
        c /= factorial(k) * factorial((n + abs(m)) // 2 - k) * factorial((n - abs(m)) // 2 - k)
        R += c * rho**(n - 2*k)
        
    if m >= 0:
        return R * np.cos(m * phi)
    else:
        return R * np.sin(-m * phi)

def generate_random_phase_screen(X: np.ndarray, Y: np.ndarray, pupil_radius: float, max_order: int = 15, amplitude_range: tuple = (-0.5, 0.5)):
    """
    Generates a random phase screen using Zernike polynomials.
    
    Args:
        X, Y: Grid coordinates.
        pupil_radius: Radius to normalize coordinates.
        max_order: Maximum Noll index for Zernike polynomials.
        amplitude_range: Range for random coefficients.
    """
    rho = np.sqrt(X**2 + Y**2) / pupil_radius
    phi = np.arctan2(Y, X)
    
    # Mask outside pupil (optional, but good for Zernike definition)
    # For phase screen over whole window, we might just let it extrapolate or clip rho
    rho = np.clip(rho, 0, 1.0) 
    
    phase = np.zeros_like(X)
    
    # Noll index mapping (simplified for low orders)
    # (n, m) pairs for Noll indices 1..15
    noll_indices = [
        (0,0), # 1: Piston
        (1,1), # 2: Tilt X
        (1,-1), # 3: Tilt Y
        (2,0), # 4: Defocus
        (2,-2), # 5: Astigmatism
        (2,2), # 6: Astigmatism
        (3,-1), # 7: Coma
        (3,1), # 8: Coma
        (3,-3), # 9: Trefoil
        (3,3), # 10: Trefoil
        (4,0), # 11: Spherical
        (4,2), # 12
        (4,-2), # 13
        (4,4), # 14
        (4,-4) # 15
    ]
    
    coeffs = {}

    max_noll_index = int(max_order)
    if max_noll_index < 1:
        raise ValueError(f"max_order must be >= 1, got {max_order}")

    # Skip piston (Z1) but keep tilt terms and all modes up to the requested
    # maximum Noll index.
    for noll_index in range(2, max_noll_index + 1):
        if noll_index > len(noll_indices):
            break
        n, m = noll_indices[noll_index - 1]
        
        # Random coefficient
        coef = np.random.uniform(amplitude_range[0], amplitude_range[1])
        coeffs[f"Z{noll_index}"] = coef
        
        z_term = zernike_polynomial(n, m, rho, phi)
        phase += coef * z_term
        
    return phase, coeffs
