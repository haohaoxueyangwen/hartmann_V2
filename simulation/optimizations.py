#!/usr/bin/env python3
"""
Performance optimization utilities for Hartmann simulation.

This module provides:
1. FFT acceleration using pyfftw or scipy.fft
2. Frequency grid caching
3. Numba JIT compilation for hot paths
4. Optimized blur functions using scipy
5. Vectorized operations

Usage:
    from simulation.optimizations import (
        get_fft_backend,
        CachedFrequencyGrid,
        fast_gaussian_blur,
        optimized_zncc_batch,
    )
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# FFT Backend Selection
# =============================================================================

_FFT_BACKEND = None
_PYFFTW_AVAILABLE = False
_SCIPY_FFT_AVAILABLE = False

# Try importing PyFFTW
try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as pyfftw_fft
    _PYFFTW_AVAILABLE = True
    # Enable caching for FFTW
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(30)  # Keep plans alive for 30 seconds
except ImportError:
    pyfftw = None
    pyfftw_fft = None

# Try importing SciPy
try:
    import scipy.fft as scipy_fft
    _SCIPY_FFT_AVAILABLE = True
except ImportError:
    scipy_fft = None


def get_fft_backend() -> str:
    """Return the name of the active FFT backend."""
    global _FFT_BACKEND
    if _FFT_BACKEND is not None:
        return _FFT_BACKEND
    
    if _PYFFTW_AVAILABLE:
        _FFT_BACKEND = 'pyfftw'
    elif _SCIPY_FFT_AVAILABLE:
        _FFT_BACKEND = 'scipy'
    else:
        _FFT_BACKEND = 'numpy'
    
    return _FFT_BACKEND


def fft2(arr: np.ndarray) -> np.ndarray:
    """Optimized 2D FFT using best available backend."""
    backend = get_fft_backend()
    if backend == 'pyfftw':
        return pyfftw_fft.fft2(arr)
    elif backend == 'scipy':
        return scipy_fft.fft2(arr)
    else:
        return np.fft.fft2(arr)


def ifft2(arr: np.ndarray) -> np.ndarray:
    """Optimized 2D inverse FFT using best available backend."""
    backend = get_fft_backend()
    if backend == 'pyfftw':
        return pyfftw_fft.ifft2(arr)
    elif backend == 'scipy':
        return scipy_fft.ifft2(arr)
    else:
        return np.fft.ifft2(arr)


def fftshift(arr: np.ndarray) -> np.ndarray:
    """FFT shift with best backend."""
    backend = get_fft_backend()
    if backend == 'scipy':
        return scipy_fft.fftshift(arr)
    else:
        return np.fft.fftshift(arr)


def ifftshift(arr: np.ndarray) -> np.ndarray:
    """Inverse FFT shift with best backend."""
    backend = get_fft_backend()
    if backend == 'scipy':
        return scipy_fft.ifftshift(arr)
    else:
        return np.fft.ifftshift(arr)


def fftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """FFT frequencies with best backend."""
    backend = get_fft_backend()
    if backend == 'scipy':
        return scipy_fft.fftfreq(n, d)
    else:
        return np.fft.fftfreq(n, d)


# =============================================================================
# Frequency Grid Caching
# =============================================================================

class CachedFrequencyGrid:
    """
    Cache for frequency grids to avoid repeated computation.
    
    For a 2048x2048 grid, this saves ~5ms per call.
    """
    _cache: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}
    _max_cache_size = 10
    
    @classmethod
    def get(cls, shape: Tuple[int, int], dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get or create frequency grids."""
        # Round dx/dy to avoid floating point key issues
        key = (shape[0], shape[1], round(dx, 12), round(dy, 12))
        
        if key in cls._cache:
            return cls._cache[key]
        
        ny, nx = shape
        freq_x = fftfreq(nx, d=dx)
        freq_y = fftfreq(ny, d=dy)
        freq_grid_x, freq_grid_y = np.meshgrid(freq_x, freq_y)
        
        # Manage cache size
        if len(cls._cache) >= cls._max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(cls._cache))
            del cls._cache[oldest_key]
        
        cls._cache[key] = (freq_grid_x, freq_grid_y)
        return freq_grid_x, freq_grid_y
    
    @classmethod
    def clear(cls):
        """Clear the cache."""
        cls._cache.clear()


# =============================================================================
# Optimized Gaussian Blur
# =============================================================================

def fast_gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """
    Fast Gaussian blur using scipy.ndimage (much faster than convolution).
    
    Falls back to separable convolution if scipy unavailable.
    """
    if sigma is None or sigma <= 0:
        return arr.copy() if isinstance(arr, np.ndarray) else np.asarray(arr)
    
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(arr.astype(np.float64), sigma=sigma, mode='reflect')
    except ImportError:
        # Fallback to optimized separable convolution
        return _separable_gaussian_blur(arr, sigma)


def _separable_gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Optimized separable Gaussian blur without scipy."""
    arr = np.asarray(arr, dtype=np.float64)
    
    rad = int(np.ceil(3.0 * sigma))
    x = np.arange(-rad, rad + 1)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    
    # Use numpy's optimized convolve
    # Pad and convolve along each axis
    
    # Along axis 1 (columns)
    padded = np.pad(arr, ((0, 0), (rad, rad)), mode='reflect')
    result = np.empty_like(arr)
    for i in range(arr.shape[0]):
        result[i, :] = np.convolve(padded[i, :], kernel, mode='valid')
    
    # Along axis 0 (rows)
    padded = np.pad(result, ((rad, rad), (0, 0)), mode='reflect')
    for j in range(arr.shape[1]):
        result[:, j] = np.convolve(padded[:, j], kernel, mode='valid')
    
    return result


# =============================================================================
# Optimized Box Blur (for detector sampling)
# =============================================================================

def fast_box_blur(arr: np.ndarray, wx: float, wy: float) -> np.ndarray:
    """
    Fast box blur using scipy.ndimage.uniform_filter.
    """
    if wx <= 1.0 and wy <= 1.0:
        return arr.copy() if isinstance(arr, np.ndarray) else np.asarray(arr)
    
    try:
        from scipy.ndimage import uniform_filter
        # uniform_filter uses size parameter
        size_y = max(1, int(round(wy)))
        size_x = max(1, int(round(wx)))
        return uniform_filter(arr.astype(np.float64), size=(size_y, size_x), mode='reflect')
    except ImportError:
        # Fallback to manual implementation
        return _manual_box_blur(arr, wx, wy)


def _manual_box_blur(arr: np.ndarray, wx: float, wy: float) -> np.ndarray:
    """Manual box blur using integral image for efficiency."""
    arr = np.asarray(arr, dtype=np.float64)
    
    # Use integral image for O(1) box blur per pixel
    kx = max(1, int(round(wx)))
    ky = max(1, int(round(wy)))
    
    # Pad array
    pad_x = kx // 2
    pad_y = ky // 2
    padded = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')
    
    # Compute integral image
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    
    # Compute box sums
    ny, nx = arr.shape
    result = np.empty_like(arr)
    
    for i in range(ny):
        for j in range(nx):
            y1, y2 = i, i + ky
            x1, x2 = j, j + kx
            
            total = integral[y2, x2]
            if y1 > 0:
                total -= integral[y1 - 1, x2]
            if x1 > 0:
                total -= integral[y2, x1 - 1]
            if y1 > 0 and x1 > 0:
                total += integral[y1 - 1, x1 - 1]
            
            result[i, j] = total / (kx * ky)
    
    return result


# =============================================================================
# Optimized Bilinear Interpolation
# =============================================================================

def fast_resample_bilinear(
    arr: np.ndarray,
    output_coords_y: np.ndarray,
    output_coords_x: np.ndarray
) -> np.ndarray:
    """
    Fast bilinear resampling using scipy.ndimage.map_coordinates.
    """
    try:
        from scipy.ndimage import map_coordinates
        
        coords = np.array([output_coords_y.ravel(), output_coords_x.ravel()])
        result = map_coordinates(arr.astype(np.float64), coords, order=1, mode='constant', cval=0.0)
        return result.reshape(output_coords_y.shape)
    except ImportError:
        # Fallback to manual bilinear
        return _manual_bilinear(arr, output_coords_y, output_coords_x)


def _manual_bilinear(
    arr: np.ndarray,
    iy_float: np.ndarray,
    ix_float: np.ndarray
) -> np.ndarray:
    """Manual bilinear interpolation (vectorized)."""
    arr = np.asarray(arr, dtype=np.float64)
    ny, nx = arr.shape
    
    i0 = np.floor(ix_float).astype(np.int32)
    j0 = np.floor(iy_float).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, nx - 1)
    j1 = np.clip(j0 + 1, 0, ny - 1)
    i0 = np.clip(i0, 0, nx - 1)
    j0 = np.clip(j0, 0, ny - 1)
    
    fx = ix_float - np.floor(ix_float)
    fy = iy_float - np.floor(iy_float)
    
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy
    
    result = (
        w00 * arr[j0, i0] +
        w10 * arr[j0, i1] +
        w01 * arr[j1, i0] +
        w11 * arr[j1, i1]
    )
    
    return result


# =============================================================================
# Numba JIT Compilation (optional)
# =============================================================================

_NUMBA_AVAILABLE = False

try:
    from numba import jit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    # Create dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(*args):
        return range(*args)


@jit(nopython=True, parallel=True, cache=False)
def _numba_zncc_single(template: np.ndarray, patch: np.ndarray) -> float:
    """Numba-accelerated single ZNCC computation."""
    t_flat = template.ravel()
    p_flat = patch.ravel()
    
    n = t_flat.shape[0]
    if n == 0:
        return -1.0
    
    t_mean = 0.0
    p_mean = 0.0
    for i in range(n):
        t_mean += t_flat[i]
        p_mean += p_flat[i]
    t_mean /= n
    p_mean /= n
    
    t_norm_sq = 0.0
    p_norm_sq = 0.0
    dot_prod = 0.0
    
    for i in range(n):
        t_centered = t_flat[i] - t_mean
        p_centered = p_flat[i] - p_mean
        t_norm_sq += t_centered * t_centered
        p_norm_sq += p_centered * p_centered
        dot_prod += t_centered * p_centered
    
    denom = np.sqrt(t_norm_sq * p_norm_sq)
    if denom < 1e-12:
        return -1.0
    
    return dot_prod / denom


@jit(nopython=True, parallel=True, cache=False)
def _numba_refine_centroids_batch(
    arr: np.ndarray,
    centers_x: np.ndarray,
    centers_y: np.ndarray,
    half: int,
    margin: int,
    gamma: float
) -> np.ndarray:
    """
    Numba-accelerated batch centroid refinement.
    
    Returns array of shape (N, 3) with (refined_x, refined_y, background).
    """
    n = len(centers_x)
    results = np.empty((n, 3), dtype=np.float64)
    ny, nx = arr.shape
    total_half = half + margin
    
    for idx in prange(n):
        cx = int(round(centers_x[idx]))
        cy = int(round(centers_y[idx]))
        
        x0 = max(0, cx - total_half)
        x1 = min(nx, cx + total_half + 1)
        y0 = max(0, cy - total_half)
        y1 = min(ny, cy + total_half + 1)
        
        # Extract patch
        patch = arr[y0:y1, x0:x1].copy()
        ph, pw = patch.shape
        
        inner_x0 = max(0, cx - half) - x0
        inner_y0 = max(0, cy - half) - y0
        inner_x1 = min(nx, cx + half + 1) - x0
        inner_y1 = min(ny, cy + half + 1) - y0
        
        # Compute background from ring
        bg_sum = 0.0
        bg_count = 0
        for i in range(ph):
            for j in range(pw):
                if not (inner_y0 <= i < inner_y1 and inner_x0 <= j < inner_x1):
                    bg_sum += patch[i, j]
                    bg_count += 1
        
        background = bg_sum / max(bg_count, 1)
        
        # Subtract background and apply gamma
        weight_sum = 0.0
        x_weighted = 0.0
        y_weighted = 0.0
        
        for i in range(inner_y0, inner_y1):
            for j in range(inner_x0, inner_x1):
                val = patch[i, j] - background
                if val < 0:
                    val = 0.0
                val = val ** gamma
                
                weight_sum += val
                x_weighted += val * (x0 + j)
                y_weighted += val * (y0 + i)
        
        if weight_sum > 0:
            results[idx, 0] = x_weighted / weight_sum
            results[idx, 1] = y_weighted / weight_sum
        else:
            results[idx, 0] = centers_x[idx]
            results[idx, 1] = centers_y[idx]
        results[idx, 2] = background
    
    return results


# =============================================================================
# Zernike Polynomial Optimization
# =============================================================================

# Pre-computed factorial table for Zernike calculations
_FACTORIAL_CACHE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880,
    3628800, 39916800, 479001600, 6227020800, 87178291200,
    1307674368000, 20922789888000, 355687428096000
], dtype=np.float64)


def factorial_cached(n: int) -> float:
    """Get factorial from cache (faster than math.factorial for small n)."""
    if n < 0:
        return 0.0
    if n < len(_FACTORIAL_CACHE):
        return _FACTORIAL_CACHE[n]
    # Fall back to numpy for large n
    return float(np.math.factorial(n))


# Pre-computed Noll index mapping
NOLL_TO_NM = {
    1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0),
    5: (2, -2), 6: (2, 2), 7: (3, -1), 8: (3, 1),
    9: (3, -3), 10: (3, 3), 11: (4, 0), 12: (4, 2),
    13: (4, -2), 14: (4, 4), 15: (4, -4),
}


def zernike_radial_fast(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """
    Fast Zernike radial polynomial using cached factorials.
    """
    m = abs(m)
    if (n - m) % 2 != 0:
        return np.zeros_like(rho)
    
    result = np.zeros_like(rho, dtype=np.float64)
    s_max = (n - m) // 2
    
    for s in range(s_max + 1):
        sign = (-1.0) ** s
        num = factorial_cached(n - s)
        den = factorial_cached(s) * factorial_cached((n + m) // 2 - s) * factorial_cached((n - m) // 2 - s)
        coef = sign * num / den
        result += coef * (rho ** (n - 2 * s))
    
    return result


def zernike_polynomial_fast(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Fast Zernike polynomial evaluation using Noll index.
    """
    if j not in NOLL_TO_NM:
        return np.zeros_like(rho)
    
    n, m = NOLL_TO_NM[j]
    R_nm = zernike_radial_fast(n, abs(m), rho)
    
    if m == 0:
        return R_nm
    elif m > 0:
        return R_nm * np.cos(m * theta)
    else:
        return R_nm * np.sin(abs(m) * theta)


# =============================================================================
# Mask Caching
# =============================================================================

class MaskCache:
    """
    Cache for Hartmann masks to avoid recomputation.
    
    Particularly useful when only translation changes between pulses.
    """
    _cache: Dict[Tuple, np.ndarray] = {}
    _max_size = 50
    
    @classmethod
    def get_key(cls, shift_x: float, shift_y: float, spacing: float, 
                aperture: float, rotation: float = 0.0) -> Tuple:
        """Generate cache key from parameters."""
        return (
            round(shift_x, 9),
            round(shift_y, 9),
            round(spacing, 9),
            round(aperture, 9),
            round(rotation, 6)
        )
    
    @classmethod
    def get(cls, key: Tuple) -> Optional[np.ndarray]:
        """Get cached mask if exists."""
        return cls._cache.get(key)
    
    @classmethod
    def put(cls, key: Tuple, mask: np.ndarray):
        """Store mask in cache."""
        if len(cls._cache) >= cls._max_size:
            # Remove oldest
            oldest = next(iter(cls._cache))
            del cls._cache[oldest]
        cls._cache[key] = mask
    
    @classmethod
    def clear(cls):
        """Clear all cached masks."""
        cls._cache.clear()


# =============================================================================
# Parallel Frame Accumulation
# =============================================================================

def fast_accumulate_frames(frames: list, dtype=np.float64) -> np.ndarray:
    """
    Fast frame accumulation using pre-allocated array.
    """
    if not frames:
        return None
    
    result = np.zeros_like(frames[0], dtype=dtype)
    for f in frames:
        np.add(result, f, out=result)
    return result


# =============================================================================
# Module Info
# =============================================================================

def print_optimization_info():
    """Print information about available optimizations."""
    print("=" * 60)
    print("Optimization Module Status")
    print("=" * 60)
    
    print(f"\nFFT Backend: {get_fft_backend()}")
    print(f"  - pyfftw: {'Available' if _PYFFTW_AVAILABLE else 'Not installed'}")
    print(f"  - scipy.fft: {'Available' if _SCIPY_FFT_AVAILABLE else 'Not installed'}")
    
    print(f"\nNumba JIT: {'Available' if _NUMBA_AVAILABLE else 'Not installed'}")
    
    try:
        from scipy.ndimage import gaussian_filter, uniform_filter, map_coordinates
        print(f"SciPy ndimage: Available (fast blur/resample)")
    except ImportError:
        print(f"SciPy ndimage: Not available (using fallback)")
    
    print(f"\nCaching:")
    print(f"  - Frequency grid cache: Active (max {CachedFrequencyGrid._max_cache_size} entries)")
    print(f"  - Mask cache: Active (max {MaskCache._max_size} entries)")
    
    print("=" * 60)


if __name__ == '__main__':
    print_optimization_info()
