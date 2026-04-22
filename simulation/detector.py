from __future__ import annotations


import numpy as np
from typing import Tuple, List, Optional, Union, Dict, Any, Sequence
import logging
from dataclasses import dataclass
import math
import warnings
from scipy.ndimage import uniform_filter, map_coordinates
from scipy.signal import fftconvolve
from simulation.blur import gaussian_blur

logger = logging.getLogger(__name__)
_POISSON_NORMAL_APPROX_THRESHOLD = 1e6



@dataclass
class PSFParams:
    type: str = 'gaussian'
    sigma_m: Optional[float] = None
    na: Optional[float] = None

@dataclass
class NoiseConfig:
    enable_poisson: bool = True
    gain_e_per_unit: float = 1.0
    enable_readout: bool = True
    readout_sigma_e: float = 0.0
    dark_current_e: float = 0.0
    fixed_bias_e: float = 0.0

@dataclass
class ADCConfig:
    bit_depth: int = 16
    full_well_e: Optional[float] = None
    gain_e_per_adu: float = 1.0

@dataclass
class BinningConfig:
    bin_size: Optional[int] = None


def _normalize_binning_policy(policy: Optional[str]) -> str:
    value = str(policy or 'trim').lower()
    if value not in {'trim', 'error', 'pad'}:
        raise ValueError(
            f"Unsupported binning policy {policy!r}; expected 'trim', 'error', or 'pad'."
        )
    return value

def _box_kernel_1d(width: float) -> np.ndarray:
    """
    （），。

    :
        width: ，“”（ pixel_size/dx  pixel_size/dy）

    :
        -  width <= 1 ， [1.0]，（）。
        -  width > 1 ， + ，，。
    """
    w = float(width)
    if w <= 1.0:
        return np.array([1.0], dtype=float)
    n_int = int(np.floor(w))
    frac = w - n_int
    k = np.concatenate(([0.5 * frac], np.ones(n_int, dtype=float), [0.5 * frac]))
    k /= np.sum(k)
    return k


def _box_blur_separable(arr: np.ndarray, wx: float, wy: float) -> np.ndarray:
    """
    使用 scipy.ndimage 优化的 box blur；对非整数宽度保留可分离卷积。

    :
        arr: （）
        wx: （：）
        wy: （：）

    :
        （）
    """
    arr = np.asarray(arr, dtype=np.float64)
    
    if wx >= 1.0 and wy >= 1.0:
        size_x = max(1, int(round(wx)))
        size_y = max(1, int(round(wy)))
        return uniform_filter(arr, size=(size_y, size_x), mode='reflect')
    
    # Fallback to separable convolution
    kx = _box_kernel_1d(wx)
    ky = _box_kernel_1d(wy)
    rad_x = len(kx) // 2
    rad_y = len(ky) // 2
    pad_x = ((0, 0), (rad_x, rad_x))
    arr_xpad = np.pad(arr, pad_x, mode='reflect')
    out_x = np.empty_like(arr, dtype=float)
    for i in range(arr.shape[0]):
        out_x[i, :] = np.convolve(arr_xpad[i, :], kx, mode='valid')
    pad_y = ((rad_y, rad_y), (0, 0))
    out_x_ypad = np.pad(out_x, pad_y, mode='reflect')
    out = np.empty_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        out[:, j] = np.convolve(out_x_ypad[:, j], ky, mode='valid')
    return out

def create_detector_region(detector_config,
                           center_x_m: float = 0.0,
                           center_y_m: float = 0.0,
                           return_grid: bool = False) -> Union[Tuple[int, int, float, float], 
                                                               Tuple[int, int, float, float, np.ndarray, np.ndarray]]:
    """
     - 。
    
    ，
    (binning)，。
    
    :
    ----
    detector_config : 
        ，、、
    center_x_m : float,  0.0
        X ()
    center_y_m : float,  0.0
        Y ()
    return_grid : bool,  False
        
    
    :
    ----
    Union[Tuple[int, int, float, float], Tuple[int, int, float, float, np.ndarray, np.ndarray]] :
        - nx, ny:  (binning)
        - dx, dy:  ()
        - return_grid=True: (nx, ny, dx, dy, X, Y) X,Y
    
    :
    --------
    >>>
    >>> config = {
    ...     'pixel_size': 12e-6,
    ...     'nx': 1024, 'ny': 1024,
    ...     'binning': 2
    ... }
    >>> 
    >>>
    >>> nx, ny, dx, dy = create_detector_region(config)
    >>> print(f": {nx}x{ny}, : {dx*1e6:.1f}x{dy*1e6:.1f} μm")
    >>> 
    >>>
    >>> nx, ny, dx, dy, X, Y = create_detector_region(config, return_grid=True)
    >>> print(f": X=[{X.min():.3f}, {X.max():.3f}] m")
    >>> print(f": Y=[{Y.min():.3f}, {Y.max():.3f}] m")
    
    :
    --------
    :
    - pixel_size:  ()
    - nx, ny: 
    - binning:  (，1)
    
    :
    --------
    1. :
       nx_eff = nx // binning
       ny_eff = ny // binning
    
    2. :
       dx_eff = pixel_size * binning
       dy_eff = pixel_size * binning
    
    3.  (return_grid=True):
       x = (i - nx_eff/2 + 0.5) * dx_eff + center_x_m
       y = (j - ny_eff/2 + 0.5) * dy_eff + center_y_m
    """
    pixel_size = detector_config.pixel_size
    nx_raw = detector_config.nx
    ny_raw = detector_config.ny
    
    binning = getattr(detector_config, 'binning', 1)
    binning_policy = _normalize_binning_policy(getattr(detector_config, 'binning_policy', 'trim'))
    if binning <= 0:
        raise ValueError(f"binning must be >= 1, got {binning}")
    if (nx_raw % binning) != 0 or (ny_raw % binning) != 0:
        if binning_policy == 'error':
            raise ValueError(
                f"Detector size {nx_raw}x{ny_raw} is not divisible by binning={binning} "
                "and binning_policy='error'."
            )
        if binning_policy == 'pad':
            warnings.warn(
                f"Detector size {nx_raw}x{ny_raw} is not divisible by binning={binning}; "
                "detector region will be padded to the next bin boundary.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"Detector size {nx_raw}x{ny_raw} is not divisible by binning={binning}; "
                "trailing pixels will be discarded.",
                RuntimeWarning,
                stacklevel=2,
            )

    if binning_policy == 'pad':
        nx = int(math.ceil(nx_raw / binning))
        ny = int(math.ceil(ny_raw / binning))
    else:
        nx = nx_raw // binning
        ny = ny_raw // binning
    dx = pixel_size * binning
    dy = pixel_size * binning
    
    if not return_grid:
        return nx, ny, dx, dy
    
    x = np.arange(nx) - nx/2 + 0.5
    y = np.arange(ny) - ny/2 + 0.5
    
    x = x * dx + center_x_m
    y = y * dy + center_y_m
    
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    return nx, ny, dx, dy, X, Y


def sample_detector(intensity: np.ndarray,
                   dx: float,
                   dy: float,
                   pixel_size: float = 12e-6,
                   detector_nx: Optional[int] = None,
                   detector_ny: Optional[int] = None,
                   center_x_m: float = 0.0,
                   center_y_m: float = 0.0,
                   mode: str = 'nearest') -> np.ndarray:
    """
    ：。

    ，，
    ，
    ``sample_detector_fov``  / 。

    ：，
    。
    """
    ny_in, nx_in = intensity.shape
    size_x = nx_in * dx
    size_y = ny_in * dy

    if detector_nx is None:
        detector_nx = max(1, int(round(size_x / pixel_size)))
    if detector_ny is None:
        detector_ny = max(1, int(round(size_y / pixel_size)))

    return sample_detector_fov(
        intensity,
        dx,
        dy,
        pixel_size=pixel_size,
        detector_nx=detector_nx,
        detector_ny=detector_ny,
        center_x_m=center_x_m,
        center_y_m=center_y_m,
        mode=mode,
    )


def sample_detector_fov(intensity: np.ndarray,
                        dx: float,
                        dy: float,
                        pixel_size: float,
                        detector_nx: int,
                        detector_ny: int,
                        center_x_m: float = 0.0,
                        center_y_m: float = 0.0,
                        mode: str = 'nearest',
                        upsample_factor: int = 1,
                        bin_size: Optional[int] = None,
                        interpolation: str = 'bilinear',
                        binning_policy: str = 'trim') -> np.ndarray:
    """
    （/）

    ：
    -  (detector_nx, detector_ny, pixel_size) ；
    -  (shape, dx, dy) ；
    - ：“”（）；
    - ：“”（0）；
    - /，；
    - ：
        * 'nearest' —— （）；
        * 'area'    —— ；
        * 'srw'     —— SRW：，
                       “→binning”，。

    :
        intensity: ， (ny_in, nx_in)
        dx, dy: （）
        pixel_size: （）
        detector_nx, detector_ny: 
        center_x_m, center_y_m: （0）
        mode: ，'nearest'、'area'  'srw'
        upsample_factor: 'srw' （>=1）
        bin_size: 'srw' binning（ upsample_factor）
        interpolation: 'srw' （'bilinear'  'nearest'）

    :
        detector_image:  (detector_ny, detector_nx) 
    """
    ny_in, nx_in = intensity.shape
    width_in = nx_in * dx
    height_in = ny_in * dy

    x_det = (np.arange(detector_nx) - detector_nx / 2 + 0.5) * pixel_size + center_x_m
    y_det = (np.arange(detector_ny) - detector_ny / 2 + 0.5) * pixel_size + center_y_m

    x_min_in = -width_in / 2.0
    x_max_in = width_in / 2.0
    y_min_in = -height_in / 2.0
    y_max_in = height_in / 2.0

    mode_l = str(mode).lower()
    if mode_l == 'srw':
        if upsample_factor is None or upsample_factor < 1:
            upsample_factor = 1
        if bin_size is None:
            bin_size = upsample_factor

        fine_pixel = pixel_size / float(upsample_factor)
        nx_fine = int(detector_nx * upsample_factor)
        ny_fine = int(detector_ny * upsample_factor)

        D_up = _resample_to_detector(
            intensity,
            dx, dy,
            pixel_size=fine_pixel,
            detector_nx=nx_fine,
            detector_ny=ny_fine,
            center_x_m=center_x_m,
            center_y_m=center_y_m,
            interpolation=interpolation,
        )

        D = _binning(D_up, bin_size, policy=binning_policy)
        return D

    if mode_l == 'area':
        wx = float(pixel_size) / float(dx)
        wy = float(pixel_size) / float(dy)
        intensity = _box_blur_separable(intensity, wx, wy)

    ix_float = x_det / dx + (nx_in / 2.0) - 0.5
    iy_float = y_det / dy + (ny_in / 2.0) - 0.5
    ix = np.round(ix_float).astype(int)
    iy = np.round(iy_float).astype(int)

    valid_x = (x_det >= x_min_in) & (x_det < x_max_in)
    valid_y = (y_det >= y_min_in) & (y_det < y_max_in)

    ix_clipped = np.clip(ix, 0, nx_in - 1)
    iy_clipped = np.clip(iy, 0, ny_in - 1)

    IX, IY = np.meshgrid(ix_clipped, iy_clipped)

    detector_image = intensity[IY, IX].astype(float)
    valid_2d = np.outer(valid_y.astype(bool), valid_x.astype(bool))
    detector_image[~valid_2d] = 0.0

    return detector_image


def _resample_to_detector(intensity: np.ndarray,
                          dx: float,
                          dy: float,
                          pixel_size: float,
                          detector_nx: int,
                          detector_ny: int,
                          center_x_m: float = 0.0,
                          center_y_m: float = 0.0,
                          interpolation: str = 'bilinear') -> np.ndarray:
    """
    使用scipy.ndimage优化的重采样函数。

    -  'bilinear'  'nearest' ；
    - ，；
    -  0。
    """
    ny_in, nx_in = intensity.shape
    width_in = nx_in * dx
    height_in = ny_in * dy

    x_det = (np.arange(detector_nx) - detector_nx / 2 + 0.5) * pixel_size + center_x_m
    y_det = (np.arange(detector_ny) - detector_ny / 2 + 0.5) * pixel_size + center_y_m

    x_min_in = -width_in / 2.0
    x_max_in = width_in / 2.0
    y_min_in = -height_in / 2.0
    y_max_in = height_in / 2.0

    ix_float = x_det / dx + (nx_in / 2.0) - 0.5
    iy_float = y_det / dy + (ny_in / 2.0) - 0.5

    IXf, IYf = np.meshgrid(ix_float, iy_float)

    valid_x = (x_det >= x_min_in) & (x_det < x_max_in)
    valid_y = (y_det >= y_min_in) & (y_det < y_max_in)
    valid_2d = np.outer(valid_y.astype(bool), valid_x.astype(bool))

    order = 0 if str(interpolation).lower() == 'nearest' else 1
    coords = np.array([IYf.ravel(), IXf.ravel()])
    out = map_coordinates(
        intensity.astype(np.float64),
        coords,
        order=order,
        mode='grid-constant',
        cval=0.0,
    ).reshape(IYf.shape)

    out[~valid_2d] = 0.0
    return out


def _binning(img: np.ndarray, bin_size: int, policy: str = 'trim') -> np.ndarray:
    if bin_size <= 1:
        return img
    binning_policy = _normalize_binning_policy(policy)
    ny, nx = img.shape
    rem_y = ny % bin_size
    rem_x = nx % bin_size
    working = np.asarray(img, dtype=np.float64)

    if rem_y != 0 or rem_x != 0:
        if binning_policy == 'error':
            raise ValueError(
                f"Image shape {ny}x{nx} is not divisible by bin_size={bin_size} "
                "and binning policy is 'error'."
            )
        if binning_policy == 'pad':
            pad_y = (bin_size - rem_y) % bin_size
            pad_x = (bin_size - rem_x) % bin_size
            warnings.warn(
                f"Image shape {ny}x{nx} is not divisible by bin_size={bin_size}; padding with zeros before binning.",
                RuntimeWarning,
                stacklevel=2,
            )
            working = np.pad(working, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0.0)
        else:
            warnings.warn(
                f"Image shape {ny}x{nx} is not divisible by bin_size={bin_size}; trimming remainder pixels before binning.",
                RuntimeWarning,
                stacklevel=2,
            )
            working = working[:ny - rem_y, :nx - rem_x]

    ny_w, nx_w = working.shape
    ny_b = ny_w // bin_size
    nx_b = nx_w // bin_size
    if ny_b <= 0 or nx_b <= 0:
        return working[:max(ny_b * bin_size, 0), :max(nx_b * bin_size, 0)]
    out = working.reshape(ny_b, bin_size, nx_b, bin_size).mean(axis=(1, 3))
    return out


def apply_optical_psf(intensity: np.ndarray, dx: float, dy: float, psf: Optional[PSFParams] = None) -> np.ndarray:
    arr = np.asarray(intensity, dtype=np.float64)
    if psf is None:
        return arr
    t = str(getattr(psf, 'type', 'gaussian') or 'gaussian').lower()
    ny, nx = arr.shape
    y = (np.arange(ny) - ny/2) * dy
    x = (np.arange(nx) - nx/2) * dx
    X, Y = np.meshgrid(x, y, indexing='xy')
    if t == 'gaussian':
        sigma = float(getattr(psf, 'sigma_m', 0.0) or 0.0)
        if sigma <= 0.0:
            return arr
        K = np.exp(-(X*X + Y*Y) / (2.0 * sigma * sigma))
    else:
        return arr
    s = float(np.sum(K))
    if s > 0:
        K = K / s
    return fftconvolve(arr, K, mode='same')


def apply_detector_noise(electrons: np.ndarray, noise: Optional[NoiseConfig] = None) -> np.ndarray:
    arr = np.asarray(electrons, dtype=np.float64)
    if noise is None:
        return arr
    out = arr.copy()
    out = out + float(getattr(noise, 'fixed_bias_e', 0.0) or 0.0)
    out = out + float(getattr(noise, 'dark_current_e', 0.0) or 0.0)
    if bool(getattr(noise, 'enable_poisson', True)):
        gain = float(getattr(noise, 'gain_e_per_unit', 1.0) or 1.0)
        lam = np.maximum(out * gain, 0.0)
        if np.any(lam > _POISSON_NORMAL_APPROX_THRESHOLD):
            warnings.warn(
                "Poisson lambda exceeds the safe threshold; using normal approximation for high-count pixels.",
                RuntimeWarning,
                stacklevel=2,
            )
            poisson_samples = np.empty_like(lam, dtype=np.float64)
            use_poisson = lam <= _POISSON_NORMAL_APPROX_THRESHOLD
            if np.any(use_poisson):
                poisson_samples[use_poisson] = np.random.poisson(lam[use_poisson]).astype(np.float64)
            if np.any(~use_poisson):
                high_lam = lam[~use_poisson]
                approx = np.random.normal(loc=high_lam, scale=np.sqrt(high_lam))
                poisson_samples[~use_poisson] = np.maximum(approx, 0.0)
            out = poisson_samples
        else:
            out = np.random.poisson(lam).astype(np.float64)
        out = out / gain
    if bool(getattr(noise, 'enable_readout', True)):
        sigma = float(getattr(noise, 'readout_sigma_e', 0.0) or 0.0)
        if sigma > 0.0:
            out = out + np.random.normal(0.0, sigma, size=out.shape)
    return out


def quantize_and_saturate(electrons: np.ndarray, adc: Optional[ADCConfig] = None) -> np.ndarray:
    arr = np.asarray(electrons, dtype=np.float64)
    if adc is None:
        return arr
    full_well = getattr(adc, 'full_well_e', None)
    if full_well is not None:
        arr = np.clip(arr, 0.0, float(full_well))
    gain = float(getattr(adc, 'gain_e_per_adu', 1.0) or 1.0)
    adu = arr / gain
    max_adu = float((1 << int(getattr(adc, 'bit_depth', 16))) - 1)
    adu = np.clip(adu, 0.0, max_adu)
    return np.round(adu).astype(np.float64)


def _gaussian_blur_separable_px(arr: np.ndarray, sigma_px: float) -> np.ndarray:
    return gaussian_blur(arr, sigma_px)


def realistic_sample(intensity: np.ndarray,
                     dx: float,
                     dy: float,
                     detector_pixel_size: float,
                     detector_nx: int,
                     detector_ny: int,
                     sigma_pixels: float = 2.0,
                     mode: str = 'nearest') -> np.ndarray:
    D = sample_detector_fov(
        intensity,
        dx,
        dy,
        pixel_size=detector_pixel_size,
        detector_nx=detector_nx,
        detector_ny=detector_ny,
        mode=mode,
    )
    if float(sigma_pixels) > 0:
        D = _gaussian_blur_separable_px(D, float(sigma_pixels))
    return D


def compute_pulses_per_frame(pulse_rep_rate: float, detector_fps: float) -> int:
    return max(1, int(pulse_rep_rate / detector_fps))


def accumulate_first_N_frames(frames: List[np.ndarray], N: int) -> Optional[np.ndarray]:
    if not frames or N <= 0:
        return None
    
    N = min(N, len(frames))
    result = frames[0].copy()
    for i in range(1, N):
        result += frames[i]
    
    return result


def determine_pulses_per_frame(pulses_per_frame: Optional[int],
                               pulse_rep_rate: Optional[float],
                               detector_fps: Optional[float]) -> int:
    if pulses_per_frame is not None:
        return pulses_per_frame
    elif pulse_rep_rate is not None and detector_fps is not None:
        return compute_pulses_per_frame(pulse_rep_rate, detector_fps)
    else:
        return 1


def accumulate_by_pulses(per_pulse_frames: List[np.ndarray],
                         pulses_per_frame: int,
                         wrap: bool = True) -> Optional[np.ndarray]:
    if not per_pulse_frames or pulses_per_frame <= 0:
        return None
    
    n_frames = len(per_pulse_frames)
    n_accumulated = n_frames // pulses_per_frame
    
    if n_accumulated == 0:
        return None
    
    accumulated = []
    for i in range(n_accumulated):
        frame_sum = per_pulse_frames[i * pulses_per_frame].copy()
        for j in range(1, pulses_per_frame):
            frame_sum += per_pulse_frames[i * pulses_per_frame + j]
        accumulated.append(frame_sum)
    
    return accumulated[0] if len(accumulated) == 1 else accumulated


def integrate_by_rates(per_pulse_frames: List[np.ndarray], 
                      pulse_rep_rate: float, 
                      detector_fps: float) -> Optional[np.ndarray]:
    pulses_per_frame = compute_pulses_per_frame(pulse_rep_rate, detector_fps)
    return accumulate_by_pulses(per_pulse_frames, pulses_per_frame)


def group_pulses_fixed(per_pulse_frames: List[np.ndarray], pulses_per_group: int) -> List[np.ndarray]:
    if not per_pulse_frames or pulses_per_group <= 0:
        return []
    
    grouped = []
    n_groups = len(per_pulse_frames) // pulses_per_group
    
    for i in range(n_groups):
        group_sum = per_pulse_frames[i * pulses_per_group].copy()
        for j in range(1, pulses_per_group):
            group_sum += per_pulse_frames[i * pulses_per_group + j]
        grouped.append(group_sum)
    
    return grouped




def _otsu_threshold(img: np.ndarray) -> float:
    """Compute Otsu threshold on normalized [0,1] image."""
    arr = np.asarray(img, dtype=np.float64)
    m = np.max(arr)
    if m <= 0:
        return 0.0
    arr = arr / m
    hist, edges = np.histogram(arr.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = arr.size
    prob = hist / total
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    weight1 = np.cumsum(prob)
    weight2 = np.cumsum(prob[::-1])[::-1]
    mean = np.cumsum(prob * bin_centers)
    mean2 = (np.cumsum((prob * bin_centers)[::-1])[::-1])
    inter_class_var = weight1[:-1] * weight2[1:] * (mean[:-1]/(weight1[:-1]+1e-12) - mean2[1:]/(weight2[1:]+1e-12))**2
    k = int(np.argmax(inter_class_var))
    return float(bin_centers[k]) * float(m)


def _compute_threshold(img: np.ndarray, cfg: Optional[Any]) -> float:
    """Determine threshold based on CentroidConfig if provided."""
    arr = np.asarray(img, dtype=np.float64)
    vmax = float(np.max(arr))
    if vmax <= 0:
        return 0.0
    if cfg is not None and hasattr(cfg, 'threshold_type'):
        ttype = str(getattr(cfg, 'threshold_type', 'relative'))
        tval = float(getattr(cfg, 'threshold_value', 0.5))
        if ttype == 'absolute':
            return tval
        if ttype == 'otsu':
            return _otsu_threshold(arr)
        # default relative
        return vmax * tval
    return vmax * 0.5


def _label_components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Label 8-connected components and return pixel coordinate lists per component."""
    Ny, Nx = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for y in range(Ny):
        for x in range(Nx):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            comp: List[Tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny = cy + dy
                        nx = cx + dx
                        if 0 <= ny < Ny and 0 <= nx < Nx:
                            if mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
            comps.append(comp)
    return comps


def _shift_bool(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    Ny, Nx = arr.shape
    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    p = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=False)
    y0 = pad_top - dy
    x0 = pad_left - dx
    return p[y0:y0+Ny, x0:x0+Nx]


def _binary_dilate(mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return mask.copy()
    r = ksize // 2
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            out |= _shift_bool(mask, dy, dx)
    return out


def _binary_erode(mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return mask.copy()
    r = ksize // 2
    out = np.ones_like(mask, dtype=bool)
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            out &= _shift_bool(mask, dy, dx)
    return out


def binary_morph_open_close(mask: np.ndarray, open_ksize: int, close_ksize: int) -> np.ndarray:
    m = mask.astype(bool)
    if int(open_ksize) > 1:
        m = _binary_dilate(_binary_erode(m, int(open_ksize)), int(open_ksize))
    if int(close_ksize) > 1:
        m = _binary_erode(_binary_dilate(m, int(close_ksize)), int(close_ksize))
    return m


def _component_bbox(coords: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    ys = np.array([p[0] for p in coords], dtype=int)
    xs = np.array([p[1] for p in coords], dtype=int)
    miny = int(np.min(ys))
    maxy = int(np.max(ys))
    minx = int(np.min(xs))
    maxx = int(np.max(xs))
    return minx, miny, maxx, maxy


def compute_component_quality(img: np.ndarray, coords: List[Tuple[int, int]]) -> Dict[str, float]:
    ys = np.array([p[0] for p in coords], dtype=int)
    xs = np.array([p[1] for p in coords], dtype=int)
    vals = img[ys, xs]
    peak = float(np.max(vals))
    sI = float(np.sum(vals))
    area = int(len(coords))
    minx, miny, maxx, maxy = _component_bbox(coords)
    Ny, Nx = img.shape
    pad = 2
    x0 = max(0, minx - pad)
    y0 = max(0, miny - pad)
    x1 = min(Nx - 1, maxx + pad)
    y1 = min(Ny - 1, maxy + pad)
    ring = img[y0:y1+1, x0:x1+1].copy()
    ring_mask = np.zeros_like(ring, dtype=bool)
    ring_mask[(ys - y0, xs - x0)] = True
    bg = ring[~ring_mask]
    if bg.size <= 0:
        bg_mean = 0.0
        bg_std = 1e-9
    else:
        bg_mean = float(np.mean(bg))
        bg_std = float(np.std(bg)) + 1e-9
    bbox_area = float((maxx - minx + 1) * (maxy - miny + 1))
    compact = float(area) / max(bbox_area, 1.0)
    vmax = float(np.max(img)) + 1e-9
    contrast = (peak - bg_mean) / (peak + bg_mean + 1e-9)
    snr = (peak - bg_mean) / bg_std
    return {
        'sum_intensity': sI,
        'area': float(area),
        'peak': peak,
        'bg_mean': bg_mean,
        'bg_std': bg_std,
        'compactness': compact,
        'contrast': float(contrast),
        'snr': float(snr),
    }


def non_maximum_suppression_points(points: List[Dict[str, Any]], radius_px: float) -> List[Dict[str, Any]]:
    if radius_px is None or float(radius_px) <= 0.0:
        return points
    pts = sorted(points, key=lambda d: float(d.get('sum_intensity', 0.0)), reverse=True)
    kept: List[Dict[str, Any]] = []
    r2 = float(radius_px) * float(radius_px)
    for p in pts:
        x = float(p['x_px'])
        y = float(p['y_px'])
        ok = True
        for q in kept:
            dx = x - float(q['x_px'])
            dy = y - float(q['y_px'])
            if dx * dx + dy * dy <= r2:
                ok = False
                break
        if ok:
            kept.append(p)
    return kept


def dog_response(img: np.ndarray, s1: float, s2: float) -> np.ndarray:
    a = gaussian_blur(img, float(s1))
    b = gaussian_blur(img, float(s2))
    return a - b


def find_local_maxima(resp: np.ndarray, min_val: float, nms_radius_px: float) -> List[Tuple[float, float, float]]:
    r = np.asarray(resp, dtype=np.float64)
    m = r >= float(min_val)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            m &= r >= _shift_bool(r, dy, dx)
    ys, xs = np.nonzero(m)
    seeds: List[Tuple[float, float, float]] = [(float(xs[i]), float(ys[i]), float(r[ys[i], xs[i]])) for i in range(len(xs))]
    seeds.sort(key=lambda t: t[2], reverse=True)
    if nms_radius_px is None or float(nms_radius_px) <= 0.0:
        return seeds
    kept: List[Tuple[float, float, float]] = []
    r2 = float(nms_radius_px) * float(nms_radius_px)
    for x, y, s in seeds:
        ok = True
        for qx, qy, _ in kept:
            dx = x - qx
            dy = y - qy
            if dx * dx + dy * dy <= r2:
                ok = False
                break
        if ok:
            kept.append((x, y, s))
    return kept


def parabolic_subpixel(win: np.ndarray, peak_y: int, peak_x: int) -> Tuple[float, float]:
    Ny, Nx = win.shape
    y = int(peak_y)
    x = int(peak_x)
    if y <= 0 or y >= Ny - 1 or x <= 0 or x >= Nx - 1:
        return float(x), float(y)
    zxm1 = float(win[y, x - 1])
    zx = float(win[y, x])
    zxp1 = float(win[y, x + 1])
    zym1 = float(win[y - 1, x])
    zy = float(win[y, x])
    zyp1 = float(win[y + 1, x])
    
    denom_x = zxm1 - 2.0 * zx + zxp1
    if abs(denom_x) < 1e-9:
        dx = 0.0
    else:
        dx = 0.5 * (zxm1 - zxp1) / denom_x

    denom_y = zym1 - 2.0 * zy + zyp1
    if abs(denom_y) < 1e-9:
        dy = 0.0
    else:
        dy = 0.5 * (zym1 - zyp1) / denom_y
        
    # Limit shift to +/- 1.0 to avoid wild excursions
    dx = float(np.clip(dx, -1.0, 1.0))
    dy = float(np.clip(dy, -1.0, 1.0))
    
    return float(x + dx), float(y + dy)


def frst_response(img: np.ndarray, r_list: Sequence[int], alpha: float, beta: float) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    Ny, Nx = arr.shape
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) * 0.5
    gy[1:-1, :] = (arr[2:, :] - arr[:-2, :]) * 0.5
    mag = np.sqrt(gx * gx + gy * gy) + 1e-12
    ux = gx / mag
    uy = gy / mag
    resp = np.zeros_like(arr)
    for r in r_list:
        tx = np.clip(np.round(np.arange(Nx)[None, :] - r * ux), 0, Nx - 1).astype(int)
        ty = np.clip(np.round(np.arange(Ny)[:, None] - r * uy), 0, Ny - 1).astype(int)
        acc = np.zeros_like(arr)
        np.add.at(acc, (ty, tx), mag)
        resp += acc / (1.0 + alpha)
    if beta is not None and beta > 0:
        resp = gaussian_blur(resp, float(beta))
    return resp












def analyze_spot_resolution(intensity_roi: np.ndarray,
                            pixel_size: float,
                            method: str = 'fwhm') -> dict:
    """
     - ROI。
    
    ，FWHM、RMS。
    。
    
    :
    ----
    intensity_roi : np.ndarray
        ROI，2D
    pixel_size : float
         ()
    method : str,  'fwhm'
        ：'fwhm'()、'rms'(RMS)、'encircled_energy'()
    
    :
    ----
    dict : 
        
    
    :
    --------
    >>> roi = np.random.rand(50, 50)
    >>> pixel_size = 12e-6
    >>> 
    >>>
    >>> result = analyze_spot_resolution(roi, pixel_size, 'fwhm')
    >>> print(f"XFWHM: {result['fwhm_x']*1e6:.1f} μm")
    >>> print(f"YFWHM: {result['fwhm_y']*1e6:.1f} μm")
    >>> 
    >>>
    >>> result = analyze_spot_resolution(roi, pixel_size, 'rms')
    >>> print(f"RMS: {result['rms_width']*1e6:.1f} μm")
    
    :
    -----------
    method，：
    - 'fwhm': fwhm_x, fwhm_y, fwhm_avg, centroid_x, centroid_y
    - 'rms': rms_width, rms_x, rms_y, centroid_x, centroid_y
    - 'encircled_energy': ee_50, ee_80, ee_90, centroid_x, centroid_y
    """
    cx = intensity_roi.shape[1] / 2.0
    cy = intensity_roi.shape[0] / 2.0
    
    result = {
        'method': method,
        'centroid_x': cx * pixel_size,
        'centroid_y': cy * pixel_size,
        'pixel_size': pixel_size
    }
    
    if method == 'fwhm':
        ny, nx = intensity_roi.shape
        
        cy_int = int(round(cy))
        cy_int = max(0, min(ny-1, cy_int))
        x_profile = intensity_roi[cy_int, :]
        
        cx_int = int(round(cx))
        cx_int = max(0, min(nx-1, cx_int))
        y_profile = intensity_roi[:, cx_int]
        
        def compute_fwhm(profile):
            max_val = np.max(profile)
            half_max = max_val / 2.0
            
            above_half = profile >= half_max
            if not np.any(above_half):
                return 0.0
            
            indices = np.where(above_half)[0]
            if len(indices) < 2:
                return 0.0
            
            return (indices[-1] - indices[0]) * pixel_size
        
        fwhm_x = compute_fwhm(x_profile)
        fwhm_y = compute_fwhm(y_profile)
        fwhm_avg = (fwhm_x + fwhm_y) / 2.0
        
        result.update({
            'fwhm_x': fwhm_x,
            'fwhm_y': fwhm_y,
            'fwhm_avg': fwhm_avg
        })
    
    elif method == 'rms':
        ny, nx = intensity_roi.shape
        y_coords, x_coords = np.mgrid[0:ny, 0:nx]
        
        x_phys = (x_coords - cx) * pixel_size
        y_phys = (y_coords - cy) * pixel_size
        
        total_intensity = np.sum(intensity_roi)
        if total_intensity > 0:
            rms_x = np.sqrt(np.sum(intensity_roi * x_phys**2) / total_intensity)
            rms_y = np.sqrt(np.sum(intensity_roi * y_phys**2) / total_intensity)
            rms_width = np.sqrt(rms_x**2 + rms_y**2)
        else:
            rms_x = rms_y = rms_width = 0.0
        
        result.update({
            'rms_width': rms_width,
            'rms_x': rms_x,
            'rms_y': rms_y
        })
    
    elif method == 'encircled_energy':
        ny, nx = intensity_roi.shape
        y_coords, x_coords = np.mgrid[0:ny, 0:nx]
        
        distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2) * pixel_size
        
        flat_distances = distances.flatten()
        flat_intensity = intensity_roi.flatten()
        
        sorted_indices = np.argsort(flat_distances)
        sorted_distances = flat_distances[sorted_indices]
        sorted_intensity = flat_intensity[sorted_indices]
        
        cumulative_energy = np.cumsum(sorted_intensity)
        total_energy = cumulative_energy[-1]
        
        if total_energy > 0:
            energy_fraction = cumulative_energy / total_energy
            
            def find_radius(fraction):
                idx = np.searchsorted(energy_fraction, fraction)
                if idx < len(sorted_distances):
                    return sorted_distances[idx]
                return sorted_distances[-1]
            
            ee_50 = find_radius(0.5)
            ee_80 = find_radius(0.8)
            ee_90 = find_radius(0.9)
        else:
            ee_50 = ee_80 = ee_90 = 0.0
        
        result.update({
            'ee_50': ee_50,
            'ee_80': ee_80,
            'ee_90': ee_90
        })
    
    return result


def batch_resolution_analysis(detector_image: np.ndarray,
                              roi_list: list,
                              pixel_size: float,
                              method: str = 'fwhm') -> list:
    """
     - ROI。
    
    ROI，
    。
    
    :
    ----
    detector_image : np.ndarray
        ，2D
    roi_list : list
        ROI，ROI(y_min, y_max, x_min, x_max)
    pixel_size : float
         ()
    method : str,  'fwhm'
        ：'fwhm'、'rms'、'encircled_energy'
    
    :
    ----
    list : 
        ROI
    
    :
    --------
    >>> rois = [(10, 30, 15, 35), (50, 70, 55, 75)]
    >>> results = batch_resolution_analysis(image, rois, 12e-6, 'fwhm')
    >>> 
    >>> for i, result in enumerate(results):
    ...     print(f"ROI {i}: FWHM = {result['fwhm_avg']*1e6:.1f} μm")
    """
    results = []
    
    for idx, (y0, y1, x0, x1) in enumerate(roi_list):
        roi = detector_image[y0:y1, x0:x1]
        res = analyze_spot_resolution(roi, pixel_size, method=method)
        res['roi_index'] = idx
        results.append(res)
    
    return results
def compute_displacements_xcorr(
    det_ref: np.ndarray,
    det_sim: np.ndarray,
    centers: List[Dict[str, Any]],
    window_radius_px: int = 15,
    search_radius_px: int = 10,
    min_score: float = 0.2,
    pixel_size_m: Optional[float] = None,
    subpixel: bool = False,
    *,
    template_radius_px: Optional[int] = None,
    template_margin_px: int = 2,
    core_radius_px: Optional[float] = None,
    core_radius_fraction: Optional[float] = None,
) -> List[Dict[str, Any]]:
    img_ref = np.asarray(det_ref, dtype=np.float64)
    img_sim = np.asarray(det_sim, dtype=np.float64)
    ny, nx = img_ref.shape
    r_tpl = int(template_radius_px) if template_radius_px is not None else int(window_radius_px)
    r_tpl = max(1, r_tpl)
    r = r_tpl + int(max(0, template_margin_px))
    R = int(search_radius_px)

    # Optional core mask to suppress edge influence; built once and reused.
    mask_core = None
    if core_radius_px is not None or core_radius_fraction is not None:
        rc = None
        if core_radius_px is not None:
            rc = float(core_radius_px)
        elif core_radius_fraction is not None:
            rc = float(core_radius_fraction) * float(r_tpl)
        if rc is not None and rc > 0:
            rc = min(rc, float(r))
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            mask_core = (xx * xx + yy * yy) <= (rc * rc)
            if not np.any(mask_core):
                mask_core = None

    def zncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
        if mask is not None:
            m = mask.astype(bool)
            if not np.any(m):
                return -1.0
            avals = np.asarray(a, dtype=np.float64)[m]
            bvals = np.asarray(b, dtype=np.float64)[m]
            if avals.size == 0 or bvals.size == 0:
                return -1.0
            a0 = avals - float(np.mean(avals))
            b0 = bvals - float(np.mean(bvals))
            da = float(np.linalg.norm(a0))
            db = float(np.linalg.norm(b0))
            if da <= 1e-12 or db <= 1e-12:
                return -1.0
            return float(np.sum(a0 * b0) / (da * db))
        a0 = a - float(np.mean(a))
        b0 = b - float(np.mean(b))
        da = float(np.sqrt(np.sum(a0 * a0)))
        db = float(np.sqrt(np.sum(b0 * b0)))
        if da <= 1e-12 or db <= 1e-12:
            return -1.0
        return float(np.sum(a0 * b0) / (da * db))

    out: List[Dict[str, Any]] = []
    for i, c in enumerate(centers):
        x0 = int(round(float(c.get('x_px', 0.0))))
        y0 = int(round(float(c.get('y_px', 0.0))))
        x1 = x0 - r
        x2 = x0 + r
        y1 = y0 - r
        y2 = y0 + r
        if x1 < 0 or y1 < 0 or x2 >= nx or y2 >= ny:
            out.append({
                'idx': i,
                'x_ref_px': float(x0),
                'y_ref_px': float(y0),
                'status': 'out_of_bounds'
            })
            continue
        pref = img_ref[y1:y2+1, x1:x2+1]
        # normalize once per spot to ensure mean-removed template
        best_s = -1.0
        best_dx = 0
        best_dy = 0
        best_x = x0
        best_y = y0
        scores_map = None
        if subpixel:
            scores_map = np.full((2*R+1, 2*R+1), -1.0, dtype=np.float64)
        for dy in range(-R, R+1):
            yc = y0 + dy
            ys1 = yc - r
            ys2 = yc + r
            if ys1 < 0 or ys2 >= ny:
                continue
            for dx in range(-R, R+1):
                xc = x0 + dx
                xs1 = xc - r
                xs2 = xc + r
                if xs1 < 0 or xs2 >= nx:
                    continue
                psim = img_sim[ys1:ys2+1, xs1:xs2+1]
                s = zncc(pref, psim, mask_core)
                if subpixel:
                    scores_map[dy+R, dx+R] = s
                if s > best_s:
                    best_s = s
                    best_dx = dx
                    best_dy = dy
                    best_x = xc
                    best_y = yc
        status = 'ok' if best_s >= float(min_score) and best_s >= 0 else 'low_score'
        dx_px = float(best_dx)
        dy_px = float(best_dy)
        dx_m = None if pixel_size_m is None else dx_px * float(pixel_size_m)
        dy_m = None if pixel_size_m is None else dy_px * float(pixel_size_m)
        if subpixel and scores_map is not None:
            cy = best_dy + R
            cx = best_dx + R
            # Use the robust parabolic_subpixel function
            px, py = parabolic_subpixel(scores_map, cy, cx)
            # px, py are indices in scores_map. 
            # Center of scores_map is at (R, R) which corresponds to (best_dx, best_dy) relative to (0,0)
            # Actually, scores_map indices: 0..2R. Center R.
            # best_dx corresponds to index cx = best_dx + R.
            # The function returns absolute index in the window.
            # So shift = px - cx.
            
            off_x = px - float(cx)
            off_y = py - float(cy)
            
            dx_px += off_x
            dy_px += off_y
            dx_m = None if pixel_size_m is None else dx_px * float(pixel_size_m)
            dy_m = None if pixel_size_m is None else dy_px * float(pixel_size_m)
        out.append({
            'idx': i,
            'x_ref_px': float(x0),
            'y_ref_px': float(y0),
            'x_sim_px': float(best_x),
            'y_sim_px': float(best_y),
            'dx_px': dx_px,
            'dy_px': dy_px,
            'dx_m': dx_m,
            'dy_m': dy_m,
            'score': float(best_s),
            'status': status,
        })
    return out

def _box_filter_2d(img: np.ndarray, ksize: int) -> np.ndarray:
    """Fast box blur using integral image (ksize must be odd and >= 1)."""
    if ksize is None or ksize <= 1:
        return np.asarray(img, dtype=np.float64)
    k = int(max(1, ksize))
    if k % 2 == 0:
        k += 1
    arr = np.asarray(img, dtype=np.float64)
    Ny, Nx = arr.shape
    pad = k // 2
    # pad with edge values
    padded = np.pad(arr, ((pad, pad), (pad, pad)), mode='edge')
    # integral image
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    # add leading zero row/col to simplify rectangle sums
    integral_z = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    integral_z[1:, 1:] = integral
    # compute window sums via integral image
    y0 = np.arange(0, Ny)
    x0 = np.arange(0, Nx)
    # vectorized computation via slicing
    y1 = y0 + k
    x1 = x0 + k
    # create 2D grids
    Y0, X0 = np.meshgrid(y0, x0, indexing='ij')
    Y1 = Y0 + k
    X1 = X0 + k
    sum_window = (
        integral_z[Y1, X1]
        - integral_z[Y0, X1]
        - integral_z[Y1, X0]
        + integral_z[Y0, X0]
    )
    return sum_window / (k * k)


def _adaptive_threshold_mask(img: np.ndarray, cfg: Optional[Any]) -> np.ndarray:
    """Tile-based adaptive thresholding to capture dim spots.

    Uses local max with relative threshold per tile, supports local Otsu.
    """
    arr = np.asarray(img, dtype=np.float64)
    Ny, Nx = arr.shape
    tile_px = int(getattr(cfg, 'adaptive_tile_px', 32)) if cfg is not None else 32
    tile_px = max(8, tile_px)
    # optional pre-blur to suppress noise
    pre_k = int(getattr(cfg, 'pre_blur_ksize', 0)) if cfg is not None else 0
    work = _box_filter_2d(arr, pre_k) if pre_k and pre_k > 1 else arr
    mask = np.zeros_like(work, dtype=bool)
    ttype = str(getattr(cfg, 'threshold_type', 'relative')) if cfg is not None else 'relative'
    tval = float(getattr(cfg, 'threshold_value', 0.3)) if cfg is not None else 0.3
    for y0 in range(0, Ny, tile_px):
        y1 = min(Ny, y0 + tile_px)
        for x0 in range(0, Nx, tile_px):
            x1 = min(Nx, x0 + tile_px)
            tile = work[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            vmax = float(np.max(tile))
            if vmax <= 0:
                continue
            if ttype == 'absolute':
                thr = tval
            elif ttype == 'otsu':
                thr = _otsu_threshold(tile)
            else:
                thr = vmax * tval
            mask[y0:y1, x0:x1] |= (tile >= thr)
    # also include global threshold to catch very bright spots spanning tiles
    thr_global = _compute_threshold(work, cfg)
    mask |= (work >= thr_global)
    # simple noise cleanup: remove isolated pixels (<=1 neighbors)
    Ny, Nx = mask.shape
    cleaned = mask.copy()
    for y in range(Ny):
        for x in range(Nx):
            if not mask[y, x]:
                continue
            cnt = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny = y + dy
                    nx = x + dx
                    if 0 <= ny < Ny and 0 <= nx < Nx and mask[ny, nx]:
                        cnt += 1
            if cnt <= 1:
                cleaned[y, x] = False
    return cleaned
