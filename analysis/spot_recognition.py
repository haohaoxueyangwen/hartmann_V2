"""
Spot recognition pipeline for Hartmanngram images.

Implements the centroid-extraction flow described in Huang et al.,
Optics Express 30(23):41061-41084 (2022):
1. Pre-filter + intensity binarization (Otsu or adaptive Gaussian)
2. Connected-component labelling with area / signal filtering
3. Coarse centroid per label
4. Cropping with balanced neighbouring background
5. Gamma-weighted fine centroid detection
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.ndimage as ndi
except Exception:  # pragma: no cover - SciPy should be available per project deps
    ndi = None

from simulation.blur import gaussian_blur


@dataclass
class SpotRecognitionConfig:
    """Configuration for Hartmanngram spot recognition."""

    binarization_method: str = "otsu"  # "otsu" | "adaptive"
    gaussian_sigma_px: float = 1.0
    adaptive_window_px: int = 41
    adaptive_bias: float = 0.0
    min_area_px: int = 12
    max_area_px: Optional[int] = None
    connectivity: int = 2  # 1: 4-connect, 2: 8-connect
    min_signal: float = 0.0
    crop_half_size_px: int = 12
    background_margin_px: int = 4
    gamma: float = 1.7
    clip_negative: bool = True
    row_grouping_tol_px: float = 15.0
    col_grouping_tol_px: float = 15.0
    max_spots: Optional[int] = None
    prefer_high_signal: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpotRecognitionResult:
    centers: List[Dict[str, Any]]
    binary_mask: np.ndarray
    label_map: np.ndarray
    method: str


def run_spot_recognition(
    detector_image: np.ndarray,
    pixel_size_m: Optional[float],
    config: SpotRecognitionConfig,
) -> SpotRecognitionResult:
    """
    Execute the recognition pipeline on a detector image.

    Parameters
    ----------
    detector_image:
        2D detector intensity array (ny, nx).
    pixel_size_m:
        Detector pixel size in meters (used to convert to physical units).
    config:
        SpotRecognitionConfig controlling filtering / thresholds.
    """
    _ensure_ndimage_available()
    arr = np.asarray(detector_image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("detector_image must be 2-D")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0] = 0.0

    work = arr
    if float(config.gaussian_sigma_px) > 0.0:
        work = gaussian_blur(arr, float(config.gaussian_sigma_px))

    method_pref = str(config.binarization_method or "otsu").lower()
    if method_pref == "auto":
        method_sequence = ["otsu", "adaptive"]
    else:
        method_sequence = [method_pref]
        if method_pref == "otsu":
            method_sequence.append("adaptive")

    best_result: Optional[SpotRecognitionResult] = None
    for method_name in method_sequence:
        mask = _binarize_image(work, config, method=method_name)
        result = _extract_spots_from_mask(arr, mask, pixel_size_m, config, method_name)
        if best_result is None or len(result.centers) > len(best_result.centers):
            best_result = result
        if result.centers:
            return result

    if best_result is not None:
        return best_result
    empty_mask = np.zeros_like(arr, dtype=np.uint8)
    empty_labels = np.zeros_like(arr, dtype=np.uint16)
    return SpotRecognitionResult([], empty_mask, empty_labels, method_pref)


def _extract_spots_from_mask(
    arr: np.ndarray,
    mask: np.ndarray,
    pixel_size_m: Optional[float],
    config: SpotRecognitionConfig,
    method_name: str,
) -> SpotRecognitionResult:
    if not np.any(mask):
        empty = np.zeros_like(mask, dtype=np.uint16)
        return SpotRecognitionResult([], mask.astype(np.uint8), empty, method_name)

    label_map, num_labels = _label_components(mask, config.connectivity)
    if num_labels <= 0:
        empty = np.zeros_like(mask, dtype=np.uint16)
        return SpotRecognitionResult([], mask.astype(np.uint8), empty, method_name)

    slices = ndi.find_objects(label_map)
    label_ids = np.arange(1, num_labels + 1, dtype=np.int32)
    areas = np.bincount(label_map.ravel()).tolist()
    signals = ndi.sum(arr, label_map, index=label_ids)
    coms = ndi.center_of_mass(arr, label_map, index=label_ids)

    Ny, Nx = arr.shape
    kept: List[Dict[str, Any]] = []
    for label, signal_value, slc, center_of_mass in zip(label_ids, signals, slices, coms):
        area = float(areas[label]) if label < len(areas) else 0.0
        if area <= 0:
            continue
        if area < float(config.min_area_px):
            continue
        if config.max_area_px is not None and area > float(config.max_area_px):
            continue
        signal = float(signal_value)
        if signal < float(config.min_signal):
            continue
        if slc is None:
            continue
        coarse_y, coarse_x = center_of_mass
        if not np.isfinite(coarse_x) or not np.isfinite(coarse_y):
            continue
        coarse_x = float(np.clip(coarse_x, 0.0, Nx - 1.0))
        coarse_y = float(np.clip(coarse_y, 0.0, Ny - 1.0))
        refined_x, refined_y, bg = _refine_centroid(arr, coarse_x, coarse_y, config)
        bbox = [
            int(slc[1].start),
            int(slc[0].start),
            int(slc[1].stop - 1),
            int(slc[0].stop - 1),
        ]
        entry = {
            "_orig_label": int(label),
            "label": len(kept) + 1,
            "area_px": float(area),
            "signal": signal,
            "bbox": bbox,
            "coarse_x_px": coarse_x,
            "coarse_y_px": coarse_y,
            "x_px": refined_x,
            "y_px": refined_y,
            "status": "ok",
            "background": float(bg),
        }
        if pixel_size_m is not None:
            entry["x_m"] = (refined_x - Nx / 2.0 + 0.5) * float(pixel_size_m)
            entry["y_m"] = (refined_y - Ny / 2.0 + 0.5) * float(pixel_size_m)
        else:
            entry["x_m"] = None
            entry["y_m"] = None
        kept.append(entry)

    if not kept:
        empty = np.zeros_like(mask, dtype=np.uint16)
        return SpotRecognitionResult([], mask.astype(np.uint8), empty, method_name)

    if config.max_spots is not None and len(kept) > int(config.max_spots):
        if config.prefer_high_signal:
            kept.sort(key=lambda d: float(d["signal"]), reverse=True)
        kept = kept[: int(config.max_spots)]

    kept.sort(key=lambda d: (float(d["y_px"]), float(d["x_px"])))
    y_orders = _assign_orders([float(c["y_px"]) for c in kept], float(config.row_grouping_tol_px))
    x_orders = _assign_orders([float(c["x_px"]) for c in kept], float(config.col_grouping_tol_px))
    for idx, c in enumerate(kept):
        c["order_y"] = y_orders[idx]
        c["order_x"] = x_orders[idx]

    filtered_map = np.zeros_like(label_map, dtype=np.uint16)
    for new_label, c in enumerate(kept, start=1):
        orig_label = c.pop("_orig_label", None)
        if orig_label is None:
            continue
        filtered_map[label_map == orig_label] = new_label
        c["label"] = new_label

    binary_mask = (filtered_map > 0).astype(np.uint8)
    return SpotRecognitionResult(kept, binary_mask, filtered_map, method_name)


def _ensure_ndimage_available() -> None:
    if ndi is None:
        raise RuntimeError(
            "scipy.ndimage is required for spot recognition. "
            "Please install SciPy (pip install scipy)."
        )


def _binarize_image(
    arr: np.ndarray,
    config: SpotRecognitionConfig,
    method: Optional[str] = None,
) -> np.ndarray:
    method = str(method or config.binarization_method or "otsu").lower()
    if method not in {"otsu", "adaptive"}:
        raise ValueError(f"Unsupported binarization_method: {config.binarization_method}")
    if method == "otsu":
        thresh = _otsu_threshold(arr)
        return arr > thresh
    # adaptive: gaussian local mean minus bias
    win = max(3, int(config.adaptive_window_px))
    if win % 2 == 0:
        win += 1
    local_mean = _local_mean(arr, win)
    bias = float(config.adaptive_bias)
    return arr > (local_mean - bias)


def _otsu_threshold(arr: np.ndarray, bins: int = 512) -> float:
    data = np.asarray(arr, dtype=np.float64)
    if data.size == 0:
        return 0.0
    min_val = float(np.min(data))
    max_val = float(np.max(data))
    if max_val <= min_val:
        return max_val
    hist, bin_edges = np.histogram(data, bins=bins, range=(min_val, max_val))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return max_val
    cumulative = np.cumsum(hist)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    cumulative_mean = np.cumsum(hist * bin_centers)
    global_mean = cumulative_mean[-1]
    denominator = cumulative * (total - cumulative)
    denominator[denominator == 0] = np.nan
    between = (global_mean * cumulative - cumulative_mean) ** 2 / denominator
    idx = int(np.nanargmax(between))
    return float(bin_centers[idx])


def _local_mean(arr: np.ndarray, window: int) -> np.ndarray:
    if ndi is not None:
        return ndi.uniform_filter(arr, size=window, mode="nearest")
    # Fallback: separable box filter via cumulative sums
    pad = window // 2
    padded = np.pad(arr, pad_width=pad, mode="edge")
    cumsum = padded.cumsum(axis=0).cumsum(axis=1)
    h, w = arr.shape
    out = np.empty_like(arr, dtype=np.float64)
    for y in range(h):
        y0 = y
        y1 = y + window
        for x in range(w):
            x0 = x
            x1 = x + window
            total = cumsum[y1, x1]
            if y0 > 0:
                total -= cumsum[y0 - 1, x1]
            if x0 > 0:
                total -= cumsum[y1, x0 - 1]
            if y0 > 0 and x0 > 0:
                total += cumsum[y0 - 1, x0 - 1]
            out[y, x] = total / (window * window)
    return out


def _label_components(mask: np.ndarray, connectivity: int) -> Tuple[np.ndarray, int]:
    struct = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    if int(connectivity) == 1:
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    label_map, num = ndi.label(mask.astype(np.int32), structure=struct)
    return label_map.astype(np.int32), int(num)


def _refine_centroid(
    arr: np.ndarray,
    coarse_x: float,
    coarse_y: float,
    config: SpotRecognitionConfig,
) -> Tuple[float, float, float]:
    Ny, Nx = arr.shape
    half = max(1, int(config.crop_half_size_px))
    margin = max(0, int(config.background_margin_px))
    total_half = half + margin
    cx = int(round(coarse_x))
    cy = int(round(coarse_y))
    x0 = max(0, cx - total_half)
    x1 = min(Nx, cx + total_half + 1)
    y0 = max(0, cy - total_half)
    y1 = min(Ny, cy + total_half + 1)
    patch = arr[y0:y1, x0:x1].copy()
    inner_x0 = max(0, cx - half) - x0
    inner_y0 = max(0, cy - half) - y0
    inner_x1 = min(Nx, cx + half + 1) - x0
    inner_y1 = min(Ny, cy + half + 1) - y0
    inner_mask = np.zeros_like(patch, dtype=bool)
    inner_mask[inner_y0:inner_y1, inner_x0:inner_x1] = True
    ring_mask = np.ones_like(patch, dtype=bool)
    ring_mask[inner_y0:inner_y1, inner_x0:inner_x1] = False
    bg_vals = patch[ring_mask]
    if bg_vals.size == 0:
        background = 0.0
    else:
        # Use 20th percentile to estimate background, safer against neighbor contamination than median
        background = float(np.percentile(bg_vals, 20))
    patch = patch - background
    if config.clip_negative:
        patch[patch < 0.0] = 0.0
    patch *= inner_mask
    if float(config.gamma) != 1.0:
        patch = np.power(patch, float(config.gamma), where=inner_mask, out=patch)
    weight = np.sum(patch)
    if weight <= 0.0:
        return coarse_x, coarse_y, background
    ys, xs = np.mgrid[y0:y1, x0:x1]
    x_refined = float(np.sum(patch * xs) / weight)
    y_refined = float(np.sum(patch * ys) / weight)
    return x_refined, y_refined, background


def _assign_orders(values: Sequence[float], tolerance: float) -> List[int]:
    if not values:
        return []
    idx_sorted = sorted(range(len(values)), key=lambda i: values[i])
    groups: List[Dict[str, Any]] = []
    tol = max(0.0, float(tolerance))
    for idx in idx_sorted:
        val = float(values[idx])
        assigned = False
        for group in groups:
            if abs(val - group["mean"]) <= tol:
                count = group["count"]
                group["mean"] = (group["mean"] * count + val) / (count + 1)
                group["count"] = count + 1
                group["members"].append(idx)
                assigned = True
                break
        if not assigned:
            groups.append({"mean": val, "count": 1, "members": [idx]})
    groups.sort(key=lambda g: g["mean"])
    orders = [0] * len(values)
    for order, group in enumerate(groups):
        for idx in group["members"]:
            orders[idx] = order
    return orders


def run_guided_spot_recognition(
    detector_image: np.ndarray,
    pixel_size_m: Optional[float],
    config: SpotRecognitionConfig,
    known_centers: List[Dict[str, Any]],
) -> SpotRecognitionResult:
    """
    Execute recognition guided by known approximate positions.
    
    Instead of blind peak finding, this method iterates over the provided
    known centers (e.g. from mask geometry), extracts a ROI for each,
    and performs centroid refinement. This is much more robust for
    reference images where the ideal grid is known.

    Parameters
    ----------
    detector_image:
        2D detector intensity array.
    pixel_size_m:
        Detector pixel size in meters.
    config:
        SpotRecognitionConfig (uses crop_half_size_px, gamma, etc.).
    known_centers:
        List of dicts with keys 'x_px', 'y_px', and optional metadata like 'pulse_idx'.
    """
    _ensure_ndimage_available()
    arr = np.asarray(detector_image, dtype=np.float64)
    Ny, Nx = arr.shape
    
    # Pre-processing (optional, consistent with blind run)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0] = 0.0
    
    kept: List[Dict[str, Any]] = []
    
    # Create a blank label map for visualization consistency
    label_map = np.zeros((Ny, Nx), dtype=np.uint16)
    
    for idx, center_info in enumerate(known_centers, start=1):
        cx = float(center_info.get('x_px', 0.0))
        cy = float(center_info.get('y_px', 0.0))
        
        # Basic bounds check
        if not (0 <= cx < Nx and 0 <= cy < Ny):
            continue
            
        # Refine centroid
        refined_x, refined_y, bg = _refine_centroid(arr, cx, cy, config)
        
        # Calculate signal in the ROI (approximate)
        half = max(1, int(config.crop_half_size_px))
        x0 = max(0, int(round(refined_x)) - half)
        x1 = min(Nx, int(round(refined_x)) + half + 1)
        y0 = max(0, int(round(refined_y)) - half)
        y1 = min(Ny, int(round(refined_y)) + half + 1)
        
        # Mark label map (just the ROI box for visualization)
        label_map[y0:y1, x0:x1] = idx
        
        patch = arr[y0:y1, x0:x1]
        signal = float(np.sum(patch))
        area = float(patch.size) # Rough approximation
        
        bbox = [y0, x0, y1-1, x1-1]
        
        entry = {
            "label": idx,
            "area_px": area,
            "signal": signal,
            "bbox": bbox,
            "coarse_x_px": cx,
            "coarse_y_px": cy,
            "x_px": refined_x,
            "y_px": refined_y,
            "status": "ok",
            "background": float(bg),
        }
        
        # Copy over any extra metadata from the input (e.g. pulse_idx)
        for k, v in center_info.items():
            if k not in entry and k not in ('x_px', 'y_px'):
                entry[k] = v
        
        if pixel_size_m is not None:
            entry["x_m"] = (refined_x - Nx / 2.0 + 0.5) * float(pixel_size_m)
            entry["y_m"] = (refined_y - Ny / 2.0 + 0.5) * float(pixel_size_m)
        else:
            entry["x_m"] = None
            entry["y_m"] = None
            
        kept.append(entry)

    # Assign orders (grid indexing) - only if not already provided
    if kept and 'order_y' not in kept[0]:
        kept.sort(key=lambda d: (float(d["y_px"]), float(d["x_px"])))
        y_orders = _assign_orders([float(c["y_px"]) for c in kept], float(config.row_grouping_tol_px))
        x_orders = _assign_orders([float(c["x_px"]) for c in kept], float(config.col_grouping_tol_px))
        for i, c in enumerate(kept):
            c["order_y"] = y_orders[i]
            c["order_x"] = x_orders[i]

    binary_mask = (label_map > 0).astype(np.uint8)
    return SpotRecognitionResult(kept, binary_mask, label_map, "guided_geometry")


__all__ = [
    "SpotRecognitionConfig",
    "SpotRecognitionResult",
    "run_spot_recognition",
    "run_guided_spot_recognition",
]
