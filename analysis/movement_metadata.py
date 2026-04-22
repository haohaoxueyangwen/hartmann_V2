"""
Shared movement metadata parser.

Parses simulation_params.json into a unified MovementInfo structure,
so that recover.py and analysis modules share a single source of truth
for movement trajectory data.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class MovementInfo:
    """Unified movement description parsed from simulation_params.json.

    Raw metadata convention:
    - offsets use (dx, dy), unit meters or pixels
    - initial_offset_xy_m is the absolute mechanical offset of pulse 0
    - pulse_offsets_xy_*[k] are legacy per-pulse offsets relative to initial_offset_xy_m

    Accessor convention:
    - get_global_offsets_*() always return the actual per-pulse absolute offsets,
      i.e. initial_offset_xy_m + relative pulse offset
    """
    mode: str = "translation"                          # "none" | "translation" | "trajectory"
    is_uniform: bool = True
    n_pulses: int = 1
    initial_offset_xy_m: Tuple[float, float] = (0.0, 0.0)

    # Legacy/raw per-pulse offsets relative to initial_offset_xy_m (length == n_pulses, or None)
    pulse_offsets_xy_m: Optional[List[Tuple[float, float]]] = None
    pulse_offsets_xy_px: Optional[np.ndarray] = None   # shape (n_pulses, 2) or None

    # Uniform-step compat fields (valid when is_uniform=True)
    uniform_step_m: Optional[float] = None
    uniform_angle_deg: Optional[float] = None

    # Per-step details (length == n_pulses-1, or None)
    step_lengths_m: Optional[List[float]] = None
    step_angles_deg: Optional[List[float]] = None

    generated_from: Optional[str] = None


def _validate_offsets_shape(offsets: np.ndarray, n_pulses: int, *, name: str) -> np.ndarray:
    arr = np.asarray(offsets, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_pulses, 2), got {arr.shape}")
    if arr.shape[0] != int(n_pulses):
        raise ValueError(f"{name} length {arr.shape[0]} does not match n_pulses={n_pulses}")
    return arr


def _initial_offset_array_m(info: MovementInfo) -> np.ndarray:
    arr = np.asarray(info.initial_offset_xy_m, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError(f"initial_offset_xy_m must have shape (2,), got {arr.shape}")
    return arr


def parse_movement_info(params: dict) -> MovementInfo:
    """Parse movement info from a simulation_params.json dict.

    Handles both old-format (angle_deg + step_m only) and new-format
    (movement_mode + movement_pulse_offsets_xy_m) params.
    """
    movement_mode = str(params.get('movement_mode', 'translation') or 'translation').lower()
    n_pulses = int(params.get('n_pulses', 1))
    pixel_size_m = params.get('pixel_size_m')

    # Initial offset
    raw_init = params.get('movement_initial_offset_xy_m')
    if raw_init is not None and len(raw_init) == 2:
        initial_offset_xy_m = (float(raw_init[0]), float(raw_init[1]))
    else:
        initial_offset_xy_m = (0.0, 0.0)

    # Uniform compat fields
    step_m = params.get('step_m')
    angle_deg = params.get('angle_deg')
    uniform_step_m = params.get('movement_uniform_step_m', step_m)
    uniform_angle_deg = params.get('movement_uniform_angle_deg', angle_deg)
    if uniform_step_m is not None:
        uniform_step_m = float(uniform_step_m)
    if uniform_angle_deg is not None:
        uniform_angle_deg = float(uniform_angle_deg)

    # Per-pulse offsets
    raw_offsets_absolute = params.get('movement_pulse_absolute_offsets_xy_m')
    raw_offsets = params.get('movement_pulse_offsets_xy_m')
    pulse_offsets_xy_m: Optional[List[Tuple[float, float]]] = None
    if raw_offsets_absolute is not None:
        abs_arr = _validate_offsets_shape(
            np.asarray(raw_offsets_absolute, dtype=np.float64),
            n_pulses,
            name="movement_pulse_absolute_offsets_xy_m",
        )
        init_arr = np.asarray(initial_offset_xy_m, dtype=np.float64)
        rel_arr = abs_arr - init_arr[np.newaxis, :]
        pulse_offsets_xy_m = [(float(o[0]), float(o[1])) for o in rel_arr]
    elif raw_offsets is not None:
        pulse_offsets_xy_m = [(float(o[0]), float(o[1])) for o in raw_offsets]

    # Per-step details
    raw_step_lengths = params.get('movement_step_lengths_m')
    step_lengths_m = [float(x) for x in raw_step_lengths] if raw_step_lengths is not None else None

    raw_step_angles = params.get('movement_step_angles_deg')
    step_angles_deg = [float(x) for x in raw_step_angles] if raw_step_angles is not None else None

    generated_from = params.get('movement_generated_from')

    # Determine is_uniform
    is_uniform_explicit = params.get('movement_is_uniform')
    if is_uniform_explicit is not None:
        is_uniform = bool(is_uniform_explicit)
    elif movement_mode == 'trajectory':
        lengths_uniform = False
        if step_lengths_m is not None:
            arr = np.asarray(step_lengths_m, dtype=np.float64)
            lengths_uniform = bool(arr.size == 0 or np.allclose(arr, arr[0], rtol=1e-9, atol=1e-15))

        angles_uniform = False
        if step_angles_deg is not None:
            arr = np.asarray(step_angles_deg, dtype=np.float64)
            if arr.size == int(n_pulses):
                arr = arr[1:]
            angles_uniform = bool(arr.size == 0 or np.allclose(arr, arr[0], rtol=1e-9, atol=1e-12))

        if step_lengths_m is not None and step_angles_deg is not None:
            is_uniform = bool(lengths_uniform and angles_uniform)
        elif uniform_step_m is not None and uniform_angle_deg is not None:
            is_uniform = True
        else:
            is_uniform = False
    else:
        is_uniform = True

    # Convert offsets to pixels
    pulse_offsets_xy_px: Optional[np.ndarray] = None
    if pulse_offsets_xy_m is not None and pixel_size_m not in (None, 0, 0.0):
        ps = float(pixel_size_m)
        pulse_offsets_xy_px = np.array(
            [[o[0] / ps, o[1] / ps] for o in pulse_offsets_xy_m],
            dtype=np.float64,
        )

    return MovementInfo(
        mode=movement_mode,
        is_uniform=is_uniform,
        n_pulses=n_pulses,
        initial_offset_xy_m=initial_offset_xy_m,
        pulse_offsets_xy_m=pulse_offsets_xy_m,
        pulse_offsets_xy_px=pulse_offsets_xy_px,
        uniform_step_m=uniform_step_m,
        uniform_angle_deg=uniform_angle_deg,
        step_lengths_m=step_lengths_m,
        step_angles_deg=step_angles_deg,
        generated_from=generated_from,
    )


def get_global_offsets_m(info: MovementInfo) -> np.ndarray:
    """Return per-pulse absolute offsets as (n_pulses, 2) array in meters."""
    initial_offset = _initial_offset_array_m(info)

    if info.pulse_offsets_xy_m is not None:
        relative_offsets = _validate_offsets_shape(
            np.asarray(info.pulse_offsets_xy_m, dtype=np.float64),
            info.n_pulses,
            name="pulse_offsets_xy_m",
        )
        return relative_offsets + initial_offset[np.newaxis, :]

    if info.uniform_step_m is not None and info.uniform_angle_deg is not None:
        step = float(info.uniform_step_m)
        ang = np.deg2rad(float(info.uniform_angle_deg))
        dx_per = step * np.cos(ang)
        dy_per = step * np.sin(ang)
        k = np.arange(info.n_pulses, dtype=np.float64)
        relative_offsets = np.column_stack([k * dx_per, k * dy_per])
        return relative_offsets + initial_offset[np.newaxis, :]

    if info.mode == "trajectory":
        raise ValueError("trajectory movement requires per-pulse offsets or uniform compat fields")

    return np.repeat(initial_offset[np.newaxis, :], info.n_pulses, axis=0)


def get_global_offsets_px(info: MovementInfo) -> np.ndarray:
    """Return per-pulse absolute offsets as (n_pulses, 2) array in pixels.

    This helper is only safe when the metadata already contains explicit pixel
    offsets and the initial mechanical offset is zero. Otherwise use
    get_global_offsets_px_with_pixelsize(...) so the initial offset can be
    converted from meters.
    """
    initial_offset = _initial_offset_array_m(info)
    if not np.allclose(initial_offset, 0.0, rtol=1e-9, atol=1e-15):
        raise ValueError(
            "movement metadata includes non-zero initial_offset_xy_m; "
            "use get_global_offsets_px_with_pixelsize(...) instead."
        )
    if info.pulse_offsets_xy_px is not None:
        return _validate_offsets_shape(
            np.asarray(info.pulse_offsets_xy_px, dtype=np.float64),
            info.n_pulses,
            name="pulse_offsets_xy_px",
        )
    if info.mode == "none":
        return np.zeros((info.n_pulses, 2), dtype=np.float64)
    raise ValueError(
        "pixel-size-free movement metadata does not contain pixel offsets; "
        "use get_global_offsets_px_with_pixelsize(...) instead."
    )


def get_global_offsets_px_with_pixelsize(info: MovementInfo, pixel_size_m: float) -> np.ndarray:
    """Like get_global_offsets_px but explicitly converts from meters if needed."""
    ps = float(pixel_size_m)
    if ps <= 0:
        raise ValueError(f"pixel_size_m must be positive, got {ps}")

    offsets_m = get_global_offsets_m(info)
    return np.asarray(offsets_m, dtype=np.float64) / ps
