from __future__ import annotations
"""Legacy spacing solver for uniform constant-step movement only.

This module is not trajectory-aware. It should only be used for the older
uniform translation design problem and must not be used as the primary design
tool for non-uniform pulse trajectories.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
from math import sin, cos, radians, sqrt
from analysis.zernike_reconstruction import energy_ev_to_wavelength_m

@dataclass
class SolverConfig:
    n_frames: int
    step_m: float
    angle_deg: float
    duty_area: float
    wavelength_m: Optional[float]
    photon_energy_ev: Optional[float]
    z_m: float
    min_diameter_m: float
    alpha_sep: float
    max_iter: int
    tol: float

def _ensure_wavelength(cfg: SolverConfig) -> float:
    if cfg.wavelength_m is not None:
        return float(cfg.wavelength_m)
    if cfg.photon_energy_ev is not None:
        return float(energy_ev_to_wavelength_m(cfg.photon_energy_ev))
    return float(energy_ev_to_wavelength_m(12000.0))

def _p_dmin(duty_area: float, min_diameter_m: float) -> float:
    return float(min_diameter_m) * sqrt(np.pi / max(1e-18, 4.0 * float(duty_area)))

def _p_psf(alpha_sep: float, wavelength_m: float, z_m: float, step_m: float, duty_area: float) -> float:
    return float(alpha_sep) * float(wavelength_m) * float(z_m) / max(1e-18, float(step_m)) * sqrt(np.pi / max(1e-18, 4.0 * float(duty_area)))

def _r_sep(alpha_sep: float, wavelength_m: float, z_m: float, p_m: float, duty_area: float) -> float:
    d = float(p_m) * sqrt(max(0.0, 4.0 * float(duty_area)) / np.pi)
    return float(alpha_sep) * float(wavelength_m) * float(z_m) / max(1e-18, d)

def _delta_component(p_m: float, step_m: float, angle_deg: float, n_frames: int, component: str) -> float:
    theta = radians(float(angle_deg))
    if component == 'x':
        v = float(p_m) * cos(theta)
    else:
        v = float(p_m) * sin(theta)
    s = float(step_m)
    k_min = -max(0, int(n_frames) - 1)
    k_max = max(0, int(n_frames) - 1)
    k0 = int(round(v / max(1e-18, s)))
    ks = [max(k_min, min(k_max, k0 + dk)) for dk in (-1, 0, 1)]
    deltas = [abs(v - k * s) for k in set(ks)]
    return float(min(deltas))

def solve_min_spacing(cfg: SolverConfig) -> Dict[str, Any]:
    if cfg.step_m is None or cfg.angle_deg is None:
        raise ValueError(
            "spacing_solver is a legacy uniform-step tool only; "
            "non-uniform trajectory design must use full pulse offsets instead."
        )
    lam = _ensure_wavelength(cfg)
    D = float(cfg.duty_area)
    s = float(cfg.step_m)
    z = float(cfg.z_m)
    theta = radians(float(cfg.angle_deg))
    p1 = _p_dmin(D, float(cfg.min_diameter_m))
    p2 = _p_psf(float(cfg.alpha_sep), lam, z, s, D)
    p = max(p1, p2)
    for _ in range(int(cfg.max_iter)):
        rs = _r_sep(float(cfg.alpha_sep), lam, z, p, D)
        dx = _delta_component(p, s, cfg.angle_deg, cfg.n_frames, 'x')
        dy = _delta_component(p, s, cfg.angle_deg, cfg.n_frames, 'y')
        gx = sqrt((p * sin(theta)) ** 2 + dx ** 2)
        gy = sqrt((p * cos(theta)) ** 2 + dy ** 2)
        need_x = 0.0
        need_y = 0.0
        if gx < rs and abs(sin(theta)) > 1e-18:
            val = rs * rs - dx * dx
            if val > 0.0:
                need_x = sqrt(val) / abs(sin(theta))
        if gy < rs and abs(cos(theta)) > 1e-18:
            val = rs * rs - dy * dy
            if val > 0.0:
                need_y = sqrt(val) / abs(cos(theta))
        p_next = max(p, p1, p2, need_x, need_y)
        if abs(p_next - p) <= float(cfg.tol):
            p = p_next
            break
        p = p_next
    d = p * sqrt(max(0.0, 4.0 * D) / np.pi)
    rs = _r_sep(float(cfg.alpha_sep), lam, z, p, D)
    dx = _delta_component(p, s, cfg.angle_deg, cfg.n_frames, 'x')
    dy = _delta_component(p, s, cfg.angle_deg, cfg.n_frames, 'y')
    theta_sin = sin(theta)
    theta_cos = cos(theta)
    gx = sqrt((p * theta_sin) ** 2 + dx ** 2)
    gy = sqrt((p * theta_cos) ** 2 + dy ** 2)
    ok = min(s, gx, gy) >= rs and d >= float(cfg.min_diameter_m)
    return {
        'spacing_m': float(p),
        'diameter_m': float(d),
        'r_sep_m': float(rs),
        'checks': {
            'p_dmin_m': float(p1),
            'p_psf_m': float(p2),
            'gx_m': float(gx),
            'gy_m': float(gy),
            'delta_x_m': float(dx),
            'delta_y_m': float(dy),
            'step_m': float(s)
        },
        'ok': bool(ok)
    }

def recommend_default() -> Dict[str, Any]:
    cfg = SolverConfig(
        n_frames=10,
        step_m=120e-6,
        angle_deg=19.0,
        duty_area=0.005,
        wavelength_m=None,
        photon_energy_ev=12000.0,
        z_m=0.3,
        min_diameter_m=3e-6,
        alpha_sep=2.44,
        max_iter=128,
        tol=1e-9,
    )
    return solve_min_spacing(cfg)

if __name__ == '__main__':
    import json
    res = recommend_default()
    print(json.dumps(res, indent=2))
