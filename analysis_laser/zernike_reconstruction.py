from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from analysis_laser.serialization import to_builtin_serializable
from analysis_laser.zernike_basis import BasisFunction, build_zernike_basis_functions
from simulation.detector import analyze_spot_resolution


@dataclass
class RoiSpec:
    idx: int
    x0: int
    x1: int
    y0: int
    y1: int
    x_center_px: float
    y_center_px: float
    status: str = "ok"

def build_rois_around_centers(
    centers: Sequence[dict],
    roi_radius_px: int,
    nx: int,
    ny: int,
) -> List[RoiSpec]:
    r = int(max(1, roi_radius_px))
    rois: List[RoiSpec] = []
    for idx, c in enumerate(centers):
        try:
            cx = float(c.get("x_px", np.nan))
            cy = float(c.get("y_px", np.nan))
        except Exception:
            rois.append(RoiSpec(idx, 0, 0, 0, 0, 0.0, 0.0, status="invalid_center"))
            continue
        if not np.isfinite(cx) or not np.isfinite(cy):
            rois.append(RoiSpec(idx, 0, 0, 0, 0, 0.0, 0.0, status="invalid_center"))
            continue
        x0 = max(0, int(round(cx - r)))
        x1 = min(nx, int(round(cx + r + 1)))
        y0 = max(0, int(round(cy - r)))
        y1 = min(ny, int(round(cy + r + 1)))
        if x1 - x0 <= 1 or y1 - y0 <= 1:
            rois.append(RoiSpec(idx, x0, x1, y0, y1, cx, cy, status="out_of_bounds"))
            continue
        rois.append(RoiSpec(idx, x0, x1, y0, y1, cx, cy, status="ok"))
    return rois


def compute_resolution_on_frame(
    detector_image: np.ndarray,
    centers: Sequence[dict],
    pixel_size_m: float,
    roi_radius_px: int,
    methods: Iterable[str],
) -> Dict[str, Any]:
    img = np.asarray(detector_image, dtype=float)
    ny, nx = img.shape
    rois = build_rois_around_centers(centers, roi_radius_px, nx, ny)

    per_spot: List[Dict[str, Any]] = []
    fwhm_vals: List[float] = []
    for roi in rois:
        entry: Dict[str, Any] = {
            "idx": roi.idx,
            "x_center_px": float(roi.x_center_px),
            "y_center_px": float(roi.y_center_px),
            "roi": {
                "x0": roi.x0,
                "x1": roi.x1,
                "y0": roi.y0,
                "y1": roi.y1,
            },
            "status": roi.status,
        }
        if roi.status != "ok":
            per_spot.append(entry)
            continue
        patch = img[roi.y0 : roi.y1, roi.x0 : roi.x1]
        metrics: Dict[str, Any] = {}
        for m in methods:
            res = analyze_spot_resolution(patch, pixel_size_m, method=str(m))
            metrics[m] = to_builtin_serializable(res)
            if m == "fwhm":
                fv = res.get("fwhm_avg")
                if fv is not None:
                    fwhm_vals.append(float(fv))
        entry["metrics"] = metrics
        per_spot.append(entry)

    summary: Dict[str, Any] = {
        "count_total": len(per_spot),
        "count_valid": int(sum(1 for p in per_spot if p.get("status") == "ok")),
    }
    if fwhm_vals:
        summary["fwhm_mean_m"] = float(np.mean(fwhm_vals))
        summary["fwhm_median_m"] = float(np.median(fwhm_vals))
        summary["fwhm_min_m"] = float(np.min(fwhm_vals))
        summary["fwhm_max_m"] = float(np.max(fwhm_vals))

    return {"per_spot": per_spot, "summary": summary}


# ---- Wavefront recovery helpers ------------------------------------------------

def zernike_basis(n_terms: int = 15, radius: float = 1.0) -> List[BasisFunction]:
    """
    Backward-compatible wrapper around the shared canonical Zernike basis.
    """
    return build_zernike_basis_functions(n_terms=n_terms, radius=radius)


def energy_ev_to_wavelength_m(energy_ev: float) -> float:
    hc = 1.2398419843320026e-6  # [m·eV]
    return hc / float(energy_ev)


def reconstruct_wavefront_from_displacements(
    displacements: Sequence[dict],
    *,
    pixel_size_m: float,
    propagation_distance_m: float,
    wavelength_m: float,
    detector_nx: int,
    detector_ny: int,
    pupil_radius_m: Optional[float] = None,
    n_modes: int = 11,
    grid_size: int = 128,
    grid_margin: float = 1.15,
) -> Dict[str, Any]:
    
    dxs: List[float] = []
    dys: List[float] = []
    xs: List[float] = []
    ys: List[float] = []
    for d in displacements:
        if str(d.get("status", "ok")) != "ok":
            continue
        dx_px = float(d.get("dx_px", 0.0))
        dy_px = float(d.get("dy_px", 0.0))
        x_ref_px = float(d.get("x_ref_px", 0.0))
        y_ref_px = float(d.get("y_ref_px", 0.0))
        xs.append((x_ref_px - detector_nx / 2.0 + 0.5) * pixel_size_m)
        ys.append((y_ref_px - detector_ny / 2.0 + 0.5) * pixel_size_m)
        dxs.append(dx_px * pixel_size_m)
        dys.append(dy_px * pixel_size_m)

    if not xs:
        return {"status": "no_valid_spots"}

    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    slope_x = np.asarray(dxs, dtype=float) / float(propagation_distance_m)
    slope_y = np.asarray(dys, dtype=float) / float(propagation_distance_m)

    # Determine pupil radius if not provided
    if pupil_radius_m is None:
        r_sq = xs_arr**2 + ys_arr**2
        pupil_radius_m = float(np.sqrt(np.max(r_sq))) * 1.05 # 5% margin
    
    # Use Zernike basis
    basis = zernike_basis(n_terms=n_modes, radius=pupil_radius_m)

    n = len(xs_arr)
    m = len(basis)
    A = np.zeros((2 * n, m), dtype=float)
    b = np.zeros((2 * n,), dtype=float)
    for i in range(n):
        x = xs_arr[i]
        y = ys_arr[i]
        for j, bf in enumerate(basis):
            A[2 * i, j] = float(bf.d_dx(x, y))
            A[2 * i + 1, j] = float(bf.d_dy(x, y))
        b[2 * i] = slope_x[i]
        b[2 * i + 1] = slope_y[i]

    try:
        coeffs, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError as exc:
        return {"status": f"lstsq_failed: {exc}"}

    pred = A @ coeffs
    res_x = slope_x - pred[0::2]
    res_y = slope_y - pred[1::2]
    residual_rms = float(np.sqrt(np.mean(res_x**2 + res_y**2)))

    extent = grid_margin * float(pupil_radius_m)
    grid_lin = np.linspace(-extent, extent, int(max(16, grid_size)))
    Xg, Yg = np.meshgrid(grid_lin, grid_lin)
    
    # Mask outside pupil
    Rg = np.sqrt(Xg**2 + Yg**2)
    mask = Rg <= pupil_radius_m

    wf = np.zeros_like(Xg, dtype=float)
    for coef, bf in zip(coeffs, basis):
        wf += float(coef) * bf.phase(Xg, Yg)
    
    # Apply mask
    wf[~mask] = np.nan
    
    phase = (2.0 * np.pi / float(wavelength_m)) * wf

    wf_valid = wf[mask]
    wf_rms = float(np.sqrt(np.mean(wf_valid**2))) if wf_valid.size > 0 else 0.0
    wf_pv = float(np.ptp(wf_valid)) if wf_valid.size > 0 else 0.0
    phase_valid = phase[mask]
    phase_rms = float(np.sqrt(np.mean(phase_valid**2))) if phase_valid.size > 0 else 0.0

    coeff_list = [
        {"name": bf.name, "value_m": float(c)}
        for bf, c in zip(basis, coeffs)
    ]
    return {
        "status": "ok",
        "coefficients": coeff_list,
        "slope_rms_rad": float(np.sqrt(np.mean(slope_x**2 + slope_y**2))),
        "residual_rms_rad": residual_rms,
        "wavefront_rms_m": wf_rms,
        "wavefront_pv_m": wf_pv,
        "phase_rms_rad": phase_rms,
        "pupil_radius_m": pupil_radius_m,
        "grid": {
            "x_m": grid_lin,
            "y_m": grid_lin,
        },
        "maps": {
            "wavefront_m": wf,
            "phase_rad": phase,
        },
    }
