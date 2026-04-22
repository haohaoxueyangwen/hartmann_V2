"""
Canonical Zernike basis definitions shared across optimization and recovery.

Repository contract:
- coordinates are `(x, y)`
- the exported basis order matches the wavefront-reconstruction path
- basis functions use the same normalization in every caller
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


ArrayFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ZernikeMode:
    noll_index: int
    n: int
    m: int
    name: str


@dataclass
class BasisFunction:
    name: str
    phase: ArrayFunc
    d_dx: ArrayFunc
    d_dy: ArrayFunc


# Canonical repository order. This intentionally matches the basis used by
# wavefront reconstruction so coefficient ordering is consistent everywhere.
ZERNIKE_MODES: List[ZernikeMode] = [
    ZernikeMode(1, 0, 0, "Z1 (Piston)"),
    ZernikeMode(2, 1, 1, "Z2 (Tilt X)"),
    ZernikeMode(3, 1, -1, "Z3 (Tilt Y)"),
    ZernikeMode(4, 2, 0, "Z4 (Defocus)"),
    ZernikeMode(5, 2, -2, "Z5 (Astig 45)"),
    ZernikeMode(6, 2, 2, "Z6 (Astig 0)"),
    ZernikeMode(7, 3, -1, "Z7 (Coma Y)"),
    ZernikeMode(8, 3, 1, "Z8 (Coma X)"),
    ZernikeMode(9, 3, -3, "Z9 (Trefoil Y)"),
    ZernikeMode(10, 3, 3, "Z10 (Trefoil X)"),
    ZernikeMode(11, 4, 0, "Z11 (Spherical)"),
    ZernikeMode(12, 4, 2, "Z12 (Sec Astig 0)"),
    ZernikeMode(13, 4, -2, "Z13 (Sec Astig 45)"),
    ZernikeMode(14, 4, 4, "Z14 (Tetrafoil X)"),
    ZernikeMode(15, 4, -4, "Z15 (Tetrafoil Y)"),
]

# Backward-compatible `(n, m)` export used by legacy code.
ZERNIKE_INDICES: List[Tuple[int, int]] = [(mode.n, mode.m) for mode in ZERNIKE_MODES]


def zernike_basis_func(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Evaluate a normalized Zernike basis function on the unit disk.

    The normalization and mode ordering match the repository's canonical
    wavefront-reconstruction basis.
    """
    rho = np.asarray(rho, dtype=float)
    theta = np.asarray(theta, dtype=float)

    if n == 0 and m == 0:
        return np.ones_like(rho)
    if n == 1 and m == 1:
        return 2.0 * rho * np.cos(theta)
    if n == 1 and m == -1:
        return 2.0 * rho * np.sin(theta)
    if n == 2 and m == 0:
        return np.sqrt(3.0) * (2.0 * rho**2 - 1.0)
    if n == 2 and m == -2:
        return np.sqrt(6.0) * rho**2 * np.sin(2.0 * theta)
    if n == 2 and m == 2:
        return np.sqrt(6.0) * rho**2 * np.cos(2.0 * theta)
    if n == 3 and m == -1:
        return np.sqrt(8.0) * (3.0 * rho**2 - 2.0) * rho * np.sin(theta)
    if n == 3 and m == 1:
        return np.sqrt(8.0) * (3.0 * rho**2 - 2.0) * rho * np.cos(theta)
    if n == 3 and m == -3:
        return np.sqrt(8.0) * rho**3 * np.sin(3.0 * theta)
    if n == 3 and m == 3:
        return np.sqrt(8.0) * rho**3 * np.cos(3.0 * theta)
    if n == 4 and m == 0:
        return np.sqrt(5.0) * (6.0 * rho**4 - 6.0 * rho**2 + 1.0)
    if n == 4 and m == 2:
        return np.sqrt(10.0) * (4.0 * rho**2 - 3.0) * rho**2 * np.cos(2.0 * theta)
    if n == 4 and m == -2:
        return np.sqrt(10.0) * (4.0 * rho**2 - 3.0) * rho**2 * np.sin(2.0 * theta)
    if n == 4 and m == 4:
        return np.sqrt(10.0) * rho**4 * np.cos(4.0 * theta)
    if n == 4 and m == -4:
        return np.sqrt(10.0) * rho**4 * np.sin(4.0 * theta)
    return np.zeros_like(rho)


def build_zernike_basis_functions(n_terms: int = 15, radius: float = 1.0) -> List[BasisFunction]:
    """
    Build canonical Zernike basis functions in Cartesian coordinates.

    Parameters
    ----------
    n_terms : int
        Number of modes to return, starting from Z1.
    radius : float
        Physical or pixel radius used to normalize `x` and `y`.
    """
    R = float(radius)
    if R <= 0.0:
        raise ValueError("radius must be positive")

    c_defocus = np.sqrt(3.0)
    c_astig = np.sqrt(6.0)
    c_coma = np.sqrt(8.0)
    c_spherical = np.sqrt(5.0)
    c_order4 = np.sqrt(10.0)

    basis = [
        BasisFunction(
            "Z1 (Piston)",
            lambda x, y: np.ones_like(x, dtype=float),
            lambda x, y: np.zeros_like(x, dtype=float),
            lambda x, y: np.zeros_like(y, dtype=float),
        ),
        BasisFunction(
            "Z2 (Tilt X)",
            lambda x, y: 2.0 * (x / R),
            lambda x, y: np.full_like(x, 2.0 / R, dtype=float),
            lambda x, y: np.zeros_like(y, dtype=float),
        ),
        BasisFunction(
            "Z3 (Tilt Y)",
            lambda x, y: 2.0 * (y / R),
            lambda x, y: np.zeros_like(x, dtype=float),
            lambda x, y: np.full_like(y, 2.0 / R, dtype=float),
        ),
        BasisFunction(
            "Z4 (Defocus)",
            lambda x, y: c_defocus * (2.0 * ((x**2 + y**2) / R**2) - 1.0),
            lambda x, y: c_defocus * (4.0 * x / R**2),
            lambda x, y: c_defocus * (4.0 * y / R**2),
        ),
        BasisFunction(
            "Z5 (Astig 45)",
            lambda x, y: c_astig * (2.0 * x * y / R**2),
            lambda x, y: c_astig * (2.0 * y / R**2),
            lambda x, y: c_astig * (2.0 * x / R**2),
        ),
        BasisFunction(
            "Z6 (Astig 0)",
            lambda x, y: c_astig * ((x**2 - y**2) / R**2),
            lambda x, y: c_astig * (2.0 * x / R**2),
            lambda x, y: c_astig * (-2.0 * y / R**2),
        ),
        BasisFunction(
            "Z7 (Coma Y)",
            lambda x, y: c_coma * (3.0 * (x**2 + y**2) / R**2 - 2.0) * (y / R),
            lambda x, y: c_coma * (6.0 * x * y / R**3),
            lambda x, y: c_coma * ((3.0 * x**2 + 9.0 * y**2) / R**3 - 2.0 / R),
        ),
        BasisFunction(
            "Z8 (Coma X)",
            lambda x, y: c_coma * (3.0 * (x**2 + y**2) / R**2 - 2.0) * (x / R),
            lambda x, y: c_coma * ((9.0 * x**2 + 3.0 * y**2) / R**3 - 2.0 / R),
            lambda x, y: c_coma * (6.0 * x * y / R**3),
        ),
        BasisFunction(
            "Z9 (Trefoil Y)",
            lambda x, y: c_coma * (3.0 * x**2 * y - y**3) / R**3,
            lambda x, y: c_coma * (6.0 * x * y / R**3),
            lambda x, y: c_coma * (3.0 * x**2 - 3.0 * y**2) / R**3,
        ),
        BasisFunction(
            "Z10 (Trefoil X)",
            lambda x, y: c_coma * (x**3 - 3.0 * x * y**2) / R**3,
            lambda x, y: c_coma * (3.0 * x**2 - 3.0 * y**2) / R**3,
            lambda x, y: c_coma * (-6.0 * x * y / R**3),
        ),
        BasisFunction(
            "Z11 (Spherical)",
            lambda x, y: c_spherical * (6.0 * ((x**2 + y**2) / R**2) ** 2 - 6.0 * ((x**2 + y**2) / R**2) + 1.0),
            lambda x, y: c_spherical * (24.0 * (x**2 + y**2) * x / R**4 - 12.0 * x / R**2),
            lambda x, y: c_spherical * (24.0 * (x**2 + y**2) * y / R**4 - 12.0 * y / R**2),
        ),
        BasisFunction(
            "Z12 (Sec Astig 0)",
            lambda x, y: c_order4 * (4.0 * ((x**2 + y**2) / R**2) - 3.0) * ((x**2 - y**2) / R**2),
            lambda x, y: c_order4 * (2.0 * x / R**2) * (8.0 * x**2 / R**2 - 3.0),
            lambda x, y: c_order4 * (2.0 * y / R**2) * (-8.0 * y**2 / R**2 + 3.0),
        ),
        BasisFunction(
            "Z13 (Sec Astig 45)",
            lambda x, y: c_order4 * (4.0 * ((x**2 + y**2) / R**2) - 3.0) * (2.0 * x * y / R**2),
            lambda x, y: c_order4 * (2.0 * y / R**2) * (12.0 * x**2 / R**2 + 4.0 * y**2 / R**2 - 3.0),
            lambda x, y: c_order4 * (2.0 * x / R**2) * (12.0 * y**2 / R**2 + 4.0 * x**2 / R**2 - 3.0),
        ),
        BasisFunction(
            "Z14 (Tetrafoil X)",
            lambda x, y: c_order4 * (x**4 - 6.0 * x**2 * y**2 + y**4) / R**4,
            lambda x, y: c_order4 * (4.0 * x**3 - 12.0 * x * y**2) / R**4,
            lambda x, y: c_order4 * (4.0 * y**3 - 12.0 * x**2 * y) / R**4,
        ),
        BasisFunction(
            "Z15 (Tetrafoil Y)",
            lambda x, y: c_order4 * (4.0 * x**3 * y - 4.0 * x * y**3) / R**4,
            lambda x, y: c_order4 * (12.0 * x**2 * y - 4.0 * y**3) / R**4,
            lambda x, y: c_order4 * (4.0 * x**3 - 12.0 * x * y**2) / R**4,
        ),
    ]
    return basis[: max(0, min(int(n_terms), len(basis)))]


def compute_zernike_basis_matrix(
    shape: Tuple[int, int],
    centers: np.ndarray,
    n_modes: int = 15,
) -> np.ndarray:
    """
    Compute the canonical Zernike basis matrix evaluated at spot centers.

    Parameters
    ----------
    shape : tuple
        Image shape `(H, W)` used to derive the normalization radius.
    centers : ndarray
        Spot centers, shape `(n_spots, 2)` as `(x_px, y_px)`.
    n_modes : int
        Number of Zernike modes to use, starting from Z1.
    """
    h, w = shape
    centers_xy = np.asarray(centers, dtype=float).reshape(-1, 2)
    basis_functions = build_zernike_basis_functions(
        n_terms=n_modes,
        radius=0.5 * float(min(h, w)),
    )
    if centers_xy.size == 0:
        return np.zeros((0, len(basis_functions)), dtype=float)

    x = centers_xy[:, 0] - w / 2.0
    y = centers_xy[:, 1] - h / 2.0
    basis = np.zeros((centers_xy.shape[0], len(basis_functions)), dtype=float)
    for idx, bf in enumerate(basis_functions):
        basis[:, idx] = bf.phase(x, y)
    return basis


def zernike_coeffs_to_shifts(
    coeffs: np.ndarray,
    basis: np.ndarray,
    n_frames: int,
    n_spots: int,
) -> np.ndarray:
    """
    Convert Zernike coefficients to per-spot, per-frame shifts.
    """
    n_modes = basis.shape[1]
    coeffs_3d = np.asarray(coeffs, dtype=float).reshape((n_frames, 2, n_modes))
    shifts = np.zeros((n_frames, n_spots, 2), dtype=float)
    for t in range(n_frames):
        shifts[t, :, 0] = basis @ coeffs_3d[t, 0, :]
        shifts[t, :, 1] = basis @ coeffs_3d[t, 1, :]
    return shifts


__all__ = [
    "BasisFunction",
    "ZERNIKE_INDICES",
    "ZERNIKE_MODES",
    "ZernikeMode",
    "build_zernike_basis_functions",
    "compute_zernike_basis_matrix",
    "zernike_basis_func",
    "zernike_coeffs_to_shifts",
]
