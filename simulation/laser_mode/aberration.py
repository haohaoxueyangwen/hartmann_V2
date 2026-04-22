from __future__ import annotations
"""
激光 4f 中继模式 — Zernike 多项式与随机相位屏。

自包含实现，不依赖 simulation/aberration.py 或 analysis/zernike_basis.py。
"""
import numpy as np
from math import factorial
from typing import Tuple

# Noll index → (n, m) 映射表，Noll 1~15
_NOLL_TABLE = [
    (0, 0),    # 1: Piston
    (1, 1),    # 2: Tilt X
    (1, -1),   # 3: Tilt Y
    (2, 0),    # 4: Defocus
    (2, -2),   # 5: Astig 45
    (2, 2),    # 6: Astig 0
    (3, -1),   # 7: Coma Y
    (3, 1),    # 8: Coma X
    (3, -3),   # 9: Trefoil Y
    (3, 3),    # 10: Trefoil X
    (4, 0),    # 11: Spherical
    (4, 2),    # 12
    (4, -2),   # 13
    (4, 4),    # 14
    (4, -4),   # 15
]


def _radial_polynomial(n: int, m_abs: int, rho: np.ndarray) -> np.ndarray:
    """Zernike 径向多项式 R_n^|m|(rho)。"""
    R = np.zeros_like(rho, dtype=float)
    for s in range((n - m_abs) // 2 + 1):
        coeff = ((-1)**s * factorial(n - s) /
                 (factorial(s) *
                  factorial((n + m_abs) // 2 - s) *
                  factorial((n - m_abs) // 2 - s)))
        R = R + coeff * rho**(n - 2 * s)
    return R


def zernike_noll(noll_index: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    计算 Noll 排序的 Zernike 多项式。

    参数:
        noll_index: Noll 索引 (1-based, 1~15)
        rho: 归一化径向坐标
        theta: 角坐标 (rad)

    返回:
        Z: 与 rho 同形状的数组
    """
    if noll_index < 1 or noll_index > len(_NOLL_TABLE):
        return np.zeros_like(rho)
    n, m = _NOLL_TABLE[noll_index - 1]
    if (n - abs(m)) % 2 != 0:
        return np.zeros_like(rho)
    R = _radial_polynomial(n, abs(m), rho)
    if m >= 0:
        return R * np.cos(m * theta)
    else:
        return R * np.sin(abs(m) * theta)


def generate_random_phase_screen(X: np.ndarray, Y: np.ndarray,
                                  pupil_radius: float,
                                  max_order: int = 15,
                                  amplitude_range: Tuple[float, float] = (-1.0, 1.0)
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 Zernike 多项式生成随机相位屏。

    跳过 piston (Z1) 和 tilt (Z2, Z3)。
    rho 裁到 [0, 1]，pupil 外相位为零。

    参数:
        X, Y: 坐标网格 (m)
        pupil_radius: 归一化半径 (m)
        max_order: 最大 Noll 索引 (1-based)
        amplitude_range: 系数随机范围

    返回:
        (phase_screen, coefficients)
    """
    rho = np.sqrt(X**2 + Y**2) / pupil_radius
    theta = np.arctan2(Y, X)
    pupil = rho <= 1.0
    rho_clipped = np.clip(rho, 0, 1.0)

    n_modes = min(max_order, len(_NOLL_TABLE))
    coeffs = np.random.uniform(amplitude_range[0], amplitude_range[1], size=n_modes)
    # 跳过 piston、tilt
    coeffs[0] = 0.0   # Z1 piston
    coeffs[1] = 0.0   # Z2 tilt X
    coeffs[2] = 0.0   # Z3 tilt Y

    phase = np.zeros_like(X, dtype=float)
    for i in range(n_modes):
        if coeffs[i] == 0.0:
            continue
        Z = zernike_noll(i + 1, rho_clipped, theta)
        phase += coeffs[i] * Z

    # pupil 外相位置零
    phase *= pupil

    return phase, coeffs
