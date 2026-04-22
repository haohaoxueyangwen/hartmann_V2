from __future__ import annotations
"""
激光 4f 中继模式 — 有限尺寸 Hartmann 掩膜。

与现有 hartmann_simulator.py 的 make_hartmann_mask 的关键区别:
- 只在 mask_window 范围内生成孔阵（不是无限平铺）
- mask_window 边界跟随板一起平移
"""
import numpy as np
from typing import Tuple


def make_finite_hartmann_centers(mask_window: float,
                                 pitch: float,
                                 offset: Tuple[float, float] = (0.0, 0.0)
                                 ) -> np.ndarray:
    """
    生成有限掩膜内的子孔径中心坐标。

    参数:
        mask_window: 板物理尺寸 (m)
        pitch: 子孔径间距 (m)
        offset: 整体偏移 (m)

    返回:
        centers: (N, 2) 数组，每行 (x, y)
    """
    n = int(np.floor(mask_window / pitch))
    if n < 1:
        return np.zeros((0, 2), dtype=float)
    coords = (np.arange(n) - (n - 1) / 2.0) * pitch
    cx = coords + offset[0]
    cy = coords + offset[1]
    # 只保留在 mask_window/2 内的
    half = mask_window / 2.0
    cx = cx[np.abs(cx) <= half]
    cy = cy[np.abs(cy) <= half]
    centers = np.array([(x, y) for y in cy for x in cx], dtype=float)
    return centers


def make_finite_hartmann_mask(X: np.ndarray, Y: np.ndarray,
                               mask_window: float,
                               pitch: float,
                               hole_diameter: float,
                               shift_x_m: float = 0.0,
                               shift_y_m: float = 0.0,
                               shape: str = 'circle') -> np.ndarray:
    """
    有限尺寸 Hartmann 掩膜。

    板的有限支撑和孔阵一起平移。mask_window 外严格为零。

    参数:
        X, Y: 计算网格 (sim_window 尺度)
        mask_window: 板物理尺寸 (m)
        pitch: 子孔径间距 (m)
        hole_diameter: 孔径直径 (m)
        shift_x_m, shift_y_m: 板整体平移 (m)
        shape: 'circle' | 'square'

    返回:
        mask: 与 X 同形状的实数数组，0 或 1
    """
    # 板的有限支撑（跟随平移）
    half = mask_window / 2.0
    board_mask = ((np.abs(X - shift_x_m) <= half) &
                  (np.abs(Y - shift_y_m) <= half))

    # 在板局部坐标系中生成中心
    centers = make_finite_hartmann_centers(mask_window, pitch)
    if len(centers) == 0:
        return np.zeros_like(X, dtype=float)

    radius = hole_diameter / 2.0
    mask = np.zeros_like(X, dtype=float)

    for cx, cy in centers:
        # 中心跟随板平移
        cx_abs = cx + shift_x_m
        cy_abs = cy + shift_y_m
        if shape == 'circle':
            hole = ((X - cx_abs)**2 + (Y - cy_abs)**2) <= radius**2
        elif shape == 'square':
            hole = (np.abs(X - cx_abs) <= radius) & (np.abs(Y - cy_abs) <= radius)
        else:
            raise ValueError(f"Unknown shape: {shape}")
        mask[hole] = 1.0

    # 乘以板边界
    mask *= board_mask

    return mask
