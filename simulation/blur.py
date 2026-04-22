#!/usr/bin/env python3
from __future__ import annotations
"""
统一的模糊工具模块

- apply_gaussian_blur: 使用 scipy.ndimage.gaussian_filter（若可用）对探测器强度图像进行模糊
- gaussian_blur: 可分离一维核实现的二维高斯模糊（无第三方依赖），用于掩模/图像平滑
"""

from typing import Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)


def apply_gaussian_blur(intensity: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    对探测器强度图像应用高斯模糊（优先使用 scipy）。

    参数:
    -----
    intensity : np.ndarray
        输入的强度分布，形状为 (ny, nx)
    sigma : float, 默认 2.0
        高斯核标准差（像素）

    返回:
    -----
    np.ndarray
        模糊后的强度分布，形状与输入相同
    """
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(intensity, sigma=sigma)
    except ImportError:
        logger.warning("scipy 不可用，跳过高斯模糊处理")
        return intensity.copy()


def gaussian_blur(arr: np.ndarray, sigma_pix: Optional[float] = 1.0) -> np.ndarray:
    """
    可分离一维核实现的二维高斯模糊（无依赖）。

    参数:
    ----
    arr : np.ndarray
        输入的二维数组
    sigma_pix : float, 默认 1.0
        高斯核的标准差（像素）

    返回:
    ----
    np.ndarray
        模糊后的数组，与输入同形状
    """
    if sigma_pix is None or sigma_pix <= 0:
        return arr.copy()

    s = float(sigma_pix)
    rad = int(np.ceil(3.0 * s))
    N = 2 * rad + 1
    x = np.arange(N) - rad
    k = np.exp(-(x**2) / (2.0 * s * s))
    k /= np.sum(k)

    arr = np.asarray(arr, dtype=float)

    # 沿 x 方向卷积（列）
    pad_x = ((0, 0), (rad, rad))
    arr_xpad = np.pad(arr, pad_x, mode='reflect')
    out_x = np.empty_like(arr, dtype=float)
    for i in range(arr.shape[0]):
        out_x[i, :] = np.convolve(arr_xpad[i, :], k, mode='valid')

    # 沿 y 方向卷积（行）
    pad_y = ((rad, rad), (0, 0))
    out_x_ypad = np.pad(out_x, pad_y, mode='reflect')
    out = np.empty_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        out[:, j] = np.convolve(out_x_ypad[:, j], k, mode='valid')

    return out


__all__ = ["apply_gaussian_blur", "gaussian_blur"]