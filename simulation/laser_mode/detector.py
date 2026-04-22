from __future__ import annotations
"""
激光 4f 中继模式 — 探测器采样。

自包含实现，支持 nearest / bilinear / srw 模式。
不依赖 simulation/detector.py。
"""
import numpy as np
from typing import Optional


def _resample_bilinear(intensity: np.ndarray,
                       dx_in: float,
                       pixel_out: float,
                       nx_out: int,
                       ny_out: int,
                       order: int = 1) -> np.ndarray:
    """
    插值重采样到目标像素网格。

    输入场以原点为中心，输出探测器也以原点为中心。
    order: 0=nearest, 1=bilinear
    """
    from scipy.ndimage import map_coordinates

    ny_in, nx_in = intensity.shape

    # 输出像素中心坐标 (m)
    x_out = (np.arange(nx_out) - nx_out / 2.0 + 0.5) * pixel_out
    y_out = (np.arange(ny_out) - ny_out / 2.0 + 0.5) * pixel_out

    # 转换为输入数组的浮点索引
    # 输入坐标: pixel i 对应 x = (i - nx_in/2) * dx_in
    # 所以 i = x / dx_in + nx_in / 2
    ix = x_out / dx_in + nx_in / 2.0
    iy = y_out / dx_in + ny_in / 2.0

    IX, IY = np.meshgrid(ix, iy)
    coords = np.array([IY.ravel(), IX.ravel()])

    result = map_coordinates(intensity, coords, order=order, mode='constant', cval=0.0)
    return result.reshape(ny_out, nx_out)


def _bin_array(arr: np.ndarray, bin_size: int) -> np.ndarray:
    """将 2D 数组按 bin_size×bin_size 块求和。"""
    if bin_size <= 1:
        return arr
    ny, nx = arr.shape
    # 裁切到 bin_size 的整数倍
    ny_trim = (ny // bin_size) * bin_size
    nx_trim = (nx // bin_size) * bin_size
    arr_trim = arr[:ny_trim, :nx_trim]
    return arr_trim.reshape(ny_trim // bin_size, bin_size,
                            nx_trim // bin_size, bin_size).sum(axis=(1, 3))


def sample_detector(intensity: np.ndarray,
                    dx: float,
                    pixel_size: float,
                    detector_nx: int,
                    detector_ny: int,
                    mode: str = 'srw',
                    upsample_factor: int = 2,
                    bin_size: Optional[int] = None,
                    interpolation: str = 'bilinear') -> np.ndarray:
    """
    将传播网格上的强度分布采样到探测器像素。

    参数:
        intensity: 传播网格强度 (ny_in, nx_in)
        dx: 传播网格间距 (m)
        pixel_size: 探测器像素尺寸 (m)
        detector_nx, detector_ny: 探测器像素数
        mode: 'nearest' | 'bilinear' | 'srw'
        upsample_factor: srw 模式上采样倍数
        bin_size: srw 模式 binning 尺寸 (默认 = upsample_factor)
        interpolation: 插值方法 'bilinear' | 'nearest'

    返回:
        detector_image: (detector_ny, detector_nx)
    """
    mode_l = str(mode).lower()
    interp_order = 1 if interpolation == 'bilinear' else 0

    if mode_l == 'srw':
        if bin_size is None:
            bin_size = upsample_factor
        fine_pixel = pixel_size / float(upsample_factor)
        nx_fine = detector_nx * upsample_factor
        ny_fine = detector_ny * upsample_factor
        D_fine = _resample_bilinear(intensity, dx, fine_pixel, nx_fine, ny_fine,
                                     order=interp_order)
        D = _bin_array(D_fine, bin_size)
        return D

    elif mode_l == 'bilinear':
        return _resample_bilinear(intensity, dx, pixel_size, detector_nx, detector_ny,
                                   order=interp_order)

    elif mode_l == 'nearest':
        # 最近邻：直接取最近的输入像素
        ny_in, nx_in = intensity.shape
        x_out = (np.arange(detector_nx) - detector_nx / 2.0 + 0.5) * pixel_size
        y_out = (np.arange(detector_ny) - detector_ny / 2.0 + 0.5) * pixel_size
        ix = np.clip(np.round(x_out / dx + nx_in / 2.0).astype(int), 0, nx_in - 1)
        iy = np.clip(np.round(y_out / dx + ny_in / 2.0).astype(int), 0, ny_in - 1)
        return intensity[np.ix_(iy, ix)]

    else:
        raise ValueError(f"Unknown sampling mode: {mode}")
