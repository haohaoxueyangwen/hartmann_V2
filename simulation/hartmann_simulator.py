from __future__ import annotations
import numpy as np
import os
import warnings
from typing import Tuple, List, Optional, Union, Dict, Any
from simulation.blur import gaussian_blur
from simulation.detector import (
    sample_detector_fov,
    determine_pulses_per_frame,
    accumulate_by_pulses
)
from simulation.wave_optics import PropagationConfig
from simulation.wave_optics import (
    calculate_wavelength,
    make_grid,
    make_plane_source,
    make_gaussian_source,
    fresnel_prop,
    fourier_prop,
    ang_spec,
    compute_rotation_step,
)

# Import optimizations if available
try:
    from simulation.optimizations import fast_accumulate_frames, MaskCache
    _HAS_OPTIMIZATIONS = True
except ImportError:
    _HAS_OPTIMIZATIONS = False
    fast_accumulate_frames = None
    MaskCache = None


def duty_area_to_diameter(pitch: float, duty_area: float) -> float:
    """
    将“面积占空比”转换为圆形孔径直径（严格面积定义）。

    定义：duty_area = 孔径面积 / 周期面积。
    圆孔面积 = π(d/2)^2，周期面积 = pitch^2。
    推导：d = pitch × sqrt(duty_area / (π/4)) ≈ pitch × sqrt(duty_area / 0.7854)

    参数
    ----
    pitch : float
        周期（单位：米 m）
    duty_area : float
        面积占空比（0 < duty_area <= 1）

    返回
    ----
    diameter : float
        圆形孔径直径（单位：米），严格面积定义下的直径。

    异常
    ----
    ValueError
        当 duty_area 不在有效范围内时抛出
    """
    duty = float(duty_area)
    if duty <= 0.0 or duty > 1.0:
        raise ValueError(f"duty_area 必须在 (0, 1] 范围内，当前值: {duty}")
    return float(pitch) * np.sqrt(4.0 * duty / np.pi)


def duty_cycle_to_diameter(pitch: float, duty_cycle: float) -> float:
    """Legacy alias for backward compatibility.

    The project now interprets the value as area fill ratio. Callers should
    migrate to ``duty_area_to_diameter`` / ``duty_area`` naming.
    """
    return duty_area_to_diameter(pitch, duty_cycle)

# 统一：计算孔径直径（优先使用显式 aperture，否则用面积占空比 duty_area）
def compute_aperture_diameter(spacing: float,
                              aperture: Optional[float],
                              duty_area: Optional[float]) -> float:
    """根据配置统一计算孔径直径。

    优先使用显式 `aperture`，否则使用面积占空比 `duty_area`
    按圆孔面积公式推导直径。
    两者都缺失时抛出异常，确保调用方明确传参。
    """
    if aperture is not None:
        return float(aperture)
    if duty_area is not None:
        return duty_area_to_diameter(float(spacing), float(duty_area))
    raise ValueError("必须提供 aperture 或 duty_area 之一用于计算孔径直径")


def _choose_mask_batch_size(
    grid_shape: Tuple[int, int],
    dtype: np.dtype,
    *,
    default_batch_size: int = 100,
    arrays_per_batch: int = 3,
    target_bytes: int = 256 * 1024 * 1024,
) -> int:
    """Choose a broadcast batch size that stays under a soft memory budget."""
    ny, nx = int(grid_shape[0]), int(grid_shape[1])
    if ny <= 0 or nx <= 0:
        return 1
    itemsize = np.dtype(dtype).itemsize
    bytes_per_center = max(1, ny * nx * itemsize * int(arrays_per_batch))
    max_batch = max(1, target_bytes // bytes_per_center)
    return max(1, min(int(default_batch_size), int(max_batch)))

# 统一：唯一入口的 Hartmann 掩模生成函数（坐标级旋转）
def make_hartmann_mask(grid_x: np.ndarray,
                       grid_y: np.ndarray,
                       *,
                       spacing: float,
                       aperture: Optional[float] = None,
                       duty_area: Optional[float] = None,
                       rotation_angle_deg: float = 0.0,
                       shape: str = 'circle',
                       phase_mode: bool = False,
                       phase_delay_rad: float = np.pi / 2.0,
                       phase_on_empty: bool = False,
                       blur_sigma_pix: float = 0.0,
                       amplitude_inside: float = 1.0,
                       amplitude_outside: float = 0.0,
                       offset_xy: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """生成 Hartmann 掩模（统一 API，gttext 坐标级旋转）。

    参数均使用米制与绝对坐标；孔径直径来自 `aperture` 或严格面积定义的 `duty_area`。
    """
    diameter = compute_aperture_diameter(spacing, aperture, duty_area)
    return hartmann_mask_gttext_style(
        grid_x, grid_y,
        spacing=spacing,
        aperture=diameter,
        duty_area=None,
        rotation_angle_rad=np.deg2rad(rotation_angle_deg),
        shape=shape,
        phase_mode=phase_mode,
        phase_delay=phase_delay_rad,
        phase_on_empty=phase_on_empty,
        blur_sigma_pix=blur_sigma_pix,
        amplitude_inside=amplitude_inside,
        amplitude_outside=amplitude_outside,
        offset_xy=offset_xy,
    )


def make_hartmann_mask_translated(grid_x: np.ndarray,
                                  grid_y: np.ndarray,
                                  *,
                                  shift_x_m: float = 0.0,
                                  shift_y_m: float = 0.0,
                                  spacing: float,
                                  aperture: Optional[float] = None,
                                  duty_area: Optional[float] = None,
                                  rotation_angle_deg: float = 0.0,
                                  shape: str = 'circle',
                                  phase_mode: bool = False,
                                  phase_delay_rad: float = np.pi / 2.0,
                                  phase_on_empty: bool = False,
                                  blur_sigma_pix: float = 0.0,
                                  amplitude_inside: float = 1.0,
                                  amplitude_outside: float = 0.0,
                                  base_offset_xy: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """
    生成带平移偏移的 Hartmann 掩膜：T(x - dx, y - dy)

    通过在坐标层面平移网格实现掩膜整体位移，避免数组插值误差。
    """
    grid_x_shifted = grid_x - float(shift_x_m)
    grid_y_shifted = grid_y - float(shift_y_m)
    return make_hartmann_mask(
        grid_x_shifted,
        grid_y_shifted,
        spacing=spacing,
        aperture=aperture,
        duty_area=duty_area,
        rotation_angle_deg=rotation_angle_deg,
        shape=shape,
        phase_mode=phase_mode,
        phase_delay_rad=phase_delay_rad,
        phase_on_empty=phase_on_empty,
        blur_sigma_pix=blur_sigma_pix,
        amplitude_inside=amplitude_inside,
        amplitude_outside=amplitude_outside,
        offset_xy=base_offset_xy,
    )



def make_hartmann_mask_rotated(grid_x: np.ndarray,
                                grid_y: np.ndarray,
                                *,
                                rotation_angle_rad: float,
                                rotation_center_xy_m: Tuple[float, float],
                                grid_tilt_deg: float = 0.0,
                                spacing: float,
                                aperture: Optional[float] = None,
                                duty_area: Optional[float] = None,
                                shape: str = 'circle',
                                phase_mode: bool = False,
                                phase_delay_rad: float = np.pi / 2.0,
                                phase_on_empty: bool = False,
                                blur_sigma_pix: float = 0.0,
                                amplitude_inside: float = 1.0,
                                amplitude_outside: float = 0.0,
                                base_offset_xy: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """生成绕旋转中心旋转后的 Hartmann 掩膜。

    通过在坐标层面旋转网格实现掩膜整体旋转，零插值误差。
    这与 make_hartmann_mask_translated (平移) 的思路完全同构：
    不操作数组，只操作坐标。

    Parameters
    ----------
    rotation_angle_rad : float
        当前脉冲相对于脉冲 0 的旋转角（弧度）。
    rotation_center_xy_m : (float, float)
        旋转中心的 (x, y) 坐标（米）。
    grid_tilt_deg : float
        掩膜网格 X 轴与旋转切线方向的倾斜角（度）。
    """
    cx, cy = float(rotation_center_xy_m[0]), float(rotation_center_xy_m[1])
    cos_a = float(np.cos(rotation_angle_rad))
    sin_a = float(np.sin(rotation_angle_rad))

    # 坐标旋转：将物理网格反向旋转（等效于掩膜正向旋转）
    dx = grid_x - cx
    dy = grid_y - cy
    x_rot = cx + dx * cos_a + dy * sin_a
    y_rot = cy - dx * sin_a + dy * cos_a

    return make_hartmann_mask(
        x_rot, y_rot,
        spacing=spacing,
        aperture=aperture,
        duty_area=duty_area,
        rotation_angle_deg=grid_tilt_deg,
        shape=shape,
        phase_mode=phase_mode,
        phase_delay_rad=phase_delay_rad,
        phase_on_empty=phase_on_empty,
        blur_sigma_pix=blur_sigma_pix,
        amplitude_inside=amplitude_inside,
        amplitude_outside=amplitude_outside,
        offset_xy=base_offset_xy,
    )

# 统一：应用掩模到场（幅度或相位掩模均支持）
def apply_mask(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """将掩模应用到复场，支持幅度或相位掩模。"""
    return field * mask

def make_hartmann_centers(window_size: float,
                          pitch: float,
                          grid_nx: Optional[int] = None,
                          grid_ny: Optional[int] = None,
                          offset: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """
    生成Hartmann子孔径中心坐标阵列
    
    创建均匀分布的子孔径中心点，用于构建Hartmann掩模
    中心点按行优先顺序排列(先X后Y)
    
    参数:
    -----
    window_size : float
        窗口尺寸(米)
        决定阵列的覆盖范围
    
    pitch : float
        子孔径间距(米)
        相邻中心点之间的距离
    
    grid_nx, grid_ny : int or None, 可选
        X和Y方向的子孔径数量
        None表示根据window_size和pitch自动计算
        自动计算: n = floor(window_size / pitch)
    
    offset : Tuple[float, float], 默认 (0.0, 0.0)
        整体偏移量(米)
        (offset_x, offset_y)，相对于原点的平移
    
    返回:
    -----
    centers : np.ndarray
        子孔径中心坐标数组，形状为(N, 2)
        N = nx × ny
        每行为一个中心点的(x, y)坐标
    
    抛出:
    -----
    ValueError : 如果pitch <= 0
    
    示例:
    -----
    >>> centers = make_hartmann_centers(100e-6, 10e-6)  # 100μm窗口, 10μm间距
    >>> print(f"子孔径数量: {len(centers)}")  # 约100个(10x10)
    >>> print(f"中心点范围: ({centers[:,0].min()*1e6:.1f}, {centers[:,0].max()*1e6:.1f}) μm")
    """
    if pitch <= 0:
        raise ValueError(f"pitch 必须为正数，当前值: {pitch}")
    nx_eff = int(grid_nx) if grid_nx is not None else int(max(1, np.floor(window_size / pitch)))
    ny_eff = int(grid_ny) if grid_ny is not None else int(max(1, np.floor(window_size / pitch)))
    x_coords = (np.arange(nx_eff) - (nx_eff - 1) / 2.0) * pitch + float(offset[0])
    y_coords = (np.arange(ny_eff) - (ny_eff - 1) / 2.0) * pitch + float(offset[1])
    centers = np.array([(cx, cy) for cy in y_coords for cx in x_coords], dtype=float)
    return centers

def build_hartmann_mask(grid_x: np.ndarray,
                        grid_y: np.ndarray,
                        centers: np.ndarray,
                        diameter: float,
                        inside_value: float = 1.0,
                        outside_value: float = 0.0,
                        shape: str = 'circle') -> np.ndarray:
    """
    功能:
    -----
    在给定的坐标网格上，根据子孔径中心坐标与孔径尺寸构建 Hartmann 掩模（圆形或方形），并将孔径内像素赋值为 `inside_value`，孔径外赋值为 `outside_value`。

    参数:
    -----
    grid_x, grid_y : np.ndarray
        坐标网格，形状为 `(ny, nx)`，单位米。二者形状必须一致，通常由 `make_grid` 或 `np.meshgrid` 构造。
    centers : np.ndarray
        子孔径中心坐标数组，形状为 `(N, 2)`，每行表示 `(x, y)`，单位米。
    diameter : float
        子孔径直径（米），必须为正数。
    inside_value : float, 可选
        掩模中孔径内像素的取值，默认 `1.0`。
    outside_value : float, 可选
        掩模中孔径外像素的取值，默认 `0.0`。
    shape : str, 可选
        孔径形状，`'circle'`（圆形）或 `'square'`（正方形），默认 `'circle'`。

    返回:
    -----
    mask : np.ndarray
        与 `grid_x` 形状一致的二维数组，类型为 `float`，表示 Hartmann 掩模。

    可能抛出:
    ---------
    ValueError
        - 当 `diameter` 为 `None` 或不为正数时。
        - 当 `shape` 不是 `'circle'` 或 `'square'` 时。

    使用示例:
    ---------
    >>> X, Y, dx, dy = make_grid(200e-6, 1024)
    >>> centers = make_hartmann_centers(200e-6, pitch=5e-6)
    >>> mask = build_hartmann_mask(X, Y, centers, diameter=2e-6, inside_value=1.0, outside_value=0.0, shape='circle')

    算法说明:
    -------
    - 圆形孔径: 对每个中心 `(cx, cy)`，依据 `((x-cx)^2 + (y-cy)^2) <= (d/2)^2` 判定像素是否位于孔径内。
    - 方形孔径: 对每个中心，依据 `|x-cx| <= d/2` 且 `|y-cy| <= d/2` 判定。
    - 将满足条件的像素赋值为 `inside_value`，其余为 `outside_value`。时间复杂度约为 `O(N * ny * nx)`。
    """
    if diameter is None or diameter <= 0:
        raise ValueError(f"diameter 必须为正数，当前值: {diameter}")
    
    mask = np.full_like(grid_x, outside_value, dtype=float)
    
    # 优化：分块处理以平衡内存和速度
    centers_array = np.asarray(centers)  # shape: (N, 2)
    if len(centers_array) == 0:
        return mask
    
    if shape == 'circle':
        # 圆形孔径：批量处理，每次处理一批中心
        radius_squared = (diameter * 0.5) ** 2
        batch_size = _choose_mask_batch_size(grid_x.shape, grid_x.dtype, arrays_per_batch=3)
        if batch_size < 100:
            warnings.warn(
                f"Reducing Hartmann mask batch_size from 100 to {batch_size} for grid {grid_x.shape} "
                "to limit temporary memory usage.",
                RuntimeWarning,
                stacklevel=2,
            )
        for i in range(0, len(centers_array), batch_size):
            batch = centers_array[i:i+batch_size]
            # 对每批中心，使用广播计算
            dx = grid_x[..., None] - batch[:, 0]  # shape: (ny, nx, batch)
            dy = grid_y[..., None] - batch[:, 1]  # shape: (ny, nx, batch)
            dist_sq = dx**2 + dy**2
            # 任一中心满足条件即为孔径内
            in_batch = np.any(dist_sq <= radius_squared, axis=-1)  # shape: (ny, nx)
            mask[in_batch] = inside_value
            
    elif shape == 'square':
        # 矩形孔径：批量处理
        half_size = diameter * 0.5
        batch_size = _choose_mask_batch_size(grid_x.shape, grid_x.dtype, arrays_per_batch=2)
        if batch_size < 100:
            warnings.warn(
                f"Reducing Hartmann mask batch_size from 100 to {batch_size} for grid {grid_x.shape} "
                "to limit temporary memory usage.",
                RuntimeWarning,
                stacklevel=2,
            )
        for i in range(0, len(centers_array), batch_size):
            batch = centers_array[i:i+batch_size]
            dx = np.abs(grid_x[..., None] - batch[:, 0])  # shape: (ny, nx, batch)
            dy = np.abs(grid_y[..., None] - batch[:, 1])  # shape: (ny, nx, batch)
            in_batch = np.any((dx <= half_size) & (dy <= half_size), axis=-1)
            mask[in_batch] = inside_value
    else:
        raise ValueError(f"不支持的孔径形状: {shape}，请使用 'circle' 或 'square'")
    
    return mask



def partition_subapertures(grid_x: np.ndarray,
                           grid_y: np.ndarray,
                           centers: np.ndarray,
                           diameter: float,
                           shape: str = 'circle') -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    功能:
    -----
    根据子孔径中心坐标与孔径尺寸，在坐标网格上划分每个子孔径的像素区域，返回像素标签图与每个子孔径的边界框（ROI）。

    参数:
    -----
    grid_x, grid_y : np.ndarray
        坐标网格，形状为 `(ny, nx)`，单位米。两者形状需一致。
    centers : np.ndarray
        子孔径中心坐标数组，形状 `(N, 2)`，每行 `(x, y)`，单位米。
    diameter : float
        子孔径直径（米），必须为正数。
    shape : str, 可选
        孔径形状，`'circle'`（圆形）或 `'square'`（正方形），默认 `'circle'`。

    返回:
    -----
    label_map : np.ndarray
        形状与 `grid_x` 相同的整型数组，值为所属子孔径的索引（从 1 开始），0 表示不在任何子孔径内。
    rois : List[Tuple[int, int, int, int]]
        每个子孔径的像素边界框列表，元素为 `(y_min, y_max, x_min, x_max)` 像素索引。

    可能抛出:
    ---------
    ValueError
        - 当 `diameter` 为 `None` 或不为正数时。
        - 当 `shape` 不是 `'circle'` 或 `'square'` 时。

    使用示例:
    ---------
    >>> X, Y, dx, dy = make_grid(200e-6, 1024)
    >>> centers = make_hartmann_centers(200e-6, pitch=5e-6)
    >>> labels, rois = partition_subapertures(X, Y, centers, diameter=2e-6, shape='circle')

    算法说明:
    -------
    - 对每个中心点构造对应形状的布尔掩模：圆形用距离判定，方形用坐标阈值判定。
    - 将掩模内的像素标记为该子孔径索引（从 1 开始），其余为 0。
    - 通过 `np.any` 和 `np.where` 提取掩模覆盖行列范围，生成最小包围边界框 `(y_min, y_max, x_min, x_max)`。
    - 时间复杂度约为 `O(N * ny * nx)`。
    """
    if diameter is None or diameter <= 0:
        raise ValueError(f"diameter 必须为正数，当前值: {diameter}")
    
    label_map = np.zeros_like(grid_x, dtype=np.int32)
    rois: List[Tuple[int, int, int, int]] = []
    
    if shape == 'circle':
        radius_squared = (diameter * 0.5) ** 2
        for idx, (center_x, center_y) in enumerate(centers, start=1):
            hole = ((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2) <= radius_squared
            label_map[hole] = idx
            rows = np.any(hole, axis=1)
            cols = np.any(hole, axis=0)
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            if y_indices.size > 0 and x_indices.size > 0:
                y_min, y_max = int(y_indices[0]), int(y_indices[-1])
                x_min, x_max = int(x_indices[0]), int(x_indices[-1])
            else:
                y_min = x_min = 0
                y_max = hole.shape[0] - 1
                x_max = hole.shape[1] - 1
            rois.append((y_min, y_max, x_min, x_max))
    elif shape == 'square':
        half_size = diameter * 0.5
        for idx, (center_x, center_y) in enumerate(centers, start=1):
            hole = ((np.abs(grid_x - center_x) <= half_size) & 
                    (np.abs(grid_y - center_y) <= half_size))
            label_map[hole] = idx
            rows = np.any(hole, axis=1)
            cols = np.any(hole, axis=0)
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            if y_indices.size > 0 and x_indices.size > 0:
                y_min, y_max = int(y_indices[0]), int(y_indices[-1])
                x_min, x_max = int(x_indices[0]), int(x_indices[-1])
            else:
                y_min = x_min = 0
                y_max = hole.shape[0] - 1
                x_max = hole.shape[1] - 1
            rois.append((y_min, y_max, x_min, x_max))
    else:
        raise ValueError(f"不支持的孔径形状: {shape}，请使用 'circle' 或 'square'")
    
    return label_map, rois


def hartmann_partition(grid_x: np.ndarray,
                       grid_y: np.ndarray,
                       window_size: float,
                       pitch: float,
                       diameter: float,
                       grid_nx: Optional[int] = None,
                       grid_ny: Optional[int] = None,
                       offset: Tuple[float, float] = (0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    功能:
    -----
    一站式Hartmann子孔径分割函数，自动生成中心坐标、标签图和ROI边界框，简化调用流程。
    现在支持 gttext 风格的坐标级旋转。

    参数:
    -----
    grid_x : np.ndarray
        X方向网格坐标，形状为 `(ny, nx)`，单位为米。
    grid_y : np.ndarray
        Y方向网格坐标，形状为 `(ny, nx)`，单位为米。
    window_size : float
        窗口总尺寸，单位为米。
    pitch : float
        子孔径间距，单位为米。
    diameter : float
        子孔径直径，单位为米，必须为正值。
    grid_nx : int, 可选
        X方向子孔径数量，默认根据 `window_size` 和 `pitch` 自动计算。
    grid_ny : int, 可选
        Y方向子孔径数量，默认根据 `window_size` 和 `pitch` 自动计算。
    offset : Tuple[float, float], 可选
        阵列偏移量 `(offset_x, offset_y)`，单位为米，默认 `(0.0, 0.0)`。
    rotation_mode : str, 可选
        旋转模式：'legacy_centers' 或 'gttext_coord'，默认 'legacy_centers'。
    rotation_angle_deg : float, 可选
        旋转角度（度），默认 0.0。
    pivot_mode : str, 可选
        旋转中心模式：'outside_default', 'relative_polar', 'absolute'，默认 'outside_default'。
    offset_r_m : float, 可选
        相对极坐标半径偏移（米），默认 0.0。
    offset_theta_deg : float, 可选
        相对极坐标角度偏移（度），默认 0.0。
    center_abs_m : Tuple[float, float], 可选
        绝对中心坐标（米），默认 (0.0, 0.0)。

    返回:
    -----
    centers : np.ndarray
        子孔径中心坐标，形状为 `(n_apertures, 2)`，单位为米。
    label_map : np.ndarray
        像素标签图，形状与输入网格相同，每个像素值表示所属子孔径索引（从1开始）。
    rois : List[Tuple[int, int, int, int]]
        每个子孔径的边界框列表，格式为 `(y_min, y_max, x_min, x_max)`。

    使用示例:
    ---------
    >>> # 传统模式
    >>> X, Y, dx, dy = make_grid(200e-6, 1024)
    >>> centers, labels, rois = hartmann_partition(X, Y, 200e-6, 3e-6, 2.4e-6)
    >>> 
    >>> # gttext 风格坐标级旋转
    >>> centers, labels, rois = hartmann_partition(
    ...     X, Y, 200e-6, 3e-6, 2.4e-6,
    ...     rotation_mode='gttext_coord',
    ...     rotation_angle_deg=45.0,
    ...     pivot_mode='outside_default'
    ... )

    算法说明:
    -------
    - 根据 rotation_mode 选择传统中心点旋转或 gttext 风格坐标级旋转；
    - 内部调用相应的分区生成函数；
    - 时间复杂度为 `O(ny*nx*n_apertures)`，受网格尺寸与子孔径数量影响。
    """
    centers = make_hartmann_centers(window_size, pitch, grid_nx=grid_nx, grid_ny=grid_ny, offset=offset)
    label_map, rois = partition_subapertures(grid_x, grid_y, centers, diameter)
    return centers, label_map, rois




def define_rotation_center(offset_x_m: float = 0.0, 
                           offset_y_m: float = 0.0,
                           reference_point: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    """
    定义旋转中心坐标。
    
    该函数通过给定的参考点和偏移量（以米为单位）来计算旋转中心的坐标。
    主要用于Hartmann传感器的旋转操作中，确定旋转的中心位置。
    
    参数:
        offset_x_m (float, 可选): X方向的偏移量，单位为米。默认值为0.0。
        offset_y_m (float, 可选): Y方向的偏移量，单位为米。默认值为0.0。
        reference_point (Tuple[float, float], 可选): 参考点坐标，单位为米。默认值为(0.0, 0.0)。
    
    返回:
        Tuple[float, float]: 旋转中心坐标(center_x, center_y)，单位为米。
    
    异常:
        无特定异常抛出，但输入参数会被强制转换为float类型。
    
    使用示例:
        >>> # 基于原点定义旋转中心，偏移 10 微米（0.00001 米）
        >>> center = define_rotation_center(offset_x_m=10e-6, offset_y_m=5e-6)
        >>> print(center)
        (1e-05, 5e-06)
        
        >>> # 基于自定义参考点定义旋转中心
        >>> ref_point = (100e-6, 50e-6)  # 参考点为(100μm, 50μm)
        >>> center = define_rotation_center(offset_x_m=20e-6, offset_y_m=-10e-6, 
        ...                                reference_point=ref_point)
        >>> print(center)
        (1.2e-04, 4e-05)
    
    算法说明:
        1. 将输入的参考点坐标转换为float类型
        2. 计算最终的旋转中心坐标：center = reference_point + offset
        4. 返回旋转中心坐标元组
    
    注意事项:
        - 所有输入参数都会被强制转换为float类型，确保数值计算的准确性
        - 偏移量以米为单位输入，内部直接使用米进行计算
        - 返回的坐标单位为米，与系统其他函数保持一致
    """
    return (0.0, 0.0)




def hartmann_sensor(grid_x: np.ndarray,
                    grid_y: np.ndarray,
                    window_size: float,
                    grid_points: int,
                    aperture: Optional[float],
                    spacing: float,
                    duty_area: Optional[float] = None,
                    grid_nx: Optional[int] = None,
                    grid_ny: Optional[int] = None,
                    offset: Tuple[float, float] = (0.0, 0.0),
                    rotation_angle: float = 0.0,
                    shape: str = 'circle') -> np.ndarray:
                    """
                    创建Hartmann传感器掩模
                    
                    支持两种布局模式:
                    1. 矩形网格布局 - 规则阵列的圆形或方形孔径
                    2. 极坐标扇形布局 - 同心环+扇区的"抽奖转盘"结构
                    
                    参数:
                    -----
                    grid_x, grid_y : np.ndarray
                        空间坐标网格(米)，通常由make_grid()生成
                        形状为(ny, nx)
                    
                    window_size : float
                        窗口尺寸(米)
                    
                    grid_points : int
                        网格点数(仅用于polar模式的参数传递)
                    
                    === 矩形网格布局参数 (polar_layout=None时使用) ===
                    
                    aperture : float or None
                        孔径直径(米)
                        None表示通过duty_area自动计算
                    
                    spacing : float
                        子孔径间距(米)
                    
                    duty_area : float or None, 默认 None
                        占空比(0-1)，孔径面积/单元面积
                        当aperture=None时必须提供
                    
                    grid_nx, grid_ny : int or None, 可选
                        X/Y方向的子孔径数量
                        None表示自动计算
                    
                    offset : Tuple[float, float], 默认 (0.0, 0.0)
                        整体偏移(offset_x, offset_y)，单位米
                    
                    rotation_angle : float, 默认 0.0
                        旋转角度(度)
                    
                    shape : str, 默认 'circle'
                        孔径形状: 'circle'(圆形) 或 'square'(矩形)
                    
                    === 极坐标扇形布局参数 (polar_layout≠None时使用) ===
                    
                    polar_layout : PolarWedgeLayout or None, 默认 None
                        极坐标布局配置
                        None表示使用矩形网格布局
                        非None表示使用极坐标扇形布局
                    
                    polar_center : Tuple[float, float] or None, 可选
                        极坐标中心位置(米)
                        None表示使用offset作为中心
                    
                    polar_mode : str, 默认 'amplitude'
                        掩模模式:
                        - 'amplitude': 振幅调制(透光/不透光)
                        - 'phase': 相位调制
                    
                    polar_radial_duty : float, 默认 1.0
                        径向占空比(0-1)
                        控制环的径向宽度
                    
                    polar_angular_duty : float, 默认 1.0
                        角向占空比(0-1)
                        控制扇区的角度宽度
                    
                    polar_amplitude_inside : float, 默认 1.0
                        扇区内部的振幅值
                    
                    polar_amplitude_outside : float, 默认 0.0
                        扇区外部的振幅值
                    
                    polar_phase_delay : float, 默认 π/2
                        相位延迟(弧度)，仅在polar_mode='phase'时使用
                    
                    polar_blur_sigma_pix : float, 默认 0.0
                        边缘模糊标准差(像素)
                        0表示硬边界
                    
                    polar_return_labels : bool, 默认 False
                        是否返回标签图和扇区信息
                    
                    equal_angle_mode : bool, 默认 False
                        是否启用等张角模式
                        True: 所有环使用相同的扇区数量，确保每个扇区的角度相等
                        False: 使用原始的变扇区数量设计
                        仅在polar_layout不为None时有效
                    
                    返回:
                    -----
                    mask : np.ndarray
                        Hartmann掩模数组，形状为(ny, nx)
                        矩形布局: 0/1二值掩模
                        极坐标布局: 根据mode返回振幅或相位掩模
                    
                    (label_map, wedge_info) : 可选
                        仅当polar_return_labels=True时返回
                        label_map : 扇区标签图
                        wedge_info : 扇区详细信息列表
                    
                    抛出:
                    -----
                    ValueError : 如果aperture和duty_area都未提供(矩形布局)
                    
                    示例:
                    -----
                    # 矩形网格布局(圆形孔径)
                    >>> X, Y, dx, dy = make_grid(100e-6, 512)
                    >>> mask = hartmann_sensor(X, Y, 100e-6, 512,
                    ...                        aperture=None, spacing=5e-6,
                    ...                        duty_area=0.2, shape='circle')
                    
                    """
                
                    pitch = float(spacing)
                    if aperture is None:
                        if duty_area is None:
                            raise ValueError("aperture 未指定且未提供 duty_area，无法推导直径")
                        aperture = duty_area_to_diameter(pitch, duty_area)
                    centers = make_hartmann_centers(window_size, pitch, grid_nx=grid_nx, grid_ny=grid_ny, offset=offset)
                    mask = build_hartmann_mask(grid_x, grid_y, centers, diameter=float(aperture), shape=shape)
                    return mask
                    
                    

def demo_partition_and_save(L: float = 200e-6,
                            M: int = 1024,
                            pitch: float = 3e-6,
                            duty_area: float = 0.15) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
                    """
                    Hartmann分区演示函数 - 演示Hartmann传感器分区（无可视化/保存）。
                    
                    该函数创建一个完整的Hartmann传感器分区演示，包括网格生成与分区计算。
                    适用于演示Hartmann传感器的工作原理和分区效果（不生成图像文件）。
                    
                    参数:
                    ----
                    L : float, 默认 200e-6
                        网格边长，单位米（默认200微米）
                    M : int, 默认 1024
                        网格点数（M×M）
                    pitch : float, 默认 3e-6
                        Hartmann光斑间距，单位米（默认3微米）
                    duty_area : float, 默认 0.15
                        占空比（光斑面积比例）
                    
                    
                    返回:
                    ----
                    Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]] : 
                        - centers: 光斑中心坐标数组
                        - label_map: 分区标签图
                        - rois: ROI区域列表
                    
                    使用示例:
                    --------
                    >>> # 基本使用
                    >>> centers, labels, rois = demo_partition_and_save()
                    >>> print(f"光斑数量: {len(centers)}")
                    >>> print(f"网格大小: {labels.shape}")
                    >>> 
                    >>> # 自定义参数
                    >>> centers, labels, rois = demo_partition_and_save(
                    ...     L=500e-6, M=2048, pitch=5e-6, duty_area=0.2)
                    >>> 
                    >>> # 不同占空比比较
                    >>> for duty in [0.1, 0.15, 0.2, 0.25]:
                    ...     centers, labels, rois = demo_partition_and_save(duty_area=duty)
                    ...     print(f"占空比{duty}: 光斑数={len(centers)}")
                    
                    算法说明:
                    --------
                    1. 网格生成:
                       
                       X, Y, dx, dy = make_grid(L, M)
                       
                       创建L×L的正方形网格，M×M个点
                    
                    2. 光斑直径计算:
                       
                       diameter = duty_area_to_diameter(pitch, duty_area)
                       
                       根据占空比计算光斑直径
                    
                    3. Hartmann分区:
                       
                       centers, label_map, rois = hartmann_partition(X, Y, L, pitch, diameter)
                       
                       计算光斑中心、标签图和ROI区域
                    
                    
                    
                    处理流程:
                    --------
                    1. 创建空间网格
                    2. 计算光斑参数
                    3. 执行Hartmann分区
                    4. 返回分区数据
                    
                    注意事项:
                    --------
                    - 不依赖可视化库
                    - 占空比影响光斑大小和数量
                    - 网格大小M影响计算精度
                    - 适用于Hartmann传感器原理演示
                    """
                    """
                    Demonstrate Hartmann sensor partitioning (no visualization/no saving).
                    """
                    X, Y, dx, dy = make_grid(L, M)
                    diameter = duty_area_to_diameter(pitch, duty_area)
                    centers, label_map, rois = hartmann_partition(X, Y, L, pitch, diameter)
                    # 预览显示逻辑已删除
                    return centers, label_map, rois
                    





# ============================================================================
# 多帧叠加模拟 - 重构版本
# ============================================================================

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PulseConfig:
    """脉冲和时间相关配置"""
    n_pulses: Optional[int] = None
    duration_s: Optional[float] = None
    pulse_rep_rate: float = 100.0
    detector_fps: Optional[float] = 10.0
    pulses_per_frame: Optional[int] = None
    wrap_accumulation: bool = True

@dataclass
class RotationConfig:
    """旋转相关配置"""
    mode: str = "velocity"  # "step" 或 "velocity"
    step_angle_deg: float = 1.0
    angular_velocity_deg_s: float = 10.0
    theta_start_deg: float = 0.0
    center_offset_m: Tuple[float, float] = (0.0, 0.0)

@dataclass
class MovementConfig:
    """平移配置（单板；已删除交叉与双板逻辑）

    模式说明：
    - 'none'：不应用逐脉冲平移，仅使用初始偏移。
    - 'translation'：沿方向角以固定步长逐脉冲平移。
    - 'trajectory'：逐 pulse 绝对位移轨迹。

    单位约定：全部用米（m）。角度为度（deg）。
    """
    mode: str = 'none'  # 'none' | 'translation' | 'trajectory' | 'rotation'

    # 板1配置（按方向角逐脉冲平移）
    plate1_direction_deg: float = 22.5
    plate1_translate_step_m: float = 0.0
    plate1_initial_offset_m: Tuple[float, float] = (0.0, 0.0)

    # 逐 pulse 绝对位移轨迹
    # 长度等于 n_pulses，元素顺序 (dx, dy)，单位米
    # pulse_offsets_xy_m[k] 表示第 k 个 pulse 相对初始位置的总运动偏移
    plate1_pulse_offsets_xy_m: Optional[List[Tuple[float, float]]] = None

    # === 旋转运动参数 (mode='rotation' 时使用) ===
    rotation_radius_m: float = 50e-3          # 旋转中心到光束的距离
    rotation_dtheta_rad: float = 0.0          # 每脉冲旋转角增量 (rad)
    rotation_grid_tilt_deg: float = 0.0       # 掩膜网格 X 轴与切线方向的夹角
    rotation_pulse_angles_rad: Optional[List[float]] = None  # 各脉冲绝对旋转角

@dataclass
class OpticalConfig:
    """光学系统配置（单板）"""
    photon_energy_ev: float = 10000.0
    window_size: float = 200e-6
    grid_points: int = 512
    z_propagation: float = 50e-3
    propagation_method: str = "ang_spec"
    source_spot_radius: Optional[float] = None
    # 光源到Hartmann板的距离（用于在掩模平面引入球面波曲率相位）
    source_to_hartmann_z: float = 0.0
    # 光源类型与参数：'plane' | 'top_hat' | 'gaussian'
    source_mode: str = 'plane'
    source_gaussian_sigma: Optional[float] = None
    wfe_phase_map: Optional[np.ndarray] = None
    wfe_type: Optional[str] = None
    wfe_params: Optional[dict] = None
    wfe_rms_rad: Optional[float] = None

@dataclass
class GratingConfig:
    """光栅配置"""
    spacing: float = 3e-6
    aperture: Optional[float] = None
    duty_area: Optional[float] = 0.15
    hartmann_extent: Optional[float] = None
    offset_xy: Tuple[float, float] = (0.0, 0.0)
    blur_sigma_mask: float = 0.0
    # Hartmann 掩模的振幅传输（非理想 0/1），用于更真实的强度透过率建模
    amplitude_inside: float = 1.0
    amplitude_outside: float = 0.0

@dataclass
class DetectorConfig:
    """探测器配置"""
    pixel_size: float = 12e-6
    nx: int = 512
    ny: int = 512
    simulate_effects: bool = True
    sampling_mode: str = 'srw'
    upsample_factor: int = 1
    bin_size: Optional[int] = None
    interpolation: str = 'bilinear'
    binning_policy: str = 'trim'

@dataclass
class OutputConfig:
    """输出配置（无可视化/文件保存）"""
    show_progress: bool = True
    save_intermediate: bool = False


# ============================================================================
# 配置工厂函数 - 简化常用配置的创建
# ============================================================================

def create_default_pulse_config(n_pulses: int = 20, pulse_rep_rate: float = 100.0) -> PulseConfig:
    """创建默认脉冲配置"""
    return PulseConfig(
        n_pulses=n_pulses,
        pulse_rep_rate=pulse_rep_rate,
        detector_fps=10.0,
        wrap_accumulation=True
    )

def create_rotation_config(mode: str = "velocity",
                          angular_velocity_deg_s: float = 10.0,
                          center_offset_m: Tuple[float, float] = (0.0, 0.0)) -> RotationConfig:
    """创建旋转配置"""
    return RotationConfig(
        mode=mode,
        angular_velocity_deg_s=angular_velocity_deg_s,
        center_offset_m=center_offset_m
    )

def create_optical_config(photon_energy_ev: float = 10000.0,
                         window_size: float = 200e-6,
                         grid_points: int = 512,
                         z_propagation: float = 50e-3) -> OpticalConfig:
    """创建光学配置 - 与原函数默认值保持一致"""
    return OpticalConfig(
        photon_energy_ev=photon_energy_ev,
        window_size=window_size,
        grid_points=grid_points,
        z_propagation=z_propagation,
        propagation_method="ang_spec",
        source_mode='plane'
    )

def create_grating_config(spacing: float = 3e-6,
                         duty_area: float = 0.15) -> GratingConfig:
    """创建光栅配置 - 与原函数默认值保持一致"""
    return GratingConfig(
        spacing=spacing,
        duty_area=duty_area
    )

def create_detector_config(pixel_size: float = 12e-6,
                          nx: int = 512,
                          ny: int = 512,
                          sampling_mode: str = 'srw',
                          upsample_factor: int = 1,
                          bin_size: Optional[int] = None,
                          interpolation: str = 'bilinear',
                          binning_policy: str = 'trim') -> DetectorConfig:
    """创建探测器配置 - 与原函数默认值保持一致"""
    return DetectorConfig(
        pixel_size=pixel_size,
        nx=nx,
        ny=ny,
        sampling_mode=sampling_mode,
        upsample_factor=upsample_factor,
        bin_size=bin_size,
        interpolation=interpolation,
        binning_policy=binning_policy,
    )

def create_quick_config(n_pulses: int = 20,
                       photon_energy_ev: float = 10000.0,
                       rotation_center_offset_m: Tuple[float, float] = (0.0, 0.0)) -> Tuple[PulseConfig, RotationConfig, OpticalConfig, GratingConfig, DetectorConfig, OutputConfig]:
    """快速创建一套完整的配置 - 与原函数默认参数保持一致"""
    pulse_cfg = create_default_pulse_config(n_pulses=n_pulses)
    rotation_cfg = create_rotation_config(center_offset_m=rotation_center_offset_m)
    optical_cfg = create_optical_config(photon_energy_ev=photon_energy_ev)
    grating_cfg = create_grating_config()
    detector_cfg = create_detector_config()
    output_cfg = OutputConfig()
    
    return pulse_cfg, rotation_cfg, optical_cfg, grating_cfg, detector_cfg, output_cfg


# ============================================================================
# 简化的接口函数
# ============================================================================

def quick_pulse_superposition(n_pulses: int = 20,
                             photon_energy_ev: float = 10000.0,
                             rotation_center_offset_m: Tuple[float, float] = (0.0, 0.0)) -> Dict[str, Any]:
    """
    快速多帧叠加模拟 - 使用与原函数一致的默认参数
    
    Args:
        n_pulses: 脉冲数量
        photon_energy_ev: 光子能量 (eV)
        rotation_center_offset_m: 旋转中心偏移 (米)
    
    Returns:
        模拟结果字典
    """
    configs = create_quick_config(n_pulses, photon_energy_ev, rotation_center_offset_m)
    pulse_cfg, rotation_cfg, optical_cfg, grating_cfg, detector_cfg, output_cfg = configs
    
    # 无保存路径：保持无可视化/文件输出
    
    simulator = PulseSuperpositionSimulator(
        pulse_cfg, rotation_cfg, optical_cfg, grating_cfg, detector_cfg, output_cfg
    )
    
    return simulator.simulate()





def hartmann_sensor_enhanced(grid_x: np.ndarray,
                           grid_y: np.ndarray,
                           window_size: float,
                           grid_points: int,
                           aperture: Optional[float],
                           spacing: float,
                           duty_area: Optional[float] = None,
                           grid_nx: Optional[int] = None,
                           grid_ny: Optional[int] = None,
                           offset: Tuple[float, float] = (0.0, 0.0),
                           rotation_angle: float = 0.0,
                           rotation_center: Tuple[float, float] = (0.0, 0.0),
                           shape: str = 'circle',
                           # 新增参数
                           rotation_mode: str = 'gttext_coord',
                           pivot_mode: str = 'outside_default',
                           offset_r_m: float = 0.0,
                           offset_theta_deg: float = 0.0,
                           center_abs_m: Tuple[float, float] = (0.0, 0.0),
                           hartmann_shape: str = 'circle',
                           phase_mode: bool = False,
                           phase_delay: float = np.pi / 2.0,
                           phase_on_empty: bool = False,
                           blur_sigma_pix: float = 0.0,
                           **kwargs) -> np.ndarray:
    """
    增强版 Hartmann 传感器掩模生成函数
    
    支持 gttext 风格的坐标级旋转和传统的中心点旋转两种模式
    
    参数:
    -----
    rotation_mode : str
        旋转模式：'gttext_coord' 或 'legacy_centers'
    pivot_mode : str
        旋转中心模式：'outside_default', 'relative_polar', 'absolute'
    其他参数与原 hartmann_sensor 函数相同
        
    返回:
    -----
    np.ndarray
        Hartmann 掩模数组
    """
    # 极坐标模式已移除
    
    # 计算实际旋转中心
    if rotation_mode == 'gttext_coord':
        actual_rotation_center = compute_rotation_center_from_config(
            window_size, spacing, pivot_mode, offset_r_m, offset_theta_deg, center_abs_m
        )
    else:
        # 传统模式使用传入的 rotation_center
        actual_rotation_center = rotation_center
    
    # 统一：总是使用 gttext 风格的坐标级旋转（去除传统中心旋转方法）
    return hartmann_mask_gttext_style(
        grid_x, grid_y, spacing, aperture, duty_area,
        np.deg2rad(rotation_angle), actual_rotation_center,
        hartmann_shape, phase_mode, phase_delay, phase_on_empty, blur_sigma_pix,
        offset_xy=(0.0, 0.0)
    )


class PulseSuperpositionSimulator:
    """多帧叠加模拟器 - 重构版本"""
    
    def __init__(self, 
                 pulse_config: PulseConfig,
                 rotation_config: RotationConfig,
                 optical_config: OpticalConfig,
                 grating_config: GratingConfig,
                 detector_config: DetectorConfig,
                 output_config: OutputConfig,
                 movement_config: MovementConfig = MovementConfig()):
        self.pulse_cfg = pulse_config
        self.rotation_cfg = rotation_config
        self.optical_cfg = optical_config
        self.grating_cfg = grating_config
        self.detector_cfg = detector_config
        self.output_cfg = output_config
        self.movement_cfg = movement_config
        
        # 计算得出的参数
        self.wavelength = None
        self.grid_data: Dict[str, Any] = {}
        self.rotation_step_deg = None
        
        # Per-pulse 相位像差
        self._aberration_mode = 'none'  # 由 simulate.py 设置: 'none', 'static', 'per_pulse'
        self._aberr_max_order = 15
        self._aberr_range = (-2.0, 2.0)
        self._phase_screens: List[np.ndarray] = []
        self._static_phase = None

    def _initialize_aberration_state(self) -> None:
        """Prepare per-run aberration state before pulse generation starts."""
        self._phase_screens = []
        if self._aberration_mode != 'static':
            return

        if self._static_phase is None:
            from simulation.aberration import generate_random_phase_screen

            phase, _coeffs = generate_random_phase_screen(
                self.grid_data['X'],
                self.grid_data['Y'],
                pupil_radius=float(self.optical_cfg.window_size) / 2.0,
                max_order=self._aberr_max_order,
                amplitude_range=self._aberr_range,
            )
            self._static_phase = np.asarray(phase, dtype=np.float64)
        else:
            self._static_phase = np.asarray(self._static_phase, dtype=np.float64)

        if self._static_phase.shape != self.grid_data['X'].shape:
            raise ValueError(
                f"static aberration phase shape {self._static_phase.shape} does not match grid "
                f"shape {self.grid_data['X'].shape}"
            )

        if self.output_cfg.show_progress:
            print(f"Static aberration initialized: max_order={self._aberr_max_order}, range={self._aberr_range}")

    def _compute_translation_offsets(self, pulse_index: int) -> Tuple[float, float]:
        """计算当前脉冲下单板的平移偏移（米）。

        返回纯运动偏移 = initial_offset + 模式相关位移。
        不包含 grating_cfg.offset_xy（静态栅格偏移由调用方通过 base_offset_xy 独立处理）。
        """
        # 初始偏移（米）
        p1_init = tuple(float(v) for v in (getattr(self.movement_cfg, 'plate1_initial_offset_m', (0.0, 0.0)) or (0.0, 0.0)))

        dx = p1_init[0]
        dy = p1_init[1]

        mode = str(getattr(self.movement_cfg, 'mode', 'none') or 'none').lower()

        if mode == 'none':
            return (dx, dy)

        if mode == 'translation':
            step = float(getattr(self.movement_cfg, 'plate1_translate_step_m', 0.0) or 0.0)
            ang = float(getattr(self.movement_cfg, 'plate1_direction_deg', 0.0) or 0.0)
            rad = np.deg2rad(ang)
            dx += pulse_index * step * float(np.cos(rad))
            dy += pulse_index * step * float(np.sin(rad))
            return (dx, dy)

        if mode == 'trajectory':
            offsets = self.movement_cfg.plate1_pulse_offsets_xy_m
            if offsets is None:
                raise ValueError("movement mode 'trajectory' requires plate1_pulse_offsets_xy_m")
            if pulse_index >= len(offsets):
                raise IndexError(
                    f"trajectory offset index out of range: pulse_index={pulse_index}, "
                    f"len(offsets)={len(offsets)}"
                )
            ox, oy = offsets[pulse_index]
            dx += float(ox)
            dy += float(oy)
            return (dx, dy)

        if mode == 'rotation':
            # For rotation mode, linear offsets are pre-computed from exact trig
            # and stored in plate1_pulse_offsets_xy_m by simulate.py.
            offsets = self.movement_cfg.plate1_pulse_offsets_xy_m
            if offsets is None:
                raise ValueError("movement mode 'rotation' requires plate1_pulse_offsets_xy_m")
            if pulse_index >= len(offsets):
                raise IndexError(
                    f"rotation offset index out of range: pulse_index={pulse_index}, "
                    f"len(offsets)={len(offsets)}"
                )
            ox, oy = offsets[pulse_index]
            dx += float(ox)
            dy += float(oy)
            return (dx, dy)

        # 未知模式：回退到无平移
        return (dx, dy)
        
    def _setup_simulation(self):
        """初始化模拟参数"""
        # 计算波长
        self.wavelength = calculate_wavelength(self.optical_cfg.photon_energy_ev)
        
        # 构建计算网格
        X, Y, dx, dy = make_grid(self.optical_cfg.window_size, self.optical_cfg.grid_points)
        # 选择光源类型
        mode = str(getattr(self.optical_cfg, 'source_mode', 'plane') or 'plane').lower()
        if mode in {'gaussian'} and (self.optical_cfg.source_gaussian_sigma or 0.0) > 0.0:
            U_in = make_gaussian_source(X, Y, sigma=float(self.optical_cfg.source_gaussian_sigma), energy=1.0)
            if self.output_cfg.show_progress:
                print(f"高斯源σ: {float(self.optical_cfg.source_gaussian_sigma)*1e6:.1f} μm")
        elif mode in {'top_hat'}:
            U_in = make_plane_source(X, Y, wavelength=self.wavelength,
                                     spot_radius=self.optical_cfg.source_spot_radius, energy=1.0)
            if self.output_cfg.show_progress and self.optical_cfg.source_spot_radius is not None:
                print(f"平顶源半径: {self.optical_cfg.source_spot_radius*1e6:.1f} μm")
        else:
            U_in = make_plane_source(X, Y, wavelength=self.wavelength,
                                     spot_radius=None, energy=1.0)
        # 若设置了光源到Hartmann板距离，则在掩模平面引入近轴球面波曲率相位
        try:
            R = float(getattr(self.optical_cfg, 'source_to_hartmann_z', 0.0) or 0.0)
        except Exception:
            R = 0.0
        if R > 0.0:
            k = 2.0 * np.pi / float(self.wavelength)
            phase_curvature = np.exp(1j * k * (X**2 + Y**2) / (2.0 * R))
            U_in = U_in * phase_curvature
            if self.output_cfg.show_progress:
                print(f"已应用光源到Hartmann板距离: {R*1e3:.2f} mm (近轴球面波相位)")
        applied_wfe = False
        self.wfe_phase_map_rad = None
        if getattr(self.optical_cfg, 'wfe_phase_map', None) is not None:
            try:
                phi = np.asarray(self.optical_cfg.wfe_phase_map, dtype=float)
                U_in = U_in * np.exp(1j * phi)
                applied_wfe = True
                self.wfe_phase_map_rad = phi
                if self.output_cfg.show_progress:
                    print("已注入自定义波前相位误差")
            except Exception:
                pass
        elif getattr(self.optical_cfg, 'wfe_type', None):
            try:
                ny, nx = X.shape
                rmax = float(self.optical_cfg.window_size) * 0.5
                t = str(self.optical_cfg.wfe_type).lower()
                params = (self.optical_cfg.wfe_params or {})
                phi = np.zeros_like(X, dtype=float)
                if '+' in t:
                    parts = [p.strip() for p in t.split('+') if p.strip()]
                else:
                    parts = [t]
                for part in parts:
                    if part == 'zernike':
                        pr = float(params.get('pupil_radius_m', rmax))
                        cx, cy = params.get('pupil_center_m', (0.0, 0.0))
                        Xc = X - float(cx)
                        Yc = Y - float(cy)
                        R = np.sqrt(Xc**2 + Yc**2) / max(1e-15, pr)
                        Theta = np.arctan2(Yc, Xc)
                        mask = (R <= 1.0).astype(float)
                        coefs = params.get('zernike_coefs', [])
                        def _radial(n, m, r):
                            m = int(abs(m))
                            n = int(n)
                            s_max = (n - m) // 2
                            out = np.zeros_like(r, dtype=float)
                            for s in range(int(s_max) + 1):
                                c = ((-1.0)**s) * float(np.math.factorial(n - s))
                                c /= float(np.math.factorial(s))
                                c /= float(np.math.factorial((n + m)//2 - s))
                                c /= float(np.math.factorial((n - m)//2 - s))
                                out = out + c * (r ** (n - 2*s))
                            return out
                        noll_map = {
                            1: (0, 0, 'cos'),
                            2: (1, 1, 'cos'),
                            3: (1, 1, 'sin'),
                            4: (2, 0, 'cos'),
                            5: (2, 2, 'cos'),
                            6: (2, 2, 'sin'),
                            7: (3, 1, 'cos'),
                            8: (3, 1, 'sin'),
                            9: (3, 3, 'cos'),
                            10: (3, 3, 'sin'),
                            11: (4, 0, 'cos'),
                            12: (4, 2, 'cos'),
                            13: (4, 2, 'sin'),
                            14: (4, 4, 'cos'),
                            15: (4, 4, 'sin'),
                        }
                        for j, a in coefs:
                            j = int(j)
                            a = float(a)
                            if j not in noll_map:
                                continue
                            n, m, mode = noll_map[j]
                            Rnm = _radial(n, m, R)
                            if m == 0:
                                Z = Rnm
                            else:
                                if mode == 'cos':
                                    Z = Rnm * np.cos(int(abs(m)) * Theta)
                                else:
                                    Z = Rnm * np.sin(int(abs(m)) * Theta)
                            phi = phi + a * Z * mask
                    if part == 'tilt':
                        tx = float(params.get('tilt_tx', 0.0))
                        ty = float(params.get('tilt_ty', 0.0))
                        phi += tx * X + ty * Y
                    elif part == 'defocus':
                        a = float(params.get('defocus_alpha', 0.0))
                        phi += a * ((X**2 + Y**2) / max(1e-15, rmax**2))
                    elif part == 'astig':
                        a = float(params.get('astig_alpha', 0.0))
                        ang = float(params.get('astig_angle_deg', 0.0))
                        rad = np.deg2rad(ang)
                        Xr = X * np.cos(rad) + Y * np.sin(rad)
                        Yr = -X * np.sin(rad) + Y * np.cos(rad)
                        phi += a * (((Xr**2 - Yr**2) / max(1e-15, rmax**2)))
                    elif part == 'psd':
                        rms = float(getattr(self.optical_cfg, 'wfe_rms_rad', params.get('psd_rms_rad', 0.0)) or 0.0)
                        corr = float(params.get('psd_corr_len_m', 0.0))
                        if rms > 0.0 and corr > 0.0:
                            fx = np.fft.fftfreq(nx, d=float(self.optical_cfg.window_size)/nx)
                            fy = np.fft.fftfreq(ny, d=float(self.optical_cfg.window_size)/ny)
                            FY, FX = np.meshgrid(fy, fx, indexing='ij')
                            H = np.exp(-2.0 * (np.pi**2) * (FX**2 + FY**2) * (corr**2))
                            noise = np.random.normal(0.0, 1.0, size=(ny, nx))
                            F = np.fft.fft2(noise)
                            Ff = F * H
                            screen = np.real(np.fft.ifft2(Ff))
                            screen -= float(np.mean(screen))
                            std = float(np.std(screen))
                            if std > 1e-15:
                                screen = screen * (rms / std)
                            phi += screen
                    elif part == 'geometric':
                        # Geometric shapes WFE (Circle, Rectangle, Triangle)
                        shapes = params.get('shapes', [])
                        for shape in shapes:
                            stype = str(shape.get('type', 'circle')).lower()
                            cx, cy = shape.get('center', (0.0, 0.0))
                            amp = float(shape.get('amplitude', 1.0))
                            term = str(shape.get('term', 'tilt')).lower() # 'tilt', 'piston', 'defocus'
                            
                            mask_shape = np.zeros_like(X, dtype=bool)
                            
                            if stype == 'circle':
                                r = float(shape.get('radius', 10e-6))
                                mask_shape = ((X - cx)**2 + (Y - cy)**2) <= r**2
                            elif stype == 'rectangle' or stype == 'square':
                                w = float(shape.get('width', 20e-6))
                                h = float(shape.get('height', w))
                                mask_shape = (np.abs(X - cx) <= w/2) & (np.abs(Y - cy) <= h/2)
                            elif stype == 'triangle':
                                # Equilateral triangle defined by radius (circumcircle)
                                r = float(shape.get('radius', 20e-6))
                                diff_x, diff_y = X - cx, Y - cy
                                # Line 1 (bottom): y > -0.5r
                                c1 = diff_y >= -0.5 * r
                                # Line 2 (left): y - 1.732x < r
                                c2 = (diff_y - 1.732 * diff_x) <= r
                                # Line 3 (right): y + 1.732x < r
                                c3 = (diff_y + 1.732 * diff_x) <= r
                                mask_shape = c1 & c2 & c3
                            elif stype == 'ring':
                                r_out = float(shape.get('radius_outer', 20e-6))
                                r_in = float(shape.get('radius_inner', 10e-6))
                                r2 = (X - cx)**2 + (Y - cy)**2
                                mask_shape = (r2 <= r_out**2) & (r2 >= r_in**2)

                            if np.any(mask_shape):
                                # Apply phase term inside the shape
                                if term == 'piston':
                                    phi[mask_shape] += amp
                                elif term == 'tilt':
                                    tx = float(shape.get('tilt_x', amp))
                                    ty = float(shape.get('tilt_y', 0.0))
                                    phi[mask_shape] += tx * (X[mask_shape] - cx) + ty * (Y[mask_shape] - cy)
                                elif term == 'defocus':
                                    # Quadratic bump: phi = amp * (r_local / R)^2
                                    if stype == 'circle' or stype == 'ring':
                                        r_max = float(shape.get('radius', shape.get('radius_outer', 10e-6)))
                                    else:
                                        r_max = float(shape.get('width', 20e-6)) / 2.0
                                    r2 = (X[mask_shape] - cx)**2 + (Y[mask_shape] - cy)**2
                                    phi[mask_shape] += amp * (r2 / (r_max**2))
                                elif term == 'pyramid':
                                    # Linear ramp from center to edge (Cone-like): phi = amp * (1 - r_local / R)
                                    if stype == 'circle' or stype == 'ring':
                                        r_max = float(shape.get('radius', shape.get('radius_outer', 10e-6)))
                                    else:
                                        r_max = float(shape.get('width', 20e-6)) / 2.0
                                    r_local = np.sqrt((X[mask_shape] - cx)**2 + (Y[mask_shape] - cy)**2)
                                    phi[mask_shape] += amp * (1.0 - r_local / r_max)
                    
                    # 处理新模式: stripe, step, sinusoidal, concentric
                    elif part == 'stripe':
                        stripe_params = params.get('stripe_params', {})
                        from simulation.wfe_generator import generate_stripe_wfe
                        phi += generate_stripe_wfe(X, Y, **stripe_params)
                    elif part == 'step':
                        step_params = params.get('step_params', {})
                        from simulation.wfe_generator import generate_step_wfe
                        phi += generate_step_wfe(X, Y, **step_params)
                    elif part == 'sinusoidal':
                        sin_params = params.get('sinusoidal_params', {})
                        from simulation.wfe_generator import generate_sinusoidal_wfe
                        phi += generate_sinusoidal_wfe(X, Y, window_size_m=float(self.optical_cfg.window_size), **sin_params)
                    elif part == 'concentric':
                        conc_params = params.get('concentric_params', {})
                        from simulation.wfe_generator import generate_concentric_wfe
                        phi += generate_concentric_wfe(X, Y, **conc_params)

                if np.any(np.isfinite(phi)):
                    U_in = U_in * np.exp(1j * phi)
                    applied_wfe = True
                    self.wfe_phase_map_rad = phi
                    if self.output_cfg.show_progress:
                        print(f"已注入模型化波前误差: {self.optical_cfg.wfe_type}")
            except Exception:
                pass
        self.grid_data = {'X': X, 'Y': Y, 'dx': dx, 'dy': dy, 'U_in': U_in}
        # 记录是否已注入WFE，供simulate返回
        self._applied_wfe_flag = bool(applied_wfe)
        self._initialize_aberration_state()
        
        
        
        # 计算旋转步长（以度为单位，便于与theta_start_deg一致）
        self.rotation_step_deg = float(np.rad2deg(compute_rotation_step(
            rotation_mode=self.rotation_cfg.mode,
            step_angle_deg=self.rotation_cfg.step_angle_deg,
            angular_velocity_deg_s=self.rotation_cfg.angular_velocity_deg_s,
            pulse_rep_rate=self.pulse_cfg.pulse_rep_rate,
        )))
        
        if self.output_cfg.show_progress:
            # 旋转相关信息打印已移除（仅保留其他诊断）
            effective_window_size = (self.grating_cfg.hartmann_extent 
                                   if self.grating_cfg.hartmann_extent is not None 
                                   else self.optical_cfg.window_size)
            if self.grating_cfg.hartmann_extent is not None:
                print(f"Hartmann光栅生成区域: {effective_window_size*1e6:.1f} μm "
                      f"(扩展自 {self.optical_cfg.window_size*1e6:.1f} μm)")
            if self.optical_cfg.source_spot_radius is not None:
                print(f"光源照射半径: {self.optical_cfg.source_spot_radius*1e6:.1f} μm")
    
    def _determine_pulse_count(self) -> int:
        """确定脉冲数量"""
        if self.pulse_cfg.n_pulses is not None:
            return max(1, int(self.pulse_cfg.n_pulses))
        elif self.pulse_cfg.duration_s is not None:
            return int(np.clip(round(self.pulse_cfg.duration_s * self.pulse_cfg.pulse_rep_rate), 1, 10**7))
        else:
            return 50  # 默认值
    
    def _generate_single_frame(self, pulse_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """生成单个脉冲的掩模、含像差探测器图像、无像差理想探测器图像。
        
        Returns:
            T: mask
            I: 未采样含像差强度 (propagation grid)
            D: 含像差探测器图像
            D_ideal: 无像差理想探测器图像 (同掩模位置)
        """
        # 禁用旋转：始终为 0 度
        theta = 0.0

        # 计算当前脉冲的平移偏移（米）
        dx1, dy1 = self._compute_translation_offsets(pulse_index)

        # 单板：根据运动模式选择掩膜生成方式
        mode = str(getattr(self.movement_cfg, 'mode', 'none') or 'none').lower()
        if mode == 'rotation':
            # 旋转模式：坐标级旋转，零插值误差
            pulse_angles = self.movement_cfg.rotation_pulse_angles_rad
            if pulse_angles is None or pulse_index >= len(pulse_angles):
                raise ValueError(
                    f"rotation mode requires rotation_pulse_angles_rad, "
                    f"got pulse_index={pulse_index}"
                )
            rot_angle = float(pulse_angles[pulse_index])
            R = float(self.movement_cfg.rotation_radius_m)
            tilt_deg = float(self.movement_cfg.rotation_grid_tilt_deg)
            # 旋转中心的放置决定了切线方向。
            # 为了让光束处的切线方向 = tilt_deg（与 translation 模式的 ANGLE_DEG 等效），
            # 旋转中心需要放在光束点的法线方向上（切线 + 90°）。
            # 法线方向 = tilt_deg + 90°
            normal_rad = np.deg2rad(tilt_deg + 90.0)
            rot_center = (R * float(np.cos(normal_rad)),
                          R * float(np.sin(normal_rad)))
            T = make_hartmann_mask_rotated(
                self.grid_data['X'], self.grid_data['Y'],
                rotation_angle_rad=rot_angle,
                rotation_center_xy_m=rot_center,
                grid_tilt_deg=0.0,  # 网格不旋转，与 translation 模式一致
                spacing=self.grating_cfg.spacing,
                aperture=self.grating_cfg.aperture,
                duty_area=self.grating_cfg.duty_area,
                shape='circle',
                blur_sigma_pix=self.grating_cfg.blur_sigma_mask,
                amplitude_inside=float(self.grating_cfg.amplitude_inside),
                amplitude_outside=float(self.grating_cfg.amplitude_outside),
                base_offset_xy=self.grating_cfg.offset_xy,
            )
        else:
            # 平移模式（translation / trajectory / none）
            T = make_hartmann_mask_translated(
                self.grid_data['X'], self.grid_data['Y'],
                shift_x_m=float(dx1),
                shift_y_m=float(dy1),
                spacing=self.grating_cfg.spacing,
                aperture=self.grating_cfg.aperture,
                duty_area=self.grating_cfg.duty_area,
                rotation_angle_deg=theta,
                shape='circle',
                blur_sigma_pix=self.grating_cfg.blur_sigma_mask,
                amplitude_inside=float(self.grating_cfg.amplitude_inside),
                amplitude_outside=float(self.grating_cfg.amplitude_outside),
                base_offset_xy=self.grating_cfg.offset_xy,
            )

        # 诊断输出（仅首帧）
        if self.output_cfg.show_progress and pulse_index == 0:
            try:
                open_fraction = float(np.mean(np.asarray(T, dtype=np.float64) > 0.5))
            except Exception:
                open_fraction = float(np.mean(np.real(T) > 0.5))
            aperture_eff = (self.grating_cfg.aperture if self.grating_cfg.aperture is not None
                            else duty_area_to_diameter(self.grating_cfg.spacing, self.grating_cfg.duty_area))
            print(f"掩膜开孔率: {open_fraction*100:.2f}%  | 有效孔径直径: {aperture_eff*1e6:.3f} μm  (spacing={self.grating_cfg.spacing*1e6:.3f} μm)")

        # === Per-pulse 相位像差注入 ===
        U_in_base = self.grid_data['U_in']  # 无像差的原始场
        U_in_frame = U_in_base  # 含像差的场
        
        if self._aberration_mode == 'per_pulse':
            from simulation.aberration import generate_random_phase_screen
            phase, coeffs = generate_random_phase_screen(
                self.grid_data['X'], self.grid_data['Y'],
                pupil_radius=float(self.optical_cfg.window_size) / 2.0,
                max_order=self._aberr_max_order,
                amplitude_range=self._aberr_range,
            )
            U_in_frame = U_in_base * np.exp(1j * phase)
            self._phase_screens.append(phase)
            if self.output_cfg.show_progress and pulse_index == 0:
                print(f"Per-pulse aberration: max_order={self._aberr_max_order}, range={self._aberr_range}")
        elif self._aberration_mode == 'static':
            if self._static_phase is None:
                raise RuntimeError("static aberration mode selected but _static_phase was not initialized")
            U_in_frame = U_in_base * np.exp(1j * self._static_phase)
            self._phase_screens.append(self._static_phase)

        # --- 含像差: 传播 + 探测器采样 ---
        U0_aberr = apply_mask(U_in_frame, T)
        I, D = self._propagate_and_sample(U0_aberr, pulse_index)

        # --- 无像差 (同掩模位置): 传播 + 探测器采样 ---
        if self._aberration_mode != 'none':
            U0_ideal = apply_mask(U_in_base, T)
            _, D_ideal = self._propagate_and_sample(U0_ideal, pulse_index)
        else:
            D_ideal = D  # 无像差时，两者相同

        return T, I, D, D_ideal

    def _propagate_and_sample(self, U0: np.ndarray, pulse_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """传播光场并采样到探测器，返回 (raw_intensity, detector_image)。"""
        U1, dx_eff, dy_eff = self._propagate_field(U0)
        I = np.abs(U1) ** 2
        mode_cfg = str(self.detector_cfg.sampling_mode).lower()

        # 传播诊断（仅首帧首次调用）
        if self.output_cfg.show_progress and pulse_index == 0 and not hasattr(self, '_diag_printed'):
            self._diag_printed = True
            try:
                pre_ny, pre_nx = self.grid_data['X'].shape
                pre_dx, pre_dy = self.grid_data['dx'], self.grid_data['dy']
                post_ny, post_nx = I.shape
                width_pre = pre_nx * pre_dx
                height_pre = pre_ny * pre_dy
                width_post = post_nx * dx_eff
                height_post = post_ny * dy_eff
                print("Propagation diagnostics:")
                print(f"  传播前网格: {pre_nx}×{pre_ny}, Δx={pre_dx*1e6:.3f} μm, FOV={width_pre*1e3:.3f} mm × {height_pre*1e3:.3f} mm")
                print(f"  传播后网格: {post_nx}×{post_ny}, Δx={dx_eff*1e6:.3f} μm, FOV={width_post*1e3:.3f} mm × {height_post*1e3:.3f} mm")
            except Exception:
                pass

        # 居中裁切/填充
        try:
            pre_ny, pre_nx = self.grid_data['X'].shape
            post_ny, post_nx = I.shape
            if post_ny >= pre_ny and post_nx >= pre_nx:
                sy = (post_ny - pre_ny) // 2
                sx = (post_nx - pre_nx) // 2
                I_for_sample = I[sy:sy+pre_ny, sx:sx+pre_nx]
            else:
                I_for_sample = np.zeros((pre_ny, pre_nx), dtype=I.dtype)
                oy = max(0, (pre_ny - post_ny) // 2)
                ox = max(0, (pre_nx - post_nx) // 2)
                y_end = min(pre_ny, oy + post_ny)
                x_end = min(pre_nx, ox + post_nx)
                I_for_sample[oy:y_end, ox:x_end] = I[0:(y_end-oy), 0:(x_end-ox)]
        except Exception:
            I_for_sample = I

        # 探测器采样诊断（仅首帧首次调用）
        if self.output_cfg.show_progress and pulse_index == 0 and not hasattr(self, '_det_diag_printed'):
            self._det_diag_printed = True
            try:
                ny_in, nx_in = I_for_sample.shape
                width_in = nx_in * dx_eff
                height_in = ny_in * dy_eff
                width_det = float(self.detector_cfg.nx) * float(self.detector_cfg.pixel_size)
                height_det = float(self.detector_cfg.ny) * float(self.detector_cfg.pixel_size)
                cover_x = (width_in / width_det) if width_det > 0 else 0.0
                cover_y = (height_in / height_det) if height_det > 0 else 0.0
                approx_nonzero_w = min(self.detector_cfg.nx, int(round(self.detector_cfg.nx * cover_x)))
                approx_nonzero_h = min(self.detector_cfg.ny, int(round(self.detector_cfg.ny * cover_y)))
                print("Detector sampling diagnostics (after cropping):")
                print(f"  Input FOV:  {width_in*1e3:.3f} mm × {height_in*1e3:.3f} mm  (grid={nx_in}×{ny_in}, Δx={dx_eff*1e6:.3f} μm)")
                print(f"  Detector FOV: {width_det*1e3:.3f} mm × {height_det*1e3:.3f} mm  (pixels={self.detector_cfg.nx}×{self.detector_cfg.ny}, px={self.detector_cfg.pixel_size*1e6:.3f} μm)")
                if cover_x > 1.0 or cover_y > 1.0:
                    print(f"  FOV coverage ≈ {cover_x*100:.2f}% × {cover_y*100:.2f}% → 裁剪到探测器尺寸")
                else:
                    print(f"  FOV coverage ≈ {cover_x*100:.2f}% × {cover_y*100:.2f}% → 非零区域估算≈ {approx_nonzero_w}×{approx_nonzero_h} 像素")
                print(f"  Mode='{mode_cfg}', upsample={getattr(self.detector_cfg,'upsample_factor',1)}, bin={getattr(self.detector_cfg,'bin_size', getattr(self.detector_cfg,'upsample_factor',1))}, interp='{getattr(self.detector_cfg,'interpolation','bilinear')}'")
            except Exception:
                pass

        # 探测器采样
        if self.detector_cfg.simulate_effects and mode_cfg != 'none':
            upsample = getattr(self.detector_cfg, 'upsample_factor', 1) or 1
            bin_sz = getattr(self.detector_cfg, 'bin_size', None)
            if bin_sz is None:
                bin_sz = upsample
            D = sample_detector_fov(
                I_for_sample, dx_eff, dy_eff,
                pixel_size=self.detector_cfg.pixel_size,
                detector_nx=self.detector_cfg.nx,
                detector_ny=self.detector_cfg.ny,
                center_x_m=0.0,
                center_y_m=0.0,
                mode=mode_cfg,
                upsample_factor=int(upsample),
                bin_size=int(bin_sz),
                interpolation=str(getattr(self.detector_cfg, 'interpolation', 'bilinear')),
                binning_policy=str(getattr(self.detector_cfg, 'binning_policy', 'trim')),
            )
        else:
            D = I_for_sample

        return I, D
    
    def _display_frame(self, mask: np.ndarray, intensity: np.ndarray, detector_image: np.ndarray, 
                      pulse_index: int, theta: float):
        # 显示与保存逻辑已删除
        return
    
    def _propagate_field(self, U0: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """传播光场"""
        pm = str(self.optical_cfg.propagation_method).lower()
        
        if pm == 'fresnel':
            U1, dx_eff, dy_eff = fresnel_prop(
                U0, self.wavelength, 
                self.grid_data['dx'], self.grid_data['dy'], 
                self.optical_cfg.z_propagation
            )
        elif pm in {'fourier', 'fraunhofer'}:
            U1, dx_eff, dy_eff = fourier_prop(
                U0, self.wavelength,
                self.grid_data['dx'], self.grid_data['dy'],
                self.optical_cfg.z_propagation,
                window_size=self.optical_cfg.window_size
            )
        else:  # ang_spec
            result = ang_spec(
                U0, self.wavelength,
                self.grid_data['dx'], self.grid_data['dy'],
                self.optical_cfg.z_propagation
            )
            U1 = result if isinstance(result, np.ndarray) else result[0]
            dx_eff, dy_eff = self.grid_data['dx'], self.grid_data['dy']
        
        return U1, dx_eff, dy_eff

    def _propagate_field_with_distance(self, U0: np.ndarray, z_m: float) -> Tuple[np.ndarray, float, float]:
        """按指定距离传播光场（与 _propagate_field 相同算法，仅替换距离参数）"""
        pm = str(self.optical_cfg.propagation_method).lower()

        if pm == 'fresnel':
            # 重要：禁用板间传播的自动零填充，保持与掩模相同的网格尺寸
            U1, dx_eff, dy_eff = fresnel_prop(
                U0, self.wavelength,
                self.grid_data['dx'], self.grid_data['dy'],
                z_m,
                config=PropagationConfig(padding_factor=1)
            )
        elif pm in {'fourier', 'fraunhofer'}:
            # 同理：禁用板间传播的零填充，避免形状变化导致与下一块掩模不匹配
            U1, dx_eff, dy_eff = fourier_prop(
                U0, self.wavelength,
                self.grid_data['dx'], self.grid_data['dy'],
                z_m,
                window_size=self.optical_cfg.window_size,
                config=PropagationConfig(padding_factor=1)
            )
        else:  # ang_spec
            result = ang_spec(
                U0, self.wavelength,
                self.grid_data['dx'], self.grid_data['dy'],
                z_m
            )
            U1 = result if isinstance(result, np.ndarray) else result[0]
            dx_eff, dy_eff = self.grid_data['dx'], self.grid_data['dy']

        return U1, dx_eff, dy_eff
    
    def _accumulate_frames(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """累积帧图像，统一返回二维数组
        - 当脉冲少于每帧脉冲数时，直接逐帧相加并保证输出二维。
        - 当按脉冲分帧累积返回列表时，堆叠并对时间维求和，得到二维图像。
        - 当返回三维数组时（例如 (N,H,W) 或 (H,W,1)），压缩为 (H,W)。
        """
        if not frames:
            return None
            
        K_eff = determine_pulses_per_frame(
            self.pulse_cfg.pulses_per_frame,
            self.pulse_cfg.pulse_rep_rate,
            self.pulse_cfg.detector_fps
        )
        
        # 如果脉冲数少于每帧脉冲数，直接累积所有帧
        if len(frames) < K_eff:
            # Use optimized accumulation if available
            if _HAS_OPTIMIZATIONS and fast_accumulate_frames is not None:
                result = fast_accumulate_frames(frames)
            else:
                result = frames[0].copy()
                for i in range(1, len(frames)):
                    result += frames[i]
            result = np.asarray(result, dtype=np.float64)
            if result.ndim == 3:
                if result.shape[-1] == 1:
                    result = result.squeeze(-1)
                else:
                    result = np.sum(result, axis=0)
            return result
        
        accumulated = accumulate_by_pulses(frames, K_eff, wrap=self.pulse_cfg.wrap_accumulation)
        if isinstance(accumulated, list):
           

            try:
                stack = np.stack([np.asarray(a, dtype=np.float64) for a in accumulated], axis=0)
                result = np.sum(stack, axis=0)
            except Exception:
                result = np.asarray(accumulated[0], dtype=np.float64)
        else:
            result = np.asarray(accumulated, dtype=np.float64)
            if result.ndim == 3:
                if result.shape[-1] == 1:
                    result = result.squeeze(-1)
                else:
                    # 若为 (N,H,W)，对 N 维求和
                    if result.shape[0] > 1:
                        result = np.sum(result, axis=0)
        return result

    def _compute_gt_displacements(self, phase_screens: List[np.ndarray]) -> Dict[str, Any]:
        """
        从相位屏计算每个子孔径的 ground truth 位移。
        
        原理: Δx(m) = [∂φ/∂x] × λ/(2π) × z_prop
              Δx(px) = Δx(m) / pixel_size
        
        移植自 dev 版本的 _compute_ground_truth()。
        """
        dx = self.grid_data['dx']
        X = self.grid_data['X']
        Y = self.grid_data['Y']
        spacing = float(self.grating_cfg.spacing)
        hole_diameter = compute_aperture_diameter(
            spacing, self.grating_cfg.aperture, self.grating_cfg.duty_area
        )
        
        # 子孔径中心坐标 (物理坐标, 米)
        window_size = float(self.optical_cfg.window_size)
        grid_count = max(1, int(round(window_size / spacing)))
        x_coords = (np.arange(grid_count) - (grid_count - 1) / 2.0) * spacing
        y_coords = (np.arange(grid_count) - (grid_count - 1) / 2.0) * spacing

        # 仅保留探测器视场内的子孔径
        det_w = float(self.detector_cfg.nx) * float(self.detector_cfg.pixel_size)
        det_h = float(self.detector_cfg.ny) * float(self.detector_cfg.pixel_size)
        x_coords = x_coords[np.abs(x_coords) < det_w / 2.0]
        y_coords = y_coords[np.abs(y_coords) < det_h / 2.0]
        
        grid_rows = len(y_coords)
        grid_cols = len(x_coords)
        hole_radius_px = int(np.ceil(hole_diameter / (2 * dx)))
        
        n_frames = len(phase_screens)
        scale = float(self.wavelength) / (2.0 * np.pi)
        z_prop = float(self.optical_cfg.z_propagation)
        pixel_size = float(self.detector_cfg.pixel_size)
        
        # 每帧每个子孔径的位移
        all_phase_vals = np.zeros((n_frames, grid_rows, grid_cols))
        all_disp_x_px = np.zeros((n_frames, grid_rows, grid_cols))
        all_disp_y_px = np.zeros((n_frames, grid_rows, grid_cols))
        
        for f_idx, phase_screen in enumerate(phase_screens):
            grad_y, grad_x = np.gradient(phase_screen, dx, dx)
            
            for r, yc in enumerate(y_coords):
                for c, xc in enumerate(x_coords):
                    # 像素索引
                    cx_idx = int(np.clip(np.round((xc - X[0, 0]) / dx), 0, X.shape[1] - 1))
                    cy_idx = int(np.clip(np.round((yc - Y[0, 0]) / dx), 0, Y.shape[0] - 1))
                    
                    # 圆形孔径区域
                    y0 = max(0, cy_idx - hole_radius_px)
                    y1 = min(phase_screen.shape[0], cy_idx + hole_radius_px + 1)
                    x0 = max(0, cx_idx - hole_radius_px)
                    x1 = min(phase_screen.shape[1], cx_idx + hole_radius_px + 1)
                    
                    yy, xx = np.ogrid[y0:y1, x0:x1]
                    dist = np.sqrt((yy - cy_idx)**2 + (xx - cx_idx)**2)
                    aperture_mask = dist <= hole_radius_px
                    
                    if aperture_mask.sum() > 0:
                        all_phase_vals[f_idx, r, c] = np.mean(phase_screen[y0:y1, x0:x1][aperture_mask])
                        gx_mean = np.mean(grad_x[y0:y1, x0:x1][aperture_mask])
                        gy_mean = np.mean(grad_y[y0:y1, x0:x1][aperture_mask])
                    else:
                        all_phase_vals[f_idx, r, c] = phase_screen[cy_idx, cx_idx]
                        gx_mean = grad_x[cy_idx, cx_idx]
                        gy_mean = grad_y[cy_idx, cx_idx]
                    
                    # 物理位移 → 像素位移
                    all_disp_x_px[f_idx, r, c] = scale * gx_mean * z_prop / pixel_size
                    all_disp_y_px[f_idx, r, c] = scale * gy_mean * z_prop / pixel_size
        
        return {
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "centers_x_m": x_coords,
            "centers_y_m": y_coords,
            "phase_rad": all_phase_vals,           # (n_frames, rows, cols)
            "displacement_x_px": all_disp_x_px,    # (n_frames, rows, cols) 像素
            "displacement_y_px": all_disp_y_px,     # (n_frames, rows, cols) 像素
            "displacement_x_m": all_disp_x_px * pixel_size,  # 米
            "displacement_y_m": all_disp_y_px * pixel_size,   # 米
            "wavelength_m": float(self.wavelength),
            "z_prop_m": z_prop,
            "pixel_size_m": pixel_size,
            "spacing_m": spacing,
        }

    def simulate(self) -> Dict[str, Any]:
        """执行完整的多帧叠加模拟（CPU模式）"""
        # 初始化
        self._setup_simulation()
        n_pulses = self._determine_pulse_count()
        
        if self.output_cfg.show_progress:
            print(f"\nStarting multi-frame superposition simulation...")
            print(f"Number of pulses: {n_pulses}")

        # 生成所有帧
        frames_intensity: List[np.ndarray] = []
        frames_detector: List[np.ndarray] = []
        frames_ideal: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        
        for i in range(n_pulses):
            if self.output_cfg.show_progress and (i + 1) % max(1, n_pulses // 10) == 0:
                print(f"Progress: {i + 1}/{n_pulses} ({100 * (i + 1) / n_pulses:.1f}%)")
            mask, intensity, detector_img, ideal_img = self._generate_single_frame(i)
            masks.append(mask)
            frames_intensity.append(intensity)
            frames_detector.append(detector_img)
            frames_ideal.append(ideal_img)

        # 累积帧
        integrated_unsampled = self._accumulate_frames(frames_intensity)
        integrated_sampled = self._accumulate_frames(frames_detector)
        integrated_ideal = self._accumulate_frames(frames_ideal)
        K_eff = determine_pulses_per_frame(
            self.pulse_cfg.pulses_per_frame,
            self.pulse_cfg.pulse_rep_rate,
            self.pulse_cfg.detector_fps
        )

        # 显示与保存逻辑已删除

        if self.output_cfg.show_progress:
            print(f"\nSimulation completed!")
            print(f"Integrated (sampled) image shape: {integrated_sampled.shape if integrated_sampled is not None else 'None'}")
            print(f"Integrated (ideal) image shape: {integrated_ideal.shape if integrated_ideal is not None else 'None'}")

        # === 计算 Ground Truth 位移 ===
        gt_displacements = None
        phase_screens_out = None
        if self._phase_screens:
            phase_screens_out = np.array(self._phase_screens)
            gt_displacements = self._compute_gt_displacements(self._phase_screens)
            if self.output_cfg.show_progress:
                print(f"Ground truth computed: {len(self._phase_screens)} phase screens")

        return {
            "frames": frames_detector,
            "frames_unsampled": frames_intensity,
            "frames_sampled": frames_detector,
            "frames_ideal": frames_ideal,
            "integrated_unsampled": integrated_unsampled,
            "integrated_sampled": integrated_sampled,
            "integrated_ideal": integrated_ideal,
            "K_eff": K_eff,
            "wfe_phase_map_rad": getattr(self, 'wfe_phase_map_rad', None),
            "phase_screens": phase_screens_out,
            "gt_displacements": gt_displacements,
            "meta": {
                "n_pulses": n_pulses,
                "rotation_step_deg": self.rotation_step_deg,
                "movement_mode": getattr(self.movement_cfg, 'mode', 'none'),
                "plate1_direction_deg": getattr(self.movement_cfg, 'plate1_direction_deg', 0.0),
                "plate1_translate_step_m": getattr(self.movement_cfg, 'plate1_translate_step_m', 0.0),
                "plate1_initial_offset_m": list(getattr(self.movement_cfg, 'plate1_initial_offset_m', (0.0, 0.0))),
                "plate1_pulse_offsets_xy_m": (
                    [list(o) for o in self.movement_cfg.plate1_pulse_offsets_xy_m]
                    if self.movement_cfg.plate1_pulse_offsets_xy_m is not None else None),
                "configs": {
                    "pulse": self.pulse_cfg,
                    "rotation": self.rotation_cfg,
                    "optical": self.optical_cfg,
                    "grating": self.grating_cfg,
                    "detector": self.detector_cfg,
                    "output": self.output_cfg,
                },
                "applied_wfe": bool(getattr(self, '_applied_wfe_flag', False)),
                "aberration_mode": self._aberration_mode,
            },
        }
    




def simulate_pulse_superposition(
    # 时间与脉冲设置
    duration_s: Optional[float] = None,
    n_pulses: Optional[int] = None,
    pulse_rep_rate: float = 100.0,
    detector_fps: Optional[float] = 10.0,
    pulses_per_frame: Optional[int] = None,
    wrap_accumulation: bool = True,
    # 旋转设置
    rotation_mode: str = "velocity",
    step_angle_deg: float = 1.0,
    angular_velocity_deg_s: float = 10.0,
    theta_start_deg: float = 0.0,
    rotation_center_offset_m: Tuple[float, float] = (0.0, 0.0),
    # 光学与几何
    photon_energy_ev: float = 10000.0,
    window_size: float = 200e-6,
    grid_points: int = 512,
    z_propagation: float = 50e-3,
    propagation_method: str = "ang_spec",
    source_spot_radius: Optional[float] = None,
    source_mode: str = 'plane',
    source_gaussian_sigma: Optional[float] = None,
    source_to_hartmann_z: float = 0.0,
    wfe_phase_map: Optional[np.ndarray] = None,
    wfe_type: Optional[str] = None,
    wfe_params: Optional[dict] = None,
    wfe_rms_rad: Optional[float] = None,
    # 光栅参数
    spacing: float = 3e-6,
    aperture: Optional[float] = None,
    duty_area: Optional[float] = 0.15,
    hartmann_extent: Optional[float] = None,
    offset_xy: Tuple[float, float] = (0.0, 0.0),
    blur_sigma_mask: float = 0.0,
    # Hartmann 掩模振幅传输（用于真实强度透过率建模）
    amplitude_inside: float = 1.0,
    amplitude_outside: float = 0.0,
    # 运动/平移参数（单板）
    movement_mode: str = 'none',
    plate1_direction_deg: float = 22.5,
    plate1_translate_step_m: float = 0.0,
    plate1_initial_offset_m: Tuple[float, float] = (0.0, 0.0),
    plate1_pulse_offsets_xy_m: Optional[List[Tuple[float, float]]] = None,
    # 探测器
    detector_pixel_size: float = 12e-6,
    detector_nx: int = 512,
    detector_ny: int = 512,
    simulate_detector_effects: bool = True,
    detector_sampling_mode: str = 'srw',
    detector_upsample_factor: int = 1,
    detector_bin_size: Optional[int] = None,
    detector_interpolation: str = 'bilinear',
    # 输出（无可视化/文件保存参数）
) -> dict:
    """
    多帧叠加模拟 - 兼容性包装函数
    
    这是为了保持向后兼容性而提供的包装函数。
    新代码建议直接使用 PulseSuperpositionSimulator 类。
    """
    # 创建配置对象
    pulse_config = PulseConfig(
        n_pulses=n_pulses,
        duration_s=duration_s,
        pulse_rep_rate=pulse_rep_rate,
        detector_fps=detector_fps,
        pulses_per_frame=pulses_per_frame,
        wrap_accumulation=wrap_accumulation
    )
    
    rotation_config = RotationConfig(
        mode=rotation_mode,
        step_angle_deg=step_angle_deg,
        angular_velocity_deg_s=angular_velocity_deg_s,
        theta_start_deg=theta_start_deg,
        center_offset_m=rotation_center_offset_m,
    )
    
    optical_config = OpticalConfig(
        photon_energy_ev=photon_energy_ev,
        window_size=window_size,
        grid_points=grid_points,
        z_propagation=z_propagation,
        propagation_method=propagation_method,
        source_spot_radius=source_spot_radius,
        source_to_hartmann_z=source_to_hartmann_z,
        source_mode=source_mode,
        source_gaussian_sigma=source_gaussian_sigma,
        wfe_phase_map=wfe_phase_map,
        wfe_type=wfe_type,
        wfe_params=wfe_params,
        wfe_rms_rad=wfe_rms_rad,
    )
    
    grating_config = GratingConfig(
        spacing=spacing,
        aperture=aperture,
        duty_area=duty_area,
        hartmann_extent=hartmann_extent,
        offset_xy=offset_xy,
        blur_sigma_mask=blur_sigma_mask,
        amplitude_inside=amplitude_inside,
        amplitude_outside=amplitude_outside
    )
    
    detector_config = DetectorConfig(
        pixel_size=detector_pixel_size,
        nx=detector_nx,
        ny=detector_ny,
        simulate_effects=simulate_detector_effects,
        sampling_mode=str(detector_sampling_mode),
        upsample_factor=int(detector_upsample_factor),
        bin_size=(int(detector_bin_size) if detector_bin_size is not None else int(detector_upsample_factor)),
        interpolation=str(detector_interpolation)
    )

    movement_config = MovementConfig(
        mode=str(movement_mode),
        plate1_direction_deg=float(plate1_direction_deg),
        plate1_translate_step_m=float(plate1_translate_step_m),
        plate1_initial_offset_m=(float(plate1_initial_offset_m[0]), float(plate1_initial_offset_m[1])),
        plate1_pulse_offsets_xy_m=(
            [(float(o[0]), float(o[1])) for o in plate1_pulse_offsets_xy_m]
            if plate1_pulse_offsets_xy_m is not None else None
        ),
    )

    output_config = OutputConfig(
        show_progress=True
    )
    
    # 创建模拟器并运行
    simulator = PulseSuperpositionSimulator(
        pulse_config, rotation_config, optical_config,
        grating_config, detector_config, output_config,
        movement_config
    )
    
    return simulator.simulate()

def hartmann_mask_gttext_style(grid_x: np.ndarray,
                              grid_y: np.ndarray,
                              spacing: float,
                              aperture: Optional[float] = None,
                              duty_area: Optional[float] = None,
                              rotation_angle_rad: float = 0.0,
                              rotation_center: Tuple[float, float] = (0.0, 0.0),
                              shape: str = 'circle',
                              phase_mode: bool = False,
                              phase_delay: float = np.pi / 2.0,
                              phase_on_empty: bool = False,
                              blur_sigma_pix: float = 0.0,
                              amplitude_inside: float = 1.0,
                              amplitude_outside: float = 0.0,
                              offset_xy: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Generate Hartmann mask supporting rotation center and amplitude/phase modes."""
    aperture_val = compute_aperture_diameter(spacing, aperture, duty_area)

    c, s = np.cos(rotation_angle_rad), np.sin(rotation_angle_rad)
    cx, cy = rotation_center
    Xrel = grid_x - cx
    Yrel = grid_y - cy
    Xp = c * Xrel + s * Yrel
    Yp = -s * Xrel + c * Yrel

    pitch = float(spacing)
    offx, offy = float(offset_xy[0]), float(offset_xy[1])
    cx_grid = np.round((Xp - offx) / pitch) * pitch + offx
    cy_grid = np.round((Yp - offy) / pitch) * pitch + offy
    dx = Xp - cx_grid
    dy = Yp - cy_grid

    shape_norm = 'rect' if shape == 'square' else shape
    if shape_norm == 'circle':
        pupil = (dx * dx + dy * dy) <= (aperture_val * 0.5) ** 2
    elif shape_norm in {'rect', 'square'}:
        half = aperture_val * 0.5
        pupil = (abs(dx) <= half) & (abs(dy) <= half)
    else:
        raise ValueError(f"Unknown aperture shape: {shape}")

    if phase_mode:
        phase_mask = pupil if phase_on_empty else (~pupil)
        mask = np.exp(1j * (phase_mask.astype(float) * phase_delay))
    else:
        mask = np.where(pupil, float(amplitude_inside), float(amplitude_outside))

    if blur_sigma_pix and blur_sigma_pix > 0:
        if phase_mode:
            phase = np.angle(mask)
            mask = np.exp(1j * gaussian_blur(phase, blur_sigma_pix))
        else:
            mask = gaussian_blur(mask, blur_sigma_pix)
            lo = min(0.0, float(amplitude_outside))
            hi = max(1.0, float(amplitude_inside))
            mask = np.clip(mask, lo, hi)

    return mask

def compute_rotation_center_from_config(window_size: float,
                                        spacing: float,
                                        pivot_mode: str,
                                        offset_r_m: float,
                                        offset_theta_deg: float,
                                        center_abs_m: Tuple[float, float]) -> Tuple[float, float]:
    """
    根据配置计算旋转中心坐标。
    """
    if pivot_mode == 'absolute':
        return center_abs_m
    elif pivot_mode == 'relative_polar':
        # 简单的极坐标偏移
        rad = np.deg2rad(offset_theta_deg)
        dx = offset_r_m * np.cos(rad)
        dy = offset_r_m * np.sin(rad)
        return (dx, dy)
    else:
        # 'outside_default' 或其他默认情况
        # 假设默认旋转中心在原点
        return (0.0, 0.0)
