from __future__ import annotations
"""
光斑模板提取与渲染模块 (Spot Template Extraction and Rendering Module)

================================================================================
模块概述
================================================================================

本模块实现了从累积Hartmann图像中提取光斑模板，并用于优化重建的功能。
相比传统高斯假设，模板方法可以更精确地拟合真实光斑形状（如Airy斑、像差变形等）。

核心思想:
---------
1. 从参考图像中提取一个真实光斑作为"模板"
2. 使用该模板替代高斯函数进行图像渲染
3. 在优化过程中，通过插值实现亚像素精度的模板平移
4. 利用预计算的模板梯度实现快速解析梯度计算

工作流程:
---------
                    ┌─────────────────┐
                    │   参考图像      │
                    │ (Reference)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ 自动选择最佳光斑 │
                    │ (auto_select)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   提取模板      │
                    │ (extract)       │
                    │ - 背景减除      │
                    │ - 窗函数加权    │
                    │ - 归一化        │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   准备模板      │
                    │ (prepare)       │
                    │ - 预计算梯度    │
                    │ ∂T/∂y, ∂T/∂x    │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐   ┌───────▼───────┐
│  渲染图像     │   │   计算梯度      │   │   目标函数    │
│ (render)      │   │ (sample_rois)   │   │ (objective)   │
│ 亚像素插值    │   │ 解析梯度        │   │ L-BFGS-B优化  │
└───────────────┘   └─────────────────┘   └───────────────┘

================================================================================
关键算法原理
================================================================================

1. 背景减除 (Background Subtraction)
------------------------------------
   从模板边缘区域（环形区域）估计背景水平，并从整个patch中减去。
   
   原理：光斑中心强度高，边缘强度低趋近背景。
         使用边缘像素的中值作为背景估计（对异常值鲁棒）。
   
   数学表达：
       bg = median(border_pixels)
       patch = patch - bg
       patch = max(patch, 0)  # 确保非负

2. Hanning窗加权 (Windowing)
----------------------------
   在提取的patch上应用2D Hanning窗函数。
   
   原理：减少模板边缘的跳变，避免在优化时产生边界伪影。
         Hanning窗在边界平滑过渡到0，减少Gibbs现象。
   
   数学表达：
       window_1d[n] = 0.5 * (1 - cos(2π*n/(N-1)))
       window_2d = outer(window_y, window_x)
       patch = patch * window_2d

3. 归一化 (Normalization)
-------------------------
   将模板归一化为单位"能量"。
   
   模式:
   - 'sum':  归一化使 sum(template) = 1  (用于亮度加权)
   - 'max':  归一化使 max(template) = 1  (保持峰值)
   - None:   不归一化

4. 亚像素插值 (Subpixel Interpolation)
--------------------------------------
   使用 scipy.ndimage.map_coordinates 实现光斑中心的亚像素平移。
   
   原理：当光斑中心 `(center_x_px, center_y_px)` 不在整数像素位置时，
         需要对模板进行插值采样。
   
   坐标变换：
       ty = (Y_roi - center_y_px) + center_offset
       tx = (X_roi - center_x_px) + center_offset
   
   其中 center_offset = (template_size - 1) / 2
   确保当 `(center_x_px, center_y_px)` 与像素中心对齐时，
   采样位置正好是模板中心。

5. 解析梯度计算 (Analytical Gradient)
--------------------------------------
   目标函数: L = ||I_pred - I_obs||²
   
   其中 I_pred = Σ T((y-center_y_px), (x-center_x_px))  对所有光斑求和
   
   求导 (使用链式法则):
       ∂L/∂center_y = 2 * Σ (I_pred - I_obs) * ∂I_pred/∂center_y
                    = 2 * Σ (I_pred - I_obs) * (-∂T/∂y)
              = -2 * Σ diff * dTdy
   
   类似地:
       ∂L/∂center_x = -2 * Σ diff * dTdx
   
   注意负号来自于链式法则：
       ∂T(y-center_y, x-center_x)/∂center_y
       = -∂T/∂(y-center_y) = -∂T/∂y

================================================================================
使用示例
================================================================================

```python
from spot_template import (
    extract_spot_template,
    prepare_template,
    auto_select_template_center,
    sample_template_rois,
    fast_render_template_spots
)
import numpy as np

# 1. 从参考图像自动选择最佳光斑位置
reference_image = np.load('reference.npy')
half_size = 10  # 模板半径
center = auto_select_template_center(reference_image, half_size, spacing=40)
print(f"Selected template center: {center}")

# 2. 提取模板
template = extract_spot_template(
    reference_image, 
    center, 
    half_size,
    bg_ring=2,        # 背景估计使用的边缘宽度
    normalize='sum',  # 归一化方式
    window=True       # 应用Hanning窗
)

# 3. 准备模板（预计算梯度）
template_pack = prepare_template(template)
# template_pack = (template, dTdy, dTdx, radius, center_offset)

# 4. 渲染图像
H, W = reference_image.shape
Y_grid, X_grid = np.mgrid[:H, :W].astype(np.float64)
spot_centers = np.array([[200.3, 100.5], [250.8, 150.2]])  # (x, y) 亚像素精度
rendered = fast_render_template_spots(
    (H, W), spot_centers, template_pack, Y_grid, X_grid
)

# 5. 采样模板及其梯度（用于优化）
cx, cy = 200.3, 100.5
Y_roi = Y_grid[90:112, 190:212]
X_roi = X_grid[90:112, 190:212]
T_roi, dTdy_roi, dTdx_roi = sample_template_rois(template_pack, Y_roi, X_roi, cx, cy)
```

================================================================================
"""

import numpy as np
from typing import Tuple, Optional, Union
from analysis.optimization_common import assert_xy_points

__all__ = [
    'extract_spot_template',
    'prepare_template', 
    'auto_select_template_center',
    'sample_template_rois',
    'fast_render_template_spots',
    'TemplatePack',
]


# Type alias for prepared template data
TemplatePack = Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]


def _hann2d(h: int, w: int) -> np.ndarray:
    """
    生成2D Hanning窗函数。
    
    Hanning窗的定义:
        w[n] = 0.5 * (1 - cos(2π*n/(N-1)))
    
    2D窗通过1D窗的外积构造:
        W_2D = outer(W_y, W_x)
    
    Parameters
    ----------
    h : int
        窗口高度
    w : int
        窗口宽度
        
    Returns
    -------
    window : ndarray of shape (h, w)
        2D Hanning窗，值域 [0, 1]
    """
    wy = np.hanning(h) if h > 1 else np.ones(1)
    wx = np.hanning(w) if w > 1 else np.ones(1)
    return np.outer(wy, wx)


def extract_spot_template(
    image: np.ndarray,
    center: Tuple[float, float],
    half_size: int,
    bg_ring: int = 2,
    normalize: Optional[str] = 'sum',
    window: bool = True
) -> np.ndarray:
    """
    从图像中提取光斑模板。
    
    处理步骤:
    1. 以 center 为中心，提取 (2*half_size+1) x (2*half_size+1) 的方形区域
    2. 使用边缘环形区域估计背景并减除
    3. 裁剪负值到0
    4. 可选应用Hanning窗加权
    5. 可选归一化
    
    背景减除原理:
    ┌─────────────────────┐
    │  bg_ring 宽度边缘   │  ← 使用这部分像素估计背景
    │  ┌───────────────┐  │
    │  │               │  │
    │  │   光斑区域    │  │
    │  │               │  │
    │  └───────────────┘  │
    └─────────────────────┘
    
    Parameters
    ----------
    image : ndarray
        输入图像 (H, W)
    center : tuple of (x, y)
        光斑中心坐标（像素单位，遵循项目 `(x_px, y_px)` 约定）
    half_size : int
        模板半径，输出尺寸为 (2*half_size+1) x (2*half_size+1)
    bg_ring : int, default=2
        用于背景估计的边缘宽度（像素）。设为0禁用背景减除。
    normalize : str or None, default='sum'
        归一化模式:
        - 'sum': sum(template) = 1
        - 'max': max(template) = 1
        - None: 不归一化
    window : bool, default=True
        是否应用Hanning窗加权
        
    Returns
    -------
    template : ndarray of shape (2*half_size+1, 2*half_size+1)
        提取的光斑模板
        
    Raises
    ------
    ValueError
        如果模板窗口超出图像边界或参数无效
    
    Notes
    -----
    - 选择 center 时应确保周围有足够空间
    - half_size 应大于光斑的视觉半径
    - bg_ring 不应太大，否则会包含其他光斑的信号
    """
    # 类型转换
    image = image.astype(np.float64, copy=False)
    center_x_px, center_y_px = float(center[0]), float(center[1])
    half_size = int(half_size)
    
    if half_size <= 0:
        raise ValueError(f"half_size must be > 0, got {half_size}")

    h, w = image.shape
    
    # 计算提取区域边界（四舍五入到最近整数）
    y0 = int(round(center_y_px)) - half_size
    y1 = int(round(center_y_px)) + half_size + 1
    x0 = int(round(center_x_px)) - half_size
    x1 = int(round(center_x_px)) + half_size + 1
    
    # 边界检查
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        raise ValueError(
            f"Template window [{y0}:{y1}, {x0}:{x1}] exceeds image bounds [{h}, {w}]. "
            "Choose a more central spot or smaller half_size."
        )

    # 提取patch
    patch = image[y0:y1, x0:x1].copy()

    # 背景减除
    bg_ring = int(bg_ring)
    if bg_ring > 0 and patch.shape[0] > 2 * bg_ring and patch.shape[1] > 2 * bg_ring:
        # 提取边缘像素
        top = patch[:bg_ring, :]
        bottom = patch[-bg_ring:, :]
        left = patch[:, :bg_ring]
        right = patch[:, -bg_ring:]
        
        # 使用中值估计背景（对异常值鲁棒）
        border_pixels = np.concatenate([
            top.ravel(), bottom.ravel(), 
            left.ravel(), right.ravel()
        ])
        bg = np.median(border_pixels)
        patch -= bg

    # 裁剪负值（背景可能高估）
    patch = np.clip(patch, 0.0, None)
    
    # Hanning窗加权
    if window:
        patch *= _hann2d(patch.shape[0], patch.shape[1])

    # 归一化
    if normalize == 'sum':
        denom = float(patch.sum())
        if denom > 0:
            patch /= denom
    elif normalize == 'max':
        denom = float(patch.max())
        if denom > 0:
            patch /= denom
    elif normalize is None:
        pass
    else:
        raise ValueError(f"Unknown normalize mode: {normalize}")

    return patch


def prepare_template(template: np.ndarray) -> TemplatePack:
    """
    准备模板用于优化：预计算梯度场。
    
    梯度预计算原理:
    ---------------
    在优化过程中，需要反复计算目标函数关于光斑位置的梯度。
    对于模板模型:
        ∂I/∂cy = -∂T/∂y  (负号来自链式法则)
        ∂I/∂cx = -∂T/∂x
    
    预计算梯度场 dTdy, dTdx 可以避免每次迭代都重新计算。
    
    Parameters
    ----------
    template : ndarray
        必须是方阵且边长为奇数，如 (21, 21)
        
    Returns
    -------
    template_pack : tuple
        包含 (template, dTdy, dTdx, radius, center_offset)
        - template: 原始模板
        - dTdy: ∂T/∂y 梯度场
        - dTdx: ∂T/∂x 梯度场
        - radius: 模板半径 = (size - 1) / 2
        - center_offset: 等于 radius，用于坐标变换
        
    Raises
    ------
    ValueError
        如果模板不是方阵或边长不是奇数
    
    Notes
    -----
    梯度使用 np.gradient 计算，采用中心差分格式:
        ∂f/∂x[i] ≈ (f[i+1] - f[i-1]) / 2
    边界处使用单侧差分。
    """
    template = template.astype(np.float64, copy=False)
    
    if template.ndim != 2:
        raise ValueError("template must be a 2D array")
    if template.shape[0] != template.shape[1] or template.shape[0] % 2 != 1:
        raise ValueError(
            f"template must be square with odd size, got {template.shape}. "
            f"Use half_size such that 2*half_size+1 is odd."
        )
    
    # 计算梯度场（中心差分）
    dTdy, dTdx = np.gradient(template)
    
    # 计算模板参数
    radius = (template.shape[0] - 1) // 2
    center_offset = float(radius)
    
    return template, dTdy, dTdx, radius, center_offset


def auto_select_template_center(
    image: np.ndarray,
    half_size: int,
    spacing: Optional[float] = None,
    max_candidates: int = 200
) -> Tuple[float, float]:
    """
    自动选择最佳光斑位置用于模板提取。
    
    选择策略:
    ---------
    1. 局部极值检测：找到图像中所有局部最大值点
    2. 阈值过滤：只保留亮度超过99分位数的点
    3. 边缘过滤：排除靠近边界的点（确保模板提取有足够空间）
    4. 得分排序：计算每个候选点的"信噪比"得分
       score = peak_intensity - ring_median
    5. 选择得分最高的点
    
    打分原理:
    ---------
    一个理想的模板光斑应该:
    - 峰值强度高（光斑明亮）
    - 周围背景低（避免重叠光斑干扰）
    - 距离边界足够远（可以提取完整模板）
    
    得分 = 峰值强度 - 环形区域中值
    环形区域: half_size * 0.55 < r < half_size
    
    Parameters
    ----------
    image : ndarray
        输入图像
    half_size : int
        模板半径
    spacing : float, optional
        光斑间距（用于调整局部极值检测窗口大小）
    max_candidates : int, default=200
        最大候选点数量
        
    Returns
    -------
    center : tuple of (x, y)
        最佳光斑中心坐标，遵循项目 `(x_px, y_px)` 约定
        
    Raises
    ------
    ValueError
        如果没有找到满足条件的光斑
    """
    from scipy.ndimage import maximum_filter
    
    image = image.astype(np.float64, copy=False)
    h, w = image.shape
    half_size = int(half_size)
    margin = half_size + 2  # 安全边距

    # 确定局部极值检测窗口大小
    if spacing is None:
        filter_size = max(9, 2 * half_size + 1)
    else:
        filter_size = max(9, int(spacing * 0.8))

    # Step 1: 局部极值检测
    local_max = maximum_filter(image, size=filter_size)
    thresh = np.percentile(image, 99)  # 只考虑最亮的1%
    mask = (image == local_max) & (image > thresh)
    peaks = np.array(np.where(mask)).T  # (N, 2) 格式
    
    if peaks.size == 0:
        raise ValueError("No peak found for template auto-selection.")

    # Step 2: 边缘过滤
    keep = (
        (peaks[:, 0] >= margin) &
        (peaks[:, 0] < h - margin) &
        (peaks[:, 1] >= margin) &
        (peaks[:, 1] < w - margin)
    )
    peaks = peaks[keep]
    
    if peaks.size == 0:
        raise ValueError(
            "No peak remains after margin filtering. "
            "Try reducing half_size or using an image with more central spots."
        )

    # Step 3: 按强度排序，保留top候选
    values = image[peaks[:, 0], peaks[:, 1]]
    order = np.argsort(values)[::-1][:max_candidates]
    peaks = peaks[order]
    values = values[order]

    # Step 4: 计算打分用的环形区域掩模
    yy, xx = np.ogrid[-half_size:half_size + 1, -half_size:half_size + 1]
    rr = np.sqrt(yy**2 + xx**2)
    r1 = max(1.0, 0.55 * half_size)  # 内径
    r2 = float(half_size)             # 外径
    ring = (rr >= r1) & (rr <= r2)
    
    if not ring.any():
        ring = rr >= 0  # 回退到整个区域

    # Step 5: 对每个候选点打分
    best_score = -np.inf
    best = None
    
    for (py, px), pv in zip(peaks, values):
        # 提取该点周围的patch
        patch = image[py - half_size:py + half_size + 1, 
                     px - half_size:px + half_size + 1]
        
        # 计算环形区域的中值
        ring_level = np.median(patch[ring])
        
        # 得分 = 峰值 - 背景
        score = float(pv) - float(ring_level)
        
        if score > best_score:
            best_score = score
            best = (float(px), float(py))

    if best is None:
        raise ValueError("Template auto-selection failed unexpectedly.")
    
    return best


def _sample_map_coordinates(
    arr: np.ndarray, 
    y_coords: np.ndarray, 
    x_coords: np.ndarray
) -> np.ndarray:
    """
    使用双线性插值在任意浮点坐标处采样数组。
    
    底层使用 scipy.ndimage.map_coordinates 实现。
    
    Parameters
    ----------
    arr : ndarray
        待采样的2D数组
    y_coords : ndarray
        采样点的y坐标（可以是浮点数）
    x_coords : ndarray
        采样点的x坐标（可以是浮点数）
        
    Returns
    -------
    sampled : ndarray
        与 y_coords/x_coords 形状相同的采样结果
    """
    from scipy.ndimage import map_coordinates
    
    coords = np.vstack([y_coords.ravel(), x_coords.ravel()])
    out = map_coordinates(arr, coords, order=1, mode='constant', cval=0.0)
    return out.reshape(y_coords.shape)


def sample_template_rois(
    template_pack: TemplatePack,
    Y_roi: np.ndarray,
    X_roi: np.ndarray,
    center_x_px: float,
    center_y_px: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在指定ROI区域采样模板及其梯度（支持亚像素中心位置）。
    
    坐标变换原理:
    -------------
    模板是以其中心为原点定义的，即:
        template[radius, radius] 对应 (0, 0) 位置
    
    当光斑中心在 `(center_x_px, center_y_px)` 时，图像坐标 `(Y_roi, X_roi)`
    对应的模板坐标为:
        ty = (Y_roi - center_y_px) + center_offset
        tx = (X_roi - center_x_px) + center_offset
    
    其中 center_offset = radius 确保坐标正确映射。
    
    示例:
    -----
    假设 template.shape = (21, 21), 则 radius = 10, center_offset = 10
    
    当 `(Y_roi[i,j], X_roi[i,j]) = (center_y_px, center_x_px)` 时:
        ty = (center_y_px - center_y_px) + 10 = 10
        tx = (center_x_px - center_x_px) + 10 = 10
    即采样模板的中心点，正确！
    
    当 `(Y_roi[i,j], X_roi[i,j]) = (center_y_px+1, center_x_px+2)` 时:
        ty = (center_y_px+1 - center_y_px) + 10 = 11
        tx = (center_x_px+2 - center_x_px) + 10 = 12
    即采样模板中心偏移 (1, 2) 的位置，正确！
    
    Parameters
    ----------
    template_pack : TemplatePack
        prepare_template() 的返回值
    Y_roi : ndarray
        ROI区域的Y坐标网格
    X_roi : ndarray
        ROI区域的X坐标网格
    center_x_px : float
        光斑中心X坐标（可以是浮点数实现亚像素精度）
    center_y_px : float
        光斑中心Y坐标
        
    Returns
    -------
    T_roi : ndarray
        采样的模板值
    dTdy_roi : ndarray
        采样的 ∂T/∂y 值
    dTdx_roi : ndarray
        采样的 ∂T/∂x 值
    """
    template, dTdy, dTdx, _, center_offset = template_pack
    
    # 坐标变换：从图像坐标到模板坐标
    ty = (Y_roi - center_y_px) + center_offset
    tx = (X_roi - center_x_px) + center_offset
    
    # 双线性插值采样
    T_roi = _sample_map_coordinates(template, ty, tx)
    dTdy_roi = _sample_map_coordinates(dTdy, ty, tx)
    dTdx_roi = _sample_map_coordinates(dTdx, ty, tx)
    
    return T_roi, dTdy_roi, dTdx_roi


def fast_render_template_spots(
    shape: Tuple[int, int],
    centers: np.ndarray,
    template_pack: TemplatePack,
    Y_grid: np.ndarray,
    X_grid: np.ndarray
) -> np.ndarray:
    """
    使用模板快速渲染多个光斑。
    
    对于每个光斑中心，提取其周围的ROI区域，
    使用亚像素插值采样模板，并累加到输出图像。
    
    Parameters
    ----------
    shape : tuple of (H, W)
        输出图像尺寸
    centers : ndarray of shape (n_spots, 2)
        光斑中心坐标 `[(x, y), ...]`，支持浮点数（亚像素精度）
    template_pack : TemplatePack
        prepare_template() 的返回值
    Y_grid : ndarray of shape (H, W)
        Y坐标网格 (通常由 np.mgrid[:H, :W] 生成)
    X_grid : ndarray of shape (H, W)
        X坐标网格
        
    Returns
    -------
    image : ndarray of shape (H, W)
        渲染的图像
        
    Notes
    -----
    时间复杂度: O(n_spots * template_size^2)
    
    对于大量光斑，使用ROI裁剪而非全图计算可显著提速。
    """
    h, w = shape
    grid = np.zeros(shape, dtype=np.float64)
    _, _, _, radius, _ = template_pack
    centers_xy = assert_xy_points(centers, name="centers")

    for center_x_px, center_y_px in centers_xy:
        # 计算ROI边界
        y_min = max(0, int(center_y_px) - radius)
        y_max = min(h, int(center_y_px) + radius + 1)
        x_min = max(0, int(center_x_px) - radius)
        x_max = min(w, int(center_x_px) + radius + 1)
        
        if y_min >= y_max or x_min >= x_max:
            continue
        
        # 提取ROI区域的坐标网格
        Y_roi = Y_grid[y_min:y_max, x_min:x_max]
        X_roi = X_grid[y_min:y_max, x_min:x_max]
        
        # 采样模板（只需要模板值，不需要梯度）
        T_roi, _, _ = sample_template_rois(
            template_pack,
            Y_roi,
            X_roi,
            center_x_px,
            center_y_px,
        )
        
        # 累加到输出图像
        grid[y_min:y_max, x_min:x_max] += T_roi

    return grid


# =============================================================================
# 便捷函数
# =============================================================================

def create_template_from_image(
    image: np.ndarray,
    half_size: int = 10,
    spacing: Optional[float] = None,
    **extract_kwargs
) -> TemplatePack:
    """
    一站式模板创建：自动选择光斑 → 提取模板 → 准备梯度。
    
    Parameters
    ----------
    image : ndarray
        参考图像
    half_size : int, default=10
        模板半径
    spacing : float, optional
        光斑间距（用于优化选择算法）
    **extract_kwargs
        传递给 extract_spot_template 的额外参数
        
    Returns
    -------
    template_pack : TemplatePack
        准备好的模板包
        
    Example
    -------
    >>> template_pack = create_template_from_image(ref_image, half_size=12)
    >>> rendered = fast_render_template_spots(shape, centers, template_pack, Y, X)
    """
    # 自动选择最佳光斑
    center = auto_select_template_center(image, half_size, spacing=spacing)
    
    # 提取模板
    template = extract_spot_template(image, center, half_size, **extract_kwargs)
    
    # 准备模板（预计算梯度）
    return prepare_template(template)
