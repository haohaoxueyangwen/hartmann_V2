from __future__ import annotations
"""
波前误差生成器模块 (Wavefront Error Generator)

本模块用于生成各种类型的波前误差相位图，支持：
1. 简单模式 (simple): 倾斜、离焦、像散
2. 高级模式 (advanced): Zernike多项式 + PSD噪声
3. 几何模式 (geometric): 可视化明显的几何形状（三角形、圆形、矩形、圆环）
4. 条纹模式 (stripe): 水波纹状条纹，每条不同相位值，用于精度标定
5. 阶梯模式 (step): 阶梯状相位变化
6. 正弦波模式 (sinusoidal): 正弦波形式的周期性相位
7. 同心环模式 (concentric): 同心圆环，每环不同相位

使用方法:
    from simulation.wfe_generator import generate_wfe_phase, WFEConfig

    # 创建配置
    config = WFEConfig.geometric_default()
    
    # 生成相位图
    phase_map = generate_wfe_phase(X, Y, config)
    
    # 条纹模式（用于精度标定）
    config = preset_stripe_calibration(n_stripes=10, max_phase=10.0)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union, Literal


# =============================================================================
# 预设配置
# =============================================================================

# 简单模式默认参数
SIMPLE_WFE_DEFAULTS = {
    'tilt_tx': 200.0,       # X方向倾斜 (rad/m)
    'tilt_ty': 0.0,         # Y方向倾斜 (rad/m)
    'defocus_alpha': 0.5,   # 离焦系数
    'astig_alpha': 0.25,    # 像散系数
    'astig_angle_deg': 0.0, # 像散角度 (度)
}

# 高级模式默认参数
ADVANCED_WFE_DEFAULTS = {
    'zernike_coefs': [(2, 8.0), (3, 6.0), (4, 2.0), (5, 1.5), (6, 1.2), (7, 1.0), (8, 1.0)],
    'pupil_radius_m': 8e-3,
    'psd_corr_len_m': 40e-6,
    'psd_rms_rad': 0.0,
}

# 几何模式默认参数 - 用于DPC验证的明显特征
GEOMETRIC_WFE_DEFAULTS = {
    'shapes': [
        {
            'type': 'triangle',
            'center': (-2000e-6, 2000e-6),  # 左上角
            'radius': 1500e-6,
            'term': 'pyramid',
            'amplitude': 20.0,
        },
        {
            'type': 'circle',
            'center': (2000e-6, 2000e-6),   # 右上角
            'radius': 1500e-6,
            'term': 'defocus',
            'amplitude': 15.0,
        },
        {
            'type': 'rectangle',
            'center': (0.0, -2000e-6),      # 底部
            'width': 4000e-6,
            'height': 1000e-6,
            'term': 'tilt',
            'tilt_x': 5000.0,
            'tilt_y': 0.0,
        },
    ]
}

# 条纹模式默认参数 - 水波纹状条纹，用于精度标定
STRIPE_WFE_DEFAULTS = {
    'n_stripes': 10,           # 条纹数量
    'direction': 'horizontal', # 条纹方向: 'horizontal', 'vertical', 'diagonal'
    'phase_values': None,      # 每条条纹的相位值列表（rad），None则自动生成
    'min_phase': 0.0,          # 自动生成时的最小相位 (rad)
    'max_phase': 10.0,         # 自动生成时的最大相位 (rad)
    'profile': 'step',         # 条纹内部轮廓: 'step'(阶梯), 'linear'(线性渐变), 'smooth'(平滑过渡)
    'angle_deg': 0.0,          # 条纹角度（用于diagonal方向）
}

# 阶梯模式默认参数
STEP_WFE_DEFAULTS = {
    'n_steps': 5,              # 阶梯数量
    'direction': 'horizontal', # 阶梯方向
    'step_values': None,       # 每级阶梯的相位值（rad）
    'min_phase': 0.0,
    'max_phase': 5.0,
}

# 正弦波模式默认参数
SINUSOIDAL_WFE_DEFAULTS = {
    'amplitude': 5.0,          # 振幅 (rad)
    'period_m': 2000e-6,       # 周期 (m)
    'direction': 'horizontal', # 波传播方向
    'phase_offset': 0.0,       # 初相位 (rad)
    'n_periods': None,         # 周期数（覆盖period_m）
}

# 同心环模式默认参数
CONCENTRIC_WFE_DEFAULTS = {
    'n_rings': 5,              # 环数量
    'center': (0.0, 0.0),      # 中心位置
    'max_radius_m': 6000e-6,   # 最大半径
    'phase_values': None,      # 每环相位值
    'min_phase': 0.0,
    'max_phase': 10.0,
    'profile': 'step',         # 'step' 或 'smooth'
}


# =============================================================================
# 配置数据类
# =============================================================================

@dataclass
class WFEConfig:
    """波前误差配置类"""
    
    mode: Literal['simple', 'advanced', 'geometric', 'stripe', 'step', 'sinusoidal', 'concentric', 'none'] = 'none'
    """WFE模式: 'simple', 'advanced', 'geometric', 'stripe', 'step', 'sinusoidal', 'concentric', 'none'"""
    
    # 简单模式参数
    tilt_tx: float = 0.0
    """X方向倾斜 (rad/m)"""
    tilt_ty: float = 0.0
    """Y方向倾斜 (rad/m)"""
    defocus_alpha: float = 0.0
    """离焦系数"""
    astig_alpha: float = 0.0
    """像散系数"""
    astig_angle_deg: float = 0.0
    """像散角度 (度)"""
    
    # 高级模式参数
    zernike_coefs: List[Tuple[int, float]] = field(default_factory=list)
    """Zernike多项式系数列表 [(noll_index, amplitude), ...]"""
    pupil_radius_m: float = 8e-3
    """光瞳半径 (m)"""
    psd_corr_len_m: float = 0.0
    """PSD相关长度 (m)"""
    psd_rms_rad: float = 0.0
    """PSD RMS值 (rad)"""
    
    # 几何模式参数
    shapes: List[Dict[str, Any]] = field(default_factory=list)
    """几何形状列表，每个形状是一个字典"""
    
    # 条纹/阶梯/正弦/同心环模式参数
    stripe_params: Dict[str, Any] = field(default_factory=dict)
    """条纹模式参数"""
    step_params: Dict[str, Any] = field(default_factory=dict)
    """阶梯模式参数"""
    sinusoidal_params: Dict[str, Any] = field(default_factory=dict)
    """正弦波模式参数"""
    concentric_params: Dict[str, Any] = field(default_factory=dict)
    """同心环模式参数"""
    
    # 全局倾斜叠加
    global_tilt_tx: float = 0.0
    """全局X倾斜，叠加在所有模式上 (rad/m)"""
    global_tilt_ty: float = 0.0
    """全局Y倾斜，叠加在所有模式上 (rad/m)"""
    
    @classmethod
    def none(cls) -> 'WFEConfig':
        """创建无WFE的配置"""
        return cls(mode='none')
    
    @classmethod
    def simple(cls, 
               tilt_tx: float = 200.0,
               tilt_ty: float = 0.0,
               defocus_alpha: float = 0.5,
               astig_alpha: float = 0.25,
               astig_angle_deg: float = 0.0) -> 'WFEConfig':
        """创建简单模式配置"""
        return cls(
            mode='simple',
            tilt_tx=tilt_tx,
            tilt_ty=tilt_ty,
            defocus_alpha=defocus_alpha,
            astig_alpha=astig_alpha,
            astig_angle_deg=astig_angle_deg,
        )
    
    @classmethod
    def advanced(cls,
                 zernike_coefs: List[Tuple[int, float]] = None,
                 pupil_radius_m: float = 8e-3,
                 psd_corr_len_m: float = 40e-6,
                 psd_rms_rad: float = 0.0) -> 'WFEConfig':
        """创建高级模式配置"""
        if zernike_coefs is None:
            zernike_coefs = ADVANCED_WFE_DEFAULTS['zernike_coefs']
        return cls(
            mode='advanced',
            zernike_coefs=zernike_coefs,
            pupil_radius_m=pupil_radius_m,
            psd_corr_len_m=psd_corr_len_m,
            psd_rms_rad=psd_rms_rad,
        )
    
    @classmethod
    def geometric(cls, shapes: List[Dict[str, Any]] = None) -> 'WFEConfig':
        """创建几何模式配置"""
        if shapes is None:
            shapes = GEOMETRIC_WFE_DEFAULTS['shapes']
        return cls(mode='geometric', shapes=shapes)
    
    @classmethod
    def geometric_default(cls) -> 'WFEConfig':
        """创建默认的几何模式配置（用于DPC验证）"""
        return cls.geometric(GEOMETRIC_WFE_DEFAULTS['shapes'])
    
    @classmethod
    def stripe(cls, 
               n_stripes: int = 10,
               direction: str = 'horizontal',
               phase_values: List[float] = None,
               min_phase: float = 0.0,
               max_phase: float = 10.0,
               profile: str = 'step',
               angle_deg: float = 0.0) -> 'WFEConfig':
        """
        创建条纹模式配置（水波纹状，用于精度标定）
        
        Parameters
        ----------
        n_stripes : int
            条纹数量
        direction : str
            条纹方向: 'horizontal', 'vertical', 'diagonal'
        phase_values : List[float], optional
            每条条纹的相位值（rad），None则自动从min_phase到max_phase线性分布
        min_phase, max_phase : float
            自动生成相位值时的范围 (rad)
        profile : str
            条纹内部轮廓: 'step'(阶跃), 'linear'(线性渐变), 'smooth'(平滑过渡)
        angle_deg : float
            条纹角度（用于diagonal方向）
        """
        params = {
            'n_stripes': n_stripes,
            'direction': direction,
            'phase_values': phase_values,
            'min_phase': min_phase,
            'max_phase': max_phase,
            'profile': profile,
            'angle_deg': angle_deg,
        }
        return cls(mode='stripe', stripe_params=params)
    
    @classmethod
    def step(cls,
             n_steps: int = 5,
             direction: str = 'horizontal',
             step_values: List[float] = None,
             min_phase: float = 0.0,
             max_phase: float = 5.0) -> 'WFEConfig':
        """
        创建阶梯模式配置
        
        Parameters
        ----------
        n_steps : int
            阶梯数量
        direction : str
            阶梯方向: 'horizontal', 'vertical'
        step_values : List[float], optional
            每级阶梯的相位值（rad）
        """
        params = {
            'n_steps': n_steps,
            'direction': direction,
            'step_values': step_values,
            'min_phase': min_phase,
            'max_phase': max_phase,
        }
        return cls(mode='step', step_params=params)
    
    @classmethod
    def sinusoidal(cls,
                   amplitude: float = 5.0,
                   period_m: float = 2000e-6,
                   direction: str = 'horizontal',
                   phase_offset: float = 0.0,
                   n_periods: int = None) -> 'WFEConfig':
        """
        创建正弦波模式配置
        
        Parameters
        ----------
        amplitude : float
            振幅 (rad)
        period_m : float
            周期 (m)
        direction : str
            波传播方向: 'horizontal', 'vertical', 'radial'
        phase_offset : float
            初相位 (rad)
        n_periods : int, optional
            周期数（如果指定，会覆盖period_m计算）
        """
        params = {
            'amplitude': amplitude,
            'period_m': period_m,
            'direction': direction,
            'phase_offset': phase_offset,
            'n_periods': n_periods,
        }
        return cls(mode='sinusoidal', sinusoidal_params=params)
    
    @classmethod
    def concentric(cls,
                   n_rings: int = 5,
                   center: Tuple[float, float] = (0.0, 0.0),
                   max_radius_m: float = 6000e-6,
                   phase_values: List[float] = None,
                   min_phase: float = 0.0,
                   max_phase: float = 10.0,
                   profile: str = 'step') -> 'WFEConfig':
        """
        创建同心环模式配置
        
        Parameters
        ----------
        n_rings : int
            环数量
        center : Tuple[float, float]
            中心位置 (m)
        max_radius_m : float
            最大半径 (m)
        phase_values : List[float], optional
            每环相位值（rad）
        profile : str
            'step'(阶跃) 或 'smooth'(平滑过渡)
        """
        params = {
            'n_rings': n_rings,
            'center': center,
            'max_radius_m': max_radius_m,
            'phase_values': phase_values,
            'min_phase': min_phase,
            'max_phase': max_phase,
            'profile': profile,
        }
        return cls(mode='concentric', concentric_params=params)
    
    def with_global_tilt(self, tx: float, ty: float = 0.0) -> 'WFEConfig':
        """添加全局倾斜（返回新配置）"""
        import copy
        new_cfg = copy.deepcopy(self)
        new_cfg.global_tilt_tx = tx
        new_cfg.global_tilt_ty = ty
        return new_cfg
    
    def to_wfe_type_string(self) -> str:
        """转换为hartmann_simulator使用的wfe_type字符串格式"""
        parts = []
        if self.mode == 'simple':
            if self.tilt_tx != 0 or self.tilt_ty != 0:
                parts.append('tilt')
            if self.defocus_alpha != 0:
                parts.append('defocus')
            if self.astig_alpha != 0:
                parts.append('astig')
        elif self.mode == 'advanced':
            if self.zernike_coefs:
                parts.append('zernike')
            if self.psd_rms_rad > 0 and self.psd_corr_len_m > 0:
                parts.append('psd')
        elif self.mode == 'geometric':
            parts.append('geometric')
        elif self.mode == 'stripe':
            parts.append('stripe')
        elif self.mode == 'step':
            parts.append('step')
        elif self.mode == 'sinusoidal':
            parts.append('sinusoidal')
        elif self.mode == 'concentric':
            parts.append('concentric')
        
        # 全局倾斜
        if self.global_tilt_tx != 0 or self.global_tilt_ty != 0:
            if 'tilt' not in parts:
                parts.append('tilt')
        
        return '+'.join(parts) if parts else ''
    
    def to_wfe_params(self) -> Dict[str, Any]:
        """转换为hartmann_simulator使用的wfe_params字典格式"""
        params = {}
        
        if self.mode == 'simple':
            params['tilt_tx'] = self.tilt_tx + self.global_tilt_tx
            params['tilt_ty'] = self.tilt_ty + self.global_tilt_ty
            params['defocus_alpha'] = self.defocus_alpha
            params['astig_alpha'] = self.astig_alpha
            params['astig_angle_deg'] = self.astig_angle_deg
        elif self.mode == 'advanced':
            params['zernike_coefs'] = self.zernike_coefs
            params['pupil_radius_m'] = self.pupil_radius_m
            params['psd_corr_len_m'] = self.psd_corr_len_m
            params['psd_rms_rad'] = self.psd_rms_rad
            # 添加全局倾斜
            if self.global_tilt_tx != 0 or self.global_tilt_ty != 0:
                params['tilt_tx'] = self.global_tilt_tx
                params['tilt_ty'] = self.global_tilt_ty
        elif self.mode == 'geometric':
            params['shapes'] = self.shapes
            # 全局倾斜作为额外参数
            if self.global_tilt_tx != 0 or self.global_tilt_ty != 0:
                params['tilt_tx'] = self.global_tilt_tx
                params['tilt_ty'] = self.global_tilt_ty
        elif self.mode == 'stripe':
            params['stripe_params'] = self.stripe_params
            if self.global_tilt_tx != 0 or self.global_tilt_ty != 0:
                params['tilt_tx'] = self.global_tilt_tx
                params['tilt_ty'] = self.global_tilt_ty
        elif self.mode == 'step':
            params['step_params'] = self.step_params
            if self.global_tilt_tx != 0 or self.global_tilt_ty != 0:
                params['tilt_tx'] = self.global_tilt_tx
                params['tilt_ty'] = self.global_tilt_ty
        elif self.mode == 'sinusoidal':
            params['sinusoidal_params'] = self.sinusoidal_params
            if self.global_tilt_tx != 0 or self.global_tilt_ty != 0:
                params['tilt_tx'] = self.global_tilt_tx
                params['tilt_ty'] = self.global_tilt_ty
        elif self.mode == 'concentric':
            params['concentric_params'] = self.concentric_params
            if self.global_tilt_tx != 0 or self.global_tilt_ty != 0:
                params['tilt_tx'] = self.global_tilt_tx
                params['tilt_ty'] = self.global_tilt_ty
        else:
            # none模式，但可能有全局倾斜
            if self.global_tilt_tx != 0 or self.global_tilt_ty != 0:
                params['tilt_tx'] = self.global_tilt_tx
                params['tilt_ty'] = self.global_tilt_ty
        
        return params


# =============================================================================
# 几何形状掩模生成
# =============================================================================

def create_circle_mask(X: np.ndarray, Y: np.ndarray, 
                       center: Tuple[float, float], 
                       radius: float) -> np.ndarray:
    """创建圆形掩模"""
    cx, cy = center
    return ((X - cx)**2 + (Y - cy)**2) <= radius**2


def create_rectangle_mask(X: np.ndarray, Y: np.ndarray,
                          center: Tuple[float, float],
                          width: float, height: float) -> np.ndarray:
    """创建矩形掩模"""
    cx, cy = center
    return (np.abs(X - cx) <= width/2) & (np.abs(Y - cy) <= height/2)


def create_triangle_mask(X: np.ndarray, Y: np.ndarray,
                         center: Tuple[float, float],
                         radius: float) -> np.ndarray:
    """创建等边三角形掩模（尖朝上）"""
    cx, cy = center
    diff_x, diff_y = X - cx, Y - cy
    # 三条边的半平面
    c1 = diff_y >= -0.5 * radius  # 底边
    c2 = (diff_y - 1.732 * diff_x) <= radius  # 左边
    c3 = (diff_y + 1.732 * diff_x) <= radius  # 右边
    return c1 & c2 & c3


def create_ring_mask(X: np.ndarray, Y: np.ndarray,
                     center: Tuple[float, float],
                     radius_outer: float, radius_inner: float) -> np.ndarray:
    """创建圆环掩模"""
    cx, cy = center
    r2 = (X - cx)**2 + (Y - cy)**2
    return (r2 <= radius_outer**2) & (r2 >= radius_inner**2)


def create_ellipse_mask(X: np.ndarray, Y: np.ndarray,
                        center: Tuple[float, float],
                        semi_major: float, semi_minor: float,
                        angle_deg: float = 0.0) -> np.ndarray:
    """创建椭圆掩模"""
    cx, cy = center
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx, dy = X - cx, Y - cy
    # 旋转坐标
    x_rot = dx * cos_a + dy * sin_a
    y_rot = -dx * sin_a + dy * cos_a
    return (x_rot / semi_major)**2 + (y_rot / semi_minor)**2 <= 1.0


def create_polygon_mask(X: np.ndarray, Y: np.ndarray,
                        vertices: List[Tuple[float, float]]) -> np.ndarray:
    """创建多边形掩模（射线法）"""
    from matplotlib.path import Path
    vertices = np.array(vertices)
    path = Path(vertices)
    points = np.column_stack([X.ravel(), Y.ravel()])
    mask = path.contains_points(points).reshape(X.shape)
    return mask


# =============================================================================
# 相位项生成
# =============================================================================

def apply_piston(phi: np.ndarray, mask: np.ndarray, amplitude: float) -> None:
    """应用活塞相位（常数相位偏移）"""
    phi[mask] += amplitude


def apply_tilt(phi: np.ndarray, mask: np.ndarray, 
               X: np.ndarray, Y: np.ndarray,
               center: Tuple[float, float],
               tilt_x: float, tilt_y: float = 0.0) -> None:
    """应用倾斜相位"""
    cx, cy = center
    phi[mask] += tilt_x * (X[mask] - cx) + tilt_y * (Y[mask] - cy)


def apply_defocus(phi: np.ndarray, mask: np.ndarray,
                  X: np.ndarray, Y: np.ndarray,
                  center: Tuple[float, float],
                  amplitude: float, r_max: float) -> None:
    """应用离焦相位（二次曲面）"""
    cx, cy = center
    r2 = (X[mask] - cx)**2 + (Y[mask] - cy)**2
    phi[mask] += amplitude * (r2 / (r_max**2))


def apply_pyramid(phi: np.ndarray, mask: np.ndarray,
                  X: np.ndarray, Y: np.ndarray,
                  center: Tuple[float, float],
                  amplitude: float, r_max: float) -> None:
    """应用棱锥相位（线性锥形）"""
    cx, cy = center
    r_local = np.sqrt((X[mask] - cx)**2 + (Y[mask] - cy)**2)
    phi[mask] += amplitude * (1.0 - r_local / r_max)


def apply_spherical(phi: np.ndarray, mask: np.ndarray,
                    X: np.ndarray, Y: np.ndarray,
                    center: Tuple[float, float],
                    amplitude: float, r_max: float) -> None:
    """应用球面相位（四次项）"""
    cx, cy = center
    r2 = (X[mask] - cx)**2 + (Y[mask] - cy)**2
    rho4 = (r2 / (r_max**2))**2
    phi[mask] += amplitude * (6 * rho4 - 6 * (r2 / (r_max**2)) + 1)


def apply_coma(phi: np.ndarray, mask: np.ndarray,
               X: np.ndarray, Y: np.ndarray,
               center: Tuple[float, float],
               amplitude: float, r_max: float,
               angle_deg: float = 0.0) -> None:
    """应用彗差相位"""
    cx, cy = center
    dx, dy = X[mask] - cx, Y[mask] - cy
    r = np.sqrt(dx**2 + dy**2)
    rho = r / r_max
    theta = np.arctan2(dy, dx) - np.deg2rad(angle_deg)
    phi[mask] += amplitude * (3 * rho**3 - 2 * rho) * np.cos(theta)


# =============================================================================
# 主生成函数
# =============================================================================

def generate_geometric_wfe(X: np.ndarray, Y: np.ndarray, 
                           shapes: List[Dict[str, Any]]) -> np.ndarray:
    """
    根据几何形状配置生成WFE相位图
    
    Parameters
    ----------
    X : np.ndarray
        X坐标网格
    Y : np.ndarray
        Y坐标网格
    shapes : List[Dict]
        形状配置列表，每个形状包含:
        - type: 'circle', 'rectangle', 'triangle', 'ring', 'ellipse'
        - center: (cx, cy) 中心坐标
        - radius/width/height/等几何参数
        - term: 'piston', 'tilt', 'defocus', 'pyramid', 'spherical', 'coma'
        - amplitude: 相位幅度
        - 其他相位项特定参数
    
    Returns
    -------
    np.ndarray
        相位图 (rad)
    """
    phi = np.zeros_like(X, dtype=float)
    
    for shape in shapes:
        stype = str(shape.get('type', 'circle')).lower()
        center = shape.get('center', (0.0, 0.0))
        cx, cy = float(center[0]), float(center[1])
        amplitude = float(shape.get('amplitude', 1.0))
        term = str(shape.get('term', 'piston')).lower()
        
        # 创建形状掩模
        mask = np.zeros_like(X, dtype=bool)
        
        if stype == 'circle':
            radius = float(shape.get('radius', 10e-6))
            mask = create_circle_mask(X, Y, (cx, cy), radius)
            r_max = radius
        elif stype in ('rectangle', 'square'):
            width = float(shape.get('width', 20e-6))
            height = float(shape.get('height', width))
            mask = create_rectangle_mask(X, Y, (cx, cy), width, height)
            r_max = width / 2.0
        elif stype == 'triangle':
            radius = float(shape.get('radius', 20e-6))
            mask = create_triangle_mask(X, Y, (cx, cy), radius)
            r_max = radius
        elif stype == 'ring':
            r_out = float(shape.get('radius_outer', 20e-6))
            r_in = float(shape.get('radius_inner', 10e-6))
            mask = create_ring_mask(X, Y, (cx, cy), r_out, r_in)
            r_max = r_out
        elif stype == 'ellipse':
            semi_major = float(shape.get('semi_major', 20e-6))
            semi_minor = float(shape.get('semi_minor', 10e-6))
            angle = float(shape.get('angle_deg', 0.0))
            mask = create_ellipse_mask(X, Y, (cx, cy), semi_major, semi_minor, angle)
            r_max = semi_major
        elif stype == 'polygon':
            vertices = shape.get('vertices', [])
            if vertices:
                mask = create_polygon_mask(X, Y, vertices)
            r_max = 1.0  # 需要用户指定
        else:
            continue
        
        if not np.any(mask):
            continue
        
        # 应用相位项
        if term == 'piston':
            apply_piston(phi, mask, amplitude)
        elif term == 'tilt':
            tilt_x = float(shape.get('tilt_x', amplitude))
            tilt_y = float(shape.get('tilt_y', 0.0))
            apply_tilt(phi, mask, X, Y, (cx, cy), tilt_x, tilt_y)
        elif term == 'defocus':
            r_max_override = float(shape.get('radius', shape.get('radius_outer', shape.get('width', r_max * 2) / 2)))
            apply_defocus(phi, mask, X, Y, (cx, cy), amplitude, r_max_override)
        elif term == 'pyramid':
            r_max_override = float(shape.get('radius', shape.get('radius_outer', shape.get('width', r_max * 2) / 2)))
            apply_pyramid(phi, mask, X, Y, (cx, cy), amplitude, r_max_override)
        elif term == 'spherical':
            apply_spherical(phi, mask, X, Y, (cx, cy), amplitude, r_max)
        elif term == 'coma':
            angle = float(shape.get('coma_angle_deg', 0.0))
            apply_coma(phi, mask, X, Y, (cx, cy), amplitude, r_max, angle)
    
    return phi


def generate_simple_wfe(X: np.ndarray, Y: np.ndarray,
                        tilt_tx: float = 0.0, tilt_ty: float = 0.0,
                        defocus_alpha: float = 0.0,
                        astig_alpha: float = 0.0, astig_angle_deg: float = 0.0,
                        rmax: float = None) -> np.ndarray:
    """
    生成简单模式的WFE相位图
    
    Parameters
    ----------
    X, Y : np.ndarray
        坐标网格
    tilt_tx, tilt_ty : float
        倾斜系数 (rad/m)
    defocus_alpha : float
        离焦系数
    astig_alpha : float
        像散系数
    astig_angle_deg : float
        像散角度 (度)
    rmax : float, optional
        归一化半径，默认使用窗口半径
    
    Returns
    -------
    np.ndarray
        相位图 (rad)
    """
    phi = np.zeros_like(X, dtype=float)
    
    if rmax is None:
        rmax = max(np.abs(X).max(), np.abs(Y).max())
    
    # 倾斜
    if tilt_tx != 0 or tilt_ty != 0:
        phi += tilt_tx * X + tilt_ty * Y
    
    # 离焦
    if defocus_alpha != 0:
        phi += defocus_alpha * ((X**2 + Y**2) / max(1e-15, rmax**2))
    
    # 像散
    if astig_alpha != 0:
        rad = np.deg2rad(astig_angle_deg)
        Xr = X * np.cos(rad) + Y * np.sin(rad)
        Yr = -X * np.sin(rad) + Y * np.cos(rad)
        phi += astig_alpha * ((Xr**2 - Yr**2) / max(1e-15, rmax**2))
    
    return phi


def generate_zernike_wfe(X: np.ndarray, Y: np.ndarray,
                         zernike_coefs: List[Tuple[int, float]],
                         pupil_radius_m: float) -> np.ndarray:
    """
    生成Zernike多项式WFE相位图
    
    Parameters
    ----------
    X, Y : np.ndarray
        坐标网格
    zernike_coefs : List[Tuple[int, float]]
        Zernike系数列表 [(noll_index, amplitude), ...]
    pupil_radius_m : float
        光瞳半径 (m)
    
    Returns
    -------
    np.ndarray
        相位图 (rad)
    """
    phi = np.zeros_like(X, dtype=float)
    
    R = np.sqrt(X**2 + Y**2) / pupil_radius_m
    Theta = np.arctan2(Y, X)
    mask = R <= 1.0
    
    # Noll索引到(n, m, mode)的映射
    noll_map = {
        1: (0, 0, 'cos'),    # Piston
        2: (1, 1, 'cos'),    # Tilt X
        3: (1, 1, 'sin'),    # Tilt Y
        4: (2, 0, 'cos'),    # Defocus
        5: (2, 2, 'cos'),    # Astig 0
        6: (2, 2, 'sin'),    # Astig 45
        7: (3, 1, 'cos'),    # Coma X
        8: (3, 1, 'sin'),    # Coma Y
        9: (3, 3, 'cos'),    # Trefoil X
        10: (3, 3, 'sin'),   # Trefoil Y
        11: (4, 0, 'cos'),   # Spherical
        12: (4, 2, 'cos'),   # Secondary Astig 0
        13: (4, 2, 'sin'),   # Secondary Astig 45
        14: (4, 4, 'cos'),   # Quadfoil X
        15: (4, 4, 'sin'),   # Quadfoil Y
    }
    
    def _radial(n, m, rho):
        """计算径向多项式"""
        m_abs = abs(m)
        result = np.zeros_like(rho)
        for k in range((n - m_abs) // 2 + 1):
            num = ((-1) ** k) * np.math.factorial(n - k)
            den = (np.math.factorial(k) * 
                   np.math.factorial((n + m_abs) // 2 - k) * 
                   np.math.factorial((n - m_abs) // 2 - k))
            result += (num / den) * (rho ** (n - 2 * k))
        return result
    
    for j, a in zernike_coefs:
        if j not in noll_map:
            continue
        n, m, mode = noll_map[j]
        Rnm = _radial(n, m, R)
        if m == 0:
            Z = Rnm
        else:
            if mode == 'cos':
                Z = Rnm * np.cos(abs(m) * Theta)
            else:
                Z = Rnm * np.sin(abs(m) * Theta)
        phi += a * Z * mask
    
    return phi


def generate_psd_wfe(X: np.ndarray, Y: np.ndarray,
                     rms_rad: float, corr_len_m: float,
                     window_size_m: float = None) -> np.ndarray:
    """
    生成PSD噪声WFE相位图
    
    Parameters
    ----------
    X, Y : np.ndarray
        坐标网格
    rms_rad : float
        RMS值 (rad)
    corr_len_m : float
        相关长度 (m)
    window_size_m : float, optional
        窗口大小，默认从X推断
    
    Returns
    -------
    np.ndarray
        相位图 (rad)
    """
    if rms_rad <= 0 or corr_len_m <= 0:
        return np.zeros_like(X)
    
    ny, nx = X.shape
    if window_size_m is None:
        window_size_m = 2 * np.abs(X).max()
    
    fx = np.fft.fftfreq(nx, d=window_size_m / nx)
    fy = np.fft.fftfreq(ny, d=window_size_m / ny)
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    
    H = np.exp(-2.0 * (np.pi**2) * (FX**2 + FY**2) * (corr_len_m**2))
    noise = np.random.normal(0.0, 1.0, size=(ny, nx))
    F = np.fft.fft2(noise)
    Ff = F * H
    screen = np.real(np.fft.ifft2(Ff))
    screen -= np.mean(screen)
    std = np.std(screen)
    if std > 1e-15:
        screen = screen * (rms_rad / std)
    
    return screen


# =============================================================================
# 条纹/阶梯/正弦/同心环 相位生成
# =============================================================================

def generate_stripe_wfe(X: np.ndarray, Y: np.ndarray,
                        n_stripes: int = 10,
                        direction: str = 'horizontal',
                        phase_values: List[float] = None,
                        min_phase: float = 0.0,
                        max_phase: float = 10.0,
                        profile: str = 'step',
                        angle_deg: float = 0.0) -> np.ndarray:
    """
    生成条纹状WFE相位图（水波纹状，用于精度标定）
    
    Parameters
    ----------
    X, Y : np.ndarray
        坐标网格
    n_stripes : int
        条纹数量
    direction : str
        条纹方向: 'horizontal'(水平条纹), 'vertical'(垂直条纹), 'diagonal'(斜条纹)
    phase_values : List[float], optional
        每条条纹的相位值（rad），长度应为n_stripes
        如果为None，则自动从min_phase到max_phase线性分布
    min_phase, max_phase : float
        自动生成相位值时的范围 (rad)
    profile : str
        条纹内部轮廓: 'step'(阶跃), 'linear'(线性渐变), 'smooth'(平滑过渡)
    angle_deg : float
        条纹角度，仅用于diagonal方向
    
    Returns
    -------
    np.ndarray
        相位图 (rad)
    """
    phi = np.zeros_like(X, dtype=float)
    
    # 生成相位值列表
    if phase_values is None:
        phase_values = np.linspace(min_phase, max_phase, n_stripes)
    else:
        phase_values = np.array(phase_values)
    
    # 计算坐标轴
    if direction == 'horizontal':
        coord = Y
    elif direction == 'vertical':
        coord = X
    elif direction == 'diagonal':
        angle = np.deg2rad(angle_deg)
        coord = X * np.cos(angle) + Y * np.sin(angle)
    else:
        coord = Y  # 默认水平
    
    # 归一化坐标到[0, n_stripes]
    coord_min, coord_max = coord.min(), coord.max()
    coord_norm = (coord - coord_min) / (coord_max - coord_min) * n_stripes
    
    if profile == 'step':
        # 阶跃式条纹
        stripe_idx = np.clip(np.floor(coord_norm).astype(int), 0, n_stripes - 1)
        phi = phase_values[stripe_idx]
    elif profile == 'linear':
        # 线性渐变条纹（在条纹内部线性变化）
        stripe_idx = np.floor(coord_norm).astype(int)
        stripe_frac = coord_norm - stripe_idx
        stripe_idx = np.clip(stripe_idx, 0, n_stripes - 1)
        next_idx = np.clip(stripe_idx + 1, 0, n_stripes - 1)
        phi = phase_values[stripe_idx] * (1 - stripe_frac) + phase_values[next_idx] * stripe_frac
    elif profile == 'smooth':
        # 平滑过渡（正弦插值）
        stripe_idx = np.floor(coord_norm).astype(int)
        stripe_frac = coord_norm - stripe_idx
        stripe_idx = np.clip(stripe_idx, 0, n_stripes - 1)
        next_idx = np.clip(stripe_idx + 1, 0, n_stripes - 1)
        # 使用正弦平滑
        smooth_frac = (1 - np.cos(np.pi * stripe_frac)) / 2
        phi = phase_values[stripe_idx] * (1 - smooth_frac) + phase_values[next_idx] * smooth_frac
    
    return phi


def generate_step_wfe(X: np.ndarray, Y: np.ndarray,
                      n_steps: int = 5,
                      direction: str = 'horizontal',
                      step_values: List[float] = None,
                      min_phase: float = 0.0,
                      max_phase: float = 5.0) -> np.ndarray:
    """
    生成阶梯状WFE相位图
    
    Parameters
    ----------
    X, Y : np.ndarray
        坐标网格
    n_steps : int
        阶梯数量
    direction : str
        阶梯方向: 'horizontal', 'vertical'
    step_values : List[float], optional
        每级阶梯的相位值（rad）
    
    Returns
    -------
    np.ndarray
        相位图 (rad)
    """
    if step_values is None:
        step_values = np.linspace(min_phase, max_phase, n_steps)
    
    return generate_stripe_wfe(X, Y, n_stripes=n_steps, direction=direction,
                               phase_values=step_values, profile='step')


def generate_sinusoidal_wfe(X: np.ndarray, Y: np.ndarray,
                            amplitude: float = 5.0,
                            period_m: float = 2000e-6,
                            direction: str = 'horizontal',
                            phase_offset: float = 0.0,
                            n_periods: int = None,
                            window_size_m: float = None) -> np.ndarray:
    """
    生成正弦波WFE相位图
    
    Parameters
    ----------
    X, Y : np.ndarray
        坐标网格
    amplitude : float
        振幅 (rad)
    period_m : float
        周期 (m)
    direction : str
        波传播方向: 'horizontal', 'vertical', 'radial'
    phase_offset : float
        初相位 (rad)
    n_periods : int, optional
        周期数（如果指定，会覆盖period_m）
    window_size_m : float, optional
        窗口大小 (m)
    
    Returns
    -------
    np.ndarray
        相位图 (rad)
    """
    if window_size_m is None:
        window_size_m = 2 * np.abs(X).max()
    
    # 如果指定了周期数，计算周期
    if n_periods is not None and n_periods > 0:
        period_m = window_size_m / n_periods
    
    if direction == 'horizontal':
        phi = amplitude * np.sin(2 * np.pi * Y / period_m + phase_offset)
    elif direction == 'vertical':
        phi = amplitude * np.sin(2 * np.pi * X / period_m + phase_offset)
    elif direction == 'radial':
        R = np.sqrt(X**2 + Y**2)
        phi = amplitude * np.sin(2 * np.pi * R / period_m + phase_offset)
    else:
        phi = amplitude * np.sin(2 * np.pi * Y / period_m + phase_offset)
    
    return phi


def generate_concentric_wfe(X: np.ndarray, Y: np.ndarray,
                            n_rings: int = 5,
                            center: Tuple[float, float] = (0.0, 0.0),
                            max_radius_m: float = 6000e-6,
                            phase_values: List[float] = None,
                            min_phase: float = 0.0,
                            max_phase: float = 10.0,
                            profile: str = 'step') -> np.ndarray:
    """
    生成同心环WFE相位图
    
    Parameters
    ----------
    X, Y : np.ndarray
        坐标网格
    n_rings : int
        环数量
    center : Tuple[float, float]
        中心位置 (m)
    max_radius_m : float
        最大半径 (m)
    phase_values : List[float], optional
        每环相位值（rad）
    min_phase, max_phase : float
        自动生成相位值时的范围 (rad)
    profile : str
        'step'(阶跃) 或 'smooth'(平滑过渡)
    
    Returns
    -------
    np.ndarray
        相位图 (rad)
    """
    phi = np.zeros_like(X, dtype=float)
    
    # 生成相位值列表
    if phase_values is None:
        phase_values = np.linspace(min_phase, max_phase, n_rings)
    else:
        phase_values = np.array(phase_values)
    
    cx, cy = center
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    # 归一化半径到[0, n_rings]
    R_norm = R / max_radius_m * n_rings
    R_norm = np.clip(R_norm, 0, n_rings - 1e-10)
    
    if profile == 'step':
        ring_idx = np.floor(R_norm).astype(int)
        ring_idx = np.clip(ring_idx, 0, n_rings - 1)
        phi = phase_values[ring_idx]
    elif profile == 'smooth':
        ring_idx = np.floor(R_norm).astype(int)
        ring_frac = R_norm - ring_idx
        ring_idx = np.clip(ring_idx, 0, n_rings - 1)
        next_idx = np.clip(ring_idx + 1, 0, n_rings - 1)
        smooth_frac = (1 - np.cos(np.pi * ring_frac)) / 2
        phi = phase_values[ring_idx] * (1 - smooth_frac) + phase_values[next_idx] * smooth_frac
    
    return phi


def generate_wfe_phase(X: np.ndarray, Y: np.ndarray, 
                       config: WFEConfig,
                       window_size_m: float = None) -> np.ndarray:
    """
    根据配置生成完整的WFE相位图
    
    Parameters
    ----------
    X, Y : np.ndarray
        坐标网格
    config : WFEConfig
        WFE配置
    window_size_m : float, optional
        窗口大小 (m)
    
    Returns
    -------
    np.ndarray
        相位图 (rad)
    """
    phi = np.zeros_like(X, dtype=float)
    
    if window_size_m is None:
        window_size_m = 2 * np.abs(X).max()
    rmax = window_size_m / 2
    
    if config.mode == 'simple':
        phi += generate_simple_wfe(
            X, Y,
            tilt_tx=config.tilt_tx,
            tilt_ty=config.tilt_ty,
            defocus_alpha=config.defocus_alpha,
            astig_alpha=config.astig_alpha,
            astig_angle_deg=config.astig_angle_deg,
            rmax=rmax
        )
    elif config.mode == 'advanced':
        if config.zernike_coefs:
            phi += generate_zernike_wfe(X, Y, config.zernike_coefs, config.pupil_radius_m)
        if config.psd_rms_rad > 0 and config.psd_corr_len_m > 0:
            phi += generate_psd_wfe(X, Y, config.psd_rms_rad, config.psd_corr_len_m, window_size_m)
    elif config.mode == 'geometric':
        phi += generate_geometric_wfe(X, Y, config.shapes)
    elif config.mode == 'stripe':
        params = config.stripe_params
        phi += generate_stripe_wfe(
            X, Y,
            n_stripes=params.get('n_stripes', 10),
            direction=params.get('direction', 'horizontal'),
            phase_values=params.get('phase_values'),
            min_phase=params.get('min_phase', 0.0),
            max_phase=params.get('max_phase', 10.0),
            profile=params.get('profile', 'step'),
            angle_deg=params.get('angle_deg', 0.0)
        )
    elif config.mode == 'step':
        params = config.step_params
        phi += generate_step_wfe(
            X, Y,
            n_steps=params.get('n_steps', 5),
            direction=params.get('direction', 'horizontal'),
            step_values=params.get('step_values'),
            min_phase=params.get('min_phase', 0.0),
            max_phase=params.get('max_phase', 5.0)
        )
    elif config.mode == 'sinusoidal':
        params = config.sinusoidal_params
        phi += generate_sinusoidal_wfe(
            X, Y,
            amplitude=params.get('amplitude', 5.0),
            period_m=params.get('period_m', 2000e-6),
            direction=params.get('direction', 'horizontal'),
            phase_offset=params.get('phase_offset', 0.0),
            n_periods=params.get('n_periods'),
            window_size_m=window_size_m
        )
    elif config.mode == 'concentric':
        params = config.concentric_params
        phi += generate_concentric_wfe(
            X, Y,
            n_rings=params.get('n_rings', 5),
            center=params.get('center', (0.0, 0.0)),
            max_radius_m=params.get('max_radius_m', 6000e-6),
            phase_values=params.get('phase_values'),
            min_phase=params.get('min_phase', 0.0),
            max_phase=params.get('max_phase', 10.0),
            profile=params.get('profile', 'step')
        )
    
    # 全局倾斜
    if config.global_tilt_tx != 0 or config.global_tilt_ty != 0:
        phi += config.global_tilt_tx * X + config.global_tilt_ty * Y
    
    return phi


# =============================================================================
# 便捷预设
# =============================================================================

def preset_dpc_verification() -> WFEConfig:
    """
    DPC验证用预设：包含三角形、圆形、矩形等明显特征
    """
    return WFEConfig.geometric([
        {
            'type': 'triangle',
            'center': (-2000e-6, 2000e-6),
            'radius': 1500e-6,
            'term': 'pyramid',
            'amplitude': 20.0,
        },
        {
            'type': 'circle',
            'center': (2000e-6, 2000e-6),
            'radius': 1500e-6,
            'term': 'defocus',
            'amplitude': 15.0,
        },
        {
            'type': 'rectangle',
            'center': (0.0, -2000e-6),
            'width': 4000e-6,
            'height': 1000e-6,
            'term': 'tilt',
            'tilt_x': 5000.0,
            'tilt_y': 0.0,
        },
    ])


def preset_ring_pattern() -> WFEConfig:
    """
    圆环图案预设
    """
    return WFEConfig.geometric([
        {
            'type': 'ring',
            'center': (0.0, 0.0),
            'radius_outer': 3000e-6,
            'radius_inner': 1500e-6,
            'term': 'defocus',
            'amplitude': 10.0,
        },
    ])


def preset_checkerboard(n_cells: int = 4, cell_size_m: float = 2000e-6) -> WFEConfig:
    """
    棋盘图案预设
    """
    shapes = []
    half_n = n_cells // 2
    for i in range(-half_n, half_n):
        for j in range(-half_n, half_n):
            if (i + j) % 2 == 0:
                shapes.append({
                    'type': 'rectangle',
                    'center': ((i + 0.5) * cell_size_m, (j + 0.5) * cell_size_m),
                    'width': cell_size_m * 0.9,
                    'height': cell_size_m * 0.9,
                    'term': 'piston',
                    'amplitude': 5.0,
                })
    return WFEConfig.geometric(shapes)


def preset_lens_array(n_lenses: int = 3, lens_radius_m: float = 1500e-6, 
                      spacing_m: float = 4000e-6) -> WFEConfig:
    """
    透镜阵列预设（多个离焦圆形）
    """
    shapes = []
    half_n = n_lenses // 2
    for i in range(-half_n, half_n + 1):
        for j in range(-half_n, half_n + 1):
            shapes.append({
                'type': 'circle',
                'center': (i * spacing_m, j * spacing_m),
                'radius': lens_radius_m,
                'term': 'defocus',
                'amplitude': 15.0,
            })
    return WFEConfig.geometric(shapes)


# =============================================================================
# 条纹/阶梯/正弦/同心环 预设（用于精度标定）
# =============================================================================

def preset_stripe_calibration(n_stripes: int = 10,
                               min_phase: float = 0.0,
                               max_phase: float = 10.0,
                               direction: str = 'horizontal',
                               profile: str = 'step') -> WFEConfig:
    """
    条纹标定预设 - 用于测量系统对不同相位误差的识别精度
    
    每条条纹具有不同的相位值（从min_phase到max_phase线性分布），
    可以用来判断系统能识别多大的波前误差。
    
    Parameters
    ----------
    n_stripes : int
        条纹数量，越多则相位梯度越密
    min_phase, max_phase : float
        相位范围 (rad)
    direction : str
        'horizontal' 水平条纹 | 'vertical' 垂直条纹
    profile : str
        'step' 阶跃 | 'linear' 线性渐变 | 'smooth' 平滑过渡
    
    Examples
    --------
    # 10条水平条纹，相位从0到10rad
    config = preset_stripe_calibration(n_stripes=10, max_phase=10.0)
    
    # 查看每条条纹的相位值
    # 条纹0: 0 rad, 条纹1: 1.11 rad, ..., 条纹9: 10 rad
    """
    return WFEConfig.stripe(
        n_stripes=n_stripes,
        direction=direction,
        min_phase=min_phase,
        max_phase=max_phase,
        profile=profile
    )


def preset_concentric_calibration(n_rings: int = 10,
                                   min_phase: float = 0.0,
                                   max_phase: float = 10.0,
                                   max_radius_m: float = 6000e-6,
                                   profile: str = 'step') -> WFEConfig:
    """
    同心环标定预设 - 用于测量径向误差识别精度
    
    每个环具有不同的相位值，从中心向外逐渐增加。
    
    Parameters
    ----------
    n_rings : int
        环数量
    min_phase, max_phase : float
        相位范围 (rad)
    max_radius_m : float
        最大半径 (m)
    profile : str
        'step' 阶跃 | 'smooth' 平滑过渡
    """
    return WFEConfig.concentric(
        n_rings=n_rings,
        min_phase=min_phase,
        max_phase=max_phase,
        max_radius_m=max_radius_m,
        profile=profile
    )


def preset_sinusoidal_test(amplitude: float = 5.0,
                           n_periods: int = 5,
                           direction: str = 'horizontal') -> WFEConfig:
    """
    正弦波测试预设 - 用于测试周期性误差响应
    
    Parameters
    ----------
    amplitude : float
        振幅 (rad)
    n_periods : int
        窗口内的周期数
    direction : str
        'horizontal' | 'vertical' | 'radial'
    """
    return WFEConfig.sinusoidal(
        amplitude=amplitude,
        n_periods=n_periods,
        direction=direction
    )


def preset_step_gradient(n_steps: int = 5,
                         min_phase: float = 0.0,
                         max_phase: float = 5.0,
                         direction: str = 'horizontal') -> WFEConfig:
    """
    阶梯渐变预设 - 用于测试离散相位阶梯响应
    
    Parameters
    ----------
    n_steps : int
        阶梯数量
    min_phase, max_phase : float
        相位范围 (rad)
    direction : str
        'horizontal' | 'vertical'
    """
    return WFEConfig.step(
        n_steps=n_steps,
        min_phase=min_phase,
        max_phase=max_phase,
        direction=direction
    )


def preset_custom_stripes(phase_values: List[float],
                          direction: str = 'horizontal',
                          profile: str = 'step') -> WFEConfig:
    """
    自定义条纹预设 - 指定每条条纹的精确相位值
    
    Parameters
    ----------
    phase_values : List[float]
        每条条纹的相位值列表 (rad)
        例如: [0.1, 0.5, 1.0, 2.0, 5.0] 生成5条具有这些相位的条纹
    direction : str
        'horizontal' | 'vertical'
    profile : str
        'step' | 'linear' | 'smooth'
    
    Examples
    --------
    # 创建具有特定相位值的条纹，用于精确标定
    config = preset_custom_stripes([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    """
    return WFEConfig.stripe(
        n_stripes=len(phase_values),
        direction=direction,
        phase_values=phase_values,
        profile=profile
    )


# =============================================================================
# X射线系统优化预设 (针对短波长、小位移特性)
# =============================================================================

def preset_xray_enhanced_geometric(wavelength_m: float = 1.0e-10,
                                   z_prop_m: float = 0.8,
                                   target_displacement_px: float = 0.5,
                                   pixel_size_m: float = 6.5e-6) -> WFEConfig:
    """
    X射线增强几何WFE - 针对短波长优化的大幅度倾斜
    
    针对X射线波长极短（~0.1 nm）的特点，使用大幅度相位梯度以产生可检测的位移。
    
    Parameters
    ----------
    wavelength_m : float
        波长 (m)，默认 0.1 nm (12 keV)
    z_prop_m : float
        传播距离 (m)
    target_displacement_px : float
        目标位移 (像素)
    pixel_size_m : float
        探测器像素尺寸 (m)
    
    Returns
    -------
    WFEConfig
        配置了大幅度倾斜的几何WFE
    
    Notes
    -----
    位移公式: Δx = z × λ/(2π) × ∂φ/∂x
    对于 λ=0.1nm, z=0.8m, 要产生 0.5px (3.25μm) 位移:
    需要相位梯度 ∂φ/∂x ≈ 2.5×10^5 rad/m
    """
    # 计算所需相位梯度
    target_displacement_m = target_displacement_px * pixel_size_m
    target_angle_rad = target_displacement_m / z_prop_m
    target_gradient_rad_per_m = target_angle_rad * (2.0 * np.pi) / wavelength_m
    
    # 使用目标梯度的不同倍数创建多个几何形状
    tilt_triangle = target_gradient_rad_per_m * 0.8   # 三角形：80%强度
    tilt_circle = target_gradient_rad_per_m * 0.6     # 圆形：60%强度
    tilt_rect = target_gradient_rad_per_m * 1.0       # 矩形：100%强度
    tilt_global = target_gradient_rad_per_m * 0.4     # 全局：40%强度
    
    return WFEConfig.geometric(shapes=[
        {
            'type': 'triangle',
            'center': (-2000e-6, 2000e-6),
            'radius': 1500e-6,
            'term': 'tilt',
            'tilt_x': tilt_triangle,
            'tilt_y': 0.0,
        },
        {
            'type': 'circle',
            'center': (2000e-6, 2000e-6),
            'radius': 1500e-6,
            'term': 'tilt',
            'tilt_x': -tilt_circle,  # 反向
            'tilt_y': 0.0,
        },
        {
            'type': 'rectangle',
            'center': (0.0, -2000e-6),
            'width': 4000e-6,
            'height': 1000e-6,
            'term': 'tilt',
            'tilt_x': 0.0,
            'tilt_y': tilt_rect,
        },
    ]).with_global_tilt(tx=tilt_global, ty=0.0)


def preset_xray_default() -> WFEConfig:
    """
    X射线默认WFE预设 - 针对12keV X射线优化
    
    预设参数:
    - 波长: 0.1033 nm (12 keV)
    - 传播距离: 800 mm
    - 目标位移: 0.5 像素
    - 像素尺寸: 6.5 μm
    
    产生的位移约 0.2-0.5 像素，适合亚像素互相关检测
    """
    return preset_xray_enhanced_geometric(
        wavelength_m=1.2398e-6 / 12000.0,  # 12 keV
        z_prop_m=0.8,
        target_displacement_px=0.5,
        pixel_size_m=6.5e-6
    )


def preset_xray_strong() -> WFEConfig:
    """
    X射线强WFE预设 - 产生较大位移用于测试
    
    目标位移: 1.0 像素
    适用于验证算法，不适合真实光学系统
    """
    return preset_xray_enhanced_geometric(
        wavelength_m=1.2398e-6 / 12000.0,
        z_prop_m=0.8,
        target_displacement_px=1.0,
        pixel_size_m=6.5e-6
    )


def preset_xray_weak() -> WFEConfig:
    """
    X射线弱WFE预设 - 产生微小位移用于精度测试
    
    目标位移: 0.2 像素
    适用于测试亚像素精度极限
    """
    return preset_xray_enhanced_geometric(
        wavelength_m=1.2398e-6 / 12000.0,
        z_prop_m=0.8,
        target_displacement_px=0.2,
        pixel_size_m=6.5e-6
    )


def preset_xray_tilt_only(tilt_x_rad_per_m: float = 100000.0,
                          tilt_y_rad_per_m: float = 0.0) -> WFEConfig:
    """
    X射线纯倾斜WFE - 用于测试全局倾斜响应
    
    Parameters
    ----------
    tilt_x_rad_per_m : float
        X方向相位梯度 (rad/m)
        默认 100k rad/m → 约 0.2 像素位移
    tilt_y_rad_per_m : float
        Y方向相位梯度 (rad/m)
    
    Examples
    --------
    # 纯X倾斜，产生约0.4像素位移
    config = preset_xray_tilt_only(tilt_x_rad_per_m=200000.0)
    
    # X-Y联合倾斜
    config = preset_xray_tilt_only(tilt_x_rad_per_m=150000.0, 
                                    tilt_y_rad_per_m=150000.0)
    """
    return WFEConfig.simple(
        tilt_tx=tilt_x_rad_per_m,
        tilt_ty=tilt_y_rad_per_m,
        defocus_alpha=0.0,
        astig_alpha=0.0
    )


def preset_xray_custom_shapes(shapes_config: List[Dict[str, Any]],
                               global_tilt_x: float = 0.0,
                               global_tilt_y: float = 0.0) -> WFEConfig:
    """
    X射线自定义几何形状WFE
    
    Parameters
    ----------
    shapes_config : List[Dict]
        形状配置列表，每个字典包含:
        - type: 'triangle'|'circle'|'rectangle'|'ring'
        - center: (x, y) 中心位置 (m)
        - radius/width/height: 尺寸 (m)
        - term: 'tilt'|'piston'|'defocus'|'pyramid'
        - tilt_x, tilt_y: 倾斜参数 (rad/m)
        - amplitude: 振幅 (rad)
    global_tilt_x, global_tilt_y : float
        全局倾斜 (rad/m)
    
    Examples
    --------
    # 创建自定义三角形和圆形
    shapes = [
        {'type': 'triangle', 'center': (0, 2000e-6), 'radius': 1000e-6,
         'term': 'tilt', 'tilt_x': 200000.0},
        {'type': 'circle', 'center': (0, -2000e-6), 'radius': 1500e-6,
         'term': 'piston', 'amplitude': 50.0}
    ]
    config = preset_xray_custom_shapes(shapes, global_tilt_x=100000.0)
    """
    return WFEConfig.geometric(shapes=shapes_config).with_global_tilt(
        tx=global_tilt_x, ty=global_tilt_y
    )
