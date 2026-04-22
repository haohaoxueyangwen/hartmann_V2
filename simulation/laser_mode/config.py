from __future__ import annotations
"""
激光 4f 中继模式 — 配置定义。

分拆为多个 dataclass，仿照现有 hartmann_simulator.py 的风格。
"""
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class OpticalConfig:
    """4f 中继光学配置"""
    wavelength_m: float = 527e-9
    sim_window_m: float = 18e-3           # 传播计算窗口
    grid_points: int = 4096
    # 4f 中继
    relay_z0_m: float = 2e-3             # 掩膜到物平面
    relay_f1_m: float = 50e-3            # L1 焦距
    relay_f2_m: float = 200e-3           # L2 焦距
    relay_lens_aperture_m: float = 25.4e-3
    relay_fourier_stop_enable: bool = False
    relay_fourier_stop_diameter_m: Optional[float] = None
    relay_fourier_stop_x_m: float = 0.0
    relay_fourier_stop_y_m: float = 0.0
    relay_lens1_aberration_coeffs: Optional[List[float]] = None  # Noll Z4 起, 弧度
    relay_lens2_aberration_coeffs: Optional[List[float]] = None
    # 光源
    source_type: str = 'plane'           # 'plane' | 'gaussian' | 'tophat'
    source_radius_m: float = 2.33e-3     # 照射半径（独立于 mask_window）

    @property
    def magnification_signed(self) -> float:
        """横向放大率（负 = 倒像）"""
        return -self.relay_f2_m / self.relay_f1_m

    @property
    def M_abs(self) -> float:
        return abs(self.magnification_signed)

    @property
    def z_eff_m(self) -> float:
        """等效杠杆臂 = |M| × z₀"""
        return self.M_abs * self.relay_z0_m

    @property
    def dx(self) -> float:
        return self.sim_window_m / self.grid_points


@dataclass
class MaskConfig:
    """Hartmann 掩膜配置"""
    mask_window_m: float = 3e-3          # 板物理尺寸（有限）
    pitch_m: float = 200e-6
    hole_diameter_m: float = 100e-6


@dataclass
class DetectorConfig:
    """探测器配置"""
    pixel_size: float = 6.5e-6
    nx: int = 2048
    ny: int = 2048
    sampling_mode: str = 'srw'           # 'nearest' | 'bilinear' | 'srw'
    upsample_factor: int = 2
    bin_size: Optional[int] = None
    interpolation: str = 'bilinear'


@dataclass
class AberrationConfig:
    """像差配置"""
    mode: str = 'per_pulse'              # 'none' | 'per_pulse' | 'static'
    phase_plate_to_mask_m: float = 0.0   # 相位板到 Hartmann 掩膜距离
    max_order: int = 15
    amplitude_range: Tuple[float, float] = (-1.0, 1.0)
    pupil_radius_m: Optional[float] = None  # None = mask_window/2


@dataclass
class MovementConfig:
    """运动配置"""
    mode: str = 'translation'
    n_pulses: int = 10
    step_m: float = 60e-6
    angle_deg: float = 19.0


@dataclass
class OutputConfig:
    """输出配置"""
    show_progress: bool = True
    save_each_frame: bool = True


def validate_configs(optical: OpticalConfig,
                     mask: MaskConfig,
                     detector: DetectorConfig) -> None:
    """
    启动前一致性检查。不通过则 raise ValueError。

    检查项:
    1. sim_window 能容纳放大像
    2. 探测器 FOV 能看到整个掩膜
    3. 透镜相位采样充分
    4. 单孔分辨率足够
    5. 子孔径数量 (警告)
    6. 照射范围覆盖 (警告)
    7. 频谱面光阑参数自洽
    """
    M = optical.M_abs
    dx = optical.dx

    # 1. sim_window 容纳放大像
    required = M * mask.mask_window_m * 1.2
    if optical.sim_window_m < required:
        raise ValueError(
            f"sim_window {optical.sim_window_m*1e3:.1f}mm < "
            f"M×mask×1.2 = {required*1e3:.1f}mm")

    # 2. 探测器 FOV 覆盖掩膜（横纵轴分别检查）
    det_fov_x = detector.pixel_size * detector.nx
    det_fov_y = detector.pixel_size * detector.ny
    obj_fov_x = det_fov_x / M
    obj_fov_y = det_fov_y / M
    if obj_fov_x < mask.mask_window_m:
        raise ValueError(
            f"X 方向物方可见范围 {obj_fov_x*1e3:.2f}mm < mask_window {mask.mask_window_m*1e3:.1f}mm. "
            f"增大探测器 nx 或减小 mask_window")
    if obj_fov_y < mask.mask_window_m:
        raise ValueError(
            f"Y 方向物方可见范围 {obj_fov_y*1e3:.2f}mm < mask_window {mask.mask_window_m*1e3:.1f}mm. "
            f"增大探测器 ny 或减小 mask_window")

    # 3. 透镜相位采样 (基于 beam NA，不是 λ/2)
    dx_lens_limit = optical.wavelength_m * optical.relay_f1_m / mask.mask_window_m
    if dx > dx_lens_limit:
        raise ValueError(
            f"dx={dx*1e6:.2f}µm > 透镜相位采样极限 {dx_lens_limit*1e6:.2f}µm. "
            f"增大 grid_points 或减小 sim_window")

    # 4. 孔分辨率
    px_per_hole = mask.hole_diameter_m / dx
    if px_per_hole < 4:
        raise ValueError(
            f"每孔仅 {px_per_hole:.1f} px，需 ≥4")

    # 5. 子孔径数量
    n_spots = int(mask.mask_window_m / mask.pitch_m)
    if n_spots < 8:
        warnings.warn(
            f"仅 {n_spots} spots/axis，Zernike 重建可能不足")

    # 6. 照射范围
    min_radius = mask.mask_window_m / (2.0 ** 0.5)
    if optical.source_radius_m < min_radius:
        warnings.warn(
            f"source_radius {optical.source_radius_m*1e3:.2f}mm < "
            f"mask 对角线半径 {min_radius*1e3:.2f}mm，板四角可能照不到")

    # 7. 频谱面光阑
    if optical.relay_fourier_stop_enable:
        stop_d = optical.relay_fourier_stop_diameter_m
        if stop_d is None or float(stop_d) <= 0.0:
            raise ValueError(
                "启用频谱面光阑时，relay_fourier_stop_diameter_m 必须为正数")
