from __future__ import annotations
# config_simulate_laser.py — Step 1 (Laser): 4f 中继仿真配置
#
# 控制 simulate_laser.py 的所有参数

from typing import Optional, Tuple

# =============================================================================
# 1. 输出控制
# =============================================================================
OUTPUT_DIR: str = 'result/laser'
SHOW_PROGRESS: bool = True
SAVE_EACH_FRAME: bool = True

# =============================================================================
# 2. 脉冲与运动
# =============================================================================
N_PULSES: int = 10
# translation 模式：运动方向角（度）
ANGLE_DEG: float = 19.0

# translation 模式：每个 pulse 相对前一 pulse 的平移步长（米）
STEP_M: float = 50e-6
# 运动模式：当前仅实现 "translation"
MOVEMENT_MODE: str = 'translation'

# =============================================================================
# 3. 光学设置
# =============================================================================
# 激光波长（米）
WAVELENGTH_M: float = 527e-9
SOURCE_TYPE: str = 'plane'              # 'plane' | 'gaussian' | 'tophat'
# 入射光束有效半径（米）；用于有限照明范围
SOURCE_RADIUS_M: float = 2.33e-3

# 传播计算窗口（米）。4f relay 全链路都在该窗口内传播。
# 一般应满足：SIM_WINDOW_M >= |M| * MASK_WINDOW_M * 安全系数
SIM_WINDOW_M: float = 18e-3
GRID_POINTS: int = 4096

# --- 4f relay 参数 ---
# 掩膜后先自由传播 z0，再把 z0 平面 relay 到探测器
RELAY_Z0_M: float = 5e-3
# 第一片透镜焦距（米）
RELAY_F1_M: float = 50e-3
# 第二片透镜焦距（米）；横向放大率 M = -f2 / f1
RELAY_F2_M: float = 100e-3
# 透镜通光孔径直径（米）；None = 不截光
RELAY_LENS_APERTURE_M: Optional[float] = 25.4e-3
# 两片透镜各自的额外 Zernike 像差系数（弧度），从 Noll Z4 开始
# 例如 [defocus, astig45, astig0, comaY, comaX, ...]
RELAY_LENS1_ABERRATION_COEFFS = None
RELAY_LENS2_ABERRATION_COEFFS = None

# =============================================================================
# 4. 掩膜配置
# =============================================================================
# Hartmann 板的有限物理尺寸（米）
MASK_WINDOW_M: float = 3e-3
# 子孔径节距（米）
HARTMANN_PITCH_M: float = 250e-6
# 子孔径直径（米）
HARTMANN_HOLE_DIAMETER_M: float = 100e-6

# =============================================================================
# 5. 探测器
# =============================================================================
# 探测器像元尺寸（米）
DETECTOR_PIXEL_M: float = 6.5e-6
DETECTOR_NX: int = 2048
DETECTOR_NY: int = 2048
# 探测器采样模式：'nearest' | 'bilinear' | 'srw'
DETECTOR_SAMPLING_MODE: str = 'srw'
# 内部重采样倍率；仅对部分采样模式生效
DETECTOR_UPSAMPLE: int = 2
DETECTOR_BIN_SIZE: Optional[int] = None
DETECTOR_INTERPOLATION: str = 'bilinear'

# =============================================================================
# 6. 相位像差 / WFE
# =============================================================================
PHASE_ABERRATION_MODE: str = 'per_pulse'  # 'none' | 'static' | 'per_pulse'
# 相位板到 Hartmann 掩膜的距离（米）。
# 当前约定：随机/静态相位屏定义在“相位板平面”，随后用角谱法传播到掩膜平面。
# 若设为 0，则退化为原先“直接加载到 Hartmann 平面”的做法。
PHASE_PLATE_TO_MASK_M: float = 100e-3
# 随机 Zernike 最高 Noll 阶数
PHASE_ABERRATION_MAX_ORDER: int = 15
# 每个 Zernike 系数的随机范围（弧度）
# ±1.0: PV~6 rad, 位移~10px，超出搜索范围
# ±0.7: PV~4 rad, 位移~7px，在搜索半径内且SNR充足
# ±0.3 太小，位移<0.2 px，信噪比不足
PHASE_ABERRATION_RANGE: Tuple[float, float] = (-0.7, 0.7)
# 像差 pupil 半径（米）；None = MASK_WINDOW_M / 2 (1.5mm, 仅覆盖78%光斑)
# 设为 2.7mm → 直径5.4mm，超过掩膜对角线4.2mm，覆盖全部144个子孔径
PHASE_ABERRATION_PUPIL_RADIUS_M: Optional[float] = 2.7e-3
