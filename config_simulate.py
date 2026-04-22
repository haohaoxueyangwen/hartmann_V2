from __future__ import annotations
# config_simulate.py — Step 1: 光传输模拟配置
#
# 控制 simulate.py 的所有参数

from typing import Tuple, Optional

# =============================================================================
# 1. 输出控制
# =============================================================================
OUTPUT_DIR: str = 'result'
SAVE_EACH_FRAME: bool = True
SAVE_ACCUMULATED_PNG: bool = True

# =============================================================================
# 2. 脉冲与运动
# =============================================================================
N_PULSES: int = 10
# translation 模式：运动方向角，同时也是网格倾角（使光斑沿运动方向不重叠）
# rotation 模式下不使用此参数，改用 ROTATION_GRID_TILT_DEG
ANGLE_DEG: float = 19.0
# 当前默认 baseline 采用 sweep 推荐工作点：
# step = 8 um, z = 450 mm, WFE = 1 rad, Hartmann pitch = 25 um
STEP_M: float = 8e-6             # translation 模式每脉冲步进位移
EXPOSURE_TIME_S: float = 0.01

# 运动模式: "none" | "translation" | "trajectory" | "rotation"
MOVEMENT_MODE: str = "translation"
# 初始偏移 (dx, dy)，单位米，在所有模式下生效
MOVEMENT_INITIAL_OFFSET_XY_M: Tuple[float, float] = (0.0, 0.0)
# 逐步间隔参数 [s01, s12, ...]，长度必须等于 N_PULSES-1
# simulate.py 会结合 ANGLE_DEG、总位移和微小角度扰动自动生成每个 pulse 的绝对位移轨迹
MOVEMENT_STEP_INTERVALS = None
# "relative": 将 MOVEMENT_STEP_INTERVALS 视为比例/权重，并缩放到总位移
# "absolute": 将 MOVEMENT_STEP_INTERVALS 视为每一步的实际位移（米）
MOVEMENT_STEP_INTERVALS_MODE: str = "relative"
# trajectory + step intervals 模式下使用：所有步长之和（米）
MOVEMENT_TOTAL_DISPLACEMENT_M = None
# 每一步相对于 ANGLE_DEG 的随机角度扰动标准差（度）
MOVEMENT_ANGLE_JITTER_STD_DEG: float = 0.0
# 随机角度扰动的随机种子；None 表示每次运行随机
MOVEMENT_ANGLE_JITTER_SEED = None

# === 旋转运动参数 (MOVEMENT_MODE="rotation" 时使用) ===
# 旋转中心到光束轴的距离（米），即圆盘半径
ROTATION_RADIUS_M: float = 50e-3            # 50 mm
# 旋转角速度（rad/s）; None = 由 STEP_M / ROTATION_RADIUS_M 自动推算
ROTATION_ANGULAR_VELOCITY: Optional[float] = None
# rotation 模式：掩膜网格 X 轴与旋转切线方向的夹角（度）
# 物理含义类似 translation 模式的 ANGLE_DEG，但两者独立配置
ROTATION_GRID_TILT_DEG: float = 19.0
# 旋转方向: 1 = 逆时针, -1 = 顺时针
ROTATION_DIRECTION: int = 1

# =============================================================================
# 3. 光学设置
# =============================================================================
PHOTON_ENERGY_EV: float = 10_000
WINDOW_SIZE_M: float = 1000e-6
GRID_POINTS: int = 4096
Z_PROP_M: float = 450e-3

# 传播方法: 'fresnel' or 'anglar_spectrum'
PROPAGATION_METHOD: str = 'fresnel'

# 光源类型: 'plane', 'gaussian', 'tophat'
SOURCE_TYPE: str = 'plane'

# =============================================================================
# 4. 掩模配置
# =============================================================================
# 掩模类型: 'hartmann' (小孔阵列) or 'microlens' (微透镜阵列)
MASK_TYPE: str = 'hartmann'

# --- Hartmann 掩模参数 ---
HARTMANN_PITCH_M: float = 25e-6
HARTMANN_HOLE_DIAMETER_M: float = 13e-6

# --- 微透镜阵列参数 ---
MICROLENS_PITCH_M: float = 150e-6
MICROLENS_FOCAL_LENGTH_M: float = 10e-3
MICROLENS_FILL_FACTOR: float = 1.0
MICROLENS_SHAPE: str = 'square'  # 'circle' or 'square'



# =============================================================================
# 5. 探测器
# =============================================================================
DETECTOR_PIXEL_M: float = 0.65e-6
DETECTOR_NX: int = 1024
DETECTOR_NY: int = 1024

# 探测器效应管线
ENABLE_DETECTOR_EFFECTS: bool = True
DETECTOR_PSF_SIGMA: float = 0.0       # 像素单位, 0 = 禁用
DETECTOR_UPSAMPLE: int = 1
DETECTOR_INTERPOLATION: str = 'bilinear'

# =============================================================================
# 6. 噪声模型
# =============================================================================
FLUX_DENSITY: float = 1e10            # photons/s/pixel
READOUT_NOISE_ELECTRONS: float = 0.5
DARK_CURRENT: float = 0.0

# ADC
ADC_BITS: int = 16
ADC_SATURATION: float = 65535.0

# =============================================================================
# 7. 相位像差 / WFE
# =============================================================================
PHASE_ABERRATION_MODE: str = 'per_pulse'  # 'static' or 'per_pulse'， 
PHASE_ABERRATION_MAX_ORDER: int = 15
PHASE_ABERRATION_RANGE: Tuple[float, float] = (-1.0, 1.0)
PRINT_ABERRATION_INFO: bool = True

# WFE 预设: '' = 随机 Zernike; 设为预设名称以使用
WFE_PRESET: str = ''

# =============================================================================
# 8. 简化高斯光斑仿真 (--simple 模式)
# =============================================================================
SIMPLE_SPOT_IMAGE_SHAPE: Tuple[int, int] = (1024, 1024)
SIMPLE_SPOT_SPACING: int = 40
SIMPLE_SPOT_SIZE: float = 5.0
SIMPLE_SPOT_MARGIN: int = 20
SIMPLE_SPOT_SHIFT_MAGNITUDE: float = 10.0  # 原始代码默认值: 10
SIMPLE_SPOT_SHIFT_ANGLE_DEG: float = 20.0
SIMPLE_SPOT_MAX_SHIFT: float = 5.0
SIMPLE_SPOT_WAVEFRONT_RMS: float = 1.5
SIMPLE_SPOT_NOISE_LEVEL: float = 0.0  # 不加噪声
