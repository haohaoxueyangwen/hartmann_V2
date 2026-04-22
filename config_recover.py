from __future__ import annotations
# config_recover.py — Step 2: 相位恢复配置
#
# 控制 recover.py 的所有参数
# 每个参数对应的算法原理见 step2_algorithms.md

from typing import Optional

# =============================================================================
# 1. 位移检测方法
# =============================================================================
# 方法: 'centroid', 'cross_correlation'
DISPLACEMENT_METHOD: str = 'cross_correlation'

# 运行所有方法进行对比
RUN_ALL_DISPLACEMENT_METHODS: bool = True

# =============================================================================
# 2. 光斑检测参数 (detect_beamlets)
# =============================================================================
# 检测前高斯平滑 sigma (像素单位)
DETECT_SMOOTH_SIGMA: float = 1.0

# 阈值因子: threshold = mean + factor × std
DETECT_THRESHOLD_FACTOR: float = 0.3

# =============================================================================
# 3. 互相关参数 (cross_correlation_shifts)
# 默认使用 guided local ZNCC：
#   1. patch 负责保留 spot 形状
#   2. search_radius 限制错配到邻近 spot
#   3. 单帧 ideal-vs-target 路径不再使用经验尺度补偿
XCORR_USE_GUIDED: bool = True
XCORR_CORRECTION_FACTOR: float = 1.0
# 仅作为 legacy decomposition fallback 使用。默认 accumulated 主路径已切换为
# local linearized residual，不再依赖 decomposition 后的经验放大因子。
XCORR_DECOMPOSITION_CORRECTION_FACTOR: float = 1.0
XCORR_PATCH_HALF: int = 7   # decomposition 场景下过大的 patch 会引入外围串扰；7 在 run_002 上更稳

# 搜索半径与旧接口兼容；真正限制错配的是这个半径，而不是 patch 大小
XCORR_WINDOW_SIZE: int = 32
XCORR_SEARCH_RADIUS: int = 8
XCORR_SUBPIXEL: bool = True
XCORR_SUBPIXEL_METHOD: str = 'quadratic2d'  # 'quadratic2d' | 'parabolic'
XCORR_SUBPIXEL_RADIUS: int = 2

# Local ZNCC 预处理
XCORR_BG_PERCENTILE: float = 10.0
XCORR_GAMMA: float = 1.0
XCORR_WEIGHT_ALPHA: float = 0.75

# 低置信度 spot 的空间修复
XCORR_MIN_SCORE: float = 0.10
XCORR_MIN_PEAK_MARGIN: float = 0.015
XCORR_MIN_PEAK_SHARPNESS: float = 0.010
XCORR_ENABLE_SPATIAL_REPAIR: bool = True
XCORR_SPATIAL_OUTLIER_PX: float = 2.0


# =============================================================================
# 4. 斜率积分方法 (integrate_slopes)
# =============================================================================
# 'southwell' : Southwell 最小二乘重建 (推荐, 无路径累积误差)
# 'trapezoidal' : 梯形积分 (旧方法, 误差沿积分路径累积)
INTEGRATION_METHOD: str = 'southwell'
# 当某些 spot 互相关峰值置信度较低时，Southwell 会自动做加权最小二乘。
# 该 floor 避免边界观测被完全丢弃，导致整体解过度漂移。
SOUTHWELL_CONFIDENCE_FLOOR: float = 0.05

# 当前验证目标是 DPC 位移本身；默认不输出相位积分产物或相位比较图。
OUTPUT_DPC_ONLY: bool = True
SAVE_DPC_FIGURES: bool = True
# 仅当 GT 位移模长不小于该阈值时，才统计百分比型 DPC 向量相对误差，
# 避免极小位移被百分比指标夸大。
DPC_RELATIVE_ERROR_MIN_TRUTH_PX: float = 0.2
# 方向误差的逐孔可信度判断需要屏蔽极小位移 spot；
# 该阈值以下的孔记为 low-signal，而不是简单判定为失败。
DPC_DIRECTION_LOW_SIGNAL_TRUTH_PX: float = 0.05
# 逐孔 trust mask 的经验阈值：
# trusted: 同时满足 good 阈值
# review:  满足 warn 阈值但未达到 good
# suspicious: 超出 warn 阈值
DPC_TRUST_MAG_ERROR_GOOD_PX: float = 0.01
DPC_TRUST_MAG_ERROR_WARN_PX: float = 0.02
DPC_TRUST_DIRECTION_GOOD_DEG: float = 5.0
DPC_TRUST_DIRECTION_WARN_DEG: float = 10.0

# =============================================================================
# 5. 优化求解器 (shift_optimization) — 独立流程 B
# =============================================================================
# 求解器选择: 'direct', 'zernike', 'auto'
#   direct : 逐光斑逐帧独立参数，适合少光斑少帧
#   zernike: 用 Zernike 多项式约束，参数更少，适合重叠/大规模
#   auto   : 根据重叠率和参数量自动选择
OPTIMIZATION_SOLVER: str = 'auto'

# 最大位移约束 (像素)
OPTIMIZATION_MAX_SHIFT: float = 5.0

# --- 正则化权重 ---
# 空间正则化: 相邻光斑位移应平滑相似
OPTIMIZATION_LAMBDA_SPATIAL: float = 0.1

# 时间正则化: 相邻帧位移应平滑变化
# 默认关闭：当前任务允许不同 pulse 的波前独立变化
OPTIMIZATION_LAMBDA_TEMPORAL: float = 0.0

# Zernike 系数正则化 (仅 zernike 求解器有效)
OPTIMIZATION_LAMBDA_COEFF: float = 0.01

# --- 优化迭代控制 ---
OPTIMIZATION_MAX_ITER_DIRECT: int = 200
OPTIMIZATION_MAX_ITER_ZERNIKE: int = 500
OPTIMIZATION_FTOL: float = 1e-9    # 函数值收敛容差
OPTIMIZATION_GTOL: float = 1e-9    # 梯度收敛容差

# 是否运行优化分析
RUN_OPTIMIZATION_ANALYSIS: bool = True

# =============================================================================
# 6. 光斑模型参数
# =============================================================================
# 光斑 σ 缩放因子 (相对于自动估计值)
SPOT_SIGMA_SCALE: float = 0.8

# 光斑模型类型: 'gaussian' (快) or 'template' (精确)
OPTIMIZATION_SPOT_MODEL: str = 'template'

# 优化求解器用 Zernike 模式数 (最多 15)
OPTIMIZATION_N_ZERNIKE: int = 15

# =============================================================================
# 7. Accumulated 图像的 trajectory-aware 分解
# =============================================================================
# 当没有 aberrated per-frame 图像，但有每个 pulse 的 ideal 图像时，
# 默认直接在 accumulated residual 上求解每个 pulse/spot 的局部位移：
#   accumulated - ideal ~= sum_p(-Ix_p * dx_p - Iy_p * dy_p)
# 这使用真实 ideal spot 形状，不假设高斯，也不需要互相关/decomposition 的尺度补偿。
ENABLE_ACCUMULATED_DECOMPOSITION: bool = True
ACCUMULATED_RECOVERY_METHOD: str = 'linearized_residual'  # 'linearized_residual' | 'decomposition'
ACCUMULATED_DECOMPOSITION_METHOD: str = 'iterative'   # 'weighted' | 'iterative' | 'template'
ACCUMULATED_LINEARIZED_ROI_EXPAND_PX: int = 10
ACCUMULATED_LINEARIZED_RIDGE_RELATIVE: float = 1e-2
ACCUMULATED_LINEARIZED_WEIGHT_POWER: float = 0.5
ACCUMULATED_LINEARIZED_MIN_WEIGHT: float = 0.02
ACCUMULATED_LINEARIZED_SIGNAL_QUANTILE: float = 0.20
ACCUMULATED_LINEARIZED_SMOOTH_SIGMA: float = 0.0
ACCUMULATED_LINEARIZED_CONFIDENCE_CONDITION_SCALE: float = 20.0
# 先验权重采用“相对局部数据项尺度”的无量纲系数。
# spatial: 同一 pulse 下相邻 spot 的 dx/dy 平滑
# temporal: 同一 spot 沿 pulse/trajectory 的弱平滑
ACCUMULATED_LINEARIZED_SPATIAL_PRIOR_RELATIVE: float = 0.01
ACCUMULATED_LINEARIZED_TEMPORAL_PRIOR_RELATIVE: float = 0.002
# 坏点会自动降权：confidence 越低，其局部数据项越容易被先验拉回。
ACCUMULATED_LINEARIZED_CONFIDENCE_DATA_POWER: float = 1.0
ACCUMULATED_LINEARIZED_CONFIDENCE_DATA_FLOOR: float = 0.5

# 主恢复路径选择
# accumulated: 真实主路径，只使用单张含像差 accumulated 图和无像差先验
# per_frame_eval: 仅用于仿真评估，上界基线，可直接使用逐帧含像差图
# auto: 优先 accumulated，只有不满足条件时才退回 per_frame_eval
PRIMARY_RECOVERY_MODE: str = 'accumulated'   # 'accumulated' | 'per_frame_eval' | 'auto'

# 若仿真目录中存在逐帧含像差图，是否额外计算一个 oracle 式 per-frame 评估基线
# 该基线仅用于对比/评估，不参与主恢复。
SAVE_PER_FRAME_EVAL_BASELINE: bool = False

# Step 2 结束后是否自动触发 Step 3 分析。
# 默认关闭：当前 X-ray 主流程只保留 guided cross-correlation 最佳恢复结果，
# 避免批量扫描时重新生成 centroid/phase_correlation 等旧方法对比。
RUN_POST_RECOVERY_ANALYSIS: bool = False

# post-recovery 分析模式:
# match: 按当前恢复路径选择对应分析分支
# both: 同时运行 decomposition 和 optimization 两条 Step 3 报告
# none: 不自动运行 Step 3
POST_RECOVERY_ANALYSIS_MODE: str = 'match'   # 'match' | 'both' | 'none'

# per-pulse 参考图构建
# auto: 优先使用保存的每 pulse ideal 图；否则使用单参考平移；再否则由 ideal_accumulated + trajectory 反卷积估计
REFERENCE_SEQUENCE_MODE: str = 'per_pulse'   # 'auto' | 'per_pulse' | 'single_reference' | 'deconvolve'
REFERENCE_SHIFT_ORDER: int = 1
REFERENCE_DECONV_REG: float = 1e-3

# accumulated optimization 是否使用 decomposition + xcorr 生成初始化
OPTIMIZATION_USE_DECOMPOSITION_INIT: bool = True

# =============================================================================
# 8. 光斑识别 (用于高级分析)
# =============================================================================
SPOT_BINARIZE_METHOD: str = 'otsu'     # 'otsu' or 'adaptive'
SPOT_MIN_AREA: int = 4
SPOT_MAX_AREA: int = 10000
SPOT_SCORE_THRESHOLD: float = 0.3

# =============================================================================
# 9. Zernike 波前重建参数 (zernike_reconstruction)
# =============================================================================
# 重建输出用 Zernike 模式数 (与优化求解器的 OPTIMIZATION_N_ZERNIKE 独立)
RECONSTRUCTION_N_ZERNIKE: int = 15

# 波前重建网格分辨率
ZERNIKE_GRID_SIZE: int = 128

# 瞳孔边缘余量因子 (1.0 = 恰好到边界)
ZERNIKE_GRID_MARGIN: float = 1.15

# 瞳孔半径 (米), None = 自动从光斑位置估算
ZERNIKE_PUPIL_RADIUS_M: Optional[float] = None

# =============================================================================
# 10. 去倾斜 (tip/tilt removal)
# =============================================================================
# 是否自动去除倾斜 (piston + tilt)
REMOVE_TIP_TILT: bool = True
