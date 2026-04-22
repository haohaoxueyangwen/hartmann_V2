# OUTPUT_GUIDE.md

这份文档只解释**当前统一版仓库、当前默认配置、当前真实结果目录**里已经产出的输出，不再混入旧版 `phase-first`、`decomposition`、`legacy demo` 的口径。

当前默认链路：

- `simulate.py`
- `recover.py`
- `OUTPUT_DPC_ONLY = True`
- 主恢复方法：`cross_correlation`

当前 authoritative 示例目录：

- `/Users/yaaang/Desktop/model2/result/xray/sweeps_20260422_001/wfe/wfe_1p0rad_step8p0um_z450mm`

文中所有 shape、文件名、字段名，都按这个目录里的真实文件来写。

## 1. 总原则

### 1.1 哪些文件是真正的“数据”

- `.npy` / `.npz` / `.csv` / `.json` 是定量数据文件。
- `.png` 是可视化结果或预览图。
- 如果某个二维 `.npy` 同时有同名 `.png`，那么 `.png` 只是这个二维数组的图像化版本，本身不增加新的定量信息。
- 三维 stack 类文件通常只有 `.npy`，因为它们不是单张二维图。

### 1.2 当前目录结构

```text
run_dir/
├── accumulated.npy
├── accumulated.png
├── ideal_accumulated.npy
├── ideal_accumulated.png
├── reference_ideal.npy
├── reference_ideal.png
├── simulation_params.json
├── phase_screens.npy
├── phase_screens.png
├── gt_displacements.npz
├── gt_displacements.png
├── ideal_centers.npy
├── spot_grid_indices.npy
├── frames/
│   ├── frame_000.npy
│   ├── frame_000.png
│   ├── frame_000_ideal.npy
│   ├── frame_000_ideal.png
│   ├── frame_001.npy
│   ├── frame_001.png
│   ├── frame_001_ideal.npy
│   ├── frame_001_ideal.png
│   ├── frame_002.npy
│   ├── frame_002.png
│   ├── frame_002_ideal.npy
│   ├── frame_002_ideal.png
│   ├── frame_003.npy
│   ├── frame_003.png
│   ├── frame_003_ideal.npy
│   ├── frame_003_ideal.png
│   ├── frame_004.npy
│   ├── frame_004.png
│   ├── frame_004_ideal.npy
│   ├── frame_004_ideal.png
│   ├── frame_005.npy
│   ├── frame_005.png
│   ├── frame_005_ideal.npy
│   ├── frame_005_ideal.png
│   ├── frame_006.npy
│   ├── frame_006.png
│   ├── frame_006_ideal.npy
│   ├── frame_006_ideal.png
│   ├── frame_007.npy
│   ├── frame_007.png
│   ├── frame_007_ideal.npy
│   ├── frame_007_ideal.png
│   ├── frame_008.npy
│   ├── frame_008.png
│   ├── frame_008_ideal.npy
│   ├── frame_008_ideal.png
│   ├── frame_009.npy
│   ├── frame_009.png
│   ├── frame_009_ideal.npy
│   └── frame_009_ideal.png
└── analysis/
    ├── accumulated_recovery_cross_correlation.json
    ├── all_displacement_x_cross_correlation.npy
    ├── all_displacement_x_cross_correlation.png
    ├── all_displacement_y_cross_correlation.npy
    ├── all_displacement_y_cross_correlation.png
    ├── all_frames_disp_summary_cross_correlation.png
    ├── all_shift_confidence_cross_correlation.npy
    ├── all_shift_confidence_cross_correlation.png
    ├── all_shifts_cross_correlation.npy
    ├── all_shifts_cross_correlation.png
    ├── displacement_comparison_cross_correlation.png
    ├── displacement_x_cross_correlation.npy
    ├── displacement_x_cross_correlation.png
    ├── displacement_y_cross_correlation.npy
    ├── displacement_y_cross_correlation.png
    ├── dpc_bias_x_per_spot_px_cross_correlation.npy
    ├── dpc_bias_x_per_spot_px_cross_correlation.png
    ├── dpc_bias_y_per_spot_px_cross_correlation.npy
    ├── dpc_bias_y_per_spot_px_cross_correlation.png
    ├── dpc_condition_number_per_spot_cross_correlation.npy
    ├── dpc_condition_number_per_spot_cross_correlation.png
    ├── dpc_error_mag_stack_px_cross_correlation.npy
    ├── dpc_error_x_stack_px_cross_correlation.npy
    ├── dpc_error_y_stack_px_cross_correlation.npy
    ├── dpc_low_signal_mask_per_spot_cross_correlation.npy
    ├── dpc_low_signal_mask_per_spot_cross_correlation.png
    ├── dpc_mean_abs_parallel_error_px_per_spot_cross_correlation.npy
    ├── dpc_mean_abs_parallel_error_px_per_spot_cross_correlation.png
    ├── dpc_mean_abs_perpendicular_error_px_per_spot_cross_correlation.npy
    ├── dpc_mean_abs_perpendicular_error_px_per_spot_cross_correlation.png
    ├── dpc_mean_angle_deg_per_spot_cross_correlation.npy
    ├── dpc_mean_angle_deg_per_spot_cross_correlation.png
    ├── dpc_mean_confidence_per_spot_cross_correlation.npy
    ├── dpc_mean_confidence_per_spot_cross_correlation.png
    ├── dpc_mean_cosine_per_spot_cross_correlation.npy
    ├── dpc_mean_cosine_per_spot_cross_correlation.png
    ├── dpc_mean_parallel_error_px_per_spot_cross_correlation.npy
    ├── dpc_mean_parallel_error_px_per_spot_cross_correlation.png
    ├── dpc_mean_perpendicular_error_px_per_spot_cross_correlation.npy
    ├── dpc_mean_perpendicular_error_px_per_spot_cross_correlation.png
    ├── dpc_mean_relative_error_pct_per_spot_cross_correlation.npy
    ├── dpc_mean_relative_error_pct_per_spot_cross_correlation.png
    ├── dpc_median_abs_direction_error_deg_per_spot_cross_correlation.npy
    ├── dpc_median_abs_direction_error_deg_per_spot_cross_correlation.png
    ├── dpc_median_abs_magnitude_error_px_per_spot_cross_correlation.npy
    ├── dpc_median_abs_magnitude_error_px_per_spot_cross_correlation.png
    ├── dpc_p95_relative_error_pct_per_spot_cross_correlation.npy
    ├── dpc_p95_relative_error_pct_per_spot_cross_correlation.png
    ├── dpc_parallel_error_stack_px_cross_correlation.npy
    ├── dpc_perpendicular_error_stack_px_cross_correlation.npy
    ├── dpc_recovered_x_stack_px_cross_correlation.npy
    ├── dpc_recovered_y_stack_px_cross_correlation.npy
    ├── dpc_relative_error_pct_stack_cross_correlation.npy
    ├── dpc_residual_ratio_per_spot_cross_correlation.npy
    ├── dpc_residual_ratio_per_spot_cross_correlation.png
    ├── dpc_rmse_mag_per_spot_px_cross_correlation.npy
    ├── dpc_rmse_mag_per_spot_px_cross_correlation.png
    ├── dpc_rmse_x_per_spot_px_cross_correlation.npy
    ├── dpc_rmse_x_per_spot_px_cross_correlation.png
    ├── dpc_rmse_y_per_spot_px_cross_correlation.npy
    ├── dpc_rmse_y_per_spot_px_cross_correlation.png
    ├── dpc_scale_vec_per_spot_cross_correlation.npy
    ├── dpc_scale_vec_per_spot_cross_correlation.png
    ├── dpc_scale_x_per_spot_cross_correlation.npy
    ├── dpc_scale_x_per_spot_cross_correlation.png
    ├── dpc_scale_y_per_spot_cross_correlation.npy
    ├── dpc_scale_y_per_spot_cross_correlation.png
    ├── dpc_spot_metrics_cross_correlation.csv
    ├── dpc_spot_summary_cross_correlation.png
    ├── dpc_spot_trust_maps_cross_correlation.png
    ├── dpc_spot_trust_summary_cross_correlation.csv
    ├── dpc_trust_code_per_spot_cross_correlation.npy
    ├── dpc_truth_equivalent_slope_mrad_per_spot_cross_correlation.npy
    ├── dpc_truth_equivalent_slope_mrad_per_spot_cross_correlation.png
    ├── dpc_truth_mag_per_spot_px_cross_correlation.npy
    ├── dpc_truth_mag_per_spot_px_cross_correlation.png
    ├── dpc_truth_mag_stack_px_cross_correlation.npy
    ├── dpc_truth_phase_local_rms_rad_per_spot_cross_correlation.npy
    ├── dpc_truth_phase_local_rms_rad_per_spot_cross_correlation.png
    ├── dpc_truth_x_stack_px_cross_correlation.npy
    ├── dpc_truth_y_stack_px_cross_correlation.npy
    ├── linearized_condition_numbers_cross_correlation.npy
    ├── linearized_condition_numbers_cross_correlation.png
    ├── linearized_residual_ratios_cross_correlation.npy
    ├── linearized_residual_ratios_cross_correlation.png
    ├── metrics_cross_correlation.json
    ├── shift_confidence_cross_correlation.npy
    ├── shift_confidence_cross_correlation.png
    ├── per_frame_displacement/
    │   ├── displacement_frame_000.png
    │   ├── displacement_frame_001.png
    │   ├── displacement_frame_002.png
    │   ├── displacement_frame_003.png
    │   ├── displacement_frame_004.png
    │   ├── displacement_frame_005.png
    │   ├── displacement_frame_006.png
    │   ├── displacement_frame_007.png
    │   ├── displacement_frame_008.png
    │   ├── displacement_frame_009.png
    │   ├── per_frame_rmse_cross_correlation.png
    │   ├── quiver_overview_cross_correlation.png
    │   └── quiver_frames/
    │       ├── quiver_frame_000_cross_correlation.png
    │       ├── quiver_frame_001_cross_correlation.png
    │       ├── quiver_frame_002_cross_correlation.png
    │       ├── quiver_frame_003_cross_correlation.png
    │       ├── quiver_frame_004_cross_correlation.png
    │       ├── quiver_frame_005_cross_correlation.png
    │       ├── quiver_frame_006_cross_correlation.png
    │       ├── quiver_frame_007_cross_correlation.png
    │       ├── quiver_frame_008_cross_correlation.png
    │       └── quiver_frame_009_cross_correlation.png
    └── statistics/
        ├── dpc_angle_error_summary_cross_correlation.csv
        ├── dpc_angle_error_summary_cross_correlation.png
        ├── dpc_error_vs_truth_bins_cross_correlation.csv
        ├── dpc_error_vs_truth_bins_cross_correlation.png
        ├── dpc_pointwise_direction_and_magnitude_error_cross_correlation.csv
        ├── dpc_pointwise_direction_and_magnitude_error_cross_correlation.png
        ├── dpc_signal_vs_error_scale_cross_correlation.csv
        ├── dpc_signal_vs_error_scale_cross_correlation.png
        ├── dpc_spot_trust_scatter_cross_correlation.png
        ├── dpc_truth_wavefront_context_cross_correlation.png
        ├── error_statistics_cross_correlation.png
        ├── metrics_all_pulses_cross_correlation.png
        ├── per_frame_error_percentiles_cross_correlation.png
        ├── per_frame_metrics_cross_correlation.csv
        ├── per_frame_metrics_cross_correlation.json
        └── spot_metric_histograms_cross_correlation.png
```

### 1.3 当前主线的坐标和单位

- 探测器原始图像：`(1024, 1024)`，单位是像素强度。
- Hartmann spot 网格：`(26, 26)`。
- 当前 pulse 数：`10`。
- `*_px` 表示像素位移。
- `*_m` 表示物理长度。
- `*_deg` 表示角度，单位度。
- `*_mrad` 表示毫弧度。
- 没有单位后缀时，要以字段说明为准。

## 2. 根目录原始输出

### 2.1 观测输入与理想参考

| 文件 | shape | 数据含义 |
|------|-------|----------|
| `accumulated.npy` | `(1024, 1024)` | 多脉冲含像差图像的累计结果。当前恢复的核心输入。 |
| `ideal_accumulated.npy` | `(1024, 1024)` | 多脉冲无像差图像的累计结果。用于 accumulated recovery 的理想参考。 |
| `reference_ideal.npy` | `(1024, 1024)` | 单帧无像差理想参考图。局部互相关时的 spot 参考。 |
| `accumulated.png` / `ideal_accumulated.png` / `reference_ideal.png` | 预览图 | 上述三张二维图像的可视化。 |

### 2.2 仿真参数与高分辨率真值

| 文件 | shape | 数据含义 |
|------|-------|----------|
| `simulation_params.json` | JSON | 当前 case 的全部仿真参数。包括波长、传播距离、像素尺寸、Hartmann pitch、孔径、扫描步长、pulse 数、运动配置等。 |
| `phase_screens.npy` | `(10, 4096, 4096)` | 每个 pulse 的高分辨率 phase screen 真值，单位 rad。它不是 Hartmann 采样网格，而是原始连续波前离散化后的高分辨率屏。 |
| `phase_screens.png` | 预览图 | `phase_screens.npy` 的代表性帧预览。 |

### 2.3 Hartmann 网格真值与网格几何

| 文件 | shape | 数据含义 |
|------|-------|----------|
| `gt_displacements.npz` | 压缩字典 | 当前 DPC 评估的真值主文件。存放每帧每孔的真值位移，以及 Hartmann 网格上的真值 phase。 |
| `gt_displacements.png` | 预览图 | `gt_displacements.npz` 中 Hartmann 网格真值的快速预览。 |
| `ideal_centers.npy` | `(676, 2)` | 每个 spot 的理想中心坐标，按 `(x_px, y_px)` 存储。 |
| `spot_grid_indices.npy` | `(676, 2)` | 每个 spot 对应的 `(row, col)` 网格索引。它把一维 spot 列表映射回 `26 x 26` 网格。 |

### 2.4 `gt_displacements.npz` 内部键

| 键 | shape | 数据含义 |
|----|-------|----------|
| `grid_rows` / `grid_cols` | 标量 | 当前 Hartmann 网格维度。主线是 `26 x 26`。 |
| `centers_x_m` | `(26,)` | 各列 spot 中心的物理 x 坐标。 |
| `centers_y_m` | `(26,)` | 各行 spot 中心的物理 y 坐标。 |
| `phase_rad` | `(10, 26, 26)` | Hartmann 网格上每帧每孔的真值 phase，单位 rad。 |
| `displacement_x_px` | `(10, 26, 26)` | 每帧每孔的真值 dX，单位 px。 |
| `displacement_y_px` | `(10, 26, 26)` | 每帧每孔的真值 dY，单位 px。 |
| `displacement_x_m` | `(10, 26, 26)` | 每帧每孔的真值 dX，单位 m。 |
| `displacement_y_m` | `(10, 26, 26)` | 每帧每孔的真值 dY，单位 m。 |
| `wavelength_m` | 标量 | 仿真波长。 |
| `z_prop_m` | 标量 | 传播距离。 |
| `pixel_size_m` | 标量 | 探测器像素尺寸。 |
| `spacing_m` | 标量 | Hartmann pitch。 |

### 2.5 `frames/`

| 文件 | shape | 数据含义 |
|------|-------|----------|
| `frames/frame_000.npy` ~ `frames/frame_009.npy` | `(1024, 1024)` | 每个 pulse 的含像差瞬时探测器图像。 |
| `frames/frame_000_ideal.npy` ~ `frames/frame_009_ideal.npy` | `(1024, 1024)` | 每个 pulse 的无像差理想瞬时探测器图像。 |
| 对应 `.png` | 预览图 | 每帧图像的快速可视化。 |

## 3. `analysis/` 下的主结果

### 3.1 主结果 JSON

#### `accumulated_recovery_cross_correlation.json`

这是 accumulated recovery 的元信息摘要，不是数组结果本身。当前键包括：

| 键 | 数据含义 |
|----|----------|
| `recovery_mode` | 当前恢复模式。 |
| `accumulated_recovery_method` | 累计图恢复方法。 |
| `reference_sequence_source` | 参考序列来源。 |
| `median_condition_number` | 线性化求解条件数的中位数。 |
| `median_residual_ratio` | 线性化残差比中位数。 |
| `median_data_scale` | 数据尺度中位数。 |
| `spatial_prior_weight` | 空间先验权重。 |
| `temporal_prior_weight` | 时间先验权重。 |

#### `metrics_cross_correlation.json`

这是当前恢复的总指标汇总，最适合程序化读取。顶层键包括：

| 键 | 数据含义 |
|----|----------|
| `recovery_mode` | 当前恢复模式。 |
| `primary_recovery_mode` | 主恢复模式说明。 |
| `used_per_frame_aberrated_for_recovery` | 是否使用逐帧含像差图参与恢复。 |
| `reference_sequence_source` | 参考序列来源。 |
| `decomposition_method` | 分解方法记录。当前 DPC-only 主线下主要是兼容性记录。 |
| `accumulated_recovery_method` | 累计图恢复方法。 |
| `metrics_source` | 指标统计来源。 |
| `output_mode` | 输出模式。 |
| `linearized_median_condition_number` | 全局线性化条件数中位数。 |
| `linearized_median_residual_ratio` | 全局线性化残差比中位数。 |
| `linearized_median_data_scale` | 全局数据尺度中位数。 |
| `linearized_spatial_prior_weight` | 空间先验权重。 |
| `linearized_temporal_prior_weight` | 时间先验权重。 |
| `displacement_metrics` | 全局逐帧位移恢复指标块。 |
| `dpc_spot_metrics` | 逐孔 DPC 与 trust 统计块。 |

`displacement_metrics` 内部键：

| 键 | 数据含义 |
|----|----------|
| `mode` | 指标统计模式。 |
| `n_frames` | 纳入统计的帧数。 |
| `grid_used` | 实际使用的 spot 网格。 |
| `rmse_dx_px` / `rmse_dy_px` / `rmse_total_px` | 全局 dX / dY / 向量总 RMSE。 |
| `mean_err_px` | 全局平均向量误差。 |
| `max_err_px` | 全局最大向量误差。 |
| `mean_corr_dx` / `mean_corr_dy` | 全局平均相关系数。 |
| `median_frame_rmse_px` / `p95_frame_rmse_px` | 逐帧 RMSE 的中位数 / P95。 |
| `best_frame_index` / `best_frame_rmse_px` | 最优帧索引及其 RMSE。 |
| `worst_frame_index` / `worst_frame_rmse_px` | 最差帧索引及其 RMSE。 |
| `per_frame_rmse` | 每帧 RMSE 数组。 |
| `per_frame_metrics` | 每帧详细指标对象列表。 |

`dpc_spot_metrics` 内部键：

| 键 | 数据含义 |
|----|----------|
| `mode` | 当前 DPC 统计模式。 |
| `n_frames` | 纳入统计的帧数。 |
| `grid_used` | 实际使用的 spot 网格。 |
| `median_truth_mag_px` / `p95_truth_mag_px` / `max_truth_mag_px` | 样本级真值位移大小统计。 |
| `median_sample_abs_error_px` / `p95_sample_abs_error_px` / `max_sample_abs_error_px` | 样本级绝对误差统计。 |
| `median_sample_angle_deg` / `p95_abs_direction_error_deg` | 样本级方向误差统计。 |
| `median_sample_abs_parallel_error_px` / `p95_sample_abs_parallel_error_px` | 样本级平行误差统计。 |
| `median_sample_abs_perpendicular_error_px` / `p95_sample_abs_perpendicular_error_px` | 样本级垂直误差统计。 |
| `median_mean_abs_error_px` / `p95_mean_abs_error_px` | 逐孔平均绝对误差的分布统计。 |
| `median_mean_abs_parallel_error_px` / `p95_mean_abs_parallel_error_px` | 逐孔平均绝对平行误差的分布统计。 |
| `median_mean_abs_perpendicular_error_px` / `p95_mean_abs_perpendicular_error_px` | 逐孔平均绝对垂直误差的分布统计。 |
| `median_spot_rmse_px` / `p95_spot_rmse_px` / `max_spot_rmse_px` | 逐孔 RMSE 的分布统计。 |
| `median_abs_bias_x_px` / `median_abs_bias_y_px` | 逐孔偏置绝对值中位数。 |
| `median_scale_x` / `median_scale_y` / `median_scale_vec` | 逐孔尺度因子的中位数。 |
| `p05_scale_vec` / `p95_scale_vec` | 向量尺度因子的分位数。 |
| `median_mean_angle_deg` / `p95_mean_angle_deg` | 逐孔平均方向误差的分布统计。 |
| `median_mean_cosine` | 逐孔平均 cosine similarity 中位数。 |
| `median_confidence` | 逐孔平均互相关置信度中位数。 |
| `median_condition_number` / `median_residual_ratio` | 数值稳定性指标中位数。 |
| `median_truth_equivalent_slope_mrad` | 真值等效 slope 中位数。 |
| `median_truth_phase_local_rms_rad` | 每孔局部 phase RMS 中位数。 |
| `median_mean_relative_error_pct` | 逐孔平均相对误差百分比中位数。 |
| `median_signal_to_error_ratio` | 信号与误差比的中位数。 |
| `fraction_sample_truth_lt_0p05_px` / `fraction_sample_truth_lt_0p10_px` / `fraction_sample_truth_lt_0p20_px` | 低信号样本占比。 |
| `fraction_sample_abs_error_lt_0p01_px` / `fraction_sample_abs_error_lt_0p02_px` / `fraction_sample_abs_error_lt_0p05_px` | 误差低于阈值的样本占比。 |
| `fraction_sample_angle_lt_5_deg` / `fraction_sample_angle_lt_10_deg` / `fraction_sample_angle_lt_15_deg` | 方向误差低于阈值的样本占比。 |
| `relative_error_truth_threshold_px` | 相对误差统计使用的真值门限。 |
| `n_low_signal_spots` / `n_review_spots` / `n_suspicious_spots` / `n_trusted_spots` | trust 分类计数。 |
| `worst_spots` | 最差 spot 列表，便于快速定位异常孔位。 |

### 3.2 最值得先看的主图

| 文件 | 数据含义 |
|------|----------|
| `displacement_comparison_cross_correlation.png` | 代表性帧 dX / dY 的真值、恢复、误差和散点对照主图。 |
| `all_frames_disp_summary_cross_correlation.png` | 所有 10 帧真值位移、恢复位移和误差总览。 |
| `dpc_spot_summary_cross_correlation.png` | 每孔 DPC 统计总览图，集中展示 truth、RMSE、parallel/perpendicular、scale、angle、confidence 等二维图。 |
| `dpc_spot_trust_maps_cross_correlation.png` | 每孔真值位移、每孔幅值误差、每孔方向误差、trust mask 的 2x2 总览。最适合回答“哪些孔可信”。 |

### 3.3 代表性恢复数组

这类文件主要记录代表性二维结果和逐帧原始恢复结果。

| 文件 | shape | 数据含义 |
|------|-------|----------|
| `displacement_x_cross_correlation.npy` | `(26, 26)` | 代表性帧的 dX，单位 m。 |
| `displacement_y_cross_correlation.npy` | `(26, 26)` | 代表性帧的 dY，单位 m。 |
| `shift_confidence_cross_correlation.npy` | `(26, 26)` | 代表性帧的局部互相关置信度，无单位。 |
| `all_displacement_x_cross_correlation.npy` | `(10, 26, 26)` | 全部帧的 dX，单位 m。 |
| `all_displacement_y_cross_correlation.npy` | `(10, 26, 26)` | 全部帧的 dY，单位 m。 |
| `all_shift_confidence_cross_correlation.npy` | `(10, 26, 26)` | 全部帧的局部互相关置信度。 |
| `all_shifts_cross_correlation.npy` | `(10, 26, 26, 2)` | 全部帧的原始位移向量场，单位 px。`[..., 0] = dX`，`[..., 1] = dY`。 |

同名 `.png` 的含义：

| 文件 | 数据含义 |
|------|----------|
| `displacement_x_cross_correlation.png` / `displacement_y_cross_correlation.png` | 代表性帧 dX / dY 二维图。 |
| `shift_confidence_cross_correlation.png` | 代表性帧置信度图。 |
| `all_displacement_x_cross_correlation.png` / `all_displacement_y_cross_correlation.png` | 全帧 dX / dY 汇总预览。 |
| `all_shift_confidence_cross_correlation.png` | 全帧置信度汇总预览。 |
| `all_shifts_cross_correlation.png` | 全帧原始位移向量场汇总预览。 |

## 4. `analysis/` 里的 DPC stack 数据

这一层是**逐帧逐孔样本级数据**，shape 主要是 `(10, 26, 26)`。它们是后续所有 per-spot 统计、直方图、散点图、trust 图的原始来源。

### 4.1 真值 / 恢复 stack

| 文件 | shape | 数据含义 |
|------|-------|----------|
| `dpc_truth_x_stack_px_cross_correlation.npy` | `(10, 26, 26)` | 真值 dX stack，单位 px。 |
| `dpc_truth_y_stack_px_cross_correlation.npy` | `(10, 26, 26)` | 真值 dY stack，单位 px。 |
| `dpc_truth_mag_stack_px_cross_correlation.npy` | `(10, 26, 26)` | 真值位移模长 stack，单位 px。 |
| `dpc_recovered_x_stack_px_cross_correlation.npy` | `(10, 26, 26)` | 恢复 dX stack，单位 px。 |
| `dpc_recovered_y_stack_px_cross_correlation.npy` | `(10, 26, 26)` | 恢复 dY stack，单位 px。 |

### 4.2 误差 stack

| 文件 | shape | 数据含义 |
|------|-------|----------|
| `dpc_error_x_stack_px_cross_correlation.npy` | `(10, 26, 26)` | 逐样本 dX 误差，`recovered_dx - truth_dx`。 |
| `dpc_error_y_stack_px_cross_correlation.npy` | `(10, 26, 26)` | 逐样本 dY 误差，`recovered_dy - truth_dy`。 |
| `dpc_error_mag_stack_px_cross_correlation.npy` | `(10, 26, 26)` | 逐样本向量误差模长。 |
| `dpc_parallel_error_stack_px_cross_correlation.npy` | `(10, 26, 26)` | 误差在真值方向上的投影。正值表示沿真值方向偏大，负值表示沿真值方向偏小。 |
| `dpc_perpendicular_error_stack_px_cross_correlation.npy` | `(10, 26, 26)` | 误差在真值垂直方向上的投影。它更直接反映“方向跑偏”而不是“幅值变大/变小”。 |
| `dpc_relative_error_pct_stack_cross_correlation.npy` | `(10, 26, 26)` | 逐样本相对误差百分比。低信号处可能是 `NaN`，因为相对误差在极小真值下不稳定。 |

## 5. `analysis/` 里的每孔标量图

这一层是**把同一 spot 在 10 帧上的统计压缩成一个数**之后得到的二维图，shape 基本都是 `(26, 26)`。

同名 `.png` 的作用是把这些二维标量图可视化；`.npy` 才是定量数据本体。

### 5.1 每孔真值 / RMSE / 偏置

| 文件 | 数据含义 |
|------|----------|
| `dpc_truth_mag_per_spot_px_cross_correlation.npy` | 每孔真值位移大小的代表统计，单位 px。 |
| `dpc_rmse_x_per_spot_px_cross_correlation.npy` | 每孔 dX RMSE。 |
| `dpc_rmse_y_per_spot_px_cross_correlation.npy` | 每孔 dY RMSE。 |
| `dpc_rmse_mag_per_spot_px_cross_correlation.npy` | 每孔向量误差 RMSE。 |
| `dpc_bias_x_per_spot_px_cross_correlation.npy` | 每孔 dX 平均偏置。 |
| `dpc_bias_y_per_spot_px_cross_correlation.npy` | 每孔 dY 平均偏置。 |

### 5.2 每孔平行 / 垂直误差

| 文件 | 数据含义 |
|------|----------|
| `dpc_mean_parallel_error_px_per_spot_cross_correlation.npy` | 每孔平均平行误差，可带符号。 |
| `dpc_mean_abs_parallel_error_px_per_spot_cross_correlation.npy` | 每孔平均绝对平行误差。 |
| `dpc_mean_perpendicular_error_px_per_spot_cross_correlation.npy` | 每孔平均垂直误差，可带符号。 |
| `dpc_mean_abs_perpendicular_error_px_per_spot_cross_correlation.npy` | 每孔平均绝对垂直误差。 |

### 5.3 每孔相对误差 / scale / angle / cosine

| 文件 | 数据含义 |
|------|----------|
| `dpc_mean_relative_error_pct_per_spot_cross_correlation.npy` | 每孔平均相对误差百分比。 |
| `dpc_p95_relative_error_pct_per_spot_cross_correlation.npy` | 每孔 P95 相对误差百分比。 |
| `dpc_scale_x_per_spot_cross_correlation.npy` | 每孔 x 分量尺度因子。接近 1 表示 dX 幅值校准较准。 |
| `dpc_scale_y_per_spot_cross_correlation.npy` | 每孔 y 分量尺度因子。 |
| `dpc_scale_vec_per_spot_cross_correlation.npy` | 每孔向量模长尺度因子。接近 1 表示位移大小整体恢复较准。 |
| `dpc_mean_angle_deg_per_spot_cross_correlation.npy` | 每孔平均方向误差，单位度。 |
| `dpc_mean_cosine_per_spot_cross_correlation.npy` | 每孔平均 cosine similarity。接近 1 表示方向一致性高。 |

### 5.4 每孔置信度与数值稳定性

| 文件 | 数据含义 |
|------|----------|
| `dpc_mean_confidence_per_spot_cross_correlation.npy` | 每孔平均互相关置信度。 |
| `dpc_condition_number_per_spot_cross_correlation.npy` | 每孔线性化求解条件数。数值越大通常表示问题越病态。 |
| `dpc_residual_ratio_per_spot_cross_correlation.npy` | 每孔线性化残差比。越大说明拟合残差相对越大。 |
| `linearized_condition_numbers_cross_correlation.npy` | 线性化求解阶段直接输出的二维条件数图。 |
| `linearized_residual_ratios_cross_correlation.npy` | 线性化求解阶段直接输出的二维残差比图。 |

对应 `.png`：

- `linearized_condition_numbers_cross_correlation.png`
- `linearized_residual_ratios_cross_correlation.png`

它们是上述两张二维图的可视化。

## 6. `analysis/` 里的 trust / low-signal / truth-context 数据

这一层专门服务于问题：“每个孔的恢复位移是否可信？”

### 6.1 trust 直接相关数组

| 文件 | shape | 数据含义 |
|------|-------|----------|
| `dpc_median_abs_magnitude_error_px_per_spot_cross_correlation.npy` | `(26, 26)` | 每孔跨帧的位移幅值绝对误差中位数。它回答“位移大小恢复得准不准”。 |
| `dpc_median_abs_direction_error_deg_per_spot_cross_correlation.npy` | `(26, 26)` | 每孔跨帧的方向绝对误差中位数。它回答“方向恢复得对不对”。 |
| `dpc_low_signal_mask_per_spot_cross_correlation.npy` | `(26, 26)` | 低信号孔位掩码。1 表示该孔真值长期太小，方向误差本身不稳定，不宜直接当作失败证据。 |
| `dpc_trust_code_per_spot_cross_correlation.npy` | `(26, 26)` | 每孔 trust 分类编码：`-1 = low_signal`，`0 = suspicious`，`1 = review`，`2 = trusted`。 |

### 6.2 真值波前上下文数组

| 文件 | shape | 数据含义 |
|------|-------|----------|
| `dpc_truth_equivalent_slope_mrad_per_spot_cross_correlation.npy` | `(26, 26)` | 由真值位移反推得到的等效 slope，单位 mrad。它和 Hartmann 位移信号直接对应。 |
| `dpc_truth_phase_local_rms_rad_per_spot_cross_correlation.npy` | `(26, 26)` | 每孔局部 phase RMS，单位 rad。它用来区分“位移信号小”和“phase 幅值小”这两件事。 |

### 6.3 trust 相关图

| 文件 | 数据含义 |
|------|----------|
| `dpc_spot_trust_maps_cross_correlation.png` | 每孔真值位移、每孔幅值误差、每孔方向误差、trust mask 的总览图。 |
| `dpc_low_signal_mask_per_spot_cross_correlation.png` | 低信号孔位空间分布图。 |
| `dpc_median_abs_magnitude_error_px_per_spot_cross_correlation.png` | 每孔幅值误差中位数热图。 |
| `dpc_median_abs_direction_error_deg_per_spot_cross_correlation.png` | 每孔方向误差中位数热图。 |
| `dpc_truth_equivalent_slope_mrad_per_spot_cross_correlation.png` | 每孔等效 slope 分布图。 |
| `dpc_truth_phase_local_rms_rad_per_spot_cross_correlation.png` | 每孔局部 phase RMS 分布图。 |

## 7. `analysis/` 里的 CSV 表

### 7.1 `dpc_spot_metrics_cross_correlation.csv`

每孔一行，偏传统误差统计。

| 字段 | 数据含义 |
|------|----------|
| `row`, `col` | 该 spot 的网格位置。 |
| `mean_truth_mag_px` | 该孔跨帧真值位移模长平均值。 |
| `mean_abs_error_mag_px` | 该孔跨帧向量误差模长绝对值平均值。 |
| `rmse_mag_px` | 该孔向量误差 RMSE。 |
| `rmse_x_px`, `rmse_y_px` | 该孔 dX / dY RMSE。 |
| `bias_x_px`, `bias_y_px` | 该孔 dX / dY 平均偏置。 |
| `max_error_mag_px` | 该孔最大向量误差。 |
| `mean_parallel_error_px` | 该孔平均平行误差，可带符号。 |
| `mean_abs_parallel_error_px` | 该孔平均绝对平行误差。 |
| `p95_abs_parallel_error_px` | 该孔绝对平行误差的 P95。 |
| `mean_perpendicular_error_px` | 该孔平均垂直误差，可带符号。 |
| `mean_abs_perpendicular_error_px` | 该孔平均绝对垂直误差。 |
| `p95_abs_perpendicular_error_px` | 该孔绝对垂直误差的 P95。 |
| `mean_relative_error_pct` | 该孔平均相对误差百分比。 |
| `p95_relative_error_pct` | 该孔相对误差百分比的 P95。 |
| `scale_x`, `scale_y`, `scale_vec` | 该孔尺度因子。 |
| `corr_x`, `corr_y` | 该孔 dX / dY 的相关系数。 |
| `mean_cosine` | 该孔方向一致性的 cosine 指标。 |
| `mean_angle_deg` | 该孔平均方向误差。 |
| `mean_confidence` | 该孔平均互相关置信度。 |
| `condition_number` | 该孔数值条件数。 |
| `residual_ratio` | 该孔残差比。 |

### 7.2 `dpc_spot_trust_summary_cross_correlation.csv`

每孔一行，偏“可信度说明”。

| 字段 | 数据含义 |
|------|----------|
| `row`, `col` | 该 spot 的网格位置。 |
| `median_truth_mag_px` | 该孔真值位移模长中位数。 |
| `median_abs_magnitude_error_px` | 该孔位移幅值绝对误差中位数。 |
| `p95_abs_magnitude_error_px` | 该孔位移幅值绝对误差的 P95。 |
| `median_abs_direction_error_deg` | 该孔方向绝对误差中位数。 |
| `p95_abs_direction_error_deg` | 该孔方向绝对误差的 P95。 |
| `median_abs_parallel_error_px` | 该孔绝对平行误差中位数。 |
| `median_abs_perpendicular_error_px` | 该孔绝对垂直误差中位数。 |
| `mean_confidence` | 该孔平均互相关置信度。 |
| `condition_number` | 该孔数值条件数。 |
| `residual_ratio` | 该孔残差比。 |
| `truth_equivalent_slope_mrad` | 该孔真值等效 slope。 |
| `truth_phase_local_rms_rad` | 该孔局部 phase RMS。 |
| `low_signal` | 是否低信号孔位。 |
| `trust_code` | trust 数值编码。 |
| `trust_label` | trust 文本标签：`low_signal` / `suspicious` / `review` / `trusted`。 |

## 8. `analysis/per_frame_displacement/`

### 8.1 逐帧位移主图

| 文件 | 数据含义 |
|------|----------|
| `displacement_frame_000.png` ~ `displacement_frame_009.png` | 每一帧 dX / dY 的真值、恢复、误差详细图。 |
| `quiver_overview_cross_correlation.png` | 代表性帧的真值箭头与恢复箭头叠加总览。 |
| `per_frame_rmse_cross_correlation.png` | 各帧 RMSE 曲线。 |

### 8.2 `quiver_frames/`

| 文件 | 数据含义 |
|------|----------|
| `quiver_frame_000_cross_correlation.png` ~ `quiver_frame_009_cross_correlation.png` | 每一帧单独的矢量箭头图。通常蓝色是真值，红色是恢复值，用来直接看方向一致性。 |

## 9. `analysis/statistics/`

### 9.1 全局误差与逐帧统计

| 文件 | 数据含义 |
|------|----------|
| `metrics_all_pulses_cross_correlation.png` | 各帧 RMSE、相关系数等时序图。 |
| `error_statistics_cross_correlation.png` | 全局误差分布摘要图。 |
| `per_frame_error_percentiles_cross_correlation.png` | 每帧误差分位数曲线。 |
| `spot_metric_histograms_cross_correlation.png` | 每孔统计量的直方图集合。 |

#### `per_frame_metrics_cross_correlation.csv`

每帧一行。

| 字段 | 数据含义 |
|------|----------|
| `frame_index` | 帧索引。 |
| `rmse_dx_px`, `rmse_dy_px`, `rmse_total_px` | 该帧 dX / dY / 向量总 RMSE。 |
| `mean_err_px` | 该帧平均向量误差。 |
| `median_err_px` | 该帧向量误差中位数。 |
| `p90_err_px`, `p95_err_px` | 该帧向量误差分位数。 |
| `max_err_px` | 该帧最大向量误差。 |
| `bias_dx_px`, `bias_dy_px` | 该帧 dX / dY 平均偏置。 |
| `scale_vec` | 该帧向量尺度因子。 |
| `mean_cosine` | 该帧平均方向 cosine similarity。 |
| `mean_angle_deg` | 该帧平均方向误差。 |
| `corr_dx`, `corr_dy` | 该帧 dX / dY 相关系数。 |

#### `per_frame_metrics_cross_correlation.json`

这是上表的 JSON 版本，包含两部分：

| 键 | 数据含义 |
|----|----------|
| `summary` | 跨帧总摘要，包含 `n_frames`、`rmse_total_px`、`best_frame_index`、`worst_frame_index`、`median_frame_rmse_px`、`p95_frame_rmse_px`、`mean_corr_dx`、`mean_corr_dy` 等。 |
| `per_frame_metrics` | 每帧详细指标对象列表，字段与 CSV 一致。 |

### 9.2 逐样本 DPC 统计

#### `dpc_signal_vs_error_scale_cross_correlation.csv`

这是全局单列摘要表，格式为 `metric,value`。当前 `metric` 包括：

| metric | 数据含义 |
|--------|----------|
| `n_samples` | 参与统计的总样本数。 |
| `median_truth_mag_px`, `p95_truth_mag_px`, `max_truth_mag_px` | 真值位移大小分布。 |
| `median_error_mag_px`, `p95_error_mag_px`, `max_error_mag_px` | 绝对误差大小分布。 |
| `median_signal_to_error_ratio` | 信号与误差比中位数。 |
| `truth_lt_0p05_px_pct`, `truth_lt_0p10_px_pct`, `truth_lt_0p20_px_pct` | 低信号样本占比。 |
| `error_lt_0p01_px_pct`, `error_lt_0p02_px_pct`, `error_lt_0p05_px_pct` | 误差低于阈值的样本占比。 |

对应图片：

| 文件 | 数据含义 |
|------|----------|
| `dpc_signal_vs_error_scale_cross_correlation.png` | 真值位移分布、误差分布、CDF 和关键数字的总览图。 |

#### `dpc_error_vs_truth_bins_cross_correlation.csv`

按真值位移大小分箱后统计误差。

| 字段 | 数据含义 |
|------|----------|
| `bin_label` | 分箱标签。 |
| `bin_lo_px`, `bin_hi_px` | 分箱上下边界。 |
| `count` | 落入该分箱的样本数。 |
| `fraction_pct` | 该分箱占全部样本的百分比。 |
| `median_truth_mag_px` | 该分箱样本真值位移中位数。 |
| `median_abs_error_px`, `p95_abs_error_px` | 幅值绝对误差统计。 |
| `median_abs_parallel_error_px`, `p95_abs_parallel_error_px` | 平行误差统计。 |
| `median_abs_perpendicular_error_px`, `p95_abs_perpendicular_error_px` | 垂直误差统计。 |
| `median_angle_deg`, `p95_angle_deg` | 方向误差统计。 |

对应图片：

| 文件 | 数据含义 |
|------|----------|
| `dpc_error_vs_truth_bins_cross_correlation.png` | 按信号强度分箱的误差趋势图。 |

#### `dpc_angle_error_summary_cross_correlation.csv`

这是方向误差的分层统计表。

| 字段 | 数据含义 |
|------|----------|
| `scope` | 统计范围，如 `all_finite_samples`、`truth>=0.05`、`truth>=0.10`、`truth>=0.20`。 |
| `count` | 纳入该范围的样本数。 |
| `median_angle_deg`, `p95_angle_deg` | 方向误差中位数 / P95。 |
| `fraction_lt_5_deg_pct`, `fraction_lt_10_deg_pct`, `fraction_lt_15_deg_pct` | 方向误差低于阈值的占比。 |

对应图片：

| 文件 | 数据含义 |
|------|----------|
| `dpc_angle_error_summary_cross_correlation.png` | 方向误差分布、CDF 和分层摘要图。 |

#### `dpc_pointwise_direction_and_magnitude_error_cross_correlation.csv`

这是最细的逐样本误差表。每一帧、每一孔都一行。

| 字段 | 数据含义 |
|------|----------|
| `frame` | 帧索引。 |
| `row`, `col` | spot 网格位置。 |
| `truth_dx_px`, `truth_dy_px` | 真值位移分量。 |
| `recovered_dx_px`, `recovered_dy_px` | 恢复位移分量。 |
| `truth_mag_px`, `recovered_mag_px` | 真值 / 恢复位移模长。 |
| `magnitude_error_px` | 带符号的模长误差，`recovered_mag - truth_mag`。 |
| `abs_magnitude_error_px` | 模长绝对误差。 |
| `direction_error_deg` | 带符号方向误差。 |
| `abs_direction_error_deg` | 方向绝对误差。 |

对应图片：

| 文件 | 数据含义 |
|------|----------|
| `dpc_pointwise_direction_and_magnitude_error_cross_correlation.png` | 逐样本方向误差和幅值误差的每孔聚合图。 |

### 9.3 当前新增的 trust / 物理解释图

| 文件 | 数据含义 |
|------|----------|
| `dpc_spot_trust_scatter_cross_correlation.png` | 左图：每孔真值位移 vs 幅值误差。右图：每孔真值位移 vs 方向误差。按 trust 类别着色。 |
| `dpc_truth_wavefront_context_cross_correlation.png` | 同时展示 `true displacement`、`equivalent slope`、`local phase RMS` 的空间分布和散点关系。它用来解释低信号孔位更接近“低 slope”，而不是简单“低 phase 幅值”。 |

## 10. 当前输出里的命名规律

理解命名规律以后，读目录会快很多。

### 10.1 后缀规律

- `_stack`：逐帧逐孔三维数组，通常 shape 是 `(10, 26, 26)`。
- `_per_spot`：每孔汇总后的二维数组，shape 通常是 `(26, 26)`。
- `_px` / `_m` / `_deg` / `_mrad`：单位。
- `_cross_correlation`：当前恢复方法标签。

### 10.2 常见前缀规律

- `truth_`：真值。
- `recovered_`：恢复值。
- `error_`：恢复减真值后的误差。
- `bias_`：跨帧平均偏置。
- `rmse_`：均方根误差。
- `scale_`：恢复值相对真值的尺度因子。
- `confidence`：局部互相关置信度。
- `condition_number` / `residual_ratio`：数值稳定性指标。

## 11. 当前不再默认输出的旧文件

下面这些内容**不是当前主线默认输出**，所以本指南不再把它们当成标准结果：

- `phase_recovered_*.npy`
- `all_phase_*.npy`
- `recovery_*.png`
- `per_frame_phase/`
- `decomposition_*.json`
- `decomposition_images_*.npy`
- `eval_baseline_*.json`
- Step 3 的 `method_comparison.png`、`summary_comparison.png` 等 legacy 分解分析图

原因：

- 当前默认配置是 `OUTPUT_DPC_ONLY = True`
- 当前主目标是 DPC displacement 的可信度，不是 phase 图本身
- authoritative 结果说明不能再混入旧链路残留

## 12. 建议的理解顺序

如果你想从“原始输入”一路看懂到“可信度结论”，建议按这个顺序：

1. `simulation_params.json`
2. `accumulated.npy`
3. `gt_displacements.npz`
4. `analysis/all_shifts_cross_correlation.npy`
5. `analysis/dpc_spot_metrics_cross_correlation.csv`
6. `analysis/dpc_spot_trust_summary_cross_correlation.csv`
7. `analysis/dpc_spot_trust_maps_cross_correlation.png`
8. `analysis/statistics/dpc_signal_vs_error_scale_cross_correlation.csv`
9. `analysis/statistics/dpc_angle_error_summary_cross_correlation.csv`
10. `analysis/statistics/dpc_truth_wavefront_context_cross_correlation.png`

## 13. 当前最核心的三句话

1. 当前主线评估对象是 `true displacement / recovered displacement / recovery error`，不是 phase 图本身。
2. 低信号 spot 更接近“孔平均 slope 小”，不能简单解释成“该区域 phase / WFE 幅值小”。
3. `dpc_spot_trust_summary_cross_correlation.csv` 和 `dpc_spot_trust_maps_cross_correlation.png` 是当前最直接回答“每个孔的恢复位移是否可信”的输出。
