# X 射线模式多脉冲波前探测实验报告

> 说明：本报告主要整理的是旧 baseline `run_002` / `sweeps_20260407_002`。
> 当前 authoritative 口径已经切换到 [AI_HANDOFF_README.md](/Users/yaaang/Desktop/model2/AI_HANDOFF_README.md) 中的主线 case：
> `step = 8 um`, `z = 450 mm`, `WFE = 1 rad`。
> 因此本报告中的 error 数字不应直接与 `pointwise_demo` 或当前主线 sweep 结果混比。

## 1. 实验目的

本实验面向高重频自由电子激光（FEL）下的 X 射线 Hartmann 波前探测场景，目标分为两部分：

1. 在固定物理参数下验证当前多脉冲叠加恢复流程是否能够稳定恢复 10 脉冲累积图像中的单脉冲波前。
2. 对影响系统性能的三类主参数进行系统扫描：
   `STEP_M`、`Z_PROP_M`、输入波前幅值 `WFE`。

本轮实验不再扩展空间频率扫描，空间频率上限直接由当前 Hartmann 采样间距决定。

## 2. 实验流程与数据路径

实验流程如下：

1. 使用 [simulate.py](/Users/yaaang/Desktop/model2/simulate.py) 生成 X 射线模式多脉冲累积探测图像。
2. 使用 [recover.py](/Users/yaaang/Desktop/model2/recover.py) 对累积图像执行单脉冲分解与波前恢复。
3. 使用 [run_xray_protocol_sweeps.py](/Users/yaaang/Desktop/model2/run_xray_protocol_sweeps.py) 批量运行三组扫参。
4. 使用 [plot_xray_protocol_sweeps.py](/Users/yaaang/Desktop/model2/plot_xray_protocol_sweeps.py) 自动读取汇总 CSV 并生成总览图。

本轮批量实验结果根目录：

- [sweeps_20260407_002](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002)

基线单组结果目录：

- [run_002](/Users/yaaang/Desktop/model2/result/xray/run_002)

## 3. 基线实验配置

本报告将 [simulation_params.json](/Users/yaaang/Desktop/model2/result/xray/run_002/simulation_params.json) 作为固定配置基线。

### 3.1 仿真参数

| 参数 | 数值 |
|---|---:|
| 模式 | `full` |
| 光子能量 | `10 keV` |
| 波长 | `1.23984e-10 m` |
| 传播距离 `z` | `0.45 m` |
| 探测器像元 | `0.65 um` |
| Hartmann pitch | `25 um` |
| 孔径直径 | `13 um` |
| 菲涅耳数 | `0.7573` |
| 脉冲数 | `10` |
| 探测器尺寸 | `1024 x 1024` |
| 视场窗口 | `1.0 mm` |
| 采样网格 | `4096 x 4096` |
| 光源 | `plane` |
| 掩膜 | `hartmann` |
| 传播模型 | `fresnel` |
| 运动模式 | `translation` |
| 运动角度 | `19 deg` |
| 脉冲步长 | `7 um` |
| 总净位移 | `63 um` |
| 子孔径网格 | `26 x 26` |

### 3.2 恢复参数

当前恢复流程来自 [config_recover.py](/Users/yaaang/Desktop/model2/config_recover.py)。

| 参数 | 数值 |
|---|---:|
| 主恢复方法 | `cross_correlation` |
| `XCORR_PATCH_HALF` | `13` |
| `XCORR_CORRECTION_FACTOR` | `1.0` |
| 累积分解方法 | `iterative` |
| 参考序列模式 | `per_pulse` |

## 4. 基线单组实验结果

### 4.1 单组结果总览

基线 `run_002` 对应的是 10 脉冲平移累积情形。当前流程在该配置下已经可以稳定完成：

1. 累积图像分解。
2. 单脉冲位移恢复。
3. 单脉冲相位重建。
4. 多方法误差比较。

基线主图如下：

- 恢复总览图：[recovery_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/recovery_cross_correlation.png)
- 全帧相位总览：[all_frames_phase_summary_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/all_frames_phase_summary_cross_correlation.png)
- 全帧位移总览：[all_frames_disp_summary_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/all_frames_disp_summary_cross_correlation.png)
- 逐帧 RMSE 曲线：[per_frame_rmse_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/per_frame_displacement/per_frame_rmse_cross_correlation.png)
- 逐帧相位总览：[phase_all_frames_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/per_frame_phase/phase_all_frames_cross_correlation.png)
- 分解图像总览：[decomposition_images_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/decomposition_images_cross_correlation.png)
- 单脉冲分解示例：[frame_000_summary.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/step3_decomposition/frame_000_summary.png)
- 方法对比图：[method_comparison.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/step3_decomposition/method_comparison.png)
- 误差统计图：[error_statistics.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/step3_decomposition/error_statistics.png)
- 轨迹占据图：[trajectory_occupancy.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/step3_decomposition/trajectory_design/trajectory_occupancy.png)
- 轨迹路径图：[trajectory_path.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/step3_decomposition/trajectory_design/trajectory_path.png)

### 4.2 基线数值指标

从 [eval_baseline_cross_correlation.json](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/eval_baseline_cross_correlation.json) 与 [metrics_summary.json](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/step3_decomposition/metrics_summary.json) 得到：

#### Oracle 逐帧位移基线

| 指标 | 数值 |
|---|---:|
| `RMSE dX` | `0.03345 px` |
| `RMSE dY` | `0.03268 px` |
| `RMSE total` | `0.04677 px` |
| `Mean error` | `0.03993 px` |
| `Max error` | `0.20310 px` |
| `Mean Corr dX` | `0.9613` |
| `Mean Corr dY` | `0.9727` |

#### Step 3 分解后平均指标

| 方法 | 平均位移 RMSE (px) | 平均相位 RMSE (rad) | 平均相位相关性 | 平均空间误差 (%) |
|---|---:|---:|---:|---:|
| `centroid` | `56.9241` | `60.1476` | `0.5275` | `1116758.38` |
| `cross_correlation` | `0.0441` | `0.1111` | `0.9937` | `13.40` |
| `phase_correlation` | `0.1214` | `0.9730` | `0.6965` | `99.90` |

结论很明确：当前配置下 `cross_correlation` 是唯一稳定可用的主恢复方法。

### 4.3 分解与轨迹可分性

从 [decomposition_cross_correlation.json](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/decomposition_cross_correlation.json) 和 [trajectory_design_summary.json](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/step3_decomposition/trajectory_design/trajectory_design_summary.json) 得到：

| 指标 | 数值 |
|---|---:|
| 分解方法 | `iterative` |
| 平均分解置信度 | `0.9978` |
| `spot_sigma_px` | `4.8077 px` |
| 平均跨脉冲步长 | `10.7692 px` |
| 总可见性 | `0.9359` |
| 轨迹设计状态 | `warning` |
| 可辨识性分数 | `0.5600` |

这说明：

1. 当前 10 脉冲分解整体是稳定的。
2. 轨迹仍处于可恢复但不宽松的工作区间。
3. 后续若继续增加 `z` 或继续提高累积密度，重叠风险会明显抬升。

## 5. 基线分辨率与物理极限

以基线 `run_002` 的 `cross_correlation` 平均位移 RMSE `0.0440847 px` 作为系统实际位移分辨下限，得到：

### 5.1 分辨率

| 指标 | 计算结果 |
|---|---:|
| 位移分辨率 `ΔX_min` | `28.66 nm` |
| 角分辨率 `θ_min` | `0.06368 urad` |
| 最小可测相位梯度 | `3.23e3 rad/m` |
| 空间分辨率 | `25 um` |
| 保守位移分辨率 `3σ` | `85.97 nm` |
| 保守角分辨率 `3σ` | `0.1910 urad` |

### 5.2 动态范围与空间频率上限

| 指标 | 数值 |
|---|---:|
| 最大角动态范围 `θ_max ≈ pitch / (2z)` | `27.78 urad` |
| 动态范围估计 `θ_max / θ_min` | `436 : 1` |
| 最高可测空间频率 `f_max ≈ 1/(2 pitch)` | `2.0e4 m^-1` |
| 等效最小可测周期 `Λ_min ≈ 2 pitch` | `50 um` |

因此，本轮实验已经可以给出明确硬件上限：

1. 当前系统的空间采样分辨率由 `25 um` 的 Hartmann pitch 决定。
2. 当前系统的理论最高可测空间频率约为 `2.0e4 m^-1`。
3. 对应的最小可分辨波前周期约为 `50 um`。

## 6. 三组扫参设计

本轮只扫描三类主参数，脉冲数始终固定为 `10`。

### 6.1 扫描矩阵

| 分组 | 扫描范围 | 固定条件 |
|---|---|---|
| `STEP_M` | `7.0 -> 10.0 um`，步长 `0.5 um` | `z=450 mm`, `WFE=3 rad`, `10 pulses` |
| `Z_PROP_M` | `450 -> 1000 mm`，步长 `50 mm` | `step=7 um`, `WFE=3 rad`, `10 pulses` |
| `WFE` | `1 -> 5 rad` | `step=7 um`, `z=450 mm`, `10 pulses` |

其中：

1. `STEP_M` 主要反映多脉冲轨迹分离和位移恢复负载变化。
2. `Z_PROP_M` 直接控制灵敏度和动态范围的物理平衡。
3. `WFE` 反映算法在像差增强时的鲁棒性。

### 6.2 批量结果文件

- 汇总目录：[summaries](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/summaries)
- 扫描总览图：[sweep_overview.png](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/plots/sweep_overview.png)
- 自动摘要：[sweep_summary.md](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/plots/sweep_summary.md)

## 7. 扫参结果

### 7.1 STEP 扫描

图像：

- [step_sweep_metrics.png](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/plots/step_sweep_metrics.png)
- 原始汇总表：[step_summary.csv](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/summaries/step_summary.csv)

主要结论：

1. 最小恢复位移误差出现在 `STEP_M = 7.0 um`，对应 `0.0647 px`。
2. 最小恢复相位误差出现在 `STEP_M = 8.0 um`，对应 `0.1389 rad`。
3. 当 `STEP_M >= 9.5 um` 后，相位 RMSE 和空间误差明显上升。
4. `STEP_M = 10.0 um` 时，相位相关性已下降到 `0.8927`，表明当前轨迹间距继续拉大后恢复稳定性开始恶化。

该组扫描说明：在当前 `z=450 mm`、`WFE=3 rad`、10 脉冲条件下，`7.0-8.0 um` 是更稳妥的工作区间。

### 7.2 Z 扫描

图像：

- [z_sweep_metrics.png](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/plots/z_sweep_metrics.png)
- 原始汇总表：[z_summary.csv](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/summaries/z_summary.csv)

主要结论：

1. 最小恢复位移误差出现在 `Z_PROP_M = 500 mm`，对应 `0.0537 px`。
2. 最小相位误差和最高相位相关性都出现在 `Z_PROP_M = 450 mm`，分别为 `0.1621 rad` 和 `0.9926`。
3. 当 `z > 700 mm` 后，位移 RMSE、相位 RMSE、空间误差均明显恶化。
4. 当 `z` 增大到 `1000 mm` 时，平均位移 RMSE 已增至 `0.2942 px`，空间误差升至 `88.43%`。

该组扫描说明：当前系统在 `450-500 mm` 附近处于最优工作区间。若优先考虑整体相位恢复质量，`450 mm` 更合适；若优先考虑位移 RMSE，`500 mm` 略优。

### 7.3 WFE 扫描

图像：

- [wfe_sweep_metrics.png](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/plots/wfe_sweep_metrics.png)
- 原始汇总表：[wfe_summary.csv](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/summaries/wfe_summary.csv)

主要结论：

1. `WFE = 1 rad` 时性能最好，平均位移 RMSE `0.0250 px`，平均相位 RMSE `0.0582 rad`。
2. 随着 `WFE` 增大，位移误差整体上升，说明强像差会直接推高重建难度。
3. 到 `WFE = 5 rad` 时，平均位移 RMSE 上升到 `0.1030 px`，平均相位 RMSE 上升到 `0.3173 rad`。
4. 在 `1-5 rad` 范围内，`cross_correlation` 仍保持较高相位相关性，整体仍可工作。

该组扫描说明：当前流程对 `1-5 rad` 波前像差具有较好的稳健性，但最佳工作区仍位于低像差侧。

## 8. 综合结论

本轮 X 射线模式实验已经形成一套完整、可复现实验结果，结论如下：

1. 基线单组配置 `z=450 mm`、`step=7 um`、`10 pulses` 下，当前流程可以稳定完成 10 脉冲累积图像的单脉冲波前恢复。
2. 在所有恢复方法中，`cross_correlation` 明显优于 `centroid` 和 `phase_correlation`，应作为主方法。
3. 当前系统在基线配置下的实际位移分辨率约为 `28.66 nm`，角分辨率约为 `0.06368 urad`，动态范围约为 `436:1`。
4. 当前硬件采样上限对应最高空间频率约 `2.0e4 m^-1`，等效最小周期约 `50 um`。
5. 扫参结果表明：
   `STEP_M` 的稳定区间为 `7.0-8.0 um`；
   `Z_PROP_M` 的优选区间为 `450-500 mm`；
   `WFE` 越小越有利于恢复精度。

因此，若作为后续 X 射线模式标准工作点，本轮实验最推荐的参数组合为：

| 参数 | 推荐值 |
|---|---:|
| `N_PULSES` | `10` |
| `STEP_M` | `7.0 um` |
| `Z_PROP_M` | `450 mm` |
| `WFE` | `3.0 rad` 作为中等像差基线；`1.0 rad` 作为性能上限参考 |
| 恢复方法 | `cross_correlation` |
| 分解方法 | `iterative` |

## 9. 报告附件索引

### 9.1 单组基线图

- [recovery_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/recovery_cross_correlation.png)
- [all_frames_phase_summary_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/all_frames_phase_summary_cross_correlation.png)
- [all_frames_disp_summary_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/all_frames_disp_summary_cross_correlation.png)
- [per_frame_rmse_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/per_frame_displacement/per_frame_rmse_cross_correlation.png)
- [phase_all_frames_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/per_frame_phase/phase_all_frames_cross_correlation.png)
- [method_comparison.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/step3_decomposition/method_comparison.png)
- [error_statistics.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/step3_decomposition/error_statistics.png)
- [trajectory_occupancy.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/step3_decomposition/trajectory_design/trajectory_occupancy.png)

### 9.2 扫参图

- [sweep_overview.png](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/plots/sweep_overview.png)
- [step_sweep_metrics.png](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/plots/step_sweep_metrics.png)
- [z_sweep_metrics.png](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/plots/z_sweep_metrics.png)
- [wfe_sweep_metrics.png](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/plots/wfe_sweep_metrics.png)

### 9.3 原始汇总表

- [step_summary.csv](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/summaries/step_summary.csv)
- [z_summary.csv](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/summaries/z_summary.csv)
- [wfe_summary.csv](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/summaries/wfe_summary.csv)
