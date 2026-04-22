# 高重频自由电子激光（FEL）多脉冲波前探测算法全面评估协议

## 1. 协议目标

本协议用于系统评估基于 Hartmann / Shack-Hartmann 原理的 X 射线多脉冲波前探测算法，在高重频 SASE FEL 条件下的单帧精度、多脉冲复用能力、抗噪声能力、运动误差容忍度和计算实时性。

协议分为两部分：

1. `Classical WFS Metrics`
   对应传统波前传感中的位移精度、DPC 保真度、灵敏度-动态范围平衡和空间分辨率。
2. `FEL-Specific Multiplexing Metrics`
   对应高重频 FEL 中多脉冲叠加、强度抖动、散粒噪声、拖影和计算负担等特有问题。

最终输出不是单一 RMSE，而是一组围绕 DPC 检测能力的标准化性能曲线与关键极限参数：

- 最小可分辨位移 `ΔX_min`
- 最小可稳定分辨相位幅值 `WFE_min`
- 保守工作上限 `WFE_work,max`
- 扩展可用上限 `WFE_ext,max`
- 空间频率截止 `f_cutoff`
- 最大可分离脉冲数 `N_pulses,max`
- 最低可恢复光通量 `Φ_min`
- 在线反馈可行频率 `f_RT`

说明：

- 主评估对象是 `DPC / spot displacement`，而不是积分后的 phase 图。
- phase 积分结果只保留为补充可视化，不作为主通过判据。

## 2. 评估对象与推荐 X 射线基线参数

本协议基于当前项目中的 X 射线模式：

- 仿真入口：[simulate.py](/Users/yaaang/Desktop/model2/simulate.py)
- 仿真参数：[config_simulate.py](/Users/yaaang/Desktop/model2/config_simulate.py)
- 恢复入口：[recover.py](/Users/yaaang/Desktop/model2/recover.py)
- 输出说明：[OUTPUT_GUIDE.md](/Users/yaaang/Desktop/model2/OUTPUT_GUIDE.md)

根据当前已经完成的 `step / z / WFE` 扫描，后续正式评估不再沿用历史默认参数，而采用一组更贴近当前最佳工作窗口的推荐基线：

| 参数 | 符号 | 推荐值 |
|---|---|---:|
| 光子能量 | `E` | `10 keV` |
| 波长 | `λ` | `1.23984 Å` |
| 传播距离 | `z_prop_m` | `0.45 m` |
| Hartmann 孔间距 | `pitch_m` | `25 um` |
| 孔径直径 | `hole_diameter_m` | `13 um` |
| 探测器像元 | `pixel_size_m` | `0.65 um` |
| 探测器尺寸 | `Nx × Ny` | `1024 x 1024` |
| 仿真窗口 | `window_size_m` | `1000 um` |
| 网格采样点数 | `grid_points` | `4096` |
| 脉冲数 | `n_pulses` | `10` |
| 运动模式 | `movement_mode` | `translation` |
| 单步位移 | `step_m` | `8 um` |
| 方向角 | `angle_deg` | `19 deg` |
| 像差模式 | `phase_aberration_mode` | `per_pulse` |
| 主展示相位幅值 | `WFE_ref` | `1.0 rad` |

其中：

- `WFE = 1.0 rad` 用于主线展示；对 X 射线波前检测来说，这也不属于“小像差”
- `WFE = 2.0 rad` 用于扩展工作区展示，不预设为“中等难度”
- `WFE = 3.0 rad` 仅作为“高像差压力测试”展示，不再作为主汇报基线
- `WFE >= 4.0 rad` 只用于动态范围边界扫描，不作为标称工作点

当前 sweep 已给出的经验结论如下：

- `step` 最优点在 `8.0 um` 左右，推荐工作区间为 `8.0–9.5 um`
- `z` 最优点在 `450 mm`，保守推荐工作区间为 `450–550 mm`
- `600 mm` 以后尾部误差开始明显抬升，`>= 750 mm` 不再建议作为可信工作区
- 在当前算法下，`1 rad` 适合主线汇报，`2 rad` 适合作为扩展工作区检查，`3 rad` 已属于较大波前畸变条件

该组参数作为后续正式评估与汇报的默认基线，除非某个模块明确指定其他值。

## 3. 评估层级

为避免不同误差源互相掩盖，所有实验统一划分为三个层级。

### 3.1 L0：单帧 / Oracle 上界层

用于测定算法本身的理想上限，不引入多脉冲分解误差。

典型设置：

- `n_pulses = 1` 或直接使用 per-frame oracle 数据
- 无强度抖动
- 无多脉冲叠加串扰
- 按模块需要选择是否开启探测器噪声

### 3.2 L1：多脉冲叠加与已知轨迹层

用于评估 accumulated decomposition 与多脉冲恢复性能。

典型设置：

- 使用 `accumulated.npy`
- 使用已知 `movement_pulse_offsets_xy_m`
- 可存在多脉冲像差
- 不引入额外 SASE 强度抖动和额外运动学误差

### 3.3 L2：FEL 真实扰动层

用于评估接近真实 FEL 情形下的鲁棒性。

典型设置：

- 多脉冲累加
- shot-to-shot 强度抖动
- Poisson 散粒噪声
- 运动拖影 surrogate
- trajectory jitter 或轨迹标定误差

## 4. 比较算法与统一输出

### 4.1 比较算法

当前正式协议只保留一条主恢复链路作为评估对象：

| 类别 | 方法 |
|---|---|
| 主恢复 | `cross_correlation` |
| accumulated 恢复 | `accumulated_linearized_residual` |
| 后处理 | `weak spatial prior + weak temporal prior` |

说明：

- 这是当前已经验证为最稳定的一条 DPC 恢复链路。
- 历史上的 `centroid`、`optimization`、`phase_correlation` 不再作为主协议必跑分支；若后续需要，可在附录中做补充对比。

### 4.2 统一统计口径

所有带随机性的 sweep 点统一采用：

- 随机种子数：`20`
- 汇报统计量：`median`、`IQR`、`mean ± std`
- 同时汇报 `failure rate`

### 4.3 统一失败判据

对于每个 sweep 点，统一记录下列失败事件：

- spot 检测失败
- 网格匹配失败
- 分解失败
- DPC 求解出现 `NaN` / `Inf`
- 指标超阈值导致判定为不可用

若某项实验需要定义“崩溃点”，统一采用：

- `failure rate > 5%`
  或
- `dpc_frame_p95_rmse_px > 0.10 px`
  或
- `dpc_spot_p95_rmse_px > 0.15 px`
  或
- `|dpc_spot_median_scale_vec - 1| > 0.05`

满足以上任一条件即可记为该 sweep 点失效。

### 4.4 统一输出字段

每个 sweep 点最终应输出一条标准记录，字段如下：

- `module_id`
- `scene_level`
- `method`
- `seed`
- `dpc_rmse_total_px`
- `dpc_frame_p95_rmse_px`
- `dpc_spot_median_rmse_px`
- `dpc_spot_p95_rmse_px`
- `dpc_spot_median_mean_relative_error_pct`
- `dpc_spot_median_scale_vec`
- `dpc_spot_median_angle_deg`
- `dpc_mean_corr_dx`
- `dpc_mean_corr_dy`
- `fail_flag`
- `runtime_s`
- `peak_memory_mb`

每个模块结束后，再对该模块所有 seed 做聚合，输出：

- `median`
- `iqr`
- `mean`
- `std`
- `failure_rate`

## 5. 基线实验

在进入 9 个评估模块之前，先运行一组完整基线实验，作为后续所有 sweep 的参考目录。

### 5.1 基线实验配置

- 使用上文推荐评估基线，而不是历史默认参数
- 主展示 case 采用：
  `step = 8 um`, `z = 450 mm`, `WFE = 1.0 rad`
- 同时补充两组对照：
  `WFE = 2.0 rad`（扩展工作区）
  `WFE = 3.0 rad`（高像差压力测试）
- 输出目录固定为：
  `result/xray/eval_baseline_001`

### 5.2 基线实验执行命令

```bash
python simulate.py --output-dir result/xray/eval_baseline_001
python recover.py result/xray/eval_baseline_001 --method cross_correlation
```

### 5.3 基线实验输出物

基线实验必须至少生成：

- `accumulated.npy`
- `ideal_accumulated.npy`
- `reference_ideal.npy`
- `gt_displacements.npz`
- `simulation_params.json`
- `analysis/metrics_cross_correlation.json`
- `analysis/dpc_spot_metrics_cross_correlation.csv`
- `analysis/dpc_spot_summary_cross_correlation.png`
- `analysis/statistics/per_frame_metrics_cross_correlation.csv`
- `analysis/statistics/per_frame_error_percentiles_cross_correlation.png`
- `analysis/statistics/spot_metric_histograms_cross_correlation.png`
- `analysis/per_frame_displacement/quiver_frames/*.png`

基线实验的作用是：

1. 固化一组后续 sweep 所有曲线的参照点。
2. 验证当前 DPC 主链路在推荐工作点下稳定可用。
3. 提供实际图像、GT 位移场、逐帧 quiver 与 per-spot 误差样例。

## 6. 九大评估模块

## 6.1 模块 1：算法位移求解精度

### 6.1.1 评估目的

评估位移求解算法本身的极限精度，测定系统能够稳定分辨的最小亚像素位移。

### 6.1.2 实验层级

`L0`

### 6.1.3 实验设置

- `n_pulses = 2`
- `phase_aberration_mode = none`
- `movement_mode = translation`
- 固定其余硬件参数与当前 X 射线基线一致
- 对第二帧施加精确平移

平移采用像素单位给定，再换算到物理位移：

`step_m = shift_px * pixel_size_m`

### 6.1.4 Sweep 范围

- `shift_px ∈ [0.00, 1.00]`
- 采样步长：`0.02 px`

### 6.1.5 评价指标

- `RMSE_shift_px`
- `mean_bias_px`
- `max_abs_error_px`
- `pixel_locking_amplitude_px`

其中：

- `ΔX_min` 定义为最小非零位移，使得
  `median_abs_error <= 0.1 * shift_in`
  且 `failure_rate <= 5%`
- `pixel_locking_amplitude` 定义为误差对亚像素相位分箱后的峰-峰值

### 6.1.6 输出结果

- 位移真值 vs 恢复值散点图
- 误差随亚像素相位变化曲线
- Pixel-locking 频谱图
- `ΔX_min` 表

## 6.2 模块 2：DPC 场保真度与局部斜率响应

### 6.2.1 评估目的

评估在受控波前输入下，恢复得到的局部 DPC / spot 位移场是否保真，重点关注每个 spot 的幅值、方向、scale 与尾部误差，而不是积分 phase 的视觉效果。

### 6.2.2 实验层级

`L0`

### 6.2.3 实验设置

- `n_pulses = 1`
- `movement_mode = none`
- 其余参数固定为推荐基线
- 输入控制型 Zernike 相位畸变
- 直接比较 GT DPC 与恢复 DPC

### 6.2.4 Sweep 设计

采用三组测试：

1. 单模态测试：
   逐个激活前 15 个可重建 Zernike 模式
2. 双模态串扰测试：
   对前 15 个模式做两两组合
3. 随机组合测试：
   在前 15 个模式内随机生成线性组合

每个模式振幅统一扫描：

- `a ∈ {0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0} rad`

### 6.2.5 评价指标

- `dpc_rmse_total_px`
- `dpc_spot_p95_rmse_px`
- `dpc_spot_median_mean_relative_error_pct`
- `dpc_spot_median_scale_vec`
- `dpc_spot_median_angle_deg`
- 单阶响应斜率
- 串扰矩阵 `C_ij`

定义：

- `C_ij = recovered_response_j / input_response_i`
- 若需要输出 phase，则只作为补充图，不参与模块通过判据

### 6.2.6 输出结果

- 每阶单模态 DPC quiver 图
- `scale / angle / p95 RMSE` 随振幅变化曲线
- DPC 残差热图
- 串扰矩阵热图

## 6.3 模块 3：灵敏度与动态范围的物理平衡

### 6.3.1 评估目的

定量验证 Hartmann 传感器中灵敏度与动态范围由传播距离 `z` 强耦合的物理规律，并给出以 DPC 误差为准的工作区与扩展区。

### 6.3.2 实验层级

`L0`

### 6.3.3 实验设置

- `n_pulses = 1`
- `movement_mode = none`
- 使用控制型波前畸变输入
- 主判据仍为 DPC，不以积分 phase 为准

### 6.3.4 Sweep 范围

#### A. 输入波前幅值

- 输入波前 RMS：
  `RMS_in ∈ {0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0} rad`

#### B. 传播距离

- `z_prop_m ∈ {0.45, 0.50, 0.55, 0.60, 0.70, 0.85, 0.95} m`

### 6.3.5 评价指标

- `WFE_min(z)`
- `WFE_work,max(z)`
- `WFE_ext,max(z)`
- `dynamic_range(z)`
- `collapse_probability(z, RMS_in)`

定义：

- `WFE_min(z)` 为最小输入 RMS，使得恢复 DPC 幅值稳定高于零输入噪声底，且
  `dpc_spot_median_angle_deg <= 10 deg`
- `WFE_work,max(z)` 为满足
  `dpc_frame_p95_rmse_px <= 0.05 px`
  且 `|dpc_spot_median_scale_vec - 1| <= 0.05`
  的最大输入 RMS
- `WFE_ext,max(z)` 为满足
  `dpc_frame_p95_rmse_px <= 0.10 px`
  且 `|dpc_spot_median_scale_vec - 1| <= 0.10`
  的最大输入 RMS
- `dynamic_range(z) = WFE_work,max(z) / WFE_min(z)`

### 6.3.6 输出结果

- `WFE_min / WFE_work,max / WFE_ext,max` 随 `z` 的变化曲线
- 动态范围 vs `z` 曲线
- 崩溃相图 `z × RMS_in`
- `frame p95` 与 `spot p95` 的相图

## 6.4 模块 4：空间分辨率限制

### 6.4.1 评估目的

评估系统在子孔径采样限制下能够恢复的最大波前空间频率，给出 DPC 传递函数和 aliasing 截止频率。

### 6.4.2 实验层级

`L0`

### 6.4.3 实验设置

- `n_pulses = 1`
- `movement_mode = none`
- 输入纯正弦相位光栅

相位形式：

`phi(x, y) = A * sin(2π f x + phi0)`

### 6.4.4 Sweep 范围

- 频率 `f / f_Nyq ∈ [0.05, 0.95]`
- 振幅 `A ∈ {0.1, 0.5, 1.0, 2.0} rad`
- 方向：`x`、`y`、`45 deg`

### 6.4.5 评价指标

- `MTF_dpc(f) = A_rec / A_in`
- `dpc_rmse_total_px(f)`
- 混叠起始频率 `f_alias`
- 截止频率 `f_cutoff`

其中：

- `f_cutoff` 定义为 `MTF_dpc < 0.5` 的最低频率
- `f_alias` 定义为误差出现明显折返或相位符号错判的频率

### 6.4.6 输出结果

- DPC MTF 曲线
- 不同方向的 MTF 对比图
- aliasing 热图

## 6.5 模块 5：多脉冲混叠复用极限与串扰

### 6.5.1 评估目的

评估在高重频多脉冲叠加条件下，算法对单个 pulse 的图像分离与 DPC 恢复能力，量化最大可分离脉冲数与串扰矩阵。

### 6.5.2 实验层级

`L1`

### 6.5.3 实验设置

- 使用 `accumulated` 模式恢复
- 采用 translation 轨迹
- 每个 pulse 有独立像差
- 其余硬件参数固定为基线

### 6.5.4 Sweep 范围

- `n_pulses ∈ {2, 5, 10, 15, 20, 30}`

### 6.5.5 评价指标

- `SSIM`：分解图像与 oracle 单帧图像的结构相似度
- `per_pulse dpc_rmse_total_px`
- `per_pulse dpc_spot_p95_rmse_px`
- `per_pulse dpc_spot_median_scale_vec`
- `energy_leakage_ratio`
- `crosstalk matrix`

串扰实验定义：

仅第 `i` 个 pulse 注入大幅畸变，其他 pulse 保持无像差，则

`CT(i -> j) = ||u_rec_j|| / ||u_rec_i||, j != i`

### 6.5.6 关键性能量

- `N_pulses,max`

定义为满足以下条件的最大脉冲数：

- `median per_pulse dpc_rmse_total_px <= 0.05 px`
- `median per_pulse dpc_spot_p95_rmse_px <= 0.10 px`
- `|median per_pulse dpc_spot_median_scale_vec - 1| <= 0.05`
- `median SSIM >= 0.9`
- `max crosstalk <= 10%`
- `failure_rate <= 5%`

### 6.5.7 输出结果

- `N_pulses` vs `per_pulse dpc_rmse_total_px`
- `N_pulses` vs `SSIM`
- 串扰矩阵热图
- `N_pulses,max` 汇总表

## 6.6 模块 6：SASE 强度剧烈波动容忍度

### 6.6.1 评估目的

评估强烈 shot-to-shot 能量起伏下，分解与恢复算法对弱脉冲 DPC 的保持能力，以及在亮脉冲主导背景下的误判情况。

### 6.6.2 实验层级

`L2`

### 6.6.3 实验设置

- 固定 `n_pulses = 10`
- 每个 pulse 乘以独立随机强度权重
- 强度权重服从 Gamma 分布
- 分布均值固定为 `1`

### 6.6.4 Sweep 范围

- 强度抖动强度：
  `sigma_jitter ∈ {0%, 50%, 95%}`

### 6.6.5 评价指标

- 最暗 `5%` pulse 的 `dpc_rmse_total_px`
- 最暗 `5%` pulse 的 `dpc_spot_p95_rmse_px`
- 弱脉冲漏检率 `miss_rate_dark`
- 强脉冲主导造成的 `false_leakage`

### 6.6.6 输出结果

- `sigma_jitter` vs darkest-pulse DPC RMSE
- darkest-pulse 生存率曲线
- pulse 强度 vs 恢复误差散点图

## 6.7 模块 7：散粒本底噪声极限

### 6.7.1 评估目的

评估在低光子计数条件下，Poisson 散粒噪声对 DPC 求解的影响，确定最低可用通量。

### 6.7.2 实验层级

`L2`

### 6.7.3 实验设置

- `ENABLE_DETECTOR_EFFECTS = True`
- 保持基线几何参数不变
- 对系统通量做对数 sweep

### 6.7.4 Sweep 范围

- 每孔平均光子数：
  `10^2, 10^2.5, 10^3, 10^3.5, 10^4, 10^4.5, 10^5`

### 6.7.5 比较算法

- `cross_correlation` 主链路

### 6.7.6 评价指标

- `dpc_rmse_total_px`
- `dpc_frame_p95_rmse_px`
- `dpc_spot_p95_rmse_px`
- `failure_rate`
- `Φ_min`

其中 `Φ_min` 定义为满足以下条件的最低通量：

- `median dpc_spot_p95_rmse_px <= 0.10 px`
- `|median dpc_spot_median_scale_vec - 1| <= 0.05`
- `failure_rate <= 5%`

### 6.7.7 输出结果

- 通量 vs `dpc_rmse_total_px`
- 通量 vs `dpc_spot_p95_rmse_px`
- 通量 vs `failure_rate`
- `Φ_min` 汇总表

## 6.8 模块 8：连续 / 离散模糊拖影畸变

### 6.8.1 评估目的

评估非理想曝光条件下，沿运动方向的拖尾模糊对位移估计和 DPC 恢复带来的方向性误差。

### 6.8.2 实验层级

`L2`

### 6.8.3 实验设置

- 固定多脉冲 translation 轨迹
- 在探测器图像上沿运动方向施加各向异性拖尾 surrogate
- 同时保留无拖尾基线作对照

### 6.8.4 Sweep 范围

- 拖尾长度：
  `L_blur ∈ {0, 0.5, 1.0, 2.0, 3.0} px`

### 6.8.5 评价指标

- `RMSE_x`
- `RMSE_y`
- `RMSE_tangential`
- `RMSE_radial`
- 方向性 bias

### 6.8.6 输出结果

- 拖尾长度 vs `RMSE_x`, `RMSE_y`
- 切向 / 径向误差分离图
- 拖尾方向误差矢量图

## 6.9 模块 9：算法时效性与计算瓶颈

### 6.9.1 评估目的

评估算法在孔数和脉冲数增加时的运行时间、内存占用和在线反馈可行性。

### 6.9.2 实验层级

`L1`

### 6.9.3 实验设置

- 使用固定基线光学参数
- 扫孔数与脉冲数
- 在统一硬件环境下记录耗时与内存

### 6.9.4 Sweep 范围

#### A. 网格规模

- `N_grid ∈ {10x10, 14x14, 20x20, 28x28}`

#### B. 脉冲数

- `N_pulses ∈ {2, 5, 10, 20, 30}`

### 6.9.5 评价指标

- `runtime_s`
- `peak_memory_mb`
- 经验复杂度阶数
- 可实现反馈频率 `f_RT = 1 / runtime_s`

### 6.9.6 输出结果

- `runtime` 随 `N_grid` 变化曲线
- `runtime` 随 `N_pulses` 变化曲线
- `log-log` 拟合复杂度图
- `1 Hz / 10 Hz / 100 Hz` 实时区间判定图

## 7. 实验执行顺序

为了保证结果可归因，统一按照以下顺序执行：

1. `Baseline`
   跑一组推荐工作点与压力测试结果
2. `Classical WFS Suite`
   依次执行模块 `1 -> 2 -> 3 -> 4`
3. `FEL Multiplexing Suite`
   依次执行模块 `5 -> 6 -> 7 -> 8`
4. `Computational Suite`
   最后执行模块 `9`

该顺序保证：

- 先得到单帧理论上界
- 再引入多脉冲分解
- 再叠加 FEL 扰动
- 最后评估工程可部署性

## 8. 最终汇报格式

最终实验章节统一输出以下内容。

### 8.1 主表

主表至少包含：

| 模块 | 主指标 | 极限参数 | 最佳方法 | 备注 |
|---|---|---|---|---|
| 1 | `RMSE_shift_px` | `ΔX_min` | 主链路 | |
| 2 | `dpc_spot_p95_rmse_px`, `relative_error_pct`, `scale_vec` | - | 主链路 | |
| 3 | `dynamic_range` | `WFE_min`, `WFE_work,max`, `WFE_ext,max` | 主链路 | |
| 4 | `MTF_dpc` | `f_cutoff` | 主链路 | |
| 5 | `per_pulse DPC RMSE`, `SSIM` | `N_pulses,max` | 主链路 | |
| 6 | darkest-pulse DPC RMSE | - | 主链路 | |
| 7 | `dpc_spot_p95_rmse_px` | `Φ_min` | 主链路 | |
| 8 | `RMSE_tangential`, `RMSE_radial` | `L_blur,max` | 主链路 | |
| 9 | `runtime_s`, `peak_memory_mb` | `f_RT` | 主链路 | |

### 8.2 标准图组

最终报告至少包含以下图组：

- 位移精度与 pixel-locking 图
- 单模态 / 双模态 DPC quiver 图
- 灵敏度-动态范围相图
- DPC MTF 曲线
- 多脉冲容量与串扰热图
- 强度 jitter 鲁棒性曲线
- 通量极限曲线
- 拖尾方向误差分离图
- 运行时间复杂度曲线

## 9. 协议结论形式

本协议执行完成后，结论部分统一回答以下 9 个问题：

1. 最小可稳定分辨的亚像素位移是多少？
2. 在低阶与组合像差输入下，每个 spot 的 DPC 保真度如何？
3. 传播距离变化如何改变 `WFE_min`、保守工作上限与扩展可用上限？
4. 系统的空间频率截止在哪里？
5. 在当前硬件参数下最多能稳定分离多少个累加脉冲？
6. 对 SASE 强度剧烈波动是否具有弱脉冲保持能力？
7. 在 X 射线低光子通量下的最低可用通量是多少？
8. 沿运动方向的拖尾会造成多强的方向性误差？
9. 当前算法能否达到 1 Hz、10 Hz 或 100 Hz 的在线反馈要求？

只要这 9 个问题全部能给出量化答案，本评估协议即完成闭环。
