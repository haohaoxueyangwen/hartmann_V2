# AI Handoff README

本文档用于在新窗口中快速接续当前项目，避免重复讨论已经作废的口径。

## 1. 项目当前的真正目标

这个项目当前**不是**在主打“积分后的 phase 恢复图看起来像不像”，而是在验证：

1. 是否能从**一张 accumulated X-ray Hartmann 图像**中恢复出 `pulse-resolved DPC`
2. 恢复出来的**位移向量**是否可信
3. 方法的能力边界是什么：
   - 绝对位移误差是多少
   - 相对误差大概是多少
   - 能恢复多少个 pulse
   - 时间复用倍率是多少
   - 空间采样率 / 空间分辨率是多少

当前用户最在意的是：

- **DPC 本身是否可信**
- 小位移不能被排除在主评价之外
- 不能再用 phase RMSE 作为主结论
- 需要能拿去组会汇报的图和口径

## 2. 当前已经确定的评价口径

### 2.1 主评价对象

主评价对象已经切换为：

- `true displacement`
- `recovered displacement`
- `recovery error`

而不是：

- 积分 phase 图
- phase RMSE

### 2.2 当前主指标

当前主链路主要看：

- `median sample absolute error (px)`
- `p95 sample absolute error (px)`
- `median |parallel| error (px)`
- `median |perpendicular| error (px)`
- `median scale`
- `median angle error`

辅助指标：

- `relative error (%)`

注意：

- 相对误差现在**不是主指标**
- 只作为辅助解释用

### 2.3 已明确的判断

当前结果可以支持：

- **绝对误差是 `0.01 px` 量级**

当前结果不能支持：

- **相对精度达到 `1%`**

更准确的说法是：

- 当前主线 case 中，典型相对误差更接近 **`10%`**

所以汇报不能讲成：

- “我们已经做到 1% 的位移恢复精度”

而应该讲成：

- “我们已经证明在小位移 regime 下，绝对误差地板达到 `0.01 px` 量级，但相对精度仍大约是 `10%` 级”

## 3. 当前主线结果与推荐口径

### 3.1 主线展示工况

当前建议用于组会主线展示的 case：

- `step = 8 um`
- `z = 450 mm`
- `WFE = 1 rad`

对应目录：

- `/Users/yaaang/Desktop/result/xray/sweeps_20260415_001/wfe/wfe_1p0rad_step8p0um_z450mm`

### 3.2 主线关键结果

主线 case 当前关键数值：

- `DPC RMSE = 0.0108 px`
- `median absolute error = 0.0060 px`
- `P95 absolute error = 0.0213 px`
- `spot-level P95 absolute error = 0.0165 px`
- `median scale = 1.0076`
- `median angle error = 4.25 deg`

### 3.3 真实位移尺度

主线 case 中：

- `median true displacement = 0.0661 px`
- `P95 true displacement = 0.1286 px`
- `max true displacement = 0.4329 px`

也就是说：

- 大多数真实位移本身就在 `0.1 px` 以内

### 3.4 当前能力结论

当前最稳妥的总结是：

- 方法已经证明**可行**
- 在小位移 regime 下，绝对误差地板达到 `0.01 px` 量级
- 但当前相对精度仍然更接近 `10%`
- 所以**可以说 feasibility is demonstrated**
- 但**不能说已经达到高精度 quantitative wavefront metrology**

## 4. 时间采样 / 空间采样 / 空间分辨率

### 4.1 时间采样

当前仿真设置：

- 每次 exposure 含 `10` 个 pulses

当前方法可从一张 accumulated 图中恢复：

- `10` 帧 pulse-resolved DPC

因此当前口径是：

- **effective DPC frame-rate gain = `10x`**

注意：

- 这里是相对常规单曝光单帧 Hartmann DPC 的**时间复用增益**
- 不是具体 beamline 的绝对 Hz

### 4.2 空间采样

当前空间采样设置：

- spot grid: `26 x 26`
- 每帧 `676` 个 DPC 采样点
- 单张 accumulated 图可恢复 `6760` 个位移向量样本

### 4.3 空间分辨率

当前系统参数：

- Hartmann pitch = `25 um`
- hole diameter = `13 um`
- detector pixel = `0.65 um`
- field span ≈ `625 um x 625 um`

当前应使用的口径：

- spatial sampling density ≈ `40 samples/mm`
- Nyquist-limited smallest period ≈ `50 um`
- 当前 DPC 空间分辨率主要由 **Hartmann pitch** 决定，而不是 detector pixel size

## 5. 当前 sweep 结果与参数窗口

当前整批 X-ray sweep 目录：

- `/Users/yaaang/Desktop/result/xray/sweeps_20260415_001`

### 5.1 最佳单点

按当前新的 DPC 口径，最佳 step 点大致是：

- `step = 9.5 um`
- `z = 450 mm`
- `WFE = 1 rad`

对应大致数值：

- `median |error| = 0.00582 px`
- `P95 |error| = 0.0193 px`

### 5.2 推荐工作区间

当前推荐工作区间：

- `step = 8.0–9.5 um`
- `z = 450–550 mm`

当前趋势：

- `z` 是最敏感参数
- `WFE = 1 rad` 适合作为主线展示
- `WFE = 3 rad` 只能作为压力测试，不适合作为标称工作条件

## 6. 已完成的代码改动

### 6.1 核心算法口径

主要修改文件：

- `recover.py`
- `run_xray_protocol_sweeps.py`
- `config_recover.py`
- `xray_fel_evaluation_protocol.md`

核心变化：

1. 主评估转为 DPC 位移，不再以 phase 为主
2. 新增 per-spot / per-frame 的 DPC 误差统计
3. 新增 `parallel / perpendicular` 误差
4. 新增按真实位移分箱的误差统计
5. 新增 `signal scale vs error scale` 图
6. 新增 `angle error summary` 图
7. 主汇报相关输出命名统一为：
   - `True`
   - `Recovered`
   - `Error`
   - `simulation truth`

### 6.2 新增的关键统计图

当前主线 case 下，最重要的图有：

- `analysis/displacement_comparison_cross_correlation.png`
- `analysis/per_frame_displacement/quiver_overview_cross_correlation.png`
- `analysis/statistics/dpc_signal_vs_error_scale_cross_correlation.png`
- `analysis/statistics/dpc_error_vs_truth_bins_cross_correlation.png`
- `analysis/statistics/dpc_angle_error_summary_cross_correlation.png`
- `analysis/dpc_spot_trust_maps_cross_correlation.png`
- `analysis/statistics/dpc_spot_trust_scatter_cross_correlation.png`
- `analysis/statistics/dpc_truth_wavefront_context_cross_correlation.png`

### 6.3 新增的关键统计 CSV

- `analysis/dpc_spot_metrics_cross_correlation.csv`
- `analysis/dpc_spot_trust_summary_cross_correlation.csv`
- `analysis/statistics/dpc_error_vs_truth_bins_cross_correlation.csv`
- `analysis/statistics/dpc_signal_vs_error_scale_cross_correlation.csv`
- `analysis/statistics/dpc_angle_error_summary_cross_correlation.csv`

## 7. 新增图的含义

### 7.1 signal vs error scale 图

文件：

- `/Users/yaaang/Desktop/model2/result/xray/sweeps_20260422_001/wfe/wfe_1p0rad_step8p0um_z450mm/analysis/statistics/dpc_signal_vs_error_scale_cross_correlation.png`

作用：

- 展示大部分真实位移在什么尺度
- 展示恢复误差在什么尺度
- 展示误差和信号是否拉开量级差距

当前图上的核心数值：

- `median true displacement = 0.0645 px`
- `median error = 0.0061 px`
- `median signal / error = 10.5x`
- `81.2%` of true displacement `< 0.10 px`
- `73.6%` of error `< 0.01 px`
- `93.7%` of error `< 0.02 px`

### 7.2 angle error summary 图

文件：

- `/Users/yaaang/Desktop/model2/result/xray/sweeps_20260422_001/wfe/wfe_1p0rad_step8p0um_z450mm/analysis/statistics/dpc_angle_error_summary_cross_correlation.png`

作用：

- 展示整体角度误差分布
- 展示角度误差 CDF
- 展示角度误差随真实位移大小变化

当前图上的核心数值：

- all finite samples:
  - `median angle = 3.12 deg`
  - `p95 angle = 19.76 deg`
- true displacement `>= 0.05 px`:
  - `median angle = 2.44 deg`
  - `p95 angle = 10.83 deg`
- true displacement `>= 0.10 px`:
  - `median angle = 1.82 deg`
  - `p95 angle = 7.38 deg`

这说明：

- 小位移区角度误差天然更不稳定
- 所以汇报时不能只报“整体 p95 angle”
- 应同时说明角度误差随真实位移幅值而变化

### 7.3 per-spot trust maps 图

文件：

- `/Users/yaaang/Desktop/model2/result/xray/sweeps_20260422_001/wfe/wfe_1p0rad_step8p0um_z450mm/analysis/dpc_spot_trust_maps_cross_correlation.png`
- `/Users/yaaang/Desktop/model2/result/xray/sweeps_20260422_001/wfe/wfe_1p0rad_step8p0um_z450mm/analysis/dpc_spot_trust_summary_cross_correlation.csv`

作用：

- 直接给出每个孔的位移恢复可信度
- 区分 `low-signal` 与真正 `suspicious` 的坏孔
- 同时展示每孔的位移幅值误差、方向误差和最终 trust 分级

当前主线 case（2026-04-22 统一重跑）下：

- `676` 个 spots 中：
  - `418` 个 `trusted`
  - `145` 个 `review`
  - `21` 个 `suspicious`
  - `92` 个 `low-signal`

这张图适合直接回答：

- 哪些孔的恢复位移可以放心使用
- 哪些孔只是因为信号太小，方向误差不宜直接解释

### 7.4 truth wavefront context 图

文件：

- `/Users/yaaang/Desktop/model2/result/xray/sweeps_20260422_001/wfe/wfe_1p0rad_step8p0um_z450mm/analysis/statistics/dpc_truth_wavefront_context_cross_correlation.png`

作用：

- 对比 `true displacement`、`equivalent slope`、`local phase RMS`
- 直接解释“低信号孔位”更接近 **低斜率**，而不是简单等价于 **低 phase / 低 WFE**

当前推荐表述：

- 低信号 spot 更直接对应 **孔平均 slope 小**
- 不应简单说成“该区域波前误差幅值小”
- phase 振幅可以不小，但只要孔平均斜率接近 0，spot 位移仍会很小

## 8. 当前最适合组会的叙事方式

当前最推荐的汇报逻辑是：

1. 问题：单张 accumulated 图是否能恢复 pulse-resolved DPC
2. 方法：cross-correlation + accumulated linearized residual
3. 主线 case：`1 rad / 8 um / 450 mm`
4. 能力展示：
   - 真实位移尺度
   - 恢复误差尺度
   - 角度误差
   - per-spot trust 分布
   - 10x temporal multiplexing
   - 26x26 spatial sampling
5. 参数窗口：
   - `step = 8.0–9.5 um`
   - `z = 450–550 mm`
6. 结论：
   - feasibility 已证明
   - 绝对误差达到 `0.01 px` 量级
   - 相对精度仍约 `10%`

## 9. 当前不能夸大的地方

新窗口中一定不要再走回这些口径：

1. 不要再把 phase RMSE 当成主结论
2. 不要再说当前结果已经达到 `1%` 精度
3. 不要再说“1 px 位移能做到 0.01 px 精度”
   - 当前数据并没有真正覆盖 `1 px` 位移 regime
4. 不要把 `WFE = 3 rad` 当作主线标称条件
   - 只能作为 stress test

## 10. 当前还没完成、但适合下一步做的事

### 10.1 汇报材料整理

下一窗口最适合继续做的是：

- 直接整理 PPT 提纲
- 每页放哪张图
- 每页怎么讲
- 中英文术语统一

### 10.2 如果还要继续做数值实验

最值得补的是：

1. **大位移 regime 扫描**
   - 真正去验证 `0.2 / 0.5 / 1.0 px` 量级时的相对精度

2. **低幅 / 小信号扫描**
   - 真正定义 sensitivity / minimum detectable displacement

3. **局部坏点治理**
   - 当前局部坏 spot 仍是尾部误差来源

## 11. 当前 case 对齐（避免混淆不同 error 量级）

目前仓库和桌面结果里至少有三套容易混淆的 X-ray case：

### 11.1 当前主线 case（authoritative）

- 最新统一重跑目录：`/Users/yaaang/Desktop/model2/result/xray/sweeps_20260422_001/wfe/wfe_1p0rad_step8p0um_z450mm`
- 最新统一重跑 batch：`/Users/yaaang/Desktop/model2/result/xray/sweeps_20260422_001`
- 参数：`step = 8 um`, `z = 450 mm`, `WFE = 1 rad`
- 几何：`26 x 26` spots, Hartmann pitch `25 um`
- 当前主线 DPC 指标（2026-04-22 统一重跑）：
  - `RMSE total = 0.0107 px`
  - `median sample abs err = 0.0061 px`
  - `p95 sample abs err = 0.0217 px`

本次统一重跑还给出了完整 sweep 结论：

- `step` 扫描最佳点：`8.5 um`，`RMSE total = 0.0096 px`
- `z` 扫描最佳点：`450 mm`，`RMSE total = 0.0116 px`
- `WFE` 扫描最佳点：`1.0 rad`，`RMSE total = 0.0107 px`

总汇图和摘要位于：

- `result/xray/sweeps_20260422_001/plots/sweep_overview.png`
- `result/xray/sweeps_20260422_001/plots/sweep_summary.md`
- `result/xray/sweeps_20260422_001/RESULTS_SUMMARY.md`

这是当前组会与对外口径应优先使用的结果。

### 11.2 旧 baseline：`run_002`

- 目录：`/Users/yaaang/Desktop/result/xray/run_002`
- 参数：`step = 7 um`, `z = 450 mm`
- 几何：`26 x 26` spots, Hartmann pitch `25 um`
- 代表性 DPC 指标：
  - `RMSE total = 0.0230 px`

这个 case 可以作为旧 baseline 参考，但**不是当前主线最优口径**。

### 11.3 已移除的 legacy demo

此前仓库里有一套 `20 x 20` / `50 um pitch` 的 `pointwise_demo`。

- 参数：`step = 7 um`, `z = 450 mm`
- 量级示例：
  - `RMSE total = 0.1776 px`
  - `median sample abs err = 0.0619 px`
  - `p95 sample abs err = 0.3387 px`

这个 demo 已从统一后的单版本仓库中移除，原因是：

- 它与当前主线 `8 um / 26x26 / 25 um pitch` 不是同一 case
- 它会让 error 量级看起来和当前主线差一个层级
- 它不应再作为当前 authoritative benchmark

结论仍然不变：

- `2` 与 `2.1` 的核心恢复算法本身并没有把同一输入算出不同量级
- 看到的量级差异，主要来自 **case 本身不同**

## 12. 重要提醒

当前统一版结果输出目录在：

- `/Users/yaaang/Desktop/model2/result/xray/...`

当前仓库已经可以直接在项目内完成输出，不再依赖桌面外部目录。

## 13. 推荐新窗口的第一句

如果在新窗口继续，可以直接用这段话作为开场：

“请先阅读 `AI_HANDOFF_README.md`。当前项目主目标是验证单张 accumulated X-ray Hartmann 图像能否恢复 pulse-resolved DPC。主评估口径已经切换到 DPC displacement，不再以 phase RMSE 为主。当前 authoritative 主线 case 为 `step=8 um, z=450 mm, WFE=1 rad`，最新统一重跑位于 `result/xray/sweeps_20260422_001/wfe/wfe_1p0rad_step8p0um_z450mm`，绝对误差达到 `0.01 px` 量级，但相对精度仍约 `10%`。请在这个前提下继续整理组会汇报材料或推进下一步实验。” 
