# hartmann_V2

Hartmann 光斑仿真与波前恢复的历史完整研究代码，包含 X 射线及可见光相关模型、质心/互相关/优化恢复、光斑识别与 Zernike 后处理。

## 环境与入口

本次以 Python 3.12.7、NumPy 1.26.4、SciPy 1.13.1、Matplotlib 3.9.2 验证依赖导入与 CLI 帮助。完整物理仿真和实验数据恢复仍需对应研究任务的参数与验收记录。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe simulate.py --help
.\.venv\Scripts\python.exe recover.py --help
```

| 入口 | 功能 |
|---|---|
| `simulate.py` | 使用 `config_simulate.py` 或显式 `--config` 生成一组数据 |
| `recover.py --sim-dir ...` | 对指定仿真目录按所选方法恢复 |
| `analysis/displacement_methods.py` | 网格、质心、位移等基础计算 |
| `analysis/zernike_reconstruction.py` | 基底、波前积分与系数评估 |
| `simulation/` | 衍射、探测器、光斑及相关传播模型 |
| `visualization/` | 结果图和诊断 |

先检查配置中的单位、像元、距离、掩膜和模式。`simulate.py` 默认使用研究配置，可能是较大任务；新手先运行独立 [wavefront-starter 教学工具](https://github.com/haohaoxueyangwen/wavefront-starter) 的小型 Hartmann 合成例子。该例子仅覆盖固定网格中的孤立光斑质心位移。

结果写入 `result/` 并保持不入 Git。不同几何、参考状态或模式的结果应使用不同目录，保留原始配置和数据身份。

## 与当前研究工程的关系

这里保留已有 GitHub 源码结构。本次新增安装说明、依赖清单和环境检查，未迁入当前本地新版 GUI、共享波前模块或多脉冲实验代码，也未修改算法参数。后续科研功能按原项目任务单独维护。

2026-10-11 检查：所有跟踪 Python 文件语法、关键依赖导入、`simulate.py --help`、`recover.py --help`。这些检查不构成完整仿真或实验物理精度验收。
