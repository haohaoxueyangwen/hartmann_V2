from __future__ import annotations
"""
激光 4f 中继 Hartmann 波前传感器仿真子系统。

独立于现有 X 射线仿真链，零依赖 simulation/hartmann_simulator.py。
"""
from simulation.laser_mode.config import (
    OpticalConfig,
    MaskConfig,
    DetectorConfig,
    AberrationConfig,
    MovementConfig,
    OutputConfig,
    validate_configs,
)
from simulation.laser_mode.simulator import Laser4fSimulator
