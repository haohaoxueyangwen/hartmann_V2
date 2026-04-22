from __future__ import annotations
"""
激光 4f 中继模式 — 仿真主类。

仿照现有 hartmann_simulator.py 的 PulseSuperpositionSimulator 风格。
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from simulation.laser_mode.config import (
    OpticalConfig, MaskConfig, DetectorConfig,
    AberrationConfig, MovementConfig, OutputConfig,
    validate_configs,
)
from simulation.laser_mode.wave_optics import (
    make_grid,
    ang_spec_propagate,
    _check_edge_energy,
    propagate_4f_relay,
)
from simulation.laser_mode.mask import make_finite_hartmann_mask, make_finite_hartmann_centers
from simulation.laser_mode.aberration import generate_random_phase_screen
from simulation.laser_mode.detector import sample_detector
from simulation.laser_mode.truth import compute_gt_displacements


class Laser4fSimulator:
    """
    激光 4f 中继 Hartmann 波前传感器仿真器。

    独立于现有 X 射线仿真链，零依赖 simulation/hartmann_simulator.py。
    """

    def __init__(self,
                 optical_config: OpticalConfig,
                 mask_config: MaskConfig,
                 detector_config: DetectorConfig,
                 aberration_config: Optional[AberrationConfig] = None,
                 movement_config: Optional[MovementConfig] = None,
                 output_config: Optional[OutputConfig] = None):
        self.optical_cfg = optical_config
        self.mask_cfg = mask_config
        self.detector_cfg = detector_config
        self.aberration_cfg = aberration_config if aberration_config is not None else AberrationConfig()
        self.movement_cfg = movement_config if movement_config is not None else MovementConfig()
        self.output_cfg = output_config if output_config is not None else OutputConfig()

        # 运行时状态
        self.X: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        self.dx: float = 0.0
        self.U_source: Optional[np.ndarray] = None
        self._phase_screens: List[np.ndarray] = []
        self._static_phase: Optional[np.ndarray] = None

    # =========================================================================
    # 初始化
    # =========================================================================

    def _setup(self) -> None:
        """初始化网格、光源、验证配置。"""
        validate_configs(self.optical_cfg, self.mask_cfg, self.detector_cfg)

        # 计算网格 (sim_window 尺度)
        self.X, self.Y, self.dx = make_grid(
            self.optical_cfg.sim_window_m,
            self.optical_cfg.grid_points,
        )

        # 光源场（只在 source_radius 内有光）
        self.U_source = self._make_source()

        # 静态像差 (如果需要)
        if self.aberration_cfg.mode == 'static':
            pupil_r = self._get_pupil_radius()
            self._static_phase, _ = generate_random_phase_screen(
                self.X, self.Y, pupil_r,
                max_order=self.aberration_cfg.max_order,
                amplitude_range=self.aberration_cfg.amplitude_range,
            )

        self._phase_screens = []

        if self.output_cfg.show_progress:
            self._print_setup_diagnostics()

    def _make_source(self) -> np.ndarray:
        """生成有限照射范围的光源场。"""
        R = np.sqrt(self.X**2 + self.Y**2)
        r0 = self.optical_cfg.source_radius_m
        stype = self.optical_cfg.source_type.lower()

        if stype == 'gaussian':
            field = np.exp(-R**2 / (2.0 * r0**2))
        elif stype == 'tophat':
            # 硬截断圆形平顶光束
            field = np.where(R <= r0, 1.0, 0.0)
        else:
            # plane: super-Gaussian 软边，模拟有限光束的平滑截断
            field = np.exp(-(R / r0)**10)

        # 归一化: 总能量 Σ|U|²·dx² = 1
        energy = np.sum(np.abs(field)**2) * self.dx**2
        if energy > 1e-30:
            field = field / np.sqrt(energy)

        return field.astype(complex)

    def _get_pupil_radius(self) -> float:
        """获取像差 pupil 半径。"""
        if self.aberration_cfg.pupil_radius_m is not None:
            return self.aberration_cfg.pupil_radius_m
        return self.mask_cfg.mask_window_m / 2.0

    def _print_setup_diagnostics(self) -> None:
        """打印初始化诊断信息。"""
        o = self.optical_cfg
        m = self.mask_cfg
        d = self.detector_cfg
        M = o.magnification_signed
        n_spots = int(m.mask_window_m / m.pitch_m)
        det_fov = d.pixel_size * max(d.nx, d.ny)
        nf = (m.hole_diameter_m / 2.0)**2 / (o.relay_z0_m * o.wavelength_m)

        print(f"  Wavelength: {o.wavelength_m*1e9:.1f} nm")
        print(f"  Grid: {o.grid_points}×{o.grid_points}, "
              f"dx={self.dx*1e6:.2f} µm, "
              f"sim_window={o.sim_window_m*1e3:.1f} mm")
        print(f"  4f relay: z₀={o.relay_z0_m*1e3:.1f}mm, "
              f"f₁={o.relay_f1_m*1e3:.0f}mm, f₂={o.relay_f2_m*1e3:.0f}mm")
        print(f"  Magnification: M={M:.1f}× (signed), "
              f"z_eff={o.z_eff_m*1e3:.1f}mm")
        print(f"  Fresnel number N_F(z₀): {nf:.2f}")
        if self.aberration_cfg.phase_plate_to_mask_m > 0.0:
            print(f"  Phase plate -> mask: {self.aberration_cfg.phase_plate_to_mask_m*1e3:.1f} mm "
                  f"(angular spectrum)")
        print(f"  Mask: {m.mask_window_m*1e3:.1f}mm, "
              f"pitch={m.pitch_m*1e6:.0f}µm, "
              f"hole={m.hole_diameter_m*1e6:.0f}µm, "
              f"{n_spots}×{n_spots} spots")
        print(f"  Detector: {d.nx}×{d.ny}, "
              f"pixel={d.pixel_size*1e6:.1f}µm, "
              f"FOV={det_fov*1e3:.1f}mm")
        print(f"  Detector spot: {abs(M)*m.hole_diameter_m/d.pixel_size:.0f} px, "
              f"pitch: {abs(M)*m.pitch_m/d.pixel_size:.0f} px")

    def _propagate_phase_plate_to_mask(self,
                                       field: np.ndarray,
                                       *,
                                       show_progress: bool = False) -> np.ndarray:
        """相位板平面 -> Hartmann 掩膜平面。d=0 时直接返回。"""
        distance = float(self.aberration_cfg.phase_plate_to_mask_m)
        if distance <= 0.0:
            return field

        U = ang_spec_propagate(
            field,
            self.optical_cfg.wavelength_m,
            self.dx,
            distance,
            padding_factor=1,
        )
        edge_frac = _check_edge_energy(U)
        if show_progress and edge_frac > 0.01:
            print(f"  Phase plate -> mask: 边缘能量 {edge_frac*100:.2f}%")
        return U

    # =========================================================================
    # 单帧生成
    # =========================================================================

    def _compute_translation_offset(self, pulse_index: int) -> Tuple[float, float]:
        """计算当前脉冲的平移偏移。"""
        if self.movement_cfg.mode != 'translation':
            return 0.0, 0.0
        angle_rad = np.deg2rad(self.movement_cfg.angle_deg)
        dx_shift = self.movement_cfg.step_m * np.cos(angle_rad) * pulse_index
        dy_shift = self.movement_cfg.step_m * np.sin(angle_rad) * pulse_index
        return dx_shift, dy_shift

    def _generate_single_frame(self, pulse_index: int
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        生成单帧。

        返回:
            (detector_aberrated, detector_ideal, mask,
             phase_screen_plate, hartmann_phase)
            其中 hartmann_phase 是 Hartmann 平面的相位 (rad)，用于 GT 计算。
        """
        shift_x, shift_y = self._compute_translation_offset(pulse_index)

        # 掩膜
        mask = make_finite_hartmann_mask(
            self.X, self.Y,
            mask_window=self.mask_cfg.mask_window_m,
            pitch=self.mask_cfg.pitch_m,
            hole_diameter=self.mask_cfg.hole_diameter_m,
            shift_x_m=shift_x,
            shift_y_m=shift_y,
        )

        # 像差
        phase_screen = None
        if self.aberration_cfg.mode == 'per_pulse':
            pupil_r = self._get_pupil_radius()
            phase_screen, _ = generate_random_phase_screen(
                self.X, self.Y, pupil_r,
                max_order=self.aberration_cfg.max_order,
                amplitude_range=self.aberration_cfg.amplitude_range,
            )
            self._phase_screens.append(phase_screen)
        elif self.aberration_cfg.mode == 'static':
            phase_screen = self._static_phase
            self._phase_screens.append(phase_screen)

        show = self.output_cfg.show_progress and pulse_index == 0

        # --- 相位板平面 -> Hartmann 掩膜平面 ---
        U_plate_ideal = self.U_source
        U_mask_ideal = self._propagate_phase_plate_to_mask(
            U_plate_ideal,
            show_progress=show,
        )

        # --- 含像差: 光源 × 相位板 -> 传播到掩膜 -> 掩膜 -> 4f -> 探测器 ---
        if phase_screen is not None:
            U_plate_aberr = self.U_source * np.exp(1j * phase_screen)
            U_mask_aberr = self._propagate_phase_plate_to_mask(
                U_plate_aberr,
                show_progress=False,
            )
        else:
            U_mask_aberr = U_mask_ideal

        U_aberr = U_mask_aberr * mask
        U_out_aberr, _ = propagate_4f_relay(
            U_aberr, self.X, self.Y, self.dx,
            self.optical_cfg, show_progress=show)
        I_aberr = np.abs(U_out_aberr)**2

        D_aberr = sample_detector(
            I_aberr, self.dx,
            pixel_size=self.detector_cfg.pixel_size,
            detector_nx=self.detector_cfg.nx,
            detector_ny=self.detector_cfg.ny,
            mode=self.detector_cfg.sampling_mode,
            upsample_factor=self.detector_cfg.upsample_factor,
            bin_size=self.detector_cfg.bin_size,
            interpolation=self.detector_cfg.interpolation,
        )

        # --- 无像差: 光源 -> 传播到掩膜 -> 掩膜 -> 4f -> 探测器 ---
        if phase_screen is not None or self.aberration_cfg.phase_plate_to_mask_m > 0.0:
            U_ideal = U_mask_ideal * mask
            U_out_ideal, _ = propagate_4f_relay(
                U_ideal, self.X, self.Y, self.dx,
                self.optical_cfg, show_progress=False)
            I_ideal = np.abs(U_out_ideal)**2
            D_ideal = sample_detector(
                I_ideal, self.dx,
                pixel_size=self.detector_cfg.pixel_size,
                detector_nx=self.detector_cfg.nx,
                detector_ny=self.detector_cfg.ny,
                mode=self.detector_cfg.sampling_mode,
                upsample_factor=self.detector_cfg.upsample_factor,
                bin_size=self.detector_cfg.bin_size,
                interpolation=self.detector_cfg.interpolation,
            )
        else:
            D_ideal = D_aberr

        # 提取 Hartmann 平面相位 (用于 GT 计算)
        # 如果有像差，hartmann_phase = angle(U_mask_aberr)
        # 如果没有像差，hartmann_phase = None (GT 无意义)
        hartmann_phase = None
        if phase_screen is not None:
            hartmann_phase = np.angle(U_mask_aberr)

        return D_aberr, D_ideal, mask, phase_screen, hartmann_phase

    # =========================================================================
    # 主仿真循环
    # =========================================================================

    def simulate(self) -> Dict[str, Any]:
        """
        执行完整的多帧仿真。

        返回字典键名与现有系统对齐:
            integrated_sampled, integrated_ideal,
            frames_sampled, frames_ideal,
            wfe_phase_map_rad, phase_screens, gt_displacements
        """
        self._setup()
        n_pulses = self.movement_cfg.n_pulses

        if self.output_cfg.show_progress:
            print(f"\nStarting laser 4f simulation: {n_pulses} pulses ...")

        frames_sampled: List[np.ndarray] = []
        frames_ideal: List[np.ndarray] = []
        hartmann_phases: List[np.ndarray] = []
        integrated_sampled = np.zeros((self.detector_cfg.ny, self.detector_cfg.nx))
        integrated_ideal = np.zeros((self.detector_cfg.ny, self.detector_cfg.nx))

        for i in range(n_pulses):
            if self.output_cfg.show_progress:
                if (i + 1) % max(1, n_pulses // 10) == 0:
                    print(f"  Progress: {i+1}/{n_pulses} "
                          f"({100*(i+1)/n_pulses:.0f}%)")

            D_aberr, D_ideal, mask, phase, h_phase = self._generate_single_frame(i)

            frames_sampled.append(D_aberr)
            frames_ideal.append(D_ideal)
            integrated_sampled += D_aberr
            integrated_ideal += D_ideal
            if h_phase is not None:
                hartmann_phases.append(h_phase)

        # GT 位移 — 从 Hartmann 平面相位的梯度计算
        gt_disp = None
        wfe_phase_map_rad = None
        if len(hartmann_phases) > 0:
            gt_disp = self._compute_gt(hartmann_phases)
            # 与 X 射线 Step 1 的 aggregate 输出语义保持一致：
            # per_pulse 模式下用 frame 0 作为代表性 WFE 真值图；
            # static 模式下所有帧相同，因此取首帧即可。
            wfe_phase_map_rad = np.asarray(hartmann_phases[0], dtype=np.float64)

        # 打包结果
        phase_screens_arr = None
        if self._phase_screens:
            phase_screens_arr = np.array(self._phase_screens)

        result = {
            "integrated_sampled": integrated_sampled,
            "integrated_ideal": integrated_ideal,
            "frames_sampled": frames_sampled,
            "frames_ideal": frames_ideal,
            "wfe_phase_map_rad": wfe_phase_map_rad,
            "phase_screens": phase_screens_arr,
            "gt_displacements": gt_disp,
        }

        if self.output_cfg.show_progress:
            print(f"  Simulation complete: {n_pulses} pulses")
            if gt_disp is not None:
                dx_px = gt_disp['displacement_x_px']
                dy_px = gt_disp['displacement_y_px']
                total = np.sqrt(dx_px**2 + dy_px**2)
                print(f"  GT displacement: mean={np.nanmean(total):.3f} px, "
                      f"max={np.nanmax(total):.3f} px")

        return result

    def _compute_gt(self,
                    hartmann_phases: List[np.ndarray]) -> Optional[Dict[str, Any]]:
        """从 Hartmann 平面相位计算 GT 位移（与 X 射线同逻辑）。"""
        centers = make_finite_hartmann_centers(
            self.mask_cfg.mask_window_m,
            self.mask_cfg.pitch_m,
        )
        if len(centers) == 0:
            return None

        cx_unique = np.unique(centers[:, 0])
        cy_unique = np.unique(centers[:, 1])

        return compute_gt_displacements(
            hartmann_phase_screens=hartmann_phases,
            X=self.X,
            Y=self.Y,
            dx=self.dx,
            centers_x_m=cx_unique,
            centers_y_m=cy_unique,
            hole_diameter_m=self.mask_cfg.hole_diameter_m,
            wavelength_m=self.optical_cfg.wavelength_m,
            z_eff_m=self.optical_cfg.z_eff_m,
            pixel_size_m=self.detector_cfg.pixel_size,
            magnification_signed=self.optical_cfg.magnification_signed,
            pitch_m=self.mask_cfg.pitch_m,
        )
