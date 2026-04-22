from __future__ import annotations
"""
激光 4f 中继模式 — 波动光学传播。

自包含实现: make_grid, 角谱传播, 薄透镜, 4f 传播链。
不依赖 simulation/wave_optics.py。
"""
import numpy as np
from typing import Tuple, Optional


# =============================================================================
# 网格
# =============================================================================

def make_grid(window_size: float, grid_points: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    生成以原点为中心的均匀 2D 网格。

    返回:
        X, Y: meshgrid (grid_points, grid_points)
        dx: 网格间距 (m)
    """
    dx = window_size / grid_points
    coords = (np.arange(grid_points) - grid_points / 2) * dx
    X, Y = np.meshgrid(coords, coords)
    return X, Y, dx


# =============================================================================
# 角谱传播
# =============================================================================

def ang_spec_propagate(field: np.ndarray,
                       wavelength: float,
                       dx: float,
                       distance: float,
                       padding_factor: int = 1) -> np.ndarray:
    """
    角谱法自由空间传播。

    参数:
        field: 输入复振幅场 (ny, nx)
        wavelength: 波长 (m)
        dx: 网格间距 (m)，假设 dx=dy
        distance: 传播距离 (m)
        padding_factor: FFT padding 倍数，1=不额外 pad

    返回:
        传播后的复振幅场，与输入同尺寸。
        dx_out = dx_in（角谱法网格不变）。
    """
    ny, nx = field.shape
    k = 2.0 * np.pi / wavelength

    # padding
    if padding_factor > 1:
        ny_pad = ny * padding_factor
        nx_pad = nx * padding_factor
        field_pad = np.zeros((ny_pad, nx_pad), dtype=complex)
        oy = (ny_pad - ny) // 2
        ox = (nx_pad - nx) // 2
        field_pad[oy:oy + ny, ox:ox + nx] = field
    else:
        field_pad = field
        ny_pad, nx_pad = ny, nx
        oy, ox = 0, 0

    # 频率网格
    fx = np.fft.fftfreq(nx_pad, d=dx)
    fy = np.fft.fftfreq(ny_pad, d=dx)
    FX, FY = np.meshgrid(fx, fy)

    # 传递函数
    kz_sq = k**2 - (2.0 * np.pi * FX)**2 - (2.0 * np.pi * FY)**2
    # 倏逝波截断
    propagating = kz_sq > 0
    H = np.zeros((ny_pad, nx_pad), dtype=complex)
    H[propagating] = np.exp(1j * distance * np.sqrt(kz_sq[propagating]))

    # 传播
    field_out = np.fft.ifft2(np.fft.fft2(field_pad) * H)

    # 去 padding
    if padding_factor > 1:
        field_out = field_out[oy:oy + ny, ox:ox + nx]

    return field_out


# =============================================================================
# 边缘能量诊断
# =============================================================================

def _check_edge_energy(field: np.ndarray,
                       threshold: float = 0.01,
                       edge_width: int = 8) -> float:
    """
    检查场边缘的能量占比。

    若超过 threshold，说明 sim_window 不够大或需要额外 padding。
    返回边缘能量占总能量的比例。
    """
    intensity = np.abs(field)**2
    total = np.sum(intensity)
    if total < 1e-30:
        return 0.0
    ny, nx = intensity.shape
    edge = np.zeros((ny, nx), dtype=bool)
    edge[:edge_width, :] = True
    edge[-edge_width:, :] = True
    edge[:, :edge_width] = True
    edge[:, -edge_width:] = True
    return float(np.sum(intensity[edge]) / total)


# =============================================================================
# 薄透镜
# =============================================================================

def apply_thin_lens(field: np.ndarray,
                    X: np.ndarray, Y: np.ndarray,
                    wavelength: float,
                    focal_length: float,
                    aperture_diameter: Optional[float] = None,
                    aberration_coeffs: Optional[list] = None) -> np.ndarray:
    """
    薄透镜相位调制 + 有限孔径 + Zernike 像差。

    φ(x,y) = -k/(2f)(x²+y²) + Σ aⱼ Zⱼ(ρ,θ)  (ρ ≤ 1 内)
    孔径外截零。

    参数:
        field: 输入复振幅场
        X, Y: 坐标网格 (m)
        wavelength: 波长 (m)
        focal_length: 焦距 (m)
        aperture_diameter: 通光孔径直径 (m)，None=无限
        aberration_coeffs: Zernike 像差系数 (弧度)，Noll Z4 起
    """
    k = 2.0 * np.pi / wavelength
    R2 = X**2 + Y**2
    phase = -k / (2.0 * focal_length) * R2

    # Zernike 像差（仅在归一化半径内施加）
    if aberration_coeffs is not None and len(aberration_coeffs) > 0:
        from simulation.laser_mode.aberration import zernike_noll
        if aperture_diameter is not None:
            R_norm = aperture_diameter / 2.0
        else:
            R_norm = float(np.max(np.sqrt(R2))) * 1.1
        rho = np.sqrt(R2) / R_norm
        theta = np.arctan2(Y, X)
        pupil = (rho <= 1.0).astype(float)
        for idx, coeff in enumerate(aberration_coeffs):
            if coeff == 0.0:
                continue
            noll_index = idx + 4  # 从 Z4 开始
            Z = zernike_noll(noll_index, rho, theta)
            phase += coeff * Z * pupil

    out = field * np.exp(1j * phase)

    # 有限孔径截断
    if aperture_diameter is not None:
        out = out * (np.sqrt(R2) <= aperture_diameter / 2.0)

    return out


def apply_circular_aperture(field: np.ndarray,
                            X: np.ndarray,
                            Y: np.ndarray,
                            diameter: float,
                            *,
                            center_x: float = 0.0,
                            center_y: float = 0.0) -> np.ndarray:
    """在当前实空间平面施加圆形光阑。"""
    if diameter is None or float(diameter) <= 0.0:
        raise ValueError("circular aperture diameter must be positive")
    R = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    return field * (R <= float(diameter) / 2.0)


# =============================================================================
# 4f 传播链
# =============================================================================

def propagate_4f_relay(field: np.ndarray,
                       X: np.ndarray, Y: np.ndarray,
                       dx: float,
                       optical_cfg,
                       show_progress: bool = True) -> Tuple[np.ndarray, float]:
    """
    4f 中继传播链（全角谱法）。

    传播链:
        掩膜 --(z₀+f₁)--> L₁ --(f₁+f₂)--> L₂ --(f₂)--> 探测器

    成像: 掩膜后 z₀ 处的物平面 → 放大 M_signed = -f₂/f₁ → 探测器
    dx_out = dx_in（角谱法网格不变）。
    每步传播后检查边缘能量，超阈值警告。

    参数:
        field: 掩膜后的复振幅场
        X, Y: 计算网格
        dx: 网格间距
        optical_cfg: OpticalConfig 实例
        show_progress: 是否打印诊断

    返回:
        (U_out, dx)
    """
    z0 = optical_cfg.relay_z0_m
    f1 = optical_cfg.relay_f1_m
    f2 = optical_cfg.relay_f2_m
    wl = optical_cfg.wavelength_m
    M = optical_cfg.magnification_signed
    lens_aperture = optical_cfg.relay_lens_aperture_m
    fourier_stop_enable = bool(getattr(optical_cfg, 'relay_fourier_stop_enable', False))
    fourier_stop_diameter = getattr(optical_cfg, 'relay_fourier_stop_diameter_m', None)
    fourier_stop_x = float(getattr(optical_cfg, 'relay_fourier_stop_x_m', 0.0))
    fourier_stop_y = float(getattr(optical_cfg, 'relay_fourier_stop_y_m', 0.0))

    if show_progress:
        print(f"  4f relay: z₀={z0*1e3:.1f}mm, "
              f"f₁={f1*1e3:.0f}mm, f₂={f2*1e3:.0f}mm, "
              f"M={M:.1f}×, z_eff={optical_cfg.z_eff_m*1e3:.1f}mm")
        if fourier_stop_enable:
            print(
                f"    Fourier stop: D={float(fourier_stop_diameter)*1e6:.0f} µm, "
                f"center=({fourier_stop_x*1e6:.0f}, {fourier_stop_y*1e6:.0f}) µm"
            )

    edge_threshold = 0.01

    # Step 1: 掩膜 → L₁ (z₀ + f₁)
    U = ang_spec_propagate(field, wl, dx, z0 + f1, padding_factor=1)
    edge_frac = _check_edge_energy(U)
    if edge_frac > edge_threshold and show_progress:
        print(f"    ⚠️ mask→L1: 边缘能量 {edge_frac*100:.2f}%")

    # Step 2: L₁
    U = apply_thin_lens(U, X, Y, wl, f1,
                        aperture_diameter=lens_aperture,
                        aberration_coeffs=optical_cfg.relay_lens1_aberration_coeffs)

    # Step 3a: L₁ → 频谱面 (f₁)
    U = ang_spec_propagate(U, wl, dx, f1, padding_factor=1)
    edge_frac = _check_edge_energy(U)
    if edge_frac > edge_threshold and show_progress:
        print(f"    ⚠️ L1→Fourier: 边缘能量 {edge_frac*100:.2f}%")

    # Step 3b: 频谱面小孔光阑
    if fourier_stop_enable:
        power_before = float(np.sum(np.abs(U)**2))
        U = apply_circular_aperture(
            U, X, Y,
            float(fourier_stop_diameter),
            center_x=fourier_stop_x,
            center_y=fourier_stop_y,
        )
        if show_progress and power_before > 1e-30:
            power_after = float(np.sum(np.abs(U)**2))
            print(f"    Fourier stop transmission: {100.0 * power_after / power_before:.2f}%")

    # Step 3c: 频谱面 → L₂ (f₂)
    U = ang_spec_propagate(U, wl, dx, f2, padding_factor=1)
    edge_frac = _check_edge_energy(U)
    if edge_frac > edge_threshold and show_progress:
        print(f"    ⚠️ Fourier→L2: 边缘能量 {edge_frac*100:.2f}%")

    # Step 4: L₂
    U = apply_thin_lens(U, X, Y, wl, f2,
                        aperture_diameter=lens_aperture,
                        aberration_coeffs=optical_cfg.relay_lens2_aberration_coeffs)

    # Step 5: L₂ → 探测器 (f₂)
    U = ang_spec_propagate(U, wl, dx, f2, padding_factor=1)
    edge_frac = _check_edge_energy(U)
    if edge_frac > edge_threshold and show_progress:
        print(f"    ⚠️ L2→det: 边缘能量 {edge_frac*100:.2f}%")

    return U, dx
