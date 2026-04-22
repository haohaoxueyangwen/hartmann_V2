from __future__ import annotations
"""
激光 4f 中继模式 — GT 位移生成。

从 Hartmann 平面的相位屏计算每个子孔径的 ground truth 位移和平均相位。
与 X 射线 GT (hartmann_simulator.py:_compute_gt_displacements) 同一物理定义：

    Δx(m) = [∂φ/∂x]_avg × λ/(2π) × z_prop
    Δx(px) = Δx(m) / pixel_size

其中 z_prop 对激光 4f 模式 = z_eff = |M| × z₀。
位移定义在探测器平面，已包含放大率。

不从探测器图像反推质心。GT 与恢复算法完全解耦。
"""
import numpy as np
from typing import List, Dict, Any, Optional


def compute_gt_displacements(
    hartmann_phase_screens: List[np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    dx: float,
    centers_x_m: np.ndarray,
    centers_y_m: np.ndarray,
    hole_diameter_m: float,
    wavelength_m: float,
    z_eff_m: float,
    pixel_size_m: float,
    magnification_signed: float,
    pitch_m: float,
) -> Dict[str, Any]:
    """
    从 Hartmann 平面相位屏计算 GT 位移。

    参数:
        hartmann_phase_screens: 每帧在 Hartmann 平面的相位 (rad)，
            shape 与 X, Y 相同。如果 phase_plate_to_mask_m > 0，
            这是传播到掩膜平面后的相位，不是相位板平面的。
        X, Y: 计算网格坐标 (m)
        dx: 网格间距 (m)
        centers_x_m: 子孔径中心 x 坐标 (m)，物方
        centers_y_m: 子孔径中心 y 坐标 (m)，物方
        hole_diameter_m: 子孔径直径 (m)
        wavelength_m: 波长 (m)
        z_eff_m: 等效杠杆臂 = |M| × z₀ (m)
        pixel_size_m: 探测器像素尺寸 (m)
        magnification_signed: 横向放大率 (负 = 倒像)
        pitch_m: 子孔径间距 (m)

    返回:
        gt_dict: 与 X 射线 gt_displacements.npz 完全同构的字典，
                 包含 phase_rad 字段。
    """
    grid_cols = len(centers_x_m)
    grid_rows = len(centers_y_m)
    n_frames = len(hartmann_phase_screens)

    hole_radius_px = int(np.ceil((hole_diameter_m / 2.0) / dx))
    scale = wavelength_m / (2.0 * np.pi)

    all_phase_vals = np.full((n_frames, grid_rows, grid_cols), np.nan)
    all_disp_x_px = np.full((n_frames, grid_rows, grid_cols), np.nan)
    all_disp_y_px = np.full((n_frames, grid_rows, grid_cols), np.nan)

    for f_idx, phase_screen in enumerate(hartmann_phase_screens):
        # 相位梯度 (rad/m)
        grad_y, grad_x = np.gradient(phase_screen, dx, dx)

        for r, yc in enumerate(centers_y_m):
            for c, xc in enumerate(centers_x_m):
                # 子孔径中心在计算网格中的像素索引
                cx_idx = int(np.clip(
                    np.round((xc - X[0, 0]) / dx), 0, X.shape[1] - 1))
                cy_idx = int(np.clip(
                    np.round((yc - Y[0, 0]) / dx), 0, Y.shape[0] - 1))

                # 圆形孔径区域
                y0 = max(0, cy_idx - hole_radius_px)
                y1 = min(phase_screen.shape[0], cy_idx + hole_radius_px + 1)
                x0 = max(0, cx_idx - hole_radius_px)
                x1 = min(phase_screen.shape[1], cx_idx + hole_radius_px + 1)

                yy, xx = np.ogrid[y0:y1, x0:x1]
                dist = np.sqrt((yy - cy_idx)**2 + (xx - cx_idx)**2)
                aperture_mask = dist <= hole_radius_px

                if aperture_mask.sum() > 0:
                    all_phase_vals[f_idx, r, c] = np.mean(
                        phase_screen[y0:y1, x0:x1][aperture_mask])
                    gx_mean = np.mean(grad_x[y0:y1, x0:x1][aperture_mask])
                    gy_mean = np.mean(grad_y[y0:y1, x0:x1][aperture_mask])
                else:
                    all_phase_vals[f_idx, r, c] = phase_screen[cy_idx, cx_idx]
                    gx_mean = grad_x[cy_idx, cx_idx]
                    gy_mean = grad_y[cy_idx, cx_idx]

                # 物理位移 → 探测器像素位移
                # Δx_det(m) = M_signed × (scale × gx_mean × z₀)
                # 但 z_eff = |M| × z₀，所以:
                # Δx_det(m) = sign(M) × scale × gx_mean × z_eff
                # Δx_det(px) = Δx_det(m) / pixel_size
                sign_M = -1.0 if magnification_signed < 0 else 1.0
                all_disp_x_px[f_idx, r, c] = (
                    sign_M * scale * gx_mean * z_eff_m / pixel_size_m)
                all_disp_y_px[f_idx, r, c] = (
                    sign_M * scale * gy_mean * z_eff_m / pixel_size_m)

    return {
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "centers_x_m": centers_x_m,
        "centers_y_m": centers_y_m,
        "phase_rad": all_phase_vals,
        "displacement_x_px": all_disp_x_px,
        "displacement_y_px": all_disp_y_px,
        "displacement_x_m": all_disp_x_px * pixel_size_m,
        "displacement_y_m": all_disp_y_px * pixel_size_m,
        "wavelength_m": float(wavelength_m),
        "z_prop_m": float(z_eff_m),
        "pixel_size_m": float(pixel_size_m),
        "spacing_m": float(pitch_m),
    }
