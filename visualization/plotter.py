from __future__ import annotations
"""
Visualization utilities (strictly aligned to detector FOV)

Features:
- Show unsampled intensity cropped to detector FOV (no resampling).
- Show detector-sampled image aligned to detector FOV (as-is).
- Optionally display accumulation series (grid or simple animation).

Notes:
- Uses non-interactive Agg backend for headless image saving (no GUI flicker).
- All images saved directly to files without displaying.
"""

from typing import List, Optional, Sequence, Tuple
import os
import math
import numpy as np

# Use non-interactive backend to avoid GUI flicker
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import apply_plot_font_bump

_MATPLOTLIB_READY = True

apply_plot_font_bump()


def _ensure_matplotlib():
    return _MATPLOTLIB_READY


def _optional_save_path(save_dir: Optional[str], filename: str) -> Optional[str]:
    """Build a save path only when a save directory is provided."""
    if save_dir is None:
        return None
    return os.path.join(save_dir, filename)


def crop_to_detector_fov(
    intensity: np.ndarray,
    dx: float,
    dy: float,
    pixel_size: float,
    nx: int,
    ny: int,
) -> Tuple[np.ndarray, float, float]:
    """
    将未采样强度图按探测器视场进行居中裁剪。

    返回：裁剪后的图像，以及探测器视场尺寸 (FOV_x, FOV_y)。
    """
    if intensity is None:
        raise ValueError("intensity 不能为空")
    if intensity.ndim != 2:
        raise ValueError(f"intensity 必须是二维数组，当前维度={intensity.ndim}")

    FOV_x = float(nx) * float(pixel_size)
    FOV_y = float(ny) * float(pixel_size)

    # 需要保留的网格采样数（向下取整），并受原图尺寸上限约束
    crop_nx = int(min(intensity.shape[1], math.floor(FOV_x / float(dx))))
    crop_ny = int(min(intensity.shape[0], math.floor(FOV_y / float(dy))))
    crop_nx = max(1, crop_nx)
    crop_ny = max(1, crop_ny)

    # 居中裁剪
    Ny, Nx = intensity.shape
    sx = max(0, (Nx - crop_nx) // 2)
    sy = max(0, (Ny - crop_ny) // 2)
    ex = sx + crop_nx
    ey = sy + crop_ny
    cropped = intensity[sy:ey, sx:ex]

    return np.asarray(cropped), FOV_x, FOV_y


def show_unsampled_in_detector_fov(
    intensity: Optional[np.ndarray],
    dx: float,
    dy: float,
    pixel_size: float,
    nx: int,
    ny: int,
    *,
    use_log_scale: bool = True,
    colormap: str = "magma",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    block: bool = True,
) -> None:
    """
    Display unsampled intensity cropped to detector FOV.
    - No downsampling/resampling, only center-cropping to detector FOV size.
    - Axes aligned to detector FOV, units in μm.
    """
    if intensity is None:
        print("[viz] No unsampled intensity (integrated_unsampled=None)")
        return

    if not _ensure_matplotlib():
        return
    import matplotlib.pyplot as plt

    cropped, FOV_x, FOV_y = crop_to_detector_fov(intensity, dx, dy, pixel_size, nx, ny)
    img = np.asarray(cropped, dtype=np.float64)
    if use_log_scale:
        img = np.log1p(np.maximum(img, 0.0))

    extent = [-(FOV_x * 1e6) / 2.0, (FOV_x * 1e6) / 2.0, -(FOV_y * 1e6) / 2.0, (FOV_y * 1e6) / 2.0]

    plt.figure(figsize=(6, 6))
    plt.imshow(img, extent=extent, origin='lower', cmap=colormap, aspect='equal')
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.colorbar(label=("log(1+I)" if use_log_scale else "Intensity"))
    plt.title(title or "Unsampled intensity (cropped to detector FOV)")
    plt.tight_layout()
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        except Exception:
            pass
        try:
            plt.savefig(save_path, dpi=150)
            print(f"[viz] Saved unsampled intensity to: {save_path}")
        except Exception as e:
            print(f"[viz] Failed to save unsampled intensity: {e}")
    plt.close()


def show_detector_image(
    detector_image: Optional[np.ndarray],
    pixel_size: float,
    nx: int,
    ny: int,
    *,
    colormap: str = "magma",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    block: bool = True,
) -> None:
    """
    Display detector-sampled image in detector FOV coordinates.
    - No image modification (no scaling/resampling).
    - Axes aligned to detector FOV, units in μm.
    """
    if detector_image is None:
        print("[viz] No sampled image (integrated_sampled=None)")
        return

    if not _ensure_matplotlib():
        return
    import matplotlib.pyplot as plt

    img = np.asarray(detector_image, dtype=np.float64)
    FOV_x = float(nx) * float(pixel_size)
    FOV_y = float(ny) * float(pixel_size)
    extent = [-(FOV_x * 1e6) / 2.0, (FOV_x * 1e6) / 2.0, -(FOV_y * 1e6) / 2.0, (FOV_y * 1e6) / 2.0]

    plt.figure(figsize=(6, 6))
    plt.imshow(img, extent=extent, origin='lower', cmap=colormap, aspect='equal')
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.colorbar(label="Intensity (sampled)")
    plt.title(title or "Detector image (FOV-aligned)")
    plt.tight_layout()
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        except Exception:
            pass
        try:
            plt.savefig(save_path, dpi=150)
            print(f"[viz] Saved detector image to: {save_path}")
        except Exception as e:
            print(f"[viz] Failed to save detector image: {e}")
    plt.close()


def show_detector_comparison(
    ref_image: Optional[np.ndarray],
    sim_image: Optional[np.ndarray],
    pixel_size: float,
    nx: int,
    ny: int,
    *,
    colormap: str = "magma",
    diff_colormap: str = "seismic",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    block: bool = True,
) -> None:
    if ref_image is None or sim_image is None:
        print("[viz] Missing images for comparison")
        return
    if not _ensure_matplotlib():
        return
    import matplotlib.pyplot as plt
    ref = np.asarray(ref_image, dtype=np.float64)
    sim = np.asarray(sim_image, dtype=np.float64)
    diff = sim - ref
    FOV_x = float(nx) * float(pixel_size)
    FOV_y = float(ny) * float(pixel_size)
    extent = [-(FOV_x * 1e6) / 2.0, (FOV_x * 1e6) / 2.0, -(FOV_y * 1e6) / 2.0, (FOV_y * 1e6) / 2.0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axes[0].imshow(ref, extent=extent, origin='lower', cmap=colormap, aspect='equal')
    axes[0].set_xlabel("x (μm)")
    axes[0].set_ylabel("y (μm)")
    axes[0].set_title("Reference")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Intensity")
    im1 = axes[1].imshow(sim, extent=extent, origin='lower', cmap=colormap, aspect='equal')
    axes[1].set_xlabel("x (μm)")
    axes[1].set_ylabel("y (μm)")
    axes[1].set_title("With WFE")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Intensity")
    im2 = axes[2].imshow(diff, extent=extent, origin='lower', cmap=diff_colormap, aspect='equal')
    axes[2].set_xlabel("x (μm)")
    axes[2].set_ylabel("y (μm)")
    axes[2].set_title("Difference (WFE - Ref)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="ΔIntensity")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        except Exception:
            pass
        try:
            plt.savefig(save_path, dpi=150)
            print(f"[viz] Saved detector comparison to: {save_path}")
        except Exception as e:
            print(f"[viz] Failed to save detector comparison: {e}")
    plt.close()













def show_accumulation_series(
    frames: Optional[Sequence[np.ndarray]],
    pixel_size: float,
    nx: int,
    ny: int,
    *,
    animate: bool = False,
    interval_ms: int = 120,
    colormap: str = "magma",
    block: bool = True,
) -> None:
    # Display cumulative detector frames with optional animation.
    if not frames:
        print("[viz] No frames, cannot display accumulation series")
        return
    if not _ensure_matplotlib():
        return
    import matplotlib.pyplot as plt
    from matplotlib import animation

    series: List[np.ndarray] = []
    acc = None
    for f in frames:
        arr = np.asarray(f, dtype=np.float64)
        acc = arr if acc is None else acc + arr
        series.append(acc.copy())

    FOV_x = float(nx) * float(pixel_size)
    FOV_y = float(ny) * float(pixel_size)
    extent = [-(FOV_x * 1e6) / 2.0, (FOV_x * 1e6) / 2.0, -(FOV_y * 1e6) / 2.0, (FOV_y * 1e6) / 2.0]

    if animate:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(series[0], extent=extent, origin='lower', cmap=colormap, aspect='equal')
        plt.colorbar(im, ax=ax, label="Intensity (accum)")
        ax.set_xlabel("x (μm)")
        ax.set_ylabel("y (μm)")

        def _update(idx: int):
            im.set_data(series[idx])
            ax.set_title(f"Accumulation {idx+1}/{len(series)}")
            return [im]

        animation.FuncAnimation(fig, _update, frames=len(series), interval=interval_ms, blit=True, repeat=False)
        plt.tight_layout()
        plt.close()
        return

    n_frames = len(series)
    cols = int(np.ceil(np.sqrt(n_frames)))
    rows = int(np.ceil(n_frames / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).flatten()
    for ax, data, idx in zip(axes, series, range(n_frames)):
        im = ax.imshow(data, extent=extent, origin='lower', cmap=colormap, aspect='equal')
        ax.set_title(f"Frame {idx+1}")
        ax.set_xlabel("x (μm)")
        ax.set_ylabel("y (μm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[n_frames:]:
        ax.axis('off')
    plt.tight_layout()
    plt.close()


def show_centroid_diagnostics(
    detector_image: Optional[np.ndarray],
    pixel_size: float,
    nx: int,
    ny: int,
    *,
    centers: Optional[Sequence[dict]] = None,
    mask: Optional[np.ndarray] = None,
    seeds_xy: Optional[Sequence[Tuple[float, float]]] = None,
    colormap: str = "magma",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    block: bool = True,
) -> None:
    if detector_image is None:
        print("[viz] No sampled image for diagnostics")
        return
    if not _ensure_matplotlib():
        return
    import matplotlib.pyplot as plt
    img = np.asarray(detector_image, dtype=np.float64)
    FOV_x = float(nx) * float(pixel_size)
    FOV_y = float(ny) * float(pixel_size)
    extent = [-(FOV_x * 1e6) / 2.0, (FOV_x * 1e6) / 2.0, -(FOV_y * 1e6) / 2.0, (FOV_y * 1e6) / 2.0]
    plt.figure(figsize=(6, 6))
    plt.imshow(img, extent=extent, origin='lower', cmap=colormap, aspect='equal')
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.colorbar(label="Intensity")
    plt.title(title or "Centroid diagnostics")
    if mask is not None:
        overlay = (np.asarray(mask, dtype=float) > 0).astype(float)
        plt.imshow(overlay, extent=extent, origin='lower', cmap='Blues', alpha=0.25)
    if centers:
        xs_um = []
        ys_um = []
        for c in centers:
            cx = float(c.get('x_px', 0.0))
            cy = float(c.get('y_px', 0.0))
            x_um = (cx - nx / 2.0 + 0.5) * pixel_size * 1e6
            y_um = (cy - ny / 2.0 + 0.5) * pixel_size * 1e6
            xs_um.append(x_um)
            ys_um.append(y_um)
        if xs_um:
            plt.scatter(xs_um, ys_um, s=24, c='cyan', marker='x', linewidths=1.2, label='centroids')
    if seeds_xy:
        sx_um = []
        sy_um = []
        for sx, sy in seeds_xy:
            x_um = (float(sx) - nx / 2.0 + 0.5) * pixel_size * 1e6
            y_um = (float(sy) - ny / 2.0 + 0.5) * pixel_size * 1e6
            sx_um.append(x_um)
            sy_um.append(y_um)
        if sx_um:
            plt.scatter(sx_um, sy_um, s=18, c='red', marker='+', linewidths=1.0, label='seeds')
    if centers or seeds_xy:
        plt.legend(loc='upper right')
    plt.tight_layout()
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        except Exception:
            pass
        try:
            plt.savefig(save_path, dpi=150)
            print(f"[viz] Saved centroid diagnostics to: {save_path}")
        except Exception as exc:
            print(f"[viz] Failed to save centroid diagnostics: {exc}")
    plt.close()


def show_displacement_vectors(
    detector_image: Optional[np.ndarray],
    centers: Optional[Sequence[dict]],
    displacements: Optional[Sequence[dict]],
    pixel_size: float,
    nx: int,
    ny: int,
    *,
    colormap: str = "magma",
    color: str = "yellow",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    block: bool = True,
    amplification: float = 1.0,
) -> None:
    if detector_image is None:
        print("[viz] No sampled image (integrated_sampled=None)")
        return
    if not centers or not displacements:
        print("[viz] No centers or displacements to overlay")
        return
    if not _ensure_matplotlib():
        return
    import matplotlib.pyplot as plt
    img = np.asarray(detector_image, dtype=np.float64)
    FOV_x = float(nx) * float(pixel_size)
    FOV_y = float(ny) * float(pixel_size)
    extent = [-(FOV_x * 1e6) / 2.0, (FOV_x * 1e6) / 2.0, -(FOV_y * 1e6) / 2.0, (FOV_y * 1e6) / 2.0]
    plt.figure(figsize=(6, 6))
    plt.imshow(img, extent=extent, origin='lower', cmap=colormap, aspect='equal')
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.colorbar(label="Intensity (sampled)")
    
    final_title = title or "Displacement vectors"
    if amplification != 1.0:
        final_title += f" ({amplification}x)"
    plt.title(final_title)
    
    # Group displacements by pulse_idx
    groups = {}
    for d in displacements:
        pidx = d.get('pulse_idx')
        # Use -1 for unknown pulse index
        key = int(pidx) if pidx is not None else -1
        
        if key not in groups:
            groups[key] = {'xs': [], 'ys': [], 'us': [], 'vs': []}
            
        x_ref_px = float(d.get('x_ref_px', 0.0))
        y_ref_px = float(d.get('y_ref_px', 0.0))
        dx_px = float(d.get('dx_px', 0.0))
        dy_px = float(d.get('dy_px', 0.0))
        
        xr_um = (x_ref_px - nx / 2.0 + 0.5) * pixel_size * 1e6
        yr_um = (y_ref_px - ny / 2.0 + 0.5) * pixel_size * 1e6
        
        u_um = dx_px * pixel_size * 1e6
        v_um = dy_px * pixel_size * 1e6
        
        groups[key]['xs'].append(xr_um)
        groups[key]['ys'].append(yr_um)
        groups[key]['us'].append(u_um)
        groups[key]['vs'].append(v_um)

    q_scale = 1.0 / max(1e-6, float(amplification))
    
    sorted_keys = sorted(groups.keys())
    
    # If only one group (or no grouping info), use single color
    if len(sorted_keys) == 1:
        k = sorted_keys[0]
        data = groups[k]
        if data['xs']:
            plt.quiver(data['xs'], data['ys'], data['us'], data['vs'], 
                       angles='xy', scale_units='xy', scale=q_scale, 
                       color=color, width=0.003)
    else:
        # Use a qualitative colormap for distinct groups
        cmap = plt.get_cmap('tab10')
        
        for i, k in enumerate(sorted_keys):
            data = groups[k]
            if not data['xs']:
                continue
                
            if k == -1:
                c = color
                lbl = 'Unknown'
            else:
                c = cmap(i % 10)
                lbl = f"Pulse {k+1}"
            
            plt.quiver(data['xs'], data['ys'], data['us'], data['vs'], 
                       angles='xy', scale_units='xy', scale=q_scale, 
                       color=c, width=0.003, label=lbl)
        
        plt.legend(loc='upper right', fontsize='small', framealpha=0.8)

    plt.tight_layout()
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        except Exception:
            pass
        try:
            plt.savefig(save_path, dpi=150)
            print(f"[viz] Saved displacement overlay to: {save_path}")
        except Exception as e:
            print(f"[viz] Failed to save displacement overlay: {e}")
    plt.close()


def show_hartmann_mask(
    mask: Optional[np.ndarray],
    dx: float,
    dy: float,
    *,
    colormap: str = "gray",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    block: bool = True,
) -> None:
    """
    Display the defined Hartmann mask (layout) with axes in μm.
    - Uses grid spacing `dx`, `dy` to compute physical extent.
    - Labels and title in English; colorbar labeled as Transmission.
    """
    if mask is None:
        print("[viz] No Hartmann mask provided")
        return

    if not _ensure_matplotlib():
        return
    import matplotlib.pyplot as plt

    img = np.asarray(mask, dtype=np.float64)
    Ny, Nx = img.shape
    FOV_x = float(Nx) * float(dx)
    FOV_y = float(Ny) * float(dy)
    extent = [-(FOV_x * 1e6) / 2.0, (FOV_x * 1e6) / 2.0, -(FOV_y * 1e6) / 2.0, (FOV_y * 1e6) / 2.0]

    plt.figure(figsize=(6, 6))
    im = plt.imshow(img, extent=extent, origin='lower', cmap=colormap, aspect='equal')
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.colorbar(im, label="Transmission")
    plt.title(title or "Hartmann mask (layout)")
    plt.tight_layout()
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        except Exception:
            pass
        try:
            plt.savefig(save_path, dpi=150)
            print(f"[viz] Saved Hartmann mask to: {save_path}")
        except Exception as e:
            print(f"[viz] Failed to save Hartmann mask: {e}")
    plt.close()

def show_xcorr_scores(
    scores: Optional[np.ndarray],
    best_dx: Optional[int] = None,
    best_dy: Optional[int] = None,
    *,
    colormap: str = "viridis",
    title: Optional[str] = None,
    block: bool = True,
) -> None:
    if scores is None:
        print("[viz] No xcorr scores to display")
        return
    if not _ensure_matplotlib():
        return
    import matplotlib.pyplot as plt
    arr = np.asarray(scores, dtype=np.float64)
    R = (arr.shape[0] - 1) // 2
    extent = [-R, R, -R, R]
    plt.figure(figsize=(6, 5))
    im = plt.imshow(arr, extent=extent, origin='lower', cmap=colormap, aspect='equal')
    plt.xlabel("dx (px)")
    plt.ylabel("dy (px)")
    plt.colorbar(im, label="ZNCC score")
    plt.title(title or "Cross-correlation scores")
    if best_dx is not None and best_dy is not None:
        plt.scatter([best_dx], [best_dy], c="red", s=40, marker="x", label="best")
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.close()


def show_dpc_results(
    displacements: Sequence[dict],
    pixel_size: float,
    propagation_distance_m: float,
    nx: int,
    ny: int,
    *,
    detector_image: Optional[np.ndarray] = None,
    colormap_slope: str = "RdBu_r",
    colormap_image: str = "gray",
    title_prefix: str = "",
    save_dir: Optional[str] = None,
    block: bool = True,
    quiver_scale: float = 1.0,
    quiver_subsample: int = 1,
    verbose: bool = False,
    theoretical_phase: Optional[np.ndarray] = None,
    wavelength: Optional[float] = None,
    phase_map_size_m: Optional[float] = None,
) -> dict:
    """
    Display DPC (Differential Phase Contrast) results.
    
    DPC直接从位移场计算斜率（slope）：
        slope_x = dx_m / z
        slope_y = dy_m / z
    其中 dx_m, dy_m 是探测器上的位移（米），z 是传播距离。
    
    显示内容：
    1. Slope X 和 Slope Y 的2D colormap (插值热力图)
    2. 位移矢量场 (quiver plot)
    3. Slope magnitude (插值热力图)
    4. (可选) 理论DPC对比
    
    Parameters
    ----------
    displacements : Sequence[dict]
        位移数据列表
    pixel_size : float
        探测器像素尺寸 (m)
    propagation_distance_m : float
        传播距离 (m)
    nx, ny : int
        探测器尺寸 (像素)
    detector_image : Optional[np.ndarray]
        背景探测器图像
    colormap_slope : str
        斜率colormap
    colormap_image : str
        背景图像colormap
    title_prefix : str
        标题前缀
    save_dir : Optional[str]
        保存目录
    block : bool
        是否阻塞显示
    quiver_scale : float
        矢量放大系数
    quiver_subsample : int
        矢量子采样因子
    verbose : bool
        是否打印详细日志
    theoretical_phase : Optional[np.ndarray]
        理论相位图 (rad)，用于生成理论DPC参考图
    wavelength : Optional[float]
        波长 (m)，用于理论DPC计算 (slope = (λ/2π) * ∇φ)
        
    Returns
    -------
    dict
        包含计算的DPC数据
    """
    if not displacements:
        if verbose:
            print("[viz] No displacements for DPC visualization")
        return {}
    if not _ensure_matplotlib():
        return {}
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    # 提取数据
    x_ref_px = np.array([float(d.get('x_ref_px', 0.0)) for d in displacements])
    y_ref_px = np.array([float(d.get('y_ref_px', 0.0)) for d in displacements])
    dx_px = np.array([float(d.get('dx_px', 0.0)) for d in displacements])
    dy_px = np.array([float(d.get('dy_px', 0.0)) for d in displacements])
    
    # 转换为物理单位
    x_m = (x_ref_px - nx / 2.0 + 0.5) * pixel_size
    y_m = (y_ref_px - ny / 2.0 + 0.5) * pixel_size
    dx_m = dx_px * pixel_size
    dy_m = dy_px * pixel_size
    
    # 计算斜率 (DPC)
    # slope = displacement / propagation_distance
    slope_x_rad = dx_m / propagation_distance_m
    slope_y_rad = dy_m / propagation_distance_m
    slope_mag_rad = np.sqrt(slope_x_rad**2 + slope_y_rad**2)
    
    # 转换为 μrad 用于显示
    slope_x_urad = slope_x_rad * 1e6
    slope_y_urad = slope_y_rad * 1e6
    slope_mag_urad = slope_mag_rad * 1e6
    
    # 转换位置为 μm 用于显示
    x_um = x_m * 1e6
    y_um = y_m * 1e6
    
    FOV_x = float(nx) * float(pixel_size)
    FOV_y = float(ny) * float(pixel_size)
    extent = [-(FOV_x * 1e6) / 2.0, (FOV_x * 1e6) / 2.0, -(FOV_y * 1e6) / 2.0, (FOV_y * 1e6) / 2.0]
    
    # --- 插值生成平滑热力图 ---
    try:
        grid_size = 256
        Xi, Yi, Vi_x = _grid_interpolate(x_um, y_um, slope_x_urad, grid_size=grid_size)
        _, _, Vi_y = _grid_interpolate(x_um, y_um, slope_y_urad, grid_size=grid_size)
        _, _, Vi_mag = _grid_interpolate(x_um, y_um, slope_mag_urad, grid_size=grid_size)
        has_interp = True
    except Exception as e:
        if verbose:
            print(f"[viz] Interpolation failed: {e}, falling back to scatter")
        has_interp = False

    # --- Figure 1: Slope X and Slope Y (Heatmap or Scatter) ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
    
    # Slope X
    vmax_x = max(abs(slope_x_urad.min()), abs(slope_x_urad.max())) if len(slope_x_urad) > 0 else 1
    
    if has_interp:
        im0 = axes1[0].imshow(Vi_x, extent=[Xi.min(), Xi.max(), Yi.min(), Yi.max()], 
                              origin='lower', cmap=colormap_slope, vmin=-vmax_x, vmax=vmax_x)
        fig1.colorbar(im0, ax=axes1[0], label="Slope (μrad)")
    else:
        sc0 = axes1[0].scatter(x_um, y_um, c=slope_x_urad, cmap=colormap_slope, 
                               s=20, vmin=-vmax_x, vmax=vmax_x)
        fig1.colorbar(sc0, ax=axes1[0], label="Slope (μrad)")
        
    axes1[0].set_xlim(extent[0], extent[1])
    axes1[0].set_ylim(extent[2], extent[3])
    axes1[0].set_xlabel("x (μm)")
    axes1[0].set_ylabel("y (μm)")
    axes1[0].set_title(f"{title_prefix}Slope X (∂W/∂x)")
    axes1[0].set_aspect('equal')
    
    # Slope Y
    vmax_y = max(abs(slope_y_urad.min()), abs(slope_y_urad.max())) if len(slope_y_urad) > 0 else 1
    
    if has_interp:
        im1 = axes1[1].imshow(Vi_y, extent=[Xi.min(), Xi.max(), Yi.min(), Yi.max()], 
                              origin='lower', cmap=colormap_slope, vmin=-vmax_y, vmax=vmax_y)
        fig1.colorbar(im1, ax=axes1[1], label="Slope (μrad)")
    else:
        sc1 = axes1[1].scatter(x_um, y_um, c=slope_y_urad, cmap=colormap_slope,
                               s=20, vmin=-vmax_y, vmax=vmax_y)
        fig1.colorbar(sc1, ax=axes1[1], label="Slope (μrad)")

    axes1[1].set_xlim(extent[0], extent[1])
    axes1[1].set_ylim(extent[2], extent[3])
    axes1[1].set_xlabel("x (μm)")
    axes1[1].set_ylabel("y (μm)")
    axes1[1].set_title(f"{title_prefix}Slope Y (∂W/∂y)")
    axes1[1].set_aspect('equal')
    
    plt.tight_layout()
    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
            fig1.savefig(os.path.join(save_dir, 'dpc_slope_xy.png'), dpi=150)
        except Exception as e:
            if verbose:
                print(f"[viz] Failed to save DPC slope XY: {e}")
    plt.close()
    plt.close(fig1)
    
    # --- Figure 2: Quiver plot (Keep as is, useful for direction) ---
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    
    if detector_image is not None:
        img = np.asarray(detector_image, dtype=np.float64)
        ax2.imshow(img, extent=extent, origin='lower', cmap=colormap_image, aspect='equal', alpha=0.5)
    
    # 子采样
    idx = np.arange(0, len(x_um), quiver_subsample)
    x_sub = x_um[idx]
    y_sub = y_um[idx]
    sx_sub = slope_x_urad[idx]
    sy_sub = slope_y_urad[idx]
    mag_sub = slope_mag_urad[idx]
    
    # 使用magnitude着色
    q = ax2.quiver(x_sub, y_sub, sx_sub, sy_sub, mag_sub,
                   angles='xy', scale_units='xy', 
                   scale=1.0 / max(quiver_scale, 1e-6),
                   cmap='hot', width=0.003)
    fig2.colorbar(q, ax=ax2, label="Slope magnitude (μrad)")
    
    ax2.set_xlim(extent[0], extent[1])
    ax2.set_ylim(extent[2], extent[3])
    ax2.set_xlabel("x (μm)")
    ax2.set_ylabel("y (μm)")
    ax2.set_title(f"{title_prefix}DPC Vector Field")
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    if save_dir:
        try:
            fig2.savefig(os.path.join(save_dir, 'dpc_quiver.png'), dpi=150)
        except Exception as e:
            if verbose:
                print(f"[viz] Failed to save DPC quiver: {e}")
    plt.close()
    plt.close(fig2)
    
    # --- Figure 3: Slope magnitude (Heatmap) ---
    fig3, ax3 = plt.subplots(figsize=(7, 6))
    
    if has_interp:
        im3 = ax3.imshow(Vi_mag, extent=[Xi.min(), Xi.max(), Yi.min(), Yi.max()], 
                         origin='lower', cmap='viridis')
        fig3.colorbar(im3, ax=ax3, label="Slope (μrad)")
    else:
        sc3 = ax3.scatter(x_um, y_um, c=slope_mag_urad, cmap='viridis', s=20)
        fig3.colorbar(sc3, ax=ax3, label="Slope (μrad)")
        
    ax3.set_xlim(extent[0], extent[1])
    ax3.set_ylim(extent[2], extent[3])
    ax3.set_xlabel("x (μm)")
    ax3.set_ylabel("y (μm)")
    ax3.set_title(f"{title_prefix}Slope Magnitude |∇W|")
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    if save_dir:
        try:
            fig3.savefig(os.path.join(save_dir, 'dpc_magnitude.png'), dpi=150)
        except Exception as e:
            if verbose:
                print(f"[viz] Failed to save DPC magnitude: {e}")
    plt.close()
    plt.close(fig3)

    # --- Figure 4: Theoretical DPC (Optional) ---
    if theoretical_phase is not None and wavelength is not None:
        try:
            # 计算理论斜率
            # slope = (lambda / 2pi) * grad(phase)
            # 注意：phase map 是在仿真网格上生成的，不是探测器视场
            # 如果提供了 phase_map_size_m，使用它作为相位图的实际覆盖范围
            ny_t, nx_t = theoretical_phase.shape
            if phase_map_size_m is not None:
                # 使用实际的仿真窗口尺寸
                dx_t = phase_map_size_m / nx_t  # m
                dy_t = phase_map_size_m / ny_t  # m
            else:
                # 回退到探测器视场（可能不准确）
                dx_t = (extent[1] - extent[0]) * 1e-6 / nx_t # m
                dy_t = (extent[3] - extent[2]) * 1e-6 / ny_t # m
            
            grad_y, grad_x = np.gradient(theoretical_phase, dy_t, dx_t)
            
            theo_slope_x_rad = (wavelength / (2 * np.pi)) * grad_x
            theo_slope_y_rad = (wavelength / (2 * np.pi)) * grad_y
            theo_slope_mag_rad = np.sqrt(theo_slope_x_rad**2 + theo_slope_y_rad**2)
            
            theo_slope_x_urad = theo_slope_x_rad * 1e6
            theo_slope_y_urad = theo_slope_y_rad * 1e6
            theo_slope_mag_urad = theo_slope_mag_rad * 1e6
            
            fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
            
            # Theo Slope X
            im4_0 = axes4[0].imshow(theo_slope_x_urad, extent=extent, origin='lower', cmap=colormap_slope, vmin=-vmax_x, vmax=vmax_x)
            axes4[0].set_title("Theoretical Slope X")
            fig4.colorbar(im4_0, ax=axes4[0], label="μrad")
            
            # Theo Slope Y
            im4_1 = axes4[1].imshow(theo_slope_y_urad, extent=extent, origin='lower', cmap=colormap_slope, vmin=-vmax_y, vmax=vmax_y)
            axes4[1].set_title("Theoretical Slope Y")
            fig4.colorbar(im4_1, ax=axes4[1], label="μrad")
            
            # Theo Magnitude
            im4_2 = axes4[2].imshow(theo_slope_mag_urad, extent=extent, origin='lower', cmap='viridis')
            axes4[2].set_title("Theoretical Magnitude")
            fig4.colorbar(im4_2, ax=axes4[2], label="μrad")
            
            plt.tight_layout()
            if save_dir:
                fig4.savefig(os.path.join(save_dir, 'dpc_theoretical.png'), dpi=150)
            plt.close()
            plt.close(fig4)
            
        except Exception as e:
            if verbose:
                print(f"[viz] Failed to generate theoretical DPC: {e}")

    return {
        'slope_x_rad': slope_x_rad,
        'slope_y_rad': slope_y_rad,
        'slope_magnitude_rad': slope_mag_rad,
        'positions_m': (x_m, y_m),
        'positions_px': (x_ref_px, y_ref_px),
    }


# =============================================================================
# Enhanced DPC visualization helpers (heatmap + quiver + stats)
# =============================================================================
def _grid_interpolate(x: np.ndarray, y: np.ndarray, v: np.ndarray, grid_size: int = 200, method: str = "cubic"):
    """
    Interpolate scattered data onto a regular grid for smooth heatmaps.
    Returns Xi, Yi, Vi (2D grids) in the same units as x, y, v.
    """
    if len(x) < 3:
        raise ValueError("Not enough points for interpolation")
    xi = np.linspace(np.nanmin(x), np.nanmax(x), grid_size)
    yi = np.linspace(np.nanmin(y), np.nanmax(y), grid_size)
    Xi, Yi = np.meshgrid(xi, yi)
    from scipy.interpolate import griddata
    Vi = griddata((x, y), v, (Xi, Yi), method=method)
    return Xi, Yi, Vi


def _plot_heatmap_with_points(Xi, Yi, Vi, x, y, v, title: str, cmap: str, unit: str, save_path: Optional[str] = None):
    vmax = np.nanmax(np.abs(v))
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1e-9
    if not _ensure_matplotlib():
        return
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.imshow(
        Vi,
        extent=[Xi.min(), Xi.max(), Yi.min(), Yi.max()],
        origin="lower",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
    )
    plt.colorbar(label=f"Slope ({unit})")
    plt.scatter(x, y, c=v, cmap=cmap, vmin=-vmax, vmax=vmax, s=10, edgecolors='k', linewidths=0.2)
    plt.title(title)
    plt.xlabel(f"x ({unit})")
    plt.ylabel(f"y ({unit})")
    plt.tight_layout()
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=180)
        except Exception as e:
            print(f"[viz] Failed to save heatmap: {e}")
    plt.close()


def show_dpc_heatmap_and_quiver(
    positions_m: Tuple[np.ndarray, np.ndarray],
    slope_x_rad: np.ndarray,
    slope_y_rad: np.ndarray,
    *,
    title_prefix: str = "",
    save_dir: Optional[str] = None,
    quiver_subsample: int = 2,
    grid_size: int = 200,
    cmap: str = "seismic",
    unit_label: str = "m",
):
    """
    Render continuous heatmaps (interpolated) plus quiver overlays for DPC slopes.
    """
    x_m, y_m = positions_m
    # convert to microns for plotting convenience
    x_um = x_m * 1e6
    y_um = y_m * 1e6
    sx = slope_x_rad
    sy = slope_y_rad
    if len(x_um) < 3:
        print("[viz] Not enough points for heatmap interpolation.")
        return

    # Heatmaps
    Xi, Yi, Vi_x = _grid_interpolate(x_um, y_um, sx, grid_size=grid_size, method="cubic")
    _, _, Vi_y = _grid_interpolate(x_um, y_um, sy, grid_size=grid_size, method="cubic")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    _plot_heatmap_with_points(
        Xi,
        Yi,
        Vi_x,
        x_um,
        y_um,
        sx,
        f"{title_prefix}Slope X",
        cmap,
        "µm",
        _optional_save_path(save_dir, "heatmap_slope_x.png"),
    )
    _plot_heatmap_with_points(
        Xi,
        Yi,
        Vi_y,
        x_um,
        y_um,
        sy,
        f"{title_prefix}Slope Y",
        cmap,
        "µm",
        _optional_save_path(save_dir, "heatmap_slope_y.png"),
    )

    # Quiver on top of heatmap background (magnitude-coded arrows)
    if _ensure_matplotlib():
        import matplotlib.pyplot as plt
        idx = np.arange(0, len(x_um), max(quiver_subsample, 1))
        mag = np.sqrt(sx**2 + sy**2)
        scale = max(np.nanmax(mag[idx]), 1e-12)
        plt.figure(figsize=(6, 6))
        plt.imshow(
            Vi_x,
            extent=[Xi.min(), Xi.max(), Yi.min(), Yi.max()],
            origin="lower",
            cmap="Greys",
            alpha=0.4,
            aspect="equal",
        )
        plt.quiver(
            x_um[idx],
            y_um[idx],
            sx[idx] * 1e6,
            sy[idx] * 1e6,
            mag[idx] * 1e6,
            angles="xy",
            scale_units="xy",
            scale=1.0 / scale,
            cmap="plasma",
            width=0.0025,
        )
        plt.colorbar(label="Slope magnitude (µrad)")
        plt.xlabel("x (µm)")
        plt.ylabel("y (µm)")
        plt.title(f"{title_prefix}DPC heatmap + quiver")
        plt.tight_layout()
        if save_dir:
            try:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, "heatmap_quiver.png"), dpi=180)
            except Exception as e:
                print(f"[viz] Failed to save heatmap_quiver: {e}")
        plt.close()


def summarize_dpc_stats(sx: np.ndarray, sy: np.ndarray) -> dict:
    """
    Compute basic statistics for DPC slopes.
    Returns a dictionary with mean/std/max in microradians.
    """
    sx_u = sx * 1e6
    sy_u = sy * 1e6
    mag_u = np.sqrt(sx**2 + sy**2) * 1e6
    stats = {
        "slope_x_mean_urad": float(np.nanmean(sx_u)),
        "slope_x_std_urad": float(np.nanstd(sx_u)),
        "slope_x_max_urad": float(np.nanmax(np.abs(sx_u))),
        "slope_y_mean_urad": float(np.nanmean(sy_u)),
        "slope_y_std_urad": float(np.nanstd(sy_u)),
        "slope_y_max_urad": float(np.nanmax(np.abs(sy_u))),
        "slope_mag_mean_urad": float(np.nanmean(mag_u)),
        "slope_mag_std_urad": float(np.nanstd(mag_u)),
        "slope_mag_max_urad": float(np.nanmax(mag_u)),
    }
    return stats


def plot_dpc_comparison(
    positions_m: Tuple[np.ndarray, np.ndarray],
    meas_slope_x: np.ndarray,
    meas_slope_y: np.ndarray,
    theory_slope_x: np.ndarray,
    theory_slope_y: np.ndarray,
    *,
    save_dir: str,
    title_prefix: str = "",
):
    """
    Compare measured vs theoretical DPC: heatmaps and difference maps.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    x_m, y_m = positions_m
    x_um = x_m * 1e6
    y_um = y_m * 1e6

    # Differences
    dx = meas_slope_x - theory_slope_x
    dy = meas_slope_y - theory_slope_y

    # Heatmaps for measured/theory/diff
    for label, sx, sy in [
        ("measured", meas_slope_x, meas_slope_y),
        ("theory", theory_slope_x, theory_slope_y),
        ("diff", dx, dy),
    ]:
        Xi, Yi, Vi_x = _grid_interpolate(x_um, y_um, sx, grid_size=200, method="cubic")
        _, _, Vi_y = _grid_interpolate(x_um, y_um, sy, grid_size=200, method="cubic")
        _plot_heatmap_with_points(
            Xi,
            Yi,
            Vi_x,
            x_um,
            y_um,
            sx,
            f"{title_prefix}Slope X ({label})",
            "seismic",
            "µm",
            save_path=None if save_dir is None else os.path.join(save_dir, f"slope_x_{label}.png"),
        )
        _plot_heatmap_with_points(
            Xi,
            Yi,
            Vi_y,
            x_um,
            y_um,
            sy,
            f"{title_prefix}Slope Y ({label})",
            "seismic",
            "µm",
            save_path=None if save_dir is None else os.path.join(save_dir, f"slope_y_{label}.png"),
        )

    # Quiver comparison
    measured_dir = None if save_dir is None else os.path.join(save_dir, "measured")
    theory_dir = None if save_dir is None else os.path.join(save_dir, "theory")
    diff_dir = None if save_dir is None else os.path.join(save_dir, "diff")
    show_dpc_heatmap_and_quiver(positions_m, meas_slope_x, meas_slope_y, title_prefix=f"{title_prefix}Measured ", save_dir=measured_dir)
    show_dpc_heatmap_and_quiver(positions_m, theory_slope_x, theory_slope_y, title_prefix=f"{title_prefix}Theory ", save_dir=theory_dir)
    show_dpc_heatmap_and_quiver(positions_m, dx, dy, title_prefix=f"{title_prefix}Diff ", save_dir=diff_dir)


def show_spot_shifts(
    reference_centers: Sequence[dict],
    target_centers: Sequence[dict],
    displacements: Sequence[dict],
    pixel_size: float,
    nx: int,
    ny: int,
    *,
    pulse_idx: int = 0,
    amplification: float = 50.0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 10),
    arrow_width: float = 0.002,
    show_grid: bool = False,
    max_arrows: int = None,
) -> None:
    """
    显示光斑位移向量对比图
    
    显示参考点（无误差）、目标点（有误差）以及位移向量。
    
    Parameters
    ----------
    reference_centers : Sequence[dict]
        参考点位置列表 (无WFE)，每个dict包含 'x_px', 'y_px'
    target_centers : Sequence[dict]
        目标点位置列表 (有WFE)，每个dict包含 'x_px', 'y_px'
    displacements : Sequence[dict]
        位移数据列表，每个dict包含 'x_ref_px', 'y_ref_px', 'dx_px', 'dy_px', 'pulse_idx'
    pixel_size : float
        探测器像素尺寸 (m)
    nx, ny : int
        探测器像素数
    pulse_idx : int
        要显示的脉冲索引
    amplification : float
        位移向量放大倍数
    title : str, optional
        图标题
    save_path : str, optional
        保存路径
    figsize : tuple
        图尺寸
    arrow_width : float
        箭头宽度
    show_grid : bool
        是否显示网格
    max_arrows : int, optional
        最大显示箭头数，None表示全部显示
    """
    if not _ensure_matplotlib():
        return
    import matplotlib.pyplot as plt
    
    # 计算FOV (μm)
    FOV_x = float(nx) * float(pixel_size) * 1e6
    FOV_y = float(ny) * float(pixel_size) * 1e6
    half_x, half_y = FOV_x / 2.0, FOV_y / 2.0
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 转换参考点到μm坐标
    ref_x_um = []
    ref_y_um = []
    for c in reference_centers:
        x_px = float(c.get('x_px', c.get('cx', 0)))
        y_px = float(c.get('y_px', c.get('cy', 0)))
        x_um = (x_px - nx / 2.0 + 0.5) * pixel_size * 1e6
        y_um = (y_px - ny / 2.0 + 0.5) * pixel_size * 1e6
        ref_x_um.append(x_um)
        ref_y_um.append(y_um)
    
    # 转换目标点到μm坐标
    tgt_x_um = []
    tgt_y_um = []
    for c in target_centers:
        x_px = float(c.get('x_px', c.get('cx', 0)))
        y_px = float(c.get('y_px', c.get('cy', 0)))
        x_um = (x_px - nx / 2.0 + 0.5) * pixel_size * 1e6
        y_um = (y_px - ny / 2.0 + 0.5) * pixel_size * 1e6
        tgt_x_um.append(x_um)
        tgt_y_um.append(y_um)
    
    # 筛选指定脉冲的位移数据
    filtered_disps = [d for d in displacements if d.get('pulse_idx', 0) == pulse_idx]
    
    # 限制箭头数量
    if max_arrows is not None and len(filtered_disps) > max_arrows:
        step = len(filtered_disps) // max_arrows
        filtered_disps = filtered_disps[::step][:max_arrows]
    
    # 提取位移向量
    xs, ys, us, vs = [], [], [], []
    for d in filtered_disps:
        x_ref_px = float(d.get('x_ref_px', 0))
        y_ref_px = float(d.get('y_ref_px', 0))
        dx_px = float(d.get('dx_px', 0))
        dy_px = float(d.get('dy_px', 0))
        
        x_um = (x_ref_px - nx / 2.0 + 0.5) * pixel_size * 1e6
        y_um = (y_ref_px - ny / 2.0 + 0.5) * pixel_size * 1e6
        u_um = dx_px * pixel_size * 1e6
        v_um = dy_px * pixel_size * 1e6
        
        xs.append(x_um)
        ys.append(y_um)
        us.append(u_um)
        vs.append(v_um)
    
    # 绘制参考点
    ax.scatter(ref_x_um, ref_y_um, c='blue', s=15, marker='o', alpha=0.7, 
               label='Reference (No Error)', zorder=3)
    
    # 绘制目标点
    ax.scatter(tgt_x_um, tgt_y_um, c='red', s=15, marker='x', alpha=0.7,
               label='Target (With Error)', zorder=3)
    
    # 绘制位移向量
    if xs:
        q_scale = 1.0 / max(1e-6, float(amplification))
        ax.quiver(xs, ys, us, vs,
                  angles='xy', scale_units='xy', scale=q_scale,
                  color='green', width=arrow_width, alpha=0.8,
                  label=f'Shift (x{amplification:.1f})', zorder=2)
    
    # 设置坐标轴
    ax.set_xlim(-half_x, half_x)
    ax.set_ylim(-half_y, half_y)
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.set_aspect('equal')
    
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    # 图例
    ax.legend(loc='upper right', fontsize='small', framealpha=0.9)
    
    # 标题
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Spot Shifts - Pulse {pulse_idx}')
    
    plt.tight_layout()
    
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        except Exception:
            pass
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[viz] Saved: {save_path}")
    
    plt.close(fig)


def show_spot_shifts_clean(
    displacements: Sequence[dict],
    pixel_size: float,
    nx: int,
    ny: int,
    *,
    pulse_idx: int = None,
    amplification: float = 50.0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 10),
    arrow_color: str = 'green',
    point_color: str = 'blue',
    arrow_width: float = 0.003,
    show_grid: bool = True,
    max_arrows: int = None,
    show_magnitude_colorbar: bool = True,
) -> None:
    """
    简洁版位移向量图 - 仅显示参考点位置和位移向量
    
    Parameters
    ----------
    displacements : Sequence[dict]
        位移数据列表，每个dict包含 'x_ref_px', 'y_ref_px', 'dx_px', 'dy_px', 'pulse_idx'
    pixel_size : float
        探测器像素尺寸 (m)
    nx, ny : int
        探测器像素数
    pulse_idx : int, optional
        要显示的脉冲索引，None表示显示所有脉冲
    amplification : float
        位移向量放大倍数
    title : str, optional
        图标题
    save_path : str, optional
        保存路径
    figsize : tuple
        图尺寸
    arrow_color : str
        箭头颜色（当show_magnitude_colorbar=False时使用）
    point_color : str
        参考点颜色
    arrow_width : float
        箭头宽度
    show_grid : bool
        是否显示网格
    max_arrows : int, optional
        最大显示箭头数
    show_magnitude_colorbar : bool
        是否根据位移大小着色并显示colorbar
    """
    if not _ensure_matplotlib():
        return
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    # 计算FOV (μm)
    FOV_x = float(nx) * float(pixel_size) * 1e6
    FOV_y = float(ny) * float(pixel_size) * 1e6
    half_x, half_y = FOV_x / 2.0, FOV_y / 2.0
    
    # 筛选位移数据
    if pulse_idx is not None:
        filtered_disps = [d for d in displacements if d.get('pulse_idx', 0) == pulse_idx]
    else:
        filtered_disps = list(displacements)
    
    # 仅保留有效状态的点
    filtered_disps = [d for d in filtered_disps if d.get('status', 'ok') == 'ok']
    
    if not filtered_disps:
        print("[viz] No valid displacements to show")
        return
    
    # 限制箭头数量
    if max_arrows is not None and len(filtered_disps) > max_arrows:
        step = len(filtered_disps) // max_arrows
        filtered_disps = filtered_disps[::step][:max_arrows]
    
    # 提取位移向量
    xs, ys, us, vs, magnitudes = [], [], [], [], []
    for d in filtered_disps:
        x_ref_px = float(d.get('x_ref_px', 0))
        y_ref_px = float(d.get('y_ref_px', 0))
        dx_px = float(d.get('dx_px', 0))
        dy_px = float(d.get('dy_px', 0))
        
        x_um = (x_ref_px - nx / 2.0 + 0.5) * pixel_size * 1e6
        y_um = (y_ref_px - ny / 2.0 + 0.5) * pixel_size * 1e6
        u_um = dx_px * pixel_size * 1e6
        v_um = dy_px * pixel_size * 1e6
        mag = np.sqrt(u_um**2 + v_um**2)
        
        xs.append(x_um)
        ys.append(y_um)
        us.append(u_um)
        vs.append(v_um)
        magnitudes.append(mag)
    
    xs, ys, us, vs = np.array(xs), np.array(ys), np.array(us), np.array(vs)
    magnitudes = np.array(magnitudes)
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 绘制参考点
    ax.scatter(xs, ys, c=point_color, s=8, marker='o', alpha=0.5, zorder=2)
    
    # 绘制位移向量
    q_scale = 1.0 / max(1e-6, float(amplification))
    
    if show_magnitude_colorbar and len(magnitudes) > 0:
        # 按位移大小着色
        norm = Normalize(vmin=magnitudes.min(), vmax=magnitudes.max())
        cmap = cm.get_cmap('viridis')
        colors = cmap(norm(magnitudes))
        
        q = ax.quiver(xs, ys, us, vs,
                      angles='xy', scale_units='xy', scale=q_scale,
                      color=colors, width=arrow_width, alpha=0.9, zorder=3)
        
        # 添加colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Displacement (μm)', fontsize=10)
    else:
        ax.quiver(xs, ys, us, vs,
                  angles='xy', scale_units='xy', scale=q_scale,
                  color=arrow_color, width=arrow_width, alpha=0.8, zorder=3)
    
    # 设置坐标轴
    ax.set_xlim(-half_x, half_x)
    ax.set_ylim(-half_y, half_y)
    ax.set_xlabel('x (μm)', fontsize=11)
    ax.set_ylabel('y (μm)', fontsize=11)
    ax.set_aspect('equal')
    
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # 标题
    if title:
        ax.set_title(title, fontsize=12)
    else:
        pulse_str = f'Pulse {pulse_idx}' if pulse_idx is not None else 'All Pulses'
        ax.set_title(f'Spot Displacements - {pulse_str} (x{amplification:.0f})', fontsize=12)
    
    # 添加统计信息
    mean_mag = np.mean(magnitudes)
    max_mag = np.max(magnitudes)
    stats_text = f'N={len(xs)}, Mean={mean_mag:.3f}μm, Max={max_mag:.3f}μm'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        except Exception:
            pass
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[viz] Saved: {save_path}")
    
    plt.close(fig)
