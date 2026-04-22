from __future__ import annotations
"""
Visualization Module for Hartmann Wavefront Analysis

Provides comprehensive visualization outputs including:
- Summary comparison plots (truth vs recovered vs error)
- DPC (Differential Phase Contrast) visualization
- Method comparison charts
- Accumulated image visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
import os
from plot_style import apply_plot_font_bump

apply_plot_font_bump()

# 设置全局样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 150,
    'image.interpolation': 'nearest',
})

# 自定义配色
CMAP_PHASE = 'RdBu_r'  # 相位用红蓝
CMAP_ERROR = 'RdBu'  # 误差用对称的红蓝
CMAP_MAGNITUDE = 'inferno'  # 幅度用暖色
CMAP_IMAGE = 'gray'  # 图像用灰度


def _symmetric_vlimits(data: np.ndarray) -> Tuple[float | None, float | None]:
    """Return symmetric visualization limits around zero when data is non-zero."""
    vmax = float(np.nanmax(np.abs(data)))
    if vmax > 0:
        return -vmax, vmax
    return None, None


def _hide_image_axes(ax) -> None:
    """Hide x/y ticks for image-like panels."""
    ax.set_xticks([])
    ax.set_yticks([])


def _style_dark_panel(ax) -> None:
    """Apply the shared dark-panel styling used by accumulated-image views."""
    ax.set_facecolor('black')
    _hide_image_axes(ax)
    for spine in ax.spines.values():
        spine.set_color('white')


def _imshow_with_colorbar(
    ax,
    data: np.ndarray,
    *,
    title: str,
    unit: str,
    cmap: str,
    symmetric: bool = False,
    title_color: str | None = None,
    colorbar_label_color: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Render an image panel with a colorbar and hidden axis ticks."""
    if symmetric and vmin is None and vmax is None:
        vmin, vmax = _symmetric_vlimits(data)
    im = ax.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    title_kwargs = {"fontsize": 10}
    if title_color is not None:
        title_kwargs["color"] = title_color
    ax.set_title(title, **title_kwargs)
    _hide_image_axes(ax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label_color is not None:
        cbar.set_label(unit, fontsize=8, color=colorbar_label_color)
        cbar.ax.yaxis.set_tick_params(color=colorbar_label_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=colorbar_label_color)
    else:
        cbar.set_label(unit, fontsize=8)
        cbar.ax.tick_params(labelsize=8)
    return im, cbar


def _save_visualization_figure(fig, output_path: str, **savefig_kwargs) -> None:
    """Save, close, and log a visualization figure."""
    fig.savefig(output_path, dpi=150, bbox_inches='tight', **savefig_kwargs)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _comparison_fields_um(
    recovered: Dict[str, np.ndarray],
    truth: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Build shared phase/displacement truth-recovered-error fields for display."""
    recovered_phase_rad = np.asarray(recovered["phase_rad"], dtype=np.float64)
    truth_phase_rad = np.asarray(truth["phase_rad"], dtype=np.float64)
    recovered_displacement_x_um = np.asarray(recovered["displacement_x_m"], dtype=np.float64) * 1e6
    truth_displacement_x_um = np.asarray(truth["displacement_x_m"], dtype=np.float64) * 1e6
    recovered_displacement_y_um = np.asarray(recovered["displacement_y_m"], dtype=np.float64) * 1e6
    truth_displacement_y_um = np.asarray(truth["displacement_y_m"], dtype=np.float64) * 1e6
    return {
        "phase_truth_rad": truth_phase_rad,
        "phase_recovered_rad": recovered_phase_rad,
        "phase_error_rad": recovered_phase_rad - truth_phase_rad,
        "displacement_x_truth_um": truth_displacement_x_um,
        "displacement_x_recovered_um": recovered_displacement_x_um,
        "displacement_x_error_um": recovered_displacement_x_um - truth_displacement_x_um,
        "displacement_y_truth_um": truth_displacement_y_um,
        "displacement_y_recovered_um": recovered_displacement_y_um,
        "displacement_y_error_um": recovered_displacement_y_um - truth_displacement_y_um,
    }


def _displacement_slopes_mrad(
    recovered: Dict[str, np.ndarray],
    truth: Dict[str, np.ndarray],
    z_prop: float,
) -> Dict[str, np.ndarray]:
    """Convert displacement fields to DPC-like slope fields in mrad."""
    scale = 1e3 / float(z_prop)
    return {
        "slope_x_truth_mrad": np.asarray(truth["displacement_x_m"], dtype=np.float64) * scale,
        "slope_y_truth_mrad": np.asarray(truth["displacement_y_m"], dtype=np.float64) * scale,
        "slope_x_recovered_mrad": np.asarray(recovered["displacement_x_m"], dtype=np.float64) * scale,
        "slope_y_recovered_mrad": np.asarray(recovered["displacement_y_m"], dtype=np.float64) * scale,
    }


def create_summary_comparison(
    recovered: Dict[str, np.ndarray],
    truth: Dict[str, np.ndarray],
    output_path: str,
    title: str = "Wavefront Reconstruction Summary"
) -> None:
    """
    Create a 3x3 summary comparison plot:
    - Row 1: Phase (truth, recovered, error)
    - Row 2: DPC-X (truth, recovered, error)
    - Row 3: DPC-Y (truth, recovered, error)
    """
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3, hspace=0.3)
    fields = _comparison_fields_um(recovered, truth)
    
    # Row 1: Phase
    _add_subplot(fig, gs[0, 0], fields["phase_truth_rad"], "Truth Phase", "rad", symmetric=True, cmap=CMAP_PHASE)
    _add_subplot(fig, gs[0, 1], fields["phase_recovered_rad"], "Recovered Phase", "rad", symmetric=True, cmap=CMAP_PHASE)
    _add_subplot(fig, gs[0, 2], fields["phase_error_rad"], "Phase Error", "rad", symmetric=True, cmap=CMAP_ERROR)
    
    # Row 2: Displacement X
    _add_subplot(fig, gs[1, 0], fields["displacement_x_truth_um"], "Truth Disp-X", "µm", symmetric=True, cmap=CMAP_PHASE)
    _add_subplot(fig, gs[1, 1], fields["displacement_x_recovered_um"], "Recovered Disp-X", "µm", symmetric=True, cmap=CMAP_PHASE)
    _add_subplot(fig, gs[1, 2], fields["displacement_x_error_um"], "Disp-X Error", "µm", symmetric=True, cmap=CMAP_ERROR)
    
    # Row 3: Displacement Y
    _add_subplot(fig, gs[2, 0], fields["displacement_y_truth_um"], "Truth Disp-Y", "µm", symmetric=True, cmap=CMAP_PHASE)
    _add_subplot(fig, gs[2, 1], fields["displacement_y_recovered_um"], "Recovered Disp-Y", "µm", symmetric=True, cmap=CMAP_PHASE)
    _add_subplot(fig, gs[2, 2], fields["displacement_y_error_um"], "Disp-Y Error", "µm", symmetric=True, cmap=CMAP_ERROR)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    _save_visualization_figure(fig, output_path)


def _add_subplot(fig, gs_cell, data, title, unit, symmetric=False, cmap='coolwarm'):
    """Add a subplot to the figure."""
    ax = fig.add_subplot(gs_cell)
    im, cbar = _imshow_with_colorbar(
        ax,
        data,
        title=title,
        unit=unit,
        cmap=cmap,
        symmetric=symmetric,
    )
    cbar.ax.tick_params(labelsize=8)
    return ax


def _plot_error_histogram(ax, values: np.ndarray, *, color: str, xlabel: str, title: str, unit: str) -> None:
    """Plot a 1D error histogram with RMS in the title and a zero reference line."""
    ax.hist(values.ravel(), bins=50, color=color, edgecolor='white')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(f'{title}\nRMS={np.std(values):.4f} {unit}')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)


def _plot_error_field(ax, values: np.ndarray, *, title: str, unit: str) -> None:
    """Plot a symmetric spatial error field with a colorbar."""
    _imshow_with_colorbar(
        ax,
        values,
        title=title,
        unit=unit,
        cmap='coolwarm',
        symmetric=True,
    )


def _plot_dark_image_with_colorbar(
    ax,
    data: np.ndarray,
    *,
    title: str,
    cmap: str,
    label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Plot a dark-background image panel with a styled colorbar."""
    _imshow_with_colorbar(
        ax,
        data,
        title=title,
        unit=label,
        cmap=cmap,
        title_color='white',
        colorbar_label_color='white',
        vmin=vmin,
        vmax=vmax,
    )
    _style_dark_panel(ax)


def _imshow_rgb_panel(ax, rgb_image: np.ndarray, *, title: str) -> None:
    """Render an RGB image panel with the standard hidden-axis styling."""
    ax.imshow(rgb_image, origin='lower')
    ax.set_title(title, fontsize=12)
    _hide_image_axes(ax)


def create_dpc_visualization(
    recovered: Dict[str, np.ndarray],
    truth: Dict[str, np.ndarray],
    output_dir: str,
    z_prop: float,
    wavelength: float,
) -> None:
    """
    Create DPC (Differential Phase Contrast) visualization.
    
    DPC = phase gradient = wavefront slope
    """
    os.makedirs(output_dir, exist_ok=True)
    
    slopes = _displacement_slopes_mrad(recovered, truth, z_prop)
    slope_x_rec = slopes["slope_x_recovered_mrad"]
    slope_y_rec = slopes["slope_y_recovered_mrad"]
    slope_x_truth = slopes["slope_x_truth_mrad"]
    slope_y_truth = slopes["slope_y_truth_mrad"]
    
    # DPC magnitude
    dpc_mag_rec = np.sqrt(slope_x_rec**2 + slope_y_rec**2)
    dpc_mag_truth = np.sqrt(slope_x_truth**2 + slope_y_truth**2)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # Row 1: Truth
    _plot_field(axes[0, 0], slope_x_truth, "Truth DPC-X", "mrad", cmap=CMAP_PHASE, symmetric=True)
    _plot_field(axes[0, 1], slope_y_truth, "Truth DPC-Y", "mrad", cmap=CMAP_PHASE, symmetric=True)
    _plot_field(axes[0, 2], dpc_mag_truth, "Truth DPC Magnitude", "mrad", cmap=CMAP_MAGNITUDE)
    
    # Row 2: Recovered
    _plot_field(axes[1, 0], slope_x_rec, "Recovered DPC-X", "mrad", cmap=CMAP_PHASE, symmetric=True)
    _plot_field(axes[1, 1], slope_y_rec, "Recovered DPC-Y", "mrad", cmap=CMAP_PHASE, symmetric=True)
    _plot_field(axes[1, 2], dpc_mag_rec, "Recovered DPC Magnitude", "mrad", cmap=CMAP_MAGNITUDE)
    
    plt.suptitle("Differential Phase Contrast (DPC) Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_visualization_figure(fig, os.path.join(output_dir, "dpc_analysis.png"))
    
    # Create HSV direction plot
    _create_dpc_direction_plot(
        slope_x_rec, slope_y_rec,
        os.path.join(output_dir, "dpc_direction.png"),
        title="DPC Direction (HSV encoding)"
    )


def _plot_field(ax, data, title, unit, cmap='coolwarm', symmetric=False):
    """Plot a scalar field using the shared image-panel helper."""
    _imshow_with_colorbar(
        ax,
        data,
        title=title,
        unit=unit,
        cmap=cmap,
        symmetric=symmetric,
    )


def _sensor_image_display_limits(image: np.ndarray) -> Tuple[float, float]:
    """Return robust percentile display limits for sensor-image views."""
    return float(np.percentile(image, 1)), float(np.percentile(image, 99.5))


def _overlay_centroids(
    ax,
    centroids_xy_px: Optional[np.ndarray],
    *,
    markersize: float,
    alpha: float,
    show_legend: bool = False,
) -> None:
    """Overlay optional centroid markers using the project-wide (x, y) convention."""
    if centroids_xy_px is None or centroids_xy_px.shape[1] != 2:
        return
    ax.plot(
        centroids_xy_px[:, 0],
        centroids_xy_px[:, 1],
        'g+',
        markersize=markersize,
        alpha=alpha,
        label='Centroids',
    )
    if show_legend:
        ax.legend(loc='upper right')


def _clipped_zoom_bounds(
    *,
    center_x_px: int,
    center_y_px: int,
    zoom_size_px: int,
    image_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Return clipped `(x1, x2, y1, y2)` bounds for a zoom inset."""
    image_rows, image_cols = image_shape
    x1 = max(0, center_x_px - zoom_size_px)
    x2 = min(image_cols, center_x_px + zoom_size_px)
    y1 = max(0, center_y_px - zoom_size_px)
    y2 = min(image_rows, center_y_px + zoom_size_px)
    return x1, x2, y1, y2


def _add_sensor_zoom_inset(
    ax,
    image: np.ndarray,
    *,
    centroids_xy_px: Optional[np.ndarray],
    vmin: float,
    vmax: float,
    center_x_px: int,
    center_y_px: int,
    zoom_size_px: int,
    zoom_factor: float,
    loc: str,
    mark_loc1: int,
    mark_loc2: int,
    borderpad: float = 0.5,
) -> None:
    """Add a reusable zoomed inset panel to the main sensor-image figure."""
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

    axins = zoomed_inset_axes(ax, zoom_factor, loc=loc, borderpad=borderpad)
    axins.imshow(image, cmap='magma', origin='lower', vmin=vmin, vmax=vmax)
    _overlay_centroids(axins, centroids_xy_px, markersize=8, alpha=0.8)

    x1, x2, y1, y2 = _clipped_zoom_bounds(
        center_x_px=center_x_px,
        center_y_px=center_y_px,
        zoom_size_px=zoom_size_px,
        image_shape=image.shape,
    )
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    _hide_image_axes(axins)
    mark_inset(ax, axins, loc1=mark_loc1, loc2=mark_loc2, fc="none", ec="white", alpha=0.5)


def _sensor_zoom_centers(image_shape: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
    """Return reusable center/corner zoom centers and zoom size for sensor-image insets."""
    image_rows, image_cols = image_shape
    zoom_size_px = min(image_rows, image_cols) // 10
    center_zoom_xy = (image_cols // 2, image_rows // 2)
    margin_px = min(image_rows, image_cols) // 8
    corner_zoom_xy = (margin_px, image_rows - margin_px)
    return center_zoom_xy, corner_zoom_xy, zoom_size_px


def _create_dpc_direction_plot(dpc_x, dpc_y, output_path, title="DPC Direction"):
    """
    Create HSV-encoded DPC direction plot.
    Hue = direction, Saturation = 1, Value = magnitude (normalized)
    """
    magnitude = np.sqrt(dpc_x**2 + dpc_y**2)
    direction = np.arctan2(dpc_y, dpc_x)  # -pi to pi
    
    # Normalize magnitude for value channel
    mag_norm = magnitude / (np.nanmax(magnitude) + 1e-10)
    
    # Convert direction to hue (0 to 1)
    hue = (direction + np.pi) / (2 * np.pi)
    
    # Create HSV image
    hsv = np.zeros((*dpc_x.shape, 3))
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = 1.0  # Full saturation
    hsv[:, :, 2] = mag_norm  # Value = normalized magnitude
    
    # Convert to RGB
    rgb = mcolors.hsv_to_rgb(hsv)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    _imshow_rgb_panel(ax, rgb, title=title)
    
    # Add colorwheel legend
    _add_colorwheel_legend(fig)
    
    _save_visualization_figure(fig, output_path)


def _add_colorwheel_legend(fig):
    """Add a small colorwheel legend to show direction encoding."""
    # Create inset axes for colorwheel
    ax_cw = fig.add_axes([0.85, 0.15, 0.1, 0.1], polar=True)
    
    # Create colorwheel
    n = 256
    theta_centers = np.linspace(0, 2*np.pi, n, endpoint=False)
    width = 2 * np.pi / n
    radial_edges = np.linspace(0, 1, 3)
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    
    # Hue based on angle
    hsv_cw = np.zeros((radial_centers.size, n, 3))
    hsv_cw[:, :, 0] = theta_centers[None, :] / (2 * np.pi)
    hsv_cw[:, :, 1] = 1.0
    hsv_cw[:, :, 2] = radial_centers[:, None]
    
    rgb_cw = mcolors.hsv_to_rgb(hsv_cw)

    for ring_idx, bottom in enumerate(radial_edges[:-1]):
        height = radial_edges[ring_idx + 1] - bottom
        bars = ax_cw.bar(
            theta_centers,
            np.full(n, height, dtype=np.float64),
            width=width,
            bottom=bottom,
            align='edge',
            linewidth=0.0,
        )
        for bar, color in zip(bars, rgb_cw[ring_idx]):
            bar.set_facecolor(color)
            bar.set_edgecolor(color)

    ax_cw.set_yticks([])
    ax_cw.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax_cw.set_xticklabels(['→', '↑', '←', '↓'], fontsize=8)
    ax_cw.set_title('Dir', fontsize=8)


def create_method_comparison(
    metrics_per_method: Dict[str, List],
    output_path: str,
) -> None:
    """
    Create method comparison bar chart.
    """
    methods = list(metrics_per_method.keys())
    
    # Calculate average metrics
    avg_rmse_shift = []
    avg_phase_corr = []
    avg_rmse_phase = []
    
    for method in methods:
        metrics_list = metrics_per_method[method]
        if metrics_list:
            avg_rmse_shift.append(np.mean([m.rmse_shift_px for m in metrics_list]))
            avg_phase_corr.append(np.mean([m.corr_phase for m in metrics_list]))
            avg_rmse_phase.append(np.mean([m.rmse_phase_rad for m in metrics_list]))
        else:
            avg_rmse_shift.append(0)
            avg_phase_corr.append(0)
            avg_rmse_phase.append(0)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    x = np.arange(len(methods))
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    _plot_method_metric_bars(
        axes[0],
        x_positions=x,
        methods=methods,
        values=avg_rmse_shift,
        colors=colors,
        ylabel='RMSE Shift (px)',
        title='Shift Accuracy',
        yscale='log',
    )
    _plot_method_metric_bars(
        axes[1],
        x_positions=x,
        methods=methods,
        values=avg_phase_corr,
        colors=colors,
        ylabel='Phase Correlation',
        title='Phase Reconstruction Quality',
        ylim=(0, 1.1),
        reference_line=1.0,
    )
    _plot_method_metric_bars(
        axes[2],
        x_positions=x,
        methods=methods,
        values=avg_rmse_phase,
        colors=colors,
        ylabel='RMSE Phase (rad)',
        title='Phase Error',
    )
    
    plt.suptitle("Method Comparison", fontsize=14, fontweight='bold')
    # Inset axes from zoom views do not cooperate with tight_layout().
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.94)
    _save_visualization_figure(fig, output_path)


def _plot_method_metric_bars(
    ax,
    *,
    x_positions: np.ndarray,
    methods: List[str],
    values: List[float],
    colors,
    ylabel: str,
    title: str,
    yscale: str | None = None,
    ylim: Tuple[float, float] | None = None,
    reference_line: float | None = None,
) -> None:
    """Render one method-comparison bar panel with shared axis styling."""
    ax.bar(x_positions, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    if yscale is not None:
        ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if reference_line is not None:
        ax.axhline(y=reference_line, color='gray', linestyle='--', alpha=0.5)


def create_accumulated_visualization(
    accumulated_image: np.ndarray,
    ideal_accumulated_image: np.ndarray,
    output_dir: str,
) -> None:
    """
    Create accumulated image visualization with normalized purple/magenta colormap.
    """
    import matplotlib.colors as mcolors
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建配色
    cmap_accumulated = 'inferno'  # 黑-红-黄，适合显示光强累积
    cmap_diff = 'RdBu'  # 红蓝配色，适合显示差异
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='black')
    
    # 归一化累积图
    acc_norm = accumulated_image / (accumulated_image.max() + 1e-10)
    _plot_dark_image_with_colorbar(
        axes[0],
        acc_norm,
        title="Accumulated Image (normalized)",
        cmap=cmap_accumulated,
        label='Intensity (a.u.)',
        vmin=0,
        vmax=1,
    )
    
    ideal_norm = ideal_accumulated_image / (ideal_accumulated_image.max() + 1e-10)
    _plot_dark_image_with_colorbar(
        axes[1],
        ideal_norm,
        title="Ideal Accumulated Image (normalized)",
        cmap=cmap_accumulated,
        label='Intensity (a.u.)',
        vmin=0,
        vmax=1,
    )
    
    diff = acc_norm - ideal_norm
    vmax = np.abs(diff).max()
    _plot_dark_image_with_colorbar(
        axes[2],
        diff,
        title="Difference",
        cmap=cmap_diff,
        label='Δ (normalized)',
        vmin=-vmax,
        vmax=vmax,
    )
    
    for ax in axes:
        _style_dark_panel(ax)
    
    plt.tight_layout()
    _save_visualization_figure(
        fig,
        os.path.join(output_dir, "accumulated_comparison.png"),
        facecolor='black',
        edgecolor='none',
    )


def create_error_statistics(
    recovered: Dict[str, np.ndarray],
    truth: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """
    Create error statistics visualization.
    """
    fields = _comparison_fields_um(recovered, truth)
    phase_error_rad = fields["phase_error_rad"]
    displacement_x_error_um = fields["displacement_x_error_um"]
    displacement_y_error_um = fields["displacement_y_error_um"]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # Top row: Histograms
    _plot_error_histogram(
        axes[0, 0],
        phase_error_rad,
        color='steelblue',
        xlabel='Phase Error (rad)',
        title='Phase Error Distribution',
        unit='rad',
    )
    _plot_error_histogram(
        axes[0, 1],
        displacement_x_error_um,
        color='coral',
        xlabel='Disp-X Error (µm)',
        title='Disp-X Error Distribution',
        unit='µm',
    )
    _plot_error_histogram(
        axes[0, 2],
        displacement_y_error_um,
        color='seagreen',
        xlabel='Disp-Y Error (µm)',
        title='Disp-Y Error Distribution',
        unit='µm',
    )
    
    # Bottom row: Spatial distribution
    _plot_error_field(axes[1, 0], phase_error_rad, title='Phase Error (Spatial)', unit='rad')
    _plot_error_field(axes[1, 1], displacement_x_error_um, title='Disp-X Error (Spatial)', unit='µm')
    _plot_error_field(axes[1, 2], displacement_y_error_um, title='Disp-Y Error (Spatial)', unit='µm')
    
    for ax in axes[1, :]:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle("Error Statistics", fontsize=14, fontweight='bold')
    # Inset axes from zoom views do not cooperate with tight_layout().
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.94)
    _save_visualization_figure(fig, output_path)


def create_sensor_image_visualization(
    image: np.ndarray,
    output_path: str,
    title: str = "Hartmann Sensor Image",
    centroids: Optional[np.ndarray] = None,
    pixel_size_um: Optional[float] = None,
) -> None:
    """
    Create a detailed visualization of the sensor image with zoom-in insets.
    
    Args:
        image: Sensor image array
        output_path: Path to save the PNG
        title: Plot title
        centroids: Optional [N, 2] array of (x, y) centroids to overlay
        pixel_size_um: Optional pixel size for scale bar
    """
    # Normalize image for display
    vmin, vmax = _sensor_image_display_limits(image)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    _imshow_with_colorbar(
        ax,
        image,
        title=title,
        unit='Counts',
        cmap='magma',
        vmin=vmin,
        vmax=vmax,
    )
    
    # Overlay centroids if provided
    _overlay_centroids(ax, centroids, markersize=4, alpha=0.6, show_legend=True)
    
    # --- Add Zoom Insets ---
    # We'll add 2 insets: Center and one corner (e.g., top-left)
    
    # 1. Center Zoom
    (center_x_px, center_y_px), (corner_x_px, corner_y_px), zoom_size_px = _sensor_zoom_centers(image.shape)
    
    _add_sensor_zoom_inset(
        ax,
        image,
        centroids_xy_px=centroids,
        vmin=vmin,
        vmax=vmax,
        center_x_px=center_x_px,
        center_y_px=center_y_px,
        zoom_size_px=zoom_size_px,
        zoom_factor=3,
        loc='lower right',
        mark_loc1=2,
        mark_loc2=4,
    )
    
    # 2. Corner Zoom (Top Left area)
    # Pick a region with signal (some distance from edge)
    _add_sensor_zoom_inset(
        ax,
        image,
        centroids_xy_px=centroids,
        vmin=vmin,
        vmax=vmax,
        center_x_px=corner_x_px,
        center_y_px=corner_y_px,
        zoom_size_px=zoom_size_px,
        zoom_factor=3,
        loc='upper left',
        mark_loc1=3,
        mark_loc2=1,
        borderpad=2,
    )

    # Inset axes from zoom views do not cooperate with tight_layout().
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.94)
    _save_visualization_figure(fig, output_path)
