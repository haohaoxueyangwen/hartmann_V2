from __future__ import annotations
"""
Visualization script for Hartmann analysis results.
Creates error plots for all pulses and statistical summary figures.
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from plot_style import apply_plot_font_bump

apply_plot_font_bump()


_PREFERRED_METHOD_ORDER = [
    "centroid",
    "cross_correlation",
    "phase_correlation",
    "optimization_direct",
    "optimization_zernike",
]

_METHOD_DISPLAY_NAMES = {
    "centroid": "Centroid",
    "cross_correlation": "Cross-Corr",
    "phase_correlation": "Phase-Corr",
    "optimization_direct": "Opt-Direct",
    "optimization_zernike": "Opt-Zernike",
}


def _save_figure(fig: plt.Figure, output_dir: str, filename: str) -> None:
    """Save and close a figure with a consistent log message."""
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)
    print(f"Saved: {filename}")


def _print_visualization_notice(message: str) -> None:
    """Print a consistent visualization CLI notice."""
    print(message)


def _print_visualization_header(title: str, *, width: int = 70) -> None:
    """Print a shared boxed header for CLI visualization summaries."""
    border = "=" * width
    print("\n" + border)
    print(title)
    print(border)


def _print_visualization_table_header(columns: str, *, width: int = 70) -> None:
    """Print a shared table header separator for CLI summaries."""
    print(columns)
    print("-" * width)


def _print_visualization_section(title: str) -> None:
    """Print a compact titled section within a CLI visualization summary."""
    print(f"\n--- {title} ---")


def _imshow_panel(ax, data: np.ndarray, *, title: str, cmap: str = 'RdBu_r', symmetric: bool = False) -> None:
    """Render a pulse-map image panel with a matching colorbar."""
    if symmetric:
        vmax = float(np.max(np.abs(data)))
        vmin = -vmax
    else:
        vmin = vmax = None
    im = ax.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)


def _load_pulse_arrays_or_notice(analysis_dir: str, pulse_idx: int, method: str) -> Dict[str, np.ndarray] | None:
    """Load a pulse artifact bundle or print a consistent missing-artifact notice."""
    data = load_pulse_arrays(analysis_dir, pulse_idx, method)
    if data is None:
        _print_visualization_notice(
            f"Skipping pulse {pulse_idx} method '{method}': no saved arrays found"
        )
        return None
    return data


def _pulse_case_records(records: List[Dict], *, include_accumulated: bool = False) -> List[Dict]:
    """Filter metric records down to pulse-level entries, optionally keeping accumulated."""
    if include_accumulated:
        return [record for record in records if record.get("case")]
    return [record for record in records if record.get("case") not in (None, "accumulated")]


def _metric_values(records: List[Dict], metric_key: str) -> np.ndarray:
    """Extract a metric series as a float array from a record list."""
    return np.asarray([record[metric_key] for record in records], dtype=np.float64)


def _pulse_methods_or_none(analysis_dir: str, pulse_idx: int) -> List[str] | None:
    """Return available pulse methods or print a consistent empty-state message."""
    methods = discover_available_methods(analysis_dir, pulse_idx=pulse_idx)
    if methods:
        return methods
    _print_visualization_notice(f"No pulse reconstructions found for pulse {pulse_idx}")
    return None


def _metric_methods_or_none(analysis_dir: str, by_method: Dict[str, List[Dict]]) -> List[str] | None:
    """Return metric-bearing methods or print a consistent empty-state notice."""
    methods = _ordered_metric_methods(analysis_dir, by_method)
    if methods:
        return methods
    _print_visualization_notice("No metric-bearing methods found")
    return None


def _method_stats_or_none(
    analysis_dir: str,
    by_method: Dict[str, List[Dict]],
) -> Dict[str, Dict[str, float]] | None:
    """Return aggregated method statistics or print a consistent empty-state message."""
    method_stats = _compute_method_statistics(analysis_dir, by_method)
    if method_stats:
        return method_stats
    _print_visualization_notice("No valid method statistics found")
    return None


def _comparison_methods_or_none(method_stats: Dict[str, Dict[str, float]]) -> List[str] | None:
    """Return comparison methods or print a consistent shortfall message."""
    methods = list(method_stats.keys())
    if len(methods) >= 2:
        return methods
    _print_visualization_notice("Not enough methods available for comparison")
    return None


def _pulse_displacement_fields_um(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Build displacement truth/recovered fields in micrometers for pulse plots."""
    recovered_displacement_x_um = np.asarray(data["displacement_x_m"], dtype=np.float64) * 1e6
    recovered_displacement_y_um = np.asarray(data["displacement_y_m"], dtype=np.float64) * 1e6
    truth_displacement_x_um = np.asarray(data["truth_displacement_x_m"], dtype=np.float64) * 1e6
    truth_displacement_y_um = np.asarray(data["truth_displacement_y_m"], dtype=np.float64) * 1e6
    return {
        "recovered_displacement_x_um": recovered_displacement_x_um,
        "recovered_displacement_y_um": recovered_displacement_y_um,
        "truth_displacement_x_um": truth_displacement_x_um,
        "truth_displacement_y_um": truth_displacement_y_um,
        "displacement_x_error_um": recovered_displacement_x_um - truth_displacement_x_um,
        "displacement_y_error_um": recovered_displacement_y_um - truth_displacement_y_um,
    }


def _pulse_phase_fields(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Build phase truth/recovered/error fields for pulse plots."""
    recovered_phase_rad = np.asarray(data["phase_rad"], dtype=np.float64)
    truth_phase_rad = np.asarray(data["truth_phase_rad"], dtype=np.float64)
    return {
        "recovered_phase_rad": recovered_phase_rad,
        "truth_phase_rad": truth_phase_rad,
        "phase_error_rad": np.asarray(data["phase_err_rad"], dtype=np.float64),
    }


def _iter_pulse_method_payloads(analysis_dir: str, pulse_idx: int, methods: List[str]):
    """Yield `(row_index, method, data)` triples for methods with available pulse arrays."""
    for row_index, method in enumerate(methods):
        data = _load_pulse_arrays_or_notice(analysis_dir, pulse_idx, method)
        if data is None:
            continue
        yield row_index, method, data


def _pulse_payloads_or_none(analysis_dir: str, pulse_idx: int, methods: List[str]):
    """Materialize available pulse payloads or print a consistent empty-state notice."""
    payloads = list(_iter_pulse_method_payloads(analysis_dir, pulse_idx, methods))
    if payloads:
        return payloads
    _print_visualization_notice(f"No pulse artifact arrays available for pulse {pulse_idx}")
    return None


def _plot_pulse_map_row(
    axes_row,
    *,
    panels: List[Tuple[np.ndarray, str, bool]],
    cmap: str = 'RdBu_r',
) -> None:
    """Render one row of pulse-map panels with shared defaults."""
    for ax, (data, title, symmetric) in zip(axes_row, panels):
        _imshow_panel(ax, data, title=title, cmap=cmap, symmetric=symmetric)


def _pulse_metric_series(records: List[Dict], metric_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return pulse indices and metric values for pulse-level metric records."""
    pulse_records = _pulse_case_records(records)
    pulse_indices = np.asarray([int(record["case"].split("_")[1]) for record in pulse_records], dtype=np.int64)
    metric_values = _metric_values(pulse_records, metric_key)
    return pulse_indices, metric_values


def _plot_metric_line_panel(
    ax,
    *,
    methods: List[str],
    by_method: Dict[str, List[Dict]],
    metric_key: str,
    metric_label: str,
    colors,
) -> None:
    """Plot one pulse-indexed metric line panel across methods."""
    for method_index, method in enumerate(methods):
        pulse_indices, metric_values = _pulse_metric_series(by_method[method], metric_key)
        ax.plot(
            pulse_indices,
            metric_values,
            'o-',
            color=colors[method_index % len(colors)],
            label=method,
            markersize=6,
            linewidth=1.5,
        )
    ax.set_xlabel("Pulse Index")
    ax.set_ylabel(metric_label)
    ax.set_title(metric_label)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_metric_box_panel(
    ax,
    *,
    methods: List[str],
    by_method: Dict[str, List[Dict]],
    metric_key: str,
    metric_label: str,
    colors,
) -> None:
    """Plot one metric boxplot panel across methods."""
    data_by_method = [_pulse_metric_series(by_method[method], metric_key)[1] for method in methods]
    boxplot = ax.boxplot(data_by_method, tick_labels=methods, patch_artist=True)
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel(metric_label)
    ax.set_title(metric_label)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)


def _plot_metric_bar_panel(
    ax,
    *,
    methods: List[str],
    by_method: Dict[str, List[Dict]],
    metric_key: str,
    metric_label: str,
    colors,
) -> None:
    """Plot one mean±std metric bar panel across methods."""
    means: List[float] = []
    stds: List[float] = []
    for method in methods:
        metric_values = _pulse_metric_series(by_method[method], metric_key)[1]
        means.append(float(np.mean(metric_values)))
        stds.append(float(np.std(metric_values)))
    x_positions = np.arange(len(methods))
    ax.bar(x_positions, means, yerr=stds, capsize=5, color=colors[:len(methods)], alpha=0.7)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel(metric_label)
    ax.set_title(metric_label)
    ax.grid(True, alpha=0.3, axis='y')


def _compute_method_statistics(
    analysis_dir: str,
    by_method: Dict[str, List[Dict]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate pulse-level metrics for each available reconstruction method."""
    method_stats: Dict[str, Dict[str, float]] = {}
    for method in _ordered_metric_methods(analysis_dir, by_method):
        records = by_method[method]
        pulse_records = _pulse_case_records(records)
        if not pulse_records:
            continue
        method_stats[method] = {
            "rmse_shift_px": float(np.mean(_metric_values(pulse_records, "rmse_shift_px"))),
            "rmse_phase_rad": float(np.mean(_metric_values(pulse_records, "rmse_phase_rad"))),
            "phase_corr": float(np.mean(_metric_values(pulse_records, "phase_corr"))),
            "n_pulses": len(pulse_records),
        }
    return method_stats


def _print_method_comparison_summary(method_stats: Dict[str, Dict[str, float]]) -> List[str]:
    """Print a ranking table and return the ordered method names."""
    _print_visualization_header("METHOD COMPARISON SUMMARY")
    _print_visualization_table_header(
        f"{'Method':<20} {'RMSE Shift (px)':<18} {'RMSE Phase (rad)':<18} {'Phase Corr':<12}"
    )

    ranking = sorted(method_stats, key=lambda method: method_stats[method]["phase_corr"], reverse=True)
    for idx, method in enumerate(ranking):
        stats = method_stats[method]
        rank = "🥇" if idx == 0 else ("🥈" if idx == 1 else ("🥉" if idx == 2 else "  "))
        print(
            f"{rank} {_METHOD_DISPLAY_NAMES.get(method, method):<17} "
            f"{stats['rmse_shift_px']:<18.4f} {stats['rmse_phase_rad']:<18.4f} {stats['phase_corr']:<12.4f}"
        )

    print("=" * 70)
    return ranking


def _centroid_reconstruction_summary(all_metrics: List[Dict], pixel_size_m: float) -> Dict[str, float] | None:
    """Summarize centroid pulse-level reconstruction quality when available."""
    centroid_metrics = [
        metric
        for metric in all_metrics
        if metric["method"] == "centroid" and metric["case"] != "accumulated"
    ]
    if not centroid_metrics:
        return None
    average_rmse_shift_px = float(np.mean([metric["rmse_shift_px"] for metric in centroid_metrics]))
    average_phase_corr = float(np.mean([metric["phase_corr"] for metric in centroid_metrics]))
    return {
        "avg_rmse_shift_px": average_rmse_shift_px,
        "avg_rmse_shift_nm": average_rmse_shift_px * float(pixel_size_m) * 1e9,
        "avg_phase_corr": average_phase_corr,
    }


def _save_method_comparison_json(
    output_dir: str,
    *,
    methods: List[str],
    method_stats: Dict[str, Dict[str, float]],
    ranking: List[str],
) -> None:
    """Persist the aggregated method comparison summary as JSON."""
    comparison_data = {
        "methods": methods,
        "statistics": method_stats,
        "ranking": ranking,
    }
    with open(os.path.join(output_dir, "method_comparison.json"), "w", encoding="utf-8") as fp:
        json.dump(comparison_data, fp, indent=2)


def _plot_comparison_bar_panel(
    ax,
    *,
    labels: List[str],
    values: List[float],
    colors: List[str],
    ylabel: str,
    title: str,
    ylim: Tuple[float, float] | None = None,
    text_offset: float = 0.01,
) -> None:
    """Plot a labeled comparison bar chart for one metric."""
    bars = ax.bar(labels, values, color=colors, alpha=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=20)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + text_offset,
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=9,
        )


def load_all_pulse_metrics(analysis_dir: str) -> Tuple[List[Dict], Dict]:
    """Load metrics from all pulse directories and the summary file."""
    metrics_path = os.path.join(analysis_dir, "metrics_summary.json")
    
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    with open(metrics_path, "r") as f:
        all_metrics = json.load(f)
    
    # Group by method
    by_method = {}
    for record in all_metrics:
        method = record["method"]
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(record)
    
    return all_metrics, by_method


def load_pulse_arrays(analysis_dir: str, pulse_idx: int, method: str) -> Dict[str, np.ndarray]:
    """Load the reconstruction arrays for a specific pulse and method."""
    pulse_dir = os.path.join(analysis_dir, f"pulse_{pulse_idx:03d}")
    npz_path = os.path.join(pulse_dir, f"{method}.npz")
    
    if not os.path.exists(npz_path):
        return None
    
    return dict(np.load(npz_path))


def discover_available_methods(analysis_dir: str, pulse_idx: int | None = None) -> List[str]:
    """Discover available reconstruction methods from metrics and pulse artifacts."""
    methods = set()

    metrics_path = os.path.join(analysis_dir, "metrics_summary.json")
    if os.path.exists(metrics_path):
        _, by_method = load_all_pulse_metrics(analysis_dir)
        methods.update(by_method.keys())

    if pulse_idx is not None:
        pulse_dir = os.path.join(analysis_dir, f"pulse_{int(pulse_idx):03d}")
        if os.path.isdir(pulse_dir):
            for filename in os.listdir(pulse_dir):
                if filename.endswith(".npz"):
                    methods.add(os.path.splitext(filename)[0])

    ordered = [method for method in _PREFERRED_METHOD_ORDER if method in methods]
    extras = sorted(method for method in methods if method not in _PREFERRED_METHOD_ORDER)
    return ordered + extras


def _ordered_metric_methods(analysis_dir: str, by_method: Dict[str, List[Dict]]) -> List[str]:
    """Return metric-bearing methods in the same stable order used by pulse discovery."""
    discovered = discover_available_methods(analysis_dir)
    ordered = [method for method in discovered if method in by_method and len(by_method[method]) > 0]
    extras = sorted(method for method in by_method if method not in ordered and len(by_method[method]) > 0)
    return ordered + extras


def plot_errors_all_pulses(analysis_dir: str, output_dir: str):
    """Create error plots showing all pulses in a single figure."""
    _, by_method = load_all_pulse_metrics(analysis_dir)
    methods = _metric_methods_or_none(analysis_dir, by_method)
    if methods is None:
        return
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Reconstruction Errors Across All Pulses", fontsize=14)
    
    metrics_to_plot = [
        ("rmse_shift_px", "RMSE Shift (pixels)", axes[0, 0]),
        ("rmse_shift_um", "RMSE Shift (µm)", axes[0, 1]),
        ("rmse_phase_rad", "RMSE Phase (rad)", axes[0, 2]),
        ("phase_corr", "Phase Correlation", axes[1, 0]),
        ("spatial_error_percent", "Spatial Error (%)", axes[1, 1]),
    ]
    
    colors = plt.cm.tab10.colors
    
    for metric_key, metric_label, ax in metrics_to_plot:
        _plot_metric_line_panel(
            ax,
            methods=methods,
            by_method=by_method,
            metric_key=metric_key,
            metric_label=metric_label,
            colors=colors,
        )
    
    # Hide the last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    _save_figure(fig, output_dir, "errors_all_pulses.png")


def plot_error_statistics(analysis_dir: str, output_dir: str):
    """Create statistical summary plots (box plots, bar charts)."""
    _, by_method = load_all_pulse_metrics(analysis_dir)
    methods = _metric_methods_or_none(analysis_dir, by_method)
    if methods is None:
        return
    
    # Collect data for box plots
    metrics_to_plot = [
        ("rmse_shift_px", "RMSE Shift (pixels)"),
        ("rmse_shift_um", "RMSE Shift (µm)"),
        ("rmse_phase_rad", "RMSE Phase (rad)"),
        ("phase_corr", "Phase Correlation"),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Statistical Summary of Reconstruction Errors", fontsize=14)
    
    axes_flat = axes.flatten()
    
    colors = plt.cm.tab10.colors
    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        _plot_metric_box_panel(
            axes_flat[idx],
            methods=methods,
            by_method=by_method,
            metric_key=metric_key,
            metric_label=metric_label,
            colors=colors,
        )
    
    plt.tight_layout()
    _save_figure(fig, output_dir, "error_statistics_boxplot.png")
    
    # Bar chart with mean and std
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Mean ± Std of Reconstruction Errors", fontsize=14)
    axes_flat = axes.flatten()
    
    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        _plot_metric_bar_panel(
            axes_flat[idx],
            methods=methods,
            by_method=by_method,
            metric_key=metric_key,
            metric_label=metric_label,
            colors=colors,
        )
    
    plt.tight_layout()
    _save_figure(fig, output_dir, "error_statistics_bar.png")


def plot_displacement_maps(analysis_dir: str, output_dir: str, pulse_idx: int = 0):
    """Create 2D maps showing displacement errors for a specific pulse."""
    methods = _pulse_methods_or_none(analysis_dir, pulse_idx)
    if methods is None:
        return
    
    fig, axes = plt.subplots(len(methods), 4, figsize=(16, 4 * len(methods)))
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Displacement Analysis - Pulse {pulse_idx}", fontsize=14)
    
    payloads = _pulse_payloads_or_none(analysis_dir, pulse_idx, methods)
    if payloads is None:
        plt.close(fig)
        return

    for row, method, data in payloads:
        fields = _pulse_displacement_fields_um(data)
        _plot_pulse_map_row(
            axes[row, :],
            panels=[
                (fields["recovered_displacement_x_um"], f"{method}\nRecovered X (µm)", False),
                (fields["truth_displacement_x_um"], "Truth X (µm)", False),
                (fields["recovered_displacement_y_um"], "Recovered Y (µm)", False),
                (fields["truth_displacement_y_um"], "Truth Y (µm)", False),
            ],
        )
    
    plt.tight_layout()
    _save_figure(fig, output_dir, f"displacement_maps_pulse_{pulse_idx:03d}.png")


def plot_phase_comparison(analysis_dir: str, output_dir: str, pulse_idx: int = 0):
    """Create phase comparison plots for a specific pulse."""
    methods = _pulse_methods_or_none(analysis_dir, pulse_idx)
    if methods is None:
        return
    
    fig, axes = plt.subplots(len(methods), 3, figsize=(12, 4 * len(methods)))
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Phase Reconstruction - Pulse {pulse_idx}", fontsize=14)
    
    payloads = _pulse_payloads_or_none(analysis_dir, pulse_idx, methods)
    if payloads is None:
        plt.close(fig)
        return

    for row, method, data in payloads:
        fields = _pulse_phase_fields(data)
        _plot_pulse_map_row(
            axes[row, :],
            panels=[
                (fields["recovered_phase_rad"], f"{method}\nRecovered Phase (rad)", True),
                (fields["truth_phase_rad"], "Truth Phase (rad)", True),
                (fields["phase_error_rad"], "Phase Error (rad)", True),
            ],
        )
    
    plt.tight_layout()
    _save_figure(fig, output_dir, f"phase_comparison_pulse_{pulse_idx:03d}.png")


def plot_method_comparison(analysis_dir: str, output_dir: str):
    """
    Create comprehensive comparison of all methods including optimization.
    
    This generates:
    1. Bar chart comparing all 5 methods (centroid, xcorr, phase_corr, opt_direct, opt_zernike)
    2. Radar chart showing multi-dimensional performance
    3. Table summary
    """
    all_metrics, by_method = load_all_pulse_metrics(analysis_dir)
    
    method_stats = _method_stats_or_none(analysis_dir, by_method)
    if method_stats is None:
        return

    methods = _comparison_methods_or_none(method_stats)
    if methods is None:
        return
    labels = [_METHOD_DISPLAY_NAMES.get(m, m) for m in methods]
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Method Comparison: All Available Methods", fontsize=14)
    
    # Color palette
    colors = {
        "centroid": "#1f77b4",
        "cross_correlation": "#ff7f0e",
        "phase_correlation": "#2ca02c",
        "optimization_direct": "#d62728",
        "optimization_zernike": "#9467bd",
    }
    bar_colors = [colors.get(m, "#888888") for m in methods]
    
    # 1. RMSE Shift comparison
    rmse_vals = [method_stats[m]["rmse_shift_px"] for m in methods]
    _plot_comparison_bar_panel(
        axes[0],
        labels=labels,
        values=rmse_vals,
        colors=bar_colors,
        ylabel="RMSE Shift (pixels)",
        title="Shift Accuracy (lower is better)",
        text_offset=0.01,
    )
    
    # 2. RMSE Phase comparison
    phase_rmse_vals = [method_stats[m]["rmse_phase_rad"] for m in methods]
    _plot_comparison_bar_panel(
        axes[1],
        labels=labels,
        values=phase_rmse_vals,
        colors=bar_colors,
        ylabel="RMSE Phase (rad)",
        title="Phase Accuracy (lower is better)",
        text_offset=0.01,
    )
    
    # 3. Phase Correlation comparison
    corr_vals = [method_stats[m]["phase_corr"] for m in methods]
    _plot_comparison_bar_panel(
        axes[2],
        labels=labels,
        values=corr_vals,
        colors=bar_colors,
        ylabel="Phase Correlation",
        title="Phase Correlation (higher is better)",
        ylim=(0, 1.1),
        text_offset=0.02,
    )
    
    plt.tight_layout()
    _save_figure(fig, output_dir, "method_comparison.png")
    
    # Print summary table
    ranking = _print_method_comparison_summary(method_stats)
    _save_method_comparison_json(
        output_dir,
        methods=methods,
        method_stats=method_stats,
        ranking=ranking,
    )


def _resolve_visualization_dirs(
    *,
    analysis_dir: str | None,
    result_dir: str | None,
    output_dir: str | None,
) -> tuple[str, str, str]:
    """Resolve result / analysis / output directories for the visualization CLI."""
    resolved_result_dir = result_dir or "result"
    resolved_analysis_dir = analysis_dir or os.path.join(resolved_result_dir, "analysis")
    resolved_output_dir = output_dir or os.path.join(resolved_analysis_dir, "figures")
    return resolved_result_dir, resolved_analysis_dir, resolved_output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate summary visualizations from Hartmann analysis outputs.")
    parser.add_argument(
        "analysis_dir",
        nargs="?",
        default=None,
        help="Path to an analysis directory containing metrics_summary.json (default: <result_dir>/analysis)",
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        help="Optional result directory used by print_error_analysis and default analysis path resolution",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save generated figures (default: <analysis_dir>/figures)",
    )
    parser.add_argument(
        "--pulse-index",
        type=int,
        default=0,
        help="Pulse index used for displacement/phase map figures (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    result_dir, analysis_dir, output_dir = _resolve_visualization_dirs(
        analysis_dir=args.analysis_dir,
        result_dir=args.result_dir,
        output_dir=args.output_dir,
    )

    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualization figures...")
    
    # Generate all plots
    plot_errors_all_pulses(analysis_dir, output_dir)
    plot_error_statistics(analysis_dir, output_dir)
    plot_displacement_maps(analysis_dir, output_dir, pulse_idx=int(args.pulse_index))
    plot_phase_comparison(analysis_dir, output_dir, pulse_idx=int(args.pulse_index))
    
    # Generate method comparison (includes optimization methods)
    plot_method_comparison(analysis_dir, output_dir)
    
    # Print summary analysis
    print_error_analysis(result_dir, analysis_dir)
    
    print(f"\nAll figures saved to: {output_dir}")


def print_error_analysis(result_dir: str, analysis_dir: str):
    """Print detailed error analysis and explanation."""
    import os
    
    # Load truth to get physical parameters
    truth_path = os.path.join(result_dir, "frame_000_truth.npz")
    if not os.path.exists(truth_path):
        return
    
    truth = np.load(truth_path)
    wavelength = float(truth['wavelength_m'])
    z_prop = float(truth['z_prop_m'])
    aperture_spacing = float(truth['pupil_spacing_m'])
    pixel_size = float(truth['detector_pixel_m'])
    phase = truth['phase_rad']
    
    # Fresnel number
    fresnel_number = aperture_spacing**2 / (wavelength * z_prop)
    
    # Phase statistics
    phase_pv = phase.max() - phase.min()
    phase_waves = phase_pv / (2 * np.pi)
    
    _print_visualization_header("ERROR ANALYSIS SUMMARY")
    
    _print_visualization_section("Physical Parameters")
    print(f"  Wavelength:        {wavelength:.4e} m ({wavelength*1e10:.4f} Angstrom)")
    print(f"  Z propagation:     {z_prop:.4f} m")
    print(f"  Aperture spacing:  {aperture_spacing*1e6:.1f} um")
    print(f"  Pixel size:        {pixel_size*1e6:.2f} um")
    
    _print_visualization_section("Regime Analysis")
    print(f"  Fresnel number:    {fresnel_number:.1f}")
    if fresnel_number > 10:
        print("  => NEAR-FIELD (Fresnel) regime")
        print("     Note: For X-rays, geometrical optics still works well")
        print("     due to the tiny diffraction angles.")
    elif fresnel_number < 0.1:
        print("  => FAR-FIELD (Fraunhofer) regime")
        print("     Geometrical optics should work well.")
    else:
        print("  => TRANSITION regime")
    
    print(f"\n  Phase variation:   {phase_pv:.1f} rad = {phase_waves:.1f} waves")
    if phase_waves > 10:
        print("  => LARGE phase variation (>> 1 wave)")
        print("     Consider reducing PHASE_ABERRATION_RANGE for better accuracy.")
    elif phase_waves > 1:
        print("  => MODERATE phase variation (> 1 wave)")
    else:
        print("  => SMALL phase variation (< 1 wave) - Good for accurate reconstruction")
    
    # Load metrics to show actual performance
    metrics_path = os.path.join(analysis_dir, "metrics_summary.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, "r") as f:
            all_metrics = json.load(f)
        
        centroid_summary = _centroid_reconstruction_summary(all_metrics, pixel_size)
        if centroid_summary is not None:
            _print_visualization_section("Reconstruction Quality")
            print(
                f"  Avg RMSE shift:    {centroid_summary['avg_rmse_shift_px']:.4f} pixels "
                f"({centroid_summary['avg_rmse_shift_nm']:.2f} nm)"
            )
            print(f"  Avg Phase corr:    {centroid_summary['avg_phase_corr']:.4f}")

            if centroid_summary["avg_phase_corr"] > 0.9:
                print("  => EXCELLENT: High correlation with ground truth")
            elif centroid_summary["avg_phase_corr"] > 0.7:
                print("  => GOOD: Reasonable correlation with ground truth")
            else:
                print("  => POOR: Low correlation - check parameters")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
