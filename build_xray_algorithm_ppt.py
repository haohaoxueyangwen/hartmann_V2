#!/usr/bin/env python3
"""Build an English PPTX deck for the X-ray multi-pulse wavefront sensing report."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
PRESENTATION_DIR = ROOT / "presentation" / "xray_algorithm_ppt"
ASSETS_DIR = PRESENTATION_DIR / "assets"
OUTPUT_PPTX = ROOT / "presentation" / "xray_multi_pulse_wavefront_sensing_en.pptx"
OUTPUT_MD = PRESENTATION_DIR / "xray_algorithm_presentation_en.md"

RUN_DIR = ROOT / "result" / "xray" / "run_002"
SWEEP_DIR = ROOT / "result" / "xray" / "sweeps_20260407_002"

MPLCONFIGDIR = ROOT / ".matplotlib"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, FancyBboxPatch
import matplotlib.image as mpimg
import numpy as np
from plot_style import apply_plot_font_bump

apply_plot_font_bump()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _mean_for_method(metrics: list[dict], method: str, key: str) -> float:
    rows = [row for row in metrics if row.get("method") == method]
    return float(mean(float(row[key]) for row in rows))


def _best_row(rows: list[dict[str, str]], metric: str, maximize: bool = False) -> dict[str, str]:
    fn = max if maximize else min
    return fn(rows, key=lambda row: float(row[metric]))


def _draw_optical_layout(output: Path, sim_params: dict) -> None:
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    y = 3.2
    beam_y = 3.2

    ax.add_patch(Circle((0.9, y), 0.25, facecolor="#dbeafe", edgecolor="#2563eb", linewidth=2))
    ax.text(0.9, 4.0, "Incoming X-ray\nwavefront", ha="center", va="bottom", fontsize=16, weight="bold")
    ax.text(0.9, 2.15, "10 keV\nDistorted pulse train", ha="center", va="top", fontsize=11)

    ax.add_patch(Rectangle((2.3, 1.8), 0.3, 2.8, facecolor="#fde68a", edgecolor="#b45309", linewidth=2))
    ax.text(2.45, 4.95, "Hartmann Mask", ha="center", va="bottom", fontsize=15, weight="bold")
    ax.text(2.45, 1.45, "Pitch = 25 um\nHole = 13 um", ha="center", va="top", fontsize=11)

    ax.add_patch(Rectangle((10.9, 1.3), 0.35, 3.7, facecolor="#dcfce7", edgecolor="#15803d", linewidth=2))
    ax.text(11.075, 5.25, "Detector", ha="center", va="bottom", fontsize=15, weight="bold")
    ax.text(11.075, 0.95, "1024 x 1024\nPixel = 0.65 um", ha="center", va="top", fontsize=11)

    arrow_style = dict(arrowstyle="-|>", mutation_scale=18, linewidth=2.2, color="#111827")
    ax.add_patch(FancyArrowPatch((1.25, beam_y), (2.28, beam_y), **arrow_style))
    ax.add_patch(FancyArrowPatch((2.62, beam_y), (10.88, beam_y), **arrow_style))

    ax.text(6.7, 3.55, f"Free-space propagation\nz = {sim_params['z_prop_m']*1000:.0f} mm", ha="center", va="bottom", fontsize=15)

    for xx in np.linspace(2.85, 10.45, 12):
        ax.plot([xx, xx + 0.25], [beam_y - 0.5, beam_y + 0.5], color="#93c5fd", alpha=0.45, linewidth=1.8)

    start = np.array([2.9, 1.0])
    step_px_label = sim_params["step_m"] * 1e6
    total_pulses = sim_params["n_pulses"]
    angle = math.radians(sim_params["angle_deg"])
    step_vec = np.array([math.cos(angle), math.sin(angle)]) * 0.42
    points = [start + i * step_vec for i in range(total_pulses)]
    for i, p in enumerate(points):
        ax.add_patch(Circle(tuple(p), 0.08, facecolor="#ef4444", edgecolor="white", linewidth=1))
        if i in (0, total_pulses - 1):
            ax.text(p[0], p[1] - 0.22, f"P{i}", ha="center", va="top", fontsize=10)
    ax.add_patch(FancyArrowPatch(tuple(points[0]), tuple(points[-1]), arrowstyle="->", mutation_scale=14, linewidth=2, color="#ef4444"))
    ax.text(5.1, 1.45, f"Pulse-to-pulse translation\n{step_px_label:.1f} um per pulse, {sim_params['angle_deg']:.0f} deg, {total_pulses} pulses", ha="center", va="bottom", fontsize=12, color="#991b1b")

    footer = FancyBboxPatch((0.5, 5.35), 12.9, 0.45, boxstyle="round,pad=0.02,rounding_size=0.07", facecolor="#f3f4f6", edgecolor="#d1d5db")
    ax.add_patch(footer)
    ax.text(6.95, 5.57, "Measurement concept: wavefront -> Hartmann sampling -> propagation -> accumulated detector frame -> pulse-wise recovery", ha="center", va="center", fontsize=12, color="#111827")

    plt.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _draw_workflow(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    boxes = [
        (0.6, 4.6, 2.2, 1.2, "#dbeafe", "Wavefront Setup", "Photon energy\nWFE\nPulse train"),
        (3.2, 4.6, 2.4, 1.2, "#fde68a", "Forward Simulation", "Hartmann mask\nPropagation\nAccumulation"),
        (6.1, 4.6, 2.4, 1.2, "#fee2e2", "Accumulated Frame", "One detector image\ncontains 10 pulses"),
        (8.9, 4.6, 2.4, 1.2, "#dcfce7", "Pulse Decomposition", "Iterative pulse\nseparation"),
        (11.7, 4.6, 1.8, 1.2, "#ede9fe", "Shift Estimation", "Cross-correlation"),
        (4.7, 2.0, 2.6, 1.2, "#cffafe", "Slope / Phase Recovery", "Spot shifts -> slopes\nSlopes -> phase"),
        (8.2, 2.0, 2.8, 1.2, "#fce7f3", "Performance Metrics", "RMSE\nPhase corr.\nDynamic range"),
    ]
    for x, y, w, h, color, title, body in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor=color, edgecolor="#374151", linewidth=1.8)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + 0.84, title, ha="center", va="center", fontsize=14, weight="bold")
        ax.text(x + w / 2, y + 0.38, body, ha="center", va="center", fontsize=11)

    arrow_style = dict(arrowstyle="-|>", mutation_scale=18, linewidth=2.0, color="#4b5563")
    arrows = [
        ((2.8, 5.2), (3.2, 5.2)),
        ((5.6, 5.2), (6.1, 5.2)),
        ((8.5, 5.2), (8.9, 5.2)),
        ((11.3, 5.2), (11.7, 5.2)),
        ((12.6, 4.55), (9.85, 3.3)),
        ((7.3, 2.6), (8.2, 2.6)),
        ((9.95, 4.55), (6.0, 3.3)),
    ]
    for p1, p2 in arrows:
        ax.add_patch(FancyArrowPatch(p1, p2, **arrow_style))

    ax.text(7.0, 6.35, "Simulation and recovery workflow used in the current X-ray FEL study", ha="center", va="center", fontsize=18, weight="bold")
    ax.text(7.0, 1.0, "Pipeline mapping in code: simulate.py -> recover.py -> batch sweeps -> plotting", ha="center", va="center", fontsize=12, color="#374151")

    plt.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _make_panel(output: Path, items: list[tuple[str, Path]], ncols: int = 2, title: str | None = None) -> None:
    n = len(items)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 7.4), facecolor="white")
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = axes.reshape(nrows, ncols)
    for ax in axes.flat:
        ax.axis("off")

    for ax, (caption, path) in zip(axes.flat, items):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(caption, fontsize=13)
        ax.axis("off")

    for ax in axes.flat[n:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=18, weight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_markdown(content: str) -> None:
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(content, encoding="utf-8")


def _build_pptx() -> None:
    OUTPUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "/opt/anaconda3/bin/pandoc",
            str(OUTPUT_MD),
            "-o",
            str(OUTPUT_PPTX),
        ],
        cwd=str(PRESENTATION_DIR),
        check=True,
    )


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    sim = _read_json(RUN_DIR / "simulation_params.json")
    oracle = _read_json(RUN_DIR / "analysis" / "eval_baseline_cross_correlation.json")
    decomp = _read_json(RUN_DIR / "analysis" / "decomposition_cross_correlation.json")
    metrics = _read_json(RUN_DIR / "analysis" / "step3_decomposition" / "metrics_summary.json")

    step_rows = _read_csv(SWEEP_DIR / "summaries" / "step_summary.csv")
    z_rows = _read_csv(SWEEP_DIR / "summaries" / "z_summary.csv")
    wfe_rows = _read_csv(SWEEP_DIR / "summaries" / "wfe_summary.csv")

    cc_shift = _mean_for_method(metrics, "cross_correlation", "rmse_shift_px")
    cc_phase = _mean_for_method(metrics, "cross_correlation", "rmse_phase_rad")
    cc_corr = _mean_for_method(metrics, "cross_correlation", "phase_corr")
    pc_shift = _mean_for_method(metrics, "phase_correlation", "rmse_shift_px")
    pc_phase = _mean_for_method(metrics, "phase_correlation", "rmse_phase_rad")
    pc_corr = _mean_for_method(metrics, "phase_correlation", "phase_corr")
    cen_shift = _mean_for_method(metrics, "centroid", "rmse_shift_px")
    cen_phase = _mean_for_method(metrics, "centroid", "rmse_phase_rad")
    cen_corr = _mean_for_method(metrics, "centroid", "phase_corr")

    mean_conf = float(mean(decomp["decomposition_confidences"]))
    oracle_rmse = float(oracle["disp_metrics"]["rmse_total_px"])

    pixel = float(sim["pixel_size_m"])
    z = float(sim["z_prop_m"])
    pitch = float(sim["pitch_m"])
    lam = float(sim["wavelength_m"])
    dx_min = cc_shift * pixel
    theta_min = dx_min / z
    grad_min = (2 * math.pi / lam) * theta_min
    theta_max = pitch / (2 * z)
    dynamic_range = theta_max / theta_min
    f_max = 1 / (2 * pitch)
    min_period = 2 * pitch

    best_step_shift = _best_row(step_rows, "step3_xcorr_avg_rmse_shift_px")
    best_step_phase = _best_row(step_rows, "step3_xcorr_avg_rmse_phase_rad")
    best_z_shift = _best_row(z_rows, "step3_xcorr_avg_rmse_shift_px")
    best_z_phase = _best_row(z_rows, "step3_xcorr_avg_rmse_phase_rad")
    best_wfe_shift = _best_row(wfe_rows, "step3_xcorr_avg_rmse_shift_px")

    optical_layout = ASSETS_DIR / "xray_optical_layout_en.png"
    workflow = ASSETS_DIR / "simulation_recovery_workflow_en.png"
    baseline_panel = ASSETS_DIR / "baseline_overview_panel.png"
    recovery_panel = ASSETS_DIR / "phase_displacement_panel.png"
    method_panel = ASSETS_DIR / "method_comparison_panel.png"

    _draw_optical_layout(optical_layout, sim)
    _draw_workflow(workflow)
    _make_panel(
        baseline_panel,
        [
            ("Recovery overview", RUN_DIR / "analysis" / "recovery_cross_correlation.png"),
            ("Decomposition overview", RUN_DIR / "analysis" / "decomposition_images_cross_correlation.png"),
            ("Trajectory occupancy", RUN_DIR / "analysis" / "step3_decomposition" / "trajectory_design" / "trajectory_occupancy.png"),
            ("Pulse 0 summary", RUN_DIR / "analysis" / "step3_decomposition" / "frame_000_summary.png"),
        ],
        ncols=2,
        title="Baseline Multi-Pulse Recovery",
    )
    _make_panel(
        recovery_panel,
        [
            ("All-frame phase summary", RUN_DIR / "analysis" / "all_frames_phase_summary_cross_correlation.png"),
            ("All-frame displacement summary", RUN_DIR / "analysis" / "all_frames_disp_summary_cross_correlation.png"),
            ("Per-frame RMSE", RUN_DIR / "analysis" / "per_frame_displacement" / "per_frame_rmse_cross_correlation.png"),
            ("Per-frame phase maps", RUN_DIR / "analysis" / "per_frame_phase" / "phase_all_frames_cross_correlation.png"),
        ],
        ncols=2,
        title="Pulse-Wise Recovery Consistency",
    )
    _make_panel(
        method_panel,
        [
            ("Method comparison", RUN_DIR / "analysis" / "step3_decomposition" / "method_comparison.png"),
            ("Error statistics", RUN_DIR / "analysis" / "step3_decomposition" / "error_statistics.png"),
        ],
        ncols=2,
        title="Method Comparison",
    )

    md = f"""---
title: "Multi-Pulse Wavefront Sensing for High-Repetition-Rate X-ray FEL"
subtitle: "Algorithm, Optical Design, Experimental Workflow, and Performance Evaluation"
author: "Generated from current project results"
date: "April 7, 2026"
---

# Motivation

- High-repetition-rate FEL can produce multiple X-ray pulses within one detector integration window.
- A conventional Hartmann workflow expects one wavefront per frame, but here one frame may contain many pulses.
- The core problem is to recover pulse-wise wavefronts from one accumulated detector image.

![](../xray_algorithm_ppt/assets/baseline_overview_panel.png){{width=88%}}

# Research Objective

- Simulate accumulated multi-pulse X-ray Hartmann measurements.
- Decompose one accumulated frame into pulse-wise spot fields.
- Recover local spot displacements and reconstruct pulse-wise wavefront phase.
- Evaluate accuracy, robustness, and operating limits.

# Optical Design

- Incoming distorted X-ray wavefront is sampled by a Hartmann mask and then propagated to the detector.
- Baseline geometry: `10 keV`, `z = {sim['z_prop_m']*1000:.0f} mm`, `pitch = {pitch*1e6:.0f} um`, `hole = {sim['hole_diameter_m']*1e6:.0f} um`.
- The detector integrates `10` translated pulses into a single accumulated image.

![](../xray_algorithm_ppt/assets/xray_optical_layout_en.png){{width=92%}}

# Experimental Workflow

- Forward model: wavefront setup -> Hartmann propagation -> detector accumulation.
- Inverse model: decomposition -> cross-correlation shift estimation -> slope integration -> phase reconstruction.
- Code path: `simulate.py` -> `recover.py` -> sweep scripts -> plotting.

![](../xray_algorithm_ppt/assets/simulation_recovery_workflow_en.png){{width=92%}}

# Recovery Algorithm

1. Generate or load an accumulated detector frame.
2. Decompose the frame into pulse-wise spot components.
3. Estimate local shifts with cross-correlation.
4. Convert shifts to slopes and integrate to phase.
5. Compare against oracle and pulse-wise reference metrics.

- Main recovery method: `cross_correlation`
- Decomposition: `iterative`
- Patch half-size: `13`
- Reference mode: `per_pulse`

![](../xray_algorithm_ppt/assets/method_comparison_panel.png){{width=88%}}

# Baseline Experimental Configuration

- Propagation distance: `450 mm`
- Pulse-to-pulse step: `7 um`
- Translation angle: `19 deg`
- Number of pulses: `10`
- Sub-aperture grid: `26 x 26`
- Fresnel number: `{sim['fresnel_number']:.3f}`
- Detector pixel size: `{pixel*1e6:.2f} um`

# Baseline Result: Decomposition and Recovery

- The baseline case successfully decomposes one accumulated frame into pulse-wise spot maps.
- Mean decomposition confidence: `{mean_conf:.4f}`
- Oracle per-frame shift RMSE: `{oracle_rmse:.4f} px`
- Cross-correlation remains stable over all `10` pulses.

![](../xray_algorithm_ppt/assets/baseline_overview_panel.png){{width=88%}}

# Baseline Result: Pulse-Wise Phase and Displacement

- The recovery is consistent across all pulses rather than relying on one best frame.
- Average cross-correlation shift RMSE: `{cc_shift:.4f} px`
- Average cross-correlation phase RMSE: `{cc_phase:.4f} rad`
- Average phase correlation: `{cc_corr:.4f}`

![](../xray_algorithm_ppt/assets/phase_displacement_panel.png){{width=88%}}

# Method Comparison

- `cross_correlation` is the only reliable primary method in the current pipeline.
- `centroid` fails badly under the multi-pulse accumulated condition.
- `phase_correlation` remains usable as a comparison baseline, but is clearly weaker.

| Method | Avg. shift RMSE (px) | Avg. phase RMSE (rad) | Avg. phase corr. |
|---|---:|---:|---:|
| Centroid | {cen_shift:.4f} | {cen_phase:.4f} | {cen_corr:.4f} |
| Cross-correlation | {cc_shift:.4f} | {cc_phase:.4f} | {cc_corr:.4f} |
| Phase correlation | {pc_shift:.4f} | {pc_phase:.4f} | {pc_corr:.4f} |

![](../xray_algorithm_ppt/assets/method_comparison_panel.png){{width=78%}}

# Physical Performance and Limits

- Displacement resolution: `{dx_min*1e9:.2f} nm`
- Angular resolution: `{theta_min*1e6:.5f} urad`
- Conservative `3sigma` displacement resolution: `{3*dx_min*1e9:.2f} nm`
- Minimum measurable phase-gradient scale: `{grad_min:.2f} rad/m`
- Dynamic range: `{dynamic_range:.0f}:1`
- Highest measurable spatial frequency: `{f_max:.1f} m^-1`
- Minimum resolvable period: `{min_period*1e6:.0f} um`

# Parameter Sweep Design

- Three parameter sweeps were used to map the operating envelope.
- `STEP_M`: `7.0 -> 10.0 um`, interval `0.5 um`
- `Z_PROP_M`: `450 -> 1000 mm`, interval `50 mm`
- `WFE`: `1 -> 5 rad`
- `N_PULSES = 10` was fixed in all sweep runs.

![](../../result/xray/sweeps_20260407_002/plots/sweep_overview.png){{width=82%}}

# Sweep Result I: STEP Dependence

- Stable operating region is around `7.0 - 8.0 um`.
- Best shift RMSE appears at `STEP_M = {float(best_step_shift['step_um']):.1f} um`.
- Best phase RMSE appears at `STEP_M = {float(best_step_phase['step_um']):.1f} um`.
- Performance degrades noticeably for `STEP_M >= 9.5 um`.

![](../../result/xray/sweeps_20260407_002/plots/step_sweep_metrics.png){{width=88%}}

# Sweep Result II: Propagation Distance Dependence

- Best shift RMSE appears near `Z = {float(best_z_shift['z_mm']):.0f} mm`.
- Best phase RMSE appears near `Z = {float(best_z_phase['z_mm']):.0f} mm`.
- Performance degrades strongly beyond `700 mm`.
- This is the expected sensitivity vs. dynamic-range trade-off.

![](../../result/xray/sweeps_20260407_002/plots/z_sweep_metrics.png){{width=88%}}

# Sweep Result III: Wavefront Error Dependence

- Best overall accuracy appears at `WFE = {float(best_wfe_shift['value']):.1f} rad`.
- Shift and phase RMSE increase as aberration strength grows.
- The algorithm remains usable across the tested `1 - 5 rad` range.

![](../../result/xray/sweeps_20260407_002/plots/wfe_sweep_metrics.png){{width=88%}}

# Recommended Operating Point

- `N_PULSES = 10`
- `STEP_M = 7.0 um`
- `Z_PROP_M = 450 mm`
- `WFE = 3.0 rad` as a representative medium-aberration benchmark
- Recovery method: `cross_correlation`
- Decomposition method: `iterative`

- The current pipeline already achieves nanometer-scale displacement sensitivity on accumulated multi-pulse X-ray Hartmann data.

![](../../result/xray/sweeps_20260407_002/plots/sweep_overview.png){{width=75%}}

# Conclusion

- The current X-ray Hartmann workflow successfully recovers pulse-wise wavefront information from a single accumulated detector frame.
- Cross-correlation is the strongest recovery method in the present implementation.
- The present geometry works best near `450 - 500 mm`.
- The system already reaches high displacement sensitivity while preserving practical dynamic range.

"""

    _build_markdown(md)
    _build_pptx()
    print(f"Markdown written to: {OUTPUT_MD}")
    print(f"PPTX written to: {OUTPUT_PPTX}")


if __name__ == "__main__":
    main()
