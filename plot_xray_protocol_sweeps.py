#!/usr/bin/env python3
"""Plot the named X-ray sweep summaries produced by run_xray_protocol_sweeps.py."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MPLCONFIGDIR = ROOT / ".matplotlib"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_style import apply_plot_font_bump

apply_plot_font_bump()


def _load_summary_csv(path: Path) -> list[dict[str, float | str | None]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows: list[dict[str, float | str | None]] = []
        for row in reader:
            parsed: dict[str, float | str | None] = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = None
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
        return rows


def _sort_rows(rows: list[dict[str, float | str | None]]) -> list[dict[str, float | str | None]]:
    return sorted(rows, key=lambda row: float(row["value"]))


def _series(rows: list[dict[str, float | str | None]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=float)


def _label(rows: list[dict[str, float | str | None]], key: str, mode: str) -> str:
    best_row = min(rows, key=lambda row: float(row[key]))
    value = float(best_row[key])
    x = float(best_row["value"])
    if mode == "min":
        return f"best {key}: {value:.4g} @ {x:g}"
    best_row = max(rows, key=lambda row: float(row[key]))
    value = float(best_row[key])
    x = float(best_row["value"])
    return f"best {key}: {value:.4g} @ {x:g}"


def _plot_step(rows: list[dict[str, float | str | None]], output: Path) -> None:
    x = _series(rows, "step_um")
    shift = _series(rows, "dpc_rmse_total_px")
    max_err = _series(rows, "dpc_max_err_px")
    corr_dx = _series(rows, "dpc_mean_corr_dx")
    corr_dy = _series(rows, "dpc_mean_corr_dy")
    spatial = _series(rows, "dpc_spatial_error_percent")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(x, shift, marker="o", linewidth=2, label="DPC RMSE")
    axes[0, 0].set_title("STEP Sweep: DPC RMSE")
    axes[0, 0].set_xlabel("STEP_M (um)")
    axes[0, 0].set_ylabel("RMSE (px)")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(x, max_err, marker="o", color="#d62728", linewidth=2)
    axes[0, 1].set_title("STEP Sweep: DPC Max Error")
    axes[0, 1].set_xlabel("STEP_M (um)")
    axes[0, 1].set_ylabel("Max error (px)")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(x, corr_dx, marker="o", color="#1f77b4", linewidth=2, label="dX")
    axes[1, 0].plot(x, corr_dy, marker="s", color="#2ca02c", linewidth=2, label="dY")
    axes[1, 0].set_title("STEP Sweep: DPC Correlation")
    axes[1, 0].set_xlabel("STEP_M (um)")
    axes[1, 0].set_ylabel("Correlation")
    axes[1, 0].set_ylim(0.85, 1.01)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(x, spatial, marker="o", color="#9467bd", linewidth=2)
    axes[1, 1].set_title("STEP Sweep: DPC Spatial Error")
    axes[1, 1].set_xlabel("STEP_M (um)")
    axes[1, 1].set_ylabel("Error (%)")
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("X-ray STEP Sweep (z=450 mm, WFE=3 rad, 10 pulses)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_z(rows: list[dict[str, float | str | None]], output: Path) -> None:
    x = _series(rows, "z_mm")
    shift = _series(rows, "dpc_rmse_total_px")
    max_err = _series(rows, "dpc_max_err_px")
    corr_dx = _series(rows, "dpc_mean_corr_dx")
    corr_dy = _series(rows, "dpc_mean_corr_dy")
    shift_um = _series(rows, "dpc_rmse_total_um")
    pitch_um = _series(rows, "pitch_um")
    theta_min_urad = shift_um / x * 1000.0
    theta_max_urad = pitch_um / (2.0 * x) * 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(x, shift, marker="o", linewidth=2, label="DPC RMSE")
    axes[0, 0].set_title("Z Sweep: DPC RMSE")
    axes[0, 0].set_xlabel("Z_PROP_M (mm)")
    axes[0, 0].set_ylabel("RMSE (px)")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(x, max_err, marker="o", color="#d62728", linewidth=2)
    axes[0, 1].set_title("Z Sweep: DPC Max Error")
    axes[0, 1].set_xlabel("Z_PROP_M (mm)")
    axes[0, 1].set_ylabel("Max error (px)")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(x, corr_dx, marker="o", color="#1f77b4", linewidth=2, label="dX")
    axes[1, 0].plot(x, corr_dy, marker="s", color="#2ca02c", linewidth=2, label="dY")
    axes[1, 0].set_title("Z Sweep: DPC Correlation")
    axes[1, 0].set_xlabel("Z_PROP_M (mm)")
    axes[1, 0].set_ylabel("Correlation")
    axes[1, 0].set_ylim(0.90, 1.01)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    ax_left = axes[1, 1]
    ax_right = ax_left.twinx()
    ax_left.plot(x, theta_min_urad, marker="o", color="#ff7f0e", linewidth=2, label="theta_min")
    ax_right.plot(x, theta_max_urad, marker="s", color="#1f77b4", linewidth=2, label="theta_max")
    ax_left.set_title("Z Sweep: Angular Resolution / Dynamic Range")
    ax_left.set_xlabel("Z_PROP_M (mm)")
    ax_left.set_ylabel("theta_min (urad)", color="#ff7f0e")
    ax_right.set_ylabel("theta_max (urad)", color="#1f77b4")
    ax_left.grid(alpha=0.3)
    ax_left.tick_params(axis="y", colors="#ff7f0e")
    ax_right.tick_params(axis="y", colors="#1f77b4")

    fig.suptitle("X-ray Z Sweep (step=7 um, WFE=3 rad, 10 pulses)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_wfe(rows: list[dict[str, float | str | None]], output: Path) -> None:
    x = _series(rows, "value")
    shift = _series(rows, "dpc_rmse_total_px")
    max_err = _series(rows, "dpc_max_err_px")
    corr_dx = _series(rows, "dpc_mean_corr_dx")
    corr_dy = _series(rows, "dpc_mean_corr_dy")
    spatial = _series(rows, "dpc_spatial_error_percent")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(x, shift, marker="o", linewidth=2, label="DPC RMSE")
    axes[0, 0].set_title("WFE Sweep: DPC RMSE")
    axes[0, 0].set_xlabel("WFE amplitude (rad)")
    axes[0, 0].set_ylabel("RMSE (px)")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(x, max_err, marker="o", color="#d62728", linewidth=2)
    axes[0, 1].set_title("WFE Sweep: DPC Max Error")
    axes[0, 1].set_xlabel("WFE amplitude (rad)")
    axes[0, 1].set_ylabel("Max error (px)")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(x, corr_dx, marker="o", color="#1f77b4", linewidth=2, label="dX")
    axes[1, 0].plot(x, corr_dy, marker="s", color="#2ca02c", linewidth=2, label="dY")
    axes[1, 0].set_title("WFE Sweep: DPC Correlation")
    axes[1, 0].set_xlabel("WFE amplitude (rad)")
    axes[1, 0].set_ylabel("Correlation")
    axes[1, 0].set_ylim(0.97, 1.01)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(x, spatial, marker="o", color="#9467bd", linewidth=2)
    axes[1, 1].set_title("WFE Sweep: DPC Spatial Error")
    axes[1, 1].set_xlabel("WFE amplitude (rad)")
    axes[1, 1].set_ylabel("Error (%)")
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("X-ray WFE Sweep (step=7 um, z=450 mm, 10 pulses)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_overview(
    step_rows: list[dict[str, float | str | None]],
    z_rows: list[dict[str, float | str | None]],
    wfe_rows: list[dict[str, float | str | None]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(_series(step_rows, "step_um"), _series(step_rows, "dpc_rmse_total_px"), marker="o", linewidth=2)
    axes[0].set_title("STEP Sweep")
    axes[0].set_xlabel("STEP_M (um)")
    axes[0].set_ylabel("DPC RMSE (px)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(_series(z_rows, "z_mm"), _series(z_rows, "dpc_rmse_total_px"), marker="o", color="#d62728", linewidth=2)
    axes[1].set_title("Z Sweep")
    axes[1].set_xlabel("Z_PROP_M (mm)")
    axes[1].set_ylabel("DPC RMSE (px)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(_series(wfe_rows, "value"), _series(wfe_rows, "dpc_rmse_total_px"), marker="o", color="#2ca02c", linewidth=2)
    axes[2].set_title("WFE Sweep")
    axes[2].set_xlabel("WFE amplitude (rad)")
    axes[2].set_ylabel("DPC RMSE (px)")
    axes[2].grid(alpha=0.3)

    fig.suptitle("X-ray DPC Sweep Overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _best_row(rows: list[dict[str, float | str | None]], metric: str, maximize: bool = False) -> dict[str, float | str | None]:
    key = lambda row: float(row[metric])
    return max(rows, key=key) if maximize else min(rows, key=key)


def _write_report(
    step_rows: list[dict[str, float | str | None]],
    z_rows: list[dict[str, float | str | None]],
    wfe_rows: list[dict[str, float | str | None]],
    output: Path,
) -> None:
    step_best_shift = _best_row(step_rows, "dpc_rmse_total_px")
    step_best_corr = _best_row(step_rows, "dpc_mean_corr_dx", maximize=True)
    z_best_shift = _best_row(z_rows, "dpc_rmse_total_px")
    z_best_corr = _best_row(z_rows, "dpc_mean_corr_dx", maximize=True)
    wfe_best_shift = _best_row(wfe_rows, "dpc_rmse_total_px")
    wfe_best_corr = _best_row(wfe_rows, "dpc_mean_corr_dx", maximize=True)

    lines = [
        "# X-ray DPC Sweep Summary",
        "",
        "## STEP Sweep",
        f"- Best DPC RMSE: `{float(step_best_shift['dpc_rmse_total_px']):.4f} px` at `STEP_M={float(step_best_shift['step_um']):.1f} um`.",
        f"- Highest dX correlation: `{float(step_best_corr['dpc_mean_corr_dx']):.4f}` at `STEP_M={float(step_best_corr['step_um']):.1f} um`.",
        "",
        "## Z Sweep",
        f"- Best DPC RMSE: `{float(z_best_shift['dpc_rmse_total_px']):.4f} px` at `Z_PROP_M={float(z_best_shift['z_mm']):.0f} mm`.",
        f"- Highest dX correlation: `{float(z_best_corr['dpc_mean_corr_dx']):.4f}` at `Z_PROP_M={float(z_best_corr['z_mm']):.0f} mm`.",
        "",
        "## WFE Sweep",
        f"- Best DPC RMSE: `{float(wfe_best_shift['dpc_rmse_total_px']):.4f} px` at `WFE={float(wfe_best_shift['value']):.1f} rad`.",
        f"- Highest dX correlation: `{float(wfe_best_corr['dpc_mean_corr_dx']):.4f}` at `WFE={float(wfe_best_corr['value']):.1f} rad`.",
        "",
        "## Note",
        "- These plots and summaries evaluate DPC displacement only; phase integration metrics are intentionally omitted.",
        "- The current spatial-frequency limit is still determined by the fixed Hartmann pitch.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot summary CSVs for the X-ray sweep batch.")
    parser.add_argument("batch_root", type=Path, help="Batch root, e.g. result/xray/sweeps_20260407_002")
    args = parser.parse_args()

    batch_root = args.batch_root.resolve()
    summaries_dir = batch_root / "summaries"
    plots_dir = batch_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    step_rows = _sort_rows(_load_summary_csv(summaries_dir / "step_summary.csv"))
    z_rows = _sort_rows(_load_summary_csv(summaries_dir / "z_summary.csv"))
    wfe_rows = _sort_rows(_load_summary_csv(summaries_dir / "wfe_summary.csv"))

    _plot_step(step_rows, plots_dir / "step_sweep_metrics.png")
    _plot_z(z_rows, plots_dir / "z_sweep_metrics.png")
    _plot_wfe(wfe_rows, plots_dir / "wfe_sweep_metrics.png")
    _plot_overview(step_rows, z_rows, wfe_rows, plots_dir / "sweep_overview.png")
    _write_report(step_rows, z_rows, wfe_rows, plots_dir / "sweep_summary.md")

    print(f"Plots saved to: {plots_dir}")
    print(f"  step: {plots_dir / 'step_sweep_metrics.png'}")
    print(f"  z:    {plots_dir / 'z_sweep_metrics.png'}")
    print(f"  wfe:  {plots_dir / 'wfe_sweep_metrics.png'}")
    print(f"  all:  {plots_dir / 'sweep_overview.png'}")
    print(f"  note: {plots_dir / 'sweep_summary.md'}")


if __name__ == "__main__":
    main()
