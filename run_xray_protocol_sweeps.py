#!/usr/bin/env python3
"""
Batch-run the X-ray protocol sweeps discussed in this project.

Baseline:
    By default, inherit the current recommended working point from
    config_simulate.py:
        step = 8 um
        z = 450 mm
        WFE = 1 rad
        Hartmann pitch = 25 um

    Optionally, --base-run can still point to an existing simulation run
    if you want to inherit physical parameters from saved data.

Sweep groups:
    1. step sweep: 7.0 -> 10.0 um, step 0.5 um
       fixed z = 450 mm, fixed WFE = 1 rad
    2. z sweep: 450 -> 1000 mm, step 50 mm
       fixed step = 8 um, fixed WFE = 1 rad
    3. wfe sweep: 1 -> 5 rad (interpreted as symmetric coefficient range [-A, A])
       fixed step = 8 um, fixed z = 450 mm

Each run:
    - writes a JSON override config
    - runs simulate.py
    - runs recover.py --method cross_correlation
    - collects a compact summary row
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
SWEEP_ROOT = ROOT / "result" / "xray"


def _load_python_config(path: Path) -> dict[str, Any]:
    scope: dict[str, Any] = {}
    exec(path.read_text(encoding="utf-8"), scope)
    return {k: v for k, v in scope.items() if k.isupper()}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_range(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    cur = start
    eps = step * 1e-9
    while cur <= stop + eps:
        values.append(round(cur, 10))
        cur += step
    return values


def _safe_name_float(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def _config_to_base_params(current_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build a simulation-params-like snapshot directly from config_simulate.py."""
    return {
        "photon_energy_ev": current_cfg["PHOTON_ENERGY_EV"],
        "z_prop_m": current_cfg["Z_PROP_M"],
        "pixel_size_m": current_cfg["DETECTOR_PIXEL_M"],
        "pitch_m": current_cfg["HARTMANN_PITCH_M"],
        "hole_diameter_m": current_cfg["HARTMANN_HOLE_DIAMETER_M"],
        "n_pulses": current_cfg["N_PULSES"],
        "detector_nx": current_cfg["DETECTOR_NX"],
        "detector_ny": current_cfg["DETECTOR_NY"],
        "window_size_m": current_cfg["WINDOW_SIZE_M"],
        "grid_points": current_cfg["GRID_POINTS"],
        "propagation_method": current_cfg["PROPAGATION_METHOD"],
        "source_type": current_cfg["SOURCE_TYPE"],
        "mask_type": current_cfg["MASK_TYPE"],
        "angle_deg": current_cfg["ANGLE_DEG"],
        "step_m": current_cfg["STEP_M"],
        "wfe_preset": current_cfg.get("WFE_PRESET", ""),
        "phase_aberration_mode": current_cfg["PHASE_ABERRATION_MODE"],
        "movement_mode": current_cfg["MOVEMENT_MODE"],
    }


def _next_batch_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    idx = 1
    while True:
        candidate = root / f"sweeps_{stamp}_{idx:03d}"
        if not candidate.exists():
            return candidate
        idx += 1


@dataclass(frozen=True)
class SweepCase:
    group: str
    value: float
    value_label: str
    output_name: str
    overrides: dict[str, Any]


def _build_base_overrides(base_params: dict[str, Any], current_cfg: dict[str, Any]) -> dict[str, Any]:
    overrides = dict(current_cfg)
    mapping = {
        "PHOTON_ENERGY_EV": base_params["photon_energy_ev"],
        "Z_PROP_M": base_params["z_prop_m"],
        "DETECTOR_PIXEL_M": base_params["pixel_size_m"],
        "HARTMANN_PITCH_M": base_params["pitch_m"],
        "HARTMANN_HOLE_DIAMETER_M": base_params["hole_diameter_m"],
        "N_PULSES": base_params["n_pulses"],
        "DETECTOR_NX": base_params["detector_nx"],
        "DETECTOR_NY": base_params["detector_ny"],
        "WINDOW_SIZE_M": base_params["window_size_m"],
        "GRID_POINTS": base_params["grid_points"],
        "PROPAGATION_METHOD": base_params["propagation_method"],
        "SOURCE_TYPE": base_params["source_type"],
        "MASK_TYPE": base_params["mask_type"],
        "ANGLE_DEG": base_params["angle_deg"],
        "STEP_M": base_params["step_m"],
        "WFE_PRESET": base_params.get("wfe_preset", ""),
        "PHASE_ABERRATION_MODE": base_params["phase_aberration_mode"],
        "MOVEMENT_MODE": base_params["movement_mode"],
    }
    overrides.update(mapping)
    # Use a controlled symmetric range for the WFE sweep baseline.
    overrides["PHASE_ABERRATION_RANGE"] = [-3.0, 3.0]
    overrides["N_PULSES"] = 10
    overrides["ANGLE_DEG"] = 19.0
    return overrides


def _build_sweep_cases(base_overrides: dict[str, Any]) -> list[SweepCase]:
    cases: list[SweepCase] = []

    fixed_step_um = 8.0
    fixed_z_mm = 450.0
    fixed_wfe_rad = 1.0

    for step_um in _float_range(7.0, 10.0, 0.5):
        overrides = dict(base_overrides)
        overrides["STEP_M"] = step_um * 1e-6
        overrides["Z_PROP_M"] = fixed_z_mm * 1e-3
        overrides["PHASE_ABERRATION_RANGE"] = [-fixed_wfe_rad, fixed_wfe_rad]
        name = f"step_{_safe_name_float(step_um)}um_z{int(fixed_z_mm)}mm_wfe{_safe_name_float(fixed_wfe_rad)}rad"
        cases.append(SweepCase("step", step_um, f"{step_um:.1f} um", name, overrides))

    for z_mm in _float_range(450.0, 1000.0, 50.0):
        overrides = dict(base_overrides)
        overrides["STEP_M"] = fixed_step_um * 1e-6
        overrides["Z_PROP_M"] = z_mm * 1e-3
        overrides["PHASE_ABERRATION_RANGE"] = [-fixed_wfe_rad, fixed_wfe_rad]
        name = f"z_{int(z_mm)}mm_step{_safe_name_float(fixed_step_um)}um_wfe{_safe_name_float(fixed_wfe_rad)}rad"
        cases.append(SweepCase("z", z_mm, f"{z_mm:.0f} mm", name, overrides))

    for wfe_rad in _float_range(1.0, 5.0, 1.0):
        overrides = dict(base_overrides)
        overrides["STEP_M"] = fixed_step_um * 1e-6
        overrides["Z_PROP_M"] = fixed_z_mm * 1e-3
        overrides["PHASE_ABERRATION_RANGE"] = [-wfe_rad, wfe_rad]
        name = f"wfe_{_safe_name_float(wfe_rad)}rad_step{_safe_name_float(fixed_step_um)}um_z{int(fixed_z_mm)}mm"
        cases.append(SweepCase("wfe", wfe_rad, f"{wfe_rad:.1f} rad", name, overrides))

    return cases


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mean_metric(metrics: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(m[key]) for m in metrics if key in m]
    return float(mean(vals)) if vals else None


def _collect_summary(run_dir: Path, case: SweepCase) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "group": case.group,
        "value": case.value,
        "value_label": case.value_label,
        "output_name": case.output_name,
        "run_dir": str(run_dir.relative_to(ROOT)),
    }

    sim_params_path = run_dir / "simulation_params.json"
    if sim_params_path.exists():
        sim = _load_json(sim_params_path)
        pixel_um = float(sim["pixel_size_m"]) * 1e6
        pitch_um = float(sim["pitch_m"]) * 1e6
        summary.update(
            {
                "step_um": float(sim["step_m"]) * 1e6,
                "z_mm": float(sim["z_prop_m"]) * 1e3,
                "pitch_um": pitch_um,
                "hole_um": float(sim["hole_diameter_m"]) * 1e6,
                "pixel_um": pixel_um,
                "n_pulses": int(sim["n_pulses"]),
                "fresnel_number": float(sim["fresnel_number"]),
            }
        )

    metrics_path = run_dir / "analysis" / "metrics_cross_correlation.json"
    if metrics_path.exists():
        metrics = _load_json(metrics_path)
        disp = metrics.get("displacement_metrics", {})
        spot = metrics.get("dpc_spot_metrics", {})
        pixel_um = summary.get("pixel_um")
        pitch_um = summary.get("pitch_um")
        dpc_rmse_px = disp.get("rmse_total_px")
        dpc_rmse_um = (
            float(dpc_rmse_px) * float(pixel_um)
            if dpc_rmse_px is not None and pixel_um is not None
            else None
        )
        dpc_spatial_percent = (
            dpc_rmse_um / float(pitch_um) * 100.0
            if dpc_rmse_um is not None and pitch_um not in (None, 0, 0.0)
            else None
        )
        summary.update(
            {
                "recovery_mode": metrics.get("recovery_mode"),
                "accumulated_recovery_method": metrics.get("accumulated_recovery_method"),
                "reference_sequence_source": metrics.get("reference_sequence_source"),
                "output_mode": metrics.get("output_mode"),
                "metrics_source": metrics.get("metrics_source"),
                "dpc_rmse_total_px": dpc_rmse_px,
                "dpc_rmse_dx_px": disp.get("rmse_dx_px"),
                "dpc_rmse_dy_px": disp.get("rmse_dy_px"),
                "dpc_mean_err_px": disp.get("mean_err_px"),
                "dpc_max_err_px": disp.get("max_err_px"),
                "dpc_mean_corr_dx": disp.get("mean_corr_dx"),
                "dpc_mean_corr_dy": disp.get("mean_corr_dy"),
                "dpc_rmse_total_um": dpc_rmse_um,
                "dpc_spatial_error_percent": dpc_spatial_percent,
                "dpc_frame_median_rmse_px": disp.get("median_frame_rmse_px"),
                "dpc_frame_p95_rmse_px": disp.get("p95_frame_rmse_px"),
                "dpc_frame_best_index": disp.get("best_frame_index"),
                "dpc_frame_best_rmse_px": disp.get("best_frame_rmse_px"),
                "dpc_frame_worst_index": disp.get("worst_frame_index"),
                "dpc_frame_worst_rmse_px": disp.get("worst_frame_rmse_px"),
                "dpc_sample_median_abs_error_px": spot.get("median_sample_abs_error_px"),
                "dpc_sample_p95_abs_error_px": spot.get("p95_sample_abs_error_px"),
                "dpc_sample_median_abs_parallel_error_px": spot.get("median_sample_abs_parallel_error_px"),
                "dpc_sample_p95_abs_parallel_error_px": spot.get("p95_sample_abs_parallel_error_px"),
                "dpc_sample_median_abs_perpendicular_error_px": spot.get("median_sample_abs_perpendicular_error_px"),
                "dpc_sample_p95_abs_perpendicular_error_px": spot.get("p95_sample_abs_perpendicular_error_px"),
                "dpc_spot_median_rmse_px": spot.get("median_spot_rmse_px"),
                "dpc_spot_p95_rmse_px": spot.get("p95_spot_rmse_px"),
                "dpc_spot_max_rmse_px": spot.get("max_spot_rmse_px"),
                "dpc_spot_median_abs_error_px": spot.get("median_mean_abs_error_px"),
                "dpc_spot_p95_abs_error_px": spot.get("p95_mean_abs_error_px"),
                "dpc_spot_median_abs_parallel_error_px": spot.get("median_mean_abs_parallel_error_px"),
                "dpc_spot_p95_abs_parallel_error_px": spot.get("p95_mean_abs_parallel_error_px"),
                "dpc_spot_median_abs_perpendicular_error_px": spot.get("median_mean_abs_perpendicular_error_px"),
                "dpc_spot_p95_abs_perpendicular_error_px": spot.get("p95_mean_abs_perpendicular_error_px"),
                "dpc_spot_median_abs_bias_x_px": spot.get("median_abs_bias_x_px"),
                "dpc_spot_median_abs_bias_y_px": spot.get("median_abs_bias_y_px"),
                "dpc_spot_relative_threshold_px": spot.get("relative_error_truth_threshold_px"),
                "dpc_spot_median_truth_mag_px": spot.get("median_truth_mag_px"),
                "dpc_spot_median_relative_error_pct": spot.get("median_mean_relative_error_pct"),
                "dpc_spot_p95_relative_error_pct": spot.get("p95_mean_relative_error_pct"),
                "dpc_spot_median_scale_vec": spot.get("median_scale_vec"),
                "dpc_spot_p05_scale_vec": spot.get("p05_scale_vec"),
                "dpc_spot_p95_scale_vec": spot.get("p95_scale_vec"),
                "dpc_spot_median_angle_deg": spot.get("median_mean_angle_deg"),
                "dpc_spot_p95_angle_deg": spot.get("p95_mean_angle_deg"),
            }
        )

    accumulated_path = run_dir / "analysis" / "accumulated_recovery_cross_correlation.json"
    if accumulated_path.exists():
        accumulated = _load_json(accumulated_path)
        summary.update(
            {
                "linearized_median_condition_number": accumulated.get("median_condition_number"),
                "linearized_median_residual_ratio": accumulated.get("median_residual_ratio"),
            }
        )

    oracle_path = run_dir / "analysis" / "eval_baseline_cross_correlation.json"
    if oracle_path.exists():
        oracle = _load_json(oracle_path)
        disp = oracle.get("disp_metrics", {})
        summary.update(
            {
                "oracle_rmse_total_px": disp.get("rmse_total_px"),
                "oracle_rmse_dx_px": disp.get("rmse_dx_px"),
                "oracle_rmse_dy_px": disp.get("rmse_dy_px"),
                "oracle_mean_corr_dx": disp.get("mean_corr_dx"),
                "oracle_mean_corr_dy": disp.get("mean_corr_dy"),
            }
        )

    decomp_path = run_dir / "analysis" / "decomposition_cross_correlation.json"
    if decomp_path.exists():
        decomp = _load_json(decomp_path)
        confs = [float(x) for x in decomp.get("decomposition_confidences", [])]
        summary["decomposition_confidence_mean"] = float(mean(confs)) if confs else None
        summary["decomposition_method"] = decomp.get("decomposition_method")

    step3_metrics_path = run_dir / "analysis" / "step3_decomposition" / "metrics_summary.json"
    if step3_metrics_path.exists():
        all_metrics = _load_json(step3_metrics_path)
        xcorr_metrics = [m for m in all_metrics if m.get("method") == "cross_correlation"]
        summary.update(
            {
                "step3_xcorr_avg_rmse_shift_px": _mean_metric(xcorr_metrics, "rmse_shift_px"),
                "step3_xcorr_avg_rmse_shift_um": _mean_metric(xcorr_metrics, "rmse_shift_um"),
                "step3_xcorr_avg_rmse_phase_rad": _mean_metric(xcorr_metrics, "rmse_phase_rad"),
                "step3_xcorr_avg_phase_corr": _mean_metric(xcorr_metrics, "phase_corr"),
                "step3_xcorr_avg_spatial_error_percent": _mean_metric(xcorr_metrics, "spatial_error_percent"),
            }
        )

    traj_path = run_dir / "analysis" / "step3_decomposition" / "trajectory_design" / "trajectory_design_summary.json"
    if traj_path.exists():
        traj = _load_json(traj_path)
        summary.update(
            {
                "trajectory_design_status": traj.get("design_status"),
                "trajectory_identifiability_score": traj.get("identifiability_score"),
                "trajectory_visibility_overall": (traj.get("visibility") or {}).get("overall_ratio"),
                "trajectory_warning_overlap_fraction": (traj.get("cross_pulse_separation_px") or {}).get("warning_overlap_fraction"),
                "trajectory_spot_sigma_px": traj.get("spot_sigma_px"),
                "trajectory_step_length_px": ((traj.get("step_length_px") or {}).get("mean")),
            }
        )

    summary["wfe_range_min_rad"] = float(case.overrides["PHASE_ABERRATION_RANGE"][0])
    summary["wfe_range_max_rad"] = float(case.overrides["PHASE_ABERRATION_RANGE"][1])
    return summary


def _write_summary_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run named X-ray parameter sweeps and recover each result.")
    parser.add_argument(
        "--base-run",
        type=Path,
        default=None,
        help="Optional baseline run directory to inherit physical parameters from. Default: use config_simulate.py.",
    )
    parser.add_argument("--groups", nargs="+", choices=("step", "z", "wfe", "all"), default=("all",), help="Sweep groups to run.")
    parser.add_argument("--recover-method", default="cross_correlation", choices=("cross_correlation", "all"), help="recover.py method")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a run if its analysis directory already exists.")
    args = parser.parse_args()

    current_cfg = _load_python_config(ROOT / "config_simulate.py")
    if args.base_run is not None:
        base_run = args.base_run if args.base_run.is_absolute() else (ROOT / args.base_run)
        base_run = base_run.resolve()
        base_params = _load_json(base_run / "simulation_params.json")
        base_source = str(base_run)
    else:
        base_run = None
        base_params = _config_to_base_params(current_cfg)
        base_source = "config_simulate.py"
    base_overrides = _build_base_overrides(base_params, current_cfg)
    all_cases = _build_sweep_cases(base_overrides)

    selected_groups = {"step", "z", "wfe"} if "all" in args.groups else set(args.groups)
    cases = [case for case in all_cases if case.group in selected_groups]

    batch_dir = _next_batch_dir(SWEEP_ROOT)
    configs_dir = batch_dir / "configs"
    summaries_dir = batch_dir / "summaries"
    batch_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "base_source": base_source,
        "selected_groups": sorted(selected_groups),
        "recover_method": args.recover_method,
        "fixed_assumptions": {
            "n_pulses": 10,
            "angle_deg": 19.0,
            "step_sweep_fixed_z_mm": 450.0,
            "step_sweep_fixed_wfe_rad": 1.0,
            "z_sweep_fixed_step_um": 8.0,
            "z_sweep_fixed_wfe_rad": 1.0,
            "wfe_sweep_fixed_step_um": 8.0,
            "wfe_sweep_fixed_z_mm": 450.0,
            "wfe_definition": "PHASE_ABERRATION_RANGE = [-A, A]",
        },
    }
    _write_json(batch_dir / "batch_metadata.json", metadata)

    group_rows: dict[str, list[dict[str, Any]]] = {"step": [], "z": [], "wfe": []}

    for idx, case in enumerate(cases, start=1):
        run_dir = batch_dir / case.group / case.output_name
        cfg_path = configs_dir / case.group / f"{case.output_name}.json"
        _write_json(cfg_path, case.overrides)
        print(f"\n[{idx}/{len(cases)}] group={case.group} value={case.value_label} -> {run_dir.relative_to(ROOT)}")

        analysis_dir = run_dir / "analysis"
        if args.skip_existing and analysis_dir.exists():
            print("  Skipping existing analysis directory")
        else:
            _run(
                [sys.executable, "simulate.py", "--config", str(cfg_path), "--output-dir", str(run_dir)],
                cwd=ROOT,
            )
            _run(
                [sys.executable, "recover.py", str(run_dir), "--method", args.recover_method],
                cwd=ROOT,
            )

        summary = _collect_summary(run_dir, case)
        group_rows[case.group].append(summary)
        _write_summary_csv(summaries_dir / f"{case.group}_summary.csv", group_rows[case.group])
        _write_json(summaries_dir / f"{case.group}_summary.json", group_rows[case.group])

    print("\nBatch complete.")
    print(f"Batch root: {batch_dir.relative_to(ROOT)}")
    for group, rows in group_rows.items():
        if rows:
            print(f"  {group}: {len(rows)} runs -> {summaries_dir / f'{group}_summary.csv'}")


if __name__ == "__main__":
    main()
