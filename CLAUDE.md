# CLAUDE.md

## Project Overview

FEL-WFS: a Python simulation and analysis suite for **multi-pulse wavefront sensing** in High-Repetition-Rate Free-Electron Lasers (FEL). Supports both X-ray and visible-laser modes.

Core workflow: `simulate.py` generates detector images -> `recover.py` reconstructs wavefront phase from those images.

## Quick Reference

```bash
# Run simulation (X-ray)
python simulate.py

# Run simulation (Laser, 527nm)
python simulate_laser.py

# Run recovery (all methods)
python recover.py result/001 --method all

# Run recovery (single method)
python recover.py result/001 --method centroid
python recover.py result/001 --method cross_correlation
```

## Project Structure

```
.
├── simulate.py              # X-ray simulation entry point
├── simulate_laser.py        # Laser-mode simulation entry point
├── recover.py               # Phase recovery entry point
├── config_simulate.py       # Simulation config (X-ray)
├── config_simulate_laser.py # Simulation config (Laser)
├── config_recover.py        # Recovery config
├── plot_spot.py             # Single-spot analysis utility
├── run_simulation.py        # Batch run helper
│
├── simulation/              # Physics engine
│   ├── wave_optics.py       # Core diffraction (Fresnel, Angular Spectrum)
│   ├── hartmann_simulator.py# Multi-pulse superposition orchestrator
│   ├── mask.py              # Hartmann/Microlens/slot mask generation
│   ├── aberration.py        # Zernike/PSD phase screens
│   ├── detector.py          # CMOS/CCD effects (QE, PSF, noise, ADC)
│   ├── wfe_generator.py     # Wavefront error presets
│   ├── simple_spots.py      # Gaussian spot approximation
│   ├── optimizations.py     # Computation optimizations
│   └── laser_mode/          # Laser-specific simulation modules
│       ├── simulator.py     # Laser simulator
│       ├── wave_optics.py   # Laser propagation
│       ├── mask.py          # Laser mask generation
│       ├── detector.py      # Laser detector model
│       ├── aberration.py    # Laser aberrations
│       ├── config.py        # Laser-mode internal config
│       └── truth.py         # Ground truth for laser mode
│
├── analysis/                # Algorithm core
│   ├── displacement_methods.py    # Centroid & cross-correlation detection
│   ├── hartmann_analysis.py       # High-level wavefront reconstruction
│   ├── zernike_reconstruction.py  # Zernike coefficient solver
│   ├── zernike_basis.py           # Zernike polynomial basis
│   ├── decomposition.py           # DPC & mode decomposition
│   ├── optimization_common.py     # GridGeometry, validation utilities
│   ├── shift_optimization.py      # Optimization-based shift estimation
│   ├── movement_metadata.py       # Translation/Rotation/Trajectory kinematics
│   ├── spot_recognition.py        # Spot detection & matching
│   ├── spot_template.py           # Template-based spot analysis
│   ├── spacing_solver.py          # Grid spacing estimation
│   ├── trajectory_design.py       # Trajectory planning
│   ├── multi_frame.py             # Multi-frame analysis
│   ├── performance.py             # Performance metrics
│   ├── subpixel_common.py         # Sub-pixel refinement
│   └── serialization.py           # Data serialization
│
├── visualization/           # Reporting & plotting
│   ├── visualize_results.py # PDF/PNG report generator
│   ├── visualization.py     # Core visualization functions
│   ├── plotter.py           # Standardized plot templates
│   └── mask_visualization.py# Mask geometry debugging
│
├── result/                  # Simulation output (per-run subdirectories)
│   ├── xray/                # X-ray simulation results
│   └── laser/               # Laser simulation results
│
└── new_hartmann_grating-dev/# Legacy/dev pipeline (reference only)
```

## Result Directory Structure

Each simulation run produces a subdirectory under `result/` containing:
- `accumulated.npy` - 2D detector image (sum of all pulses)
- `ideal_centers.npy` - Reference (un-aberrated) spot positions in px
- `gt_displacements.npz` - Ground truth local shifts for benchmarking
- `wfe_phase_map.npy` - Input phase screen (if available)
- `simulation_params.json` - Full parameter snapshot (required for recovery)

## Tech Stack

- **Python 3.13** (Anaconda)
- **NumPy** - Array computation
- **SciPy** - Signal processing, optimization
- **Matplotlib** - Visualization

## Coordinate Conventions (CRITICAL)

See `COORDINATE_CONVENTIONS.md` for full details. Summary:

| Layer | Format | Usage |
|-------|--------|-------|
| Physical | `(x_m, y_m)` in meters | Optical calculations |
| Detector pixel | `(x_px, y_px)` | Spot centers, subpixel shifts |
| Array index | `[row_px, col_px] = [y_px, x_px]` | NumPy indexing only |

Key rules:
- **Never mix coordinate layers** in the same variable
- Use suffixes: `_px` (pixels), `_m` (meters), `_rad` (radians)
- Point arrays shape: `(N, 2)` as `[x, y]`
- Image access: always `image[row, col]` = `image[y, x]`
- Light propagates along **+z**
- Angles: degrees (0-360), counter-clockwise from +X

## Physical Units

- **Length**: meters (m) internally
- **Energy**: electron-volts (eV)
- **Phase**: radians (rad)
- **Wavelength**: derived as `lambda = hc / E`

## Coding Conventions

- `snake_case` for functions/variables, `PascalCase` for classes
- Type hints mandatory for new code in `analysis/` and `simulation/`
- Comments in Chinese or English are both acceptable (existing codebase uses both)
- Simulation results must save `simulation_params.json` for recovery compatibility

## Important Constraints

1. **Read before edit**: Always read `simulation_params.json` from the target result folder before modifying recovery or analysis code, to understand the physical scale.
2. **Do not modify `wave_optics.py` core propagation** without empirical validation (Fresnel number check).
3. **Movement models**: Any new movement model must be registered in `analysis/movement_metadata.py` to work with both simulation and recovery.
4. **Config files are the source of truth**: `config_simulate.py`, `config_simulate_laser.py`, and `config_recover.py` define all tunable parameters. Do not hardcode values elsewhere.
