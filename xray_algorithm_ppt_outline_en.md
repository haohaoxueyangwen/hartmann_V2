# PPT Outline: Multi-Pulse X-ray Wavefront Sensing for High-Repetition-Rate FEL

> Note: this draft still references legacy `run_002` figures.
> The current authoritative reporting baseline is the case documented in
> [AI_HANDOFF_README.md](/Users/yaaang/Desktop/model2/AI_HANDOFF_README.md):
> `step = 8 um`, `z = 450 mm`, `WFE = 1 rad`.
> If this outline is reused for the next deck pass, the visuals and headline metrics
> should be refreshed to that current baseline before presentation use.

## Deck Goal

This presentation is meant to explain the algorithm, the experimental workflow, and the optical design in a clear narrative:

1. Why multi-pulse X-ray wavefront sensing is difficult in high-repetition-rate FEL.
2. How the current Hartmann-based simulation and recovery pipeline works.
3. What optical geometry and recovery method are used.
4. How well the algorithm performs on the baseline case and under parameter sweeps.
5. What operating point is recommended for the current system.

Recommended deck length: `14-16 slides`.

## Slide-by-Slide Outline

### Slide 1. Title

**Title**
Multi-Pulse Wavefront Sensing for High-Repetition-Rate X-ray FEL Using a Hartmann Sensor

**Subtitle**
Algorithm, Optical Design, Experimental Workflow, and Performance Evaluation

**Use**
- Presenter name
- Affiliation
- Date

**Visual**
- A clean title slide with one representative figure:
  [recovery_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/recovery_cross_correlation.png)

### Slide 2. Motivation and Problem Statement

**Core message**
- Conventional Shack-Hartmann / Hartmann sensing assumes one frame corresponds to one wavefront realization.
- In high-repetition-rate FEL, detector frame rate can be lower than the pulse repetition rate.
- As a result, multiple pulses are accumulated into one detector frame, which creates spot overlap and pulse mixing.

**Key points**
- We need to recover individual pulse wavefronts from one accumulated image.
- The challenge is not only centroid estimation, but also pulse separation and crosstalk suppression.
- This is the main algorithmic contribution of the workflow.

**Visual**
- Use one accumulated detector image or decomposition overview:
  [decomposition_images_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/decomposition_images_cross_correlation.png)

### Slide 3. Research Objective

**Core message**
- The goal is to build and evaluate a multi-pulse X-ray Hartmann wavefront sensing pipeline.

**Key points**
- Simulate accumulated multi-pulse X-ray Hartmann measurements.
- Decompose the accumulated sensor image into pulse-wise spot maps.
- Recover pulse-wise displacements and reconstruct wavefront phase.
- Evaluate accuracy, robustness, and operating limits.

**Visual**
- No dense figure needed.
- Use a simple text + icon layout.

### Slide 4. Optical Design and Measurement Geometry

**Core message**
- The current X-ray setup is a Hartmann-mask-based free-space propagation geometry.

**Suggested content**
- X-ray source / incoming distorted wavefront
- Hartmann mask
- Free-space propagation distance `z`
- Detector plane
- Multi-pulse translation trajectory across pulses

**Key parameters to show**
- Photon energy: `10 keV`
- Wavelength: `0.124 nm`
- Propagation distance: `z = 0.45 m`
- Hartmann pitch: `25 um`
- Hole diameter: `13 um`
- Detector pixel size: `0.65 um`
- Number of pulses: `10`

**Visual**
- This slide should use a clean schematic to be drawn for the PPT.
- We do not yet have a dedicated optical-layout figure for the X-ray mode, so this should be created as a new diagram.

### Slide 5. Experimental Workflow

**Core message**
- The workflow has two stages: forward simulation and inverse recovery.

**Suggested flow**
`Wavefront definition -> Hartmann propagation -> detector accumulation -> pulse decomposition -> displacement recovery -> phase reconstruction -> metrics`

**Implementation mapping**
- Forward simulation: [simulate.py](/Users/yaaang/Desktop/model2/simulate.py)
- Recovery: [recover.py](/Users/yaaang/Desktop/model2/recover.py)
- Batch sweeps: [run_xray_protocol_sweeps.py](/Users/yaaang/Desktop/model2/run_xray_protocol_sweeps.py)
- Sweep plotting: [plot_xray_protocol_sweeps.py](/Users/yaaang/Desktop/model2/plot_xray_protocol_sweeps.py)

**Visual**
- Best as a new pipeline diagram for the PPT.

### Slide 6. Recovery Algorithm

**Core message**
- The current recovery pipeline consists of decomposition plus cross-correlation-based displacement estimation.

**Algorithm steps**
1. Build or load the accumulated detector image.
2. Decompose the accumulated image into pulse-wise spot components.
3. Use cross-correlation to estimate local spot shifts.
4. Convert shifts to local slopes.
5. Integrate slopes to reconstruct phase.
6. Compare recovered phase with ground truth or reference metrics.

**Method settings to show**
- Main method: `cross_correlation`
- Patch half-size: `13`
- Correction factor: `1.0`
- Decomposition method: `iterative`
- Reference mode: `per_pulse`

**Visual**
- Use:
  [frame_000_summary.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/step3_decomposition/frame_000_summary.png)
  and
  [method_comparison.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/step3_decomposition/method_comparison.png)

### Slide 7. Baseline Experimental Configuration

**Core message**
- We define one baseline X-ray configuration and use it as the reference working point for all analysis.

**Key parameters**
- `z = 450 mm`
- `step = 7 um`
- `10 pulses`
- `26 x 26` sub-apertures
- Fresnel number `~0.757`
- Translation angle `19 deg`

**Optional table**
- Simulation parameters on the left
- Recovery parameters on the right

**Visual**
- A compact parameter table.

### Slide 8. Baseline Single-Run Result: Sensor Image to Pulse Decomposition

**Core message**
- The baseline case demonstrates that one accumulated frame can be decomposed into pulse-wise spot fields with high confidence.

**Key numbers**
- Mean decomposition confidence: `0.9978`
- Overall visibility: `0.9359`
- Mean step length in detector space: `10.77 px`

**Visual**
- [recovery_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/recovery_cross_correlation.png)
- [decomposition_images_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/decomposition_images_cross_correlation.png)
- [trajectory_occupancy.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/step3_decomposition/trajectory_design/trajectory_occupancy.png)

### Slide 9. Baseline Single-Run Result: Pulse-Wise Phase and Displacement Recovery

**Core message**
- After decomposition, the algorithm recovers pulse-wise displacement and phase maps with strong consistency.

**Key figures**
- [all_frames_phase_summary_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/all_frames_phase_summary_cross_correlation.png)
- [all_frames_disp_summary_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/all_frames_disp_summary_cross_correlation.png)
- [phase_all_frames_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/per_frame_phase/phase_all_frames_cross_correlation.png)
- [per_frame_rmse_cross_correlation.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/per_frame_displacement/per_frame_rmse_cross_correlation.png)

**Key message to say**
- The baseline run is stable across all 10 pulses rather than only on a single best frame.

### Slide 10. Method Comparison

**Core message**
- Cross-correlation is clearly the best-performing method in the current system.

**Suggested comparison table**

| Method | Avg. shift RMSE (px) | Avg. phase RMSE (rad) | Avg. phase corr. |
|---|---:|---:|---:|
| Centroid | `56.9241` | `60.1476` | `0.5275` |
| Cross-correlation | `0.0441` | `0.1111` | `0.9937` |
| Phase correlation | `0.1214` | `0.9730` | `0.6965` |

**Visual**
- [method_comparison.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/step3_decomposition/method_comparison.png)
- [error_statistics.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/step3_decomposition/error_statistics.png)

### Slide 11. Physical Performance: Resolution and Limits

**Core message**
- The baseline experiment already defines the measurable resolution and physical limits of the current Hartmann system.

**Numbers to show**
- Displacement resolution: `28.66 nm`
- Angular resolution: `0.06368 urad`
- Conservative `3sigma` displacement resolution: `85.97 nm`
- Dynamic range: `436 : 1`
- Highest measurable spatial frequency: `2.0e4 m^-1`
- Minimum resolvable period: `50 um`

**Visual**
- Best as a clean summary table.
- Optionally add one small inset from:
  [z_sweep_metrics.png](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/plots/z_sweep_metrics.png)

### Slide 12. Parameter Sweep Design

**Core message**
- Three controlled sweeps were used to map the operating envelope of the algorithm.

**Sweep matrix**
- `STEP_M`: `7.0 -> 10.0 um`, interval `0.5 um`
- `Z_PROP_M`: `450 -> 1000 mm`, interval `50 mm`
- `WFE`: `1 -> 5 rad`
- `N_PULSES = 10` fixed

**Visual**
- Use:
  [sweep_overview.png](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/plots/sweep_overview.png)

### Slide 13. Sweep Result I: STEP Dependence

**Core message**
- The stable operating region for pulse-to-pulse translation is around `7.0-8.0 um`.

**Key findings**
- Best shift RMSE at `7.0 um`
- Best phase RMSE at `8.0 um`
- Performance degrades for `STEP_M >= 9.5 um`
- At `10.0 um`, phase correlation drops significantly

**Visual**
- [step_sweep_metrics.png](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/plots/step_sweep_metrics.png)

### Slide 14. Sweep Result II: Propagation Distance Dependence

**Core message**
- The best practical propagation distance lies near `450-500 mm`.

**Key findings**
- Best shift RMSE at `500 mm`
- Best phase RMSE and best phase correlation at `450 mm`
- Performance degrades strongly beyond `700 mm`
- This reflects the physical trade-off between sensitivity and dynamic range

**Visual**
- [z_sweep_metrics.png](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/plots/z_sweep_metrics.png)

### Slide 15. Sweep Result III: Wavefront Error Dependence

**Core message**
- The method remains usable from `1-5 rad`, but low-aberration conditions provide the best accuracy.

**Key findings**
- Best performance at `WFE = 1 rad`
- Shift RMSE and phase RMSE increase with stronger aberration
- The algorithm remains fairly robust in phase correlation across the tested range

**Visual**
- [wfe_sweep_metrics.png](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/plots/wfe_sweep_metrics.png)

### Slide 16. Recommended Operating Point and Takeaways

**Core message**
- We can now define a practical operating point for the current X-ray multi-pulse Hartmann system.

**Recommended settings**
- `N_PULSES = 10`
- `STEP_M = 7.0 um`
- `Z_PROP_M = 450 mm`
- `WFE = 3.0 rad` as a representative medium-aberration benchmark
- Recovery method: `cross_correlation`
- Decomposition method: `iterative`

**Final takeaways**
- The algorithm works on accumulated multi-pulse X-ray Hartmann data.
- Cross-correlation is the only reliable primary method in the current pipeline.
- The current system already reaches nanometer-scale displacement sensitivity.
- The `450-500 mm` range is the most useful operating window for the present geometry.

**Visual**
- Use a clean summary slide with one inset:
  [sweep_overview.png](/Users/yaaang/Desktop/model2/result/xray/sweeps_20260407_002/plots/sweep_overview.png)

## Optional Appendix Slides

### Appendix A. Trajectory Design and Identifiability

**Use**
- To explain why pulse overlap becomes risky even when decomposition still works.

**Visual**
- [trajectory_path.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/step3_decomposition/trajectory_design/trajectory_path.png)
- [trajectory_occupancy.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/step3_decomposition/trajectory_design/trajectory_occupancy.png)

### Appendix B. Additional Frame-Level Results

**Use**
- To show one or two representative per-frame results if the audience asks for more detail.

**Visual**
- [displacement_frame_000.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/per_frame_displacement/displacement_frame_000.png)
- [phase_frame_000.png](/Users/yaaang/Desktop/model2/result/xray/run_002/analysis/per_frame_phase/phase_frame_000.png)

## Recommended PPT Style

To keep the talk strong and readable:

1. Use one clear message per slide.
2. Avoid putting too many metrics on the same slide.
3. Keep the optical design, algorithm pipeline, and performance results visually separated.
4. Use the baseline run as the narrative anchor, then show sweeps as validation.
5. Build two new custom diagrams for the deck:
   the X-ray optical layout;
   the simulation-to-recovery workflow.

## Next Step

This outline is ready to be turned into the actual PPT.
The next build step should be:

1. Create the two missing schematic slides.
2. Arrange the existing result figures into the slide layout.
3. Turn this outline into a `.pptx` file with English text.
