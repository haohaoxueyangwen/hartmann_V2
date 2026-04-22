
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from typing import List, Sequence, Tuple

from analysis.optimization_common import coerce_global_offsets_xy
from analysis.subpixel_common import subpixel_peak_offset_2d


def calculate_expected_spot_shift(
    global_offsets_px: Sequence[Sequence[float]] | Tuple[float, float],
) -> Tuple[float, float]:
    """Return the single per-pulse shift only for uniform trajectories.

    Non-uniform trajectories do not have a single expected spot shift. Callers
    must use the full global_offsets_px sequence instead of collapsing it.
    """
    arr = np.asarray(global_offsets_px, dtype=float)
    if arr.ndim == 1 and arr.shape == (2,):
        return float(arr[0]), float(arr[1])
    arr = coerce_global_offsets_xy(global_offsets_px)
    if arr.shape[0] < 2:
        return 0.0, 0.0
    steps = np.diff(arr, axis=0)
    if not np.allclose(steps, steps[0], rtol=1e-9, atol=1e-12):
        raise ValueError(
            "non-uniform trajectory does not have a single expected spot shift; "
            "use the full global_offsets_px sequence instead."
        )
    step = steps[0]
    return float(step[0]), float(step[1])


def detect_multi_pulse_peaks(
    img: np.ndarray,
    min_distance: int = 5,
    threshold_rel: float = 0.1,
) -> List[dict]:
    arr = np.asarray(img, dtype=float)
    vmax = float(arr.max()) if arr.size else 0.0
    if vmax <= 0:
        return []
    thr = vmax * float(threshold_rel)
    k = int(max(1, min_distance))
    footprint = np.ones((2 * k + 1, 2 * k + 1))
    local_max = (arr == ndi.maximum_filter(arr, footprint=footprint)) & (arr > thr)
    ys, xs = np.nonzero(local_max)
    return [
        {"x_px": float(x), "y_px": float(y), "val": float(arr[y, x])}
        for x, y in zip(xs, ys)
    ]


def extract_template_from_image(
    image: np.ndarray,
    center: Tuple[float, float],
    radius_px: int,
    *,
    hann: bool = True,
    subtract_bg: bool = True,
) -> np.ndarray:
    img = np.asarray(image, dtype=float)
    y0, x0 = float(center[1]), float(center[0])
    r = int(radius_px)
    y1, y2 = int(round(y0 - r)), int(round(y0 + r))
    x1, x2 = int(round(x0 - r)), int(round(x0 + r))
    if y1 < 0 or x1 < 0 or y2 >= img.shape[0] or x2 >= img.shape[1]:
        raise ValueError("Template ROI out of bounds")
    tpl = img[y1 : y2 + 1, x1 : x2 + 1].copy()
    if subtract_bg:
        tpl -= tpl.mean()
    if hann:
        wx = np.hanning(tpl.shape[1])
        wy = np.hanning(tpl.shape[0])
        tpl = tpl * wy[:, None] * wx[None, :]
    tpl_norm = np.linalg.norm(tpl)
    if tpl_norm > 1e-9:
        tpl /= tpl_norm
    return tpl


def _zncc(a: np.ndarray, b: np.ndarray) -> float:
    # Remove DC components so correlation depends on shape/texture, not background level.
    ref = np.asarray(a, dtype=float)
    tgt = np.asarray(b, dtype=float)
    ref = ref - ref.mean()
    tgt = tgt - tgt.mean()
    da = np.linalg.norm(ref)
    db = np.linalg.norm(tgt)
    if da <= 1e-12 or db <= 1e-12:
        return -1.0
    return float(np.sum(ref * tgt) / (da * db))


def refine_centers_with_xcorr(
    image: np.ndarray,
    peaks: Sequence[dict],
    roi_radius: int = 15,
    search_radius: int = 20,
    template: np.ndarray | None = None,
    *,
    threshold_score: float | None = None,
    collect_scores: bool = False,
) -> List[dict] | tuple[List[dict], list[dict]]:
    img = np.asarray(image, dtype=float)
    ny, nx = img.shape
    R = int(search_radius)
    r = int(roi_radius)
    tpl = template
    if tpl is not None:
        tpl = np.asarray(tpl, dtype=float)
        tpl = tpl - tpl.mean()
        tpl_energy = np.linalg.norm(tpl)
        if tpl_energy > 1e-9:
            tpl = tpl / tpl_energy

    out: List[dict] = []
    scores_debug: list[dict] = []
    for pk in peaks:
        xc0 = float(pk.get("x_px", 0.0))
        yc0 = float(pk.get("y_px", 0.0))
        x0, y0 = int(round(xc0)), int(round(yc0))
        y1, y2 = y0 - r, y0 + r
        x1, x2 = x0 - r, x0 + r
        if y1 < 0 or x1 < 0 or y2 >= ny or x2 >= nx:
            out.append({**pk, "status": "out_of_bounds"})
            continue
        best_s = -1.0
        best_dx = 0
        best_dy = 0
        scores = np.full((2 * R + 1, 2 * R + 1), -1.0, dtype=float)
        for dy in range(-R, R + 1):
            ys1, ys2 = y1 + dy, y2 + dy
            if ys1 < 0 or ys2 >= ny:
                continue
            for dx in range(-R, R + 1):
                xs1, xs2 = x1 + dx, x2 + dx
                if xs1 < 0 or xs2 >= nx:
                    continue
                patch = img[ys1 : ys2 + 1, xs1 : xs2 + 1]
                if tpl is None:
                    patch0 = img[y1 : y2 + 1, x1 : x2 + 1]
                    s = _zncc(patch, patch0)
                else:
                    s = _zncc(patch, tpl)
                scores[dy + R, dx + R] = s
                if s > best_s:
                    best_s = s
                    best_dx = dx
                    best_dy = dy
        dx_sub, dy_sub = subpixel_peak_offset_2d(scores, best_dy + R, best_dx + R, method="parabolic")
        cx = x0 + best_dx + dx_sub
        cy = y0 + best_dy + dy_sub
        status = "ok"
        if threshold_score is not None and best_s < float(threshold_score):
            status = "low_score"
        out.append(
            {
                **pk,
                "x_px": float(cx),
                "y_px": float(cy),
                "score": float(best_s),
                "status": status,
            }
        )
        if collect_scores:
            scores_debug.append(
                {
                    "scores": scores.copy(),
                    "best_dx": int(best_dx),
                    "best_dy": int(best_dy),
                    "x0": x0,
                    "y0": y0,
                }
            )
    if collect_scores:
        return out, scores_debug
    return out


def group_spots_by_trajectory(
    refined_spots: Sequence[dict],
    ref_grid_x: Sequence[float],
    ref_grid_y: Sequence[float],
    global_offsets_px: Sequence[Sequence[float]] | Tuple[float, float],
    n_pulses: int | None = None,
    tolerance_px: float = 10.0,
) -> List[List[dict]]:
    ref_x = np.asarray(ref_grid_x, dtype=float)
    ref_y = np.asarray(ref_grid_y, dtype=float)
    offsets = coerce_global_offsets_xy(global_offsets_px, n_frames=n_pulses, name="global_offsets_px")
    frames: List[List[dict]] = [[] for _ in range(int(offsets.shape[0]))]

    if len(refined_spots) == 0 or len(ref_x) == 0:
        return frames

    spots_arr = np.asarray(
        [[float(s.get("x_px", 0.0)), float(s.get("y_px", 0.0))] for s in refined_spots],
        dtype=float,
    )
    used = np.zeros(len(refined_spots), dtype=bool)

    for ref_idx, (x0, y0) in enumerate(zip(ref_x, ref_y)):
        for k in range(int(offsets.shape[0])):
            px = x0 + offsets[k, 0]
            py = y0 + offsets[k, 1]
            d2 = np.sum((spots_arr - np.array([px, py])) ** 2, axis=1)
            cand = int(np.argmin(d2))
            if used[cand]:
                continue
            dist = float(np.sqrt(d2[cand]))
            if dist > float(tolerance_px):
                continue
            used[cand] = True
            meas_x, meas_y = spots_arr[cand]
            frames[k].append(
                {
                    "idx": int(ref_idx),
                    "x_ref_px": float(px),
                    "y_ref_px": float(py),
                    "x_sim_px": float(meas_x),
                    "y_sim_px": float(meas_y),
                    "dx_px": float(meas_x - px),
                    "dy_px": float(meas_y - py),
                    "status": "ok",
                    "score": float(refined_spots[cand].get("score", 0.0)),
                }
            )
    return frames
