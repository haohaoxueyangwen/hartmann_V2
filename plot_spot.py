#!/usr/bin/env python3
from __future__ import annotations
"""Inspect the brightest spot in a saved detector frame."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_style import apply_plot_font_bump

apply_plot_font_bump()


def extract_brightest_spot(frame: np.ndarray, window_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Extract a square window around the brightest detector pixel."""
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"frame must be a 2D array, got shape {arr.shape}")
    half_window = int(window_size) // 2
    row_peak, col_peak = np.unravel_index(np.argmax(arr), arr.shape)
    row_min = max(0, row_peak - half_window)
    row_max = min(arr.shape[0], row_peak + half_window + 1)
    col_min = max(0, col_peak - half_window)
    col_max = min(arr.shape[1], col_peak + half_window + 1)
    return arr[row_min:row_max, col_min:col_max], (int(row_peak), int(col_peak))


def save_spot_report(frame_path: Path, output_path: Path, window_size: int) -> None:
    """Generate a 2D + 1D report for the brightest spot in a frame."""
    frame = np.load(frame_path)
    spot, (row_peak, col_peak) = extract_brightest_spot(frame, window_size=window_size)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    image_artist = axes[0].imshow(spot, cmap="magma", origin="lower")
    axes[0].set_title("Brightest Spot (2D)")
    plt.colorbar(image_artist, ax=axes[0])

    row_center = spot.shape[0] // 2
    col_center = spot.shape[1] // 2
    axes[1].plot(spot[row_center, :], label="Horizontal cross-section")
    axes[1].plot(spot[:, col_center], label="Vertical cross-section")
    axes[1].set_title("Cross-sections Through Spot Center")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(
        f"Saved {output_path}. Spot shape: {spot.shape}, "
        f"brightest frame pixel: (row={row_peak}, col={col_peak})"
    )
    print("\nCentral 9x9 numerical values (normalized to 0-100):")
    half_small = 4
    row0 = max(0, row_peak - half_small)
    row1 = min(frame.shape[0], row_peak + half_small + 1)
    col0 = max(0, col_peak - half_small)
    col1 = min(frame.shape[1], col_peak + half_small + 1)
    small_spot = np.asarray(frame[row0:row1, col0:col1], dtype=np.float64)
    max_value = float(np.max(small_spot)) if small_spot.size else 0.0
    if max_value > 0:
        small_spot = small_spot / max_value * 100.0
    for row in small_spot:
        print(" ".join(f"{value:4.0f}" for value in row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the brightest spot from a detector frame.")
    parser.add_argument("frame_path", help="Path to a saved detector frame `.npy` file")
    parser.add_argument(
        "--output",
        default="spot_analysis.png",
        help="Output figure path (default: %(default)s)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Square window size around the brightest spot (default: %(default)s)",
    )
    args = parser.parse_args()

    save_spot_report(
        frame_path=Path(args.frame_path),
        output_path=Path(args.output),
        window_size=int(args.window_size),
    )


if __name__ == "__main__":
    main()
