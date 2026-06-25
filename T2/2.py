#!/usr/bin/env python3
"""Plot representative spectral curves from the Salinas hyperspectral scene.

Default inputs:
  data/Salinas_corrected.mat
  data/Salinas_gt.mat

Default output:
  T2/2.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np

from hsi_utils import infer_dataset_key, label_for_class, load_cube, load_ground_truth, parse_classes, spectral_axis


def mean_spectrum_for_class(
    cube: np.ndarray,
    gt: np.ndarray,
    class_id: int,
    max_pixels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    flat_indices = np.flatnonzero(gt.ravel() == class_id)
    if flat_indices.size == 0:
        raise ValueError(f"Class {class_id} has no pixels in ground-truth image")

    if flat_indices.size > max_pixels:
        flat_indices = rng.choice(flat_indices, size=max_pixels, replace=False)

    rows, cols = np.unravel_index(flat_indices, gt.shape)
    # This selects at most max_pixels spectra, avoiding a large cube-sized copy.
    spectra = cube[rows, cols, :].astype(np.float32, copy=False)
    return spectra.mean(axis=0)


def normalize_spectrum(spectrum: np.ndarray, method: str) -> np.ndarray:
    arr = spectrum.astype(np.float32, copy=True)
    if method == "minmax":
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            return (arr - lo) / (hi - lo)
        return np.zeros_like(arr)
    if method == "l2":
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            return arr / norm
        return arr
    return arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube", type=Path, default=Path("data/Salinas_corrected.mat"))
    parser.add_argument("--gt", type=Path, default=Path("data/Salinas_gt.mat"))
    parser.add_argument("--output", type=Path, default=Path("T2/2.png"))
    parser.add_argument("--cube-variable", help="Name of the 3-D image variable inside the .mat file.")
    parser.add_argument("--gt-variable", help="Name of the 2-D ground-truth variable inside the .mat file.")
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class ids to plot. If omitted, the five largest nonzero classes are used.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=2000,
        help="Maximum sampled pixels per class. Lower this if memory is very tight.",
    )
    parser.add_argument(
        "--x-axis",
        choices=("wavelength", "band"),
        default="wavelength",
        help="Use approximate wavelength or band number on the spectral curve x-axis.",
    )
    parser.add_argument(
        "--normalize",
        choices=("none", "minmax", "l2"),
        default="none",
        help="Optional normalization for each plotted mean spectrum.",
    )
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cube, cube_variable = load_cube(args.cube, args.cube_variable)
    gt, gt_variable = load_ground_truth(args.gt, cube.shape[:2], args.gt_variable)
    class_ids = parse_classes(args.classes, gt)
    dataset_key = infer_dataset_key(str(args.cube), cube_variable, str(args.gt), gt_variable)
    rng = np.random.default_rng(args.seed)

    x, xlabel = spectral_axis(dataset_key, cube.shape[2], prefer_wavelength=args.x_axis == "wavelength")
    ylabel = "Mean at-sensor radiance value (a.u.)"
    if args.normalize == "minmax":
        ylabel = "Min-max normalized mean spectrum"
    elif args.normalize == "l2":
        ylabel = "L2-normalized mean spectrum"

    plt.figure(figsize=(10, 6), dpi=160)
    for class_id in class_ids:
        spectrum = mean_spectrum_for_class(cube, gt, class_id, args.max_pixels, rng)
        spectrum = normalize_spectrum(spectrum, args.normalize)
        label = label_for_class(dataset_key, class_id)
        plt.plot(x, spectrum, linewidth=1.6, label=f"{class_id}: {label}")

    plt.title("Representative spectral profiles in Salinas scene")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)
    plt.close()
    print(f"Saved spectral-curve plot to {args.output}")
    print(f"Cube variable: {cube_variable}; GT variable: {gt_variable}; shape: {cube.shape}")
    print(f"Classes: {class_ids}; max_pixels/class: {args.max_pixels}")
    print(f"X-axis: {args.x_axis}; normalization: {args.normalize}")


if __name__ == "__main__":
    main()
