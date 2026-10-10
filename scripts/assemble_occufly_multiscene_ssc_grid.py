#!/usr/bin/env python3
"""Assemble an OccuFly Fig. 4-style multiscene SSC comparison grid."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# OccuFly Fig. 4 first-row order, traced in the existing qualitative export.
UIDS = [
    "scene_08_30_000299",
    "scene_09_40_000267",
    "scene_09_30_000145",
    "scene_08_40_000245",
    "scene_09_30_000233",
]

VOXEL_ROWS = [
    ("gt", "GT"),
    ("ssc1", "SSC 1"),
    ("ssc2", "SSC 2"),
    ("ssc5", "SSC 5"),
    ("o5", "O5"),
]


def load_rgb(dataset_root: Path, uid: str) -> Image.Image:
    scene, altitude, frame = uid[:8], uid[9:11], uid[-6:]
    path = dataset_root / scene / altitude / "images" / "visual" / f"{frame}.png"
    return Image.open(path).convert("RGB").resize((384, 384), Image.Resampling.BICUBIC)


def load_voxels(voxel_root: Path) -> dict[tuple[str, str], Image.Image]:
    panels = {
        (uid, variant): Image.open(voxel_root / uid / f"{variant}.png").convert("RGB")
        for uid in UIDS
        for variant, _ in VOXEL_ROWS
    }

    # Crop each scene consistently across GT and all predictions. The repeated
    # camera glyph occupies the top 70 px of every source render; omitting it
    # permits a much tighter overview without stretching or changing geometry.
    cropped = {}
    for uid in UIDS:
        scene_panels = {}
        for variant, _ in VOXEL_ROWS:
            pixels = np.asarray(panels[(uid, variant)]).copy()
            # The renderer places the same orientation glyph here in every
            # panel. Remove it for this overview; voxel content elsewhere is
            # untouched and the rendered view itself is unchanged.
            pixels[:90, 200:290] = 255
            scene_panels[variant] = Image.fromarray(pixels)
        bounds = []
        for panel in scene_panels.values():
            pixels = np.asarray(panel)
            nonwhite = np.any(pixels < 245, axis=2)
            ys, xs = np.where(nonwhite)
            bounds.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        pad = 4
        width, height = next(iter(scene_panels.values())).size
        common = (
            max(0, min(bound[0] for bound in bounds) - pad),
            max(0, min(bound[1] for bound in bounds) - pad),
            min(width, max(bound[2] for bound in bounds) + pad),
            min(height, max(bound[3] for bound in bounds) + pad),
        )
        for variant, panel in scene_panels.items():
            crop = np.asarray(panel.crop(common).convert("RGBA")).copy()
            empty = np.all(crop[:, :, :3] > 250, axis=2)
            crop[empty, 3] = 0
            cropped[(uid, variant)] = Image.fromarray(crop)
    return cropped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--voxel-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rgb = {uid: load_rgb(args.dataset_root, uid) for uid in UIDS}
    voxels = load_voxels(args.voxel_root)

    fig = plt.figure(figsize=(3.5, 3.0), facecolor="white")
    grid = fig.add_gridspec(
        6, 5,
        height_ratios=(1.0, 0.72, 0.72, 0.72, 0.72, 0.72),
        hspace=-0.105,
        wspace=0.006,
    )
    axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(5)] for row in range(6)])
    for ax in axes.ravel():
        ax.axis("off")

    for col, uid in enumerate(UIDS):
        axes[0, col].imshow(rgb[uid])
        for row, (variant, _) in enumerate(VOXEL_ROWS, start=1):
            axes[row, col].imshow(voxels[(uid, variant)])

    fig.subplots_adjust(left=0.043, right=0.997, bottom=0.004, top=0.996)
    for row, label in enumerate(("RGB", "GT", "SSC 1", "SSC 2", "SSC 5", "O5")):
        box = axes[row, 0].get_position()
        fig.text(
            0.014,
            0.5 * (box.y0 + box.y1),
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=6.0,
            fontweight="bold",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, facecolor="white")
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
