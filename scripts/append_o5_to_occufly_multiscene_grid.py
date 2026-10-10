#!/usr/bin/env python3
"""Append canonical O5 renders to the archived five-row Fig. 6 raster.

The existing five rows are preserved pixel-for-pixel.  Each O5 panel uses the
same renderer and GT-derived camera viewport; GT and O5 are cropped together
per scene so their relative scale is unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


UIDS = [
    "scene_08_30_000299",
    "scene_09_40_000267",
    "scene_09_30_000145",
    "scene_08_40_000245",
    "scene_09_30_000233",
]


def content_crop(gt: Image.Image, o5: Image.Image) -> Image.Image:
    panels = []
    bounds = []
    for source in (gt, o5):
        pixels = np.asarray(source.convert("RGB")).copy()
        pixels[:90, 200:290] = 255
        panels.append(Image.fromarray(pixels))
        nonwhite = np.any(pixels < 245, axis=2)
        ys, xs = np.where(nonwhite)
        bounds.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    pad = 4
    width, height = panels[0].size
    common = (
        max(0, min(bound[0] for bound in bounds) - pad),
        max(0, min(bound[1] for bound in bounds) - pad),
        min(width, max(bound[2] for bound in bounds) + pad),
        min(height, max(bound[3] for bound in bounds) + pad),
    )
    crop = np.asarray(panels[1].crop(common).convert("RGBA")).copy()
    crop[np.all(crop[:, :, :3] > 250, axis=2), 3] = 0
    return Image.fromarray(crop)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = Image.open(args.base).convert("RGBA")
    if base.size != (2100, 1530):
        raise ValueError(f"Expected the archived 2100x1530 grid, got {base.size}")

    row_step = 273
    row_top = 1489
    row_height = 307
    column_left = (90, 491, 893, 1294, 1695)
    column_width = 399
    canvas = Image.new("RGBA", (base.width, base.height + row_step), "white")
    # Preserve the archived raster verbatim before placing the adjacent O5 row.
    canvas.paste(base, (0, 0))

    for left, uid in zip(column_left, UIDS):
        root = args.panel_root / uid
        panel = content_crop(Image.open(root / "gt.png"), Image.open(root / "o5.png"))
        panel.thumbnail((column_width, row_height), Image.Resampling.LANCZOS)
        x = left + (column_width - panel.width) // 2
        y = row_top + (row_height - panel.height) // 2
        canvas.alpha_composite(panel, (x, y))

    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = ImageFont.truetype(str(font_path), 50)
    label = Image.new("RGBA", (120, 180), (255, 255, 255, 0))
    draw = ImageDraw.Draw(label)
    box = draw.textbbox((0, 0), "O5", font=font)
    draw.text(
        ((label.width - (box[2] - box[0])) / 2, (label.height - (box[3] - box[1])) / 2 - box[1]),
        "O5",
        fill="black",
        font=font,
    )
    label = label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    canvas.alpha_composite(label, (10, row_top + (row_height - label.height) // 2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, dpi=(600, 600))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
