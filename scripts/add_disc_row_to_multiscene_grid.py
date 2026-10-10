#!/usr/bin/env python3
"""Add the OccuFly-reported DISC row to the paper's multiscene SSC grid.

The input grid is treated as an immutable source: this script always writes a
new file.  The five DISC panels are extracted individually from the original
embedded raster in Figure 4 of the OccuFly paper, then inserted with one shared
scale so their relative proportions and orientations are preserved.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont


# Figure 4 columns in the embedded 2456 x 988 OccuFly raster.  Bounds include
# the complete prediction content but exclude the row label and neighboring
# panels.  The order is the same as the paper and our multiscene comparison.
DISC_COLUMN_BOUNDS = (
    (73, 538),
    (586, 994),
    (1063, 1438),
    (1515, 1980),
    (2027, 2423),
)

# Axes used by assemble_occufly_multiscene_ssc_grid.py at 600 dpi.
TARGET_COLUMN_BOUNDS = (
    (90, 489),
    (491, 890),
    (893, 1291),
    (1294, 1693),
    (1695, 2094),
)


def extract_figure_four(pdf_path: Path) -> Image.Image:
    document = pymupdf.open(pdf_path)
    page = document[7]  # Published paper page containing Figure 4.
    candidates = []
    for image_info in page.get_images(full=True):
        xref = image_info[0]
        payload = document.extract_image(xref)
        candidates.append((payload["width"] * payload["height"], payload))
    if not candidates:
        raise RuntimeError("No embedded image found on OccuFly paper page 8")
    payload = max(candidates, key=lambda item: item[0])[1]
    from io import BytesIO

    figure = Image.open(BytesIO(payload["image"])).convert("RGB")
    if figure.size != (2456, 988):
        raise RuntimeError(f"Unexpected Figure 4 raster size: {figure.size}")
    return figure


def crop_disc_panels(figure: Image.Image, output_dir: Path) -> list[Image.Image]:
    pixels = np.asarray(figure)
    panels = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (x0, x1) in enumerate(DISC_COLUMN_BOUNDS, start=1):
        # The DISC prediction content begins below y=680.  Starting there also
        # excludes the tail of the GT row and the camera glyph above DISC.
        region = pixels[680:988, x0:x1]
        # Select colored voxel marks, excluding the black camera-orientation
        # glyph above them.  Padding retains antialiasing at the voxel edges.
        spread = region.max(axis=2).astype(np.int16) - region.min(axis=2).astype(np.int16)
        colored = (spread > 18) & (region.min(axis=2) < 245)
        ys, xs = np.where(colored)
        if len(xs) == 0:
            raise RuntimeError(f"No DISC voxel content detected in panel {index}")
        pad = 3
        box = (
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(region.shape[1], int(xs.max()) + pad + 1),
            min(region.shape[0], int(ys.max()) + pad + 1),
        )
        panel = Image.fromarray(region).crop(box)
        panel.save(output_dir / f"disc_{index}.png")
        panels.append(panel)
    return panels


def add_row(source_grid: Path, panels: list[Image.Image], output: Path) -> None:
    source = Image.open(source_grid).convert("RGBA")
    if source.width != 2100:
        raise RuntimeError(f"Unexpected source grid width: {source.width}")

    # Insert immediately before SSC 1.  The 272 px pitch matches the existing
    # voxel rows and leaves the original pixels above and below untouched.
    insertion_y = 680
    row_height = 272
    result = Image.new("RGBA", (source.width, source.height + row_height), "white")
    result.paste(source.crop((0, 0, source.width, insertion_y)), (0, 0))
    result.paste(
        source.crop((0, insertion_y, source.width, source.height)),
        (0, insertion_y + row_height),
    )

    max_width = max(panel.width for panel in panels)
    max_height = max(panel.height for panel in panels)
    scale = min(385 / max_width, 252 / max_height)
    for panel, (x0, x1) in zip(panels, TARGET_COLUMN_BOUNDS):
        size = (round(panel.width * scale), round(panel.height * scale))
        resized = panel.resize(size, Image.Resampling.LANCZOS).convert("RGBA")
        x = round((x0 + x1 - resized.width) / 2)
        y = insertion_y + round((row_height - resized.height) / 2)
        result.alpha_composite(resized, (x, y))

    # The legacy six-row grid used negative row spacing, leaving two detached
    # SSC1 tips above the actual SSC1 row after insertion.  Clear only those
    # empty-gap fragments; the complete SSC1 panels below remain untouched.
    for column in (2, 4):
        x0, x1 = TARGET_COLUMN_BOUNDS[column]
        result.paste(Image.new("RGBA", (x1 - x0, 60), "white"), (x0, 925))

    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    # Match the other labels' approximately 50 px type.  The longer name is
    # split into two adjacent vertical lines so it fits without shrinking.
    font = ImageFont.truetype(str(font_path), 48)
    for x, text in ((0, "SSC0 -"), (43, "DISC")):
        bbox = font.getbbox(text)
        label = Image.new(
            "RGBA", (bbox[2] - bbox[0] + 4, bbox[3] - bbox[1] + 4),
            (255, 255, 255, 0),
        )
        draw = ImageDraw.Draw(label)
        draw.text((2 - bbox[0], 2 - bbox[1]), text, fill="black", font=font)
        vertical = label.rotate(90, expand=True)
        result.alpha_composite(
            vertical,
            (x, insertion_y + round((row_height - vertical.height) / 2)),
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-pdf", type=Path, required=True)
    parser.add_argument("--source-grid", type=Path, required=True)
    parser.add_argument("--panels-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    figure = extract_figure_four(args.paper_pdf)
    panels = crop_disc_panels(figure, args.panels_dir)
    add_row(args.source_grid, panels, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
