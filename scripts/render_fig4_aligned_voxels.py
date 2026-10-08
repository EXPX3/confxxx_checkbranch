#!/usr/bin/env python3
"""Render the selected OccuFly voxel grids with one fixed Fig. 4 camera.

All GT and prediction panels use the same WHD-to-world transform, perspective
camera, semantic LUT, canvas, and crop.  Predictions are stored as contiguous
train IDs and are converted back to the public OccuFly raw taxonomy here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

GRID = (192, 128, 128)
VOXEL_M = 0.5
TRAIN_TO_RAW = np.asarray(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 21, 22, 33, 34, 35, 36],
    dtype=np.uint8,
)
COLORS = {
    1:(128,0,128), 2:(204,163,72), 3:(128,0,0), 4:(192,192,192),
    5:(246,120,40), 6:(0,255,0), 7:(112,148,32), 8:(64,64,0),
    9:(255,255,0), 11:(255,16,255), 12:(255,204,153), 13:(0,128,128),
    14:(0,0,255), 16:(255,0,0), 17:(64,160,120), 21:(255,160,0),
    22:(106,0,255), 33:(128,64,128), 34:(240,120,120),
    35:(255,255,128), 36:(128,128,64),
}


def unpack_mask(path: Path) -> np.ndarray:
    return np.unpackbits(np.fromfile(path, dtype=np.uint8))[:np.prod(GRID)].reshape(GRID).astype(bool)


def visible_surface(occupied: np.ndarray) -> np.ndarray:
    interior = occupied.copy()
    for axis in range(3):
        interior &= np.roll(occupied, 1, axis=axis)
        interior &= np.roll(occupied, -1, axis=axis)
    interior[[0, -1], :, :] = False
    interior[:, [0, -1], :] = False
    interior[:, :, [0, -1]] = False
    return occupied & ~interior


def camera_projection(points: np.ndarray, width: int, height: int):
    # Oblique aerial inspection camera used consistently for every panel.
    # Calibrated against all five published Figure 4 GT crops: azimuth -60°,
    # elevation 20° around the common grid center.
    position = np.asarray([103.0, -179.0, 47.0], dtype=np.float32)
    target = np.asarray([0.0, 0.0, -28.0], dtype=np.float32)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rel = points - position
    x = rel @ right
    y = rel @ up
    depth = rel @ forward
    focal = 0.5 * height / np.tan(np.deg2rad(33.0) / 2.0)
    px = width * 0.5 + focal * x / depth
    py = height * 0.40 - focal * y / depth
    return px, py, depth, focal


def world_points(indices: np.ndarray) -> np.ndarray:
    # OccuFly is camera-centric W,H,D.  For its downward-looking aerial
    # cameras, D/camera-forward is physical down.  Figure 4 therefore uses
    # (x, y, -depth) so roofs are above ground.  Using +depth caused the
    # original upside-down result.
    camera_xyz = (
        (indices.astype(np.float32) + 0.5) * VOXEL_M
        + np.asarray([-48.0, -32.0, 0.0], dtype=np.float32)
    )
    return np.stack(
        [camera_xyz[:, 0], camera_xyz[:, 1], -camera_xyz[:, 2]], axis=1
    )


def draw_frustum(draw: ImageDraw.ImageDraw, width: int, height: int):
    # Fig. 4 uses the same black camera glyph at the top-center of every panel.
    cx = width // 2
    xy = [(cx, 10), (cx-34, 58), (cx+34, 58), (cx+22, 80), (cx-22, 80)]
    for a, b in [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]:
        draw.line([xy[a], xy[b]], fill=(0,0,0), width=4)


def projected_surface(labels_raw: np.ndarray, mask: np.ndarray, width: int, height: int):
    occupied = mask & (labels_raw > 0) & (labels_raw != 255)
    boundary = visible_surface(occupied)
    indices = np.argwhere(boundary)
    labels = labels_raw[tuple(indices.T)]
    points = world_points(indices)
    px, py, depth, focal = camera_projection(points, width, height)
    valid = depth > 1
    return px[valid], py[valid], depth[valid], labels[valid], focal


def common_viewport(items, width: int, height: int):
    """Fit one shared camera/crop to GT and every prediction for a frame."""
    projected = [projected_surface(labels, mask, width, height) for labels, mask in items]
    # Derive the viewport from GT only, then reuse it unchanged for every
    # method. Predictions must not alter the framing of the reference scene.
    xs = projected[0][0]
    ys = projected[0][1]
    # Reserve the upper band for the camera glyph, as in OccuFly Fig. 4.
    left, right, top, bottom = 12.0, width - 12.0, 82.0, height - 10.0
    xmin, xmax = np.percentile(xs, [0.05, 99.95])
    ymin, ymax = np.percentile(ys, [0.05, 99.95])
    scale = min((right - left) / max(xmax - xmin, 1.0), (bottom - top) / max(ymax - ymin, 1.0))
    source_center = np.asarray([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5])
    target_center = np.asarray([(left + right) * 0.5, (top + bottom) * 0.5])
    return projected, scale, source_center, target_center


def render_projected(projected, viewport, out_path: Path, width=489, height=343):
    px, py, depth, labels, focal = projected
    scale, source_center, target_center = viewport
    px = (px - source_center[0]) * scale + target_center[0]
    py = (py - source_center[1]) * scale + target_center[1]
    valid = (px > -8) & (px < width + 8) & (py > -8) & (py < height + 8)
    px, py, depth, labels = px[valid], py[valid], depth[valid], labels[valid]
    order = np.argsort(depth)[::-1]
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for idx in order:
        color = COLORS.get(int(labels[idx]), (255, 255, 255))
        radius = max(1, min(4, int(round(scale * 0.52 * focal / depth[idx]))))
        x, y = int(round(px[idx])), int(round(py[idx]))
        draw.rectangle((x-radius, y-radius, x+radius, y+radius), fill=color)
    draw_frustum(draw, width, height)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def load_prediction(path: Path, kind: str) -> np.ndarray:
    if kind == "ssc1":
        train = np.fromfile(path, dtype=np.uint16).reshape(GRID)
    elif kind == "ssc2":
        # Historical Table-III row 2 predicts the public raw OccuFly IDs
        # directly with 37 output channels; it must not pass through the
        # contiguous-22 ID conversion used by the later treatments.
        return np.load(path)["prediction_whd"].astype(np.uint8, copy=False)
    else:
        train = np.load(path)["prediction_whd"]
    out = np.full(GRID, 255, dtype=np.uint8)
    valid = (train >= 0) & (train < len(TRAIN_TO_RAW))
    out[valid] = TRAIN_TO_RAW[train[valid].astype(np.int64)]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    uids = [
        "scene_08_30_000299", "scene_09_40_000267", "scene_09_30_000145",
        "scene_08_40_000245", "scene_09_30_000233",
    ]
    for uid in uids:
        scene, altitude, frame = uid[:8], uid[9:11], uid[-6:]
        gt_dir = args.dataset_root / scene / altitude / "ground_truth" / frame
        gt = np.fromfile(gt_dir / f"{frame}.label", dtype=np.uint8).reshape(GRID)
        invalid = unpack_mask(gt_dir / f"{frame}.invalid")
        valid = ~invalid
        # Figure 4 shows the occupied grid, not only the supplied observation
        # surface mask.  ``render`` extracts its visible exterior itself.
        p1 = args.workspace / "raw" / "ssc1" / "sequences" / f"{scene}_{altitude}" / "predictions" / f"{frame}.label"
        grids = [("gt", gt), ("ssc1", load_prediction(p1, "ssc1"))]
        for variant in ("ssc2", "ssc4", "ssc5"):
            path = args.workspace / "raw" / variant / f"{uid}.npz"
            grids.append((variant, load_prediction(path, variant)))
        projected, scale, source_center, target_center = common_viewport(
            [(labels, valid) for _, labels in grids], 489, 343
        )
        viewport = (scale, source_center, target_center)
        for (name, _), panel in zip(grids, projected):
            render_projected(panel, viewport, args.output / uid / f"{name}.png")
        print(f"rendered {uid}", flush=True)


if __name__ == "__main__":
    main()
