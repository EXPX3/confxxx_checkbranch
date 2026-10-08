#!/usr/bin/env python3
"""Assemble five paper-ready MDE/SSC qualitative comparison figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

UIDS = [
    "scene_08_30_000299", "scene_09_40_000267", "scene_09_30_000145",
    "scene_08_40_000245", "scene_09_30_000233",
]
MDE = [
    ("mde_vj_linear", "VJ Linear Probe RGB"),
    ("mde_vj_dta_film_mvs", "VJ Video DTA-FiLM-MVS"),
    ("mde_d3_film_dpt", "D3 FiLM-DPT-RGB"),
    ("mde_mvsformer_d3hplus", "MVSFormer++/D3-H+"),
]
MDE_COMPACT_TITLES = ["MDE 1", "MDE 3", "MDE 8", "MDE 9"]
SSC = [("ssc1", "SSC 1"), ("ssc2", "SSC 2"), ("ssc4", "SSC 4"), ("ssc5", "SSC 5")]

CHECKPOINTS = {
    "mde_vj_linear": "/datatank/giridhar.vb/data/checkpoints/my_checkpoints/08_accv26_reviewer_vjepa21_rgb_depth_comparison_seed0_20260829/linear_probe/linearprobe_best.pt",
    "mde_vj_dta_film_mvs": "/datatank/giridhar.vb/data/checkpoints/my_checkpoints/09a_wacv27_posemvs_depthonly_seed0_20260829/ssc_best.pt",
    "mde_d3_film_dpt": "/datatank/giridhar.vb/data/checkpoints/my_checkpoints/02_dinov3_vith16plus_film_dpt_occufly_parity_seed0_20260829/dpt_film_best.pt",
    "mde_mvsformer_d3hplus": "/datatank/giridhar.vb/data/checkpoints/posed_mvs_depth/official_structure_occufly/seed0_screen_5epoch_ddp8/dinov3_hplus/seed_0/best.pt",
    "ssc1": "/datatank/giridhar.vb/repos/ws_jepa_occufly_test/CGFormer/logs/ssc/cgformer/dav2prior_train/tensorboard/version_0/checkpoints/best.ckpt",
    "ssc2": "/datatank/giridhar.vb/repos/ws_jepa_occufly_test/experiments_wacv27/outputs/results/ssc/fssc_rgb_context_dpt_depth_film_dpt_reproduction_fixed_20260826/ssc_epoch_009.pt",
    "ssc4": "/datatank/giridhar.vb/data/checkpoints/my_checkpoints/v5v2dinov3_seed43_20261006/best_val_miou.pt",
    "ssc5": "/datatank/giridhar.vb/data/checkpoints/my_checkpoints/07b_dinov3_mvsformerpp_voxdet_v5v2_seed2_replication_20261006/best_val_miou.pt",
}


def resize_float(array: np.ndarray, size=(384, 384), nearest=False) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32).squeeze()
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(array, mode="F").resize(size, mode), dtype=np.float32)


def read_mde(path: Path) -> np.ndarray:
    if not path.exists() and path.parent.name == "mde_d3_film_dpt":
        # SSC4 exported the dense output of the same strict D3 FiLM-DPT
        # checkpoint while producing its five SSC predictions.
        path = path.parents[1] / "ssc4" / path.name
        payload = np.load(path)
        return resize_float(payload["depth"])
    payload = np.load(path)
    return resize_float(payload["prediction"])


def load_rgb(root: Path, uid: str) -> np.ndarray:
    scene, altitude, frame = uid[:8], uid[9:11], uid[-6:]
    path = root / scene / altitude / "images" / "visual" / f"{frame}.png"
    return np.asarray(Image.open(path).convert("RGB").resize((384, 384), Image.Resampling.BICUBIC))


def load_gt_depth(workspace: Path, uid: str):
    payload = np.load(workspace / "raw" / "mde_vj_linear" / f"{uid}.npz")
    gt = resize_float(payload["ground_truth"])
    valid = resize_float(payload["valid_mask"].astype(np.float32), nearest=True) > 0.5
    return gt, valid


def load_voxel_panels(voxel_root: Path, uid: str, compact: bool) -> dict[str, Image.Image]:
    """Load voxel panels and, for print, apply one shared content crop.

    A common crop preserves the exact relative scale, camera, and orientation
    across GT and predictions while removing unused canvas around every panel.
    """
    names = ["gt", "ssc1", "ssc2", "ssc4", "ssc5"]
    panels = {name: Image.open(voxel_root / uid / f"{name}.png").convert("RGB") for name in names}
    if not compact:
        return panels

    bounds = []
    for panel in panels.values():
        pixels = np.asarray(panel)
        nonwhite = np.any(pixels < 245, axis=2)
        ys, xs = np.where(nonwhite)
        bounds.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    pad = 6
    width, height = next(iter(panels.values())).size
    common = (
        max(0, min(bound[0] for bound in bounds) - pad),
        max(0, min(bound[1] for bound in bounds) - pad),
        min(width, max(bound[2] for bound in bounds) + pad),
        min(height, max(bound[3] for bound in bounds) + pad),
    )
    return {name: panel.crop(common) for name, panel in panels.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--voxel-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uids", nargs="*", choices=UIDS, default=None)
    parser.add_argument(
        "--paper-compact",
        action="store_true",
        help="Single-column paper layout with compact labels and no repeated error headings.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    selected_uids = args.uids or UIDS
    for uid in selected_uids:
        scene, altitude, frame = uid[:8], uid[9:11], uid[-6:]
        rgb = load_rgb(args.dataset_root, uid)
        gt, valid = load_gt_depth(args.workspace, uid)
        predictions = [read_mde(args.workspace / "raw" / folder / f"{uid}.npz") for folder, _ in MDE]
        voxel_panels = load_voxel_panels(args.voxel_root, uid, args.paper_compact)

        if args.paper_compact:
            fig = plt.figure(figsize=(3.5, 2.15), facecolor="white")
            title_size = 5.8
        else:
            fig = plt.figure(figsize=(18.2, 10.9), facecolor="white")
            title_size = 13
        # Leave enough inter-row space for labels without reducing the Fig. 4
        # voxel panels' native 489:343 landscape aspect ratio.
        grid = fig.add_gridspec(
            3, 5,
            height_ratios=(1.0, 1.0, 1.0) if args.paper_compact else (1.0, 1.0, 0.78),
            hspace=-0.030 if args.paper_compact else 0.10,
            wspace=0.025,
        )
        axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(5)] for row in range(3)])
        for ax in axes.ravel():
            ax.axis("off")
        if args.paper_compact:
            for ax in axes[0]:
                ax.set_anchor("S")
            for ax in axes[2]:
                ax.set_anchor("N")
        if not args.paper_compact:
            fig.suptitle(f"OccuFly qualitative comparison: {scene.replace('_', ' ')}, altitude {altitude} m, frame {frame}", fontsize=22, fontweight="bold", y=0.995)

        axes[0, 0].imshow(rgb)
        if not args.paper_compact:
            axes[0, 0].set_title(f"RGB {altitude} m / {frame}", fontsize=title_size, fontweight="bold", pad=1.0)
        depth_image = axes[1, 0].imshow(np.where(valid, gt, np.nan), cmap="viridis", vmin=0, vmax=80)
        if not args.paper_compact:
            axes[1, 0].set_title("GT depth", fontsize=title_size, fontweight="bold", pad=1.0)
        axes[2, 0].imshow(voxel_panels["gt"])
        if not args.paper_compact:
            axes[2, 0].set_title("GT voxel grid", fontsize=title_size, fontweight="bold", pad=1.0)

        error_image = None
        for col, ((_, mde_title), prediction, (ssc_folder, ssc_title)) in enumerate(zip(MDE, predictions, SSC), start=1):
            axes[0, col].imshow(prediction, cmap="viridis", vmin=0, vmax=80)
            shown_mde_title = MDE_COMPACT_TITLES[col - 1] if args.paper_compact else mde_title
            axes[0, col].set_title(shown_mde_title, fontsize=title_size, fontweight="bold", pad=1.0)
            error = np.where(valid, np.abs(prediction - gt), np.nan)
            error_image = axes[1, col].imshow(error, cmap="magma", vmin=0, vmax=20)
            if not args.paper_compact:
                axes[1, col].set_title("Absolute error", fontsize=12)
            if args.paper_compact:
                axes[2, col].text(
                    0.5, -0.025, ssc_title, transform=axes[2, col].transAxes,
                    ha="center", va="top", fontsize=title_size, fontweight="bold",
                )
            else:
                axes[2, col].set_title(ssc_title, fontsize=title_size, fontweight="bold", pad=1.0)
            if ssc_folder is None:
                axes[2, col].set_facecolor("#f4f4f4")
                axes[2, col].text(0.5, 0.5, "SSC 2 skipped\n(no substituted model)", ha="center", va="center", fontsize=14, color="#555", transform=axes[2, col].transAxes)
            else:
                axes[2, col].imshow(voxel_panels[ssc_folder])

        fig.subplots_adjust(
            left=0.038 if args.paper_compact else 0.006,
            right=0.895 if args.paper_compact else 0.945,
            bottom=0.038 if args.paper_compact else 0.006,
            top=0.965 if args.paper_compact else 0.955,
        )
        if args.paper_compact:
            for row, label in enumerate(("RGB", "GT depth", "GT voxel grid")):
                box = axes[row, 0].get_position()
                fig.text(
                    0.013, 0.5 * (box.y0 + box.y1), label,
                    rotation=90, ha="center", va="center",
                    fontsize=5.5, fontweight="bold",
                )

        if args.paper_compact:
            row_box = axes[0, -1].get_position()
            cax1 = fig.add_axes([0.910, row_box.y0 + 0.010, 0.012, row_box.height - 0.010])
        else:
            cax1 = fig.add_axes([0.955, 0.655, 0.012, 0.285])
        cb1 = fig.colorbar(depth_image, cax=cax1, ticks=[0, 40, 80] if args.paper_compact else None)
        cb1.set_label("Depth (m)", fontsize=5.5 if args.paper_compact else 12, fontweight="bold", labelpad=1.5)
        cb1.ax.tick_params(labelsize=5.2 if args.paper_compact else 10, length=1.5)
        if args.paper_compact:
            row_box = axes[1, -1].get_position()
            cax2 = fig.add_axes([0.910, row_box.y0, 0.012, row_box.height - 0.010])
        else:
            cax2 = fig.add_axes([0.955, 0.36, 0.012, 0.285])
        cb2 = fig.colorbar(error_image, cax=cax2, ticks=[0, 10, 20] if args.paper_compact else None)
        cb2.set_label("Absolute error (m)", fontsize=5.5 if args.paper_compact else 12, fontweight="bold", labelpad=1.5)
        cb2.ax.tick_params(labelsize=5.2 if args.paper_compact else 10, length=1.5)
        out_path = args.output / f"fig6_occufly_{uid}.png"
        fig.savefig(out_path, dpi=600 if args.paper_compact else 180, facecolor="white")
        plt.close(fig)
        manifest.append({"uid": uid, "output": str(out_path), "ssc2": "rendered_from_exact_table_iii_checkpoint"})
        print(f"saved {out_path}", flush=True)
    provenance = {
        "figure_count": len(manifest),
        "uids_in_occufly_figure_4_first_row_order": UIDS,
        "figures": manifest,
        "checkpoints": CHECKPOINTS,
        "ssc2": {
            "status": "rendered",
            "table_iii_method": "VJ-FSSC-ctx-DPT-depth-FiLM_DPT",
            "selected_epoch": 9,
            "checkpoint_sha256": "72f897ae7960143d83df522dbff967b455b989a15e92e58d841bab7ce4aabe0b",
            "reported_sc_iou_percent": 34.42029733520126,
            "reported_fixed_21_miou_percent": 2.9870438712427365,
            "label_space": "37 raw-ID-indexed OccuFly channels",
        },
        "d3_film_dpt_depth_source": (
            "Dense depth exported by the strictly loaded D3 FiLM-DPT module during "
            "the selected SSC4 inference; no non-strict or substitute model was used."
        ),
        "voxel_rendering": {
            "grid_whd": [192, 128, 128],
            "voxel_size_m": 0.5,
            "camera_grid_bounds_m": {
                "camera_right_x": [-48, 48],
                "camera_down_y": [-32, 32],
                "camera_forward_depth": [0, 64],
            },
            "display_transform": "(W,H,D) -> (x, y, -depth); aerial camera-forward is physical down",
            "valid_frustum_mask_applied_to_gt_and_all_predictions": True,
            "view_calibration": "common oblique aerial view calibrated against all five OccuFly Figure 4 GT panels",
            "same_transform_camera_palette_and_canvas_for_gt_and_predictions": True,
            "panel_size_px": [489, 343],
            "reference": "OccuFly paper Figure 4 layout",
        },
    }
    if not args.paper_compact:
        (args.output / "fig6_occufly_manifest.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
