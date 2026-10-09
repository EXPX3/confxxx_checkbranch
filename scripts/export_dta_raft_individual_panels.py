#!/usr/bin/env python3
"""Export individual, publication-ready RAFT/DTA panels from the aligned checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize
from torchvision.utils import flow_to_image


PAPER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_ROOT.parent
SCRIPT_ROOT = PAPER_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from make_vjepa_raft_figure3_extension import (  # noqa: E402
    OccuFlyVideoTargetDataset,
    fit_shared_pca,
    make_model,
    pca_image,
    rgb_image,
)


RUN_DIR = (
    REPO_ROOT
    / "experiments_wacv27/outputs/mde_video/vjepa21video_spatial_alignment_grid_sample"
    / "real_dta__clip4__seed0__lr3e_4__wd1e_4"
)
OUTPUT_DIR = PAPER_ROOT / "content/images/dtavideo/raft"
REFERENCE = ("scene_08", 50, "000091")
ATTENTION_VARIANTS = [
    ("scene_08", 50, "000091"),
    ("scene_09", 40, "000708"),
    ("scene_09", 50, "000652"),
]


def dataset_for(config: dict, scene: str, height: int) -> OccuFlyVideoTargetDataset:
    return OccuFlyVideoTargetDataset(
        dataset_root=Path(config["dataset_root"]),
        scenes=[scene],
        heights=[height],
        image_size=int(config["image_size"]),
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
        png_depth_scale=float(config["png_depth_scale"]),
        clip_num_of_frames=int(config["clip_num_of_frames"]),
        clip_mode="real_dta",
        context_policy=config["context_policy"],
        skip_first_last=bool(config["skip_first_last"]),
        live_height=config.get("live_height"),
        shuffle_seed=int(config["seed"]),
    )


def select_frame(dataset: OccuFlyVideoTargetDataset, frame_id: str) -> int:
    for index, sample in enumerate(dataset.samples):
        if sample.target.frame_id == frame_id:
            return index
    raise RuntimeError(f"Frame {frame_id} is unavailable in the requested scene/height")


def save_rgb(path: Path, image: np.ndarray) -> None:
    plt.imsave(path, np.clip(image, 0.0, 1.0))


def save_scalar(path: Path, image: np.ndarray, cmap: str, vmin: float, vmax: float) -> None:
    plt.imsave(path, image, cmap=cmap, vmin=vmin, vmax=vmax)


def save_scalar_with_embedded_legend(
    path: Path,
    image: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    ticks: list[float],
    ticklabels: list[str],
    legend_title: str,
) -> None:
    """Save a square scalar panel with a compact legend attached underneath."""
    fig = plt.figure(figsize=(4.0, 4.82), dpi=120, facecolor="white")
    image_ax = fig.add_axes([0.0, 0.20, 1.0, 0.80])
    image_ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    image_ax.axis("off")
    colorbar_ax = fig.add_axes([0.075, 0.115, 0.85, 0.033])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=ticks,
    )
    tick_texts = colorbar.ax.set_xticklabels(ticklabels)
    if tick_texts:
        tick_texts[0].set_ha("left")
        tick_texts[-1].set_ha("right")
    colorbar.ax.tick_params(labelsize=7.6, length=2, pad=2)
    colorbar.ax.set_title(legend_title, fontsize=8.2, pad=2)
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)


def upsample_nearest(array: torch.Tensor, size: int = 384) -> np.ndarray:
    return (
        F.interpolate(array[None, None].float(), size=(size, size), mode="nearest")[0, 0]
        .detach()
        .cpu()
        .numpy()
    )


def save_colorbar(path: Path, cmap: str, vmin: float, vmax: float, label: str) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 0.72))
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.55, top=0.96)
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
        cax=ax,
        orientation="horizontal",
    )
    colorbar.set_label(label, fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    fig.savefig(path, dpi=180, transparent=True)
    plt.close(fig)


def attention_delta(aux) -> torch.Tensor:
    attention = aux.attentions[-1][0].float().mean(dim=0).cpu()
    return attention[1] - attention[0]


@torch.inference_mode()
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads((RUN_DIR / "run_config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config, RUN_DIR / "dta_dptfilm_best.pt", device)

    scene, height, frame_id = REFERENCE
    reference_dataset = dataset_for(config, scene, height)
    item = reference_dataset.get_visual_sample(select_frame(reference_dataset, frame_id))
    aux = model.forward_with_aux(
        item["clip"].unsqueeze(0).to(device),
        item["metadata"].unsqueeze(0).to(device),
        return_attention=True,
    )

    # The clip is [past, target, target, future]; export only unique temporal views.
    clip_vis = item["clip_vis"]
    save_rgb(OUTPUT_DIR / "01_rgb_past.png", rgb_image(clip_vis[:, 0]))
    save_rgb(OUTPUT_DIR / "02_rgb_target.png", rgb_image(clip_vis[:, 1]))
    save_rgb(OUTPUT_DIR / "03_rgb_future.png", rgb_image(clip_vis[:, 3]))

    # Use one shared magnitude normalization for the two tubelet-flow panels.
    flows = aux.flow_pixels[0].detach().float().cpu()
    flow_images = flow_to_image(flows).permute(0, 2, 3, 1).numpy()
    plt.imsave(OUTPUT_DIR / "04_raft_tubelet_flow_past_target.png", flow_images[0])
    plt.imsave(OUTPUT_DIR / "05_raft_tubelet_flow_target_future.png", flow_images[1])

    validities = aux.alignment_validities[-1][0, 0].detach().float().cpu()
    save_scalar(
        OUTPUT_DIR / "06_validity_mask_past_target.png",
        upsample_nearest(validities[0]),
        "gray",
        0.0,
        1.0,
    )
    save_scalar(
        OUTPUT_DIR / "07_validity_mask_target_future.png",
        upsample_nearest(validities[1]),
        "gray",
        0.0,
        1.0,
    )

    raw = aux.video_grids[-1][0].float().cpu()
    aligned = aux.aligned_grids[-1][0].float().cpu()
    adapted = aux.adapted_maps[-1][0].float().cpu()
    shared_pca = fit_shared_pca([raw[:, 0], raw[:, 1], aligned[:, 0], aligned[:, 1], adapted])
    # Direct V-JEPA 2.1 outputs before RAFT-based spatial alignment.
    save_rgb(OUTPUT_DIR / "08_raw_vjepa21_tubelet_0_pca_before_alignment.png", pca_image(raw[:, 0], shared_pca))
    save_rgb(OUTPUT_DIR / "09_raw_vjepa21_tubelet_1_pca_before_alignment.png", pca_image(raw[:, 1], shared_pca))
    save_rgb(OUTPUT_DIR / "08_aligned_feature_pca_past_target.png", pca_image(aligned[:, 0], shared_pca))
    save_rgb(OUTPUT_DIR / "09_aligned_feature_pca_target_future.png", pca_image(aligned[:, 1], shared_pca))
    save_rgb(OUTPUT_DIR / "10_dta_feature_pca.png", pca_image(adapted, shared_pca))

    # Channel-wise L2 change between the two raw, pre-alignment tubelet features.
    raw_feature_change = torch.linalg.vector_norm(raw[:, 1] - raw[:, 0], dim=0)
    raw_feature_change /= torch.quantile(raw_feature_change, 0.99).clamp_min(1e-6)
    save_scalar_with_embedded_legend(
        OUTPUT_DIR / "10_raw_tubelet_feature_change_before_alignment.png",
        upsample_nearest(raw_feature_change.clamp(0, 1)),
        "viridis",
        0.0,
        1.0,
        [0.0, 1.0],
        ["No change", "Higher change"],
        "Raw tubelet feature change (L2 norm)",
    )

    reference_delta = attention_delta(aux)
    save_scalar_with_embedded_legend(
        OUTPUT_DIR / f"10_dta_preference_{scene}_{height}m_{frame_id}.png",
        upsample_nearest(reference_delta),
        "coolwarm",
        -1.0,
        1.0,
        [-1.0, 0.0, 1.0],
        ["Tubelet 0\npast–target", "equal", "Tubelet 1\ntarget–future"],
        "DTA attention preference",
    )
    depth = aux.depth[0].detach().float().cpu().squeeze().numpy()
    save_scalar_with_embedded_legend(
        OUTPUT_DIR / f"11_predicted_depth_{scene}_{height}m_{frame_id}.png",
        depth,
        "magma",
        float(config["min_depth"]),
        float(config["max_depth"]),
        [0.0, 20.0, 40.0, 60.0, 80.0],
        ["0", "20", "40", "60", "80"],
        "Predicted depth (m)",
    )

    manifest: list[dict[str, object]] = [
        {
            "scene": scene,
            "height_m": height,
            "frame_id": frame_id,
            "kind": "reference_pipeline",
            "attention_mean": float(reference_delta.mean()),
            "attention_min": float(reference_delta.min()),
            "attention_max": float(reference_delta.max()),
        }
    ]
    for variant_index, (variant_scene, variant_height, variant_frame) in enumerate(ATTENTION_VARIANTS[1:], start=12):
        dataset = dataset_for(config, variant_scene, variant_height)
        variant_item = dataset.get_visual_sample(select_frame(dataset, variant_frame))
        variant_aux = model.forward_with_aux(
            variant_item["clip"].unsqueeze(0).to(device),
            variant_item["metadata"].unsqueeze(0).to(device),
            return_attention=True,
        )
        delta = attention_delta(variant_aux)
        save_scalar_with_embedded_legend(
            OUTPUT_DIR
            / f"{variant_index:02d}_dta_preference_{variant_scene}_{variant_height}m_{variant_frame}.png",
            upsample_nearest(delta),
            "coolwarm",
            -1.0,
            1.0,
            [-1.0, 0.0, 1.0],
            ["Tubelet 0\npast–target", "equal", "Tubelet 1\ntarget–future"],
            "DTA attention preference",
        )
        manifest.append(
            {
                "scene": variant_scene,
                "height_m": variant_height,
                "frame_id": variant_frame,
                "kind": "attention_variation",
                "attention_mean": float(delta.mean()),
                "attention_min": float(delta.min()),
                "attention_max": float(delta.max()),
            }
        )

    save_colorbar(
        OUTPUT_DIR / "14_dta_preference_colorbar.png",
        "coolwarm",
        -1.0,
        1.0,
        "attention preference: past–target  ←  0  →  target–future",
    )
    save_colorbar(
        OUTPUT_DIR / "15_depth_colorbar_0_80m.png",
        "magma",
        float(config["min_depth"]),
        float(config["max_depth"]),
        "predicted depth (m)",
    )

    metadata = {
        "checkpoint": str(RUN_DIR / "dta_dptfilm_best.pt"),
        "reference": {"scene": scene, "height_m": height, "frame_id": frame_id},
        "attention_scale": "fixed [-1, 1] for all exported attention maps",
        "depth_scale_m": [float(config["min_depth"]), float(config["max_depth"])],
        "validity_scale": "[0, 1] feature-grid confidence enlarged with nearest-neighbor sampling",
        "flow_scale": "shared torchvision flow_to_image normalization across both tubelet fields",
        "attention_examples": manifest,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
