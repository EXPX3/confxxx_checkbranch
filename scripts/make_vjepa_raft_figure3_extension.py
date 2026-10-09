#!/usr/bin/env python3
"""Create a visual-first extension of the earlier V-JEPA video-token Figure 3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import flow_to_image


REPO_ROOT = Path("/datatank/giridhar.vb/repos/ws_jepa_occufly_test")
ALIGN_ROOT = REPO_ROOT / "experiments_wacv27/mde_video/vjepa21video_spatial_alignment_grid_sample"
DTA_ROOT = REPO_ROOT / "experiments_wacv27/mde_video/vjepa21video_scripts/vjepa21video_dptfilm_attention"
for path in (ALIGN_ROOT / "src", DTA_ROOT / "src", DTA_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vjepa21video_dptfilm_attention.occufly_video_dataset import OccuFlyVideoTargetDataset
from vjepa21video_dptfilm_attention.paths import add_repo_paths
from vjepa21video_spatial_alignment.aligned_dta import SpatialAlignedVJepaVideoDPTFiLMDepthModel
from vjepa21video_spatial_alignment.alignment import FlowConfig


DEFAULT_RUN = (
    REPO_ROOT
    / "experiments_wacv27/outputs/mde_video/vjepa21video_spatial_alignment_grid_sample"
    / "real_dta__clip4__seed0__lr3e_4__wd1e_4"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "confxxx_checkbranch_ieeercc2027/content/images"
    / "vjepa_video_raft_figure3_extension_preview.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--scene", default="scene_08")
    parser.add_argument("--height", type=int, default=50)
    parser.add_argument("--frame-id", default="000091")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def select_sample(dataset: OccuFlyVideoTargetDataset, frame_id: str) -> int:
    for index, sample in enumerate(dataset.samples):
        if sample.target.frame_id == frame_id:
            return index
    raise RuntimeError(f"Could not find target frame {frame_id} in {len(dataset)} samples")


def make_model(config: dict, checkpoint: Path, device: torch.device):
    add_repo_paths(Path(config["repo_root"]), None, None)
    flow_config = FlowConfig(
        model=config["flow_model"],
        checkpoint=Path(config["flow_checkpoint"]) if config.get("flow_checkpoint") else None,
        pretrained=bool(config["flow_pretrained"]),
        progress=False,
        tubelet_size=int(config["tubelet_size"]),
        consistency_threshold=float(config["flow_consistency_threshold"]),
        duplicate_tolerance=float(config["flow_duplicate_tolerance"]),
        validity_threshold=float(config["alignment_validity_threshold"]),
    )
    model = SpatialAlignedVJepaVideoDPTFiLMDepthModel(
        repo_root=Path(config["repo_root"]),
        checkpoint_path=Path(config["vjepa_checkpoint"]),
        selected_layers=list(config["selected_layers"]),
        vjepa_size=config["vjepa_size"],
        freeze_backbone=True,
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
        bins_strategy=config["bins_strategy"],
        norm_strategy=config["norm_strategy"],
        n_output_channels=int(config["n_output_channels"]),
        dpt_feature_dim=int(config["dpt_feature_dim"]),
        dpt_hidden_channels=int(config["dpt_hidden_channels"]),
        dpt_post_process_channels=list(config["dpt_post_process_channels"]),
        metadata_dim=int(config["metadata_dim"]),
        temporal_num_heads=int(config["temporal_num_heads"]),
        temporal_adapter_dim=int(config["temporal_adapter_dim"]),
        temporal_mlp_ratio=float(config["temporal_mlp_ratio"]),
        temporal_dropout=float(config["temporal_dropout"]),
        temporal_init_scale=float(config["temporal_init_scale"]),
        use_batchnorm=bool(config["use_batchnorm"]),
        flow_config=flow_config,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    return model.to(device).eval()


def fit_shared_pca(maps: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.cat([item.float().permute(1, 2, 0).reshape(-1, item.shape[0]) for item in maps], dim=0)
    mean = tokens.mean(dim=0, keepdim=True)
    centered = tokens - mean
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:3].T.contiguous()
    projection = centered @ basis
    low = projection.quantile(0.01, dim=0, keepdim=True)
    high = projection.quantile(0.99, dim=0, keepdim=True)
    return mean, basis, (low, high)


def pca_image(feature: torch.Tensor, pca, size: int = 384) -> np.ndarray:
    mean, basis, limits = pca
    c, h, w = feature.shape
    tokens = feature.float().permute(1, 2, 0).reshape(-1, c)
    projected = (tokens - mean) @ basis
    low, high = limits
    rgb = ((projected - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    rgb = rgb.reshape(h, w, 3).permute(2, 0, 1).unsqueeze(0)
    rgb = F.interpolate(rgb, size=(size, size), mode="nearest")[0]
    return rgb.permute(1, 2, 0).cpu().numpy()


def rgb_image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def pair_image(first: np.ndarray, second: np.ndarray, gap: int = 12) -> np.ndarray:
    h = max(first.shape[0], second.shape[0])
    fill = 255 if np.issubdtype(first.dtype, np.integer) else 1.0
    if first.ndim == 2:
        canvas = np.full((h, first.shape[1] + gap + second.shape[1]), fill, dtype=first.dtype)
    else:
        canvas = np.full((h, first.shape[1] + gap + second.shape[1], first.shape[2]), fill, dtype=first.dtype)
    canvas[: first.shape[0], : first.shape[1]] = first
    canvas[: second.shape[0], first.shape[1] + gap :] = second
    return canvas


def clip_montage(clip_vis: torch.Tensor, gap: int = 8) -> np.ndarray:
    frames = [rgb_image(clip_vis[:, index]) for index in range(clip_vis.shape[1])]
    return np.concatenate(
        [piece for index, frame in enumerate(frames) for piece in ([frame] if index == len(frames) - 1 else [frame, np.ones((frame.shape[0], gap, 3))])],
        axis=1,
    )


def scalar_image(array: torch.Tensor, size: int = 384) -> np.ndarray:
    value = array.float().unsqueeze(0).unsqueeze(0)
    value = F.interpolate(value, size=(size, size), mode="nearest")[0, 0]
    return value.detach().cpu().numpy()


def decorate_axis(ax, label: str, panel: str | None = None) -> None:
    ax.set_title(label, fontsize=11.5, fontweight="bold", pad=6)
    ax.axis("off")
    if panel:
        ax.text(
            0.018,
            0.97,
            panel,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=13,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#cbd5e1", alpha=0.95),
        )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))
    checkpoint = args.checkpoint or (args.run_dir / "dta_dptfilm_best.pt")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = OccuFlyVideoTargetDataset(
        dataset_root=Path(config["dataset_root"]),
        scenes=[args.scene],
        heights=[args.height],
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
    item = dataset.get_visual_sample(select_sample(dataset, args.frame_id))
    clip = item["clip"].unsqueeze(0).to(device)
    metadata = item["metadata"].unsqueeze(0).to(device)
    model = make_model(config, checkpoint, device)
    aux = model.forward_with_aux(clip, metadata, return_attention=True)

    raw = aux.video_grids[-1][0].float().cpu()       # [C,2,H,W]
    aligned = aux.aligned_grids[-1][0].float().cpu() # [C,2,H,W]
    adapted = aux.adapted_maps[-1][0].float().cpu()  # [C,H,W]
    shared_pca = fit_shared_pca([raw[:, 0], raw[:, 1], aligned[:, 0], aligned[:, 1], adapted])
    raw_pca = [pca_image(raw[:, index], shared_pca) for index in range(2)]
    aligned_pca = [pca_image(aligned[:, index], shared_pca) for index in range(2)]
    adapted_pca = pca_image(adapted, shared_pca)

    raw_delta = torch.linalg.vector_norm(raw[:, 1] - raw[:, 0], dim=0)
    raw_delta /= torch.quantile(raw_delta, 0.99).clamp_min(1e-6)
    raw_delta_image = scalar_image(raw_delta.clamp(0, 1))

    flows = aux.flow_pixels[0].detach().cpu()
    flow_pair = pair_image(
        flow_to_image(flows[0].unsqueeze(0))[0].permute(1, 2, 0).numpy(),
        flow_to_image(flows[1].unsqueeze(0))[0].permute(1, 2, 0).numpy(),
    )
    valid = aux.alignment_validities[-1][0, 0].detach().cpu()
    validity_pair = pair_image(scalar_image(valid[0]), scalar_image(valid[1]))

    attention = aux.attentions[-1][0].float().mean(dim=0).cpu()  # [2,H,W]
    attention_delta = attention[1] - attention[0]
    attention_scale = torch.quantile(attention_delta.abs(), 0.99).clamp_min(1e-6)
    attention_image = scalar_image((attention_delta / attention_scale).clamp(-1, 1))

    depth = aux.depth[0].detach().float().cpu().squeeze()
    depth_image = depth.numpy()

    fig = plt.figure(figsize=(18.2, 7.7), facecolor="white")
    grid = fig.add_gridspec(2, 6, left=0.015, right=0.995, bottom=0.055, top=0.88, wspace=0.08, hspace=0.34)
    fig.suptitle(
        "V-JEPA 2.1 tubelets and RAFT-aligned dense temporal attention",
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )
    fig.text(0.5, 0.915, "OccuFly scene 08 · 50 m · target frame 000091", ha="center", fontsize=11.5, color="#4b5563")

    ax = fig.add_subplot(grid[0, 0:2])
    ax.imshow(clip_montage(item["clip_vis"]))
    decorate_axis(ax, r"four-frame clip  $[-1,0,0,+1]$", "a")
    for column, (image, label) in enumerate(
        [(raw_pca[0], "tubelet 0 PCA"), (raw_pca[1], "tubelet 1 PCA")], start=2
    ):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(image)
        decorate_axis(ax, label)
    ax = fig.add_subplot(grid[0, 4])
    ax.imshow(raw_delta_image, cmap="viridis", vmin=0, vmax=1)
    decorate_axis(ax, "tubelet feature change")
    ax = fig.add_subplot(grid[0, 5])
    ax.imshow(adapted_pca)
    decorate_axis(ax, "DTA feature PCA")

    ax = fig.add_subplot(grid[1, 0])
    ax.imshow(flow_pair)
    decorate_axis(ax, "RAFT tubelet flow", "b")
    ax = fig.add_subplot(grid[1, 1])
    ax.imshow(validity_pair, cmap="gray", vmin=0, vmax=1)
    decorate_axis(ax, "validity masks")
    ax = fig.add_subplot(grid[1, 2])
    ax.imshow(pair_image(raw_pca[0], raw_pca[1]))
    decorate_axis(ax, "before alignment")
    ax = fig.add_subplot(grid[1, 3])
    ax.imshow(pair_image(aligned_pca[0], aligned_pca[1]))
    decorate_axis(ax, "after alignment")
    ax = fig.add_subplot(grid[1, 4])
    ax.imshow(attention_image, cmap="coolwarm", vmin=-1, vmax=1)
    decorate_axis(ax, "DTA attention preference")
    ax = fig.add_subplot(grid[1, 5])
    ax.imshow(depth_image, cmap="magma", vmin=float(config["min_depth"]), vmax=float(config["max_depth"]))
    decorate_axis(ax, "predicted depth")

    fig.text(
        0.5,
        0.018,
        "Shared PCA basis for all token panels · blue/red attention favors tubelet 0/1 · white validity is retained",
        ha="center",
        fontsize=10.5,
        color="#4b5563",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
