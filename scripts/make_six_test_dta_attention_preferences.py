#!/usr/bin/env python3
"""Render aligned-DTA preferences for every OccuFly test scene/height stratum."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


PAPER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_ROOT.parent
SCRIPT_ROOT = PAPER_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from make_vjepa_raft_figure3_extension import (  # noqa: E402
    OccuFlyVideoTargetDataset,
    make_model,
    rgb_image,
)


RUN_DIR = (
    REPO_ROOT
    / "experiments_wacv27/outputs/mde_video/vjepa21video_spatial_alignment_grid_sample"
    / "real_dta__clip4__seed0__lr3e_4__wd1e_4"
)
OUTPUT = PAPER_ROOT / "content/images/dta_attention_preference_test_six.png"
CSV_OUTPUT = PAPER_ROOT / "content/images/dta_attention_preference_test_six.csv"


def select_six(dataset: OccuFlyVideoTargetDataset, scenes: list[str], heights: list[int]) -> list[int]:
    """Choose the deterministic middle sample from each scene/height stratum."""
    selected: list[int] = []
    for scene in scenes:
        for height in heights:
            candidates = [
                index
                for index, sample in enumerate(dataset.samples)
                if sample.target.scene == scene and sample.target.height == height
            ]
            if not candidates:
                raise RuntimeError(f"No test samples for {scene} at {height} m")
            selected.append(candidates[len(candidates) // 2])
    return selected


def upsample_nearest(array: torch.Tensor, size: int = 384) -> np.ndarray:
    return (
        F.interpolate(array[None, None].float(), size=(size, size), mode="nearest")[0, 0]
        .detach()
        .cpu()
        .numpy()
    )


def depth_metrics(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    prediction = prediction.float().squeeze().cpu()
    target = target.float().squeeze().cpu()
    valid = valid.bool().squeeze().cpu() & torch.isfinite(prediction) & torch.isfinite(target) & (target > 0)
    pred = prediction[valid].clamp_min(1e-6)
    truth = target[valid].clamp_min(1e-6)
    error = pred - truth
    log_error = torch.log(pred) - torch.log(truth)
    ratio = torch.maximum(pred / truth, truth / pred)
    return {
        "rmse": float(torch.sqrt(torch.mean(error.square()))),
        "mae": float(torch.mean(error.abs())),
        "abs_rel": float(torch.mean(error.abs() / truth)),
        "silog": float(100.0 * torch.sqrt(torch.clamp(torch.mean(log_error.square()) - torch.mean(log_error).square(), min=0.0))),
        "delta1": float(torch.mean((ratio < 1.25).float())),
        "valid_pixels": int(valid.sum()),
    }


@torch.inference_mode()
def main() -> None:
    config = json.loads((RUN_DIR / "run_config.json").read_text(encoding="utf-8"))
    aggregate = json.loads((RUN_DIR / "metrics.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OccuFlyVideoTargetDataset(
        dataset_root=Path(config["dataset_root"]),
        scenes=list(config["test_scenes"]),
        heights=list(config["heights"]),
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
    indices = select_six(dataset, list(config["test_scenes"]), list(config["heights"]))
    model = make_model(config, RUN_DIR / "dta_dptfilm_best.pt", device)

    records: list[dict[str, object]] = []
    for index in indices:
        item = dataset.get_visual_sample(index)
        clip = item["clip"].unsqueeze(0).to(device)
        metadata = item["metadata"].unsqueeze(0).to(device)
        aux = model.forward_with_aux(clip, metadata, return_attention=True)
        attention = aux.attentions[-1][0].float().mean(dim=0).cpu()
        delta = attention[1] - attention[0]
        display_scale = torch.quantile(delta.abs(), 0.99).clamp_min(1e-6)
        display = (delta / display_scale).clamp(-1, 1)
        validity = aux.alignment_validities[-1][0, 0].detach().float().cpu()
        metrics = depth_metrics(aux.depth[0], item["depth"], item["valid"])
        records.append(
            {
                "uid": str(item["uid"]),
                "scene": str(item["scene"]),
                "height": int(item["height"].item()),
                "frame": str(item["target_frame_id"]),
                "target": rgb_image(item["clip_vis"][:, 1]),
                "attention": upsample_nearest(display),
                "depth": aux.depth[0].detach().float().cpu().squeeze().numpy(),
                "mean_preference": float(delta.mean()),
                "red_fraction": float((delta > 0).float().mean()),
                "blue_fraction": float((delta < 0).float().mean()),
                "valid_fraction": float((validity >= float(config["alignment_validity_threshold"])).float().mean()),
                **metrics,
            }
        )

    fig, axes = plt.subplots(3, 6, figsize=(15.8, 8.55), facecolor="white")
    fig.suptitle(
        "RAFT-aligned DTA attention on all OccuFly test scene–height strata",
        fontsize=17,
        fontweight="bold",
        y=0.988,
    )
    fig.text(
        0.5,
        0.948,
        (
            f"Full test: δ₁ {aggregate['test_delta1']:.3f} · AbsRel {aggregate['test_abs_rel']:.3f} · "
            f"RMSE {aggregate['test_rmse']:.3f} m · MAE {aggregate['test_mae']:.3f} m · "
            f"SILog {aggregate['test_silog']:.3f}"
        ),
        ha="center",
        fontsize=10.5,
        color="#374151",
    )
    for column, record in enumerate(records):
        title = f"{record['scene'].replace('_', ' ')} · {record['height']} m · {record['frame']}"
        axes[0, column].imshow(record["target"])
        axes[0, column].set_title(title, fontsize=9.7, fontweight="bold", pad=5)
        axes[0, column].axis("off")

        axes[1, column].imshow(record["attention"], cmap="coolwarm", vmin=-1, vmax=1)
        axes[1, column].set_title(
            f"DTA preference · mean {record['mean_preference']:+.3f}",
            fontsize=8.8,
            pad=4,
        )
        axes[1, column].axis("off")

        axes[2, column].imshow(
            record["depth"],
            cmap="magma",
            vmin=float(config["min_depth"]),
            vmax=float(config["max_depth"]),
        )
        axes[2, column].set_title(
            f"RMSE {record['rmse']:.2f} m · AbsRel {record['abs_rel']:.3f}\n"
            f"δ₁ {record['delta1']:.3f} · SILog {record['silog']:.2f}",
            fontsize=8.5,
            pad=4,
        )
        axes[2, column].axis("off")

    fig.text(
        0.5,
        0.017,
        "Blue favors past–target · red favors target–future · deterministic middle frame per stratum · each attention map has its own display scale",
        ha="center",
        fontsize=9.8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.008, 0.04, 0.992, 0.925), h_pad=1.1, w_pad=0.45)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, facecolor="white")
    plt.close(fig)

    fieldnames = [
        "uid",
        "scene",
        "height",
        "frame",
        "rmse",
        "mae",
        "abs_rel",
        "silog",
        "delta1",
        "valid_pixels",
        "mean_preference",
        "red_fraction",
        "blue_fraction",
        "valid_fraction",
    ]
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})
    print(OUTPUT)
    print(CSV_OUTPUT)


if __name__ == "__main__":
    main()
