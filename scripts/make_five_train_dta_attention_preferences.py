#!/usr/bin/env python3
"""Reproduce aligned-DTA attention preference for five OccuFly training samples."""

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
OUTPUT = PAPER_ROOT / "content/images/dta_attention_preference_train_five.png"
CSV_OUTPUT = PAPER_ROOT / "content/images/dta_attention_preference_train_five.csv"


def select_five(dataset: OccuFlyVideoTargetDataset, train_scenes: list[str]) -> list[int]:
    """Choose a deterministic mid-sequence sample from each training scene."""
    selected: list[int] = []
    requested_heights = [30, 40, 50, 30, 40]
    for scene, height in zip(train_scenes[:5], requested_heights, strict=True):
        candidates = [
            index
            for index, sample in enumerate(dataset.samples)
            if sample.target.scene == scene and sample.target.height == height
        ]
        if not candidates:
            raise RuntimeError(f"No training samples for {scene} at {height} m")
        selected.append(candidates[len(candidates) // 2])
    return selected


def upsample_nearest(array: torch.Tensor, size: int = 384) -> np.ndarray:
    return (
        F.interpolate(array[None, None].float(), size=(size, size), mode="nearest")[0, 0]
        .detach()
        .cpu()
        .numpy()
    )


@torch.inference_mode()
def main() -> None:
    config = json.loads((RUN_DIR / "run_config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OccuFlyVideoTargetDataset(
        dataset_root=Path(config["dataset_root"]),
        scenes=list(config["train_scenes"]),
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
    indices = select_five(dataset, list(config["train_scenes"]))
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
        h = delta.shape[0]
        upper = delta[: h // 2]
        lower = delta[h // 2 :]
        records.append(
            {
                "uid": str(item["uid"]),
                "scene": str(item["scene"]),
                "height": int(item["height"].item()),
                "frame": str(item["target_frame_id"]),
                "target": rgb_image(item["clip_vis"][:, 1]),
                "display": upsample_nearest(display),
                "mean_delta": float(delta.mean()),
                "upper_mean": float(upper.mean()),
                "lower_mean": float(lower.mean()),
                "red_fraction": float((delta > 0).float().mean()),
                "blue_fraction": float((delta < 0).float().mean()),
                "valid_fraction": float((validity >= float(config["alignment_validity_threshold"])).float().mean()),
            }
        )

    fig, axes = plt.subplots(2, 5, figsize=(14.2, 6.15), facecolor="white")
    fig.suptitle(
        "DTA attention preference on five OccuFly training samples",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    for column, record in enumerate(records):
        axes[0, column].imshow(record["target"])
        axes[0, column].set_title(
            f"{record['scene'].replace('_', ' ')} · {record['height']} m · {record['frame']}",
            fontsize=10,
            fontweight="bold",
        )
        axes[0, column].axis("off")
        axes[1, column].imshow(record["display"], cmap="coolwarm", vmin=-1, vmax=1)
        axes[1, column].set_title(
            f"upper {record['upper_mean']:+.3f} · lower {record['lower_mean']:+.3f}",
            fontsize=9.5,
        )
        axes[1, column].axis("off")
    fig.text(
        0.5,
        0.018,
        "Blue favors past–target · red favors target–future · each map uses its own 99th-percentile display scale",
        ha="center",
        fontsize=10.5,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.01, 0.05, 0.99, 0.95), h_pad=1.15, w_pad=0.5)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, facecolor="white")
    plt.close(fig)

    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "uid",
            "scene",
            "height",
            "frame",
            "mean_delta",
            "upper_mean",
            "lower_mean",
            "red_fraction",
            "blue_fraction",
            "valid_fraction",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})
    print(OUTPUT)
    print(CSV_OUTPUT)


if __name__ == "__main__":
    main()
