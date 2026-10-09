#!/usr/bin/env python3
"""Render a visual-only RAFT alignment preview for the Figure 3 extension."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
from torchvision.utils import flow_to_image


def parse_args() -> argparse.Namespace:
    root = Path("/datatank/giridhar.vb/repos/datasets/OccuFly_Dataset")
    paper = Path("/datatank/giridhar.vb/repos/ws_jepa_occufly_test/confxxx_checkbranch_ieeercc2027")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=root / "scene_08/50/images/visual")
    parser.add_argument("--target-frame", default="000091")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument(
        "--output",
        type=Path,
        default=paper / "content/images/raft_alignment_occufly_scene08_50m_000091_preview.png",
    )
    return parser.parse_args()


def load_rgb(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as image_file:
        image = image_file.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def sampling_grid(flow: torch.Tensor) -> torch.Tensor:
    """Return x + flow(x), normalized for align_corners=False."""
    n, _two, h, w = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    x = (2.0 * (xx + flow[:, 0] + 0.5) / w) - 1.0
    y = (2.0 * (yy + flow[:, 1] + 0.5) / h) - 1.0
    return torch.stack((x, y), dim=-1)


def warp(source: torch.Tensor, target_to_source_flow: torch.Tensor) -> torch.Tensor:
    return F.grid_sample(
        source,
        sampling_grid(target_to_source_flow),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def consistency_mask(forward: torch.Tensor, reverse: torch.Tensor) -> torch.Tensor:
    grid = sampling_grid(forward)
    reverse_at_match = F.grid_sample(reverse, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    residual = torch.linalg.vector_norm(forward + reverse_at_match, dim=1)
    scale = torch.linalg.vector_norm(forward, dim=1) + torch.linalg.vector_norm(reverse_at_match, dim=1)
    consistent = residual <= (1.5 + 0.01 * scale)
    in_bounds = (
        (grid[..., 0] >= -1.0)
        & (grid[..., 0] <= 1.0)
        & (grid[..., 1] >= -1.0)
        & (grid[..., 1] <= 1.0)
    )
    return consistent & in_bounds


def rgb_image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def flow_image(flow: torch.Tensor) -> np.ndarray:
    image = flow_to_image(flow.detach().cpu())[0]
    return image.permute(1, 2, 0).numpy()


def residual_image(target: torch.Tensor, warped: torch.Tensor, valid: torch.Tensor) -> np.ndarray:
    residual = (target - warped).abs().mean(dim=1)[0]
    values = residual[valid[0]]
    high = torch.quantile(values, 0.99).clamp_min(1e-6) if values.numel() else residual.new_tensor(1.0)
    return (residual / high).clamp(0, 1).detach().cpu().numpy()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    target_id = int(args.target_frame)
    paths = {
        "previous": args.image_dir / f"{target_id - 1:06d}.png",
        "target": args.image_dir / f"{target_id:06d}.png",
        "next": args.image_dir / f"{target_id + 1:06d}.png",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing OccuFly frames: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = {name: load_rgb(path, args.size).unsqueeze(0).to(device) for name, path in paths.items()}
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights, progress=False).to(device).eval()

    def estimate(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first_n, second_n = weights.transforms()(first, second)
        return model(first_n, second_n)[-1].float()

    flow_t_prev = estimate(frames["target"], frames["previous"])
    flow_prev_t = estimate(frames["previous"], frames["target"])
    flow_t_next = estimate(frames["target"], frames["next"])
    flow_next_t = estimate(frames["next"], frames["target"])

    valid_prev = consistency_mask(flow_t_prev, flow_prev_t)
    valid_next = consistency_mask(flow_t_next, flow_next_t)
    warped_prev = warp(frames["previous"], flow_t_prev)
    warped_next = warp(frames["next"], flow_t_next)

    fig, axes = plt.subplots(2, 6, figsize=(18.2, 6.7), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.012, right=0.994, bottom=0.035, top=0.86, wspace=0.055, hspace=0.28)
    fig.suptitle(
        "RAFT target-coordinate alignment on the Figure 3 OccuFly clip",
        fontsize=17,
        fontweight="semibold",
        y=0.965,
    )

    top = [
        (rgb_image(frames["previous"][0]), r"$I_{t-1}$"),
        (rgb_image(frames["target"][0]), r"target $I_t$"),
        (rgb_image(frames["next"][0]), r"$I_{t+1}$"),
        (flow_image(flow_t_prev), r"flow $t\rightarrow t{-}1$"),
        (flow_image(flow_t_next), r"flow $t\rightarrow t{+}1$"),
        (
            np.concatenate(
                [valid_prev[0].detach().cpu().numpy(), valid_next[0].detach().cpu().numpy()], axis=1
            ),
            "valid correspondences",
        ),
    ]
    bottom = [
        (rgb_image(warped_prev[0]), r"warp $I_{t-1}\rightarrow t$"),
        (residual_image(frames["target"], warped_prev, valid_prev), "alignment residual"),
        (valid_prev[0].detach().cpu().numpy(), "validity mask"),
        (valid_next[0].detach().cpu().numpy(), "validity mask"),
        (residual_image(frames["target"], warped_next, valid_next), "alignment residual"),
        (rgb_image(warped_next[0]), r"warp $I_{t+1}\rightarrow t$"),
    ]

    for ax, (image, title) in zip(axes[0], top):
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=1 if image.ndim == 2 else None)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
        ax.axis("off")
    for index, (ax, (image, title)) in enumerate(zip(axes[1], bottom)):
        if "residual" in title:
            ax.imshow(image, cmap="magma", vmin=0, vmax=1)
        else:
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=1 if image.ndim == 2 else None)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
        ax.axis("off")

    fig.text(
        0.5,
        0.012,
        "RGB warps visualize the correspondence field; the implementation applies the same backward-sampling grid to V-JEPA tubelet features.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#4b5563",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
