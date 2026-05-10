from __future__ import annotations

import random
from typing import Sequence

import matplotlib.pyplot as plt
import torch


def sample_indices(total_size: int, sample_count: int, fixed: bool, seed: int) -> list[int]:
    sample_count = min(sample_count, total_size)
    if fixed:
        rng = random.Random(seed)
        return rng.sample(range(total_size), sample_count)
    return random.sample(range(total_size), sample_count)


def plot_prediction_grid(
    items: Sequence[dict],
    cols: int = 3,
    row_height: float = 4.0,
    title_fontsize_pred: int = 14,
    title_fontsize_true: int = 16,
) -> None:
    """Render prediction cards in a grid for quick qualitative checks."""
    count = len(items)
    if count == 0:
        print("No items to display.")
        return

    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(20, row_height * rows),
        constrained_layout=True,
    )
    axes = axes.flatten() if count > 1 else [axes]

    for i, item in enumerate(items):
        image = item["image"]
        pred_text = item["pred"]
        true_text = item["true"]
        conf = float(item.get("confidence", 0.0))
        correct = bool(item.get("correct", pred_text == true_text))
        pred_color = "green" if correct else "red"

        if isinstance(image, torch.Tensor):
            # RGB tensor [3, H, W]
            if image.ndim == 3 and image.size(0) == 3:
                img_np = image.permute(1, 2, 0).cpu().numpy()
                # undo normalization [-1,1] -> [0,1]
                img_np = (img_np * 0.5 + 0.5).clip(0, 1)
            # grayscale tensor [1, H, W]
            else:
                img_np = image.squeeze(0).cpu().numpy()
        else:
            img_np = image

        if getattr(img_np, "ndim", 0) == 2:
            axes[i].imshow(
                img_np,
                cmap="gray",
                interpolation="nearest",
                aspect="auto",
            )
        else:
            axes[i].imshow(
                img_np,
                interpolation="nearest",
                aspect="auto",
            )
        axes[i].axis("off")
        axes[i].text(
            0.5,
            1.04,
            f"{pred_text} ({conf * 100:.1f}%)",
            transform=axes[i].transAxes,
            ha="center",
            va="bottom",
            fontsize=title_fontsize_pred,
            fontweight="bold",
            color=pred_color,
        )
        axes[i].text(
            0.5,
            -0.08,
            true_text,
            transform=axes[i].transAxes,
            ha="center",
            va="top",
            fontsize=title_fontsize_true,
            fontweight="bold",
            color="green",
        )

        extra = item.get("extra_line")
        filename = item.get("filename")
        y_offset = -0.20

        if extra:
            axes[i].text(
                0.5,
                y_offset,
                extra,
                transform=axes[i].transAxes,
                ha="center",
                va="top",
                fontsize=12,
                color="orange",
            )
            y_offset -= 0.10

        if filename:
            axes[i].text(
                0.5,
                y_offset,
                filename,
                transform=axes[i].transAxes,
                ha="center",
                va="top",
                fontsize=10,
                color="gray",
            )

    for j in range(count, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(h_pad=0.8)
    plt.show()
