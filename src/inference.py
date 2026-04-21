from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torchvision.transforms as T
from PIL import Image

from src.decode import greedy_decode


def preprocess_image(image_path: str | Path, img_height: int, img_width: int) -> torch.Tensor:
    transform = T.Compose(
        [
            T.Grayscale(num_output_channels=1),
            T.Resize((img_height, img_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def decode_with_confidence(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    alphabet: str,
    device: torch.device,
) -> Tuple[str, float]:
    """Return decoded text + approximate confidence in [0, 1]."""
    with torch.no_grad():
        logits = model(image_tensor.to(device))  # [B, T, C]
        logits = logits.permute(1, 0, 2)  # [T, B, C]
        probs = torch.softmax(logits, dim=2)
        log_probs = torch.log(probs.clamp_min(1e-8))
        pred_text = greedy_decode(log_probs, alphabet)[0]

        # Mean probability of kept CTC tokens (non-blank and deduplicated).
        top_probs, top_idx = probs[:, 0, :].max(dim=1)
        kept_conf = []
        prev = -1
        for t in range(top_idx.size(0)):
            idx = int(top_idx[t].item())
            if idx != 0 and idx != prev:
                kept_conf.append(float(top_probs[t].item()))
            prev = idx

        confidence = float(sum(kept_conf) / len(kept_conf)) if kept_conf else float(top_probs.mean().item())

    return pred_text, confidence
