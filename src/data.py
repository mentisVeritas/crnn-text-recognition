from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from src.text_codec import encode_text

logger = logging.getLogger(__name__)


def build_image_transform(img_height: int = 32, img_width: int = 128) -> T.Compose:
    """Shared preprocessing for training and inference images."""
    return T.Compose(
        [
            T.Grayscale(num_output_channels=1),
            T.Resize((img_height, img_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


class OCRDataset(Dataset):
    """CSV + image folder dataset for CTC training."""

    def __init__(
        self,
        images_dir: str | os.PathLike[str],
        labels_path: str | os.PathLike[str],
        alphabet: str,
        img_height: int = 32,
        img_width: int = 128,
    ):
        self.images_dir = str(images_dir)
        df = pd.read_csv(labels_path)

        # Keep only fully valid rows to avoid runtime errors in __getitem__.
        df = df.dropna(subset=["image_name", "utf8string"])
        df = df[df["utf8string"].notna()]
        logger.info("Dataset loaded with %s samples (NaN filtered)", len(df))

        self.filenames = df["image_name"].values
        self.texts = df["utf8string"].values
        self.char2idx = {ch: i + 1 for i, ch in enumerate(alphabet)}  # 0 is CTC blank.
        self.transform = build_image_transform(img_height, img_width)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_name = self.filenames[idx]
        text = self.texts[idx]

        img_path = Path(self.images_dir) / str(img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        encoded = encode_text(text, self.char2idx)
        return {
            "image": image,
            "label": torch.tensor(encoded, dtype=torch.long),
            "length": torch.tensor(len(encoded), dtype=torch.long),
            "text": text,  # Useful for evaluation and debugging.
        }