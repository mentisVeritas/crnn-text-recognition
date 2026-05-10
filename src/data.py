from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from src.text_codec import encode_text

logger = logging.getLogger(__name__)


def resize_keep_ratio(img: Image.Image, img_height: int) -> Image.Image:
    w, h = img.size
    scale = img_height / h
    new_w = max(1, int(w * scale))

    img = img.resize((new_w, img_height), Image.BICUBIC)
    return img


# ---------- IMAGE TRANSFORM ----------
def build_image_transform(img_height: int = 32) -> T.Compose:
    """Resize RGB OCR images while preserving aspect ratio."""

    from functools import partial

    transform_fn = partial(resize_keep_ratio, img_height=img_height)

    return T.Compose(
        [
            T.Lambda(transform_fn),
            T.ToTensor(),
            T.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )


# ---------- DATASET ----------
class OCRDataset(Dataset):
    """CSV + image folder dataset for CTC training."""

    def __init__(
        self,
        images_dir: str | os.PathLike[str],
        labels_path: str | os.PathLike[str],
        alphabet: str,
        img_height: int = 32,
    ):
        self.images_dir = str(images_dir)

        df = pd.read_csv(
            labels_path,
            sep="\t",
            header=None,
            names=["filename", "text"],
            engine="python",
            quoting=3,
            on_bad_lines="skip"
        )

        # фильтр мусора
        df = df.dropna(subset=["filename", "text"])
        df = df[df["text"].notna()]

        logger.info("Dataset loaded with %s samples", len(df))

        self.filenames = df["filename"].values
        self.texts = df["text"].values

        # CTC: 0 — blank
        self.char2idx = {ch: i + 1 for i, ch in enumerate(alphabet)}

        self.transform = build_image_transform(img_height)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_name = self.filenames[idx]
        text = self.texts[idx]

        img_path = Path(self.images_dir) / str(img_name)

        # безопасное открытие
        with Image.open(img_path) as img:
            # keep RGB pipeline consistent between train and inference
            image = img.convert("RGB")

        image = self.transform(image)
        encoded = encode_text(text, self.char2idx)

        return {
            "image": image,
            "label": torch.tensor(encoded, dtype=torch.long),
            "length": torch.tensor(len(encoded), dtype=torch.long),
            "text": text,
        }


# ---------- CTC COLLATE ----------
def collate_fn_ctc(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    texts = [item["text"] for item in batch]

    # dynamic width padding
    max_w = max(img.shape[2] for img in images)

    padded_images = []
    for img in images:
        c, h, w = img.shape

        if w < max_w:
            # white padding after normalization
            pad = torch.ones((c, h, max_w - w), dtype=img.dtype)
            img = torch.cat([img, pad], dim=2)

        padded_images.append(img)

    images = torch.stack(padded_images)

    return {
        "image": images,
        "label": labels,
        "length": lengths,
        "text": texts,
    }


# ---------- DATALOADERS ----------
def build_dataloaders(
    config: dict,
    images_dir: str | os.PathLike[str],
    labels_path: str | os.PathLike[str],
    split_seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    dataset = OCRDataset(
        images_dir=images_dir,
        labels_path=labels_path,
        alphabet=config["alphabet"],
        img_height=config["img_height"],
    )

    val_split = float(config.get("val_split", 0.05))

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(split_seed)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        pin_memory=False,
        collate_fn=collate_fn_ctc,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=False,
        collate_fn=collate_fn_ctc,
    )

    return train_loader, val_loader