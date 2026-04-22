import logging
import os

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from src.text_codec import encode_text

logger = logging.getLogger(__name__)


def build_image_transform(img_height=32, img_width=128):
    return T.Compose(
        [
            T.Grayscale(num_output_channels=1),
            T.Resize((img_height, img_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


class OCRDataset(Dataset):
    def __init__(
        self,
        images_dir,
        labels_path,
        alphabet,
        img_height=32,
        img_width=128,
    ):
        self.images_dir = images_dir
        df = pd.read_csv(labels_path)

        # Filter out NaN values
        df = df.dropna(subset=["image_name", "utf8string"])
        df = df[df["utf8string"].notna()]  # double check

        logger.info("Dataset loaded with %s samples (NaN filtered)", len(df))

        self.filenames = df["image_name"].values
        self.texts = df["utf8string"].values
        self.char2idx = {c: i + 1 for i, c in enumerate(alphabet)}  # 0 — blank (CTC)
        self.transform = build_image_transform(img_height, img_width)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        text = self.texts[idx]

        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        encoded = encode_text(text, self.char2idx)

        return {
            "image": image,
            "label": torch.tensor(encoded, dtype=torch.long),
            "length": torch.tensor(len(encoded), dtype=torch.long),
            "text": text,  # удобно для дебага
        }