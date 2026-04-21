import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import logging

logger = logging.getLogger(__name__)


class OCRDataset(Dataset):
    def __init__(
        self,
        images_dir,
        labels_path,
        alphabet,
        img_height=32,
        img_width=128
    ):
        self.images_dir = images_dir
        df = pd.read_csv(labels_path)

        # Filter out NaN values
        df = df.dropna(subset=["image_name", "utf8string"])
        df = df[df["utf8string"].notna()]  # double check

        logging.info(f"Dataset loaded with {len(df)} samples (NaN filtered)")

        self.filenames = df["image_name"].values
        self.texts = df["utf8string"].values

        self.alphabet = alphabet
        self.char2idx = {c: i + 1 for i, c in enumerate(alphabet)}  # 0 — blank (CTC)

        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((img_height, img_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.filenames)

    def encode_text(self, text):
        # Handle NaN or non-string values
        if not isinstance(text, str):
            logger.warning(f"Non-string text encountered: {text} (type: {type(text)}), using space")
            text = " "
        
        encoded = []
        for c in text:
            if c in self.char2idx:
                encoded.append(self.char2idx[c])
            else:
                encoded.append(1)  # space for unknown
        return encoded if encoded else [1]  # return at least [1] if empty

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        text = self.texts[idx]

        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        encoded = self.encode_text(text)

        return {
            "image": image,
            "label": torch.tensor(encoded, dtype=torch.long),
            "length": torch.tensor(len(encoded), dtype=torch.long),
            "text": text  # удобно для дебага
        }