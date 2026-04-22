import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data import OCRDataset
from src.model import CRNN

logger = logging.getLogger(__name__)


def collate_fn_ctc(batch):
    images = torch.stack([item["image"] for item in batch])
    labels = [item["label"] for item in batch]
    lengths = torch.tensor([item["length"].item() for item in batch], dtype=torch.long)
    return {
        "image": images,
        "label": labels,
        "length": lengths,
    }


def build_dataloaders(
    images_dir: str,
    labels_path: str,
    config: dict,
    split_seed: int = 42,
):
    dataset = OCRDataset(
        images_dir=images_dir,
        labels_path=labels_path,
        alphabet=config["alphabet"],
        img_height=config["img_height"],
        img_width=config["img_width"],
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(split_seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn_ctc,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn_ctc,
    )
    return dataset, train_dataset, val_dataset, train_loader, val_loader


def build_model_bundle(config: dict, device: torch.device):
    num_classes = len(config["alphabet"]) + 1
    model = CRNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    return model, optimizer, criterion


def model_output_time_steps(model: nn.Module, config: dict, device: torch.device) -> int:
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 1, config["img_height"], config["img_width"], device=device)
        return int(model(dummy).size(1))


def log_ctc_capacity_warning(model: nn.Module, dataset: OCRDataset, config: dict, device: torch.device):
    time_steps = model_output_time_steps(model, config, device)
    text_lengths = [len(str(t)) for t in dataset.texts]
    too_long = sum(1 for t in text_lengths if t > time_steps)
    logger.info("Model time steps (T): %s", time_steps)
    logger.info("Samples with label length > T: %s/%s", too_long, len(text_lengths))
    if too_long > 0:
        pct = 100.0 * too_long / len(text_lengths)
        logger.warning(
            "CTC capacity: %d/%d samples (%.2f%%) have text length > time steps (%d). "
            "Consider increasing img_width.",
            too_long,
            len(text_lengths),
            pct,
            time_steps,
        )
