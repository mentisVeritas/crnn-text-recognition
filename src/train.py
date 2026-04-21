import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import logging
import os
from tqdm import tqdm

from src.data import OCRDataset
from src.model import CRNN

logger = logging.getLogger(__name__)


def _filter_batch_for_ctc(images, labels_list, lengths, max_input_len, device):
    """Keep only samples with target length <= input length for CTC."""
    valid_mask = lengths <= max_input_len
    valid_count = int(valid_mask.sum().item())
    total_count = int(lengths.numel())
    skipped = total_count - valid_count
    if valid_count == 0:
        return None, None, None, skipped

    valid_images = images[valid_mask]
    valid_labels = [labels_list[i] for i in range(total_count) if bool(valid_mask[i].item())]
    valid_lengths = lengths[valid_mask]
    labels_tensor = torch.cat(valid_labels).to(device)
    return valid_images, labels_tensor, valid_lengths, skipped


def collate_fn_ctc(batch):
    """Custom collate function for CTC loss that handles variable length sequences"""
    images = torch.stack([item['image'] for item in batch])
    labels = [item['label'] for item in batch]
    lengths = torch.tensor([item['length'].item() for item in batch])
    
    # Concatenate all labels and keep track of lengths
    labels_tensor = torch.cat(labels)
    
    return {
        'image': images,
        'label': labels,
        'length': lengths,
        'labels_tensor': labels_tensor  # concatenated labels for CTC
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    valid_steps = 0
    skipped_samples = 0
    skipped_batches = 0

    # Determine CPU device for CTCLoss (not supported on MPS)
    loss_device = torch.device("cpu") if device.type == "mps" else device

    for batch in tqdm(loader, desc="Training"):
        images = batch["image"].to(device)
        labels = batch["label"]
        lengths = batch["length"].to(device)

        # модель (on GPU/MPS)
        logits = model(images)                 # [B, W, C]
        logits = logits.permute(1, 0, 2)       # [T, B, C]
        log_probs = torch.log_softmax(logits, dim=2)

        max_input_len = int(log_probs.size(0))
        filtered = _filter_batch_for_ctc(images, labels, lengths, max_input_len, device)
        if filtered[0] is None:
            skipped_batches += 1
            skipped_samples += filtered[3]
            continue
        images, labels_tensor, target_lengths, skipped = filtered
        skipped_samples += skipped
        batch_size = images.size(0)

        # CTC targets
        input_lengths = torch.full(
            size=(batch_size,),
            fill_value=log_probs.size(0),
            dtype=torch.long
        ).to(device)

        # loss (CTCLoss not supported on MPS, move to CPU)
        if device.type == "mps":
            log_probs = log_probs.to(loss_device)
            labels_tensor = labels_tensor.to(loss_device)
            input_lengths = input_lengths.to(loss_device)
            target_lengths = target_lengths.to(loss_device)

        loss = criterion(log_probs, labels_tensor, input_lengths, target_lengths)
        if not torch.isfinite(loss):
            skipped_batches += 1
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        valid_steps += 1

    if skipped_samples > 0:
        logger.warning(
            "Training epoch skipped %d samples and %d batches due to CTC constraints/non-finite loss.",
            skipped_samples,
            skipped_batches,
        )
    if valid_steps == 0:
        raise RuntimeError(
            "All training batches were skipped. Increase img_width or reduce target text length."
        )
    return total_loss / valid_steps


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    valid_steps = 0
    skipped_samples = 0

    # Determine CPU device for CTCLoss (not supported on MPS)
    loss_device = torch.device("cpu") if device.type == "mps" else device

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images = batch["image"].to(device)
            labels = batch["label"]
            lengths = batch["length"].to(device)

            logits = model(images)
            logits = logits.permute(1, 0, 2)
            log_probs = torch.log_softmax(logits, dim=2)

            max_input_len = int(log_probs.size(0))
            filtered = _filter_batch_for_ctc(images, labels, lengths, max_input_len, device)
            if filtered[0] is None:
                skipped_samples += filtered[3]
                continue
            images, labels_tensor, target_lengths, skipped = filtered
            skipped_samples += skipped

            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=log_probs.size(0),
                dtype=torch.long
            ).to(device)

            # loss (CTCLoss not supported on MPS, move to CPU)
            if device.type == "mps":
                log_probs = log_probs.to(loss_device)
                labels_tensor = labels_tensor.to(loss_device)
                input_lengths = input_lengths.to(loss_device)
                target_lengths = target_lengths.to(loss_device)

            loss = criterion(log_probs, labels_tensor, input_lengths, target_lengths)
            if not torch.isfinite(loss):
                continue
            total_loss += loss.item()
            valid_steps += 1

    if skipped_samples > 0:
        logger.warning(
            "Validation skipped %d samples due to CTC constraints.",
            skipped_samples,
        )
    if valid_steps == 0:
        return float("inf")
    return total_loss / valid_steps


def run_training(config, project_root=None):
    """Train CRNN. Paths in config are resolved against project_root (default: cwd)."""
    root = project_root or os.getcwd()

    # Select device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: CUDA - {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using device: MPS (Apple Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU (slow - consider using GPU)")

    logger.info(f"Device selected: {device}")

    ckpt_dir = os.path.join(root, "outputs", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Resolve paths - support both relative and absolute paths
    images_dir = config["images_dir"]
    labels_path = config["labels_path"]

    if not os.path.isabs(images_dir):
        images_dir = os.path.join(root, images_dir)
    if not os.path.isabs(labels_path):
        labels_path = os.path.join(root, labels_path)

    logger.info(f"Loading dataset from:")
    logger.info(f"  Images: {images_dir}")
    logger.info(f"  Labels: {labels_path}")

    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # dataset
    dataset = OCRDataset(
        images_dir=images_dir,
        labels_path=labels_path,
        alphabet=config["alphabet"],
        img_height=config["img_height"],
        img_width=config["img_width"],
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset is empty after loading CSV. Check labels_path and column names "
            "(expected image_name, utf8string)."
        )
    logger.info(f"Loaded dataset with {len(dataset)} samples")

    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn_ctc
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn_ctc
    )

    # model
    num_classes = len(config["alphabet"]) + 1
    model = CRNN(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 1, config["img_height"], config["img_width"], device=device)
        time_steps = int(model(dummy).size(1))
    model.train()
    text_lengths = [len(str(t)) for t in dataset.texts]
    too_long = sum(1 for t in text_lengths if t > time_steps)
    if too_long > 0:
        pct = 100.0 * too_long / len(text_lengths)
        logger.warning(
            "CTC capacity warning: %d/%d samples (%.2f%%) have text length > time steps (%d). "
            "Consider increasing img_width.",
            too_long,
            len(text_lengths),
            pct,
            time_steps,
        )

    # training
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    best_val_loss = float('inf')
    logger.info(f"Starting training for {config['epochs']} epochs...")

    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        logger.info(f"Epoch {epoch+1}/{config['epochs']}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")

    logger.info("Training completed!")
    return model