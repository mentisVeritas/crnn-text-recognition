"""
Training helpers and CLI entrypoint.

This module groups training setup and training loop logic, and re-exports
selected evaluation helpers for notebook convenience.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

from src.data import build_dataloaders, OCRDataset
from src.evaluation import (
    evaluate_val_subset,
    run_leaderboard_log,
    show_hard_val_examples,
    show_leaderboard,
    visualize_val_predictions,
)
from src.trainer import run_epoch
from src.checkpointing import (
    best_val_loss_in_checkpoint_file,
    load_weights_for_inference,
)
from src.model import CRNN
from src.utils import get_device, setup_outputs_file_logging

logger = logging.getLogger(__name__)
EPOCH_LOG_NAME = "epoch_metrics.log"


def build_model_bundle(config: dict, device: torch.device):
    """Create CRNN model, Adam optimizer and CTCLoss criterion."""
    num_classes = len(config["alphabet"]) + 1
    model = CRNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    return model, optimizer, criterion


def model_output_time_steps(model: nn.Module, config: dict, device: torch.device) -> int:
    """Estimate CTC time dimension T from a dummy forward pass."""
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, config["img_height"], config["img_width"], device=device)
        return int(model(dummy).size(1))


def log_ctc_capacity_warning(model: nn.Module, dataset: OCRDataset, config: dict, device: torch.device):
    """Warn when label lengths exceed model time-steps T for CTC."""
    time_steps = model_output_time_steps(model, config, device)
    text_lengths = [len(t) for t in dataset.texts]
    too_long = sum(1 for t in text_lengths if t > time_steps)

    if too_long > 0:
        logger.warning(f"Too long samples ratio: {too_long / len(text_lengths):.2%}")

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


def train_with_checkpoints(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        device: torch.device,
        checkpoint_path: Path,
        best_path: Path,
        training: bool,
        use_tqdm: bool = False,
        resume: bool = True,
):
    """
    Main training loop with per-epoch ``checkpoint_path`` and best-on-val ``best_path``.
    """
    if not training:
        logger.info("TRAINING=False -> training skipped; load weights before eval.")
        return

    ctc_width = model_output_time_steps(model, config, device)

    best_val_loss = float("inf")
    epoch_log_path = checkpoint_path.parent.parent / "logs" / EPOCH_LOG_NAME
    epoch_log_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    on_disk_best_val: float | None = None

    # Resume path (notebook default): restore model + optimizer + epoch cursor.
    if resume and checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            logger.info(
                "Resumed from checkpoint; best_val_loss=%.4f; next loop epoch index=%d (epochs in config=%d)",
                best_val_loss,
                start_epoch,
                config["epochs"],
            )
            ctc_width = model_output_time_steps(model, config, device)
        except Exception as e:
            logger.warning("Failed to load checkpoint, starting from scratch: %s", e)
    # Fresh path (CLI default): do not load checkpoint; optionally protect on-disk best.
    elif not resume:
        logger.info(
            "resume=False: not loading %s — training from randomly initialized weights.",
            checkpoint_path,
        )
        on_disk_best_val = best_val_loss_in_checkpoint_file(best_path)
        if on_disk_best_val is not None:
            logger.info(
                "Existing %s has best_val_loss=%.4f — will update that file only if this run beats it.",
                best_path.name,
                on_disk_best_val,
            )

    if start_epoch >= config["epochs"]:
        logger.warning(
            "No training steps will run (next epoch index %d >= epochs=%d). Raise epochs or use resume with a younger checkpoint.",
            start_epoch,
            config["epochs"],
        )

    for epoch in range(start_epoch, config["epochs"]):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training=True,
            use_tqdm=use_tqdm,
        )

        train_skipped = 0
        train_steps = len(train_loader)

        val_loss = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            training=False,
            use_tqdm=use_tqdm,
        )

        val_skipped = 0
        val_steps = len(val_loader)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f (steps=%d, skipped=%d) | val_loss=%.4f (steps=%d, skipped=%d)",
            epoch + 1,
            config["epochs"],
            train_loss,
            train_steps,
            train_skipped,
            val_loss,
            val_steps,
            val_skipped,
        )
        with open(epoch_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"Epoch {epoch + 1}/{config['epochs']} | "
                f"train_loss={train_loss:.4f} "
                f"(steps={train_steps}, skipped={train_skipped}) | "
                f"val_loss={val_loss:.4f} "
                f"(steps={val_steps}, skipped={val_skipped})\n"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            write_best_file = True
            if on_disk_best_val is not None and val_loss >= on_disk_best_val:
                write_best_file = False
                logger.info(
                    "Run-best val_loss=%.4f but on-disk best is %.4f — not overwriting %s.",
                    val_loss,
                    on_disk_best_val,
                    best_path,
                )
            if write_best_file:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                    },
                    best_path,
                )
                logger.info("Saved BEST model -> %s", best_path)

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            },
            checkpoint_path,
        )

    logger.info("Best val loss: %.4f", best_val_loss)


def run_training(config, project_root=None, split_seed: int = 42):
    """
    CLI entry: train from scratch (does not load checkpoint.pth), for all epochs in config.
    best_model.pth is overwritten only when validation beats the value already stored in that file (if any).
    For resume-from-checkpoint behavior (e.g. notebook), call train_with_checkpoints(..., resume=True).
    """
    root = Path(project_root or os.getcwd())
    setup_outputs_file_logging(root)
    device = get_device()

    images_dir = config["images_dir"]
    labels_path = config["labels_path"]
    if not os.path.isabs(images_dir):
        images_dir = str(root / images_dir)
    else:
        images_dir = str(images_dir)
    if not os.path.isabs(labels_path):
        labels_path = str(root / labels_path)
    else:
        labels_path = str(labels_path)

    logger.info("Images: %s", images_dir)
    logger.info("Labels: %s", labels_path)

    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    train_loader, val_loader = build_dataloaders(
        config,
        images_dir,
        labels_path,
        split_seed=split_seed,
    )

    dataset = train_loader.dataset.dataset
    logger.info("Dataset size: %s", len(dataset))

    model, optimizer, criterion = build_model_bundle(config, device)
    logger.info("Model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")
    log_ctc_capacity_warning(model, dataset, config, device)

    ckpt_dir = root / "outputs" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / "checkpoint.pth"
    best_path = ckpt_dir / "best_model.pth"

    resume = bool(config.get("resume", False))

    logger.info("Resume checkpoints: %s", resume)

    train_with_checkpoints(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        config,
        device,
        checkpoint_path,
        best_path,
        training=True,
        use_tqdm=True,
        resume=resume,
    )
    return model


__all__ = [
    "build_model_bundle",
    "model_output_time_steps",
    "log_ctc_capacity_warning",
    "train_with_checkpoints",
    "load_weights_for_inference",
    "visualize_val_predictions",
    "show_hard_val_examples",
    "evaluate_val_subset",
    "run_leaderboard_log",
    "show_leaderboard",
    "run_training",
]
