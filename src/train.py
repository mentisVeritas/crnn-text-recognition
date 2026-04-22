"""
Training facade for notebooks/scripts.

Public API is preserved for backward compatibility:
- data/model setup helpers
- checkpointed training loop
- evaluation/leaderboard helpers
- run_training CLI entry
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from src.evaluation import (
    evaluate_val_subset,
    run_leaderboard_log,
    show_hard_val_examples,
    show_leaderboard,
    visualize_val_predictions,
)
from src.runtime import setup_outputs_file_logging
from src.training_loop import load_weights_for_inference, run_epoch, train_with_checkpoints
from src.training_setup import (
    build_dataloaders,
    build_model_bundle,
    collate_fn_ctc,
    log_ctc_capacity_warning,
    model_output_time_steps,
)
from src.utils import get_device

logger = logging.getLogger(__name__)



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

    dataset, _, _, train_loader, val_loader = build_dataloaders(
        images_dir, labels_path, config, split_seed=split_seed
    )
    logger.info("Dataset size: %s", len(dataset))

    model, optimizer, criterion = build_model_bundle(config, device)
    logger.info("Model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")
    log_ctc_capacity_warning(model, dataset, config, device)

    ckpt_dir = root / "outputs" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / "checkpoint.pth"
    best_path = ckpt_dir / "best_model.pth"

    # CLI policy: fresh init (no checkpoint resume); see function docstring for best_model guard.
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
        resume=False,
    )
    return model


__all__ = [
    "collate_fn_ctc",
    "build_dataloaders",
    "build_model_bundle",
    "model_output_time_steps",
    "log_ctc_capacity_warning",
    "run_epoch",
    "train_with_checkpoints",
    "load_weights_for_inference",
    "visualize_val_predictions",
    "show_hard_val_examples",
    "evaluate_val_subset",
    "run_leaderboard_log",
    "show_leaderboard",
    "run_training",
]
