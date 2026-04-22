import logging
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.runtime import project_root_from_checkpoint_path, setup_outputs_file_logging
from src.training_setup import model_output_time_steps

logger = logging.getLogger(__name__)


def _filter_batch_for_ctc(images, labels_list, lengths, max_input_len, target_device):
    valid_mask = lengths <= max_input_len
    valid_count = int(valid_mask.sum().item())
    total_count = int(lengths.numel())
    skipped = total_count - valid_count
    if valid_count == 0:
        return None, None, None, skipped

    valid_images = images[valid_mask]
    valid_labels = [labels_list[i] for i in range(total_count) if bool(valid_mask[i].item())]
    valid_lengths = lengths[valid_mask]
    labels_tensor = torch.cat(valid_labels).to(target_device)
    return valid_images, labels_tensor, valid_lengths, skipped


def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    ctc_width: int,
    train: bool = True,
    use_tqdm: bool = False,
):
    model.train(train)
    total_loss = 0.0
    valid_steps = 0
    skipped_samples = 0
    skipped_batches = 0
    loss_device = torch.device("cpu") if device.type == "mps" else device

    iterator = loader
    if use_tqdm:
        iterator = tqdm(loader, desc="Train" if train else "Valid")

    for batch in iterator:
        images = batch["image"].to(device)
        labels = batch["label"]
        lengths = batch["length"].to(device)

        if train:
            optimizer.zero_grad()

        filtered = _filter_batch_for_ctc(images, labels, lengths, ctc_width, device)
        if filtered[0] is None:
            skipped_batches += 1
            skipped_samples += filtered[3]
            continue

        images, labels_tensor, target_lengths, skipped = filtered
        skipped_samples += skipped

        logits = model(images).permute(1, 0, 2)
        log_probs = torch.log_softmax(logits, dim=2)
        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=log_probs.size(0),
            dtype=torch.long,
            device=device,
        )

        if device.type == "mps":
            log_probs = log_probs.to(loss_device)
            labels_tensor = labels_tensor.to(loss_device)
            input_lengths = input_lengths.to(loss_device)
            target_lengths = target_lengths.to(loss_device)

        loss = criterion(log_probs, labels_tensor, input_lengths, target_lengths)
        if not torch.isfinite(loss):
            if train:
                skipped_batches += 1
            continue

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += float(loss.item())
        valid_steps += 1

    if skipped_samples > 0:
        logger.warning(
            "Epoch skipped %d samples (and %d train batches) due to CTC / non-finite loss.",
            skipped_samples,
            skipped_batches,
        )
    if valid_steps == 0:
        if train:
            raise RuntimeError(
                "All training batches were skipped. Increase img_width or reduce target text length."
            )
        return float("inf"), skipped_samples, 0

    avg = total_loss / valid_steps
    return avg, skipped_samples, valid_steps


def _best_val_loss_in_checkpoint_file(path: Path) -> float | None:
    """Read best_val_loss from a saved dict checkpoint, if present."""
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "best_val_loss" in payload:
            return float(payload["best_val_loss"])
    except Exception:
        return None
    return None


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

    pr = project_root_from_checkpoint_path(checkpoint_path)
    if pr is not None:
        setup_outputs_file_logging(pr)

    ctc_width = model_output_time_steps(model, config, device)

    best_val_loss = float("inf")
    start_epoch = 0
    on_disk_best_val: float | None = None

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
    elif not resume:
        logger.info(
            "resume=False: not loading %s — training from randomly initialized weights.",
            checkpoint_path,
        )
        on_disk_best_val = _best_val_loss_in_checkpoint_file(best_path)
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
        train_loss, train_skipped, train_steps = run_epoch(
            model, train_loader, optimizer, criterion, device, ctc_width, train=True, use_tqdm=use_tqdm
        )
        val_loss, val_skipped, val_steps = run_epoch(
            model, val_loader, optimizer, criterion, device, ctc_width, train=False, use_tqdm=use_tqdm
        )

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


def _load_state_dict_payload(path: Path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    return payload


def load_weights_for_inference(
    model: nn.Module,
    best_path: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    load_path = best_path if best_path.exists() else checkpoint_path
    if not load_path.exists():
        raise FileNotFoundError("No checkpoint found. Train first or set TRAINING=True.")

    state = _load_state_dict_payload(load_path)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Checkpoint loaded: %s", load_path)
