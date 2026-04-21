"""
Training, checkpoints, resume, and experiment helpers.

Notebook: import from here and call ``train_with_checkpoints`` (default ``resume=True``).
Terminal: ``scripts/train.py`` → ``run_training()`` which calls ``train_with_checkpoints(..., resume=False)``.

See docstrings on those two entry points for checkpoint semantics (checkpoint.pth vs best_model.pth).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data import OCRDataset
from src.experiment_log import append_experiment_log, collect_predictions, metrics_from_rows
from src.inference import decode_with_confidence
from src.metrics import EvalMetrics
from src.model import CRNN
from src.utils import get_device
from src.visualization import plot_prediction_grid, sample_indices

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
    g = torch.Generator().manual_seed(split_seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)

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

    Args:
        training: If False, returns immediately (no epochs); use for notebook eval-only flows.
        use_tqdm: Progress bars on train/val loaders (CLI sets True).
        resume: If True and ``checkpoint_path`` exists, loads weights/optimizer and continues epochs.
            If False, starts from whatever is already in ``model`` (typically random init) and does not
            load ``checkpoint_path``. In that case, if ``best_path`` already exists, it is overwritten
            only when validation loss beats the ``best_val_loss`` stored inside that file (CLI behavior
            via ``run_training``).

    The loop runs ``epoch`` in ``range(start_epoch, config["epochs"])``; if the checkpoint is already
    past ``epochs``, no steps run (see warning log).
    """
    if not training:
        logger.info("TRAINING=False -> training skipped; load weights before eval.")
        return

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


def make_predict_fn(model: nn.Module, alphabet: str, device: torch.device):
    def predict_fn(image_tensor: torch.Tensor) -> tuple[str, float]:
        return decode_with_confidence(model, image_tensor, alphabet, device)

    return predict_fn


def visualize_val_predictions(
        model: nn.Module,
        val_dataset,
        config: dict,
        device: torch.device,
        sample_count: int = 12,
        fixed: bool = False,
        seed: int = 42,
):
    predict_fn = make_predict_fn(model, config["alphabet"], device)
    indices = sample_indices(len(val_dataset), sample_count, fixed, seed)
    rows = collect_predictions(val_dataset, indices, predict_fn)

    items = [
        {
            "image": r["image"],
            "pred": r["pred"],
            "true": r["true"],
            "confidence": r["confidence"],
            "correct": r["correct"],
        }
        for r in rows
    ]
    plot_prediction_grid(items)

    metrics = metrics_from_rows(rows)
    logger.info("Shown samples: %s", metrics.total)
    logger.info("Correct words: %s/%s", metrics.correct, metrics.total)
    logger.info("Accuracy: %.3f | CER: %.3f | WER: %.3f", metrics.accuracy, metrics.cer, metrics.wer)
    logger.info("Average confidence: %.1f%%", metrics.avg_confidence * 100)


def show_hard_val_examples(
        model: nn.Module,
        val_dataset,
        config: dict,
        device: torch.device,
        sample_count: int = 200,
        top_k: int = 12,
        fixed: bool = True,
        seed: int = 42,
):
    predict_fn = make_predict_fn(model, config["alphabet"], device)
    indices = sample_indices(len(val_dataset), sample_count, fixed, seed)
    rows = collect_predictions(val_dataset, indices, predict_fn)
    rows_sorted = sorted(rows, key=lambda x: (x["dist"], -x["confidence"]), reverse=True)
    hard = rows_sorted[: min(top_k, len(rows_sorted))]

    items = [
        {
            "image": r["image"],
            "pred": r["pred"],
            "true": r["true"],
            "confidence": r["confidence"],
            "correct": r["correct"],
            "extra_line": f"edit_distance={r['dist']}",
        }
        for r in hard
    ]
    plot_prediction_grid(items, row_height=4.2)

    mistakes = sum(1 for x in rows if not x["correct"])
    logger.info("Analyzed samples: %s | mistakes: %s/%s", len(rows), mistakes, len(rows))


def evaluate_val_subset(
        model: nn.Module,
        val_dataset,
        config: dict,
        device: torch.device,
        sample_count: int,
        fixed: bool,
        seed: int,
) -> EvalMetrics:
    predict_fn = make_predict_fn(model, config["alphabet"], device)
    indices = sample_indices(len(val_dataset), sample_count, fixed, seed)
    rows = collect_predictions(val_dataset, indices, predict_fn)
    return metrics_from_rows(rows)


def run_leaderboard_log(
        model: nn.Module,
        val_dataset,
        config: dict,
        device: torch.device,
        project_root: Path,
        best_path: Path,
        checkpoint_path: Path,
        training: bool,
        sample_count: int,
        fixed: bool,
        seed: int,
        note: str = "",
):
    log_path = project_root / "outputs" / "experiment_log.csv"
    metrics = evaluate_val_subset(model, val_dataset, config, device, sample_count, fixed, seed)

    mode = "train" if training else "inference_only"
    if best_path.exists():
        model_file = str(best_path)
    elif checkpoint_path.exists():
        model_file = str(checkpoint_path)
    else:
        model_file = "not_found"

    append_experiment_log(log_path, metrics, mode, config, sample_count, model_file, note)

    logger.info(
        "Leaderboard subset | acc=%.2f%% cer=%.2f%% wer=%.2f%% conf=%.2f%% -> %s",
        metrics.accuracy * 100,
        metrics.cer * 100,
        metrics.wer * 100,
        metrics.avg_confidence * 100,
        log_path,
    )

    show_leaderboard(log_path, top_k=10)


def show_leaderboard(log_path: Path, top_k: int = 10):
    import pandas as pd

    if not log_path.exists():
        logger.info("No experiment log yet.")
        return

    df = pd.read_csv(log_path)
    if df.empty:
        logger.info("Experiment log is empty.")
        return

    board = df.sort_values(by=["accuracy", "cer", "wer"], ascending=[False, True, True])
    logger.info("Total logged runs: %s", len(board))

    display_cols = [
        "timestamp",
        "mode",
        "sample_count",
        "accuracy",
        "cer",
        "wer",
        "avg_confidence",
        "batch_size",
        "lr",
        "model_file",
        "note",
    ]
    view = board[display_cols].head(top_k).copy()
    view["accuracy"] = (view["accuracy"] * 100).round(2)
    view["cer"] = (view["cer"] * 100).round(2)
    view["wer"] = (view["wer"] * 100).round(2)
    view["avg_confidence"] = (view["avg_confidence"] * 100).round(2)
    view["model_file"] = view["model_file"].apply(lambda p: Path(str(p)).name)

    view = view.rename(
        columns={
            "accuracy": "accuracy_%",
            "cer": "cer_%",
            "wer": "wer_%",
            "avg_confidence": "avg_conf_%",
            "model_file": "weights",
        }
    )

    try:
        from IPython.display import display

        display(view.reset_index(drop=True))
    except Exception:
        print(view.to_string(index=False))


def run_training(config, project_root=None, split_seed: int = 42):
    """
    CLI entry: train from scratch (does not load checkpoint.pth), for all epochs in config.
    best_model.pth is overwritten only when validation beats the value already stored in that file (if any).
    For resume-from-checkpoint behavior (e.g. notebook), call train_with_checkpoints(..., resume=True).
    """
    root = Path(project_root or os.getcwd())
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
