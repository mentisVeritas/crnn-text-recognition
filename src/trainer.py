import logging

import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------- CTC FILTER ----------
def filter_batch_for_ctc(
    labels,
    lengths,
    max_input_len: int,
    target_device: torch.device,
):
    """Drop samples whose target length exceeds CTC input length."""

    valid_labels = []
    valid_lengths = []
    valid_indices = []

    total_count = int(lengths.numel())

    for idx, length in enumerate(lengths.tolist()):
        if length <= max_input_len:
            valid_labels.append(labels[idx])
            valid_lengths.append(length)
            valid_indices.append(idx)

    if len(valid_labels) == 0:
        return None, None, None, total_count

    labels_tensor = torch.cat(valid_labels).to(target_device)
    lengths_tensor = torch.tensor(valid_lengths, dtype=torch.long, device=target_device)

    return labels_tensor, lengths_tensor, valid_indices, total_count


# ---------- TRAIN / VAL EPOCH ----------
def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    training: bool = True,
    use_tqdm: bool = True,
):
    model.train(training)

    total_loss = 0.0
    processed_batches = 0

    iterator = loader
    if use_tqdm:
        iterator = tqdm(
            loader,
            desc="Train" if training else "Valid",
            leave=True,
        )

    for batch in iterator:
        images = batch["image"].to(device)

        if images.ndim != 4 or images.size(1) != 3:
            raise RuntimeError(
                f"Expected RGB tensor [B,3,H,W], got shape: {tuple(images.shape)}"
            )

        labels = batch["label"]
        lengths = batch["length"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        log_probs = logits.log_softmax(2).permute(1, 0, 2)

        input_lengths = torch.full(
            size=(logits.size(0),),
            fill_value=logits.size(1),
            dtype=torch.long,
            device=device,
        )

        labels_tensor, lengths_tensor, valid_indices, _ = filter_batch_for_ctc(
            labels,
            lengths,
            int(logits.size(1)),
            device,
        )

        if labels_tensor is None:
            continue

        valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long, device=device)

        log_probs = log_probs[:, valid_indices_tensor, :]
        input_lengths = input_lengths[valid_indices_tensor]

        # CTCLoss is not implemented on MPS -> fallback to CPU
        loss_device = torch.device("cpu") if device.type == "mps" else device

        if device.type == "mps":
            log_probs = log_probs.to(loss_device)
            labels_tensor = labels_tensor.to(loss_device)
            input_lengths = input_lengths.to(loss_device)
            lengths_tensor = lengths_tensor.to(loss_device)

        loss = criterion(
            log_probs,
            labels_tensor,
            input_lengths,
            lengths_tensor,
        )

        if not torch.isfinite(loss):
            logger.warning("Skipping non-finite loss: %s", loss.item())
            continue

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += float(loss.item())
        processed_batches += 1

        if use_tqdm:
            iterator.set_postfix(loss=f"{loss.item():.4f}")

    if processed_batches == 0:
        return float("inf")

    return total_loss / processed_batches