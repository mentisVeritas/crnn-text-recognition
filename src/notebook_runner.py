"""
Logic for notebooks/experiments.ipynb — keep the notebook as thin orchestration only.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data import OCRDataset
from src.experiment_log import append_experiment_log, collect_predictions, metrics_from_rows
from src.inference import decode_with_confidence
from src.metrics import EvalMetrics
from src.model import CRNN
from src.visualization import plot_prediction_grid, sample_indices


def collate_fn_ctc(batch):
    images = torch.stack([item["image"] for item in batch])
    labels = [item["label"] for item in batch]
    lengths = torch.tensor([item["length"].item() for item in batch], dtype=torch.long)
    labels_tensor = torch.cat(labels)
    return {
        "image": images,
        "label": labels,
        "length": lengths,
        "labels_tensor": labels_tensor,
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


def print_ctc_capacity_warning(model: nn.Module, dataset: OCRDataset, config: dict, device: torch.device):
    with torch.no_grad():
        dummy = torch.zeros(1, 1, config["img_height"], config["img_width"], device=device)
        time_steps = int(model(dummy).size(1))
    text_lengths = [len(str(t)) for t in dataset.texts]
    too_long = sum(1 for t in text_lengths if t > time_steps)
    print(f"Model time steps (T): {time_steps}")
    print(f"Samples with label length > T: {too_long}/{len(text_lengths)}")


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


def run_epoch(model, loader, optimizer, criterion, device, train: bool = True):
    model.train(train)
    total_loss = 0.0
    valid_steps = 0
    skipped_samples = 0
    loss_device = torch.device("cpu") if device.type == "mps" else device

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]
        lengths = batch["length"].to(device)

        if train:
            optimizer.zero_grad()

        logits = model(images)
        logits = logits.permute(1, 0, 2)
        log_probs = torch.log_softmax(logits, dim=2)

        filtered = _filter_batch_for_ctc(images, labels, lengths, int(log_probs.size(0)), device)
        if filtered[0] is None:
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
            continue

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += float(loss.item())
        valid_steps += 1

    avg_loss = total_loss / valid_steps if valid_steps > 0 else float("inf")
    return avg_loss, skipped_samples, valid_steps


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
):
    if not training:
        print("TRAINING=False -> training is skipped. Will load saved weights in next cell.")
        return

    best_val_loss = float("inf")
    start_epoch = 0

    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        except Exception as e:
            print(f"Failed to load checkpoint, starting from scratch: {e}")

    for epoch in range(start_epoch, config["epochs"]):
        train_loss, train_skipped, train_steps = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )
        val_loss, val_skipped, val_steps = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False
        )

        print(
            f"Epoch {epoch + 1}/{config['epochs']} | "
            f"train_loss={train_loss:.4f} (steps={train_steps}, skipped={train_skipped}) | "
            f"val_loss={val_loss:.4f} (steps={val_steps}, skipped={val_skipped})"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                best_path,
            )
            print(f"Saved BEST model -> {best_path}")

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            },
            checkpoint_path,
        )

    print(f"Best val loss: {best_val_loss:.4f}")


def _load_state_dict_payload(path: Path, device: torch.device):
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
        raise FileNotFoundError("No model checkpoint found. Run training first or set TRAINING=True.")

    state = _load_state_dict_payload(load_path, device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Checkpoint loaded: {load_path}")


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
    print(f"Shown samples: {metrics.total}")
    print(f"Correct words: {metrics.correct}/{metrics.total}")
    print(f"Accuracy: {metrics.accuracy:.3f}")
    print(f"CER: {metrics.cer:.3f}")
    print(f"WER: {metrics.wer:.3f}")
    print(f"Average confidence: {metrics.avg_confidence * 100:.1f}%")


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
    print(f"Analyzed samples: {len(rows)}")
    print(f"Mistakes in analyzed set: {mistakes}/{len(rows)}")
    print("Showing hardest cases by edit distance.")


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

    print("Current run metrics on leaderboard subset:")
    print(f"Accuracy: {metrics.accuracy * 100:.2f}%")
    print(f"CER: {metrics.cer * 100:.2f}%")
    print(f"WER: {metrics.wer * 100:.2f}%")
    print(f"Average confidence: {metrics.avg_confidence * 100:.2f}%")
    print(f"Logged to: {log_path}")
    print(
        "Saved run:",
        f"acc={metrics.accuracy * 100:.2f}%",
        f"cer={metrics.cer * 100:.2f}%",
        f"wer={metrics.wer * 100:.2f}%",
        f"conf={metrics.avg_confidence * 100:.2f}%",
    )

    show_leaderboard(log_path, top_k=10)


def show_leaderboard(log_path: Path, top_k: int = 10):
    if not log_path.exists():
        print("No experiment log yet.")
        return

    import pandas as pd

    df = pd.read_csv(log_path)
    if df.empty:
        print("Experiment log is empty.")
        return

    board = df.sort_values(by=["accuracy", "cer", "wer"], ascending=[False, True, True])
    print(f"Total logged runs: {len(board)}")

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
