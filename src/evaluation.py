"""Evaluation helpers: prediction collection, visualization and leaderboard logging."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import torch

from src.inference import decode_with_confidence
from src.metrics import EvalMetrics, compute_metrics, levenshtein
from src.visualization import plot_prediction_grid, sample_indices

logger = logging.getLogger(__name__)


def _make_predict_fn(model: torch.nn.Module, alphabet: str, device: torch.device) -> Callable[[torch.Tensor], tuple[str, float]]:
    def predict_fn(image_tensor: torch.Tensor) -> tuple[str, float]:
        return decode_with_confidence(model, image_tensor, alphabet, device)

    return predict_fn


def collect_predictions(
    dataset,
    indices: Sequence[int],
    predict_fn: Callable[[torch.Tensor], tuple[str, float]],
) -> list[dict]:
    """Collect decoded predictions and per-sample metadata for selected dataset indices."""
    rows = []
    for idx in indices:
        item = dataset[idx]
        image = item["image"]
        true_text = str(item["text"])
        pred_text, conf = predict_fn(image.unsqueeze(0))
        dist = levenshtein(pred_text, true_text)
        rows.append(
            {
                "idx": idx,
                "image": image,
                "true": true_text,
                "pred": pred_text,
                "confidence": conf,
                "dist": dist,
                "correct": pred_text == true_text,
            }
        )
    return rows


def metrics_from_rows(rows) -> EvalMetrics:
    """Compute aggregate metrics from collected row dicts."""
    pairs = [(r["pred"], r["true"], float(r["confidence"])) for r in rows]
    return compute_metrics(pairs)


def append_experiment_log(
    log_path: Path,
    metrics: EvalMetrics,
    mode: str,
    config: dict,
    sample_count: int,
    model_file: str,
    note: str = "",
) -> dict:
    """Append one evaluation run to CSV leaderboard and return written row."""
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "epochs_config": int(config["epochs"]),
        "batch_size": int(config["batch_size"]),
        "lr": float(config["lr"]),
        "img_size": f"{config['img_height']}x{config['img_width']}",
        "sample_count": int(sample_count),
        "accuracy": float(metrics.accuracy),
        "cer": float(metrics.cer),
        "wer": float(metrics.wer),
        "avg_confidence": float(metrics.avg_confidence),
        "correct": int(metrics.correct),
        "total": int(metrics.total),
        "model_file": model_file,
        "note": str(note),
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if log_path.exists():
        df_old = pd.read_csv(log_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(log_path, index=False)
    return row


def load_leaderboard(log_path: Path) -> pd.DataFrame:
    """Load and sort leaderboard CSV by quality metrics."""
    if not log_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(log_path)
    if df.empty:
        return df
    return df.sort_values(by=["accuracy", "cer", "wer"], ascending=[False, True, True]).reset_index(drop=True)


def visualize_val_predictions(
    model: torch.nn.Module,
    val_dataset,
    config: dict,
    device: torch.device,
    sample_count: int = 12,
    fixed: bool = False,
    seed: int = 42,
):
    """Show random/fixed validation samples and print aggregate metrics."""
    predict_fn = _make_predict_fn(model, config["alphabet"], device)
    indices = sample_indices(len(val_dataset), sample_count, fixed, seed)
    rows = collect_predictions(val_dataset, indices, predict_fn)

    items = [
        {
            "image": row["image"],
            "pred": row["pred"],
            "true": row["true"],
            "confidence": row["confidence"],
            "correct": row["correct"],
        }
        for row in rows
    ]
    plot_prediction_grid(items)

    metrics = metrics_from_rows(rows)
    logger.info("Shown samples: %s", metrics.total)
    logger.info("Correct words: %s/%s", metrics.correct, metrics.total)
    logger.info("Accuracy: %.3f | CER: %.3f | WER: %.3f", metrics.accuracy, metrics.cer, metrics.wer)
    logger.info("Average confidence: %.1f%%", metrics.avg_confidence * 100)


def show_hard_val_examples(
    model: torch.nn.Module,
    val_dataset,
    config: dict,
    device: torch.device,
    sample_count: int = 200,
    top_k: int = 12,
    fixed: bool = True,
    seed: int = 42,
):
    """Display the hardest examples by edit distance on a validation subset."""
    predict_fn = _make_predict_fn(model, config["alphabet"], device)
    indices = sample_indices(len(val_dataset), sample_count, fixed, seed)
    rows = collect_predictions(val_dataset, indices, predict_fn)
    rows_sorted = sorted(rows, key=lambda x: (x["dist"], -x["confidence"]), reverse=True)
    hard = rows_sorted[: min(top_k, len(rows_sorted))]

    items = [
        {
            "image": row["image"],
            "pred": row["pred"],
            "true": row["true"],
            "confidence": row["confidence"],
            "correct": row["correct"],
            "extra_line": f"edit_distance={row['dist']}",
        }
        for row in hard
    ]
    plot_prediction_grid(items, row_height=4.2)

    mistakes = sum(1 for row in rows if not row["correct"])
    logger.info("Analyzed samples: %s | mistakes: %s/%s", len(rows), mistakes, len(rows))


def evaluate_val_subset(
    model: torch.nn.Module,
    val_dataset,
    config: dict,
    device: torch.device,
    sample_count: int,
    fixed: bool,
    seed: int,
) -> EvalMetrics:
    """Run inference on a subset and return aggregated quality metrics."""
    predict_fn = _make_predict_fn(model, config["alphabet"], device)
    indices = sample_indices(len(val_dataset), sample_count, fixed, seed)
    rows = collect_predictions(val_dataset, indices, predict_fn)
    return metrics_from_rows(rows)


def run_leaderboard_log(
    model: torch.nn.Module,
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
    """Evaluate subset and append one row to outputs/experiment_log.csv."""
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
    """Display top-k leaderboard rows in notebook or fallback text mode."""
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
