from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import torch

from src.metrics import EvalMetrics, compute_metrics, levenshtein


def collect_predictions(
    dataset,
    indices: Sequence[int],
    predict_fn: Callable[[torch.Tensor], tuple[str, float]],
):
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
):
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
    if not log_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(log_path)
    if df.empty:
        return df
    return df.sort_values(by=["accuracy", "cer", "wer"], ascending=[False, True, True]).reset_index(drop=True)
