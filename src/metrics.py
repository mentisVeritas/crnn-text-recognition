from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(a)][len(b)]


@dataclass
class EvalMetrics:
    accuracy: float
    cer: float
    wer: float
    avg_confidence: float
    correct: int
    total: int


def compute_metrics(pairs: Sequence[Tuple[str, str, float]]) -> EvalMetrics:
    """
    pairs: sequence of (pred_text, true_text, confidence)
    """
    total = len(pairs)
    correct = sum(1 for pred, true, _ in pairs if pred == true)
    accuracy = (correct / total) if total else 0.0

    cer_num = 0
    cer_den = 0
    for pred, true, _ in pairs:
        cer_num += levenshtein(pred, true)
        cer_den += max(1, len(true))
    cer = (cer_num / cer_den) if cer_den else 0.0

    wer_num = sum(1 for pred, true, _ in pairs if pred != true)
    wer = (wer_num / total) if total else 0.0

    avg_conf = (sum(conf for _, _, conf in pairs) / total) if total else 0.0

    return EvalMetrics(
        accuracy=accuracy,
        cer=cer,
        wer=wer,
        avg_confidence=avg_conf,
        correct=correct,
        total=total,
    )
