import logging

import torch

logger = logging.getLogger(__name__)


def encode_text(text: object, char2idx: dict[str, int]) -> list[int]:
    """Encode text into CTC labels (blank=0, unknown char -> 1)."""
    if not isinstance(text, str):
        logger.warning("Non-string text encountered: %r (type=%s), using space", text, type(text))
        text = " "

    encoded = [char2idx.get(ch, 1) for ch in text]
    return encoded if encoded else [1]


def greedy_decode(logits: torch.Tensor, alphabet: str) -> list[str]:
    """Greedy CTC decode for logits shaped [T, B, C]."""
    preds = torch.argmax(logits, dim=2)  # [T, B]
    decoded_texts = []

    for b in range(preds.size(1)):
        pred = preds[:, b]
        text = []
        prev = -1
        for p in pred:
            p = p.item()
            if p != 0 and p != prev:  # skip blank and duplicates
                text.append(alphabet[p - 1])
            prev = p
        decoded_texts.append("".join(text))

    return decoded_texts
