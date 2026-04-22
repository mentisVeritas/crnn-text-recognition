import logging

import torch

logger = logging.getLogger(__name__)


def encode_text(text, char2idx):
    """
    Encode text to CTC target indices.
    0 is reserved for CTC blank, unknown symbols map to 1.
    """
    if not isinstance(text, str):
        logger.warning("Non-string text encountered: %r (type=%s), using space", text, type(text))
        text = " "

    encoded = []
    for c in text:
        encoded.append(char2idx.get(c, 1))
    return encoded if encoded else [1]


def greedy_decode(logits, alphabet):
    """
    Greedy decoding for CTC output.
    logits: [T, B, C] where C = len(alphabet) + 1 (blank)
    """
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
