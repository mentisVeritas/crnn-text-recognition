import torch


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
        decoded_texts.append(''.join(text))

    return decoded_texts
