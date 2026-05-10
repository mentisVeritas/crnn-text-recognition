from pathlib import Path

import torch

from src.utils import project_root_from_checkpoint_path


# ---------- CHECKPOINT HELPERS ----------
def best_val_loss_in_checkpoint_file(path: Path) -> float | None:
    if not path.exists():
        return None

    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("val_loss")


def load_state_dict_payload(path: Path):
    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            return checkpoint["model_state"]

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

    return checkpoint


def load_weights_for_inference(model, checkpoint_path: str | Path):
    checkpoint_path = Path(checkpoint_path)

    state_dict = load_state_dict_payload(checkpoint_path)
    model.load_state_dict(state_dict)

    root = project_root_from_checkpoint_path(checkpoint_path)

    if root is not None:
        config_path = root / "configs" / "config.yaml"

        if config_path.exists():
            return model, config_path

    return model, None