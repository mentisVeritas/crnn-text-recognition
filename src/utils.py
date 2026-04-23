import logging
from pathlib import Path

import torch
import yaml


def load_config(config_path: str | Path) -> dict:
    """Load YAML config file into a dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_device() -> torch.device:
    """Select the best available device in priority order: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Device: CUDA - %s", torch.cuda.get_device_name(0))
        logging.info("GPU Memory: %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Device: MPS (Apple Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logging.info("Device: CPU")
    return device


def project_root_from_checkpoint_path(checkpoint_path: Path) -> Path | None:
    """
    Expect layout ``<project>/outputs/checkpoints/<file>`` and return ``<project>``.
    """
    try:
        root = Path(checkpoint_path).expanduser().resolve().parent.parent.parent
        if (root / "src").is_dir():
            return root
    except (OSError, ValueError):
        pass
    return None


def setup_outputs_file_logging(project_root: Path | str) -> Path | None:
    """
    Append INFO (and above) from the root logger to ``outputs/logs/train.log`` under project_root.

    Safe to call more than once: skips if a FileHandler for that path already exists.
    """
    root = Path(project_root).resolve()
    log_dir = root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"
    root_logger = logging.getLogger()
    try:
        target = log_file.resolve()
    except OSError:
        target = log_file

    # Avoid duplicate file handlers across repeated notebook/script calls.
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None):
            try:
                if Path(handler.baseFilename).resolve() == target:
                    return log_file
            except OSError:
                pass

    try:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    except OSError as e:
        logging.getLogger(__name__).warning("Could not open %s: %s", log_file, e)
        return None

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root_logger.addHandler(file_handler)
    if root_logger.level == logging.NOTSET:
        root_logger.setLevel(logging.INFO)
    logging.getLogger(__name__).info("File logging (append) -> %s", log_file)
    return log_file

