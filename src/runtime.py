import logging
from pathlib import Path


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
