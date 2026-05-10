import os
import sys
import logging

# Parent of scripts/ must be on path before importing ``src``.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.train import run_training
from src.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Load config and run full training pipeline."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs/config.yaml")
    config = load_config(config_path)

    run_training(config, project_root=project_root)
    logger.info("Done. Weights: outputs/checkpoints/checkpoint.pth (last epoch) and best_model.pth (best val).")


if __name__ == "__main__":
    main()
