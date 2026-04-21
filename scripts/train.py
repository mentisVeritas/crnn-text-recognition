import os
import sys
import torch
import logging

# Add parent directory to path so we can import src BEFORE importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.train import run_training
from src.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Get the project root directory (parent of scripts/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs/config.yaml")
    config = load_config(config_path)

    model = run_training(config, project_root=project_root)

    # Save final model
    checkpoints_dir = os.path.join(project_root, "outputs/checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    final_model_path = os.path.join(checkpoints_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")


if __name__ == "__main__":
    main()
