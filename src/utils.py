import logging

import yaml
import torch


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device():
    """Select the best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Device: CUDA - {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logging.info("Device: MPS (Apple Metal Performance Shaders)")
    else:
        device = torch.device('cpu')
        logging.info("Device: CPU")
    return device

