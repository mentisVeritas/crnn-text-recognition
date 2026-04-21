import torch
import logging
import argparse
import os
import sys

# Add parent directory to path so we can import src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import CRNN
from src.inference import preprocess_image, decode_with_confidence
from src.utils import load_config

logger = logging.getLogger(__name__)


def load_model(checkpoint_path, num_classes, device):
    model = CRNN(num_classes=num_classes)
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(model, image_tensor, alphabet, device):
    pred_text, conf = decode_with_confidence(model, image_tensor, alphabet, device)
    return pred_text, conf


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Predict text from image")
    parser.add_argument('--image', type=str, help='Path to image file')
    args = parser.parse_args()

    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs/config.yaml")
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(config["alphabet"]) + 1
    checkpoint_path = os.path.join(project_root, "outputs/checkpoints/best_model.pth")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Model checkpoint not found at {checkpoint_path}. Please train the model first.")
        return

    model = load_model(checkpoint_path, num_classes, device)

    if args.image:
        if not os.path.exists(args.image):
            logger.error("Image file not found.")
            return
        image_tensor = preprocess_image(args.image, config["img_height"], config["img_width"])
        predicted_text, conf = predict(model, image_tensor, config["alphabet"], device)
        logger.info(f"Predicted text: {predicted_text} (confidence={conf * 100:.1f}%)")
    else:
        # Example prediction on a few images
        images_dir = os.path.join(project_root, config["images_dir"])
        if not os.path.isdir(images_dir):
            logger.error(
                "Images directory not found: %s. Download the dataset (see README) "
                "or pass --image /path/to.png",
                images_dir,
            )
            return
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')][:5]
        if not image_files:
            logger.error("No .png files in %s", images_dir)
            return
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            image_tensor = preprocess_image(img_path, config["img_height"], config["img_width"])
            predicted_text, conf = predict(model, image_tensor, config["alphabet"], device)
            logger.info(f"{img_file}: {predicted_text} (confidence={conf * 100:.1f}%)")


if __name__ == "__main__":
    main()
