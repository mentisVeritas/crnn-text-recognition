"""
FastAPI backend for CRNN Text Recognition
Accepts image uploads and returns predicted text
"""
import io
import tempfile
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from src.model import CRNN
from src.inference import decode_with_confidence
from src.data import build_image_transform
from src.utils import load_config, get_device

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    try:
        load_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

    yield


# Initialize FastAPI app
app = FastAPI(
    title="CRNN Text Recognition",
    description="OCR using CRNN model",
    version="1.0.0",
    lifespan=lifespan
)

# Serve static files (frontend)
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Global variables for model
model = None
config = None
device = None


def load_model():
    """Load model and config on startup."""
    global model, config, device

    # Get project root
    project_root = Path(__file__).parent.resolve()

    # Load config
    config_path = project_root / "configs" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    config = load_config(config_path)
    device = get_device()

    # Load model checkpoint
    checkpoint_path = project_root / "outputs" / "checkpoints" / "best_model.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    # Initialize model
    num_classes = len(config["alphabet"]) + 1  # +1 for CTC blank
    model = CRNN(num_classes=num_classes).to(device)

    # Load weights
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # Handle checkpoint format (extract model_state if it's wrapped)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print(f"✅ Model loaded from {checkpoint_path}")
    print(f"   Device: {device}")
    print(f"   Image size: {config['img_height']}x{config['img_width']}")
    print(f"   Alphabet: {config['alphabet']}")


@app.get("/")
async def root():
    """Serve homepage."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device) if device else None,
        "img_height": config["img_height"] if config else None,
        "img_width": config["img_width"] if config else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict text from uploaded image.

    Args:
        file: Image file (PNG, JPG, etc.)

    Returns:
        JSON with predicted text and confidence
    """
    if model is None or config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read image from upload
        contents = await file.read()

        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Save temporary file for preprocessing
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            image.save(temp_path)

        if temp_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Empty image file")

        # Preprocess
        transform = build_image_transform(config["img_height"])
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        pred_text, confidence = decode_with_confidence(
            model=model,
            image_tensor=image_tensor,
            alphabet=config["alphabet"],
            device=device
        )

        # Cleanup
        temp_path.unlink(missing_ok=True)

        return {
            "text": pred_text,
            "confidence": round(confidence * 100, 2),
            "success": True,
        }

    except HTTPException:
        if 'temp_path' in locals():
            temp_path.unlink(missing_ok=True)
        raise

    except Exception as e:
        if 'temp_path' in locals():
            temp_path.unlink(missing_ok=True)

        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print("Open: http://127.0.0.1:8000")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
