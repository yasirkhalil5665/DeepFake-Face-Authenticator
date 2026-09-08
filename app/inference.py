import io
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from model_builder import create_efficientnet_b0, TinyVGG

cnn_model  = TinyVGG

from . import config

_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
])

_model = None  

def load_model() -> torch.nn.Module:
    global _model
    if _model is not None:
        return _model

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {config.MODEL_PATH}. "
            f"Set MODEL_PATH env var or place a .pth file there."
        )

    model = create_efficientnet_b0(output_shape=len(config.CLASS_NAMES), device=config.DEVICE)
    state_dict = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    _model = model
    return _model


def predict_image(image_bytes: bytes) -> dict:
    model = load_model()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(config.DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx = int(torch.argmax(probs).item())
    probabilities = {name: round(float(probs[i]), 4) for i, name in enumerate(config.CLASS_NAMES)}

    return {
        "label": config.CLASS_NAMES[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": probabilities,
    }
