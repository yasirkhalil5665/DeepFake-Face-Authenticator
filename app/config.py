"""Central config for the serving app. Kept separate from src/ (training) config."""
import os
from pathlib import Path

# Project root = parent of the app/ folder
ROOT_DIR = Path(__file__).resolve().parent.parent

# Which checkpoint to serve. Override with: MODEL_PATH=models/cnn_model_1.pth
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "models" / "efficientnet_model.pth"))

# Must match train.py's default --image-size (128) unless you retrained with a different value.
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 128))

# From the notebook: train_dataset.classes -> ['fake', 'real'] (ImageFolder sorts alphabetically).
CLASS_NAMES = ["fake", "real"]

import torch  # noqa: E402  (kept below the constants above for readability)
DEVICE = "cpu" if os.getenv("FORCE_CPU", "0") == "1" else ("cuda" if torch.cuda.is_available() else "cpu")
