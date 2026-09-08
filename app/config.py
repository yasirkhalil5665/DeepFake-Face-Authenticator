import os
import torch 
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "models" / "efficientnet_checkpoint.pth"))

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 128))

CLASS_NAMES = ["fake", "real"]

DEVICE = "cpu" if os.getenv("FORCE_CPU", "0") == "1" else ("cuda" if torch.cuda.is_available() else "cpu")

