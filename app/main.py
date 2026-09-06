from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from . import config, inference
from .schemas import PredictionResponse

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="DeepFake Face Authenticator",
    description="Classifies an uploaded face image as real or AI-generated (fake).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC_DIR / "index.html")

@asynccontextmanager
async def lifespan(app: FastAPI):
    inference.load_model()  
    yield                    


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'. Use JPEG, PNG, or WEBP.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        result = inference.predict_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not process image: {e}")

    return PredictionResponse(**result)
