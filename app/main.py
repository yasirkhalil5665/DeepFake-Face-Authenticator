from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import config, inference
from .schemas import HealthResponse, PredictionResponse

app = FastAPI(
    title="DeepFake Face Authenticator",
    description="Classifies an uploaded face image as real or AI-generated (fake).",
    version="1.0.0",
)

# Loosen this to specific origins before going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.on_event("startup")
def _startup():
    # Load the model once when the server boots, not on the first request.
    inference.load_model()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        device=config.DEVICE,
        model_path=str(config.MODEL_PATH),
    )


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
