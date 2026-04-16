"""
main.py — Plant Disease Prediction API
FastAPI backend with /predict endpoint.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from utils.model_loader import load_model, preprocess_image, predict
from utils.disease_info import load_disease_info, get_disease_solution

# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# PATHS (adjust if needed)
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "model.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "..", "model", "class_indices.json")
DISEASE_JSON_PATH = os.path.join(BASE_DIR, "..", "data", "disease_info.json")

# Globals (populated at startup)
model = None
class_indices = None
disease_db = None


# ─────────────────────────────────────────
# LIFESPAN — load model once at startup
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_indices, disease_db

    logger.info("Starting up — loading model and disease database...")
    model, class_indices = load_model(MODEL_PATH, CLASS_INDICES_PATH)
    disease_db = load_disease_info(DISEASE_JSON_PATH)
    logger.info("Startup complete. API is ready.")

    yield  # App runs here

    logger.info("Shutting down...")


# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────
app = FastAPI(
    title="Plant Disease Prediction API",
    description="Upload a plant leaf image to detect diseases and get pesticide recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS ────────────────────────────────
# Allow React frontend (all origins in dev; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Plant Disease API is running"}


@app.get("/health")
def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_loaded": class_indices is not None,
        "disease_db_loaded": disease_db is not None,
        "num_classes": len(class_indices) if class_indices else 0,
    }


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict plant disease from uploaded image.

    Request:
        - file: image (JPG, PNG, WEBP)

    Response:
        {
          "disease": "Tomato Early Blight",
          "confidence": 94.5,
          "low_confidence": false,
          "solution": {
            "disease_name": "...",
            "plant": "...",
            "symptoms": "...",
            "pesticide": "...",
            "precautions": [...],
            "severity": "..."
          },
          "top5": { ... }
        }
    """
    # ── Validate file type ────────────────────────────────────────────
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Upload JPG or PNG.",
        )

    # ── Read image bytes ──────────────────────────────────────────────
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=400, detail="Could not read uploaded file.")

    # ── Preprocess ────────────────────────────────────────────────────
    try:
        image_array = preprocess_image(image_bytes)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise HTTPException(
            status_code=422,
            detail="Could not process image. Ensure it is a valid plant photo.",
        )

    # ── Predict ───────────────────────────────────────────────────────
    try:
        result = predict(model, class_indices, image_array)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed.")

    # ── Low confidence check ──────────────────────────────────────────
    if result["low_confidence"]:
        logger.info(
            f"Low confidence ({result['confidence']}%) for class: {result['class_name']}"
        )
        return JSONResponse(
            status_code=200,
            content={
                "low_confidence": True,
                "confidence": result["confidence"],
                "message": "Confidence too low. Please try a clearer image with better lighting.",
                "top5": result["top5"],
            },
        )

    # ── Get disease solution ──────────────────────────────────────────
    solution = get_disease_solution(result["class_name"], disease_db)

    logger.info(
        f"Prediction: {result['class_name']} | Confidence: {result['confidence']}%"
    )

    return {
        "disease": solution.get("disease_name", result["class_name"]),
        "confidence": result["confidence"],
        "low_confidence": False,
        "solution": solution,
        "top5": result["top5"],
    }


# ─────────────────────────────────────────
# EXCEPTION HANDLERS
# ─────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )
