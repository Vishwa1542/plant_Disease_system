"""
model_loader.py — Singleton model loader
Loads model and class indices once, reuses across requests.
"""

import json
import logging
import numpy as np
import tensorflow as tf
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Singleton — model loaded once at startup
_model = None
_class_indices = None

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70  # Below this → "try another image"


def load_model(model_path: str, class_indices_path: str):
    """Load CNN model and class index mapping into memory."""
    global _model, _class_indices

    if _model is None:
        logger.info(f"Loading model from {model_path} ...")
        _model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully.")

    if _class_indices is None:
        with open(class_indices_path, "r") as f:
            # Keys are string indices from JSON; convert to int
            raw = json.load(f)
            _class_indices = {int(k): v for k, v in raw.items()}
        logger.info(f"Class indices loaded: {len(_class_indices)} classes.")

    return _model, _class_indices


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess raw image bytes for model inference.
    Steps:
      1. Open image from bytes
      2. Convert to RGB (handles RGBA, grayscale, etc.)
      3. Resize to 224×224
      4. Normalize to [0, 1]
      5. Expand dims → (1, 224, 224, 3)
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")                        # Ensure 3-channel RGB
    img = img.resize((IMG_SIZE, IMG_SIZE))          # Resize to model input
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0                   # Normalize pixels
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def predict(model, class_indices: dict, image_array: np.ndarray) -> dict:
    """
    Run inference and return prediction result.
    Returns:
        {
          "class_name": str,
          "confidence": float,
          "low_confidence": bool,
          "all_probs": dict  # top-5 predictions
        }
    """
    predictions = model.predict(image_array, verbose=0)  # Shape: (1, num_classes)
    probs = predictions[0]                                # Flatten batch dim

    # Top predicted class
    top_idx = int(np.argmax(probs))
    top_confidence = float(probs[top_idx])
    top_class = class_indices.get(top_idx, "Unknown")

    # Top-5 predictions for transparency
    top5_indices = np.argsort(probs)[::-1][:5]
    top5 = {
        class_indices.get(int(i), f"class_{i}"): round(float(probs[i]) * 100, 2)
        for i in top5_indices
    }

    return {
        "class_name": top_class,
        "confidence": round(top_confidence * 100, 2),  # As percentage
        "low_confidence": top_confidence < CONFIDENCE_THRESHOLD,
        "top5": top5,
    }
