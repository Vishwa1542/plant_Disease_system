"""
disease_info.py — Load and query the pesticide/disease knowledge base.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

_disease_db = None


def load_disease_info(json_path: str) -> dict:
    """Load disease JSON knowledge base into memory (singleton)."""
    global _disease_db
    if _disease_db is None:
        if not os.path.exists(json_path):
            logger.error(f"Disease info file not found: {json_path}")
            _disease_db = {}
        else:
            with open(json_path, "r") as f:
                _disease_db = json.load(f)
            logger.info(f"Disease database loaded: {len(_disease_db)} entries.")
    return _disease_db


def get_disease_solution(class_name: str, db: dict) -> dict:
    """
    Match predicted class name to disease database.
    Returns full disease info dict or a fallback if not found.
    """
    if class_name in db:
        return db[class_name]

    # Try case-insensitive match as fallback
    for key, value in db.items():
        if key.lower() == class_name.lower():
            return value

    # Not found — return generic response
    logger.warning(f"Disease '{class_name}' not found in database.")
    return {
        "disease_name": class_name.replace("_", " "),
        "plant": "Unknown",
        "symptoms": "No information available for this disease.",
        "pesticide": "Please consult a local agricultural expert.",
        "precautions": ["Isolate affected plant", "Consult an agronomist"],
        "severity": "Unknown",
    }
