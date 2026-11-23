import os
import sys
import io
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure runtime directories are configured before TF imports
os.environ["KERAS_HOME"] = str(PROJECT_ROOT / ".keras")
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib")

from src.config import (
    BEST_MODEL_PATH,
    IMAGE_SIZE,
    CLASS_NAMES,
    DISPLAY_LABELS,
    MIN_PREDICTION_CONFIDENCE
)
from src.remedies import DISEASE_INFO
from src.potato_verifier import verify_potato_leaf, check_leaf_presence

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI(
    title="Potato Leaf Disease Identification & Management API",
    description="End-to-End Deep Learning REST API for Potato Leaf Disease Diagnosis and Agronomic Treatment Guidance. "
                "Developed for Debre Berhan University, College of Computing, Department of Computer Science.",
    version="1.1.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        if not BEST_MODEL_PATH.exists():
            raise RuntimeError(f"Trained model not found at {BEST_MODEL_PATH}. Train model first.")
        _MODEL = tf.keras.models.load_model(str(BEST_MODEL_PATH))
        print(f"Loaded classifier model into memory from: {BEST_MODEL_PATH}")
    return _MODEL

@app.get("/", tags=["General"])
def root():
    return {
        "project": "Smart Agro-Vision: Potato Leaf Disease Identification",
        "institution": "Debre Berhan University - College of Computing",
        "department": "Department of Computer Science",
        "course": "Selected Topics in Computer Science",
        "status": "Online",
        "api_documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    }

@app.get("/health", tags=["Monitoring"])
def health_check():
    model_ready = BEST_MODEL_PATH.exists()
    return {
        "status": "healthy" if model_ready else "degraded",
        "model_file_exists": model_ready,
        "supported_classes": DISPLAY_LABELS,
        "input_resolution": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} RGB",
        "target_crop": "Potato (Solanum tuberosum)"
    }

def process_prediction(raw_image: Image.Image, filename: str = "image.jpg") -> dict:
    """
    Core prediction pipeline:
    1. Runs multi-tier potato leaf and biological foliage verification.
    2. Rejects non-leaf and non-potato images with structured diagnostic metrics.
    3. If verified as potato leaf, executes deep CNN disease diagnosis and returns agronomic advisory.
    """
    # 1. Multi-tier Potato Leaf & Foliage Verification
    verification = verify_potato_leaf(raw_image)
    if not verification.is_valid:
        return {
            "status": "rejected",
            "is_potato_leaf": False,
            "is_plant_leaf": verification.is_leaf,
            "rejection_category": verification.status,
            "message": verification.reason,
            "confidence_percent": 0.0,
            "prediction": None,
            "diagnostic_metrics": {
                "foliage_coverage_percent": round(verification.foliage_ratio * 100, 2),
                "excess_green_index": round(verification.exg_mean, 2),
                "organic_hue_variance": round(verification.hue_std, 2),
                "reconstruction_error": round(verification.reconstruction_mse, 5)
            },
            "agronomic_advisory": None
        }

    # 2. Preprocess verified potato leaf tensor
    image = raw_image.resize(IMAGE_SIZE)
    img_array = np.array(image, dtype=np.float32)
    img_tensor = np.expand_dims(img_array, axis=0)  # (1, 256, 256, 3)

    # 3. Diagnostic Deep Learning Inference
    model = get_model()
    probabilities = model(img_tensor, training=False).numpy()[0]  # Array of length 3

    pred_index = int(np.argmax(probabilities))
    pred_class = CLASS_NAMES[pred_index]
    display_name = DISPLAY_LABELS[pred_index]
    confidence = float(probabilities[pred_index])

    # Format probability distribution
    prob_dict = {
        disp: round(float(prob) * 100, 2)
        for disp, prob in zip(DISPLAY_LABELS, probabilities)
    }

    # Check for ambiguous / low-confidence diagnosis
    is_confident = (confidence * 100.0) >= MIN_PREDICTION_CONFIDENCE

    # 4. Fetch agronomic remedies
    remedy_info = DISEASE_INFO.get(pred_class, {})

    return {
        "status": "success",
        "is_potato_leaf": True,
        "filename": filename,
        "prediction": {
            "class_code": pred_class,
            "display_name": display_name,
            "confidence_percent": round(confidence * 100, 2),
            "is_reliable": is_confident,
            "warning": None if is_confident else f"Prediction confidence ({confidence * 100:.1f}%) is below reliable threshold ({MIN_PREDICTION_CONFIDENCE}%).",
            "probabilities_percent": prob_dict
        },
        "agronomic_advisory": remedy_info
    }

@app.post("/predict", tags=["Inference"])
async def predict_disease(file: UploadFile = File(...)):
    """
    Accepts an uploaded image and:
    1. Validates whether the image contains legitimate potato plant foliage (Solanum tuberosum).
    2. Rejects non-leaf images (faces, vehicles, objects, backgrounds) and non-potato leaves.
    3. If verified as potato leaf, returns predicted disease condition, Softmax confidence scores,
       and tailored agronomic treatment recommendations.
    """
    # 1. Flexible validation of file content type / filename extension
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp", "application/octet-stream"]
    valid_ext = any(file.filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]) if file.filename else False
    
    if file.content_type not in allowed_types and not valid_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format ({file.content_type}). Please upload a JPEG or PNG image."
        )

    try:
        # 2. Read image bytes into PIL
        contents = await file.read()
        try:
            raw_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            return JSONResponse(status_code=400, content={
                "status": "error",
                "error": "Corrupted or unreadable image file. Please provide a valid photograph in JPG or PNG format."
            })
        
        # 3. Execute prediction pipeline
        result = process_prediction(raw_image, file.filename or "uploaded_leaf.jpg")
        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)
