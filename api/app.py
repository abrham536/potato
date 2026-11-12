import os
import sys
import io
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BEST_MODEL_PATH, IMAGE_SIZE, CLASS_NAMES, DISPLAY_LABELS
from src.remedies import DISEASE_INFO

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
    version="1.0.0"
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
        print(f"Loaded model into memory from: {BEST_MODEL_PATH}")
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
        "input_resolution": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} RGB"
    }

@app.post("/predict", tags=["Inference"])
async def predict_disease(file: UploadFile = File(...)):
    """
    Accepts an uploaded potato leaf image and returns the predicted disease category,
    Softmax confidence scores, and specific agronomic treatment recommendations.
    """
    # 1. Validate file content type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Please upload a JPEG or PNG image."
        )

    try:
        # 2. Read image bytes into PIL
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        # 3. Preprocess image tensor: [0, 255] -> [0.0, 1.0]
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)  # (1, 256, 256, 3)

        # 4. Model inference
        model = get_model()
        probabilities = model.predict(img_tensor, verbose=0)[0]  # Array of length 3

        pred_index = int(np.argmax(probabilities))
        pred_class = CLASS_NAMES[pred_index]
        display_name = DISPLAY_LABELS[pred_index]
        confidence = float(probabilities[pred_index])

        # Format probability distribution
        prob_dict = {
            disp: round(float(prob) * 100, 2)
            for disp, prob in zip(DISPLAY_LABELS, probabilities)
        }

        # 5. Fetch agronomic remedies
        remedy_info = DISEASE_INFO.get(pred_class, {})

        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "prediction": {
                "class_code": pred_class,
                "display_name": display_name,
                "confidence_percent": round(confidence * 100, 2),
                "probabilities_percent": prob_dict
            },
            "agronomic_advisory": remedy_info
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)
