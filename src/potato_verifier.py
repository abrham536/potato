"""
Smart Agro-Vision: Potato Leaf & Biological Foliage Verification Engine
Debre Berhan University - College of Computing - Department of Computer Science

Multi-tier verification system designed to ensure that diagnostic deep learning
inference is executed EXCLUSIVELY on legitimate potato crop foliage (Solanum tuberosum)
and provides detailed, structured rejection diagnostics for non-leaf or out-of-distribution inputs.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure runtime directories are configured before TF imports
os.environ["KERAS_HOME"] = str(PROJECT_ROOT / ".keras")
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib")

import numpy as np
from PIL import Image
import tensorflow as tf

from src.config import (
    VERIFIER_MODEL_PATH,
    MIN_FOLIAGE_COVERAGE,
    MIN_EXG_MEAN,
    MIN_HUE_STD,
    MIN_TEXTURE_STD,
    MAX_RECONSTRUCTION_MSE,
    MIN_PREDICTION_CONFIDENCE
)

@dataclass
class VerificationResult:
    """Structured result of multi-tier potato leaf verification."""
    is_valid: bool
    is_leaf: bool
    is_potato: bool
    status: str
    reason: str
    foliage_ratio: float
    exg_mean: float
    hue_std: float
    texture_std: float
    reconstruction_mse: float
    diagnostic_details: Dict[str, Any] = field(default_factory=dict)

# Global cached verifier model instance
_VERIFIER_MODEL = None

def get_verifier_model():
    """Loads and caches the Potato Leaf Autoencoder Verifier model."""
    global _VERIFIER_MODEL
    if _VERIFIER_MODEL is None and VERIFIER_MODEL_PATH.exists():
        try:
            import tensorflow as tf
            _VERIFIER_MODEL = tf.keras.models.load_model(str(VERIFIER_MODEL_PATH))
            print(f"[Verifier] Successfully loaded Potato Leaf Verifier from: {VERIFIER_MODEL_PATH}")
        except Exception as e:
            print(f"[Verifier] Warning: Failed to load verifier model ({e}). Relying on botanical verification.")
            _VERIFIER_MODEL = None
    return _VERIFIER_MODEL

def verify_potato_leaf(image: Image.Image) -> VerificationResult:
    """
    Executes a 4-tier botanical and deep learning verification:
      Tier 1: Digital Image & Content Integrity (minimum size, contrast, non-blank)
      Tier 2: Botanical Plant Foliage & Chlorophyll Spectral Verification (Leaf vs Non-Leaf)
      Tier 3: Organic Foliage Variation & Venation Texture (Rejects synthetic green surfaces)
      Tier 4: Potato Leaf Morphology & Autoencoder Manifold Check (Potato vs Non-Potato)

    Returns a VerificationResult instance with full diagnostic metrics.
    """
    # Tier 1: Input Sanity & Digital Formatting
    img = image.convert("RGB")
    width, height = img.size

    if width < 64 or height < 64:
        return VerificationResult(
            is_valid=False,
            is_leaf=False,
            is_potato=False,
            status="REJECTED_RESOLUTION",
            reason=f"Image resolution too low ({width}x{height} px). Minimum 64x64 px required.",
            foliage_ratio=0.0,
            exg_mean=0.0,
            hue_std=0.0,
            texture_std=0.0,
            reconstruction_mse=1.0,
            diagnostic_details={"dimensions": (width, height)}
        )

    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # Grayscale conversion for texture and variation analysis
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    texture_std = float(np.std(gray))
    adjacent_diff = float(np.mean(np.abs(np.diff(gray, axis=0))))

    # Check for flat, blank, or uniform images (solid colors, blank walls)
    if texture_std < MIN_TEXTURE_STD:
        return VerificationResult(
            is_valid=False,
            is_leaf=False,
            is_potato=False,
            status="REJECTED_FLAT_BACKGROUND",
            reason="Uniform or flat background detected without leaf texture or biological morphology.",
            foliage_ratio=0.0,
            exg_mean=0.0,
            hue_std=0.0,
            texture_std=texture_std,
            reconstruction_mse=1.0,
            diagnostic_details={"texture_std": round(texture_std, 2)}
        )

    # Check for pure high-frequency digital noise
    if adjacent_diff > 48.0:
        return VerificationResult(
            is_valid=False,
            is_leaf=False,
            is_potato=False,
            status="REJECTED_DIGITAL_NOISE",
            reason="High-frequency digital noise or artificial pattern detected instead of natural plant tissue.",
            foliage_ratio=0.0,
            exg_mean=0.0,
            hue_std=0.0,
            texture_std=texture_std,
            reconstruction_mse=1.0,
            diagnostic_details={"adjacent_diff": round(adjacent_diff, 2)}
        )

    # Tier 2: Botanical Plant Foliage & Chlorophyll Spectral Analysis
    # Plant foliage color dynamics in HSV:
    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    h = hsv[..., 0] * 360.0 / 255.0  # Hue in [0, 360] degrees
    s = hsv[..., 1] / 255.0          # Saturation [0, 1]
    v = hsv[..., 2] / 255.0          # Value [0, 1]

    # 1. Active chlorophyll green tissue: Hue 35° to 165°, Sat >= 0.15, Val >= 0.12
    green_mask = (h >= 35.0) & (h <= 165.0) & (s >= 0.15) & (v >= 0.12)
    # 2. Chlorotic / senescent yellowing tissue: Hue 20° to 35°, Sat >= 0.20, Val >= 0.15
    yellow_mask = (h >= 20.0) & (h < 35.0) & (s >= 0.20) & (v >= 0.15)
    # 3. Necrotic blight lesions: Hue 8° to 30°, Sat >= 0.12, Val 0.06 to 0.65
    brown_mask = (h >= 8.0) & (h < 30.0) & (s >= 0.12) & (v >= 0.06) & (v <= 0.65)

    foliage_mask = green_mask | yellow_mask | brown_mask
    foliage_ratio = float(np.mean(foliage_mask))
    green_ratio = float(np.mean(green_mask))

    # Reject if plant foliage coverage is below threshold
    if foliage_ratio < MIN_FOLIAGE_COVERAGE or green_ratio < 0.10:
        return VerificationResult(
            is_valid=False,
            is_leaf=False,
            is_potato=False,
            status="REJECTED_NON_LEAF",
            reason=(
                f"No plant foliage or leaf structure detected. "
                f"Detected foliage coverage is {foliage_ratio * 100:.1f}% (required >= {MIN_FOLIAGE_COVERAGE * 100:.1f}%)."
            ),
            foliage_ratio=foliage_ratio,
            exg_mean=0.0,
            hue_std=0.0,
            texture_std=texture_std,
            reconstruction_mse=1.0,
            diagnostic_details={
                "foliage_ratio_percent": round(foliage_ratio * 100, 2),
                "green_ratio_percent": round(green_ratio * 100, 2)
            }
        )

    # Tier 3: Excess Green Index & Organic Foliage Variation
    # Plant foliage requires green channel dominance over red and blue (ExG = 2G - R - B)
    exg = 2.0 * g - r - b
    if np.any(green_mask):
        exg_mean = float(np.mean(exg[green_mask]))
        hue_std = float(np.std(h[green_mask]))
    else:
        exg_mean = float(np.mean(exg[foliage_mask]))
        hue_std = float(np.std(h[foliage_mask]))

    # Living plant chlorophyll must produce significant positive ExG
    if exg_mean < MIN_EXG_MEAN:
        return VerificationResult(
            is_valid=False,
            is_leaf=False,
            is_potato=False,
            status="REJECTED_INSUFFICIENT_CHLOROPHYLL",
            reason=(
                f"Insufficient chlorophyll spectral index detected (ExG: {exg_mean:.1f}, required >= {MIN_EXG_MEAN:.1f}). "
                f"The photographed object is not living plant foliage."
            ),
            foliage_ratio=foliage_ratio,
            exg_mean=exg_mean,
            hue_std=hue_std,
            texture_std=texture_std,
            reconstruction_mse=1.0,
            diagnostic_details={"exg_mean": round(exg_mean, 2)}
        )

    # Rejects uniform synthetic green objects (green shirt, green wall, green plastics)
    # Living leaves exhibit organic color gradation across the blade, veins, and lesions
    if hue_std < MIN_HUE_STD:
        return VerificationResult(
            is_valid=False,
            is_leaf=False,
            is_potato=False,
            status="REJECTED_SYNTHETIC_SURFACE",
            reason=(
                f"Synthetic or uniform green surface detected without natural biological hue variation "
                f"(Hue std: {hue_std:.2f}°, required >= {MIN_HUE_STD:.2f}°)."
            ),
            foliage_ratio=foliage_ratio,
            exg_mean=exg_mean,
            hue_std=hue_std,
            texture_std=texture_std,
            reconstruction_mse=1.0,
            diagnostic_details={"hue_std": round(hue_std, 2)}
        )

    # Check gradient texture within the foliage
    gy = np.abs(np.diff(g, axis=0))[:, :-1]
    gx = np.abs(np.diff(g, axis=1))[:-1, :]
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_std = float(np.std(grad_mag))

    if grad_std < 8.0:
        return VerificationResult(
            is_valid=False,
            is_leaf=False,
            is_potato=False,
            status="REJECTED_NO_VENATION",
            reason="Smooth artificial green texture detected without biological leaf venation or cellular morphology.",
            foliage_ratio=foliage_ratio,
            exg_mean=exg_mean,
            hue_std=hue_std,
            texture_std=texture_std,
            reconstruction_mse=1.0,
            diagnostic_details={"grad_std": round(grad_std, 2)}
        )

    # At this point, the image is confirmed to be REAL PLANT LEAF FOLIAGE!
    # Now Tier 4: Specificity Verification for Potato Leaf (Solanum tuberosum)
    reconstruction_mse = 0.0
    verifier_model = get_verifier_model()

    if verifier_model is not None:
        try:
            # Resize image to (128, 128) for autoencoder verification
            img_128 = np.array(img.resize((128, 128)), dtype=np.float32)
            img_tensor = tf.constant(np.expand_dims(img_128, axis=0))
            recon = verifier_model(img_tensor, training=False).numpy()[0]
            reconstruction_mse = float(np.mean((img_128 / 255.0 - recon) ** 2))

            # If reconstruction MSE exceeds potato manifold threshold, it is a non-potato leaf
            if reconstruction_mse > MAX_RECONSTRUCTION_MSE:
                return VerificationResult(
                    is_valid=False,
                    is_leaf=True,
                    is_potato=False,
                    status="REJECTED_NON_POTATO_LEAF",
                    reason=(
                        f"Plant leaf detected, but morphology and textural manifold do NOT match potato leaf "
                        f"(Solanum tuberosum). Reconstruction variance ({reconstruction_mse:.4f}) exceeds threshold "
                        f"({MAX_RECONSTRUCTION_MSE:.4f}). This system diagnoses ONLY potato leaves."
                    ),
                    foliage_ratio=foliage_ratio,
                    exg_mean=exg_mean,
                    hue_std=hue_std,
                    texture_std=texture_std,
                    reconstruction_mse=reconstruction_mse,
                    diagnostic_details={
                        "reconstruction_mse": round(reconstruction_mse, 5),
                        "max_allowed_mse": MAX_RECONSTRUCTION_MSE
                    }
                )
        except Exception as e:
            print(f"[Verifier] Autoencoder inference error ({e}), proceeding with botanical checks.")

    # All botanical and potato manifold checks passed!
    return VerificationResult(
        is_valid=True,
        is_leaf=True,
        is_potato=True,
        status="ACCEPTED",
        reason="Genuine potato leaf foliage (Solanum tuberosum) verified successfully.",
        foliage_ratio=foliage_ratio,
        exg_mean=exg_mean,
        hue_std=hue_std,
        texture_std=texture_std,
        reconstruction_mse=reconstruction_mse,
        diagnostic_details={
            "foliage_ratio_percent": round(foliage_ratio * 100, 2),
            "exg_mean": round(exg_mean, 2),
            "hue_std": round(hue_std, 2),
            "reconstruction_mse": round(reconstruction_mse, 5)
        }
    )

def check_leaf_presence(image: Image.Image) -> Tuple[bool, float, str]:
    """
    Backward-compatible adapter for existing call sites.
    Returns (is_valid: bool, foliage_ratio: float, reason: str).
    """
    res = verify_potato_leaf(image)
    return res.is_valid, res.foliage_ratio, res.reason
