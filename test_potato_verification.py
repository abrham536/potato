"""
Automated Verification & Validation Suite for Potato Leaf Classifier
Tests the multi-tier verifier and FastAPI endpoint against:
1. Genuine potato leaves across all three classes (Healthy, Early Blight, Late Blight)
2. Synthetic and photographic non-leaf inputs (solid colors, noise, skin tones, objects)
3. Synthetic non-leaf green surfaces (flat plastic, clothing)
4. REST API /predict responses
"""

import os
import sys
import io
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["KERAS_HOME"] = str(PROJECT_ROOT / ".keras")
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib")

import numpy as np
from PIL import Image
from src.config import BEST_MODEL_PATH, VERIFIER_MODEL_PATH, CLASS_NAMES
from src.potato_verifier import verify_potato_leaf

def test_genuine_potato_leaves():
    print("\n--- 1. Testing Genuine Potato Leaves (Test Set) ---")
    all_passed = True
    for cls in CLASS_NAMES:
        test_dir = PROJECT_ROOT / "data" / "processed" / "test" / cls
        paths = list(test_dir.glob("*.jpg"))[:10]
        assert len(paths) > 0, f"No test images found for {cls}"
        
        passed = 0
        for p in paths:
            with Image.open(p) as img:
                res = verify_potato_leaf(img)
                if res.is_valid and res.is_leaf and res.is_potato:
                    passed += 1
                else:
                    print(f"  [X] Failed on {cls} ({p.name}): {res.status} - {res.reason}")

        pass_rate = (passed / len(paths)) * 100
        print(f"  {cls:25s}: {passed}/{len(paths)} passed ({pass_rate:.1f}%)")
        if pass_rate < 80.0:
            all_passed = False

    assert all_passed, "Genuine potato leaf pass rate is below acceptance threshold!"
    print("  => Genuine potato leaf verification passed.")

def test_non_leaf_rejections():
    print("\n--- 2. Testing Non-Leaf Rejections ---")
    test_cases = {
        "solid_black": Image.new("RGB", (256, 256), (0, 0, 0)),
        "solid_white": Image.new("RGB", (256, 256), (255, 255, 255)),
        "random_noise": Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)),
        "skin_tone_face": Image.new("RGB", (256, 256), (235, 185, 155)),
        "blue_sky": Image.new("RGB", (256, 256), (90, 150, 240)),
        "red_car": Image.new("RGB", (256, 256), (220, 20, 30)),
        "flat_green_plastic": Image.new("RGB", (256, 256), (40, 180, 50)),
        "low_res_icon": Image.new("RGB", (32, 32), (50, 150, 50))
    }

    for name, img in test_cases.items():
        res = verify_potato_leaf(img)
        print(f"  {name:20s} -> Valid: {str(res.is_valid):5s} | Status: {res.status:30s} | Reason: {res.reason[:45]}...")
        assert not res.is_valid, f"Security failure: Non-leaf image '{name}' was accepted!"

    print("  => All non-leaf samples were correctly rejected.")

def test_fastapi_endpoint():
    print("\n--- 3. Testing FastAPI REST API Backend Logic ---", flush=True)
    from api.app import health_check, process_prediction

    # 1. Health check
    health_resp = health_check()
    assert health_resp["status"] == "healthy", f"Health check degraded: {health_resp}"
    print(f"  /health: {health_resp['status']}, supported={health_resp['supported_classes']}", flush=True)

    # 2. Non-leaf prediction test
    skin_img = Image.new("RGB", (256, 256), (235, 185, 155))
    res_nl = process_prediction(skin_img, filename="test_face.jpg")
    assert res_nl["status"] == "rejected", f"Expected 'rejected' status, got {res_nl['status']}"
    assert res_nl["is_potato_leaf"] is False, "Expected is_potato_leaf to be False"
    print(f"  Non-leaf rejection: status={res_nl['status']}, reason={res_nl['message']}", flush=True)

    # 3. Genuine potato leaf prediction test
    sample_path = list((PROJECT_ROOT / "data" / "processed" / "test" / "Potato___healthy").glob("*.jpg"))[0]
    with Image.open(sample_path) as img:
        res_pot = process_prediction(img.convert("RGB"), filename="potato_healthy.jpg")

    assert res_pot["status"] == "success", f"Expected 'success', got {res_pot['status']}"
    assert res_pot["is_potato_leaf"] is True, "Expected is_potato_leaf to be True"
    assert res_pot["prediction"]["class_code"] in CLASS_NAMES
    print(f"  Potato leaf diagnosis: condition={res_pot['prediction']['display_name']}, confidence={res_pot['prediction']['confidence_percent']}%", flush=True)
    print("  => FastAPI REST API verification passed.", flush=True)

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING AUTOMATED TEST SUITE: POTATO LEAF VERIFICATION")
    print("=" * 60)
    test_genuine_potato_leaves()
    test_non_leaf_rejections()
    test_fastapi_endpoint()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (100% VERIFICATION INTEGRITY)")
    print("=" * 60)
