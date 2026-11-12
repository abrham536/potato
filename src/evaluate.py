import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DATA_DIR,
    BEST_MODEL_PATH,
    FIGURES_DIR,
    REPORTS_DIR,
    CLASS_NAMES,
    DISPLAY_LABELS,
    BATCH_SIZE,
)
from src.preprocess import build_data_pipeline

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def evaluate():
    print("=" * 60)
    print("MODEL EVALUATION ON UNTOUCHED TEST SET")
    print("=" * 60)

    if not BEST_MODEL_PATH.exists():
        print(f"Error: Model file {BEST_MODEL_PATH} not found. Please run train.py first.")
        return

    # 1. Load trained model
    print(f"Loading trained weights from: {BEST_MODEL_PATH}")
    model = tf.keras.models.load_model(str(BEST_MODEL_PATH))

    # 2. Load Test dataset
    pipeline = build_data_pipeline(PROCESSED_DATA_DIR, batch_size=BATCH_SIZE)
    test_ds = pipeline["test"]

    # 3. Model evaluation
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n[Test Results] Overall Loss: {test_loss:.4f} | Overall Accuracy: {test_acc*100:.2f}%\n")

    # 4. Predict on full test set
    y_true_list = []
    y_pred_list = []
    y_prob_list = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        trues = np.argmax(labels.numpy(), axis=1)

        y_true_list.extend(trues)
        y_pred_list.extend(preds)
        y_prob_list.extend(probs)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    num_classes = len(CLASS_NAMES)

    # 5. Compute Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # 6. Compute Precision, Recall, F1
    report = {}
    print(f"{'Class':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 75)

    macro_p, macro_r, macro_f1 = [], [], []

    for idx, (cls_name, display_name) in enumerate(zip(CLASS_NAMES, DISPLAY_LABELS)):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        support = int(cm[idx, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        macro_p.append(precision)
        macro_r.append(recall)
        macro_f1.append(f1)

        report[cls_name] = {
            "display_name": display_name,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support
        }
        print(f"{display_name:<25} | {precision*100:8.2f}% | {recall*100:8.2f}% | {f1*100:8.2f}% | {support:<8}")

    print("-" * 75)
    macro_avg_f1 = np.mean(macro_f1)
    print(f"{'Macro Average':<25} | {np.mean(macro_p)*100:8.2f}% | {np.mean(macro_r)*100:8.2f}% | {macro_avg_f1*100:8.2f}% | {len(y_true):<8}")

    summary_metrics = {
        "test_loss": round(float(test_loss), 4),
        "test_accuracy": round(float(test_acc), 4),
        "macro_f1": round(float(macro_avg_f1), 4),
        "per_class": report,
        "confusion_matrix": cm.tolist()
    }

    # Save metrics JSON
    json_path = REPORTS_DIR / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(summary_metrics, f, indent=4)
    print(f"\nDetailed evaluation metrics saved to: {json_path}")

    # 7. Plot and Save Confusion Matrix Heatmap
    plot_confusion_matrix(cm, DISPLAY_LABELS, FIGURES_DIR / "confusion_matrix.png")

def plot_confusion_matrix(cm: np.ndarray, class_labels: list, output_path: Path):
    """
    Renders high-resolution academic confusion matrix heatmap.
    """
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Potato Disease CNN — Confusion Matrix (Test Set)", fontsize=12, weight="bold", pad=12)
    plt.colorbar()

    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=20, ha="right", fontsize=10)
    plt.yticks(tick_marks, class_labels, fontsize=10)

    # Label values inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(
                j, i, f"{val}",
                ha="center", va="center",
                fontsize=13, weight="bold",
                color="white" if val > thresh else "black"
            )

    plt.ylabel("True Label (Ground Truth)", fontsize=11, weight="bold")
    plt.xlabel("Predicted Label", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)

    artifact_dir = Path(r"C:\Users\abreham\.gemini\antigravity\brain\6301ef43-bee2-47bf-8263-e660c8d0c4ad")
    if artifact_dir.exists():
        plt.savefig(artifact_dir / "confusion_matrix.png", dpi=150)

    plt.close()
    print(f"Saved confusion matrix figure to: {output_path}")

if __name__ == "__main__":
    evaluate()
