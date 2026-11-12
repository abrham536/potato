import os
import sys
from pathlib import Path

# Add project root to sys.path and configure Keras environment
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import src.config

import numpy as np

def compute_class_weights(train_dir: Path, smooth_factor: float = 0.5):
    """
    Computes smoothed, balanced class weights for training:
        weight[j] = (total_samples / (num_classes * count[j])) ** smooth_factor
    Normalized so the average weight is 1.0, preventing extreme gradient dominance.
    """
    train_dir = Path(train_dir)
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    counts = {}
    total_samples = 0
    
    for cls in classes:
        cls_path = train_dir / cls
        c = len([f for f in cls_path.glob("*") if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        counts[cls] = c
        total_samples += c

    num_classes = len(classes)
    raw_weights = []
    for cls in classes:
        cnt = counts[cls]
        w = (total_samples / (num_classes * cnt)) ** smooth_factor if cnt > 0 else 1.0
        raw_weights.append(w)

    mean_weight = sum(raw_weights) / len(raw_weights)
    class_weights = {}
    print("\nSmoothed Class Distribution in Training Set:")
    for idx, cls in enumerate(classes):
        normalized_w = round(raw_weights[idx] / mean_weight, 4)
        class_weights[idx] = normalized_w
        print(f"  [{idx}] {cls:25s}: {counts[cls]:4d} images  -->  Class Weight: {normalized_w:.4f}")

    return class_weights

def build_data_pipeline(
    data_dir: Path,
    img_size=(256, 256),
    batch_size=32,
    seed=42
):
    """
    Builds clean, high-performance tf.data pipelines for Train, Val, and Test splits.
    Rescaling and data augmentation are handled directly inside the Keras Model.
    """
    import tensorflow as tf

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    print(f"\nLoading datasets from: {data_dir.resolve()}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        label_mode="categorical"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode="categorical"
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode="categorical"
    )

    AUTOTUNE = tf.data.AUTOTUNE

    return {
        "train": train_ds.cache().prefetch(buffer_size=AUTOTUNE),
        "val": val_ds.cache().prefetch(buffer_size=AUTOTUNE),
        "test": test_ds.cache().prefetch(buffer_size=AUTOTUNE),
        "class_names": train_ds.class_names
    }

if __name__ == "__main__":
    processed_dir = Path("data") / "processed"
    if (processed_dir / "train").exists():
        compute_class_weights(processed_dir / "train")
    else:
        print("Please run src/data_split.py first to create the splits.")
