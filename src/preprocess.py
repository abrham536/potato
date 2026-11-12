import os
from pathlib import Path
import numpy as np

def compute_class_weights(train_dir: Path):
    """
    Computes balanced class weights for training:
        weight[j] = total_samples / (num_classes * count[j])
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
    class_weights = {}
    print("Class Distribution in Training Set:")
    for idx, cls in enumerate(classes):
        cnt = counts[cls]
        w = total_samples / (num_classes * cnt) if cnt > 0 else 1.0
        class_weights[idx] = round(w, 4)
        print(f"  [{idx}] {cls:25s}: {cnt:4d} images  -->  Class Weight: {w:.4f}")

    return class_weights

def build_data_pipeline(
    data_dir: Path,
    img_size=(256, 256),
    batch_size=32,
    seed=42
):
    """
    Builds optimized TensorFlow tf.data pipelines for Train, Val, and Test splits.
    Includes data augmentation for the training split, normalization, and prefetching.
    """
    import tensorflow as tf
    from tensorflow.keras import layers

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

    class_names = train_ds.class_names
    print(f"Detected Classes ({len(class_names)}): {class_names}")

    # Data Augmentation Pipeline (only applied during training)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ], name="data_augmentation")

    # Rescaling layer: maps [0, 255] -> [0.0, 1.0]
    rescale_layer = layers.Rescaling(1./255)

    AUTOTUNE = tf.data.AUTOTUNE

    # Prepare Train: Augment -> Rescale -> Cache -> Prefetch
    train_pipeline = train_ds.map(
        lambda x, y: (rescale_layer(data_augmentation(x, training=True)), y),
        num_parallel_calls=AUTOTUNE
    ).cache().prefetch(buffer_size=AUTOTUNE)

    # Prepare Val: Rescale -> Cache -> Prefetch
    val_pipeline = val_ds.map(
        lambda x, y: (rescale_layer(x), y),
        num_parallel_calls=AUTOTUNE
    ).cache().prefetch(buffer_size=AUTOTUNE)

    # Prepare Test: Rescale -> Cache -> Prefetch
    test_pipeline = test_ds.map(
        lambda x, y: (rescale_layer(x), y),
        num_parallel_calls=AUTOTUNE
    ).cache().prefetch(buffer_size=AUTOTUNE)

    return {
        "train": train_pipeline,
        "val": val_pipeline,
        "test": test_pipeline,
        "class_names": class_names,
        "data_augmentation": data_augmentation,
        "raw_train_ds": train_ds
    }

if __name__ == "__main__":
    processed_dir = Path("data") / "processed"
    if (processed_dir / "train").exists():
        compute_class_weights(processed_dir / "train")
    else:
        print("Please run src/data_split.py first to create the splits.")
