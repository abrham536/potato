import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure runtime directories are configured before TF import
os.environ["KERAS_HOME"] = str(PROJECT_ROOT / ".keras")
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".matplotlib")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

from src.config import (
    PROCESSED_DATA_DIR,
    VERIFIER_MODEL_PATH,
    RANDOM_SEED
)

def build_potato_autoencoder(input_shape=(128, 128, 3)) -> tf.keras.Model:
    """
    Builds a lightweight Convolutional Autoencoder for unsupervised
    potato leaf anomaly / out-of-distribution detection.
    """
    inputs = layers.Input(shape=input_shape, name="verifier_input")
    x = layers.Rescaling(1.0 / 255.0, name="norm")(inputs)
    x = layers.RandomFlip("horizontal_and_vertical", seed=RANDOM_SEED)(x)
    x = layers.RandomRotation(0.1, seed=RANDOM_SEED)(x)

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv1")(x)
    x = layers.MaxPooling2D((2, 2), padding="same", name="enc_pool1")(x) # 64x64
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="enc_conv2")(x)
    x = layers.MaxPooling2D((2, 2), padding="same", name="enc_pool2")(x) # 32x32
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="enc_conv3")(x)
    encoded = layers.MaxPooling2D((2, 2), padding="same", name="enc_bottleneck")(x) # 16x16x64

    # Decoder
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same", name="dec_conv1")(encoded) # 32x32
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same", name="dec_conv2")(x) # 64x64
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same", name="dec_conv3")(x) # 128x128
    decoded = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same", name="dec_output")(x)

    model = models.Model(inputs, decoded, name="Potato_Leaf_Autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model

def load_split_images(split_dir: Path, img_size=(128, 128)):
    paths = list(split_dir.glob("*/*.jpg"))
    data = np.zeros((len(paths), img_size[0], img_size[1], 3), dtype=np.float32)
    for i, pt in enumerate(paths):
        with Image.open(pt) as f:
            data[i] = np.array(f.convert("RGB").resize(img_size), dtype=np.float32)
    return data

def main():
    print("=" * 60)
    print("TRAINING POTATO LEAF VERIFIER (AUTOENCODER)")
    print("=" * 60)

    train_dir = PROCESSED_DATA_DIR / "train"
    val_dir = PROCESSED_DATA_DIR / "val"
    test_dir = PROCESSED_DATA_DIR / "test"

    print(f"Loading training data from {train_dir}...")
    X_train = load_split_images(train_dir)
    print(f"Loaded {len(X_train)} training images.")

    print(f"Loading validation data from {val_dir}...")
    X_val = load_split_images(val_dir)
    print(f"Loaded {len(X_val)} validation images.")

    # Target is normalized [0, 1]
    Y_train = X_train / 255.0
    Y_val = X_val / 255.0

    model = build_potato_autoencoder()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)
    ]

    print("\nStarting autoencoder training...")
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=10,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\nSaving trained verifier model to: {VERIFIER_MODEL_PATH}")
    model.save(str(VERIFIER_MODEL_PATH))

    # Evaluate on test partition
    print(f"\nLoading test data from {test_dir}...")
    X_test = load_split_images(test_dir)
    preds = model.predict(X_test, batch_size=64, verbose=0)
    test_mse = np.mean((X_test / 255.0 - preds) ** 2, axis=(1, 2, 3))

    print(f"Test Set Potato Leaves Reconstruction MSE:")
    print(f"  - Mean: {np.mean(test_mse):.5f}")
    print(f"  - Std:  {np.std(test_mse):.5f}")
    print(f"  - Max:  {np.max(test_mse):.5f}")
    print(f"  - 99th Percentile: {np.percentile(test_mse, 99):.5f}")

    # Evaluate on non-leaf benchmarks
    print("\nEvaluating against synthetic Out-of-Distribution (non-leaf) samples:")
    non_leaves = {
        "solid_black": np.zeros((1, 128, 128, 3), dtype=np.float32),
        "solid_white": np.ones((1, 128, 128, 3), dtype=np.float32) * 255.0,
        "random_noise": np.random.uniform(0, 255, (1, 128, 128, 3)).astype(np.float32),
        "flat_green": np.full((1, 128, 128, 3), [40, 180, 50], dtype=np.float32),
        "blue_sky": np.full((1, 128, 128, 3), [100, 150, 255], dtype=np.float32)
    }
    for name, sample in non_leaves.items():
        rec = model.predict(sample, verbose=0)
        mse = float(np.mean((sample / 255.0 - rec) ** 2))
        print(f"  - {name:15s}: Reconstruction MSE = {mse:.5f}")

    print("\nPotato Leaf Verifier trained and saved successfully!")

if __name__ == "__main__":
    main()
