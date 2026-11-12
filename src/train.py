import os
import sys
import csv
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DATA_DIR,
    BEST_MODEL_PATH,
    TRAINING_LOG_PATH,
    FIGURES_DIR,
    BATCH_SIZE,
    MAX_EPOCHS,
    INITIAL_LEARNING_RATE,
    MIN_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
)
from src.preprocess import build_data_pipeline, compute_class_weights
from src.model import build_custom_cnn

import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def train():
    print("=" * 60)
    print("STARTING POTATO DISEASE CNN MODEL TRAINING")
    print("=" * 60)

    # 1. Prepare data loaders
    pipeline = build_data_pipeline(PROCESSED_DATA_DIR, batch_size=BATCH_SIZE)
    train_ds = pipeline["train"]
    val_ds = pipeline["val"]

    # 2. Compute class weights to address minority healthy class
    class_weights = compute_class_weights(PROCESSED_DATA_DIR / "train")
    print(f"\nApplied Class Weights: {class_weights}")

    # 3. Build & Compile Model
    model = build_custom_cnn()
    optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("\nModel compiled successfully with Adam optimizer and Categorical Cross-Entropy loss.")

    # 4. Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(TRAINING_LOG_PATH),
            separator=",",
            append=False
        )
    ]

    # 5. Fit Model
    print(f"\nTraining for up to {MAX_EPOCHS} epochs with batch size {BATCH_SIZE}...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    print(f"\nBest model successfully saved to: {BEST_MODEL_PATH}")

    # 6. Plot Learning Curves
    plot_training_curves(TRAINING_LOG_PATH, FIGURES_DIR / "training_curves.png")

def plot_training_curves(csv_path: Path, output_image_path: Path):
    """
    Plots training vs validation accuracy and loss curves from CSV logs.
    """
    if not csv_path.exists():
        print(f"Log file {csv_path} not found.")
        return

    epochs, acc, val_acc, loss, val_loss = [], [], [], [], []
    with open(csv_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]) + 1)
            acc.append(float(row["accuracy"]))
            val_acc.append(float(row["val_accuracy"]))
            loss.append(float(row["loss"]))
            val_loss.append(float(row["val_loss"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs, acc, "b-o", label="Training Accuracy", linewidth=2)
    ax1.plot(epochs, val_acc, "g-s", label="Validation Accuracy", linewidth=2)
    ax1.set_title("Training and Validation Accuracy", fontsize=12, weight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    # Loss Plot
    ax2.plot(epochs, loss, "r-o", label="Training Loss", linewidth=2)
    ax2.plot(epochs, val_loss, "m-s", label="Validation Loss", linewidth=2)
    ax2.set_title("Training and Validation Loss", fontsize=12, weight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=150)
    
    artifact_dir = Path(r"C:\Users\abreham\.gemini\antigravity\brain\6301ef43-bee2-47bf-8263-e660c8d0c4ad")
    if artifact_dir.exists():
        plt.savefig(artifact_dir / "training_curves.png", dpi=150)

    plt.close()
    print(f"Saved learning curves to: {output_image_path}")

if __name__ == "__main__":
    train()
