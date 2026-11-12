import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INPUT_SHAPE, NUM_CLASSES, DROPOUT_RATE
import tensorflow as tf
from tensorflow.keras import layers, models

def build_custom_cnn(
    input_shape=INPUT_SHAPE,
    num_classes=NUM_CLASSES,
    dropout_rate=DROPOUT_RATE
) -> tf.keras.Model:
    """
    Builds an end-to-end Convolutional Neural Network with built-in preprocessing,
    runtime data augmentation, hierarchical feature extraction, and Softmax classification.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name="input_image"),
        
        # 1. Internal Preprocessing & Data Augmentation (active only during training)
        layers.Rescaling(1.0 / 255.0, name="rescaling"),
        layers.RandomFlip("horizontal_and_vertical", name="random_flip"),
        layers.RandomRotation(0.15, name="random_rotation"),
        layers.RandomZoom(0.10, name="random_zoom"),

        # 2. Convolutional Block 1 (Low-level features)
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),

        # 3. Convolutional Block 2 (Leaf texture & lesion margins)
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),

        # 4. Convolutional Block 3 (Necrotic spot morphology)
        layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool3"),

        # 5. Convolutional Block 4 (Complex disease patterns)
        layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv4"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool4"),

        # 6. Classifier Head
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense128"),
        layers.Dropout(dropout_rate, name="dropout"),
        layers.Dense(num_classes, activation="softmax", name="classifier_output")
    ], name="Potato_Disease_Custom_CNN")

    return model

def build_transfer_learning_model(
    input_shape=INPUT_SHAPE,
    num_classes=NUM_CLASSES,
    dropout_rate=DROPOUT_RATE
) -> tf.keras.Model:
    """
    Builds a Transfer Learning model using the MobileNetV2 backbone pretrained on ImageNet.
    Used for academic comparative benchmarking.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=input_shape, name="input_image"),
        layers.Rescaling(1.0 / 255.0, name="rescaling"),
        base_model,
        layers.GlobalAveragePooling2D(name="gap"),
        layers.Dropout(dropout_rate, name="dropout"),
        layers.Dense(num_classes, activation="softmax", name="classifier_output")
    ], name="Potato_MobileNetV2_TransferLearning")

    return model

if __name__ == "__main__":
    cnn = build_custom_cnn()
    cnn.summary()
