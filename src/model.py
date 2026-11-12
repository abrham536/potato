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
    Builds a 4-Block Deep Convolutional Neural Network with Batch Normalization,
    Global Average Pooling, and a Softmax output classifier.
    
    Architecture:
      ConvBlock 1: Conv2D(32)  -> BatchNorm -> ReLU -> MaxPool2D(2,2)
      ConvBlock 2: Conv2D(64)  -> BatchNorm -> ReLU -> MaxPool2D(2,2)
      ConvBlock 3: Conv2D(128) -> BatchNorm -> ReLU -> MaxPool2D(2,2)
      ConvBlock 4: Conv2D(256) -> BatchNorm -> ReLU -> MaxPool2D(2,2)
      GlobalAveragePooling2D
      Dense(128) -> BatchNorm -> ReLU -> Dropout(0.3)
      Dense(3, activation='softmax')
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Block 1: Low-level edges & contours
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Block 2: Textural features & small leaf spots
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Block 3: Complex lesion geometry & margins
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # Block 4: High-level disease pattern recognition
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)

    # Spatial aggregation via Global Average Pooling (minimizes parameter count & prevents overfitting)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Dense representation head
    x = layers.Dense(128, name="dense128")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="relu_dense")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # Final Softmax Classification Head (3 mutually exclusive classes)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Potato_Disease_Custom_CNN")
    return model

def build_transfer_learning_model(
    input_shape=INPUT_SHAPE,
    num_classes=NUM_CLASSES,
    dropout_rate=DROPOUT_RATE
) -> tf.keras.Model:
    """
    Builds a Transfer Learning model using the MobileNetV2 backbone pretrained on ImageNet.
    Used for academic comparative benchmarking against our Custom CNN.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze pretrained feature extraction backbone

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Potato_MobileNetV2_TransferLearning")
    return model

if __name__ == "__main__":
    cnn_model = build_custom_cnn()
    cnn_model.summary()
