"""
Plant Disease CNN Model Training Script
========================================
Dataset: PlantVillage (Kaggle) - 26 disease classes
Model: Custom CNN with BatchNormalization
Input: 224x224 RGB images
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CONFIG = {
    "img_size": 224,
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 0.001,
    "train_dir": "data/train",       # Path to training images
    "val_dir": "data/val",           # Path to validation images
    "model_save_path": "model/model.h5",
    "class_indices_path": "model/class_indices.json",
    "confidence_threshold": 0.70,
}

IMG_SIZE = CONFIG["img_size"]
BATCH_SIZE = CONFIG["batch_size"]


# ─────────────────────────────────────────
# 1. DATA AUGMENTATION & GENERATORS
# ─────────────────────────────────────────
def create_data_generators():
    """Create training and validation data generators with augmentation."""

    # Augmentation for training data to prevent overfitting
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,          # Normalize pixel values to [0,1]
        rotation_range=20,           # Random rotation ±20 degrees
        width_shift_range=0.2,       # Horizontal shift
        height_shift_range=0.2,      # Vertical shift
        shear_range=0.15,            # Shear transformation
        zoom_range=0.15,             # Random zoom
        horizontal_flip=True,        # Mirror images
        fill_mode="nearest",         # Fill empty pixels
    )

    # Only normalize validation data (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        CONFIG["train_dir"],
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        CONFIG["val_dir"],
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen


# ─────────────────────────────────────────
# 2. CNN MODEL ARCHITECTURE
# ─────────────────────────────────────────
def build_model(num_classes: int) -> tf.keras.Model:
    """
    Build CNN model.
    Architecture:
      Conv2D → BN → MaxPool
      Conv2D → BN → MaxPool
      Conv2D → BN → MaxPool
      Flatten → Dense → Dropout → Softmax
    """
    model = models.Sequential([
        # ── Block 1 ──────────────────────────────
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # ── Block 2 ──────────────────────────────
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # ── Block 3 ──────────────────────────────
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # ── Block 4 ──────────────────────────────
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # ── Classifier ───────────────────────────
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),          # Dropout to prevent overfitting
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),  # Output layer
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


# ─────────────────────────────────────────
# 3. CALLBACKS (Early Stopping, etc.)
# ─────────────────────────────────────────
def get_callbacks():
    """Define training callbacks."""
    return [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,                 # Stop if no improvement for 5 epochs
            restore_best_weights=True,  # Restore best model weights
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,                 # Halve learning rate on plateau
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=CONFIG["model_save_path"],
            monitor="val_accuracy",
            save_best_only=True,        # Only save when accuracy improves
            verbose=1,
        ),
    ]


# ─────────────────────────────────────────
# 4. PLOT TRAINING HISTORY
# ─────────────────────────────────────────
def plot_history(history):
    """Plot training & validation accuracy/loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Loss plot
    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("model/training_history.png", dpi=150)
    plt.show()
    print("Training history plot saved to model/training_history.png")


# ─────────────────────────────────────────
# 5. MAIN TRAINING ROUTINE
# ─────────────────────────────────────────
def train():
    # Create output directory
    os.makedirs("model", exist_ok=True)

    print("=" * 50)
    print("  Plant Disease CNN — Training")
    print("=" * 50)

    # Load data
    train_gen, val_gen = create_data_generators()
    num_classes = len(train_gen.class_indices)
    print(f"\nClasses found: {num_classes}")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")

    # Save class indices mapping → used by backend for predictions
    class_indices = {v: k for k, v in train_gen.class_indices.items()}
    with open(CONFIG["class_indices_path"], "w") as f:
        json.dump(class_indices, f, indent=2)
    print(f"\nClass indices saved to {CONFIG['class_indices_path']}")

    # Build model
    model = build_model(num_classes)

    # Train
    print("\nStarting training...")
    history = model.fit(
        train_gen,
        epochs=CONFIG["epochs"],
        validation_data=val_gen,
        callbacks=get_callbacks(),
        verbose=1,
    )

    # Final evaluation
    print("\nEvaluating on validation set...")
    loss, acc = model.evaluate(val_gen, verbose=0)
    print(f"Validation Accuracy: {acc * 100:.2f}%")
    print(f"Validation Loss:     {loss:.4f}")

    # Plot results
    plot_history(history)

    print(f"\nModel saved to: {CONFIG['model_save_path']}")
    print("Training complete!")


if __name__ == "__main__":
    train()
