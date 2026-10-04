from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from app.config import MODEL_DIR, MODEL_FILE


def synthetic_image(label: str, index: int, rng: np.random.Generator) -> np.ndarray:
    image = rng.normal(0.5, 0.25, size=(224, 224, 3)).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)

    if label == "deepfake":
        shifted = np.roll(image, shift=8, axis=0)
        blurred = cv2.GaussianBlur(shifted, (7, 7), 0)
        image = blurred
        image[:, :, 0] *= 0.9
        image[:, :, 1] *= 1.1
        image = np.clip(image, 0.0, 1.0)

    if index % 5 == 0:
        image = image + np.linspace(0.0, 0.3, 224, dtype=np.float32)[:, None, None]
        image = np.clip(image, 0.0, 1.0)

    return image


def generate_synthetic_dataset(output_dir: Path, images_per_class: int = 60):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    for label in ["real", "deepfake"]:
        class_dir = output_dir / label
        if class_dir.exists():
            shutil.rmtree(class_dir)
        class_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(images_per_class):
            image = synthetic_image(label, idx, rng)
            file_path = class_dir / f"{idx:03d}.png"
            plt.imsave(file_path, image)


def build_and_train_model(train_dir: Path, epochs: int = 8, save_path: Path = MODEL_FILE):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.1,
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode="binary",
    )

    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )

    model.fit(train_generator, epochs=epochs, steps_per_epoch=max(1, len(train_generator)))
    model.save(str(save_path))
    return model


if __name__ == "__main__":
    dataset_dir = Path(__file__).resolve().parent.parent / "synthetic_dataset"
    generate_synthetic_dataset(dataset_dir)
    build_and_train_model(dataset_dir, epochs=6)
    print(f"Saved model to: {MODEL_FILE}")
