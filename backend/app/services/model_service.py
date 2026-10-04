from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from app.config import MODEL_DIR, MODEL_FILE


class DeepfakeDetectorService:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self._load_or_build_model()

    def _build_model(self):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )

        base_model.trainable = False
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.BinaryAccuracy(name="accuracy")],
        )
        return model

    def _load_or_build_model(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if MODEL_FILE.exists():
            self.model = tf.keras.models.load_model(str(MODEL_FILE), compile=False)
            self.model_loaded = True
            return

        self.model = self._build_model()
        self.model_loaded = True

    def _preprocess_image(self, image_data: bytes):
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = ImageOps.fit(image, (224, 224), method=Image.Resampling.LANCZOS)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def predict_from_bytes(self, image_data: bytes):
        if self.model is None:
            raise RuntimeError("Model is not available")

        image_array = self._preprocess_image(image_data)
        score = float(self.model.predict(image_array, verbose=0)[0][0])
        prediction = "deepfake" if score >= 0.5 else "real"
        confidence = float(max(score, 1.0 - score))

        return {
            "prediction": prediction,
            "score": round(score, 4),
            "confidence": round(confidence, 4),
        }


model_service = DeepfakeDetectorService()
