from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def read_image_to_array(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.fit(image, target_size, method=Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def ensure_sample_image_dir(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
