from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "deepfake_detector.keras"

APP_HOST = "0.0.0.0"
APP_PORT = 8000
