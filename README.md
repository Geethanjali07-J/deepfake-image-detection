# Deepfake Image Detection

A full-stack deep learning project for classifying whether an uploaded image is a real photo or a deepfake. The project includes:

- React + Vite frontend for image upload and visualization
- FastAPI backend for file processing and model inference
- TensorFlow/Keras model pipeline for deepfake image detection
- Training script with synthetic dataset generation for local experimentation

## Project structure

- `frontend/` — React app for uploading and visualizing results
- `backend/` — FastAPI API and TensorFlow model logic
- `backend/train_model.py` — script that builds and trains a sample deepfake detector
- `README.md` — project overview and setup instructions

## Features

- Image upload from the browser
- Real-time prediction through a REST API
- TensorFlow-based CNN and transfer-learning model design
- CORS-enabled FastAPI backend for frontend communication
- Simple model training workflow

## Tech stack

Frontend:
- React
- Vite
- Tailwind CSS

Backend:
- FastAPI
- Uvicorn
- TensorFlow
- OpenCV
- Pillow

## Quick start

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

Then open the local frontend URL displayed by Vite and upload an image.

## Backend API

- `GET /health` — checks service health
- `POST /predict` — uploads an image and returns prediction data

Example response:

```json
{
  "prediction": "deepfake",
  "score": 0.86,
  "confidence": 0.86
}
```

## Model note

This project includes a training pipeline and a transfer-learning model setup. For production use, replace the synthetic dataset with a real deepfake detection dataset such as FaceForensics++, DFDC, or a curated custom dataset.

## License

MIT
