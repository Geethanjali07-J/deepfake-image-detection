from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import APP_HOST, APP_PORT
from app.services.model_service import model_service

app = FastAPI(
    title="Deepfake Image Detection API",
    description="Detect if an uploaded image is real or deepfake",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Deepfake detection service is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "No file was uploaded"})

    try:
        contents = await file.read()
        result = model_service.predict_from_bytes(contents)
        return result
    except Exception as exc:  # pragma: no cover
        return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=APP_HOST, port=APP_PORT, reload=True)
