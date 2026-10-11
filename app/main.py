"""FastAPI service for the Korean food classifier.

Run from the project root:
    uvicorn app.main:app --reload --port 8000

Then open http://localhost:8000/docs to test it in the browser.
"""
import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

# predict.py, model.py and utils.py live in src/, so make that folder importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(SRC_DIR))

from predict import Predictor  # noqa: E402 (must come after the sys.path change)

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB, same limit as the frontend
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Frontend addresses that are allowed to call this API (comma separated in the env variable)
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts: load the model into memory
    app.state.predictor = Predictor()
    yield


app = FastAPI(title="Korean Food Classifier API", lifespan=lifespan)

# Without CORS the browser blocks requests from localhost:5173 to localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Simple check that the server is running and the model is loaded."""
    return {"status": "ok", "classes": app.state.predictor.class_names}


# A normal "def" (not "async def"): model inference is heavy work,
# so FastAPI runs it in a thread pool and the server stays responsive.
@app.post("/predict")
def predict_endpoint(file: UploadFile = File(...)):
    """Receive an image and return the predicted class with probabilities."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415, detail="Unsupported file type. Use JPG, PNG or WebP."
        )

    # Read one byte more than the limit, so we can detect files that are too large
    data = file.file.read(MAX_FILE_SIZE + 1)
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File is too large. Maximum size is 5 MB.")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # force decoding now, so a corrupted file fails here
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The file is not a valid image.")

    return app.state.predictor.predict(image)