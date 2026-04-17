"""
Dog Breed Classifier API  —  Deploy on Render
Start command: uvicorn main:app --host 0.0.0.0 --port $PORT

Required files in same directory:
  dogmodel.onnx   (~44 MB, exported from your .pth)
  Dog_List.txt    (73 breed names, one per line)
"""

import io
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# ── shared state loaded once at startup ──────────────────────────────────────
_state: dict = {}

NICKNAMES = {
    "Xoloitzcuintli": " (Mexican Hairless)",
    "Staffordshire-Bull-Terrier": " (Pitbull)",
    "Pembroke-Welsh-Corgi": " (Corgi)",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["session"] = ort.InferenceSession(
        "dogmodel.onnx", providers=["CPUExecutionProvider"]
    )
    with open("Dog_List.txt") as f:
        _state["labels"] = [line.strip() for line in f if line.strip()]
    print(f"✅  Model loaded — {len(_state['labels'])} breeds")
    yield
    _state.clear()


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Dog Classifier", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your Cloudflare Pages URL in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── helpers ───────────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    """Replicate the exact transforms used during training."""
    # Resize + centre-crop to 224×224  (same as torchvision pipeline)
    image = image.resize((224, 224), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    # ImageNet normalisation
    arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    arr = arr.transpose(2, 0, 1)          # HWC → CHW
    return arr[np.newaxis].astype(np.float32)   # add batch dim


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    logits = _state["session"].run(None, {"input": preprocess(image)})[0][0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()

    top3_idx = np.argsort(probs)[-3:][::-1]
    labels = _state["labels"]

    return JSONResponse({
        "predictions": [
            {
                "breed": labels[i],
                "display_name": (
                    labels[i].replace("-", " ") + NICKNAMES.get(labels[i], "")
                ),
                "probability": round(float(probs[i]), 4),
            }
            for i in top3_idx
        ]
    })
