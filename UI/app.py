# Beeonic Breast Ultrasound Triage API (decision support, NOT a diagnosis).
# Run:  uvicorn app:app --host 0.0.0.0 --port 8000
# Env:  BEEONIC_API_KEY   (if set, every /v1 call must send header X-API-Key)
#       BEEONIC_MODEL_DIR (default: ./models, folder with model_*.keras and config.json)
import base64
import hmac
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

MODEL_DIR = Path(os.environ.get("BEEONIC_MODEL_DIR", "models"))
API_KEY = os.environ.get("BEEONIC_API_KEY")          # unset = open access (development only)
MAX_BYTES = 10 * 1024 * 1024                          # 10 MB per image
MAX_BATCH = 20
STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    import keras
    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    paths = sorted(MODEL_DIR.glob("model_*.keras"))
    if not paths:
        raise RuntimeError(f"No model_*.keras found in {MODEL_DIR.resolve()}")
    STATE["cfg"] = cfg
    STATE["models"] = [keras.saving.load_model(p, compile=False) for p in paths]
    yield
    STATE.clear()


app = FastAPI(
    title="Beeonic Breast Ultrasound Triage API",
    version="2.0",
    description="Sorts a breast ultrasound into benign / malignant / normal to prioritise the radiologist worklist. "
                "Decision support only: a radiologist reads every image.",
    lifespan=lifespan,
)


def check_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and not hmac.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def preprocess(raw: bytes, size: int) -> np.ndarray:
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image larger than 10 MB")
    try:
        img = Image.open(io.BytesIO(raw)).convert("L").resize((size, size))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File is not a readable image (use PNG or JPG)")
    arr = np.asarray(img, dtype=np.float32)               # 0..255, same scale as training
    return np.repeat(arr[None, :, :, None], 3, axis=-1)   # (1, S, S, 3)


def run_models(x: np.ndarray, tta: bool):
    probs, segs = [], []
    for m in STATE["models"]:
        o = m(x, training=False)
        p, s = np.asarray(o["probs"]), np.asarray(o["seg"])
        if tta:
            o2 = m(x[:, :, ::-1, :], training=False)
            p = (p + np.asarray(o2["probs"])) / 2.0
            s = (s + np.asarray(o2["seg"])[:, :, ::-1, :]) / 2.0
        probs.append(p)
        segs.append(s)
    return np.mean(probs, axis=0)[0], np.mean(segs, axis=0)[0, :, :, 0]


def decide(p: np.ndarray, cfg: dict) -> int:
    mal = cfg["class_names"].index("malignant")
    if p[mal] >= cfg["threshold"]:
        return mal
    q = p.copy()
    q[mal] = -1.0
    return int(q.argmax())


def overlay_png_b64(x: np.ndarray, seg: np.ndarray) -> str:
    gray = x[0, :, :, 0] / 255.0
    rgb = np.stack([gray] * 3, axis=-1)
    m = (seg > 0.5)[..., None]
    red = np.array([1.0, 0.1, 0.1])
    rgb = np.where(m, 0.6 * rgb + 0.4 * red, rgb)
    buf = io.BytesIO()
    Image.fromarray((rgb * 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def analyse(raw: bytes, overlay: bool) -> dict:
    cfg = STATE["cfg"]
    x = preprocess(raw, cfg["img_size"])
    p, seg = run_models(x, cfg.get("use_tta", True))
    k = decide(p, cfg)
    name = cfg["class_names"][k]
    out = {
        "predicted_class": name,
        "probabilities": {c: round(float(v), 4) for c, v in zip(cfg["class_names"], p)},
        "malignant_probability": round(float(p[cfg["class_names"].index("malignant")]), 4),
        "decision_threshold_malignant": cfg["threshold"],
        "triage": cfg["triage"][name],
        "lesion_area_pct": round(float((seg > 0.5).mean() * 100), 2),
        "needs_extra_attention": bool(float(p.max()) < 0.55),
        "model_version": cfg["version"],
        "disclaimer": "Decision support for worklist prioritisation only. A radiologist must read every image.",
    }
    if overlay:
        out["lesion_overlay_png_base64"] = overlay_png_b64(x, seg)
    return out


@app.get("/health")
def health():
    ok = "models" in STATE
    return {"status": "ok" if ok else "loading", "n_models": len(STATE.get("models", [])),
            "version": STATE.get("cfg", {}).get("version")}


@app.get("/v1/info", dependencies=[Depends(check_key)])
def info():
    cfg = STATE["cfg"]
    return {k: cfg[k] for k in ["version", "class_names", "img_size", "threshold", "use_tta", "n_models", "triage", "test_metrics"] if k in cfg}


@app.post("/v1/predict", dependencies=[Depends(check_key)])
async def predict(file: UploadFile = File(...), include_overlay: bool = Query(False)):
    return analyse(await file.read(), include_overlay)


@app.post("/v1/predict/batch", dependencies=[Depends(check_key)])
async def predict_batch(files: List[UploadFile] = File(...)):
    if len(files) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"At most {MAX_BATCH} images per request")
    results = [{"filename": f.filename, **analyse(await f.read(), False)} for f in files]
    # worklist order: malignant first (highest malignant probability first), then benign, then normal
    rank = {"malignant": 0, "benign": 1, "normal": 2}
    results.sort(key=lambda r: (rank[r["predicted_class"]], -r["malignant_probability"]))
    return {"count": len(results), "worklist": results}


# --- Beeonic web UI ---
from fastapi.responses import FileResponse


@app.get("/", include_in_schema=False)
def web_ui():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
