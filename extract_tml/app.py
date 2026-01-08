from typing import List, Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


from .core import extract_tags  # keep relative import if app.py is inside your package

app = FastAPI(title="TML Extract API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_specs_allowed(s: Optional[str]) -> List[int]:
    if s is None or not str(s).strip():
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def decode_upload_to_bgr(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Provide a valid JPG/PNG.")
    return img


@app.post("/extract")
def extract(
    file: UploadFile = File(...),
    specs_allowed: str = Query("7,9", description='Comma-separated spec IDs, e.g. "7,9"')
):
    img = decode_upload_to_bgr(file)
    specs = parse_specs_allowed(specs_allowed)
    print(f"[INFO] Extracting TML tags with specs_allowed={specs}")

    out_json = extract_tags(
        img,
        specs_allowed=specs,
        debug=False
    )

    payload = out_json
    return JSONResponse(payload)
