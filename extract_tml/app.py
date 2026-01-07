from typing import Any, Dict, List, Optional
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


def _to_int(x: Any) -> int:
    # helps if your pipeline produces numpy ints
    return int(x)


# def normalize_detections(detections: Any) -> List[Dict[str, Any]]:
#     """
#     Convert your detections output into a JSON-friendly list.
#     Assumption: each detection has a bbox-like field.
#     Adjust mapping to match your actual detection type/keys.
#     """
#     out: List[Dict[str, Any]] = []

#     for d in detections or []:
#         # Case 1: dict-like detections
#         if isinstance(d, dict):
#             # try common keys
#             bbox = d.get("bbox") or d.get("box") or d.get("xyxy")
#             if bbox is None and all(k in d for k in ("x1", "y1", "x2", "y2")):
#                 bbox = [d["x1"], d["y1"], d["x2"], d["y2"]]

#             rec = dict(d)  # shallow copy
#             if bbox is not None:
#                 x1, y1, x2, y2 = bbox
#                 rec["bbox_xyxy"] = [_to_int(x1), _to_int(y1), _to_int(x2), _to_int(y2)]
#             out.append(rec)
#             continue

#         # Case 2: object-like detections (e.g., Candidate dataclass)
#         # Try to read typical attributes; adapt if yours differ
#         bbox = getattr(d, "bbox", None) or getattr(d, "xyxy", None)
#         x1 = getattr(d, "x1", None)
#         if bbox is None and None not in (x1, getattr(d, "y1", None), getattr(d, "x2", None), getattr(d, "y2", None)):
#             bbox = [d.x1, d.y1, d.x2, d.y2]

#         rec: Dict[str, Any] = {}
#         for k in ("spec_id", "tag", "text", "score", "color", "rule_id"):
#             if hasattr(d, k):
#                 v = getattr(d, k)
#                 # convert numpy scalars if any
#                 if isinstance(v, (np.generic,)):
#                     v = v.item()
#                 rec[k] = v

#         if bbox is not None:
#             x1, y1, x2, y2 = bbox
#             rec["bbox_xyxy"] = [_to_int(x1), _to_int(y1), _to_int(x2), _to_int(y2)]

#         out.append(rec)

#     return out


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
