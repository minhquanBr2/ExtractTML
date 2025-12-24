import re
from typing import Optional, Tuple
import numpy as np
import cv2
from .config import TAG_REGEX_LIST, OCR_WHITELIST, HSV_RANGES
from .preprocess import threshold_hsv, morph_close_open


def strip_non_alnum_spaces(s: str) -> str:
    return re.sub(r"[^A-Z0-9 ]+", " ", s.upper()).strip()


def replace_common_ocr_errors(s: str) -> str:
    return s.replace("O", "0") if re.fullmatch(r"[0-9O]+", s) else s


def extract_tag_from_text(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None
    t = strip_non_alnum_spaces(raw_text)
    t = replace_common_ocr_errors(t)

    for pat in TAG_REGEX_LIST:
        m = re.search(pat, t)
        if m:
            return m.group(0)
    return None


def normalize_tag(tag: str) -> str:
    t = tag.upper().strip().replace(" ", "")
    if t.startswith("TML"):
        m = re.match(r"^TML0*(\d+)([A-Z]?)$", t)
        if m:
            t = f"TML{int(m.group(1))}{m.group(2)}"
    return t


def ocr_preprocess(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    gray = cv2.medianBlur(gray, 3)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 5
    )
    if np.mean(bin_img) < 127:
        bin_img = 255 - bin_img
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return bin_img


def crop_with_padding(img: np.ndarray, bbox: Tuple[int, int, int, int], pad_ratio: float = 0.25) -> np.ndarray:
    x, y, w, h = bbox
    H, W = img.shape[:2]
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)
    return img[y1:y2, x1:x2].copy()


def remove_border_color(roi_bgr: np.ndarray, expected_color: str) -> np.ndarray:
    if roi_bgr.size == 0:
        return roi_bgr
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    ranges = HSV_RANGES.get(expected_color)
    if not ranges:
        return roi_bgr

    m = threshold_hsv(hsv, ranges)
    m = morph_close_open(m, k=3)
    return cv2.inpaint(roi_bgr, m, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
