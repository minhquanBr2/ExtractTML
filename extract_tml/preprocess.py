from typing import Tuple, List
import cv2
import numpy as np


def maybe_resize(img: np.ndarray, max_side: int = 2200) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = 1.0
    m = max(h, w)
    if m > max_side:
        scale = max_side / float(m)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def denoise_light(img: np.ndarray) -> np.ndarray:
    # Keep edges; mild denoise
    return cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)


def threshold_hsv(hsv: np.ndarray, ranges: List[tuple]) -> np.ndarray:
    mask_all = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (hmin, hmax), (smin, smax), (vmin, vmax) in ranges:
        lower = np.array([hmin, smin, vmin], dtype=np.uint8)
        upper = np.array([hmax, smax, vmax], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask_all = cv2.bitwise_or(mask_all, mask)
    return mask_all


def morph_close_open(mask: np.ndarray, k: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def remove_small_blobs(mask: np.ndarray, min_area: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == i] = 255
    return out


def split_stroke_fill(mask: np.ndarray, ksize: int = 3):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ksize, ksize)
    )

    # Stroke = morphological gradient
    stroke = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

    # Fill = close + hole fill (optional)
    fill = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return stroke, fill
