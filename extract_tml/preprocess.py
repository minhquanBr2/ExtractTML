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


def clean_stroke_mask(mask: np.ndarray, k_close: int = 3) -> np.ndarray:
    """
    For thin border strokes: avoid OPEN (erosion-first). Use CLOSE only to bridge gaps.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)


def clean_fill_mask(mask: np.ndarray, k: int = 3) -> np.ndarray:
    """
    For filled regions: close+open is OK to fill small holes and remove speckles.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


def remove_small_blobs(mask: np.ndarray, min_area: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == i] = 255
    return out


def remove_leader_lines(mask: np.ndarray, lengths=(15, 25, 35)) -> np.ndarray:
    """
    Removes long thin line-like components (leader lines) from a stroke mask.
    """
    mask = mask.copy()
    extracted_lines = np.zeros_like(mask)

    for L in lengths:
        # horizontal line kernel
        kx = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
        # vertical line kernel
        ky = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))

        # opening extracts line segments matching kernel orientation
        lines_h = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kx)
        lines_v = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ky)

        extracted_lines = cv2.bitwise_or(extracted_lines, lines_h)
        extracted_lines = cv2.bitwise_or(extracted_lines, lines_v)

    # remove extracted lines from original
    core = cv2.bitwise_and(mask, cv2.bitwise_not(extracted_lines))

    # small close to heal minor nicks on the shape border after subtraction
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, k, iterations=1)

    return core


def split_stroke_fill_from_raw(
    raw0: np.ndarray,
    *,
    k_stroke: int = 3,
    k_fill: int = 3,
    min_area: int = 200,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Build stroke/fill from the SAME color threshold (raw0) but with different cleaning paths.
    Returns: (stroke_mask, fill_mask, debug_dict)
    """
    # --- stroke path (preserve thin borders)
    stroke0 = remove_small_blobs(raw0, min_area=min_area)
    stroke1 = clean_stroke_mask(stroke0, k_close=k_stroke)
    # stroke2 = remove_leader_lines(stroke1)

    # --- fill path (more aggressive cleaning)
    fill0 = clean_fill_mask(raw0, k=k_fill)
    fill1 = remove_small_blobs(fill0, min_area=min_area)

    # Convert fill into "solid" by closing again (optional)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_fill, k_fill))
    fill2 = cv2.morphologyEx(fill1, cv2.MORPH_CLOSE, k, iterations=1)

    debug = {
        "raw0": raw0,
        "stroke0_blobs": stroke0,
        "stroke1_close": stroke1,
        # "stroke2_leader": stroke2,
        "fill0_closeopen": fill0,
        "fill1_blobs": fill1,
        "fill2_close": fill2,
    }
    return stroke1, fill2, debug