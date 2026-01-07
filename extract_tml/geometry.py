from typing import List, Optional, Tuple
import numpy as np
import cv2


def find_contours(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _filter_by_area(cnt: np.ndarray, min_area: int, max_area: int) -> Optional[float]:
    area = float(cv2.contourArea(cnt))
    if area < float(min_area) or area > float(max_area):
        return None
    print(f"Contour with {len(cnt)} points\tArea: {area}")
    return area


def contour_centroid(cnt: np.ndarray) -> Optional[Tuple[float, float]]:
    m = cv2.moments(cnt)
    if abs(m.get("m00", 0.0)) < 1e-6:
        return None
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return (float(cx), float(cy))


def polygon_centroid(poly: np.ndarray) -> Optional[Tuple[float, float]]:
    return contour_centroid(poly)


def approx_poly(cnt: np.ndarray, eps_ratio: float) -> np.ndarray:
    peri = cv2.arcLength(cnt, True)
    eps = eps_ratio * peri
    return cv2.approxPolyDP(cnt, eps, True)


def angle_deg(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    v1 = p0 - p1
    v2 = p2 - p1
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosang = float(np.dot(v1, v2) / (n1 * n2))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def is_rectangle_angles(poly: np.ndarray, tol_deg: float) -> bool:
    # poly shape: (N,1,2) with N=4
    if len(poly) != 4:
        return False
    pts = poly.reshape(-1, 2).astype(np.float32)
    hull = cv2.convexHull(pts.astype(np.int32), returnPoints=True).reshape(-1, 2).astype(np.float32)
    if len(hull) != 4:
        return False
    pts = hull
    for i in range(4):
        p0 = pts[(i - 1) % 4]
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        a = angle_deg(p0, p1, p2)
        if abs(a - 90.0) > tol_deg:
            return False
    return True


def compute_fill_ratio(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return 0.0
    roi = mask[y : y + h, x : x + w]
    filled = float(np.count_nonzero(roi))
    return filled / float(w * h)
