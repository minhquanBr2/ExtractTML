from typing import List, Optional, Dict, Any
from .types import Candidate
from .geometry import find_contours, _filter_by_area, contour_centroid, approx_poly, is_rectangle_angles
import cv2
import numpy as np


def core_bbox_from_contour(cnt: np.ndarray, keep_ratio: float = 0.7) -> tuple[int,int,int,int]:
    """
    Compute bbox from 'core' contour points by dropping farthest points (tail outliers).
    keep_ratio=0.7 keeps the closest 70% points to centroid.
    """
    pts = cnt.reshape(-1, 2).astype(np.float32)

    m = cv2.moments(cnt)
    if abs(m.get("m00", 0.0)) < 1e-6:
        x,y,w,h = cv2.boundingRect(cnt)
        return x,y,w,h
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    center = np.array([cx, cy], dtype=np.float32)

    d = np.linalg.norm(pts - center, axis=1)
    thresh = np.quantile(d, keep_ratio)       # keep closest keep_ratio points
    core_pts = pts[d <= thresh].astype(np.int32)

    x, y, w, h = cv2.boundingRect(core_pts)
    return int(x), int(y), int(w), int(h)


def detect_circles_from_mask(
    mask: np.ndarray,
    *,
    meta: Optional[Dict[str, Any]] = None,
    min_area: int = 25,
    max_area: int = 2000,
    circularity_min: float = 0.4,
    circularity_max: float = 2,
) -> List[Candidate]:
    meta = meta or {}
    out: List[Candidate] = []

    for cnt in find_contours(mask):
        area = _filter_by_area(cnt, min_area, max_area)
        if area is None:
            continue

        peri = float(cv2.arcLength(cnt, True))
        if peri <= 1e-6:
            continue

        circ = (4.0 * np.pi * area) / (peri * peri)
        print(f"Circularity: {circ}")
        if not (circularity_min <= circ <= circularity_max):
            continue

        cxy = contour_centroid(cnt)
        print(f"Centroid: {cxy}")
        if cxy is None:
            continue

        x, y, w, h = core_bbox_from_contour(cnt, keep_ratio=0.7)

        out.append(Candidate(
            bbox=(x, y, w, h),
            contour=cnt,
            center=cxy,
            shape="circle",
            score=area,
            color=meta.get("color"),
            use=meta.get("use"),
            rule_id=meta.get("rule_id"),
        ))
    print(f"Found {len(out)} circles.")
    return out


def detect_rects_from_mask(
    mask: np.ndarray,
    *,
    meta: Optional[Dict[str, Any]] = None,
    min_area: int = 200,
    max_area: int = 2000,
    poly_eps_ratio: float = 0.02,
    angle_tol_deg: float = 20.0,
) -> List[Candidate]:
    meta = meta or {}
    out: List[Candidate] = []

    for cnt in find_contours(mask):
        area = _filter_by_area(cnt, min_area, max_area)
        if area is None:
            continue

        poly = approx_poly(cnt, poly_eps_ratio)
        if len(poly) != 4:
            continue

        if not is_rectangle_angles(poly, tol_deg=angle_tol_deg):
            continue

        cxy = contour_centroid(poly)
        if cxy is None:
            continue

        x, y, w, h = core_bbox_from_contour(poly, keep_ratio=0.7)

        out.append(Candidate(
            bbox=(x, y, w, h),
            contour=poly,
            center=cxy,
            shape="rect",
            score=area,
            color=meta.get("color"),
            use=meta.get("use"),
            rule_id=meta.get("rule_id"),
        ))
    print(f"Found {len(out)} rectangles.")
    return out


def detect_polygons_from_mask(
    mask: np.ndarray,
    n_vertices: int,
    *,
    meta: Optional[Dict[str, Any]] = None,
    min_area: int = 200,
    max_area: int = 10000,
    poly_eps_ratio: float = 0.02,
    reject_rectangles_when_n4: bool = True,
    rect_angle_tol_deg: float = 20.0,
) -> List[Candidate]:
    if n_vertices < 3:
        raise ValueError("n_vertices must be >= 3")

    meta = meta or {}
    out: List[Candidate] = []

    for i, cnt in enumerate(find_contours(mask)):
        area = _filter_by_area(cnt, min_area, max_area)
        if area is None:
            continue

        poly = approx_poly(cnt, poly_eps_ratio)

        cxy = contour_centroid(poly)
        if cxy is None:
            continue

        x, y, w, h = core_bbox_from_contour(poly, keep_ratio=0.7)

        out.append(Candidate(
            bbox=(x, y, w, h),
            contour=poly,
            center=cxy,
            shape=f"poly{n_vertices}",
            score=area,
            color=meta.get("color"),
            use=meta.get("use"),
            rule_id=meta.get("rule_id"),
        ))
    print(f"Found {len(out)} polygons.")
    return out


def detect_by_shape(mask: np.ndarray, shape: str, meta: dict) -> List[Candidate]:
    if shape == "circle":
        return detect_circles_from_mask(mask, meta=meta)
    elif shape == "rect":
        return detect_polygons_from_mask(mask, n_vertices=4, meta=meta)
    elif shape.startswith("poly"):
        n = int(shape.replace("poly", ""))
        return detect_polygons_from_mask(mask, n, meta=meta)
    else:
        return []
