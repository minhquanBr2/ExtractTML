from typing import List, Optional, Dict, Any, Tuple
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
    # area limits are very data-dependent; make max_area big enough
    min_area: int = 400,
    max_area: int = 5000,
    circularity_min: float = 0.2,
    circularity_max: float = 1.5,
    # radius constraints (pixels) - tune these from your drawings
    min_radius: int = 10,
    max_radius: int = 160,
    # suppress duplicates / inner junk
    center_merge_dist: float = 10.0,
    # preprocessing
    close_ksize: int = 5,
    close_iters: int = 2,
) -> List[Candidate]:
    meta = meta or {}
    out: List[Candidate] = []

    # --- 1) Preprocess to make thin/broken rings continuous
    m = mask.copy().astype(np.uint8)
    if close_ksize and close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=close_iters)         # Bridges small gaps in the ring, reconnect broken arcs, makes the outer circle a single contour

    # Optional: remove tiny specks that become "circles"
    # (only if you have this helper; otherwise comment out)
    # m = remove_small_blobs(m, min_area=50)

    # --- 2) Prefer hierarchy so we can avoid “inner text” loops
    # Use RETR_TREE to get parent/child relation
    cnts, hier = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []

    hier = hier[0]  # shape (N, 4) [next, prev, child, parent]

    candidates_tmp: List[Tuple[float, float, float, float, Tuple[int,int,int,int], np.ndarray]] = []
    # (cx, cy, r, score, bbox, contour)

    for i, cnt in enumerate(cnts):
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue

        peri = float(cv2.arcLength(cnt, True))
        if peri <= 1e-6:
            continue

        circ = (4.0 * np.pi * area) / (peri * peri)
        if not (circularity_min <= circ <= circularity_max):
            continue

        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if r < min_radius or r > max_radius:                # reject junk circles like small text loops, inner holes
            continue

        # --- 3) Prefer “outer-ish” contours:
        # If this contour has a parent, it might be inner junk.
        # We won't strictly remove them (broken rings can create weird hierarchy),
        # but we down-weight them.
        parent = hier[i][3]
        parent_penalty = 0.85 if parent != -1 else 1.0

        # Score: prefer bigger circles (r) and good circularity
        score = (r * r) * circ * parent_penalty

        # Bounding box from enclosing circle (better than boundingRect for circles)
        x = int(round(cx - r))
        y = int(round(cy - r))
        w = int(round(2 * r))
        h = int(round(2 * r))

        candidates_tmp.append((cx, cy, r, score, (x, y, w, h), cnt))

    # --- 4) Suppress inner duplicates: keep biggest (by radius/score) near same center
    # Sort large-first so outer rings win
    candidates_tmp.sort(key=lambda t: (t[2], t[3]), reverse=True)

    kept: List[Tuple[float, float, float, float, Tuple[int,int,int,int], np.ndarray]] = []
    for cx, cy, r, score, bbox, cnt in candidates_tmp:
        keep = True
        for kcx, kcy, kr, kscore, kbbox, kcnt in kept:
            dx = cx - kcx
            dy = cy - kcy
            if (dx * dx + dy * dy) ** 0.5 < center_merge_dist:
                # same center neighborhood -> keep the bigger one (we sorted big-first)
                keep = False
                break
        if keep:
            kept.append((cx, cy, r, score, bbox, cnt))

    # --- 5) Build Candidates
    for cx, cy, r, score, (x, y, w, h), cnt in kept:
        out.append(Candidate(
            bbox=(x, y, w, h),
            contour=cnt,
            center=(float(cx), float(cy)),
            shape="circle",
            score=float(score),
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
    max_area: int = 5000,
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
