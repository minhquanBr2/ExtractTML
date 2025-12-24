from typing import List, Tuple
from .types import Candidate


def iou_xywh(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    union = float(aw * ah + bw * bh) - inter
    return inter / union if union > 0 else 0.0


def nms(cands: List[Candidate], iou_thresh: float = 0.4) -> List[Candidate]:
    if not cands:
        return []
    cands = sorted(cands, key=lambda c: c.score, reverse=True)
    keep: List[Candidate] = []
    for c in cands:
        if all(iou_xywh(c.bbox, k.bbox) < iou_thresh for k in keep):
            keep.append(c)
    return keep


def nms_keep_smaller(cands: List[Candidate], iou_thresh: float = 0.4) -> List[Candidate]:
    if not cands:
        return []

    def area_xywh(b):
        _, _, w, h = b
        return float(w) * float(h)

    # smallest first
    cands = sorted(cands, key=lambda c: area_xywh(c.bbox))

    keep: List[Candidate] = []
    for c in cands:
        if all(iou_xywh(c.bbox, k.bbox) < iou_thresh for k in keep):
            keep.append(c)
    return keep
