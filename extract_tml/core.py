from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np
import cv2
from collections import defaultdict

from .ocr import build_ocr
from .config import HSV_RANGES, RULE_SPEC, SHAPE_THRESH
from .preprocess import denoise_light, maybe_resize, threshold_hsv, morph_close_open, remove_small_blobs, split_stroke_fill
from .detect import detect_by_shape
from .nms import nms, nms_keep_smaller
from .postprocess import crop_with_padding, ocr_preprocess, extract_tag_from_text, normalize_tag
from .visualize import draw_bboxes_on_image, visualize_color_masks

from .types import Candidate


def resolve_rule_id(c: Candidate) -> Optional[int]:
    # find a matching rule id from RULE_SPEC
    for rid, spec in RULE_SPEC.items():
        if spec.get("shape") == c.shape and spec.get("color") == c.color and spec.get("use") == c.use:
            return rid
    return None


def dedup_results(items: List[Dict[str, Any]], dist_thresh: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        keep = True
        for ex in out:
            dx = it["center_x"] - ex["center_x"]
            dy = it["center_y"] - ex["center_y"]
            if (dx * dx + dy * dy) ** 0.5 < dist_thresh:
                keep = False
                break
        if keep:
            out.append(it)
    return out


def extract_tags(
        image_bgr: np.ndarray,
        debug: bool = False,
        return_detections: bool = False
    ) -> Dict[str, Any]:
    ocr_fn = build_ocr()

    H0, W0 = image_bgr.shape[:2]

    img = denoise_light(image_bgr)
    img, scale = maybe_resize(img, max_side=2200)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ---- build mask bank
    mask_bank = {}
    for color, ranges in HSV_RANGES.items():
        raw0 = threshold_hsv(hsv, ranges)
        raw1 = morph_close_open(raw0, k=3)
        raw = remove_small_blobs(raw1, min_area=SHAPE_THRESH["min_area"])

        stroke, fill = split_stroke_fill(raw, ksize=3)
        mask_bank[color] = {"raw": raw, "stroke": raw0, "fill": fill}
        if color == "red":
            visualize_color_masks({
                "raw0": raw0,
                "raw1": raw1,
                "raw": raw,
                "stroke": stroke,
                "fill": fill
            })

    # ---- detect by rules (rule stamps meta)
    candidates: List[Candidate] = []
    for rule_id, spec in RULE_SPEC.items():
        print(f"Checking rule {rule_id} with spec: {spec}")
        color = spec["color"]
        shape = spec["shape"]
        use = spec["use"]  # "stroke" | "fill"

        mask = mask_bank[color][use]
        meta = {"color": color, "use": use, "rule_id": rule_id}

        candidates.extend(detect_by_shape(mask, shape, meta))
    print(f"\nFounding {len(candidates)} candidates in total.")

    # ---- keep only candidates that already have rule_id
    valid = [c for c in candidates if c.rule_id is not None]
    print(f"\nFounding {len(valid)} valid candidates in total.")

    # --- VISUALIZE BEFORE NMS
    if debug:
        draw_bboxes_on_image(
            img_bgr=img,
            candidates=valid,
            title="Before NMS (all valid candidates)"
        )

    # ---- NMS per rule (avoid suppressing across different rules)
    grouped = defaultdict(list)
    for c in valid:
        grouped[c.rule_id].append(c)
    print(f"\nPrinting groups...")
    for rule_id in sorted(grouped.keys()):
        print(f"rule_id = {rule_id}:")
        print(f"Number of candidates for this rule: {len(grouped[rule_id])}")
        print()

    valid_after_nms = []
    for _, group in grouped.items():
        valid_after_nms.extend(nms_keep_smaller(group, iou_thresh=0.1))

    # --- VISUALIZE AFTER NMS
    if debug:
        draw_bboxes_on_image(
            img_bgr=img,
            candidates=valid_after_nms,
            title="After NMS (per rule)"
        )

    # ---- OCR + output
    results: List[Dict[str, Any]] = []

    for i, c in enumerate(valid_after_nms):
        roi = crop_with_padding(img, c.bbox, pad_ratio=0.25)

        if debug:
            import matplotlib.pyplot as plt
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            x, y, w, h = c.bbox
            plt.figure(figsize=(4, 4))
            plt.imshow(roi_rgb)
            plt.title(
                f"#{i} rule={c.rule_id} {c.color}/{c.use} {c.shape}\n"
                f"bbox=({x},{y},{w},{h})"
            )
            plt.axis("off")
            plt.show()

        raw_text, conf = ocr_fn(roi)
        tag = extract_tag_from_text(raw_text)
        if not tag:
            continue
        tag = normalize_tag(tag)

        cx_r, cy_r = c.center
        x_r, y_r, w_r, h_r = c.bbox

        # ---- map from resized image -> original image
        if scale != 1.0:
            cx_orig, cy_orig = cx_r / scale, cy_r / scale
            x0_orig, y0_orig, w0_orig, h0_orig = int(round(x_r / scale)), int(round(y_r / scale)), int(round(w_r / scale)), int(round(h_r / scale))
        else:
            cx_orig, cy_orig = cx_r, cy_r
            x0_orig, y0_orig, w0_orig, h0_orig = int(round(x_r)), int(round(y_r)), int(round(w_r)), int(round(h_r))
        # x1_orig = x0_orig + w0_orig
        # y1_orig = y0_orig + h0_orig

        # ---- normalized to [0..1000]
        cx_norm = int(np.clip(round(cx_orig / W0 * 1000.0), 0, 1000))
        cy_norm = int(np.clip(round(cy_orig / H0 * 1000.0), 0, 1000))
        x0_norm = int(np.clip(round(x0_orig / W0 * 1000.0), 0, 1000))
        y0_norm = int(np.clip(round(y0_orig / H0 * 1000.0), 0, 1000))
        w0_norm = int(np.clip(round(w0_orig / W0 * 1000.0), 0, 1000))
        h0_norm = int(np.clip(round(h0_orig / H0 * 1000.0), 0, 1000))

        # if w0_norm * h0_norm > 1000:
        #     continue

        bbox_orig = [x0_orig, y0_orig, w0_orig, h0_orig]
        bbox_norm = [x0_norm, y0_norm, w0_norm, h0_norm]

        results.append({"tag_id": tag, "center_x": cx_orig, "center_y": cy_orig, "bbox_orig": bbox_orig, "bbox_norm": bbox_norm})

    print(f"Number of boxes after OCR: {len(results)}")

    results = dedup_results(results, dist_thresh=10)
    results = sorted(results, key=lambda x: x["tag_id"])

    detections = []
    for r in results:
        detections.append({
            "tag_id": r["tag_id"],
            "center_x": r["center_x"],
            "center_y": r["center_y"],
            "bbox_orig": r["bbox_orig"]
        })

    print(f"Number of boxes after deduplication: {len(results)}")

    out_json = {"extracted_tml_tags": results}
    if return_detections:
        return out_json, detections
    return out_json
