from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import cv2
from .types import Candidate


def visualize_color_masks(
    masks: Dict[str, np.ndarray],
    cols: int = 3,
    figsize: tuple = (12, 8),
    title: str = "HSV Color Masks"
):
    color_names = list(masks.keys())
    n = len(color_names)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for ax in axes[n:]:
        ax.axis("off")

    for i, color in enumerate(color_names):
        ax = axes[i]
        ax.imshow(masks[color], cmap="gray")
        ax.set_title(color)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def draw_detections_on_image(image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    vis = image_bgr.copy()

    for det in detections:
        tag = det["tag_id"]
        x, y, w, h = det["bbox_orig"]
        x2, y2 = x + w, y + h

        cv2.rectangle(vis, (x, y), (x2, y2), (0, 255, 0), 2)
        label = tag

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y_text = max(0, y - 8)
        cv2.rectangle(vis, (x, y_text - th - 6), (x + tw + 6, y_text), (0, 255, 0), -1)
        cv2.putText(vis, label, (x + 3, y_text - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    return vis


def draw_bboxes_on_image(
    img_bgr: np.ndarray,
    candidates: List[Candidate],
    title: str,
    max_boxes: Optional[int] = None,
) -> None:
    vis = img_bgr.copy()
    items = candidates if max_boxes is None else candidates[:max_boxes]

    for c in items:
        x, y, w, h = c.bbox
        x2, y2 = x + w, y + h
        cv2.rectangle(vis, (x, y), (x2, y2), (0, 255, 0), 2)
        label = f"r{c.rule_id}:{c.color}/{c.use}"
        cv2.putText(
            vis,
            label,
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(14, 10))
    plt.imshow(vis_rgb)
    plt.title(f"{title} (count={len(candidates)})")
    plt.axis("off")
    plt.show()
