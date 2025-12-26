import sys
import os
import json
import cv2
from .core import extract_tags


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if len(argv) < 2:
        print('Usage: python tml_extract.py "inputs/your_image.jpg"')
        sys.exit(2)

    in_path = argv[1]

    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {in_path}")

    os.makedirs("outputs", exist_ok=True)
    base = os.path.splitext(os.path.basename(in_path))[0]
    json_path = os.path.join("outputs", f"{base}.json")
    vis_path = os.path.join("outputs", f"{base}.jpg")

    out_json, detections = extract_tags(img, debug=True, return_detections=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote JSON to: {json_path}")

    # draw + save visualization image
    from .visualize import draw_detections_on_image
    vis = draw_detections_on_image(img, detections)
    ok = cv2.imwrite(vis_path, vis)
    if not ok:
        raise RuntimeError(f"Failed to write image: {vis_path}")
    print(f"[OK] Wrote visualization to: {vis_path}")


if __name__ == "__main__":
    main()
