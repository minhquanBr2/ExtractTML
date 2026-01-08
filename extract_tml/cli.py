import sys
import os
import json
import cv2
import argparse
from .core import extract_tags


def parse_specs_allowed(s: str) -> list[int]:
    # comma format only: "7,9" (spaces allowed)
    if s is None or not str(s).strip():
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    parser = argparse.ArgumentParser(prog="tml_extract")
    parser.add_argument("input", help='Input image path, e.g. "inputs/your_image.jpg"')
    parser.add_argument("-o", "--output-dir", default="outputs", help='Output dir (default: "outputs")')
    parser.add_argument(
        "--specs-allowed",
        default="7,9",
        help='Comma-separated spec IDs (default: "7,9")',
    )
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug")
    args = parser.parse_args(argv)

    specs_allowed = parse_specs_allowed(args.specs_allowed)


    # ---- read input image
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")


    # ---- extract tags
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    json_path = os.path.join(args.output_dir, f"{base}.json")
    vis_path = os.path.join(args.output_dir, f"{base}.jpg")

    out_json, detections = extract_tags(
        img,
        specs_allowed=specs_allowed,
        debug=args.debug
    )


    # write output JSON
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
