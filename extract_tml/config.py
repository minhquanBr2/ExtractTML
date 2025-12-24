from typing import Dict, List, Tuple

# Each entry: [ ((Hmin,Hmax),(Smin,Smax),(Vmin,Vmax)), ... ]
HSV_RANGES: Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]] = {
    "red": [
        ((0, 10), (40, 255), (40, 255)),
        ((170, 180), (40, 255), (40, 255)),
    ],
    "orange": [((10, 25), (50, 255), (50, 255))],
    "purple": [((125, 155), (40, 255), (40, 255))],
    "green": [((35, 95), (40, 255), (40, 255))],
    "yellow": [((18, 40), (40, 255), (60, 255))],
    "black": [((0, 180), (0, 255), (0, 60))],  # hue any; low V
}

SHAPE_THRESH = {
    "min_area": 200,
    "max_area": 250000,
    "circle_circularity_min": 0.72,
    "circle_circularity_max": 1.25,
    "rect_angle_tol_deg": 20,
    "poly_eps_ratio": 0.02,
    "yellow_fill_ratio_min": 0.60,
}

OCR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TAG_REGEX_LIST = [
    r"TML\s*\d+[A-Z]?",   # TML1 / TML 1 / TML12A
    r"\d+[A-Z]?",         # 15A / 20 / 3
    r"[A-Z]\d+[A-Z]?",    # E3 / A12
]

# rule spec: (shape, color, fill/stroke) -> rule_id
RULE_SPEC = {
    1:  dict(shape="circle", color="red",    use="stroke"),
    2:  dict(shape="circle", color="purple", use="stroke"),
    3:  dict(shape="circle", color="green",  use="stroke"),
    4:  dict(shape="circle", color="orange", use="stroke"),
    5:  dict(shape="circle", color="black",  use="stroke"),

    6:  dict(shape="rect",   color="red",    use="stroke"),
    7:  dict(shape="rect",   color="black",  use="stroke"),
    8:  dict(shape="rect",   color="green",  use="stroke"),

    9:  dict(shape="rect",   color="yellow", use="fill"),

    10: dict(shape="poly4",  color="green",  use="stroke"),
    11: dict(shape="poly5",  color="green",  use="stroke"),
    12: dict(shape="poly6",  color="green",  use="stroke"),
}
