from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class Candidate:
    bbox: Tuple[int, int, int, int]          # x,y,w,h
    contour: np.ndarray
    center: Tuple[float, float]              # cx,cy
    shape: str                               # "circle" | "rect" | "polyN"
    score: float = 0.0

    # metadata stamped by rule engine (optional but useful)
    color: Optional[str] = None              # "red"|"yellow"|...
    use: Optional[str] = None                # "stroke"|"fill"
    rule_id: Optional[int] = None
