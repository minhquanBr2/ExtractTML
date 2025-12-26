from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class Candidate:
    bbox: Tuple[int, int, int, int]          # x,y,w,h
    shape: str                               # "circle" | "rect" | "polyN"
    score: float = 0.0

    # metadata stamped by rule engine (optional but useful)
    contour: Optional[np.ndarray] = None  # contour points
    center: Optional[Tuple[float, float]] = None              # cx,cy
    color: Optional[str] = None              # "red"|"yellow"|...
    use: Optional[str] = None                # "stroke"|"fill"
    rule_id: Optional[int] = None
