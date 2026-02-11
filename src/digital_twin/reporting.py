import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Hotspot:
    position: Tuple[float, float]
    score: float


def compute_hotspots(heat: np.ndarray, min_xy: np.ndarray, grid_size: float, k: int = 5) -> List[Hotspot]:
    flat = heat.flatten()
    if flat.size == 0:
        return []
    idx = np.argpartition(-flat, min(k, flat.size - 1))[:k]
    hotspots: List[Hotspot] = []
    h, w = heat.shape
    for i in idx:
        y = i // w
        x = i % w
        score = float(heat[y, x])
        pos = (float(min_xy[0] + x * grid_size), float(min_xy[1] + y * grid_size))
        hotspots.append(Hotspot(position=pos, score=score))
    hotspots.sort(key=lambda h: h.score, reverse=True)
    return hotspots


def save_report_json(
    out_path: Path,
    metadata: Dict,
    hotspots: List[Hotspot],
) -> None:
    payload = {
        "metadata": metadata,
        "hotspots": [
            {"position": [h.position[0], h.position[1]], "score": h.score} for h in hotspots
        ],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
