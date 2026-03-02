"""Trapezoid corridor gating for approximate ego-path prediction.

The corridor is a trapezoid in image space:
    - Wide at the bottom  (close range, just in front of the rider)
    - Narrow at the top   (far range, near the horizon)
    - Centred at center_x_ratio * frame_w

Coordinate convention: (0,0) top-left, y increases downward.
Polygon vertex order:  [BL, BR, TR, TL]
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.risk.risk_types import CorridorConfig

logger = logging.getLogger(__name__)


def build_corridor_polygon(
    frame_w: int, frame_h: int, cfg: CorridorConfig
) -> np.ndarray:
    """Build a 4-vertex trapezoid (float32, shape (4,2)) in pixel coordinates.

    Vertices:  BL, BR, TR, TL  (clockwise from bottom-left).
    """
    cx = frame_w * cfg.center_x_ratio
    bot_half = frame_w * cfg.bottom_width_ratio / 2.0
    top_half = frame_w * cfg.top_width_ratio / 2.0
    top_y = float(frame_h) * (1.0 - cfg.height_ratio)
    bot_y = float(frame_h)

    return np.array(
        [
            [cx - bot_half, bot_y],   # BL
            [cx + bot_half, bot_y],   # BR
            [cx + top_half, top_y],   # TR
            [cx - top_half, top_y],   # TL
        ],
        dtype=np.float32,
    )


def _edges_at_y(
    y: float, poly: np.ndarray
) -> Optional[tuple[float, float]]:
    """Interpolated (left_x, right_x) of the corridor at image row y.

    Returns None when y is outside the corridor's vertical span.
    Assumes poly = [BL, BR, TR, TL].
    """
    bot_y = float(poly[0, 1])
    top_y = float(poly[3, 1])

    if y < top_y or y > bot_y:
        return None

    span = bot_y - top_y
    if span < 1e-6:
        return None

    t = (y - top_y) / span   # 0 at top, 1 at bottom

    left_x = poly[3, 0] + t * (poly[0, 0] - poly[3, 0])   # TL → BL
    right_x = poly[2, 0] + t * (poly[1, 0] - poly[2, 0])  # TR → BR

    return left_x, right_x


def point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test (works for convex and concave)."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i, 0]), float(poly[i, 1])
        xj, yj = float(poly[j, 0]), float(poly[j, 1])
        if (yi > y) != (yj > y):
            intersect_x = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < intersect_x:
                inside = not inside
        j = i
    return inside


def corridor_proximity(x: float, y: float, poly: np.ndarray) -> float:
    """Proximity to the corridor centreline, in [0, 1].

    1.0 on the centreline, 0.0 at or beyond the lateral edge, linear in between.
    Returns 0.0 for points outside the vertical span of the corridor.
    """
    edges = _edges_at_y(y, poly)
    if edges is None:
        return 0.0

    left_x, right_x = edges
    half_w = (right_x - left_x) / 2.0
    if half_w < 1e-6:
        return 0.0

    center_x = (left_x + right_x) / 2.0
    dist = abs(x - center_x)
    return max(0.0, 1.0 - dist / half_w)


def corridor_membership(x: float, y: float, poly: np.ndarray) -> float:
    """Soft membership value in [0, 1].

    - 1.0  inside corridor
    - Linearly decays to 0 over one half-width beyond the lateral edge
    - Objects below the bottom of the corridor (very close) get 0.8
    - Objects above the corridor top (beyond horizon) decay to 0
    """
    bot_y = float(poly[0, 1])
    top_y = float(poly[3, 1])

    if y > bot_y:
        return 0.8   # very close — assume in path

    if y < top_y:
        dist_above = top_y - y
        span = bot_y - top_y
        return max(0.0, 1.0 - dist_above / (span + 1e-6))

    edges = _edges_at_y(y, poly)
    if edges is None:
        return 0.0

    left_x, right_x = edges
    half_w = (right_x - left_x) / 2.0
    center_x = (left_x + right_x) / 2.0

    dist = abs(x - center_x)
    if dist <= half_w:
        return 1.0
    excess = dist - half_w
    return max(0.0, 1.0 - excess / (half_w + 1e-6))


def draw_corridor(
    frame: np.ndarray,
    poly: np.ndarray,
    alpha: float = 0.15,
    color: tuple[int, int, int] = (0, 220, 80),
) -> np.ndarray:
    """Overlay a semi-transparent corridor on frame (in-place)."""
    import cv2  # local import to keep corridor.py usable without cv2 in tests

    pts = poly.astype(np.int32).reshape((-1, 1, 2))
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)
    return frame
