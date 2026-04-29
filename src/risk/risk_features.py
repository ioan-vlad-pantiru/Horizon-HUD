"""Physics-inspired feature extraction for RiskEngineV1.

All functions are pure (no side effects) so they are easy to unit-test and
replace independently when better sensors become available.

Coordinate convention (pinhole camera, y-down):
    focal_length_px = frame_w / (2 * tan(fov/2))
    distance_m      = (known_height_m * focal_length_px) / bbox_height_px
    closing_speed   = positive means object is approaching
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

from src.core.types import Track
from src.risk.risk_types import TrackSnapshot


# ── geometry helpers ──────────────────────────────────────────────────────────

def bbox_center(
    bbox_xyxy: tuple[float, float, float, float]
) -> tuple[float, float]:
    return (
        (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0,
        (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0,
    )


def bbox_height(bbox_xyxy: tuple[float, float, float, float]) -> float:
    return max(bbox_xyxy[3] - bbox_xyxy[1], 1.0)


def bbox_width(bbox_xyxy: tuple[float, float, float, float]) -> float:
    return max(bbox_xyxy[2] - bbox_xyxy[0], 1.0)


# ── camera helpers ────────────────────────────────────────────────────────────

def focal_length_from_fov(frame_w: int, fov_deg: float) -> float:
    """Pinhole focal length in pixels from horizontal FOV."""
    return frame_w / (2.0 * math.tan(math.radians(fov_deg / 2.0)))


# ── distance proxy ────────────────────────────────────────────────────────────

def distance_proxy_from_bbox(
    height_px: float,
    known_height_m: float,
    focal_length_px: float,
) -> float:
    """Monocular distance estimate via the pinhole camera model.

    distance_m ≈ (known_height_m * focal_length_px) / height_px

    Reliable when:
      - The object stands roughly upright (height_px ~ projected real height)
      - known_height_m is a reasonable average for the class
    Returns a large number (999) when height_px < 1 to signal invalid input.
    """
    if height_px < 1.0:
        return 999.0
    return (known_height_m * focal_length_px) / height_px


# ── closing speed ─────────────────────────────────────────────────────────────

def closing_speed_foe_px(
    track: Track,
    frame_w: int,
    frame_h: int,
    compensated_vel: Optional[tuple[float, float]] = None,
) -> float:
    """Closing speed in px/s via projection toward the Focus of Expansion.

    The FoE (bottom-centre of frame) is where stationary road features expand
    from as the vehicle moves forward.  A positive dot product means the object
    is moving toward the camera (approaching).
    """
    vx, vy = compensated_vel if compensated_vel is not None else track.velocity_px_s
    cx, cy = bbox_center(track.bbox_xyxy)
    dx = frame_w / 2.0 - cx
    dy = float(frame_h) - cy
    d = math.sqrt(dx * dx + dy * dy)
    if d < 1e-6:
        return 0.0
    return (vx * dx + vy * dy) / d


def closing_speed_from_growth(
    history: deque[TrackSnapshot],
    known_height_m: float,
    focal_length_px: float,
) -> Optional[float]:
    """Closing speed in m/s from the rate of bbox height change.

    From pinhole geometry:
        z = f * H / h   (H = real height, h = pixel height)
        dz/dt = -f * H / h^2 * dh/dt = -(z/h) * dh/dt

    So  v_closing = +z/h * dh/dt   (positive when object grows = approaching)
    """
    if len(history) < 2:
        return None
    newest, oldest = history[-1], history[0]
    dt = newest.timestamp - oldest.timestamp
    if dt < 1e-6:
        return None

    h_new = bbox_height(newest.bbox_xyxy)
    h_old = bbox_height(oldest.bbox_xyxy)
    dh_dt = (h_new - h_old) / dt

    dist_m = distance_proxy_from_bbox(h_new, known_height_m, focal_length_px)
    return dist_m * dh_dt / h_new


def fused_closing_speed_m(
    track: Track,
    history: deque[TrackSnapshot],
    known_height_m: float,
    focal_length_px: float,
    frame_w: int,
    frame_h: int,
    dist_m: float,
    compensated_vel: Optional[tuple[float, float]] = None,
) -> float:
    """Fused closing speed in m/s, blending FoE and bbox-growth methods.

    Both methods are converted to m/s before blending.
    Growth method is weighted higher when the history is long enough.
    """
    cspd_foe_px = closing_speed_foe_px(track, frame_w, frame_h, compensated_vel)
    cspd_foe_m = max(0.0, cspd_foe_px * dist_m / (focal_length_px + 1e-6))

    cspd_growth = closing_speed_from_growth(history, known_height_m, focal_length_px)

    if cspd_growth is not None and len(history) >= 4:
        return 0.4 * cspd_foe_m + 0.6 * max(0.0, cspd_growth)
    return cspd_foe_m


# ── TTC ───────────────────────────────────────────────────────────────────────

def ttc_proxy(
    distance_m: float,
    closing_speed_m_s: float,
    clamp: tuple[float, float] = (0.5, 15.0),
) -> Optional[float]:
    """Time-to-collision proxy (seconds).  Returns None when not approaching."""
    if closing_speed_m_s < 0.05:
        return None
    raw = distance_m / closing_speed_m_s
    return max(clamp[0], min(raw, clamp[1]))


# ── erratic motion ────────────────────────────────────────────────────────────

def erratic_score(history: deque[TrackSnapshot]) -> float:
    """Normalised velocity-magnitude variance, in [0, 1].

    High values indicate sudden acceleration/deceleration.
    A standard deviation of ≥ 80 px/s maps to ~1.0.
    """
    if len(history) < 3:
        return 0.0
    speeds = [
        math.sqrt(s.velocity_px_s[0] ** 2 + s.velocity_px_s[1] ** 2)
        for s in history
    ]
    mean = sum(speeds) / len(speeds)
    variance = sum((s - mean) ** 2 for s in speeds) / len(speeds)
    return min(math.sqrt(variance) / 80.0, 1.0)


# ── lateral risk ─────────────────────────────────────────────────────────────

def lateral_risk_score(
    track: Track,
    corridor_poly: "np.ndarray",
    compensated_vel: Optional[tuple[float, float]] = None,
    lateral_ttc_s: float = 2.0,
) -> float:
    """Score in [0, 1] for lateral motion directed toward the corridor centreline.

    High when: the object is outside the corridor AND moving inward (toward the
    centreline) fast enough to intersect within *lateral_ttc_s* seconds.

    Returns 0 when:
      - The object is already inside the corridor (path_factor handles this).
      - The object is moving away from or parallel to the centreline.
      - The object is outside the corridor's vertical span.

    Parameters
    ----------
    track
        Current track state (bbox and velocity).
    corridor_poly
        4×2 float array [BL, BR, TR, TL] from build_corridor_polygon.
    compensated_vel
        Ego-motion-compensated (vx, vy) in px/s.  Falls back to
        track.velocity_px_s when None.
    lateral_ttc_s
        Time-to-intersect threshold in seconds.  Objects whose lateral
        trajectory would cross the centreline within this window get a
        non-zero score.
    """
    import numpy as _np  # lazy import — keeps risk_features importable without numpy

    vx, _ = compensated_vel if compensated_vel is not None else track.velocity_px_s
    cx = (track.bbox_xyxy[0] + track.bbox_xyxy[2]) / 2.0
    foot_y = float(track.bbox_xyxy[3])

    # ── interpolate corridor edges at foot_y ──────────────────────────────────
    bot_y = float(corridor_poly[0, 1])
    top_y = float(corridor_poly[3, 1])

    if foot_y < top_y or foot_y > bot_y:
        return 0.0

    span = bot_y - top_y
    if span < 1e-6:
        return 0.0

    t = (foot_y - top_y) / span   # 0 at top, 1 at bottom
    left_x = float(corridor_poly[3, 0]) + t * (float(corridor_poly[0, 0]) - float(corridor_poly[3, 0]))
    right_x = float(corridor_poly[2, 0]) + t * (float(corridor_poly[1, 0]) - float(corridor_poly[2, 0]))

    center_x = (left_x + right_x) / 2.0
    half_w = (right_x - left_x) / 2.0

    signed_dist = cx - center_x

    # ── inside corridor: handled by path_factor ───────────────────────────────
    if abs(signed_dist) <= half_w:
        return 0.0

    # ── lateral closing speed toward centreline (positive = approaching) ──────
    lateral_closing = -vx * math.copysign(1.0, signed_dist)

    if lateral_closing <= 0.0:
        return 0.0

    tti = abs(signed_dist) / lateral_closing   # seconds

    if tti >= lateral_ttc_s:
        return 0.0

    return 1.0 - tti / lateral_ttc_s


# ── track confidence ──────────────────────────────────────────────────────────

def confidence_factor(track: Track, min_hits: int = 3) -> float:
    """Scalar in (0, 1] that suppresses score for uncertain tracks.

    - New tracks (few hits) get a reduced confidence.
    - Stale tracks (high time_since_update) get penalised.
    """
    hit_f = min(1.0, 0.4 + 0.6 * max(0, track.hits - 1) / max(min_hits - 1, 1))
    stale_f = max(0.3, 1.0 - track.time_since_update * 2.0)
    return hit_f * stale_f
