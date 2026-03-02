"""Shared dataclasses for RiskEngine v1.

All weights and thresholds that control scoring are collected in RiskConfig so
they can be loaded from docs/config.yaml and swapped without touching logic.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import Optional


@dataclasses.dataclass
class RiskAssessmentV1:
    track_id: int
    risk_score: float          # EMA-smoothed, clamped [0, 1]
    risk_level: str            # LOW | MEDIUM | HIGH | CRITICAL
    ttc_s: Optional[float]     # time-to-collision proxy in seconds
    distance_m: Optional[float]
    reasons: list[str]
    in_corridor: bool


@dataclasses.dataclass
class CorridorConfig:
    """Trapezoid corridor definition in normalised image coordinates."""
    bottom_width_ratio: float = 0.55   # fraction of frame width at bottom edge
    top_width_ratio: float = 0.18      # fraction of frame width at horizon edge
    height_ratio: float = 0.55         # fraction of frame height the corridor spans (from bottom up)
    center_x_ratio: float = 0.50       # bottom centre (fixed, road straight ahead)
    top_center_x_ratio: float = 0.50   # top centre (shifts with yaw — vanishing point)
    yaw_gain: float = 0.50             # screen-fraction shift per radian of yaw delta


@dataclasses.dataclass
class RiskConfig:
    """All tunable parameters for RiskEngineV1."""

    # ── scoring weights (should sum to ~1.0) ──────────────────────────────────
    w_ttc: float = 0.40
    w_distance: float = 0.25
    w_path: float = 0.20
    w_class: float = 0.10
    w_erratic: float = 0.05

    # ── TTC breakpoints (seconds) ─────────────────────────────────────────────
    # Piecewise linear: 1.0 at <=critical, 0.7 at high, 0.3 at medium, 0.0 at max
    ttc_critical_s: float = 1.5
    ttc_high_s: float = 3.0
    ttc_medium_s: float = 6.0
    ttc_max_s: float = 12.0

    # ── distance calibration ──────────────────────────────────────────────────
    fov_deg: float = 70.0
    known_heights_m: dict[str, float] = dataclasses.field(default_factory=lambda: {
        "pedestrian": 1.75,
        "cyclist": 1.70,
        "vehicle": 1.50,
        "road_obstacle": 0.40,
        "default": 1.50,
    })
    distance_near_m: float = 5.0    # full distance risk at or below this
    distance_max_m: float = 40.0    # zero distance risk at or beyond this

    # ── class priors [0, 1] ───────────────────────────────────────────────────
    class_prior: dict[str, float] = dataclasses.field(default_factory=lambda: {
        "pedestrian": 1.00,
        "cyclist": 0.85,
        "vehicle": 0.65,
        "road_obstacle": 0.45,
        "default": 0.50,
    })

    # ── EMA smoothing ─────────────────────────────────────────────────────────
    ema_alpha: float = 0.40        # higher = more responsive, less smooth

    # ── hysteresis enter / exit thresholds ────────────────────────────────────
    enter_medium: float = 0.30
    exit_medium: float = 0.25
    enter_high: float = 0.55
    exit_high: float = 0.50
    enter_critical: float = 0.80
    exit_critical: float = 0.75

    # ── persistence (escalate to HIGH/CRITICAL only if K of last M frames qualify) ──
    persist_k: int = 3
    persist_m: int = 5

    # ── output ────────────────────────────────────────────────────────────────
    top_n: int = 2             # surface only top-N hazards prominently

    # ── per-track history depth ───────────────────────────────────────────────
    history_len: int = 8


@dataclasses.dataclass
class TrackSnapshot:
    """Lightweight record of a track at one time step, used for history smoothing."""
    timestamp: float
    bbox_xyxy: tuple[float, float, float, float]
    velocity_px_s: tuple[float, float]
    raw_score: float = 0.0
