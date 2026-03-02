"""Risk assessment engine.

Produces a 0..1 risk score for each tracked object using:
    - class-based base risk (pedestrians highest)
    - distance proxy from bbox height / area
    - closing speed toward bottom-center of image
    - TTC proxy = distance / closing_speed
    - penalties for high lateral motion and erratic acceleration
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from src.core.types import RiskAssessment, Track

logger = logging.getLogger(__name__)

_DEFAULT_BASE_RISK: dict[str, float] = {
    "pedestrian": 0.45,
    "cyclist": 0.35,
    "vehicle": 0.25,
    "road_obstacle": 0.20,
}

RISK_LEVELS = [
    (0.75, "CRITICAL"),
    (0.50, "HIGH"),
    (0.25, "MEDIUM"),
    (0.00, "LOW"),
]


def _level(score: float) -> str:
    for thresh, label in RISK_LEVELS:
        if score >= thresh:
            return label
    return "LOW"


class RiskEngine:
    """Assess risk for every active track.

    Parameters
    ----------
    base_risk : dict[str, float] | None
        Per-class base risk in [0,1].
    ref_height_px : float
        Reference bbox height that maps to ~1.0 proximity factor.
    closing_weight : float
        Weight of closing-speed component.
    lateral_weight : float
        Penalty weight for lateral motion.
    ttc_clamp : tuple[float, float]
        Min / max clamp for TTC seconds.
    """

    def __init__(
        self,
        base_risk: Optional[dict[str, float]] = None,
        ref_height_px: float = 300.0,
        closing_weight: float = 0.30,
        lateral_weight: float = 0.10,
        ttc_clamp: tuple[float, float] = (0.5, 10.0),
    ) -> None:
        self._base = base_risk or _DEFAULT_BASE_RISK
        self._ref_h = ref_height_px
        self._cw = closing_weight
        self._lw = lateral_weight
        self._ttc_clamp = ttc_clamp
        self._prev_vel: dict[int, tuple[float, float]] = {}

    def assess(
        self,
        tracks: list[Track],
        frame_size: tuple[int, int],
        timestamp: float,
    ) -> list[RiskAssessment]:
        fw, fh = frame_size
        anchor_x = fw / 2.0
        anchor_y = float(fh)
        results: list[RiskAssessment] = []

        for tr in tracks:
            reasons: list[str] = []
            x1, y1, x2, y2 = tr.bbox_xyxy
            bh = max(y2 - y1, 1.0)
            bw = max(x2 - x1, 1.0)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            base = self._base.get(tr.label, 0.15)

            proximity = min(bh / self._ref_h, 1.0)
            if proximity > 0.5:
                reasons.append(f"close (h={bh:.0f}px)")

            vx, vy = tr.velocity_px_s
            dx_to_anchor = anchor_x - cx
            dy_to_anchor = anchor_y - cy
            dist_to_anchor = math.sqrt(dx_to_anchor ** 2 + dy_to_anchor ** 2)
            if dist_to_anchor > 1e-3:
                ux = dx_to_anchor / dist_to_anchor
                uy = dy_to_anchor / dist_to_anchor
            else:
                ux, uy = 0.0, 1.0
            closing_speed = vx * ux + vy * uy
            closing_factor = 0.0
            if closing_speed > 0:
                closing_factor = min(closing_speed / 200.0, 1.0)
                reasons.append(f"approaching ({closing_speed:.0f}px/s)")

            ttc: Optional[float] = None
            distance_proxy = max(1.0 - proximity, 0.01)
            if closing_speed > 1.0:
                ttc_raw = (distance_proxy * self._ref_h) / closing_speed
                ttc = max(self._ttc_clamp[0], min(ttc_raw, self._ttc_clamp[1]))
                if ttc < 2.0:
                    reasons.append(f"TTC={ttc:.1f}s")

            lateral_speed = abs(vx * (-uy) + vy * ux)
            lateral_factor = min(lateral_speed / 150.0, 1.0)
            if lateral_factor > 0.3:
                reasons.append("lateral motion")

            accel_penalty = 0.0
            prev = self._prev_vel.get(tr.track_id)
            if prev is not None:
                dvx = vx - prev[0]
                dvy = vy - prev[1]
                accel_mag = math.sqrt(dvx ** 2 + dvy ** 2)
                accel_penalty = min(accel_mag / 300.0, 0.15)
                if accel_penalty > 0.05:
                    reasons.append("erratic accel")
            self._prev_vel[tr.track_id] = (vx, vy)

            raw = (
                base * 0.30
                + proximity * 0.25
                + closing_factor * self._cw
                + lateral_factor * self._lw
                + accel_penalty
            )
            score = max(0.0, min(raw, 1.0))

            results.append(RiskAssessment(
                track_id=tr.track_id,
                risk_score=round(score, 3),
                risk_level=_level(score),
                reasons=reasons if reasons else ["low overall"],
                ttc_s=round(ttc, 2) if ttc is not None else None,
            ))

        active_ids = {tr.track_id for tr in tracks}
        self._prev_vel = {k: v for k, v in self._prev_vel.items() if k in active_ids}

        return results

