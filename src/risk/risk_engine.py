"""RiskEngine v1 — interpretable, stable, cheap ADAS risk scoring.

Scoring pipeline per track
──────────────────────────
1.  Extract physics proxies (distance_m, closing_speed_m_s, TTC).
2.  Score each feature component (each normalised to [0,1]).
3.  Weighted additive combination → raw_score.
4.  Multiply by track-confidence factor to suppress noisy tracks.
5.  EMA smoothing → smooth_score.
6.  Hysteresis level assignment to prevent flicker.
7.  Persistence gate: only escalate to HIGH / CRITICAL after K of last M
    frames exceeded the threshold.
8.  Return top-N hazards by smooth_score.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Optional

import numpy as np

from src.perception.corridor import (
    build_corridor_polygon,
    corridor_membership,
    corridor_proximity,
    point_in_polygon,
)
from src.risk.risk_features import (
    bbox_center,
    bbox_height,
    confidence_factor,
    distance_proxy_from_bbox,
    erratic_score,
    focal_length_from_fov,
    fused_closing_speed_m,
    ttc_proxy,
)
from src.risk.risk_types import (
    CorridorConfig,
    RiskAssessmentV1,
    RiskConfig,
    TrackSnapshot,
)
from src.core.types import Track

logger = logging.getLogger(__name__)

_LEVEL_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
_LEVEL_IDX: dict[str, int] = {l: i for i, l in enumerate(_LEVEL_ORDER)}


# ── component scorers ─────────────────────────────────────────────────────────

def _ttc_risk(ttc: Optional[float], cfg: RiskConfig) -> float:
    """Piecewise-linear TTC → risk mapping in [0, 1]."""
    if ttc is None:
        return 0.0
    bps = [0.0, cfg.ttc_critical_s, cfg.ttc_high_s, cfg.ttc_medium_s, cfg.ttc_max_s]
    vals = [1.0, 1.0, 0.70, 0.30, 0.0]
    if ttc <= bps[0]:
        return vals[0]
    if ttc >= bps[-1]:
        return vals[-1]
    for i in range(len(bps) - 1):
        if bps[i] <= ttc <= bps[i + 1]:
            t = (ttc - bps[i]) / (bps[i + 1] - bps[i] + 1e-12)
            return vals[i] + t * (vals[i + 1] - vals[i])
    return 0.0


def _distance_risk(dist_m: float, cfg: RiskConfig) -> float:
    """Piecewise-linear distance → risk mapping in [0, 1]."""
    if dist_m <= cfg.distance_near_m:
        return 1.0
    if dist_m >= cfg.distance_max_m:
        return 0.0
    return (cfg.distance_max_m - dist_m) / (cfg.distance_max_m - cfg.distance_near_m)


# ── hysteresis ────────────────────────────────────────────────────────────────

def _apply_hysteresis(score: float, current_idx: int, cfg: RiskConfig) -> int:
    """Return the new level index after applying hysteresis.

    Escalation uses enter thresholds (higher), de-escalation uses exit
    thresholds (lower), preventing rapid oscillation near boundaries.
    """
    enter = [0.0, cfg.enter_medium, cfg.enter_high, cfg.enter_critical]
    exit_ = [0.0, cfg.exit_medium, cfg.exit_high, cfg.exit_critical]

    # Highest level that the score qualifies to enter
    target_up = 0
    for lvl in range(1, 4):
        if score >= enter[lvl]:
            target_up = lvl

    if target_up >= current_idx:
        return target_up  # escalation (or hold) via enter thresholds

    # De-escalation: descend one level at a time, checking exit thresholds
    new_lvl = current_idx
    while new_lvl > 0:
        if score < exit_[new_lvl]:
            new_lvl -= 1
        else:
            break
    return new_lvl


# ── persistence gate ──────────────────────────────────────────────────────────

def _persistence_gate(
    desired_lvl: int,
    current_lvl: int,
    score_history: deque[float],
    cfg: RiskConfig,
) -> int:
    """Cap escalation to HIGH/CRITICAL when persistence requirement is not met.

    De-escalation and transitions to LOW/MEDIUM are always immediate.
    """
    if desired_lvl <= current_lvl or desired_lvl < 2:
        return desired_lvl   # de-escalation or LOW/MEDIUM: no persistence needed

    # Walk up from current+1 to desired, checking each escalation step
    approved = current_lvl
    for target in range(current_lvl + 1, desired_lvl + 1):
        if target < 2:
            # MEDIUM escalation never requires persistence
            approved = target
            continue
        thresh = [0.0, 0.0, cfg.enter_high, cfg.enter_critical][target]
        count = sum(1 for s in score_history if s >= thresh)
        if count < cfg.persist_k:
            break   # cannot reach this level yet
        approved = target
    return approved


# ── main engine ───────────────────────────────────────────────────────────────

class RiskEngineV1:
    """ADAS risk engine with physical proxies, EMA, hysteresis, and persistence.

    Parameters
    ----------
    risk_cfg : RiskConfig
        All scoring weights and thresholds.
    corridor_cfg : CorridorConfig
        Trapezoid corridor geometry.
    frame_size : tuple[int, int] | None
        (width, height) — if supplied at construction the corridor polygon is
        pre-built; otherwise it is rebuilt on the first update() call.
    """

    def __init__(
        self,
        risk_cfg: Optional[RiskConfig] = None,
        corridor_cfg: Optional[CorridorConfig] = None,
        frame_size: Optional[tuple[int, int]] = None,
    ) -> None:
        self._cfg = risk_cfg or RiskConfig()
        self._ccfg = corridor_cfg or CorridorConfig()
        self._corridor: Optional[np.ndarray] = None
        if frame_size is not None:
            self._corridor = build_corridor_polygon(*frame_size, self._ccfg)

        self._history: dict[int, deque[TrackSnapshot]] = {}
        self._ema: dict[int, float] = {}
        self._levels: dict[int, int] = {}          # level index per track
        self._score_buf: dict[int, deque[float]] = {}  # EMA scores for persistence

        focal = focal_length_from_fov(640, self._cfg.fov_deg)
        self._focal: float = focal

    # ── public API ────────────────────────────────────────────────────────────

    def update(
        self,
        tracks: list[Track],
        frame_size: tuple[int, int],
        timestamp: float,
        compensated_velocities: Optional[dict[int, tuple[float, float]]] = None,
        corridor_center_x: Optional[float] = None,
    ) -> list[RiskAssessmentV1]:
        """Score all active tracks and return top-N assessments.

        Parameters
        ----------
        tracks
            Active tracks from the SORT tracker.
        frame_size
            (width, height) of the current frame.
        timestamp
            Monotonic timestamp in seconds.
        compensated_velocities
            Optional per-track compensated (ego-motion removed) velocity in
            px/s.  Falls back to track.velocity_px_s when absent.
        corridor_center_x
            When provided, overrides the corridor's horizontal centre so the
            risk-scoring polygon stays in sync with the yaw-steered visual one.
        """
        fw, fh = frame_size
        self._focal = focal_length_from_fov(fw, self._cfg.fov_deg)

        if corridor_center_x is not None:
            self._ccfg.top_center_x_ratio = corridor_center_x
            self._corridor = build_corridor_polygon(fw, fh, self._ccfg)
        elif self._corridor is None or self._corridor[0, 1] != fh:
            self._corridor = build_corridor_polygon(fw, fh, self._ccfg)

        active_ids = {tr.track_id for tr in tracks}
        self._purge(active_ids)

        results: list[RiskAssessmentV1] = []
        for tr in tracks:
            tid = tr.track_id
            comp_vel = (compensated_velocities or {}).get(tid)
            assessment = self._score_track(tr, fw, fh, timestamp, comp_vel)
            results.append(assessment)

        results.sort(key=lambda a: a.risk_score, reverse=True)
        return results[: self._cfg.top_n] if self._cfg.top_n > 0 else results

    # ── per-track scoring ─────────────────────────────────────────────────────

    def _score_track(
        self,
        tr: Track,
        fw: int,
        fh: int,
        timestamp: float,
        comp_vel: Optional[tuple[float, float]],
    ) -> RiskAssessmentV1:
        tid = tr.track_id
        cfg = self._cfg

        # ── history update ────────────────────────────────────────────────────
        if tid not in self._history:
            self._history[tid] = deque(maxlen=cfg.history_len)
            self._score_buf[tid] = deque(maxlen=cfg.persist_m)
        hist = self._history[tid]
        hist.append(TrackSnapshot(
            timestamp=timestamp,
            bbox_xyxy=tr.bbox_xyxy,
            velocity_px_s=tr.velocity_px_s,
        ))

        # ── physical proxies ──────────────────────────────────────────────────
        label = tr.label or "default"
        known_h = cfg.known_heights_m.get(label, cfg.known_heights_m.get("default", 1.5))
        h_px = bbox_height(tr.bbox_xyxy)
        cx, cy = bbox_center(tr.bbox_xyxy)

        dist_m = distance_proxy_from_bbox(h_px, known_h, self._focal)
        dist_m = min(dist_m, cfg.distance_max_m)

        cspd_m = fused_closing_speed_m(
            tr, hist, known_h, self._focal, fw, fh, dist_m, comp_vel
        )
        ttc = ttc_proxy(dist_m, cspd_m, (0.5, cfg.ttc_max_s))

        # ── corridor ──────────────────────────────────────────────────────────
        foot_x = cx
        foot_y = float(tr.bbox_xyxy[3])   # bottom of bbox = "feet"
        poly = self._corridor
        prox = corridor_proximity(foot_x, foot_y, poly)
        memb = corridor_membership(foot_x, foot_y, poly)
        in_corr = point_in_polygon(foot_x, foot_y, poly)
        path_factor = prox * memb

        # ── component scores ──────────────────────────────────────────────────
        r_ttc = _ttc_risk(ttc, cfg)
        r_dist = _distance_risk(dist_m, cfg)
        r_class = cfg.class_prior.get(label, cfg.class_prior.get("default", 0.5))
        r_erratic = erratic_score(hist)
        conf = confidence_factor(tr)

        raw = (
            cfg.w_ttc * r_ttc
            + cfg.w_distance * r_dist
            + cfg.w_path * path_factor
            + cfg.w_class * r_class
            + cfg.w_erratic * r_erratic
        )
        raw = max(0.0, min(raw * conf, 1.0))

        # ── EMA smoothing ─────────────────────────────────────────────────────
        prev_ema = self._ema.get(tid, raw)
        ema = cfg.ema_alpha * raw + (1.0 - cfg.ema_alpha) * prev_ema
        self._ema[tid] = ema

        # ── hysteresis ────────────────────────────────────────────────────────
        cur_idx = self._levels.get(tid, 0)
        new_idx = _apply_hysteresis(ema, cur_idx, cfg)

        # ── persistence gate ──────────────────────────────────────────────────
        self._score_buf[tid].append(ema)
        final_idx = _persistence_gate(new_idx, cur_idx, self._score_buf[tid], cfg)
        self._levels[tid] = final_idx

        level = _LEVEL_ORDER[final_idx]

        # ── reasons ───────────────────────────────────────────────────────────
        reasons = self._build_reasons(
            dist_m, cspd_m, ttc, in_corr, prox, tr, r_erratic, cfg
        )

        return RiskAssessmentV1(
            track_id=tid,
            risk_score=round(ema, 3),
            risk_level=level,
            ttc_s=round(ttc, 2) if ttc is not None else None,
            distance_m=round(dist_m, 1),
            reasons=reasons,
            in_corridor=in_corr,
        )

    # ── reasons builder ───────────────────────────────────────────────────────

    @staticmethod
    def _build_reasons(
        dist_m: float,
        cspd_m: float,
        ttc: Optional[float],
        in_corr: bool,
        prox: float,
        tr: Track,
        r_erratic: float,
        cfg: RiskConfig,
    ) -> list[str]:
        reasons: list[str] = []
        if in_corr or prox > 0.5:
            reasons.append("in path")
        if dist_m < cfg.distance_near_m * 1.5:
            reasons.append(f"very close ({dist_m:.0f}m)")
        if ttc is not None and ttc <= cfg.ttc_high_s:
            reasons.append(f"low TTC ({ttc:.1f}s)")
        if cspd_m > 1.0:
            reasons.append(f"closing ({cspd_m:.1f}m/s)")
        if r_erratic > 0.4:
            reasons.append("erratic")
        if tr.hits < 3 or tr.time_since_update > 0.3:
            reasons.append("unstable track")
        return reasons if reasons else ["low risk"]

    # ── housekeeping ──────────────────────────────────────────────────────────

    def _purge(self, active_ids: set[int]) -> None:
        for tid in list(self._history):
            if tid not in active_ids:
                del self._history[tid]
                self._ema.pop(tid, None)
                self._levels.pop(tid, None)
                self._score_buf.pop(tid, None)

    # ── config helpers ────────────────────────────────────────────────────────

    @property
    def corridor_polygon(self) -> Optional[np.ndarray]:
        return self._corridor


def risk_config_from_dict(d: dict) -> RiskConfig:
    """Build a RiskConfig from the 'risk' section of config.yaml."""
    kh_raw = d.get("known_heights_m", {})
    kh = {str(k): float(v) for k, v in kh_raw.items()} if kh_raw else None
    cp_raw = d.get("class_prior", {})
    cp = {str(k): float(v) for k, v in cp_raw.items()} if cp_raw else None

    kwargs: dict = {
        k: d[k] for k in (
            "w_ttc", "w_distance", "w_path", "w_class", "w_erratic",
            "ttc_critical_s", "ttc_high_s", "ttc_medium_s", "ttc_max_s",
            "fov_deg",
            "distance_near_m", "distance_max_m",
            "ema_alpha",
            "enter_medium", "exit_medium",
            "enter_high", "exit_high",
            "enter_critical", "exit_critical",
            "persist_k", "persist_m",
            "top_n", "history_len",
        ) if k in d
    }
    if kh:
        kwargs["known_heights_m"] = kh
    if cp:
        kwargs["class_prior"] = cp
    return RiskConfig(**kwargs)


def corridor_config_from_dict(d: dict) -> CorridorConfig:
    """Build a CorridorConfig from the 'corridor' section of config.yaml."""
    kwargs = {
        k: d[k] for k in (
            "bottom_width_ratio", "top_width_ratio",
            "height_ratio", "center_x_ratio", "top_center_x_ratio", "yaw_gain",
        ) if k in d
    }
    return CorridorConfig(**kwargs)
