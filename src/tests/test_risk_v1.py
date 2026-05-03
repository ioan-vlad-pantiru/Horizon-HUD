"""Tests for RiskEngine v1: monotonicity, hysteresis, corridor, and feature math."""

from __future__ import annotations

import math
import time
import unittest
from collections import deque

from src.perception.corridor import (
    build_corridor_polygon,
    corridor_membership,
    corridor_proximity,
    point_in_polygon,
)
from src.risk.risk_engine import (
    RiskEngineV1,
    _apply_hysteresis,
    _persistence_gate,
    _ttc_risk,
    _distance_risk,
)
from src.risk.risk_features import (
    bbox_height,
    confidence_factor,
    distance_proxy_from_bbox,
    erratic_score,
    focal_length_from_fov,
    ttc_proxy,
)
from src.risk.risk_types import CorridorConfig, RiskConfig, TrackSnapshot
from src.core.types import Track


# ── helpers ───────────────────────────────────────────────────────────────────

def _track(
    tid: int = 1,
    bbox: tuple = (250, 200, 390, 380),
    vel: tuple = (0.0, 0.0),
    label: str = "vehicle",
    hits: int = 5,
    tsu: float = 0.0,
) -> Track:
    return Track(
        track_id=tid,
        bbox_xyxy=bbox,
        velocity_px_s=vel,
        age=hits,
        time_since_update=tsu,
        hits=hits,
        label=label,
        class_id=0,
    )


def _engine(cfg: RiskConfig | None = None, ccfg: CorridorConfig | None = None) -> RiskEngineV1:
    return RiskEngineV1(cfg or RiskConfig(), ccfg or CorridorConfig())


# ── feature math tests ────────────────────────────────────────────────────────

class TestFeatureMath(unittest.TestCase):

    def test_focal_length_fov90(self) -> None:
        fl = focal_length_from_fov(1000, 90.0)
        self.assertAlmostEqual(fl, 500.0, places=3)

    def test_distance_proxy_inverse(self) -> None:
        fl = focal_length_from_fov(640, 70.0)
        d1 = distance_proxy_from_bbox(100, 1.75, fl)
        d2 = distance_proxy_from_bbox(50,  1.75, fl)
        self.assertGreater(d2, d1)   # smaller bbox = farther

    def test_distance_proxy_invalid(self) -> None:
        self.assertEqual(distance_proxy_from_bbox(0, 1.75, 600), 999.0)

    def test_ttc_none_when_not_approaching(self) -> None:
        self.assertIsNone(ttc_proxy(10.0, 0.0))
        self.assertIsNone(ttc_proxy(10.0, -1.0))

    def test_ttc_clamped(self) -> None:
        t = ttc_proxy(100.0, 0.1, (0.5, 15.0))
        self.assertEqual(t, 15.0)

    def test_erratic_zero_short_history(self) -> None:
        h: deque[TrackSnapshot] = deque(maxlen=8)
        h.append(TrackSnapshot(0.0, (0,0,100,200), (10.0, 5.0)))
        self.assertEqual(erratic_score(h), 0.0)

    def test_erratic_high_variance(self) -> None:
        # Speed alternates 500/0 px/s: high speed variance → speed_component=1.0,
        # but heading is constant (all vx≥0) → heading_component=0.
        # Blended score = 0.5 * 1.0 + 0.5 * 0.0 = 0.5.
        h: deque[TrackSnapshot] = deque(maxlen=8)
        for i in range(6):
            v = 500.0 if i % 2 == 0 else 0.0
            h.append(TrackSnapshot(float(i), (0,0,100,200), (v, 0.0)))
        self.assertGreaterEqual(erratic_score(h), 0.5)

    def test_confidence_new_track(self) -> None:
        tr = _track(hits=1, tsu=0.0)
        self.assertLess(confidence_factor(tr), 0.5)

    def test_confidence_confirmed_track(self) -> None:
        tr = _track(hits=8, tsu=0.0)
        self.assertGreater(confidence_factor(tr), 0.9)


# ── TTC and distance risk components ─────────────────────────────────────────

class TestComponentScorers(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = RiskConfig()

    def test_ttc_risk_decreases_with_higher_ttc(self) -> None:
        r1 = _ttc_risk(1.0, self.cfg)
        r2 = _ttc_risk(3.5, self.cfg)
        r3 = _ttc_risk(8.0, self.cfg)
        self.assertGreater(r1, r2)
        self.assertGreater(r2, r3)

    def test_ttc_risk_critical_is_max(self) -> None:
        self.assertEqual(_ttc_risk(0.5, self.cfg), 1.0)
        self.assertEqual(_ttc_risk(1.5, self.cfg), 1.0)

    def test_ttc_risk_beyond_max_is_zero(self) -> None:
        self.assertEqual(_ttc_risk(100.0, self.cfg), 0.0)

    def test_ttc_risk_none_is_zero(self) -> None:
        self.assertEqual(_ttc_risk(None, self.cfg), 0.0)

    def test_distance_risk_near_is_one(self) -> None:
        self.assertEqual(_distance_risk(1.0, self.cfg), 1.0)

    def test_distance_risk_far_is_zero(self) -> None:
        self.assertEqual(_distance_risk(50.0, self.cfg), 0.0)

    def test_distance_risk_monotone(self) -> None:
        r5 = _distance_risk(5.0,  self.cfg)
        r15 = _distance_risk(15.0, self.cfg)
        r30 = _distance_risk(30.0, self.cfg)
        self.assertGreater(r5, r15)
        self.assertGreater(r15, r30)


# ── corridor tests ────────────────────────────────────────────────────────────

class TestCorridor(unittest.TestCase):

    def setUp(self) -> None:
        self.fw, self.fh = 640, 480
        self.ccfg = CorridorConfig()
        self.poly = build_corridor_polygon(self.fw, self.fh, self.ccfg)

    def test_polygon_shape(self) -> None:
        self.assertEqual(self.poly.shape, (4, 2))

    def test_center_bottom_in_polygon(self) -> None:
        # Bottom-center of frame should be inside the corridor
        self.assertTrue(point_in_polygon(320, 460, self.poly))

    def test_far_left_not_in_polygon(self) -> None:
        self.assertFalse(point_in_polygon(10, 400, self.poly))

    def test_far_right_not_in_polygon(self) -> None:
        self.assertFalse(point_in_polygon(630, 400, self.poly))

    def test_proximity_centerline_is_one(self) -> None:
        cx = int(self.fw * 0.5)
        cy = int(self.fh * 0.7)
        p = corridor_proximity(float(cx), float(cy), self.poly)
        self.assertAlmostEqual(p, 1.0, places=2)

    def test_proximity_monotone_lateral(self) -> None:
        """Moving laterally away from the centerline reduces proximity."""
        cy = float(self.fh * 0.7)
        cx = self.fw * 0.5
        p_center = corridor_proximity(cx, cy, self.poly)
        p_left = corridor_proximity(cx - 60, cy, self.poly)
        p_far_left = corridor_proximity(cx - 150, cy, self.poly)
        self.assertGreater(p_center, p_left)
        self.assertGreater(p_left, p_far_left)

    def test_proximity_zero_outside_vertical_span(self) -> None:
        # Above the corridor top
        p = corridor_proximity(320.0, 10.0, self.poly)
        self.assertEqual(p, 0.0)

    def test_membership_inside_is_one(self) -> None:
        m = corridor_membership(320.0, float(self.fh * 0.7), self.poly)
        self.assertAlmostEqual(m, 1.0, places=2)

    def test_membership_outside_decays(self) -> None:
        cy = float(self.fh * 0.7)
        m_in = corridor_membership(320.0, cy, self.poly)
        m_out = corridor_membership(50.0, cy, self.poly)
        self.assertGreater(m_in, m_out)


# ── hysteresis tests ──────────────────────────────────────────────────────────

class TestHysteresis(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = RiskConfig()

    def test_escalate_to_medium(self) -> None:
        self.assertEqual(_apply_hysteresis(0.32, 0, self.cfg), 1)   # LOW → MEDIUM

    def test_sticky_medium(self) -> None:
        # Score between exit (0.25) and enter (0.30): stay MEDIUM
        self.assertEqual(_apply_hysteresis(0.27, 1, self.cfg), 1)

    def test_deescalate_medium_to_low(self) -> None:
        self.assertEqual(_apply_hysteresis(0.20, 1, self.cfg), 0)   # MEDIUM → LOW

    def test_sticky_high(self) -> None:
        # Score between exit_high (0.50) and enter_high (0.55): stay HIGH
        self.assertEqual(_apply_hysteresis(0.52, 2, self.cfg), 2)

    def test_deescalate_critical_to_high(self) -> None:
        # Score 0.70: below enter_critical (0.80), below exit_critical (0.75) → HIGH
        self.assertEqual(_apply_hysteresis(0.70, 3, self.cfg), 2)

    def test_sticky_critical(self) -> None:
        # Score 0.78 in CRITICAL: above exit_critical (0.75) → stays CRITICAL
        self.assertEqual(_apply_hysteresis(0.78, 3, self.cfg), 3)

    def test_escalate_skips_levels(self) -> None:
        # From LOW, score of 0.85 → goes directly to CRITICAL
        self.assertEqual(_apply_hysteresis(0.85, 0, self.cfg), 3)

    def test_no_flicker_at_medium_boundary(self) -> None:
        """Alternating scores at 0.28 and 0.31 should NOT flicker."""
        level = 0
        scores = [0.31, 0.28, 0.31, 0.28, 0.31]
        levels = []
        for s in scores:
            level = _apply_hysteresis(s, level, self.cfg)
            levels.append(level)
        # After entering MEDIUM (first 0.31), score 0.28 > exit_medium (0.25) → stays MEDIUM
        self.assertTrue(all(l >= 1 for l in levels[1:]),
                        f"Flicker detected: {levels}")


# ── persistence tests ─────────────────────────────────────────────────────────

class TestPersistence(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = RiskConfig(persist_k=3, persist_m=5)

    def test_single_spike_does_not_escalate_to_high(self) -> None:
        buf: deque[float] = deque([0.30, 0.30, 0.65], maxlen=5)  # only 1 of 3 above high
        result = _persistence_gate(2, 1, buf, self.cfg)
        self.assertLess(result, 2)   # should not reach HIGH

    def test_sustained_high_scores_escalate(self) -> None:
        buf: deque[float] = deque([0.60, 0.62, 0.58, 0.65, 0.61], maxlen=5)
        result = _persistence_gate(2, 1, buf, self.cfg)
        self.assertEqual(result, 2)  # all 5 above enter_high → HIGH approved

    def test_deescalation_always_allowed(self) -> None:
        buf: deque[float] = deque([0.2, 0.2], maxlen=5)
        result = _persistence_gate(0, 2, buf, self.cfg)
        self.assertEqual(result, 0)  # de-escalation to LOW is immediate


# ── end-to-end engine monotonicity ───────────────────────────────────────────

class TestEngineMonotonicity(unittest.TestCase):

    def setUp(self) -> None:
        self.frame = (640, 480)
        self.t0 = 1000.0

    def _run(self, tracks: list[Track], n: int = 8) -> list[float]:
        engine = _engine()
        scores = []
        for i in range(n):
            hazards = engine.update(tracks, self.frame, self.t0 + i * 0.033)
            if hazards:
                scores.append(hazards[0].risk_score)
            else:
                scores.append(0.0)
        return scores

    def test_closer_track_scores_higher(self) -> None:
        far   = _track(tid=1, bbox=(260, 260, 380, 310))  # small bbox = far
        close = _track(tid=2, bbox=(100, 100, 540, 420))  # large bbox = close

        e = _engine()
        t = self.t0
        h_far   = e.update([far],   self.frame, t)
        e2 = _engine()
        h_close = e2.update([close], self.frame, t)

        if h_far and h_close:
            self.assertGreater(h_close[0].risk_score, h_far[0].risk_score)

    def test_approaching_track_scores_higher(self) -> None:
        static  = _track(tid=1, vel=(0.0,   0.0))
        closing = _track(tid=2, vel=(0.0, 120.0))   # moving toward bottom-center

        e1 = _engine()
        e2 = _engine()
        # Run several frames to build history
        r_static  = [e1.update([static],  self.frame, self.t0 + i * 0.033) for i in range(6)]
        r_closing = [e2.update([closing], self.frame, self.t0 + i * 0.033) for i in range(6)]

        final_static  = r_static[-1][0].risk_score  if r_static[-1]  else 0.0
        final_closing = r_closing[-1][0].risk_score if r_closing[-1] else 0.0
        self.assertGreater(final_closing, final_static)

    def test_pedestrian_scores_higher_than_vehicle(self) -> None:
        ped = _track(tid=1, label="pedestrian")
        veh = _track(tid=2, label="vehicle")

        e1 = _engine()
        e2 = _engine()
        r_ped = e1.update([ped], self.frame, self.t0)
        r_veh = e2.update([veh], self.frame, self.t0)

        if r_ped and r_veh:
            self.assertGreater(r_ped[0].risk_score, r_veh[0].risk_score)

    def test_in_corridor_higher_than_out(self) -> None:
        """Object in corridor should score higher than same object outside it."""
        ccfg = CorridorConfig()
        poly = build_corridor_polygon(640, 480, ccfg)

        # In corridor: bbox centered at bottom-center
        in_bbox = (260.0, 300.0, 380.0, 430.0)
        # Out of corridor: bbox far to the left
        out_bbox = (5.0, 300.0, 125.0, 430.0)

        e1 = _engine()
        e2 = _engine()
        r_in  = e1.update([_track(tid=1, bbox=in_bbox)],  (640, 480), self.t0)
        r_out = e2.update([_track(tid=2, bbox=out_bbox)], (640, 480), self.t0)

        if r_in and r_out:
            self.assertGreater(r_in[0].risk_score, r_out[0].risk_score)

    def test_score_clamped(self) -> None:
        extreme = _track(tid=1, bbox=(0, 0, 640, 480), vel=(300, 300), label="pedestrian")
        engine = _engine()
        for i in range(10):
            h = engine.update([extreme], self.frame, self.t0 + i * 0.033)
            if h:
                self.assertLessEqual(h[0].risk_score, 1.0)
                self.assertGreaterEqual(h[0].risk_score, 0.0)

    def test_hysteresis_no_rapid_flicker(self) -> None:
        """Score oscillating just above/below a threshold should not flicker levels."""
        engine = _engine()
        cfg = RiskConfig()

        # Force engine to MEDIUM by running a moderately risky track
        mid_track = _track(tid=1, bbox=(200, 250, 440, 400), vel=(0, 60))
        for i in range(12):
            engine.update([mid_track], self.frame, self.t0 + i * 0.033)

        # Now run slightly jittered score and collect levels
        levels = []
        for i in range(20):
            h = engine.update([mid_track], self.frame, self.t0 + 12 * 0.033 + i * 0.033)
            if h:
                levels.append(h[0].risk_level)

        # Count transitions
        transitions = sum(1 for a, b in zip(levels, levels[1:]) if a != b)
        self.assertLess(transitions, 8, f"Too many level transitions: {levels}")

    def test_top_n_limits_output(self) -> None:
        cfg = RiskConfig(top_n=2)
        engine = RiskEngineV1(cfg)
        tracks = [_track(tid=i) for i in range(5)]
        h = engine.update(tracks, self.frame, self.t0)
        self.assertLessEqual(len(h), 2)

    def test_empty_tracks(self) -> None:
        engine = _engine()
        h = engine.update([], (640, 480), self.t0)
        self.assertEqual(h, [])


if __name__ == "__main__":
    unittest.main()
