"""Tests for RiskEngineV1 — the engine deployed in main.py.

Six required cases (Task 1.3):
  1. Monotonicity — closer bbox = higher risk score.
  2. Hysteresis — no flicker at the MEDIUM/LOW level boundary.
  3. Persistence gate — CRITICAL requires persist_k consecutive qualifying frames.
  4. Corridor — in-path object scores higher than identical off-path object.
  5. Score bounds — risk_score always in [0.0, 1.0] across 50 random inputs.
  6. Empty tracks — update([], ...) returns [].

Additional case (Task 2.4):
  7. Lateral risk — pedestrian with strong inward lateral velocity scores higher
     than an identical stationary pedestrian at the corridor edge.
"""

from __future__ import annotations

import random
import unittest

from src.risk.risk_engine import RiskEngineV1, _apply_hysteresis
from src.risk.risk_features import lateral_risk_score
from src.risk.risk_types import CorridorConfig, RiskConfig
from src.perception.corridor import build_corridor_polygon
from src.core.types import SignalState, Track


# ── helpers ───────────────────────────────────────────────────────────────────

def _track(
    tid: int = 1,
    bbox: tuple = (250, 200, 390, 380),
    vel: tuple = (0.0, 0.0),
    label: str = "pedestrian",
    hits: int = 10,
) -> Track:
    return Track(
        track_id=tid,
        bbox_xyxy=bbox,
        velocity_px_s=vel,
        age=hits,
        time_since_update=0.0,
        hits=hits,
        label=label,
        class_id=0,
    )


def _engine(cfg: RiskConfig | None = None, ccfg: CorridorConfig | None = None) -> RiskEngineV1:
    return RiskEngineV1(cfg or RiskConfig(), ccfg or CorridorConfig())


_FRAME = (640, 480)
_T0 = 5000.0
_DT = 1 / 30


# ── 1. Monotonicity ───────────────────────────────────────────────────────────

class TestMonotonicity(unittest.TestCase):
    """Closer bbox (taller in pixels) must score higher than a distant one."""

    def test_closer_bbox_higher_risk(self) -> None:
        # Large bbox ≈ close object; small bbox ≈ distant object.
        # Both are at the same horizontal position (corridor centre).
        close = _track(tid=1, bbox=(100, 50, 540, 430))   # nearly fills 640×480 frame
        distant = _track(tid=2, bbox=(270, 220, 370, 260))  # tiny bbox = far away

        e1 = _engine()
        e2 = _engine()
        r_close = e1.update([close], _FRAME, _T0)
        r_distant = e2.update([distant], _FRAME, _T0)

        self.assertTrue(r_close, "Close track produced no hazards")
        self.assertTrue(r_distant, "Distant track produced no hazards")
        self.assertGreater(
            r_close[0].risk_score,
            r_distant[0].risk_score,
            f"close={r_close[0].risk_score:.3f} should > distant={r_distant[0].risk_score:.3f}",
        )


# ── 2. Hysteresis ─────────────────────────────────────────────────────────────

class TestHysteresisNoFlicker(unittest.TestCase):
    """Score hovering just above the MEDIUM enter threshold must not oscillate."""

    def test_no_flicker_at_medium_boundary(self) -> None:
        cfg = RiskConfig()
        # Alternate scores between just-above-enter_medium (0.30) and
        # just-below-enter_medium but above exit_medium (0.25).
        scores = [0.31, 0.27, 0.31, 0.27, 0.31, 0.27, 0.31, 0.27, 0.31, 0.27,
                  0.31, 0.27, 0.31, 0.27, 0.31, 0.27, 0.31, 0.27, 0.31, 0.27]

        level = 0
        levels: list[int] = []
        for s in scores:
            level = _apply_hysteresis(s, level, cfg)
            levels.append(level)

        # After the first escalation to MEDIUM (score 0.31), every subsequent
        # 0.27 is above exit_medium (0.25) so the level should stay MEDIUM.
        self.assertTrue(
            all(l >= 1 for l in levels[1:]),
            f"Level flickered back to LOW: {levels}",
        )


# ── 3. Persistence gate ───────────────────────────────────────────────────────

class TestPersistenceGate(unittest.TestCase):
    """CRITICAL escalation requires persist_k (default 3) consecutive frames."""

    # Construct a very high-risk track: large pedestrian bbox at corridor centre
    # with strong downward velocity.  EMA converges quickly.
    _HIGH_RISK_TRACK = _track(
        tid=1,
        bbox=(50, 20, 590, 460),   # fills most of 640×480
        vel=(0.0, 300.0),          # fast downward = approaching
        label="pedestrian",
        hits=15,
    )

    def _run_frames(self, n: int) -> str:
        """Run the engine for n frames with the high-risk braking track; return last level.

        Brake signal is passed so the raw score (≈0.82) clears enter_critical=0.80
        with the rebalanced weights (w_ttc=0.30, w_class=0.05, w_signal=0.10).
        The persistence gate logic — not the score level — is what this test verifies.
        """
        engine = _engine()
        last_level = "LOW"
        signal_states = {1: SignalState(brake=True, left=False, right=False, confidence=0.9)}
        for i in range(n):
            results = engine.update(
                [self._HIGH_RISK_TRACK], _FRAME, _T0 + i * _DT,
                signal_states=signal_states,
            )
            if results:
                last_level = results[0].risk_level
        return last_level

    def test_two_frames_not_critical(self) -> None:
        """2 frames above enter_critical threshold should NOT reach CRITICAL."""
        level = self._run_frames(2)
        self.assertNotEqual(
            level, "CRITICAL",
            f"Got CRITICAL after only 2 frames (persist_k=3); level={level}",
        )

    def test_three_frames_reaches_critical(self) -> None:
        """3 frames (== persist_k) above enter_critical threshold MUST reach CRITICAL."""
        level = self._run_frames(3)
        self.assertEqual(
            level, "CRITICAL",
            f"Expected CRITICAL after 3 frames (persist_k=3); got {level}",
        )


# ── 4. Corridor ───────────────────────────────────────────────────────────────

class TestCorridor(unittest.TestCase):
    """Object at the corridor centre must score higher than an identical off-path one."""

    def test_in_path_scores_higher_than_off_path(self) -> None:
        # Both objects have the same bbox size (same distance) and zero velocity.
        # The only difference is horizontal position.
        in_path = _track(tid=1, bbox=(260.0, 280.0, 380.0, 420.0))   # corridor centre
        off_path = _track(tid=2, bbox=(5.0, 280.0, 125.0, 420.0))    # far left edge

        e1 = _engine()
        e2 = _engine()
        r_in = e1.update([in_path], _FRAME, _T0)
        r_out = e2.update([off_path], _FRAME, _T0)

        self.assertTrue(r_in, "In-path track produced no hazards")
        self.assertTrue(r_out, "Off-path track produced no hazards")
        self.assertGreater(
            r_in[0].risk_score,
            r_out[0].risk_score,
            f"in={r_in[0].risk_score:.3f} should > off={r_out[0].risk_score:.3f}",
        )


# ── 5. Score bounds ───────────────────────────────────────────────────────────

class TestScoreBounds(unittest.TestCase):
    """risk_score must always be in [0.0, 1.0] for any track input."""

    def test_score_always_in_0_1(self) -> None:
        rng = random.Random(42)
        engine = _engine()
        frame_size = (640, 480)

        for i in range(50):
            x1 = rng.uniform(0, 600)
            y1 = rng.uniform(0, 440)
            x2 = x1 + rng.uniform(10, 640 - x1)
            y2 = y1 + rng.uniform(10, 480 - y1)
            vx = rng.uniform(-400, 400)
            vy = rng.uniform(-400, 400)
            label = rng.choice(["pedestrian", "vehicle", "cyclist", "road_obstacle"])

            tr = _track(tid=1, bbox=(x1, y1, x2, y2), vel=(vx, vy), label=label, hits=5)
            results = engine.update([tr], frame_size, _T0 + i * _DT)
            for r in results:
                self.assertGreaterEqual(r.risk_score, 0.0, f"score {r.risk_score} < 0 on iter {i}")
                self.assertLessEqual(r.risk_score, 1.0, f"score {r.risk_score} > 1 on iter {i}")


# ── 6. Empty tracks ───────────────────────────────────────────────────────────

class TestEmptyTracks(unittest.TestCase):
    """update() with an empty track list must return an empty list."""

    def test_empty_tracks_returns_empty(self) -> None:
        engine = _engine()
        result = engine.update([], _FRAME, _T0)
        self.assertEqual(result, [])

    def test_empty_tracks_after_active_track(self) -> None:
        """Engine that saw tracks previously must still return [] on empty input."""
        engine = _engine()
        engine.update([_track(tid=1)], _FRAME, _T0)
        result = engine.update([], _FRAME, _T0 + _DT)
        self.assertEqual(result, [])


# ── 7. Lateral risk (Task 2.4) ────────────────────────────────────────────────

class TestLateralRisk(unittest.TestCase):
    """Pedestrian moving inward from the corridor edge must score materially higher
    than an identical stationary pedestrian."""

    def test_lateral_inward_scores_higher_than_stationary(self) -> None:
        ccfg = CorridorConfig()
        poly = build_corridor_polygon(640, 480, ccfg)

        # Place pedestrian just outside the right corridor edge at mid-frame height.
        # cx ≈ 550, which is well outside the ~487-px right edge at y=350.
        edge_stationary = _track(tid=1, bbox=(490.0, 270.0, 610.0, 430.0), vel=(0.0, 0.0))
        edge_lateral    = _track(tid=2, bbox=(490.0, 270.0, 610.0, 430.0), vel=(-200.0, 0.0))
        # vx = -200 px/s (moving left = toward corridor centre at x≈320)

        # Test the feature function directly for a crisp unit test.
        r_stationary = lateral_risk_score(edge_stationary, poly)
        r_lateral = lateral_risk_score(edge_lateral, poly)

        self.assertEqual(r_stationary, 0.0, "Stationary object should have zero lateral score")
        self.assertGreater(r_lateral, 0.0, "Inward-moving object must have positive lateral score")

    def test_lateral_outward_scores_zero(self) -> None:
        """Object moving away from the corridor must score zero."""
        ccfg = CorridorConfig()
        poly = build_corridor_polygon(640, 480, ccfg)
        # Object to the right, moving further right (away from centre)
        track_away = _track(tid=3, bbox=(490.0, 270.0, 610.0, 430.0), vel=(100.0, 0.0))
        self.assertEqual(lateral_risk_score(track_away, poly), 0.0)

    def test_object_inside_corridor_scores_zero(self) -> None:
        """Object already inside the corridor must score zero (path_factor handles it)."""
        ccfg = CorridorConfig()
        poly = build_corridor_polygon(640, 480, ccfg)
        inside = _track(tid=4, bbox=(270.0, 280.0, 370.0, 420.0), vel=(-50.0, 0.0))
        self.assertEqual(lateral_risk_score(inside, poly), 0.0)

    def test_engine_lateral_inward_scores_higher(self) -> None:
        """End-to-end: engine scores lateral-inward pedestrian higher than stationary."""
        e1 = _engine()
        e2 = _engine()

        stationary = _track(tid=1, bbox=(490.0, 270.0, 610.0, 430.0), vel=(0.0, 0.0))
        lateral    = _track(tid=2, bbox=(490.0, 270.0, 610.0, 430.0), vel=(-200.0, 0.0))

        r_stat = e1.update([stationary], _FRAME, _T0)
        r_lat  = e2.update([lateral],   _FRAME, _T0)

        self.assertTrue(r_stat, "Stationary track produced no hazards")
        self.assertTrue(r_lat,  "Lateral track produced no hazards")
        self.assertGreater(
            r_lat[0].risk_score,
            r_stat[0].risk_score,
            f"lateral={r_lat[0].risk_score:.3f} should > stationary={r_stat[0].risk_score:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
