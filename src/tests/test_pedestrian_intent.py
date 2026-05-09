"""Tests for the pedestrian crossing intent module.

Covers:
  - pedestrian_crossing_score() unit tests (pure function)
  - Engine integration: intent propagated to RiskAssessmentV1
  - Non-pedestrian tracks produce crossing_prob == 0.0
"""

from __future__ import annotations

import unittest
from collections import deque

import numpy as np

from src.core.types import Track
from src.perception.corridor import build_corridor_polygon
from src.risk.risk_features import pedestrian_crossing_score
from src.risk.risk_engine import RiskEngineV1
from src.risk.risk_types import CorridorConfig, RiskAssessmentV1, RiskConfig, TrackSnapshot

_FRAME_W, _FRAME_H = 640, 480
_TS = 0.0333  # one 30-fps frame


# ── helpers ───────────────────────────────────────────────────────────────────

def _default_corridor() -> np.ndarray:
    return build_corridor_polygon(_FRAME_W, _FRAME_H, CorridorConfig())


def _make_hist(
    snapshots: list[tuple[float, float, float, float, float, float, float]],
) -> "deque[TrackSnapshot]":
    """Build a deque of TrackSnapshots from (t, x1, y1, x2, y2, vx, vy) tuples."""
    hist: deque[TrackSnapshot] = deque(maxlen=16)
    for t, x1, y1, x2, y2, vx, vy in snapshots:
        hist.append(TrackSnapshot(
            timestamp=t,
            bbox_xyxy=(x1, y1, x2, y2),
            velocity_px_s=(vx, vy),
        ))
    return hist


def _make_track(
    track_id: int = 1,
    bbox: tuple[float, float, float, float] = (100.0, 200.0, 160.0, 380.0),
    velocity: tuple[float, float] = (0.0, 0.0),
    label: str = "pedestrian",
    hits: int = 6,
) -> Track:
    return Track(
        track_id=track_id,
        bbox_xyxy=bbox,
        velocity_px_s=velocity,
        age=6,
        time_since_update=0.0,
        hits=hits,
        label=label,
    )


def _run_engine(
    tracks: list[Track],
    n_warmup: int = 6,
    velocities: list[tuple[float, float]] | None = None,
) -> list[RiskAssessmentV1]:
    engine = RiskEngineV1(RiskConfig(), CorridorConfig())
    result: list[RiskAssessmentV1] = []
    for i in range(n_warmup):
        ts = i * _TS
        # Optionally vary velocity each frame so the engine history sees motion
        frame_tracks = tracks
        if velocities and i < len(velocities):
            vx, vy = velocities[i]
            frame_tracks = [
                Track(
                    track_id=tr.track_id,
                    bbox_xyxy=tr.bbox_xyxy,
                    velocity_px_s=(vx, vy),
                    age=tr.age,
                    time_since_update=tr.time_since_update,
                    hits=tr.hits,
                    label=tr.label,
                    class_id=tr.class_id,
                )
                for tr in tracks
            ]
        result = engine.update(frame_tracks, (_FRAME_W, _FRAME_H), ts)
    return result


# ── unit tests for pedestrian_crossing_score() ────────────────────────────────

class TestPedestrianCrossingScore(unittest.TestCase):

    def setUp(self):
        self._poly = _default_corridor()
        # Corridor centre x at y=380 ≈ 320 (frame centre).
        # Track at x ≈ 130 (well left of corridor centre).
        self._base_bbox = (100.0, 200.0, 160.0, 380.0)

    def _stationary_hist(self, n: int = 5) -> "deque[TrackSnapshot]":
        rows = [(i * _TS,) + self._base_bbox + (0.0, 0.0) for i in range(n)]
        return _make_hist(rows)

    def test_stationary_pedestrian_scores_zero(self):
        hist = self._stationary_hist()
        score = pedestrian_crossing_score(hist, self._poly, _FRAME_W)
        self.assertAlmostEqual(score, 0.0)

    def test_lateral_motion_toward_corridor_raises_score(self):
        # Pedestrian left of corridor centre (cx≈130), vx > 0 → toward road
        rows = [(i * _TS,) + self._base_bbox + (20.0, 0.0) for i in range(5)]
        hist = _make_hist(rows)
        score = pedestrian_crossing_score(hist, self._poly, _FRAME_W)
        self.assertGreater(score, 0.0)

    def test_lateral_motion_away_from_corridor_scores_zero_lateral(self):
        # Pedestrian left of corridor centre (cx≈130), vx < 0 → moving away
        rows = [(i * _TS,) + self._base_bbox + (-20.0, 0.0) for i in range(5)]
        hist = _make_hist(rows)
        score = pedestrian_crossing_score(hist, self._poly, _FRAME_W)
        # lateral component = 0; score may still be >0 due to speed component
        # but should be lower than the toward-corridor case
        rows_toward = [(i * _TS,) + self._base_bbox + (20.0, 0.0) for i in range(5)]
        score_toward = pedestrian_crossing_score(_make_hist(rows_toward), self._poly, _FRAME_W)
        self.assertLess(score, score_toward)

    def test_approaching_pedestrian_raises_score(self):
        # Growing bbox (h: 180 → 200 over 4 frames)
        rows = []
        for i in range(5):
            h = 180.0 + i * 5.0
            rows.append((i * _TS, 100.0, 200.0, 160.0, 200.0 + h, 0.0, 2.0))
        hist = _make_hist(rows)
        score = pedestrian_crossing_score(hist, self._poly, _FRAME_W)
        self.assertGreater(score, 0.0)

    def test_score_clamped_to_unit_interval(self):
        # Extreme velocities should not exceed [0, 1]
        rows = [(i * _TS,) + self._base_bbox + (1000.0, 1000.0) for i in range(5)]
        hist = _make_hist(rows)
        score = pedestrian_crossing_score(hist, self._poly, _FRAME_W)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_minimum_history_below_3_returns_zero(self):
        rows = [(0.0,) + self._base_bbox + (20.0, 0.0),
                (_TS,)  + self._base_bbox + (20.0, 0.0)]
        hist = _make_hist(rows)
        self.assertEqual(pedestrian_crossing_score(hist, self._poly, _FRAME_W), 0.0)


# ── integration tests ─────────────────────────────────────────────────────────

class TestIntentIntegration(unittest.TestCase):

    def test_crossing_pedestrian_scores_higher_than_stationary(self):
        stationary = _make_track(velocity=(0.0, 0.0))
        lateral = _make_track(velocity=(25.0, 0.0))

        score_stat = _run_engine([stationary])[0].risk_score
        score_lat = _run_engine([lateral])[0].risk_score
        self.assertGreater(score_lat, score_stat)

    def test_crossing_prob_propagated_to_assessment(self):
        # Track moving toward corridor centre → crossing_prob > 0 after warmup
        track = _make_track(velocity=(25.0, 0.0))
        result = _run_engine([track])
        self.assertTrue(len(result) > 0)
        self.assertIsNotNone(result[0].crossing_prob)
        self.assertGreater(result[0].crossing_prob, 0.0)

    def test_crossing_likely_reason_present(self):
        # Strong lateral motion + some speed → r_intent > 0.4 → "crossing likely"
        track = _make_track(velocity=(35.0, 5.0))
        result = _run_engine([track])
        self.assertTrue(len(result) > 0)
        self.assertIn("crossing likely", result[0].reasons)

    def test_vehicle_has_crossing_prob_zero(self):
        # Non-pedestrian label → r_intent must be 0.0
        track = _make_track(label="vehicle", velocity=(25.0, 0.0))
        result = _run_engine([track])
        self.assertTrue(len(result) > 0)
        self.assertEqual(result[0].crossing_prob, 0.0)

    def test_non_pedestrian_label_not_scored(self):
        # Cyclist with lateral motion → crossing_prob == 0.0
        track = _make_track(label="cyclist", velocity=(25.0, 0.0))
        result = _run_engine([track])
        self.assertTrue(len(result) > 0)
        self.assertEqual(result[0].crossing_prob, 0.0)


if __name__ == "__main__":
    unittest.main()
