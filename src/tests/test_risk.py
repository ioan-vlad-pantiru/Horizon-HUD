"""Tests for the risk engine – scoring correctness and monotonicity."""

from __future__ import annotations

import unittest

from src.risk.risk import RiskEngine, _level
from src.core.types import RiskAssessment, Track


def _make_track(
    track_id: int = 1,
    bbox_xyxy: tuple[float, float, float, float] = (100, 100, 200, 300),
    velocity_px_s: tuple[float, float] = (0.0, 0.0),
    label: str = "person",
    class_id: int = 0,
) -> Track:
    return Track(
        track_id=track_id,
        bbox_xyxy=bbox_xyxy,
        velocity_px_s=velocity_px_s,
        age=5,
        time_since_update=0.0,
        hits=5,
        label=label,
        class_id=class_id,
    )


class TestRiskLevels(unittest.TestCase):
    def test_level_boundaries(self) -> None:
        self.assertEqual(_level(0.0), "LOW")
        self.assertEqual(_level(0.24), "LOW")
        self.assertEqual(_level(0.25), "MEDIUM")
        self.assertEqual(_level(0.49), "MEDIUM")
        self.assertEqual(_level(0.50), "HIGH")
        self.assertEqual(_level(0.74), "HIGH")
        self.assertEqual(_level(0.75), "CRITICAL")
        self.assertEqual(_level(1.0), "CRITICAL")


class TestRiskMonotonicity(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RiskEngine(ref_height_px=300.0)
        self.frame_size = (640, 480)

    def test_larger_bbox_higher_risk(self) -> None:
        """A track with a larger bbox (closer) should score higher."""
        small = _make_track(track_id=1, bbox_xyxy=(200, 200, 260, 280))
        large = _make_track(track_id=2, bbox_xyxy=(100, 100, 300, 400))

        r1 = self.engine.assess([small], self.frame_size, 1.0)
        engine2 = RiskEngine(ref_height_px=300.0)
        r2 = engine2.assess([large], self.frame_size, 1.0)

        self.assertGreater(r2[0].risk_score, r1[0].risk_score)

    def test_closing_speed_increases_risk(self) -> None:
        """A track moving toward the bottom center should score higher than stationary."""
        stationary = _make_track(track_id=1, velocity_px_s=(0, 0))
        closing = _make_track(track_id=2, velocity_px_s=(0, 100))

        e1 = RiskEngine(ref_height_px=300.0)
        r_stat = e1.assess([stationary], self.frame_size, 1.0)
        e2 = RiskEngine(ref_height_px=300.0)
        r_close = e2.assess([closing], self.frame_size, 1.0)

        self.assertGreater(r_close[0].risk_score, r_stat[0].risk_score)

    def test_pedestrian_higher_base_than_vehicle(self) -> None:
        """Pedestrian should have higher base risk than vehicle, all else equal."""
        pedestrian = _make_track(track_id=1, label="pedestrian", class_id=1)
        vehicle = _make_track(track_id=2, label="vehicle", class_id=0)

        e1 = RiskEngine(ref_height_px=300.0)
        r_pedestrian = e1.assess([pedestrian], self.frame_size, 1.0)
        e2 = RiskEngine(ref_height_px=300.0)
        r_vehicle = e2.assess([vehicle], self.frame_size, 1.0)

        self.assertGreater(r_pedestrian[0].risk_score, r_vehicle[0].risk_score)

    def test_lateral_motion_penalty(self) -> None:
        """High lateral motion should increase risk."""
        calm = _make_track(track_id=1, velocity_px_s=(0, 50))
        lateral = _make_track(track_id=2, velocity_px_s=(200, 50))

        e1 = RiskEngine(ref_height_px=300.0)
        r_calm = e1.assess([calm], self.frame_size, 1.0)
        e2 = RiskEngine(ref_height_px=300.0)
        r_lat = e2.assess([lateral], self.frame_size, 1.0)

        self.assertGreaterEqual(r_lat[0].risk_score, r_calm[0].risk_score)

    def test_score_clamped_0_1(self) -> None:
        """Risk score must always be in [0, 1]."""
        extreme = _make_track(
            track_id=1,
            bbox_xyxy=(0, 0, 640, 480),
            velocity_px_s=(500, 500),
            label="person",
        )
        r = self.engine.assess([extreme], self.frame_size, 1.0)
        self.assertGreaterEqual(r[0].risk_score, 0.0)
        self.assertLessEqual(r[0].risk_score, 1.0)

    def test_reasons_populated(self) -> None:
        tr = _make_track(
            track_id=1,
            bbox_xyxy=(100, 100, 400, 450),
            velocity_px_s=(0, 150),
            label="person",
        )
        r = self.engine.assess([tr], self.frame_size, 1.0)
        self.assertIsInstance(r[0].reasons, list)
        self.assertGreater(len(r[0].reasons), 0)

    def test_empty_tracks(self) -> None:
        r = self.engine.assess([], self.frame_size, 1.0)
        self.assertEqual(len(r), 0)


class TestRiskAccelPenalty(unittest.TestCase):
    def test_erratic_accel_increases_risk(self) -> None:
        engine = RiskEngine(ref_height_px=300.0)
        tr1 = _make_track(track_id=1, velocity_px_s=(0, 50))
        engine.assess([tr1], (640, 480), 1.0)

        tr2 = _make_track(track_id=1, velocity_px_s=(0, 350))
        r = engine.assess([tr2], (640, 480), 1.033)

        has_erratic = any("erratic" in reason for reason in r[0].reasons)
        self.assertTrue(has_erratic)


if __name__ == "__main__":
    unittest.main()
