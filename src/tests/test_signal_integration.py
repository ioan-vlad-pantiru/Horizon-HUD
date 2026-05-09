"""Integration tests for signal-state → risk engine pipeline.

Covers:
  - RiskConfig default weights sum to exactly 1.0 (regression guard)
  - _signal_risk: brake=0.7, None=0.0, toward-corridor > away-corridor
  - Braking vehicle scores strictly higher than same vehicle with no signal
  - signal_state is propagated to RiskAssessmentV1
  - "braking" appears in reasons when brake=True
  - 50 random signal states: risk_score stays in [0.0, 1.0]
"""

from __future__ import annotations

import random
import unittest

import numpy as np

from src.core.types import SignalState, Track
from src.perception.corridor import build_corridor_polygon
from src.risk.risk_engine import RiskEngineV1, _signal_risk, _corridor_centre_at_y
from src.risk.risk_types import CorridorConfig, RiskAssessmentV1, RiskConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRAME_W, _FRAME_H = 640, 480
_TS = 0.0333   # one 30-fps frame

def _make_track(
    track_id: int = 1,
    bbox: tuple[float, float, float, float] = (270.0, 200.0, 370.0, 400.0),
    label: str = "vehicle",
    hits: int = 5,
) -> Track:
    return Track(
        track_id=track_id,
        bbox_xyxy=bbox,
        velocity_px_s=(0.0, 5.0),   # mild closing speed
        age=5,
        time_since_update=0.0,
        hits=hits,
        label=label,
    )


def _default_corridor() -> np.ndarray:
    return build_corridor_polygon(_FRAME_W, _FRAME_H, CorridorConfig())


def _run_engine(
    tracks: list[Track],
    signal_states: dict[int, SignalState] | None = None,
    n_warmup: int = 5,
) -> list[RiskAssessmentV1]:
    engine = RiskEngineV1(RiskConfig(), CorridorConfig())
    result = []
    for i in range(n_warmup):
        ts = i * _TS
        result = engine.update(
            tracks, (_FRAME_W, _FRAME_H), ts,
            signal_states=signal_states,
        )
    return result


# ---------------------------------------------------------------------------
# Weight-sum regression guard
# ---------------------------------------------------------------------------

class TestWeightSum(unittest.TestCase):

    def test_default_weights_sum_to_one(self):
        cfg = RiskConfig()
        total = (
            cfg.w_ttc + cfg.w_distance + cfg.w_path
            + cfg.w_class + cfg.w_erratic + cfg.w_lateral + cfg.w_signal
        )
        self.assertAlmostEqual(total, 1.0, places=10)


# ---------------------------------------------------------------------------
# _signal_risk unit tests
# ---------------------------------------------------------------------------

class TestSignalRiskFunction(unittest.TestCase):

    def setUp(self):
        self._poly = _default_corridor()
        # Track centred in the frame (inside or near corridor centre)
        self._tr_centre = _make_track(bbox=(270.0, 200.0, 370.0, 400.0))
        # Track to the right of corridor centre
        self._tr_right = _make_track(bbox=(450.0, 200.0, 550.0, 400.0))
        # Track to the left of corridor centre
        self._tr_left = _make_track(bbox=(90.0, 200.0, 190.0, 400.0))

    def test_none_signal_returns_zero(self):
        self.assertEqual(_signal_risk(None, self._tr_centre, self._poly), 0.0)

    def test_all_false_returns_zero(self):
        sig = SignalState(brake=False, left=False, right=False, confidence=0.0)
        self.assertEqual(_signal_risk(sig, self._tr_centre, self._poly), 0.0)

    def test_brake_returns_0_7(self):
        sig = SignalState(brake=True, left=False, right=False, confidence=0.9)
        self.assertAlmostEqual(_signal_risk(sig, self._tr_centre, self._poly), 0.7)

    def test_hazard_lights_returns_0_6(self):
        sig = SignalState(brake=False, left=True, right=True, confidence=0.9)
        self.assertAlmostEqual(_signal_risk(sig, self._tr_centre, self._poly), 0.6)

    def test_brake_takes_precedence_over_hazard(self):
        # brake=True → 0.7 even when both indicators also True
        sig = SignalState(brake=True, left=True, right=True, confidence=0.9)
        self.assertAlmostEqual(_signal_risk(sig, self._tr_centre, self._poly), 0.7)

    def test_toward_corridor_higher_than_away(self):
        # Track to the right: left indicator points toward centre → 0.5
        sig_toward = SignalState(brake=False, left=True, right=False, confidence=0.9)
        score_toward = _signal_risk(sig_toward, self._tr_right, self._poly)
        # Track to the right: right indicator points away from centre → 0.2
        sig_away = SignalState(brake=False, left=False, right=True, confidence=0.9)
        score_away = _signal_risk(sig_away, self._tr_right, self._poly)
        self.assertGreater(score_toward, score_away)

    def test_toward_score_is_0_5(self):
        sig = SignalState(brake=False, left=True, right=False, confidence=0.9)
        score = _signal_risk(sig, self._tr_right, self._poly)
        self.assertAlmostEqual(score, 0.5)

    def test_away_score_is_0_2(self):
        sig = SignalState(brake=False, left=False, right=True, confidence=0.9)
        score = _signal_risk(sig, self._tr_right, self._poly)
        self.assertAlmostEqual(score, 0.2)


# ---------------------------------------------------------------------------
# Corridor centre helper
# ---------------------------------------------------------------------------

class TestCorridorCentreAtY(unittest.TestCase):

    def test_centre_at_bottom_y(self):
        poly = _default_corridor()
        bot_y = float(poly[0, 1])
        centre = _corridor_centre_at_y(bot_y, poly)
        self.assertIsNotNone(centre)
        # At bottom the corridor centre should be near frame centre
        self.assertAlmostEqual(centre, _FRAME_W / 2.0, delta=20.0)

    def test_above_corridor_returns_none(self):
        poly = _default_corridor()
        top_y = float(poly[3, 1])
        result = _corridor_centre_at_y(top_y - 10.0, poly)
        self.assertIsNone(result)

    def test_below_corridor_returns_none(self):
        poly = _default_corridor()
        bot_y = float(poly[0, 1])
        result = _corridor_centre_at_y(bot_y + 10.0, poly)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Full engine integration
# ---------------------------------------------------------------------------

class TestSignalIntegration(unittest.TestCase):

    def test_braking_vehicle_scores_higher_than_no_signal(self):
        track = _make_track(1)
        sig_brake = {1: SignalState(brake=True, left=False, right=False, confidence=0.9)}

        scored_no_signal = _run_engine([track], signal_states=None)
        scored_brake = _run_engine([track], signal_states=sig_brake)

        self.assertTrue(len(scored_no_signal) > 0)
        self.assertTrue(len(scored_brake) > 0)
        self.assertGreater(scored_brake[0].risk_score, scored_no_signal[0].risk_score)

    def test_signal_state_propagated_to_assessment(self):
        track = _make_track(1)
        sig = SignalState(brake=True, left=False, right=False, confidence=0.9)
        result = _run_engine([track], signal_states={1: sig})
        self.assertTrue(len(result) > 0)
        self.assertIsNotNone(result[0].signal_state)
        self.assertTrue(result[0].signal_state.brake)

    def test_no_signal_produces_none_signal_state(self):
        track = _make_track(1)
        result = _run_engine([track], signal_states=None)
        self.assertTrue(len(result) > 0)
        self.assertIsNone(result[0].signal_state)

    def test_braking_reason_present_when_brake_true(self):
        track = _make_track(1)
        sig = {1: SignalState(brake=True, left=False, right=False, confidence=0.9)}
        result = _run_engine([track], signal_states=sig)
        self.assertTrue(len(result) > 0)
        self.assertIn("braking", result[0].reasons)

    def test_signaling_left_reason_present(self):
        track = _make_track(1)
        sig = {1: SignalState(brake=False, left=True, right=False, confidence=0.9)}
        result = _run_engine([track], signal_states=sig)
        self.assertTrue(len(result) > 0)
        self.assertIn("signaling left", result[0].reasons)

    def test_signaling_right_reason_present(self):
        track = _make_track(1)
        sig = {1: SignalState(brake=False, left=False, right=True, confidence=0.9)}
        result = _run_engine([track], signal_states=sig)
        self.assertTrue(len(result) > 0)
        self.assertIn("signaling right", result[0].reasons)

    def test_hazard_lights_reason_present(self):
        track = _make_track(1)
        sig = {1: SignalState(brake=False, left=True, right=True, confidence=0.9)}
        result = _run_engine([track], signal_states=sig)
        self.assertTrue(len(result) > 0)
        self.assertIn("hazard lights", result[0].reasons)


# ---------------------------------------------------------------------------
# Score bounds with random signal states
# ---------------------------------------------------------------------------

class TestScoreBoundsWithSignal(unittest.TestCase):

    def test_50_random_signal_states_all_in_unit_interval(self):
        rng = random.Random(0)
        track = _make_track(1)

        for _ in range(50):
            brake = rng.choice([True, False])
            left = rng.choice([True, False])
            right = rng.choice([True, False])
            confidence = rng.random()
            sig = {1: SignalState(brake=brake, left=left, right=right, confidence=confidence)}

            engine = RiskEngineV1(RiskConfig(), CorridorConfig())
            for i in range(6):
                results = engine.update([track], (_FRAME_W, _FRAME_H), i * _TS,
                                        signal_states=sig)

            self.assertTrue(len(results) > 0)
            score = results[0].risk_score
            self.assertGreaterEqual(score, 0.0, f"score {score} < 0 for {sig}")
            self.assertLessEqual(score, 1.0, f"score {score} > 1 for {sig}")


if __name__ == "__main__":
    unittest.main()
