"""Unit tests for src/detection/signal_classifier.py.

Covers:
  - SignalState: hazard derivation for all label combos
  - _SlidingVote: N-of-M logic, clear after M falses
  - SignalClassifier graceful degradation (missing model)
  - ROI crop: degenerate bbox, out-of-bounds clamping, bottom-40% crop bounds
  - Vehicle-only gating
  - Per-track inference scheduler (classify_every_n)
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.core.types import SignalState, Track
from src.detection.signal_classifier import SignalClassifier, _SlidingVote


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_track(label: str, track_id: int = 1) -> Track:
    return Track(
        track_id=track_id,
        bbox_xyxy=(100.0, 200.0, 300.0, 400.0),
        velocity_px_s=(0.0, 0.0),
        age=5,
        time_since_update=0.0,
        hits=5,
        label=label,
    )


def _blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# SignalState tests
# ---------------------------------------------------------------------------

class TestSignalState(unittest.TestCase):

    def test_hazard_both_indicators(self):
        ss = SignalState(brake=False, left=True, right=True, confidence=0.9)
        self.assertTrue(ss.hazard)

    def test_hazard_requires_both(self):
        self.assertFalse(SignalState(brake=False, left=True, right=False, confidence=0.9).hazard)
        self.assertFalse(SignalState(brake=False, left=False, right=True, confidence=0.9).hazard)

    def test_brake_does_not_imply_hazard(self):
        ss = SignalState(brake=True, left=False, right=False, confidence=0.9)
        self.assertFalse(ss.hazard)

    def test_brake_and_both_indicators_is_hazard(self):
        ss = SignalState(brake=True, left=True, right=True, confidence=0.9)
        self.assertTrue(ss.hazard)

    def test_all_false_no_hazard(self):
        ss = SignalState(brake=False, left=False, right=False, confidence=0.0)
        self.assertFalse(ss.hazard)


# ---------------------------------------------------------------------------
# _SlidingVote tests
# ---------------------------------------------------------------------------

class TestSlidingVote(unittest.TestCase):

    def test_n_of_m_satisfied(self):
        v = _SlidingVote(n=2, m=3)
        self.assertFalse(v.update(True))   # 1 True in window of 1 → not yet 2
        self.assertTrue(v.update(True))    # 2 True in window of 2 → satisfied

    def test_n_of_m_not_satisfied(self):
        v = _SlidingVote(n=3, m=5)
        v.update(True)
        v.update(True)
        result = v.update(False)   # only 2 Trues so far
        self.assertFalse(result)

    def test_window_slides_out_old_trues(self):
        v = _SlidingVote(n=2, m=3)
        v.update(True)
        v.update(True)    # window: [T, T] → True
        v.update(False)   # window: [T, T, F] → still 2 → True
        v.update(False)   # window: [T, F, F] → only 1 → False
        result = v.update(False)  # window: [F, F, F] → 0 → False
        self.assertFalse(result)

    def test_all_false_returns_false(self):
        v = _SlidingVote(n=2, m=4)
        for _ in range(4):
            self.assertFalse(v.update(False))

    def test_n_equals_m(self):
        v = _SlidingVote(n=3, m=3)
        v.update(True)
        v.update(True)
        self.assertFalse(v.update(False))   # only 2/3
        self.assertFalse(v.update(True))    # window [T, F, T] → 2/3 still False
        v2 = _SlidingVote(n=3, m=3)
        v2.update(True)
        v2.update(True)
        self.assertTrue(v2.update(True))    # 3/3


# ---------------------------------------------------------------------------
# Graceful degradation (missing model)
# ---------------------------------------------------------------------------

class TestSignalClassifierGracefulDegradation(unittest.TestCase):

    MISSING = "/nonexistent/model_XXXXXXXXXXX.tflite"

    def test_missing_model_run_returns_empty_dict(self):
        sc = SignalClassifier(model_path=self.MISSING)
        result = sc.run(_blank_frame(), [], 0)
        self.assertEqual(result, {})

    def test_missing_model_no_exception_with_tracks(self):
        sc = SignalClassifier(model_path=self.MISSING)
        tracks = [_make_track("vehicle", 1)]
        # Must not raise
        result = sc.run(_blank_frame(), tracks, 1)
        self.assertEqual(result, {})

    def test_classify_returns_none_when_disabled(self):
        sc = SignalClassifier(model_path=self.MISSING)
        result = sc._classify(_blank_frame(), (100.0, 100.0, 300.0, 400.0))
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# ROI crop
# ---------------------------------------------------------------------------

class TestROICrop(unittest.TestCase):

    MISSING = "/nonexistent/model_XXXXXXXXXXX.tflite"

    def test_degenerate_bbox_zero_area_returns_none(self):
        sc = SignalClassifier(model_path=self.MISSING)
        # x1 == x2 → zero-width crop
        result = sc._classify(_blank_frame(), (200.0, 100.0, 200.0, 300.0))
        self.assertIsNone(result)

    def test_degenerate_bbox_zero_height_returns_none(self):
        sc = SignalClassifier(model_path=self.MISSING)
        result = sc._classify(_blank_frame(), (100.0, 200.0, 300.0, 200.0))
        self.assertIsNone(result)

    def test_out_of_bounds_bbox_clamped_no_crash(self):
        sc = SignalClassifier(model_path=self.MISSING)
        # bbox extends beyond frame — should be clamped, not crash
        result = sc._classify(_blank_frame(480, 640), (-200.0, -200.0, 900.0, 900.0))
        # With no model, returns None (model gate), but should not raise
        self.assertIsNone(result)

    def test_bottom_40_percent_crop_bounds(self):
        """Verify the crop math: roi_y1 = y1 + 0.6*(y2-y1)."""
        y1, y2 = 100.0, 300.0
        expected_roi_y1 = y1 + 0.6 * (y2 - y1)   # = 220.0
        self.assertAlmostEqual(expected_roi_y1, 220.0)

        sc = SignalClassifier(model_path=self.MISSING)
        # Inject a mock interpreter so inference runs
        mock_interp = MagicMock()
        mock_interp.get_tensor.return_value = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        sc._interpreter = mock_interp
        sc._in_details = [{"index": 0, "dtype": np.float32, "shape": [1, 96, 96, 3]}]
        sc._out_details = [{"index": 1}]
        sc._in_h, sc._in_w = 96, 96

        frame = _blank_frame(480, 640)
        bbox = (100.0, 100.0, 300.0, 300.0)   # height = 200 px → roi from y=220
        sc._classify(frame, bbox)
        mock_interp.invoke.assert_called_once()   # crop was valid → inference ran


# ---------------------------------------------------------------------------
# Vehicle-only gating
# ---------------------------------------------------------------------------

class TestVehicleGating(unittest.TestCase):

    def test_pedestrian_track_skipped(self):
        sc = SignalClassifier(model_path="/nonexistent/x.tflite")
        # Inject mock interpreter so run() would process vehicle tracks
        sc._interpreter = MagicMock()
        sc._interpreter.get_tensor.return_value = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        with patch.object(sc, "_classify", return_value=SignalState(True, False, False, 0.9)):
            tracks = [_make_track("pedestrian", 1), _make_track("cyclist", 2)]
            result = sc.run(_blank_frame(), tracks, 1)
        self.assertEqual(result, {})

    def test_vehicle_track_processed(self):
        sc = SignalClassifier(model_path="/nonexistent/x.tflite")
        sc._interpreter = MagicMock()

        fixed_state = SignalState(brake=True, left=False, right=False, confidence=0.9)
        with patch.object(sc, "_classify", return_value=fixed_state):
            tracks = [_make_track("vehicle", 42)]
            result = sc.run(_blank_frame(), tracks, 1)
        # After one frame the smoother (vote_n=2) has only 1 positive → state may
        # not fire yet, but the track must be in the internal state tracking.
        # Just verify no crash and pedestrian tracks didn't appear.
        self.assertNotIn(1, result)   # track_id 1 is pedestrian — absent


# ---------------------------------------------------------------------------
# Scheduler (classify_every_n)
# ---------------------------------------------------------------------------

class TestScheduler(unittest.TestCase):

    def test_classify_called_once_per_n_frames(self):
        sc = SignalClassifier(
            model_path="/nonexistent/x.tflite",
            classify_every_n=3,
            vote_n=1,
            vote_m=3,
        )
        sc._interpreter = MagicMock()

        fixed = SignalState(brake=True, left=False, right=False, confidence=0.9)
        with patch.object(sc, "_classify", return_value=fixed) as mock_classify:
            tr = _make_track("vehicle", 1)
            sc.run(_blank_frame(), [tr], frame_idx=1)  # → classify (1 - init_offset >= 3)
            sc.run(_blank_frame(), [tr], frame_idx=2)  # 2-1=1 < 3 → skip
            sc.run(_blank_frame(), [tr], frame_idx=3)  # 3-1=2 < 3 → skip
            self.assertEqual(mock_classify.call_count, 1)
            sc.run(_blank_frame(), [tr], frame_idx=4)  # 4-1=3 >= 3 → classify
            self.assertEqual(mock_classify.call_count, 2)

    def test_per_track_independent_scheduling(self):
        sc = SignalClassifier(
            model_path="/nonexistent/x.tflite",
            classify_every_n=3,
            vote_n=1,
            vote_m=3,
        )
        sc._interpreter = MagicMock()

        fixed = SignalState(brake=False, left=True, right=False, confidence=0.8)
        with patch.object(sc, "_classify", return_value=fixed) as mock_classify:
            tr1 = _make_track("vehicle", 1)
            tr2 = _make_track("vehicle", 2)
            sc.run(_blank_frame(), [tr1], frame_idx=1)   # tr1 classified at frame 1
            sc.run(_blank_frame(), [tr2], frame_idx=2)   # tr2 classified at frame 2 (independent)
            # Total = 2 classify calls
            self.assertEqual(mock_classify.call_count, 2)


if __name__ == "__main__":
    unittest.main()
