"""Vehicle rear-signal classifier.

Supports two backends, chosen automatically by file extension:
  .onnx   — ONNX Runtime (preferred on macOS; uses CoreML/ANE when available)
  .tflite — TFLite INT8 (preferred on Raspberry Pi)

When the model file is absent or no runtime is available the classifier
degrades gracefully: run() returns {} and never raises.

Pipeline per frame
──────────────────
1. Vehicle-only gating — non-vehicle labels are skipped entirely.
2. Per-track scheduler — inference runs at most every classify_every_n frames;
   the previous result is reused between steps.
3. ROI crop — bottom 40% of the vehicle bounding box (rear of vehicle).
4. Resize to model input size and run inference.
5. _SlidingVote smoother — emit a label only after N positives in the last M
   frames, applied independently to brake / left / right.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Optional

import cv2
import numpy as np

from src.core.types import SignalState, Track

logger = logging.getLogger(__name__)

_VEHICLE_LABELS: frozenset[str] = frozenset({"vehicle"})


# ---------------------------------------------------------------------------
# Backend loaders
# ---------------------------------------------------------------------------

def _load_ort_session(model_path: str):
    """Load an ONNX Runtime InferenceSession, preferring CoreML on macOS."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("SignalClassifier: onnxruntime not installed — try: pip install onnxruntime")
        return None

    # Try CoreML (Apple ANE/GPU) then CPU
    for providers in (
        ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ):
        try:
            sess = ort.InferenceSession(model_path, providers=providers)
            active = sess.get_providers()
            logger.info("SignalClassifier (ONNX): loaded %s  providers=%s", model_path, active)
            return sess
        except Exception:
            continue
    return None


def _load_tflite_interpreter(model_path: str):
    """Load a TFLite Interpreter via whichever runtime is available."""
    Interpreter = None
    for loader in (
        lambda: __import__("ai_edge_litert.interpreter", fromlist=["Interpreter"]).Interpreter,
        lambda: __import__("tflite_runtime.interpreter", fromlist=["Interpreter"]).Interpreter,
        lambda: __import__("tensorflow.lite.python.interpreter",
                           fromlist=["Interpreter"]).Interpreter,
    ):
        try:
            Interpreter = loader()
            break
        except ImportError:
            continue

    if Interpreter is None:
        logger.warning("SignalClassifier: no TFLite runtime found")
        return None

    try:
        interp = Interpreter(model_path=model_path, num_threads=2)
        interp.allocate_tensors()
        logger.info("SignalClassifier (TFLite): loaded %s", model_path)
        return interp
    except Exception as exc:
        logger.warning("SignalClassifier: TFLite load failed (%s)", exc)
        return None


# ---------------------------------------------------------------------------
# Temporal smoother
# ---------------------------------------------------------------------------

class _SlidingVote:
    """Returns True when at least N of the last M inputs were True."""

    def __init__(self, n: int, m: int) -> None:
        self._n = n
        self._buf: deque[bool] = deque(maxlen=m)

    def update(self, value: bool) -> bool:
        self._buf.append(value)
        return sum(self._buf) >= self._n


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class SignalClassifier:
    """Classify brake / left-indicator / right-indicator from vehicle bbox crops.

    Parameters
    ----------
    model_path
        Path to a .onnx or .tflite model file.
        Absent file → classifier disabled gracefully.
    classify_every_n
        Run inference at most once every N frames per track.
    vote_n, vote_m
        Temporal smoother: require vote_n True values in the last vote_m frames
        per signal channel per track.
    """

    def __init__(
        self,
        model_path: str,
        classify_every_n: int = 3,
        vote_n: int = 2,
        vote_m: int = 5,
    ) -> None:
        self._classify_every_n = classify_every_n
        self._vote_n = vote_n
        self._vote_m = vote_m
        self._session = None      # ONNX Runtime session
        self._interpreter = None  # TFLite interpreter
        self._in_h = 96
        self._in_w = 96
        self._ort_input_name: str = "input"

        # Per-track state
        self._last_frame: dict[int, int] = {}
        self._cached: dict[int, Optional[SignalState]] = {}
        self._brake_vote: dict[int, _SlidingVote] = {}
        self._left_vote: dict[int, _SlidingVote] = {}
        self._right_vote: dict[int, _SlidingVote] = {}

        if not os.path.isfile(model_path):
            logger.warning("SignalClassifier: model not found at %s — disabled", model_path)
            return

        ext = os.path.splitext(model_path)[1].lower()
        if ext == ".onnx":
            self._session = _load_ort_session(model_path)
            if self._session is not None:
                inp = self._session.get_inputs()[0]
                self._ort_input_name = inp.name
                shape = inp.shape  # [batch, C, H, W]
                self._in_h = int(shape[2]) if len(shape) >= 4 else 96
                self._in_w = int(shape[3]) if len(shape) >= 4 else 96
        else:
            self._interpreter = _load_tflite_interpreter(model_path)
            if self._interpreter is not None:
                in_det = self._interpreter.get_input_details()
                shape = in_det[0]["shape"]  # [batch, H, W, C]
                self._in_h = int(shape[1])
                self._in_w = int(shape[2])
                self._in_details = in_det
                self._out_details = self._interpreter.get_output_details()

    @property
    def _enabled(self) -> bool:
        return self._session is not None or self._interpreter is not None

    # ── public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        frame_idx: int,
    ) -> dict[int, SignalState]:
        """Return {track_id: SignalState} for vehicle tracks that have a result."""
        if not self._enabled:
            return {}

        result: dict[int, SignalState] = {}
        active_ids: set[int] = set()

        for tr in tracks:
            active_ids.add(tr.track_id)
            if tr.label not in _VEHICLE_LABELS:
                continue

            tid = tr.track_id
            self._ensure_track_state(tid)

            frames_since = frame_idx - self._last_frame.get(tid, -(self._classify_every_n + 1))
            if frames_since >= self._classify_every_n:
                raw = self._classify(frame, tr.bbox_xyxy)
                self._last_frame[tid] = frame_idx
                if raw is not None:
                    brake = self._brake_vote[tid].update(raw.brake)
                    left = self._left_vote[tid].update(raw.left)
                    right = self._right_vote[tid].update(raw.right)
                    self._cached[tid] = SignalState(
                        brake=brake, left=left, right=right, confidence=raw.confidence,
                    )

            cached = self._cached.get(tid)
            if cached is not None:
                result[tid] = cached

        self._purge(active_ids)
        return result

    # ── internals ──────────────────────────────────────────────────────────────

    def _ensure_track_state(self, tid: int) -> None:
        if tid not in self._brake_vote:
            self._brake_vote[tid] = _SlidingVote(self._vote_n, self._vote_m)
            self._left_vote[tid] = _SlidingVote(self._vote_n, self._vote_m)
            self._right_vote[tid] = _SlidingVote(self._vote_n, self._vote_m)
            self._cached[tid] = None

    def _crop_roi(
        self, frame: np.ndarray, bbox_xyxy: tuple[float, float, float, float]
    ) -> Optional[np.ndarray]:
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        roi_y1 = y1 + 0.6 * (y2 - y1)
        x1c, y1c = max(0, int(x1)), max(0, int(roi_y1))
        x2c, y2c = min(fw, int(x2)), min(fh, int(y2))
        if x2c <= x1c or y2c <= y1c:
            return None
        roi = frame[y1c:y2c, x1c:x2c]
        return None if roi.size == 0 else roi

    def _classify(
        self,
        frame: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> Optional[SignalState]:
        """Crop rear ROI, run inference, return raw (unsmoothed) SignalState."""
        if self._session is None and self._interpreter is None:
            return None

        roi = self._crop_roi(frame, bbox_xyxy)
        if roi is None:
            return None

        resized = cv2.resize(roi, (self._in_w, self._in_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        try:
            if self._session is not None:
                out = self._run_ort(rgb)
            else:
                out = self._run_tflite(rgb)
        except Exception as exc:
            logger.debug("SignalClassifier inference error: %s", exc)
            return None

        if out is None:
            return None

        # Apply sigmoid if model outputs raw logits
        if out.min() < 0.0 or out.max() > 1.0:
            out = 1.0 / (1.0 + np.exp(-np.clip(out, -20.0, 20.0)))

        brake = bool(out[0] >= 0.5)
        left = bool(out[1] >= 0.5)
        right = bool(out[2] >= 0.5)
        confidence = float((float(out[0]) + float(out[1]) + float(out[2])) / 3.0)
        return SignalState(brake=brake, left=left, right=right, confidence=confidence)

    # ImageNet normalisation constants (must match training preprocessing)
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _run_ort(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        # ONNX Runtime expects NCHW float32, ImageNet-normalised
        inp = rgb.astype(np.float32) / 255.0
        inp = (inp - self._MEAN) / self._STD
        inp = np.expand_dims(inp.transpose(2, 0, 1), axis=0)  # HWC → NCHW
        out = self._session.run(None, {self._ort_input_name: inp})[0]
        return out[0].astype(np.float32)

    def _run_tflite(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        inp_dtype = self._in_details[0]["dtype"]
        if inp_dtype == np.uint8:
            inp = np.expand_dims(rgb.astype(np.uint8), axis=0)
        else:
            inp = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)
        self._interpreter.set_tensor(self._in_details[0]["index"], inp)
        self._interpreter.invoke()
        return self._interpreter.get_tensor(self._out_details[0]["index"])[0].astype(np.float32)

    def _purge(self, active_ids: set[int]) -> None:
        for tid in list(self._last_frame):
            if tid not in active_ids:
                self._last_frame.pop(tid, None)
                self._cached.pop(tid, None)
                self._brake_vote.pop(tid, None)
                self._left_vote.pop(tid, None)
                self._right_vote.pop(tid, None)
