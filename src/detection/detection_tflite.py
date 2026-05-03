"""Detection adapter: TFLite when available, deterministic DummyDetector otherwise."""

from __future__ import annotations

import logging
import math
import os
from typing import Optional, Protocol

import cv2
import numpy as np

from src.core.types import Detection

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_MODEL = os.path.join(_PROJECT_ROOT, "models", "best_int8.tflite")

_DEFAULT_LABELS: dict[int, str] = {
    0: "vehicle",
    1: "pedestrian",
    2: "cyclist",
    3: "road_obstacle",
}


class DetectorInterface(Protocol):
    def infer(self, frame_bgr: np.ndarray, timestamp: float) -> list[Detection]: ...


class DummyDetector:
    """Returns deterministic synthetic detections for pipeline testing."""

    def __init__(self, score_thresh: float = 0.25, labels: Optional[dict[int, str]] = None) -> None:
        self.score_thresh = score_thresh
        self._labels = labels or _DEFAULT_LABELS
        self._frame_count = 0

    def infer(self, frame_bgr: np.ndarray, timestamp: float) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        self._frame_count += 1
        t = self._frame_count

        dets: list[Detection] = []

        cx = w * 0.5 + w * 0.15 * math.sin(t * 0.05)
        cy = h * 0.55 + h * 0.05 * math.sin(t * 0.03)
        bw = w * (0.10 + 0.03 * math.sin(t * 0.02))
        bh = h * (0.15 + 0.04 * math.sin(t * 0.02))
        dets.append(Detection(
            bbox_xyxy=(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2),
            score=0.85,
            class_id=1,
            label="pedestrian",
        ))

        cx2 = w * 0.25 + w * 0.1 * math.cos(t * 0.04)
        cy2 = h * 0.6
        bw2 = w * (0.18 + 0.02 * math.sin(t * 0.015))
        bh2 = h * 0.12
        dets.append(Detection(
            bbox_xyxy=(cx2 - bw2 / 2, cy2 - bh2 / 2, cx2 + bw2 / 2, cy2 + bh2 / 2),
            score=0.72,
            class_id=0,
            label="vehicle",
        ))

        cx3 = w * 0.75
        cy3 = h * 0.45
        bw3 = w * 0.06
        bh3 = h * 0.08
        dets.append(Detection(
            bbox_xyxy=(cx3 - bw3 / 2, cy3 - bh3 / 2, cx3 + bw3 / 2, cy3 + bh3 / 2),
            score=0.60,
            class_id=2,
            label="cyclist",
        ))

        return [d for d in dets if d.score >= self.score_thresh]


class TFLiteDetector:
    """Loads detector.tflite and runs inference.

    Falls back to DummyDetector if tflite runtime is unavailable or if
    output decoding fails.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        score_thresh: float = 0.25,
        nms_thresh: float = 0.45,
        labels: Optional[dict[int, str]] = None,
    ) -> None:
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self._labels = labels or _DEFAULT_LABELS
        self._interpreter = None
        self._is_yolov8 = False
        self._num_classes = 0
        self._fallback = DummyDetector(score_thresh, labels)

        try:
            Interpreter = self._load_interpreter_class()
            if Interpreter is None:
                raise ImportError("No TFLite runtime found")
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            self._interpreter = Interpreter(model_path=model_path, num_threads=4)
            self._interpreter.allocate_tensors()
            self._in_details = self._interpreter.get_input_details()
            self._out_details = self._interpreter.get_output_details()
            self._in_h = self._in_details[0]["shape"][1]
            self._in_w = self._in_details[0]["shape"][2]
            out_shape = self._out_details[0]["shape"]
            # YOLOv8: [1, 4+nc, num_anchors]  SSD: 3 tensors
            self._is_yolov8 = len(self._out_details) == 1 and len(out_shape) == 3
            if self._is_yolov8:
                self._num_classes = out_shape[1] - 4
            logger.info(
                "TFLiteDetector loaded %s  input=%dx%d  yolov8=%s",
                model_path, self._in_w, self._in_h, self._is_yolov8,
            )
        except Exception as exc:
            logger.warning("TFLiteDetector init failed (%s), using DummyDetector", exc)
            self._interpreter = None

    @staticmethod
    def _load_interpreter_class() -> Optional[type]:
        try:
            from ai_edge_litert.interpreter import Interpreter
            return Interpreter
        except ImportError:
            pass
        try:
            from tflite_runtime.interpreter import Interpreter
            return Interpreter
        except ImportError:
            pass
        try:
            from tensorflow.lite.python.interpreter import Interpreter
            return Interpreter
        except ImportError:
            pass
        return None

    def infer(self, frame_bgr: np.ndarray, timestamp: float) -> list[Detection]:
        if self._interpreter is None:
            return self._fallback.infer(frame_bgr, timestamp)
        try:
            return self._run_tflite(frame_bgr)
        except Exception as exc:
            logger.warning("TFLite inference failed (%s), falling back to DummyDetector", exc)
            return self._fallback.infer(frame_bgr, timestamp)

    def _run_tflite(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._in_w, self._in_h))

        inp_dtype = self._in_details[0]["dtype"]
        if inp_dtype == np.uint8:
            inp = np.expand_dims(resized.astype(np.uint8), axis=0)
        else:
            inp = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        self._interpreter.set_tensor(self._in_details[0]["index"], inp)
        self._interpreter.invoke()

        if self._is_yolov8:
            return self._decode_yolov8(h, w)
        return self._decode_ssd(h, w)

    def _decode_yolov8(self, frame_h: int, frame_w: int) -> list[Detection]:
        """Decode YOLOv8 TFLite output.

        Output tensor shape: [1, 4 + num_classes, num_anchors]
        Channels 0-3: cx, cy, w, h  (in model-input pixel units)
        Channels 4+:  class scores (raw logits or post-sigmoid floats)
        """
        raw = self._interpreter.get_tensor(self._out_details[0]["index"])[0]
        # raw: [4+nc, num_anchors]
        cx = raw[0]
        cy = raw[1]
        bw = raw[2]
        bh = raw[3]
        class_scores = raw[4:]  # [nc, num_anchors]

        # onnx2tf normalises bbox coords to [0,1]; multiply by frame dimensions.
        # (Original PyTorch pixel-space coords are divided by input_size during
        # ONNX->TF conversion, so we recover them by multiplying by frame size.)
        dets: list[Detection] = []
        class_ids = np.argmax(class_scores, axis=0)
        best_scores = class_scores[class_ids, np.arange(class_scores.shape[1])]

        mask = best_scores >= self.score_thresh
        for i in np.where(mask)[0]:
            cid = int(class_ids[i])
            score = float(best_scores[i])
            x1 = float((cx[i] - bw[i] / 2) * frame_w)
            y1 = float((cy[i] - bh[i] / 2) * frame_h)
            x2 = float((cx[i] + bw[i] / 2) * frame_w)
            y2 = float((cy[i] + bh[i] / 2) * frame_h)
            label = self._labels.get(cid, f"class_{cid}")
            dets.append(Detection(
                bbox_xyxy=(x1, y1, x2, y2),
                score=score,
                class_id=cid,
                label=label,
            ))
        return dets

    def _decode_ssd(self, frame_h: int, frame_w: int) -> list[Detection]:
        """Decode SSD-MobileNet style outputs.

        # TODO: Adjust tensor indices if your SSD model differs from the
        # standard layout (boxes[0], classes[1], scores[2]).
        """
        try:
            boxes = self._interpreter.get_tensor(self._out_details[0]["index"])[0]
            classes = self._interpreter.get_tensor(self._out_details[1]["index"])[0].astype(int)
            scores = self._interpreter.get_tensor(self._out_details[2]["index"])[0]
        except (IndexError, KeyError) as exc:
            logger.warning("SSD output decode failed: %s", exc)
            return []

        dets: list[Detection] = []
        for i, score in enumerate(scores):
            if score < self.score_thresh:
                continue
            y1, x1, y2, x2 = boxes[i]
            label = self._labels.get(classes[i], f"class_{classes[i]}")
            dets.append(Detection(
                bbox_xyxy=(
                    float(x1 * frame_w),
                    float(y1 * frame_h),
                    float(x2 * frame_w),
                    float(y2 * frame_h),
                ),
                score=float(score),
                class_id=int(classes[i]),
                label=label,
            ))
        return dets

    @property
    def is_real(self) -> bool:
        return self._interpreter is not None


def create_detector(
    detector_type: str = "tflite",
    score_thresh: float = 0.25,
    nms_thresh: float = 0.45,
    labels: Optional[dict[int, str]] = None,
    model_path: Optional[str] = None,
) -> DetectorInterface:
    if detector_type == "dummy":
        logger.info("Using DummyDetector by config")
        return DummyDetector(score_thresh, labels)
    kwargs: dict = dict(score_thresh=score_thresh, nms_thresh=nms_thresh, labels=labels)
    if model_path:
        kwargs["model_path"] = model_path
    return TFLiteDetector(**kwargs)
