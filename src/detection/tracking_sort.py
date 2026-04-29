"""SORT-style multi-object tracker with per-track Kalman filter.

State vector: [cx, cy, w, h, vx, vy, vw, vh]
    cx, cy – bbox center in pixels
    w, h   – bbox width, height
    vx…vh  – respective velocities (pixels / s)

Uses IoU-based matching with the Hungarian algorithm (scipy).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.core.types import Detection, Track

logger = logging.getLogger(__name__)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iou_matrix(dets: np.ndarray, trks: np.ndarray) -> np.ndarray:
    n, m = len(dets), len(trks)
    mat = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            mat[i, j] = _iou(dets[i], trks[j])
    return mat


def nms(detections: list[Detection], iou_thresh: float = 0.45) -> list[Detection]:
    if not detections:
        return []
    boxes = np.array([d.bbox_xyxy for d in detections])
    scores = np.array([d.score for d in detections])
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        ious = np.array([_iou(boxes[i], boxes[j]) for j in order[1:]])
        order = order[1:][ious < iou_thresh]
    return [detections[k] for k in keep]


class _KalmanTrack:
    _next_id: int = 1

    def __init__(self, det: Detection, timestamp: float) -> None:
        self.track_id = _KalmanTrack._next_id
        _KalmanTrack._next_id += 1

        cx = (det.bbox_xyxy[0] + det.bbox_xyxy[2]) / 2.0
        cy = (det.bbox_xyxy[1] + det.bbox_xyxy[3]) / 2.0
        w = det.bbox_xyxy[2] - det.bbox_xyxy[0]
        h = det.bbox_xyxy[3] - det.bbox_xyxy[1]

        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 50.0, 50.0, 50.0, 50.0])

        self.label = det.label
        self.class_id = det.class_id
        self.score = det.score
        self.hits = 1
        self.age = 0
        self.time_since_update = 0.0
        self._last_ts = timestamp

    @staticmethod
    def _make_F(dt: float) -> np.ndarray:
        F = np.eye(8)
        for i in range(4):
            F[i, i + 4] = dt
        return F

    @staticmethod
    def _make_Q(dt: float) -> np.ndarray:
        q_pos = 1.0
        q_vel = 5.0
        Q = np.diag([q_pos, q_pos, q_pos, q_pos, q_vel, q_vel, q_vel, q_vel]) * dt
        return Q

    _H = np.eye(4, 8)
    _R = np.diag([4.0, 4.0, 4.0, 4.0])

    def predict(self, timestamp: float) -> np.ndarray:
        dt = max(timestamp - self._last_ts, 1e-4)
        F = self._make_F(dt)
        Q = self._make_Q(dt)
        self.x = F @ self.x
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            self.P = F @ self.P @ F.T + Q
        self.P = np.nan_to_num(self.P, nan=1e3, posinf=1e6, neginf=0.0)
        self.age += 1
        self.time_since_update += dt
        self._last_ts = timestamp
        return self._get_bbox()

    def update(self, det: Detection, timestamp: float) -> None:
        cx = (det.bbox_xyxy[0] + det.bbox_xyxy[2]) / 2.0
        cy = (det.bbox_xyxy[1] + det.bbox_xyxy[3]) / 2.0
        w = det.bbox_xyxy[2] - det.bbox_xyxy[0]
        h = det.bbox_xyxy[3] - det.bbox_xyxy[1]
        z = np.array([cx, cy, w, h])

        H = self._H
        R = self._R
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P

        self.hits += 1
        self.time_since_update = 0.0
        self.label = det.label
        self.class_id = det.class_id
        self.score = det.score

    def _get_bbox(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def to_track(self) -> Track:
        bb = self._get_bbox()
        return Track(
            track_id=self.track_id,
            bbox_xyxy=(float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])),
            velocity_px_s=(float(self.x[4]), float(self.x[5])),
            age=self.age,
            time_since_update=self.time_since_update,
            hits=self.hits,
            label=self.label,
            class_id=self.class_id,
        )


class SORTTracker:
    """Simple Online and Realtime Tracking.

    Parameters
    ----------
    iou_thresh : float
        Minimum IoU for a detection–track match.
    max_age : int
        Maximum number of frames an unmatched track is kept alive before
        it is pruned.  Combined with *frame_duration_s*, the keep-alive
        window in seconds is ``max_age * frame_duration_s``.
    min_hits : int
        Minimum consecutive hits before a track is reported.
    nms_thresh : float
        NMS IoU threshold applied before tracking.
    frame_duration_s : float
        Expected duration of one frame in seconds (default 1/30 for 30 fps).
        Used to convert the frame-count ``max_age`` into a seconds threshold
        for ``time_since_update`` pruning.  Adjust when running at a fixed
        non-30-fps rate (e.g. ``1/15`` for 15 fps on Raspberry Pi).
    """

    def __init__(
        self,
        iou_thresh: float = 0.3,
        max_age: int = 5,
        min_hits: int = 2,
        nms_thresh: float = 0.45,
        frame_duration_s: float = 1 / 30,
    ) -> None:
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.nms_thresh = nms_thresh
        self.frame_duration_s = frame_duration_s
        self._tracks: list[_KalmanTrack] = []

    def update(self, detections: list[Detection], timestamp: float) -> list[Track]:
        detections = nms(detections, self.nms_thresh)

        pred_boxes = np.array([t.predict(timestamp) for t in self._tracks]) if self._tracks else np.empty((0, 4))
        det_boxes = np.array([d.bbox_xyxy for d in detections]) if detections else np.empty((0, 4))

        matched_d: set[int] = set()
        matched_t: set[int] = set()

        if len(det_boxes) > 0 and len(pred_boxes) > 0:
            iou_mat = _iou_matrix(det_boxes, pred_boxes)
            cost = 1.0 - iou_mat
            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                if iou_mat[r, c] >= self.iou_thresh:
                    self._tracks[c].update(detections[r], timestamp)
                    matched_d.add(r)
                    matched_t.add(c)

        for d_i, det in enumerate(detections):
            if d_i not in matched_d:
                self._tracks.append(_KalmanTrack(det, timestamp))

        max_staleness_s = self.max_age * self.frame_duration_s
        self._tracks = [
            t for i, t in enumerate(self._tracks)
            if i in matched_t or t.time_since_update < max_staleness_s
        ]

        return [t.to_track() for t in self._tracks if t.hits >= self.min_hits]
