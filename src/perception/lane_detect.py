"""Hough-line based lane boundary detector.

Returns the left and right lane boundary x-positions (normalised to frame
width) at the bottom and top of the detection ROI, so the corridor polygon
can be built directly from the detected road edges.

Output
------
detect() → (lx_bot, rx_bot, lx_top, rx_top) in [0, 1] or None.
All four values are EMA-smoothed; None is returned only when the detector
has been stale for more than `stale_frames` consecutive frames.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np


class LaneDetector:
    def __init__(
        self,
        roi_top_ratio: float = 0.50,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 15,
        hough_min_length: int = 30,
        hough_max_gap: int = 25,
        slope_min: float = 0.3,
        slope_max: float = 3.0,
        ema_alpha: float = 0.40,
        stale_frames: int = 8,
    ) -> None:
        self._roi_top = roi_top_ratio
        self._canny_low = canny_low
        self._canny_high = canny_high
        self._hough_thresh = hough_threshold
        self._hough_min = hough_min_length
        self._hough_gap = hough_max_gap
        self._slope_min = slope_min
        self._slope_max = slope_max
        self._alpha = ema_alpha
        self._stale_frames = stale_frames

        self._ema: Optional[tuple[float, float, float, float]] = None
        self._frames_since_detect: int = stale_frames + 1

    def detect(
        self,
        frame_bgr: np.ndarray,
        debug_frame: Optional[np.ndarray] = None,
    ) -> Optional[tuple[float, float, float, float]]:
        """Return (lx_bot, rx_bot, lx_top, rx_top) ratios or None if stale."""
        h, w = frame_bgr.shape[:2]
        roi_y = int(h * self._roi_top)
        roi_h = h - roi_y

        roi = frame_bgr[roi_y:h, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self._canny_low, self._canny_high)

        lines = cv2.HoughLinesP(
            edges,
            rho=2,
            theta=math.pi / 180,
            threshold=self._hough_thresh,
            minLineLength=self._hough_min,
            maxLineGap=self._hough_gap,
        )

        if debug_frame is not None:
            cv2.line(debug_frame, (0, roi_y), (w, roi_y), (255, 255, 0), 1)

        raw = self._fit_boundaries(lines, w, roi_h, roi_y, debug_frame)

        if raw is not None:
            if self._ema is None:
                self._ema = raw
            else:
                self._ema = tuple(
                    self._alpha * r + (1 - self._alpha) * e
                    for r, e in zip(raw, self._ema)
                )  # type: ignore[assignment]
            self._frames_since_detect = 0
        else:
            self._frames_since_detect += 1

        if self._frames_since_detect > self._stale_frames:
            return None
        assert self._ema is not None
        return self._ema

    def _fit_boundaries(
        self,
        lines: Optional[np.ndarray],
        roi_w: int,
        roi_h: int,
        roi_y_offset: int,
        debug_frame: Optional[np.ndarray] = None,
    ) -> Optional[tuple[float, float, float, float]]:
        if lines is None:
            return None

        left_pts: list[tuple[float, float, float, float]] = []
        right_pts: list[tuple[float, float, float, float]] = []

        for line in lines:
            x1, y1, x2, y2 = line[0].astype(float)
            dx = x2 - x1
            if abs(dx) < 1e-6:
                continue
            slope = (y2 - y1) / dx
            if abs(slope) < self._slope_min or abs(slope) > self._slope_max:
                continue
            if slope < 0:
                left_pts.append((x1, y1, x2, y2))
                if debug_frame is not None:
                    cv2.line(debug_frame,
                             (int(x1), int(y1) + roi_y_offset),
                             (int(x2), int(y2) + roi_y_offset),
                             (255, 80, 80), 1)
            else:
                right_pts.append((x1, y1, x2, y2))
                if debug_frame is not None:
                    cv2.line(debug_frame,
                             (int(x1), int(y1) + roi_y_offset),
                             (int(x2), int(y2) + roi_y_offset),
                             (80, 80, 255), 1)

        if not left_pts or not right_pts:
            return None

        left_line = self._average_line(left_pts, roi_h)
        right_line = self._average_line(right_pts, roi_h)

        if left_line is None or right_line is None:
            return None

        lx_bot, lx_top = left_line
        rx_bot, rx_top = right_line

        if lx_bot >= rx_bot:
            return None
        if lx_top > rx_top:
            lx_top, rx_top = rx_top, lx_top

        if debug_frame is not None:
            cv2.line(debug_frame,
                     (int(lx_bot), roi_y_offset + roi_h),
                     (int(lx_top), roi_y_offset), (0, 255, 255), 2)
            cv2.line(debug_frame,
                     (int(rx_bot), roi_y_offset + roi_h),
                     (int(rx_top), roi_y_offset), (0, 255, 255), 2)
            cv2.circle(debug_frame, (int((lx_bot + rx_bot) / 2), roi_y_offset + roi_h), 6, (0, 255, 0), -1)
            cv2.circle(debug_frame, (int((lx_top + rx_top) / 2), roi_y_offset), 6, (0, 255, 0), -1)

        clamp = lambda v: max(0.05, min(0.95, v / roi_w))
        return clamp(lx_bot), clamp(rx_bot), clamp(lx_top), clamp(rx_top)

    @staticmethod
    def _average_line(
        pts: list[tuple[float, float, float, float]],
        roi_h: int,
    ) -> Optional[tuple[float, float]]:
        xs: list[float] = []
        ys: list[float] = []
        for x1, y1, x2, y2 in pts:
            xs += [x1, x2]
            ys += [y1, y2]
        if len(xs) < 4:
            return None
        try:
            m, b = np.polyfit(np.array(ys), np.array(xs), 1)
        except np.linalg.LinAlgError:
            return None
        return float(m * roi_h + b), float(b)
