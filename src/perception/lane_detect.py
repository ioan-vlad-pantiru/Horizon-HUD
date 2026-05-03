"""Hough-line based lane centre estimator.

Detects left/right lane boundaries in the lower ROI of a frame and returns
the normalised x-centre of the lane at both the bottom and top of the ROI.
Designed to run in <5 ms on a Raspberry Pi 4 at 640×480.

Output
------
detect() returns (bottom_cx_ratio, top_cx_ratio) in [0, 1] frame-width units,
or None when detection is not confident enough.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np


class LaneDetector:
    def __init__(
        self,
        roi_top_ratio: float = 0.45,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 15,
        hough_min_length: int = 30,
        hough_max_gap: int = 25,
        slope_min: float = 0.3,
        slope_max: float = 3.0,
        ema_alpha: float = 0.25,
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

        self._ema_bot: Optional[float] = None
        self._ema_top: Optional[float] = None
        self._frames_since_detect: int = stale_frames + 1

    def detect(
        self,
        frame_bgr: np.ndarray,
        debug_frame: Optional[np.ndarray] = None,
    ) -> Optional[tuple[float, float]]:
        """Return (bottom_cx_ratio, top_cx_ratio) or None if not confident."""
        h, w = frame_bgr.shape[:2]
        roi_y = int(h * self._roi_top)

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

        result = self._fit_lane_centre(lines, w, h - roi_y, roi_y, debug_frame)

        if debug_frame is not None:
            cv2.line(debug_frame, (0, roi_y), (w, roi_y), (255, 255, 0), 1)

        if result is not None:
            raw_bot, raw_top = result
            if self._ema_bot is None:
                self._ema_bot = raw_bot
                self._ema_top = raw_top
            else:
                self._ema_bot = self._alpha * raw_bot + (1 - self._alpha) * self._ema_bot
                self._ema_top = self._alpha * raw_top + (1 - self._alpha) * self._ema_top
            self._frames_since_detect = 0
        else:
            self._frames_since_detect += 1

        if self._frames_since_detect > self._stale_frames:
            return None
        assert self._ema_bot is not None and self._ema_top is not None
        return self._ema_bot, self._ema_top

    def _fit_lane_centre(
        self,
        lines: Optional[np.ndarray],
        roi_w: int,
        roi_h: int,
        roi_y_offset: int,
        debug_frame: Optional[np.ndarray] = None,
    ) -> Optional[tuple[float, float]]:
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
            # In image coords (y down): left lane → negative slope, right → positive
            if slope < 0:
                left_pts.append((x1, y1, x2, y2))
                if debug_frame is not None:
                    cv2.line(debug_frame,
                             (int(x1), int(y1) + roi_y_offset),
                             (int(x2), int(y2) + roi_y_offset),
                             (255, 0, 0), 1)
            else:
                right_pts.append((x1, y1, x2, y2))
                if debug_frame is not None:
                    cv2.line(debug_frame,
                             (int(x1), int(y1) + roi_y_offset),
                             (int(x2), int(y2) + roi_y_offset),
                             (0, 0, 255), 1)

        if not left_pts or not right_pts:
            return None

        left_line = self._average_line(left_pts, roi_h)
        right_line = self._average_line(right_pts, roi_h)

        if left_line is None or right_line is None:
            return None

        lx_bot, lx_top = left_line
        rx_bot, rx_top = right_line

        if lx_bot >= rx_bot or lx_top >= rx_top:
            return None

        if debug_frame is not None:
            cv2.line(debug_frame,
                     (int(lx_bot), roi_y_offset + roi_h),
                     (int(lx_top), roi_y_offset), (0, 255, 255), 2)
            cv2.line(debug_frame,
                     (int(rx_bot), roi_y_offset + roi_h),
                     (int(rx_top), roi_y_offset), (0, 255, 255), 2)
            cx_px = int((lx_bot + rx_bot) / 2)
            cv2.circle(debug_frame, (cx_px, roi_y_offset + roi_h), 6, (0, 255, 0), -1)
            cx_px_top = int((lx_top + rx_top) / 2)
            cv2.circle(debug_frame, (cx_px_top, roi_y_offset), 6, (0, 255, 0), -1)

        cx_bot = ((lx_bot + rx_bot) / 2.0) / roi_w
        cx_top = ((lx_top + rx_top) / 2.0) / roi_w

        cx_bot = max(0.1, min(0.9, cx_bot))
        cx_top = max(0.1, min(0.9, cx_top))

        return cx_bot, cx_top

    @staticmethod
    def _average_line(
        pts: list[tuple[float, float, float, float]],
        roi_h: int,
    ) -> Optional[tuple[float, float]]:
        """Fit a single line through all segment endpoints; return (x_at_bottom, x_at_top)."""
        xs: list[float] = []
        ys: list[float] = []
        for x1, y1, x2, y2 in pts:
            xs += [x1, x2]
            ys += [y1, y2]

        if len(xs) < 4:
            return None

        xs_arr = np.array(xs)
        ys_arr = np.array(ys)

        # fit x = m*y + b  (more stable than y=mx+b for near-vertical lines)
        try:
            coeffs = np.polyfit(ys_arr, xs_arr, 1)
        except np.linalg.LinAlgError:
            return None

        m, b = coeffs
        x_bot = m * roi_h + b
        x_top = m * 0.0 + b

        return float(x_bot), float(x_top)
