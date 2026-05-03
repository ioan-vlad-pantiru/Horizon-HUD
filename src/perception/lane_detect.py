"""Sliding-window histogram lane detector.

Only processes the bottom 20% of the frame where the road surface is
guaranteed — avoiding sky, lamp posts, and distant features entirely.

Output
------
detect() → (lx_bot, rx_bot, lx_top, rx_top, top_y_ratio, bot_y_ratio) or None.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class LaneDetector:
    def __init__(
        self,
        roi_top_ratio: float = 0.80,
        n_strips: int = 4,
        min_peak_pixels: int = 20,
        min_strips_per_side: int = 2,
        min_convergence_ratio: float = 0.03,
        ema_alpha: float = 0.15,
        stale_frames: int = 10,
    ) -> None:
        self._roi_top = roi_top_ratio
        self._n_strips = n_strips
        self._min_peak = min_peak_pixels
        self._min_strips = min_strips_per_side
        self._min_conv = min_convergence_ratio
        self._alpha = ema_alpha
        self._stale_frames = stale_frames

        self._ema: Optional[tuple[float, float, float, float]] = None
        self._last_top_y: float = 0.5
        self._last_bot_y: float = 1.0
        self._frames_since_detect: int = stale_frames + 1

    def detect(
        self,
        frame_bgr: np.ndarray,
        debug_frame: Optional[np.ndarray] = None,
    ) -> Optional[tuple[float, float, float, float, float, float]]:
        """Return (lx_bot, rx_bot, lx_top, rx_top, top_y_ratio, bot_y_ratio) or None."""
        h, w = frame_bgr.shape[:2]
        roi_y = int(h * self._roi_top)
        roi_h = h - roi_y
        roi = frame_bgr[roi_y:h, :]

        mask = self._color_mask(roi)

        if debug_frame is not None:
            cv2.line(debug_frame, (0, roi_y), (w, roi_y), (0, 255, 255), 1)
            vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            debug_frame[roi_y:h, :] = cv2.addWeighted(
                debug_frame[roi_y:h, :], 0.5, vis, 0.5, 0
            )

        raw = self._sliding_window(mask, w, roi_h, roi_y, h, debug_frame)

        if raw is not None:
            lx_b, rx_b, lx_t, rx_t, top_y, bot_y = raw
            smooth = (lx_b, rx_b, lx_t, rx_t)
            self._ema = smooth if self._ema is None else tuple(
                self._alpha * r + (1 - self._alpha) * e
                for r, e in zip(smooth, self._ema)  # type: ignore[arg-type]
            )  # type: ignore[assignment]
            self._last_top_y = top_y
            self._last_bot_y = bot_y
            self._frames_since_detect = 0
        else:
            self._frames_since_detect += 1

        if self._frames_since_detect > self._stale_frames:
            return None
        assert self._ema is not None
        lx_b, rx_b, lx_t, rx_t = self._ema
        return lx_b, rx_b, lx_t, rx_t, self._last_top_y, self._last_bot_y  # type: ignore[return-value]

    @staticmethod
    def _color_mask(roi_bgr: np.ndarray) -> np.ndarray:
        """Strict white+yellow mask to isolate lane markings only."""
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        white  = cv2.inRange(hsv, (0,   0, 200), (180, 30, 255))
        yellow = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
        mask = cv2.bitwise_or(white, yellow)
        return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    def _sliding_window(
        self,
        mask: np.ndarray,
        frame_w: int,
        roi_h: int,
        roi_y_offset: int,
        frame_h: int,
        debug_frame: Optional[np.ndarray] = None,
    ) -> Optional[tuple[float, float, float, float, float, float]]:
        mid = frame_w // 2
        strip_h = roi_h // self._n_strips

        left_pts:  list[tuple[float, float]] = []
        right_pts: list[tuple[float, float]] = []

        for i in range(self._n_strips):
            y_bot = roi_h - i * strip_h
            y_top = y_bot - strip_h
            strip = mask[y_top:y_bot, :]
            hist = np.sum(strip, axis=0).astype(np.float32)
            strip_cy = roi_y_offset + (y_top + y_bot) // 2

            lx = int(np.argmax(hist[:mid]))
            rx = mid + int(np.argmax(hist[mid:]))

            if hist[lx] >= self._min_peak * 255:
                left_pts.append((float(lx), float(strip_cy)))
                if debug_frame is not None:
                    cv2.circle(debug_frame, (lx, strip_cy), 6, (255, 80, 80), -1)

            if hist[rx] >= self._min_peak * 255:
                right_pts.append((float(rx), float(strip_cy)))
                if debug_frame is not None:
                    cv2.circle(debug_frame, (rx, strip_cy), 6, (80, 80, 255), -1)

        if len(left_pts) < self._min_strips or len(right_pts) < self._min_strips:
            return None

        all_ys = [p[1] for p in left_pts + right_pts]
        top_y = min(all_ys)
        bot_y = max(all_ys)

        lx_bot, lx_top = self._extrapolate(left_pts, bot_y, top_y)
        rx_bot, rx_top = self._extrapolate(right_pts, bot_y, top_y)

        if lx_bot >= rx_bot or lx_top >= rx_top:
            return None
        lane_w = rx_bot - lx_bot
        if lane_w < frame_w * 0.10 or lane_w > frame_w * 0.85:
            return None
        # Reject vertical objects: real lane lines converge inward (x changes between top/bottom)
        if abs(lx_bot - lx_top) < frame_w * self._min_conv:
            return None
        if abs(rx_bot - rx_top) < frame_w * self._min_conv:
            return None

        if debug_frame is not None:
            cv2.line(debug_frame, (int(lx_bot), int(bot_y)), (int(lx_top), int(top_y)), (0, 255, 100), 2)
            cv2.line(debug_frame, (int(rx_bot), int(bot_y)), (int(rx_top), int(top_y)), (0, 255, 100), 2)

        c = lambda v: max(0.05, min(0.95, v / frame_w))
        return c(lx_bot), c(rx_bot), c(lx_top), c(rx_top), top_y / frame_h, bot_y / frame_h

    @staticmethod
    def _extrapolate(
        pts: list[tuple[float, float]],
        y_bot: float,
        y_top: float,
    ) -> tuple[float, float]:
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        try:
            m, b = np.polyfit(ys, xs, 1)
        except np.linalg.LinAlgError:
            mean_x = float(np.mean(xs))
            return mean_x, mean_x
        return float(m * y_bot + b), float(m * y_top + b)
