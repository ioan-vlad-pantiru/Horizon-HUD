"""Ego-motion compensation using orientation deltas.

Approximates the image-plane shift caused by camera rotation between frames
and removes it from tracked bounding boxes / velocities so that residual
motion reflects real-world object movement only.

The approach:
    1. Compute delta roll, pitch, yaw between consecutive frames.
    2. Build a small-angle rotation matrix R from delta_rpy.
    3. Project the image center through K^-1, rotate, re-project through K.
    4. The pixel-level translation of the image center is the ego-motion
       offset that gets subtracted from each bbox.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from src.core.types import Orientation

logger = logging.getLogger(__name__)


def default_K(frame_w: int, frame_h: int, fov_deg: float = 70.0) -> np.ndarray:
    """Build a pinhole camera intrinsic matrix from frame size and horizontal FOV."""
    fx = frame_w / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    fy = fx
    cx = frame_w / 2.0
    cy = frame_h / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _rotation_matrix(dr: float, dp: float, dy: float) -> np.ndarray:
    """Small-angle rotation matrix from delta roll, pitch, yaw (radians)."""
    cr, sr = math.cos(dr), math.sin(dr)
    cp, sp = math.cos(dp), math.sin(dp)
    cy, sy = math.cos(dy), math.sin(dy)

    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Ry @ Rx @ Rz


class MotionCompensator:
    """Removes ego-rotation artefacts from bounding boxes.

    Parameters
    ----------
    K : np.ndarray | None
        3x3 camera intrinsic matrix.  If None, built from frame size + FOV.
    fov_deg : float
        Horizontal FOV used when K is not provided.
    enabled : bool
        Whether compensation is active.
    """

    def __init__(
        self,
        K: np.ndarray | None = None,
        fov_deg: float = 70.0,
        enabled: bool = True,
    ) -> None:
        self._K = K
        self.fov_deg = fov_deg
        self.enabled = enabled
        self._prev_orient: Orientation | None = None

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        logger.info("Motion compensation %s", "ON" if self.enabled else "OFF")
        return self.enabled

    def _get_K(self, fw: int, fh: int) -> np.ndarray:
        if self._K is not None:
            return self._K
        return default_K(fw, fh, self.fov_deg)

    def update_orientation(self, orient: Orientation) -> tuple[float, float]:
        """Store new orientation and return (dx, dy) ego-motion offset in pixels.

        Returns (0, 0) when disabled or on the very first frame.
        """
        if self._prev_orient is None or not self.enabled:
            self._prev_orient = orient
            return 0.0, 0.0

        dr = orient.roll - self._prev_orient.roll
        dp = orient.pitch - self._prev_orient.pitch
        dy = orient.yaw - self._prev_orient.yaw
        self._prev_orient = orient
        self._delta_rpy = (dr, dp, dy)
        return self._compute_offset(dr, dp, dy)

    def _compute_offset(self, dr: float, dp: float, dy: float) -> tuple[float, float]:
        K = self._get_K(640, 480)
        R = _rotation_matrix(dr, dp, dy)
        H = K @ R @ np.linalg.inv(K)
        cx = K[0, 2]
        cy = K[1, 2]
        pt = H @ np.array([cx, cy, 1.0])
        pt /= pt[2]
        return float(pt[0] - cx), float(pt[1] - cy)

    def compensate_bbox(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        dx: float,
        dy: float,
    ) -> tuple[float, float, float, float]:
        """Shift bbox by the ego-motion offset."""
        return (
            bbox_xyxy[0] - dx,
            bbox_xyxy[1] - dy,
            bbox_xyxy[2] - dx,
            bbox_xyxy[3] - dy,
        )

    def compensate_velocity(
        self,
        vx: float,
        vy: float,
        dx: float,
        dy: float,
        dt: float,
    ) -> tuple[float, float]:
        """Remove ego-rotation component from pixel velocity."""
        if dt < 1e-9:
            return vx, vy
        return vx - dx / dt, vy - dy / dt
