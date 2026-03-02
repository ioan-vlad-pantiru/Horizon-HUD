"""Complementary-filter orientation estimator.

Fuses gyroscope integration with accelerometer gravity-vector correction
and optional magnetometer yaw correction.

Coordinate frame (same as imu_sim):
    x -> right,  y -> down,  z -> forward.
    Roll  = rotation about z-axis  (lean left/right).
    Pitch = rotation about x-axis  (nose up/down).
    Yaw   = rotation about y-axis  (heading).

Gravity at rest points along +y, so:
    pitch_accel = atan2(-az, ay)
    roll_accel  = atan2(ax, ay)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from src.core.types import IMUReading, Orientation

logger = logging.getLogger(__name__)


class OrientationEstimator:
    """Complementary filter for roll, pitch, yaw.

    Parameters
    ----------
    alpha : float
        Weight of gyro integration vs accel/mag correction.
        Higher → trusts gyro more (0..1, typical 0.96-0.98).
    use_mag : bool
        If True, fuse magnetometer for yaw correction.
    """

    def __init__(self, alpha: float = 0.98, use_mag: bool = True) -> None:
        self.alpha = alpha
        self.use_mag = use_mag
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._prev_ts: Optional[float] = None

    def reset(self) -> None:
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._prev_ts = None

    def update(self, imu: IMUReading) -> Orientation:
        ax, ay, az = imu.accel
        gx, gy, gz = imu.gyro

        if self._prev_ts is None:
            dt = 0.0
        else:
            dt = max(imu.timestamp - self._prev_ts, 0.0)
        self._prev_ts = imu.timestamp

        roll_gyro = self._roll + gz * dt
        pitch_gyro = self._pitch + gx * dt
        yaw_gyro = self._yaw + gy * dt

        g_len = math.sqrt(ax * ax + ay * ay + az * az)
        if g_len > 1e-6:
            roll_accel = math.atan2(ax, ay)
            pitch_accel = math.atan2(-az, ay)
        else:
            roll_accel = self._roll
            pitch_accel = self._pitch

        a = self.alpha
        self._roll = a * roll_gyro + (1.0 - a) * roll_accel
        self._pitch = a * pitch_gyro + (1.0 - a) * pitch_accel

        if self.use_mag:
            mx, my, mz = imu.mag
            mag_len = math.sqrt(mx * mx + my * my + mz * mz)
            if mag_len > 1e-6:
                yaw_mag = math.atan2(mx, -mz)
                self._yaw = a * yaw_gyro + (1.0 - a) * yaw_mag
            else:
                self._yaw = yaw_gyro
        else:
            self._yaw = yaw_gyro

        return Orientation(
            roll=self._roll,
            pitch=self._pitch,
            yaw=self._yaw,
            timestamp=imu.timestamp,
        )
