"""Simulated IMU sensor with selectable driving scenarios.

Coordinate frame (right-hand, camera-aligned):
    x -> right
    y -> down
    z -> forward (into the scene)

Gravity vector at rest: accel = (0, +9.81, 0)  (y-down convention).
Gyro units: rad/s.  Magnetometer units: µT (arbitrary scale in sim).
"""

from __future__ import annotations

import logging
import math
from enum import Enum, auto
from typing import Optional

import numpy as np

from src.core.types import IMUReading

logger = logging.getLogger(__name__)

_GRAVITY = 9.81


class Scenario(Enum):
    STRAIGHT_VIBRATION = auto()
    GENTLE_LEAN = auto()
    ACCEL_BRAKE = auto()
    CONSTANT_YAW = auto()


_SCENARIO_ORDER = list(Scenario)


class IMUSimulator:
    """Generates synthetic IMU readings for each scenario.

    Parameters
    ----------
    scenario : Scenario
        Initial driving scenario.
    noise_accel : float
        Std-dev of Gaussian noise added to accelerometer (m/s²).
    noise_gyro : float
        Std-dev of Gaussian noise added to gyroscope (rad/s).
    noise_mag : float
        Std-dev of Gaussian noise added to magnetometer (µT).
    update_hz : float
        Nominal sensor rate (used only for logging / doc).
    seed : int | None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        scenario: Scenario = Scenario.STRAIGHT_VIBRATION,
        noise_accel: float = 0.05,
        noise_gyro: float = 0.005,
        noise_mag: float = 0.3,
        update_hz: float = 100.0,
        seed: Optional[int] = None,
    ) -> None:
        self.scenario = scenario
        self.noise_accel = noise_accel
        self.noise_gyro = noise_gyro
        self.noise_mag = noise_mag
        self.update_hz = update_hz
        self._rng = np.random.default_rng(seed)
        self._t0: Optional[float] = None
        logger.info("IMUSimulator created  scenario=%s", self.scenario.name)

    def cycle_scenario(self) -> Scenario:
        idx = _SCENARIO_ORDER.index(self.scenario)
        self.scenario = _SCENARIO_ORDER[(idx + 1) % len(_SCENARIO_ORDER)]
        self._t0 = None
        logger.info("IMU scenario cycled -> %s", self.scenario.name)
        return self.scenario

    def read(self, timestamp: float) -> IMUReading:
        if self._t0 is None:
            self._t0 = timestamp
        t = timestamp - self._t0

        accel, gyro, mag = self._generate(t)

        accel += self._rng.normal(0.0, self.noise_accel, 3)
        gyro += self._rng.normal(0.0, self.noise_gyro, 3)
        mag += self._rng.normal(0.0, self.noise_mag, 3)

        return IMUReading(
            accel=tuple(accel.tolist()),
            gyro=tuple(gyro.tolist()),
            mag=tuple(mag.tolist()),
            timestamp=timestamp,
        )

    def _generate(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.scenario == Scenario.STRAIGHT_VIBRATION:
            return self._straight_vibration(t)
        if self.scenario == Scenario.GENTLE_LEAN:
            return self._gentle_lean(t)
        if self.scenario == Scenario.ACCEL_BRAKE:
            return self._accel_brake(t)
        if self.scenario == Scenario.CONSTANT_YAW:
            return self._constant_yaw(t)
        return self._straight_vibration(t)

    def _straight_vibration(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vib = 0.15 * math.sin(2 * math.pi * 30 * t)
        accel = np.array([0.0, _GRAVITY + vib, 0.0])
        gyro = np.zeros(3)
        mag = np.array([20.0, 0.0, -40.0])
        return accel, gyro, mag

    def _gentle_lean(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        roll_angle = math.radians(10) * math.sin(2 * math.pi * 0.25 * t)
        roll_rate = math.radians(10) * 2 * math.pi * 0.25 * math.cos(2 * math.pi * 0.25 * t)
        accel = np.array([
            _GRAVITY * math.sin(roll_angle),
            _GRAVITY * math.cos(roll_angle),
            0.0,
        ])
        gyro = np.array([0.0, 0.0, roll_rate])
        mag = np.array([
            20.0 * math.cos(roll_angle),
            20.0 * math.sin(roll_angle),
            -40.0,
        ])
        return accel, gyro, mag

    def _accel_brake(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cycle = t % 8.0
        if cycle < 4.0:
            az = 2.0
            pitch_angle = math.radians(-5)
        else:
            az = -3.0
            pitch_angle = math.radians(8)
        accel = np.array([
            0.0,
            _GRAVITY * math.cos(pitch_angle),
            az + _GRAVITY * math.sin(pitch_angle),
        ])
        pitch_rate = 0.0
        gyro = np.array([pitch_rate, 0.0, 0.0])
        mag = np.array([20.0, 0.0, -40.0])
        return accel, gyro, mag

    def _constant_yaw(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        yaw_rate = math.radians(20)
        yaw_angle = yaw_rate * t
        lean = math.radians(8)
        accel = np.array([
            _GRAVITY * math.sin(lean),
            _GRAVITY * math.cos(lean),
            0.0,
        ])
        gyro = np.array([0.0, yaw_rate, 0.0])
        mag = np.array([
            20.0 * math.cos(yaw_angle),
            0.0,
            -40.0 * math.sin(yaw_angle),
        ])
        return accel, gyro, mag
