"""Tests for motion compensation math and orientation estimator."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.perception.imu_sim import IMUSimulator, Scenario
from src.perception.motion_comp import MotionCompensator, _rotation_matrix, default_K
from src.perception.orientation import OrientationEstimator
from src.core.types import IMUReading, Orientation


class TestDefaultK(unittest.TestCase):
    def test_principal_point_at_center(self) -> None:
        K = default_K(640, 480, 70.0)
        self.assertAlmostEqual(K[0, 2], 320.0)
        self.assertAlmostEqual(K[1, 2], 240.0)

    def test_focal_length_positive(self) -> None:
        K = default_K(1920, 1080, 90.0)
        self.assertGreater(K[0, 0], 0)
        self.assertAlmostEqual(K[0, 0], K[1, 1])

    def test_fov_90_focal(self) -> None:
        K = default_K(1000, 500, 90.0)
        expected_fx = 1000 / (2.0 * math.tan(math.radians(45)))
        self.assertAlmostEqual(K[0, 0], expected_fx, places=5)


class TestRotationMatrix(unittest.TestCase):
    def test_identity_at_zero(self) -> None:
        R = _rotation_matrix(0, 0, 0)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_small_yaw_rotation(self) -> None:
        dy = math.radians(1)
        R = _rotation_matrix(0, 0, dy)
        self.assertAlmostEqual(R[0, 0], math.cos(dy), places=6)

    def test_determinant_one(self) -> None:
        R = _rotation_matrix(0.1, -0.2, 0.3)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=6)

    def test_orthogonal(self) -> None:
        R = _rotation_matrix(0.05, 0.1, -0.05)
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=6)


class TestMotionCompensator(unittest.TestCase):
    def test_no_rotation_no_offset(self) -> None:
        mc = MotionCompensator(fov_deg=70.0, enabled=True)
        o1 = Orientation(0.0, 0.0, 0.0, 0.0)
        o2 = Orientation(0.0, 0.0, 0.0, 0.01)
        mc.update_orientation(o1)
        dx, dy = mc.update_orientation(o2)
        self.assertAlmostEqual(dx, 0.0, places=3)
        self.assertAlmostEqual(dy, 0.0, places=3)

    def test_yaw_produces_horizontal_offset(self) -> None:
        mc = MotionCompensator(fov_deg=70.0, enabled=True)
        mc.update_orientation(Orientation(0, 0, 0, 0.0))
        dx, dy = mc.update_orientation(Orientation(0, 0, math.radians(2), 0.01))
        self.assertNotAlmostEqual(dx, 0.0, places=1)
        self.assertAlmostEqual(dy, 0.0, places=1)

    def test_disabled_returns_zero(self) -> None:
        mc = MotionCompensator(fov_deg=70.0, enabled=False)
        mc.update_orientation(Orientation(0, 0, 0, 0.0))
        dx, dy = mc.update_orientation(Orientation(0.1, 0.1, 0.1, 0.01))
        self.assertEqual(dx, 0.0)
        self.assertEqual(dy, 0.0)

    def test_compensate_bbox_shifts(self) -> None:
        mc = MotionCompensator()
        bbox = (100.0, 100.0, 200.0, 200.0)
        out = mc.compensate_bbox(bbox, 5.0, 3.0)
        self.assertAlmostEqual(out[0], 95.0)
        self.assertAlmostEqual(out[1], 97.0)

    def test_compensate_velocity(self) -> None:
        mc = MotionCompensator()
        vx, vy = mc.compensate_velocity(100.0, 50.0, 10.0, 5.0, 0.033)
        self.assertAlmostEqual(vx, 100.0 - 10.0 / 0.033, places=1)

    def test_toggle(self) -> None:
        mc = MotionCompensator(enabled=True)
        self.assertFalse(mc.toggle())
        self.assertTrue(mc.toggle())


class TestOrientationEstimator(unittest.TestCase):
    def test_stationary_converges_to_zero(self) -> None:
        est = OrientationEstimator(alpha=0.98, use_mag=True)
        imu = IMUReading(
            accel=(0.0, 9.81, 0.0),
            gyro=(0.0, 0.0, 0.0),
            mag=(20.0, 0.0, -40.0),
            timestamp=0.0,
        )
        for i in range(200):
            o = est.update(IMUReading(
                accel=imu.accel, gyro=imu.gyro, mag=imu.mag,
                timestamp=0.01 * (i + 1),
            ))
        self.assertAlmostEqual(o.roll, 0.0, places=1)
        self.assertAlmostEqual(o.pitch, 0.0, places=1)

    def test_constant_gyro_drifts_yaw(self) -> None:
        est = OrientationEstimator(alpha=0.98, use_mag=False)
        for i in range(100):
            o = est.update(IMUReading(
                accel=(0.0, 9.81, 0.0),
                gyro=(0.0, math.radians(10), 0.0),
                mag=(0.0, 0.0, 0.0),
                timestamp=0.01 * i,
            ))
        self.assertGreater(abs(o.yaw), math.radians(5))

    def test_reset(self) -> None:
        est = OrientationEstimator()
        est.update(IMUReading((1, 9, 0), (0.1, 0.1, 0.1), (20, 0, -40), 0.0))
        est.reset()
        o = est.update(IMUReading((0, 9.81, 0), (0, 0, 0), (20, 0, -40), 1.0))
        self.assertAlmostEqual(o.roll, 0.0, places=2)


class TestIMUSimulator(unittest.TestCase):
    def test_all_scenarios_run(self) -> None:
        for sc in Scenario:
            sim = IMUSimulator(scenario=sc, seed=42)
            r = sim.read(0.0)
            self.assertEqual(len(r.accel), 3)
            self.assertEqual(len(r.gyro), 3)

    def test_cycle(self) -> None:
        sim = IMUSimulator(scenario=Scenario.STRAIGHT_VIBRATION)
        s2 = sim.cycle_scenario()
        self.assertEqual(s2, Scenario.GENTLE_LEAN)

    def test_deterministic_with_seed(self) -> None:
        a = IMUSimulator(scenario=Scenario.STRAIGHT_VIBRATION, seed=123)
        b = IMUSimulator(scenario=Scenario.STRAIGHT_VIBRATION, seed=123)
        r_a = a.read(1.0)
        r_b = b.read(1.0)
        self.assertEqual(r_a.accel, r_b.accel)


if __name__ == "__main__":
    unittest.main()
