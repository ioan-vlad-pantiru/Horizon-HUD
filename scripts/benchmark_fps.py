"""FPS benchmark script (Task 2.3).

Measures per-component latency on synthetic frames so the thesis can
include a runtime table arguing real-time suitability on Raspberry Pi.

Usage
-----
    python scripts/benchmark_fps.py [--frames 200] [--model models/best_int8.tflite]

Output
------
Prints a table: component | mean ms | std ms | % of frame budget (at 30 fps).
Also prints overall FPS achieved.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.detection.detection_tflite import create_detector
from src.detection.tracking_sort import SORTTracker
from src.perception.imu_sim import IMUSimulator, Scenario
from src.perception.motion_comp import MotionCompensator
from src.perception.orientation import OrientationEstimator
from src.risk.risk_engine import RiskEngineV1
from src.risk.risk_types import CorridorConfig, RiskConfig


def _perf() -> float:
    return time.perf_counter()


def _stats(times_s: list[float]) -> tuple[float, float]:
    """Return (mean_ms, std_ms)."""
    n = len(times_s)
    if n == 0:
        return 0.0, 0.0
    mean = sum(times_s) / n
    var = sum((t - mean) ** 2 for t in times_s) / n
    return mean * 1000.0, math.sqrt(var) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark per-component FPS on synthetic 640×480 frames."
    )
    parser.add_argument("--frames", type=int, default=200,
                        help="Number of frames to run (default: 200)")
    parser.add_argument("--model", default=None, metavar="PATH",
                        help="TFLite model path.  Omit to use the dummy detector.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    fw, fh = args.width, args.height
    n_frames = args.frames
    FRAME_BUDGET_MS = 1000.0 / 30.0   # 33.3 ms at 30 fps

    # ── component construction ────────────────────────────────────────────────
    model_path: Optional[str] = None
    if args.model:
        p = Path(args.model)
        model_path = str(_PROJECT_ROOT / p) if not p.is_absolute() else str(p)

    detector = create_detector(
        detector_type="tflite" if model_path else "dummy",
        score_thresh=0.25,
        nms_thresh=0.45,
        model_path=model_path,
    )

    tracker = SORTTracker(iou_thresh=0.3, max_age=5, min_hits=2)
    imu = IMUSimulator(scenario=Scenario.STRAIGHT_VIBRATION)
    orient_est = OrientationEstimator()
    compensator = MotionCompensator(enabled=True)
    risk_engine = RiskEngineV1(RiskConfig(), CorridorConfig())

    # ── timing accumulators ───────────────────────────────────────────────────
    t_detection: list[float] = []
    t_tracking: list[float] = []
    t_imu: list[float] = []
    t_motion: list[float] = []
    t_risk: list[float] = []
    t_total: list[float] = []

    # Pre-build a black frame (no camera needed)
    black_frame = np.zeros((fh, fw, 3), dtype=np.uint8)

    print(f"Benchmarking {n_frames} frames at {fw}×{fh} …\n")
    t_run_start = _perf()

    for i in range(n_frames):
        ts = float(i) / 30.0
        t_frame_start = _perf()

        # Detection
        t0 = _perf()
        detections = detector.infer(black_frame, ts)
        t_detection.append(_perf() - t0)

        # Tracking
        t0 = _perf()
        tracks = tracker.update(detections, ts)
        t_tracking.append(_perf() - t0)

        # IMU read + orientation
        t0 = _perf()
        imu_reading = imu.read(ts)
        orient = orient_est.update(imu_reading)
        t_imu.append(_perf() - t0)

        # Motion compensation
        t0 = _perf()
        # Pass frame dims when supported (requires Task 1.1 fix); fall back gracefully.
        try:
            ego_dx, ego_dy = compensator.update_orientation(orient, frame_w=fw, frame_h=fh)
        except TypeError:
            ego_dx, ego_dy = compensator.update_orientation(orient)
        dt = 1.0 / 30.0
        comp_vels = {}
        for tr in tracks:
            vx, vy = compensator.compensate_velocity(
                tr.velocity_px_s[0], tr.velocity_px_s[1], ego_dx, ego_dy, dt
            )
            comp_vels[tr.track_id] = (vx, vy)
        t_motion.append(_perf() - t0)

        # Risk scoring
        t0 = _perf()
        risk_engine.update(tracks, (fw, fh), ts, compensated_velocities=comp_vels or None)
        t_risk.append(_perf() - t0)

        t_total.append(_perf() - t_frame_start)

    t_run_elapsed = _perf() - t_run_start
    overall_fps = n_frames / t_run_elapsed

    # ── print results ─────────────────────────────────────────────────────────
    rows = [
        ("Detection",           t_detection),
        ("Tracking (SORT)",     t_tracking),
        ("IMU + orientation",   t_imu),
        ("Motion compensation", t_motion),
        ("Risk scoring",        t_risk),
        ("--- Total pipeline",  t_total),
    ]

    col1 = max(len(r[0]) for r in rows) + 2
    header = f"{'Component':<{col1}}  {'mean (ms)':>10}  {'std (ms)':>9}  {'% budget':>9}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, times in rows:
        mean_ms, std_ms = _stats(times)
        pct = mean_ms / FRAME_BUDGET_MS * 100.0
        print(f"{name:<{col1}}  {mean_ms:>10.3f}  {std_ms:>9.3f}  {pct:>8.1f}%")
    print(sep)
    print(f"\nOverall FPS: {overall_fps:.1f}  (wall clock over {n_frames} frames)")
    model_label = model_path if model_path else "dummy detector (no model)"
    print(f"Model: {model_label}")
    print(f"Frame budget @ 30 fps: {FRAME_BUDGET_MS:.1f} ms")


if __name__ == "__main__":
    main()
