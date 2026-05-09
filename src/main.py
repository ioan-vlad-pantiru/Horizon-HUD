"""Horizon-HUD perception demo – single runnable entrypoint.

Usage:
    python -m src.main --source webcam
    python -m src.main --source webcam --imu phone
    python -m src.main --source /dev/video0 --imu hardware
    python -m src.main --source path/to/video.mp4 --config docs/config.yaml
    python -m src.main --source path/to/video.mp4 --save output.mp4

IMU sources:
    simulator  (default) simulated IMU with selectable scenario
    phone      iPhone streams via WebSocket – open printed URL in Safari
    hardware   MPU-9250 on Raspberry Pi 4 Model B via I2C (requires smbus2)

Keyboard controls:
    q  – quit
    m  – toggle ego-motion compensation
    i  – cycle IMU simulator scenario (simulator mode only)
    d  – toggle dummy / tflite detector
    r  – toggle risk overlay
    c  – toggle corridor visualisation
    z  – recalibrate straight-ahead yaw reference
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Optional

import cv2
import numpy as np
import yaml

from src.perception.corridor import build_corridor_polygon, draw_corridor, yaw_to_center_x
from src.perception.lane_detect import LaneDetector
from src.detection.detection_tflite import DummyDetector, TFLiteDetector, create_detector
from src.detection.signal_classifier import SignalClassifier
from src.perception.imu_sim import IMUSimulator, Scenario
from src.perception.motion_comp import MotionCompensator
from src.perception.orientation import OrientationEstimator
from src.risk.risk_engine import (
    RiskEngineV1,
    corridor_config_from_dict,
    risk_config_from_dict,
)
from src.risk.risk_types import RiskAssessmentV1
from src.detection.tracking_sort import SORTTracker
from src.core.types import Orientation, Track

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "docs", "config.yaml")

_RISK_COLORS: dict[str, tuple[int, int, int]] = {
    "LOW":      (0, 200, 0),
    "MEDIUM":   (0, 200, 255),
    "HIGH":     (0, 100, 255),
    "CRITICAL": (0, 0, 255),
}


# ── config loading ─────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _scenario_from_str(name: str) -> Scenario:
    mapping = {
        "straight_vibration": Scenario.STRAIGHT_VIBRATION,
        "gentle_lean":        Scenario.GENTLE_LEAN,
        "accel_brake":        Scenario.ACCEL_BRAKE,
        "constant_yaw":       Scenario.CONSTANT_YAW,
    }
    return mapping.get(name.lower(), Scenario.STRAIGHT_VIBRATION)


# ── pipeline factory ───────────────────────────────────────────────────────────

def _create_imu(source: str, cfg: dict[str, Any]):
    """Instantiate the correct IMU reader based on *source*."""
    imu_cfg = cfg.get("imu", {})
    if source == "phone":
        from src.perception.iphone_imu import iPhoneIMUReader
        return iPhoneIMUReader(
            port=imu_cfg.get("http_port", 8766),
            yaw_axis=imu_cfg.get("phone_yaw_axis", "gamma"),
        )
    if source == "hardware":
        from src.perception.hardware_imu import MPU9250Reader
        return MPU9250Reader(
            bus=imu_cfg.get("i2c_bus", 1),
            address=imu_cfg.get("i2c_address", 0x68),
        )
    return IMUSimulator(
        scenario=_scenario_from_str(imu_cfg.get("scenario", "straight_vibration")),
        noise_accel=imu_cfg.get("noise_accel", 0.05),
        noise_gyro=imu_cfg.get("noise_gyro", 0.005),
        noise_mag=imu_cfg.get("noise_mag", 0.3),
        update_hz=imu_cfg.get("update_hz", 100.0),
    )


def _build_pipeline(cfg: dict[str, Any], imu_source: str = "simulator") -> tuple:
    det_cfg = cfg.get("detector", {})
    labels = det_cfg.get("labels")
    if labels:
        labels = {int(k): v for k, v in labels.items()}
    model_path = det_cfg.get("model")
    if model_path:
        model_path = (
            os.path.join(_PROJECT_ROOT, model_path)
            if not os.path.isabs(model_path)
            else model_path
        )
    detector = create_detector(
        detector_type=det_cfg.get("type", "tflite"),
        score_thresh=det_cfg.get("score_thresh", 0.25),
        nms_thresh=det_cfg.get("nms_thresh", 0.45),
        labels=labels,
        model_path=model_path,
    )

    trk_cfg = cfg.get("tracker", {})
    tracker = SORTTracker(
        iou_thresh=trk_cfg.get("iou_thresh", 0.3),
        max_age=trk_cfg.get("max_age", 5),
        min_hits=trk_cfg.get("min_hits", 2),
        nms_thresh=det_cfg.get("nms_thresh", 0.45),
        frame_duration_s=trk_cfg.get("frame_duration_s", 1 / 30),
    )

    imu = _create_imu(imu_source, cfg)

    ori_cfg = cfg.get("orientation", {})
    orient_est = OrientationEstimator(
        alpha=ori_cfg.get("alpha", 0.98),
        use_mag=ori_cfg.get("use_mag", True),
    )

    mot_cfg = cfg.get("motion", {})
    K = None
    if "K" in mot_cfg and mot_cfg["K"] is not None:
        K = np.array(mot_cfg["K"], dtype=np.float64)
    compensator = MotionCompensator(
        K=K,
        fov_deg=mot_cfg.get("fov_deg", 70.0),
        enabled=mot_cfg.get("enabled", True),
    )

    risk_cfg = risk_config_from_dict(cfg.get("risk", {}))
    corridor_cfg = corridor_config_from_dict(cfg.get("corridor", {}))
    risk_engine = RiskEngineV1(risk_cfg, corridor_cfg)

    signal_classifier: Optional[SignalClassifier] = None
    sig_cfg = cfg.get("signal_classifier", {})
    sig_model = sig_cfg.get("model")
    if sig_model:
        sig_model_path = (
            os.path.join(_PROJECT_ROOT, sig_model)
            if not os.path.isabs(sig_model)
            else sig_model
        )
        try:
            signal_classifier = SignalClassifier(
                model_path=sig_model_path,
                classify_every_n=int(sig_cfg.get("classify_every_n", 3)),
                vote_n=int(sig_cfg.get("vote_n", 2)),
                vote_m=int(sig_cfg.get("vote_m", 5)),
            )
        except Exception as exc:
            logger.warning("SignalClassifier construction failed (%s) — disabled", exc)

    return detector, tracker, imu, orient_est, compensator, risk_engine, corridor_cfg, signal_classifier


# ── visualisation ──────────────────────────────────────────────────────────────

def _draw_all_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    top_hazard_ids: set[int],
) -> None:
    """Draw thin dim boxes for all non-hazard tracks."""
    for tr in tracks:
        if tr.track_id in top_hazard_ids:
            continue
        x1, y1, x2, y2 = (int(v) for v in tr.bbox_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
        cv2.putText(
            frame, f"#{tr.track_id}", (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA,
        )


def _draw_hazards(
    frame: np.ndarray,
    hazards: list[RiskAssessmentV1],
    tracks: list[Track],
) -> None:
    """Draw prominent overlay for top-N hazards."""
    tr_map = {tr.track_id: tr for tr in tracks}

    for i, ra in enumerate(hazards):
        tr = tr_map.get(ra.track_id)
        if tr is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in tr.bbox_xyxy)
        color = _RISK_COLORS.get(ra.risk_level, (200, 200, 200))
        thickness = 3 if ra.risk_level in ("HIGH", "CRITICAL") else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Bold warning banner on CRITICAL
        if ra.risk_level == "CRITICAL":
            cv2.rectangle(frame, (x1, y1 - 24), (x2, y1), color, -1)
            cv2.putText(
                frame, "⚠ CRITICAL", (x1 + 2, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA,
            )

        speed = math.sqrt(tr.velocity_px_s[0] ** 2 + tr.velocity_px_s[1] ** 2)
        ttc_str = f"TTC:{ra.ttc_s:.1f}s" if ra.ttc_s is not None else "TTC:--"
        dist_str = f"{ra.distance_m:.0f}m" if ra.distance_m is not None else "--"

        sig_tags: list[str] = []
        if ra.signal_state is not None:
            if ra.signal_state.brake:
                sig_tags.append("[BRAKE]")
            if ra.signal_state.hazard:
                sig_tags.append("[HAZARD]")
            elif ra.signal_state.left:
                sig_tags.append("[L]")
            elif ra.signal_state.right:
                sig_tags.append("[R]")

        line1 = f"#{tr.track_id} {tr.label}  {ra.risk_level}"
        line2 = f"{dist_str}  {ttc_str}  spd:{speed:.0f}"
        reasons_text = "  ".join(ra.reasons[:3])
        line3 = (reasons_text + "  " + "  ".join(sig_tags)).strip() if sig_tags else reasons_text

        cv2.putText(frame, line1, (x1, y1 - 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, color, 1, cv2.LINE_AA)
        cv2.putText(frame, line2, (x1, y1 - 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, color, 1, cv2.LINE_AA)
        cv2.putText(frame, line3, (x1, y1 - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (200, 200, 200), 1, cv2.LINE_AA)


def _draw_status(
    frame: np.ndarray,
    orient: Orientation,
    comp_enabled: bool,
    imu_name: str,
    detector_name: str,
    show_risk: bool,
    show_corridor: bool,
    frame_idx: int,
    yaw_delta: float = 0.0,
    lane_source: str = "imu",
) -> None:
    h = frame.shape[0]
    rpy = (
        f"R:{math.degrees(orient.roll):+6.1f}  "
        f"P:{math.degrees(orient.pitch):+6.1f}  "
        f"Y:{math.degrees(orient.yaw):+6.1f}  "
        f"dY:{math.degrees(yaw_delta):+6.1f}"
    )
    cv2.putText(frame, rpy, (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    status = (
        f"IMU:{imu_name}  "
        f"comp:{'ON' if comp_enabled else 'OFF'}  "
        f"det:{detector_name}  "
        f"risk:{'ON' if show_risk else 'OFF'}  "
        f"corr:{'ON' if show_corridor else 'OFF'}  "
        f"lane:{lane_source}  "
        f"f:{frame_idx}"
    )
    cv2.putText(frame, status, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)


# ── JSONL logging ──────────────────────────────────────────────────────────────

def _open_log(cfg: dict[str, Any], explicit_path: Optional[str]) -> Optional[Any]:
    if explicit_path:
        os.makedirs(os.path.dirname(os.path.abspath(explicit_path)), exist_ok=True)
        return open(explicit_path, "w")
    log_cfg = cfg.get("log", {})
    if not log_cfg.get("enabled", False):
        return None
    log_dir = os.path.join(_PROJECT_ROOT, log_cfg.get("directory", "docs/logs"))
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"horizon_hud_{ts}.jsonl")
    logger.info("JSONL log: %s", path)
    return open(path, "w")


def _jsonl_record(
    timestamp: float,
    frame_idx: int,
    tracks: list[Track],
    hazards: list[RiskAssessmentV1],
    orient: Orientation,
    corridor_poly: Optional[np.ndarray],
) -> str:
    hazard_map = {r.track_id: r for r in hazards}
    record: dict[str, Any] = {
        "ts": round(timestamp, 4),
        "frame": frame_idx,
        "orientation": {
            "roll": round(orient.roll, 4),
            "pitch": round(orient.pitch, 4),
            "yaw": round(orient.yaw, 4),
        },
        "tracks": [],
        "hazards": [],
    }
    for tr in tracks:
        entry: dict[str, Any] = {
            "id": tr.track_id,
            "label": tr.label,
            "bbox": [round(v, 1) for v in tr.bbox_xyxy],
            "vel": [round(v, 2) for v in tr.velocity_px_s],
            "hits": tr.hits,
            "tsu": round(tr.time_since_update, 3),
        }
        record["tracks"].append(entry)
    for ra in hazards:
        sig = ra.signal_state
        record["hazards"].append({
            "id": ra.track_id,
            "score": ra.risk_score,
            "level": ra.risk_level,
            "ttc": ra.ttc_s,
            "dist_m": ra.distance_m,
            "in_corridor": ra.in_corridor,
            "reasons": ra.reasons,
            "signal": {
                "brake": sig.brake,
                "left": sig.left,
                "right": sig.right,
                "confidence": round(sig.confidence, 3),
            } if sig is not None else None,
        })
    return json.dumps(record, default=lambda x: float(x) if hasattr(x, 'item') else x)


# ── camera utilities ───────────────────────────────────────────────────────────

class _Picamera2Capture:
    """Minimal cv2.VideoCapture-compatible wrapper around picamera2."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        from picamera2 import Picamera2
        self._cam = Picamera2()
        config = self._cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self._cam.configure(config)
        self._cam.start()
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        try:
            frame = self._cam.capture_array()
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, bgr
        except Exception:
            return False, None

    def release(self) -> None:
        if self._opened:
            self._cam.stop()
            self._opened = False


def _list_cameras_darwin() -> None:
    """Print available camera indices on macOS by probing 0-4."""
    logger.info("Probing available cameras (0-4):")
    found = False
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, _ = cap.read()
            label = f"index {i}"
            if i == 0:
                label += "  ← built-in FaceTime"
            elif i == 1:
                label += "  ← likely iPhone Continuity Camera"
            logger.info("  --source %d   %s", i, label if ret else "(no frames)")
            cap.release()
            found = True
    if not found:
        logger.info("  No cameras found.")


# ── main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Horizon-HUD perception demo")
    parser.add_argument("--source", default="webcam",
                        help="'webcam' or path to video file")
    parser.add_argument("--config", default=_DEFAULT_CONFIG,
                        help="YAML config path")
    parser.add_argument("--imu", default=None,
                        choices=["simulator", "phone", "hardware"],
                        help="IMU source: simulator (default), phone, or hardware")
    parser.add_argument("--jsonl", default=None,
                        help="Optional output JSONL path (overrides config log)")
    parser.add_argument("--save", default=None, metavar="PATH",
                        help="Save output video with overlays to this path (e.g. out.mp4)")
    parser.add_argument("--eval", action="store_true",
                        help="Enable evaluation mode: print TP/FP/lead-time stats at shutdown")
    parser.add_argument("--gt", default=None, metavar="PATH",
                        help="Ground-truth JSONL for --eval mode.  Each record: "
                             '{"frame": N, "hazard_ids": [...]}')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-28s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = _load_config(args.config)

    imu_source = args.imu or cfg.get("imu", {}).get("source", "simulator")
    detector, tracker, imu, orient_est, compensator, risk_engine, corridor_cfg, signal_classifier = (
        _build_pipeline(cfg, imu_source)
    )

    # ── open video source ──────────────────────────────────────────────────────
    src_lower = args.source.lower()
    if src_lower == "picamera2":
        try:
            cap = _Picamera2Capture()
        except Exception as exc:
            logger.error("Failed to open picamera2: %s", exc)
            sys.exit(1)
        is_webcam = True
        source = "picamera2"
    else:
        if src_lower == "webcam":
            cam_index = 0
        elif src_lower.lstrip("-").isdigit():
            cam_index = int(args.source)
        else:
            cam_index = None   # file path

        is_webcam = cam_index is not None
        source = cam_index if is_webcam else args.source

        if is_webcam and sys.platform == "darwin":
            cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error("Cannot open video source: %s", args.source)
        if is_webcam and sys.platform == "darwin":
            logger.error(
                "macOS: go to System Settings -> Privacy & Security -> Camera "
                "and enable Terminal access, then rerun."
            )
            _list_cameras_darwin()
        sys.exit(1)

    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        logger.error(
            "Camera opened but returned no frames.  "
            "Grant Camera permission to Terminal and rerun."
        )
        cap.release()
        sys.exit(1)

    logger.info("Video source opened: %s (camera index %s)", args.source, source)
    if is_webcam and sys.platform == "darwin" and source == 0:
        logger.info(
            "Using built-in camera (index 0). "
            "For iPhone Continuity Camera try --source 1"
        )

    # ── detector toggle ────────────────────────────────────────────────────────
    dummy_det = DummyDetector(
        score_thresh=cfg.get("detector", {}).get("score_thresh", 0.25)
    )
    tflite_det: Optional[TFLiteDetector] = (
        detector if isinstance(detector, TFLiteDetector) and detector.is_real else None
    )
    active_detector = detector
    detector_name = (
        "tflite"
        if isinstance(active_detector, TFLiteDetector) and active_detector.is_real
        else "dummy"
    )

    # ── eval mode ─────────────────────────────────────────────────────────────
    gt_index: dict[int, set[int]] = {}   # frame_idx → set of hazardous track IDs
    if args.eval and args.gt:
        with open(args.gt) as _gt_f:
            for _line in _gt_f:
                _line = _line.strip()
                if _line:
                    _rec = json.loads(_line)
                    gt_index[int(_rec["frame"])] = set(_rec.get("hazard_ids", []))
        logger.info("Loaded GT for %d frames from %s", len(gt_index), args.gt)
    elif args.eval:
        logger.warning("--eval set but no --gt provided; TP/FP metrics will be trivially 0.")

    # Eval accumulators (only used when --eval is set)
    _eval_tp_frames = 0   # frames where HIGH/CRITICAL fired AND GT hazard present
    _eval_fp_frames = 0   # frames where HIGH/CRITICAL fired AND no GT hazard
    _eval_fn_frames = 0   # frames where GT hazard present AND no HIGH/CRITICAL
    # lead_times[track_id] = (gt_first_frame, alert_first_frame)
    _eval_gt_first: dict[int, int] = {}
    _eval_alert_first: dict[int, int] = {}

    # ── log file ───────────────────────────────────────────────────────────────
    log_file = _open_log(cfg, args.jsonl)

    # ── lane detector ──────────────────────────────────────────────────────────
    lane_detector = LaneDetector()

    # ── video writer ───────────────────────────────────────────────────────────
    writer: Optional[cv2.VideoWriter] = None
    if args.save:
        fh0, fw0 = test_frame.shape[:2]
        src_fps = cap.get(cv2.CAP_PROP_FPS) if not is_webcam else 0.0
        out_fps = src_fps if src_fps > 0 else 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, out_fps, (fw0, fh0))
        if not writer.isOpened():
            logger.error("Cannot open video writer for path: %s", args.save)
            writer = None
        else:
            logger.info("Saving output video to %s  (%.0f FPS, %dx%d)", args.save, out_fps, fw0, fh0)

    # ── display toggles ────────────────────────────────────────────────────────
    show_risk = True
    show_corridor = False
    show_lane_debug = False

    prev_time = time.monotonic()
    pending_frame: Optional[np.ndarray] = test_frame
    frame_idx = 0

    yaw_ref: Optional[float] = None
    corridor_poly: Optional[np.ndarray] = None
    lane_source: str = "imu"

    try:
        while True:
            # ── read frame ────────────────────────────────────────────────────
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.info("End of video stream")
                    break

            now = time.monotonic()
            dt = max(now - prev_time, 1e-4)
            prev_time = now
            frame_idx += 1
            fh, fw = frame.shape[:2]

            # ── detection + tracking ──────────────────────────────────────────
            det_interval = cfg.get("detector", {}).get("detect_every_n_frames", 3)
            if frame_idx % det_interval == 0:
                detections = active_detector.infer(frame, now)
            else:
                detections = []
            tracks = tracker.update(detections, now)

            # ── IMU + orientation ─────────────────────────────────────────────
            imu_reading = imu.read(now)
            orient = orient_est.update(imu_reading)

            # ── lane-detection corridor (freeze on last-known if stale) ─────
            if yaw_ref is None:
                yaw_ref = orient.yaw
            yaw_delta = orient.yaw - yaw_ref
            center_x = yaw_to_center_x(yaw_delta, corridor_cfg.yaw_gain)

            lane_result = lane_detector.detect(frame, debug_frame=frame if show_lane_debug else None)
            if lane_result is not None:
                lx_bot, rx_bot, lx_top, rx_top, top_y_r, bot_y_r = lane_result
                corridor_poly = np.array([
                    [lx_bot * fw, bot_y_r * fh],
                    [rx_bot * fw, bot_y_r * fh],
                    [rx_top * fw, top_y_r * fh],
                    [lx_top * fw, top_y_r * fh],
                ], dtype=np.float32)
                lane_source = "lane"
            else:
                corridor_poly = build_corridor_polygon(fw, fh, corridor_cfg)
                lane_source = "imu"

            # ── motion compensation ───────────────────────────────────────────
            ego_dx, ego_dy = compensator.update_orientation(orient, frame_w=fw, frame_h=fh)
            comp_vels: dict[int, tuple[float, float]] = {}
            if compensator.enabled:
                new_tracks = []
                for tr in tracks:
                    vx, vy = compensator.compensate_velocity(
                        tr.velocity_px_s[0], tr.velocity_px_s[1], ego_dx, ego_dy, dt
                    )
                    nb = compensator.compensate_bbox(tr.bbox_xyxy, ego_dx, ego_dy)
                    comp_vels[tr.track_id] = (vx, vy)
                    new_tracks.append(Track(
                        track_id=tr.track_id,
                        bbox_xyxy=nb,
                        velocity_px_s=(vx, vy),
                        age=tr.age,
                        time_since_update=tr.time_since_update,
                        hits=tr.hits,
                        label=tr.label,
                        class_id=tr.class_id,
                    ))
                tracks = new_tracks

            # ── signal classification ─────────────────────────────────────────
            signal_states = (
                signal_classifier.run(frame, tracks, frame_idx)
                if signal_classifier is not None else {}
            )

            # ── risk assessment ───────────────────────────────────────────────
            hazards = risk_engine.update(
                tracks, (fw, fh), now,
                compensated_velocities=comp_vels if comp_vels else None,
                corridor_center_x=center_x,
                signal_states=signal_states if signal_states else None,
            )
            top_ids = {ra.track_id for ra in hazards}

            # ── eval mode tracking ────────────────────────────────────────────
            if args.eval:
                gt_hazard_ids = gt_index.get(frame_idx, set())
                # Record GT first-seen per hazard track
                for tid in gt_hazard_ids:
                    if tid not in _eval_gt_first:
                        _eval_gt_first[tid] = frame_idx
                # Record alert first-seen per track
                alerted_ids = {
                    ra.track_id for ra in hazards
                    if ra.risk_level in ("HIGH", "CRITICAL")
                }
                for tid in alerted_ids:
                    if tid not in _eval_alert_first:
                        _eval_alert_first[tid] = frame_idx
                # Frame-level TP/FP/FN
                if alerted_ids:
                    if gt_hazard_ids:
                        _eval_tp_frames += 1
                    else:
                        _eval_fp_frames += 1
                elif gt_hazard_ids:
                    _eval_fn_frames += 1

            # ── visualisation ─────────────────────────────────────────────────
            if show_corridor and corridor_poly is not None:
                draw_corridor(frame, corridor_poly)

            _draw_all_tracks(frame, tracks, top_ids if show_risk else set())

            if show_risk:
                _draw_hazards(frame, hazards, tracks)

            _draw_status(
                frame, orient,
                compensator.enabled,
                getattr(imu, "source_name", None) or imu.scenario.name,
                detector_name,
                show_risk, show_corridor, frame_idx,
                yaw_delta=yaw_delta,
                lane_source=lane_source,
            )

            if writer is not None:
                writer.write(frame)

            cv2.imshow("Horizon-HUD", frame)

            # ── logging ───────────────────────────────────────────────────────
            if log_file:
                log_file.write(
                    _jsonl_record(now, frame_idx, tracks, hazards, orient, corridor_poly)
                    + "\n"
                )

            # ── keyboard ──────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                compensator.toggle()
            elif key == ord("i"):
                imu.cycle_scenario()
                orient_est.reset()
                yaw_ref = None
            elif key == ord("z"):
                yaw_ref = orient.yaw
                logger.info("Yaw recalibrated (ref=%.3f rad)", yaw_ref)
            elif key == ord("d"):
                if detector_name == "dummy" and tflite_det is not None:
                    active_detector = tflite_det
                    detector_name = "tflite"
                else:
                    active_detector = dummy_det
                    detector_name = "dummy"
                logger.info("Detector switched -> %s", detector_name)
            elif key == ord("r"):
                show_risk = not show_risk
                logger.info("Risk overlay %s", "ON" if show_risk else "OFF")
            elif key == ord("c"):
                show_corridor = not show_corridor
                logger.info("Corridor overlay %s", "ON" if show_corridor else "OFF")
            elif key == ord("l"):
                show_lane_debug = not show_lane_debug
                logger.info("Lane debug overlay %s", "ON" if show_lane_debug else "OFF")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            logger.info("Output video saved: %s", args.save)
        cv2.destroyAllWindows()
        if log_file:
            log_file.close()
        if hasattr(imu, "close"):
            imu.close()
        logger.info("Shutdown complete  frames=%d", frame_idx)

        if args.eval:
            total_alert_frames = _eval_tp_frames + _eval_fp_frames
            total_gt_frames = _eval_tp_frames + _eval_fn_frames
            tpr = _eval_tp_frames / total_gt_frames if total_gt_frames > 0 else float("nan")
            fpr = _eval_fp_frames / max(frame_idx - total_gt_frames, 1)

            # Lead time: GT first-seen → alert first-seen, for tracks in both sets
            lead_times = []
            for tid, gt_f in _eval_gt_first.items():
                alert_f = _eval_alert_first.get(tid)
                if alert_f is not None:
                    lead_times.append(alert_f - gt_f)
            mean_lead = sum(lead_times) / len(lead_times) if lead_times else float("nan")

            print("\n" + "=" * 52)
            print("EVAL SUMMARY")
            print("=" * 52)
            print(f"  Total frames processed    : {frame_idx}")
            print(f"  Frames with GT hazard     : {total_gt_frames}")
            print(f"  Frames with alert fired   : {total_alert_frames}")
            print(f"  True positive frames (TP) : {_eval_tp_frames}")
            print(f"  False positive frames (FP): {_eval_fp_frames}")
            print(f"  False negative frames (FN): {_eval_fn_frames}")
            print(f"  True positive rate (TPR)  : {tpr:.3f}" if tpr == tpr else "  True positive rate (TPR)  : n/a (no GT hazards)")
            print(f"  False positive rate (FPR) : {fpr:.3f}")
            print(f"  Mean lead time            : {mean_lead:.1f} frames" if mean_lead == mean_lead else "  Mean lead time            : n/a")
            print("=" * 52)


if __name__ == "__main__":
    main()
