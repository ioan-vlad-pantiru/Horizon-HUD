#!/usr/bin/env python3
"""
hud_test.py

Test object detection (SSD-MobileNet V1 COCO) and pose estimation (MoveNet Lightning INT8)
on your laptop/webcam.

Outputs:
 - Red boxes + blue arrows for cars (motion vectors)
 - Green boxes for other COCO objects (people, animals…)
 - Yellow dots for each person's 17 pose keypoints
 - ⚠️ Warning when a detected car gets too close
Press 'q' to quit.
"""

import os
import urllib.request
import zipfile
import requests
import numpy as np
import cv2
import tensorflow as tf

# Use LiteRT interpreter (new recommended approach)
try:
    from ai_edge_litert.interpreter import Interpreter
    print("Using LiteRT interpreter")
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        print("Using tflite_runtime.Interpreter")
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
        print("Using tensorflow.lite.Interpreter (deprecated)")

# ─── PROXIMITY WARNING THRESHOLD ────────────────────────────────────────────────
# If a car's bounding-box width > this fraction of frame width, issue: warning
CLOSE_WARN_RATIO = 0.3
SCORE_THRESHOLD  = 0.5

# ─── URLS & PATHS ───────────────────────────────────────────────────────────────
# (Download code is commented out; assume models and labels are already in place.)

# 1) SSD MobileNet V1 quantized COCO 300×300
MODEL_ZIP_URL      = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
MODEL_ZIP          = "coco_ssd_mobilenet_v1.zip"
DETECT_MODEL_FILE  = "detect.tflite"
LABELS_FILE        = "labelmap.txt"   # ensure you have the correct COCO label file

# 2) MoveNet SinglePose Lightning INT8
POSE_MODEL_URL     = "https://github.com/tensorflow/examples/raw/master/lite/examples/pose_estimation/raspberry_pi/models/movenet_singlepose_lightning.tflite"
POSE_MODEL_FILE    = "pose.tflite"

# ─── MODEL LOADING ───────────────────────────────────────────────────────────────

def load_models():
    # Detector
    det_int = Interpreter(model_path=DETECT_MODEL_FILE)
    det_int.allocate_tensors()
    det_in  = det_int.get_input_details()
    det_out = det_int.get_output_details()
    H_det, W_det = det_in[0]['shape'][1:3]

    # Pose
    pose_int = Interpreter(model_path=POSE_MODEL_FILE)
    pose_int.allocate_tensors()
    pose_in  = pose_int.get_input_details()
    pose_out = pose_int.get_output_details()
    H_pose, W_pose = pose_in[0]['shape'][1:3]

    return (det_int, det_in, det_out, H_det, W_det,
            pose_int, pose_in, pose_out, H_pose, W_pose)

def load_labels():
    with open(LABELS_FILE, "r") as f:
        return [l.strip() for l in f.readlines()]

# ─── INFERENCE & DRAWING ─────────────────────────────────────────────────────────

def detect_and_draw(frame, det_int, det_in, det_out, H_det, W_det, labels, prev_cars):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (W_det, H_det))
    inp = np.expand_dims(small, 0)
    if det_in[0]['dtype'] == np.uint8:
        inp = inp.astype(np.uint8)
    else:
        inp = (np.float32(inp) - 127.5) / 127.5

    det_int.set_tensor(det_in[0]['index'], inp)
    det_int.invoke()

    boxes   = det_int.get_tensor(det_out[0]['index'])[0]
    classes = det_int.get_tensor(det_out[1]['index'])[0].astype(int)
    scores  = det_int.get_tensor(det_out[2]['index'])[0]

    h, w, _ = frame.shape
    new_cars = {}

    for i, score in enumerate(scores):
        if score < SCORE_THRESHOLD:
            continue

        y1, x1, y2, x2 = boxes[i]
        x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        label = labels[classes[i]]

        # Handle cars: only show outline when too close
        if label == "car":
            # track centroid for future motion arrows if desired
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            new_cars[i] = (cx, cy)

            box_width = x2 - x1
            if (box_width / w) > CLOSE_WARN_RATIO:
                # draw thick red outline only
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # skip normal drawing and labels for cars
            continue

        # Non-car detections (people, animals, obstacles, etc.)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label}:{int(score*100)}%",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return frame, new_cars


def estimate_pose(frame, pose_int, pose_in, pose_out):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (pose_in[0]['shape'][2], pose_in[0]['shape'][1]))
    inp = np.expand_dims(small, 0).astype(np.uint8)

    pose_int.set_tensor(pose_in[0]['index'], inp)
    pose_int.invoke()
    kpts = pose_int.get_tensor(pose_out[0]['index'])[0][0]  # [17,3]

    h, w, _ = frame.shape
    for y, x, c in kpts:
        if c < 0.3:
            continue
        cv2.circle(frame, (int(x*w), int(y*h)), 3, (0,255,255), -1)
    return frame

# ─── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    # prepare_models()  # models must be downloaded beforehand
    (det_int, det_in, det_out, H_det, W_det,
     pose_int, pose_in, pose_out, H_pose, W_pose) = load_models()
    labels = load_labels()

    cap = cv2.VideoCapture(0)
    prev_cars = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, prev_cars = detect_and_draw(
            frame, det_int, det_in, det_out, H_det, W_det, labels, prev_cars
        )
        frame = estimate_pose(frame, pose_int, pose_in, pose_out)

        cv2.imshow("HUD Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()