"""
Test YOLOv8 model on images and save visualizations.
Safety-focused: uses low confidence threshold to maximize recall.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Test YOLOv8 model on images")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--source", type=str, required=True, help="Image/dir/video path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save-dir", type=str, default="models/yolo_experiments/test_results")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=args.save_dir,
        name="predictions",
        exist_ok=True,
        verbose=True,
    )

    for r in results:
        print(f"\nImage: {r.path}")
        if r.boxes is not None and len(r.boxes):
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                name = r.names[cls_id]
                print(f"  {name:14s} conf={conf:.3f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        else:
            print("  No detections")


if __name__ == "__main__":
    main()
