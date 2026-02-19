"""
Train YOLOv8n for Horizon-HUD safety detection.
Optimized for high recall on safety-critical classes (vehicle, pedestrian).
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n for Horizon-HUD")
    parser.add_argument("--data", type=str, default="ml/config/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--project", type=str, default="models/yolo_experiments")
    parser.add_argument("--name", type=str, default="horizon_v1")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.resume:
        model = YOLO(args.resume)
    else:
        model = YOLO("yolov8n.pt")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        amp=False,

        # Safety-critical: maximize recall
        conf=0.001,
        iou=0.6,

        # Augmentation for road scenes
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.1,

        # Training
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        weight_decay=0.0005,
        patience=20,
        save=True,
        save_period=5,
        val=True,
        plots=True,
        verbose=True,
        workers=args.workers,
    )

    print(f"\nTraining complete. Best model: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
