"""
Export YOLOv8 model to TFLite for Raspberry Pi 5 deployment.
Supports INT8 quantization for maximum inference speed.
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 to TFLite")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--int8", action="store_true", default=True)
    parser.add_argument("--half", action="store_true", default=False)
    args = parser.parse_args()

    model = YOLO(args.model)

    model.export(
        format="tflite",
        imgsz=args.imgsz,
        int8=args.int8,
        half=args.half,
    )

    print(f"\nExported TFLite model (imgsz={args.imgsz}, int8={args.int8})")
    print("Deploy this .tflite file to Raspberry Pi 5")


if __name__ == "__main__":
    main()
