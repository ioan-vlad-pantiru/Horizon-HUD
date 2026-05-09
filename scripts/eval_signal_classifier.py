"""Evaluate a signal classifier ONNX model on the TLD test split.

Uses the same preprocessing as SignalClassifier._run_ort() (divide by 255,
no ImageNet mean/std) so the reported metrics reflect actual deployment
accuracy rather than training-time accuracy.

Usage
-----
    python scripts/eval_signal_classifier.py \
        --model  models/signal_int8_quant.onnx \
        --data_dir data/tld_full

Optional flags
--------------
    --imagenet_norm   Apply ImageNet mean/std normalisation instead of /255
                      (use this to evaluate the float32 model as trained)
    --threshold 0.5   Classification threshold (default 0.5)
    --batch 256       Batch size for DataLoader
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise SystemExit("onnxruntime not installed — pip install onnxruntime")

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow not installed — pip install pillow")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_path: Path, imagenet_norm: bool, size: int = 96) -> np.ndarray:
    img = Image.open(img_path).convert("RGB").resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0          # HWC, [0,1]
    if imagenet_norm:
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = arr.transpose(2, 0, 1)                            # NCHW
    return arr


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f = 2 * p * r / (p + r + 1e-8)
    return p, r, f


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         default="models/signal_int8_quant.onnx")
    parser.add_argument("--data_dir",      default="data/tld_full")
    parser.add_argument("--threshold",     type=float, default=0.5)
    parser.add_argument("--imagenet_norm", action="store_true",
                        help="Use ImageNet mean/std instead of /255 only")
    parser.add_argument("--batch",         type=int, default=256)
    args = parser.parse_args()

    model_path = Path(args.model)
    data_dir   = Path(args.data_dir)
    manifest   = data_dir / "manifest.csv"
    img_dir    = data_dir / "images"

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not manifest.exists():
        raise SystemExit(f"manifest.csv not found in {data_dir}")

    # Load test records
    records: list[dict] = []
    with open(manifest) as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                records.append({
                    "path":  img_dir / row["filename"],
                    "label": np.array(
                        [int(row["brake"]), int(row["left"]), int(row["right"])],
                        dtype=np.float32,
                    ),
                })

    if not records:
        raise SystemExit("No test records found in manifest.csv")

    print(f"Model  : {model_path}")
    print(f"Data   : {data_dir}  ({len(records)} test samples)")
    norm_str = "ImageNet mean/std" if args.imagenet_norm else "/255 only (deployment)"
    print(f"Norm   : {norm_str}")
    print(f"Thresh : {args.threshold}\n")

    # Load ONNX model
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp_name  = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape
    in_h = int(inp_shape[2]) if len(inp_shape) >= 4 else 96
    in_w = int(inp_shape[3]) if len(inp_shape) >= 4 else 96

    # Accumulate per-class TP/FP/FN
    channels  = ["brake", "left", "right"]
    tp = {c: 0 for c in channels}
    fp = {c: 0 for c in channels}
    fn = {c: 0 for c in channels}

    # Process one sample at a time (quantized model has fixed batch size = 1)
    total = len(records)
    for idx, rec in enumerate(records):
        img   = preprocess(rec["path"], args.imagenet_norm, size=in_h)
        img   = img[np.newaxis, ...]                              # (1, C, H, W)
        label = rec["label"]                                      # (3,)

        logits = sess.run(None, {inp_name: img})[0][0]            # (3,)
        # Apply sigmoid if model outputs raw logits
        if logits.min() < 0.0 or logits.max() > 1.0:
            logits = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))

        preds = (logits >= args.threshold).astype(np.float32)     # (3,)

        for i, c in enumerate(channels):
            if preds[i] == 1 and label[i] == 1: tp[c] += 1
            if preds[i] == 1 and label[i] == 0: fp[c] += 1
            if preds[i] == 0 and label[i] == 1: fn[c] += 1

        if (idx + 1) % 500 == 0:
            print(f"\r  {idx + 1}/{total}", end="", flush=True)

    print(f"\r  {total}/{total} done\n")

    # Report
    print(f"{'Class':<10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("-" * 42)
    for c in channels:
        p, r, f = _prf(tp[c], fp[c], fn[c])
        flag = ""
        if c == "brake"  and p < 0.80: flag = "  ← below target (0.80)"
        if c in ("left", "right") and f < 0.75: flag = "  ← below target (0.75)"
        print(f"{c:<10}  {p:>10.3f}  {r:>8.3f}  {f:>8.3f}{flag}")


if __name__ == "__main__":
    main()
