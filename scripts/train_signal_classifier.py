"""Train a MobileNetV3-Small signal classifier and export to TFLite INT8.

Input
-----
    data/tld_signal/
        images/          96×96 PNG crops from prepare_tld_dataset.py
        manifest.csv     columns: filename, brake, left, right, split

Output
------
    models/signal_int8.tflite   TFLite INT8 model ready for SignalClassifier

Usage
-----
    python scripts/train_signal_classifier.py \\
        --data_dir data/tld_signal \\
        --out models/signal_int8.tflite \\
        --epochs 30

Targets
-------
    brake precision >= 0.80
    left / right F1 >= 0.75
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import torchvision.models as tv_models
    import torchvision.transforms as T
    from PIL import Image
except ImportError:
    print("ERROR: PyTorch / torchvision not installed.")
    print("  pip install torch torchvision pillow")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SignalDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        augment: bool = False,
    ) -> None:
        manifest = data_dir / "manifest.csv"
        img_dir = data_dir / "images"

        self._img_dir = img_dir
        self._records: list[dict] = []

        with open(manifest) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self._records.append({
                        "filename": row["filename"],
                        "label": torch.tensor(
                            [int(row["brake"]), int(row["left"]), int(row["right"])],
                            dtype=torch.float32,
                        ),
                    })

        base_tf: list = [
            T.Resize((96, 96)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        aug_tf: list = [
            T.Resize((96, 96)),
            T.ColorJitter(brightness=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self._transform = T.Compose(aug_tf if augment else base_tf)
        self._augment = augment

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self._records[idx]
        img = Image.open(self._img_dir / rec["filename"]).convert("RGB")
        label = rec["label"].clone()

        if self._augment and torch.rand(1).item() < 0.5:
            # Horizontal flip: swap left <-> right channels
            img = T.functional.hflip(img)
            label[1], label[2] = label[2].clone(), label[1].clone()

        return self._transform(img), label


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_model(num_classes: int = 3) -> nn.Module:
    model = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
    # Replace the classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / max(len(loader.dataset), 1)  # type: ignore[arg-type]


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    preds_arr = np.concatenate(all_preds, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)

    metrics: dict[str, float] = {}
    for i, name in enumerate(["brake", "left", "right"]):
        tp = float(((preds_arr[:, i] == 1) & (labels_arr[:, i] == 1)).sum())
        fp = float(((preds_arr[:, i] == 1) & (labels_arr[:, i] == 0)).sum())
        fn = float(((preds_arr[:, i] == 0) & (labels_arr[:, i] == 1)).sum())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        metrics[f"{name}_precision"] = precision
        metrics[f"{name}_recall"] = recall
        metrics[f"{name}_f1"] = f1

    val_loss = total_loss / max(len(loader.dataset), 1)  # type: ignore[arg-type]
    return val_loss, metrics


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _export_onnx(model: nn.Module, onnx_path: Path, device: torch.device) -> None:
    # ONNX export must run on CPU (MPS/CUDA not supported by the exporter).
    # Opset 18 is required by the dynamo-based exporter in PyTorch >= 2.1.
    cpu_model = model.cpu().eval()
    dummy = torch.zeros(1, 3, 96, 96)
    with torch.no_grad():
        torch.onnx.export(
            cpu_model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
        )
    model.to(device)
    print(f"ONNX model saved: {onnx_path}")


def _quantize_onnx(onnx_path: Path, train_loader: DataLoader) -> Path:
    """Post-training INT8 quantization via onnxruntime.quantization.

    Tries static (best accuracy) then dynamic (simpler, still 2-3× faster on ARM).
    """
    try:
        import onnx
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_dynamic,
            quantize_static,
        )
    except ImportError:
        print("WARNING: onnxruntime.quantization not available — skipping INT8 export.")
        print("  pip install onnxruntime")
        return onnx_path

    out_path = onnx_path.with_stem(onnx_path.stem + "_quant")
    preprocessed = onnx_path.with_stem(onnx_path.stem + "_pre")

    # Run ONNX shape inference first — fixes shape mismatches that break the quantizer
    try:
        model = onnx.load(str(onnx_path))
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save(model, str(preprocessed))
    except Exception as exc:
        print(f"  Shape inference failed ({exc}), using original model.")
        preprocessed = onnx_path

    # Calibration reader: yield exactly one sample at a time.
    # The model was exported with fixed batch=1 so we cannot feed larger batches.
    class _CalibReader(CalibrationDataReader):
        def __init__(self, loader: DataLoader, n: int = 200) -> None:
            self._samples: list[np.ndarray] = []
            for imgs, _ in loader:
                for img in imgs:
                    # img shape: (C, H, W) — add batch dim → (1, C, H, W)
                    self._samples.append(img.unsqueeze(0).numpy())
                    if len(self._samples) >= n:
                        break
                if len(self._samples) >= n:
                    break
            self._idx = 0

        def get_next(self):
            if self._idx >= len(self._samples):
                return None
            data = {"input": self._samples[self._idx]}
            self._idx += 1
            return data

    try:
        quantize_static(
            str(preprocessed),
            str(out_path),
            calibration_data_reader=_CalibReader(train_loader),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )
        print(f"INT8 ONNX (static) saved: {out_path}  "
              f"({out_path.stat().st_size / 1024:.1f} KB)")
    except Exception as exc:
        print(f"  Static quantization failed ({exc}), trying dynamic …")
        try:
            quantize_dynamic(
                str(preprocessed), str(out_path),
                weight_type=QuantType.QInt8,
            )
            print(f"INT8 ONNX (dynamic) saved: {out_path}  "
                  f"({out_path.stat().st_size / 1024:.1f} KB)")
        except Exception as exc2:
            print(f"  Dynamic quantization also failed ({exc2}) — using float32 model.")
            if preprocessed.exists() and preprocessed != onnx_path:
                preprocessed.unlink()
            return onnx_path

    if preprocessed.exists() and preprocessed != onnx_path:
        preprocessed.unlink()

    return out_path


def _convert_to_tflite(
    onnx_path: Path,
    tflite_path: Path,
    representative_loader: DataLoader,
    device: torch.device,
) -> None:
    """Convert ONNX → TF SavedModel → TFLite INT8 via onnx2tf."""
    try:
        import onnx2tf
    except ImportError:
        print("WARNING: onnx2tf not installed — skipping TFLite conversion.")
        print("  pip install onnx2tf")
        print(f"  Then manually convert {onnx_path} to TFLite INT8.")
        return

    tf_saved_dir = onnx_path.parent / "tf_saved_model"
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(tf_saved_dir),
        non_verbose=True,
    )
    print(f"TF SavedModel written: {tf_saved_dir}")

    try:
        import tensorflow as tf
    except ImportError:
        print("WARNING: TensorFlow not installed — cannot quantise to INT8.")
        return

    def representative_dataset():
        count = 0
        for imgs, _ in representative_loader:
            if count >= 500:
                break
            for img in imgs:
                yield [img.unsqueeze(0).cpu().numpy()]
                count += 1
                if count >= 500:
                    break

    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_saved_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)
    print(f"TFLite INT8 model saved: {tflite_path}  ({len(tflite_model) / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train signal classifier")
    parser.add_argument("--data_dir", default="data/tld_signal")
    parser.add_argument("--out", default="models/signal_int8.onnx")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "manifest.csv").exists():
        print(f"ERROR: manifest.csv not found in {data_dir}")
        print("Run prepare_tld_dataset.py first.")
        sys.exit(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── datasets ──────────────────────────────────────────────────────────────
    train_ds = SignalDataset(data_dir, "train", augment=True)
    val_ds = SignalDataset(data_dir, "val", augment=False)
    test_ds = SignalDataset(data_dir, "test", augment=False)

    print(f"Dataset sizes: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    pin = device.type == "cuda"   # pin_memory only works on CUDA
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers)

    # ── model ─────────────────────────────────────────────────────────────────
    model = _build_model(num_classes=3).to(device)

    # pos_weight corrects for class imbalance (left/right are rare).
    # For each channel: pos_weight = n_negative / n_positive.
    labels_mat = torch.stack([train_ds[i][1] for i in range(len(train_ds))])
    n_pos = labels_mat.sum(0).clamp(min=1)
    n_neg = len(train_ds) - n_pos
    pos_weight = (n_neg / n_pos).clamp(max=10.0).to(device)
    print(f"pos_weight: brake={pos_weight[0]:.1f}  left={pos_weight[1]:.1f}  right={pos_weight[2]:.1f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW([
        {"params": model.classifier.parameters(), "lr": args.lr_head},
        {"params": model.features.parameters(), "lr": args.lr_backbone},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── training ──────────────────────────────────────────────────────────────
    # Score = brake_precision + mean(left_F1, right_F1); higher is better.
    # This matches the thesis targets directly instead of optimising val loss,
    # which diverges from the metrics once the model starts to overfit.
    best_score = -1.0
    best_state: dict | None = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = _eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        score = (
            val_metrics["brake_precision"]
            + (val_metrics["left_f1"] + val_metrics["right_f1"]) / 2
        )

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"brake_P={val_metrics['brake_precision']:.3f}  "
            f"left_F1={val_metrics['left_f1']:.3f}  "
            f"right_F1={val_metrics['right_f1']:.3f}  "
            f"score={score:.3f}  ({elapsed:.1f}s)"
        )

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping after {epoch} epochs (no improvement for {args.patience}).")
                break

    # ── restore best weights ──────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── test evaluation ───────────────────────────────────────────────────────
    _, test_metrics = _eval_epoch(model, test_loader, criterion, device)
    print("\n── Test-set evaluation ──────────────────────────────────────────")
    for channel in ("brake", "left", "right"):
        print(
            f"  {channel:5s}  "
            f"precision={test_metrics[f'{channel}_precision']:.3f}  "
            f"recall={test_metrics[f'{channel}_recall']:.3f}  "
            f"F1={test_metrics[f'{channel}_f1']:.3f}"
        )
    # Target checks
    if test_metrics["brake_precision"] < 0.80:
        print("WARNING: brake precision below target (0.80)")
    if test_metrics["left_f1"] < 0.75 or test_metrics["right_f1"] < 0.75:
        print("WARNING: indicator F1 below target (0.75)")

    # ── export ────────────────────────────────────────────────────────────────
    onnx_path = Path(args.out).with_suffix(".onnx")
    _export_onnx(model, onnx_path, device)

    # INT8 quantization for Pi 4 deployment
    int8_path = _quantize_onnx(onnx_path, train_loader)
    if int8_path != onnx_path:
        print(f"\nFor Pi 4: use {int8_path} in config.yaml (3-4× faster on ARM NEON)")
        print(f"For Mac:  use {onnx_path} (float32, CoreML backend)")

    # TFLite conversion (optional, Linux/Pi works better than macOS for this)
    tflite_path = Path(args.out).with_suffix(".tflite")
    _convert_to_tflite(onnx_path, tflite_path, train_loader, device)


if __name__ == "__main__":
    main()
