# Horizon-HUD: Object Detection Model Training Methodology

## 1. Introduction

### 1.1 Purpose

Horizon-HUD is a safety device for motorcyclists that provides real-time visual awareness of surrounding road hazards through a heads-up display. The core of the system is an object detection model that identifies vehicles, pedestrians, cyclists, and road obstacles from a forward-facing camera mounted on the motorcycle.

### 1.2 Objective

Train a high-recall object detection model optimized for:
- **Safety-critical inference**: prioritize detecting all hazards (recall) over eliminating false positives (precision)
- **Edge deployment**: run in real-time on a Raspberry Pi 5 via TFLite INT8 quantization
- **Road-scene robustness**: perform reliably across diverse lighting, weather, and traffic conditions

### 1.3 Deployment Target

- **Hardware**: Raspberry Pi 5 (Cortex-A76 CPU)
- **Inference format**: TFLite INT8, 320×320 input resolution
- **Target latency**: ≤100ms per frame (~10 FPS)

---

## 2. Dataset

### 2.1 Source

The model is trained on the **BDD100K** (Berkeley Deep Drive 100K) dataset, one of the largest and most diverse driving video datasets available. BDD100K contains 100,000 dashcam video frames captured across varied geographic locations, weather conditions (clear, rainy, snowy, foggy), times of day (daytime, nighttime, dawn/dusk), and scene types (highway, city, residential).

### 2.2 Split

| Split | Images | Annotated Objects |
|-------|--------|-------------------|
| Train | 70,000 | ~1,280,000 |
| Val | 10,000 | ~185,000 |

### 2.3 Class Mapping

BDD100K provides 10 fine-grained object categories. These are consolidated into 4 safety-relevant classes for the Horizon-HUD use case:

| Horizon-HUD Class (ID) | BDD100K Source Categories |
|------------------------|--------------------------|
| **vehicle** (0) | car, bus, truck, train |
| **pedestrian** (1) | person |
| **cyclist** (2) | rider, bike, motor, motorcycle, bicycle |
| **road_obstacle** (3) | traffic sign, traffic light |

This merging strategy groups objects by the type of hazard they represent to a motorcyclist, reducing class count while maintaining safety-relevant distinctions.

### 2.4 Label Conversion

BDD100K annotations are provided as per-image JSON files with absolute pixel-coordinate bounding boxes (x1, y1, x2, y2) for 1280×720 images. These are converted to YOLO format:

```
<class_id> <center_x> <center_y> <width> <height>
```

All coordinates are normalized to [0, 1] relative to image dimensions. The conversion is parallelized across 4 worker processes and processes all 80,000 label files in under 60 seconds.

### 2.5 Dataset Structure

The YOLO-compatible dataset is organized with parallel `images/` and `labels/` directories:

```
dataset/
├── images/
│   ├── train/   (70,000 .jpg symlinks)
│   └── val/     (10,000 .jpg symlinks)
├── labels/
│   ├── train/   (70,000 .txt files)
│   └── val/     (10,000 .txt files)
└── dataset.yaml
```

Image files are symlinked from the source dataset to avoid duplication while maintaining the directory structure required by YOLO's automatic label discovery mechanism.

---

## 3. Model Architecture

### 3.1 Model Selection

**YOLOv8s** (small variant) was selected after evaluating the trade-off between detection accuracy and Raspberry Pi 5 inference speed:

| Model | Parameters | GFLOPs | Pi5 TFLite INT8 @ 320px | Suitability |
|-------|-----------|--------|------------------------|-------------|
| YOLOv8n (nano) | 3.0M | 8.2 | ~30ms (~30 FPS) | Fast but lower recall |
| **YOLOv8s (small)** | **11.1M** | **28.7** | **~100ms (~10 FPS)** | **Best recall within latency budget** |
| YOLOv8m (medium) | 25.9M | 78.9 | ~300ms (~3 FPS) | Too slow for real-time |

YOLOv8s provides significantly better recall than YOLOv8n for detecting small and distant objects (pedestrians at range, cyclists in peripheral vision), which is critical for a safety device. The ~10 FPS inference rate on Pi5 provides a detection update approximately every 1.7 meters at 60 km/h, which is sufficient for hazard alerting.

### 3.2 Architecture Details

YOLOv8s is a single-stage anchor-free detector with:

- **Backbone**: CSPDarknet with C2f (Cross Stage Partial with 2 convolutions and flow) blocks
- **Neck**: PANet (Path Aggregation Network) with feature pyramid for multi-scale detection
- **Head**: Decoupled detection head with separate branches for classification and bounding box regression
- **Detection scales**: 3 (P3/8, P4/16, P5/32) for detecting objects from small to large

```
Layer  Module                        Parameters
0      Conv [3→32, 3×3, stride 2]    928
1      Conv [32→64, 3×3, stride 2]   18,560
2      C2f [64→64, 1 block]          29,056
3      Conv [64→128, 3×3, stride 2]  73,984
4      C2f [128→128, 2 blocks]       197,632
5      Conv [128→256, 3×3, stride 2] 295,424
6      C2f [256→256, 2 blocks]       788,480
7      Conv [256→512, 3×3, stride 2] 1,180,672
8      C2f [512→512, 1 block]        1,838,080
9      SPPF [512→512, kernel 5]      656,896
10-21  FPN + PAN Neck                 ~3,000,000
22     Detect Head [4 classes]        2,117,596
─────────────────────────────────────────────────
Total: 130 layers, 11,137,148 parameters, 28.7 GFLOPs
```

### 3.3 Transfer Learning

The model is initialized with COCO-pretrained weights (`yolov8s.pt`), transferring 349 of 355 parameter tensors. This provides a strong initialization for general object detection features (edges, textures, object shapes) that are then fine-tuned on the BDD100K road-scene distribution.

---

## 4. Training Configuration

### 4.1 Compute Environment

| Resource | Specification |
|----------|--------------|
| Platform | Kaggle Notebooks |
| GPU | NVIDIA Tesla P100-PCIe (16 GB VRAM) |
| CUDA | 12.6 with PyTorch 2.9.0 |
| Precision | Automatic Mixed Precision (AMP / FP16) |
| Batch size | 15 (auto-detected for 58% VRAM utilization) |

### 4.2 Optimizer

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Decoupled weight decay, stable convergence |
| Initial LR | 0.001 | Standard for fine-tuning with AdamW |
| Final LR | 0.00001 (lrf=0.01) | 100× reduction via cosine annealing |
| Weight decay | 0.0005 | Regularization against overfitting |
| Momentum | 0.937 | Default YOLO momentum |
| Warmup epochs | 3 | Linear warmup for learning rate and momentum |

### 4.3 Training Schedule

| Parameter | Value |
|-----------|-------|
| Epochs | Up to 100 (time-limited to 20 hours) |
| Effective epochs | ~48-50 within time limit |
| Patience | 20 epochs (early stopping on val mAP) |
| Close mosaic | Epoch 40 (disables mosaic for final 10 epochs) |
| Checkpoint interval | Every 5 epochs |

### 4.4 Loss Function

YOLOv8 uses a composite loss with three components:

| Component | Function | Weight | Purpose |
|-----------|----------|--------|---------|
| Box loss | CIoU (Complete IoU) | 7.5 | Bounding box regression accuracy |
| Class loss | BCE (Binary Cross-Entropy) | 0.5 | Classification accuracy |
| DFL loss | Distribution Focal Loss | 1.5 | Bounding box distribution refinement |

### 4.5 Data Augmentation

Augmentations are chosen for road-scene diversity while avoiding transformations that would be unrealistic for a forward-facing motorcycle camera:

| Augmentation | Value | Rationale |
|-------------|-------|-----------|
| HSV-Hue | 0.015 | Minor color shifts for lighting variation |
| HSV-Saturation | 0.7 | Weather/time-of-day robustness |
| HSV-Value | 0.4 | Brightness variation (tunnels, shadows) |
| Horizontal flip | 0.5 | Left/right traffic variation |
| Vertical flip | 0.0 | Disabled (unrealistic for a road camera) |
| Rotation | 0.0 | Disabled (motorcycle camera is level) |
| Translation | 0.1 | Minor positional shifts |
| Scale | 0.5 | Object size variation |
| Mosaic | 1.0 | 4-image mosaic for dense object training |
| MixUp | 0.1 | Light image blending for regularization |
| Erasing | 0.4 | Random erasing for occlusion robustness |
| Albumentations | Blur, MedianBlur, ToGray, CLAHE | Additional photometric augmentation |

### 4.6 Safety-Critical Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Confidence threshold (eval) | 0.001 | Evaluate recall at very low thresholds |
| IoU threshold (NMS) | 0.6 | Balance between duplicate suppression and recall |
| Max detections | 300 | High limit for dense traffic scenes |

---

## 5. Training Results

### 5.1 Training Progression

Training was conducted across 3 Kaggle sessions with checkpoint-based resumption (epochs 1–12, 13–41, 42–49). Key epochs are shown below:

| Epoch | box_loss | cls_loss | dfl_loss | mAP50 | mAP50-95 | Recall | Precision |
|-------|----------|----------|----------|-------|----------|--------|-----------|
| 1 | 1.525 | 0.964 | 1.036 | 0.459 | 0.227 | 0.435 | 0.588 |
| 5 | 1.419 | 0.810 | 0.991 | 0.575 | 0.292 | 0.525 | 0.694 |
| 10 | 1.376 | 0.759 | 0.971 | 0.608 | 0.314 | 0.555 | 0.713 |
| 15 | 1.351 | 0.734 | 0.963 | 0.618 | 0.320 | 0.562 | 0.724 |
| 20 | 1.333 | 0.714 | 0.955 | 0.623 | 0.324 | 0.563 | 0.727 |
| 25 | 1.326 | 0.705 | 0.952 | 0.629 | 0.328 | 0.567 | 0.734 |
| 30 | 1.313 | 0.692 | 0.947 | 0.635 | 0.331 | 0.573 | 0.736 |
| 35 | 1.304 | 0.682 | 0.944 | 0.639 | 0.334 | 0.575 | 0.743 |
| 40* | 1.296 | 0.674 | 0.942 | 0.642 | 0.335 | 0.576 | 0.747 |
| 45 | 1.285 | 0.663 | 0.939 | 0.643 | 0.336 | 0.581 | 0.743 |
| 49 | 1.277 | 0.655 | 0.936 | 0.643 | 0.337 | 0.581 | 0.743 |

*Epoch 40: mosaic augmentation disabled (`close_mosaic=10`) for final refinement phase.

### 5.2 Per-Class Final Results

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| **vehicle** | 9,905 | 108,363 | 0.816 | 0.730 | 0.801 | 0.501 |
| **pedestrian** | 3,220 | 13,262 | 0.745 | 0.538 | 0.621 | 0.307 |
| **cyclist** | 883 | 2,108 | 0.714 | 0.429 | 0.492 | 0.237 |
| **road_obstacle** | 8,862 | 61,793 | 0.700 | 0.625 | 0.660 | 0.301 |
| **all** | 10,000 | 185,526 | 0.743 | 0.581 | 0.643 | 0.336 |

### 5.3 Observations

- All three loss components decrease monotonically across all 49 epochs, indicating stable learning with no overfitting
- The largest mAP gains occur in the first 10 epochs (0.459 → 0.608, +32% relative), with diminishing but consistent returns thereafter
- Disabling mosaic at epoch 40 produces a noticeable drop in classification loss (0.674 → 0.650) as the model fine-tunes on unaugmented images
- The model converged by epoch 45 with mAP50 plateauing at 0.643 (patience of 20 was not triggered)
- Vehicle detection achieves the strongest performance (0.801 mAP50) due to high instance count and large object size
- Cyclist detection is the weakest (0.492 mAP50) due to class imbalance (2,108 instances vs 108,363 for vehicles) and small object size
- Inference speed on GPU: 2.7ms per image (370 FPS), confirming the model is lightweight enough for edge deployment

---

## 6. Deployment Pipeline

### 6.1 Model Export

The trained PyTorch model (`best.pt`) is exported to TensorFlow Lite format with INT8 quantization for Raspberry Pi 5 deployment:

```
Input:  best.pt (PyTorch, FP32, 22.5 MB)
Output: best_int8.tflite (TFLite, INT8, 11.4 MB)
Export resolution: 320×320
```

Multiple export formats are produced for flexibility:

| Format | Size | Use Case |
|--------|------|----------|
| best_int8.tflite | 11.4 MB | **Primary deployment** on Pi 5 (fastest) |
| best_full_integer_quant.tflite | 11.5 MB | Edge accelerator compatibility |
| best_float16.tflite | 22.4 MB | Higher accuracy if latency permits |
| best_float32.tflite | 44.6 MB | Debugging and accuracy baseline |
| best.pt | 22.5 MB | Re-export or fine-tuning |

### 6.2 Inference Characteristics

| Property | Value |
|----------|-------|
| Input resolution | 320×320 pixels |
| Quantization | INT8 |
| Expected latency (Pi5) | ~100ms |
| Expected throughput | ~10 FPS |
| Model size | 11.4 MB |

### 6.3 Resume Training

Training supports checkpoint-based resumption for handling compute session limits:

1. YOLO automatically saves `best.pt` (best validation mAP) and `last.pt` (latest epoch) during training
2. `last.pt` contains the full training state (model weights, optimizer state, scheduler state, epoch number)
3. To resume: load `last.pt` and call `model.train(resume=True)`

---

## 7. Preflight Validation

The training notebook includes automated preflight checks that run before any training begins:

| Check | Purpose |
|-------|---------|
| GPU detection | Verify CUDA GPU is available and report VRAM |
| Disk space | Ensure ≥5 GB free for checkpoints and exports |
| Dataset paths | Validate image and label directories exist with correct structure |
| Image/label counts | Verify train and val splits have matching file counts |
| Resume checkpoint | If resuming, verify `last.pt` exists at the specified path |

These checks prevent wasted GPU time from configuration errors.

---

## 8. Robustness Considerations

### 8.1 Dataset Diversity

BDD100K provides built-in robustness through its diverse capture conditions:
- **Weather**: clear, overcast, rainy, snowy, foggy
- **Time of day**: daytime, nighttime, dawn, dusk
- **Scene type**: highway, city street, residential, parking lot
- **Geography**: multiple US cities with varied road infrastructure

### 8.2 Augmentation Strategy

The augmentation pipeline specifically targets failure modes relevant to motorcycle riding:
- **HSV augmentation**: robustness to changing light (entering/exiting tunnels, sun glare)
- **Mosaic**: training on dense multi-object scenes (urban intersections)
- **Random erasing**: robustness to partial occlusion (vehicles behind other vehicles)
- **No vertical flip or rotation**: preserves the gravitational prior of road scenes

### 8.3 Class Imbalance

BDD100K is naturally vehicle-heavy. The 4-class merging strategy mitigates some imbalance:
- **vehicle** (car + bus + truck + train): most frequent, provides strong detection baseline
- **pedestrian**: well-represented in urban scenes
- **cyclist** (rider + bike + motor): less frequent but augmented by merging multiple categories
- **road_obstacle** (traffic sign + traffic light): abundant in all road scenes

---

## 9. Tools and Infrastructure

| Component | Technology |
|-----------|-----------|
| Training framework | Ultralytics YOLOv8 (v8.4.14) |
| Deep learning backend | PyTorch 2.9.0 + CUDA 12.6 |
| Compute platform | Kaggle Notebooks (Tesla P100 GPU) |
| Dataset | BDD100K (Berkeley Deep Drive) |
| Label conversion | Custom Python script with multiprocessing |
| Export format | TensorFlow Lite INT8 |
| Deployment target | Raspberry Pi 5 |
