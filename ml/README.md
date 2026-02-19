# Machine Learning Pipeline - General Object Detection

ML architecture for Horizon-HUD General Object Detection module.

## Overview

This module implements the training, evaluation, and deployment pipeline for object detection models targeting:
- Vehicle
- Pedestrian  
- Cyclist
- Road obstacle

## Architecture

```
ml/
├── config/              # Configuration files
│   ├── model_config.yaml      # Model and training configuration
│   └── experiment_config.yaml # Experiment-specific settings
├── data/                # Dataset loaders
│   └── bdd100k_loader.py      # BDD100K dataset loader
├── training/            # Training pipeline
│   ├── train.py              # Main training script
│   ├── model_builder.py      # Model architecture builders
│   └── losses.py             # Loss functions
├── evaluation/          # Evaluation framework
│   ├── evaluate.py          # Model evaluation
│   └── benchmark_pi.py       # Raspberry Pi 5 benchmarking
├── conversion/          # TFLite conversion
│   └── convert_to_tflite.py # Keras to TFLite converter
└── utils/              # Utilities
    └── class_mapping.py      # Class mapping and label management
```

## Quick Start

### 1. Prepare Dataset

Ensure BDD100K dataset is downloaded and structured:
```
data/bdd100k/
├── images/
│   └── 100k/
│       ├── train/
│       ├── val/
│       └── test/
└── labels/
    ├── det_train.json
    ├── det_val.json
    └── det_test.json
```

### 2. Create Experiment Config

Copy and modify `config/experiment_config.yaml`:
```yaml
experiment:
  name: "my_experiment"
  # ... configure experiment settings
```

### 3. Train Model

```bash
python ml/training/train.py \
  --config ml/config/experiment_config.yaml \
  --dataset-root data/bdd100k \
  [--resume path/to/checkpoint.h5]
```

### 4. Evaluate Model

```bash
python ml/evaluation/evaluate.py \
  --model models/experiments/my_experiment/best_model.h5 \
  --dataset-root data/bdd100k \
  --split test \
  --config ml/config/model_config.yaml \
  --output evaluation_results.json
```

### 5. Convert to TFLite

```bash
python ml/conversion/convert_to_tflite.py \
  --model models/experiments/my_experiment/best_model.h5 \
  --output models/detector.tflite \
  --quantization int8 \
  --dataset-root data/bdd100k \
  --config ml/config/model_config.yaml
```

### 6. Benchmark on Raspberry Pi 5

```bash
python ml/evaluation/benchmark_pi.py \
  --model models/detector.tflite \
  --input-size 320 320 \
  --num-runs 1000 \
  --num-threads 4
```

## Model Selection

### Baseline: SSD MobileNetV2
- Input: 320x320 or 384x384
- Quantization: INT8 (preferred) or FP16
- Good speed/accuracy tradeoff

### Alternatives
- EfficientDet-Lite0: Better accuracy, higher latency
- YOLO-nano: Strong accuracy, may have latency variance

## Acceptance Criteria

Model must satisfy on Raspberry Pi 5:
- Mean latency ≤ 50ms
- P95 latency ≤ 75ms
- Stable FPS (low jitter)
- Acceptable recall for vehicle and pedestrian classes

## Output Format

Detection outputs:
- Bounding boxes (normalized coordinates)
- Class IDs (0: vehicle, 1: pedestrian, 2: cyclist, 3: road_obstacle)
- Confidence scores

## Class Stability

**IMPORTANT**: Class IDs are fixed and must not change:
- 0: vehicle
- 1: pedestrian
- 2: cyclist
- 3: road_obstacle

Changing these IDs will break downstream modules (Tracking, Risk Scoring).
