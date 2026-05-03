# Horizon-HUD

Real-time heads-up display system for motorcycles using camera, IMU, and TensorFlow Lite for object detection, tracking, ego-motion compensation, and risk assessment.

## Setup

```bash
git clone https://github.com/yourusername/Horizon-HUD.git
cd Horizon-HUD
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Minimum dependencies for the perception pipeline (no full TF needed):

```bash
pip install numpy opencv-python scipy pyyaml
```

## Running

### Webcam demo (works without TFLite – uses DummyDetector)

```bash
python -m src.main --source webcam
```

### Video file

```bash
python -m src.main --source path/to/video.mp4
```

### Custom config

```bash
python -m src.main --source webcam --config docs/config.yaml
```

### JSONL telemetry output

```bash
python -m src.main --source webcam --jsonl out.jsonl
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `m` | Toggle ego-motion compensation on/off |
| `i` | Cycle IMU simulator scenario |
| `d` | Toggle dummy / TFLite detector |

## Tests

```bash
python -m pytest src/tests/ -v
```

Key test files:

| File | What it covers |
|------|----------------|
| `src/tests/test_risk_engine_v1.py` | `RiskEngineV1` (deployed engine) — monotonicity, hysteresis, persistence gate, corridor, score bounds, lateral risk |
| `src/tests/test_risk_v1.py` | Feature math, component scorers, corridor geometry, hysteresis/persistence units |
| `src/tests/test_motion_math.py` | Camera matrix, rotation math, motion compensator |

## Evaluation

### Collect ground-truth distances

Place known objects at measured distances from the camera (tape measure or LiDAR). Record a short video clip and annotate a CSV:

```
frame,bbox_x1,bbox_y1,bbox_x2,bbox_y2,true_distance_m,label
1,200,100,300,380,8.5,pedestrian
2,180,120,310,410,6.2,pedestrian
```

### Run distance validation

```bash
python scripts/validate_distance.py --csv ground_truth.csv
# Sweep FOV to minimise RMSE:
python scripts/validate_distance.py --csv ground_truth.csv --fov 65
# Save scatter plot:
python scripts/validate_distance.py --csv ground_truth.csv --plot
```

### Run the ablation study

First produce a telemetry log:

```bash
python -m src.main --source video.mp4 --jsonl run.jsonl
```

Then replay under ablation conditions:

```bash
python scripts/ablation_study.py --log run.jsonl
```

Prints a markdown table comparing CRITICAL/HIGH alert counts and latency across five conditions: baseline, no ego-motion, no corridor, no persistence gate, no hysteresis.

### Run the FPS benchmark

```bash
python scripts/benchmark_fps.py --model models/best_int8.tflite
python scripts/benchmark_fps.py --model models/best_float16.tflite
```

Prints per-component mean/std latency and % of 30-fps frame budget. Use with both quantised and float models to report quantisation impact.

### Export per-run summary CSV

```bash
python scripts/summarise_log.py --log run.jsonl --out summary.csv
```

### Evaluate against ground truth

Annotate a JSONL file with hazardous track IDs per frame:

```json
{"frame": 42, "hazard_ids": [3, 7]}
{"frame": 43, "hazard_ids": [3]}
```

Then run with `--eval`:

```bash
python -m src.main --source video.mp4 --gt gt.jsonl --eval
```

At shutdown, prints TPR, FPR, and mean lead time.

## Architecture

```
src/
  core/types.py              Core dataclasses (Detection, Track, IMUReading, Orientation)
  detection/
    detection_tflite.py      TFLite detector with DummyDetector fallback
    tracking_sort.py         SORT tracker (Kalman + IoU + Hungarian)
  perception/
    imu_sim.py               Simulated IMU with 4 driving scenarios
    hardware_imu.py          MPU-9250 reader for Raspberry Pi
    iphone_imu.py            iPhone IMU via WebSocket bridge
    orientation.py           Complementary filter (gyro + accel + mag)
    motion_comp.py           Ego-motion compensation via orientation deltas
    corridor.py              Trapezoid corridor geometry and membership scoring
  risk/
    risk_features.py         Physics proxies: distance, TTC, closing speed, lateral risk, erratic score
    risk_types.py            RiskConfig, CorridorConfig, RiskAssessmentV1 dataclasses
    risk_engine.py           RiskEngineV1 — weighted scoring, EMA, hysteresis, persistence gate
    risk.py                  Legacy RiskEngine V0 (retained for reference)
  tests/
    test_risk_engine_v1.py   RiskEngineV1 unit and integration tests (deployed engine)
    test_risk_v1.py          Feature math and component scorer tests
    test_motion_math.py      Camera matrix, rotation math, motion compensator tests
  main.py                    Pipeline runner, visualization, JSONL logging, --eval mode
scripts/
  validate_distance.py       Distance proxy validation against ground-truth CSV
  ablation_study.py          JSONL log replay under ablated pipeline conditions
  benchmark_fps.py           Per-component latency benchmark on synthetic frames
  summarise_log.py           Export per-frame CSV and aggregate stats from JSONL log
docs/
  config.yaml                All tunable parameters
```

## Configuration

All tunable parameters live in `docs/config.yaml`: detector thresholds, tracker settings, IMU noise, orientation filter alpha, motion compensation FOV, risk weights.

## Enabling a Real IMU

The IMU simulator (`src/imu_sim.py`) implements the same interface any real provider must satisfy:

```python
def read(self, timestamp: float) -> IMUReading
```

To plug in a GY-91, MPU9250, or similar:

1. Create `src/imu_hw.py` with a class that reads from the I2C/SPI bus and returns `IMUReading`.
2. In `src/main.py`, replace `IMUSimulator(...)` with your hardware class. The rest of the pipeline (orientation estimator, motion compensator, risk engine) requires no changes.
3. Calibrate accelerometer/gyro biases and magnetometer hard/soft iron offsets in your hardware class before returning readings.

## Limitations

- IMU is simulated; real sensor noise profiles will differ.
- Motion compensation uses a pinhole camera model with small-angle rotation approximation.
- Distance proxy in risk scoring is based on bbox height, not true depth.
- DummyDetector produces fixed synthetic detections – useful only for pipeline testing.
- TFLite output decoding assumes SSD-MobileNet tensor layout; adjust `_decode_outputs` in `detection_tflite.py` for other models.

## Raspberry Pi Setup

```bash
sudo apt update
sudo apt install python3-opencv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main --source webcam
```
