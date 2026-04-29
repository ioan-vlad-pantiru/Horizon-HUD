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

## Architecture

```
src/
  types.py              Core dataclasses (Detection, Track, IMUReading, Orientation, RiskAssessment)
  detection_tflite.py   TFLite detector with DummyDetector fallback
  tracking_sort.py      SORT tracker (Kalman + IoU + Hungarian)
  imu_sim.py            Simulated IMU with 4 driving scenarios
  orientation.py        Complementary filter (gyro + accel + mag)
  motion_comp.py        Ego-motion compensation via orientation deltas
  risk.py               Risk scoring engine (proximity, closing speed, TTC, lateral, accel)
  main.py               Pipeline runner and visualization
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
