# Horizon-HUD — Thesis Improvement Plan

This document is a task list for Claude Code. Work through it top-to-bottom. Each section is self-contained. The project root is assumed to be `Horizon-HUD/`.

---

## Git & GitHub Workflow — Non-Negotiable Rules

Follow these rules for every single task in this plan, without exception.

**Branch base:** `main` is protected. Never create a branch off `main`. All branches must be created from `dev` (or from another feature branch that itself stems from `dev`). Before starting any task, ensure your local `dev` is up to date: `git checkout dev && git pull origin dev`.

**Branch naming:** Use the format `dev/<short-description>`, e.g. `dev/fix-motion-comp-frame-size`, `dev/lateral-risk-feature`, `dev/eval-ablation-script`. Keep names lowercase, hyphen-separated, no spaces.

**One branch per task:** Each numbered task in this plan gets its own branch and its own PR. Do not batch multiple tasks into one branch.

**Commits:** Write clear, imperative commit messages that describe what changed and why, e.g. `fix: pass actual frame dims to MotionCompensator._compute_offset` or `feat: add lateral closing speed score to RiskEngineV1`. Reference the task number in the message body where helpful.

**Pull Requests:** Every branch must be merged via a PR into `dev`, never pushed directly. The PR description should briefly state: what problem it solves, what was changed, and how to verify it (e.g. which test to run). Do not merge the PR yourself — leave it open for review unless explicitly told otherwise.

**Never merge to `main` directly:** Promotion from `dev` to `main` happens separately and is not part of this plan. All PRs target `dev` as the base branch.

**Run tests before opening a PR:** Always run `python -m pytest src/tests/ -v` from the project root before pushing. A PR that breaks existing tests should not be opened until the failures are resolved.

**Summary of the flow for each task:**
```
git checkout dev
git pull origin dev
git checkout -b dev/<task-branch-name>
# ... make changes ...
python -m pytest src/tests/ -v   # must pass
git add -p                        # stage intentionally, not with git add .
git commit -m "type: description"
git push origin dev/<task-branch-name>
# Open PR → base: dev
```

---

## 0. Context

Horizon-HUD is a real-time motorcycle heads-up display system. It chains:
- YOLOv8 TFLite object detection → SORT tracker → ego-motion compensation (IMU + complementary filter) → trapezoid corridor path prediction → risk scoring engine (TTC, distance, EMA, hysteresis, persistence gate).

The thesis goal is to demonstrate a complete, evaluated ADAS prototype for motorcycles. The improvements below fall into three tiers: **bug fixes** (must do), **evaluation** (needed for a thesis defence), and **feature enhancements** (differentiation). Do them in order.

---

## TIER 1 — Bug Fixes (Critical)

### 1.1 Fix hardcoded frame size in `motion_comp.py`

**File:** `src/perception/motion_comp.py`

**Problem:** `_compute_offset` on line 99 always calls `self._get_K(640, 480)`, so the camera intrinsic matrix is wrong for every resolution other than 640×480. Motion compensation silently produces incorrect pixel offsets at other resolutions.

**Fix:**
- Add `frame_w: int` and `frame_h: int` parameters to `_compute_offset(self, dr, dp, dy, frame_w, frame_h)`.
- Update the call inside `_compute_offset` to `K = self._get_K(frame_w, frame_h)`.
- Store `frame_w` and `frame_h` on the instance in `update_orientation` so `_compute_offset` can use actual dimensions.
- The simplest approach: store them as `self._fw` and `self._fh`, set to 640/480 as defaults, updated each call to `update_orientation`. Pass them through to `_compute_offset`.
- Update the call in `src/main.py` inside the main loop: after reading the frame (`fh, fw = frame.shape[:2]`), pass `fw` and `fh` into the compensator. The compensator's `update_orientation` should accept optional `frame_w` / `frame_h` kwargs and store them.

### 1.2 Clarify/fix SORT tracker max_age pruning logic

**File:** `src/detection/tracking_sort.py`

**Problem:** Line 206-209:
```python
self._tracks = [
    t for i, t in enumerate(self._tracks)
    if i in matched_t or t.time_since_update < self.max_age * 0.1
]
```
`time_since_update` is measured in seconds, but `max_age` is a frame-count parameter. With `max_age=5` this keeps unmatched tracks for only 0.5 seconds, which may drop tracks that have briefly left the frame. The intent appears to be keeping unmatched tracks alive for up to `max_age` frames at ~30fps (≈ 0.167s per frame), meaning the threshold should be something like `max_age / fps`. 

**Fix:** Replace the hard-coded `0.1` multiplier with a `frame_duration_s` parameter (default `1/30`) on `SORTTracker.__init__`, so the threshold is `max_age * self.frame_duration_s`. Document the parameter clearly. Update `config.yaml` and `_build_pipeline` in `main.py` if needed.

### 1.3 Fix test suite to cover the deployed engine

**File:** `src/tests/test_risk.py`

**Problem:** This file imports and tests `src.risk.risk.RiskEngine` (the old V0 engine), not `src.risk.risk_engine.RiskEngineV1`, which is the engine actually running in `main.py`. The CI test suite is not testing the deployed code.

**Fix:**
- Create a new file `src/tests/test_risk_engine_v1.py`.
- Write tests that import `from src.risk.risk_engine import RiskEngineV1` and `from src.risk.risk_types import RiskConfig, CorridorConfig`.
- Required test cases:
  1. **Monotonicity — closer bbox = higher risk**: create two tracks with identical velocities, one with a large bbox (close object) and one with a small bbox (distant), assert the close one scores higher.
  2. **Hysteresis — no flicker on level boundary**: run 20 frames of a track hovering just above the MEDIUM enter threshold, verify the level does not oscillate between LOW and MEDIUM once stable.
  3. **Persistence gate — CRITICAL requires K consecutive frames**: feed a track with a score just above `enter_critical` for only 2 frames (below `persist_k=3`), assert it does NOT reach CRITICAL. Feed 3+ frames, assert it does.
  4. **Corridor — in-path object scores higher than off-path**: same bbox size/velocity, one centered in the corridor, one at the frame edge.
  5. **Score bounds**: assert risk_score is always in [0.0, 1.0] across 50 random track inputs.
  6. **Empty tracks**: assert `update([], (640, 480), 0.0)` returns `[]`.

---

## TIER 2 — Evaluation (Thesis-Essential)

### 2.1 Distance proxy validation script

**File to create:** `scripts/validate_distance.py`

This script validates the `distance_proxy_from_bbox` function against known ground-truth distances. It is used to generate the error characterisation table for the thesis.

**Implementation:**
- Accept a CSV file as input with columns: `frame, bbox_x1, bbox_y1, bbox_x2, bbox_y2, true_distance_m, label`.
- For each row, compute `distance_proxy_from_bbox(bbox_height, known_height_m, focal_length_px)` using the label's known height from `RiskConfig.known_heights_m` and a configurable `fov_deg` (default 70.0).
- Compute MAE, RMSE, and mean percentage error.
- Print a summary table and optionally save a plot of `estimated_m` vs `true_distance_m` with a y=x reference line (use matplotlib if available, otherwise skip the plot).
- Add a `--fov` argument so the FOV can be swept to find the calibration that minimises error.

**Also add a README note** in `README.md` under a new "Evaluation" section explaining how to collect ground-truth distances (tape measure or LiDAR) and run this script.

### 2.2 Ablation study script

**File to create:** `scripts/ablation_study.py`

This script replays a JSONL telemetry log (produced with `--jsonl`) and re-scores tracks under different pipeline configurations to quantify the contribution of each component.

**Conditions to compare (run sequentially on the same log):**
1. **Baseline**: full pipeline as-is.
2. **No ego-motion compensation**: set `compensated_velocities=None` in every `risk_engine.update` call.
3. **No corridor**: set corridor config to `bottom_width_ratio=1.0, top_width_ratio=1.0` (full-frame corridor, so all objects are always "in path").
4. **No persistence gate**: set `persist_k=1` (any single frame above threshold escalates).
5. **No hysteresis**: set all enter/exit thresholds equal.

**Metrics to report per condition:**
- Total CRITICAL alerts fired.
- Total HIGH alerts fired.
- Alert rate (alerts per 100 frames).
- Mean frames from first detection to first HIGH alert (latency).

Print a markdown table to stdout. The script should be runnable as `python scripts/ablation_study.py --log path/to/log.jsonl`.

### 2.3 FPS benchmark script

**File to create:** `scripts/benchmark_fps.py`

**Purpose:** Measure per-component latency so the thesis can include a runtime table (critical for arguing real-time suitability on Raspberry Pi).

**Implementation:**
- Run the full pipeline on a synthetic 640×480 black frame (no real camera needed) for `--frames N` iterations (default 200).
- Use `time.perf_counter()` to independently time each stage: detection, tracking, IMU read + orientation, motion compensation, risk scoring.
- Print a table: component | mean ms | std ms | % of frame budget.
- Also print overall FPS achieved.
- Accept `--model` to switch between `models/best_int8.tflite` and `models/best_float16.tflite` so quantisation impact can be reported.

### 2.4 Lateral closing speed feature

**Files:** `src/risk/risk_features.py`, `src/risk/risk_types.py`, `src/risk/risk_engine.py`, `docs/config.yaml`

**Problem:** A pedestrian stepping laterally into the motorcycle's path is a high-priority hazard. Currently the risk engine has no lateral velocity term. The old `test_risk.py` tests `test_lateral_motion_penalty` but it tests the V0 engine — `RiskEngineV1` underscores lateral threats.

**Implementation:**

In `risk_features.py`, add:
```python
def lateral_risk_score(
    track: Track,
    corridor_poly: np.ndarray,
    compensated_vel: Optional[tuple[float, float]] = None,
) -> float:
    """Score in [0, 1] for lateral motion directed toward the corridor centreline.

    High when: object is outside or near the corridor edge AND moving
    inward (toward centreline) fast enough to intersect within ~2 seconds.
    """
```
- Compute the object's lateral velocity (`vx` from compensated or raw velocity).
- Compute the signed distance from the corridor centreline at the object's y position.
- If the object is moving toward the centreline and the time-to-intersect is below a threshold, return a proportional score.
- Return 0 if the object is already inside the corridor (handled by path_factor) or moving away.

In `risk_types.py`, add `w_lateral: float = 0.10` to `RiskConfig` and reduce `w_erratic` to `0.05` or redistribute as appropriate so weights still sum to ~1.0. Add `lateral_ttc_s: float = 2.0` threshold.

In `risk_engine.py`, call `lateral_risk_score` in `_score_track` and include it in the weighted sum.

In `config.yaml`, add `w_lateral: 0.10` and `lateral_ttc_s: 2.0` under the `risk:` section.

In `test_risk_engine_v1.py` (from task 1.3), add a test: a pedestrian at the corridor edge with strong inward lateral velocity should score materially higher than the same pedestrian stationary.

---

## TIER 3 — Enhancements (Differentiation)

### 3.1 Improve `erratic_score` with heading change variance

**File:** `src/risk/risk_features.py`

**Current limitation:** `erratic_score` only measures variance in speed magnitude. A vehicle swerving (changing heading rapidly) without changing speed returns a low erratic score.

**Fix:** Blend speed variance with heading (direction) variance:
```python
def erratic_score(history: deque[TrackSnapshot]) -> float:
    """Blended erratic score: speed variance + heading change variance."""
```
- Compute speeds as before.
- Compute headings as `math.atan2(vy, vx)` for each snapshot where speed > 0.
- Compute angular differences between consecutive headings (wrapping with `math.atan2(sin, cos)` to handle ±π wrap).
- Normalise heading variance: a standard deviation of π/4 rad/s (45°/s) → ~1.0.
- Return `0.5 * speed_component + 0.5 * heading_component`, both clamped to [0, 1].

### 3.2 Add `--eval` mode to `main.py` for annotated video replay

**File:** `src/main.py`

Add an `--eval` CLI flag. When set, `main.py` also reads a ground-truth JSONL (passed via `--gt path/to/gt.jsonl`) where each record has `{"frame": N, "hazard_ids": [...]}` — the set of track IDs that are genuinely hazardous in that frame.

At shutdown, print:
- True positive rate: frames where a CRITICAL/HIGH alert fired AND a ground-truth hazard was present.
- False positive rate: frames where CRITICAL/HIGH fired with no ground-truth hazard.
- Mean lead time: average frames between first ground-truth hazard frame and first HIGH/CRITICAL alert.

This is lightweight but gives the thesis a proper evaluation number without requiring a full MOTA-style dataset.

### 3.3 Export a per-run summary CSV from JSONL

**File to create:** `scripts/summarise_log.py`

Reads a JSONL telemetry file produced by the `--jsonl` flag and outputs a CSV with one row per frame:
`frame, timestamp, n_tracks, n_hazards, top_risk_score, top_risk_level, top_track_id, roll_deg, pitch_deg, yaw_deg`

Also prints aggregate stats: total frames, mean FPS (from timestamps), total CRITICAL events, total HIGH events, longest continuous HIGH/CRITICAL streak.

Usage: `python scripts/summarise_log.py --log out.jsonl --out summary.csv`

### 3.4 Update README with Evaluation section

After the above scripts exist, add an "## Evaluation" section to `README.md` covering:
- How to produce a telemetry log: `python -m src.main --source video.mp4 --jsonl run.jsonl`
- How to run the distance validation: `python scripts/validate_distance.py --csv ground_truth.csv`
- How to run the ablation study: `python scripts/ablation_study.py --log run.jsonl`
- How to run the FPS benchmark: `python scripts/benchmark_fps.py --model models/best_int8.tflite`
- How to run the full test suite: `python -m pytest src/tests/ -v`

Also update the "Architecture" section to include the new `scripts/` tools and the `src/tests/test_risk_engine_v1.py` test file.

---

## Implementation Order

1. Tasks 1.1, 1.2, 1.3 — bug fixes. Run `python -m pytest src/tests/ -v` to confirm green after each.
2. Task 2.4 — lateral score feature (requires editing existing source files, so best done before adding eval scripts that depend on stable scoring).
3. Tasks 2.1, 2.2, 2.3 — evaluation scripts (new files, no risk of breaking existing code).
4. Tasks 3.1, 3.2, 3.3, 3.4 — enhancements and documentation.

At each step, run the full test suite to confirm nothing regressed.
