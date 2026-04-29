"""Ablation study script (Task 2.2).

Replays a JSONL telemetry log and re-scores tracks under five pipeline
configurations to quantify the contribution of each component.

Usage
-----
    python scripts/ablation_study.py --log path/to/run.jsonl

Conditions
----------
  1. Baseline          – full pipeline as-is.
  2. No ego-motion     – compensated_velocities=None in every update() call.
  3. No corridor       – corridor set to full-frame width (all objects in path).
  4. No persistence    – persist_k=1 (any single frame above threshold escalates).
  5. No hysteresis     – enter == exit for all levels.

Metrics reported per condition
--------------------------------
  - Total CRITICAL alerts fired.
  - Total HIGH alerts fired.
  - Alert rate (alerts per 100 frames).
  - Mean frames from first detection of a track to first HIGH/CRITICAL alert.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.types import Track
from src.risk.risk_engine import RiskEngineV1
from src.risk.risk_types import CorridorConfig, RiskConfig


# ── JSONL loading ─────────────────────────────────────────────────────────────

def _load_frames(log_path: str) -> list[dict[str, Any]]:
    frames = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def _frame_to_tracks(record: dict[str, Any]) -> list[Track]:
    tracks = []
    for t in record.get("tracks", []):
        tracks.append(Track(
            track_id=t["id"],
            bbox_xyxy=tuple(t["bbox"]),
            velocity_px_s=tuple(t["vel"]),
            age=t.get("hits", 1),
            time_since_update=t.get("tsu", 0.0),
            hits=t.get("hits", 1),
            label=t.get("label", "default"),
            class_id=0,
        ))
    return tracks


# ── condition factories ───────────────────────────────────────────────────────

def _make_engine(condition: str) -> RiskEngineV1:
    risk_cfg = RiskConfig()
    corridor_cfg = CorridorConfig()

    if condition == "no_corridor":
        corridor_cfg = dataclasses.replace(
            corridor_cfg,
            bottom_width_ratio=1.0,
            top_width_ratio=1.0,
        )
    elif condition == "no_persistence":
        risk_cfg = dataclasses.replace(risk_cfg, persist_k=1)
    elif condition == "no_hysteresis":
        risk_cfg = dataclasses.replace(
            risk_cfg,
            enter_medium=risk_cfg.exit_medium,
            enter_high=risk_cfg.exit_high,
            enter_critical=risk_cfg.exit_critical,
        )

    return RiskEngineV1(risk_cfg, corridor_cfg)


# ── replay ────────────────────────────────────────────────────────────────────

def _replay(frames: list[dict], condition: str) -> dict[str, Any]:
    engine = _make_engine(condition)
    no_comp = condition == "no_ego_motion"

    n_frames = len(frames)
    critical_count = 0
    high_count = 0
    alert_frames = 0  # frames with at least one HIGH or CRITICAL alert

    # For latency: first_seen[track_id] = frame_idx when track first appeared
    first_seen: dict[int, int] = {}
    # first_alerted[track_id] = frame_idx when track first reached HIGH or CRITICAL
    first_alerted: dict[int, int] = {}

    for frame_idx, record in enumerate(frames):
        tracks = _frame_to_tracks(record)
        ts = record.get("ts", float(frame_idx) / 30.0)
        frame_size = (640, 480)  # assume standard frame size

        # Track first-seen
        for tr in tracks:
            if tr.track_id not in first_seen:
                first_seen[tr.track_id] = frame_idx

        comp_vels = None
        if not no_comp:
            # Re-use raw velocities as "compensated" (no separate comp data in log)
            comp_vels = {tr.track_id: tr.velocity_px_s for tr in tracks}

        hazards = engine.update(tracks, frame_size, ts, compensated_velocities=comp_vels)

        frame_has_alert = False
        for ha in hazards:
            if ha.risk_level == "CRITICAL":
                critical_count += 1
                frame_has_alert = True
            elif ha.risk_level == "HIGH":
                high_count += 1
                frame_has_alert = True
            # Record first alert per track
            if ha.risk_level in ("HIGH", "CRITICAL") and ha.track_id not in first_alerted:
                first_alerted[ha.track_id] = frame_idx

        if frame_has_alert:
            alert_frames += 1

    alert_rate = (alert_frames / n_frames * 100.0) if n_frames > 0 else 0.0

    # Mean detection-to-alert latency in frames
    latencies = []
    for tid, alerted_at in first_alerted.items():
        seen_at = first_seen.get(tid)
        if seen_at is not None:
            latencies.append(alerted_at - seen_at)
    mean_latency = sum(latencies) / len(latencies) if latencies else float("nan")

    return {
        "condition": condition,
        "n_frames": n_frames,
        "critical": critical_count,
        "high": high_count,
        "alert_rate": alert_rate,
        "mean_latency_frames": mean_latency,
    }


# ── main ─────────────────────────────────────────────────────────────────────

CONDITIONS = [
    ("baseline",        "Baseline (full pipeline)"),
    ("no_ego_motion",   "No ego-motion compensation"),
    ("no_corridor",     "No corridor (full-frame path)"),
    ("no_persistence",  "No persistence gate (persist_k=1)"),
    ("no_hysteresis",   "No hysteresis (enter == exit)"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation study: replay a JSONL log under different pipeline configs."
    )
    parser.add_argument("--log", required=True, metavar="PATH",
                        help="JSONL telemetry log produced with --jsonl flag")
    args = parser.parse_args()

    print(f"Loading log: {args.log}")
    frames = _load_frames(args.log)
    print(f"Loaded {len(frames)} frames.\n")

    results = []
    for key, label in CONDITIONS:
        print(f"Running condition: {label} …", end=" ", flush=True)
        r = _replay(frames, key)
        r["label"] = label
        results.append(r)
        print("done.")

    # ── markdown table ────────────────────────────────────────────────────────
    print()
    header = (
        f"| {'Condition':<38} | {'CRITICAL':>8} | {'HIGH':>8} | "
        f"{'Alert rate':>10} | {'Mean latency (frames)':>21} |"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print("|" + "-" * 40 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 23 + "|")
    for r in results:
        lat = f"{r['mean_latency_frames']:.1f}" if r['mean_latency_frames'] == r['mean_latency_frames'] else "n/a"
        print(
            f"| {r['label']:<38} | {r['critical']:>8} | {r['high']:>8} | "
            f"{r['alert_rate']:>9.1f}% | {lat:>21} |"
        )
    print(sep)


if __name__ == "__main__":
    main()
