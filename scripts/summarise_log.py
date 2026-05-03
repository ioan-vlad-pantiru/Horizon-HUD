"""Per-run summary CSV generator (Task 3.3).

Reads a JSONL telemetry file produced by the --jsonl flag and outputs
a CSV with one row per frame plus aggregate statistics.

Usage
-----
    python scripts/summarise_log.py --log run.jsonl --out summary.csv

Output CSV columns
------------------
    frame, timestamp, n_tracks, n_hazards,
    top_risk_score, top_risk_level, top_track_id,
    roll_deg, pitch_deg, yaw_deg

Aggregate stats (printed to stdout)
------------------------------------
    - Total frames
    - Mean FPS (from timestamps)
    - Total CRITICAL events
    - Total HIGH events
    - Longest continuous HIGH/CRITICAL streak (frames)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise a Horizon-HUD JSONL telemetry log into a CSV."
    )
    parser.add_argument("--log", required=True, metavar="PATH",
                        help="JSONL telemetry log (produced with --jsonl flag)")
    parser.add_argument("--out", required=True, metavar="PATH",
                        help="Output CSV path")
    return parser.parse_args()


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    args = _parse_args()

    records = _load_jsonl(args.log)
    if not records:
        print("ERROR: JSONL log is empty.", file=sys.stderr)
        sys.exit(1)

    # ── CSV output ────────────────────────────────────────────────────────────
    FIELDNAMES = [
        "frame", "timestamp",
        "n_tracks", "n_hazards",
        "top_risk_score", "top_risk_level", "top_track_id",
        "roll_deg", "pitch_deg", "yaw_deg",
    ]

    total_critical = 0
    total_high = 0
    streak = 0
    max_streak = 0

    timestamps = [r.get("ts", 0.0) for r in records]

    with open(args.out, "w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for rec in records:
            orient = rec.get("orientation", {})
            hazards = rec.get("hazards", [])
            tracks = rec.get("tracks", [])

            # Find highest-scoring hazard
            if hazards:
                top = max(hazards, key=lambda h: h.get("score", 0.0))
                top_score = round(top.get("score", 0.0), 4)
                top_level = top.get("level", "LOW")
                top_tid = top.get("id", -1)
            else:
                top_score = 0.0
                top_level = "LOW"
                top_tid = -1

            # Count alert events
            for h in hazards:
                lvl = h.get("level", "LOW")
                if lvl == "CRITICAL":
                    total_critical += 1
                elif lvl == "HIGH":
                    total_high += 1

            # Streak tracking
            frame_high = any(h.get("level") in ("HIGH", "CRITICAL") for h in hazards)
            if frame_high:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

            writer.writerow({
                "frame": rec.get("frame", 0),
                "timestamp": rec.get("ts", 0.0),
                "n_tracks": len(tracks),
                "n_hazards": len(hazards),
                "top_risk_score": top_score,
                "top_risk_level": top_level,
                "top_track_id": top_tid,
                "roll_deg": round(math.degrees(orient.get("roll", 0.0)), 3),
                "pitch_deg": round(math.degrees(orient.get("pitch", 0.0)), 3),
                "yaw_deg": round(math.degrees(orient.get("yaw", 0.0)), 3),
            })

    # ── aggregate stats ───────────────────────────────────────────────────────
    n_frames = len(records)

    # FPS from timestamps (use first and last)
    if n_frames >= 2 and timestamps[-1] > timestamps[0]:
        mean_fps = (n_frames - 1) / (timestamps[-1] - timestamps[0])
    else:
        mean_fps = float("nan")

    print(f"Log file    : {args.log}")
    print(f"CSV output  : {args.out}")
    print(f"Total frames: {n_frames}")
    print(f"Mean FPS    : {mean_fps:.1f}" if mean_fps == mean_fps else "Mean FPS    : n/a")
    print(f"CRITICAL    : {total_critical} events")
    print(f"HIGH        : {total_high} events")
    print(f"Longest HIGH/CRITICAL streak: {max_streak} frames")


if __name__ == "__main__":
    main()
