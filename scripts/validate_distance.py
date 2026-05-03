"""Distance proxy validation script (Task 2.1).

Computes monocular distance estimates from bounding boxes and compares
them against known ground-truth distances to characterise measurement error.

Usage
-----
    python scripts/validate_distance.py --csv ground_truth.csv [--fov 70.0]

CSV format (one row per detection)
-----------------------------------
    frame, bbox_x1, bbox_y1, bbox_x2, bbox_y2, true_distance_m, label

Example
-------
    frame,bbox_x1,bbox_y1,bbox_x2,bbox_y2,true_distance_m,label
    1,200,100,300,380,8.5,pedestrian
    2,180,120,310,410,6.2,pedestrian
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

# ── allow running as a top-level script from the project root ─────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.risk.risk_features import distance_proxy_from_bbox, focal_length_from_fov
from src.risk.risk_types import RiskConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate distance_proxy_from_bbox against ground-truth distances."
    )
    parser.add_argument("--csv", required=True, metavar="PATH",
                        help="Ground-truth CSV file (see script docstring for format)")
    parser.add_argument("--fov", type=float, default=70.0,
                        help="Horizontal FOV in degrees (default: 70.0)")
    parser.add_argument("--frame-w", type=int, default=640, dest="frame_w",
                        help="Frame width in pixels used to compute focal length (default: 640)")
    parser.add_argument("--plot", action="store_true",
                        help="Save a scatter plot of estimated vs true distances")
    parser.add_argument("--out-plot", default="distance_validation.png", dest="out_plot",
                        help="Output path for the plot (default: distance_validation.png)")
    return parser.parse_args()


def _load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "frame": int(row["frame"]),
                "bbox": (float(row["bbox_x1"]), float(row["bbox_y1"]),
                         float(row["bbox_x2"]), float(row["bbox_y2"])),
                "true_distance_m": float(row["true_distance_m"]),
                "label": row["label"].strip().lower(),
            })
    return rows


def main() -> None:
    args = _parse_args()
    cfg = RiskConfig()
    focal = focal_length_from_fov(args.frame_w, args.fov)

    rows = _load_csv(args.csv)
    if not rows:
        print("ERROR: CSV is empty or has no valid rows.", file=sys.stderr)
        sys.exit(1)

    results = []
    for row in rows:
        x1, y1, x2, y2 = row["bbox"]
        height_px = max(y2 - y1, 1.0)
        label = row["label"]
        known_h = cfg.known_heights_m.get(label, cfg.known_heights_m.get("default", 1.5))
        est = distance_proxy_from_bbox(height_px, known_h, focal)
        true_d = row["true_distance_m"]
        error = est - true_d
        pct = abs(error) / true_d * 100.0 if true_d > 0 else 0.0
        results.append({
            "frame": row["frame"],
            "label": label,
            "height_px": height_px,
            "known_h_m": known_h,
            "est_m": est,
            "true_m": true_d,
            "error_m": error,
            "abs_error_m": abs(error),
            "pct_error": pct,
        })

    n = len(results)
    mae = sum(r["abs_error_m"] for r in results) / n
    rmse = math.sqrt(sum(r["error_m"] ** 2 for r in results) / n)
    mpe = sum(r["pct_error"] for r in results) / n

    # ── summary table ─────────────────────────────────────────────────────────
    col_w = [6, 12, 10, 10, 8, 8, 8]
    header = f"{'frame':>6}  {'label':<12}  {'est (m)':>8}  {'true (m)':>8}  {'err (m)':>7}  {'|err|':>6}  {'%err':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['frame']:>6}  {r['label']:<12}  {r['est_m']:>8.2f}  "
            f"{r['true_m']:>8.2f}  {r['error_m']:>7.2f}  "
            f"{r['abs_error_m']:>6.2f}  {r['pct_error']:>5.1f}%"
        )
    print("-" * len(header))
    print(f"\nN = {n}  |  FOV = {args.fov}°  |  frame_w = {args.frame_w}px  |  focal = {focal:.1f}px")
    print(f"MAE  = {mae:.3f} m")
    print(f"RMSE = {rmse:.3f} m")
    print(f"MPE  = {mpe:.1f} %  (mean absolute percentage error)")

    # ── optional plot ─────────────────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nWarning: matplotlib not available — skipping plot.", file=sys.stderr)
            return
        true_vals = [r["true_m"] for r in results]
        est_vals = [r["est_m"] for r in results]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(true_vals, est_vals, alpha=0.7, edgecolors="k", linewidths=0.5)
        lim_max = max(max(true_vals), max(est_vals)) * 1.1
        ax.plot([0, lim_max], [0, lim_max], "r--", label="y = x (perfect)")
        ax.set_xlabel("True distance (m)")
        ax.set_ylabel("Estimated distance (m)")
        ax.set_title(f"Distance proxy vs ground truth  (FOV={args.fov}°, RMSE={rmse:.2f}m)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_plot, dpi=150)
        print(f"\nPlot saved: {args.out_plot}")


if __name__ == "__main__":
    main()
