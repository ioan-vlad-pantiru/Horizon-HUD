"""Prepare the TLD-LOKI dataset for signal-classifier training.

Downloads TLD-LOKI.zip from ChaiJohn/TLD on HuggingFace, then streams
through it in-place (no full extraction needed).  Stops as soon as
--max_crops crops have been collected.

Dataset structure (COCO format)
--------------------------------
  TLD-LOKI/
    group_NNN/
      scenario_NNN/
        _annotations.coco.json   ← COCO per-scenario labels
        image_XXXX_...jpg

Category names that carry signal state:
  "brake"  → brake=1, left=0, right=0
  "go"     → brake=0, left=0, right=0  (no active signal)
  "left"   → brake=0, left=1, right=0
  "right"  → brake=0, left=0, right=1
  "cars*" / "vehicle" → skip (vehicle bbox only, no signal info)

Usage
-----
  python scripts/prepare_tld_dataset.py --token hf_XXXX
  python scripts/prepare_tld_dataset.py --local_zip /path/to/TLD-LOKI.zip
  python scripts/prepare_tld_dataset.py --token hf_XXXX --max_crops 20000

Output
------
  data/tld_signal/
      images/        96×96 PNG rear-ROI crops
      manifest.csv   filename, brake, left, right, split
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
except ImportError:
    print("ERROR: pip install huggingface-hub")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: pip install opencv-python numpy")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ID = "ChaiJohn/TLD"
ALL_ZIPS = [
    ("TLD-LOKI.zip",    "~11.7 GB"),
    ("TLD-YT-part1.zip","~26 GB"),
    ("TLD-YT-part2.zip","~26 GB"),
]
ROI_BOTTOM_FRACTION = 0.40
TARGET_SIZE = (96, 96)
SPLIT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_MAX_CROPS = 5_000

# Maps COCO category name → (brake, left, right).
# Names that don't appear here are skipped (generic vehicle boxes).
_NAME_TO_LABEL: dict[str, tuple[int, int, int]] = {
    "brake": (1, 0, 0),
    "go":    (0, 0, 0),
    "left":  (0, 1, 0),
    "right": (0, 0, 1),
}

_GATED_HELP = """
Access denied (gated dataset).

  1. Visit https://huggingface.co/datasets/ChaiJohn/TLD and request access.
  2. Generate a token at https://huggingface.co/settings/tokens (Read scope).
  3. Re-run:  python scripts/prepare_tld_dataset.py --token hf_XXXX
"""


# ---------------------------------------------------------------------------
# COCO parsing helpers
# ---------------------------------------------------------------------------

def _build_label_map(categories: list[dict]) -> dict[int, tuple[int, int, int]]:
    """Map category_id → (brake, left, right) using category name."""
    result: dict[int, tuple[int, int, int]] = {}
    for cat in categories:
        label = _NAME_TO_LABEL.get(cat["name"])
        if label is not None:
            result[cat["id"]] = label
    return result


def _parse_coco(data: dict) -> list[dict]:
    """Extract per-image annotated bboxes from one COCO JSON dict.

    Returns a flat list of dicts:
        { filename, brake, left, right, bbox_xyxy: (x1,y1,x2,y2) }
    Only annotations whose category name maps to a signal state are kept.
    """
    label_map = _build_label_map(data.get("categories", []))
    if not label_map:
        return []

    id_to_file: dict[int, str] = {
        img["id"]: img["file_name"]
        for img in data.get("images", [])
    }

    # Group signal-labeled bboxes by image_id
    by_image: dict[int, list[dict]] = {}
    for ann in data.get("annotations", []):
        label = label_map.get(ann["category_id"])
        if label is None:
            continue
        iid = ann["image_id"]
        x, y, w, h = ann["bbox"]   # COCO: [x_min, y_min, width, height]
        by_image.setdefault(iid, []).append({
            "brake": label[0], "left": label[1], "right": label[2],
            "bbox_xyxy": (x, y, x + w, y + h),
        })

    records: list[dict] = []
    for iid, anns in by_image.items():
        fname = id_to_file.get(iid)
        if fname is None:
            continue
        for ann in anns:
            records.append({**ann, "filename": fname})
    return records


# ---------------------------------------------------------------------------
# ROI crop
# ---------------------------------------------------------------------------

def _crop_rear_roi(
    img: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
) -> np.ndarray | None:
    """Crop bottom 40% of the bounding box and resize to TARGET_SIZE."""
    fh, fw = img.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    roi_y1 = y1 + (1.0 - ROI_BOTTOM_FRACTION) * (y2 - y1)
    x1c, y1c = max(0, int(x1)), max(0, int(roi_y1))
    x2c, y2c = min(fw, int(x2)), min(fh, int(y2))
    if x2c <= x1c or y2c <= y1c:
        return None
    roi = img[y1c:y2c, x1c:x2c]
    return cv2.resize(roi, TARGET_SIZE) if roi.size > 0 else None


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def _stratified_split(records: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    for i, rec in enumerate(shuffled):
        rec["split"] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return shuffled


# ---------------------------------------------------------------------------
# Core processing — streams through the zip
# ---------------------------------------------------------------------------

def _report_labels(crops: list[dict]) -> None:
    if not crops:
        return
    d = {
        "brake": sum(1 for r in crops if r["brake"]),
        "left":  sum(1 for r in crops if r["left"]),
        "right": sum(1 for r in crops if r["right"]),
        "none":  sum(1 for r in crops if not r["brake"] and not r["left"] and not r["right"]),
    }
    print(f"  Running total: {len(crops)} crops — {d}")


def _process_zip(
    zip_path: str,
    img_dir: Path,
    max_crops: int,
    start_idx: int = 0,
) -> list[dict]:
    """Stream through the zip file, crop rear ROIs, return manifest rows."""
    all_crops: list[dict] = []
    crop_idx = start_idx

    with zipfile.ZipFile(zip_path) as zf:
        names_set = set(zf.namelist())
        ann_files = sorted(n for n in names_set if n.endswith("_annotations.coco.json"))
        print(f"  {len(ann_files)} scenarios in zip.  "
              f"Collecting until {max_crops} total crops …")

        for ann_name in ann_files:
            if crop_idx >= max_crops:
                break

            # Parse COCO annotation
            with zf.open(ann_name) as f:
                coco_data = json.load(f)
            records = _parse_coco(coco_data)
            if not records:
                continue

            scenario_dir = ann_name.rsplit("/", 1)[0] + "/"

            # Process each annotated bbox
            loaded_images: dict[str, np.ndarray] = {}
            for rec in records:
                if crop_idx >= max_crops:
                    break

                img_entry = scenario_dir + rec["filename"]
                if img_entry not in names_set:
                    continue

                # Load image (cache within this scenario to avoid re-reading)
                if img_entry not in loaded_images:
                    with zf.open(img_entry) as f:
                        buf = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    loaded_images[img_entry] = img

                img = loaded_images[img_entry]
                crop = _crop_rear_roi(img, rec["bbox_xyxy"])
                if crop is None:
                    continue

                out_fname = f"crop_{crop_idx:07d}.png"
                cv2.imwrite(str(img_dir / out_fname), crop)
                all_crops.append({
                    "filename": out_fname,
                    "brake": rec["brake"],
                    "left": rec["left"],
                    "right": rec["right"],
                })
                crop_idx += 1

            loaded_images.clear()

            if crop_idx % 500 == 0 and crop_idx > start_idx:
                print(f"  {crop_idx}/{max_crops} crops …")

    return all_crops


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _download_and_process(
    zip_name: str,
    token: str | None,
    out_dir: Path,
    img_dir: Path,
    keep_zip: bool,
    max_crops: int,
    existing_crops: list[dict],
) -> list[dict]:
    """Download one zip, process it, delete it, return accumulated crops."""
    zip_path = out_dir / zip_name
    print(f"\nDownloading {zip_name} from {REPO_ID} …")
    try:
        zip_path_str = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset",
            filename=zip_name, token=token,
            local_dir=str(out_dir),
        )
    except GatedRepoError:
        print(_GATED_HELP)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Download failed: {exc}")
        sys.exit(1)

    crops = _process_zip(zip_path_str, img_dir, max_crops, start_idx=len(existing_crops))

    if not keep_zip:
        try:
            os.remove(zip_path_str)
            print(f"Deleted {zip_name}")
        except OSError:
            pass

    return existing_crops + crops


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TLD signal dataset")
    parser.add_argument("--out_dir", default="data/tld_signal")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token (required for gated dataset)")
    parser.add_argument("--local_zip", default=None,
                        help="Path to a single already-downloaded zip (skips download)")
    parser.add_argument("--all", dest="all_zips", action="store_true",
                        help="Download and process all three zips (~64 GB total)")
    parser.add_argument("--keep_zip", action="store_true",
                        help="Do not delete zips after processing")
    parser.add_argument("--max_crops", type=int, default=DEFAULT_MAX_CROPS,
                        help=f"Total crop target across all zips (default {DEFAULT_MAX_CROPS}; "
                             "0 = no limit)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    max_crops = args.max_crops if args.max_crops > 0 else 10_000_000

    all_crops: list[dict] = []

    # ── single local zip ──────────────────────────────────────────────────────
    if args.local_zip:
        print(f"Using local zip: {args.local_zip}")
        all_crops = _process_zip(args.local_zip, img_dir, max_crops)

    # ── all three zips from HuggingFace ──────────────────────────────────────
    elif args.all_zips:
        token = args.token or os.environ.get("HF_TOKEN")
        if not token:
            print("ERROR: --token hf_XXXX or HF_TOKEN env var required.")
            sys.exit(1)
        total_gb = sum(float(s.strip("~<> GB").split()[0]) for _, s in ALL_ZIPS)
        print(f"Processing all {len(ALL_ZIPS)} zips (~{total_gb:.0f} GB total, "
              f"target {max_crops} crops).")
        for zip_name, size_hint in ALL_ZIPS:
            if len(all_crops) >= max_crops:
                break
            print(f"\n── {zip_name} {size_hint} ──")
            all_crops = _download_and_process(
                zip_name, token, out_dir, img_dir,
                args.keep_zip, max_crops, all_crops,
            )
            _report_labels(all_crops)

    # ── default: just TLD-LOKI.zip ────────────────────────────────────────────
    else:
        token = args.token or os.environ.get("HF_TOKEN")
        if not token:
            print("ERROR: --token hf_XXXX or HF_TOKEN env var required.")
            sys.exit(1)
        all_crops = _download_and_process(
            ALL_ZIPS[0][0], token, out_dir, img_dir,
            args.keep_zip, max_crops, [],
        )

    if not all_crops:
        print("ERROR: No crops produced.  The annotation format may have changed —\n"
              "       inspect a COCO JSON inside the zip and update _NAME_TO_LABEL.")
        sys.exit(1)

    # ── split + write manifest ────────────────────────────────────────────────
    all_crops = _stratified_split(all_crops, args.seed)

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "brake", "left", "right", "split"])
        writer.writeheader()
        writer.writerows(all_crops)

    counts = {s: sum(1 for r in all_crops if r["split"] == s) for s in ("train", "val", "test")}
    print(f"\nDone.  {len(all_crops)} crops → {out_dir}")
    print(f"  train={counts['train']}  val={counts['val']}  test={counts['test']}")
    print(f"  manifest: {manifest_path}")
    label_counts = {
        "brake": sum(1 for r in all_crops if r["brake"]),
        "left":  sum(1 for r in all_crops if r["left"]),
        "right": sum(1 for r in all_crops if r["right"]),
        "none":  sum(1 for r in all_crops if not r["brake"] and not r["left"] and not r["right"]),
    }
    print(f"  label distribution: {label_counts}")


if __name__ == "__main__":
    main()
