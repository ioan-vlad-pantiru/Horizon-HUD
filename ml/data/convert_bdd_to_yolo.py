"""
Convert BDD100K per-image JSON labels to YOLO format.
Maps BDD categories to Horizon-HUD safety classes.
"""

import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

BDD_TO_HORIZON = {
    "car": 0,
    "bus": 0,
    "truck": 0,
    "train": 0,
    "person": 1,
    "rider": 2,
    "bike": 2,
    "motor": 2,
    "motorcycle": 2,
    "bicycle": 2,
    "traffic sign": 3,
    "traffic light": 3,
}

IMG_W = 1280
IMG_H = 720


def convert_one(args: tuple) -> int:
    label_path, out_dir = args
    with open(label_path, "r") as f:
        ann = json.load(f)

    frames = ann.get("frames", [])
    if not frames:
        return 0
    objects = frames[0].get("objects", [])

    lines = []
    for obj in objects:
        cat = obj.get("category", "").lower()
        cls_id = BDD_TO_HORIZON.get(cat)
        if cls_id is None:
            continue

        box = obj.get("box2d", {})
        if not box:
            continue

        x1 = max(0.0, float(box.get("x1", 0)))
        y1 = max(0.0, float(box.get("y1", 0)))
        x2 = min(float(IMG_W), float(box.get("x2", 0)))
        y2 = min(float(IMG_H), float(box.get("y2", 0)))

        if x2 <= x1 or y2 <= y1:
            continue

        cx = ((x1 + x2) / 2.0) / IMG_W
        cy = ((y1 + y2) / 2.0) / IMG_H
        w = (x2 - x1) / IMG_W
        h = (y2 - y1) / IMG_H

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    out_path = out_dir / (label_path.stem + ".txt")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def main():
    parser = argparse.ArgumentParser(description="Convert BDD100K labels to YOLO format")
    parser.add_argument("--labels-root", type=str, required=True,
                        help="BDD100K per-image JSON labels root (e.g. datasets/100k_labels)")
    parser.add_argument("--images-root", type=str, required=True,
                        help="BDD100K images root (e.g. datasets/100k)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output YOLO dataset root (e.g. datasets/horizon_yolo)")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    labels_root = Path(args.labels_root)
    images_root = Path(args.images_root)
    output_root = Path(args.output)
    total_labels = 0
    total_objects = 0

    for split in args.splits:
        split_dir = labels_root / split
        if not split_dir.exists():
            print(f"Skipping {split} (not found: {split_dir})")
            continue

        out_labels = output_root / "labels" / split
        out_labels.mkdir(parents=True, exist_ok=True)

        out_images = output_root / "images" / split
        if not out_images.exists():
            out_images.parent.mkdir(parents=True, exist_ok=True)
            src = (images_root / split).resolve()
            out_images.symlink_to(src)
            print(f"Symlinked {out_images} -> {src}")

        json_files = sorted(split_dir.glob("*.json"))
        print(f"{split}: converting {len(json_files)} label files...")

        work = [(p, out_labels) for p in json_files]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(convert_one, w): w for w in work}
            for future in as_completed(futures):
                n = future.result()
                total_objects += n
                total_labels += 1

    print(f"\nDone: {total_labels} files, {total_objects} objects converted.")
    print(f"YOLO dataset at: {output_root}")


if __name__ == "__main__":
    main()
