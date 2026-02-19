"""
Create labelmap.txt file for Horizon-HUD detection classes.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.utils.class_mapping import create_labelmap_file


def main():
    parser = argparse.ArgumentParser(
        description="Create labelmap.txt for Horizon-HUD classes"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="labels/horizon_labelmap.txt",
        help="Output path for labelmap.txt"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_labelmap_file(str(output_path))
    print(f"Labelmap created at {output_path}")
    print("\nClasses:")
    print("  0: vehicle")
    print("  1: pedestrian")
    print("  2: cyclist")
    print("  3: road_obstacle")


if __name__ == "__main__":
    main()
