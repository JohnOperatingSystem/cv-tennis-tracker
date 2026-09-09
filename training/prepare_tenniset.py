"""Export compact TenniSet manifests and a weak-label review queue."""

import argparse
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tenniset import TenniSetAnnotations  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_DIR / "external" / "tenniset" / "annotations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "external" / "tenniset" / "manifests",
    )
    parser.add_argument("--overrides", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = TenniSetAnnotations(args.annotations, args.overrides)
    paths = dataset.export(args.output)
    print(f"Exported {len(dataset.events)} temporal events")
    print(f"Exported {len(dataset.points)} points")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
