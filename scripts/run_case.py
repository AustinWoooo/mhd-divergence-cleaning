#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="test_glm_1d",
        help="CMake target to build and run",
    )
    parser.add_argument(
        "--plot-glm-1d",
        action="store_true",
        help="Run scripts/plot_glm_1d.py after the executable",
    )
    args = parser.parse_args()

    subprocess.run(
        ["cmake", "--build", "build", "--target", args.target],
        cwd=ROOT,
        check=True,
    )

    exe = ROOT / "build" / args.target
    subprocess.run([str(exe)], cwd=ROOT, check=True)

    if args.plot_glm_1d:
        subprocess.run(
            ["python3", "scripts/plot_glm_1d.py"],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
