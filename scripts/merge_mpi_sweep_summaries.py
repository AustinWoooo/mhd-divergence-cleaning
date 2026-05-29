#!/usr/bin/env python3
"""Merge rank-local MPI sweep summaries into one CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-dir",
        default="results/mhd_sweep_mpi/summaries",
        help="directory containing summary_rank_*.csv files",
    )
    parser.add_argument(
        "--output",
        default="results/mhd_sweep_mpi/mpi_sweep_summary.csv",
    )
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    files = sorted(summary_dir.glob("summary_rank_*.csv"))
    if not files:
        raise FileNotFoundError(f"no rank summaries found in {summary_dir}")

    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in files:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            rows.extend(reader)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
