#!/usr/bin/env python3
"""Validate rank-local summaries from the optional MPI sweep runner."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys


REQUIRED_COLUMNS = {
    "method",
    "nx",
    "ny",
    "rank",
    "runtime_sec",
    "status",
    "final_divB_L2",
    "final_divB_Linf",
    "min_pressure",
    "energy_drift",
}


def fail(message: str) -> None:
    print(f"MPI sweep summary check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def finite_float(row: dict[str, str], column: str, source: Path) -> float:
    try:
        value = float(row[column])
    except (KeyError, ValueError) as exc:
        fail(f"{source} has invalid {column}: {exc}")
    if not math.isfinite(value):
        fail(f"{source} has nonfinite {column}: {value}")
    return value


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            fail(f"{path} missing columns: {', '.join(sorted(missing))}")
        return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-dir",
        default="results/mhd_sweep_mpi/summaries",
        help="directory containing summary_rank_*.csv files",
    )
    parser.add_argument("--min-rows", type=int, default=1)
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    files = sorted(summary_dir.glob("summary_rank_*.csv"))
    if not files:
        fail(f"no rank summaries found in {summary_dir}")

    total_rows = 0
    for path in files:
        rows = read_rows(path)
        total_rows += len(rows)
        for row in rows:
            if row["status"] != "finished":
                fail(f"{path} has non-finished job status: {row['status']}")
            for column in (
                "runtime_sec",
                "final_divB_L2",
                "final_divB_Linf",
                "min_pressure",
                "energy_drift",
            ):
                finite_float(row, column, path)
            if finite_float(row, "min_pressure", path) <= 0.0:
                fail(f"{path} has non-positive min_pressure")

    if total_rows < args.min_rows:
        fail(f"expected at least {args.min_rows} rows, found {total_rows}")

    print(f"MPI sweep summary check passed ({total_rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
