#!/usr/bin/env python3
"""Validate performance-scaling CSVs and generated figures."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys


REQUIRED_COLUMNS = {
    "problem",
    "method",
    "reconstruction",
    "limiter",
    "glm_ch_factor",
    "glm_cd",
    "glm_cr",
    "glm_subcycles",
    "glm_ch",
    "glm_cp",
    "glm_effective_cd",
    "glm_effective_cr",
    "nx",
    "ny",
    "ncell",
    "tfinal",
    "steps",
    "status",
    "total_wall_time_sec",
    "initialization_time_sec",
    "hydro_time_sec",
    "cleaning_time_sec",
    "diagnostics_compute_time_sec",
    "diagnostics_write_time_sec",
    "snapshot_write_time_sec",
    "summary_write_time_sec",
    "output_time_sec",
    "total_cell_updates",
    "seconds_per_step",
    "cell_updates_per_second",
    "cleaning_subcycles_total",
    "projection_iterations_total",
    "final_L2_fv",
    "final_Linf_fv",
    "final_L2_norm_fv",
    "final_Linf_norm_fv",
    "peak_L2_norm_fv",
    "peak_Linf_norm_fv",
    "time_integrated_L2_norm_fv",
    "min_pressure",
    "min_density",
    "has_nonfinite",
    "has_negative_density",
    "has_negative_pressure",
    "energy_drift",
    "failure_reason",
    "summary_file",
    "diagnostic_file",
}

DEFAULT_FIGURE_STEMS = [
    "performance_walltime_vs_cells",
    "performance_cell_updates_per_second",
    "performance_seconds_per_step",
    "performance_method_breakdown",
    "performance_method_breakdown_normalized",
    "performance_cleaning_overhead_fraction",
]


def fail(message: str) -> None:
    print(f"performance scaling check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def finite_float(row: dict[str, str], column: str, path: Path) -> float:
    try:
        value = float(row[column])
    except (KeyError, ValueError) as exc:
        fail(f"{path} has invalid {column}: {exc}")
    if not math.isfinite(value):
        fail(f"{path} has nonfinite {column}: {value}")
    return value


def positive_int(row: dict[str, str], column: str, path: Path) -> int:
    value = finite_float(row, column, path)
    integer = int(round(value))
    if integer <= 0:
        fail(f"{path} has non-positive {column}: {value}")
    return integer


def infer_figure_suffix(csv_path: Path) -> str:
    if csv_path.stem == "performance_scaling_all_methods":
        return "all_methods"
    return ""


def figure_path(stem: str, suffix: str = "", resolution: int | None = None) -> str:
    clean_suffix = suffix.strip("_")
    parts = ["figures/mhd_runner/performance/" + stem]
    if clean_suffix:
        parts.append(clean_suffix)
    if resolution is not None:
        parts.append(str(resolution))
    return "_".join(parts) + ".png"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="results/mhd_runner/performance/performance_scaling.csv",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Do not fail if a benchmark row has status != finished.",
    )
    parser.add_argument(
        "--figures",
        nargs="*",
        default=None,
        help="Figure files expected to exist.",
    )
    parser.add_argument(
        "--figure-suffix",
        default=None,
        help="Expected figure suffix. Defaults to inferring from the CSV filename.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        fail(f"missing {csv_path}")

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            fail(f"{csv_path} missing columns: {', '.join(sorted(missing))}")

    if not rows:
        fail(f"{csv_path} has no data rows")

    resolutions = set()
    for row in rows:
        nx = positive_int(row, "nx", csv_path)
        ny = positive_int(row, "ny", csv_path)
        ncell = positive_int(row, "ncell", csv_path)
        steps = positive_int(row, "steps", csv_path)
        total_cell_updates = positive_int(row, "total_cell_updates", csv_path)
        if nx == ny:
            resolutions.add(nx)

        if ncell != nx * ny:
            fail(f"ncell={ncell} but nx*ny={nx * ny}")
        if total_cell_updates != ncell * steps:
            fail(
                "total_cell_updates="
                f"{total_cell_updates} but ncell*steps={ncell * steps}"
            )

        for column in (
            "total_wall_time_sec",
            "seconds_per_step",
            "cell_updates_per_second",
        ):
            if finite_float(row, column, csv_path) <= 0.0:
                fail(f"{column} must be positive")

        for column in (
            "initialization_time_sec",
            "hydro_time_sec",
            "cleaning_time_sec",
            "diagnostics_compute_time_sec",
            "diagnostics_write_time_sec",
            "snapshot_write_time_sec",
            "summary_write_time_sec",
            "output_time_sec",
        ):
            if finite_float(row, column, csv_path) < 0.0:
                fail(f"{column} must be non-negative")

        if not args.allow_failures and row["status"] != "finished":
            fail(
                f"unexpected benchmark failure for {row['problem']} "
                f"{row['method']} N={nx}: status={row['status']}"
            )

    suffix = infer_figure_suffix(csv_path) if args.figure_suffix is None else args.figure_suffix
    expected_figures = (
        list(args.figures)
        if args.figures is not None
        else [figure_path(stem, suffix) for stem in DEFAULT_FIGURE_STEMS]
    )
    for n in sorted(resolutions):
        expected_figures.extend(
            [
                figure_path("performance_method_breakdown", suffix, n),
                figure_path("performance_method_breakdown_normalized", suffix, n),
            ]
        )

    for figure in expected_figures:
        path = Path(figure)
        if not path.exists():
            fail(f"missing figure {path}")
        if path.stat().st_size <= 0:
            fail(f"empty figure {path}")

    print(f"performance scaling check passed ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
