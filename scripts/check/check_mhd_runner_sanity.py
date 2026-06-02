#!/usr/bin/env python3
"""Small scientific sanity checks for mhd_runner smoke outputs.

This script intentionally checks only short mhd_runner smoke outputs. It is not
a performance benchmark and does not replace full result generation.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys


MAIN_METHODS = [
    "none",
    "hyperbolic_glm",
    "mixed_glm",
    "parabolic",
    "elliptic_projection",
    "powell_source",
    "mixed_eglm",
    "gi_mixed_eglm",
]

REQUIRED_DIVERGENCE_COLUMNS = {
    "step",
    "time",
    "dt",
    "L1_fv",
    "L2_fv",
    "Linf_fv",
    "L1_norm_fv",
    "L2_norm_fv",
    "Linf_norm_fv",
    "total_energy",
    "min_density",
    "min_pressure",
    "has_nonfinite",
    "has_negative_density",
    "has_negative_pressure",
}

REQUIRED_SUMMARY_COLUMNS = {
    "problem",
    "method",
    "status",
    "final_time_reached",
    "final_L2_fv",
    "min_pressure",
    "min_density",
}


def fail(message: str) -> None:
    print(f"sanity check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        fail(f"missing {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        fail(f"{path} has no data rows")
    return rows


def finite_float(row: dict[str, str], column: str, path: Path) -> float:
    try:
        value = float(row[column])
    except (KeyError, ValueError) as exc:
        fail(f"{path} has nonnumeric {column}: {exc}")
    if not math.isfinite(value):
        fail(f"{path} has nonfinite {column}: {value}")
    return value


def final_value(rows: list[dict[str, str]], column: str, path: Path) -> float:
    return finite_float(rows[-1], column, path)


def check_required_columns(path: Path, rows: list[dict[str, str]], required: set[str]) -> None:
    missing = required.difference(rows[0].keys())
    if missing:
        fail(f"{path} missing columns: {', '.join(sorted(missing))}")


def check_method_outputs(results_dir: Path, prefix: str, method: str) -> float:
    div_path = results_dir / "divergence" / f"{prefix}_{method}.csv"
    summary_path = results_dir / "summaries" / f"{prefix}_{method}_summary.csv"

    div_rows = read_rows(div_path)
    check_required_columns(div_path, div_rows, REQUIRED_DIVERGENCE_COLUMNS)

    for row in div_rows:
        for column in ("time", "L2_norm_fv", "min_pressure", "min_density"):
            finite_float(row, column, div_path)

    final_l2_norm = final_value(div_rows, "L2_norm_fv", div_path)
    final_min_p = final_value(div_rows, "min_pressure", div_path)
    final_min_rho = final_value(div_rows, "min_density", div_path)
    if final_min_p <= 0.0:
        fail(f"{div_path} ended with non-positive pressure: {final_min_p}")
    if final_min_rho <= 0.0:
        fail(f"{div_path} ended with non-positive density: {final_min_rho}")

    if any(int(float(row["has_nonfinite"])) != 0 for row in div_rows):
        fail(f"{div_path} reported nonfinite state")
    if any(int(float(row["has_negative_density"])) != 0 for row in div_rows):
        fail(f"{div_path} reported negative density")
    if any(int(float(row["has_negative_pressure"])) != 0 for row in div_rows):
        fail(f"{div_path} reported negative pressure")

    summary_rows = read_rows(summary_path)
    check_required_columns(summary_path, summary_rows, REQUIRED_SUMMARY_COLUMNS)
    summary = summary_rows[-1]
    if summary["method"] != method:
        fail(f"{summary_path} method column is {summary['method']}, expected {method}")
    if summary["status"] != "finished":
        fail(f"{summary_path} status is {summary['status']}, expected finished")
    if int(float(summary["final_time_reached"])) != 1:
        fail(f"{summary_path} did not reach final time")

    return final_l2_norm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/mhd_runner")
    parser.add_argument("--prefix", default="mhd_ot_smoke")
    parser.add_argument("--methods", nargs="*", default=MAIN_METHODS)
    parser.add_argument(
        "--check-reduction",
        action="store_true",
        help=(
            "Require mixed_glm and elliptic_projection to reduce final "
            "normalized FV divB relative to none. Use this on a "
            "divergence-contaminated smoke problem, not necessarily on "
            "very short Orszag-Tang smoke output."
        ),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    final_l2_norm: dict[str, float] = {}
    for method in args.methods:
        final_l2_norm[method] = check_method_outputs(results_dir, args.prefix, method)

    if args.check_reduction:
        for method in ("none", "mixed_glm", "elliptic_projection"):
            if method not in final_l2_norm:
                fail(f"--check-reduction requires {method} output")

        none = final_l2_norm["none"]
        mixed = final_l2_norm["mixed_glm"]
        projection = final_l2_norm["elliptic_projection"]

        if not mixed < none:
            fail(f"mixed_glm final L2_norm_fv={mixed:g} is not below none={none:g}")
        if not projection < 0.1 * none:
            fail(
                "elliptic_projection final L2_norm_fv="
                f"{projection:g} is not at least 10x below none={none:g}"
            )

    for method in ("mixed_eglm", "gi_mixed_eglm"):
        value = final_l2_norm.get(method)
        if value is not None and not math.isfinite(value):
            fail(f"{method} final L2_norm_fv is nonfinite")

    print("mhd_runner smoke sanity checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
