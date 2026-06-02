#!/usr/bin/env python3
"""Validate and rank GLM-family parameter sweep CSV output."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys


REQUIRED_COLUMNS = {
    "problem",
    "method",
    "nx",
    "reconstruction",
    "limiter",
    "glm_ch_factor",
    "glm_cd",
    "glm_cr",
    "glm_subcycles",
    "glm_ch",
    "glm_cp",
    "final_L2_norm_fv",
    "peak_L2_norm_fv",
    "time_integrated_L2_norm_fv",
    "final_Linf_norm_fv",
    "peak_Linf_norm_fv",
    "min_pressure",
    "min_density",
    "energy_drift",
    "total_wall_time_sec",
    "seconds_per_step",
    "cell_updates_per_second",
    "status",
    "failure_reason",
}


def fail(message: str) -> None:
    print(f"GLM parameter sweep check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def to_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def flag_value(row: dict[str, str], key: str) -> int:
    value = to_float(row, key, 0.0)
    if not math.isfinite(value):
        return 0
    return int(round(value))


def is_robust(row: dict[str, str]) -> bool:
    if row.get("status", "") != "finished":
        return False
    if flag_value(row, "final_time_reached") != 1:
        return False
    if flag_value(row, "has_nonfinite") != 0:
        return False
    if flag_value(row, "has_negative_density") != 0:
        return False
    if flag_value(row, "has_negative_pressure") != 0:
        return False
    if to_float(row, "min_pressure") <= 0.0:
        return False
    if to_float(row, "min_density") <= 0.0:
        return False
    return True


def finite_or_inf(row: dict[str, str], key: str) -> float:
    value = to_float(row, key)
    return value if math.isfinite(value) else math.inf


def rank_key(row: dict[str, str], max_energy_drift: float) -> tuple[object, ...]:
    energy_drift = abs(finite_or_inf(row, "energy_drift"))
    return (
        not is_robust(row),
        finite_or_inf(row, "final_L2_norm_fv"),
        finite_or_inf(row, "time_integrated_L2_norm_fv"),
        energy_drift > max_energy_drift,
        energy_drift,
        finite_or_inf(row, "total_wall_time_sec"),
    )


def format_optional(row: dict[str, str], key: str) -> str:
    value = to_float(row, key)
    if not math.isfinite(value):
        return "unused"
    return f"{value:g}"


def describe(row: dict[str, str]) -> str:
    return (
        f"ch_factor={format_optional(row, 'glm_ch_factor')}, "
        f"cd={format_optional(row, 'glm_cd')}, "
        f"cr={format_optional(row, 'glm_cr')}, "
        f"subcycles={format_optional(row, 'glm_subcycles')}, "
        f"ch={format_optional(row, 'glm_ch')}, "
        f"cp={format_optional(row, 'glm_cp')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="results/mhd_runner/glm_sweep/glm_parameter_sweep.csv",
    )
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--method", default=None)
    parser.add_argument(
        "--max-energy-drift",
        type=float,
        default=1.0e-8,
        help="Rows above this absolute drift are still shown but rank after comparable low-drift rows.",
    )
    parser.add_argument(
        "--require-robust",
        action="store_true",
        help="Fail if no robust row exists for any reported method.",
    )
    args = parser.parse_args()

    if args.top <= 0:
        fail("--top must be positive")
    if args.max_energy_drift < 0.0:
        fail("--max-energy-drift must be non-negative")

    path = Path(args.csv_path)
    if not path.exists():
        fail(f"missing {path}")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            fail(f"{path} missing columns: {', '.join(sorted(missing))}")

    if not rows:
        fail(f"{path} has no data rows")

    if args.method is not None:
        rows = [row for row in rows if row.get("method") == args.method]
        if not rows:
            fail(f"{path} has no rows for method={args.method}")

    by_method: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    for method in sorted(by_method):
        ranked = sorted(by_method[method], key=lambda row: rank_key(row, args.max_energy_drift))
        robust_count = sum(1 for row in ranked if is_robust(row))
        if args.require_robust and robust_count == 0:
            fail(f"no robust parameter setting for {method}")

        print(f"best parameter setting for {method} ({robust_count}/{len(ranked)} robust rows):")
        for row in ranked[: args.top]:
            robust = "yes" if is_robust(row) else "no"
            print(
                "  "
                f"N={row['nx']}, robust={robust}, "
                f"final_L2_norm_fv={format_optional(row, 'final_L2_norm_fv')}, "
                f"time_integrated_L2_norm_fv={format_optional(row, 'time_integrated_L2_norm_fv')}, "
                f"energy_drift={format_optional(row, 'energy_drift')}, "
                f"wall={format_optional(row, 'total_wall_time_sec')}s, "
                f"{describe(row)}"
            )

    print(f"checked {len(rows)} GLM sweep rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
