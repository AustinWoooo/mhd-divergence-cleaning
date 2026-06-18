#!/usr/bin/env python3
"""Run the all-methods performance sweep for each report benchmark problem."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import run_performance_scaling as perf


DEFAULT_PROBLEMS = ["divergence_advection", "field_loop", "orszag_tang"]


def per_problem_csv(problem: str) -> Path:
    return perf.PERF_DIR / f"performance_scaling_all_methods_{problem}.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", nargs="+", default=DEFAULT_PROBLEMS)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--methods", nargs="+", default=perf.METHOD_ORDER)
    parser.add_argument("--reconstruction", default="plm", choices=["pcm", "plm"])
    parser.add_argument("--limiter", default="mc", choices=["minmod", "vanleer", "mc"])
    parser.add_argument("--tfinal", type=float, default=0.05)
    parser.add_argument("--diagnostic-stride", type=int, default=100)
    parser.add_argument("--glm-ch-factor", type=float, default=4.0)
    parser.add_argument("--glm-cd", type=float, default=None)
    parser.add_argument("--glm-cr", type=float, default=None)
    parser.add_argument("--glm-subcycles", type=int, default=1)
    parser.add_argument("--runner", default=str(perf.DEFAULT_RUNNER))
    parser.add_argument(
        "--output-csv",
        default=str(perf.PERF_DIR / "performance_scaling_all_methods.csv"),
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop on the first failed benchmark command instead of preserving failure rows.",
    )
    args = parser.parse_args()

    script = Path(__file__).resolve().with_name("run_performance_scaling.py")
    merged_rows: list[dict[str, str]] = []

    for index, problem in enumerate(args.problems):
        output_csv = per_problem_csv(problem)
        cmd = [
            "python3",
            str(script),
            "--problem",
            problem,
            "--resolutions",
            *[str(n) for n in args.resolutions],
            "--methods",
            *args.methods,
            "--reconstruction",
            args.reconstruction,
            "--limiter",
            args.limiter,
            "--tfinal",
            str(args.tfinal),
            "--diagnostic-stride",
            str(args.diagnostic_stride),
            "--glm-ch-factor",
            str(args.glm_ch_factor),
            "--glm-subcycles",
            str(args.glm_subcycles),
            "--runner",
            args.runner,
            "--output-csv",
            str(output_csv),
            "--figure-suffix",
            f"all_methods_{problem}",
        ]
        if args.glm_cd is not None:
            cmd.extend(["--glm-cd", str(args.glm_cd)])
        if args.glm_cr is not None:
            cmd.extend(["--glm-cr", str(args.glm_cr)])
        if args.dry_run:
            cmd.append("--dry-run")
        if args.skip_build or index > 0:
            cmd.append("--skip-build")
        if not args.stop_on_failure:
            cmd.append("--continue-on-failure")

        print(" ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=perf.ROOT, check=False)
        if result.returncode != 0:
            message = f"{problem} performance sweep exited with code {result.returncode}"
            if args.stop_on_failure:
                raise SystemExit(message)
            print(f"WARNING: {message}")

        if args.dry_run:
            continue
        if not output_csv.exists():
            message = f"missing per-problem performance CSV {output_csv}"
            if args.stop_on_failure:
                raise SystemExit(message)
            print(f"WARNING: {message}")
            continue
        merged_rows.extend(
            perf.keep_current_methods(read_rows(output_csv), f"{problem} performance CSV")
        )

    if args.dry_run:
        return 0
    if not merged_rows:
        raise SystemExit("no performance rows were collected")
    merged_rows = perf.keep_current_methods(merged_rows, "merged performance CSV")
    if not merged_rows:
        raise SystemExit("no current-method performance rows were collected")

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = perf.ROOT / output_csv
    perf.write_csv(merged_rows, output_csv)
    perf.plot_walltime(merged_rows, "all_methods")
    perf.plot_throughput(merged_rows, "all_methods")
    perf.plot_seconds_per_step(merged_rows, "all_methods")
    perf.plot_breakdown(merged_rows, "all_methods")
    perf.plot_breakdowns_by_resolution(merged_rows, "all_methods")
    perf.plot_breakdown_normalized(merged_rows, "all_methods")
    perf.plot_normalized_breakdowns_by_resolution(merged_rows, "all_methods")
    perf.plot_cleaning_overhead(merged_rows, "all_methods")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
