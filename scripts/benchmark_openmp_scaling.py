#!/usr/bin/env python3
"""Benchmark OpenMP strong scaling for the 2D MHD runner."""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"
DEFAULT_RUNNER = BUILD_DIR / "mhd_runner_cli"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "openmp_scaling_runs"
DEFAULT_CSV = ROOT / "results" / "openmp_scaling.csv"
DEFAULT_RAW_CSV = ROOT / "results" / "openmp_scaling_raw.csv"
DEFAULT_FIG_DIR = ROOT / "figures" / "openmp_scaling"
DEFAULT_REPORT = ROOT / "results" / "openmp_scaling_report.md"

CSV_FIELDS = [
    "problem",
    "method",
    "reconstruction",
    "limiter",
    "nx",
    "ny",
    "ncell",
    "tfinal",
    "threads",
    "repeat_count",
    "runtime_statistic",
    "selected_solver_time_sec",
    "best_solver_time_sec",
    "median_solver_time_sec",
    "selected_total_wall_time_sec",
    "hydro_time_sec",
    "cleaning_time_sec",
    "iteration_count",
    "time_per_iteration_sec",
    "final_residual_L2_fv",
    "final_residual_Linf_fv",
    "energy_drift",
    "speedup",
    "parallel_efficiency",
    "residual_abs_diff_from_1",
    "residual_rel_diff_from_1",
    "correctness_ok",
    "status",
    "summary_file",
]

RAW_FIELDS = [
    "problem",
    "method",
    "reconstruction",
    "limiter",
    "nx",
    "ny",
    "ncell",
    "tfinal",
    "threads",
    "repeat",
    "solver_time_sec",
    "total_wall_time_sec",
    "hydro_time_sec",
    "cleaning_time_sec",
    "iteration_count",
    "time_per_iteration_sec",
    "final_residual_L2_fv",
    "final_residual_Linf_fv",
    "energy_drift",
    "status",
    "summary_file",
]


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print(" ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-80:])
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {' '.join(cmd)}\n{tail}"
        )
    return result


def ensure_build(runner: Path, skip_build: bool) -> None:
    if skip_build and runner.exists():
        return
    if not (BUILD_DIR / "CMakeCache.txt").exists():
        run(["cmake", "-S", ".", "-B", "build"])
    run(["cmake", "--build", "build", "--target", runner.name, "--parallel"])


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = as_float(row, key, float(default))
    if not math.isfinite(value):
        return default
    return int(round(value))


def read_summary(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"summary CSV has no data rows: {path}")
    return rows[-1]


def write_csv(rows: list[dict[str, object]], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def benchmark_one(
    args: argparse.Namespace,
    runner: Path,
    output_root: Path,
    n: int,
    threads: int,
    repeat: int,
) -> dict[str, object]:
    prefix = f"openmp_{args.problem}_{args.method}_n{n}_t{threads}_r{repeat}"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_DYNAMIC"] = "FALSE"
    env.setdefault("OMP_PROC_BIND", "close")
    env.setdefault("OMP_PLACES", "cores")

    cmd = [
        str(runner),
        "--performance-mode",
        "--output-root",
        str(output_root),
        "--output-prefix",
        prefix,
        "--nx",
        str(n),
        "--ny",
        str(n),
        "--tfinal",
        str(args.tfinal),
        "--diagnostic-stride",
        str(args.diagnostic_stride),
        "--reconstruction",
        args.reconstruction,
        "--limiter",
        args.limiter,
        args.problem,
        args.method,
    ]
    run(cmd, env=env)

    summary_file = output_root / "summaries" / f"{prefix}_{args.method}_summary.csv"
    if not summary_file.exists():
        raise FileNotFoundError(f"missing runner summary: {summary_file}")

    summary = read_summary(summary_file)
    hydro = as_float(summary, "hydro_time_sec")
    cleaning = as_float(summary, "cleaning_time_sec", 0.0)
    solver_time = hydro + cleaning
    steps = as_int(summary, "steps")
    return {
        "problem": args.problem,
        "method": args.method,
        "reconstruction": args.reconstruction,
        "limiter": args.limiter,
        "nx": as_int(summary, "nx", n),
        "ny": as_int(summary, "ny", n),
        "ncell": as_int(summary, "ncell", n * n),
        "tfinal": args.tfinal,
        "threads": threads,
        "repeat": repeat,
        "solver_time_sec": solver_time,
        "total_wall_time_sec": as_float(summary, "total_wall_time_sec"),
        "hydro_time_sec": hydro,
        "cleaning_time_sec": cleaning,
        "iteration_count": steps,
        "time_per_iteration_sec": solver_time / steps if steps > 0 else math.nan,
        "final_residual_L2_fv": as_float(summary, "final_L2_fv"),
        "final_residual_Linf_fv": as_float(summary, "final_Linf_fv"),
        "energy_drift": as_float(summary, "energy_drift"),
        "status": summary.get("status", ""),
        "summary_file": str(summary_file.relative_to(ROOT)),
    }


def select_run(rows: list[dict[str, object]], statistic: str) -> dict[str, object]:
    times = [float(row["solver_time_sec"]) for row in rows]
    selected_time = min(times) if statistic == "best" else statistics.median(times)
    return min(rows, key=lambda row: abs(float(row["solver_time_sec"]) - selected_time))


def aggregate_rows(
    raw_rows: list[dict[str, object]],
    statistic: str,
    residual_rtol: float,
    residual_atol: float,
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in raw_rows:
        grouped.setdefault((int(row["nx"]), int(row["threads"])), []).append(row)

    aggregated: list[dict[str, object]] = []
    for (n, threads), rows in sorted(grouped.items()):
        selected = select_run(rows, statistic)
        times = [float(row["solver_time_sec"]) for row in rows]
        steps = int(selected["iteration_count"])
        selected_time = float(selected["solver_time_sec"])
        aggregated.append(
            {
                **{key: selected[key] for key in [
                    "problem",
                    "method",
                    "reconstruction",
                    "limiter",
                    "nx",
                    "ny",
                    "ncell",
                    "tfinal",
                    "threads",
                    "hydro_time_sec",
                    "cleaning_time_sec",
                    "iteration_count",
                    "final_residual_L2_fv",
                    "final_residual_Linf_fv",
                    "energy_drift",
                    "status",
                    "summary_file",
                ]},
                "repeat_count": len(rows),
                "runtime_statistic": statistic,
                "selected_solver_time_sec": selected_time,
                "best_solver_time_sec": min(times),
                "median_solver_time_sec": statistics.median(times),
                "selected_total_wall_time_sec": selected["total_wall_time_sec"],
                "time_per_iteration_sec": selected_time / steps if steps > 0 else math.nan,
            }
        )

    baseline_by_n = {
        int(row["nx"]): float(row["selected_solver_time_sec"])
        for row in aggregated
        if int(row["threads"]) == 1
    }
    residual_by_n = {
        int(row["nx"]): float(row["final_residual_L2_fv"])
        for row in aggregated
        if int(row["threads"]) == 1
    }

    for row in aggregated:
        n = int(row["nx"])
        threads = int(row["threads"])
        t1 = baseline_by_n.get(n, math.nan)
        speedup = t1 / float(row["selected_solver_time_sec"]) if math.isfinite(t1) else math.nan
        residual = float(row["final_residual_L2_fv"])
        base_residual = residual_by_n.get(n, math.nan)
        abs_diff = abs(residual - base_residual) if math.isfinite(base_residual) else math.nan
        scale = max(abs(base_residual), residual_atol) if math.isfinite(base_residual) else math.nan
        rel_diff = abs_diff / scale if math.isfinite(scale) and scale > 0.0 else math.nan
        ok = (
            math.isfinite(abs_diff)
            and (abs_diff <= residual_atol or rel_diff <= residual_rtol)
        )
        row["speedup"] = speedup
        row["parallel_efficiency"] = speedup / threads if math.isfinite(speedup) else math.nan
        row["residual_abs_diff_from_1"] = abs_diff
        row["residual_rel_diff_from_1"] = rel_diff
        row["correctness_ok"] = int(ok)

    return aggregated


def rows_by_size(rows: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    out: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        out.setdefault(int(row["nx"]), []).append(row)
    for grouped in out.values():
        grouped.sort(key=lambda row: int(row["threads"]))
    return dict(sorted(out.items()))


def plot_metric(
    rows: list[dict[str, object]],
    fig_dir: Path,
    filename: str,
    y_key: str,
    ylabel: str,
    *,
    ideal: bool = False,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4.8))
    max_thread = 1
    for n, grouped in rows_by_size(rows).items():
        threads = [int(row["threads"]) for row in grouped]
        values = [float(row[y_key]) for row in grouped]
        max_thread = max(max_thread, max(threads))
        plt.plot(threads, values, marker="o", label=f"N={n}")
    if ideal:
        xs = sorted({int(row["threads"]) for row in rows})
        if xs:
            plt.plot(xs, xs, "k--", linewidth=1.0, label="ideal linear")
    plt.xlabel("OpenMP threads")
    plt.ylabel(ylabel)
    plt.xticks(sorted({int(row["threads"]) for row in rows}))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    for suffix in ("png", "pdf"):
        plt.savefig(fig_dir / f"{filename}.{suffix}", dpi=180)
    plt.close()


def make_plots(rows: list[dict[str, object]], fig_dir: Path) -> None:
    plot_metric(
        rows,
        fig_dir,
        "openmp_runtime_vs_threads",
        "selected_solver_time_sec",
        "solver time (s)",
    )
    plot_metric(
        rows,
        fig_dir,
        "openmp_speedup_vs_threads",
        "speedup",
        "speedup",
        ideal=True,
    )
    plot_metric(
        rows,
        fig_dir,
        "openmp_efficiency_vs_threads",
        "parallel_efficiency",
        "parallel efficiency",
    )
    plot_metric(
        rows,
        fig_dir,
        "openmp_time_per_iteration_vs_threads",
        "time_per_iteration_sec",
        "solver time per iteration (s)",
    )


def write_report(rows: list[dict[str, object]], path: Path, fig_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sizes = sorted({int(row["nx"]) for row in rows})
    threads = sorted({int(row["threads"]) for row in rows})
    best_by_size = []
    for n, grouped in rows_by_size(rows).items():
        best = max(grouped, key=lambda row: float(row["speedup"]))
        best_by_size.append(
            f"N={n}: {float(best['speedup']):.2f}x at {int(best['threads'])} threads"
        )
    all_ok = all(int(row["correctness_ok"]) == 1 for row in rows)
    correctness = (
        "The final divergence residuals matched the one-thread baseline within the configured tolerance."
        if all_ok
        else "Some final divergence residuals differed from the one-thread baseline beyond the configured tolerance; inspect the CSV before using those rows."
    )
    paragraph = (
        "OpenMP tests intra-node shared-memory parallelism by running the same "
        f"{rows[0]['problem']} / {rows[0]['method']} solver workload on thread "
        f"counts {threads} for grid sizes {sizes}. The ideal strong-scaling "
        "speedup is linear in the number of threads, but real scaling is limited "
        "by memory bandwidth, synchronization and barrier overhead, reduction "
        "overhead in diagnostics, cache effects, and serial portions of the code. "
        "In this run, the best measured speedups were "
        + "; ".join(best_by_size)
        + ". Larger grids usually scale better because each thread receives more "
        "stencil and Riemann-solver work relative to fixed OpenMP overhead, while "
        "parallel efficiency usually decreases at high thread counts as those "
        "overheads and bandwidth limits become more important. "
        + correctness
    )
    with path.open("w") as handle:
        handle.write("# OpenMP Scaling Result\n\n")
        handle.write(paragraph + "\n\n")
        handle.write("Figures:\n")
        for name in [
            "openmp_runtime_vs_threads.png",
            "openmp_speedup_vs_threads.png",
            "openmp_efficiency_vs_threads.png",
            "openmp_time_per_iteration_vs_threads.png",
        ]:
            handle.write(f"- `{fig_dir.relative_to(ROOT) / name}`\n")


def positive_ints(values: list[str]) -> list[int]:
    out = [int(value) for value in values]
    if any(value <= 0 for value in out):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", default=["128", "256", "512"])
    parser.add_argument("--threads", nargs="+", default=["1", "2", "4", "8", "16", "32"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--statistic", choices=["median", "best"], default="median")
    parser.add_argument("--problem", default="divergence_advection")
    parser.add_argument("--method", default="none")
    parser.add_argument("--reconstruction", default="plm", choices=["pcm", "plm"])
    parser.add_argument("--limiter", default="mc", choices=["minmod", "vanleer", "mc"])
    parser.add_argument("--tfinal", type=float, default=0.02)
    parser.add_argument("--diagnostic-stride", type=int, default=1000000)
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-csv", default=str(DEFAULT_CSV))
    parser.add_argument("--raw-output-csv", default=str(DEFAULT_RAW_CSV))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIG_DIR))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT))
    parser.add_argument("--residual-rtol", type=float, default=1.0e-10)
    parser.add_argument("--residual-atol", type=float, default=1.0e-12)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    sizes = positive_ints(args.sizes)
    requested_threads = positive_ints(args.threads)
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.tfinal <= 0.0:
        raise SystemExit("--tfinal must be positive")

    cpu_count = available_cpus()
    threads = [thread for thread in requested_threads if thread <= cpu_count]
    skipped = [thread for thread in requested_threads if thread > cpu_count]
    if not threads:
        raise SystemExit(f"no requested thread counts fit available CPUs ({cpu_count})")
    if skipped:
        print(f"Skipping thread counts above available CPUs ({cpu_count}): {skipped}")

    runner = Path(args.runner)
    if not runner.is_absolute():
        runner = ROOT / runner
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    ensure_build(runner, args.skip_build)

    raw_rows: list[dict[str, object]] = []
    for n in sizes:
        for thread in threads:
            for repeat in range(1, args.repeats + 1):
                raw_rows.append(benchmark_one(args, runner, output_root, n, thread, repeat))

    aggregate = aggregate_rows(
        raw_rows,
        args.statistic,
        args.residual_rtol,
        args.residual_atol,
    )

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = ROOT / output_csv
    raw_csv = Path(args.raw_output_csv)
    if not raw_csv.is_absolute():
        raw_csv = ROOT / raw_csv
    fig_dir = Path(args.figure_dir)
    if not fig_dir.is_absolute():
        fig_dir = ROOT / fig_dir
    report_md = Path(args.report_md)
    if not report_md.is_absolute():
        report_md = ROOT / report_md

    write_csv(raw_rows, raw_csv, RAW_FIELDS)
    write_csv(aggregate, output_csv, CSV_FIELDS)
    make_plots(aggregate, fig_dir)
    write_report(aggregate, report_md, fig_dir)

    print(f"Wrote {output_csv.relative_to(ROOT)}")
    print(f"Wrote {raw_csv.relative_to(ROOT)}")
    print(f"Wrote figures in {fig_dir.relative_to(ROOT)}")
    print(f"Wrote {report_md.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
