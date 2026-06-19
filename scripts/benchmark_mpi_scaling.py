#!/usr/bin/env python3
"""Benchmark MPI strong scaling for the 2D MHD runner.

Companion to benchmark_openmp_scaling.py.  Where the OpenMP benchmark varies the
number of shared-memory threads on a single process, this one varies the number
of MPI ranks: the same fixed-size simulation is split across a 2D Cartesian
domain decomposition (mhd_runner_mpi), so it measures distributed-memory strong
scaling -- including halo-exchange communication, collective reductions, and
load imbalance.

Each rank runs with OMP_NUM_THREADS=1 by default so the measurement is pure MPI
(not a mix of MPI and OpenMP).  Solver time is hydro_time_sec + cleaning_time_sec
from the rank-0 summary, which includes the halo exchanges performed inside each
RK / cleaning substep but excludes the gather-to-root diagnostic I/O.
"""

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
BUILD_DIR = ROOT / "build-mpi"
DEFAULT_RUNNER = BUILD_DIR / "mhd_runner_mpi"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "mpi_scaling_runs"
DEFAULT_CSV = ROOT / "results" / "mpi_scaling.csv"
DEFAULT_RAW_CSV = ROOT / "results" / "mpi_scaling_raw.csv"
DEFAULT_FIG_DIR = ROOT / "figures" / "mpi_scaling"
DEFAULT_REPORT = ROOT / "results" / "mpi_scaling_report.md"

CSV_FIELDS = [
    "problem",
    "method",
    "nx",
    "ny",
    "ncell",
    "tfinal",
    "ranks",
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
    "nx",
    "ny",
    "ncell",
    "tfinal",
    "ranks",
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
        run(["cmake", "-S", ".", "-B", BUILD_DIR.name, "-DENABLE_MPI=ON"])
    run(["cmake", "--build", BUILD_DIR.name, "--target", runner.name, "--parallel"])


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


def mpirun_command(args: argparse.Namespace, ranks: int) -> list[str]:
    cmd = [args.mpirun, "-np", str(ranks)]
    if args.oversubscribe:
        cmd.append("--oversubscribe")
    if args.mpirun_extra:
        cmd.extend(args.mpirun_extra.split())
    return cmd


def benchmark_one(
    args: argparse.Namespace,
    runner: Path,
    output_root: Path,
    n: int,
    ranks: int,
    repeat: int,
) -> dict[str, object]:
    prefix = f"mpi_{args.problem}_{args.method}_n{n}_p{ranks}_r{repeat}"

    # Pure-MPI measurement: one OpenMP thread per rank unless overridden.
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(args.threads_per_rank))
    env["OMP_DYNAMIC"] = "FALSE"

    cmd = mpirun_command(args, ranks) + [
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
        "nx": as_int(summary, "nx", n),
        "ny": as_int(summary, "ny", n),
        "ncell": as_int(summary, "ncell", n * n),
        "tfinal": args.tfinal,
        "ranks": ranks,
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
        grouped.setdefault((int(row["nx"]), int(row["ranks"])), []).append(row)

    aggregated: list[dict[str, object]] = []
    for (n, ranks), rows in sorted(grouped.items()):
        selected = select_run(rows, statistic)
        times = [float(row["solver_time_sec"]) for row in rows]
        steps = int(selected["iteration_count"])
        selected_time = float(selected["solver_time_sec"])
        aggregated.append(
            {
                **{key: selected[key] for key in [
                    "problem",
                    "method",
                    "nx",
                    "ny",
                    "ncell",
                    "tfinal",
                    "ranks",
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
        if int(row["ranks"]) == 1
    }
    residual_by_n = {
        int(row["nx"]): float(row["final_residual_L2_fv"])
        for row in aggregated
        if int(row["ranks"]) == 1
    }

    for row in aggregated:
        n = int(row["nx"])
        ranks = int(row["ranks"])
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
        row["parallel_efficiency"] = speedup / ranks if math.isfinite(speedup) else math.nan
        row["residual_abs_diff_from_1"] = abs_diff
        row["residual_rel_diff_from_1"] = rel_diff
        row["correctness_ok"] = int(ok)

    return aggregated


def rows_by_size(rows: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    out: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        out.setdefault(int(row["nx"]), []).append(row)
    for grouped in out.values():
        grouped.sort(key=lambda row: int(row["ranks"]))
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
    for n, grouped in rows_by_size(rows).items():
        ranks = [int(row["ranks"]) for row in grouped]
        values = [float(row[y_key]) for row in grouped]
        plt.plot(ranks, values, marker="o", label=f"N={n}")
    if ideal:
        xs = sorted({int(row["ranks"]) for row in rows})
        if xs:
            plt.plot(xs, xs, "k--", linewidth=1.0, label="ideal linear")
    plt.xlabel("MPI ranks")
    plt.ylabel(ylabel)
    plt.xticks(sorted({int(row["ranks"]) for row in rows}))
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
        "mpi_runtime_vs_ranks",
        "selected_solver_time_sec",
        "solver time (s)",
    )
    plot_metric(
        rows,
        fig_dir,
        "mpi_speedup_vs_ranks",
        "speedup",
        "speedup",
        ideal=True,
    )
    plot_metric(
        rows,
        fig_dir,
        "mpi_efficiency_vs_ranks",
        "parallel_efficiency",
        "parallel efficiency",
    )
    plot_metric(
        rows,
        fig_dir,
        "mpi_time_per_iteration_vs_ranks",
        "time_per_iteration_sec",
        "solver time per iteration (s)",
    )


def write_report(rows: list[dict[str, object]], path: Path, fig_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sizes = sorted({int(row["nx"]) for row in rows})
    ranks = sorted({int(row["ranks"]) for row in rows})
    best_by_size = []
    for n, grouped in rows_by_size(rows).items():
        best = max(grouped, key=lambda row: float(row["speedup"]))
        best_by_size.append(
            f"N={n}: {float(best['speedup']):.2f}x at {int(best['ranks'])} ranks"
        )
    all_ok = all(int(row["correctness_ok"]) == 1 for row in rows)
    correctness = (
        "The final divergence residuals matched the one-rank baseline within the configured tolerance, confirming the decomposition reproduces the serial solution."
        if all_ok
        else "Some final divergence residuals differed from the one-rank baseline beyond the configured tolerance; inspect the CSV before using those rows."
    )
    paragraph = (
        "MPI tests distributed-memory parallelism by splitting the same "
        f"{rows[0]['problem']} / {rows[0]['method']} simulation across a 2D "
        f"Cartesian domain decomposition for rank counts {ranks} and grid sizes "
        f"{sizes}. The ideal strong-scaling speedup is linear in the number of "
        "ranks, but real scaling is limited by halo-exchange communication "
        "(which grows with the per-rank boundary while compute grows with the "
        "per-rank area), the latency of the collective MPI_Allreduce used for the "
        "CFL and positivity checks every step, load imbalance, and the serial "
        "gather-to-root I/O. In this run, the best measured speedups were "
        + "; ".join(best_by_size)
        + ". Larger grids usually scale better because each rank owns more "
        "interior cells relative to its fixed-width halo, raising the "
        "computation-to-communication ratio, while parallel efficiency usually "
        "drops at high rank counts as surface-to-volume and collective-latency "
        "costs grow. "
        + correctness
    )
    with path.open("w") as handle:
        handle.write("# MPI Scaling Result\n\n")
        handle.write(paragraph + "\n\n")
        handle.write("Figures:\n")
        for name in [
            "mpi_runtime_vs_ranks.png",
            "mpi_speedup_vs_ranks.png",
            "mpi_efficiency_vs_ranks.png",
            "mpi_time_per_iteration_vs_ranks.png",
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
    parser.add_argument("--ranks", nargs="+", default=["1", "2", "4", "8"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--statistic", choices=["median", "best"], default="median")
    parser.add_argument("--problem", default="divergence_advection")
    parser.add_argument("--method", default="none")
    parser.add_argument("--tfinal", type=float, default=0.02)
    parser.add_argument("--diagnostic-stride", type=int, default=1000000)
    parser.add_argument("--threads-per-rank", type=int, default=1,
                        help="OMP_NUM_THREADS per rank (1 = pure MPI)")
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--oversubscribe", action="store_true",
                        help="pass --oversubscribe to mpirun (allows ranks > cores)")
    parser.add_argument("--mpirun-extra", default="",
                        help="extra space-separated flags forwarded to mpirun")
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-csv", default=str(DEFAULT_CSV))
    parser.add_argument("--raw-output-csv", default=str(DEFAULT_RAW_CSV))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIG_DIR))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT))
    parser.add_argument("--residual-rtol", type=float, default=1.0e-8)
    parser.add_argument("--residual-atol", type=float, default=1.0e-12)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    sizes = positive_ints(args.sizes)
    requested_ranks = positive_ints(args.ranks)
    if 1 not in requested_ranks:
        raise SystemExit("--ranks must include 1 (the strong-scaling baseline)")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.tfinal <= 0.0:
        raise SystemExit("--tfinal must be positive")

    cpu_count = available_cpus()
    if args.oversubscribe:
        ranks = requested_ranks
    else:
        ranks = [r for r in requested_ranks if r <= cpu_count]
        skipped = [r for r in requested_ranks if r > cpu_count]
        if skipped:
            print(
                f"Skipping rank counts above available CPUs ({cpu_count}): "
                f"{skipped} (use --oversubscribe to force them)"
            )
    if not ranks:
        raise SystemExit(f"no requested rank counts fit available CPUs ({cpu_count})")

    runner = Path(args.runner)
    if not runner.is_absolute():
        runner = ROOT / runner
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    ensure_build(runner, args.skip_build)

    raw_rows: list[dict[str, object]] = []
    skipped_combos: list[str] = []
    for n in sizes:
        for rank_count in ranks:
            for repeat in range(1, args.repeats + 1):
                try:
                    raw_rows.append(
                        benchmark_one(args, runner, output_root, n, rank_count, repeat)
                    )
                except RuntimeError as exc:
                    # Most likely: N not divisible by the MPI process-grid that
                    # MPI_Dims_create chose for this rank count.  Skip the combo
                    # (keeping the rest of the sweep) and report it.
                    combo = f"N={n}, ranks={rank_count}"
                    if combo not in skipped_combos:
                        skipped_combos.append(combo)
                    print(f"  skipping {combo}: {str(exc).splitlines()[0]}")
                    break  # no point repeating a combo that cannot run

    if skipped_combos:
        print(f"Skipped incompatible combinations: {skipped_combos}")
    if not raw_rows:
        raise SystemExit("no benchmark runs succeeded; check MPI build and grid/rank divisibility")

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
