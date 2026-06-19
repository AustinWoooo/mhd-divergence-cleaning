#!/usr/bin/env python3
"""Run unified OpenMP, MPI, and hybrid strong-scaling benchmarks."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import statistics
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SERIAL_BUILD_DIR = ROOT / "build"
MPI_BUILD_DIR = ROOT / "build-mpi"
SERIAL_RUNNER = SERIAL_BUILD_DIR / "mhd_runner_cli"
MPI_RUNNER = MPI_BUILD_DIR / "mhd_runner_mpi"

DEFAULT_RUN_ROOT = ROOT / "results" / "performance" / "runs"
DEFAULT_RAW_CSV = ROOT / "results" / "performance" / "openmp_mpi_scaling_raw.csv"
DEFAULT_CSV = ROOT / "results" / "performance" / "openmp_mpi_scaling.csv"
DEFAULT_REPORT = ROOT / "results" / "performance" / "openmp_mpi_scaling_report.md"
DEFAULT_FIG_DIR = ROOT / "figures" / "performance"

REQUIRED_METHODS = ["none", "mixed_glm", "elliptic_projection"]
HYBRID_METHODS = ["mixed_glm", "elliptic_projection"]

METHOD_COLORS = {
    "none": "#4c78a8",
    "mixed_glm": "#54a24b",
    "elliptic_projection": "#9467bd",
    "parabolic": "#e45756",
    "mixed_eglm": "#e377c2",
    "gi_mixed_eglm": "#b79f00",
}

METHOD_MARKERS = {
    "none": "o",
    "mixed_glm": "s",
    "elliptic_projection": "P",
    "parabolic": "^",
    "mixed_eglm": "D",
    "gi_mixed_eglm": "X",
}

FAMILY_LABELS = {
    "openmp": "OpenMP",
    "mpi": "MPI",
    "hybrid": "Hybrid",
}

RAW_FIELDS = [
    "family",
    "problem",
    "method",
    "nx",
    "ny",
    "ncell",
    "mpi_ranks",
    "omp_threads_per_rank",
    "total_cores",
    "repeat",
    "config_label",
    "launcher",
    "selected_solver_time_sec",
    "selected_total_wall_time_sec",
    "hydro_time_sec",
    "cleaning_time_sec",
    "diagnostics_compute_time_sec",
    "output_time_sec",
    "seconds_per_step",
    "cell_updates_per_second",
    "steps",
    "total_cell_updates",
    "final_L1_fv",
    "final_L2_fv",
    "final_Linf_fv",
    "energy_drift",
    "projection_iterations_total",
    "projection_true_residual",
    "projection_converged",
    "status",
    "failure_reason",
    "summary_file",
]

AGG_FIELDS = [
    "family",
    "problem",
    "method",
    "nx",
    "ny",
    "ncell",
    "mpi_ranks",
    "omp_threads_per_rank",
    "total_cores",
    "config_label",
    "repeat_count",
    "successful_repeats",
    "runtime_statistic",
    "solver_time_sec",
    "best_solver_time_sec",
    "median_solver_time_sec",
    "wall_time_sec",
    "hydro_time_sec",
    "cleaning_time_sec",
    "diagnostics_compute_time_sec",
    "output_time_sec",
    "seconds_per_step",
    "cell_updates_per_second",
    "steps",
    "total_cell_updates",
    "speedup",
    "parallel_efficiency",
    "final_L1_fv",
    "final_L2_fv",
    "final_Linf_fv",
    "energy_drift",
    "projection_iterations_total",
    "projection_true_residual",
    "projection_converged",
    "status",
    "failure_reason",
    "summary_file",
]


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def positive_int_list(values: list[str]) -> list[int]:
    out = [int(value) for value in values]
    if any(value <= 0 for value in out):
        raise argparse.ArgumentTypeError("all integer values must be positive")
    return out


def parse_config_label(label: str) -> tuple[int, int]:
    try:
        rank_text, thread_text = label.lower().split("x", 1)
        ranks = int(rank_text)
        threads = int(thread_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid hybrid config '{label}', expected RxT such as 2x4"
        ) from exc
    if ranks <= 0 or threads <= 0:
        raise argparse.ArgumentTypeError("hybrid ranks/threads must be positive")
    return ranks, threads


def choose_mpi_launcher(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for name in ("mpirun", "mpiexec"):
        path = shutil.which(name)
        if path:
            return path
    return None


def run_command(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    print(" ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def ensure_serial_build(skip_build: bool) -> None:
    if skip_build and SERIAL_RUNNER.exists():
        return
    if not (SERIAL_BUILD_DIR / "CMakeCache.txt").exists():
        result = run_command(["cmake", "-S", ".", "-B", SERIAL_BUILD_DIR.name])
        if result.returncode != 0:
            raise RuntimeError(result.stdout)
    result = run_command(
        ["cmake", "--build", SERIAL_BUILD_DIR.name, "--target", SERIAL_RUNNER.name, "--parallel"]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout)


def ensure_mpi_build(skip_build: bool) -> None:
    if skip_build and MPI_RUNNER.exists():
        return
    if not (MPI_BUILD_DIR / "CMakeCache.txt").exists():
        result = run_command(
            ["cmake", "-S", ".", "-B", MPI_BUILD_DIR.name, "-DENABLE_MPI=ON"]
        )
        if result.returncode != 0:
            raise RuntimeError(result.stdout)
    result = run_command(
        ["cmake", "--build", MPI_BUILD_DIR.name, "--target", MPI_RUNNER.name, "--parallel"]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout)


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


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def base_env(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_DYNAMIC"] = "FALSE"
    env["OMP_PROC_BIND"] = "spread"
    env["OMP_PLACES"] = "cores"
    return env


def summary_row_to_raw(
    *,
    family: str,
    problem: str,
    method: str,
    ranks: int,
    threads: int,
    repeat: int,
    launcher: str,
    config_label: str,
    summary: dict[str, str],
    summary_file: Path,
) -> dict[str, object]:
    status = summary.get("status", "")
    return {
        "family": family,
        "problem": problem,
        "method": method,
        "nx": as_int(summary, "nx"),
        "ny": as_int(summary, "ny"),
        "ncell": as_int(summary, "ncell"),
        "mpi_ranks": ranks,
        "omp_threads_per_rank": threads,
        "total_cores": ranks * threads,
        "repeat": repeat,
        "config_label": config_label,
        "launcher": launcher,
        "selected_solver_time_sec": as_float(summary, "hydro_time_sec", 0.0)
        + as_float(summary, "cleaning_time_sec", 0.0),
        "selected_total_wall_time_sec": as_float(summary, "total_wall_time_sec"),
        "hydro_time_sec": as_float(summary, "hydro_time_sec"),
        "cleaning_time_sec": as_float(summary, "cleaning_time_sec"),
        "diagnostics_compute_time_sec": as_float(summary, "diagnostics_compute_time_sec"),
        "output_time_sec": as_float(summary, "output_time_sec"),
        "seconds_per_step": as_float(summary, "seconds_per_step"),
        "cell_updates_per_second": as_float(summary, "cell_updates_per_second"),
        "steps": as_int(summary, "steps"),
        "total_cell_updates": as_int(summary, "total_cell_updates"),
        "final_L1_fv": as_float(summary, "final_L1_fv"),
        "final_L2_fv": as_float(summary, "final_L2_fv"),
        "final_Linf_fv": as_float(summary, "final_Linf_fv"),
        "energy_drift": as_float(summary, "energy_drift"),
        "projection_iterations_total": as_int(summary, "projection_iterations_total"),
        "projection_true_residual": as_float(summary, "projection_true_residual"),
        "projection_converged": (
            1 if method == "elliptic_projection" and status == "finished" else 0
            if method == "elliptic_projection" else ""
        ),
        "status": status,
        "failure_reason": summary.get("failure_reason", ""),
        "summary_file": display_path(summary_file),
    }


def skipped_row(
    *,
    family: str,
    problem: str,
    method: str,
    nx: int,
    ny: int,
    ranks: int,
    threads: int,
    repeat: int,
    launcher: str,
    reason: str,
) -> dict[str, object]:
    return {
        "family": family,
        "problem": problem,
        "method": method,
        "nx": nx,
        "ny": ny,
        "ncell": nx * ny,
        "mpi_ranks": ranks,
        "omp_threads_per_rank": threads,
        "total_cores": ranks * threads,
        "repeat": repeat,
        "config_label": f"{ranks}x{threads}",
        "launcher": launcher,
        "selected_solver_time_sec": "",
        "selected_total_wall_time_sec": "",
        "hydro_time_sec": "",
        "cleaning_time_sec": "",
        "diagnostics_compute_time_sec": "",
        "output_time_sec": "",
        "seconds_per_step": "",
        "cell_updates_per_second": "",
        "steps": "",
        "total_cell_updates": "",
        "final_L1_fv": "",
        "final_L2_fv": "",
        "final_Linf_fv": "",
        "energy_drift": "",
        "projection_iterations_total": "",
        "projection_true_residual": "",
        "projection_converged": "",
        "status": "skipped",
        "failure_reason": reason,
        "summary_file": "",
    }


def failed_row(
    *,
    family: str,
    problem: str,
    method: str,
    nx: int,
    ny: int,
    ranks: int,
    threads: int,
    repeat: int,
    launcher: str,
    reason: str,
) -> dict[str, object]:
    row = skipped_row(
        family=family,
        problem=problem,
        method=method,
        nx=nx,
        ny=ny,
        ranks=ranks,
        threads=threads,
        repeat=repeat,
        launcher=launcher,
        reason=reason,
    )
    row["status"] = "failed"
    return row


def runner_summary_path(run_root: Path, prefix: str, method: str) -> Path:
    return run_root / "summaries" / f"{prefix}_{method}_summary.csv"


def prefix_for_run(
    family: str,
    problem: str,
    method: str,
    nx: int,
    ranks: int,
    threads: int,
    repeat: int,
) -> str:
    return f"perf_{family}_{problem}_{method}_n{nx}_r{ranks}_t{threads}_rep{repeat}"


def benchmark_serial_or_openmp(
    *,
    family: str,
    runner: Path,
    run_root: Path,
    problem: str,
    method: str,
    nx: int,
    ny: int,
    ranks: int,
    threads: int,
    repeat: int,
    tfinal: float,
    diagnostic_stride: int,
    timeout: float,
) -> dict[str, object]:
    prefix = prefix_for_run(family, problem, method, nx, ranks, threads, repeat)
    cmd = [
        str(runner),
        "--performance-mode",
        "--no-snapshots",
        "--output-root",
        str(run_root),
        "--output-prefix",
        prefix,
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--tfinal",
        str(tfinal),
        "--diagnostic-stride",
        str(diagnostic_stride),
        problem,
        method,
    ]
    try:
        result = run_command(cmd, env=base_env(threads), timeout=timeout)
    except subprocess.TimeoutExpired:
        return failed_row(
            family=family,
            problem=problem,
            method=method,
            nx=nx,
            ny=ny,
            ranks=ranks,
            threads=threads,
            repeat=repeat,
            launcher="none",
            reason=f"timeout>{int(timeout)}s",
        )

    if result.returncode != 0:
        return failed_row(
            family=family,
            problem=problem,
            method=method,
            nx=nx,
            ny=ny,
            ranks=ranks,
            threads=threads,
            repeat=repeat,
            launcher="none",
            reason=(result.stdout.splitlines()[-1] if result.stdout else f"exit={result.returncode}")[:240],
        )

    summary_file = runner_summary_path(run_root, prefix, method)
    if not summary_file.exists():
        return failed_row(
            family=family,
            problem=problem,
            method=method,
            nx=nx,
            ny=ny,
            ranks=ranks,
            threads=threads,
            repeat=repeat,
            launcher="none",
            reason="missing_summary_csv",
        )

    summary = read_summary(summary_file)
    return summary_row_to_raw(
        family=family,
        problem=problem,
        method=method,
        ranks=ranks,
        threads=threads,
        repeat=repeat,
        launcher="none",
        config_label=f"{ranks}x{threads}",
        summary=summary,
        summary_file=summary_file,
    )


def benchmark_mpi_or_hybrid(
    *,
    family: str,
    launcher: str,
    runner: Path,
    run_root: Path,
    problem: str,
    method: str,
    nx: int,
    ny: int,
    ranks: int,
    threads: int,
    repeat: int,
    tfinal: float,
    diagnostic_stride: int,
    timeout: float,
) -> dict[str, object]:
    prefix = prefix_for_run(family, problem, method, nx, ranks, threads, repeat)
    cmd = [
        launcher,
        "-np",
        str(ranks),
        str(runner),
        "--performance-mode",
        "--no-snapshots",
        "--output-root",
        str(run_root),
        "--output-prefix",
        prefix,
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--tfinal",
        str(tfinal),
        "--diagnostic-stride",
        str(diagnostic_stride),
        problem,
        method,
    ]
    try:
        result = run_command(cmd, env=base_env(threads), timeout=timeout)
    except subprocess.TimeoutExpired:
        return failed_row(
            family=family,
            problem=problem,
            method=method,
            nx=nx,
            ny=ny,
            ranks=ranks,
            threads=threads,
            repeat=repeat,
            launcher=Path(launcher).name,
            reason=f"timeout>{int(timeout)}s",
        )

    if result.returncode != 0:
        tail = result.stdout.splitlines()[-1] if result.stdout else f"exit={result.returncode}"
        return failed_row(
            family=family,
            problem=problem,
            method=method,
            nx=nx,
            ny=ny,
            ranks=ranks,
            threads=threads,
            repeat=repeat,
            launcher=Path(launcher).name,
            reason=tail[:240],
        )

    summary_file = runner_summary_path(run_root, prefix, method)
    if not summary_file.exists():
        return failed_row(
            family=family,
            problem=problem,
            method=method,
            nx=nx,
            ny=ny,
            ranks=ranks,
            threads=threads,
            repeat=repeat,
            launcher=Path(launcher).name,
            reason="missing_summary_csv",
        )

    summary = read_summary(summary_file)
    return summary_row_to_raw(
        family=family,
        problem=problem,
        method=method,
        ranks=ranks,
        threads=threads,
        repeat=repeat,
        launcher=Path(launcher).name,
        config_label=f"{ranks}x{threads}",
        summary=summary,
        summary_file=summary_file,
    )


def aggregate_rows(raw_rows: list[dict[str, object]], statistic: str) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in raw_rows:
        key = (
            row["family"],
            row["problem"],
            row["method"],
            int(row["nx"]),
            int(row["ny"]),
            int(row["mpi_ranks"]),
            int(row["omp_threads_per_rank"]),
        )
        grouped.setdefault(key, []).append(row)

    aggregate: list[dict[str, object]] = []
    baseline_by_family: dict[tuple[str, str, int, int], float] = {}
    baseline_by_mpi_like: dict[tuple[str, int, int], float] = {}
    baseline_by_openmp: dict[tuple[str, int, int], float] = {}

    for rows in grouped.values():
        success = [
            row for row in rows
            if str(row.get("status", "")) == "finished"
            and row.get("selected_solver_time_sec") not in ("", None)
        ]
        representative = success if success else rows
        sample = representative[0]
        record = {
            "family": sample["family"],
            "problem": sample["problem"],
            "method": sample["method"],
            "nx": sample["nx"],
            "ny": sample["ny"],
            "ncell": sample["ncell"],
            "mpi_ranks": sample["mpi_ranks"],
            "omp_threads_per_rank": sample["omp_threads_per_rank"],
            "total_cores": sample["total_cores"],
            "config_label": sample["config_label"],
            "repeat_count": len(rows),
            "successful_repeats": len(success),
            "runtime_statistic": statistic,
            "status": sample["status"] if not success else "finished",
            "failure_reason": sample.get("failure_reason", "") if not success else "",
            "summary_file": sample.get("summary_file", ""),
        }

        if success:
            times = [float(row["selected_solver_time_sec"]) for row in success]
            target = min(times) if statistic == "best" else statistics.median(times)
            selected = min(success, key=lambda row: abs(float(row["selected_solver_time_sec"]) - target))
            record.update({
                "solver_time_sec": float(selected["selected_solver_time_sec"]),
                "best_solver_time_sec": min(times),
                "median_solver_time_sec": statistics.median(times),
                "wall_time_sec": float(selected["selected_total_wall_time_sec"]),
                "hydro_time_sec": float(selected["hydro_time_sec"]),
                "cleaning_time_sec": float(selected["cleaning_time_sec"]),
                "diagnostics_compute_time_sec": float(selected["diagnostics_compute_time_sec"]),
                "output_time_sec": float(selected["output_time_sec"]),
                "seconds_per_step": float(selected["seconds_per_step"]),
                "cell_updates_per_second": float(selected["cell_updates_per_second"]),
                "steps": int(selected["steps"]),
                "total_cell_updates": int(selected["total_cell_updates"]),
                "final_L1_fv": float(selected["final_L1_fv"]),
                "final_L2_fv": float(selected["final_L2_fv"]),
                "final_Linf_fv": float(selected["final_Linf_fv"]),
                "energy_drift": float(selected["energy_drift"]),
                "projection_iterations_total": int(selected["projection_iterations_total"]),
                "projection_true_residual": selected["projection_true_residual"],
                "projection_converged": selected["projection_converged"],
                "summary_file": selected["summary_file"],
            })
        else:
            for key in AGG_FIELDS:
                record.setdefault(key, "")
        aggregate.append(record)

    for row in aggregate:
        if row["status"] != "finished":
            continue
        key = (str(row["family"]), str(row["method"]), int(row["nx"]), int(row["ny"]))
        if int(row["total_cores"]) == 1:
            baseline_by_family[key] = float(row["solver_time_sec"])
            family = str(row["family"])
            method = str(row["method"])
            dims = (method, int(row["nx"]), int(row["ny"]))
            if family == "openmp":
                baseline_by_openmp[dims] = float(row["solver_time_sec"])
            if family in {"mpi", "hybrid"}:
                baseline_by_mpi_like[dims] = float(row["solver_time_sec"])

    for row in aggregate:
        if row["status"] != "finished":
            row["speedup"] = ""
            row["parallel_efficiency"] = ""
            continue
        family = str(row["family"])
        method = str(row["method"])
        nx = int(row["nx"])
        ny = int(row["ny"])
        own_key = (family, method, nx, ny)
        baseline = baseline_by_family.get(own_key)
        if baseline is None:
            dims = (method, nx, ny)
            if family == "hybrid":
                baseline = baseline_by_mpi_like.get(dims, baseline_by_openmp.get(dims))
            elif family == "mpi":
                baseline = baseline_by_mpi_like.get(dims, baseline_by_openmp.get(dims))
            else:
                baseline = baseline_by_openmp.get(dims, baseline_by_mpi_like.get(dims))
        if baseline is None or baseline <= 0.0:
            row["speedup"] = ""
            row["parallel_efficiency"] = ""
            continue
        speedup = baseline / float(row["solver_time_sec"])
        row["speedup"] = speedup
        row["parallel_efficiency"] = speedup / int(row["total_cores"])

    aggregate.sort(
        key=lambda row: (
            str(row["family"]),
            str(row["method"]),
            int(row["nx"]),
            int(row["mpi_ranks"]),
            int(row["omp_threads_per_rank"]),
        )
    )
    return aggregate


def finished_rows(rows: list[dict[str, object]], family: str) -> list[dict[str, object]]:
    return [row for row in rows if row["family"] == family and row["status"] == "finished"]


def largest_common_grid(rows: list[dict[str, object]], methods: list[str]) -> int | None:
    if not rows:
        return None
    per_method = {
        method: {int(row["nx"]) for row in rows if row["method"] == method}
        for method in methods
    }
    common = set.intersection(*(values for values in per_method.values() if values)) if all(per_method.values()) else set()
    return max(common) if common else None


def plot_family_lines(
    rows: list[dict[str, object]],
    *,
    family: str,
    methods: list[str],
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
    ideal: bool = False,
) -> bool:
    selected_grid = largest_common_grid(rows, methods)
    if selected_grid is None:
        return False
    subset = [row for row in rows if int(row["nx"]) == selected_grid and row["method"] in methods]
    if not subset:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    x_values_seen: set[int] = set()
    for method in methods:
        series = sorted(
            [row for row in subset if row["method"] == method],
            key=lambda row: int(row[x_key]),
        )
        if not series:
            continue
        xs = [int(row[x_key]) for row in series]
        ys = [float(row[y_key]) for row in series]
        x_values_seen.update(xs)
        ax.plot(
            xs,
            ys,
            marker=METHOD_MARKERS.get(method, "o"),
            color=METHOD_COLORS.get(method, None),
            linewidth=1.9,
            label=method,
        )
    if ideal and x_values_seen:
        xs = sorted(x_values_seen)
        ax.plot(xs, xs, "k--", linewidth=1.0, label="ideal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(x_values_seen))
    ax.set_title(f"{title}\nproblem={subset[0]['problem']}, grid={selected_grid}x{selected_grid}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_hybrid_runtime(
    rows: list[dict[str, object]],
    *,
    methods: list[str],
    output_path: Path,
) -> bool:
    selected_grid = largest_common_grid(rows, methods)
    if selected_grid is None:
        return False
    subset = [row for row in rows if int(row["nx"]) == selected_grid and row["method"] in methods]
    labels = sorted({str(row["config_label"]) for row in subset}, key=lambda text: (int(text.split("x")[0]), int(text.split("x")[1])))
    if not labels:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    for method in methods:
        series = {str(row["config_label"]): float(row["solver_time_sec"]) for row in subset if row["method"] == method}
        xs = list(range(len(labels)))
        ys = [series.get(label, math.nan) for label in labels]
        ax.plot(
            xs,
            ys,
            marker=METHOD_MARKERS.get(method, "o"),
            color=METHOD_COLORS.get(method, None),
            linewidth=1.9,
            label=method,
        )
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels)
    ax.set_xlabel("MPI ranks x OpenMP threads")
    ax.set_ylabel("solver time (s)")
    ax.set_title(f"Hybrid runtime comparison\nproblem={subset[0]['problem']}, grid={selected_grid}x{selected_grid}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_method_overhead(rows: list[dict[str, object]], output_path: Path) -> bool:
    required = ["none", "mixed_glm", "elliptic_projection"]
    finished = [row for row in rows if row["status"] == "finished" and row["method"] in required]
    if not finished:
        return False
    common_grids = None
    for family in ("openmp", "mpi", "hybrid"):
        family_rows = [row for row in finished if row["family"] == family]
        if not family_rows:
            continue
        grids = {int(row["nx"]) for row in family_rows}
        common_grids = grids if common_grids is None else common_grids.intersection(grids)
    if not common_grids:
        common_grids = {int(row["nx"]) for row in finished}
    grid = max(common_grids)

    candidate_core_sets = []
    for family in ("openmp", "mpi", "hybrid"):
        family_rows = [row for row in finished if row["family"] == family and int(row["nx"]) == grid]
        if family_rows:
            candidate_core_sets.append({int(row["total_cores"]) for row in family_rows})
    if not candidate_core_sets:
        return False
    common_cores = set.intersection(*candidate_core_sets) if len(candidate_core_sets) > 1 else candidate_core_sets[0]
    if not common_cores:
        return False
    total_cores = max(common_cores)

    chosen: dict[tuple[str, str], float] = {}
    for family in ("openmp", "mpi", "hybrid"):
        family_rows = [
            row for row in finished
            if row["family"] == family and int(row["nx"]) == grid and int(row["total_cores"]) == total_cores
        ]
        for method in required:
            matches = [row for row in family_rows if row["method"] == method]
            if not matches:
                continue
            if family == "hybrid":
                row = min(matches, key=lambda item: float(item["solver_time_sec"]))
            else:
                row = matches[0]
            chosen[(family, method)] = float(row["solver_time_sec"])

    families = [family for family in ("openmp", "mpi", "hybrid") if any((family, method) in chosen for method in required)]
    if not families:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    x = list(range(len(required)))
    width = 0.22
    offsets = {
        family: (index - (len(families) - 1) / 2.0) * width
        for index, family in enumerate(families)
    }
    for family in families:
        ys = [chosen.get((family, method), math.nan) for method in required]
        ax.bar(
            [value + offsets[family] for value in x],
            ys,
            width=width,
            label=FAMILY_LABELS[family],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(required)
    ax.set_ylabel("solver time (s)")
    ax.set_title(f"Method overhead comparison\nproblem={finished[0]['problem']}, grid={grid}x{grid}, total cores={total_cores}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def make_plots(aggregate: list[dict[str, object]], fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    openmp_rows = finished_rows(aggregate, "openmp")
    mpi_rows = finished_rows(aggregate, "mpi")
    hybrid_rows = finished_rows(aggregate, "hybrid")
    if plot_family_lines(
        openmp_rows,
        family="openmp",
        methods=sorted({str(row["method"]) for row in openmp_rows}),
        x_key="omp_threads_per_rank",
        y_key="speedup",
        xlabel="OpenMP threads",
        ylabel="speedup",
        title="OpenMP speedup by method",
        output_path=fig_dir / "openmp_speedup_by_method.png",
        ideal=True,
    ):
        outputs.append(fig_dir / "openmp_speedup_by_method.png")
    if plot_family_lines(
        mpi_rows,
        family="mpi",
        methods=sorted({str(row["method"]) for row in mpi_rows}),
        x_key="mpi_ranks",
        y_key="speedup",
        xlabel="MPI ranks",
        ylabel="speedup",
        title="MPI strong scaling by method",
        output_path=fig_dir / "mpi_speedup_by_method.png",
        ideal=True,
    ):
        outputs.append(fig_dir / "mpi_speedup_by_method.png")
    if plot_family_lines(
        mpi_rows,
        family="mpi",
        methods=sorted({str(row["method"]) for row in mpi_rows}),
        x_key="mpi_ranks",
        y_key="parallel_efficiency",
        xlabel="MPI ranks",
        ylabel="parallel efficiency",
        title="MPI parallel efficiency by method",
        output_path=fig_dir / "mpi_efficiency_by_method.png",
    ):
        outputs.append(fig_dir / "mpi_efficiency_by_method.png")
    if plot_hybrid_runtime(
        hybrid_rows,
        methods=[method for method in HYBRID_METHODS if any(row["method"] == method for row in hybrid_rows)],
        output_path=fig_dir / "hybrid_runtime_comparison.png",
    ):
        outputs.append(fig_dir / "hybrid_runtime_comparison.png")
    if plot_method_overhead(aggregate, fig_dir / "method_overhead_comparison.png"):
        outputs.append(fig_dir / "method_overhead_comparison.png")
    return outputs


def write_report(
    aggregate: list[dict[str, object]],
    *,
    path: Path,
    figures: list[Path],
    skipped_configs: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    finished = [row for row in aggregate if row["status"] == "finished"]
    with path.open("w") as handle:
        handle.write("# Parallel Scaling Workflow\n\n")
        if finished:
            families = sorted({str(row["family"]) for row in finished})
            methods = sorted({str(row["method"]) for row in finished})
            grids = sorted({f"{row['nx']}x{row['ny']}" for row in finished})
            handle.write(
                "Finished benchmark families: "
                + ", ".join(families)
                + ". Methods: "
                + ", ".join(methods)
                + ". Grids: "
                + ", ".join(grids)
                + ".\n\n"
            )
        if skipped_configs:
            handle.write("Skipped configurations:\n")
            for item in skipped_configs:
                handle.write(f"- {item}\n")
            handle.write("\n")
        handle.write("Figures:\n")
        for figure in figures:
            handle.write(f"- `{display_path(figure)}`\n")


def default_sizes(smoke: bool, include_512: bool) -> list[int]:
    if smoke:
        return [64]
    sizes = [128, 256]
    if include_512:
        sizes.append(512)
    return sizes


def default_threads(smoke: bool) -> list[int]:
    return [1, 2] if smoke else [1, 2, 4, 8, 16]


def default_ranks(smoke: bool) -> list[int]:
    return [1, 2] if smoke else [1, 2, 4, 8, 16]


def default_hybrid_configs(smoke: bool, cpu_count: int) -> list[tuple[int, int]]:
    if smoke:
        candidates = [(1, 2), (2, 1)]
    else:
        budgets = [budget for budget in (8, 16) if budget <= cpu_count]
        if not budgets:
            budgets = [max(1, min(cpu_count, 4))]
        candidates = []
        for budget in budgets:
            config = 1
            while config <= budget:
                if budget % config == 0:
                    candidates.append((config, budget // config))
                config *= 2
    unique: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for ranks, threads in candidates:
        if ranks * threads <= cpu_count and (ranks, threads) not in seen:
            seen.add((ranks, threads))
            unique.append((ranks, threads))
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--openmp", action="store_true")
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--problem", default="divergence_advection")
    parser.add_argument("--methods", nargs="+", default=REQUIRED_METHODS)
    parser.add_argument("--sizes", nargs="+")
    parser.add_argument("--openmp-threads", nargs="+")
    parser.add_argument("--mpi-ranks", nargs="+")
    parser.add_argument("--hybrid-configs", nargs="+")
    parser.add_argument("--include-512", action="store_true")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--statistic", choices=["best", "median"], default="best")
    parser.add_argument("--tfinal", type=float, default=0.02)
    parser.add_argument("--diagnostic-stride", type=int, default=1_000_000)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--mpirun", default=None)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--output-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--raw-output-csv", default=str(DEFAULT_RAW_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_CSV))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIG_DIR))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    families = {
        "openmp": args.openmp,
        "mpi": args.mpi,
        "hybrid": args.hybrid,
    }
    if not any(families.values()):
        for key in families:
            families[key] = True

    cpu_count = available_cpus()
    sizes = positive_int_list(args.sizes) if args.sizes else default_sizes(args.smoke, args.include_512)
    openmp_threads = positive_int_list(args.openmp_threads) if args.openmp_threads else default_threads(args.smoke)
    mpi_ranks = positive_int_list(args.mpi_ranks) if args.mpi_ranks else default_ranks(args.smoke)
    hybrid_configs = (
        [parse_config_label(label) for label in args.hybrid_configs]
        if args.hybrid_configs else default_hybrid_configs(args.smoke, cpu_count)
    )
    repeats = args.repeats if args.repeats is not None else (1 if args.smoke else 3)
    if repeats <= 0:
        raise SystemExit("--repeats must be positive")

    run_root = Path(args.output_root)
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    raw_csv = Path(args.raw_output_csv)
    if not raw_csv.is_absolute():
        raw_csv = ROOT / raw_csv
    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = ROOT / output_csv
    fig_dir = Path(args.figure_dir)
    if not fig_dir.is_absolute():
        fig_dir = ROOT / fig_dir
    report_md = Path(args.report_md)
    if not report_md.is_absolute():
        report_md = ROOT / report_md

    launcher = choose_mpi_launcher(args.mpirun)
    if (families["mpi"] or families["hybrid"]) and launcher is None:
        raise SystemExit("MPI or hybrid scaling requested, but neither mpirun nor mpiexec is available")

    ensure_serial_build(args.skip_build)
    if families["mpi"] or families["hybrid"]:
        ensure_mpi_build(args.skip_build)

    raw_rows: list[dict[str, object]] = []
    skipped_configs: list[str] = []

    def record_skip(row: dict[str, object]) -> None:
        raw_rows.append(row)
        skipped_configs.append(
            f"{row['family']} {row['method']} {row['nx']}x{row['ny']} {row['config_label']}: {row['failure_reason']}"
        )

    bounded_threads = []
    for thread in openmp_threads:
        if thread <= cpu_count:
            bounded_threads.append(thread)
        else:
            for size in sizes:
                for method in args.methods:
                    record_skip(
                        skipped_row(
                            family="openmp",
                            problem=args.problem,
                            method=method,
                            nx=size,
                            ny=size,
                            ranks=1,
                            threads=thread,
                            repeat=1,
                            launcher="none",
                            reason=f"threads>{cpu_count}_cpus",
                        )
                    )

    bounded_ranks = []
    for ranks in mpi_ranks:
        if ranks <= cpu_count:
            bounded_ranks.append(ranks)
        else:
            for size in sizes:
                for method in args.methods:
                    record_skip(
                        skipped_row(
                            family="mpi",
                            problem=args.problem,
                            method=method,
                            nx=size,
                            ny=size,
                            ranks=ranks,
                            threads=1,
                            repeat=1,
                            launcher=Path(launcher or "mpirun").name,
                            reason=f"ranks>{cpu_count}_cpus",
                        )
                    )

    bounded_hybrid: list[tuple[int, int]] = []
    for ranks, threads in hybrid_configs:
        total = ranks * threads
        if total <= cpu_count:
            bounded_hybrid.append((ranks, threads))
        else:
            for size in sizes:
                for method in HYBRID_METHODS:
                    record_skip(
                        skipped_row(
                            family="hybrid",
                            problem=args.problem,
                            method=method,
                            nx=size,
                            ny=size,
                            ranks=ranks,
                            threads=threads,
                            repeat=1,
                            launcher=Path(launcher or "mpirun").name,
                            reason=f"total_cores>{cpu_count}_cpus",
                        )
                    )

    if families["openmp"]:
        for method in args.methods:
            for size in sizes:
                for threads in bounded_threads:
                    for repeat in range(1, repeats + 1):
                        raw_rows.append(
                            benchmark_serial_or_openmp(
                                family="openmp",
                                runner=SERIAL_RUNNER,
                                run_root=run_root,
                                problem=args.problem,
                                method=method,
                                nx=size,
                                ny=size,
                                ranks=1,
                                threads=threads,
                                repeat=repeat,
                                tfinal=args.tfinal,
                                diagnostic_stride=args.diagnostic_stride,
                                timeout=args.timeout_seconds,
                            )
                        )

    if families["mpi"]:
        for method in args.methods:
            for size in sizes:
                for ranks in bounded_ranks:
                    for repeat in range(1, repeats + 1):
                        raw_rows.append(
                            benchmark_mpi_or_hybrid(
                                family="mpi",
                                launcher=launcher or "mpirun",
                                runner=MPI_RUNNER,
                                run_root=run_root,
                                problem=args.problem,
                                method=method,
                                nx=size,
                                ny=size,
                                ranks=ranks,
                                threads=1,
                                repeat=repeat,
                                tfinal=args.tfinal,
                                diagnostic_stride=args.diagnostic_stride,
                                timeout=args.timeout_seconds,
                            )
                        )

    if families["hybrid"]:
        for method in [candidate for candidate in HYBRID_METHODS if candidate in args.methods]:
            for size in sizes:
                for ranks, threads in bounded_hybrid:
                    for repeat in range(1, repeats + 1):
                        raw_rows.append(
                            benchmark_mpi_or_hybrid(
                                family="hybrid",
                                launcher=launcher or "mpirun",
                                runner=MPI_RUNNER,
                                run_root=run_root,
                                problem=args.problem,
                                method=method,
                                nx=size,
                                ny=size,
                                ranks=ranks,
                                threads=threads,
                                repeat=repeat,
                                tfinal=args.tfinal,
                                diagnostic_stride=args.diagnostic_stride,
                                timeout=args.timeout_seconds,
                            )
                        )

    aggregate = aggregate_rows(raw_rows, args.statistic)
    write_csv(raw_csv, RAW_FIELDS, raw_rows)
    write_csv(output_csv, AGG_FIELDS, aggregate)
    figures = make_plots(aggregate, fig_dir)
    write_report(aggregate, path=report_md, figures=figures, skipped_configs=skipped_configs)

    print(f"Wrote {display_path(raw_csv)}")
    print(f"Wrote {display_path(output_csv)}")
    print(f"Wrote {display_path(report_md)}")
    for figure in figures:
        print(f"Wrote {display_path(figure)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
