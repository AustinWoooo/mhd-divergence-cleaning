#!/usr/bin/env python3
"""Run serial MHD performance-scaling cases and plot benchmark figures."""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"
DEFAULT_RUNNER = BUILD_DIR / "mhd_runner_cli"
RESULTS_DIR = ROOT / "results" / "mhd_runner"
PERF_DIR = RESULTS_DIR / "performance"
SUMMARY_DIR = RESULTS_DIR / "summaries"
DIV_DIR = RESULTS_DIR / "divergence"
FIG_DIR = ROOT / "figures" / "mhd_runner"

PROBLEM_PREFIX = {
    "orszag_tang": "ot",
    "field_loop": "fl",
    "divergence_advection": "da",
}

METHOD_ORDER = [
    "none",
    "parabolic",
    "hyperbolic_glm",
    "mixed_glm",
    "mixed_eglm",
    "gi_mixed_eglm",
    "elliptic_projection",
    "powell_source",
    "powell_source_limited",
    "powell_source_subcycled",
]

METHOD_LABELS = {
    "none": "none",
    "parabolic": "parabolic",
    "hyperbolic_glm": "hyperbolic GLM",
    "mixed_glm": "mixed GLM",
    "mixed_eglm": "mixed EGLM",
    "gi_mixed_eglm": "GI mixed EGLM",
    "elliptic_projection": "elliptic projection",
    "powell_source": "Powell",
    "powell_source_limited": "Powell limited",
    "powell_source_subcycled": "Powell subcycled",
}

METHOD_MARKERS = {
    "none": "o",
    "parabolic": "v",
    "hyperbolic_glm": ">",
    "mixed_glm": "s",
    "mixed_eglm": "^",
    "gi_mixed_eglm": "D",
    "elliptic_projection": "P",
    "powell_source": "X",
    "powell_source_limited": "*",
    "powell_source_subcycled": "h",
}

TIMING_COLORS = {
    "hydro": "#4c78a8",
    "cleaning": "#f58518",
    "diagnostics": "#54a24b",
}

CSV_FIELDS = [
    "problem",
    "method",
    "reconstruction",
    "limiter",
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
    "energy_drift",
    "failure_reason",
    "summary_file",
    "diagnostic_file",
]


def run(cmd: list[str], dry_run: bool = False, check: bool = True) -> subprocess.CompletedProcess[str] | None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        return subprocess.run(cmd, cwd=ROOT, check=check)
    return None


def ensure_build(skip_build: bool, runner: Path) -> None:
    if skip_build and runner.exists():
        return
    if not (BUILD_DIR / "CMakeCache.txt").exists():
        run(["cmake", "-S", ".", "-B", "build"])
    run(["cmake", "--build", "build", "--target", runner.name, "--parallel"])


def canonical_method(method: str) -> str:
    return "mixed_eglm" if method == "eglm" else method


def benchmark_prefix(problem: str, reconstruction: str, n: int, method: str) -> str:
    short = PROBLEM_PREFIX.get(problem, problem)
    return f"perf_{short}_{reconstruction}_n{n}_{canonical_method(method)}"


def read_summary(prefix: str, method: str) -> tuple[dict[str, str], Path]:
    canonical = canonical_method(method)
    exact = SUMMARY_DIR / f"{prefix}_{canonical}_summary.csv"
    if exact.exists():
        path = exact
    else:
        matches = sorted(SUMMARY_DIR.glob(f"{prefix}_*_summary.csv"))
        if not matches:
            raise FileNotFoundError(f"no summary CSV found for prefix {prefix}")
        path = matches[0]

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"{path} has no data rows")
    return rows[-1], path


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


def read_divergence_metrics(path: Path) -> dict[str, float]:
    metrics = {
        "final_L2_norm_fv": math.nan,
        "final_Linf_norm_fv": math.nan,
        "peak_L2_norm_fv": math.nan,
        "peak_Linf_norm_fv": math.nan,
        "time_integrated_L2_norm_fv": math.nan,
    }
    if not path.exists():
        print(
            f"WARNING: missing divergence diagnostic CSV {path}; "
            "normalized divergence metrics set to NaN"
        )
        return metrics

    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        print(
            f"WARNING: could not read divergence diagnostic CSV {path}: {exc}; "
            "normalized divergence metrics set to NaN"
        )
        return metrics

    if not rows:
        print(
            f"WARNING: divergence diagnostic CSV {path} has no rows; "
            "normalized divergence metrics set to NaN"
        )
        return metrics

    required = {"time", "L2_norm_fv"}
    missing = required.difference(rows[0].keys())
    if missing:
        print(
            f"WARNING: divergence diagnostic CSV {path} missing columns "
            f"{', '.join(sorted(missing))}; normalized divergence metrics set to NaN"
        )
        return metrics
    has_linf = "Linf_norm_fv" in rows[0]
    if not has_linf:
        print(
            f"WARNING: divergence diagnostic CSV {path} missing column Linf_norm_fv; "
            "Linf normalized divergence metrics set to NaN"
        )

    times = [as_float(row, "time") for row in rows]
    l2_values = [as_float(row, "L2_norm_fv") for row in rows]

    metrics["final_L2_norm_fv"] = l2_values[-1]

    finite_l2 = [value for value in l2_values if math.isfinite(value)]
    if finite_l2:
        metrics["peak_L2_norm_fv"] = max(finite_l2)

    if has_linf:
        linf_values = [as_float(row, "Linf_norm_fv") for row in rows]
        metrics["final_Linf_norm_fv"] = linf_values[-1]
        finite_linf = [value for value in linf_values if math.isfinite(value)]
        if finite_linf:
            metrics["peak_Linf_norm_fv"] = max(finite_linf)

    integral = 0.0
    have_interval = False
    for i in range(1, len(rows)):
        t0 = times[i - 1]
        t1 = times[i]
        y0 = l2_values[i - 1]
        y1 = l2_values[i]
        if all(math.isfinite(value) for value in (t0, t1, y0, y1)):
            integral += 0.5 * (y0 + y1) * (t1 - t0)
            have_interval = True
    if have_interval:
        metrics["time_integrated_L2_norm_fv"] = integral
    elif len(rows) == 1 and math.isfinite(l2_values[0]):
        metrics["time_integrated_L2_norm_fv"] = 0.0

    return metrics


def make_performance_row(
    summary: dict[str, str],
    summary_path: Path,
    problem: str,
    method: str,
    reconstruction: str,
    limiter: str,
    n: int,
    tfinal: float,
    prefix: str,
) -> dict[str, str | int | float]:
    canonical = canonical_method(method)
    diagnostic_path = DIV_DIR / f"{prefix}_{canonical}.csv"
    divergence_metrics = read_divergence_metrics(diagnostic_path)
    ncell = n * n

    row = {
        "problem": problem,
        "method": canonical,
        "reconstruction": reconstruction,
        "limiter": limiter,
        "nx": n,
        "ny": n,
        "ncell": ncell,
        "tfinal": tfinal,
        "steps": as_int(summary, "steps"),
        "status": summary.get("status", ""),
        "total_wall_time_sec": as_float(summary, "total_wall_time_sec"),
        "initialization_time_sec": as_float(summary, "initialization_time_sec"),
        "hydro_time_sec": as_float(summary, "hydro_time_sec"),
        "cleaning_time_sec": as_float(summary, "cleaning_time_sec"),
        "diagnostics_compute_time_sec": as_float(summary, "diagnostics_compute_time_sec"),
        "diagnostics_write_time_sec": as_float(summary, "diagnostics_write_time_sec"),
        "snapshot_write_time_sec": as_float(summary, "snapshot_write_time_sec"),
        "summary_write_time_sec": as_float(summary, "summary_write_time_sec"),
        "output_time_sec": as_float(summary, "output_time_sec"),
        "total_cell_updates": as_int(summary, "total_cell_updates"),
        "seconds_per_step": as_float(summary, "seconds_per_step"),
        "cell_updates_per_second": as_float(summary, "cell_updates_per_second"),
        "cleaning_subcycles_total": as_int(summary, "cleaning_subcycles_total"),
        "projection_iterations_total": as_int(summary, "projection_iterations_total"),
        "final_L2_fv": as_float(summary, "final_L2_fv"),
        "final_Linf_fv": as_float(summary, "final_Linf_fv"),
        "final_L2_norm_fv": divergence_metrics["final_L2_norm_fv"],
        "final_Linf_norm_fv": divergence_metrics["final_Linf_norm_fv"],
        "peak_L2_norm_fv": divergence_metrics["peak_L2_norm_fv"],
        "peak_Linf_norm_fv": divergence_metrics["peak_Linf_norm_fv"],
        "time_integrated_L2_norm_fv": divergence_metrics["time_integrated_L2_norm_fv"],
        "min_pressure": as_float(summary, "min_pressure"),
        "min_density": as_float(summary, "min_density"),
        "energy_drift": as_float(summary, "energy_drift"),
        "failure_reason": summary.get("failure_reason", ""),
        "summary_file": str(summary_path.relative_to(ROOT)),
        "diagnostic_file": str(diagnostic_path.relative_to(ROOT)),
    }
    return row


def write_csv(rows: list[dict[str, str | int | float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


def finite_float(value: object) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def method_sort_key(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), method)


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def figure_path(stem: str, figure_suffix: str = "", resolution: int | None = None) -> Path:
    clean_suffix = figure_suffix.strip("_")
    parts = [stem]
    if clean_suffix:
        parts.append(clean_suffix)
    if resolution is not None:
        parts.append(str(resolution))
    return FIG_DIR / ("_".join(parts) + ".png")


def group_rows(rows: list[dict[str, str | int | float]]) -> dict[str, list[dict[str, str | int | float]]]:
    grouped_unsorted: dict[str, list[dict[str, str | int | float]]] = {}
    for row in rows:
        grouped_unsorted.setdefault(str(row["method"]), []).append(row)

    grouped: dict[str, list[dict[str, str | int | float]]] = {}
    for method in sorted(grouped_unsorted, key=method_sort_key):
        group = grouped_unsorted[method]
        group.sort(key=lambda r: int(r["ncell"]))
        grouped[method] = group
    return grouped


def benchmark_subtitle(rows: list[dict[str, str | int | float]]) -> str:
    if not rows:
        return ""
    first = rows[0]
    problem = str(first["problem"]).replace("_", " ")
    reconstruction = str(first["reconstruction"]).upper()
    limiter = str(first["limiter"])
    tfinal = first["tfinal"]
    return f"{problem}; {reconstruction}, limiter={limiter}; t_final={tfinal}"


def timing_components(
    rows: list[dict[str, str | int | float]]
) -> tuple[list[float], list[float], list[float]]:
    hydro = [float(r["hydro_time_sec"]) for r in rows]
    cleaning = [float(r["cleaning_time_sec"]) for r in rows]
    diagnostics = [
        float(r["diagnostics_compute_time_sec"]) + float(r["output_time_sec"])
        for r in rows
    ]
    return hydro, cleaning, diagnostics


def normalized_timing_components(
    rows: list[dict[str, str | int | float]], context: str
) -> tuple[list[dict[str, str | int | float]], list[float], list[float], list[float]]:
    hydro, cleaning, diagnostics = timing_components(rows)
    kept_rows: list[dict[str, str | int | float]] = []
    hydro_fraction: list[float] = []
    cleaning_fraction: list[float] = []
    diagnostics_fraction: list[float] = []

    for row, h, c, d in zip(rows, hydro, cleaning, diagnostics):
        total_measured = h + c + d
        if total_measured <= 0.0:
            print(
                "Skipping normalized timing row "
                f"{context} {row['method']} {row['nx']}^2: "
                f"total_measured={total_measured}"
            )
            continue
        kept_rows.append(row)
        hydro_fraction.append(h / total_measured)
        cleaning_fraction.append(c / total_measured)
        diagnostics_fraction.append(d / total_measured)

    return kept_rows, hydro_fraction, cleaning_fraction, diagnostics_fraction


def draw_timing_stack(
    ax: plt.Axes,
    x: list[int],
    hydro: list[float],
    cleaning: list[float],
    diagnostics: list[float],
) -> None:
    ax.bar(x, hydro, label="hydro/RK/HLLD", color=TIMING_COLORS["hydro"])
    ax.bar(
        x,
        cleaning,
        bottom=hydro,
        label="cleaning",
        color=TIMING_COLORS["cleaning"],
    )
    bottoms = [h + c for h, c in zip(hydro, cleaning)]
    ax.bar(
        x,
        diagnostics,
        bottom=bottoms,
        label="diagnostics/output",
        color=TIMING_COLORS["diagnostics"],
    )


def resolution_values(rows: list[dict[str, str | int | float]]) -> list[int]:
    return sorted({int(r["nx"]) for r in rows if int(r["nx"]) == int(r["ny"])})


def plot_walltime(rows: list[dict[str, str | int | float]], figure_suffix: str = "") -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for method, group in group_rows(rows).items():
        xs = [int(r["ncell"]) for r in group]
        ys = [float(r["total_wall_time_sec"]) for r in group]
        ax.loglog(
            xs,
            ys,
            marker=METHOD_MARKERS.get(method, "o"),
            linewidth=1.8,
            markersize=5.5,
            label=method_label(method),
        )
    ax.set_xlabel("number of cells")
    ax.set_ylabel("wall-clock time [s]")
    ax.set_title(f"MHD runner performance scaling\n{benchmark_subtitle(rows)}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out = figure_path("performance_walltime_vs_cells", figure_suffix)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_throughput(rows: list[dict[str, str | int | float]], figure_suffix: str = "") -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for method, group in group_rows(rows).items():
        xs = [int(r["ncell"]) for r in group]
        ys = [float(r["cell_updates_per_second"]) for r in group]
        ax.semilogx(
            xs,
            ys,
            marker=METHOD_MARKERS.get(method, "o"),
            linewidth=1.8,
            markersize=5.5,
            label=method_label(method),
        )
    ax.set_xlabel("number of cells")
    ax.set_ylabel("cell updates per second")
    ax.set_title(f"MHD runner throughput\n{benchmark_subtitle(rows)}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out = figure_path("performance_cell_updates_per_second", figure_suffix)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_seconds_per_step(rows: list[dict[str, str | int | float]], figure_suffix: str = "") -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for method, group in group_rows(rows).items():
        xs = [int(r["ncell"]) for r in group]
        ys = [float(r["seconds_per_step"]) for r in group]
        ax.loglog(
            xs,
            ys,
            marker=METHOD_MARKERS.get(method, "o"),
            linewidth=1.8,
            markersize=5.5,
            label=method_label(method),
        )
    ax.set_xlabel("number of cells")
    ax.set_ylabel("seconds per accepted step")
    ax.set_title(f"Cost per accepted time step\n{benchmark_subtitle(rows)}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out = figure_path("performance_seconds_per_step", figure_suffix)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_breakdown(rows: list[dict[str, str | int | float]], figure_suffix: str = "") -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (int(r["ncell"]), method_sort_key(str(r["method"]))))
    labels = [f'{r["nx"]}^2\n{method_label(str(r["method"]))}' for r in rows]
    hydro, cleaning, diagnostics = timing_components(rows)

    fig, ax = plt.subplots(figsize=(max(8.0, 0.68 * len(rows)), 5.4))
    x = list(range(len(rows)))
    draw_timing_stack(ax, x, hydro, cleaning, diagnostics)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("time [s]")
    ax.set_title(f"Performance timing breakdown\n{benchmark_subtitle(rows)}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    out = figure_path("performance_method_breakdown", figure_suffix)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_breakdown_normalized(
    rows: list[dict[str, str | int | float]], figure_suffix: str = ""
) -> Path | None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: (int(r["ncell"]), method_sort_key(str(r["method"]))))
    (
        kept_rows,
        hydro_fraction,
        cleaning_fraction,
        diagnostics_fraction,
    ) = normalized_timing_components(sorted_rows, "overall")
    if not kept_rows:
        print("Skipping normalized timing breakdown: no rows with positive measured time")
        return None

    labels = [f'{r["nx"]}^2\n{method_label(str(r["method"]))}' for r in kept_rows]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.68 * len(kept_rows)), 5.4))
    x = list(range(len(kept_rows)))
    draw_timing_stack(ax, x, hydro_fraction, cleaning_fraction, diagnostics_fraction)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("fraction of measured time")
    ax.set_title(f"Normalized performance timing breakdown\n{benchmark_subtitle(kept_rows)}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    out = figure_path("performance_method_breakdown_normalized", figure_suffix)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_breakdown_for_resolution(
    rows: list[dict[str, str | int | float]], resolution: int, figure_suffix: str = ""
) -> Path | None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    expected_methods = sorted({str(r["method"]) for r in rows}, key=method_sort_key)
    selected = [
        r for r in rows
        if int(r["nx"]) == resolution and int(r["ny"]) == resolution
    ]
    selected.sort(key=lambda r: method_sort_key(str(r["method"])))
    if not selected:
        print(f"Skipping {resolution}^2 timing breakdown: no matching rows")
        return None

    present = {str(r["method"]) for r in selected}
    missing = [m for m in expected_methods if m not in present]
    if missing:
        labels = ", ".join(method_label(m) for m in missing)
        print(f"Skipping missing methods for {resolution}^2 timing breakdown: {labels}")

    labels = [method_label(str(r["method"])) for r in selected]
    hydro, cleaning, diagnostics = timing_components(selected)

    fig_width = max(6.8, 1.05 * len(selected) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    x = list(range(len(selected)))
    draw_timing_stack(ax, x, hydro, cleaning, diagnostics)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("time [s]")
    ax.set_title(
        f"Performance timing breakdown\n{benchmark_subtitle(selected)}; resolution={resolution}^2"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    out = figure_path("performance_method_breakdown", figure_suffix, resolution)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_breakdown_normalized_for_resolution(
    rows: list[dict[str, str | int | float]], resolution: int, figure_suffix: str = ""
) -> Path | None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    expected_methods = sorted({str(r["method"]) for r in rows}, key=method_sort_key)
    selected = [
        r for r in rows
        if int(r["nx"]) == resolution and int(r["ny"]) == resolution
    ]
    selected.sort(key=lambda r: method_sort_key(str(r["method"])))
    if not selected:
        print(f"Skipping normalized {resolution}^2 timing breakdown: no matching rows")
        return None

    present = {str(r["method"]) for r in selected}
    missing = [m for m in expected_methods if m not in present]
    if missing:
        labels = ", ".join(method_label(m) for m in missing)
        print(f"Skipping missing methods for normalized {resolution}^2 timing breakdown: {labels}")

    (
        kept_rows,
        hydro_fraction,
        cleaning_fraction,
        diagnostics_fraction,
    ) = normalized_timing_components(selected, f"{resolution}^2")
    if not kept_rows:
        print(
            f"Skipping normalized {resolution}^2 timing breakdown: "
            "no rows with positive measured time"
        )
        return None

    labels = [method_label(str(r["method"])) for r in kept_rows]
    fig_width = max(6.8, 1.05 * len(kept_rows) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    x = list(range(len(kept_rows)))
    draw_timing_stack(ax, x, hydro_fraction, cleaning_fraction, diagnostics_fraction)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("fraction of measured time")
    ax.set_title(
        "Normalized performance timing breakdown\n"
        f"{benchmark_subtitle(kept_rows)}; resolution={resolution}^2"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    out = figure_path("performance_method_breakdown_normalized", figure_suffix, resolution)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_breakdowns_by_resolution(
    rows: list[dict[str, str | int | float]], figure_suffix: str = ""
) -> list[Path]:
    paths: list[Path] = []
    for resolution in resolution_values(rows):
        path = plot_breakdown_for_resolution(rows, resolution, figure_suffix)
        if path is not None:
            paths.append(path)
    return paths


def plot_normalized_breakdowns_by_resolution(
    rows: list[dict[str, str | int | float]], figure_suffix: str = ""
) -> list[Path]:
    paths: list[Path] = []
    for resolution in resolution_values(rows):
        path = plot_breakdown_normalized_for_resolution(rows, resolution, figure_suffix)
        if path is not None:
            paths.append(path)
    return paths


def plot_cleaning_overhead(rows: list[dict[str, str | int | float]], figure_suffix: str = "") -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for method, group in group_rows(rows).items():
        xs = [int(r["ncell"]) for r in group]
        ys = []
        for r in group:
            total = finite_float(r["total_wall_time_sec"]) or 0.0
            cleaning = finite_float(r["cleaning_time_sec"]) or 0.0
            ys.append(cleaning / total if total > 0.0 else math.nan)
        ax.semilogx(
            xs,
            ys,
            marker=METHOD_MARKERS.get(method, "o"),
            linewidth=1.8,
            markersize=5.5,
            label=method_label(method),
        )
    ax.set_xlabel("number of cells")
    ax.set_ylabel("cleaning time / total wall time")
    ax.set_title(f"Divergence-cleaning overhead fraction\n{benchmark_subtitle(rows)}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out = figure_path("performance_cleaning_overhead_fraction", figure_suffix)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="divergence_advection")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["none", "mixed_glm", "mixed_eglm", "elliptic_projection"],
    )
    parser.add_argument("--reconstruction", default="plm", choices=["pcm", "plm"])
    parser.add_argument("--limiter", default="mc", choices=["minmod", "vanleer", "mc"])
    parser.add_argument("--tfinal", type=float, default=0.05)
    parser.add_argument("--diagnostic-stride", type=int, default=100)
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER))
    parser.add_argument(
        "--output-csv",
        default=str(PERF_DIR / "performance_scaling.csv"),
    )
    parser.add_argument(
        "--figure-suffix",
        default="",
        help="Append this suffix before .png, e.g. all_methods.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Keep successful benchmark rows if one case fails or reports failed status.",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.tfinal <= 0.0:
        raise SystemExit("--tfinal must be positive")
    if args.diagnostic_stride <= 0:
        raise SystemExit("--diagnostic-stride must be positive")
    if any(n <= 0 for n in args.resolutions):
        raise SystemExit("--resolutions must be positive")
    if len(set(args.resolutions)) < 3:
        print(
            "WARNING: fewer than 3 grid sizes; this is a sanity run, not a meaningful scaling study.",
            flush=True,
        )

    runner = Path(args.runner)
    if not runner.is_absolute():
        runner = ROOT / runner

    if not args.dry_run:
        ensure_build(args.skip_build, runner)

    rows: list[dict[str, str | int | float]] = []
    failures: list[str] = []
    for n in args.resolutions:
        for method in args.methods:
            prefix = benchmark_prefix(args.problem, args.reconstruction, n, method)
            cmd = [
                str(runner),
                "--performance-mode",
                "--no-snapshots",
                "--diagnostic-stride",
                str(args.diagnostic_stride),
                "--nx",
                str(n),
                "--ny",
                str(n),
                "--tfinal",
                str(args.tfinal),
                "--output-prefix",
                prefix,
                "--reconstruction",
                args.reconstruction,
                "--limiter",
                args.limiter,
                args.problem,
                method,
            ]
            result = run(cmd, dry_run=args.dry_run, check=not args.continue_on_failure)
            if result is not None and result.returncode != 0:
                message = f"{args.problem} {method} {n}^2 exited with code {result.returncode}"
                failures.append(message)
                print(f"WARNING: {message}")
                continue
            if args.dry_run:
                continue
            try:
                summary, summary_path = read_summary(prefix, method)
            except (FileNotFoundError, RuntimeError) as exc:
                if not args.continue_on_failure:
                    raise
                message = f"{args.problem} {method} {n}^2 missing summary: {exc}"
                failures.append(message)
                print(f"WARNING: {message}")
                continue

            row = make_performance_row(
                summary,
                summary_path,
                args.problem,
                method,
                args.reconstruction,
                args.limiter,
                n,
                args.tfinal,
                prefix,
            )
            if row["status"] != "finished":
                message = (
                    f"{args.problem} {row['method']} {n}^2 status={row['status']} "
                    f"failure_reason={row['failure_reason']}"
                )
                failures.append(message)
                print(f"WARNING: {message}")
                if args.continue_on_failure:
                    continue
            rows.append(row)

    if args.dry_run:
        return 0
    if not rows:
        raise SystemExit("no successful benchmark rows were collected")

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = ROOT / output_csv
    write_csv(rows, output_csv)
    plot_walltime(rows, args.figure_suffix)
    plot_throughput(rows, args.figure_suffix)
    plot_seconds_per_step(rows, args.figure_suffix)
    plot_breakdown(rows, args.figure_suffix)
    plot_breakdowns_by_resolution(rows, args.figure_suffix)
    plot_breakdown_normalized(rows, args.figure_suffix)
    plot_normalized_breakdowns_by_resolution(rows, args.figure_suffix)
    plot_cleaning_overhead(rows, args.figure_suffix)

    if failures:
        print("Benchmark failures/skips:")
        for failure in failures:
            print(f"  - {failure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
