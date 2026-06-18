#!/usr/bin/env python3
"""Generate a small set of report-ready figures from existing results.

This script is intentionally selective.  It avoids the broad visual-inventory
approach and only plots compatible datasets with a clear scientific purpose.
It does not run simulations.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = Path("results")
FIGURES = Path("figures")
OUT = FIGURES / "report_clean"
TRADEOFF_OUT = FIGURES / "mhd_runner" / "tradeoff"
PARETO_OUT = FIGURES / "mhd_runner" / "pareto"

CURRENT_METHODS = [
    "none",
    "hyperbolic_glm",
    "parabolic",
    "mixed_glm",
    "mixed_eglm",
    "gi_mixed_eglm",
    "elliptic_projection",
    "powell_source",
]

OT_PARETO_NO_PROJECTION_NO_POWELL_METHODS = [
    "none",
    "hyperbolic_glm",
    "parabolic",
    "mixed_glm",
    "mixed_eglm",
    "gi_mixed_eglm",
]

METHOD_ORDER = CURRENT_METHODS.copy()
REPORT_METHODS = CURRENT_METHODS.copy()
PRESSURE_METHODS = CURRENT_METHODS.copy()
PERFORMANCE_METHODS = CURRENT_METHODS.copy()
SCORECARD_METHODS = CURRENT_METHODS.copy()
SNAPSHOT_COMPARE_METHODS = CURRENT_METHODS.copy()

LABELS = {
    "none": "No cleaning",
    "hyperbolic_glm": "Hyperbolic GLM",
    "mixed_glm": "Mixed GLM",
    "parabolic": "Parabolic",
    "elliptic_projection": "Projection",
    "powell_source": "Powell",
    "mixed_eglm": "Mixed EGLM",
    "gi_mixed_eglm": "GI-Mixed EGLM",
}

COLORS = {
    "none": "#1f77b4",
    "hyperbolic_glm": "#ff7f0e",
    "mixed_glm": "#2ca02c",
    "parabolic": "#d62728",
    "elliptic_projection": "#9467bd",
    "powell_source": "#8c564b",
    "mixed_eglm": "#e377c2",
    "gi_mixed_eglm": "#bcbd22",
}

LINESTYLES = {
    "none": "-",
    "hyperbolic_glm": "--",
    "mixed_glm": "-.",
    "parabolic": ":",
    "elliptic_projection": "-",
    "powell_source": "--",
    "mixed_eglm": "-.",
    "gi_mixed_eglm": ":",
}

PROBLEM_PREFIX = {
    "orszag_tang": "mhd_ot",
    "field_loop": "mhd_fl",
    "divergence_advection": "mhd_da",
    "blast_wave": "mhd_blast",
}

PROBLEM_LABEL = {
    "orszag_tang": "Orszag-Tang",
    "field_loop": "Field-loop advection",
    "divergence_advection": "Divergence advection",
    "blast_wave": "Blast wave",
}

TARGET_FINAL_TIME = {
    "orszag_tang": 0.5,
    "field_loop": 0.5,
    "divergence_advection": 0.5,
    "blast_wave": 0.2,
}

PROBLEM_SCORECARD_LABEL = {
    "orszag_tang": "Orszag-Tang",
    "field_loop": "field-loop",
    "divergence_advection": "divergence-advection",
}

GENERATED: list[str] = []
SKIPPED: list[str] = []
PLOT_METHODS_INCLUDED: dict[str, list[str]] = {}
WARNED_INPUTS: set[str] = set()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.8),
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.2,
            "savefig.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def method_sort_key(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), method)


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        SKIPPED.append(f"missing {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        SKIPPED.append(f"could not read {path}: {exc}")
        return None


def warn_expected_csv(problem: str, method: str, path: Path, reason: str = "missing CSV") -> None:
    message = f"warning: {reason} for problem={problem} method={method} path={path}"
    if message in WARNED_INPUTS:
        return
    WARNED_INPUTS.add(message)
    print(message)
    SKIPPED.append(f"{problem}/{method}: {reason}: {path}")


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    GENERATED.append(str(png))
    GENERATED.append(str(pdf))


def save_png(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    GENERATED.append(str(path))


def finite_xy(df: pd.DataFrame, xcol: str, ycol: str, positive_y: bool = False):
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    if positive_y:
        mask &= y > 0.0
    return x[mask], y[mask]


def history_csv_path(problem: str, method: str) -> Path:
    prefix = PROBLEM_PREFIX[problem]
    return RESULTS / "mhd_runner" / "divergence" / f"{prefix}_{method}.csv"


def load_problem_history(problem: str, method: str, warn_missing: bool = False) -> pd.DataFrame | None:
    path = history_csv_path(problem, method)
    if not path.exists():
        if warn_missing:
            warn_expected_csv(problem, method, path)
            return None
        return read_csv(path)
    return read_csv(path)


def method_completed(problem: str, method: str) -> bool | None:
    summary = read_csv(FIGURES / "mhd_runner" / "data" / "mhd_runner_summary.csv")
    if summary is None or not {"problem", "method", "completed"}.issubset(summary.columns):
        return None
    row = summary[(summary["problem"] == problem) & (summary["method"] == method)]
    if row.empty:
        return None
    return bool(int(row.iloc[0]["completed"]))


def legend_below(ax: plt.Axes, ncol: int = 3) -> None:
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=ncol,
        frameon=False,
        handlelength=3.0,
    )


def plot_history_comparison(
    problem: str,
    methods: list[str],
    ycol: str,
    ylabel: str,
    title: str,
    stem: str,
    yscale: str = "linear",
    derived_energy_drift: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    plotted = []
    for method in methods:
        df = load_problem_history(problem, method, warn_missing=True)
        if df is None or df.empty or "time" not in df.columns:
            if df is not None and (df.empty or "time" not in df.columns):
                warn_expected_csv(problem, method, history_csv_path(problem, method), "invalid or empty history CSV")
            continue
        if derived_energy_drift:
            if "total_energy" not in df.columns:
                warn_expected_csv(problem, method, history_csv_path(problem, method), "history CSV lacks total_energy")
                continue
            energy = pd.to_numeric(df["total_energy"], errors="coerce")
            finite = energy[np.isfinite(energy)]
            if finite.empty:
                warn_expected_csv(problem, method, history_csv_path(problem, method), "history CSV has no finite total_energy")
                continue
            e0 = float(finite.iloc[0])
            df = df.copy()
            df["relative_energy_drift"] = (energy - e0) / max(abs(e0), 1.0e-30)
            plot_col = "relative_energy_drift"
        else:
            plot_col = ycol
        if plot_col not in df.columns:
            warn_expected_csv(problem, method, history_csv_path(problem, method), f"history CSV lacks {plot_col}")
            continue
        x, y = finite_xy(df, "time", plot_col, positive_y=(yscale == "log"))
        if x.empty:
            warn_expected_csv(problem, method, history_csv_path(problem, method), f"history CSV has no finite {plot_col}")
            continue
        ax.plot(
            x,
            y,
            label=LABELS.get(method, method),
            color=COLORS.get(method),
            linestyle=LINESTYLES.get(method, "-"),
        )
        plotted.append(method)

    if not plotted:
        plt.close(fig)
        SKIPPED.append(f"{stem}: no compatible data")
        return
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.28)
    if yscale == "log":
        ax.set_yscale("log")
    elif yscale == "symlog":
        ax.set_yscale("symlog", linthresh=1.0e-10)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    legend_below(ax, ncol=min(3, len(plotted)))
    save(fig, stem)
    PLOT_METHODS_INCLUDED[stem] = plotted
    print(f"{stem}: included methods: {', '.join(plotted)}")


def pick_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def load_run_summary(problem: str, method: str) -> pd.Series | None:
    prefix = PROBLEM_PREFIX[problem]
    path = RESULTS / "mhd_runner" / "summaries" / f"{prefix}_{method}_summary.csv"
    df = read_csv(path)
    if df is None or df.empty:
        return None
    return df.iloc[0]


def plot_ot_min_pressure() -> None:
    """Pressure positivity diagnostic matched to scripts/plot_pressure_diagnostics.py.

    The time-history files currently contain only ``min_pressure`` for OT, while
    the run summaries contain ``min_raw_pressure`` and failure metadata.  Plot the
    available time history and overlay raw-pressure failure markers when the run
    summary reports a failed hydro positivity check.
    """

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    plotted = []
    marked_failure = False

    for method in PRESSURE_METHODS:
        df = load_problem_history("orszag_tang", method, warn_missing=True)
        if df is None or df.empty or "time" not in df.columns:
            if df is not None and (df.empty or "time" not in df.columns):
                warn_expected_csv("orszag_tang", method, history_csv_path("orszag_tang", method), "invalid or empty history CSV")
            continue

        pcol = pick_existing_column(
            df,
            [
                "min_raw_pressure",
                "min_pressure",
                "min_p",
                "min_pressure_after_full_step",
            ],
        )
        if pcol is None:
            warn_expected_csv("orszag_tang", method, history_csv_path("orszag_tang", method), "history CSV lacks pressure column")
            continue

        x, y = finite_xy(df, "time", pcol)
        if x.empty:
            continue

        ax.plot(
            x,
            y,
            label=LABELS.get(method, method),
            color=COLORS.get(method),
            linestyle=LINESTYLES.get(method, "-"),
            linewidth=2.0,
        )
        plotted.append(method)

        summary = load_run_summary("orszag_tang", method)
        if summary is None:
            continue
        status = str(summary.get("status", "")).lower()
        failure_time = pd.to_numeric(pd.Series([summary.get("failure_time")]), errors="coerce").iloc[0]
        raw_failure_pressure = pd.to_numeric(
            pd.Series([summary.get("min_raw_pressure")]), errors="coerce"
        ).iloc[0]
        if status == "failed" and np.isfinite(failure_time) and np.isfinite(raw_failure_pressure):
            ax.scatter(
                [failure_time],
                [raw_failure_pressure],
                marker="x",
                s=70,
                color=COLORS.get(method),
                linewidth=2.0,
                zorder=5,
                label="failure marker" if not marked_failure else None,
            )
            marked_failure = True

    if not plotted:
        plt.close(fig)
        SKIPPED.append("ot_min_pressure: no pressure histories found")
        return

    ax.axhline(0.0, color="black", linewidth=1.2, alpha=0.8)
    ax.set_yscale("symlog", linthresh=1.0e-6)
    ax.set_xlabel("time")
    ax.set_ylabel(r"minimum raw pressure $\min(p)$")
    ax.set_title(
        "Orszag-Tang: pressure positivity diagnostic\n"
        r"$p < 0$ marks a hydro positivity failure"
    )
    ax.grid(True, which="both", alpha=0.3)
    legend_below(ax, ncol=3)
    save(fig, "ot_min_pressure")
    PLOT_METHODS_INCLUDED["ot_min_pressure"] = plotted
    print(f"ot_min_pressure: included methods: {', '.join(plotted)}")


def plot_figure_1_to_5() -> None:
    # Figure 1: report registry comparison. Failed methods still show the
    # available prefix of their history instead of disappearing from the legend.
    plot_history_comparison(
        "orszag_tang",
        REPORT_METHODS,
        "L2_norm_fv",
        r"normalized FV $L_2(\nabla\cdot B)$",
        "Orszag-Tang divergence control",
        "ot_divergence_main",
        yscale="log",
    )

    plot_ot_min_pressure()

    plot_history_comparison(
        "orszag_tang",
        REPORT_METHODS,
        "relative_energy_drift",
        "relative total-energy drift",
        "Orszag-Tang energy drift",
        "ot_energy_drift",
        yscale="symlog",
        derived_energy_drift=True,
    )

    plot_history_comparison(
        "field_loop",
        REPORT_METHODS,
        "L2_norm_fv",
        r"normalized FV $L_2(\nabla\cdot B)$",
        "Field-loop divergence control",
        "field_loop_divergence",
        yscale="log",
    )

    plot_history_comparison(
        "divergence_advection",
        REPORT_METHODS,
        "L2_norm_fv",
        r"normalized FV $L_2(\nabla\cdot B)$",
        "Divergence-advection cleaning",
        "divergence_advection_divergence",
        yscale="log",
    )


def load_mhd_runner_summary() -> pd.DataFrame | None:
    return read_csv(FIGURES / "mhd_runner" / "data" / "mhd_runner_summary.csv")


def refresh_performance_divergence_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if "diagnostic_file" not in df.columns:
        return df

    out = df.copy()
    for idx, row in out.iterrows():
        diag_value = row.get("diagnostic_file", "")
        if not isinstance(diag_value, str) or not diag_value:
            continue
        diag_path = Path(diag_value)
        if not diag_path.is_absolute():
            diag_path = Path(diag_value)
        if not diag_path.exists():
            continue
        try:
            div = pd.read_csv(diag_path)
        except Exception:
            continue
        if div.empty or not {"time", "L2_norm_fv", "Linf_norm_fv"}.issubset(div.columns):
            continue

        l2 = pd.to_numeric(div["L2_norm_fv"], errors="coerce")
        linf = pd.to_numeric(div["Linf_norm_fv"], errors="coerce")
        time = pd.to_numeric(div["time"], errors="coerce")

        out.at[idx, "final_L2_norm_fv"] = l2.iloc[-1]
        out.at[idx, "final_Linf_norm_fv"] = linf.iloc[-1]
        out.at[idx, "peak_L2_norm_fv"] = l2.max()
        out.at[idx, "peak_Linf_norm_fv"] = linf.max()

        valid = pd.DataFrame({"time": time, "l2": l2}).dropna()
        if len(valid) >= 2:
            out.at[idx, "time_integrated_L2_norm_fv"] = np.trapezoid(
                valid["l2"].to_numpy(),
                valid["time"].to_numpy(),
            )
        elif len(valid) == 1:
            out.at[idx, "time_integrated_L2_norm_fv"] = 0.0

    return out


def load_performance_table() -> pd.DataFrame | None:
    path = RESULTS / "mhd_runner" / "performance" / "performance_scaling_all_methods.csv"
    df = read_csv(path)
    if df is None or df.empty:
        return None
    df = refresh_performance_divergence_metrics(df)
    required = {"problem", "method", "nx", "total_wall_time_sec", "final_L2_norm_fv"}
    if not required.issubset(df.columns):
        SKIPPED.append("performance benchmark lacks required columns")
        return None
    df["problem"] = df["problem"].astype(str)
    df["method"] = df["method"].astype(str)
    return df


def select_largest_resolution_benchmark(df: pd.DataFrame, problem: str) -> pd.DataFrame:
    problem_df = df[df["problem"] == problem].copy()
    if problem_df.empty:
        return problem_df
    nx = pd.to_numeric(problem_df["nx"], errors="coerce")
    if nx.dropna().empty:
        return problem_df.iloc[0:0].copy()
    max_n = int(nx.max())
    problem_df = problem_df[nx == max_n].copy()
    problem_df = problem_df[problem_df["method"].isin(PERFORMANCE_METHODS)].copy()
    problem_df["_order"] = problem_df["method"].map(lambda m: method_sort_key(m)[0])
    problem_df = problem_df.sort_values(["_order", "method"])
    return problem_df.drop_duplicates(subset=["method"], keep="last")


def load_performance_benchmark() -> pd.DataFrame | None:
    df = load_performance_table()
    if df is None or df.empty:
        return None
    problem = "divergence_advection" if "divergence_advection" in set(df["problem"]) else str(df["problem"].iloc[0])
    df = select_largest_resolution_benchmark(df, problem)
    if df.empty:
        return None
    return df.sort_values("_order")


def save_scorecard_figure(df: pd.DataFrame, problem: str, stem: str) -> None:
    required = {"final_L2_norm_fv", "min_pressure", "energy_drift", "total_wall_time_sec", "method"}
    if not required.issubset(df.columns):
        SKIPPED.append(f"{stem}: missing required scorecard columns")
        return

    metrics = [
        ("final_L2_norm_fv", r"final normalized $L_2(\nabla\cdot B)$", "log"),
        ("min_pressure", "minimum pressure", "linear"),
        ("energy_drift_abs", r"$|\Delta E/E_0|$", "log"),
        ("total_wall_time_sec", "wall time [s]", "log"),
    ]
    df["energy_drift_abs"] = pd.to_numeric(df.get("energy_drift", np.nan), errors="coerce").abs()
    df = df[df["method"].isin(SCORECARD_METHODS)].copy()
    df["_order"] = df["method"].map(lambda m: method_sort_key(m)[0])
    df = df.sort_values("_order")
    methods = list(df["method"])
    x = np.arange(len(df))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    for ax, (col, ylabel, scale) in zip(axes.ravel(), metrics):
        vals = pd.to_numeric(df[col], errors="coerce")
        colors = [COLORS.get(m, "#777777") for m in methods]
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        if scale == "log":
            positive = vals[vals > 0]
            if not positive.empty:
                ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", which="both", alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in methods], rotation=35, ha="right")
    title_problem = PROBLEM_SCORECARD_LABEL.get(problem, problem.replace("_", "-"))
    fig.suptitle(f"Method scorecard: {title_problem} performance benchmark", y=0.98)
    fig.text(
        0.5,
        0.01,
        f"Benchmark for all panels: {problem}, PLM/van Leer, largest available resolution in performance_scaling_all_methods.csv.",
        ha="center",
        fontsize=9,
    )
    save(fig, stem)
    PLOT_METHODS_INCLUDED[stem] = methods
    print(f"{stem}: included methods: {', '.join(methods)}")


def plot_scorecard() -> None:
    df = load_performance_table()
    if df is None or df.empty:
        SKIPPED.append("method_scorecard: no compatible performance benchmark")
        return
    problems = sorted(df["problem"].dropna().unique())
    for problem in problems:
        problem_df = select_largest_resolution_benchmark(df, str(problem))
        if problem_df.empty:
            SKIPPED.append(f"method_scorecard_{problem}: no largest-resolution rows")
            continue
        missing = [method for method in SCORECARD_METHODS if method not in set(problem_df["method"])]
        if missing:
            labels = ", ".join(LABELS.get(method, method) for method in missing)
            SKIPPED.append(f"method_scorecard_{problem}: missing methods skipped: {labels}")
        stem = f"method_scorecard_{problem}"
        save_scorecard_figure(problem_df, str(problem), stem)
        if problem == "divergence_advection":
            save_scorecard_figure(problem_df, str(problem), "method_scorecard")


def plot_divergence_energy_tradeoff() -> None:
    summary = load_mhd_runner_summary()
    if summary is None:
        return
    df = summary[(summary["problem"] == "orszag_tang")].copy()
    df = df[df["method"].isin(REPORT_METHODS)]
    if "completed" in df.columns:
        df = df[pd.to_numeric(df["completed"], errors="coerce") == 1].copy()
    for col in ["final_L2_norm_fv", "energy_drift"]:
        if col not in df.columns:
            SKIPPED.append(f"divergence_energy_tradeoff: missing {col}")
            return
    df["_order"] = df["method"].map(lambda m: method_sort_key(str(m))[0])
    df = df.sort_values("_order")
    x = pd.to_numeric(df["final_L2_norm_fv"], errors="coerce")
    y = pd.to_numeric(df["energy_drift"], errors="coerce").abs()
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    df, x, y = df[mask], x[mask], y[mask]
    if df.empty:
        SKIPPED.append("divergence_energy_tradeoff: no finite completed OT data")
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    label_floor = {
        "hyperbolic_glm": 2.0e-14,
        "mixed_glm": 7.0e-15,
    }
    offsets = {
        "none": (-8, 3),
        "hyperbolic_glm": (6, 8),
        "mixed_glm": (6, -2),
        "parabolic": (6, 3),
        "elliptic_projection": (6, 3),
    }
    horizontal_alignment = {"none": "right"}
    for _, row in df.iterrows():
        method = str(row["method"])
        yplot = max(abs(float(row["energy_drift"])), 1.0e-14)
        yplot = label_floor.get(method, yplot)
        ax.scatter(
            row["final_L2_norm_fv"],
            yplot,
            s=85,
            color=COLORS.get(method),
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            LABELS.get(method, method),
            (row["final_L2_norm_fv"], yplot),
            xytext=offsets.get(method, (6, 4)),
            textcoords="offset points",
            fontsize=8.5,
            ha=horizontal_alignment.get(method, "left"),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(x.min()) * 0.35, float(x.max()) * 2.0)
    ax.set_ylim(3.0e-15, 2.0e-1)
    ax.set_xlabel(r"final normalized FV $L_2(\nabla\cdot B)$")
    ax.set_ylabel(r"$|\Delta E/E_0|$ (floor $10^{-14}$)")
    ax.set_title("Orszag-Tang accuracy-conservation trade-off")
    ax.grid(True, which="both", alpha=0.28)
    save(fig, "divergence_energy_tradeoff")
    included = [str(method) for method in df["method"]]
    PLOT_METHODS_INCLUDED["divergence_energy_tradeoff"] = included
    print(f"divergence_energy_tradeoff: included methods: {', '.join(included)}")


def plot_accuracy_cost_pareto() -> None:
    df = load_performance_benchmark()
    if df is None or df.empty:
        SKIPPED.append("accuracy_cost_pareto: no compatible performance benchmark")
        return
    needed = {"total_wall_time_sec", "final_L2_norm_fv", "energy_drift", "method"}
    if not needed.issubset(df.columns):
        SKIPPED.append("accuracy_cost_pareto: missing required performance columns")
        return
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    for _, row in df.iterrows():
        method = str(row["method"])
        x = float(row["total_wall_time_sec"])
        y = float(row["final_L2_norm_fv"])
        if not (np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0):
            continue
        drift = abs(float(row.get("energy_drift", 0.0))) if np.isfinite(row.get("energy_drift", np.nan)) else 0.0
        size = 70.0 + 900.0 * min(drift, 0.01)
        ax.scatter(
            x,
            y,
            s=size,
            color=COLORS.get(method),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
            zorder=3,
        )
        ax.scatter([], [], s=70, color=COLORS.get(method), edgecolor="black", linewidth=0.5, label=LABELS.get(method, method))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall time [s]")
    ax.set_ylabel(r"final normalized FV $L_2(\nabla\cdot B)$")
    ax.set_title("Accuracy-cost Pareto view")
    ax.grid(True, which="both", alpha=0.28)
    ax.text(
        0.02,
        0.02,
        "Benchmark: divergence_advection, largest available N; marker size scales with |energy drift|.",
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
    )
    legend_below(ax, ncol=3)
    save(fig, "accuracy_cost_pareto")
    included = [str(method) for method in df["method"]]
    PLOT_METHODS_INCLUDED["accuracy_cost_pareto"] = included
    print(f"accuracy_cost_pareto: included methods: {', '.join(included)}")


def clean_pareto_data(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    needed = {"total_wall_time_sec", "final_L2_norm_fv", "energy_drift", "method"}
    if not needed.issubset(df.columns):
        return df.iloc[0:0].copy()
    out = df[df["method"].isin(methods)].copy()
    out["total_wall_time_sec"] = pd.to_numeric(
        out["total_wall_time_sec"], errors="coerce"
    )
    out["final_L2_norm_fv"] = pd.to_numeric(out["final_L2_norm_fv"], errors="coerce")
    out["energy_drift"] = pd.to_numeric(out["energy_drift"], errors="coerce")
    out = out[
        np.isfinite(out["total_wall_time_sec"])
        & np.isfinite(out["final_L2_norm_fv"])
        & (out["total_wall_time_sec"] > 0.0)
        & (out["final_L2_norm_fv"] > 0.0)
    ].copy()
    out["_order"] = out["method"].map(lambda m: method_sort_key(str(m))[0])
    return out.sort_values(["_order", "method"])


def plot_accuracy_cost_pareto_variant(
    df: pd.DataFrame,
    problem: str,
    *,
    include_projection: bool,
) -> None:
    methods = CURRENT_METHODS.copy()
    projection_label = "with Projection"
    output_suffix = "with_projection"
    if not include_projection:
        methods = [method for method in methods if method != "elliptic_projection"]
        projection_label = "without Projection"
        output_suffix = "without_projection"

    plot_df = clean_pareto_data(df, methods)
    stem = f"pareto_{problem}_{output_suffix}"
    if plot_df.empty:
        SKIPPED.append(f"{stem}: no finite largest-resolution rows")
        return

    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    for _, row in plot_df.iterrows():
        method = str(row["method"])
        drift = row.get("energy_drift", np.nan)
        drift_abs = abs(float(drift)) if np.isfinite(drift) else 0.0
        size = 70.0 + 900.0 * min(drift_abs, 0.01)
        ax.scatter(
            row["total_wall_time_sec"],
            row["final_L2_norm_fv"],
            s=size,
            color=COLORS.get(method),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
            zorder=3,
        )
        ax.scatter(
            [],
            [],
            s=70,
            color=COLORS.get(method),
            edgecolor="black",
            linewidth=0.5,
            label=LABELS.get(method, method),
        )

    nx_values = pd.to_numeric(plot_df["nx"], errors="coerce").dropna().astype(int)
    resolution = int(nx_values.iloc[0]) if not nx_values.empty else -1
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall time [s]")
    ax.set_ylabel(r"final normalized FV $L_2(\nabla\cdot B)$")
    ax.set_title(f"Accuracy-cost Pareto view -- {problem} ({projection_label})")
    ax.grid(True, which="both", alpha=0.28)
    ax.text(
        0.02,
        0.02,
        "largest available N; marker size scales with |energy drift|.",
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
    )
    legend_below(ax, ncol=3)

    out = TRADEOFF_OUT / f"{stem}.png"
    save_png(fig, out)
    included = [str(method) for method in plot_df["method"]]
    PLOT_METHODS_INCLUDED[stem] = included
    print(f"{stem}: selected largest N = {resolution}")
    print(f"{stem}: included methods: {', '.join(included)}")
    diagnostic_files = [
        str(value)
        for value in plot_df.get("diagnostic_file", pd.Series(dtype=object)).dropna()
        if str(value)
    ]
    if diagnostic_files:
        print(f"{stem}: diagnostic CSVs used for final divergence:")
        for path in diagnostic_files:
            print(f"  {path}")


def plot_problem_accuracy_cost_paretos() -> None:
    df = load_performance_table()
    if df is None or df.empty:
        SKIPPED.append("problem accuracy-cost paretos: no compatible performance benchmark")
        return

    print(
        "Problem-specific Pareto performance CSV: "
        + str(RESULTS / "mhd_runner" / "performance" / "performance_scaling_all_methods.csv")
    )
    for problem in ["divergence_advection", "field_loop", "orszag_tang"]:
        problem_df = select_largest_resolution_benchmark(df, problem)
        if problem_df.empty:
            SKIPPED.append(f"pareto_{problem}: no largest-resolution rows")
            continue
        for include_projection in (True, False):
            plot_accuracy_cost_pareto_variant(
                problem_df,
                problem,
                include_projection=include_projection,
            )


def plot_ot_pareto_without_projection_and_powell() -> None:
    df = load_performance_table()
    stem = "accuracy_cost_pareto_orszag_tang_no_projection_no_powell"
    if df is None or df.empty:
        SKIPPED.append(f"{stem}: no compatible performance benchmark")
        return

    problem = "orszag_tang"
    problem_df = select_largest_resolution_benchmark(df, problem)
    plot_df = clean_pareto_data(
        problem_df,
        OT_PARETO_NO_PROJECTION_NO_POWELL_METHODS,
    )
    if plot_df.empty:
        SKIPPED.append(f"{stem}: no finite largest-resolution rows")
        return

    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    for _, row in plot_df.iterrows():
        method = str(row["method"])
        drift = row.get("energy_drift", np.nan)
        drift_abs = abs(float(drift)) if np.isfinite(drift) else 0.0
        size = 70.0 + 900.0 * min(drift_abs, 0.01)
        ax.scatter(
            row["total_wall_time_sec"],
            row["final_L2_norm_fv"],
            s=size,
            color=COLORS.get(method),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
            zorder=3,
        )
        ax.scatter(
            [],
            [],
            s=70,
            color=COLORS.get(method),
            edgecolor="black",
            linewidth=0.5,
            label=LABELS.get(method, method),
        )

    nx_values = pd.to_numeric(plot_df["nx"], errors="coerce").dropna().astype(int)
    resolution = int(nx_values.iloc[0]) if not nx_values.empty else -1
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall time [s]")
    ax.set_ylabel(r"final normalized FV $L_2(\nabla\cdot B)$")
    ax.set_title("Accuracy-cost Pareto view -- orszag_tang (without Projection and Powell)")
    ax.grid(True, which="both", alpha=0.28)
    ax.text(
        0.02,
        0.02,
        "largest available N; marker size scales with |energy drift|.",
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
    )
    legend_below(ax, ncol=3)

    out = PARETO_OUT / f"{stem}.png"
    save_png(fig, out)

    included = [str(method) for method in plot_df["method"]]
    PLOT_METHODS_INCLUDED[stem] = included
    print(
        f"{stem}: performance CSV used: "
        f"{RESULTS / 'mhd_runner' / 'performance' / 'performance_scaling_all_methods.csv'}"
    )
    print(f"{stem}: selected problem = {problem}")
    print(f"{stem}: selected largest N = {resolution}")
    print(f"{stem}: included methods: {', '.join(included)}")
    diagnostic_files = [
        str(value)
        for value in plot_df.get("diagnostic_file", pd.Series(dtype=object)).dropna()
        if str(value)
    ]
    if diagnostic_files:
        print(f"{stem}: diagnostic CSVs used for final divergence:")
        for path in diagnostic_files:
            print(f"  {path}")


def plot_robustness_heatmap() -> None:
    summary = load_mhd_runner_summary()
    if summary is None:
        return
    required = {"problem", "method", "completed"}
    if not required.issubset(summary.columns):
        SKIPPED.append("robustness heatmap: mhd_runner_summary lacks required columns")
        return
    df = summary[summary["method"].isin(METHOD_ORDER)].copy()
    table = df.pivot_table(
        index="problem",
        columns="method",
        values="completed",
        aggfunc="last",
    )
    problems = ["orszag_tang", "field_loop", "divergence_advection"]
    methods = METHOD_ORDER
    table = table.reindex(index=problems, columns=methods)

    fig, ax = plt.subplots(figsize=(1.0 * len(methods) + 2.5, 3.4))
    arr = table.to_numpy(dtype=float)
    plot_arr = np.where(np.isnan(arr), -1.0, arr)
    cmap = mcolors.ListedColormap(["#bdbdbd", "#d95f02", "#1b9e77"])
    norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    im = ax.imshow(plot_arr, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([LABELS[m] for m in methods], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(problems)))
    ax.set_yticklabels([PROBLEM_LABEL[p] for p in problems])
    ax.set_title("Robustness by problem and method")
    for j in range(len(problems)):
        for i in range(len(methods)):
            value = arr[j, i]
            if np.isnan(value):
                text = "N/A"
                color = "black"
            else:
                text = "pass" if value >= 0.5 else "fail"
                color = "white"
            ax.text(i, j, text, ha="center", va="center", fontsize=8, color=color, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["N/A / no data", "failed", "completed"])
    save(fig, "robustness_problem_method")
    PLOT_METHODS_INCLUDED["robustness_problem_method"] = methods
    print(f"robustness_problem_method: included methods: {', '.join(methods)}")


def snapshot_csv_path(stem: str) -> Path:
    return RESULTS / "mhd_runner" / "snapshots" / f"{stem}_final.csv"


def load_snapshot(stem: str) -> pd.DataFrame | None:
    path = snapshot_csv_path(stem)
    return read_csv(path)


def as_summary_float(summary: pd.Series | None, key: str) -> float:
    if summary is None:
        return float("nan")
    value = pd.to_numeric(pd.Series([summary.get(key)]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else float("nan")


def summary_bool(summary: pd.Series | None, key: str) -> bool | None:
    value = as_summary_float(summary, key)
    if not np.isfinite(value):
        return None
    return bool(int(round(value)))


def last_history_time(problem: str, method: str) -> float:
    path = history_csv_path(problem, method)
    if not path.exists():
        return float("nan")
    try:
        df = pd.read_csv(path, usecols=["time"])
    except Exception:
        return float("nan")
    if df.empty:
        return float("nan")
    values = pd.to_numeric(df["time"], errors="coerce")
    finite = values[np.isfinite(values)]
    return float(finite.iloc[-1]) if not finite.empty else float("nan")


def format_time(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    return f"{value:.4g}"


def snapshot_run_state(problem: str, method: str) -> dict[str, object]:
    summary = load_run_summary(problem, method)
    status = str(summary.get("status", "unknown")).strip().lower() if summary is not None else "unknown"
    final_time_reached = summary_bool(summary, "final_time_reached")
    failure_time = as_summary_float(summary, "failure_time")
    history_time = last_history_time(problem, method)
    stop_time = failure_time if np.isfinite(failure_time) else history_time
    snapshot_write_time = as_summary_float(summary, "snapshot_write_time_sec")
    performance_mode = summary_bool(summary, "performance_mode")
    target_time = TARGET_FINAL_TIME.get(problem, float("nan"))
    success = status == "finished" and final_time_reached is True
    reason = str(summary.get("failure_reason", "")).strip() if summary is not None else ""
    return {
        "summary": summary,
        "status": status,
        "success": success,
        "final_time_reached": final_time_reached,
        "failure_time": failure_time,
        "history_time": history_time,
        "stop_time": stop_time,
        "snapshot_write_time": snapshot_write_time,
        "performance_mode": performance_mode,
        "target_time": target_time,
        "reason": reason,
    }


def load_final_snapshot_for_plot(problem: str, method: str) -> tuple[pd.DataFrame | None, str | None, dict[str, object]]:
    prefix = PROBLEM_PREFIX[problem]
    stem = f"{prefix}_{method}"
    path = snapshot_csv_path(stem)
    state = snapshot_run_state(problem, method)

    if not bool(state["success"]):
        stop_time = state["stop_time"]
        placeholder = f"FAILED\nt={format_time(float(stop_time))}"
        if path.exists():
            SKIPPED.append(
                f"snapshot {problem}/{method}: existing {path} ignored because "
                f"status={state['status']} final_time_reached={state['final_time_reached']}"
            )
        return None, placeholder, state

    if not path.exists():
        if state["performance_mode"] is True or (
            np.isfinite(float(state["snapshot_write_time"]))
            and float(state["snapshot_write_time"]) <= 0.0
        ):
            reason = "missing snapshot CSV; generation workflow did not request snapshots"
        else:
            reason = "missing snapshot CSV for successful final-time run"
        warn_expected_csv(problem, method, path, reason)
        return None, "N/A", state

    df = load_snapshot(stem)
    if df is None or df.empty:
        warn_expected_csv(problem, method, path, "invalid or empty snapshot CSV")
        return None, "N/A", state
    return add_snapshot_derived(df), None, state


def snapshot_grid(df: pd.DataFrame, field: str):
    if field not in df.columns:
        return None
    nx = int(df["i"].max()) + 1
    ny = int(df["j"].max()) + 1
    ordered = df.sort_values(["j", "i"])
    z = pd.to_numeric(ordered[field], errors="coerce").to_numpy().reshape(ny, nx)
    x = ordered[ordered["j"] == ordered["j"].min()].sort_values("i")["x"].to_numpy()
    y = ordered[ordered["i"] == ordered["i"].min()].sort_values("j")["y"].to_numpy()
    return x, y, z


def add_snapshot_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "divB_fv" in out.columns:
        out["abs_divB_fv"] = np.abs(pd.to_numeric(out["divB_fv"], errors="coerce"))
        if "Bmag" in out.columns:
            bmag = pd.to_numeric(out["Bmag"], errors="coerce").abs()
        elif {"Bx", "By", "Bz"}.issubset(out.columns):
            bx = pd.to_numeric(out["Bx"], errors="coerce")
            by = pd.to_numeric(out["By"], errors="coerce")
            bz = pd.to_numeric(out["Bz"], errors="coerce")
            bmag = np.sqrt(bx * bx + by * by + bz * bz)
        elif {"Bx", "By"}.issubset(out.columns):
            bx = pd.to_numeric(out["Bx"], errors="coerce")
            by = pd.to_numeric(out["By"], errors="coerce")
            bmag = np.sqrt(bx * bx + by * by)
        else:
            bmag = None
        if bmag is not None:
            xs = np.sort(out["x"].unique())
            ys = np.sort(out["y"].unique())
            dx = float(np.median(np.diff(xs))) if len(xs) > 1 else 1.0
            dy = float(np.median(np.diff(ys))) if len(ys) > 1 else 1.0
            h = math.sqrt(dx * dy)
            out["norm_abs_divB_fv"] = h * out["abs_divB_fv"] / (np.asarray(bmag) + 1.0e-30)
    return out


def annotate_missing_panel(
    ax: plt.Axes,
    text: str = "N/A",
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] = (0.0, 1.0),
) -> None:
    ax.set_facecolor("#f1f1f1")
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#555555",
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cfcfcf")


def plot_snapshot_comparison(
    problem: str,
    methods: list[str] | None = None,
    output_stem: str | None = None,
) -> None:
    prefix = PROBLEM_PREFIX[problem]
    methods = methods or SNAPSHOT_COMPARE_METHODS
    output_stem = output_stem or f"snapshot_compare_{problem}"
    loaded: list[dict[str, object]] = []
    missing: list[str] = []
    failed: list[str] = []
    for method in methods:
        df, placeholder, state = load_final_snapshot_for_plot(problem, method)
        if df is None:
            loaded.append(
                {
                    "method": method,
                    "df": None,
                    "placeholder": placeholder or "N/A",
                    "state": state,
                }
            )
            if placeholder and placeholder.startswith("FAILED"):
                failed.append(method)
            else:
                missing.append(method)
        else:
            loaded.append(
                {
                    "method": method,
                    "df": df,
                    "placeholder": None,
                    "state": state,
                }
            )
    rows = [
        ("rho", r"density $\rho$", "viridis", "linear"),
        ("p", r"pressure $p$", "plasma", "linear"),
        ("norm_abs_divB_fv", r"normalized $|\nabla\cdot B|$", "magma", "log"),
    ]
    ncols = len(methods)
    fig_width = 2.6 * ncols + 1.4
    fig_height = 2.55 * len(rows) + 1.1
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(
        len(rows),
        len(loaded) + 1,
        width_ratios=[*[1.0 for _ in loaded], 0.06],
        wspace=0.06,
        hspace=0.08,
    )
    axes = np.empty((len(rows), len(loaded)), dtype=object)
    caxes = []
    for r in range(len(rows)):
        for c in range(len(loaded)):
            axes[r, c] = fig.add_subplot(gs[r, c])
        caxes.append(fig.add_subplot(gs[r, len(loaded)]))
    for r, (field, label, cmap, scale) in enumerate(rows):
        grids = []
        valid_positions = []
        for c, item in enumerate(loaded):
            df = item["df"]
            if df is None:
                continue
            grid = snapshot_grid(df, field)
            if grid is None:
                continue
            grids.append(grid)
            valid_positions.append(c)
        values = np.concatenate([z[np.isfinite(z)].ravel() for _, _, z in grids]) if grids else np.array([])
        values = values[np.isfinite(values)]
        if values.size == 0:
            for c, item in enumerate(loaded):
                method = str(item["method"])
                placeholder = str(item.get("placeholder") or "N/A")
                annotate_missing_panel(axes[r, c], placeholder)
                if r == 0:
                    axes[r, c].set_title(LABELS[method], fontsize=17)
                if c == 0:
                    axes[r, c].set_ylabel(label + "\n$y$", fontsize=16)
                if r == len(rows) - 1:
                    axes[r, c].set_xlabel("$x$", fontsize=15)
            caxes[r].axis("off")
            SKIPPED.append(f"{output_stem}: no finite values for {field}")
            continue
        if scale == "log":
            positive = values[values > 0]
            if positive.size == 0:
                norm = mcolors.LogNorm(vmin=1.0e-14, vmax=1.0)
            else:
                vmin = max(float(positive.min()), 1.0e-14)
                vmax = max(float(positive.max()), vmin * 10.0)
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=float(values.min()), vmax=float(values.max()))
        last_im = None
        grid_by_position = dict(zip(valid_positions, grids))
        xlim = (
            min(float(np.nanmin(x)) for x, _, _ in grids),
            max(float(np.nanmax(x)) for x, _, _ in grids),
        )
        ylim = (
            min(float(np.nanmin(y)) for _, y, _ in grids),
            max(float(np.nanmax(y)) for _, y, _ in grids),
        )
        for c, item in enumerate(loaded):
            method = str(item["method"])
            ax = axes[r, c]
            if c not in grid_by_position:
                placeholder = str(loaded[c].get("placeholder") or "N/A")
                annotate_missing_panel(ax, placeholder, xlim=xlim, ylim=ylim)
                if r == 0:
                    ax.set_title(LABELS[method], fontsize=17)
                if c == 0:
                    ax.set_ylabel(label + "\n$y$", fontsize=16)
                if r == len(rows) - 1:
                    ax.set_xlabel("$x$", fontsize=15)
                continue
            x, y, z = grid_by_position[c]
            if scale == "log":
                z = np.where(z > 0.0, z, norm.vmin)
            last_im = ax.pcolormesh(x, y, z, shading="auto", cmap=cmap, norm=norm)
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(LABELS[method], fontsize=17)
            if c == 0:
                ax.set_ylabel(label + "\n$y$", fontsize=16)
            else:
                ax.set_yticklabels([])
            if r == len(rows) - 1:
                ax.set_xlabel("$x$", fontsize=15)
            else:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=12)
        if last_im is not None:
            cbar = fig.colorbar(last_im, cax=caxes[r], label=label)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label(label, fontsize=15)
        else:
            caxes[r].axis("off")
    fig.suptitle(f"{PROBLEM_LABEL[problem]} final snapshots", fontsize=20)
    if missing:
        labels = ", ".join(LABELS[m] for m in missing)
        SKIPPED.append(f"{output_stem}: missing snapshots for {labels}")
    if failed:
        labels = ", ".join(
            f"{LABELS[m]} at t={format_time(float(snapshot_run_state(problem, m)['stop_time']))}"
            for m in failed
        )
        SKIPPED.append(f"{output_stem}: failed/early snapshots shown as placeholders for {labels}")
    save(fig, output_stem)
    included = [str(item["method"]) for item in loaded if item["df"] is not None]
    PLOT_METHODS_INCLUDED[output_stem] = included
    print(f"{output_stem}: included methods: {', '.join(included)}")


def plot_snapshot_comparisons() -> None:
    for problem in ["divergence_advection", "field_loop", "orszag_tang", "blast_wave"]:
        plot_snapshot_comparison(problem)


def plot_ot_snapshot_panel() -> None:
    plot_snapshot_comparison(
        "orszag_tang",
        methods=["none", "mixed_glm", "elliptic_projection"],
        output_stem="ot_snapshot_density_pressure_divb",
    )


def plot_projection_diagnostics() -> None:
    df = load_problem_history("orszag_tang", "elliptic_projection")
    if df is None or df.empty:
        return
    cols = [
        ("projection_iterations_step", "SOR iterations / step", "linear"),
        ("projection_true_residual", "projection true residual", "log"),
        ("L2_norm_fv", r"normalized FV $L_2(\nabla\cdot B)$", "log"),
    ]
    if not all(col in df.columns for col, _, _ in cols):
        SKIPPED.append("projection diagnostics: missing projection columns")
        return
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.0), sharex=True)
    for ax, (col, ylabel, scale) in zip(axes, cols):
        x, y = finite_xy(df, "time", col, positive_y=(scale == "log"))
        ax.plot(x, y, color=COLORS["elliptic_projection"])
        if scale == "log":
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.28)
    axes[-1].set_xlabel("time")
    axes[0].set_title("Projection diagnostics on Orszag-Tang")
    save(fig, "projection_diagnostics")


def plot_convergence_scaling() -> None:
    path = RESULTS / "mhd_runner" / "convergence" / "divergence_advection_convergence.csv"
    df = read_csv(path)
    if df is None or df.empty:
        return
    required = {"reconstruction", "dx", "final_L2_norm_fv"}
    if not required.issubset(df.columns):
        SKIPPED.append("convergence: divergence-advection convergence CSV lacks required columns")
        return
    fig, ax = plt.subplots(figsize=(7.3, 5.0))
    for reconstruction, group in df.groupby("reconstruction"):
        group = group.copy()
        group["dx"] = pd.to_numeric(group["dx"], errors="coerce")
        group["final_L2_norm_fv"] = pd.to_numeric(group["final_L2_norm_fv"], errors="coerce")
        group = group[np.isfinite(group["dx"]) & np.isfinite(group["final_L2_norm_fv"]) & (group["dx"] > 0) & (group["final_L2_norm_fv"] > 0)]
        group = group.sort_values("dx")
        if len(group) < 2:
            continue
        coeff = np.polyfit(np.log(group["dx"]), np.log(group["final_L2_norm_fv"]), 1)
        ax.loglog(
            group["dx"],
            group["final_L2_norm_fv"],
            marker="o",
            linewidth=2.2,
            label=f"{reconstruction.upper()} empirical slope={coeff[0]:.2f}",
        )
    ax.invert_xaxis()
    ax.set_xlabel("grid spacing $\\Delta x$")
    ax.set_ylabel(r"final normalized FV $L_2(\nabla\cdot B)$")
    ax.set_title("Empirical divergence scaling")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=False)
    save(fig, "convergence_divergence_scaling")


def write_run_summary() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Report Clean Figure Generation", ""]
    lines.append("## Generated")
    for item in GENERATED:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Methods Included")
    if PLOT_METHODS_INCLUDED:
        for stem, methods in sorted(PLOT_METHODS_INCLUDED.items()):
            labels = ", ".join(methods)
            lines.append(f"- {stem}: {labels}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Skipped / Notes")
    if SKIPPED:
        for item in SKIPPED:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    (OUT / "REPORT_CLEAN_INDEX.md").write_text("\n".join(lines))


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    plot_figure_1_to_5()
    plot_scorecard()
    plot_divergence_energy_tradeoff()
    plot_accuracy_cost_pareto()
    plot_problem_accuracy_cost_paretos()
    plot_ot_pareto_without_projection_and_powell()
    plot_robustness_heatmap()
    plot_snapshot_comparisons()
    plot_ot_snapshot_panel()
    plot_projection_diagnostics()
    plot_convergence_scaling()
    write_run_summary()
    print(f"generated {len(GENERATED)} files")
    for item in GENERATED:
        print(item)
    if SKIPPED:
        print("skipped/notes:")
        for item in SKIPPED:
            print(f"- {item}")


if __name__ == "__main__":
    main()
