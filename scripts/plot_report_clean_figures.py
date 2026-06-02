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

METHOD_ORDER = [
    "none",
    "hyperbolic_glm",
    "mixed_glm",
    "parabolic",
    "elliptic_projection",
    "powell_source",
    "powell_source_limited",
    "mixed_eglm",
    "gi_mixed_eglm",
]

CORE_METHODS = [
    "none",
    "hyperbolic_glm",
    "mixed_glm",
    "parabolic",
    "elliptic_projection",
]

PRESSURE_METHODS = [
    "none",
    "hyperbolic_glm",
    "mixed_glm",
    "parabolic",
    "elliptic_projection",
    "powell_source",
    "powell_source_subcycled",
    "powell_source_limited",
]

GLM_FAMILY_METHODS = [
    "none",
    "hyperbolic_glm",
    "mixed_glm",
    "mixed_eglm",
    "gi_mixed_eglm",
]

PERFORMANCE_METHODS = [
    "none",
    "hyperbolic_glm",
    "mixed_glm",
    "mixed_eglm",
    "gi_mixed_eglm",
    "parabolic",
    "elliptic_projection",
    "powell_source",
    "powell_source_limited",
    "powell_source_subcycled",
]

SCORECARD_METHODS = [
    "none",
    "hyperbolic_glm",
    "mixed_glm",
    "parabolic",
    "elliptic_projection",
    "mixed_eglm",
    "gi_mixed_eglm",
]

LABELS = {
    "none": "No cleaning",
    "hyperbolic_glm": "Hyperbolic GLM",
    "mixed_glm": "Mixed GLM",
    "parabolic": "Parabolic",
    "elliptic_projection": "Projection",
    "powell_source": "Powell",
    "powell_source_limited": "Limited Powell",
    "powell_source_subcycled": "Subcycled Powell",
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
    "powell_source_limited": "#17becf",
    "powell_source_subcycled": "#7f7f7f",
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
    "powell_source_limited": "--",
    "powell_source_subcycled": "-.",
    "mixed_eglm": "-.",
    "gi_mixed_eglm": ":",
}

PROBLEM_PREFIX = {
    "orszag_tang": "mhd_ot",
    "field_loop": "mhd_fl",
    "divergence_advection": "mhd_da",
}

PROBLEM_LABEL = {
    "orszag_tang": "Orszag-Tang",
    "field_loop": "Field-loop advection",
    "divergence_advection": "Divergence advection",
}

GENERATED: list[str] = []
SKIPPED: list[str] = []


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


def finite_xy(df: pd.DataFrame, xcol: str, ycol: str, positive_y: bool = False):
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    if positive_y:
        mask &= y > 0.0
    return x[mask], y[mask]


def load_problem_history(problem: str, method: str) -> pd.DataFrame | None:
    prefix = PROBLEM_PREFIX[problem]
    return read_csv(RESULTS / "mhd_runner" / "divergence" / f"{prefix}_{method}.csv")


def method_completed(problem: str, method: str) -> bool | None:
    summary = read_csv(FIGURES / "mhd_runner" / "mhd_runner_summary.csv")
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
        df = load_problem_history(problem, method)
        if df is None or df.empty or "time" not in df.columns:
            continue
        if method == "powell_source" and problem == "orszag_tang":
            completed = method_completed(problem, method)
            if completed is False:
                continue
        if derived_energy_drift:
            if "total_energy" not in df.columns:
                continue
            energy = pd.to_numeric(df["total_energy"], errors="coerce")
            finite = energy[np.isfinite(energy)]
            if finite.empty:
                continue
            e0 = float(finite.iloc[0])
            df = df.copy()
            df["relative_energy_drift"] = (energy - e0) / max(abs(e0), 1.0e-30)
            plot_col = "relative_energy_drift"
        else:
            plot_col = ycol
        if plot_col not in df.columns:
            continue
        x, y = finite_xy(df, "time", plot_col, positive_y=(yscale == "log"))
        if x.empty:
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
        df = load_problem_history("orszag_tang", method)
        if df is None or df.empty or "time" not in df.columns:
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


def plot_figure_1_to_5() -> None:
    # Figure 1: core OT divergence comparison. Powell variants fail on OT and are
    # excluded from the main final-value comparison.
    plot_history_comparison(
        "orszag_tang",
        CORE_METHODS,
        "L2_norm_fv",
        r"normalized FV $L_2(\nabla\cdot B)$",
        "Orszag-Tang divergence control",
        "ot_divergence_main",
        yscale="log",
    )

    plot_ot_min_pressure()

    plot_history_comparison(
        "orszag_tang",
        CORE_METHODS,
        "relative_energy_drift",
        "relative total-energy drift",
        "Orszag-Tang energy drift",
        "ot_energy_drift",
        yscale="symlog",
        derived_energy_drift=True,
    )

    plot_history_comparison(
        "field_loop",
        GLM_FAMILY_METHODS,
        "L2_norm_fv",
        r"normalized FV $L_2(\nabla\cdot B)$",
        "Field-loop divergence control",
        "field_loop_divergence",
        yscale="log",
    )

    da_methods = [
        "none",
        "hyperbolic_glm",
        "mixed_glm",
        "parabolic",
        "elliptic_projection",
        "mixed_eglm",
        "gi_mixed_eglm",
    ]
    plot_history_comparison(
        "divergence_advection",
        da_methods,
        "L2_norm_fv",
        r"normalized FV $L_2(\nabla\cdot B)$",
        "Divergence-advection cleaning",
        "divergence_advection_divergence",
        yscale="log",
    )


def load_mhd_runner_summary() -> pd.DataFrame | None:
    return read_csv(FIGURES / "mhd_runner" / "mhd_runner_summary.csv")


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


def load_performance_benchmark() -> pd.DataFrame | None:
    path = RESULTS / "mhd_runner" / "performance" / "performance_scaling_all_methods.csv"
    df = read_csv(path)
    if df is None or df.empty:
        return None
    df = refresh_performance_divergence_metrics(df)
    required = {"method", "nx", "total_wall_time_sec", "final_L2_norm_fv"}
    if not required.issubset(df.columns):
        SKIPPED.append("performance benchmark lacks required columns")
        return None
    max_n = int(pd.to_numeric(df["nx"], errors="coerce").max())
    df = df[pd.to_numeric(df["nx"], errors="coerce") == max_n].copy()
    df["method"] = df["method"].astype(str)
    df = df[df["method"].isin(PERFORMANCE_METHODS)].copy()
    df["_order"] = df["method"].map(lambda m: method_sort_key(m)[0])
    return df.sort_values("_order")


def plot_scorecard() -> None:
    df = load_performance_benchmark()
    if df is None or df.empty:
        SKIPPED.append("method_scorecard: no compatible performance benchmark")
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
    fig.suptitle("Method scorecard: divergence-advection performance benchmark", y=0.98)
    fig.text(
        0.5,
        0.01,
        "Same benchmark for all panels: divergence_advection, PLM/van Leer, largest available resolution in performance_scaling_all_methods.csv.",
        ha="center",
        fontsize=9,
    )
    save(fig, "method_scorecard")


def plot_divergence_energy_tradeoff() -> None:
    summary = load_mhd_runner_summary()
    if summary is None:
        return
    df = summary[(summary["problem"] == "orszag_tang")].copy()
    df = df[df["method"].isin(CORE_METHODS)]
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
    problems = [p for p in ["orszag_tang", "field_loop", "divergence_advection"] if p in table.index]
    methods = [m for m in METHOD_ORDER if m in table.columns]
    table = table.loc[problems, methods]

    fig, ax = plt.subplots(figsize=(1.0 * len(methods) + 2.5, 3.4))
    arr = table.to_numpy(dtype=float)
    cmap = mcolors.ListedColormap(["#d95f02", "#1b9e77"])
    im = ax.imshow(arr, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([LABELS[m] for m in methods], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(problems)))
    ax.set_yticklabels([PROBLEM_LABEL[p] for p in problems])
    ax.set_title("Robustness by problem and method")
    for j in range(len(problems)):
        for i in range(len(methods)):
            value = arr[j, i]
            text = "pass" if value >= 0.5 else "fail"
            ax.text(i, j, text, ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, ticks=[0, 1])
    cbar.ax.set_yticklabels(["failed", "completed"])
    save(fig, "robustness_problem_method")


def load_snapshot(stem: str) -> pd.DataFrame | None:
    path = RESULTS / "mhd_runner" / "snapshots" / f"{stem}_final.csv"
    return read_csv(path)


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
            xs = np.sort(out["x"].unique())
            ys = np.sort(out["y"].unique())
            dx = float(np.median(np.diff(xs))) if len(xs) > 1 else 1.0
            dy = float(np.median(np.diff(ys))) if len(ys) > 1 else 1.0
            h = math.sqrt(dx * dy)
            out["norm_abs_divB_fv"] = h * out["abs_divB_fv"] / (out["Bmag"].abs() + 1.0e-30)
    return out


def plot_ot_snapshot_panel() -> None:
    entries = [
        ("mhd_ot_none", "none"),
        ("mhd_ot_mixed_glm", "mixed_glm"),
        ("mhd_ot_elliptic_projection", "elliptic_projection"),
    ]
    loaded = []
    for stem, method in entries:
        df = load_snapshot(stem)
        if df is None:
            return
        loaded.append((method, add_snapshot_derived(df)))
    rows = [
        ("rho", r"density $\rho$", "viridis", "linear"),
        ("p", r"pressure $p$", "plasma", "linear"),
        ("norm_abs_divB_fv", r"normalized $|\nabla\cdot B|$", "magma", "log"),
    ]
    fig = plt.figure(figsize=(11.2, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(
        len(rows),
        len(loaded) + 1,
        width_ratios=[1.0, 1.0, 1.0, 0.055],
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
        for _, df in loaded:
            grid = snapshot_grid(df, field)
            if grid is None:
                SKIPPED.append(f"ot snapshot: missing field {field}")
                plt.close(fig)
                return
            grids.append(grid)
        values = np.concatenate([z[np.isfinite(z)].ravel() for _, _, z in grids])
        if scale == "log":
            positive = values[values > 0]
            norm = mcolors.LogNorm(vmin=max(float(positive.min()), 1.0e-14), vmax=float(positive.max()))
        else:
            norm = mcolors.Normalize(vmin=float(values.min()), vmax=float(values.max()))
        last_im = None
        for c, ((method, _), (x, y, z)) in enumerate(zip(loaded, grids)):
            ax = axes[r, c]
            last_im = ax.pcolormesh(x, y, z, shading="auto", cmap=cmap, norm=norm)
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(LABELS[method])
            if c == 0:
                ax.set_ylabel(label + "\n$y$")
            else:
                ax.set_yticklabels([])
            if r == len(rows) - 1:
                ax.set_xlabel("$x$")
            else:
                ax.set_xticklabels([])
        fig.colorbar(last_im, cax=caxes[r], label=label)
    save(fig, "ot_snapshot_density_pressure_divb")


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
    plot_robustness_heatmap()
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
