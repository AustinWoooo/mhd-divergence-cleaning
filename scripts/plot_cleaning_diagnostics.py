#!/usr/bin/env python3
"""
Generate report-oriented diagnostics for the MHD cleaning comparison.

The script is intentionally tolerant of partial result directories:
missing files and missing columns are reported as warnings, and the
remaining figures are still written.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIVERGENCE_DIR = Path("results/mhd_runner/divergence")
FAILURE_DIR = Path("results/mhd_runner/failures")
SUMMARY_DIR = Path("results/mhd_runner/summaries")
FIGURE_DIR = Path("figures/mhd_runner")

PROBLEM = "orszag_tang"
PREFIX = "mhd_ot"

METHODS = [
    "none",
    "hyperbolic_glm",
    "mixed_glm",
    "parabolic",
    "elliptic_projection",
    "powell_source",
    "mixed_eglm",
    "gi_mixed_eglm",
]

LABELS = {
    "none": "None",
    "hyperbolic_glm": "Hyperbolic GLM",
    "mixed_glm": "Mixed GLM",
    "parabolic": "Parabolic",
    "elliptic_projection": "Projection",
    "powell_source": "Powell",
    "mixed_eglm": "Mixed EGLM",
    "gi_mixed_eglm": "GI Mixed EGLM",
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

STAGE_COLORS = {
    "after_hydro": "#4c78a8",
    "before_cleaning": "#f58518",
    "after_cleaning": "#e45756",
    "unknown": "#9d9d9d",
}


def warn(message: str) -> None:
    print(f"warning: {message}")


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        warn(f"missing {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - diagnostic script
        warn(f"could not read {path}: {exc}")
        return None


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def first_column(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def finite_xy(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[pd.Series, pd.Series]:
    x = numeric(df, x_col)
    y = numeric(df, y_col)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def method_label(method: str) -> str:
    return LABELS.get(method, method)


def load_divergence(method: str) -> pd.DataFrame | None:
    return read_csv(DIVERGENCE_DIR / f"{PREFIX}_{method}.csv")


def load_failures() -> pd.DataFrame:
    rows = []
    for path in sorted(FAILURE_DIR.glob("*.csv")):
        df = read_csv(path)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["source_file"] = str(path)
        rows.append(df)
    if not rows:
        warn(f"no failure CSVs found in {FAILURE_DIR}")
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def load_summaries() -> pd.DataFrame:
    rows = []
    for path in sorted(SUMMARY_DIR.glob("*_summary.csv")):
        df = read_csv(path)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["source_file"] = str(path)
        rows.append(df)
    if not rows:
        warn(f"no summary CSVs found in {SUMMARY_DIR}")
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def save_figure(fig: plt.Figure, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_l2_divergence() -> Path | None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    plotted = False
    for method in METHODS:
        df = load_divergence(method)
        if df is None:
            continue
        if "time" not in df.columns or "L2_fv" not in df.columns:
            warn(f"{PREFIX}_{method}.csv lacks time or L2_fv")
            continue
        time, l2 = finite_xy(df, "time", "L2_fv")
        mask = l2 > 0.0
        time = time[mask]
        l2 = l2[mask]
        if time.empty:
            warn(f"{PREFIX}_{method}.csv has no positive finite L2_fv values")
            continue
        ax.plot(
            time,
            l2,
            label=method_label(method),
            color=COLORS.get(method),
            linewidth=1.9,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel("FV L2(div B)")
    ax.set_title("Orszag-Tang FV divergence error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    return save_figure(fig, "cleaning_ot_l2_fv.png")


def plot_normalized_l2_divergence() -> Path | None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    plotted = False
    for method in METHODS:
        df = load_divergence(method)
        if df is None:
            continue
        y_col = first_column(df, ["L2_norm_fv", "L2_fv_norm", "L2_normalized_fv"])
        if "time" not in df.columns or y_col is None:
            warn(f"{PREFIX}_{method}.csv lacks normalized L2 divergence columns")
            continue
        time, l2 = finite_xy(df, "time", y_col)
        mask = l2 > 0.0
        time = time[mask]
        l2 = l2[mask]
        if time.empty:
            warn(f"{PREFIX}_{method}.csv has no positive finite {y_col} values")
            continue
        ax.plot(
            time,
            l2,
            label=method_label(method),
            color=COLORS.get(method),
            linewidth=1.9,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel("normalized FV L2(div B)")
    ax.set_title("Orszag-Tang normalized FV divergence error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    return save_figure(fig, "cleaning_ot_l2_norm_fv.png")


def failure_summary(failures: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "problem",
        "method",
        "energy_policy",
        "failure_stage",
        "failure_time",
        "i",
        "j",
        "pressure",
        "rho",
        "internal_energy",
        "magnetic_energy",
        "kinetic_energy",
        "divB",
    ]
    if failures.empty:
        out = pd.DataFrame(columns=columns)
        out.to_csv(FIGURE_DIR / "cleaning_failure_stage_summary.csv", index=False)
        return out

    df = failures.copy()
    if "failure_stage" not in df.columns:
        df["failure_stage"] = df.get("stage", "unknown")
    if "failure_time" not in df.columns:
        df["failure_time"] = df.get("time", np.nan)
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    out = df[columns].copy()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "cleaning_failure_stage_summary.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}")
    return out


def plot_failure_times(failures: pd.DataFrame) -> Path | None:
    table = failure_summary(failures)
    if table.empty:
        return None
    table = table.sort_values(["method", "energy_policy"], kind="stable")
    times = pd.to_numeric(table["failure_time"], errors="coerce")
    labels = [
        f"{method_label(str(row.method))}\n{str(row.energy_policy).replace('_', ' ')}"
        for row in table.itertuples(index=False)
    ]
    stages = table["failure_stage"].fillna("unknown").astype(str)
    colors = [STAGE_COLORS.get(stage, STAGE_COLORS["unknown"]) for stage in stages]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(table))
    bars = ax.bar(x, times, color=colors)
    for bar, stage, time_value in zip(bars, stages, times):
        if np.isfinite(time_value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                stage,
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("failure time")
    ax.set_title("Orszag-Tang bad-state time by method and energy policy")
    ax.grid(True, axis="y", alpha=0.25)
    return save_figure(fig, "cleaning_failure_times.png")


def plot_failure_energy(failures: pd.DataFrame) -> Path | None:
    if failures.empty:
        return None
    required = ["total_energy", "kinetic_energy", "magnetic_energy", "internal_energy"]
    if not any(column in failures.columns for column in required):
        warn("failure CSVs lack energy decomposition columns")
        return None

    rows = failures.copy()
    rows["failure_time"] = pd.to_numeric(rows.get("time", np.nan), errors="coerce")
    rows["stage"] = rows.get("stage", "unknown")
    labels = []
    for row in rows.itertuples(index=False):
        method = method_label(str(getattr(row, "method", "unknown")))
        policy = str(getattr(row, "energy_policy", "unknown")).replace("_", " ")
        stage = str(getattr(row, "stage", "unknown"))
        time = getattr(row, "failure_time", np.nan)
        labels.append(f"{method}\n{policy}\n{stage}, t={time:.4g}")

    values = np.vstack([pd.to_numeric(rows.get(col, np.nan), errors="coerce") for col in required])
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(rows))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(required))
    for offset, column, data in zip(offsets, required, values):
        ax.bar(x + offset, data, width=width, label=column.replace("_", " "))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("cell energy density")
    ax.set_title("Energy decomposition at first bad cell")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    return save_figure(fig, "cleaning_failure_energy_decomposition.png")


def orszag_tang_summaries(summaries: pd.DataFrame) -> pd.DataFrame:
    if summaries.empty or "problem" not in summaries.columns:
        return pd.DataFrame()
    problem = summaries["problem"].astype(str)
    smoke = summaries.get("source_file", pd.Series("", index=summaries.index)).astype(str)
    mask = (problem == PROBLEM) & ~smoke.str.contains("_smoke_", regex=False)
    return summaries[mask].copy()


def plot_energy_drift(summaries: pd.DataFrame) -> Path | None:
    df = orszag_tang_summaries(summaries)
    if df.empty:
        warn("no full Orszag-Tang summaries available for energy drift plot")
        return None
    for col in ["energy_initial", "energy_final", "energy_drift"]:
        if col not in df.columns:
            warn(f"summaries lack {col}; skipping energy drift plot")
            return None
    df["method_order"] = df["method"].map({method: i for i, method in enumerate(METHODS)})
    df = df.sort_values("method_order")
    initial = pd.to_numeric(df["energy_initial"], errors="coerce")
    final = pd.to_numeric(df["energy_final"], errors="coerce")
    relative = pd.to_numeric(df["energy_drift"], errors="coerce")
    absolute = final - initial

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True)
    labels = [method_label(str(method)) for method in df["method"]]
    x = np.arange(len(df))
    colors = [COLORS.get(str(method), "#777777") for method in df["method"]]
    axes[0].bar(x, absolute, color=colors)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("energy_final - energy_initial")
    axes[0].set_title("absolute drift")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(x, relative, color=colors)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("reported relative drift")
    axes[1].set_title("relative drift")
    axes[1].grid(True, axis="y", alpha=0.25)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    fig.suptitle("Orszag-Tang total-energy drift")
    return save_figure(fig, "cleaning_energy_drift.png")


def plot_projection_convergence(summaries: pd.DataFrame) -> Path | None:
    df = orszag_tang_summaries(summaries)
    if df.empty or "projection_iterations_total" not in df.columns:
        warn("no projection iteration totals available")
        return None
    df = df.copy()
    iterations = pd.to_numeric(df["projection_iterations_total"], errors="coerce")
    df = df[iterations.fillna(0.0) > 0.0]
    if df.empty:
        warn("projection iteration totals are all zero")
        return None
    x = np.arange(len(df))
    labels = [method_label(str(method)) for method in df["method"]]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(x, pd.to_numeric(df["projection_iterations_total"], errors="coerce"), color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("total SOR iterations")
    ax.set_title("Projection convergence work, Orszag-Tang")
    ax.grid(True, axis="y", alpha=0.25)
    return save_figure(fig, "cleaning_projection_iterations_total.png")


def plot_projection_history() -> Path | None:
    candidates = [
        "projection_iterations",
        "projection_iterations_total",
        "projection_final_residual",
        "projection_residual",
    ]
    plotted = False
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)
    for method in METHODS:
        df = load_divergence(method)
        if df is None or "time" not in df.columns:
            continue
        iter_col = first_column(df, ["projection_iterations", "projection_iters"])
        res_col = first_column(df, ["projection_final_residual", "projection_residual"])
        if iter_col is None and res_col is None:
            continue
        if iter_col is not None:
            t, y = finite_xy(df, "time", iter_col)
            if not t.empty:
                axes[0].plot(t, y, label=method_label(method), color=COLORS.get(method))
                plotted = True
        if res_col is not None:
            t, y = finite_xy(df, "time", res_col)
            y = y[y > 0.0]
            t = t.loc[y.index]
            if not t.empty:
                axes[1].plot(t, y, label=method_label(method), color=COLORS.get(method))
                axes[1].set_yscale("log")
                plotted = True
    if not plotted:
        plt.close(fig)
        warn(f"no projection history columns found: {', '.join(candidates)}")
        return None
    axes[0].set_ylabel("iterations")
    axes[1].set_ylabel("final residual")
    axes[1].set_xlabel("time")
    axes[0].set_title("Projection convergence history")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    return save_figure(fig, "cleaning_projection_convergence_history.png")


def plot_subcycle_totals(summaries: pd.DataFrame) -> Path | None:
    df = orszag_tang_summaries(summaries)
    if df.empty or "cleaning_subcycles_total" not in df.columns:
        warn("no cleaning subcycle totals available")
        return None
    df["method_order"] = df["method"].map({method: i for i, method in enumerate(METHODS)})
    df = df.sort_values("method_order")
    subcycles = pd.to_numeric(df["cleaning_subcycles_total"], errors="coerce")
    if not np.isfinite(subcycles).any():
        warn("cleaning subcycle totals are not finite")
        return None
    x = np.arange(len(df))
    labels = [method_label(str(method)) for method in df["method"]]
    colors = [COLORS.get(str(method), "#777777") for method in df["method"]]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(x, subcycles, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("total cleaning subcycles")
    ax.set_title("Orszag-Tang cleaning subcycle workload")
    ax.grid(True, axis="y", alpha=0.25)
    return save_figure(fig, "cleaning_subcycles_total.png")


def plot_subcycle_history() -> Path | None:
    plotted = False
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for method in METHODS:
        df = load_divergence(method)
        if df is None or "time" not in df.columns:
            continue
        sub_col = first_column(df, ["cleaning_subcycles", "subcycles", "nsub"])
        if sub_col is None:
            continue
        time, subcycles = finite_xy(df, "time", sub_col)
        if time.empty:
            continue
        ax.step(time, subcycles, where="post", label=method_label(method), color=COLORS.get(method))
        plotted = True
    if not plotted:
        plt.close(fig)
        warn("no per-time cleaning subcycle columns found")
        return None
    ax.set_xlabel("time")
    ax.set_ylabel("subcycles per MHD step")
    ax.set_title("Cleaning subcycles vs time")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    return save_figure(fig, "cleaning_subcycles_vs_time.png")


def write_combined_summary(summaries: pd.DataFrame, failures: pd.DataFrame) -> Path:
    rows = []
    failure_lookup = {}
    if not failures.empty:
        f = failures.copy()
        if "failure_stage" not in f.columns:
            f["failure_stage"] = f.get("stage", "unknown")
        if "failure_time" not in f.columns:
            f["failure_time"] = f.get("time", np.nan)
        for row in f.itertuples(index=False):
            method = str(getattr(row, "method", ""))
            policy = str(getattr(row, "energy_policy", ""))
            failure_lookup[(method, policy)] = row

    ot_summaries = orszag_tang_summaries(summaries)
    for method in METHODS:
        div = load_divergence(method)
        summary_row = None
        if not ot_summaries.empty and "method" in ot_summaries.columns:
            matches = ot_summaries[ot_summaries["method"].astype(str) == method]
            if not matches.empty:
                summary_row = matches.iloc[0]
        row = {
            "problem": PROBLEM,
            "method": method,
            "divergence_file": str(DIVERGENCE_DIR / f"{PREFIX}_{method}.csv"),
            "divergence_rows": 0,
            "final_time": np.nan,
            "final_time_reached": np.nan,
            "final_L1_fv": np.nan,
            "final_L2_fv": np.nan,
            "final_Linf_fv": np.nan,
            "final_L2_norm_fv": np.nan,
            "energy_initial": np.nan,
            "energy_final": np.nan,
            "energy_drift_reported": np.nan,
            "energy_drift_absolute": np.nan,
            "min_pressure": np.nan,
            "min_density": np.nan,
            "failure_time": np.nan,
            "failure_reason": "",
            "cleaning_subcycles_total": np.nan,
            "projection_iterations_total": np.nan,
        }
        if div is not None and not div.empty:
            row["divergence_rows"] = len(div)
            row["final_time"] = numeric(div, "time").iloc[-1] if "time" in div else np.nan
            for dst, src in [
                ("final_L1_fv", "L1_fv"),
                ("final_L2_fv", "L2_fv"),
                ("final_Linf_fv", "Linf_fv"),
            ]:
                if src in div:
                    row[dst] = numeric(div, src).iloc[-1]
            norm_col = first_column(div, ["L2_norm_fv", "L2_fv_norm", "L2_normalized_fv"])
            if norm_col is not None:
                row["final_L2_norm_fv"] = numeric(div, norm_col).iloc[-1]
        if summary_row is not None:
            for col in [
                "energy_initial",
                "energy_final",
                "energy_drift",
                "final_L1_fv",
                "final_L2_fv",
                "final_Linf_fv",
                "final_time_reached",
                "min_pressure",
                "min_density",
                "failure_time",
                "failure_reason",
                "cleaning_subcycles_total",
                "projection_iterations_total",
            ]:
                if col in summary_row.index:
                    key = "energy_drift_reported" if col == "energy_drift" else col
                    row[key] = summary_row[col]
            try:
                row["energy_drift_absolute"] = float(row["energy_final"]) - float(row["energy_initial"])
            except (TypeError, ValueError):
                pass
        rows.append(row)

    out = pd.DataFrame(rows)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "cleaning_diagnostics_summary.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}")
    return path


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    failures = load_failures()
    summaries = load_summaries()

    written = [
        plot_l2_divergence(),
        plot_normalized_l2_divergence(),
        plot_failure_times(failures),
        plot_failure_energy(failures),
        plot_energy_drift(summaries),
        plot_projection_history(),
        plot_projection_convergence(summaries),
        plot_subcycle_history(),
        plot_subcycle_totals(summaries),
        write_combined_summary(summaries, failures),
    ]
    count = sum(path is not None for path in written)
    print(f"completed cleaning diagnostics: {count} outputs written")


if __name__ == "__main__":
    main()
