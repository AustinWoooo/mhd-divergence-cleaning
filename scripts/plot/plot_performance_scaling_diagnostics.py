#!/usr/bin/env python3
"""Plot quantitative performance-scaling diagnostics from benchmark CSVs."""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "results" / "mhd_runner" / "performance" / "performance_scaling_all_methods.csv"
FIG_DIR = ROOT / "figures" / "mhd_runner" / "performance"

METHOD_ORDER = [
    "none",
    "parabolic",
    "hyperbolic_glm",
    "mixed_glm",
    "mixed_eglm",
    "gi_mixed_eglm",
    "elliptic_projection",
    "powell_source",
]

PROBLEM_ORDER = [
    "divergence_advection",
    "field_loop",
    "orszag_tang",
]

METHOD_LABELS = {
    "none": "None (HLLD only)",
    "parabolic": "Parabolic",
    "hyperbolic_glm": "Hyperbolic GLM",
    "mixed_glm": "Mixed GLM",
    "mixed_eglm": "Mixed EGLM",
    "gi_mixed_eglm": "GI Mixed EGLM",
    "elliptic_projection": "Projection",
    "powell_source": "Powell",
}

METHOD_COLORS = {
    "none": "#4c78a8",
    "parabolic": "#e45756",
    "hyperbolic_glm": "#f58518",
    "mixed_glm": "#54a24b",
    "mixed_eglm": "#e377c2",
    "gi_mixed_eglm": "#b79f00",
    "elliptic_projection": "#9467bd",
    "powell_source": "#8c564b",
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
}


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def positive_float(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(np.isfinite(values), np.nan)


def load_scaling_data(path: Path = INPUT_CSV) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing performance CSV: {path}")

    df = pd.read_csv(path)
    required = {
        "problem",
        "method",
        "nx",
        "ny",
        "ncell",
        "hydro_time_sec",
        "cleaning_time_sec",
        "diagnostics_compute_time_sec",
        "output_time_sec",
    }
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"{path} missing columns: {', '.join(sorted(missing))}")

    supported = set(METHOD_ORDER)
    df = df[df["method"].isin(supported)].copy()
    if df.empty:
        raise SystemExit("no supported method rows found in performance CSV")

    numeric_cols = [
        "nx",
        "ny",
        "ncell",
        "hydro_time_sec",
        "cleaning_time_sec",
        "diagnostics_compute_time_sec",
        "output_time_sec",
    ]
    for column in numeric_cols:
        df[column] = positive_float(df[column])

    df["diagnostics_output_time_sec"] = (
        df["diagnostics_compute_time_sec"] + df["output_time_sec"]
    )
    df["component_total_time_sec"] = (
        df["hydro_time_sec"]
        + df["cleaning_time_sec"]
        + df["diagnostics_output_time_sec"]
    )
    return df


def aggregate_suite(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["method", "nx", "ny", "ncell"], as_index=False)
        .agg(
            problem_count=("problem", "nunique"),
            hydro_time_sec=("hydro_time_sec", "sum"),
            cleaning_time_sec=("cleaning_time_sec", "sum"),
            diagnostics_output_time_sec=("diagnostics_output_time_sec", "sum"),
            component_total_time_sec=("component_total_time_sec", "sum"),
        )
    )
    grouped["cleaning_fraction"] = (
        grouped["cleaning_time_sec"] / grouped["component_total_time_sec"]
    )
    grouped = grouped.sort_values(
        ["method", "ncell"],
        key=lambda s: s.map(lambda v: METHOD_ORDER.index(v) if v in METHOD_ORDER else v)
        if s.name == "method"
        else s,
    )
    return grouped.reset_index(drop=True)


def aggregate_by_problem(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["problem", "method", "nx", "ny", "ncell"], as_index=False)
        .agg(
            hydro_time_sec=("hydro_time_sec", "sum"),
            cleaning_time_sec=("cleaning_time_sec", "sum"),
            diagnostics_output_time_sec=("diagnostics_output_time_sec", "sum"),
            component_total_time_sec=("component_total_time_sec", "sum"),
        )
    )
    grouped["cleaning_fraction"] = (
        grouped["cleaning_time_sec"] / grouped["component_total_time_sec"]
    )
    grouped["problem_order"] = grouped["problem"].map(
        lambda value: PROBLEM_ORDER.index(value)
        if value in PROBLEM_ORDER
        else len(PROBLEM_ORDER)
    )
    grouped["method_order"] = grouped["method"].map(
        lambda value: METHOD_ORDER.index(value)
        if value in METHOD_ORDER
        else len(METHOD_ORDER)
    )
    grouped = grouped.sort_values(["problem_order", "method_order", "ncell"])
    return grouped.drop(columns=["problem_order", "method_order"]).reset_index(drop=True)


def resolution_ticks(data: pd.DataFrame) -> tuple[list[float], list[str]]:
    resolutions = (
        data[["nx", "ny", "ncell"]]
        .drop_duplicates()
        .sort_values("ncell")
    )
    ticks = [float(row.ncell) for row in resolutions.itertuples()]
    labels = [f"{int(row.nx)}^2" if int(row.nx) == int(row.ny) else str(int(row.ncell))
              for row in resolutions.itertuples()]
    return ticks, labels


def setup_resolution_axis(ax: plt.Axes, data: pd.DataFrame) -> None:
    ticks, labels = resolution_ticks(data)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("grid resolution")
    ax.grid(True, which="both", alpha=0.3)


def plot_method_lines(
    data: pd.DataFrame,
    y_column: str,
    ylabel: str,
    title: str,
    output_name: str,
    *,
    log_y: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for method in METHOD_ORDER:
        group = data[data["method"] == method].sort_values("ncell")
        if group.empty:
            continue
        y = group[y_column].astype(float)
        if log_y:
            valid = y > 0.0
            group = group[valid]
            y = y[valid]
        if group.empty:
            continue
        ax.plot(
            group["ncell"],
            y,
            marker=METHOD_MARKERS.get(method, "o"),
            color=METHOD_COLORS.get(method),
            linewidth=1.9,
            markersize=5.8,
            label=method_label(method),
        )

    setup_resolution_axis(ax, data)
    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8.2, ncol=2)
    fig.tight_layout()

    out = FIG_DIR / output_name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_projection_slowdown(data: pd.DataFrame) -> Path:
    pivot = data.pivot(index="ncell", columns="method", values="component_total_time_sec")
    nx_by_ncell = data.drop_duplicates("ncell").set_index("ncell")["nx"].to_dict()
    ratios = {
        "Projection / None": pivot["elliptic_projection"] / pivot["none"],
        "Projection / Parabolic": pivot["elliptic_projection"] / pivot["parabolic"],
        "Projection / Mixed GLM": pivot["elliptic_projection"] / pivot["mixed_glm"],
    }

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    styles = [
        ("Projection / None", "#4c78a8", "o"),
        ("Projection / Parabolic", "#e45756", "v"),
        ("Projection / Mixed GLM", "#54a24b", "s"),
    ]
    for label, color, marker in styles:
        series = ratios[label].dropna().sort_index()
        ax.plot(
            series.index,
            series.values,
            marker=marker,
            color=color,
            linewidth=2.0,
            markersize=6.0,
            label=label,
        )

    ticks = sorted(pivot.index.astype(float))
    tick_labels = [f"{int(nx_by_ncell[t])}^2" for t in ticks]
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("grid resolution")
    ax.set_ylabel("slowdown ratio")
    ax.set_title("Projection slowdown relative to baseline methods")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = FIG_DIR / "performance_projection_slowdown_vs_resolution.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_normalized_growth(
    data: pd.DataFrame,
    y_column: str,
    ylabel: str,
    title: str,
    output_name: str,
    *,
    baseline_resolution: int = 64,
    log_y: bool = False,
) -> Path:
    values_df = normalized_growth_frame(data, y_column, baseline_resolution)

    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    plotted = False
    for method in METHOD_ORDER:
        group = values_df[values_df["method"] == method].sort_values("ncell")
        if group.empty:
            continue
        ax.plot(
            group["ncell"],
            group["normalized_value"],
            marker=METHOD_MARKERS.get(method, "o"),
            color=METHOD_COLORS.get(method),
            linewidth=1.9,
            markersize=5.8,
            label=method_label(method),
        )
        plotted = True

    if not plotted:
        raise SystemExit(f"no normalized growth data plotted for {output_name}")

    setup_resolution_axis(ax, values_df)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.55)
    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8.2, ncol=2)
    fig.tight_layout()

    out = FIG_DIR / output_name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def normalized_growth_frame(
    data: pd.DataFrame,
    y_column: str,
    baseline_resolution: int = 64,
) -> pd.DataFrame:
    baseline_ncell = baseline_resolution * baseline_resolution
    keep_ncells = {
        baseline_ncell,
        128 * 128,
        256 * 256,
    }
    rows = []
    for method in METHOD_ORDER:
        group = data[
            (data["method"] == method)
            & (data["ncell"].isin(keep_ncells))
        ].sort_values("ncell")
        if group.empty:
            continue
        baseline = group.loc[group["ncell"] == baseline_ncell, y_column]
        if (
            baseline.empty
            or not np.isfinite(float(baseline.iloc[0]))
            or float(baseline.iloc[0]) <= 0.0
        ):
            print(
                f"Skipping normalized growth for {method}: "
                f"missing positive {baseline_resolution}^2 baseline"
            )
            continue
        normalized = group.copy()
        normalized["normalized_value"] = (
            normalized[y_column].astype(float) / float(baseline.iloc[0])
        )
        rows.append(normalized)
    if not rows:
        return pd.DataFrame(columns=list(data.columns) + ["normalized_value"])
    return pd.concat(rows, ignore_index=True)


def plot_problem_total_growth(problem_data: pd.DataFrame, problem: str) -> list[Path]:
    outputs = []
    for log_y, suffix, title_suffix in (
        (False, "", ""),
        (True, "_logy", " (log y)"),
    ):
        outputs.append(
            plot_normalized_growth(
                problem_data,
                "component_total_time_sec",
                r"$T_{\rm total}(N) / T_{\rm total}(64^2)$",
                f"Normalized total-time growth from 64^2 -- {problem}{title_suffix}",
                f"performance_total_time_growth_normalized_64_{problem}{suffix}.png",
                log_y=log_y,
            )
        )
    return outputs


def print_problem_total_growth_values(problem_data: pd.DataFrame) -> None:
    print("\nNormalized total-time growth by problem:")
    for problem in PROBLEM_ORDER:
        data = problem_data[problem_data["problem"] == problem]
        if data.empty:
            continue
        values = normalized_growth_frame(data, "component_total_time_sec")
        print(f"\n{problem}")
        print(f"{'method':<22} {'128^2':>10} {'256^2':>10}")
        for method in METHOD_ORDER:
            group = values[values["method"] == method].set_index("nx")
            if group.empty:
                continue
            value_128 = group.loc[128, "normalized_value"] if 128 in group.index else math.nan
            value_256 = group.loc[256, "normalized_value"] if 256 in group.index else math.nan
            print(f"{method:<22} {float(value_128):10.3f} {float(value_256):10.3f}")


def fit_exponents(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        group = data[data["method"] == method].sort_values("ncell")
        row = {"method": method}
        for column, name in (
            ("component_total_time_sec", "alpha_total"),
            ("cleaning_time_sec", "alpha_cleaning"),
        ):
            valid = group[(group["ncell"] > 0.0) & (group[column] > 0.0)]
            if len(valid) < 2:
                row[name] = math.nan
                continue
            x = np.log(valid["ncell"].astype(float).to_numpy())
            y = np.log(valid[column].astype(float).to_numpy())
            row[name] = float(np.polyfit(x, y, 1)[0])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_exponents(exponents: pd.DataFrame) -> Path:
    x = np.arange(len(METHOD_ORDER))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.bar(
        x - width / 2,
        exponents["alpha_total"],
        width,
        label=r"total $\alpha$",
        color="#4c78a8",
    )
    ax.bar(
        x + width / 2,
        exponents["alpha_cleaning"],
        width,
        label=r"cleaning $\alpha$",
        color="#f58518",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(m) for m in METHOD_ORDER], rotation=35, ha="right")
    ax.set_ylabel(r"fitted exponent in $T \propto N_{\rm cell}^{\alpha}$")
    ax.set_title("Performance scaling exponents")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = FIG_DIR / "performance_scaling_exponents_all_methods.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def print_exponents(exponents: pd.DataFrame) -> None:
    print("\nFitted scaling exponents using T ~ N_cell^alpha:")
    print(f"{'method':<22} {'alpha_total':>12} {'alpha_cleaning':>16}")
    for row in exponents.itertuples():
        print(
            f"{row.method:<22} "
            f"{row.alpha_total:12.3f} "
            f"{row.alpha_cleaning:16.3f}"
        )


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_scaling_data(INPUT_CSV)
    suite = aggregate_suite(raw)
    by_problem = aggregate_by_problem(raw)

    methods = [method for method in METHOD_ORDER if method in set(suite["method"])]
    print(f"Input CSV: {INPUT_CSV}")
    print("Plotted methods: " + ", ".join(methods))
    print(
        "Aggregated benchmark suite: "
        + ", ".join(sorted(raw["problem"].unique()))
    )
    print(
        "The all-method normalized growth figures use combined multi-problem data "
        "summed over: "
        + ", ".join(sorted(raw["problem"].unique()))
    )

    plot_method_lines(
        suite,
        "component_total_time_sec",
        "total time [s]",
        "Total time vs resolution",
        "performance_total_time_vs_resolution_all_methods.png",
        log_y=True,
    )
    plot_method_lines(
        suite,
        "cleaning_time_sec",
        "cleaning time [s]",
        "Cleaning time vs resolution",
        "performance_cleaning_time_vs_resolution_all_methods.png",
        log_y=True,
    )
    plot_method_lines(
        suite,
        "cleaning_fraction",
        "cleaning time / total measured time",
        "Cleaning fraction vs resolution",
        "performance_cleaning_fraction_vs_resolution_all_methods.png",
    )
    plot_normalized_growth(
        suite,
        "component_total_time_sec",
        r"$T_{\rm total}(N) / T_{\rm total}(64^2)$",
        "Normalized total-time growth from 64^2",
        "performance_total_time_growth_normalized_64_all_methods.png",
    )
    plot_normalized_growth(
        suite,
        "cleaning_time_sec",
        r"$T_{\rm cleaning}(N) / T_{\rm cleaning}(64^2)$",
        "Normalized cleaning-time growth from 64^2",
        "performance_cleaning_time_growth_normalized_64_all_methods.png",
    )
    for problem in PROBLEM_ORDER:
        problem_data = by_problem[by_problem["problem"] == problem]
        if problem_data.empty:
            print(f"Skipping per-problem normalized total-time growth: no rows for {problem}")
            continue
        plot_problem_total_growth(problem_data, problem)
    plot_projection_slowdown(suite)
    exponents = fit_exponents(suite)
    plot_exponents(exponents)
    print_problem_total_growth_values(by_problem)
    print_exponents(exponents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
