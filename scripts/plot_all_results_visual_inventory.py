#!/usr/bin/env python3
"""Generate a broad visual inventory of all existing result CSVs.

This script is intentionally comprehensive rather than curated.  It reads
existing files under results/ and figures/mhd_runner/, writes figures under
figures/visual_inventory/, and records a machine-readable + human-readable
index of generated plots.  It does not run simulations.
"""

from __future__ import annotations

import math
import os
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(".")
RESULTS = ROOT / "results"
FIG_MHD = ROOT / "figures" / "mhd_runner"
OUT = ROOT / "figures" / "visual_inventory"

GAMMA = 5.0 / 3.0

METHOD_ORDER = [
    "none",
    "hyperbolic_glm",
    "mixed_glm",
    "mixed_eglm",
    "gi_mixed_eglm",
    "parabolic",
    "elliptic_projection",
    "powell_source",
]

METHOD_LABELS = {
    "none": "None",
    "hyperbolic_glm": "Hyperbolic GLM",
    "mixed_glm": "Mixed GLM",
    "mixed_eglm": "Mixed EGLM",
    "gi_mixed_eglm": "GI Mixed EGLM",
    "parabolic": "Parabolic",
    "elliptic_projection": "Projection",
    "powell_source": "Powell",
    "eglm": "EGLM",
}

METHOD_COLORS = {
    "none": "#1f77b4",
    "hyperbolic_glm": "#ff7f0e",
    "mixed_glm": "#2ca02c",
    "mixed_eglm": "#e377c2",
    "gi_mixed_eglm": "#bcbd22",
    "parabolic": "#d62728",
    "elliptic_projection": "#9467bd",
    "powell_source": "#8c564b",
}

FAMILY = {
    "none": "no cleaning",
    "hyperbolic_glm": "GLM family",
    "mixed_glm": "GLM family",
    "mixed_eglm": "EGLM/source family",
    "gi_mixed_eglm": "EGLM/source family",
    "parabolic": "parabolic",
    "elliptic_projection": "elliptic projection",
    "powell_source": "Powell/source family",
    "eglm": "EGLM/source family",
}

PROBLEM_LABELS = {
    "orszag_tang": "Orszag-Tang",
    "field_loop": "Field loop",
    "divergence_advection": "Divergence advection",
    "glm_1d": "GLM 1D pulse",
    "glm_2d": "GLM 2D pulse",
    "unknown": "Unknown",
}

PREFERRED_TIME_DIAGNOSTICS = [
    ("L1_fv", "FV L1(divB)", "log"),
    ("L2_fv", "FV L2(divB)", "log"),
    ("Linf_fv", "FV Linf(divB)", "log"),
    ("L1_norm_fv", "normalized FV L1(divB)", "log"),
    ("L2_norm_fv", "normalized FV L2(divB)", "log"),
    ("Linf_norm_fv", "normalized FV Linf(divB)", "log"),
    ("min_pressure", "minimum pressure", "linear"),
    ("min_density", "minimum density", "linear"),
    ("dt", "time step", "log"),
    ("total_mass_rel", "relative mass error", "symlog"),
    ("total_energy_rel", "relative energy drift", "symlog"),
    ("total_momentum_x_rel", "momentum x error", "symlog"),
    ("total_momentum_y_rel", "momentum y error", "symlog"),
    ("total_momentum_z_rel", "momentum z error", "symlog"),
    ("cleaning_subcycles_step", "cleaning subcycles / step", "linear"),
    ("projection_iterations_step", "projection iterations / step", "linear"),
    ("projection_solver_update_residual", "projection update residual", "log"),
    ("projection_final_residual", "projection final residual", "log"),
    ("projection_true_residual", "projection true residual", "log"),
    ("has_nonfinite", "nonfinite flag", "linear"),
    ("has_negative_density", "negative density flag", "linear"),
    ("has_negative_pressure", "negative pressure flag", "linear"),
]

SUMMARY_TRADEOFFS = [
    ("final_L2_fv", "energy_drift"),
    ("final_L2_fv", "min_pressure"),
    ("final_L2_fv", "total_wall_time_sec"),
    ("energy_drift", "total_wall_time_sec"),
    ("min_pressure", "total_wall_time_sec"),
    ("final_L2_norm_fv", "energy_drift"),
    ("final_L2_norm_fv", "min_pressure"),
    ("final_L2_norm_fv", "total_wall_time_sec"),
]

IMPORTANT_SUMMARY_COLUMNS = {
    "final_L1_fv",
    "final_L2_fv",
    "final_Linf_fv",
    "final_L2_norm_fv",
    "final_Linf_norm_fv",
    "min_pressure",
    "min_density",
    "min_raw_pressure",
    "energy_drift",
    "total_wall_time_sec",
    "seconds_per_step",
    "cell_updates_per_second",
    "steps",
    "cleaning_subcycles_total",
    "projection_iterations_total",
    "projection_true_residual",
    "failure_time",
    "retry_count",
}


@dataclass
class Meta:
    path: Path
    stem: str
    problem: str = "unknown"
    method: str = "unknown"
    run_kind: str = "unknown"
    reconstruction: str = ""
    resolution: str = ""
    prefix: str = ""


figure_rows: list[dict[str, str]] = []
inventory_rows: list[dict[str, str]] = []
warnings: list[str] = []


def warn(message: str) -> None:
    warnings.append(message)
    print(f"warning: {message}")


def sanitize(text: object) -> str:
    s = str(text)
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


def method_sort_key(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), method)


def save_figure(
    fig: plt.Figure,
    rel_path: str | Path,
    group: str,
    inputs: list[Path] | tuple[Path, ...],
    diagnostic: str,
    methods: list[str] | tuple[str, ...],
    problem: str,
    description: str,
    useful: str = "maybe",
    backup: str = "maybe",
    also_pdf: bool = False,
) -> None:
    path = OUT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    figure_rows.append(
        {
            "filename": str(path),
            "figure_group": group,
            "input_data_files": ";".join(str(p) for p in inputs),
            "diagnostic": diagnostic,
            "methods_included": ";".join(sorted(set(methods), key=method_sort_key)),
            "problem_included": problem,
            "description": description,
            "likely_useful_for_final_presentation": useful,
            "likely_backup_only": backup,
        }
    )


def read_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        warn(f"could not parse {path}: {exc}")
        return None


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def finite_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def add_relative_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "total_mass",
        "total_energy",
        "total_momentum_x",
        "total_momentum_y",
        "total_momentum_z",
    ]:
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors="coerce")
            finite = vals[np.isfinite(vals)]
            if len(finite):
                ref = float(finite.iloc[0])
                denom = max(abs(ref), 1.0e-30)
                out[f"{col}_rel"] = (vals - ref) / denom
    return out


def apply_scale(ax: plt.Axes, scale: str) -> None:
    if scale == "log":
        ax.set_yscale("log")
    elif scale == "symlog":
        ax.set_yscale("symlog", linthresh=1.0e-12)


def parse_meta(path: Path) -> Meta:
    stem = path.stem
    stem = stem.removesuffix("_summary").removesuffix("_final")
    meta = Meta(path=path, stem=stem, prefix=stem)

    if stem.startswith("mhd_ot"):
        meta.problem = "orszag_tang"
    elif stem.startswith("mhd_fl"):
        meta.problem = "field_loop"
    elif stem.startswith("mhd_da") or stem.startswith("perf_da") or stem.startswith("mpi_da"):
        meta.problem = "divergence_advection"
    elif stem.startswith("glm_1d"):
        meta.problem = "glm_1d"
    elif stem.startswith("glm_2d"):
        meta.problem = "glm_2d"

    if "_conv_" in stem:
        meta.run_kind = "convergence"
    elif "_smoke_" in stem:
        meta.run_kind = "smoke"
    elif stem.startswith("perf_"):
        meta.run_kind = "performance"
    elif stem.startswith("mpi_"):
        meta.run_kind = "mpi"
    elif stem.startswith("mhd_"):
        meta.run_kind = "science"
    elif stem.startswith("glm_"):
        meta.run_kind = "standalone_glm"

    for token in ("pcm", "plm"):
        if re.search(rf"(^|_){token}(_|$)", stem):
            meta.reconstruction = token

    m = re.search(r"_n(\d+)", stem)
    if m:
        meta.resolution = m.group(1)
    m = re.search(r"_(\d+)x(\d+)", stem)
    if m:
        meta.resolution = f"{m.group(1)}x{m.group(2)}"

    candidates = sorted(METHOD_ORDER + ["eglm"], key=len, reverse=True)
    for method in candidates:
        if stem == method or stem.endswith("_" + method):
            meta.method = "mixed_eglm" if method == "eglm" else method
            break

    if meta.problem == "glm_1d":
        meta.method = stem.replace("glm_1d_", "")
    elif meta.problem == "glm_2d":
        meta.method = stem.replace("glm_2d_", "")

    return meta


def group_result_files() -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(RESULTS.rglob("*.csv")):
        parts = path.parts
        if len(parts) >= 2 and parts[1] == "glm_1d":
            groups["glm_1d"].append(path)
        elif len(parts) >= 4 and parts[1] == "glm_2d":
            groups[f"glm_2d/{parts[2]}"].append(path)
        elif len(parts) >= 4 and parts[1] == "mhd_runner":
            groups[f"mhd_runner/{parts[2]}"].append(path)
        elif len(parts) >= 2 and parts[1] == "mhd_sweep_mpi":
            groups["mhd_sweep_mpi/summaries" if "summaries" in parts else "mhd_sweep_mpi/root"].append(path)
        else:
            groups["other"].append(path)
    return groups


def write_data_inventory(groups: dict[str, list[Path]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Result Data Inventory", ""]
    for group, paths in sorted(groups.items()):
        cols: dict[str, int] = defaultdict(int)
        parsed = 0
        examples = []
        for path in paths:
            df = read_csv(path, nrows=5)
            if df is None:
                continue
            parsed += 1
            examples.append(path)
            for col in df.columns:
                cols[col] += 1
        inventory_rows.append(
            {
                "group": group,
                "file_count": str(len(paths)),
                "parsed_file_count": str(parsed),
                "example_files": ";".join(str(p) for p in examples[:5]),
                "columns": ";".join(f"{c}:{n}" for c, n in sorted(cols.items())),
            }
        )
        lines += [
            f"## {group}",
            f"- files: {len(paths)}",
            f"- parsed: {parsed}",
            f"- examples: {', '.join(str(p) for p in examples[:5])}",
            f"- columns: {', '.join(f'{c} ({n})' for c, n in sorted(cols.items()))}",
            "",
        ]
    pd.DataFrame(inventory_rows).to_csv(OUT / "result_inventory.csv", index=False)
    (OUT / "RESULT_INVENTORY.md").write_text("\n".join(lines))


def plot_time_series_collection(
    files: list[Path],
    out_dir: str,
    group_name: str,
    problem: str,
) -> None:
    data = []
    for path in files:
        df = read_csv(path)
        if df is None or "time" not in df.columns:
            continue
        meta = parse_meta(path)
        data.append((path, meta, add_relative_columns(df)))

    if not data:
        warn(f"no time-series data for {group_name}")
        return

    # Per-run diagnostics.
    for path, meta, df in data:
        present = [(c, label, scale) for c, label, scale in PREFERRED_TIME_DIAGNOSTICS if c in df.columns]
        if not present:
            continue
        n = len(present)
        ncols = 3
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.2 * nrows), squeeze=False)
        x = finite_series(df, "time")
        for ax, (col, label, scale) in zip(axes.ravel(), present):
            y = finite_series(df, col)
            mask = np.isfinite(x) & np.isfinite(y)
            if scale == "log":
                mask &= y > 0.0
            if mask.any():
                ax.plot(x[mask], y[mask], lw=1.4, color=METHOD_COLORS.get(meta.method, "#333333"))
            ax.set_title(label)
            ax.set_xlabel("time")
            ax.grid(True, which="both", alpha=0.25)
            apply_scale(ax, scale)
        for ax in axes.ravel()[n:]:
            ax.set_axis_off()
        fig.suptitle(f"{meta.stem}: time-series diagnostics", y=1.0)
        save_figure(
            fig,
            f"{out_dir}/per_run/{sanitize(meta.stem)}_diagnostics.png",
            group_name,
            [path],
            "all available time-series diagnostics",
            [meta.method],
            meta.problem,
            f"Per-run diagnostic panel for {meta.stem}.",
            useful="maybe",
            backup="yes",
        )

    # Comparison by problem/run-kind.
    by_problem_kind: dict[tuple[str, str], list[tuple[Path, Meta, pd.DataFrame]]] = defaultdict(list)
    for item in data:
        by_problem_kind[(item[1].problem, item[1].run_kind)].append(item)

    compare_cols = [
        ("L1_fv", "FV L1(divB)", "log"),
        ("L2_fv", "FV L2(divB)", "log"),
        ("Linf_fv", "FV Linf(divB)", "log"),
        ("L2_norm_fv", "normalized FV L2(divB)", "log"),
        ("min_pressure", "minimum pressure", "linear"),
        ("min_density", "minimum density", "linear"),
        ("total_energy_rel", "relative energy drift", "symlog"),
        ("total_mass_rel", "relative mass error", "symlog"),
        ("total_momentum_x_rel", "relative x-momentum error", "symlog"),
        ("total_momentum_y_rel", "relative y-momentum error", "symlog"),
        ("total_momentum_z_rel", "relative z-momentum error", "symlog"),
        ("projection_iterations_step", "projection iterations / step", "linear"),
        ("cleaning_subcycles_step", "cleaning subcycles / step", "linear"),
    ]
    for (prob, kind), items in sorted(by_problem_kind.items()):
        for col, label, scale in compare_cols:
            if not any(col in df.columns for _, _, df in items):
                continue
            fig, ax = plt.subplots(figsize=(9.5, 5.5))
            methods = []
            inputs = []
            plotted = 0
            for path, meta, df in sorted(items, key=lambda x: (method_sort_key(x[1].method), x[1].stem)):
                if col not in df.columns:
                    continue
                x = finite_series(df, "time")
                y = finite_series(df, col)
                mask = np.isfinite(x) & np.isfinite(y)
                if scale == "log":
                    mask &= y > 0.0
                if not mask.any():
                    continue
                extra = []
                if meta.reconstruction:
                    extra.append(meta.reconstruction.upper())
                if meta.resolution:
                    extra.append(f"N={meta.resolution}")
                suffix = f" ({', '.join(extra)})" if extra else ""
                ax.plot(
                    x[mask],
                    y[mask],
                    lw=1.5,
                    label=method_label(meta.method) + suffix,
                    color=METHOD_COLORS.get(meta.method),
                    alpha=0.9,
                )
                inputs.append(path)
                methods.append(meta.method)
                plotted += 1
            if plotted == 0:
                plt.close(fig)
                continue
            ax.set_xlabel("time")
            ax.set_ylabel(label)
            ax.set_title(f"{PROBLEM_LABELS.get(prob, prob)} / {kind}: {label}")
            ax.grid(True, which="both", alpha=0.25)
            apply_scale(ax, scale)
            ax.legend(fontsize=7, ncol=2)
            save_figure(
                fig,
                f"{out_dir}/compare_by_problem/{sanitize(prob)}_{sanitize(kind)}_{sanitize(col)}.png",
                group_name,
                inputs,
                label,
                methods,
                prob,
                f"Comparison of {label} for {prob} {kind} runs.",
                useful="maybe",
                backup="maybe",
                also_pdf=(col in {"L2_fv", "L2_norm_fv", "min_pressure", "total_energy_rel"}),
            )

    # Comparison by method across problems/run kinds.
    by_method: dict[str, list[tuple[Path, Meta, pd.DataFrame]]] = defaultdict(list)
    for item in data:
        by_method[item[1].method].append(item)
    for method, items in sorted(by_method.items(), key=lambda x: method_sort_key(x[0])):
        for col, label, scale in [
            ("L2_norm_fv", "normalized FV L2(divB)", "log"),
            ("L2_fv", "FV L2(divB)", "log"),
            ("min_pressure", "minimum pressure", "linear"),
            ("total_energy_rel", "relative energy drift", "symlog"),
        ]:
            if not any(col in df.columns for _, _, df in items):
                continue
            fig, ax = plt.subplots(figsize=(9.5, 5.5))
            inputs = []
            plotted = 0
            for path, meta, df in sorted(items, key=lambda x: (x[1].problem, x[1].run_kind, x[1].stem)):
                if col not in df.columns:
                    continue
                x = finite_series(df, "time")
                y = finite_series(df, col)
                mask = np.isfinite(x) & np.isfinite(y)
                if scale == "log":
                    mask &= y > 0.0
                if not mask.any():
                    continue
                label_text = f"{PROBLEM_LABELS.get(meta.problem, meta.problem)} {meta.run_kind}"
                if meta.reconstruction or meta.resolution:
                    label_text += f" {meta.reconstruction} {meta.resolution}".strip()
                ax.plot(x[mask], y[mask], lw=1.4, label=label_text, alpha=0.85)
                inputs.append(path)
                plotted += 1
            if plotted == 0:
                plt.close(fig)
                continue
            ax.set_xlabel("time")
            ax.set_ylabel(label)
            ax.set_title(f"{method_label(method)} across result sets: {label}")
            ax.grid(True, which="both", alpha=0.25)
            apply_scale(ax, scale)
            ax.legend(fontsize=7, ncol=2)
            save_figure(
                fig,
                f"{out_dir}/compare_by_method/{sanitize(method)}_{sanitize(col)}.png",
                group_name,
                inputs,
                label,
                [method],
                "multiple",
                f"Comparison of {label} for {method_label(method)} across problems/run types.",
                useful="maybe",
                backup="yes",
            )


def plot_glm_histories() -> None:
    glm1 = sorted((RESULTS / "glm_1d").glob("*.csv"))
    glm2 = sorted((RESULTS / "glm_2d" / "divergence").glob("*.csv"))
    plot_time_series_collection(glm1, "glm_1d", "glm_1d", "glm_1d")
    plot_time_series_collection(glm2, "glm_2d", "glm_2d", "glm_2d")

    for files, out_dir, group, metrics in [
        (glm1, "glm_1d/comparisons", "glm_1d", ["L1", "L2", "Linf"]),
        (
            glm2,
            "glm_2d/comparisons",
            "glm_2d",
            [
                "L1_centered",
                "L2_centered",
                "Linf_centered",
                "L1_fv",
                "L2_fv",
                "Linf_fv",
                "L1_norm_fv",
                "L2_norm_fv",
                "Linf_norm_fv",
            ],
        ),
    ]:
        loaded = []
        for path in files:
            df = read_csv(path)
            if df is not None and "time" in df.columns:
                loaded.append((path, parse_meta(path), df))
        for metric in metrics:
            if not any(metric in df.columns for _, _, df in loaded):
                continue
            fig, ax = plt.subplots(figsize=(8.0, 5.0))
            inputs = []
            methods = []
            for path, meta, df in sorted(loaded, key=lambda x: method_sort_key(x[1].method)):
                if metric not in df.columns:
                    continue
                x = finite_series(df, "time")
                y = finite_series(df, metric)
                mask = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
                if not mask.any():
                    continue
                ax.semilogy(x[mask], y[mask], lw=1.8, label=method_label(meta.method), color=METHOD_COLORS.get(meta.method))
                inputs.append(path)
                methods.append(meta.method)
            ax.set_xlabel("time")
            ax.set_ylabel(metric)
            ax.set_title(f"{group}: {metric} comparison")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=8)
            save_figure(
                fig,
                f"{out_dir}/{sanitize(metric)}_comparison.png",
                group,
                inputs,
                metric,
                methods,
                group,
                f"Standalone {group} comparison for {metric}.",
                useful="maybe",
                backup="maybe",
                also_pdf=(metric in {"L2", "L2_fv", "L2_norm_fv"}),
            )


def combine_summary_tables() -> pd.DataFrame:
    paths = sorted((RESULTS / "mhd_runner" / "summaries").glob("*.csv"))
    extra = [
        FIG_MHD / "mhd_runner_summary.csv",
        FIG_MHD / "cleaning_diagnostics_summary.csv",
    ]
    rows = []
    for path in paths + [p for p in extra if p.exists()]:
        df = read_csv(path)
        if df is None or df.empty:
            continue
        meta = parse_meta(path)
        df = df.copy()
        df["source_file"] = str(path)
        if "problem" not in df.columns:
            df["problem"] = meta.problem
        if "method" not in df.columns:
            df["method"] = meta.method
        df["run_kind"] = df.get("run_kind", meta.run_kind)
        df["parsed_stem"] = meta.stem
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def plot_summary_visualizations() -> pd.DataFrame:
    df = combine_summary_tables()
    if df.empty:
        warn("no summary CSVs found")
        return df

    # Normalize a few columns from figure-level summaries.
    if "final_L2_norm_fv" not in df.columns and "final_L2_fv" in df.columns:
        df["final_L2_norm_fv"] = np.nan

    for col in df.columns:
        if col not in {"problem", "method", "status", "failure_reason", "energy_policy", "source_file", "run_kind", "parsed_stem", "reconstruction", "limiter"}:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0 or df[col].notna().sum() == 0:
                df[col] = converted

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = [c for c in numeric_cols if c not in {"final_time_reached"}]

    # All numeric summary columns as bar charts.
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        if not np.isfinite(values).any():
            continue
        fig, ax = plt.subplots(figsize=(max(10.0, min(28.0, 0.20 * len(df))), 5.2))
        labels = [
            f"{str(r.get('problem', ''))[:3]}:{method_label(str(r.get('method', '')))}:{str(r.get('parsed_stem', i))[-12:]}"
            for i, r in df.iterrows()
        ]
        x = np.arange(len(df))
        colors = [METHOD_COLORS.get(str(m), "#777777") for m in df.get("method", pd.Series([""] * len(df)))]
        ax.bar(x, values.to_numpy(dtype=float), color=colors, alpha=0.85)
        ax.set_ylabel(col)
        ax.set_title(f"Summary numeric column: {col}")
        if len(df) <= 80:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=80, ha="right", fontsize=6)
        else:
            ax.set_xticks([])
            ax.set_xlabel("summary rows")
        if col.lower().startswith("final_l") or "residual" in col or "wall_time" in col or "seconds" in col:
            positive = values[values > 0]
            if len(positive) and positive.max() / max(positive.min(), 1e-300) > 100:
                ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.25)
        save_figure(
            fig,
            f"summaries/all_numeric_columns/{sanitize(col)}.png",
            "summaries",
            [Path(p) for p in df["source_file"].dropna().unique()],
            col,
            [str(m) for m in df.get("method", [])],
            "multiple",
            f"Bar chart for summary column {col} across all summary rows.",
            useful="maybe" if col in IMPORTANT_SUMMARY_COLUMNS else "no",
            backup="yes" if col not in IMPORTANT_SUMMARY_COLUMNS else "maybe",
        )

    # Heatmaps by problem x method for important columns.
    if {"problem", "method"}.issubset(df.columns):
        for col in sorted(IMPORTANT_SUMMARY_COLUMNS & set(df.columns)):
            vals = pd.to_numeric(df[col], errors="coerce")
            if not np.isfinite(vals).any():
                continue
            tmp = df.copy()
            tmp[col] = vals
            table = tmp.pivot_table(index="problem", columns="method", values=col, aggfunc="last")
            if table.empty:
                continue
            cols = sorted(table.columns, key=method_sort_key)
            table = table[cols]
            fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 3, 0.55 * len(table.index) + 2.5))
            arr = table.to_numpy(dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                plt.close(fig)
                continue
            norm = None
            if (col.lower().startswith("final_l") or "residual" in col) and np.nanmax(finite) > 0:
                positive = finite[finite > 0]
                if positive.size:
                    norm = mcolors.LogNorm(vmin=max(float(np.nanmin(positive)), 1e-300), vmax=float(np.nanmax(positive)))
            im = ax.imshow(arr, aspect="auto", cmap="viridis", norm=norm)
            ax.set_xticks(np.arange(len(cols)))
            ax.set_xticklabels([method_label(c) for c in cols], rotation=35, ha="right")
            ax.set_yticks(np.arange(len(table.index)))
            ax.set_yticklabels([PROBLEM_LABELS.get(str(p), str(p)) for p in table.index])
            ax.set_title(f"Summary heatmap: {col}")
            fig.colorbar(im, ax=ax, shrink=0.85)
            save_figure(
                fig,
                f"summaries/heatmaps/problem_method_{sanitize(col)}.png",
                "summaries",
                [Path(p) for p in df["source_file"].dropna().unique()],
                col,
                list(cols),
                "multiple",
                f"Problem by method heatmap for {col}.",
                useful="yes" if col in IMPORTANT_SUMMARY_COLUMNS else "maybe",
                backup="maybe",
                also_pdf=(col in {"final_L2_fv", "final_L2_norm_fv", "min_pressure", "energy_drift"}),
            )

    # Robustness heatmaps from summary status/final_time_reached.
    if {"problem", "method"}.issubset(df.columns):
        complete = pd.Series(np.nan, index=df.index)
        if "final_time_reached" in df.columns:
            complete = pd.to_numeric(df["final_time_reached"], errors="coerce")
        elif "completed" in df.columns:
            complete = pd.to_numeric(df["completed"], errors="coerce")
        if np.isfinite(complete).any():
            tmp = df.copy()
            tmp["completed_numeric"] = complete
            table = tmp.pivot_table(index="problem", columns="method", values="completed_numeric", aggfunc="last")
            cols = sorted(table.columns, key=method_sort_key)
            table = table[cols]
            fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 3, 0.55 * len(table.index) + 2.5))
            im = ax.imshow(table.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_xticks(np.arange(len(cols)))
            ax.set_xticklabels([method_label(c) for c in cols], rotation=35, ha="right")
            ax.set_yticks(np.arange(len(table.index)))
            ax.set_yticklabels([PROBLEM_LABELS.get(str(p), str(p)) for p in table.index])
            ax.set_title("Robustness heatmap: completion status")
            fig.colorbar(im, ax=ax, shrink=0.85, label="completed")
            save_figure(
                fig,
                "failures_and_robustness/robustness_problem_method_completion.png",
                "failures_and_robustness",
                [Path(p) for p in df["source_file"].dropna().unique()],
                "completion status",
                list(cols),
                "multiple",
                "Problem by method completion/failure heatmap from summary files.",
                useful="yes",
                backup="no",
                also_pdf=True,
            )

    # Trade-off scatter plots.
    for xcol, ycol in SUMMARY_TRADEOFFS:
        if xcol not in df.columns or ycol not in df.columns:
            continue
        x = pd.to_numeric(df[xcol], errors="coerce")
        y = pd.to_numeric(df[ycol], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            continue
        fig, ax = plt.subplots(figsize=(8.0, 5.8))
        methods = []
        for method, group in df[mask].groupby("method", dropna=False):
            idx = group.index
            ax.scatter(
                x.loc[idx],
                y.loc[idx],
                label=method_label(str(method)),
                color=METHOD_COLORS.get(str(method), None),
                s=45,
                alpha=0.82,
                edgecolor="black",
                linewidth=0.3,
            )
            methods.append(str(method))
        if xcol.lower().startswith("final_l"):
            ax.set_xscale("log")
        if ycol.lower().startswith("final_l") or ycol in {"total_wall_time_sec", "seconds_per_step"}:
            positive = y[y > 0]
            if len(positive) and positive.max() / max(positive.min(), 1e-300) > 100:
                ax.set_yscale("log")
        if "energy_drift" in ycol:
            ax.set_yscale("symlog", linthresh=1e-10)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(f"Trade-off: {xcol} vs {ycol}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        save_figure(
            fig,
            f"summaries/tradeoffs/{sanitize(xcol)}_vs_{sanitize(ycol)}.png",
            "summaries",
            [Path(p) for p in df["source_file"].dropna().unique()],
            f"{xcol} vs {ycol}",
            methods,
            "multiple",
            f"Summary trade-off scatter for {xcol} and {ycol}.",
            useful="yes",
            backup="no",
            also_pdf=True,
        )

    return df


def derive_snapshot_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"u", "v", "w"}.issubset(out.columns):
        out["vmag"] = np.sqrt(out["u"] ** 2 + out["v"] ** 2 + out["w"] ** 2)
    if {"Bx", "By", "Bz"}.issubset(out.columns):
        out["Bmag_derived"] = np.sqrt(out["Bx"] ** 2 + out["By"] ** 2 + out["Bz"] ** 2)
        out["magnetic_pressure"] = 0.5 * out["Bmag_derived"] ** 2
    if {"rho", "u", "v", "w"}.issubset(out.columns):
        out["kinetic_energy_density"] = 0.5 * out["rho"] * (out["u"] ** 2 + out["v"] ** 2 + out["w"] ** 2)
    if "p" in out.columns:
        out["internal_energy_density"] = out["p"] / (GAMMA - 1.0)
    if "divB_fv" in out.columns:
        out["abs_divB_fv"] = np.abs(out["divB_fv"])
        if "Bmag" in out.columns:
            dx = infer_spacing(out, "x")
            dy = infer_spacing(out, "y")
            h = math.sqrt(dx * dy) if dx and dy else 1.0
            out["normalized_abs_divB_fv"] = h * out["abs_divB_fv"] / (out["Bmag"].abs() + 1e-30)
    return out


def infer_spacing(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    vals = np.sort(pd.to_numeric(df[col], errors="coerce").dropna().unique())
    if len(vals) < 2:
        return None
    diffs = np.diff(vals)
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if len(diffs) else None


def snapshot_to_grid(df: pd.DataFrame, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not {"i", "j", field}.issubset(df.columns):
        return None
    nx = int(pd.to_numeric(df["i"], errors="coerce").max()) + 1
    ny = int(pd.to_numeric(df["j"], errors="coerce").max()) + 1
    ordered = df.sort_values(["j", "i"])
    if len(ordered) != nx * ny:
        warn(f"snapshot grid is incomplete for field {field}")
        return None
    x = ordered[ordered["j"] == ordered["j"].min()].sort_values("i")["x"].to_numpy() if "x" in ordered.columns else np.arange(nx)
    y = ordered[ordered["i"] == ordered["i"].min()].sort_values("j")["y"].to_numpy() if "y" in ordered.columns else np.arange(ny)
    z = pd.to_numeric(ordered[field], errors="coerce").to_numpy().reshape(ny, nx)
    return x, y, z


def field_norm(values: np.ndarray, field: str, shared_limits: tuple[float, float] | None = None):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return mcolors.Normalize(vmin=0, vmax=1)
    vmin = float(np.nanmin(finite)) if shared_limits is None else shared_limits[0]
    vmax = float(np.nanmax(finite)) if shared_limits is None else shared_limits[1]
    if vmin == vmax:
        vmax = vmin + 1.0
    if field in {"abs_divB_fv", "normalized_abs_divB_fv", "Bmag", "Bmag_derived", "magnetic_pressure", "kinetic_energy_density"}:
        positive = finite[finite > 0]
        if positive.size and float(np.nanmax(positive)) / max(float(np.nanmin(positive)), 1e-300) > 100:
            return mcolors.LogNorm(vmin=max(float(np.nanmin(positive)), 1e-300), vmax=max(vmax, max(float(np.nanmin(positive)) * 10, 1e-300)))
    if field in {"divB_fv", "Bx", "By", "Bz", "u", "v", "w", "psi"} and vmin < 0 < vmax:
        lim = max(abs(vmin), abs(vmax))
        return mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def plot_snapshots() -> None:
    paths = sorted((RESULTS / "mhd_runner" / "snapshots").glob("*.csv"))
    if not paths:
        warn("no snapshot CSVs found")
        return
    loaded = []
    for path in paths:
        df = read_csv(path)
        if df is None or not {"i", "j"}.issubset(df.columns):
            continue
        loaded.append((path, parse_meta(path), derive_snapshot_fields(df)))

    preferred_fields = [
        "rho",
        "p",
        "u",
        "v",
        "w",
        "vmag",
        "Bx",
        "By",
        "Bz",
        "Bmag",
        "Bmag_derived",
        "psi",
        "divB_fv",
        "abs_divB_fv",
        "normalized_abs_divB_fv",
        "magnetic_pressure",
        "kinetic_energy_density",
        "internal_energy_density",
    ]

    for path, meta, df in loaded:
        fields = [f for f in preferred_fields if f in df.columns]
        if not fields:
            continue
        ncols = 4
        nrows = math.ceil(len(fields) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows), squeeze=False)
        for ax, field in zip(axes.ravel(), fields):
            grid = snapshot_to_grid(df, field)
            if grid is None:
                ax.set_axis_off()
                continue
            x, y, z = grid
            im = ax.pcolormesh(x, y, z, shading="auto", cmap="coolwarm" if field in {"divB_fv", "Bx", "By", "Bz", "u", "v", "w", "psi"} else "viridis", norm=field_norm(z, field))
            ax.set_title(field)
            ax.set_aspect("equal")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
        for ax in axes.ravel()[len(fields):]:
            ax.set_axis_off()
        fig.suptitle(f"{meta.stem}: snapshot fields", y=1.0)
        save_figure(
            fig,
            f"snapshots/per_run/{sanitize(meta.stem)}_fields.png",
            "snapshots",
            [path],
            "all spatial fields",
            [meta.method],
            meta.problem,
            f"Spatial maps for all available and derived snapshot fields in {meta.stem}.",
            useful="maybe",
            backup="yes",
        )

    # Compare every common field across science snapshots for each problem.
    science = [(p, m, d) for p, m, d in loaded if m.run_kind == "science"]
    by_problem: dict[str, list[tuple[Path, Meta, pd.DataFrame]]] = defaultdict(list)
    for item in science:
        by_problem[item[1].problem].append(item)
    for prob, items in sorted(by_problem.items()):
        common_fields = sorted(set().union(*(set(df.columns) for _, _, df in items)))
        fields = [f for f in preferred_fields if f in common_fields]
        for field in fields:
            valid = [(p, m, d) for p, m, d in items if field in d.columns and snapshot_to_grid(d, field) is not None]
            if len(valid) < 2:
                continue
            methods = [m.method for _, m, _ in valid]
            ncols = min(4, len(valid))
            nrows = math.ceil(len(valid) / ncols)
            arrays = [snapshot_to_grid(d, field)[2] for _, _, d in valid]  # type: ignore[index]
            finite = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays if np.isfinite(a).any()])
            shared = (float(np.nanmin(finite)), float(np.nanmax(finite))) if finite.size else None
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.7 * nrows), squeeze=False)
            for ax, (path, meta, df), arr in zip(axes.ravel(), valid, arrays):
                x, y, z = snapshot_to_grid(df, field)  # type: ignore[misc]
                im = ax.pcolormesh(x, y, z, shading="auto", cmap="coolwarm" if field in {"divB_fv", "Bx", "By", "Bz", "u", "v", "w", "psi"} else "viridis", norm=field_norm(z, field, shared))
                ax.set_title(method_label(meta.method))
                ax.set_aspect("equal")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
            for ax in axes.ravel()[len(valid):]:
                ax.set_axis_off()
            fig.suptitle(f"{PROBLEM_LABELS.get(prob, prob)}: {field} across methods", y=1.0)
            save_figure(
                fig,
                f"snapshots/compare_by_field/{sanitize(prob)}_{sanitize(field)}_all_science_methods.png",
                "snapshots",
                [p for p, _, _ in valid],
                field,
                methods,
                prob,
                f"Shared-scale method comparison for snapshot field {field} in {prob}.",
                useful="yes" if field in {"rho", "p", "abs_divB_fv", "normalized_abs_divB_fv", "Bmag"} else "maybe",
                backup="maybe",
                also_pdf=(field in {"rho", "p", "abs_divB_fv", "normalized_abs_divB_fv"}),
            )

    # Focused method-family panels.
    focus_groups = {
        "none_mixed_projection": ["none", "mixed_glm", "elliptic_projection"],
        "glm_eglm_family": ["none", "hyperbolic_glm", "mixed_glm", "mixed_eglm", "gi_mixed_eglm"],
        "source_term_family": ["none", "mixed_eglm", "gi_mixed_eglm", "powell_source"],
    }
    for prob, items in sorted(by_problem.items()):
        by_method = {m.method: (p, m, d) for p, m, d in items}
        for group_name, methods in focus_groups.items():
            selected = [by_method[m] for m in methods if m in by_method]
            if len(selected) < 2:
                continue
            for field in ["rho", "p", "abs_divB_fv", "normalized_abs_divB_fv", "Bmag", "vmag"]:
                valid = [(p, m, d) for p, m, d in selected if field in d.columns and snapshot_to_grid(d, field) is not None]
                if len(valid) < 2:
                    continue
                arrays = [snapshot_to_grid(d, field)[2] for _, _, d in valid]  # type: ignore[index]
                finite = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays if np.isfinite(a).any()])
                shared = (float(np.nanmin(finite)), float(np.nanmax(finite))) if finite.size else None
                fig, axes = plt.subplots(1, len(valid), figsize=(4.0 * len(valid), 3.7), squeeze=False)
                for ax, (path, meta, df) in zip(axes.ravel(), valid):
                    x, y, z = snapshot_to_grid(df, field)  # type: ignore[misc]
                    im = ax.pcolormesh(x, y, z, shading="auto", cmap="coolwarm" if field in {"divB_fv"} else "viridis", norm=field_norm(z, field, shared))
                    ax.set_title(method_label(meta.method))
                    ax.set_aspect("equal")
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
                fig.suptitle(f"{PROBLEM_LABELS.get(prob, prob)}: {group_name} / {field}", y=1.02)
                save_figure(
                    fig,
                    f"snapshots/focused_panels/{sanitize(prob)}_{sanitize(group_name)}_{sanitize(field)}.png",
                    "snapshots",
                    [p for p, _, _ in valid],
                    field,
                    [m.method for _, m, _ in valid],
                    prob,
                    f"Focused snapshot comparison for {group_name}, field {field}.",
                    useful="yes" if group_name == "none_mixed_projection" and field in {"rho", "p", "abs_divB_fv"} else "maybe",
                    backup="maybe",
                )


def plot_convergence() -> None:
    paths = sorted((RESULTS / "mhd_runner" / "convergence").glob("*.csv"))
    for path in paths:
        df = read_csv(path)
        if df is None or df.empty:
            continue
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or c in {"N", "dx"}]
        ycols = [c for c in numeric_cols if c not in {"N", "dx", "tfinal", "estimated_slope"}]
        x_candidates = [c for c in ["dx", "N"] if c in df.columns]
        for xcol in x_candidates:
            for ycol in ycols:
                x = pd.to_numeric(df[xcol], errors="coerce")
                y = pd.to_numeric(df[ycol], errors="coerce")
                mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                if mask.sum() < 2:
                    continue
                fig, ax = plt.subplots(figsize=(7.5, 5.3))
                group_col = "reconstruction" if "reconstruction" in df.columns else None
                if group_col:
                    for name, g in df[mask].groupby(group_col):
                        gx = pd.to_numeric(g[xcol], errors="coerce")
                        gy = pd.to_numeric(g[ycol], errors="coerce")
                        order = np.argsort(gx)
                        slope = np.nan
                        if len(gx) >= 2:
                            coeff = np.polyfit(np.log(gx), np.log(gy), 1)
                            slope = coeff[0]
                        ax.loglog(gx.iloc[order], gy.iloc[order], marker="o", lw=1.7, label=f"{name} empirical slope={slope:.2f}")
                else:
                    order = np.argsort(x[mask])
                    coeff = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
                    ax.loglog(x[mask].iloc[order], y[mask].iloc[order], marker="o", lw=1.7, label=f"empirical slope={coeff[0]:.2f}")
                if xcol == "dx":
                    ax.invert_xaxis()
                ax.set_xlabel(xcol)
                ax.set_ylabel(ycol)
                ax.set_title(f"{path.stem}: {ycol} vs {xcol}")
                ax.grid(True, which="both", alpha=0.25)
                ax.legend(fontsize=8)
                save_figure(
                    fig,
                    f"convergence/{sanitize(path.stem)}_{sanitize(ycol)}_vs_{sanitize(xcol)}.png",
                    "convergence",
                    [path],
                    f"{ycol} vs {xcol}",
                    [],
                    parse_meta(path).problem,
                    f"Convergence plot with empirical slopes for {path.name}.",
                    useful="yes" if ycol in {"final_L2_norm_fv", "l2_B_vector"} else "maybe",
                    backup="maybe",
                    also_pdf=(ycol in {"final_L2_norm_fv", "l2_B_vector"}),
                )

    hrsc = RESULTS / "mhd_runner" / "hrsc" / "pcm_vs_plm_summary.csv"
    if hrsc.exists():
        df = read_csv(hrsc)
        if df is not None:
            for col in [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in {"N"}]:
                fig, ax = plt.subplots(figsize=(6.8, 4.8))
                labels = df["reconstruction"].astype(str) if "reconstruction" in df.columns else np.arange(len(df)).astype(str)
                ax.bar(labels, pd.to_numeric(df[col], errors="coerce"), color="#4c78a8")
                ax.set_ylabel(col)
                ax.set_title(f"HRSC PCM vs PLM: {col}")
                ax.grid(True, axis="y", alpha=0.25)
                save_figure(
                    fig,
                    f"convergence/hrsc_pcm_vs_plm_{sanitize(col)}.png",
                    "convergence",
                    [hrsc],
                    col,
                    list(df.get("method", [])),
                    "field_loop",
                    f"PCM vs PLM summary for {col}.",
                    useful="maybe",
                    backup="maybe",
                )


def plot_performance() -> None:
    paths = sorted((RESULTS / "mhd_runner" / "performance").glob("*.csv"))
    for path in paths:
        df = read_csv(path)
        if df is None or df.empty:
            continue
        if "method" not in df.columns:
            continue
        metrics = [
            "total_wall_time_sec",
            "seconds_per_step",
            "cell_updates_per_second",
            "cleaning_time_sec",
            "hydro_time_sec",
            "diagnostics_compute_time_sec",
            "output_time_sec",
            "final_L2_norm_fv",
            "min_pressure",
            "energy_drift",
            "projection_iterations_total",
            "cleaning_subcycles_total",
        ]
        for xcol in ["ncell", "nx"]:
            if xcol not in df.columns:
                continue
            for ycol in [m for m in metrics if m in df.columns]:
                x = pd.to_numeric(df[xcol], errors="coerce")
                y = pd.to_numeric(df[ycol], errors="coerce")
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 2:
                    continue
                fig, ax = plt.subplots(figsize=(8.5, 5.5))
                methods = []
                for method, g in df[mask].groupby("method"):
                    gx = pd.to_numeric(g[xcol], errors="coerce")
                    gy = pd.to_numeric(g[ycol], errors="coerce")
                    order = np.argsort(gx)
                    ax.plot(gx.iloc[order], gy.iloc[order], marker="o", lw=1.7, label=method_label(str(method)), color=METHOD_COLORS.get(str(method)))
                    methods.append(str(method))
                if xcol == "ncell":
                    ax.set_xscale("log")
                if ycol in {"total_wall_time_sec", "seconds_per_step", "cell_updates_per_second", "final_L2_norm_fv"}:
                    positive = y[y > 0]
                    if len(positive):
                        ax.set_yscale("log")
                elif ycol == "energy_drift":
                    ax.set_yscale("symlog", linthresh=1e-10)
                ax.set_xlabel(xcol)
                ax.set_ylabel(ycol)
                ax.set_title(f"{path.stem}: {ycol} vs {xcol}")
                ax.grid(True, which="both", alpha=0.25)
                ax.legend(fontsize=8, ncol=2)
                save_figure(
                    fig,
                    f"performance/{sanitize(path.stem)}_{sanitize(ycol)}_vs_{sanitize(xcol)}.png",
                    "performance",
                    [path],
                    f"{ycol} vs {xcol}",
                    methods,
                    "divergence_advection",
                    f"Performance scaling plot for {ycol} vs {xcol}.",
                    useful="yes" if ycol in {"total_wall_time_sec", "cell_updates_per_second", "final_L2_norm_fv"} else "maybe",
                    backup="maybe",
                    also_pdf=(ycol in {"total_wall_time_sec", "cell_updates_per_second"}),
                )

        timing_cols = [c for c in ["hydro_time_sec", "cleaning_time_sec", "diagnostics_compute_time_sec", "diagnostics_write_time_sec", "output_time_sec"] if c in df.columns]
        if timing_cols and {"nx", "method"}.issubset(df.columns):
            for nx, g in df.groupby("nx"):
                fig, ax = plt.subplots(figsize=(max(8.0, 0.7 * len(g)), 5.2))
                g = g.sort_values("method", key=lambda s: s.map(lambda m: method_sort_key(str(m))[0]))
                bottom = np.zeros(len(g))
                x = np.arange(len(g))
                for col in timing_cols:
                    vals = pd.to_numeric(g[col], errors="coerce").fillna(0).to_numpy()
                    ax.bar(x, vals, bottom=bottom, label=col)
                    bottom += vals
                ax.set_xticks(x)
                ax.set_xticklabels([method_label(str(m)) for m in g["method"]], rotation=35, ha="right")
                ax.set_ylabel("seconds")
                ax.set_title(f"{path.stem}: runtime breakdown N={nx}")
                ax.legend(fontsize=8)
                ax.grid(True, axis="y", alpha=0.25)
                save_figure(
                    fig,
                    f"performance/{sanitize(path.stem)}_runtime_breakdown_n{sanitize(nx)}.png",
                    "performance",
                    [path],
                    "runtime breakdown",
                    [str(m) for m in g["method"]],
                    "divergence_advection",
                    f"Runtime component breakdown for N={nx}.",
                    useful="maybe",
                    backup="maybe",
                )

        if {"total_wall_time_sec", "final_L2_norm_fv", "method"}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(8.0, 5.5))
            methods = []
            for method, g in df.groupby("method"):
                x = pd.to_numeric(g["total_wall_time_sec"], errors="coerce")
                y = pd.to_numeric(g["final_L2_norm_fv"], errors="coerce")
                mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                if mask.any():
                    ax.scatter(x[mask], y[mask], s=55, label=method_label(str(method)), color=METHOD_COLORS.get(str(method)), edgecolor="black", linewidth=0.3)
                    methods.append(str(method))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("wall time [s]")
            ax.set_ylabel("final normalized FV L2(divB)")
            ax.set_title(f"{path.stem}: accuracy-cost scatter")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=8, ncol=2)
            save_figure(
                fig,
                f"performance/{sanitize(path.stem)}_accuracy_cost_final_L2_vs_walltime.png",
                "performance",
                [path],
                "accuracy-cost",
                methods,
                "divergence_advection",
                "Accuracy-cost scatter using final normalized divergence and wall time.",
                useful="yes",
                backup="no",
                also_pdf=True,
            )

    # MPI sweep summaries.
    mpi_paths = sorted((RESULTS / "mhd_sweep_mpi").rglob("*.csv"))
    if mpi_paths:
        frames = []
        for p in mpi_paths:
            df = read_csv(p)
            if df is not None:
                df = df.copy()
                df["source_file"] = str(p)
                frames.append(df)
        if frames:
            df = pd.concat(frames, ignore_index=True, sort=False)
            for col in ["runtime_sec", "final_divB_L2", "final_divB_Linf", "min_pressure", "energy_drift"]:
                if col not in df.columns:
                    continue
                fig, ax = plt.subplots(figsize=(8.0, 4.8))
                labels = [f"r{r}:{method_label(str(m))}" for r, m in zip(df.get("rank", range(len(df))), df.get("method", [""] * len(df)))]
                ax.bar(np.arange(len(df)), pd.to_numeric(df[col], errors="coerce"), color="#4c78a8")
                ax.set_xticks(np.arange(len(df)))
                ax.set_xticklabels(labels, rotation=35, ha="right")
                ax.set_ylabel(col)
                ax.set_title(f"MPI sweep: {col}")
                if col.startswith("final_divB") or col == "runtime_sec":
                    positive = pd.to_numeric(df[col], errors="coerce")
                    if (positive > 0).any():
                        ax.set_yscale("log")
                ax.grid(True, axis="y", alpha=0.25)
                save_figure(
                    fig,
                    f"performance/mpi_sweep_{sanitize(col)}.png",
                    "performance",
                    mpi_paths,
                    col,
                    [str(m) for m in df.get("method", [])],
                    "divergence_advection",
                    f"MPI coarse-grained sweep summary for {col}.",
                    useful="maybe",
                    backup="yes",
                )


def plot_failures_and_families(summary_df: pd.DataFrame) -> None:
    failure_dir = RESULTS / "mhd_runner" / "failures"
    failure_paths = sorted(failure_dir.glob("*.csv")) if failure_dir.exists() else []
    if not failure_paths:
        warn("results/mhd_runner/failures is missing or empty; generating robustness figures only from summaries")
    else:
        frames = []
        for p in failure_paths:
            df = read_csv(p)
            if df is not None:
                df["source_file"] = str(p)
                frames.append(df)
        if frames:
            df = pd.concat(frames, ignore_index=True, sort=False)
            for col in ["failure_time", "pressure", "rho", "internal_energy", "magnetic_energy", "kinetic_energy", "divB"]:
                if col not in df.columns:
                    continue
                fig, ax = plt.subplots(figsize=(9.0, 5.0))
                labels = df.get("method", pd.Series(np.arange(len(df)).astype(str))).astype(str)
                ax.bar(np.arange(len(df)), pd.to_numeric(df[col], errors="coerce"))
                ax.set_xticks(np.arange(len(df)))
                ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
                ax.set_ylabel(col)
                ax.set_title(f"Failure diagnostics: {col}")
                ax.grid(True, axis="y", alpha=0.25)
                save_figure(
                    fig,
                    f"failures_and_robustness/failure_csv_{sanitize(col)}.png",
                    "failures_and_robustness",
                    failure_paths,
                    col,
                    list(labels),
                    "multiple",
                    f"Failure CSV diagnostic {col}.",
                    useful="maybe",
                    backup="yes",
                )

    if summary_df.empty or "method" not in summary_df.columns:
        return
    df = summary_df.copy()
    df["family"] = df["method"].astype(str).map(lambda m: FAMILY.get(m, "other"))
    for col in ["final_L2_fv", "final_L2_norm_fv", "min_pressure", "min_density", "energy_drift", "total_wall_time_sec", "cleaning_subcycles_total", "projection_iterations_total"]:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if not np.isfinite(vals).any():
            continue
        tmp = df.copy()
        tmp[col] = vals
        table = tmp.groupby("family")[col].median().sort_values()
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        x = np.arange(len(table))
        ax.bar(x, table.to_numpy(dtype=float), color="#4c78a8")
        ax.set_xticks(x)
        ax.set_xticklabels(table.index.astype(str), rotation=25, ha="right")
        ax.set_ylabel(f"median {col}")
        ax.set_title(f"Method-family median: {col}")
        if col.lower().startswith("final_l") or col == "total_wall_time_sec":
            positive = table[table > 0]
            if len(positive):
                ax.set_yscale("log")
        elif col == "energy_drift":
            ax.set_yscale("symlog", linthresh=1e-10)
        ax.grid(True, axis="y", alpha=0.25)
        save_figure(
            fig,
            f"method_families/family_median_{sanitize(col)}.png",
            "method_families",
            [Path(p) for p in df["source_file"].dropna().unique()] if "source_file" in df.columns else [],
            col,
            list(df["method"].astype(str).unique()),
            "multiple",
            f"Family-level median comparison for {col}.",
            useful="maybe",
            backup="maybe",
        )


def write_figure_index() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(figure_rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "filename",
                "figure_group",
                "input_data_files",
                "diagnostic",
                "methods_included",
                "problem_included",
                "description",
                "likely_useful_for_final_presentation",
                "likely_backup_only",
            ]
        )
    df.to_csv(OUT / "figure_index.csv", index=False)
    lines = ["# Visual Inventory Figure Index", ""]
    lines.append(f"Generated figures: {len(df)}")
    lines.append("")
    for group, g in df.groupby("figure_group", dropna=False):
        lines.append(f"## {group}")
        for _, row in g.iterrows():
            lines.append(f"- `{row['filename']}`")
            lines.append(f"  - diagnostic: {row['diagnostic']}")
            lines.append(f"  - problem: {row['problem_included']}")
            lines.append(f"  - methods: {row['methods_included']}")
            lines.append(f"  - description: {row['description']}")
            lines.append(f"  - final-useful: {row['likely_useful_for_final_presentation']}; backup-only: {row['likely_backup_only']}")
        lines.append("")
    if warnings:
        lines += ["## Warnings", ""]
        for message in warnings:
            lines.append(f"- {message}")
    (OUT / "FIGURE_INDEX.md").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    groups = group_result_files()
    write_data_inventory(groups)

    plot_glm_histories()

    mhd_divergence = sorted((RESULTS / "mhd_runner" / "divergence").glob("*.csv"))
    plot_time_series_collection(mhd_divergence, "mhd_runner_divergence", "mhd_runner_divergence", "multiple")

    summary_df = plot_summary_visualizations()
    plot_snapshots()
    plot_convergence()
    plot_performance()
    plot_failures_and_families(summary_df)

    write_figure_index()
    print(f"Generated {len(figure_rows)} figures under {OUT}")
    print(f"Wrote {OUT / 'figure_index.csv'}")
    print(f"Wrote {OUT / 'FIGURE_INDEX.md'}")
    if warnings:
        print(f"Completed with {len(warnings)} warnings")


if __name__ == "__main__":
    main()
