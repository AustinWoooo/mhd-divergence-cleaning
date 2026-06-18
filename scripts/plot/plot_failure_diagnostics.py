#!/usr/bin/env python3
"""Plot the run-failure diagnostics in results/mhd_runner/failures.

The runner aborts a divB-cleaning method as soon as a full step leaves the
state unphysical (negative pressure/density or non-finite).  When that happens
it writes, per method:

  * <prefix>_<policy>_failure.csv        -- one row: where/when it failed, plus
                                            the min pressure recorded after each
                                            sub-stage of the step.
  * <prefix>_first_bad_cell_diagnostic.csv  -- before/after state of the cell
                                            that first went bad.
This script turns those records into three figures:

  figures/mhd_runner/failures/failure_stage_pressure.png
      Min pressure after each sub-stage, per method.  Shows *where* in the step
      positivity is lost (e.g. fine after hydro, negative after the cleaning B
      update for the low-beta blast).
  figures/mhd_runner/failures/failure_time.png
      When (sim time / step) each method died, annotated with the reason.
Run from the repository root:

    python scripts/plot/plot_failure_diagnostics.py
"""

from pathlib import Path
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".")
FAIL_DIR = ROOT / "results" / "mhd_runner" / "failures"
DIAG_DIR = ROOT / "results" / "mhd_runner" / "divergence"
FIG_DIR = ROOT / "figures" / "mhd_runner" / "failures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Sub-stage columns written into every *_failure.csv, in execution order.
STAGE_COLUMNS = [
    "min_pressure_before_hydro",
    "min_pressure_after_hydro",
    "min_pressure_after_cleaning_B",
    "min_pressure_after_energy_repair",
    "min_pressure_after_full_step",
]
STAGE_LABELS = [
    "before\nhydro",
    "after\nhydro",
    "after\ncleaning B",
    "after\nenergy repair",
    "after\nfull step",
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

COLORS = {
    "none": "tab:blue",
    "parabolic": "tab:red",
    "hyperbolic_glm": "tab:orange",
    "mixed_glm": "tab:green",
    "mixed_eglm": "tab:pink",
    "gi_mixed_eglm": "tab:olive",
    "elliptic_projection": "tab:purple",
    "powell_source": "tab:brown",
}

# Plot methods in a stable, readable order regardless of glob order on disk.
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


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def method_color(method: str) -> str:
    return COLORS.get(method, "tab:gray")


def method_sort_key(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def load_failures() -> pd.DataFrame:
    """Concatenate blast-wave *_failure.csv rows for supported methods."""
    rows = []
    supported = set(METHOD_ORDER)
    for path in sorted(FAIL_DIR.glob("mhd_blast_*_failure.csv")):
        df = pd.read_csv(path)
        if not df.empty:
            df = df[
                (df["problem"] == "blast_wave")
                & (df["method"].isin(supported))
            ].copy()
        if not df.empty:
            df["source_file"] = path.name
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values("method", key=lambda s: s.map(method_sort_key))
    return out.reset_index(drop=True)


def stage_rows_for_supported_methods(failures: pd.DataFrame) -> pd.DataFrame:
    """Return one stage row per supported blast method.

    Methods without a failure record completed the blast-wave diagnostic, so
    they have no failing sub-stage.  For those rows, use the minimum pressure
    reached over the run as a positive reference at each stage category.
    """
    by_method = {
        str(row["method"]): row.copy()
        for _, row in failures.iterrows()
    }
    rows = []
    for method in METHOD_ORDER:
        if method in by_method:
            row = by_method[method]
            row["completed_without_failure"] = False
            rows.append(row)
            continue

        diag_path = DIAG_DIR / f"mhd_blast_{method}.csv"
        if not diag_path.exists():
            print(f"  warning: no failure or diagnostic CSV for {method}; skipping")
            continue
        diag = pd.read_csv(diag_path)
        if "min_pressure" not in diag.columns:
            print(f"  warning: {diag_path} has no min_pressure column; skipping")
            continue
        min_pressure = pd.to_numeric(diag["min_pressure"], errors="coerce").min()
        row = pd.Series(
            {
                "problem": "blast_wave",
                "method": method,
                "stage": "completed",
                "step": "",
                "time": pd.to_numeric(diag.get("time"), errors="coerce").max()
                if "time" in diag.columns
                else np.nan,
                "reason": "completed_without_positivity_loss",
                "source_file": diag_path.name,
                "completed_without_failure": True,
                **{column: min_pressure for column in STAGE_COLUMNS},
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def plot_stage_pressure(failures: pd.DataFrame) -> None:
    """Min pressure after each sub-stage, one line per method (symlog)."""
    stage_rows = stage_rows_for_supported_methods(failures)
    present = [c for c in STAGE_COLUMNS if c in stage_rows.columns]
    if not present:
        print("  (no stage min-pressure columns; skipping stage figure)")
        return

    x = np.arange(len(present))
    fig, ax = plt.subplots(figsize=(10.8, 6.0))

    plotted_methods = []
    for _, row in stage_rows.iterrows():
        method = row["method"]
        y = pd.to_numeric(row[present], errors="coerce").to_numpy(dtype=float)
        completed = bool(row.get("completed_without_failure", False))
        ax.plot(
            x, y,
            marker="s" if completed else "o",
            markersize=5.5,
            linewidth=1.8 if completed else 2.2,
            linestyle="--" if completed else "-",
            alpha=0.78 if completed else 1.0,
            color=method_color(method),
            label=method_label(method),
        )
        plotted_methods.append(method)

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LABELS[STAGE_COLUMNS.index(c)] for c in present])
    ax.set_ylabel("min pressure in domain")
    ax.set_title(
        "MHD blast wave -- where each method loses positivity\n"
        "(min pressure after each sub-stage of the failing step)"
    )
    ax.grid(True, which="both", axis="y", alpha=0.3)

    # Shade the region below zero (unphysical) to make crossings obvious.
    ax.axhspan(ax.get_ylim()[0], 0.0, color="red", alpha=0.05)
    ax.text(
        0.99,
        0.02,
        "Dashed lines: completed runs; value is min pressure over run",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

    out = FIG_DIR / "failure_stage_pressure.png"
    plt.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    print(f"  stage methods: {', '.join(plotted_methods)}")


def plot_failure_time(failures: pd.DataFrame) -> None:
    """Horizontal bar of failure time per method, annotated with step/reason."""
    if "time" not in failures.columns:
        print("  (no time column; skipping failure-time figure)")
        return

    methods = failures["method"].tolist()
    times = pd.to_numeric(failures["time"], errors="coerce").to_numpy(dtype=float)
    y = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(methods) + 2.0))
    ax.barh(
        y, times,
        color=[method_color(m) for m in methods],
        edgecolor="black", linewidth=0.5,
    )

    for yi, (_, row) in zip(y, failures.iterrows()):
        step = row.get("step", "")
        reason = str(row.get("reason", "")).replace("_", " ")
        stage = str(row.get("failure_stage", "")).replace("_", " ")
        note = f"step {step} | {reason}"
        if stage and stage != "nan":
            note += f"\n@ {stage}"
        ax.text(
            times[yi], yi, "  " + note,
            va="center", ha="left", fontsize=8,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([method_label(m) for m in methods])
    ax.invert_yaxis()
    ax.set_xlabel("simulation time at failure")
    ax.set_title("MHD blast wave -- when each cleaning method aborts")
    ax.grid(True, axis="x", alpha=0.3)
    ax.margins(x=0.25)

    out = FIG_DIR / "failure_time.png"
    plt.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_limiter_timeseries() -> None:
    """Pressure-limited Powell limiter time series, if present."""
    stats_files = sorted(FAIL_DIR.glob("*_limiter_stats.csv"))
    if not stats_files:
        print("  (no *_limiter_stats.csv; skipping limiter figure)")
        return

    for path in stats_files:
        df = pd.read_csv(path)
        if df.empty or "time" not in df.columns:
            continue
        t = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)

        fig, (ax_p, ax_d) = plt.subplots(
            2, 1, figsize=(9, 6.5), sharex=True
        )

        # Top: min pressure before/after the limited source term.
        if "min_pressure_before_source" in df.columns:
            ax_p.plot(
                t, df["min_pressure_before_source"],
                color="tab:blue", label="min p before source",
            )
        if "min_pressure_after_source" in df.columns:
            ax_p.plot(
                t, df["min_pressure_after_source"],
                color="tab:red", label="min p after source",
            )
        ax_p.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax_p.set_yscale("symlog", linthresh=1e-6)
        ax_p.set_ylabel("min pressure")
        ax_p.set_title(
            f"Pressure-limited Powell limiter history\n({path.name})"
        )
        ax_p.grid(True, which="both", alpha=0.3)
        ax_p.legend(loc="best", fontsize=9)

        # Bottom: max|divB| (log) with theta_min on a twin axis.
        if "max_abs_divB" in df.columns:
            ax_d.semilogy(
                t, df["max_abs_divB"],
                color="tab:purple", label="max |divB|",
            )
            ax_d.set_ylabel("max |divB|", color="tab:purple")
            ax_d.tick_params(axis="y", labelcolor="tab:purple")
        ax_d.set_xlabel("simulation time")
        ax_d.grid(True, which="both", alpha=0.3)

        if "theta_min" in df.columns:
            ax_t = ax_d.twinx()
            ax_t.plot(
                t, df["theta_min"],
                color="tab:green", alpha=0.7, label="theta_min",
            )
            ax_t.set_ylabel("limiter theta_min", color="tab:green")
            ax_t.tick_params(axis="y", labelcolor="tab:green")
            ax_t.set_ylim(-0.05, 1.05)

        out = FIG_DIR / "limiter_timeseries.png"
        plt.tight_layout()
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")


def main() -> None:
    if not FAIL_DIR.is_dir():
        raise SystemExit(f"failures directory not found: {FAIL_DIR}")

    failures = load_failures()
    if failures.empty:
        raise SystemExit(f"no *_failure.csv rows found in {FAIL_DIR}")

    print(f"Loaded {len(failures)} failure record(s) from {FAIL_DIR}")
    plot_stage_pressure(failures)
    plot_failure_time(failures)
    plot_limiter_timeseries()


if __name__ == "__main__":
    main()
