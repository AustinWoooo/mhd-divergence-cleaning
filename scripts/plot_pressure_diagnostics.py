#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path(".")
RESULTS = ROOT / "results" / "mhd_runner"
DIV_DIR = RESULTS / "divergence"
SNAP_DIR = RESULTS / "snapshots"
FAIL_DIR = RESULTS / "failures"
FIG_DIR = ROOT / "figures" / "mhd_runner"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHODS_MAIN = [
    "none",
    "hyperbolic_glm",
    "mixed_glm",
    "parabolic",
    "elliptic_projection",
    "powell_source",
    "powell_source_subcycled",
    "powell_source_limited",
]

METHODS_READY = [
    "none",
    "hyperbolic_glm",
    "mixed_glm",
]

METHOD_LABELS = {
    "none": "None",
    "hyperbolic_glm": "Hyperbolic GLM",
    "mixed_glm": "Mixed GLM",
    "parabolic": "Parabolic",
    "elliptic_projection": "Projection",
    "powell_source": "Powell",
    "powell_source_subcycled": "Powell subcycled",
    "powell_source_limited": "Powell limited",
}

COLORS = {
    "none": "tab:gray",
    "hyperbolic_glm": "tab:blue",
    "mixed_glm": "tab:orange",
    "parabolic": "tab:green",
    "elliptic_projection": "tab:red",
    "powell_source": "tab:purple",
    "powell_source_subcycled": "tab:brown",
    "powell_source_limited": "tab:cyan",
}

LINESTYLES = {
    "none": "-",
    "hyperbolic_glm": "-",
    "mixed_glm": "-",
    "parabolic": "--",
    "elliptic_projection": "--",
    "powell_source": ":",
    "powell_source_subcycled": "-.",
    "powell_source_limited": "--",
}


def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of columns {candidates} found. Available = {list(df.columns)}")
    return None


def read_csv(path):
    if not path.exists():
        print(f"missing: {path}")
        return None
    return pd.read_csv(path)


def plot_ot_min_pressure():
    fig, ax = plt.subplots(figsize=(8, 5))

    plotted = 0
    for method in METHODS_MAIN:
        path = DIV_DIR / f"mhd_ot_{method}.csv"
        df = read_csv(path)
        if df is None or len(df) == 0:
            continue

        tcol = pick_col(df, ["time", "t"])
        pcol = pick_col(df, [
            "min_raw_pressure",
            "min_pressure",
            "min_p",
            "min_pressure_after_full_step",
        ], required=False)

        if pcol is None:
            print(f"skip {method}: no pressure column")
            continue

        t = df[tcol].to_numpy()
        p = df[pcol].to_numpy()

        finite = np.isfinite(t) & np.isfinite(p)
        if not finite.any():
            continue

        ax.plot(
            t[finite],
            p[finite],
            lw=1.8,
            color=COLORS.get(method, None),
            ls=LINESTYLES.get(method, "-"),
            label=METHOD_LABELS.get(method, method),
        )
        plotted += 1

    ax.axhline(0.0, color="black", lw=1.2, alpha=0.8)
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel("time")
    ax.set_ylabel(r"minimum raw pressure $\min(p)$")
    ax.set_title(
        "Orszag–Tang: pressure positivity diagnostic\n"
        r"$p<0$ marks a hydro positivity failure"
    )
    ax.grid(True, which="both", alpha=0.3)
    if plotted:
        ax.legend(fontsize=8.5, ncol=2)
    plt.tight_layout()

    out = FIG_DIR / "mhd_runner_ot_min_pressure.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def find_failure_file(method):
    candidates = sorted(FAIL_DIR.glob(f"mhd_ot_{method}_*_failure.csv"))
    if not candidates:
        return None
    # Prefer preserve_thermal_pressure if present.
    for p in candidates:
        if "preserve_thermal_pressure" in p.name:
            return p
    return candidates[0]


def final_value(df, candidates):
    c = pick_col(df, candidates, required=False)
    if c is None or len(df) == 0:
        return np.nan
    vals = df[c].to_numpy()
    vals = vals[np.isfinite(vals)]
    return float(vals[-1]) if vals.size else np.nan


def plot_ot_pressure_failure_stages():
    methods = [
        "parabolic",
        "elliptic_projection",
        "powell_source",
        "powell_source_subcycled",
        "powell_source_limited",
    ]
    stage_cols = [
        ("before_hydro", ["min_pressure_before_hydro"]),
        ("after_hydro", ["min_pressure_after_hydro"]),
        ("after_cleaning_B", ["min_pressure_after_cleaning_B_update", "min_pressure_after_cleaning_B"]),
        ("after_energy_repair", ["min_pressure_after_energy_repair"]),
        ("after_full_step", ["min_pressure_after_full_step"]),
    ]

    rows = []
    for method in methods:
        fpath = find_failure_file(method)
        if fpath is None:
            print(f"no failure file for {method}")
            continue
        df = pd.read_csv(fpath)
        row = {"method": method, "file": fpath.name}
        for stage, candidates in stage_cols:
            row[stage] = final_value(df, candidates)
        rows.append(row)

    if not rows:
        print("no failure-stage data found")
        return

    table = pd.DataFrame(rows)
    x = np.arange(len(table))
    width = 0.16

    fig, ax = plt.subplots(figsize=(9, 5))
    for k, (stage, _) in enumerate(stage_cols):
        vals = table[stage].to_numpy(dtype=float)
        ax.bar(x + (k - 2) * width, vals, width=width, label=stage)

    ax.axhline(0.0, color="black", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in table["method"]])
    ax.set_ylabel(r"minimum raw pressure at failure step")
    ax.set_title(
        "Orszag–Tang: where does pressure first become negative?\n"
        "Failure occurs after the hydro step, not after the projection/parabolic repair"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    out = FIG_DIR / "mhd_runner_ot_pressure_failure_stages.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def to_2d(df, col):
    i = df["i"].to_numpy(dtype=int)
    j = df["j"].to_numpy(dtype=int)
    nx = int(i.max()) + 1
    ny = int(j.max()) + 1

    if "x" in df.columns:
        xvals = df.sort_values("i").groupby("i")["x"].first().to_numpy()
    else:
        xvals = np.arange(nx)

    if "y" in df.columns:
        yvals = df.sort_values("j").groupby("j")["y"].first().to_numpy()
    else:
        yvals = np.arange(ny)

    Z = np.full((ny, nx), np.nan)
    Z[j, i] = df[col].to_numpy()
    return xvals, yvals, Z


def plot_ot_pressure_maps():
    methods = METHODS_READY
    dfs = {}

    for method in methods:
        path = SNAP_DIR / f"mhd_ot_{method}_final.csv"
        df = read_csv(path)
        if df is None:
            return
        pcol = pick_col(df, ["p", "pressure", "raw_pressure"])
        dfs[method] = (df, pcol)

    arrays = []
    for method, (df, pcol) in dfs.items():
        _, _, Z = to_2d(df, pcol)
        arrays.append(Z)

    all_vals = np.concatenate([Z[np.isfinite(Z)].ravel() for Z in arrays])
    vmin = np.nanpercentile(all_vals, 1)
    vmax = np.nanpercentile(all_vals, 99)
    if vmin == vmax:
        vmax = vmin + 1.0

    fig, axes = plt.subplots(1, len(methods), figsize=(12, 4), constrained_layout=True)

    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        df, pcol = dfs[method]
        x, y, Z = to_2d(df, pcol)

        im = ax.pcolormesh(
            x, y, Z,
            shading="auto",
            cmap="plasma",
            norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
        )
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        finite = Z[np.isfinite(Z)]
        if finite.size:
            ax.text(
                0.03, 0.97,
                rf"$\min p={finite.min():.2e}$",
                transform=ax.transAxes,
                va="top",
                color="white",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55),
            )

    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, label=r"pressure $p$")
    fig.suptitle(
        r"Orszag–Tang: final pressure field, $t=0.5$",
        fontsize=12,
    )

    out = FIG_DIR / "mhd_runner_ot_pressure_maps.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def main():
    plot_ot_min_pressure()
    plot_ot_pressure_failure_stages()
    plot_ot_pressure_maps()


if __name__ == "__main__":
    main()
