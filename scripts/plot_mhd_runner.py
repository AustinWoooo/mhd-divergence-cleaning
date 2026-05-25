#!/usr/bin/env python3
"""
plot_mhd_runner.py
==================
Visualise results produced by the integrated HLLD + GLM runner
(tests/test_mhd_runner.cpp → build/test_mhd_runner).

Four figures are saved:

  figures/mhd_runner/mhd_runner_ot_divB.png
      Orszag-Tang: L2(divB_fv) vs time for all cleaning methods.

  figures/mhd_runner/mhd_runner_ot_snapshot.png
      Orszag-Tang: 2D final-state maps (density, pressure, |divB|)
      for NONE vs the best finite-speed cleaning method.

  figures/mhd_runner/mhd_runner_bw_divB.png
      Brio-Wu strip: L2(divB_fv) vs time for all cleaning methods.
      (Initial divB = 0 in this problem; the plot shows how much
       each method accumulates during the run.)

  figures/mhd_runner/mhd_runner_bw_profiles.png
      Brio-Wu strip: 1D final-state profiles along the middle row
      (rho, p, u, By) for NONE vs MIXED_GLM.

  figures/mhd_runner/mhd_runner_fl_divB.png
  figures/mhd_runner/mhd_runner_fl_snapshot.png
      Field-loop advection diagnostics and final maps.

  figures/mhd_runner/mhd_runner_da_divB.png
  figures/mhd_runner/mhd_runner_da_snapshot.png
      Divergence-advection diagnostics and final maps.

Usage
-----
  # 1. Build and run the C++ runner first:
  #    cmake -B build && cmake --build build --target test_mhd_runner
  #    ./build/test_mhd_runner
  # 2. Then:
  python scripts/plot_mhd_runner.py

Both files are expected at the project root when you run the script.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
#  Problem configuration (must match test_mhd_runner.cpp)
# ---------------------------------------------------------------------------

OT_PREFIX = "mhd_ot"
BW_PREFIX = "mhd_bw"
FL_PREFIX = "mhd_fl"
DA_PREFIX = "mhd_da"

CLEANING_TYPES = [
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
    "none":                "None (HLLD only)",
    "hyperbolic_glm":      "Hyperbolic GLM",
    "mixed_glm":           "Mixed GLM",
    "parabolic":           "Parabolic",
    "elliptic_projection": "Elliptic Projection",
    "powell_source":       "Powell Source",
    "mixed_eglm":          "Mixed EGLM",
    "gi_mixed_eglm":       "GI Mixed EGLM",
}

# 8-colour palette drawn from tab10 (colourblind-friendly order).
COLORS = {
    "none":                "#1f77b4",
    "hyperbolic_glm":      "#ff7f0e",
    "mixed_glm":           "#2ca02c",
    "parabolic":           "#d62728",
    "elliptic_projection": "#9467bd",
    "powell_source":       "#8c564b",
    "mixed_eglm":          "#e377c2",
    "gi_mixed_eglm":       "#bcbd22",
}

LINESTYLES = {
    "none":                "-",
    "hyperbolic_glm":      "--",
    "mixed_glm":           "-.",
    "parabolic":           ":",
    "elliptic_projection": "-",
    "powell_source":       "--",
    "mixed_eglm":          "-.",
    "gi_mixed_eglm":       ":",
}

DIVERG_DIR   = Path("results/mhd_runner/divergence")
SNAPSHOT_DIR = Path("results/mhd_runner/snapshots")
FIGURES_DIR  = Path("figures/mhd_runner")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def load_diag(prefix: str, cleaning: str) -> pd.DataFrame | None:
    path = DIVERG_DIR / f"{prefix}_{cleaning}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_snapshot(prefix: str, cleaning: str) -> pd.DataFrame | None:
    path = SNAPSHOT_DIR / f"{prefix}_{cleaning}_final.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def to_2d(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reshape a column from the snapshot CSV to a 2D array.

    The C++ writer loops outer-j, inner-i, so the flat data is in
    row-major order with j as the slow index.

    Returns
    -------
    x1d : (nx,) x-coordinates
    y1d : (ny,) y-coordinates
    Z   : (ny, nx) field values, Z[j_idx, i_idx]
    """
    nx = df["i"].max() + 1
    ny = df["j"].max() + 1
    x1d = df[df["j"] == 0].sort_values("i")["x"].values      # (nx,)
    y1d = df[df["i"] == 0].sort_values("j")["y"].values      # (ny,)
    Z   = df.sort_values(["j", "i"])[col].values.reshape(ny, nx)
    return x1d, y1d, Z


def plot_problem_divB(prefix: str, title: str, output_name: str):
    fig, ax = plt.subplots(figsize=(8, 5))

    plotted = 0
    for ct in CLEANING_TYPES:
        df = load_diag(prefix, ct)
        if df is None:
            continue
        ycol = "L2_norm_fv" if "L2_norm_fv" in df.columns else "L2_fv"
        ax.semilogy(
            df["time"],
            df[ycol],
            color=COLORS[ct],
            ls=LINESTYLES[ct],
            lw=1.8,
            label=LABELS[ct],
        )
        plotted += 1

    if plotted == 0:
        print(f"  [{prefix}_divB] no data found — run test_mhd_runner first.")
        plt.close()
        return

    ax.set_xlabel("time", fontsize=12)
    ax.set_ylabel(r"$L_2$ normalized FV $\nabla\!\cdot\!\mathbf{B}$", fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    out = FIGURES_DIR / output_name
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


def plot_problem_snapshot(prefix: str, title: str, output_name: str):
    pairs = [("none", "None"), ("mixed_glm", "Mixed GLM")]

    dfs = {}
    for ct, _ in pairs:
        df = load_snapshot(prefix, ct)
        if df is None:
            print(f"  [{prefix}_snapshot] missing {prefix}_{ct}_final.csv — skipping.")
            return
        dfs[ct] = df

    plot_rows = [
        ("rho",     r"Density $\rho$",        "viridis", False),
        ("p",       r"Pressure $P$",          "plasma",  False),
        ("Bmag",    r"$|\mathbf{B}|$",        "magma",   False),
        ("divB_fv", r"$|\nabla\!\cdot\!B|$",  "hot_r",   True),
    ]

    fig, axes = plt.subplots(len(plot_rows), 2, figsize=(10, 13))
    fig.suptitle(title, fontsize=12, y=0.998)

    for row, (col, ylabel, cmap, use_log) in enumerate(plot_rows):
        Zs = {}
        for ct, _ in pairs:
            x1d, y1d, Z = to_2d(dfs[ct], col)
            if col == "divB_fv":
                Z = np.abs(Z)
            Zs[ct] = Z

        all_vals = np.concatenate([Z.ravel() for Z in Zs.values()])
        finite_vals = all_vals[np.isfinite(all_vals)]
        if finite_vals.size == 0:
            finite_vals = np.array([0.0])
        vmin = finite_vals.min()
        vmax = finite_vals.max()

        if use_log:
            positive = finite_vals[finite_vals > 0]
            eps = max(float(positive.min()) if positive.size else 1e-12, 1e-14)
            norm = mcolors.LogNorm(vmin=eps, vmax=max(vmax, eps * 10))
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        ims = []
        for col_idx, (ct, ctlabel) in enumerate(pairs):
            ax = axes[row, col_idx]
            im = ax.pcolormesh(x1d, y1d, Zs[ct], cmap=cmap, norm=norm, shading="auto")
            ax.set_aspect("equal")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ims.append(im)

            if col_idx == 0:
                ax.set_ylabel(ylabel + r" / $y$", fontsize=10)
            if row == len(plot_rows) - 1:
                ax.set_xlabel(r"$x$", fontsize=10)
            if row == 0:
                ax.set_title(ctlabel, fontsize=12, fontweight="bold")

        fig.colorbar(ims[-1], ax=axes[row, :], shrink=0.72, pad=0.02, aspect=28)

    plt.tight_layout()
    out = FIGURES_DIR / output_name
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
#  Figure 1  — Orszag-Tang: divB evolution
# ---------------------------------------------------------------------------

def plot_ot_divB():
    fig, ax = plt.subplots(figsize=(8, 5))

    plotted = 0
    for ct in CLEANING_TYPES:
        df = load_diag(OT_PREFIX, ct)
        if df is None:
            continue
        ax.semilogy(
            df["time"],
            df["L2_fv"],
            color=COLORS[ct],
            ls=LINESTYLES[ct],
            lw=1.8,
            label=LABELS[ct],
        )
        plotted += 1

    if plotted == 0:
        print("  [ot_divB] no data found — run test_mhd_runner first.")
        plt.close()
        return

    ax.set_xlabel("time", fontsize=12)
    ax.set_ylabel(r"$L_2\!\left(\nabla\!\cdot\!\mathbf{B}\right)$ (FV)", fontsize=12)
    ax.set_title(
        r"Orszag-Tang Vortex  —  HLLD + GLM cleaning comparison"
        "\n"
        r"$N=128\times128$,  $\gamma=5/3$,  $t_{\rm end}=0.5$",
        fontsize=11,
    )
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    out = FIGURES_DIR / "mhd_runner_ot_divB.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
#  Figure 2  — Orszag-Tang: 2D final-state snapshot
# ---------------------------------------------------------------------------

def plot_ot_snapshot():
    # Show NONE vs MIXED_GLM (a representative finite-speed method)
    pairs = [("none", "None"), ("mixed_glm", "Mixed GLM")]

    dfs = {}
    for ct, _ in pairs:
        df = load_snapshot(OT_PREFIX, ct)
        if df is None:
            print(f"  [ot_snapshot] missing {OT_PREFIX}_{ct}_final.csv — skipping.")
            return
        dfs[ct] = df

    plot_rows = [
        ("rho",     r"Density  $\rho$",         "viridis",  False),
        ("p",       r"Pressure $P$",             "plasma",   False),
        ("divB_fv", r"$|\nabla\!\cdot\!B|$ (FV)","hot_r",   True),
    ]

    fig, axes = plt.subplots(len(plot_rows), 2, figsize=(10, 12))
    fig.suptitle(
        r"Orszag-Tang Vortex  —  final state at $t=0.5$"
        "\n"
        r"HLLD solver,  $N=128\times128$,  $\gamma=5/3$",
        fontsize=12, y=0.998,
    )

    for row, (col, ylabel, cmap, use_log) in enumerate(plot_rows):
        Zs = {}
        for ct, _ in pairs:
            x1d, y1d, Z = to_2d(dfs[ct], col)
            if col == "divB_fv":
                Z = np.abs(Z)
            Zs[ct] = Z

        all_vals = np.concatenate([Z.ravel() for Z in Zs.values()])
        vmin = all_vals.min()
        vmax = all_vals.max()

        if use_log:
            eps  = max(float(all_vals[all_vals > 0].min()) if (all_vals > 0).any() else 1e-12, 1e-14)
            norm = mcolors.LogNorm(vmin=eps, vmax=max(vmax, eps * 10))
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        ims = []
        for col_idx, (ct, ctlabel) in enumerate(pairs):
            ax = axes[row, col_idx]
            im = ax.pcolormesh(x1d, y1d, Zs[ct], cmap=cmap, norm=norm, shading="auto")
            ax.set_aspect("equal")
            ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
            ims.append(im)

            if col_idx == 0:
                ax.set_ylabel(ylabel + r"  /  $y$", fontsize=10)
            if row == len(plot_rows) - 1:
                ax.set_xlabel(r"$x$", fontsize=10)
            if row == 0:
                ax.set_title(ctlabel, fontsize=12, fontweight="bold")

            if col == "divB_fv":
                L2 = float(np.sqrt(np.mean(Zs[ct] ** 2)))
                ax.text(
                    0.03, 0.97,
                    rf"$L_2={L2:.2e}$",
                    transform=ax.transAxes,
                    va="top", fontsize=8.5, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55),
                )

        fig.colorbar(ims[-1], ax=axes[row, :], shrink=0.72, pad=0.02, aspect=28)

    plt.tight_layout()
    out = FIGURES_DIR / "mhd_runner_ot_snapshot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
#  Figure 3  — Brio-Wu: divB evolution
# ---------------------------------------------------------------------------

def plot_bw_divB():
    fig, ax = plt.subplots(figsize=(8, 5))

    plotted = 0
    for ct in CLEANING_TYPES:
        df = load_diag(BW_PREFIX, ct)
        if df is None:
            continue
        # Brio-Wu strip is 1D-in-2D: initial divB ≈ 0 and stays tiny.
        # Use symlog so we can see structure even when values hit 0.
        ax.plot(
            df["time"],
            df["L2_fv"],
            color=COLORS[ct],
            ls=LINESTYLES[ct],
            lw=1.8,
            label=LABELS[ct],
        )
        plotted += 1

    if plotted == 0:
        print("  [bw_divB] no data found — run test_mhd_runner first.")
        plt.close()
        return

    ax.set_yscale("symlog", linthresh=1e-14)
    ax.set_xlabel("time", fontsize=12)
    ax.set_ylabel(r"$L_2\!\left(\nabla\!\cdot\!\mathbf{B}\right)$ (FV)", fontsize=12)
    ax.set_title(
        r"Brio-Wu Shock Tube (2D strip)  —  HLLD + GLM cleaning comparison"
        "\n"
        r"$N=128\times128$,  $\gamma=2$,  $t_{\rm end}=0.2$",
        fontsize=11,
    )
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    out = FIGURES_DIR / "mhd_runner_bw_divB.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
#  Figure 4  — Brio-Wu: 1D shock profiles
# ---------------------------------------------------------------------------

def plot_bw_profiles():
    # Compare NONE vs MIXED_GLM along the middle row of the 2D strip.
    comparisons = [
        ("none",      "None (HLLD only)", "tab:blue", "-"),
        ("mixed_glm", "Mixed GLM",        "tab:red",  "--"),
    ]

    dfs = {}
    for ct, _, _, _ in comparisons:
        df = load_snapshot(BW_PREFIX, ct)
        if df is None:
            print(f"  [bw_profiles] missing {BW_PREFIX}_{ct}_final.csv — skipping.")
            return
        dfs[ct] = df

    # Extract middle y-row.
    def middle_row(df):
        ny = df["j"].max() + 1
        j_mid = ny // 2
        return df[df["j"] == j_mid].sort_values("i")

    rows = {ct: middle_row(dfs[ct]) for ct, *_ in comparisons}

    plot_vars = [
        ("rho", r"Density $\rho$"),
        ("p",   r"Pressure $P$"),
        ("u",   r"Velocity $u_x$"),
        ("By",  r"Transverse field $B_y$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    fig.suptitle(
        r"Brio-Wu Shock Tube (2D strip, middle row)  —  final state at $t=0.2$"
        "\n"
        r"HLLD solver,  $N=128\times128$,  $\gamma=2$",
        fontsize=12,
    )

    ax_flat = axes.ravel()
    for k, (col, ylabel) in enumerate(plot_vars):
        ax = ax_flat[k]
        for ct, label, color, ls in comparisons:
            r = rows[ct]
            ax.plot(r["x"].values, r[col].values,
                    color=color, ls=ls, lw=1.6, label=label)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for ax in axes[1]:
        ax.set_xlabel(r"$x$", fontsize=11)

    plt.tight_layout()
    out = FIGURES_DIR / "mhd_runner_bw_profiles.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    print("=== Orszag-Tang vortex ===")
    plot_ot_divB()
    plot_ot_snapshot()

    print("\n=== Brio-Wu shock tube ===")
    plot_bw_divB()
    plot_bw_profiles()

    print("\n=== Field-loop advection ===")
    plot_problem_divB(
        FL_PREFIX,
        r"Field-Loop Advection  —  HLLD + cleaning comparison"
        "\n"
        r"$N=128\times128$,  $\gamma=5/3$,  $t_{\rm end}=0.5$",
        "mhd_runner_fl_divB.png",
    )
    plot_problem_snapshot(
        FL_PREFIX,
        r"Field-Loop Advection  —  final state at $t=0.5$",
        "mhd_runner_fl_snapshot.png",
    )

    print("\n=== Divergence advection ===")
    plot_problem_divB(
        DA_PREFIX,
        r"Divergence Advection  —  HLLD + cleaning comparison"
        "\n"
        r"$N=128\times128$,  $\gamma=5/3$,  $t_{\rm end}=0.5$",
        "mhd_runner_da_divB.png",
    )
    plot_problem_snapshot(
        DA_PREFIX,
        r"Divergence Advection  —  final state at $t=0.5$",
        "mhd_runner_da_snapshot.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
