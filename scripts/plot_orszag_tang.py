#!/usr/bin/env python3
"""
plot_orszag_tang.py
===================
Load 2D Orszag-Tang Vortex results and produce side-by-side comparison.

Usage:
    # 1. Build and run C++ simulation first:
    #    cmake -B build && cmake --build build -t orszag_tang
    #    ./build/orszag_tang
    # 2. Then plot:
    python scripts/plot_orszag_tang.py

Output: figures/orszag_tang_comparison.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


CSV = {
    "noGLM":   "results/orszag_tang/noGLM.csv",
    "withGLM": "results/orszag_tang/withGLM.csv",
}


def load(path):
    p = Path(path)
    if not p.exists():
        print(f"ERROR: {path} not found. Run ./build/orszag_tang first.")
        sys.exit(1)
    df = pd.read_csv(p)
    N = int(round(np.sqrt(len(df))))
    assert N * N == len(df), f"Expected square grid, got {len(df)} rows"
    return df, N


def to2d(df, col, N):
    """Reshape flat column to (Ny, Nx) for pcolormesh (row=y, col=x)."""
    return df[col].values.reshape(N, N).T   # [i,j] -> transpose -> [j,i]


def main():
    figdir = Path("figures/orszag_tang")
    figdir.mkdir(parents=True, exist_ok=True)

    print("Loading CSV files...")
    df_no, N = load(CSV["noGLM"])
    df_gl, _ = load(CSV["withGLM"])
    print(f"  Grid: {N}x{N}")

    # 1D coordinate vectors
    x = df_no["x"].values.reshape(N, N)[:, 0]   # x_i  (N,)
    y = df_no["y"].values.reshape(N, N)[0, :]    # y_j  (N,)

    def fields(df):
        rho  = to2d(df, "rho",  N)
        Bx   = to2d(df, "Bx",   N)
        By   = to2d(df, "By",   N)
        divB = to2d(df, "divB", N)
        return rho, Bx**2 + By**2, np.abs(divB)

    rho_no, Bmag_no, adivB_no = fields(df_no)
    rho_gl, Bmag_gl, adivB_gl = fields(df_gl)

    # Statistics
    l2_no = np.sqrt(np.mean(adivB_no**2))
    l2_gl = np.sqrt(np.mean(adivB_gl**2))
    linf_no = adivB_no.max()
    linf_gl = adivB_gl.max()
    print(f"  L2  |divB|  no GLM : {l2_no:.4e}")
    print(f"  L2  |divB| with GLM: {l2_gl:.4e}  (x{l2_no/l2_gl:.1f} reduction)")
    print(f"  Linf|divB|  no GLM : {linf_no:.4e}")
    print(f"  Linf|divB| with GLM: {linf_gl:.4e}  (x{linf_no/linf_gl:.1f} reduction)")

    # ---- Figure: 3 rows x 2 cols ----
    fig, axes = plt.subplots(3, 2, figsize=(10, 13))
    fig.suptitle(
        r"Orszag-Tang Vortex  ($N=128\times128$,  $\gamma=5/3$,  $t=0.5$)"
        "\nHLLD Solver — GLM Divergence Cleaning Comparison",
        fontsize=13, y=0.995)

    plot_rows = [
        (rho_no,   rho_gl,   r"Density  $\rho$",             "viridis", False),
        (Bmag_no,  Bmag_gl,  r"Magnetic pressure  $|B|^2$",  "inferno", False),
        (adivB_no, adivB_gl, r"$|\nabla\!\cdot\!B|$",        "hot_r",   True),
    ]
    col_labels = ["No GLM", "With GLM"]

    for row, (Z_no, Z_gl, ylabel, cmap, use_log) in enumerate(plot_rows):
        Zs = [Z_no, Z_gl]
        vmin = min(Z_no.min(), Z_gl.min())
        vmax = max(Z_no.max(), Z_gl.max())
        if use_log:
            eps = max(vmin, 1e-12)
            norm = mcolors.LogNorm(vmin=eps, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        ims = []
        for col, Z in enumerate(Zs):
            ax = axes[row, col]
            im = ax.pcolormesh(x, y, Z, cmap=cmap, norm=norm, shading="auto")
            ax.set_aspect("equal")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ims.append(im)

            if col == 0:
                ax.set_ylabel(ylabel + "\n" + r"$y$", fontsize=10)
            if row == 2:
                ax.set_xlabel(r"$x$", fontsize=10)

        # Shared colorbar spanning both columns in this row
        fig.colorbar(ims[-1], ax=axes[row, :], shrink=0.75, pad=0.02, aspect=30)

        # Column titles (only on first row)
        if row == 0:
            axes[row, 0].set_title("No GLM", fontsize=12, fontweight="bold")
            axes[row, 1].set_title("With GLM", fontsize=12, fontweight="bold")

    # Annotate L2 norm on the |divB| row
    axes[2, 0].text(0.03, 0.97,
                    r"$L_2=$" + f" {l2_no:.2e}",
                    transform=axes[2, 0].transAxes,
                    va="top", fontsize=9, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
    axes[2, 1].text(0.03, 0.97,
                    r"$L_2=$" + f" {l2_gl:.2e}",
                    transform=axes[2, 1].transAxes,
                    va="top", fontsize=9, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))

    plt.tight_layout()
    out = figdir / "orszag_tang_comparison.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
