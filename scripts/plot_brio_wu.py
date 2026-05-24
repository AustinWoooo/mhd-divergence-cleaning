#!/usr/bin/env python3
"""
plot_brio_wu.py
===============
Read the three C++ HLLD simulation outputs and plot a comparison:
  - Reference   : clean IC, no div-B error
  - No GLM      : sinusoidal Bx perturbation, no cleaning
  - With GLM    : same perturbation, hyperbolic-GLM cleaning active

Usage
-----
  # 1. Build and run the C++ simulation first:
  #    g++ -std=c++17 -O2 -Iinclude src/hlld.cpp src/main.cpp -o brio_wu
  #    ./brio_wu
  # 2. Then plot:
  python scripts/plot_brio_wu.py

Output: figures/brio_wu_comparison.png
"""

import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


CSV_FILES = {
    "reference": "results/brio_wu/brio_wu_reference.csv",
    "noGLM":     "results/brio_wu/brio_wu_noGLM.csv",
    "GLM":       "results/brio_wu/brio_wu_GLM.csv",
}


def load_all():
    data = {}
    for key, path in CSV_FILES.items():
        p = Path(path)
        if not p.exists():
            print(f"ERROR: {path} not found.")
            print("       Run the C++ simulation first:  ./brio_wu")
            sys.exit(1)
        data[key] = pd.read_csv(p)
        print(f"  Loaded {len(data[key])} cells  ← {path}")
    return data


def main():
    figdir = Path("figures/brio_wu")
    figdir.mkdir(parents=True, exist_ok=True)

    print("Loading CSV files …")
    data = load_all()

    x   = data["reference"]["x"].values
    ref = data["reference"]
    nog = data["noGLM"]
    glm = data["GLM"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    fig.suptitle(
        r"Brio-Wu Shock Tube  ($\gamma = 2$,  $N = 400$,  $t = 0.1$)"
        "\nHLLD Solver — GLM Divergence Cleaning Comparison",
        fontsize=12)

    plot_vars = [
        ("rho", r"Density $\rho$",        axes[0, 0]),
        ("p",   r"Gas Pressure $P$",       axes[0, 1]),
        ("u",   r"Velocity $u_x$",         axes[1, 0]),
        ("By",  r"Transverse Field $B_y$", axes[1, 1]),
    ]

    style = [
        ("Reference (clean IC)",            "k",         1.8, "-"),
        (r"No GLM  (+$\delta B_x$ error)",  "tab:red",   1.2, "--"),
        (r"With GLM (+$\delta B_x$ error)", "tab:blue",  1.2, "-."),
    ]

    for col, ylabel, ax in plot_vars:
        for df, (label, color, lw, ls) in zip([ref, nog, glm], style):
            ax.plot(x, df[col].values, color=color, lw=lw, ls=ls, label=label)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=8.5, loc="best")
        ax.grid(True, alpha=0.3)

    for ax in axes[1]:
        ax.set_xlabel(r"$x$", fontsize=11)

    plt.tight_layout()
    out = figdir / "brio_wu_comparison.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.close()


if __name__ == "__main__":
    main()
