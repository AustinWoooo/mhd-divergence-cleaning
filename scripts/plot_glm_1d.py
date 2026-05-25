import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base = Path("results/glm_1d")
figdir = Path("figures/glm_1d")
figdir.mkdir(parents=True, exist_ok=True)

cases = [
    "none",
    "hyperbolic_glm",
    "mixed_glm",
    "parabolic",
]

metrics = ["L1", "L2", "Linf"]

for metric in metrics:
    plt.figure()

    for case in cases:
        path = base / f"glm_1d_{case}.csv"
        df = pd.read_csv(path)
        plt.plot(df["time"], df[metric], label=case)

    plt.xlabel("time")
    plt.ylabel(rf"${metric}(\nabla\cdot B)$")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()

    out = figdir / f"glm_1d_{metric}_comparison.png"
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")