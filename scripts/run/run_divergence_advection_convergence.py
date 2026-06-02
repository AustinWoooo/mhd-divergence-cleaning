#!/usr/bin/env python3
"""
Run divergence-advection convergence and PCM-vs-PLM evidence cases.

Outputs:
  results/mhd_runner/convergence/divergence_advection_convergence.csv
  figures/mhd_runner/convergence/divergence_advection_convergence.png
  figures/mhd_runner/divB/pcm_vs_plm_divB.png
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"
EXE = BUILD_DIR / "test_mhd_runner"
RESULTS_DIR = ROOT / "results" / "mhd_runner"
DIV_DIR = RESULTS_DIR / "divergence"
SNAP_DIR = RESULTS_DIR / "snapshots"
CONV_DIR = RESULTS_DIR / "convergence"
HRSC_DIR = RESULTS_DIR / "hrsc"
FIG_CONV_DIR = ROOT / "figures" / "mhd_runner" / "convergence"
FIG_DIVB_DIR = ROOT / "figures" / "mhd_runner" / "divB"

TFINAL = 0.5
CLEANING_METHOD = "mixed_glm"
PROBLEM = "divergence_advection"
PREFIX = "mhd_da_conv"


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def ensure_build(skip_build: bool) -> None:
    if skip_build and EXE.exists():
        return
    if not (BUILD_DIR / "CMakeCache.txt").exists():
        run(["cmake", "-S", ".", "-B", "build"])
    run(["cmake", "--build", "build", "--target", "test_mhd_runner", "--parallel"])


def periodic_delta(x: np.ndarray, center: float) -> np.ndarray:
    return (x - center + 0.5) % 1.0 - 0.5

def final_divergence_error(prefix: str) -> tuple[float, float]:
    diag_path = DIV_DIR / f"{prefix}_{CLEANING_METHOD}.csv"
    if not diag_path.exists():
        raise FileNotFoundError(diag_path)

    df = pd.read_csv(diag_path)

    if "L2_norm_fv" in df.columns:
        l2_norm = float(df["L2_norm_fv"].iloc[-1])
    else:
        l2_norm = float("nan")

    if "L2_fv" in df.columns:
        l2_raw = float(df["L2_fv"].iloc[-1])
    else:
        l2_raw = float("nan")

    return l2_raw, l2_norm

def estimate_slope(rows: list[dict[str, float]]) -> float:
    xs = np.array([row["dx"] for row in rows], dtype=float)
    ys = np.array([row["final_L2_norm_fv"] for row in rows], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0.0) & (ys > 0.0)
    if int(mask.sum()) < 2:
        return float("nan")
    coeff = np.polyfit(np.log(xs[mask]), np.log(ys[mask]), 1)
    return float(coeff[0])


def run_divergence_advection(reconstruction: str, n: int) -> str:
    prefix = f"{PREFIX}_{reconstruction}_n{n}"
    run(
        [
            str(EXE),
            "--nx",
            str(n),
            "--ny",
            str(n),
            "--tfinal",
            str(TFINAL),
            "--output-prefix",
            prefix,
            "--reconstruction",
            reconstruction,
            PROBLEM,
            CLEANING_METHOD,
        ]
    )
    return prefix


def write_convergence_csv(rows: list[dict[str, float]]) -> Path:
    out = CONV_DIR / "divergence_advection_convergence.csv"
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "reconstruction",
        "N",
        "dx",
        "tfinal",
        "final_L2_fv",
        "final_L2_norm_fv",
        "estimated_slope",
        "diagnostic_file",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")
    return out


def plot_convergence(df: pd.DataFrame) -> Path:
    FIG_CONV_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    for reconstruction, group in df.groupby("reconstruction"):
        group = group.sort_values("dx")
        slope = float(group["estimated_slope"].iloc[0])
        ax.loglog(
            group["dx"],
            group["final_L2_norm_fv"],
            marker="o",
            lw=1.8,
            label=f"{reconstruction.upper()} slope={slope:.2f}",
        )

    ax.invert_xaxis()
    ax.set_xlabel("dx")
    ax.set_ylabel(r"final normalized FV $L_2(\nabla\cdot B)$")
    ax.set_title("Divergence-advection convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    out = FIG_CONV_DIR / "divergence_advection_convergence.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def final_value(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or len(df) == 0:
        return float("nan")
    return float(df[column].iloc[-1])


def write_hrsc_summary() -> Path:
    HRSC_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for reconstruction in ("pcm", "plm"):
        prefix = f"mhd_fl_conv_{reconstruction}_n128"
        diag_path = DIV_DIR / f"{prefix}_{CLEANING_METHOD}.csv"
        if not diag_path.exists():
            raise FileNotFoundError(diag_path)
        df = pd.read_csv(diag_path)
        energy0 = float(df["total_energy"].iloc[0])
        energy1 = final_value(df, "total_energy")
        rows.append(
            {
                "problem": "field_loop",
                "method": CLEANING_METHOD,
                "reconstruction": reconstruction,
                "N": 128,
                "final_time": final_value(df, "time"),
                "final_L2_norm_fv": final_value(df, "L2_norm_fv"),
                "min_pressure": float(df["min_pressure"].min()),
                "energy_drift": (energy1 - energy0) / max(abs(energy0), 1.0e-30),
                "diagnostic_file": str(diag_path.relative_to(ROOT)),
            }
        )

    out = HRSC_DIR / "pcm_vs_plm_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")
    return out


def plot_hrsc_comparison() -> Path:
    FIG_DIVB_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    styles = {"pcm": ("#d62728", "--"), "plm": ("#2ca02c", "-")}
    for reconstruction in ("pcm", "plm"):
        prefix = f"mhd_fl_conv_{reconstruction}_n128"
        df = pd.read_csv(DIV_DIR / f"{prefix}_{CLEANING_METHOD}.csv")
        color, ls = styles[reconstruction]
        ax.semilogy(
            df["time"],
            df["L2_norm_fv"],
            color=color,
            ls=ls,
            lw=2.0,
            label=reconstruction.upper(),
        )
    ax.set_xlabel("time")
    ax.set_ylabel(r"$L_2$ normalized FV $\nabla\cdot B$")
    ax.set_title("Field-loop PCM vs PLM reconstruction")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out = FIG_DIVB_DIR / "pcm_vs_plm_divB.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[32, 64, 128, 256],
        help="Divergence-advection resolutions to run.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Use an existing build/test_mhd_runner if present.",
    )
    args = parser.parse_args()

    ensure_build(args.skip_build)

    rows: list[dict[str, float]] = []

    for reconstruction in ("pcm", "plm"):
        rows_for_reconstruction: list[dict[str, float]] = []

        for n in args.resolutions:
            prefix = run_divergence_advection(reconstruction, n)
            l2_raw, l2_norm = final_divergence_error(prefix)

            rows_for_reconstruction.append(
                {
                    "reconstruction": reconstruction,
                    "N": n,
                    "dx": 1.0 / n,
                    "tfinal": TFINAL,
                    "final_L2_fv": l2_raw,
                    "final_L2_norm_fv": l2_norm,
                    "estimated_slope": float("nan"),
                    "diagnostic_file": str(
                        (DIV_DIR / f"{prefix}_{CLEANING_METHOD}.csv").relative_to(ROOT)
                    ),
                }
            )

        slope = estimate_slope(rows_for_reconstruction)
        print(f"{reconstruction.upper()} divergence-advection slope: {slope:.3f}")

        for row in rows_for_reconstruction:
            row["estimated_slope"] = slope

        rows.extend(rows_for_reconstruction)

    csv_path = write_convergence_csv(rows)
    df = pd.read_csv(csv_path)
    plot_convergence(df)


if __name__ == "__main__":
    main()
