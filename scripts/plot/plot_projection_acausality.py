#!/usr/bin/env python3
"""Plot blast-wave elliptic-projection correction diagnostics.

This visualizes the instantaneous nonlocal elliptic correction
delta_B = B_after_projection - B_before_projection.  It is a numerical-method
diagnostic, not a physical-relativity statement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def snapshot_grid(df: pd.DataFrame, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx = int(df["i"].max()) + 1
    ny = int(df["j"].max()) + 1
    x = np.sort(df["x"].unique())
    y = np.sort(df["y"].unique())
    z = df.pivot(index="j", columns="i", values=field).to_numpy()
    if z.shape != (ny, nx):
        raise ValueError(f"unexpected grid shape for {field}: {z.shape}")
    return x, y, z


def radial_profile(df: pd.DataFrame, nbins: int) -> pd.DataFrame:
    r = pd.to_numeric(df["r"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["abs_dB_projection"], errors="coerce").to_numpy()
    finite = np.isfinite(r) & np.isfinite(y)
    r = r[finite]
    y = y[finite]
    if r.size == 0:
        raise ValueError("snapshot has no finite radial data")

    edges = np.linspace(0.0, r.max(), nbins + 1)
    rows: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (r >= lo) & (r < hi)
        if not np.any(mask):
            continue
        vals = y[mask]
        rows.append(
            {
                "r_mid": 0.5 * (lo + hi),
                "mean_abs_dB_projection": float(np.mean(vals)),
                "max_abs_dB_projection": float(np.max(vals)),
            }
        )
    return pd.DataFrame(rows)


def choose_summary_row(summary: pd.DataFrame, requested_time: float | None) -> pd.Series:
    with_snapshots = summary[summary["snapshot_file"].fillna("").astype(str) != ""].copy()
    if with_snapshots.empty:
        raise ValueError("summary has no rows with projection snapshot files")

    if requested_time is not None:
        time = pd.to_numeric(with_snapshots["time"], errors="coerce")
        idx = (time - requested_time).abs().idxmin()
        return with_snapshots.loc[idx]

    frac = pd.to_numeric(with_snapshots["outside_fraction"], errors="coerce").fillna(-1.0)
    return with_snapshots.loc[frac.idxmax()]


def add_radius_circle(ax, row: pd.Series, radius: float, **kwargs) -> None:
    circle = plt.Circle(
        (float(row["xc"]), float(row["yc"])),
        radius,
        fill=False,
        **kwargs,
    )
    ax.add_patch(circle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/mhd_runner")
    parser.add_argument("--fig-dir", default="figures/mhd_runner/projection_diagnostics")
    parser.add_argument("--prefix", default="mhd_blast")
    parser.add_argument("--method", default="elliptic_projection")
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--nbins", type=int, default=64)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    diag_dir = results_root / "projection_diagnostics"
    summary_path = diag_dir / f"{args.prefix}_{args.method}_projection_causal_summary.csv"
    summary = read_csv(summary_path)
    row = choose_summary_row(summary, args.time)

    snapshot_path = Path(str(row["snapshot_file"]))
    if not snapshot_path.is_absolute():
        snapshot_path = diag_dir / snapshot_path
    snapshot = read_csv(snapshot_path)
    profile = radial_profile(snapshot, args.nbins)

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    x, y, abs_db = snapshot_grid(snapshot, "abs_dB_projection")
    positive = abs_db[np.isfinite(abs_db) & (abs_db > 0.0)]
    norm = None
    if positive.size:
        norm = LogNorm(
            vmin=max(float(positive.min()), 1.0e-14),
            vmax=float(positive.max()),
        )

    time = float(row["time"])
    r0 = float(row["r0"])
    r_causal = float(row["r_causal"])
    outside_fraction = float(row["outside_fraction"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    im = ax.imshow(
        abs_db,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="magma",
        norm=norm,
        aspect="equal",
    )
    add_radius_circle(
        ax,
        row,
        r_causal,
        color="cyan",
        linewidth=1.8,
        label="estimated causal radius",
    )
    add_radius_circle(
        ax,
        row,
        r0,
        color="white",
        linewidth=1.2,
        linestyle="--",
        label="initial blast radius",
    )
    ax.set_title(r"$|\Delta B_{\rm projection}|$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    ax.plot(profile["r_mid"], profile["mean_abs_dB_projection"], label="mean")
    ax.plot(profile["r_mid"], profile["max_abs_dB_projection"], label="max")
    ax.axvline(r0, color="0.45", linestyle="--", linewidth=1.0, label="initial radius")
    ax.axvline(
        r_causal,
        color="tab:cyan",
        linestyle="-",
        linewidth=1.4,
        label="estimated causal radius",
    )
    ax.set_yscale("log")
    ax.set_xlabel("radius")
    ax.set_ylabel(r"$|\Delta B_{\rm projection}|$")
    ax.set_title("Radial projection correction")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(
        pd.to_numeric(summary["time"], errors="coerce"),
        pd.to_numeric(summary["outside_fraction"], errors="coerce"),
        color="tab:purple",
    )
    ax.scatter([time], [outside_fraction], color="black", s=24, zorder=3)
    ax.set_xlabel("time")
    ax.set_ylabel("outside correction fraction")
    ax.set_ylim(bottom=0.0)
    ax.set_title("Correction outside estimated causal radius")
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Blast-wave elliptic projection: instantaneous nonlocal correction\n"
        f"snapshot t={time:.6g}, outside fraction={outside_fraction:.3g}",
        y=1.02,
    )
    fig.tight_layout()

    out = fig_dir / f"{args.prefix}_{args.method}_projection_acausality.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    profile_out = fig_dir / f"{args.prefix}_{args.method}_projection_radial_profile.csv"
    profile.to_csv(profile_out, index=False)

    print(f"Saved {out}")
    print(f"Saved {profile_out}")
    print(f"Used {snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
