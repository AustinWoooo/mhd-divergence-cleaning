#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

ROOT = Path(".")
RESULTS = ROOT / "results" / "mhd_runner"
DIV_DIR = RESULTS / "divergence"
SNAP_DIR = RESULTS / "snapshots"
FAIL_DIR = RESULTS / "failures"
FIG_DIR = ROOT / "figures" / "mhd_runner"
FIG_PRESSURE_DIR = FIG_DIR / "pressure"
FIG_SNAP_DIR = FIG_DIR / "snapshots"
FIG_DATA_DIR = FIG_DIR / "data"
for _d in (FIG_PRESSURE_DIR, FIG_SNAP_DIR, FIG_DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

STAGE_SUMMARY_CSV = FIG_DATA_DIR / "cleaning_failure_stage_summary.csv"
RUNNER_SUMMARY_CSV = FIG_DATA_DIR / "mhd_runner_summary.csv"
T_FINAL_OT = 0.5

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
    "powell_source_subcycled": "Powell subcycled control",
    "powell_source_limited": "Pressure-limited Powell",
}

_METHOD_SHORT = {
    "none": "none",
    "hyperbolic_glm": "hyp. GLM",
    "mixed_glm": "mixed GLM",
    "mixed_eglm": "EGLM",
    "gi_mixed_eglm": "GI-EGLM",
    "parabolic": "parabolic",
    "elliptic_projection": "projection",
    "powell_source": "Powell",
    "powell_source_limited": "Powell (lim.)",
}

_METHOD_ORDER = [
    "none", "hyperbolic_glm", "mixed_glm", "mixed_eglm", "gi_mixed_eglm",
    "parabolic", "elliptic_projection", "powell_source", "powell_source_limited",
]

_POLICY_SHORT = {
    "conserve_total_energy": "conserve",
    "preserve_thermal_pressure": "preserve",
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

    out = FIG_PRESSURE_DIR / "mhd_runner_ot_min_pressure.png"
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

    out = FIG_PRESSURE_DIR / "mhd_runner_ot_pressure_failure_stages.png"
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

    out = FIG_SNAP_DIR / "mhd_runner_ot_pressure_maps.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


# --- Clean aggregate diagnostics ----------------------------------------------


def _load_clean_failures():
    """Return failure summary with garbage rows removed: one row per method/policy, OT only.

    The raw CSV is contaminated with per-step limiter-stat rows that have empty
    method and problem fields.  This function keeps only the properly-labeled rows
    and returns the earliest failure time per (method, energy_policy) pair.
    """
    if not STAGE_SUMMARY_CSV.exists():
        print(f"missing: {STAGE_SUMMARY_CSV}")
        return pd.DataFrame()
    df = pd.read_csv(STAGE_SUMMARY_CSV)
    df = df[df["method"].notna() & df["problem"].notna()].copy()
    df = df[df["problem"] == "orszag_tang"].copy()
    df["failure_time"] = pd.to_numeric(df["failure_time"], errors="coerce")
    df = df.sort_values("failure_time")
    df = df.groupby(["method", "energy_policy"], as_index=False).first()
    return df


def _get_stage_pressures(method, energy_policy):
    """Return (p_before, p_after_hydro, p_after_cleaning) from the first row of the failure file."""
    fname = f"mhd_ot_{method}_{energy_policy}_failure.csv"
    path = FAIL_DIR / fname
    nan = float("nan")
    if not path.exists():
        return nan, nan, nan
    df = pd.read_csv(path)
    if df.empty:
        return nan, nan, nan
    row = df.iloc[0]

    def _get(col):
        if col not in df.columns:
            return nan
        try:
            return float(row[col])
        except (ValueError, TypeError):
            return nan

    return (
        _get("min_pressure_before_hydro"),
        _get("min_pressure_after_hydro"),
        _get("min_pressure_after_cleaning_B"),
    )


def print_failure_table():
    """Print compact terminal table: method, policy, failure_time, stage, cell indices, stage pressures, energy decomposition."""
    fails = _load_clean_failures()
    if fails.empty:
        print("no failure data found")
        return

    hdr = (
        f"{'method':<25} {'policy':<28} {'t_fail':>8} {'stage':<24}"
        f" {'i':>4} {'j':>4}"
        f" {'p_before':>12} {'p_aft_hydro':>12} {'p_aft_clean':>12}"
        f" {'E_tot':>10} {'E_kin':>10} {'E_mag':>10} {'E_int':>10}"
    )
    print()
    print("Orszag-Tang first bad-state summary")
    print(hdr)
    print("-" * len(hdr))

    def _f(v):
        return f"{v:>12.4e}" if np.isfinite(v) else f"{'N/A':>12}"

    for _, row in fails.iterrows():
        method = str(row["method"])
        policy = str(row["energy_policy"])
        t_fail = float(row["failure_time"]) if pd.notna(row["failure_time"]) else float("nan")
        stage = str(row.get("failure_stage") or "")
        i_val = int(row["i"]) if pd.notna(row["i"]) else -1
        j_val = int(row["j"]) if pd.notna(row["j"]) else -1
        e_kin = float(row["kinetic_energy"]) if pd.notna(row["kinetic_energy"]) else float("nan")
        e_mag = float(row["magnetic_energy"]) if pd.notna(row["magnetic_energy"]) else float("nan")
        e_int = float(row["internal_energy"]) if pd.notna(row["internal_energy"]) else float("nan")
        e_tot = e_kin + e_mag + e_int
        p_b, p_h, p_c = _get_stage_pressures(method, policy)
        print(
            f"{method:<25} {policy:<28} {t_fail:>8.5f} {stage:<24}"
            f" {i_val:>4} {j_val:>4}"
            f" {_f(p_b)} {_f(p_h)} {_f(p_c)}"
            f" {_f(e_tot)} {_f(e_kin)} {_f(e_mag)} {_f(e_int)}"
        )
    print()


def plot_ot_failure_times_clean():
    """One bar per method from the runner summary.

    Methods that completed to t=0.5 are shown hatched at T_FINAL_OT.
    Methods that terminated early are shown at their final time.
    Saves to mhd_runner_ot_failure_times_clean.png.
    """
    if not RUNNER_SUMMARY_CSV.exists():
        print(f"missing: {RUNNER_SUMMARY_CSV}")
        return

    ot = pd.read_csv(RUNNER_SUMMARY_CSV)
    ot = ot[ot["problem"] == "orszag_tang"].copy()

    labels, times, is_completed_list = [], [], []
    for method in _METHOD_ORDER:
        row = ot[ot["method"] == method]
        if row.empty:
            continue
        row = row.iloc[0]
        comp = int(row.get("completed", 0)) == 1
        t = T_FINAL_OT if comp else float(row["final_time"])
        labels.append(_METHOD_SHORT.get(method, method))
        times.append(t)
        is_completed_list.append(comp)

    if not labels:
        print("no runner summary data for orszag_tang")
        return

    x = np.arange(len(labels))
    colors = ["tab:green" if c else "tab:red" for c in is_completed_list]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for xi, t, col, comp in zip(x, times, colors, is_completed_list):
        ax.bar(xi, t, color=col, hatch="///" if comp else "", edgecolor="black", lw=0.7, alpha=0.82)

    ax.axhline(T_FINAL_OT, color="gray", lw=1.2, ls="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("time")
    ax.set_title(
        "Orszag–Tang: run termination time by method\n"
        r"hatched = completed to $t=0.5$; solid = stopped at first bad state"
    )
    ax.set_ylim(0, T_FINAL_OT * 1.12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        handles=[
            Patch(fc="tab:green", hatch="///", ec="black", label=f"completed (t={T_FINAL_OT})"),
            Patch(fc="tab:red", ec="black", label="terminated at bad state"),
        ],
        fontsize=9,
    )
    plt.tight_layout()

    out = FIG_PRESSURE_DIR / "mhd_runner_ot_failure_times_clean.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def plot_ot_bad_cell_energy_clean():
    """Stacked bar chart of energy decomposition at first bad cell, one bar per method/policy.

    powell_source_limited is excluded: its cell energies are unphysical
    (E_kin ~ 6500, E_mag ~ 2000) and would suppress all other bars.
    Saves to mhd_runner_ot_bad_cell_energy_clean.png.
    """
    fails = _load_clean_failures()
    if fails.empty:
        print("no failure energy data found")
        return

    fails = fails[fails["method"] != "powell_source_limited"].copy()

    method_rank = {m: i for i, m in enumerate(_METHOD_ORDER)}
    fails["_rank"] = fails["method"].map(method_rank).fillna(99)
    fails = fails.sort_values(["_rank", "energy_policy"]).reset_index(drop=True)

    e_kin = pd.to_numeric(fails["kinetic_energy"], errors="coerce").to_numpy()
    e_mag = pd.to_numeric(fails["magnetic_energy"], errors="coerce").to_numpy()
    e_int = pd.to_numeric(fails["internal_energy"], errors="coerce").to_numpy()
    e_tot = e_kin + e_mag + e_int

    labels = []
    for _, row in fails.iterrows():
        short = _METHOD_SHORT.get(str(row["method"]), str(row["method"]))
        pol = _POLICY_SHORT.get(str(row["energy_policy"]), str(row["energy_policy"]))
        labels.append(f"{short}\n({pol})")

    x = np.arange(len(fails))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x, e_kin, label=r"$E_{\rm kin}$", color="tab:blue", alpha=0.85)
    ax.bar(x, e_mag, bottom=e_kin, label=r"$E_{\rm mag}$", color="tab:orange", alpha=0.85)

    e_int_pos = np.where(e_int >= 0, e_int, 0.0)
    e_int_neg = np.where(e_int < 0, e_int, 0.0)
    if e_int_pos.any():
        ax.bar(x, e_int_pos, bottom=e_kin + e_mag, label=r"$E_{\rm int}$ ($\geq$0)", color="tab:green", alpha=0.85)
    ax.bar(x, e_int_neg, label=r"$E_{\rm int}$ (<0)", color="tab:red", alpha=0.85)

    ax.scatter(x, e_tot, color="black", zorder=5, marker="D", s=40, label=r"$E_{\rm total}$")

    ax.axhline(0.0, color="black", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("cell energy density")
    ax.set_title(
        "Orszag–Tang: energy decomposition at first bad cell\n"
        r"(powell\_source\_limited excluded: $E_{\rm kin}\approx6500$, $E_{\rm mag}\approx2000$ — unphysical)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    plt.tight_layout()

    out = FIG_PRESSURE_DIR / "mhd_runner_ot_bad_cell_energy_clean.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def main():
    plot_ot_min_pressure()
    plot_ot_pressure_failure_stages()
    plot_ot_pressure_maps()
    print_failure_table()
    plot_ot_failure_times_clean()
    plot_ot_bad_cell_energy_clean()


if __name__ == "__main__":
    main()
