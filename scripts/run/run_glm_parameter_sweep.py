#!/usr/bin/env python3
"""Run GLM-family parameter sweeps through mhd_runner_cli.

This script only varies exposed GLM tuning parameters. It does not regenerate
report figures and does not introduce new cleaning methods.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = ROOT / "build"
DEFAULT_RUNNER = BUILD_DIR / "mhd_runner_cli"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "mhd_runner"
SWEEP_DIR = DEFAULT_OUTPUT_ROOT / "glm_sweep"
SUMMARY_DIR = DEFAULT_OUTPUT_ROOT / "summaries"
DIV_DIR = DEFAULT_OUTPUT_ROOT / "divergence"

METHODS = [
    "hyperbolic_glm",
    "mixed_glm",
    "mixed_eglm",
    "gi_mixed_eglm",
]

CSV_FIELDS = [
    "problem",
    "method",
    "nx",
    "ny",
    "reconstruction",
    "limiter",
    "damping_mode",
    "glm_ch_factor",
    "glm_cd",
    "glm_cr",
    "glm_subcycles",
    "glm_ch",
    "glm_cp",
    "glm_effective_cd",
    "glm_effective_cr",
    "status",
    "final_time_reached",
    "final_L2_norm_fv",
    "peak_L2_norm_fv",
    "time_integrated_L2_norm_fv",
    "final_Linf_norm_fv",
    "peak_Linf_norm_fv",
    "min_pressure",
    "min_density",
    "has_nonfinite",
    "has_negative_density",
    "has_negative_pressure",
    "energy_drift",
    "total_wall_time_sec",
    "seconds_per_step",
    "cell_updates_per_second",
    "cleaning_subcycles_total",
    "failure_reason",
    "summary_file",
    "diagnostic_file",
]


def run(cmd: list[str], dry_run: bool = False) -> subprocess.CompletedProcess[str] | None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return None
    return subprocess.run(cmd, cwd=ROOT, check=False)


def ensure_build(skip_build: bool, runner: Path) -> None:
    if skip_build and runner.exists():
        return
    if not (BUILD_DIR / "CMakeCache.txt").exists():
        subprocess.run(["cmake", "-S", ".", "-B", "build"], cwd=ROOT, check=True)
    subprocess.run(
        ["cmake", "--build", "build", "--target", runner.name, "--parallel"],
        cwd=ROOT,
        check=True,
    )


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = as_float(row, key, float(default))
    if not math.isfinite(value):
        return default
    return int(round(value))


def fmt_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def prefix_for(
    problem: str,
    method: str,
    n: int,
    ch_factor: float,
    subcycles: int,
    damping_mode: str,
    damping_value: float | None,
) -> str:
    damping_token = damping_mode
    if damping_value is not None:
        damping_token += fmt_token(damping_value)
    return (
        f"glm_sweep_{problem}_n{n}_{method}_"
        f"ch{fmt_token(ch_factor)}_sub{subcycles}_{damping_token}"
    )


def read_last_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"{path} has no data rows")
    return rows[-1]


def read_divergence_metrics(path: Path) -> dict[str, float]:
    metrics = {
        "final_L2_norm_fv": math.nan,
        "peak_L2_norm_fv": math.nan,
        "time_integrated_L2_norm_fv": math.nan,
        "final_Linf_norm_fv": math.nan,
        "peak_Linf_norm_fv": math.nan,
        "has_nonfinite": math.nan,
        "has_negative_density": math.nan,
        "has_negative_pressure": math.nan,
    }
    if not path.exists():
        return metrics

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return metrics

    times = [as_float(row, "time") for row in rows]
    l2 = [as_float(row, "L2_norm_fv") for row in rows]
    linf = [as_float(row, "Linf_norm_fv") for row in rows]

    if l2:
        metrics["final_L2_norm_fv"] = l2[-1]
        finite_l2 = [value for value in l2 if math.isfinite(value)]
        if finite_l2:
            metrics["peak_L2_norm_fv"] = max(finite_l2)
    if linf:
        metrics["final_Linf_norm_fv"] = linf[-1]
        finite_linf = [value for value in linf if math.isfinite(value)]
        if finite_linf:
            metrics["peak_Linf_norm_fv"] = max(finite_linf)

    integral = 0.0
    have_interval = False
    for i in range(1, len(rows)):
        values = (times[i - 1], times[i], l2[i - 1], l2[i])
        if all(math.isfinite(value) for value in values):
            integral += 0.5 * (l2[i - 1] + l2[i]) * (times[i] - times[i - 1])
            have_interval = True
    if have_interval:
        metrics["time_integrated_L2_norm_fv"] = integral
    elif len(rows) == 1 and math.isfinite(l2[0]):
        metrics["time_integrated_L2_norm_fv"] = 0.0

    for column in ("has_nonfinite", "has_negative_density", "has_negative_pressure"):
        if column in rows[0]:
            metrics[column] = max(as_float(row, column, 0.0) for row in rows)

    return metrics


def make_row(
    summary: dict[str, str],
    summary_path: Path,
    diagnostic_path: Path,
    problem: str,
    method: str,
    n: int,
    reconstruction: str,
    limiter: str,
    damping_mode: str,
) -> dict[str, object]:
    div = read_divergence_metrics(diagnostic_path)
    return {
        "problem": problem,
        "method": method,
        "nx": n,
        "ny": n,
        "reconstruction": reconstruction,
        "limiter": limiter,
        "damping_mode": damping_mode,
        "glm_ch_factor": as_float(summary, "glm_ch_factor"),
        "glm_cd": as_float(summary, "glm_cd"),
        "glm_cr": as_float(summary, "glm_cr"),
        "glm_subcycles": as_int(summary, "glm_subcycles", 1),
        "glm_ch": as_float(summary, "glm_ch"),
        "glm_cp": as_float(summary, "glm_cp"),
        "glm_effective_cd": as_float(summary, "glm_effective_cd"),
        "glm_effective_cr": as_float(summary, "glm_effective_cr"),
        "status": summary.get("status", ""),
        "final_time_reached": as_int(summary, "final_time_reached"),
        "final_L2_norm_fv": div["final_L2_norm_fv"],
        "peak_L2_norm_fv": div["peak_L2_norm_fv"],
        "time_integrated_L2_norm_fv": div["time_integrated_L2_norm_fv"],
        "final_Linf_norm_fv": div["final_Linf_norm_fv"],
        "peak_Linf_norm_fv": div["peak_Linf_norm_fv"],
        "min_pressure": as_float(summary, "min_pressure"),
        "min_density": as_float(summary, "min_density"),
        "has_nonfinite": div["has_nonfinite"],
        "has_negative_density": div["has_negative_density"],
        "has_negative_pressure": div["has_negative_pressure"],
        "energy_drift": as_float(summary, "energy_drift"),
        "total_wall_time_sec": as_float(summary, "total_wall_time_sec"),
        "seconds_per_step": as_float(summary, "seconds_per_step"),
        "cell_updates_per_second": as_float(summary, "cell_updates_per_second"),
        "cleaning_subcycles_total": as_int(summary, "cleaning_subcycles_total"),
        "failure_reason": summary.get("failure_reason", ""),
        "summary_file": str(summary_path.relative_to(ROOT)),
        "diagnostic_file": str(diagnostic_path.relative_to(ROOT)),
    }


def make_process_failure_row(
    problem: str,
    method: str,
    n: int,
    reconstruction: str,
    limiter: str,
    damping_mode: str,
    ch_factor: float,
    cd: float | None,
    cr: float | None,
    subcycles: int,
    returncode: int,
) -> dict[str, object]:
    row = {field: math.nan for field in CSV_FIELDS}
    row.update(
        {
            "problem": problem,
            "method": method,
            "nx": n,
            "ny": n,
            "reconstruction": reconstruction,
            "limiter": limiter,
            "damping_mode": damping_mode,
            "glm_ch_factor": ch_factor,
            "glm_cd": cd if cd is not None else math.nan,
            "glm_cr": cr if cr is not None else math.nan,
            "glm_subcycles": subcycles,
            "status": "process_failed",
            "final_time_reached": 0,
            "failure_reason": f"runner_exit_code_{returncode}",
            "summary_file": "",
            "diagnostic_file": "",
        }
    )
    return row


def damping_cases(mode: str, cd_values: list[float], cr_values: list[float]) -> list[tuple[str, float | None]]:
    if mode == "none":
        return [("none", None)]
    cases: list[tuple[str, float | None]] = []
    if mode in {"cd", "both"}:
        cases.extend(("cd", value) for value in cd_values)
    if mode in {"cr", "both"}:
        cases.extend(("cr", value) for value in cr_values)
    return cases


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="divergence_advection")
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--glm-ch-factors", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--glm-subcycles", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--glm-cd-values", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.5, 0.8])
    parser.add_argument("--glm-cr-values", nargs="+", type=float, default=[0.05, 0.1, 0.18, 0.3, 0.5, 1.0])
    parser.add_argument("--damping-mode", choices=["cd", "cr", "both", "none"], default="cd")
    parser.add_argument("--reconstruction", choices=["pcm", "plm"], default="plm")
    parser.add_argument("--limiter", choices=["minmod", "vanleer", "mc"], default="mc")
    parser.add_argument("--tfinal", type=float, default=0.05)
    parser.add_argument("--diagnostic-stride", type=int, default=100)
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--output-csv",
        default=str(SWEEP_DIR / "glm_parameter_sweep.csv"),
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.tfinal <= 0.0:
        raise SystemExit("--tfinal must be positive")
    if args.diagnostic_stride <= 0:
        raise SystemExit("--diagnostic-stride must be positive")
    if any(value <= 0.0 for value in args.glm_ch_factors):
        raise SystemExit("--glm-ch-factors must be positive")
    if any(value < 1 for value in args.glm_subcycles):
        raise SystemExit("--glm-subcycles values must be >= 1")
    if any(not (0.0 < value < 1.0) for value in args.glm_cd_values):
        raise SystemExit("--glm-cd-values must satisfy 0 < value < 1")
    if any(value <= 0.0 for value in args.glm_cr_values):
        raise SystemExit("--glm-cr-values must be positive")

    runner = Path(args.runner)
    if not runner.is_absolute():
        runner = ROOT / runner
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    summary_dir = output_root / "summaries"
    div_dir = output_root / "divergence"

    if not args.dry_run:
        ensure_build(args.skip_build, runner)

    all_damping_cases = damping_cases(args.damping_mode, args.glm_cd_values, args.glm_cr_values)
    rows: list[dict[str, object]] = []

    for n in args.resolutions:
        for method in args.methods:
            method_damping_cases = (
                [("unused", None)]
                if method == "hyperbolic_glm" and args.damping_mode != "none"
                else all_damping_cases
            )
            for ch_factor in args.glm_ch_factors:
                for subcycles in args.glm_subcycles:
                    for damping_mode, damping_value in method_damping_cases:
                        cd = damping_value if damping_mode == "cd" else None
                        cr = damping_value if damping_mode == "cr" else None
                        prefix = prefix_for(
                            args.problem,
                            method,
                            n,
                            ch_factor,
                            subcycles,
                            damping_mode,
                            damping_value,
                        )
                        cmd = [
                            str(runner),
                            "--performance-mode",
                            "--no-snapshots",
                            "--diagnostic-stride",
                            str(args.diagnostic_stride),
                            "--output-root",
                            str(output_root),
                            "--nx",
                            str(n),
                            "--ny",
                            str(n),
                            "--tfinal",
                            str(args.tfinal),
                            "--output-prefix",
                            prefix,
                            "--reconstruction",
                            args.reconstruction,
                            "--limiter",
                            args.limiter,
                            "--glm-ch-factor",
                            str(ch_factor),
                            "--glm-subcycles",
                            str(subcycles),
                        ]
                        if cd is not None:
                            cmd.extend(["--glm-cd", str(cd)])
                        if cr is not None:
                            cmd.extend(["--glm-cr", str(cr)])
                        cmd.extend([args.problem, method])

                        result = run(cmd, dry_run=args.dry_run)
                        if args.dry_run:
                            continue
                        if result is not None and result.returncode != 0:
                            if not args.continue_on_error:
                                raise SystemExit(
                                    f"{method} N={n} exited with code {result.returncode}"
                                )
                            rows.append(
                                make_process_failure_row(
                                    args.problem,
                                    method,
                                    n,
                                    args.reconstruction,
                                    args.limiter,
                                    damping_mode,
                                    ch_factor,
                                    cd,
                                    cr,
                                    subcycles,
                                    result.returncode,
                                )
                            )
                            continue

                        summary_path = summary_dir / f"{prefix}_{method}_summary.csv"
                        diagnostic_path = div_dir / f"{prefix}_{method}.csv"
                        try:
                            summary = read_last_csv_row(summary_path)
                        except (OSError, RuntimeError) as exc:
                            if not args.continue_on_error:
                                raise
                            row = make_process_failure_row(
                                args.problem,
                                method,
                                n,
                                args.reconstruction,
                                args.limiter,
                                damping_mode,
                                ch_factor,
                                cd,
                                cr,
                                subcycles,
                                0,
                            )
                            row["status"] = "missing_summary"
                            row["failure_reason"] = str(exc)
                            rows.append(row)
                            continue

                        rows.append(
                            make_row(
                                summary,
                                summary_path,
                                diagnostic_path,
                                args.problem,
                                method,
                                n,
                                args.reconstruction,
                                args.limiter,
                                damping_mode,
                            )
                        )

    if args.dry_run:
        return 0
    if not rows:
        raise SystemExit("no sweep rows were collected")

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = ROOT / output_csv
    write_csv(rows, output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
