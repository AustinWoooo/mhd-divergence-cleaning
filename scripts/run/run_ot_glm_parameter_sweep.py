#!/usr/bin/env python3
"""Run and rank an Orszag-Tang-specific GLM parameter sweep.

This script only orchestrates existing mhd_runner_cli controls.  It does not
modify GLM kernels or solver physics.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from pathlib import Path
import subprocess
from typing import Iterable

from run_glm_parameter_sweep import (
    ROOT,
    DEFAULT_RUNNER,
    DEFAULT_OUTPUT_ROOT,
    as_float,
    as_int,
    ensure_build,
    make_process_failure_row,
    read_divergence_metrics,
    read_last_csv_row,
)


PROBLEM = "orszag_tang"
METHODS = [
    "hyperbolic_glm",
    "mixed_glm",
    "mixed_eglm",
    "gi_mixed_eglm",
]
MIXED_METHODS = {"mixed_glm", "mixed_eglm", "gi_mixed_eglm"}
BASELINE_METHODS = ["none", "parabolic", "mixed_glm", "elliptic_projection"]

RANKED_FIELDS = [
    "problem",
    "method",
    "nx",
    "status",
    "glm_ch_factor",
    "glm_cr",
    "glm_cd",
    "glm_subcycles",
    "glm_ch",
    "glm_cp",
    "final_L2_norm_fv",
    "peak_L2_norm_fv",
    "time_integrated_L2_norm_fv",
    "final_Linf_norm_fv",
    "min_pressure",
    "energy_drift",
    "total_wall_time_sec",
    "diagnostic_file",
]


def existing_performance_row(method: str, nx: int) -> dict[str, object] | None:
    path = DEFAULT_OUTPUT_ROOT / "performance" / "performance_scaling_all_methods.csv"
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates = [
        row for row in rows
        if row.get("problem") == PROBLEM
        and row.get("method") == method
        and row.get("nx") == str(nx)
    ]
    if not candidates:
        return None
    row = candidates[-1]
    out: dict[str, object] = dict(row)
    out.setdefault("final_time_reached", 1 if row.get("status") == "finished" else 0)
    out.setdefault("has_nonfinite", 0)
    out.setdefault("has_negative_density", 0)
    out.setdefault("has_negative_pressure", 0)
    return out


def finite_float(row: dict[str, object], key: str, default: float = math.nan) -> float:
    value = row.get(key, default)
    if value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_int(row: dict[str, object], key: str, default: int = 0) -> int:
    value = finite_float(row, key, float(default))
    if not math.isfinite(value):
        return default
    return int(round(value))


def token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def param_key(row: dict[str, object]) -> tuple[str, float, int, str, float | None]:
    cd = finite_float(row, "glm_cd")
    cr = finite_float(row, "glm_cr")
    if math.isfinite(cd):
        mode = "cd"
        damping = cd
    elif math.isfinite(cr):
        mode = "cr"
        damping = cr
    else:
        mode = "unused"
        damping = None
    return (
        str(row["method"]),
        finite_float(row, "glm_ch_factor"),
        finite_int(row, "glm_subcycles", 1),
        mode,
        damping,
    )


def prefix_for_case(
    stage: str,
    method: str,
    n: int,
    ch_factor: float,
    subcycles: int,
    damping_mode: str,
    damping_value: float | None,
) -> str:
    damping = damping_mode
    if damping_value is not None:
        damping += token(damping_value)
    return (
        f"ot_glm_sweep_{stage}_n{n}_{method}_"
        f"ch{token(ch_factor)}_sub{subcycles}_{damping}"
    )


def prefix_for_baseline(method: str, n: int) -> str:
    return f"ot_glm_sweep_baseline_n{n}_{method}"


def robust(row: dict[str, object]) -> bool:
    return (
        row.get("status") == "finished"
        and finite_int(row, "final_time_reached", 1) == 1
        and finite_float(row, "min_pressure") > 0.0
        and finite_float(row, "min_density") > 0.0
        and finite_int(row, "has_nonfinite") == 0
        and finite_int(row, "has_negative_density") == 0
        and finite_int(row, "has_negative_pressure") == 0
        and math.isfinite(finite_float(row, "final_L2_norm_fv"))
        and math.isfinite(finite_float(row, "time_integrated_L2_norm_fv"))
    )


def cmd_for_case(
    runner: Path,
    output_root: Path,
    prefix: str,
    method: str,
    n: int,
    tfinal: float,
    diagnostic_stride: int,
    reconstruction: str,
    limiter: str,
    ch_factor: float | None = None,
    subcycles: int | None = None,
    damping_mode: str = "unused",
    damping_value: float | None = None,
) -> list[str]:
    cmd = [
        str(runner),
        "--performance-mode",
        "--no-snapshots",
        "--diagnostic-stride",
        str(diagnostic_stride),
        "--output-root",
        str(output_root),
        "--nx",
        str(n),
        "--ny",
        str(n),
        "--tfinal",
        str(tfinal),
        "--output-prefix",
        prefix,
        "--reconstruction",
        reconstruction,
        "--limiter",
        limiter,
    ]
    if ch_factor is not None:
        cmd.extend(["--glm-ch-factor", str(ch_factor)])
    if subcycles is not None:
        cmd.extend(["--glm-subcycles", str(subcycles)])
    if damping_mode == "cd" and damping_value is not None:
        cmd.extend(["--glm-cd", str(damping_value)])
    elif damping_mode == "cr" and damping_value is not None:
        cmd.extend(["--glm-cr", str(damping_value)])
    cmd.extend([PROBLEM, method])
    return cmd


def collect_row(
    output_root: Path,
    prefix: str,
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
    if returncode != 0:
        return make_process_failure_row(
            PROBLEM,
            method,
            n,
            reconstruction,
            limiter,
            damping_mode,
            ch_factor,
            cd,
            cr,
            subcycles,
            returncode,
        )

    summary_path = output_root / "summaries" / f"{prefix}_{method}_summary.csv"
    diagnostic_path = output_root / "divergence" / f"{prefix}_{method}.csv"
    try:
        summary = read_last_csv_row(summary_path)
        div = read_divergence_metrics(diagnostic_path)
        try:
            summary_file = str(summary_path.relative_to(ROOT))
        except ValueError:
            summary_file = str(summary_path)
        try:
            diagnostic_file = str(diagnostic_path.relative_to(ROOT))
        except ValueError:
            diagnostic_file = str(diagnostic_path)
        return {
            "problem": PROBLEM,
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
            "summary_file": summary_file,
            "diagnostic_file": diagnostic_file,
        }
    except (OSError, RuntimeError, ValueError) as exc:
        row = make_process_failure_row(
            PROBLEM,
            method,
            n,
            reconstruction,
            limiter,
            damping_mode,
            ch_factor,
            cd,
            cr,
            subcycles,
            0,
        )
        row["status"] = "missing_summary"
        row["failure_reason"] = str(exc)
        return row


def run_one(
    cmd: list[str],
    quiet: bool,
    print_command: bool,
) -> int:
    if print_command:
        print(" ".join(cmd), flush=True)
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.PIPE if quiet else None
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=stdout,
        stderr=stderr,
        text=True,
        check=False,
    )
    if result.returncode != 0 and quiet and result.stderr:
        print(result.stderr[-2000:], flush=True)
    return result.returncode


def run_cases(
    cases: list[dict[str, object]],
    jobs: int,
    quiet: bool,
    dry_run: bool,
    print_commands: bool = False,
) -> list[dict[str, object]]:
    if dry_run:
        for case in cases:
            print(" ".join(case["cmd"]))  # type: ignore[index]
        return []

    rows: list[dict[str, object]] = []
    if jobs == 1:
        for index, case in enumerate(cases, start=1):
            returncode = run_one(case["cmd"], quiet, print_commands)  # type: ignore[arg-type,index]
            rows.append(case["collect"](returncode))  # type: ignore[index,operator]
            if index % 100 == 0 or index == len(cases):
                print(f"  completed {index}/{len(cases)} cases", flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        future_to_case = {
            pool.submit(run_one, case["cmd"], quiet, print_commands): case  # type: ignore[arg-type,index]
            for case in cases
        }
        for index, future in enumerate(as_completed(future_to_case), start=1):
            case = future_to_case[future]
            returncode = future.result()
            rows.append(case["collect"](returncode))  # type: ignore[index,operator]
            if index % 100 == 0 or index == len(cases):
                print(f"  completed {index}/{len(cases)} cases", flush=True)
    return rows


def damping_cases(
    method: str,
    cr_values: Iterable[float],
    cd_values: Iterable[float],
) -> list[tuple[str, float | None]]:
    if method not in MIXED_METHODS:
        return [("unused", None)]
    return [("cr", value) for value in cr_values] + [
        ("cd", value) for value in cd_values
    ]


def make_sweep_cases(args: argparse.Namespace, runner: Path, output_root: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for n in args.coarse_resolutions:
        for method in METHODS:
            for ch_factor in args.glm_ch_factors:
                for subcycles in args.glm_subcycles:
                    for damping_mode, damping_value in damping_cases(
                        method,
                        args.glm_cr_values,
                        args.glm_cd_values,
                    ):
                        cd = damping_value if damping_mode == "cd" else None
                        cr = damping_value if damping_mode == "cr" else None
                        prefix = prefix_for_case(
                            "coarse",
                            method,
                            n,
                            ch_factor,
                            subcycles,
                            damping_mode,
                            damping_value,
                        )
                        cmd = cmd_for_case(
                            runner,
                            output_root,
                            prefix,
                            method,
                            n,
                            args.tfinal,
                            args.diagnostic_stride,
                            args.reconstruction,
                            args.limiter,
                            ch_factor,
                            subcycles,
                            damping_mode,
                            damping_value,
                        )
                        cases.append(
                            {
                                "cmd": cmd,
                                "collect": lambda returncode, prefix=prefix, method=method, n=n,
                                damping_mode=damping_mode, ch_factor=ch_factor, cd=cd, cr=cr,
                                subcycles=subcycles: collect_row(
                                    output_root,
                                    prefix,
                                    method,
                                    n,
                                    args.reconstruction,
                                    args.limiter,
                                    damping_mode,
                                    ch_factor,
                                    cd,
                                    cr,
                                    subcycles,
                                    returncode,
                                ),
                            }
                        )
    return cases


def make_baseline_cases(
    args: argparse.Namespace,
    runner: Path,
    output_root: Path,
    resolutions: Iterable[int],
    methods: Iterable[str],
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for n in resolutions:
        for method in methods:
            prefix = prefix_for_baseline(method, n)
            cmd = cmd_for_case(
                runner,
                output_root,
                prefix,
                method,
                n,
                args.tfinal,
                args.diagnostic_stride,
                args.reconstruction,
                args.limiter,
                4.0,
                1,
                "cr" if method in MIXED_METHODS else "unused",
                0.1 if method in MIXED_METHODS else None,
            )
            cases.append(
                {
                    "cmd": cmd,
                    "collect": lambda returncode, prefix=prefix, method=method, n=n: collect_row(
                        output_root,
                        prefix,
                        method,
                        n,
                        args.reconstruction,
                        args.limiter,
                        "baseline",
                        4.0,
                        None,
                        0.1 if method in MIXED_METHODS else None,
                        1,
                        returncode,
                    ),
                }
            )
    return cases


def baseline_by_resolution(rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    for row in rows:
        if row.get("method") == "none" and robust(row):
            out[int(row["nx"])] = row
    return out


def row_rank_key(row: dict[str, object], baselines: dict[int, dict[str, object]]) -> tuple[object, ...]:
    n = int(row["nx"])
    baseline = baselines.get(n)
    baseline_final = finite_float(baseline or {}, "final_L2_norm_fv", math.inf)
    baseline_integral = finite_float(
        baseline or {},
        "time_integrated_L2_norm_fv",
        math.inf,
    )
    baseline_pressure = finite_float(baseline or {}, "min_pressure", -math.inf)
    final = finite_float(row, "final_L2_norm_fv", math.inf)
    integral = finite_float(row, "time_integrated_L2_norm_fv", math.inf)
    min_pressure = finite_float(row, "min_pressure", -math.inf)
    energy = abs(finite_float(row, "energy_drift", math.inf))
    wall = finite_float(row, "total_wall_time_sec", math.inf)
    return (
        not robust(row),
        not (final < baseline_final),
        not (integral < baseline_integral),
        max(0.0, baseline_pressure - min_pressure),
        energy,
        wall,
        final,
        integral,
    )


def setting_rank_key(
    rows: list[dict[str, object]],
    baselines: dict[int, dict[str, object]],
) -> tuple[object, ...]:
    robust_rows = [row for row in rows if robust(row)]
    if len(robust_rows) != len(rows):
        return (True, math.inf, math.inf, math.inf, math.inf, math.inf)

    final_ratios = []
    integral_ratios = []
    pressure_drops = []
    energy = []
    wall = []
    for row in robust_rows:
        baseline = baselines.get(int(row["nx"]), {})
        final_base = finite_float(baseline, "final_L2_norm_fv", math.inf)
        integral_base = finite_float(
            baseline,
            "time_integrated_L2_norm_fv",
            math.inf,
        )
        pressure_base = finite_float(baseline, "min_pressure", -math.inf)
        final_ratios.append(finite_float(row, "final_L2_norm_fv", math.inf) / final_base)
        integral_ratios.append(
            finite_float(row, "time_integrated_L2_norm_fv", math.inf) / integral_base
        )
        pressure_drops.append(max(0.0, pressure_base - finite_float(row, "min_pressure", -math.inf)))
        energy.append(abs(finite_float(row, "energy_drift", math.inf)))
        wall.append(finite_float(row, "total_wall_time_sec", math.inf))

    return (
        False,
        not all(value < 1.0 for value in final_ratios),
        not all(value < 1.0 for value in integral_ratios),
        sum(final_ratios) / len(final_ratios),
        sum(integral_ratios) / len(integral_ratios),
        max(pressure_drops),
        max(energy),
        sum(wall) / len(wall),
    )


def select_top_settings(
    rows: list[dict[str, object]],
    baselines: dict[int, dict[str, object]],
    top: int,
) -> list[tuple[str, float, int, str, float | None]]:
    by_setting: dict[tuple[str, float, int, str, float | None], list[dict[str, object]]] = {}
    required_n = {64, 128}
    for row in rows:
        if int(row["nx"]) not in required_n:
            continue
        by_setting.setdefault(param_key(row), []).append(row)

    selected: list[tuple[str, float, int, str, float | None]] = []
    for method in METHODS:
        method_settings = [
            (setting, setting_rows)
            for setting, setting_rows in by_setting.items()
            if setting[0] == method
            and {int(row["nx"]) for row in setting_rows} == required_n
        ]
        method_settings.sort(key=lambda item: setting_rank_key(item[1], baselines))
        selected.extend(setting for setting, _ in method_settings[:top])
    return selected


def make_fine_cases(
    args: argparse.Namespace,
    runner: Path,
    output_root: Path,
    settings: list[tuple[str, float, int, str, float | None]],
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for method, ch_factor, subcycles, damping_mode, damping_value in settings:
        cd = damping_value if damping_mode == "cd" else None
        cr = damping_value if damping_mode == "cr" else None
        prefix = prefix_for_case(
            "fine",
            method,
            args.fine_resolution,
            ch_factor,
            subcycles,
            damping_mode,
            damping_value,
        )
        cmd = cmd_for_case(
            runner,
            output_root,
            prefix,
            method,
            args.fine_resolution,
            args.tfinal,
            args.diagnostic_stride,
            args.reconstruction,
            args.limiter,
            ch_factor,
            subcycles,
            damping_mode,
            damping_value,
        )
        cases.append(
            {
                "cmd": cmd,
                "collect": lambda returncode, prefix=prefix, method=method,
                damping_mode=damping_mode, ch_factor=ch_factor, cd=cd, cr=cr,
                subcycles=subcycles: collect_row(
                    output_root,
                    prefix,
                    method,
                    args.fine_resolution,
                    args.reconstruction,
                    args.limiter,
                    damping_mode,
                    ch_factor,
                    cd,
                    cr,
                    subcycles,
                    returncode,
                ),
            }
        )
    return cases


def write_ranked(rows: list[dict[str, object]], output: Path, baselines: dict[int, dict[str, object]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    robust_rows = [row for row in rows if robust(row)]
    robust_rows.sort(key=lambda row: row_rank_key(row, baselines))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANKED_FIELDS)
        writer.writeheader()
        for row in robust_rows:
            writer.writerow({field: row.get(field, "") for field in RANKED_FIELDS})
    print(f"Wrote {output}")


def best_by_method(
    rows: list[dict[str, object]],
    baselines: dict[int, dict[str, object]],
    nx: int,
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for method in METHODS:
        method_rows = [
            row for row in rows
            if row.get("method") == method and int(row["nx"]) == nx and robust(row)
        ]
        if method_rows:
            out[method] = sorted(method_rows, key=lambda row: row_rank_key(row, baselines))[0]
    return out


def format_row(label: str, row: dict[str, object] | None) -> str:
    if row is None:
        return f"{label}: missing"
    return (
        f"{label}: method={row.get('method')}, nx={row.get('nx')}, "
        f"final={finite_float(row, 'final_L2_norm_fv'):.6g}, "
        f"integral={finite_float(row, 'time_integrated_L2_norm_fv'):.6g}, "
        f"min_p={finite_float(row, 'min_pressure'):.6g}, "
        f"energy={finite_float(row, 'energy_drift'):.6g}, "
        f"wall={finite_float(row, 'total_wall_time_sec'):.3g}s, "
        f"ch_factor={finite_float(row, 'glm_ch_factor'):.6g}, "
        f"cr={finite_float(row, 'glm_cr'):.6g}, "
        f"cd={finite_float(row, 'glm_cd'):.6g}, "
        f"sub={finite_int(row, 'glm_subcycles', 1)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-resolutions", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--fine-resolution", type=int, default=256)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument(
        "--glm-ch-factors",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    )
    parser.add_argument("--glm-subcycles", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument(
        "--glm-cr-values",
        nargs="+",
        type=float,
        default=[0.01, 0.03, 0.05, 0.1, 0.18, 0.3, 0.5, 1.0],
    )
    parser.add_argument(
        "--glm-cd-values",
        nargs="+",
        type=float,
        default=[0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8],
    )
    parser.add_argument("--top-per-method", type=int, default=5)
    parser.add_argument("--reconstruction", choices=["pcm", "plm"], default="plm")
    parser.add_argument("--limiter", choices=["minmod", "vanleer", "mc"], default="vanleer")
    parser.add_argument("--tfinal", type=float, default=0.05)
    parser.add_argument("--diagnostic-stride", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER))
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT / "glm_sweep" / "ot_runs"),
    )
    parser.add_argument(
        "--ranked-output-csv",
        default=str(DEFAULT_OUTPUT_ROOT / "glm_sweep" / "ot_glm_parameter_sweep_ranked_summary.csv"),
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet-runner", action="store_true", default=True)
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument(
        "--include-elliptic-baseline",
        action="store_true",
        help="Also rerun elliptic_projection baselines. This is expensive at nx=256.",
    )
    args = parser.parse_args()

    args.methods = [method for method in args.methods if method in METHODS]
    if set(args.methods) != set(METHODS):
        raise SystemExit("this OT sweep intentionally supports only GLM-family methods")
    if args.tfinal <= 0.0:
        raise SystemExit("--tfinal must be positive")
    if args.diagnostic_stride <= 0:
        raise SystemExit("--diagnostic-stride must be positive")
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    if args.top_per_method < 1:
        raise SystemExit("--top-per-method must be >= 1")

    runner = Path(args.runner)
    if not runner.is_absolute():
        runner = ROOT / runner
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    ranked_output = Path(args.ranked_output_csv)
    if not ranked_output.is_absolute():
        ranked_output = ROOT / ranked_output

    if not args.dry_run:
        ensure_build(args.skip_build, runner)

    baseline_methods = ["none", "parabolic", "mixed_glm"]
    if args.include_elliptic_baseline:
        baseline_methods.append("elliptic_projection")
    baseline_resolutions = sorted(set(args.coarse_resolutions + [args.fine_resolution]))
    baseline_cases = make_baseline_cases(
        args,
        runner,
        output_root,
        baseline_resolutions,
        baseline_methods,
    )
    print(f"Running {len(baseline_cases)} baseline cases", flush=True)
    baseline_rows = run_cases(
        baseline_cases,
        args.jobs,
        args.quiet_runner,
        args.dry_run,
        args.print_commands,
    )

    coarse_cases = make_sweep_cases(args, runner, output_root)
    print(f"Running {len(coarse_cases)} coarse GLM cases", flush=True)
    coarse_rows = run_cases(
        coarse_cases,
        args.jobs,
        args.quiet_runner,
        args.dry_run,
        args.print_commands,
    )
    if args.dry_run:
        return 0

    baselines = baseline_by_resolution(baseline_rows)
    missing_baselines = sorted(set(args.coarse_resolutions + [args.fine_resolution]) - set(baselines))
    if missing_baselines:
        raise SystemExit(f"missing robust none baselines for nx={missing_baselines}")

    top_settings = select_top_settings(coarse_rows, baselines, args.top_per_method)
    print("Selected fine-grid settings:")
    for setting in top_settings:
        print(f"  {setting}")

    fine_cases = make_fine_cases(args, runner, output_root, top_settings)
    print(f"Running {len(fine_cases)} fine GLM cases", flush=True)
    fine_rows = run_cases(
        fine_cases,
        args.jobs,
        args.quiet_runner,
        args.dry_run,
        args.print_commands,
    )

    all_rows = baseline_rows + coarse_rows + fine_rows
    write_ranked(all_rows, ranked_output, baselines)

    best = best_by_method(fine_rows, baselines, args.fine_resolution)
    baseline_lookup = {
        (row.get("method"), int(row["nx"])): row
        for row in baseline_rows
        if robust(row)
    }
    print(f"\nOT nx={args.fine_resolution} comparison:")
    print(format_row("none", baseline_lookup.get(("none", args.fine_resolution))))
    print(format_row("parabolic", baseline_lookup.get(("parabolic", args.fine_resolution))))
    print(format_row("current mixed_glm ch4 cr0.1", baseline_lookup.get(("mixed_glm", args.fine_resolution))))
    print(format_row("best OT-tuned mixed_glm", best.get("mixed_glm")))
    print(format_row("best OT-tuned mixed_eglm", best.get("mixed_eglm")))
    print(format_row("best OT-tuned gi_mixed_eglm", best.get("gi_mixed_eglm")))
    elliptic = baseline_lookup.get(("elliptic_projection", args.fine_resolution))
    if elliptic is None:
        elliptic = existing_performance_row("elliptic_projection", args.fine_resolution)
    print(format_row("elliptic_projection", elliptic))

    none_256 = baseline_lookup.get(("none", args.fine_resolution))
    if none_256 is not None:
        none_final = finite_float(none_256, "final_L2_norm_fv")
        improved = [
            row for row in best.values()
            if finite_float(row, "final_L2_norm_fv") < none_final
        ]
        if not improved:
            print(
                f"\nNo OT-tuned GLM row improved final_L2_norm_fv over no cleaning at nx={args.fine_resolution}."
            )
            print(
                "Suggested next step: stage-split GLM cleaning, such as applying the same GLM cleaning "
                "operator after each RK stage or a Strang-style cleaning split."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
