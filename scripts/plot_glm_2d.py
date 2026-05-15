#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def pick_time_column(df):
    if "time" in df.columns:
        return "time"
    if "t" in df.columns:
        return "t"
    raise KeyError(f"No time column found. Columns = {list(df.columns)}")


def pick_metric_column(df, metric, operator, normalized=False):
    """
    metric: L1, L2, Linf
    operator: fv, centered, auto
    normalized: if True, use L2_norm_fv etc.
    """

    candidates = []

    if operator != "auto":
        if normalized:
            candidates.append(f"{metric}_norm_{operator}")
        else:
            candidates.append(f"{metric}_{operator}")

    if operator == "auto":
        if normalized:
            candidates += [
                f"{metric}_norm_fv",
                f"{metric}_norm_centered",
                f"{metric}_norm",
            ]
        else:
            candidates += [
                f"{metric}_fv",
                f"{metric}_centered",
                metric,
                f"{metric}_divB",
                f"{metric}_divb",
            ]

    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError(
        f"No matching column found for metric={metric}, "
        f"operator={operator}, normalized={normalized}. "
        f"Columns = {list(df.columns)}"
    )


def pretty_label(path, label):
    if label is not None:
        return label

    name = Path(path).stem

    # Remove common prefix.
    name = name.replace("glm_2d_", "")
    name = name.replace("glm_1d_", "")

    return name


def main():
    parser = argparse.ArgumentParser(
        description="Plot GLM divergence-cleaning diagnostics."
    )

    parser.add_argument(
        "files",
        nargs="*",
        help="diagnostic CSV files. If omitted, use results/divergence/glm_2d_*.csv",
    )

    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="custom labels. Must match number of files.",
    )

    parser.add_argument(
        "--metric",
        default="L2",
        choices=["L1", "L2", "Linf"],
        help="which norm to plot",
    )

    parser.add_argument(
        "--operator",
        default="fv",
        choices=["fv", "centered", "auto"],
        help="which discrete divergence operator to plot",
    )

    parser.add_argument(
        "--normalized",
        action="store_true",
        help="plot normalized divergence norm",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="output figure path",
    )

    parser.add_argument(
        "--title",
        default=None,
        help="plot title",
    )

    args = parser.parse_args()

    files = args.files

    if len(files) == 0:
        files = sorted(str(p) for p in Path("results/divergence").glob("glm_2d_*.csv"))

    if len(files) == 0:
        raise FileNotFoundError("No input files found.")

    if args.labels is not None and len(args.labels) != len(files):
        raise ValueError("Number of labels must match number of files.")

    labels = args.labels

    if args.output is None:
        suffix = "norm" if args.normalized else "raw"
        args.output = (
            f"figures/glm_2d_{args.metric}_{args.operator}_{suffix}_compare.png"
        )

    plt.figure(figsize=(8, 5))

    for k, file in enumerate(files):
        df = pd.read_csv(file)

        time_col = pick_time_column(df)

        metric_col = pick_metric_column(
            df,
            metric=args.metric,
            operator=args.operator,
            normalized=args.normalized,
        )

        label = pretty_label(
            file,
            labels[k] if labels is not None else None,
        )

        plt.semilogy(
            df[time_col],
            df[metric_col],
            marker="o" if len(df) <= 5 else None,
            linewidth=1.8,
            label=label,
        )

    op_text = args.operator
    if args.operator == "fv":
        op_text = "finite-volume"
    elif args.operator == "centered":
        op_text = "centered"

    if args.normalized:
        ylabel = rf"${args.metric}$ norm of normalized {op_text} $\nabla\cdot\mathbf{{B}}$"
    else:
        ylabel = rf"${args.metric}$ norm of {op_text} $\nabla\cdot\mathbf{{B}}$"

    plt.xlabel("time")
    plt.ylabel(ylabel)

    if args.title is not None:
        plt.title(args.title)

    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=250)

    print(f"Saved {out}")


if __name__ == "__main__":
    main()