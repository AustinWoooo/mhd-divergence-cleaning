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


def pick_metric_column(df, metric):
    candidates = [
        metric,
        f"{metric}_divB",
        f"{metric}_divb",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError(f"No {metric} column found. Columns = {list(df.columns)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="diagnostics csv files")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--metric", default="L2", choices=["L1", "L2", "Linf"])
    parser.add_argument("--output", default="figures/compare_cleaning.png")
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.files):
        raise ValueError("Number of labels must match number of files.")

    labels = args.labels
    if labels is None:
        labels = [Path(f).stem for f in args.files]

    plt.figure(figsize=(7, 5))

    for file, label in zip(args.files, labels):
        df = pd.read_csv(file)
        time_col = pick_time_column(df)
        metric_col = pick_metric_column(df, args.metric)

        plt.plot(df[time_col], df[metric_col], label=label)

    plt.yscale("log")
    plt.xlabel("time")
    plt.ylabel(rf"${args.metric}(\nabla\cdot\mathbf{{B}})$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)

    print(f"Saved {out}")


if __name__ == "__main__":
    main()

