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
        f"{metric}_norm_fv",
        f"{metric}_fv",
        metric,
        f"{metric}_norm_centered",
        f"{metric}_centered",
        f"{metric}_divB",
        f"{metric}_divb",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError(
        f"No compatible {metric} column found. "
        f"Tried {candidates}. Columns = {list(df.columns)}"
    )


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

    metric_cols = []

    for file, label in zip(args.files, labels):
        df = pd.read_csv(file)
        time_col = pick_time_column(df)
        metric_col = pick_metric_column(df, args.metric)
        metric_cols.append(metric_col)

        print(f"{label}: using time={time_col}, metric={metric_col}")

        plt.plot(df[time_col], df[metric_col], label=label)

    selected_metrics = set(metric_cols)
    ylabel_metric = metric_cols[0] if len(selected_metrics) == 1 else args.metric

    plt.yscale("log")
    plt.xlabel("time")
    plt.ylabel(f"{ylabel_metric}(divB)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)

    print(f"Saved {out}")


if __name__ == "__main__":
    main()
