import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from label_utils import normalize_label_name


DEFAULT_METRICS = ["bdq", "bsq", "bpq", "aji"]


def load_metric_tables(csv_paths):
    import pandas as pd

    frames = []
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        if "experiment" not in df.columns:
            df["experiment"] = csv_path.stem.replace("_per_case", "")
        if "label" in df.columns:
            df["label"] = df["label"].map(normalize_label_name)
        frames.append(df)
    if not frames:
        raise ValueError("No metric CSV files were provided.")
    return pd.concat(frames, ignore_index=True)


def save_boxplots(df, metrics, output_dir):
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["group"] = df["experiment"].astype(str) + "\n" + df["label"].astype(str)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for ax, metric in zip(axes.flat, metrics):
        groups = []
        labels = []
        for group_name, group_df in df.groupby("group", sort=False):
            groups.append(group_df[metric].dropna().to_numpy())
            labels.append(group_name)
        if groups:
            ax.boxplot(groups, labels=labels, patch_artist=True)
        ax.set_title(f"{metric.upper()} Distribution")
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3, linestyle="--")
    fig.suptitle("Per-case Metric Distributions", fontsize=14)
    fig.savefig(output_dir / "metric_boxplots.png", dpi=220)
    plt.close(fig)


def save_heatmap(df, metrics, output_dir):
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    mean_df = (
        df.groupby(["experiment", "label"], as_index=False)[metrics]
        .mean(numeric_only=True)
        .sort_values(["experiment", "label"])
    )
    row_labels = [
        f"{row.experiment} | {row.label}" for row in mean_df.itertuples(index=False)
    ]
    values = mean_df[metrics].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, max(4, len(row_labels) * 0.45)))
    im = ax.imshow(values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(metrics)), labels=[m.upper() for m in metrics])
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_title("Mean Metric Heatmap")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "metric_heatmap.png", dpi=220)
    plt.close(fig)


def save_grouped_bars(df, metrics, output_dir):
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    mean_df = (
        df.groupby(["experiment", "label"], as_index=False)[metrics]
        .mean(numeric_only=True)
        .sort_values(["label", "experiment"])
    )
    labels = list(mean_df["label"].drop_duplicates())

    fig, axes = plt.subplots(
        len(labels),
        1,
        figsize=(13, 4 * max(1, len(labels))),
        constrained_layout=True,
    )
    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        label_df = mean_df[mean_df["label"] == label].reset_index(drop=True)
        x = np.arange(len(label_df))
        width = 0.8 / max(1, len(metrics))
        for idx, metric in enumerate(metrics):
            ax.bar(
                x + (idx - (len(metrics) - 1) / 2) * width,
                label_df[metric].to_numpy(dtype=float),
                width=width,
                label=metric.upper(),
            )
        ax.set_xticks(x, labels=label_df["experiment"].tolist())
        ax.set_title(f"Mean Metrics for {label}")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, linestyle="--", axis="y")
        ax.legend()
    fig.savefig(output_dir / "metric_grouped_bars.png", dpi=220)
    plt.close(fig)


def save_ratio_chart(df, metrics, output_dir):
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    mean_df = (
        df.groupby(["experiment", "label"], as_index=False)[metrics]
        .mean(numeric_only=True)
        .sort_values(["experiment", "label"])
    )

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(12, 3.8 * max(1, len(metrics))),
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        pivot = (
            mean_df.pivot(index="experiment", columns="label", values=metric)
            .fillna(0.0)
            .sort_index()
        )
        values = np.clip(pivot.to_numpy(dtype=float), a_min=0.0, a_max=None)
        denom = np.clip(values.sum(axis=1, keepdims=True), a_min=1e-8, a_max=None)
        ratios = values / denom
        bottom = np.zeros(len(pivot))
        x = np.arange(len(pivot))
        for label_idx, label_name in enumerate(pivot.columns):
            ax.bar(x, ratios[:, label_idx], bottom=bottom, label=label_name)
            bottom += ratios[:, label_idx]
        ax.set_xticks(x, labels=pivot.index.tolist())
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Share")
        ax.set_title(f"{metric.upper()} Structure Ratio")
        ax.grid(alpha=0.3, linestyle="--", axis="y")

    axes[0].legend()
    fig.savefig(output_dir / "metric_ratio_chart.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        action="append",
        required=True,
        help="path to a per-case evaluation CSV exported by infer.py; can be passed multiple times",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/figures",
        help="directory where plots will be saved",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="metric columns to visualize",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    df = load_metric_tables(args.csv)
    save_boxplots(df, args.metrics, output_dir)
    save_heatmap(df, args.metrics, output_dir)
    save_grouped_bars(df, args.metrics, output_dir)
    save_ratio_chart(df, args.metrics, output_dir)


if __name__ == "__main__":
    main()
