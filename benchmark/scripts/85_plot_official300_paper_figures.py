#!/usr/bin/env python3
"""Plot reproducible paper figures from generated official_300 paper metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_JSON = (
    ROOT
    / "benchmark"
    / "data"
    / "prototype_eval_results"
    / "official_300repo_release_unified_v1_paper_artifacts"
    / "paper_metrics.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "figures" / "generated"

SYSTEM_ORDER = ["BM25", "FAISS", "Mem0", "Graphiti", "Oracle Gold Context"]
SYSTEM_COLORS = {
    "BM25": "#4C78A8",
    "FAISS": "#F58518",
    "Mem0": "#54A24B",
    "Graphiti": "#E45756",
    "Oracle Gold Context": "#B279A2",
}
TASK_ORDER = [
    "single_state_lookup",
    "cross_version_comparison",
    "temporal_version_ordering",
]
TASK_TITLES = {
    "single_state_lookup": "Single-state lookup",
    "cross_version_comparison": "Cross-version comparison",
    "temporal_version_ordering": "Temporal ordering",
}
DECOUPLING_ORDER = [
    "all_recalled__correct",
    "all_recalled__incorrect",
    "partial_recalled__correct",
    "partial_recalled__incorrect",
    "zero_recalled__correct",
    "zero_recalled__incorrect",
]
DECOUPLING_LABELS = {
    "all_recalled__correct": "All + Correct",
    "all_recalled__incorrect": "All + Incorrect",
    "partial_recalled__correct": "Partial + Correct",
    "partial_recalled__incorrect": "Partial + Incorrect",
    "zero_recalled__correct": "Zero + Correct",
    "zero_recalled__incorrect": "Zero + Incorrect",
}
DECOUPLING_COLORS = {
    "all_recalled__correct": "#1B9E77",
    "all_recalled__incorrect": "#66A61E",
    "partial_recalled__correct": "#E6AB02",
    "partial_recalled__incorrect": "#D95F02",
    "zero_recalled__correct": "#7570B3",
    "zero_recalled__incorrect": "#E7298A",
}


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_system_entry(metrics: dict, system_name: str) -> dict:
    for item in metrics.get("systems") or []:
        if item.get("paper_name") == system_name:
            return item
    raise KeyError(f"Missing system entry for {system_name}")


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")


def compute_axis_limits(values: list[float], *, floor: float = 0.0, ceil: float = 1.0) -> tuple[float, float]:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return (floor, ceil)
    low = min(valid)
    high = max(valid)
    if math.isclose(low, high):
        pad = 0.05
    else:
        pad = max(0.03, 0.12 * (high - low))
    lower = max(floor, low - pad)
    upper = min(ceil, high + pad)
    if upper - lower < 0.12:
        center = (upper + lower) / 2
        lower = max(floor, center - 0.06)
        upper = min(ceil, center + 0.06)
    return (lower, upper)


def plot_task_topk_acc(metrics: dict, output_dir: Path) -> list[str]:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), sharex="col")
    for column, task in enumerate(TASK_ORDER):
        top_ks = metrics["task_top_ks"][task]
        acc_ax = axes[0][column]
        cov_ax = axes[1][column]
        all_acc_values: list[float] = []
        all_cov_values: list[float] = []
        for system_name in SYSTEM_ORDER:
            entry = get_system_entry(metrics, system_name)
            acc_values = [entry["task_topk_metrics"][task][str(k)]["acc"] for k in top_ks]
            cov_values = [entry["task_topk_metrics"][task][str(k)]["cov"] for k in top_ks]
            all_acc_values.extend(acc_values)
            all_cov_values.extend(cov_values)
            acc_ax.plot(
                top_ks,
                acc_values,
                marker="o",
                linewidth=2,
                markersize=5,
                color=SYSTEM_COLORS[system_name],
                label=system_name,
            )
            cov_ax.plot(
                top_ks,
                cov_values,
                marker="s",
                linewidth=2,
                markersize=4.5,
                linestyle="--",
                color=SYSTEM_COLORS[system_name],
                label=system_name,
            )
        acc_ax.set_title(TASK_TITLES[task], fontsize=11)
        acc_ax.set_xticks(top_ks)
        cov_ax.set_xticks(top_ks)
        acc_ax.set_ylim(*compute_axis_limits(all_acc_values))
        cov_ax.set_ylim(*compute_axis_limits(all_cov_values))
        acc_ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
        cov_ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
        cov_ax.set_xlabel("Retrieval top-k")
    axes[0][0].set_ylabel("ACC")
    axes[1][0].set_ylabel("COV")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(
        "ACC and COV by task family and retrieval top-k",
        y=1.04,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_dir / "official300_task_topk_acc")
    plt.close(fig)
    return [
        str((output_dir / "official300_task_topk_acc.png").resolve()),
        str((output_dir / "official300_task_topk_acc.svg").resolve()),
        str((output_dir / "official300_task_topk_acc.pdf").resolve()),
    ]


def plot_decoupling_maink(metrics: dict, output_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    x_positions = list(range(len(SYSTEM_ORDER)))
    bottoms = [0.0 for _ in SYSTEM_ORDER]

    for bucket in DECOUPLING_ORDER:
        heights = []
        for system_name in SYSTEM_ORDER:
            entry = get_system_entry(metrics, system_name)
            heights.append(entry["decoupling_maink"]["rates"][bucket])
        ax.bar(
            x_positions,
            heights,
            bottom=bottoms,
            color=DECOUPLING_COLORS[bucket],
            label=DECOUPLING_LABELS[bucket],
            width=0.72,
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_xticks(x_positions)
    ax.set_xticklabels(SYSTEM_ORDER, rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of questions")
    ax.set_title("Retrieval completeness and answer correctness at main retrieval top-k")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    save_figure(fig, output_dir / "official300_decoupling_maink")
    plt.close(fig)
    return [
        str((output_dir / "official300_decoupling_maink.png").resolve()),
        str((output_dir / "official300_decoupling_maink.svg").resolve()),
        str((output_dir / "official300_decoupling_maink.pdf").resolve()),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper figures from official_300 paper metrics.")
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    metrics = load_metrics(args.metrics_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "task_topk_acc": plot_task_topk_acc(metrics, args.output_dir),
        "decoupling_maink": plot_decoupling_maink(metrics, args.output_dir),
    }
    manifest_path = args.output_dir / "official300_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
