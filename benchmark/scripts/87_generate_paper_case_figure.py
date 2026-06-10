#!/usr/bin/env python3
"""Generate an illustrative benchmark-instance figure for the paper."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "paper" / "figures" / "generated"
OUTPUT_STEM = OUTPUT_DIR / "rhaphire_win11debloat_case"


MEMORY_CARDS = [
    ("1", "2025.08.01", "adds an option to disable AI features in Paint & Notepad", "#2F6FDB"),
    ("2", "2025.08.16", "adds an option to disable telemetry, ads, MSN news feed, and AI in Microsoft Edge", "#31A354"),
    ("3", "2026.02.01", "introduces a full graphical user interface (GUI)", "#FF7F0E"),
    ("4", "2026.02.04", "adds an option to disable BitLocker automatic device encryption", "#4C78A8"),
    ("5", "2026.02.12", "expands DisableEdgeAds to disable additional Microsoft Edge clutter and ads", "#2CA25F"),
    ("6", "2026.05.10", "adds System Registry backup & restore and Start Menu layout backup & restore", "#8A60D1"),
]

QUESTION_BOXES = [
    (
        "1) Single-State Lookup",
        "Which release adds an option to disable AI features in Paint & Notepad?",
        ["1"],
        "#2F6FDB",
    ),
    (
        "2) Cross-Version Comparison",
        "How does the Microsoft Edge change in 2025.08.16 differ from 2026.02.12?",
        ["2", "5"],
        "#31A354",
    ),
    (
        "3) Temporal Ordering",
        "Order these states: Paint/Notepad AI disable; full GUI; BitLocker device encryption disable; backup & restore.",
        ["1", "3", "4", "6"],
        "#FF7F0E",
    ),
]


def save_figure(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")


def add_round_box(ax, xy, width, height, *, edgecolor, facecolor="white", radius=0.02, linewidth=1.8):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    return patch


def add_memory_card(ax, x, y, label, date_text, body_text, color):
    card_w, card_h = 0.30, 0.105
    add_round_box(ax, (x, y), card_w, card_h, edgecolor=color, facecolor="white", radius=0.016, linewidth=1.2)
    badge = Circle((x + 0.024, y + card_h - 0.028), 0.019, facecolor=color, edgecolor=color)
    ax.add_patch(badge)
    ax.text(x + 0.024, y + card_h - 0.028, label, ha="center", va="center", fontsize=12, color="white", fontweight="bold")
    add_round_box(ax, (x + 0.060, y + card_h - 0.043), 0.10, 0.028, edgecolor=color, facecolor=color, radius=0.007, linewidth=0)
    ax.text(x + 0.110, y + card_h - 0.029, date_text, ha="center", va="center", fontsize=11.5, color="white", fontweight="bold")
    ax.text(
        x + 0.060,
        y + card_h - 0.055,
        fill(body_text, width=33),
        ha="left",
        va="top",
        fontsize=11,
        color="#222222",
        linespacing=1.28,
    )


def add_question_box(ax, x, y, title, query, gold_ids, color):
    box_w, box_h = 0.285, 0.205
    add_round_box(ax, (x, y), box_w, box_h, edgecolor=color, facecolor="white", radius=0.02, linewidth=1.5)
    ax.text(x + 0.022, y + box_h - 0.045, title, ha="left", va="center", fontsize=13.5, color=color, fontweight="bold")
    ax.plot([x + 0.02, x + box_w - 0.02], [y + box_h - 0.072, y + box_h - 0.072], color=color, linewidth=1.2, linestyle=(0, (1.5, 2.5)))
    ax.text(x + 0.02, y + box_h - 0.105, "Query:", ha="left", va="center", fontsize=11.5, color=color, fontweight="bold")
    ax.text(
        x + 0.02,
        y + box_h - 0.122,
        fill(query, width=36),
        ha="left",
        va="top",
        fontsize=10.8,
        color="#222222",
        linespacing=1.25,
    )
    ax.plot([x + 0.02, x + box_w - 0.02], [y + 0.052, y + 0.052], color="#C9D2DF", linewidth=0.8)
    ax.text(x + 0.02, y + 0.028, "Gold memory:", ha="left", va="center", fontsize=11.5, color=color, fontweight="bold")
    pill_x = x + 0.145
    for gold_id in gold_ids:
        add_round_box(ax, (pill_x, y + 0.012), 0.022, 0.027, edgecolor=color, facecolor="white", radius=0.006, linewidth=1.1)
        ax.text(pill_x + 0.011, y + 0.026, gold_id, ha="center", va="center", fontsize=11.2, color=color, fontweight="bold")
        pill_x += 0.031


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(14.8, 8.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.5, 0.955, "Illustrative TSSMRBench Instance", ha="center", va="center", fontsize=28, fontweight="bold", color="#10284B")
    ax.text(0.5, 0.908, "Repository: Raphire/Win11Debloat", ha="center", va="center", fontsize=17, fontweight="bold", color="#1D5AA6")

    timeline_x = 0.50
    ax.plot([timeline_x, timeline_x], [0.17, 0.85], color="#424242", linewidth=2.5, zorder=1)
    ax.add_patch(FancyArrowPatch((timeline_x, 0.85), (timeline_x, 0.88), arrowstyle="-|>", mutation_scale=18, linewidth=1.4, color="#424242"))
    ax.add_patch(FancyArrowPatch((timeline_x, 0.17), (timeline_x, 0.14), arrowstyle="-|>", mutation_scale=18, linewidth=1.4, color="#424242"))
    ax.text(timeline_x - 0.05, 0.86, "Earlier", ha="center", va="center", fontsize=14, color="#555555")
    ax.text(timeline_x - 0.045, 0.15, "Later", ha="center", va="center", fontsize=14, color="#555555")

    y_positions = [0.80, 0.70, 0.54, 0.41, 0.29, 0.16]
    card_x = 0.03
    question_x = 0.68
    question_y = [0.62, 0.37, 0.11]

    for (label, date_text, body_text, color), y in zip(MEMORY_CARDS, y_positions):
        add_memory_card(ax, card_x, y, label, date_text, body_text, color)
        ax.text(timeline_x - 0.055, y + 0.045, date_text, ha="right", va="center", fontsize=12, color="#222222")
        node = Circle((timeline_x, y + 0.045), 0.018, facecolor="white", edgecolor=color, linewidth=1.5, zorder=3)
        ax.add_patch(node)
        ax.text(timeline_x, y + 0.045, label, ha="center", va="center", fontsize=11.5, color=color, fontweight="bold", zorder=4)

    ax.text(timeline_x - 0.07, 0.625, "⋮", ha="center", va="center", fontsize=30, color="#8A8A8A")
    ax.text(timeline_x - 0.07, 0.235, "⋮", ha="center", va="center", fontsize=30, color="#8A8A8A")

    for (title, query, gold_ids, color), y in zip(QUESTION_BOXES, question_y):
        add_question_box(ax, question_x, y, title, query, gold_ids, color)

    arrow_specs = [
        ((0.66, 0.72), (0.58, 0.72), "#2F6FDB"),
        ((0.66, 0.49), (0.58, 0.49), "#31A354"),
        ((0.66, 0.27), (0.58, 0.27), "#FF7F0E"),
    ]
    for start, end, color in arrow_specs:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=22, linewidth=1.5, color=color))

    add_round_box(ax, (0.20, 0.03), 0.58, 0.048, edgecolor="#D7DDE7", facecolor="#FBFCFE", radius=0.012, linewidth=1.0)
    ax.text(
        0.49,
        0.054,
        "Only task-relevant memory states are shown; omitted releases are collapsed with ellipses.",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#3A4A5D",
    )

    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
