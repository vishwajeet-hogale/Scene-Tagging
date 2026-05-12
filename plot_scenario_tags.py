#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze scenario tag distribution across all JSON files and generate plots.

Outputs:
- Console summary
- analysis/scenario_tag_analysis/tag_counts.csv
- analysis/scenario_tag_analysis/num_tags_per_file.csv
- analysis/scenario_tag_analysis/summary.json
- analysis/scenario_tag_analysis/plots/tag_counts_bar.png
- analysis/scenario_tag_analysis/plots/tags_per_file_bar.png
- analysis/scenario_tag_analysis/plots/top_combinations_bar.png
- analysis/scenario_tag_analysis/plots/tag_family_breakdown.png
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    plt = None


# ---------------------------------------------------------------------------
# Tag family definitions (keep in sync with add_scenario_tags.py)
# ---------------------------------------------------------------------------

TAG_FAMILIES: Dict[str, List[str]] = {
    # Curvature is a lane-segment-level property, not a frame-level tag.
    # It lives in scenario_meta.lane_segments, not scenario_tags.
    "topology": [
        "low topological complexity",
        "medium topological complexity",
        "high topological complexity",
        "topology_unknown",
    ],
    "lighting": [
        "well lit",
        "poorly lit",
        "lighting_unknown",
    ],
    "occlusion": [
        "no occlusion",
        "low occlusion",
        "high occlusion",
        "occlusion_unknown",
    ],
}

TAG_COLORS: Dict[str, str] = {
    # curvature
    "straight": "#2ca02c",
    "straight with an angle": "#f0a000",
    "curve left": "#1f77b4",
    "curve right": "#d62728",
    "sharp left": "#0a2d5a",
    "sharp right": "#5c0505",
    "curvature_unknown": "#888888",
    # topology
    "low topological complexity": "#9ecae1",
    "medium topological complexity": "#6baed6",
    "high topological complexity": "#08519c",
    "topology_unknown": "#bbbbbb",
    # lighting
    "well lit": "#fee08b",
    "poorly lit": "#4575b4",
    "lighting_unknown": "#aaaaaa",
    # occlusion
    "no occlusion": "#c7e9c0",
    "low occlusion": "#74c476",
    "high occlusion": "#006d2c",
    "occlusion_unknown": "#999999",
}

FAMILY_COLORS: Dict[str, str] = {
    "curvature": "#e07b39",
    "topology": "#4e79a7",
    "lighting": "#f28e2b",
    "occlusion": "#59a14f",
}

# ---------------------------------------------------------------------------
# Lane-segment level tag definitions
# ---------------------------------------------------------------------------

LANE_SEG_TAG_FAMILIES: Dict[str, List[str]] = {
    "curvature_tag": [
        "straight",
        "straight with an angle",
        "curve left",
        "curve right",
        "sharp left",
        "sharp right",
        "curvature_unknown",
    ],
    "slope_tag": [
        "straight",
        "straight with an angle",
        "curve left",
        "curve right",
        "sharp left",
        "sharp right",
        "curvature_unknown",
    ],
    "kappa_tag": [
        "straight",
        "straight with an angle",
        "curve left",
        "curve right",
        "sharp left",
        "sharp right",
        "curvature_unknown",
    ],
}


def _resolve_family(tag: str) -> Optional[str]:
    for fam, members in TAG_FAMILIES.items():
        if tag in members:
            return fam
    return None


# ---------------------------------------------------------------------------
# File discovery & parsing
# ---------------------------------------------------------------------------


def analyze_lane_segments(files: List[Path]) -> Dict[str, Any]:
    """Collect lane-segment-level statistics from all -ls.json files."""
    curvature_counts: Counter = Counter()
    slope_counts: Counter = Counter()
    kappa_counts: Counter = Counter()
    segments_per_file: Counter = Counter()
    intersection_counts: Counter = Counter()
    length_values: List[float] = []
    heading_values: List[float] = []
    kappa_values: List[float] = []  # value_m_inv (signed curvature)

    total_segments = 0
    files_with_lane_segs = 0

    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        lane_segs = (data.get("scenario_meta") or {}).get("lane_segments")
        if not isinstance(lane_segs, list):
            segments_per_file[0] += 1
            continue

        files_with_lane_segs += 1
        segments_per_file[len(lane_segs)] += 1
        total_segments += len(lane_segs)

        for seg in lane_segs:
            if not isinstance(seg, dict):
                continue

            ct = normalize_tag(seg.get("curvature_tag") or "curvature_unknown")
            st = normalize_tag(seg.get("slope_tag") or "curvature_unknown")
            kt = normalize_tag(seg.get("kappa_tag") or "curvature_unknown")
            curvature_counts[ct] += 1
            slope_counts[st] += 1
            kappa_counts[kt] += 1

            is_int = seg.get("is_intersection_or_connector", False)
            intersection_counts["intersection/connector" if is_int else "road"] += 1

            length = seg.get("length_m")
            if isinstance(length, (int, float)) and np.isfinite(length):
                length_values.append(float(length))

            heading = seg.get("heading_deg")
            if isinstance(heading, (int, float)) and np.isfinite(heading):
                heading_values.append(float(heading))

            kappa = seg.get("value_m_inv")
            if isinstance(kappa, (int, float)) and np.isfinite(kappa):
                kappa_values.append(float(kappa))

    return {
        "total_segments": total_segments,
        "files_with_lane_segs": files_with_lane_segs,
        "curvature_tag_counts": dict(
            sorted(curvature_counts.items(), key=lambda kv: -kv[1])
        ),
        "slope_tag_counts": dict(
            sorted(slope_counts.items(), key=lambda kv: -kv[1])
        ),
        "kappa_tag_counts": dict(
            sorted(kappa_counts.items(), key=lambda kv: -kv[1])
        ),
        "segments_per_file": dict(
            sorted(segments_per_file.items(), key=lambda kv: kv[0])
        ),
        "intersection_counts": dict(intersection_counts),
        "length_m_values": length_values,
        "heading_deg_values": heading_values,
        "kappa_m_inv_values": kappa_values,
    }

def find_json_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*-ls.json") if p.is_file())


def normalize_tag(tag: str) -> str:
    return " ".join(tag.strip().split()).lower()


def analyze_files(files: List[Path]) -> Dict[str, Any]:
    tag_counts: Counter = Counter()
    tags_per_file: Counter = Counter()
    combo_counts: Counter = Counter()
    family_counts: Dict[str, Counter] = {fam: Counter() for fam in TAG_FAMILIES}

    files_with_scenario_tags = 0
    files_without_scenario_tags = 0
    files_with_empty_scenario_tags = 0
    files_parsed = 0
    files_failed = 0

    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            files_failed += 1
            continue

        if not isinstance(data, dict):
            files_failed += 1
            continue

        files_parsed += 1
        raw_tags = data.get("scenario_tags")

        if raw_tags is None:
            files_without_scenario_tags += 1
            tags_per_file[0] += 1
            continue

        if not isinstance(raw_tags, list):
            files_failed += 1
            continue

        files_with_scenario_tags += 1
        normalized_tags = sorted({
            normalize_tag(t)
            for t in raw_tags
            if isinstance(t, str) and t.strip()
        })

        if not normalized_tags:
            files_with_empty_scenario_tags += 1

        tags_per_file[len(normalized_tags)] += 1
        for t in normalized_tags:
            tag_counts[t] += 1
            fam = _resolve_family(t)
            if fam:
                family_counts[fam][t] += 1

        combo_key = " | ".join(normalized_tags) if normalized_tags else "<empty>"
        combo_counts[combo_key] += 1

    return {
        "files_total": len(files),
        "files_parsed": files_parsed,
        "files_failed": files_failed,
        "files_with_scenario_tags": files_with_scenario_tags,
        "files_without_scenario_tags": files_without_scenario_tags,
        "files_with_empty_scenario_tags": files_with_empty_scenario_tags,
        "unique_tags": len(tag_counts),
        "tag_counts": dict(sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "tags_per_file": dict(sorted(tags_per_file.items(), key=lambda kv: kv[0])),
        "family_counts": {
            fam: dict(sorted(fc.items(), key=lambda kv: -kv[1]))
            for fam, fc in family_counts.items()
        },
        "top_tag_combinations": [
            {"combination": k, "count": v}
            for k, v in combo_counts.most_common(25)
        ],
    }


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_tag_counts_csv(tag_counts: Dict[str, int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_tag", "datapoint_count"])
        for tag, count in tag_counts.items():
            writer.writerow([tag, count])


def write_tags_per_file_csv(tags_per_file: Dict[int, int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_tags_in_file", "file_count"])
        for n_tags, file_count in tags_per_file.items():
            writer.writerow([n_tags, file_count])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _bar_chart(
    labels: List[str],
    values: List[float],
    colors: List[str],
    out_path: Path,
    title: str,
    xlabel: str,
    total: Optional[int] = None,
    dpi: int = 120,
) -> None:
    """Horizontal bar chart with count and percentage annotations."""
    if not labels:
        return
    n = len(labels)
    fig_h = max(3.5, 0.45 * n + 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=dpi)

    y = np.arange(n)
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    denom = total if total else (max(values) or 1)
    x_max = max(values) * 1.18 if values else 1
    ax.set_xlim(0, x_max)
    for bar, v in zip(bars, values):
        pct = 100.0 * v / denom if denom else 0
        ax.text(
            bar.get_width() + x_max * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(v):,}  ({pct:.1f}%)",
            va="center", ha="left", fontsize=8, color="#333333",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def plot_tag_counts_bar(tag_counts: Dict[str, int], out_path: Path, top_n: int = 40) -> None:
    """Bar chart of the top-N most frequent scenario tags."""
    items = list(tag_counts.items())[:top_n]
    if not items:
        return
    labels, values = zip(*items)
    colors = [TAG_COLORS.get(lbl, "#5b8db8") for lbl in labels]
    total = sum(tag_counts.values())
    _bar_chart(
        list(labels), [float(v) for v in values], colors, out_path,
        title=f"Scenario tag counts  (top {min(top_n, len(items))} of {len(tag_counts)} unique tags)",
        xlabel="Count (frames)",
        total=total,
    )


def plot_tags_per_file_bar(tags_per_file: Dict[int, int], out_path: Path) -> None:
    """Bar chart showing how many files have 0, 1, 2 … tags."""
    if not tags_per_file:
        return
    keys = sorted(tags_per_file.keys())
    values = [tags_per_file[k] for k in keys]
    labels = [str(k) for k in keys]
    total = sum(values)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.bar(keys, values, color="#4e79a7", edgecolor="white", linewidth=0.5)
    ax.set_xticks(keys)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of scenario tags per file", fontsize=9)
    ax.set_ylabel("File count", fontsize=9)
    ax.set_title("Distribution of tag count per file", fontsize=11, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    for x, v in zip(keys, values):
        pct = 100.0 * v / total if total else 0
        ax.text(x, v + total * 0.005, f"{v:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=8, color="#333333")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def plot_top_combinations_bar(
    combos: List[Dict[str, Any]], out_path: Path, top_n: int = 20
) -> None:
    """Horizontal bar chart of the most common tag combinations."""
    combos = combos[:top_n]
    if not combos:
        return
    labels = [c["combination"] for c in combos]
    values = [float(c["count"]) for c in combos]
    # Shorten long combination strings
    display_labels = [
        (lbl if len(lbl) <= 70 else lbl[:67] + "…") for lbl in labels
    ]
    total = sum(values)
    colors = ["#4e79a7"] * len(values)
    _bar_chart(
        display_labels, values, colors, out_path,
        title=f"Top {min(top_n, len(combos))} tag combinations by frequency",
        xlabel="Count (frames)",
        total=int(total),
    )


def plot_tag_family_breakdown(
    family_counts: Dict[str, Dict[str, int]],
    out_path: Path,
) -> None:
    """One subplot per tag family, showing tag distribution within each family."""
    families = [fam for fam in TAG_FAMILIES if family_counts.get(fam)]
    if not families:
        return

    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), dpi=120)
    if n == 1:
        axes = [axes]

    for ax, fam in zip(axes, families):
        # Use canonical order from TAG_FAMILIES, filtered to what's present
        canonical = [t for t in TAG_FAMILIES[fam] if t in family_counts[fam]]
        # Add any extra tags not in the canonical list
        for t in family_counts[fam]:
            if t not in canonical:
                canonical.append(t)

        values = [family_counts[fam].get(t, 0) for t in canonical]
        colors = [TAG_COLORS.get(t, FAMILY_COLORS.get(fam, "#888888")) for t in canonical]
        total = sum(values) or 1

        y = np.arange(len(canonical))
        bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [t.replace(" ", "\n", 1) if len(t) > 14 else t for t in canonical],
            fontsize=8,
        )
        ax.invert_yaxis()
        ax.set_title(fam.capitalize(), fontsize=10, fontweight="bold",
                     color=FAMILY_COLORS.get(fam, "#333"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Frames", fontsize=8)
        x_max = max(values) * 1.25 if values else 1
        ax.set_xlim(0, x_max)
        for bar, v in zip(bars, values):
            pct = 100.0 * v / total
            ax.text(
                bar.get_width() + x_max * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{v:,} ({pct:.0f}%)",
                va="center", ha="left", fontsize=7.5, color="#333",
            )

    fig.suptitle("Tag distribution by family", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def plot_tag_family_pie_grid(
    family_counts: Dict[str, Dict[str, int]],
    out_path: Path,
) -> None:
    """2x2 grid of pie charts, one per tag family."""
    families = [fam for fam in TAG_FAMILIES if family_counts.get(fam)]
    if not families:
        return

    ncols = 2
    nrows = (len(families) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), dpi=120)
    axes_flat = np.array(axes).flatten()

    for ax, fam in zip(axes_flat, families):
        canonical = [t for t in TAG_FAMILIES[fam] if t in family_counts[fam]]
        for t in family_counts[fam]:
            if t not in canonical:
                canonical.append(t)
        values = [family_counts[fam].get(t, 0) for t in canonical]
        colors = [TAG_COLORS.get(t, "#888888") for t in canonical]
        total = sum(values)

        wedge_props = {"edgecolor": "white", "linewidth": 1.2}
        wedges, texts, autotexts = ax.pie(
            values,
            labels=None,
            colors=colors,
            autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
            startangle=90,
            wedgeprops=wedge_props,
            pctdistance=0.75,
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_color("white")
            at.set_fontweight("bold")

        legend_labels = [
            f"{t}  ({family_counts[fam].get(t, 0):,})" for t in canonical
        ]
        ax.legend(wedges, legend_labels, loc="lower center",
                  bbox_to_anchor=(0.5, -0.18), fontsize=7.5, ncol=1,
                  frameon=False)
        ax.set_title(
            f"{fam.capitalize()}  (n={total:,})",
            fontsize=10, fontweight="bold",
            color=FAMILY_COLORS.get(fam, "#333"),
        )

    # Hide any unused axes
    for ax in axes_flat[len(families):]:
        ax.set_visible(False)

    fig.suptitle("Scenario tag distribution — per family", fontsize=13,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Lane-segment level plots
# ---------------------------------------------------------------------------

_CURVATURE_ORDER = [
    "straight",
    "curve left",
    "curve right",
    "sharp left",
    "sharp right",
]


def _ordered_items(
    counts: Dict[str, int], order: List[str]
) -> Tuple[List[str], List[int]]:
    """Return (labels, values) for tags listed in order (others are excluded)."""
    labels, values = [], []
    for tag in order:
        if tag in counts:
            labels.append(tag)
            values.append(counts[tag])
    return labels, values


def plot_lane_seg_tag_bars(lane_summary: Dict[str, Any], out_path: Path) -> None:
    """Three horizontal bar charts: curvature_tag, slope_tag, kappa_tag."""
    families = [
        ("curvature_tag", lane_summary["curvature_tag_counts"], "Curvature tag"),
        ("slope_tag",     lane_summary["slope_tag_counts"],     "Slope tag"),
        ("kappa_tag",     lane_summary["kappa_tag_counts"],     "Kappa tag"),
    ]
    families = [(k, d, t) for k, d, t in families if d]
    if not families:
        return

    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), dpi=120)
    if n == 1:
        axes = [axes]

    for ax, (key, counts, title) in zip(axes, families):
        labels, values = _ordered_items(counts, _CURVATURE_ORDER)
        colors = [TAG_COLORS.get(lbl, "#888888") for lbl in labels]
        total = sum(values) or 1
        y = np.arange(len(labels))
        bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Lane segments", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        x_max = max(values) * 1.28 if values else 1
        ax.set_xlim(0, x_max)
        for bar, v in zip(bars, values):
            pct = 100.0 * v / total
            ax.text(
                bar.get_width() + x_max * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{v:,} ({pct:.1f}%)",
                va="center", ha="left", fontsize=7.5, color="#333",
            )

    fig.suptitle(
        f"Lane-segment tag distributions  (n={lane_summary['total_segments']:,} segments)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def plot_lane_seg_curvature_pies(lane_summary: Dict[str, Any], out_path: Path) -> None:
    """Three pie charts for curvature_tag, slope_tag, kappa_tag side by side."""
    families = [
        ("curvature_tag", lane_summary["curvature_tag_counts"], "Curvature tag"),
        ("slope_tag",     lane_summary["slope_tag_counts"],     "Slope tag"),
        ("kappa_tag",     lane_summary["kappa_tag_counts"],     "Kappa tag"),
    ]
    families = [(k, d, t) for k, d, t in families if d]
    if not families:
        return

    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), dpi=120)
    if n == 1:
        axes = [axes]

    for ax, (key, counts, title) in zip(axes, families):
        labels, values = _ordered_items(counts, _CURVATURE_ORDER)
        colors = [TAG_COLORS.get(lbl, "#888888") for lbl in labels]
        total = sum(values)
        wedge_props = {"edgecolor": "white", "linewidth": 1.2}
        wedges, _, autotexts = ax.pie(
            values,
            labels=None,
            colors=colors,
            autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
            startangle=90,
            wedgeprops=wedge_props,
            pctdistance=0.75,
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_color("white")
            at.set_fontweight("bold")
        legend_labels = [f"{lbl}  ({v:,})" for lbl, v in zip(labels, values)]
        ax.legend(wedges, legend_labels, loc="lower center",
                  bbox_to_anchor=(0.5, -0.22), fontsize=7.5, ncol=1, frameon=False)
        ax.set_title(f"{title}  (n={total:,})", fontsize=10, fontweight="bold")

    fig.suptitle("Lane-segment tag pies", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def plot_segments_per_file(lane_summary: Dict[str, Any], out_path: Path) -> None:
    """Histogram of lane-segment count per frame."""
    spf = lane_summary["segments_per_file"]
    if not spf:
        return
    keys = sorted(spf.keys())
    values = [spf[k] for k in keys]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    ax.bar(keys, values, color="#4e79a7", edgecolor="white", linewidth=0.5, width=0.8)
    ax.set_xlabel("Lane segments per frame", fontsize=9)
    ax.set_ylabel("Frame count", fontsize=9)
    ax.set_title(
        f"Distribution of lane-segment count per frame  (total frames={total:,})",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    # Only annotate if there aren't too many bars
    if len(keys) <= 60:
        for x, v in zip(keys, values):
            if v > 0:
                ax.text(x, v, f"{v}", ha="center", va="bottom", fontsize=6, color="#333")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def plot_intersection_bar(lane_summary: Dict[str, Any], out_path: Path) -> None:
    """Bar chart: intersection/connector vs road segments."""
    counts = lane_summary["intersection_counts"]
    if not counts:
        return
    order = ["road", "intersection/connector"]
    labels = [k for k in order if k in counts] + [k for k in counts if k not in order]
    values = [counts[k] for k in labels]
    colors = ["#4e79a7", "#f28e2b"][:len(labels)]
    total = sum(values) or 1

    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=120)
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Lane segments", fontsize=9)
    ax.set_title("Road vs intersection/connector segments", fontsize=11,
                 fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    x_max = max(values) * 1.25 if values else 1
    ax.set_xlim(0, x_max)
    for bar, v in zip(bars, values):
        pct = 100.0 * v / total
        ax.text(
            bar.get_width() + x_max * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{v:,} ({pct:.1f}%)",
            va="center", ha="left", fontsize=8.5, color="#333",
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def plot_lane_seg_length_histogram(lane_summary: Dict[str, Any], out_path: Path) -> None:
    """Histogram of lane-segment lengths in metres."""
    vals = lane_summary["length_m_values"]
    if not vals:
        return
    arr = np.array(vals, dtype=float)
    # Clip to 99th percentile for readability
    p99 = float(np.percentile(arr, 99))
    arr_clipped = arr[arr <= p99]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.hist(arr_clipped, bins=60, color="#59a14f", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Length (m)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(
        f"Lane-segment length distribution  (n={len(arr):,}, clipped at p99={p99:.1f} m)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(float(np.median(arr)), color="#e07b39", linewidth=1.2,
               linestyle="--", label=f"median={np.median(arr):.1f} m")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def plot_lane_seg_heading_histogram(lane_summary: Dict[str, Any], out_path: Path) -> None:
    """Polar + linear histograms of lane-segment heading angles."""
    vals = lane_summary["heading_deg_values"]
    if not vals:
        return
    arr = np.deg2rad(np.array(vals, dtype=float))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=120,
                             subplot_kw={"projection": None})
    # Linear histogram
    ax_lin = axes[0]
    ax_lin.hist(np.degrees(arr), bins=72, range=(-180, 180),
                color="#4e79a7", edgecolor="white", linewidth=0.3)
    ax_lin.set_xlabel("Heading (degrees)", fontsize=9)
    ax_lin.set_ylabel("Count", fontsize=9)
    ax_lin.set_title("Heading distribution (linear)", fontsize=10, fontweight="bold")
    ax_lin.set_xticks(range(-180, 181, 45))
    ax_lin.spines[["top", "right"]].set_visible(False)

    # Polar histogram
    fig2, ax_pol = plt.subplots(figsize=(5, 5), dpi=120,
                                subplot_kw={"projection": "polar"})
    bin_edges = np.linspace(-np.pi, np.pi, 37)
    counts, _ = np.histogram(arr, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = bin_edges[1] - bin_edges[0]
    ax_pol.bar(bin_centers, counts, width=width, color="#4e79a7",
               edgecolor="white", linewidth=0.3, alpha=0.85)
    ax_pol.set_theta_zero_location("N")
    ax_pol.set_theta_direction(-1)
    ax_pol.set_title(f"Heading (polar, n={len(vals):,})", fontsize=10,
                     fontweight="bold", pad=15)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)

    polar_path = out_path.parent / (out_path.stem + "_polar.png")
    fig2.tight_layout()
    fig2.savefig(str(polar_path), bbox_inches="tight")
    plt.close(fig2)


def plot_lane_seg_kappa_histogram(lane_summary: Dict[str, Any], out_path: Path) -> None:
    """Histogram of signed curvature values (value_m_inv)."""
    vals = lane_summary["kappa_m_inv_values"]
    if not vals:
        return
    arr = np.array(vals, dtype=float)
    p1, p99 = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
    arr_clipped = arr[(arr >= p1) & (arr <= p99)]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.hist(arr_clipped, bins=80, color="#e07b39", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Curvature κ (m⁻¹)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(
        f"Lane-segment curvature distribution  (n={len(arr):,}, clipped to [p1, p99])",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.axvline(0, color="#333", linewidth=0.8, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def generate_lane_seg_plots(lane_summary: Dict[str, Any], plot_dir: Path) -> None:
    if plt is None:
        print("[WARN] matplotlib not available — skipping lane-segment plots.")
        return
    plot_dir.mkdir(parents=True, exist_ok=True)

    p = plot_dir / "lane_seg_tag_bars.png"
    plot_lane_seg_tag_bars(lane_summary, p)
    print(f"[WROTE] {p}")

    p = plot_dir / "lane_seg_tag_pies.png"
    plot_lane_seg_curvature_pies(lane_summary, p)
    print(f"[WROTE] {p}")

    p = plot_dir / "lane_seg_segments_per_file.png"
    plot_segments_per_file(lane_summary, p)
    print(f"[WROTE] {p}")

    p = plot_dir / "lane_seg_intersection_bar.png"
    plot_intersection_bar(lane_summary, p)
    print(f"[WROTE] {p}")

    p = plot_dir / "lane_seg_length_hist.png"
    plot_lane_seg_length_histogram(lane_summary, p)
    print(f"[WROTE] {p}")

    p = plot_dir / "lane_seg_heading_hist.png"
    plot_lane_seg_heading_histogram(lane_summary, p)
    print(f"[WROTE] {p}")

    p = plot_dir / "lane_seg_kappa_hist.png"
    plot_lane_seg_kappa_histogram(lane_summary, p)
    print(f"[WROTE] {p}")


def generate_all_plots(summary: Dict[str, Any], plot_dir: Path) -> None:
    if plt is None:
        print("[WARN] matplotlib not available — skipping all plots.")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    p = plot_dir / "tag_counts_bar.png"
    plot_tag_counts_bar(summary["tag_counts"], p)
    print(f"[WROTE] {p}")

    p = plot_dir / "tags_per_file_bar.png"
    plot_tags_per_file_bar(summary["tags_per_file"], p)
    print(f"[WROTE] {p}")

    p = plot_dir / "top_combinations_bar.png"
    plot_top_combinations_bar(summary["top_tag_combinations"], p)
    print(f"[WROTE] {p}")

    p = plot_dir / "tag_family_breakdown.png"
    plot_tag_family_breakdown(summary["family_counts"], p)
    print(f"[WROTE] {p}")

    p = plot_dir / "tag_family_pies.png"
    plot_tag_family_pie_grid(summary["family_counts"], p)
    print(f"[WROTE] {p}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(summary: Dict[str, Any]) -> None:
    print("=== Scenario Tag Analysis ===")
    print(f"Total JSON files:            {summary['files_total']}")
    print(f"Parsed JSON files:           {summary['files_parsed']}")
    print(f"Failed JSON files:           {summary['files_failed']}")
    print(f"With scenario_tags:          {summary['files_with_scenario_tags']}")
    print(f"Without scenario_tags:       {summary['files_without_scenario_tags']}")
    print(f"Empty scenario_tags lists:   {summary['files_with_empty_scenario_tags']}")
    print(f"Unique scenario tags:        {summary['unique_tags']}")

    print("\nTop tags:")
    tag_items = list(summary["tag_counts"].items())[:20]
    if not tag_items:
        print("  <none>")
    for tag, count in tag_items:
        print(f"  {tag:40s} {count}")

    print("\nTags per file distribution:")
    for n_tags, file_count in summary["tags_per_file"].items():
        print(f"  {n_tags:2d} tags: {file_count}")

    print("\nTag family breakdown:")
    for fam, fc in summary["family_counts"].items():
        total = sum(fc.values()) or 1
        print(f"  [{fam}]")
        for tag, cnt in fc.items():
            print(f"    {tag:40s} {cnt:6d}  ({100.0 * cnt / total:5.1f}%)")

    print("\nTop combinations:")
    combos = summary["top_tag_combinations"][:10]
    if not combos:
        print("  <none>")
    for c in combos:
        print(f"  {c['count']:6d}  {c['combination']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and plot scenario tag distribution across JSON files."
    )
    parser.add_argument("--target_root", type=str, required=True,
                        help="Root directory to recursively scan for JSON files.")
    parser.add_argument("--output_dir", type=str,
                        default="analysis/scenario_tag_analysis",
                        help="Directory for CSV/JSON/PNG outputs.")
    parser.add_argument("--skip_plots", action="store_true",
                        help="Skip PNG generation (write CSVs and JSON only).")
    parser.add_argument("--top_tags", type=int, default=40,
                        help="Maximum tags shown in the tag-counts bar chart.")
    args = parser.parse_args()

    root = Path(args.target_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Invalid target_root: {root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Scanning: {root}")
    files = find_json_files(root)
    print(f"[INFO] Found {len(files)} -ls.json files")

    summary = analyze_files(files)

    # Remove family_counts from the JSON dump (it's redundant with tag_counts)
    summary_for_json = {k: v for k, v in summary.items() if k != "family_counts"}

    write_tag_counts_csv(summary["tag_counts"], output_dir / "tag_counts.csv")
    write_tags_per_file_csv(summary["tags_per_file"], output_dir / "num_tags_per_file.csv")

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_for_json, f, ensure_ascii=False, indent=2)

    print(f"[WROTE] {output_dir / 'tag_counts.csv'}")
    print(f"[WROTE] {output_dir / 'num_tags_per_file.csv'}")
    print(f"[WROTE] {output_dir / 'summary.json'}")

    if not args.skip_plots:
        generate_all_plots(summary, output_dir / "plots")

    # ------------------------------------------------------------------
    # Lane-segment level analysis
    # ------------------------------------------------------------------
    print(f"\n[INFO] Analysing lane-segment level data …")
    lane_summary = analyze_lane_segments(files)
    print(f"[INFO] Total lane segments: {lane_summary['total_segments']:,}")

    # Dump lane-segment summary (exclude large numeric arrays)
    lane_summary_for_json = {
        k: v for k, v in lane_summary.items()
        if k not in ("length_m_values", "heading_deg_values", "kappa_m_inv_values")
    }
    lane_json_path = output_dir / "lane_seg_summary.json"
    with lane_json_path.open("w", encoding="utf-8") as f:
        json.dump(lane_summary_for_json, f, ensure_ascii=False, indent=2)
    print(f"[WROTE] {lane_json_path}")

    if not args.skip_plots:
        generate_lane_seg_plots(lane_summary, output_dir / "plots")

    print_summary(summary)
    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
