#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Taxonomy analysis framework for OpenLane-V2 scenario tags.

Reads every *-ls.json under --dataset_root that has scenario_meta populated
by add_scenario_tags.py, and writes CSVs + a small set of plots into
--output_dir.

Outputs
-------
frames.csv              : One row per frame (master frame table).
lane_segments.csv       : One row per lane_segment across all frames.
prevalence_frames.csv   : Counts / % of each tag at frame level.
prevalence_segments.csv : Counts / % of each tag at segment level.
cooccurrence_frames.csv : Raw co-occurrence counts for every tag-pair
                          combination at frame level.
association_matrix.csv  : Cramer's V pairwise association matrix for the
                          three categorical frame-level tags.
plots/
  prevalence_frames.png
  prevalence_segments.png
  association_matrix.png
  per_tag_topology_score.png   (topology score distribution per topology tag)
  per_segment_heading_hist.png (heading distribution per segment curvature tag)

Usage
-----
python analyze_taxonomy.py
    --dataset_root /path/to/OpenLaneV2/train
    --output_dir   /path/to/output

If --dataset_root points at a single sequence dir it analyses just that
sequence. If it points at the split root (train/ or val/) it analyses all.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# -----------------------------------------------------------------------------
# Tag families (keep in sync with add_scenario_tags.py)
# -----------------------------------------------------------------------------

FRAME_TAG_FAMILIES = {
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

SEGMENT_TAG_VALUES = [
    "straight",
    "curve left",
    "curve right",
    "sharp left",
    "sharp right",
]


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def find_ls_jsons(root: Path) -> List[Path]:
    """All *-ls.json files under `root`."""
    return sorted(p for p in root.rglob("*-ls.json") if p.is_file())


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] {path}: read failed ({e})")
        return None


def sequence_id_from_path(p: Path) -> str:
    """
    Infer sequence id from path like .../<seq>/info/<stem>-ls.json
    Falls back to the immediate parent directory name.
    """
    parts = p.parts
    if "info" in parts:
        idx = parts.index("info")
        if idx >= 1:
            return parts[idx - 1]
    return p.parent.name


def frame_id_from_path(p: Path) -> str:
    stem = p.stem  # "<timestamp>-ls"
    if stem.endswith("-ls"):
        stem = stem[:-3]
    return stem


def safe_get(d: Any, *keys: Any, default: Any = None) -> Any:
    """Follow a chain of keys/indexes through a dict, returning default on miss."""
    cur = d
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k)
        elif isinstance(cur, list) and isinstance(k, int) and 0 <= k < len(cur):
            cur = cur[k]
        else:
            return default
    return cur if cur is not None else default


# -----------------------------------------------------------------------------
# Row extraction
# -----------------------------------------------------------------------------

def extract_frame_row(path: Path, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build one frame-level row; returns None if scenario_meta is absent."""
    meta = data.get("scenario_meta")
    if not isinstance(meta, dict):
        return None

    topo = meta.get("topology_complexity") or {}
    light = meta.get("lighting") or {}
    occ = meta.get("occlusion") or {}

    tags = data.get("scenario_tags") or []
    def _match_tag(family: str) -> Optional[str]:
        vals = FRAME_TAG_FAMILIES[family]
        for t in tags:
            if isinstance(t, str) and t.strip().lower() in vals:
                return t.strip().lower()
        return None

    topo_tag = _match_tag("topology") or "topology_unknown"
    light_tag = _match_tag("lighting") or "lighting_unknown"
    occ_tag = _match_tag("occlusion") or "occlusion_unknown"

    graph_stats = safe_get(topo, "graph", default={}) or {}
    lane_segments = meta.get("lane_segments") or []

    row = {
        "frame_id": frame_id_from_path(path),
        "sequence_id": sequence_id_from_path(path),
        "json_path": str(path),

        # Topology
        "topology_tag": topo_tag,
        "topology_score": topo.get("value"),
        "topology_method": topo.get("method"),
        "num_lane_segments": graph_stats.get("num_segments"),
        "num_connectors_in_bev": graph_stats.get("num_connectors_in_bev"),
        "num_splits_in_bev": graph_stats.get("num_splits_in_bev"),
        "num_merges_in_bev": graph_stats.get("num_merges_in_bev"),
        "dist_to_nearest_connector_m": graph_stats.get("dist_to_nearest_connector_m"),
        "ego_segment_idx": graph_stats.get("ego_segment_idx"),

        # Lighting
        "lighting_tag": light_tag,
        "lighting_score": light.get("score"),
        "mean_luminance": safe_get(light, "features", "mean_luminance"),
        "dark_ratio": safe_get(light, "features", "dark_ratio"),

        # Occlusion
        "occlusion_tag": occ_tag,
        "occlusion_ratio": occ.get("value"),
        "num_vehicles_detected": occ.get("total_vehicle_boxes"),
        "occlusion_status": occ.get("status"),

        # Per-frame segment counts (aggregated from lane_segments array)
        "num_tagged_segments": len(lane_segments),
    }

    # Add per-frame breakdown of segment-curvature counts.
    seg_counts: Counter = Counter()
    for seg in lane_segments:
        if isinstance(seg, dict):
            t = seg.get("curvature_tag", "curvature_unknown")
            seg_counts[t] += 1
    for tag_val in SEGMENT_TAG_VALUES:
        row[f"segs_{tag_val.replace(' ', '_')}"] = seg_counts.get(tag_val, 0)

    return row


def extract_segment_rows(path: Path, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per lane_segment under this frame."""
    meta = data.get("scenario_meta")
    if not isinstance(meta, dict):
        return []
    lane_segments = meta.get("lane_segments") or []
    rows = []
    frame_id = frame_id_from_path(path)
    seq_id = sequence_id_from_path(path)
    for seg in lane_segments:
        if not isinstance(seg, dict):
            continue
        rows.append({
            "frame_id": frame_id,
            "sequence_id": seq_id,
            "segment_id": seg.get("id"),
            "curvature_tag": seg.get("curvature_tag", "curvature_unknown"),
            "heading_deg": seg.get("heading_deg"),
            "value_m_inv": seg.get("value_m_inv"),
            "slope_tag": seg.get("slope_tag"),
            "kappa_tag": seg.get("kappa_tag"),
            "length_m": seg.get("length_m"),
            "status": seg.get("status"),
            "is_intersection_or_connector": seg.get("is_intersection_or_connector"),
        })
    return rows


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def cramers_v(contingency: np.ndarray) -> float:
    """
    Bias-corrected Cramer's V from a contingency table of counts.
    Returns 0.0 for degenerate cases.
    """
    if contingency.size == 0 or contingency.sum() == 0:
        return 0.0
    chi2 = _chi2_from_table(contingency)
    n = float(contingency.sum())
    r, k = contingency.shape
    if min(r, k) <= 1:
        return 0.0
    # Bias correction (Bergsma & Wicher)
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return 0.0
    return float(math.sqrt(phi2corr / denom))


def _chi2_from_table(contingency: np.ndarray) -> float:
    row_sums = contingency.sum(axis=1, keepdims=True)
    col_sums = contingency.sum(axis=0, keepdims=True)
    total = contingency.sum()
    if total == 0:
        return 0.0
    expected = row_sums @ col_sums / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.where(expected > 0, (contingency - expected) ** 2 / expected, 0.0)
    return float(chi2.sum())


def contingency_table(a: Sequence[str], b: Sequence[str],
                      a_vals: Sequence[str], b_vals: Sequence[str]) -> np.ndarray:
    a_idx = {v: i for i, v in enumerate(a_vals)}
    b_idx = {v: i for i, v in enumerate(b_vals)}
    table = np.zeros((len(a_vals), len(b_vals)), dtype=np.int64)
    for av, bv in zip(a, b):
        ai = a_idx.get(av)
        bi = b_idx.get(bv)
        if ai is None or bi is None:
            continue
        table[ai, bi] += 1
    return table


# -----------------------------------------------------------------------------
# CSV writers
# -----------------------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
        for r in rows[1:]:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

TAG_COLORS = {
    # curvature
    "straight": "#2ca02c",
    "straight with an angle": "#f0a000",
    "curve left": "#1f77b4",
    "curve right": "#d62728",
    "curvature_unknown": "#888888",
    # topology
    "low topological complexity": "#9ecae1",
    "medium topological complexity": "#4292c6",
    "high topological complexity": "#08519c",
    "topology_unknown": "#888888",
    # lighting
    "well lit": "#fee08b",
    "poorly lit": "#4575b4",
    "lighting_unknown": "#888888",
    # occlusion
    "no occlusion": "#a1d99b",
    "low occlusion": "#31a354",
    "high occlusion": "#006d2c",
    "occlusion_unknown": "#888888",
}


def build_tag_cooccurrence_matrix(
    items: List[Dict[str, Any]],
    tag_fields: List[Tuple[str, List[str]]],
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a per-tag co-occurrence matrix across a set of (field_name, tag_values) pairs.

    Each "item" (frame or segment) carries one tag per field. The returned matrix
    is NxN where N = total number of tag values across all fields. Cell (i,j) =
    number of items that have both tag i and tag j simultaneously.

    Items are expected to have exactly one tag per field, so within-field cells
    are the prevalence of that tag (diagonal blocks are diagonal within each family).

    Returns
    -------
    labels    : list of N tag strings (e.g. "curvature: straight", "topology: low ...")
    counts    : NxN raw co-occurrence counts
    lift      : NxN lift matrix: lift(i,j) = P(i AND j) / (P(i) * P(j))
                (nan where either marginal is 0; 1.0 means independent)
    cond_prob : NxN P(j | i), rows sum to 1 where the row marginal is > 0
    """
    labels: List[str] = []
    label_to_idx: Dict[Tuple[str, str], int] = {}
    for fam, vals in tag_fields:
        for v in vals:
            label_to_idx[(fam, v)] = len(labels)
            labels.append(f"{fam}: {v}")

    n = len(labels)
    counts = np.zeros((n, n), dtype=np.int64)
    for it in items:
        active_idx: List[int] = []
        for fam, _vals in tag_fields:
            field = f"{fam}_tag" if not fam.startswith("seg_") else fam
            v = it.get(field)
            if v is None:
                continue
            idx = label_to_idx.get((fam, v))
            if idx is None:
                continue
            active_idx.append(idx)
        for i in active_idx:
            for j in active_idx:
                counts[i, j] += 1

    total = float(len(items)) if items else 1.0
    diag = np.diag(counts).astype(np.float64)
    p_i = diag / total  # marginal prob of each tag

    # Lift: P(i,j) / (P(i)P(j))
    with np.errstate(divide="ignore", invalid="ignore"):
        joint = counts.astype(np.float64) / total
        outer = np.outer(p_i, p_i)
        lift = np.where(outer > 0, joint / outer, np.nan)

    # Conditional probability: P(j | i) = count(i,j) / count(i)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond_prob = np.where(diag[:, None] > 0, counts.astype(np.float64) / diag[:, None], np.nan)

    return labels, counts, lift, cond_prob


def plot_conditional_probability_matrix(
    labels: List[str],
    cond_prob: np.ndarray,
    counts: np.ndarray,
    out_path: Path,
    title: str,
    family_boundaries: Optional[List[int]] = None,
    display_labels: Optional[List[str]] = None,
) -> None:
    """Plot a conditional-probability heatmap in the 'P(col | row)' style.

    Cell (i, j) reads as "given row tag i, what % of those frames also have
    column tag j?". Within-family cells are masked with a dash.
    """
    if plt is None:
        return
    n = len(labels)
    labels_shown = display_labels if display_labels is not None else labels

    # Mask within-family cells (same family => either the diagonal at 100% or
    # mutually-exclusive off-diagonals at 0% — uninformative either way)
    within_family_mask = np.zeros((n, n), dtype=bool)
    if family_boundaries:
        fam_of = np.zeros(n, dtype=np.int64)
        fam_idx = 0
        for i in range(n):
            while fam_idx < len(family_boundaries) and i >= family_boundaries[fam_idx]:
                fam_idx += 1
            fam_of[i] = fam_idx
        for i in range(n):
            for j in range(n):
                if fam_of[i] == fam_of[j]:
                    within_family_mask[i, j] = True
    else:
        np.fill_diagonal(within_family_mask, True)

    # Replace masked cells with NaN for display; a single viridis-style cmap
    # over [0, 1] reads as a clean percentage gradient.
    disp = np.where(within_family_mask, np.nan, cond_prob)
    figsize = (max(8.0, 0.65 * n + 2), max(7.0, 0.6 * n + 2))
    fig, ax = plt.subplots(figsize=figsize, dpi=110,
                           facecolor="#1a1d2e")
    ax.set_facecolor("#1a1d2e")

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#2a2d3e")  # masked cells

    im = ax.imshow(disp, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels_shown, rotation=90, ha="center",
                       fontsize=9, color="white")
    ax.set_yticklabels(labels_shown, fontsize=9, color="white")
    ax.xaxis.tick_top()
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    for i in range(n):
        for j in range(n):
            if within_family_mask[i, j]:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color="#888")
                continue
            p = cond_prob[i, j]
            if np.isnan(p):
                continue
            pct = int(round(100.0 * p))
            # Text color: white on dark (low p), black on bright (high p)
            color = "black" if p > 0.55 else "white"
            ax.text(j, i, f"{pct}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    # Family-boundary lines in white
    if family_boundaries:
        for b in family_boundaries:
            ax.axhline(b - 0.5, color="white", linewidth=1.0, alpha=0.4)
            ax.axvline(b - 0.5, color="white", linewidth=1.0, alpha=0.4)

    cbar = fig.colorbar(im, ax=ax, label="P(column | row)",
                        fraction=0.035, pad=0.04)
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")

    ax.set_title(title, color="white", fontsize=11, pad=16)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_cooccurrence_matrix(
    labels: List[str],
    values: np.ndarray,
    counts: np.ndarray,
    out_path: Path,
    title: str,
    cmap: str = "RdBu_r",
    center: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    family_boundaries: Optional[List[int]] = None,
    mask_within_family: bool = True,
) -> None:
    """Plot an NxN co-occurrence matrix with cell-level numbers.

    When mask_within_family=True and family_boundaries is given, cells within
    the same family (diagonal plus mutually-exclusive off-diagonal cells) are
    masked so the colormap auto-scales to the informative cross-family region
    only. This avoids the common-tag self-lift tautology where diagonal values
    of 1/P(tag) dominate the scale without conveying information.
    """
    if plt is None:
        return
    n = len(labels)

    # Identify which cells are "within family" (and therefore uninformative):
    #   - the diagonal (self-lift = 1/P, tautological)
    #   - off-diagonal cells where both tags are in the same family (mutually
    #     exclusive, co-occurrence always 0)
    within_family_mask = np.zeros((n, n), dtype=bool)
    if family_boundaries:
        fam_of = np.zeros(n, dtype=np.int64)
        fam_idx = 0
        for i in range(n):
            while fam_idx < len(family_boundaries) and i >= family_boundaries[fam_idx]:
                fam_idx += 1
            fam_of[i] = fam_idx
        for i in range(n):
            for j in range(n):
                if fam_of[i] == fam_of[j]:
                    within_family_mask[i, j] = True
    else:
        # Without family info, still mask diagonal.
        np.fill_diagonal(within_family_mask, True)

    # Values we color by — within-family cells are masked to NaN so they don't
    # influence vmin/vmax auto-scaling.
    if mask_within_family:
        values_colored = np.where(within_family_mask, np.nan, values)
    else:
        values_colored = values

    figsize = (max(7.0, 0.55 * n + 3), max(6.0, 0.55 * n + 2))
    fig, ax = plt.subplots(figsize=figsize, dpi=110)

    # Replace nans with center for display
    disp = np.where(np.isnan(values_colored), center, values_colored)
    finite_colored = values_colored[np.isfinite(values_colored)]
    if vmin is None:
        vmin = float(np.min(finite_colored)) if finite_colored.size > 0 else 0.0
    if vmax is None:
        vmax = float(np.max(finite_colored)) if finite_colored.size > 0 else 2.0
    # Symmetric range around center for divergent cmap (RdBu_r)
    if cmap == "RdBu_r":
        spread = max(abs(vmax - center), abs(vmin - center), 0.5)
        vmin_plot = center - spread
        vmax_plot = center + spread
    else:
        vmin_plot = vmin
        vmax_plot = vmax


    im = ax.imshow(disp, cmap=cmap, vmin=vmin_plot, vmax=vmax_plot, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(n):
        for j in range(n):
            v = values[i, j]
            c = counts[i, j]
            is_masked = mask_within_family and within_family_mask[i, j]
            if is_masked:
                # Gray out the cell so it reads as "not compared"
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           color="#e8e8e8", zorder=1.5))
                if i == j:
                    # Show the marginal count for reference on the diagonal
                    txt = f"n={int(c)}"
                else:
                    txt = ""
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=6.5, color="#666666")
                continue
            if np.isnan(v):
                txt = "-"
            elif c == 0:
                txt = "0"
            else:
                txt = f"{v:.2f}\n(n={int(c)})"
            # Text color: black on light backgrounds, white on dark
            disp_v = disp[i, j]
            dist_from_center = abs(disp_v - (vmin_plot + vmax_plot) / 2.0)
            range_half = (vmax_plot - vmin_plot) / 2.0
            color = "white" if range_half > 0 and dist_from_center / range_half > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color=color)

    # Draw family-boundary lines
    if family_boundaries:
        for b in family_boundaries:
            ax.axhline(b - 0.5, color="black", linewidth=1.2)
            ax.axvline(b - 0.5, color="black", linewidth=1.2)

    fig.colorbar(im, ax=ax, label="lift" if cmap == "RdBu_r" else "value")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def write_matrix_csv(path: Path, labels: List[str], matrix: np.ndarray, value_fmt: str = "{:.4f}") -> None:
    """Write an NxN matrix as CSV with labels as header row and first column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag"] + labels)
        for i, lab in enumerate(labels):
            row = [lab]
            for j in range(len(labels)):
                v = matrix[i, j]
                if isinstance(v, (int, np.integer)):
                    row.append(str(int(v)))
                elif np.isnan(v):
                    row.append("")
                else:
                    row.append(value_fmt.format(float(v)))
            w.writerow(row)


def plot_prevalence(counts: Dict[str, Dict[str, int]], out_path: Path, title: str) -> None:
    if plt is None:
        return
    fams = list(counts.keys())
    n = len(fams)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), dpi=100)
    if n == 1:
        axes = [axes]
    for ax, fam in zip(axes, fams):
        fam_counts = counts[fam]
        labels = list(fam_counts.keys())
        vals = [fam_counts[k] for k in labels]
        colors = [TAG_COLORS.get(lbl, "#888888") for lbl in labels]
        ax.barh(range(len(labels)), vals, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([lbl.replace(" ", "\n", 1) for lbl in labels], fontsize=8)
        ax.invert_yaxis()
        total = sum(vals) or 1
        for i, v in enumerate(vals):
            ax.text(v, i, f"  {v} ({100.0 * v / total:.1f}%)",
                    va="center", ha="left", fontsize=8)
        ax.set_title(fam)
        ax.margins(x=0.15)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def plot_association_matrix(matrix: np.ndarray, labels: List[str], out_path: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if matrix[i, j] < 0.5 else "black",
                    fontsize=10)
    fig.colorbar(im, ax=ax, label="Cramér's V")
    ax.set_title("Pairwise tag association (Cramér's V)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def plot_hist_per_tag(frames: List[Dict[str, Any]], field: str,
                      tag_field: str, tag_order: List[str],
                      out_path: Path, xlabel: str,
                      clip: Optional[Tuple[float, float]] = None) -> None:
    if plt is None:
        return
    by_tag: Dict[str, List[float]] = defaultdict(list)
    for r in frames:
        v = r.get(field)
        t = r.get(tag_field)
        if v is None or not isinstance(v, (int, float)):
            continue
        if clip is not None:
            v = max(clip[0], min(clip[1], float(v)))
        by_tag[t].append(float(v))
    tags_present = [t for t in tag_order if t in by_tag]
    if not tags_present:
        return
    fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
    for t in tags_present:
        ax.hist(by_tag[t], bins=30, alpha=0.55,
                color=TAG_COLORS.get(t, "#888888"), label=f"{t} (n={len(by_tag[t])})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(f"{xlabel} distribution by {tag_field}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, type=str,
                    help="Path to the val/ dir, or a single sequence dir.")
    ap.add_argument("--output_dir", required=True, type=str,
                    help="Directory where CSVs and plots will be written.")
    ap.add_argument("--skip_plots", action="store_true",
                    help="Write only CSVs, no PNGs.")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = find_ls_jsons(root)
    print(f"[INFO] Found {len(files)} -ls.json files under {root}")
    if not files:
        print("[ERROR] No -ls.json files found. Is the path correct and tagging complete?")
        return

    frame_rows: List[Dict[str, Any]] = []
    segment_rows: List[Dict[str, Any]] = []

    skipped_no_meta = 0
    for i, fp in enumerate(files, 1):
        data = load_json(fp)
        if data is None:
            continue
        row = extract_frame_row(fp, data)
        if row is None:
            skipped_no_meta += 1
            continue
        frame_rows.append(row)
        segment_rows.extend(extract_segment_rows(fp, data))
        if i % 500 == 0 or i == len(files):
            print(f"[PROGRESS] {i}/{len(files)}  frames_accepted={len(frame_rows)}")

    if skipped_no_meta:
        print(f"[WARN] {skipped_no_meta} files had no scenario_meta (not tagged yet).")

    if not frame_rows:
        print("[ERROR] No tagged frames. Run add_scenario_tags.py first.")
        return

    # --- Master tables ---
    write_csv(out / "frames.csv", frame_rows)
    write_csv(out / "lane_segments.csv", segment_rows)
    print(f"[WROTE] {out / 'frames.csv'}  ({len(frame_rows)} rows)")
    print(f"[WROTE] {out / 'lane_segments.csv'}  ({len(segment_rows)} rows)")

    # --- Prevalence tables ---
    frame_prev: Dict[str, Dict[str, int]] = {}
    for fam, vals in FRAME_TAG_FAMILIES.items():
        field = f"{fam}_tag"
        c = Counter(r.get(field) for r in frame_rows)
        frame_prev[fam] = {v: int(c.get(v, 0)) for v in vals}
    prev_rows = []
    for fam, d in frame_prev.items():
        total = sum(d.values()) or 1
        for tag_val, cnt in d.items():
            prev_rows.append({
                "family": fam,
                "tag": tag_val,
                "count": cnt,
                "pct": f"{100.0 * cnt / total:.2f}",
            })
    write_csv(out / "prevalence_frames.csv", prev_rows)
    print(f"[WROTE] {out / 'prevalence_frames.csv'}")

    seg_counter = Counter(r.get("curvature_tag") for r in segment_rows)
    seg_total = sum(seg_counter.values()) or 1
    seg_prev_rows = []
    for tag_val in SEGMENT_TAG_VALUES:
        cnt = int(seg_counter.get(tag_val, 0))
        seg_prev_rows.append({
            "tag": tag_val,
            "count": cnt,
            "pct": f"{100.0 * cnt / seg_total:.2f}",
        })
    write_csv(out / "prevalence_segments.csv", seg_prev_rows)
    print(f"[WROTE] {out / 'prevalence_segments.csv'}")

    # --- Co-occurrence counts (frame level, all pairs) ---
    cooc_rows = []
    fams = list(FRAME_TAG_FAMILIES.keys())
    for i, fa in enumerate(fams):
        for fb in fams[i + 1:]:
            a_seq = [r[f"{fa}_tag"] for r in frame_rows]
            b_seq = [r[f"{fb}_tag"] for r in frame_rows]
            a_vals = FRAME_TAG_FAMILIES[fa]
            b_vals = FRAME_TAG_FAMILIES[fb]
            table = contingency_table(a_seq, b_seq, a_vals, b_vals)
            for i_a, av in enumerate(a_vals):
                for i_b, bv in enumerate(b_vals):
                    cooc_rows.append({
                        "family_a": fa, "tag_a": av,
                        "family_b": fb, "tag_b": bv,
                        "count": int(table[i_a, i_b]),
                    })
    write_csv(out / "cooccurrence_frames.csv", cooc_rows)
    print(f"[WROTE] {out / 'cooccurrence_frames.csv'}")

    # --- Cramer's V association matrix (frame-level, 4x4) ---
    labels = fams  # curvature, topology, lighting, occlusion
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for i, fa in enumerate(labels):
        for j, fb in enumerate(labels):
            if i == j:
                matrix[i, j] = 1.0
                continue
            a_seq = [r[f"{fa}_tag"] for r in frame_rows]
            b_seq = [r[f"{fb}_tag"] for r in frame_rows]
            a_vals = FRAME_TAG_FAMILIES[fa]
            b_vals = FRAME_TAG_FAMILIES[fb]
            table = contingency_table(a_seq, b_seq, a_vals, b_vals)
            matrix[i, j] = cramers_v(table)

    assoc_rows = [{"family": labels[i],
                   **{labels[j]: f"{matrix[i, j]:.4f}" for j in range(len(labels))}}
                  for i in range(len(labels))]
    write_csv(out / "association_matrix.csv", assoc_rows)
    print(f"[WROTE] {out / 'association_matrix.csv'}")

    # --- Per-tag co-occurrence matrices (frame-level, 15x15) ---
    frame_tag_fields = [(fam, FRAME_TAG_FAMILIES[fam]) for fam in fams]
    tag_labels, tag_counts, tag_lift, tag_cond = build_tag_cooccurrence_matrix(
        frame_rows, frame_tag_fields,
    )
    # Family-boundary indices (for drawing the lines in the heatmap)
    boundaries = []
    acc = 0
    for fam, vals in frame_tag_fields:
        acc += len(vals)
        boundaries.append(acc)
    write_matrix_csv(out / "cooccurrence_matrix_frames_counts.csv", tag_labels,
                     tag_counts, value_fmt="{:.0f}")
    write_matrix_csv(out / "cooccurrence_matrix_frames_lift.csv", tag_labels,
                     tag_lift, value_fmt="{:.4f}")
    write_matrix_csv(out / "cooccurrence_matrix_frames_cond_prob.csv", tag_labels,
                     tag_cond, value_fmt="{:.4f}")
    print(f"[WROTE] {out / 'cooccurrence_matrix_frames_counts.csv'}")
    print(f"[WROTE] {out / 'cooccurrence_matrix_frames_lift.csv'}")
    print(f"[WROTE] {out / 'cooccurrence_matrix_frames_cond_prob.csv'}")

    # Per-tag co-occurrence for lane-segment-level (only one family: segment curvature)
    seg_tag_fields = [("segment_curvature", SEGMENT_TAG_VALUES)]
    seg_items_for_cooc = [
        {"segment_curvature_tag": r.get("curvature_tag", "unknown")} for r in segment_rows
    ]
    seg_tag_labels, seg_tag_counts, seg_tag_lift, seg_tag_cond = build_tag_cooccurrence_matrix(
        seg_items_for_cooc, seg_tag_fields,
    )
    write_matrix_csv(out / "cooccurrence_matrix_segments_counts.csv", seg_tag_labels,
                     seg_tag_counts, value_fmt="{:.0f}")
    write_matrix_csv(out / "cooccurrence_matrix_segments_lift.csv", seg_tag_labels,
                     seg_tag_lift, value_fmt="{:.4f}")
    print(f"[WROTE] {out / 'cooccurrence_matrix_segments_counts.csv'}")
    print(f"[WROTE] {out / 'cooccurrence_matrix_segments_lift.csv'}")

    # --- Plots ---
    if plt is None:
        print("[WARN] matplotlib not installed, skipping plots.")
    elif args.skip_plots:
        print("[INFO] --skip_plots set, no PNGs written.")
    else:
        plot_prevalence(frame_prev, out / "plots" / "prevalence_frames.png",
                        title=f"Frame-level tag prevalence (n={len(frame_rows)} frames)")
        plot_prevalence(
            {"lane-segment curvature": {v: int(seg_counter.get(v, 0)) for v in SEGMENT_TAG_VALUES}},
            out / "plots" / "prevalence_segments.png",
            title=f"Lane-segment curvature prevalence (n={len(segment_rows)} segments)",
        )
        plot_association_matrix(matrix, labels, out / "plots" / "association_matrix.png")
        plot_cooccurrence_matrix(
            tag_labels, tag_lift, tag_counts,
            out / "plots" / "cooccurrence_matrix_frames_lift.png",
            title=("Frame-level tag co-occurrence — lift "
                   "(red = over-represented, blue = under-represented, white ≈ 1.0 = independent)"),
            cmap="RdBu_r", center=1.0,
            family_boundaries=boundaries,
        )

        # Short display labels for the conditional-probability heatmap.
        # Dropped the family prefix; shortened "topological complexity" to "topo".
        short_label_map = {
            "topology: low topological complexity": "low topo",
            "topology: high topological complexity": "high topo",
            "topology: topology_unknown": "topo_unknown",
            "lighting: well lit": "well lit",
            "lighting: poorly lit": "poorly lit",
            "lighting: lighting_unknown": "light_unknown",
            "occlusion: no occlusion": "no occlusion",
            "occlusion: low occlusion": "low occlusion",
            "occlusion: high occlusion": "high occlusion",
            "occlusion: occlusion_unknown": "occ_unknown",
        }
        short_labels = [short_label_map.get(l, l) for l in tag_labels]
        plot_conditional_probability_matrix(
            tag_labels, tag_cond, tag_counts,
            out / "plots" / "conditional_probability_matrix_frames.png",
            title="P(column tag | row tag) — row-wise conditional probabilities",
            family_boundaries=boundaries,
            display_labels=short_labels,
        )
        plot_cooccurrence_matrix(
            seg_tag_labels, seg_tag_lift, seg_tag_counts,
            out / "plots" / "cooccurrence_matrix_segments_lift.png",
            title="Lane-segment curvature co-occurrence — lift",
            cmap="RdBu_r", center=1.0,
        )
        plot_hist_per_tag(
            frame_rows, "topology_score", "topology_tag",
            FRAME_TAG_FAMILIES["topology"],
            out / "plots" / "per_tag_topology_score.png",
            xlabel="topology score", clip=(0.0, 1.0),
        )
        plot_hist_per_tag(
            segment_rows, "heading_deg", "curvature_tag",
            SEGMENT_TAG_VALUES,
            out / "plots" / "per_segment_heading_hist.png",
            xlabel="lane-segment heading (deg)", clip=(-180.0, 180.0),
        )
        print(f"[WROTE] plots in {out / 'plots'}")

    # --- Short stdout summary ---
    print("\n=== Frame-level prevalence ===")
    for fam, d in frame_prev.items():
        total = sum(d.values()) or 1
        print(f"  [{fam}] n={total}")
        for k, v in sorted(d.items(), key=lambda kv: -kv[1]):
            print(f"     {k:32s} : {v:5d}  ({100.0 * v / total:5.1f}%)")

    print("\n=== Cramer's V (frame-level 3x3) ===")
    header = "          " + "  ".join(f"{l[:10]:>10}" for l in labels)
    print(header)
    for i, lab in enumerate(labels):
        row = f"{lab[:10]:>10}  " + "  ".join(f"{matrix[i, j]:10.3f}" for j in range(len(labels)))
        print(row)

    print(f"\n[INFO] Done. Outputs in: {out}")


if __name__ == "__main__":
    main()