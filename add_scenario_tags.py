#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenLane-V2 scenario tagger (curvature + topology complexity + lighting + occlusion)

Tags written:
-------------
- Curvature:
    "curvature_unknown"
    "straight"
    "curve"
    "sharp curve"

- Topology complexity:
    "topology_unknown"
    "low topological complexity"
    "medium topological complexity"
    "high topological complexity"

- Lighting:
    "well lit"
    "poorly lit"
    "lighting_unknown"

- Occlusion:
    "low occlusion"
    "high occlusion"
    "occlusion_unknown"

Meta written:
-------------
scenario_meta.curvature
scenario_meta.topology_complexity
scenario_meta.lighting
scenario_meta.occlusion

Usage:
------
Dry run (test on first 10 frames, write debug viz):
python add_scenario_tags.py \
  --dataset_root /path/to/train \
  --dry_run \
  --debug_viz_n 10 \
  --debug_viz_dir ./debug_viz

Full write:
python add_scenario_tags.py \
  --dataset_root /path/to/train
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend, safe for batch runs
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
except Exception:
    plt = None
    Rectangle = None
    Line2D = None


# =========================================================
# Generic helpers
# =========================================================

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def angle_wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def angle_diff(a: float, b: float) -> float:
    return angle_wrap(a - b)


def wrap_angle_arr(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x.copy()
    kernel = np.ones(w, dtype=np.float64) / w
    xpad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(xpad, kernel, mode="valid")


def is_point_like(p: Any) -> bool:
    return isinstance(p, (list, tuple)) and len(p) >= 2 and all(
        isinstance(v, (int, float)) for v in p[:2]
    )


def is_polyline_like(x: Any) -> bool:
    return isinstance(x, list) and len(x) >= 2 and all(is_point_like(p) for p in x)


def polyline_to_xy(poly: List[List[float]]) -> np.ndarray:
    arr = np.array(poly, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    return arr[:, :2]


def remove_duplicate_points(xy: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if len(xy) <= 1:
        return xy
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = d > eps
    return xy[keep]


def resample_polyline(xy: np.ndarray, ds: float = 0.5) -> np.ndarray:
    if len(xy) < 2:
        return np.zeros((0, 2), dtype=np.float64)

    xy = remove_duplicate_points(xy)
    if len(xy) < 2:
        return np.zeros((0, 2), dtype=np.float64)

    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = s[-1]
    if total < ds:
        return xy.copy()

    s_new = np.arange(0.0, total + 1e-9, ds)
    x_new = np.interp(s_new, s, xy[:, 0])
    y_new = np.interp(s_new, s, xy[:, 1])
    return np.stack([x_new, y_new], axis=1)


def smooth_polyline(xy: np.ndarray, window: int = 5) -> np.ndarray:
    if len(xy) < max(5, window):
        return xy
    x = moving_average(xy[:, 0], window)
    y = moving_average(xy[:, 1], window)
    return np.stack([x, y], axis=1)


def nearest_y_at_x(xy: np.ndarray, x_target: float, max_dx: float = float("inf")) -> float:
    if len(xy) == 0:
        return float("nan")
    idx = int(np.argmin(np.abs(xy[:, 0] - x_target)))
    dx = abs(float(xy[idx, 0]) - x_target)
    if dx > max_dx:
        return float("nan")
    return float(xy[idx, 1])


def heading_at_x(xy: np.ndarray, x_target: float) -> float:
    if len(xy) < 3:
        return float("nan")
    idx = int(np.argmin(np.abs(xy[:, 0] - x_target)))
    i0 = max(0, idx - 1)
    i1 = min(len(xy) - 1, idx + 1)
    if i1 == i0:
        return float("nan")
    dx = xy[i1, 0] - xy[i0, 0]
    dy = xy[i1, 1] - xy[i0, 1]
    if abs(dx) + abs(dy) < 1e-8:
        return float("nan")
    return float(math.atan2(dy, dx))


def cluster_1d(values: List[float], eps: float) -> List[List[float]]:
    if not values:
        return []
    vals = sorted(values)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - groups[-1][-1]) <= eps:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups


def dedupe_polylines(polys: List[np.ndarray], tol: float = 0.2) -> List[np.ndarray]:
    kept = []
    seen = set()
    for p in polys:
        if len(p) < 2:
            continue
        key = (
            round(float(p[0, 0]) / tol), round(float(p[0, 1]) / tol),
            round(float(p[-1, 0]) / tol), round(float(p[-1, 1]) / tol),
            len(p)
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(p)
    return kept


# =========================================================
# Schema-tolerant polyline / relation extraction
# =========================================================

KEY_HINTS = [
    "centerline", "center_line", "polyline", "points", "xyz", "geometry",
    "lane_centerline", "lane_center_line", "lane_points", "coords"
]


def maybe_extract_polyline_from_dict(d: Dict[str, Any]) -> List[List[float]]:
    for k, v in d.items():
        kl = k.lower()
        if any(h in kl for h in KEY_HINTS) and is_polyline_like(v):
            return v

    for _, v in d.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                kkl = kk.lower()
                if any(h in kkl for h in KEY_HINTS) and is_polyline_like(vv):
                    return vv
    return []


def collect_polylines(obj: Any, out: List[List[List[float]]]) -> None:
    if isinstance(obj, dict):
        poly = maybe_extract_polyline_from_dict(obj)
        if poly:
            out.append(poly)
        for v in obj.values():
            collect_polylines(v, out)
    elif isinstance(obj, list):
        if is_polyline_like(obj):
            out.append(obj)  # type: ignore
        else:
            for it in obj:
                collect_polylines(it, out)


# =========================================================
# Curvature
# =========================================================

def forward_crop(xy: np.ndarray, x_min: float, x_max: float, y_abs_max: float) -> np.ndarray:
    if len(xy) == 0:
        return xy
    m = (xy[:, 0] >= x_min) & (xy[:, 0] <= x_max) & (np.abs(xy[:, 1]) <= y_abs_max)
    return xy[m]


def curvature_samples(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(xy) < 4:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    dxy = np.diff(xy, axis=0)
    ds = np.linalg.norm(dxy, axis=1)
    if np.sum(ds > 1e-6) < 3:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    theta = np.arctan2(dxy[:, 1], dxy[:, 0])
    dtheta = wrap_angle_arr(np.diff(theta))
    ds_mid = 0.5 * (ds[:-1] + ds[1:])
    kappa = np.divide(dtheta, ds_mid, out=np.zeros_like(dtheta), where=ds_mid > 1e-6)

    s = np.concatenate(([0.0], np.cumsum(ds)))
    s_mid = 0.5 * (s[1:-1] + s[2:])
    return kappa, s_mid


def classify_curvature(heading: float, k_frame: float, thr: Dict[str, float]) -> Tuple[str, str]:
    if np.isfinite(heading):
        a = abs(heading)
        straight_rad = thr.get("straight_rad", math.radians(2.0))
        angled_rad = thr.get("angled_rad", math.radians(8.0))
        if a < straight_rad:
            return "straight", "slope"
        if a < angled_rad:
            return "straight with an angle", "slope"
        return ("curve left" if heading > 0 else "curve right"), "slope"

    if np.isfinite(k_frame):
        kappa_straight = thr.get("legacy_straight_m_inv", 0.003)
        if abs(k_frame) < kappa_straight:
            return "straight", "kappa"
        return ("curve left" if k_frame > 0 else "curve right"), "kappa"

    return "curvature_unknown", "none"


def _segment_start_end_heading(xy: np.ndarray) -> Tuple[float, float]:
    if len(xy) < 2:
        return float("nan"), float("nan")
    for i in range(1, min(len(xy), 5)):
        dx = xy[i, 0] - xy[0, 0]
        dy = xy[i, 1] - xy[0, 1]
        if abs(dx) + abs(dy) > 1e-6:
            start_h = math.atan2(dy, dx)
            break
    else:
        return float("nan"), float("nan")
    n = len(xy)
    for j in range(1, min(n, 5)):
        dx = xy[n - 1, 0] - xy[n - 1 - j, 0]
        dy = xy[n - 1, 1] - xy[n - 1 - j, 1]
        if abs(dx) + abs(dy) > 1e-6:
            end_h = math.atan2(dy, dx)
            break
    else:
        end_h = start_h
    return float(start_h), float(end_h)


def build_ego_lane_chain(
    data: Dict[str, Any],
    max_length_m: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "status": "ok",
        "start_segment_idx": None,
        "segments_used": 0,
        "chain_length_m": 0.0,
        "num_branches_encountered": 0,
    }

    segments = _get_lane_segments(data)
    if not segments:
        info["status"] = "no_lane_segments"
        return np.zeros((0, 2), dtype=np.float64), info

    lsls_value = data.get("topology_lsls")
    if lsls_value is None:
        lsls_value = _get_annotation_dict(data).get("topology_lsls", [])
    lsls = _to_numpy_matrix(lsls_value)

    start_candidates = []
    for idx, seg in enumerate(segments):
        xy = _segment_centerline_xy(seg)
        if len(xy) < 2:
            continue
        dists = np.linalg.norm(xy[:, :2], axis=1)
        closest = int(np.argmin(dists))
        d = float(dists[closest])
        if d > 5.0:
            continue
        i0 = max(0, closest - 1)
        i1 = min(len(xy) - 1, closest + 1)
        if i1 == i0:
            continue
        dx = xy[i1, 0] - xy[i0, 0]
        dy = xy[i1, 1] - xy[i0, 1]
        if abs(dx) + abs(dy) < 1e-6:
            continue
        local_h = math.atan2(dy, dx)
        if abs(local_h) > math.radians(60.0):
            continue
        trimmed = xy[closest:]
        if len(trimmed) < 2:
            continue
        score = d * 2.0 + abs(local_h)
        start_candidates.append((score, idx, trimmed))

    if not start_candidates:
        info["status"] = "no_start_segment"
        return np.zeros((0, 2), dtype=np.float64), info

    start_candidates.sort(key=lambda c: c[0])
    _, start_idx, start_xy = start_candidates[0]
    info["start_segment_idx"] = int(start_idx)

    n = len(segments)
    has_graph = lsls.size > 0 and lsls.shape[0] >= n and lsls.shape[1] >= n
    if has_graph:
        lsls_bin = (lsls[:n, :n] > 0).astype(np.int32)
    else:
        lsls_bin = None

    visited = {start_idx}
    chain_pieces: List[np.ndarray] = [start_xy]
    total_length = float(np.sum(np.linalg.norm(np.diff(start_xy, axis=0), axis=1)))
    current_idx = start_idx
    current_xy = start_xy

    while total_length < max_length_m and lsls_bin is not None:
        _, parent_end_h = _segment_start_end_heading(current_xy)
        if not np.isfinite(parent_end_h):
            break
        successors = np.where(lsls_bin[current_idx] > 0)[0].tolist()
        successors = [s for s in successors if s not in visited]
        if not successors:
            break
        if len(successors) > 1:
            info["num_branches_encountered"] += 1

        best_succ = None
        best_diff = math.inf
        for s in successors:
            s_xy = _segment_centerline_xy(segments[s])
            if len(s_xy) < 2:
                continue
            s_start_h, _ = _segment_start_end_heading(s_xy)
            if not np.isfinite(s_start_h):
                continue
            diff = abs(angle_diff(s_start_h, parent_end_h))
            if diff < best_diff:
                best_diff = diff
                best_succ = (s, s_xy)

        if best_succ is None:
            break
        next_idx, next_xy = best_succ
        visited.add(next_idx)
        chain_pieces.append(next_xy)
        total_length += float(np.sum(np.linalg.norm(np.diff(next_xy, axis=0), axis=1)))
        current_idx = next_idx
        current_xy = next_xy

    stitched: List[np.ndarray] = [chain_pieces[0]]
    for piece in chain_pieces[1:]:
        last_point = stitched[-1][-1]
        if len(piece) > 0 and np.linalg.norm(piece[0] - last_point) < 0.5:
            stitched.append(piece[1:])
        else:
            stitched.append(piece)
    chain_xy = np.vstack([p for p in stitched if len(p) > 0]) if stitched else np.zeros((0, 2), dtype=np.float64)

    if len(chain_xy) >= 2:
        info["chain_length_m"] = float(np.sum(np.linalg.norm(np.diff(chain_xy, axis=0), axis=1)))
    info["segments_used"] = len(chain_pieces)
    return chain_xy, info


def _polyline_slope_heading(xy: np.ndarray, x_mid: float, fit_half: float = 5.0) -> float:
    if len(xy) < 4:
        return float("nan")
    mask = (xy[:, 0] >= x_mid - fit_half) & (xy[:, 0] <= x_mid + fit_half)
    pts = xy[mask]
    if len(pts) < 4:
        return float("nan")
    dx_span = float(pts[:, 0].max() - pts[:, 0].min())
    dy_span = float(pts[:, 1].max() - pts[:, 1].min())
    if dx_span < 2.0 or dy_span >= dx_span:
        return float("nan")
    idx_sorted = np.argsort(pts[:, 0])
    x_sorted = pts[idx_sorted, 0]
    if x_sorted[-1] - x_sorted[0] <= 0:
        return float("nan")
    m, _c = np.polyfit(pts[:, 0], pts[:, 1], 1)
    h = float(math.atan(m))
    if abs(h) >= math.pi / 2.0:
        return float("nan")
    return h


def compute_curvature(data: Dict[str, Any], args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    segments = _get_lane_segments(data)
    base_meta = {
        "method": "bev_lane_segment_curvature_aggregate_v1",
        "status": "ok",
        "decision_source": "none",
        "num_segments_total": len(segments),
        "num_segments_in_bev": 0,
        "num_segments_classified": 0,
        "num_straight_segments": 0,
        "num_curve_left_segments": 0,
        "num_curve_right_segments": 0,
        "num_sharp_left_segments": 0,
        "num_sharp_right_segments": 0,
        "num_curve_segments": 0,
        "num_sharp_segments": 0,
        "segment_tag_counts": {},
        "bev_crop_m": {
            "x_min": float(args.topo_bev_x_min),
            "x_max": float(args.topo_bev_x_max),
            "y_min": float(args.topo_bev_y_min),
            "y_max": float(args.topo_bev_y_max),
        },
        "thresholds": {
            "slope_deg": {"straight": 2.0, "angled": float(args.curv_thr_angled_deg)},
            "kappa_m_inv": {"straight": float(args.curv_thr_straight)},
        },
    }

    if not segments:
        return "curvature_unknown", {
            **base_meta,
            "status": "no_lane_segments",
            "decision_source": "no_lane_segments",
        }

    thr_straight_rad = math.radians(2.0)
    thr_angled_rad = math.radians(args.curv_thr_angled_deg)
    thr_kappa_straight = float(args.curv_thr_straight)

    seg_tag_counts: Dict[str, int] = {}
    for seg in segments:
        cropped_xy = _segment_points_in_bev_box(
            seg,
            x_min=float(args.topo_bev_x_min),
            x_max=float(args.topo_bev_x_max),
            y_min=float(args.topo_bev_y_min),
            y_max=float(args.topo_bev_y_max),
        )
        if len(cropped_xy) < 2:
            continue

        base_meta["num_segments_in_bev"] += 1
        classified = _classify_single_polyline_curvature(
            xy=cropped_xy,
            thr_straight_rad=thr_straight_rad,
            thr_angled_rad=thr_angled_rad,
            thr_kappa_straight=thr_kappa_straight,
            resample_ds=args.curv_resample_ds,
            smooth_window=args.curv_smooth_window,
            is_connector=bool(seg.get("is_intersection_or_connector", False)),
        )
        seg_tag = classified.get("curvature_tag", "curvature_unknown")
        seg_tag_counts[seg_tag] = seg_tag_counts.get(seg_tag, 0) + 1

        if seg_tag != "curvature_unknown":
            base_meta["num_segments_classified"] += 1

    base_meta["segment_tag_counts"] = seg_tag_counts
    base_meta["num_straight_segments"] = int(seg_tag_counts.get("straight", 0))
    base_meta["num_curve_left_segments"] = int(seg_tag_counts.get("curve left", 0))
    base_meta["num_curve_right_segments"] = int(seg_tag_counts.get("curve right", 0))
    base_meta["num_sharp_left_segments"] = int(seg_tag_counts.get("sharp left", 0))
    base_meta["num_sharp_right_segments"] = int(seg_tag_counts.get("sharp right", 0))
    base_meta["num_curve_segments"] = (
        base_meta["num_curve_left_segments"] + base_meta["num_curve_right_segments"]
    )
    base_meta["num_sharp_segments"] = (
        base_meta["num_sharp_left_segments"] + base_meta["num_sharp_right_segments"]
    )

    if base_meta["num_segments_in_bev"] == 0:
        return "curvature_unknown", {
            **base_meta,
            "status": "no_segments_in_bev",
            "decision_source": "no_segments_in_bev",
        }

    if base_meta["num_segments_classified"] == 0:
        return "curvature_unknown", {
            **base_meta,
            "status": "no_classifiable_segments_in_bev",
            "decision_source": "no_classifiable_segments_in_bev",
        }

    if base_meta["num_sharp_segments"] > 0:
        return "sharp curve", {
            **base_meta,
            "decision_source": "sharp_segment_present_in_bev",
        }

    if base_meta["num_curve_segments"] > 0:
        return "curve", {
            **base_meta,
            "decision_source": "curved_segment_present_in_bev",
        }

    if base_meta["num_straight_segments"] > 0:
        return "straight", {
            **base_meta,
            "decision_source": "all_classified_segments_straight_in_bev",
        }

    return "curvature_unknown", {
        **base_meta,
        "status": "unresolved_after_aggregation",
        "decision_source": "unresolved_after_aggregation",
    }


def _classify_single_polyline_curvature(
    xy: np.ndarray,
    thr_straight_rad: float,
    thr_angled_rad: float,
    thr_kappa_straight: float,
    resample_ds: float,
    smooth_window: int,
    min_length_m: float = 0.3,
    is_connector: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "curvature_tag": "curvature_unknown",
        "heading_deg": None,
        "heading_rad": None,
        "value_m_inv": None,
        "slope_tag": None,
        "kappa_tag": None,
        "status": "ok",
        "length_m": 0.0,
    }

    if xy is None or xy.ndim != 2 or xy.shape[0] < 2 or xy.shape[1] < 2:
        result["status"] = "invalid_polyline"
        return result

    xy = xy[:, :2].astype(np.float64)
    length_m = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    result["length_m"] = length_m

    if length_m < min_length_m:
        result["status"] = "segment_too_short"
        result["curvature_tag"] = "straight"
        return result

    xy_r = resample_polyline(xy, ds=resample_ds)
    if length_m >= 10.0 and len(xy_r) >= max(5, smooth_window):
        xy_r = smooth_polyline(xy_r, window=smooth_window)
    if len(xy_r) < 2:
        result["status"] = "too_few_samples_after_resample"
        return result

    h_slope = float("nan")
    if len(xy_r) >= 4:
        x_range = float(xy_r[:, 0].max() - xy_r[:, 0].min())
        if x_range >= 1.5:
            x_lo = float(xy_r[:, 0].min())
            x_hi = float(xy_r[:, 0].max())
            x_mid_local = 0.5 * (x_lo + x_hi)
            fit_half = min(5.0, 0.45 * x_range)
            mask = (xy_r[:, 0] >= x_mid_local - fit_half) & (xy_r[:, 0] <= x_mid_local + fit_half)
            pts = xy_r[mask]
            if len(pts) >= 4:
                dx_span = float(pts[:, 0].max() - pts[:, 0].min())
                dy_span = float(pts[:, 1].max() - pts[:, 1].min())
                if dx_span >= 1.5 and dy_span < dx_span:
                    idx_sorted = np.argsort(pts[:, 0])
                    if pts[idx_sorted, 0][-1] - pts[idx_sorted, 0][0] > 0:
                        m, _c = np.polyfit(pts[:, 0], pts[:, 1], 1)
                        h_fit = float(math.atan(m))
                        if abs(h_fit) < math.pi / 2.0:
                            h_slope = h_fit

    if not np.isfinite(h_slope):
        dx = float(xy[-1, 0] - xy[0, 0])
        dy = float(xy[-1, 1] - xy[0, 1])
        if abs(dx) + abs(dy) > 1e-6:
            h_slope = float(math.atan2(dy, dx))

    kappa_arr, _ = curvature_samples(xy_r)

    def _peak_biased(ks: np.ndarray) -> float:
        if len(ks) < 2:
            return float("nan")
        mags = np.abs(ks)
        peak_mag = float(np.percentile(mags, 75))
        if peak_mag < 1e-9:
            return 0.0
        hi_mask = mags >= peak_mag * 0.5
        if not np.any(hi_mask):
            hi_mask = mags >= np.percentile(mags, 50)
        src = ks[hi_mask]
        pos = float(np.sum(src > 0))
        neg = float(np.sum(src < 0))
        sign = 1.0 if pos >= neg else -1.0
        return sign * peak_mag

    k_val = _peak_biased(kappa_arr)

    def _slope_tag(h: float) -> str:
        if not np.isfinite(h):
            return "unknown"
        a = abs(h)
        if a > math.pi / 2.0:
            a = math.pi - a
        if a < thr_straight_rad:
            return "straight"
        if a < thr_angled_rad:
            return "straight with an angle"
        return "curve left" if h > 0 else "curve right"

    def _kappa_tag(k: float) -> str:
        if not np.isfinite(k):
            return "unknown"
        if abs(k) < thr_kappa_straight:
            return "straight"
        return "curve left" if k > 0 else "curve right"

    s_tag = _slope_tag(h_slope)
    k_tag = _kappa_tag(k_val)

    if s_tag == k_tag and s_tag != "unknown":
        final = s_tag
    elif k_tag == "straight" and s_tag == "straight with an angle":
        final = "straight"
    elif k_tag in ("curve left", "curve right") and s_tag in ("straight", "straight with an angle", "unknown"):
        final = k_tag
    elif s_tag in ("curve left", "curve right") and k_tag in ("straight", "unknown"):
        final = s_tag
    elif s_tag != "unknown":
        final = s_tag
    elif k_tag != "unknown":
        final = k_tag
    else:
        final = "straight"

    # Collapse any leftover "straight with an angle" into "straight"
    if final == "straight with an angle":
        final = "straight"

    # Connector / intersection segments: curved connectors are intersection
    # turns — tag as "sharp" instead of "curve".
    if is_connector:
        if final == "curve left":
            final = "sharp left"
        elif final == "curve right":
            final = "sharp right"

    result["curvature_tag"] = final
    result["slope_tag"] = s_tag
    result["kappa_tag"] = k_tag
    result["heading_rad"] = float(h_slope) if np.isfinite(h_slope) else None
    result["heading_deg"] = float(math.degrees(h_slope)) if np.isfinite(h_slope) else None
    result["value_m_inv"] = float(k_val) if np.isfinite(k_val) else None
    return result


def compute_lane_segment_curvature(
    data: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    segments = _get_lane_segments(data)
    if not segments:
        return []

    thr_straight_rad = math.radians(2.0)
    thr_angled_rad = math.radians(args.curv_thr_angled_deg)
    thr_kappa_straight = float(args.curv_thr_straight)

    out: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments):
        seg_id = seg.get("id")
        if seg_id is None:
            seg_id = f"idx_{i}"
        xy = _segment_centerline_xy(seg)
        is_connector = bool(seg.get("is_intersection_or_connector", False))
        classified = _classify_single_polyline_curvature(
            xy=xy,
            thr_straight_rad=thr_straight_rad,
            thr_angled_rad=thr_angled_rad,
            thr_kappa_straight=thr_kappa_straight,
            resample_ds=args.curv_resample_ds,
            smooth_window=args.curv_smooth_window,
            is_connector=is_connector,
        )
        entry = {"id": seg_id, **classified}
        if "is_intersection_or_connector" in seg:
            entry["is_intersection_or_connector"] = is_connector
        out.append(entry)
    return out


# =========================================================
# Topology complexity
# =========================================================

def heading_modes(headings: List[float], mode_sep_deg: float) -> int:
    if not headings:
        return 0
    sep = math.radians(mode_sep_deg)
    modes = []
    for h in headings:
        assigned = False
        for m in modes:
            if abs(angle_diff(h, m)) <= sep:
                assigned = True
                break
        if not assigned:
            modes.append(h)
    return len(modes)


def _get_annotation_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    annotation = data.get("annotation")
    if isinstance(annotation, dict):
        return annotation
    return {}


def _get_lane_segments(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    lane_segments = data.get("lane_segment")
    if not isinstance(lane_segments, list):
        lane_segments = _get_annotation_dict(data).get("lane_segment")
    if isinstance(lane_segments, list):
        return [item for item in lane_segments if isinstance(item, dict)]
    return []


def _to_numpy_matrix(value: Any) -> np.ndarray:
    try:
        arr = np.array(value, dtype=np.float64)
    except Exception:
        return np.zeros((0, 0), dtype=np.float64)
    if arr.ndim != 2:
        return np.zeros((0, 0), dtype=np.float64)
    return arr


def _segment_centerline_xy(segment: Dict[str, Any]) -> np.ndarray:
    centerline = segment.get("centerline")
    if not is_polyline_like(centerline):
        return np.zeros((0, 2), dtype=np.float64)
    return polyline_to_xy(centerline)


def _segment_centerline_xyz(segment: Dict[str, Any]) -> np.ndarray:
    """Return Nx3 array of centerline points. Uses the stored z if available,
    otherwise falls back to z=0 (flat-ground assumption)."""
    centerline = segment.get("centerline")
    if not is_polyline_like(centerline):
        return np.zeros((0, 3), dtype=np.float64)
    arr = np.array(centerline, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return np.zeros((0, 3), dtype=np.float64)
    if arr.shape[1] >= 3:
        return arr[:, :3]
    # Only x,y available — pad with z=0
    return np.column_stack([arr[:, :2], np.zeros(len(arr), dtype=np.float64)])


def _segment_points_in_bev_box(
    segment: Dict[str, Any],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    """Return centerline points that fall inside an ego-frame BEV crop."""
    xy = _segment_centerline_xy(segment)
    if len(xy) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    mask = (
        (xy[:, 0] >= x_min) & (xy[:, 0] <= x_max) &
        (xy[:, 1] >= y_min) & (xy[:, 1] <= y_max)
    )
    return xy[mask]


def _find_ego_segment_idx(segments: List[Dict[str, Any]]) -> Optional[int]:
    """
    Return the index of the lane_segment the ego is currently on, or None if
    ego isn't clearly on any segment.

    Ego is at (0, 0) in ego frame. We pick the segment whose centerline passes
    closest to the origin, with the tangent at the closest point pointing
    roughly forward (|heading| < 60°). Mirrors the logic in build_ego_lane_chain.
    """
    candidates = []
    for idx, seg in enumerate(segments):
        xy = _segment_centerline_xy(seg)
        if len(xy) < 2:
            continue
        dists = np.linalg.norm(xy[:, :2], axis=1)
        closest = int(np.argmin(dists))
        d = float(dists[closest])
        if d > 5.0:
            continue
        i0 = max(0, closest - 1)
        i1 = min(len(xy) - 1, closest + 1)
        if i1 == i0:
            continue
        dx = xy[i1, 0] - xy[i0, 0]
        dy = xy[i1, 1] - xy[i0, 1]
        if abs(dx) + abs(dy) < 1e-6:
            continue
        local_h = math.atan2(dy, dx)
        if abs(local_h) > math.radians(60.0):
            continue
        score = d * 2.0 + abs(local_h)
        candidates.append((score, idx))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


def _lane_segment_graph_stats(data: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Per-frame, ego-centric topology analysis.

    The driving signal is whether the ego is currently on an intersection/connector
    segment, or how close the nearest upcoming connector is along ego's forward path.

    Returns:
      - ego_on_connector: ego's current lane_segment has is_intersection_or_connector=True
      - dist_to_nearest_connector_m: forward distance (along ego +x) from ego origin
          to the closest point on any connector segment's centerline inside the
          BEV crop. None if no connector exists in the crop.
      - ego_segment_idx: which segment ego sits on (None if we can't tell)
      - num_segments_in_bev: debug count of lane segments intersecting the BEV crop
      - num_connectors_in_bev: debug count of connector segments intersecting the BEV crop
            - ego_connector_choice_count: number of related connector choices sharing
                    a predecessor or successor neighborhood with the ego connector
    """
    segments = _get_lane_segments(data)

    x_min = float(args.topo_bev_x_min)
    x_max = float(args.topo_bev_x_max)
    y_min = float(args.topo_bev_y_min)
    y_max = float(args.topo_bev_y_max)

    base = {
        "num_segments": len(segments),
        "bev_crop_m": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        },
        "ego_segment_idx": None,
        "ego_on_connector": False,
        "dist_to_nearest_connector_m": None,
        "num_segments_in_bev": 0,
        "num_connectors_in_bev": 0,
        "ego_connector_choice_count": 0,
        "ego_connector_related_connectors": 0,
        "ego_connector_pred_fanout_max": 0,
        "ego_connector_succ_fanin_max": 0,
        "num_splits_in_bev": 0,
        "num_merges_in_bev": 0,
    }

    if len(segments) == 0:
        return base

    n = len(segments)
    lsls_value = data.get("topology_lsls")
    if lsls_value is None:
        lsls_value = _get_annotation_dict(data).get("topology_lsls", [])
    lsls = _to_numpy_matrix(lsls_value)
    has_graph = lsls.size > 0 and lsls.shape[0] >= n and lsls.shape[1] >= n
    lsls_bin = (lsls[:n, :n] > 0).astype(np.int32) if has_graph else None

    connector_flags = [
        bool(seg.get("is_intersection_or_connector", False)) for seg in segments
    ]

    ego_idx = _find_ego_segment_idx(segments)
    base["ego_segment_idx"] = ego_idx
    if ego_idx is not None:
        base["ego_on_connector"] = connector_flags[ego_idx]

    # Distance to the nearest connector segment inside the ego-frame BEV crop.
    # We still bias the search toward points at or ahead of ego by requiring
    # x >= max(-2m, crop x_min); the -2m slack handles the case where ego is
    # right at the start of a connector.
    min_forward_dist = float("inf")
    n_seg_in_bev = 0
    n_conn_in_bev = 0
    forward_x_min = max(-2.0, x_min)
    in_bev_mask = np.zeros(n, dtype=bool)
    for idx, (seg, is_conn) in enumerate(zip(segments, connector_flags)):
        pts_in_bev = _segment_points_in_bev_box(seg, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        if len(pts_in_bev) == 0:
            continue
        in_bev_mask[idx] = True
        n_seg_in_bev += 1
        if not is_conn:
            continue
        n_conn_in_bev += 1
        # forward mask: points at or ahead of ego, limited to the BEV crop
        fwd = pts_in_bev[pts_in_bev[:, 0] >= forward_x_min]
        if len(fwd) == 0:
            continue
        dists = np.linalg.norm(fwd[:, :2], axis=1)
        d = float(np.min(dists))
        if d < min_forward_dist:
            min_forward_dist = d

    if np.isfinite(min_forward_dist):
        base["dist_to_nearest_connector_m"] = float(min_forward_dist)
    base["num_segments_in_bev"] = n_seg_in_bev
    base["num_connectors_in_bev"] = n_conn_in_bev

    # ------------------------------------------------------------------
    # Split / merge detection within the BEV crop.
    # A split: a non-connector segment in the BEV crop has >1 successor
    #   that also has points in the BEV crop.
    # A merge: a non-connector segment in the BEV crop has >1 predecessor
    #   that also has points in the BEV crop.
    # We only look at non-connector segments because connectors already
    # signal "high" complexity by themselves.
    # ------------------------------------------------------------------
    num_splits_in_bev = 0
    num_merges_in_bev = 0
    if lsls_bin is not None:
        for idx in range(n):
            if not in_bev_mask[idx] or connector_flags[idx]:
                continue
            successors_in_bev = [
                j for j in np.where(lsls_bin[idx] > 0)[0]
                if in_bev_mask[j]
            ]
            predecessors_in_bev = [
                j for j in np.where(lsls_bin[:, idx] > 0)[0]
                if in_bev_mask[j]
            ]
            if len(successors_in_bev) > 1:
                num_splits_in_bev += 1
            if len(predecessors_in_bev) > 1:
                num_merges_in_bev += 1
    base["num_splits_in_bev"] = num_splits_in_bev
    base["num_merges_in_bev"] = num_merges_in_bev

    if ego_idx is not None and base["ego_on_connector"] and lsls_bin is not None and in_bev_mask[ego_idx]:
        pred_fanout_max = 0
        succ_fanin_max = 0
        related_connectors = {int(ego_idx)}

        predecessors = np.where(lsls_bin[:, ego_idx] > 0)[0].tolist()
        successors = np.where(lsls_bin[ego_idx] > 0)[0].tolist()

        for pred_idx in predecessors:
            sibling_connectors = [
                int(conn_idx)
                for conn_idx in np.where(lsls_bin[pred_idx] > 0)[0].tolist()
                if connector_flags[conn_idx] and in_bev_mask[conn_idx]
            ]
            if not sibling_connectors:
                continue
            pred_fanout_max = max(pred_fanout_max, len(sibling_connectors))
            related_connectors.update(sibling_connectors)

        for succ_idx in successors:
            sibling_connectors = [
                int(conn_idx)
                for conn_idx in np.where(lsls_bin[:, succ_idx] > 0)[0].tolist()
                if connector_flags[conn_idx] and in_bev_mask[conn_idx]
            ]
            if not sibling_connectors:
                continue
            succ_fanin_max = max(succ_fanin_max, len(sibling_connectors))
            related_connectors.update(sibling_connectors)

        base["ego_connector_choice_count"] = len(related_connectors)
        base["ego_connector_related_connectors"] = len(related_connectors)
        base["ego_connector_pred_fanout_max"] = pred_fanout_max
        base["ego_connector_succ_fanin_max"] = succ_fanin_max

    return base


def compute_topology_complexity(data: Dict[str, Any], args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    """
    Per-frame topology tagging (3 tiers, evaluated within the ego BEV crop):

    - high:   any is_intersection_or_connector segment is present in the BEV crop
              (intersection / turning lane).  A mix of intersection + split/merge
              also falls here.
    - medium: no connectors in the crop, but at least one non-connector segment
              has a split (fanout>1) or merge (fanin>1) with another BEV-crop
              segment — i.e. road branches or lanes join within the crop.
    - low:    only straight-through segments with no branching in the BEV crop.
    """
    stats = _lane_segment_graph_stats(data, args)

    if stats["num_segments"] == 0:
        return "topology_unknown", {
            "value": None,
            "confidence": 0.0,
            "method": "bev_topology_3tier",
            "status": "no_lane_segments",
            "graph": stats,
        }

    connectors_in_bev = int(stats.get("num_connectors_in_bev", 0))
    splits_in_bev     = int(stats.get("num_splits_in_bev", 0))
    merges_in_bev     = int(stats.get("num_merges_in_bev", 0))
    has_branching     = (splits_in_bev + merges_in_bev) > 0

    if connectors_in_bev > 0:
        tag      = "high topological complexity"
        score    = 1.0
        decision = "connector_present_in_bev"
    elif has_branching:
        tag      = "medium topological complexity"
        score    = 0.5
        decision = "split_or_merge_in_bev"
    else:
        tag      = "low topological complexity"
        score    = 0.0
        decision = "no_connector_no_branching_in_bev"

    meta = {
        "value": float(score),
        "confidence": 1.0 if stats["ego_segment_idx"] is not None else 0.3,
        "method": "bev_topology_3tier",
        "status": "ok",
        "decision_source": decision,
        "graph": stats,
        "classification_thresholds": {
            "bev_crop_m": {
                "x_min": float(args.topo_bev_x_min),
                "x_max": float(args.topo_bev_x_max),
                "y_min": float(args.topo_bev_y_min),
                "y_max": float(args.topo_bev_y_max),
            },
            "high_if_connector_in_bev": True,
            "medium_if_split_or_merge_in_bev": True,
        },
    }
    return tag, meta


# =========================================================
# Lighting
# =========================================================

def compute_lighting(image_path: Path, args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    if cv2 is None:
        return "lighting_unknown", {
            "label": "lighting_unknown",
            "score": None,
            "confidence": 0.0,
            "method": "composite_luminance_contrast_darktail_v1",
            "status": "opencv_not_available",
            "image_path": str(image_path),
        }

    if not image_path.exists():
        return "lighting_unknown", {
            "label": "lighting_unknown",
            "score": None,
            "confidence": 0.0,
            "method": "composite_luminance_contrast_darktail_v1",
            "status": "image_missing",
            "image_path": str(image_path),
        }

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return "lighting_unknown", {
            "label": "lighting_unknown",
            "score": None,
            "confidence": 0.0,
            "method": "composite_luminance_contrast_darktail_v1",
            "status": "image_read_failed",
            "image_path": str(image_path),
        }

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32) / 255.0

    mu = float(np.mean(L))
    p10 = float(np.percentile(L, 10))
    p90 = float(np.percentile(L, 90))
    spread = float(p90 - p10)
    dark_ratio = float(np.mean(L < args.light_dark_thresh))
    sat_low = float(np.mean(L < args.light_sat_low_thresh))

    score = (
        0.40 * (1.0 - mu) +
        0.25 * (1.0 - p10) +
        0.20 * dark_ratio +
        0.15 * (1.0 - spread)
    )
    score = float(clamp(score, 0.0, 1.0))

    label = "poorly lit" if score >= args.light_score_thresh else "well lit"
    confidence = float(clamp(abs(score - args.light_score_thresh) / args.light_conf_band, 0.0, 1.0))

    return label, {
        "label": label,
        "score": score,
        "confidence": confidence,
        "method": "composite_luminance_contrast_darktail_v1",
        "status": "ok",
        "image_path": str(image_path),
        "features": {
            "mean_luminance": mu,
            "p10_luminance": p10,
            "p90_luminance": p90,
            "contrast_spread": spread,
            "dark_ratio": dark_ratio,
            "saturated_dark_ratio": sat_low,
            "dark_thresh": args.light_dark_thresh,
            "sat_low_thresh": args.light_sat_low_thresh,
        },
        "thresholds": {
            "score_thresh_poorly_lit": args.light_score_thresh,
            "confidence_band": args.light_conf_band
        }
    }


def _get_camera_sensor(data: Dict[str, Any], camera_name: str) -> Dict[str, Any]:
    sensor = data.get("sensor")
    if not isinstance(sensor, dict):
        return {}
    camera = sensor.get(camera_name)
    if not isinstance(camera, dict):
        return {}
    return camera


def resolve_image_path(json_path: Path, data: Dict[str, Any], camera_name: str, ext: str) -> Path:
    camera = _get_camera_sensor(data, camera_name)
    sensor_image_path = camera.get("image_path")
    if isinstance(sensor_image_path, str) and sensor_image_path:
        sensor_path = Path(sensor_image_path)
        seq_dir = json_path.parent.parent
        candidates = []
        if sensor_path.is_absolute():
            candidates.append(sensor_path)
        if len(json_path.parents) >= 3:
            candidates.append(json_path.parents[2] / sensor_path)
        candidates.append(seq_dir / "image" / camera_name / sensor_path.name)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[-1]

    stem = json_path.stem
    if stem.endswith("-ls"):
        stem = stem[:-3]
    seq_dir = json_path.parent.parent
    return seq_dir / "image" / camera_name / f"{stem}.{ext}"


# =========================================================
# Occlusion
# =========================================================

# YOLO COCO classes considered vehicles: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

# -----------------------------------------------------------------------
# Fixed calibration fallback for all 7 ring cameras.
# -----------------------------------------------------------------------

# Format: camera_name -> (K 3x3, dist3 [k1,k2,k3], R 3x3, T 3-vec)
FIXED_CALIB: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _build_fixed_calib() -> None:
    K = np.array([
        [1.77753967e+03, 0.00000000e+00, 7.77762878e+02],
        [0.00000000e+00, 1.77753967e+03, 1.01631311e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
    ], dtype=np.float64)
    dist3 = np.array([-0.24479243, -0.19577468, 0.30131747], dtype=np.float64)

    cam_specs = [
        ("ring_front_center",    1.77,   0.00,   1.38,    0.0),
        ("ring_front_left",      1.52,   0.34,   1.38,   55.0),
        ("ring_front_right",     1.52,  -0.34,   1.38,  -55.0),
        ("ring_side_left",       0.00,   0.90,   1.38,   90.0),
        ("ring_side_right",      0.00,  -0.90,   1.38,  -90.0),
        ("ring_rear_left",      -1.50,   0.34,   1.38,  125.0),
        ("ring_rear_right",     -1.50,  -0.34,   1.38, -125.0),
    ]

    for name, tx, ty, tz, yaw_deg in cam_specs:
        T = np.array([tx, ty, tz], dtype=np.float64)
        yaw = math.radians(yaw_deg)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cam_z = np.array([cy,  sy,  0.0])
        cam_x = np.array([sy, -cy,  0.0])
        cam_y = np.array([0.0, 0.0, -1.0])
        R = np.stack([cam_x, cam_y, cam_z], axis=1)

        if name == "ring_front_center":
            R = np.array([
                [-8.48589870e-04,  1.00005773e-02,  9.99949633e-01],
                [-9.99998983e-01, -1.15429954e-03, -8.37087508e-04],
                [ 1.14587004e-03, -9.99949327e-01,  1.00015467e-02],
            ], dtype=np.float64)

        FIXED_CALIB[name] = (K.copy(), dist3.copy(), R, T)


_build_fixed_calib()

RING_CAMERA_NAMES = [
    "ring_front_left",
    "ring_front_center",
    "ring_front_right",
    "ring_side_left",
    "ring_side_right",
    "ring_rear_left",
    "ring_rear_right",
]


def collect_lane_arrays(data: Dict[str, Any]) -> List[np.ndarray]:
    raw: List[List[List[float]]] = []
    collect_polylines(data, raw)
    lanes = []
    for p in raw:
        arr = np.array(p, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
            lanes.append(arr[:, :3] if arr.shape[1] >= 3 else arr[:, :2])
    return lanes


MAX_LANE_RANGE_M = 50.0  # Drop lane points farther than this from ego origin


def project_lane_xyz_to_image_uv(
    lane_xyz: np.ndarray,
    K: np.ndarray,
    dist3: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    if lane_xyz.ndim != 2 or lane_xyz.shape[1] < 3 or lane_xyz.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float64)

    xyz = lane_xyz[:, :3].astype(np.float64)

    # Discard points beyond MAX_LANE_RANGE_M from ego (origin in ego frame).
    # Distant points project near the vanishing point / horizon and produce
    # misleading near-horizontal lines that appear to float in the air.
    ego_dist = np.linalg.norm(xyz, axis=1)
    xyz = xyz.copy()
    xyz[ego_dist > MAX_LANE_RANGE_M] = np.nan

    cam = (R.T @ (xyz - t.reshape(1, 3)).T).T

    x_c, y_c, z_c = cam[:, 0], cam[:, 1], cam[:, 2]
    uv = np.full((xyz.shape[0], 2), np.nan, dtype=np.float64)

    valid = (z_c > 1e-6) & np.isfinite(x_c) & np.isfinite(y_c)
    if not np.any(valid):
        return uv

    x = x_c[valid] / z_c[valid]
    y = y_c[valid] / z_c[valid]

    k1, k2, k3 = dist3.tolist()
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3)
    xd = x * radial
    yd = y * radial

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    uv[valid, 0] = fx * xd + cx
    uv[valid, 1] = fy * yd + cy
    return uv


def get_camera_projection_params(
    data: Dict[str, Any],
    camera_name: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    camera = _get_camera_sensor(data, camera_name)
    intrinsic = camera.get("intrinsic")
    extrinsic = camera.get("extrinsic")

    if isinstance(intrinsic, dict) and isinstance(extrinsic, dict):
        K = np.array(intrinsic.get("K", []), dtype=np.float64)
        distortion = np.array(intrinsic.get("distortion", []), dtype=np.float64).reshape(-1)
        R = np.array(extrinsic.get("rotation", []), dtype=np.float64)
        t = np.array(extrinsic.get("translation", []), dtype=np.float64).reshape(-1)

        if K.shape == (3, 3) and R.shape == (3, 3) and t.shape == (3,) and distortion.size >= 3:
            # Distortion is stored as 3 values [k1, k2, k3] (no tangential p1/p2).
            # OpenCV's 5-param format [k1, k2, p1, p2, k3] is NOT used by this dataset.
            # Previously reading index 4 silently fell back to k3=0.0, causing negative
            # radial values at large angles (side/rear cameras) and wrong projections.
            k1 = distortion[0]
            k2 = distortion[1]
            k3 = distortion[4] if distortion.size >= 5 else distortion[2]
            return K, np.array([k1, k2, k3], dtype=np.float64), R, t

    if camera_name in FIXED_CALIB:
        return FIXED_CALIB[camera_name]

    return None


def project_3d_lanes_to_image(
    lanes_3d: List[np.ndarray],
    img_w: int,
    img_h: int,
    projection_params: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> List[List[np.ndarray]]:
    """Returns a list of groups. Each group is a list of contiguous in-image
    segments that all came from the same original 3-D lane. Callers that need
    to filter by total visible length should sum across the group; callers that
    only need a flat list of polylines can flatten with a list comprehension."""
    out: List[List[np.ndarray]] = []
    for lane in lanes_3d:
        if lane.ndim != 2 or lane.shape[0] < 2 or lane.shape[1] < 3:
            continue
        uv = project_lane_xyz_to_image_uv(lane, *projection_params)
        mask = (
            np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) &
            (uv[:, 0] >= 0) & (uv[:, 0] < img_w) &
            (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
        )
        # Split on gaps so we never interpolate across image boundaries.
        kept_indices = np.where(mask)[0]
        if len(kept_indices) < 2:
            continue
        breaks = np.where(np.diff(kept_indices) > 1)[0] + 1
        group = [uv[seg_idx] for seg_idx in np.split(kept_indices, breaks)
                 if len(seg_idx) >= 2]
        if group:
            out.append(group)
    return out


def resample_polyline_px(xy: np.ndarray, ds_px: float) -> np.ndarray:
    if len(xy) < 2:
        return np.zeros((0, 2), dtype=np.float64)

    xy = remove_duplicate_points(xy)
    if len(xy) < 2:
        return np.zeros((0, 2), dtype=np.float64)

    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = s[-1]
    if total < ds_px:
        return xy.copy()

    s_new = np.arange(0.0, total + 1e-9, ds_px)
    x_new = np.interp(s_new, s, xy[:, 0])
    y_new = np.interp(s_new, s, xy[:, 1])
    return np.stack([x_new, y_new], axis=1)


def detect_vehicle_boxes(
    model: Any,
    image_bgr: np.ndarray,
    conf_thr: float,
    imgsz: int,
    device: str,
) -> List[Tuple[float, float, float, float, float, int]]:
    if model is None:
        return []

    results = model.predict(source=image_bgr, conf=conf_thr, imgsz=imgsz, device=device, verbose=False)
    out: List[Tuple[float, float, float, float, float, int]] = []
    if not results:
        return out

    boxes = results[0].boxes
    if boxes is None:
        return out

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
    conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
    cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.array(boxes.cls).astype(int)

    for b, s, c in zip(xyxy, conf, cls):
        if int(c) not in VEHICLE_CLASS_IDS:
            continue
        x1, y1, x2, y2 = map(float, b)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        out.append((x1, y1, x2, y2, float(s), int(c)))
    return out


def dilate_boxes(
    boxes: List[Tuple[float, float, float, float, float, int]],
    dilate_px: float,
    w: int,
    h: int,
) -> List[Tuple[float, float, float, float, float, int]]:
    out = []
    for x1, y1, x2, y2, s, c in boxes:
        out.append((
            max(0.0, x1 - dilate_px),
            max(0.0, y1 - dilate_px),
            min(float(w - 1), x2 + dilate_px),
            min(float(h - 1), y2 + dilate_px),
            s,
            c,
        ))
    return out


def point_in_any_box(u: float, v: float, boxes: List[Tuple[float, float, float, float, float, int]]) -> bool:
    for x1, y1, x2, y2, _, _ in boxes:
        if x1 <= u <= x2 and y1 <= v <= y2:
            return True
    return False


def compute_occlusion(
    lane_polys_img: List[List[np.ndarray]],
    boxes: List[Tuple[float, float, float, float, float, int]],
    image_w: int,
    image_h: int,
    sample_ds_px: float,
    min_lane_length_px: float,
) -> Dict[str, Any]:
    """lane_polys_img is a list of groups (one per original 3-D lane). Each
    group is a list of contiguous in-image segments as returned by
    project_3d_lanes_to_image. The min_lane_length_px filter is applied to the
    *total* visible length of the group, not each split segment individually."""
    candidate_lanes = len(lane_polys_img)
    in_frame_lanes = 0
    short_lanes = 0
    total_samples = 0
    occ_samples = 0
    lanes_used = 0
    visible_lengths_px: List[float] = []

    for seg_group in lane_polys_img:
        # Sum visible length across all contiguous segments of this lane.
        group_len = sum(
            float(np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1)))
            for s in seg_group if len(s) >= 2
        )
        if group_len == 0.0:
            continue
        in_frame_lanes += 1
        visible_lengths_px.append(group_len)
        if group_len < min_lane_length_px:
            short_lanes += 1
            continue

        lanes_used += 1
        for pts in seg_group:
            if len(pts) < 2:
                continue
            samples = resample_polyline_px(pts, ds_px=sample_ds_px)
            if len(samples) == 0:
                continue
            total_samples += len(samples)
            for u, v in samples:
                if point_in_any_box(float(u), float(v), boxes):
                    occ_samples += 1

    ratio = None if total_samples == 0 else float(occ_samples / total_samples)
    return {
        "ratio": ratio,
        "total_samples": int(total_samples),
        "occluded_samples": int(occ_samples),
        "lanes_used": int(lanes_used),
        "candidate_lanes": int(candidate_lanes),
        "in_frame_lanes": int(in_frame_lanes),
        "short_lanes": int(short_lanes),
        "max_visible_length_px": float(max(visible_lengths_px)) if visible_lengths_px else 0.0,
        "mean_visible_length_px": float(np.mean(visible_lengths_px)) if visible_lengths_px else 0.0,
    }


# BGR colors used in OpenCV drawings
_COLOR_BOX   = (0, 0, 255)
_COLOR_PT_OK = (0, 255, 0)
_COLOR_PT_OC = (0, 128, 255)
_COLOR_TEXT  = (255, 255, 255)


def render_occlusion_debug_frame(
    jpath: Path,
    data: Dict[str, Any],
    occ_tag: str,
    occ_ratio: Optional[float],
    args: argparse.Namespace,
    model: Any,
    out_path: Path,
    viz_cache: Optional[Dict[str, Any]] = None,
    curvature_tag: Optional[str] = None,
    topo_tag: Optional[str] = None,
) -> bool:
    """viz_cache: optional dict of {cam_name: {"boxes": ..., "lane_polys_img": ...}}
    populated by compute_occlusion_for_frame. When provided, YOLO inference and
    lane projection are skipped, ensuring the visualization matches the stored tag."""
    if cv2 is None:
        return False

    lanes_3d = [
        _segment_centerline_xyz(seg)
        for seg in _get_lane_segments(data)
        if len(_segment_centerline_xyz(seg)) >= 2
    ]

    LAYOUT = [
        ("ring_front_left",  "ring_front_center",  "ring_front_right"),
        ("ring_side_left",   None,                  "ring_side_right"),
        ("ring_rear_left",   None,                  "ring_rear_right"),
    ]

    annotated: Dict[str, Optional[np.ndarray]] = {}

    for cam_name in RING_CAMERA_NAMES:
        try:
            img_path = resolve_image_path(jpath, data, camera_name=cam_name, ext=args.image_ext)
        except Exception:
            annotated[cam_name] = None
            continue
        if not img_path.exists():
            annotated[cam_name] = None
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            annotated[cam_name] = None
            continue

        h, w = img.shape[:2]
        canvas = img.copy()

        if viz_cache is not None and cam_name in viz_cache:
            cam_cache = viz_cache[cam_name]
            lane_groups: List[List[np.ndarray]] = cam_cache["lane_polys_img"]
            boxes: List[Tuple[float, float, float, float, float, int]] = cam_cache["boxes"]
        else:
            proj_params = get_camera_projection_params(data, cam_name)
            lane_groups = []
            if proj_params is not None:
                lane_groups = project_3d_lanes_to_image(lanes_3d, w, h, proj_params)
            boxes = []
            if model is not None:
                boxes = detect_vehicle_boxes(
                    model=model, image_bgr=img,
                    conf_thr=args.bbox_score_thr,
                    imgsz=args.det_imgsz,
                    device=args.device,
                )
                boxes = dilate_boxes(boxes, args.bbox_dilate_px, w, h)

        # Flatten groups to a plain list of polylines for drawing.
        lane_polys_flat = [s for group in lane_groups for s in group]

        for x1, y1, x2, y2, score, cls_id in boxes:
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), _COLOR_BOX, 2)
            cv2.putText(canvas, f"{score:.2f}",
                        (int(x1), max(0, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOR_BOX, 1, cv2.LINE_AA)

        cam_total = 0
        cam_occ = 0
        for poly in lane_polys_flat:
            if len(poly) < 2:
                continue
            samples = resample_polyline_px(poly, ds_px=args.sample_ds_px)
            for u, v in samples:
                u_i, v_i = int(round(u)), int(round(v))
                if not (0 <= u_i < w and 0 <= v_i < h):
                    continue
                cam_total += 1
                if point_in_any_box(float(u), float(v), boxes):
                    cam_occ += 1
                    color = _COLOR_PT_OC
                else:
                    color = _COLOR_PT_OK
                cv2.circle(canvas, (u_i, v_i), 3, color, -1)

        cam_ratio = cam_occ / cam_total if cam_total > 0 else None
        ratio_str = f"{cam_ratio:.1%}" if cam_ratio is not None else "n/a"
        label = f"{cam_name} | occ={ratio_str} | {cam_occ}/{cam_total} pts | {len(boxes)} boxes"
        cv2.rectangle(canvas, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(canvas, label, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_TEXT, 1, cv2.LINE_AA)

        annotated[cam_name] = canvas

    cell_h, cell_w = 0, 0
    for img in annotated.values():
        if img is not None:
            cell_h, cell_w = img.shape[:2]
            break
    if cell_h == 0 or cell_w == 0:
        return False

    grid = np.zeros((3 * cell_h, 3 * cell_w, 3), dtype=np.uint8)
    for r, row_cams in enumerate(LAYOUT):
        for c, cam_name in enumerate(row_cams):
            if cam_name is None:
                cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                cv2.putText(cell, "(ego)", (cell_w // 3, cell_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1)
            else:
                img = annotated.get(cam_name)
                if img is None:
                    cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                    cv2.putText(cell, f"{cam_name} (missing)", (10, cell_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
                else:
                    if img.shape[:2] != (cell_h, cell_w):
                        img = cv2.resize(img, (cell_w, cell_h))
                    cell = img
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            grid[y0:y1, x0:x1] = cell

    header_h = 88
    header = np.zeros((header_h, grid.shape[1], 3), dtype=np.uint8)
    ratio_str = f"{occ_ratio:.1%}" if isinstance(occ_ratio, (int, float)) else "n/a"
    occ_color = (0, 0, 220) if occ_tag == "high occlusion" else (0, 180, 0)
    occ_text = (
        f"OCC: {occ_tag.upper()}  |  ratio={ratio_str}  |  "
        f"threshold={args.occlusion_ratio_threshold}  |  {jpath.stem}"
    )
    cv2.putText(header, occ_text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, occ_color, 2, cv2.LINE_AA)
    if curvature_tag is not None:
        curvature_color = (
            (0, 80, 255) if curvature_tag == "sharp curve"
            else (180, 0, 180) if curvature_tag == "curve"
            else (0, 200, 100)
        )
        cv2.putText(header, f"CURV: {curvature_tag.upper()}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, curvature_color, 2, cv2.LINE_AA)
    if topo_tag is not None:
        topo_color = (
            (0, 80, 255) if topo_tag == "high topological complexity"
            else (0, 165, 255) if topo_tag == "medium topological complexity"
            else (0, 200, 100)
        )
        cv2.putText(header, f"TOPO: {topo_tag.upper()}", (10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, topo_color, 2, cv2.LINE_AA)
    composite = np.vstack([header, grid])

    max_w = 4096
    if composite.shape[1] > max_w:
        scale = max_w / composite.shape[1]
        composite = cv2.resize(composite, (max_w, int(composite.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(out_path), composite)


def compute_occlusion_for_frame(
    data: Dict[str, Any],
    jpath: Path,
    args: argparse.Namespace,
    model: Any,
    build_viz_cache: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    base_meta = {
        "value": None,
        "label": "occlusion_unknown",
        "ratio_threshold": args.occlusion_ratio_threshold,
        "confidence": 0.0,
        "method": "multi_camera_lane_overlap",
    }

    if cv2 is None:
        return "occlusion_unknown", {**base_meta, "status": "opencv_not_available"}

    if model is None:
        return "occlusion_unknown", {
            **base_meta, "status": "detector_unavailable",
            "detector_model": args.det_model,
        }

    lanes_3d = [
        _segment_centerline_xyz(seg)
        for seg in _get_lane_segments(data)
        if len(_segment_centerline_xyz(seg)) >= 2
    ]

    total_samples_sum = 0
    occ_samples_sum = 0
    total_boxes = 0
    per_camera: List[Dict[str, Any]] = []
    viz_cache: Dict[str, Any] = {}

    for cam_name in RING_CAMERA_NAMES:
        cam_entry: Dict[str, Any] = {"camera": cam_name}

        try:
            img_path = resolve_image_path(jpath, data, camera_name=cam_name, ext=args.image_ext)
        except Exception as e:
            cam_entry["status"] = f"resolve_error: {e}"
            per_camera.append(cam_entry)
            continue

        if not img_path.exists():
            cam_entry["status"] = "image_missing"
            per_camera.append(cam_entry)
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            cam_entry["status"] = "image_read_failed"
            per_camera.append(cam_entry)
            continue

        h, w = image.shape[:2]

        proj_params = get_camera_projection_params(data, cam_name)
        if proj_params is None:
            cam_entry["status"] = "no_calibration"
            per_camera.append(cam_entry)
            continue

        lane_groups = project_3d_lanes_to_image(lanes_3d, w, h, proj_params)
        if not lane_groups:
            cam_entry["status"] = "no_lanes_in_view"
            cam_entry["projected_lanes"] = 0
            per_camera.append(cam_entry)
            continue

        boxes = detect_vehicle_boxes(
            model=model, image_bgr=image,
            conf_thr=args.bbox_score_thr, imgsz=args.det_imgsz, device=args.device,
        )
        boxes = dilate_boxes(boxes, args.bbox_dilate_px, w, h)

        if build_viz_cache:
            viz_cache[cam_name] = {"boxes": boxes, "lane_polys_img": lane_groups}

        occ_stats = compute_occlusion(
            lane_polys_img=lane_groups, boxes=boxes,
            image_w=w, image_h=h,
            sample_ds_px=args.sample_ds_px,
            min_lane_length_px=args.min_lane_length_px,
        )

        cam_entry.update({
            "status": "ok",
            "projected_lanes": len(lane_groups),
            "lanes_used": occ_stats["lanes_used"],
            "total_samples": occ_stats["total_samples"],
            "occluded_samples": occ_stats["occluded_samples"],
            "vehicle_boxes": len(boxes),
            "ratio": occ_stats["ratio"],
        })

        total_samples_sum += occ_stats["total_samples"]
        occ_samples_sum += occ_stats["occluded_samples"]
        total_boxes += len(boxes)
        per_camera.append(cam_entry)

    if total_samples_sum < args.min_total_samples:
        tag = "low occlusion"
        status = "insufficient_lane_samples"
        ratio: Optional[float] = (
            occ_samples_sum / total_samples_sum if total_samples_sum > 0 else None
        )
    else:
        ratio = occ_samples_sum / total_samples_sum
        tag = "high occlusion" if ratio >= args.occlusion_ratio_threshold else "low occlusion"
        status = "ok"

    confidence = float(clamp(
        total_samples_sum / float(max(1, args.min_total_samples * 3)), 0.0, 1.0
    )) if total_samples_sum > 0 else 0.0

    meta = {
        **base_meta,
        "value": float(ratio) if ratio is not None else None,
        "label": tag,
        "confidence": confidence,
        "status": status,
        "total_samples": total_samples_sum,
        "occluded_samples": occ_samples_sum,
        "total_vehicle_boxes": total_boxes,
        "num_3d_lanes": len(lanes_3d),
        "per_camera": per_camera,
        "_viz_cache": viz_cache if build_viz_cache else None,
    }
    return tag, meta


# =========================================================
# JSON update helpers
# =========================================================

FRAME_CURVATURE_TAGS = {
    "curvature_unknown",
    "straight",
    "curve",
    "sharp curve",
}

LEGACY_FRAME_CURVATURE_TAGS = {
    "curves",
    "sharp curves",
    "curve left",
    "curve right",
    "sharp left",
    "sharp right",
    "straight with an angle",
    "shape_curves",
    "sharp_curves",
    "low curvature left",
    "low curvature right",
    "medium curvature left",
    "medium curvature right",
    "high curvature left",
    "high curvature right",
}

TOPOLOGY_TAGS = {
    "topology_unknown",
    "low topological complexity",
    "medium topological complexity",
    "high topological complexity",
}

LEGACY_TOPOLOGY_TAGS: set = set()

LIGHTING_TAGS = {
    "well lit",
    "poorly lit",
    "lighting_unknown",
}

OCCLUSION_TAGS = {
    "low occlusion",
    "high occlusion",
    "occlusion_unknown",
}


def upsert_tag_family(tags: List[Any], new_tag: str, family_set: set) -> List[Any]:
    kept = []
    for t in tags:
        if isinstance(t, str) and t.strip().lower() in family_set:
            continue
        kept.append(t)
    kept.append(new_tag)

    out = []
    seen = set()
    for t in kept:
        key = t.strip().lower() if isinstance(t, str) else str(t)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


CURVATURE_TAG_COLORS = {
    "straight": "#2ca02c",
    "curve": "#9467bd",
    "sharp curve": "#7f2704",
    "curve left": "#1f77b4",
    "curve right": "#d62728",
    "sharp left": "#0a2d5a",    # dark blue
    "sharp right": "#5c0505",   # dark red
    "curvature_unknown": "#888888",
}

RING_CAMERA_LAYOUT = [
    (0, 0, "ring_front_left",  "FRONT LEFT"),
    (0, 1, "ring_front_center","FRONT CENTER"),
    (0, 2, "ring_front_right", "FRONT RIGHT"),
    (1, 0, "ring_side_left",   "SIDE LEFT"),
    (1, 2, "ring_side_right",  "SIDE RIGHT"),
    (2, 0, "ring_rear_left",   "REAR LEFT"),
    (2, 2, "ring_rear_right",  "REAR RIGHT"),
]


def visualize_frame(
    image_path: Path,
    data: Dict[str, Any],
    frame_curv_tag: str,
    frame_curv_meta: Dict[str, Any],
    per_segment: List[Dict[str, Any]],
    output_path: Path,
    jpath: Optional[Path] = None,
    camera_ext: str = "jpg",
) -> bool:
    if plt is None or cv2 is None:
        return False

    def _load_rgb(path: Path) -> Optional[np.ndarray]:
        if path is None or not path.exists():
            return None
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    segments = _get_lane_segments(data)
    seg_tag_by_id: Dict[Any, str] = {}
    for entry in per_segment:
        seg_tag_by_id[entry.get("id")] = entry.get("curvature_tag", "curvature_unknown")

    banner_color = CURVATURE_TAG_COLORS.get(frame_curv_tag, "#888888")

    fig = plt.figure(figsize=(20, 12), dpi=100)
    gs = fig.add_gridspec(
        nrows=3, ncols=4,
        width_ratios=[1.0, 1.0, 1.0, 2.2],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.08, hspace=0.15,
    )

    subtitle_parts = [
        f"bev_segs={frame_curv_meta.get('num_segments_in_bev', 0)}",
        f"classified={frame_curv_meta.get('num_segments_classified', 0)}",
        f"curves={frame_curv_meta.get('num_curve_segments', 0)}",
        f"sharp={frame_curv_meta.get('num_sharp_segments', 0)}",
        f"decision={frame_curv_meta.get('decision_source', '-')}",
    ]
    fig.suptitle(
        f"FRAME: {frame_curv_tag.upper()}    |    {' | '.join(subtitle_parts)}",
        color=banner_color, fontsize=15, fontweight="bold", y=0.995,
    )

    for row, col, cam_name, label in RING_CAMERA_LAYOUT:
        ax = fig.add_subplot(gs[row, col])
        cam_img = None
        if jpath is not None:
            try:
                p = resolve_image_path(jpath, data, camera_name=cam_name, ext=camera_ext)
                cam_img = _load_rgb(p)
            except Exception:
                cam_img = None
        if cam_img is None and image_path is not None:
            try:
                guess = image_path.parent.parent / cam_name / image_path.name
                cam_img = _load_rgb(guess)
            except Exception:
                cam_img = None

        if cam_img is not None:
            ax.imshow(cam_img)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_facecolor("#f0f0f0")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=9, pad=3)
        is_front_center = (cam_name == "ring_front_center")
        for spine in ax.spines.values():
            spine.set_edgecolor(banner_color if is_front_center else "#cccccc")
            spine.set_linewidth(3.5 if is_front_center else 1.0)

    ax_bev = fig.add_subplot(gs[:, 3])

    bev_x_min, bev_x_max = -5.0, 55.0
    bev_half_lat = 25.0

    for i, seg in enumerate(segments):
        xy = _segment_centerline_xy(seg)
        if len(xy) < 2:
            continue
        seg_id = seg.get("id")
        if seg_id is None:
            seg_id = f"idx_{i}"
        tag = seg_tag_by_id.get(seg_id, "curvature_unknown")
        color = CURVATURE_TAG_COLORS.get(tag, "#888888")
        disp_x = -xy[:, 1]
        disp_y = xy[:, 0]
        ax_bev.plot(disp_x, disp_y, color=color, linewidth=1.6, alpha=0.88, zorder=2)

    chain_xy, _chain_info = build_ego_lane_chain(data, max_length_m=40.0)
    if len(chain_xy) >= 2:
        ax_bev.plot(
            -chain_xy[:, 1], chain_xy[:, 0],
            color="#ffcc00", linewidth=3.0, alpha=0.75,
            linestyle=(0, (6, 4)),
            zorder=3,
            label="ego chain (frame tag)",
        )

    if Rectangle is not None:
        ego_rect = Rectangle((-0.9, -2.0), 1.8, 4.0, linewidth=1.2,
                             edgecolor="black", facecolor="#333333", zorder=5)
        ax_bev.add_patch(ego_rect)
        ax_bev.annotate("", xy=(0, 3.5), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                        zorder=6)

    ax_bev.set_xlim(-bev_half_lat, bev_half_lat)
    ax_bev.set_ylim(bev_x_min, bev_x_max)
    ax_bev.set_aspect("equal")
    ax_bev.grid(True, alpha=0.3, zorder=0)
    ax_bev.set_xlabel("← left     lateral (m)     right →")
    ax_bev.set_ylabel("forward (m)")
    ax_bev.set_title(f"BEV lane segments ({len(per_segment)} tagged)", fontsize=11)

    if Line2D is not None:
        legend_items = [
            Line2D([0], [0], color=col, lw=2.5, label=tag)
            for tag, col in CURVATURE_TAG_COLORS.items()
        ]
        legend_items.append(Line2D([0], [0], color="#ffcc00", lw=3.0,
                                   linestyle=(0, (6, 4)), label="ego chain"))
        ax_bev.legend(handles=legend_items, loc="lower right", fontsize=9,
                      framealpha=0.85)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(str(output_path), dpi=100, bbox_inches="tight")
    except Exception as e:
        print(f"[WARN] Failed to save viz {output_path}: {e}")
        plt.close(fig)
        return False
    plt.close(fig)
    return True


def update_json_fields(
    data: Dict[str, Any],
    curvature_tag: str,
    curvature_meta: Dict[str, Any],
    topo_tag: str,
    topo_meta: Dict[str, Any],
    lighting_tag: str,
    lighting_meta: Dict[str, Any],
    occlusion_tag: str,
    occlusion_meta: Dict[str, Any],
    lane_segment_curvature: Optional[List[Dict[str, Any]]] = None,
) -> None:
    if "scenario_tags" not in data or not isinstance(data["scenario_tags"], list):
        data["scenario_tags"] = []

    tags = data["scenario_tags"]
    tags = upsert_tag_family(tags, curvature_tag, FRAME_CURVATURE_TAGS | LEGACY_FRAME_CURVATURE_TAGS)
    tags = upsert_tag_family(tags, topo_tag, TOPOLOGY_TAGS | LEGACY_TOPOLOGY_TAGS)
    tags = upsert_tag_family(tags, lighting_tag, LIGHTING_TAGS)
    tags = upsert_tag_family(tags, occlusion_tag, OCCLUSION_TAGS)
    data["scenario_tags"] = tags

    if "scenario_meta" not in data or not isinstance(data["scenario_meta"], dict):
        data["scenario_meta"] = {}

    data["scenario_meta"]["curvature"] = curvature_meta
    data["scenario_meta"]["topology_complexity"] = topo_meta
    data["scenario_meta"]["lighting"] = lighting_meta
    data["scenario_meta"]["occlusion"] = occlusion_meta

    if lane_segment_curvature:
        data["scenario_meta"]["lane_segments"] = lane_segment_curvature


# =========================================================
# Batch processing
# =========================================================

def find_json_files(root: Path) -> List[Path]:
    """Find all *-ls.json files. Naturally skips folders (e.g. 11xxx) that have none."""
    return sorted(p for p in root.rglob("*-ls.json") if p.is_file())


def load_topology_source_data(jpath: Path, frame_data: Dict[str, Any]) -> Dict[str, Any]:
    # jpath is always a -ls.json file; return as-is.
    return frame_data


def process_file(
    jpath: Path,
    args: argparse.Namespace,
    detector_model: Any,
    debug_viz_dir: Optional[Path] = None,
) -> Tuple[bool, Dict[str, str]]:
    try:
        with jpath.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, {"error": f"read_error: {e}"}

    if not isinstance(data, dict):
        return False, {"error": "root_not_dict"}

    try:
        topology_data = load_topology_source_data(jpath, data)

        frame_curv_tag, frame_curv_meta = compute_curvature(topology_data, args)
        topo_tag, topo_meta = compute_topology_complexity(topology_data, args)

        img_path = resolve_image_path(jpath, topology_data, camera_name=args.camera_name, ext=args.image_ext)
        light_tag, light_meta = compute_lighting(img_path, args)
        occ_tag, occ_meta = compute_occlusion_for_frame(
            topology_data, jpath, args, detector_model,
            build_viz_cache=debug_viz_dir is not None,
        )

        per_segment: List[Dict[str, Any]] = []
        if _get_lane_segments(topology_data):
            per_segment = compute_lane_segment_curvature(topology_data, args)

        # --- Debug occlusion viz ---
        if debug_viz_dir is not None:
            stem = jpath.stem
            if stem.endswith("-ls"):
                stem = stem[:-3]
            occ_viz_path = debug_viz_dir / f"{stem}_occlusion.png"
            occ_ratio = occ_meta.get("value")
            success = render_occlusion_debug_frame(
                jpath=jpath,
                data=topology_data,
                occ_tag=occ_tag,
                occ_ratio=occ_ratio,
                args=args,
                model=detector_model,
                out_path=occ_viz_path,
                viz_cache=occ_meta.get("_viz_cache"),
                curvature_tag=frame_curv_tag,
                topo_tag=topo_tag,
            )
            if not success:
                print(f"[WARN] Occlusion debug viz failed for {jpath.stem}")

        # --- Curvature BEV viz ---
        if args.viz_output_dir and per_segment:
            viz_dir = Path(args.viz_output_dir)
            stem = jpath.stem
            if stem.endswith("-ls"):
                stem = stem[:-3]
            viz_path = viz_dir / f"{stem}.png"
            visualize_frame(
                image_path=img_path,
                data=data,
                frame_curv_tag=frame_curv_tag,
                frame_curv_meta=frame_curv_meta,
                per_segment=per_segment,
                output_path=viz_path,
                jpath=jpath,
                camera_ext=args.image_ext,
            )

        # Topology debug: short label and dist-to-connector for quick visual eyeballing
        topo_graph = topo_meta.get("graph") or {}
        topo_dist = topo_graph.get("dist_to_nearest_connector_m")
        topo_dist_str = f"{topo_dist:5.1f}m" if isinstance(topo_dist, (int, float)) else "  n/a"
        topo_short = {
            "high topological complexity": "HIGH",
            "medium topological complexity": "MED ",
            "low topological complexity": "LOW ",
            "topology_unknown": "UNK ",
        }.get(topo_tag, "??? ")

        print(
            f"curv={frame_curv_tag} | segs_tagged={len(per_segment)} | occ={occ_tag} | "
            f"topo={topo_short} dist={topo_dist_str} | img={img_path}"
        )

        # Per-segment curvature breakdown
        if per_segment:
            seg_tally: Dict[str, int] = {}
            for entry in per_segment:
                t = entry.get("curvature_tag", "curvature_unknown")
                seg_tally[t] = seg_tally.get(t, 0) + 1
            tally_str = "  ".join(f"{t}={n}" for t, n in sorted(seg_tally.items(), key=lambda kv: -kv[1]))
            print(f"  [LANE_SEGS] {tally_str}")
            # Print each segment's id, tag, connector flag, heading
            for entry in per_segment:
                seg_id = entry.get("id", "?")
                seg_tag = entry.get("curvature_tag", "?")
                seg_conn = "conn" if entry.get("is_intersection_or_connector") else "road"
                seg_hdeg = entry.get("heading_deg")
                seg_h_str = f"{seg_hdeg:+.1f}°" if isinstance(seg_hdeg, (int, float)) else "n/a"
                seg_len = entry.get("length_m", 0.0)
                print(f"    {seg_id:>12s}  {seg_tag:22s}  {seg_conn:4s}  h={seg_h_str:>8s}  len={seg_len:.1f}m")

        update_json_fields(
            data=data,
            curvature_tag=frame_curv_tag,
            curvature_meta=frame_curv_meta,
            topo_tag=topo_tag,
            topo_meta=topo_meta,
            lighting_tag=light_tag,
            lighting_meta=light_meta,
            occlusion_tag=occ_tag,
            occlusion_meta=occ_meta,
            lane_segment_curvature=per_segment,
        )

        if not args.dry_run:
            with jpath.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        return True, {
            "curvature": frame_curv_tag,
            "topology": topo_tag,
            "lighting": light_tag,
            "occlusion": occ_tag,
            "lane_segment_curvature_counts": _tally_segment_tags(per_segment),
            "lane_segment_unknown_status": _tally_unknown_status(per_segment),
        }
    except Exception as e:
        import traceback
        return False, {"error": f"proc_error: {e}\n{traceback.format_exc()}"}


def _tally_segment_tags(per_segment: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entry in per_segment:
        tag = entry.get("curvature_tag", "curvature_unknown")
        counts[tag] = counts.get(tag, 0) + 1
    return counts


def _tally_unknown_status(per_segment: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entry in per_segment:
        if entry.get("curvature_tag") == "curvature_unknown":
            s = entry.get("status", "unknown_status")
            counts[s] = counts.get(s, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true")

    # --- Debug / test mode ---
    parser.add_argument("--debug_viz_n", type=int, default=0,
                        help="If > 0: process only this many frames and write per-camera "
                             "occlusion debug images (YOLO boxes + lane sample points) "
                             "to --debug_viz_dir. Useful to validate the pipeline before "
                             "a full run. Implies --dry_run.")
    parser.add_argument("--debug_viz_dir", type=str, default="./debug_viz_occlusion",
                        help="Output directory for occlusion debug images when "
                             "--debug_viz_n is set.")

    # image settings
    parser.add_argument("--camera_name", type=str, default="ring_front_center")
    parser.add_argument("--image_ext", type=str, default="jpg")

    # curvature params
    parser.add_argument("--curv_forward_min", type=float, default=5.0)
    parser.add_argument("--curv_forward_max", type=float, default=40.0)
    parser.add_argument("--curv_lateral_max", type=float, default=20.0)
    parser.add_argument("--curv_resample_ds", type=float, default=0.5)
    parser.add_argument("--curv_smooth_window", type=int, default=5)
    parser.add_argument("--curv_thr_straight", type=float, default=0.003)
    parser.add_argument("--curv_thr_angled_deg", type=float, default=8.0)
    parser.add_argument("--curv_ego_chain_max_m", type=float, default=40.0)
    parser.add_argument("--curv_ego_chain_min_m", type=float, default=15.0)

    # curvature BEV visualization
    parser.add_argument("--viz_output_dir", type=str, default=None)

    # topology params
    parser.add_argument("--topo_bev_x_min", type=float, default=-25.0,
                        help="Topology BEV crop min x (m) in ego frame.")
    parser.add_argument("--topo_bev_x_max", type=float, default=25.0,
                        help="Topology BEV crop max x (m) in ego frame.")
    parser.add_argument("--topo_bev_y_min", type=float, default=-25.0,
                        help="Topology BEV crop min y (m) in ego frame.")
    parser.add_argument("--topo_bev_y_max", type=float, default=25.0,
                        help="Topology BEV crop max y (m) in ego frame.")
    # lighting params
    parser.add_argument("--light_dark_thresh", type=float, default=0.16)
    parser.add_argument("--light_sat_low_thresh", type=float, default=0.04)
    parser.add_argument("--light_score_thresh", type=float, default=0.55)
    parser.add_argument("--light_conf_band", type=float, default=0.20)

    # occlusion / detector params
    parser.add_argument("--det_model", type=str, default="yolov8n.pt")
    parser.add_argument("--det_imgsz", type=int, default=960)
    parser.add_argument("--bbox_score_thr", type=float, default=0.25,
                        help="YOLO confidence threshold for vehicle detections.")
    parser.add_argument("--bbox_dilate_px", type=float, default=2.0,
                        help="Dilate YOLO boxes by this many pixels before overlap check.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sample_ds_px", type=float, default=3.0,
                        help="Spacing (px) between lane sample points for overlap check.")
    parser.add_argument("--min_lane_length_px", type=float, default=30.0,
                        help="Ignore projected lane polylines shorter than this (px).")
    parser.add_argument("--min_total_samples", type=int, default=40,
                        help="Min pooled lane samples across all cameras to produce a tag. "
                             "Below this, defaults to 'low occlusion'.")
    parser.add_argument("--occlusion_ratio_threshold", type=float, default=0.75,
                        help="Fraction of lane samples occluded (across all cameras) at or "
                             "above which the frame is tagged 'high occlusion'.")

    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"
        print(f"[INFO] --device auto resolved to: {args.device}")

    debug_mode = args.debug_viz_n > 0
    if debug_mode:
        args.dry_run = True
        print(f"[INFO] Debug mode: processing first {args.debug_viz_n} frames, "
              f"writing occlusion viz to: {args.debug_viz_dir}")

    root = Path(args.dataset_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Invalid dataset_root: {root}")

    files = find_json_files(root)
    if debug_mode:
        files = files[:args.debug_viz_n]
        print(f"[INFO] Debug mode: capped to {len(files)} files.")
    else:
        print(f"[INFO] Found {len(files)} -ls.json files under: {root}")

    if cv2 is None:
        print("[WARN] OpenCV (cv2) not installed — lighting will be 'lighting_unknown'.")
    if YOLO is None:
        print("[WARN] Ultralytics not installed — occlusion will be 'occlusion_unknown'.")
    if args.viz_output_dir:
        if plt is None:
            print("[WARN] matplotlib not installed — --viz_output_dir will be ignored.")
        else:
            print(f"[INFO] Curvature BEV visualizations -> {args.viz_output_dir}")

    detector_model = None
    if YOLO is not None:
        try:
            detector_model = YOLO(args.det_model)
            print(f"[INFO] Loaded detector: {args.det_model}")
        except Exception as e:
            print(f"[WARN] Failed to load detector '{args.det_model}': {e}. Occlusion will be unknown.")

    debug_viz_dir: Optional[Path] = Path(args.debug_viz_dir) if debug_mode else None

    ok, fail = 0, 0
    frame_curv_counts: Dict[str, int] = {}
    topo_counts: Dict[str, int] = {}
    light_counts: Dict[str, int] = {}
    occ_counts: Dict[str, int] = {}
    seg_curv_counts: Dict[str, int] = {}
    seg_unknown_status: Dict[str, int] = {}
    total_segments_tagged = 0

    for i, fp in enumerate(files, 1):
        success, result = process_file(fp, args, detector_model, debug_viz_dir=debug_viz_dir)
        if success:
            ok += 1
            c = result["curvature"]
            t = result["topology"]
            l = result["lighting"]
            o = result["occlusion"]
            frame_curv_counts[c] = frame_curv_counts.get(c, 0) + 1
            topo_counts[t] = topo_counts.get(t, 0) + 1
            light_counts[l] = light_counts.get(l, 0) + 1
            occ_counts[o] = occ_counts.get(o, 0) + 1
            for tag, n in (result.get("lane_segment_curvature_counts") or {}).items():
                seg_curv_counts[tag] = seg_curv_counts.get(tag, 0) + n
                total_segments_tagged += n
            for st, n in (result.get("lane_segment_unknown_status") or {}).items():
                seg_unknown_status[st] = seg_unknown_status.get(st, 0) + n
        else:
            fail += 1
            print(f"[WARN] {fp}: {result.get('error', 'unknown_error')}")

        if i % 200 == 0 or i == len(files):
            print(f"[PROGRESS] {i}/{len(files)} | ok={ok} fail={fail}")
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    print("\n=== Summary ===")
    print(f"Total files: {len(files)}")
    print(f"Success:     {ok}")
    print(f"Failed:      {fail}")

    print("\nFrame curvature tags:")
    for k, v in sorted(frame_curv_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:28s} : {v}")

    print("\nTopology tags:")
    for k, v in sorted(topo_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:28s} : {v}")

    print("\nLighting tags:")
    for k, v in sorted(light_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:28s} : {v}")

    print("\nOcclusion tags:")
    for k, v in sorted(occ_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:28s} : {v}")

    if total_segments_tagged > 0:
        print(f"\nLane-segment curvature tags (total {total_segments_tagged} segments):")
        for k, v in sorted(seg_curv_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            pct = 100.0 * v / total_segments_tagged
            print(f"  {k:28s} : {v} ({pct:.1f}%)")
        if seg_unknown_status:
            unk_total = sum(seg_unknown_status.values())
            print(f"\n  Unknown breakdown ({unk_total} segments):")
            for st, n in sorted(seg_unknown_status.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"    {st:28s} : {n}")

    if args.dry_run:
        print("\n[NOTE] Dry run: no files were modified.")
    if debug_mode and debug_viz_dir:
        print(f"\n[INFO] Occlusion debug images written to: {debug_viz_dir}")
        print("       Green dots = unoccluded lane samples")
        print("       Orange dots = occluded lane samples (inside vehicle box)")
        print("       Red boxes = detected vehicles")


if __name__ == "__main__":
    main()