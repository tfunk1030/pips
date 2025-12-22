import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import cv2
import numpy as np

@dataclass
class Config:
    # You set these ONCE per display scale:
    board_crop: Tuple[int, int, int, int]  # x, y, w, h (crop that contains ONLY the grid)
    rows: int
    cols: int
    samples_per_cell: int = 9  # 3x3 samples


def apply_meanshift_filtering(
    img_bgr: np.ndarray,
    spatial_radius: int = 10,
    color_radius: int = 20
) -> np.ndarray:
    """
    Apply pyrMeanShiftFiltering for edge-preserving color smoothing.

    This preprocessing step makes colors more uniform within regions while
    preserving edges between regions, improving subsequent clustering accuracy.

    Args:
        img_bgr: Input image in BGR format (OpenCV default)
        spatial_radius: Spatial window radius - controls how far spatially to consider
        color_radius: Color window radius - controls color difference tolerance

    Returns:
        Filtered image with smoothed colors and preserved edges
    """
    # pyrMeanShiftFiltering requires 8-bit, 3-channel image
    if img_bgr.dtype != np.uint8:
        img_bgr = img_bgr.astype(np.uint8)

    filtered = cv2.pyrMeanShiftFiltering(
        img_bgr,
        sp=spatial_radius,
        sr=color_radius
    )
    return filtered

def sample_cell_color(img_bgr: np.ndarray, r: int, c: int, rows: int, cols: int, samples: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    cell_w = w / cols
    cell_h = h / rows

    # sample a small grid of points in the middle of the cell to avoid grid lines
    s = int(math.sqrt(samples))
    s = max(2, s)
    xs = np.linspace((c + 0.3) * cell_w, (c + 0.7) * cell_w, s)
    ys = np.linspace((r + 0.3) * cell_h, (r + 0.7) * cell_h, s)

    pts = []
    for yy in ys:
        for xx in xs:
            pts.append(img_bgr[int(yy), int(xx)].astype(np.float32))

    pts = np.array(pts, dtype=np.float32)
    return pts.mean(axis=0)

def kmeans_cluster(colors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    # colors: (N, 3)
    Z = colors.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return labels.flatten(), centers

def build_regions(labels: np.ndarray, rows: int, cols: int) -> Dict[int, List[Tuple[int,int]]]:
    # labels indexed by r*cols+c
    regions: Dict[int, List[Tuple[int,int]]] = {}
    for r in range(rows):
        for c in range(cols):
            lab = int(labels[r*cols + c])
            regions.setdefault(lab, []).append((r, c))
    return regions

def regions_to_yaml_like(regions: Dict[int, List[Tuple[int,int]]], centers_bgr: np.ndarray) -> Dict:
    # Output a structure you can easily map into your existing pips_puzzle.yaml format
    out = {
        "grid": {
            "rows": None,
            "cols": None,
        },
        "regions": []
    }
    for rid, cells in sorted(regions.items(), key=lambda x: len(x[1]), reverse=True):
        bgr = centers_bgr[rid].tolist()
        out["regions"].append({
            "id": rid,
            "approx_color_bgr": [round(x, 1) for x in bgr],
            "cells": [{"r": r, "c": c} for (r, c) in cells],
            # fill these manually (or OCR later)
            "constraint": {"op": None, "value": None},
        })
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("screenshot", help="Full screenshot PNG/JPG")
    ap.add_argument("--config", default="shot_config.json", help="Config JSON with crop + rows/cols")
    ap.add_argument("--k", type=int, default=None, help="Number of color regions; if omitted, guesses")
    ap.add_argument("--out", default="regions_extracted.json", help="Output json")
    ap.add_argument("--spatial-radius", type=int, default=10,
                    help="Spatial window radius for meanshift filtering (default: 10)")
    ap.add_argument("--color-radius", type=int, default=20,
                    help="Color window radius for meanshift filtering (default: 20)")
    ap.add_argument("--no-preprocess", action="store_true",
                    help="Disable pyrMeanShiftFiltering preprocessing")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg = Config(**cfg)

    img = cv2.imread(args.screenshot, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("Could not read screenshot")

    x, y, w, h = cfg.board_crop
    board = img[y:y+h, x:x+w].copy()

    # Apply pyrMeanShiftFiltering for edge-preserving color smoothing
    # This makes colors more uniform within regions while preserving edges,
    # improving clustering accuracy for complex layouts
    if not args.no_preprocess:
        board = apply_meanshift_filtering(
            board,
            spatial_radius=args.spatial_radius,
            color_radius=args.color_radius
        )

    # Collect one representative color per cell
    colors = []
    for r in range(cfg.rows):
        for c in range(cfg.cols):
            colors.append(sample_cell_color(board, r, c, cfg.rows, cfg.cols, cfg.samples_per_cell))
    colors = np.array(colors, dtype=np.float32)

    # Guess k if not provided: try a small range and pick elbow-ish by inertia
    if args.k is None:
        best_k = 8
        best_score = None
        for k in range(6, 18):
            labels, centers = kmeans_cluster(colors, k)
            # compute compactness
            diffs = colors - centers[labels]
            score = float((diffs * diffs).sum())
            if best_score is None or score < best_score:
                best_score = score
                best_k = k
        k = best_k
    else:
        k = args.k

    labels, centers = kmeans_cluster(colors, k)
    regions = build_regions(labels, cfg.rows, cfg.cols)

    out = regions_to_yaml_like(regions, centers)
    out["grid"]["rows"] = cfg.rows
    out["grid"]["cols"] = cfg.cols

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} (k={k}, regions={len(regions)})")

if __name__ == "__main__":
    main()
