import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import cv2
import numpy as np
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class ClusteringConfidence:
    """
    Confidence metrics for screenshot clustering results.

    Attributes:
        overall: Overall confidence score (0.0 to 1.0)
        silhouette: Silhouette score (-1.0 to 1.0, higher is better)
        intra_cluster_variance: Average variance within clusters
        method_confidence: Confidence based on clustering method used
    """
    overall: float = 0.0
    silhouette: float = 0.0
    intra_cluster_variance: float = 0.0
    method_confidence: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall": round(self.overall, 4),
            "silhouette": round(self.silhouette, 4),
            "intra_cluster_variance": round(self.intra_cluster_variance, 4),
            "method_confidence": round(self.method_confidence, 4)
        }


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
    """
    Cluster colors using KMeans algorithm.

    Args:
        colors: Array of colors (N, 3)
        k: Number of clusters

    Returns:
        Tuple of (labels, centers)
    """
    Z = colors.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return labels.flatten(), centers


def dbscan_cluster(
    colors: np.ndarray,
    eps: float = 15.0,
    min_samples: int = 2
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Cluster colors using DBSCAN algorithm.

    DBSCAN automatically determines the number of clusters based on color density.

    Args:
        colors: Array of colors (N, 3) in LAB format
        eps: Maximum distance between samples to be considered in same neighborhood
        min_samples: Minimum samples in neighborhood to form a core point

    Returns:
        Tuple of (labels, cluster_centers, n_clusters)
    """
    colors_float = colors.astype(np.float32)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(colors_float)

    unique_labels = set(labels)
    unique_labels.discard(-1)
    n_clusters = len(unique_labels)

    if n_clusters > 0:
        centers = np.zeros((n_clusters, colors.shape[1]), dtype=np.float32)
        for i, label in enumerate(sorted(unique_labels)):
            mask = labels == label
            centers[i] = colors_float[mask].mean(axis=0)
    else:
        centers = np.zeros((0, colors.shape[1]), dtype=np.float32)

    return labels, centers, n_clusters


def meanshift_cluster(
    colors: np.ndarray,
    quantile: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Cluster colors using MeanShift algorithm.

    MeanShift automatically determines the number of clusters based on density modes.

    Args:
        colors: Array of colors (N, 3) in LAB format
        quantile: Quantile for bandwidth estimation

    Returns:
        Tuple of (labels, cluster_centers, n_clusters)
    """
    colors_float = colors.astype(np.float32)

    if len(colors_float) < 2:
        return np.zeros(len(colors_float), dtype=np.int32), np.zeros((0, colors.shape[1]), dtype=np.float32), 0

    bandwidth = estimate_bandwidth(colors_float, quantile=quantile)
    if bandwidth <= 0:
        bandwidth = 10.0

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = ms.fit_predict(colors_float)
    centers = ms.cluster_centers_.astype(np.float32)
    n_clusters = len(centers)

    return labels, centers, n_clusters


def compute_clustering_confidence(
    colors: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    method: str
) -> ClusteringConfidence:
    """
    Compute confidence metrics for clustering results.

    Args:
        colors: Original color values (N, 3)
        labels: Cluster label for each point
        centers: Cluster center coordinates
        method: Clustering method used

    Returns:
        ClusteringConfidence with computed metrics
    """
    confidence = ClusteringConfidence()
    n_samples = len(labels)
    n_clusters = len(centers)

    if n_samples < 2 or n_clusters < 2:
        confidence.overall = 0.1 if n_clusters >= 1 else 0.0
        return confidence

    colors_float = colors.astype(np.float32)

    # Silhouette score
    try:
        sil_score = silhouette_score(colors_float, labels)
        confidence.silhouette = sil_score
        silhouette_normalized = (sil_score + 1.0) / 2.0
    except Exception:
        confidence.silhouette = 0.0
        silhouette_normalized = 0.5

    # Intra-cluster variance
    total_variance = 0.0
    for i in range(n_clusters):
        mask = labels == i
        if np.sum(mask) > 0:
            cluster_points = colors_float[mask]
            cluster_center = centers[i]
            distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
            total_variance += np.mean(distances ** 2)
    avg_variance = total_variance / n_clusters if n_clusters > 0 else 0.0
    confidence.intra_cluster_variance = avg_variance
    variance_score = max(0.0, 1.0 - avg_variance / 100.0)

    # Method confidence
    method_scores = {
        "dbscan": 1.0,
        "meanshift": 0.85,
        "kmeans": 0.7
    }
    confidence.method_confidence = method_scores.get(method, 0.5)

    # Overall confidence
    confidence.overall = (
        0.50 * silhouette_normalized +
        0.30 * variance_score +
        0.20 * confidence.method_confidence
    )

    return confidence


def cluster_colors_adaptive(
    colors: np.ndarray,
    eps: float = 15.0,
    min_samples: int = 2,
    meanshift_quantile: float = 0.3,
    fallback_k: int = 8,
    confidence_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, str, ClusteringConfidence]:
    """
    Cluster colors using adaptive method selection with fallback pipeline.

    Tries DBSCAN first, then MeanShift, then KMeans as final fallback.
    Logs fallback triggers when methods fail or produce low confidence.

    Args:
        colors: Array of colors (N, 3) in LAB format
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        meanshift_quantile: Quantile for MeanShift bandwidth estimation
        fallback_k: Number of clusters for KMeans fallback
        confidence_threshold: Minimum confidence to accept without warning

    Returns:
        Tuple of (labels, centers, method_used, confidence)
    """
    n_samples = len(colors)
    logger.info(f"Starting adaptive clustering on {n_samples} color samples")

    # Try DBSCAN first
    logger.debug(f"Attempting DBSCAN clustering (eps={eps}, min_samples={min_samples})")
    labels, centers, n_clusters = dbscan_cluster(colors, eps=eps, min_samples=min_samples)

    noise_ratio = np.sum(labels == -1) / len(labels) if len(labels) > 0 else 1.0

    if n_clusters >= 2 and noise_ratio < 0.3:
        # DBSCAN succeeded - remap noise points
        if noise_ratio > 0:
            noise_mask = labels == -1
            for i in np.where(noise_mask)[0]:
                distances = np.linalg.norm(centers - colors[i], axis=1)
                labels[i] = np.argmin(distances)
            logger.debug(f"DBSCAN: remapped {np.sum(noise_mask)} noise points")

        confidence = compute_clustering_confidence(colors, labels, centers, "dbscan")
        logger.info(f"DBSCAN succeeded: {n_clusters} clusters, noise_ratio={noise_ratio:.3f}, confidence={confidence.overall:.3f}")

        if confidence.overall < confidence_threshold:
            logger.warning(f"DBSCAN confidence ({confidence.overall:.3f}) below threshold ({confidence_threshold})")

        return labels, centers, "dbscan", confidence

    # DBSCAN fallback triggered
    logger.warning(f"DBSCAN fallback triggered: n_clusters={n_clusters}, noise_ratio={noise_ratio:.3f}")

    # Try MeanShift
    logger.debug(f"Attempting MeanShift clustering (quantile={meanshift_quantile})")
    labels, centers, n_clusters = meanshift_cluster(colors, quantile=meanshift_quantile)

    if n_clusters >= 2:
        confidence = compute_clustering_confidence(colors, labels, centers, "meanshift")
        logger.info(f"MeanShift succeeded: {n_clusters} clusters, confidence={confidence.overall:.3f}")

        if confidence.overall < confidence_threshold:
            logger.warning(f"MeanShift confidence ({confidence.overall:.3f}) below threshold ({confidence_threshold})")

        return labels, centers, "meanshift", confidence

    # MeanShift fallback triggered
    logger.warning(f"MeanShift fallback triggered: n_clusters={n_clusters}")

    # Fall back to KMeans
    logger.debug(f"Falling back to KMeans clustering (k={fallback_k})")
    labels, centers = kmeans_cluster(colors, k=fallback_k)
    confidence = compute_clustering_confidence(colors, labels, centers, "kmeans")
    logger.info(f"KMeans fallback used: {fallback_k} clusters, confidence={confidence.overall:.3f}")

    if confidence.overall < confidence_threshold:
        logger.warning(f"KMeans confidence ({confidence.overall:.3f}) below threshold - all methods exhausted")

    return labels, centers, "kmeans", confidence


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
    ap.add_argument("--k", type=int, default=None, help="Number of color regions; if omitted, uses adaptive clustering")
    ap.add_argument("--out", default="regions_extracted.json", help="Output json")
    ap.add_argument("--spatial-radius", type=int, default=10,
                    help="Spatial window radius for meanshift filtering (default: 10)")
    ap.add_argument("--color-radius", type=int, default=20,
                    help="Color window radius for meanshift filtering (default: 20)")
    ap.add_argument("--no-preprocess", action="store_true",
                    help="Disable pyrMeanShiftFiltering preprocessing")
    ap.add_argument("--use-adaptive", action="store_true",
                    help="Use adaptive clustering (DBSCAN -> MeanShift -> KMeans fallback)")
    ap.add_argument("--eps", type=float, default=15.0,
                    help="DBSCAN epsilon parameter for adaptive clustering (default: 15.0)")
    ap.add_argument("--confidence-threshold", type=float, default=0.5,
                    help="Minimum confidence score to accept clustering (default: 0.5)")
    ap.add_argument("--use-lab", action="store_true", default=True,
                    help="Use LAB color space for clustering (default: True)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Enable verbose logging output")
    args = ap.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info(f"Processing screenshot: {args.screenshot}")

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg = Config(**cfg)

    img = cv2.imread(args.screenshot, cv2.IMREAD_COLOR)
    if img is None:
        logger.error(f"Could not read screenshot: {args.screenshot}")
        raise SystemExit("Could not read screenshot")

    x, y, w, h = cfg.board_crop
    board = img[y:y+h, x:x+w].copy()
    logger.debug(f"Cropped board: {w}x{h} from ({x}, {y})")

    # Apply pyrMeanShiftFiltering for edge-preserving color smoothing
    # This makes colors more uniform within regions while preserving edges,
    # improving clustering accuracy for complex layouts
    if not args.no_preprocess:
        logger.debug(f"Applying meanshift filtering (spatial={args.spatial_radius}, color={args.color_radius})")
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
    logger.debug(f"Sampled {len(colors)} cell colors from {cfg.rows}x{cfg.cols} grid")

    # Convert to LAB color space for better perceptual clustering
    if args.use_lab:
        colors_uint8 = np.clip(colors, 0, 255).astype(np.uint8)
        colors_reshaped = colors_uint8.reshape(-1, 1, 3)
        colors_lab = cv2.cvtColor(colors_reshaped, cv2.COLOR_BGR2LAB)
        colors = colors_lab.reshape(-1, 3).astype(np.float32)
        logger.debug("Converted colors to LAB color space")

    # Use adaptive clustering or traditional KMeans
    method_used = "kmeans"
    confidence = None

    if args.use_adaptive or args.k is None:
        # Use adaptive clustering with fallback pipeline
        logger.info("Using adaptive clustering pipeline (DBSCAN -> MeanShift -> KMeans)")
        fallback_k = args.k if args.k else 8

        labels, centers, method_used, confidence = cluster_colors_adaptive(
            colors,
            eps=args.eps,
            min_samples=2,
            fallback_k=fallback_k,
            confidence_threshold=args.confidence_threshold
        )
        k = len(set(labels))

        # Convert LAB centers back to BGR for output
        if args.use_lab and len(centers) > 0:
            centers_uint8 = np.clip(centers, 0, 255).astype(np.uint8)
            centers_reshaped = centers_uint8.reshape(-1, 1, 3)
            centers_bgr = cv2.cvtColor(centers_reshaped, cv2.COLOR_LAB2BGR)
            centers = centers_bgr.reshape(-1, 3).astype(np.float32)
    else:
        # Use traditional KMeans with specified k
        k = args.k
        logger.info(f"Using KMeans clustering with k={k}")
        labels, centers = kmeans_cluster(colors, k)

        # Convert LAB centers back to BGR for output
        if args.use_lab and len(centers) > 0:
            centers_uint8 = np.clip(centers, 0, 255).astype(np.uint8)
            centers_reshaped = centers_uint8.reshape(-1, 1, 3)
            centers_bgr = cv2.cvtColor(centers_reshaped, cv2.COLOR_LAB2BGR)
            centers = centers_bgr.reshape(-1, 3).astype(np.float32)

    regions = build_regions(labels, cfg.rows, cfg.cols)

    out = regions_to_yaml_like(regions, centers)
    out["grid"]["rows"] = cfg.rows
    out["grid"]["cols"] = cfg.cols

    # Add clustering metadata
    out["clustering"] = {
        "method": method_used,
        "n_clusters": k
    }
    if confidence:
        out["clustering"]["confidence"] = confidence.to_dict()

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Log final summary
    confidence_str = f", confidence={confidence.overall:.3f}" if confidence else ""
    logger.info(f"Clustering complete: method={method_used}, k={k}, regions={len(regions)}{confidence_str}")

    if confidence and confidence.overall < args.confidence_threshold:
        logger.warning(
            f"Final clustering confidence ({confidence.overall:.3f}) "
            f"below threshold ({args.confidence_threshold}) - results may be inaccurate"
        )

    print(f"Wrote {args.out} (method={method_used}, k={k}, regions={len(regions)})")

if __name__ == "__main__":
    main()
