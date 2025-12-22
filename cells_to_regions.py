"""
Cell-to-region clustering module.

This module provides color-based clustering for puzzle cells, using DBSCAN
as the primary method (automatic cluster count detection), MeanShift as
secondary fallback, and KMeans as final fallback.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from collections import defaultdict
import json
from typing import Tuple, List, Dict, Optional


def dbscan_cluster(
    colors: np.ndarray,
    eps: float = 15.0,
    min_samples: int = 2
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Cluster colors using DBSCAN algorithm.

    DBSCAN automatically determines the number of clusters based on color density,
    making it ideal for puzzles with unknown region counts. It also identifies
    noise points that don't belong to any cluster.

    Args:
        colors: Array of colors (N, 3) in BGR or LAB format
        eps: Maximum distance between samples to be considered in same neighborhood.
             For LAB colors, 15-25 works well.
        min_samples: Minimum samples in neighborhood to form a core point.
             Lower values (2-3) for small puzzles, higher (5+) for large grids.

    Returns:
        Tuple of (labels, cluster_centers, n_clusters):
        - labels: Cluster label for each input color (-1 for noise)
        - cluster_centers: Mean color of each cluster (n_clusters, 3)
        - n_clusters: Number of clusters found (excluding noise)
    """
    colors_float = colors.astype(np.float32)

    # Fit DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(colors_float)

    # Get unique labels (excluding noise label -1)
    unique_labels = set(labels)
    unique_labels.discard(-1)
    n_clusters = len(unique_labels)

    # Compute cluster centers as mean of member colors
    if n_clusters > 0:
        centers = np.zeros((n_clusters, colors.shape[1]), dtype=np.float32)
        for i, label in enumerate(sorted(unique_labels)):
            mask = labels == label
            centers[i] = colors_float[mask].mean(axis=0)
    else:
        # No clusters found - return empty centers
        centers = np.zeros((0, colors.shape[1]), dtype=np.float32)

    return labels, centers, n_clusters


def meanshift_cluster(
    colors: np.ndarray,
    bandwidth: Optional[float] = None,
    quantile: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Cluster colors using MeanShift algorithm.

    MeanShift automatically determines the number of clusters based on density
    modes in the color space. It finds cluster centers by iteratively shifting
    points toward the mode (peak) of the local density. Unlike DBSCAN, it
    assigns all points to clusters (no noise points).

    Args:
        colors: Array of colors (N, 3) in BGR or LAB format
        bandwidth: Kernel bandwidth for MeanShift. If None, automatically
                   estimated using estimate_bandwidth with the given quantile.
        quantile: Quantile used for automatic bandwidth estimation (0-1).
                  Higher values = larger bandwidth = fewer clusters.
                  Default 0.3 works well for puzzle color segmentation.

    Returns:
        Tuple of (labels, cluster_centers, n_clusters):
        - labels: Cluster label for each input color (no noise, all assigned)
        - cluster_centers: Mean color of each cluster (n_clusters, 3)
        - n_clusters: Number of clusters found
    """
    colors_float = colors.astype(np.float32)

    # Estimate bandwidth if not provided
    if bandwidth is None:
        # estimate_bandwidth requires at least 2 samples
        if len(colors_float) < 2:
            return np.zeros(len(colors_float), dtype=np.int32), np.zeros((0, colors.shape[1]), dtype=np.float32), 0
        bandwidth = estimate_bandwidth(colors_float, quantile=quantile)
        # Ensure bandwidth is positive (can be 0 for very uniform colors)
        if bandwidth <= 0:
            bandwidth = 10.0  # Default fallback for LAB color space

    # Fit MeanShift
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = ms.fit_predict(colors_float)
    centers = ms.cluster_centers_.astype(np.float32)
    n_clusters = len(centers)

    return labels, centers, n_clusters


def kmeans_cluster(
    colors: np.ndarray,
    k: int,
    random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster colors using KMeans algorithm.

    KMeans requires specifying the number of clusters upfront. Use as fallback
    when DBSCAN fails or when cluster count is known.

    Args:
        colors: Array of colors (N, 3) in BGR or LAB format
        k: Number of clusters to find
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (labels, centers):
        - labels: Cluster label for each input color
        - centers: Cluster center colors (k, 3)
    """
    colors_float = colors.astype(np.float32)

    # Use OpenCV's kmeans for consistency with other cv-service code
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centers = cv2.kmeans(
        colors_float, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    return labels.flatten(), centers


def cluster_colors_adaptive(
    colors: np.ndarray,
    eps: float = 15.0,
    min_samples: int = 2,
    meanshift_quantile: float = 0.3,
    fallback_k: int = 6
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Cluster colors using adaptive method selection.

    Tries DBSCAN first (automatic cluster detection), then MeanShift as
    secondary fallback, and finally KMeans as the last resort.

    Args:
        colors: Array of colors (N, 3)
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        meanshift_quantile: Quantile for MeanShift bandwidth estimation
        fallback_k: Number of clusters for KMeans fallback

    Returns:
        Tuple of (labels, centers, method_used):
        - labels: Cluster label for each input color
        - centers: Cluster center colors
        - method_used: "dbscan", "meanshift", or "kmeans"
    """
    # Try DBSCAN first
    labels, centers, n_clusters = dbscan_cluster(colors, eps=eps, min_samples=min_samples)

    # Check if DBSCAN produced reasonable results
    noise_ratio = np.sum(labels == -1) / len(labels) if len(labels) > 0 else 1.0

    if n_clusters >= 2 and noise_ratio < 0.3:
        # DBSCAN succeeded - remap noise points to nearest cluster
        if noise_ratio > 0:
            noise_mask = labels == -1
            for i in np.where(noise_mask)[0]:
                distances = np.linalg.norm(centers - colors[i], axis=1)
                labels[i] = np.argmin(distances)
        return labels, centers, "dbscan"

    # Try MeanShift as secondary fallback
    labels, centers, n_clusters = meanshift_cluster(colors, quantile=meanshift_quantile)

    if n_clusters >= 2:
        # MeanShift succeeded
        return labels, centers, "meanshift"

    # Fall back to KMeans as last resort
    labels, centers = kmeans_cluster(colors, k=fallback_k)
    return labels, centers, "kmeans"


def extract_cells(img: np.ndarray) -> List[Tuple[int, int, int, int, int, int]]:
    """
    Extract cell bounding boxes from puzzle image.

    Args:
        img: Input image in BGR format

    Returns:
        List of (x, y, w, h, cx, cy) tuples for each detected cell,
        sorted by row then column
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 40 < w < 120 and abs(w - h) < 15:
            cx, cy = x + w // 2, y + h // 2
            cells.append((x, y, w, h, cx, cy))

    # Sort by row (y) then column (x)
    cells.sort(key=lambda c: (c[5], c[4]))
    return cells


def sample_cell_colors(
    img: np.ndarray,
    cells: List[Tuple[int, int, int, int, int, int]],
    use_lab: bool = True
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Sample representative colors from each cell.

    Args:
        img: Input image in BGR format
        cells: List of cell bounding boxes (x, y, w, h, cx, cy)
        use_lab: If True, convert to LAB color space for perceptual uniformity

    Returns:
        Tuple of (colors, coords):
        - colors: Array of sampled colors (N, 3)
        - coords: List of cell center coordinates (cx, cy)
    """
    colors = []
    coords = []

    for x, y, w, h, cx, cy in cells:
        # Sample from center of cell to avoid grid lines
        margin = 10
        patch = img[y + margin:y + h - margin, x + margin:x + w - margin]

        if patch.size == 0:
            continue

        avg = patch.mean(axis=(0, 1))
        colors.append(avg)
        coords.append((cx, cy))

    colors = np.array(colors, dtype=np.float32)

    # Convert to LAB color space for better perceptual clustering
    if use_lab and len(colors) > 0:
        # Reshape for cv2.cvtColor: (N, 1, 3) -> convert -> (N, 3)
        colors_uint8 = np.clip(colors, 0, 255).astype(np.uint8)
        colors_reshaped = colors_uint8.reshape(-1, 1, 3)
        colors_lab = cv2.cvtColor(colors_reshaped, cv2.COLOR_BGR2LAB)
        colors = colors_lab.reshape(-1, 3).astype(np.float32)

    return colors, coords


def build_regions(
    coords: List[Tuple[int, int]],
    labels: np.ndarray
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Group cell coordinates by cluster label.

    Args:
        coords: List of (cx, cy) cell center coordinates
        labels: Cluster label for each cell

    Returns:
        Dictionary mapping label -> list of (cx, cy) coordinates
    """
    regions: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for (cx, cy), lab in zip(coords, labels):
        regions[int(lab)].append((cx, cy))
    return dict(regions)


def main(
    img_path: str = "IMG_2050.png",
    output_path: str = "regions_raw.json",
    eps: float = 15.0,
    min_samples: int = 2,
    fallback_k: int = 6,
    use_lab: bool = True
) -> Dict:
    """
    Main entry point for cell-to-region clustering.

    Args:
        img_path: Path to input image
        output_path: Path to output JSON file
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        fallback_k: Number of clusters for KMeans fallback
        use_lab: Use LAB color space for clustering

    Returns:
        Dictionary with clustering results
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    # Extract cells
    cells = extract_cells(img)
    if len(cells) == 0:
        raise ValueError("No cells detected in image")

    # Sample colors
    colors, coords = sample_cell_colors(img, cells, use_lab=use_lab)

    # Cluster using adaptive method
    labels, centers, method = cluster_colors_adaptive(
        colors,
        eps=eps,
        min_samples=min_samples,
        fallback_k=fallback_k
    )

    # Build regions
    regions = build_regions(coords, labels)

    # Convert LAB centers back to BGR for output
    if use_lab and len(centers) > 0:
        centers_uint8 = np.clip(centers, 0, 255).astype(np.uint8)
        centers_reshaped = centers_uint8.reshape(-1, 1, 3)
        centers_bgr = cv2.cvtColor(centers_reshaped, cv2.COLOR_LAB2BGR)
        centers = centers_bgr.reshape(-1, 3).astype(np.float32)

    # Prepare output
    out = {
        "method": method,
        "n_clusters": len(regions),
        "regions": [
            {
                "id": i,
                "cells": v,
                "color": centers[i].tolist() if i < len(centers) else [0, 0, 0]
            }
            for i, v in regions.items()
        ]
    }

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {output_path} (method={method}, regions={len(regions)})")
    return out


if __name__ == "__main__":
    main()
