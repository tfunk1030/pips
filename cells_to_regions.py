"""
Cell-to-region clustering module.

This module provides color-based clustering for puzzle cells, using DBSCAN
as the primary method (automatic cluster count detection), MeanShift as
secondary fallback, and KMeans as final fallback.

Includes confidence scoring for clustering results to enable intelligent
fallback decisions and quality assessment.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
from collections import defaultdict
from dataclasses import dataclass, field
import json
import logging
from typing import Tuple, List, Dict, Optional

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class ClusterConfidence:
    """
    Confidence metrics for clustering results.

    Attributes:
        overall: Overall confidence score (0.0 to 1.0)
        intra_cluster_variance: Average variance within clusters (lower is better)
        inter_cluster_separation: Average distance between cluster centers (higher is better)
        silhouette: Silhouette score (-1.0 to 1.0, higher is better)
        noise_ratio: Ratio of noise points (DBSCAN only, lower is better)
        cluster_balance: Measure of cluster size balance (0.0 to 1.0, higher is better)
        method_confidence: Confidence based on clustering method used
    """
    overall: float = 0.0
    intra_cluster_variance: float = 0.0
    inter_cluster_separation: float = 0.0
    silhouette: float = 0.0
    noise_ratio: float = 0.0
    cluster_balance: float = 0.0
    method_confidence: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall": round(self.overall, 4),
            "intra_cluster_variance": round(self.intra_cluster_variance, 4),
            "inter_cluster_separation": round(self.inter_cluster_separation, 4),
            "silhouette": round(self.silhouette, 4),
            "noise_ratio": round(self.noise_ratio, 4),
            "cluster_balance": round(self.cluster_balance, 4),
            "method_confidence": round(self.method_confidence, 4)
        }


@dataclass
class ClusteringResult:
    """
    Complete clustering result with labels, centers, method info, and confidence.

    Attributes:
        labels: Cluster label for each input point
        centers: Cluster center coordinates
        method: Clustering method used ("dbscan", "meanshift", or "kmeans")
        n_clusters: Number of clusters found
        confidence: Confidence metrics for the clustering
    """
    labels: np.ndarray
    centers: np.ndarray
    method: str
    n_clusters: int
    confidence: ClusterConfidence = field(default_factory=ClusterConfidence)


def compute_cluster_confidence(
    colors: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    method: str,
    original_labels: Optional[np.ndarray] = None
) -> ClusterConfidence:
    """
    Compute confidence metrics for clustering results.

    Calculates multiple metrics to assess clustering quality:
    - Intra-cluster variance: How tight are the clusters?
    - Inter-cluster separation: How well separated are cluster centers?
    - Silhouette score: Combined measure of cluster cohesion and separation
    - Noise ratio: Percentage of points marked as noise (DBSCAN)
    - Cluster balance: How balanced are cluster sizes?
    - Method confidence: Baseline confidence based on method used

    Args:
        colors: Original color values (N, 3)
        labels: Cluster label for each point (after noise remapping)
        centers: Cluster center coordinates (K, 3)
        method: Clustering method used ("dbscan", "meanshift", "kmeans")
        original_labels: Labels before noise remapping (for DBSCAN noise ratio)

    Returns:
        ClusterConfidence with all computed metrics
    """
    confidence = ClusterConfidence()
    n_samples = len(labels)
    n_clusters = len(centers)

    # Handle edge cases
    if n_samples < 2 or n_clusters < 2:
        confidence.overall = 0.1 if n_clusters >= 1 else 0.0
        return confidence

    colors_float = colors.astype(np.float32)

    # 1. Intra-cluster variance (lower is better)
    # Average within-cluster sum of squared distances
    total_variance = 0.0
    cluster_sizes = []
    for i in range(n_clusters):
        mask = labels == i
        if np.sum(mask) > 0:
            cluster_points = colors_float[mask]
            cluster_center = centers[i]
            distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
            total_variance += np.mean(distances ** 2)
            cluster_sizes.append(np.sum(mask))
    avg_variance = total_variance / n_clusters if n_clusters > 0 else 0.0
    # Normalize: variance of 0 = 1.0, variance of 100+ = ~0.1
    confidence.intra_cluster_variance = avg_variance
    variance_score = max(0.0, 1.0 - avg_variance / 100.0)

    # 2. Inter-cluster separation (higher is better)
    # Average pairwise distance between cluster centers
    if n_clusters >= 2:
        center_distances = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                dist = np.linalg.norm(centers[i] - centers[j])
                center_distances.append(dist)
        avg_separation = np.mean(center_distances) if center_distances else 0.0
        confidence.inter_cluster_separation = avg_separation
        # Normalize: separation of 50+ LAB units = 1.0
        separation_score = min(1.0, avg_separation / 50.0)
    else:
        confidence.inter_cluster_separation = 0.0
        separation_score = 0.0

    # 3. Silhouette score (-1 to 1, higher is better)
    try:
        sil_score = silhouette_score(colors_float, labels)
        confidence.silhouette = sil_score
        # Normalize to 0-1 range
        silhouette_normalized = (sil_score + 1.0) / 2.0
    except Exception:
        confidence.silhouette = 0.0
        silhouette_normalized = 0.5

    # 4. Noise ratio (DBSCAN only)
    if original_labels is not None:
        noise_count = np.sum(original_labels == -1)
        confidence.noise_ratio = noise_count / n_samples
        noise_score = 1.0 - confidence.noise_ratio
    else:
        confidence.noise_ratio = 0.0
        noise_score = 1.0

    # 5. Cluster balance (how evenly distributed are cluster sizes)
    if cluster_sizes and len(cluster_sizes) > 1:
        sizes = np.array(cluster_sizes)
        # Coefficient of variation: std/mean (lower = more balanced)
        cv = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 1.0
        # Transform: cv of 0 = 1.0, cv of 2+ = ~0.3
        confidence.cluster_balance = max(0.0, 1.0 - cv / 2.0)
    else:
        confidence.cluster_balance = 1.0

    # 6. Method confidence (baseline based on method preference)
    method_scores = {
        "dbscan": 1.0,      # Preferred: automatic cluster count
        "meanshift": 0.85,  # Secondary: automatic but slower
        "kmeans": 0.7       # Fallback: requires known cluster count
    }
    confidence.method_confidence = method_scores.get(method, 0.5)

    # 7. Compute overall confidence as weighted combination
    weights = {
        "variance": 0.20,
        "separation": 0.20,
        "silhouette": 0.30,
        "noise": 0.10,
        "balance": 0.10,
        "method": 0.10
    }
    confidence.overall = (
        weights["variance"] * variance_score +
        weights["separation"] * separation_score +
        weights["silhouette"] * silhouette_normalized +
        weights["noise"] * noise_score +
        weights["balance"] * confidence.cluster_balance +
        weights["method"] * confidence.method_confidence
    )

    return confidence


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


def cluster_colors_with_confidence(
    colors: np.ndarray,
    eps: float = 15.0,
    min_samples: int = 2,
    meanshift_quantile: float = 0.3,
    fallback_k: int = 6,
    confidence_threshold: float = 0.5
) -> ClusteringResult:
    """
    Cluster colors with confidence scoring using adaptive method selection.

    Tries DBSCAN first (automatic cluster detection), then MeanShift as
    secondary fallback, and finally KMeans as the last resort. Computes
    comprehensive confidence metrics for the clustering result.

    Fallback triggers are logged when:
    - DBSCAN produces too few clusters or high noise ratio
    - MeanShift produces too few clusters
    - Confidence score falls below threshold

    Args:
        colors: Array of colors (N, 3)
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        meanshift_quantile: Quantile for MeanShift bandwidth estimation
        fallback_k: Number of clusters for KMeans fallback
        confidence_threshold: Minimum confidence score to accept result without warning

    Returns:
        ClusteringResult with labels, centers, method, and confidence metrics
    """
    n_samples = len(colors)
    logger.info(f"Starting adaptive clustering on {n_samples} color samples")

    # Try DBSCAN first
    logger.debug(f"Attempting DBSCAN clustering (eps={eps}, min_samples={min_samples})")
    labels, centers, n_clusters = dbscan_cluster(colors, eps=eps, min_samples=min_samples)
    original_labels = labels.copy()  # Keep for noise ratio calculation

    # Check if DBSCAN produced reasonable results
    noise_ratio = np.sum(labels == -1) / len(labels) if len(labels) > 0 else 1.0

    if n_clusters >= 2 and noise_ratio < 0.3:
        # DBSCAN succeeded - remap noise points to nearest cluster
        if noise_ratio > 0:
            noise_mask = labels == -1
            for i in np.where(noise_mask)[0]:
                distances = np.linalg.norm(centers - colors[i], axis=1)
                labels[i] = np.argmin(distances)
            logger.debug(f"DBSCAN: remapped {np.sum(noise_mask)} noise points to nearest clusters")

        confidence = compute_cluster_confidence(
            colors, labels, centers, "dbscan", original_labels
        )
        logger.info(f"DBSCAN succeeded: {n_clusters} clusters, noise_ratio={noise_ratio:.3f}, confidence={confidence.overall:.3f}")

        if confidence.overall < confidence_threshold:
            logger.warning(f"DBSCAN confidence ({confidence.overall:.3f}) below threshold ({confidence_threshold})")

        return ClusteringResult(
            labels=labels,
            centers=centers,
            method="dbscan",
            n_clusters=n_clusters,
            confidence=confidence
        )

    # DBSCAN fallback triggered
    logger.warning(f"DBSCAN fallback triggered: n_clusters={n_clusters}, noise_ratio={noise_ratio:.3f} (threshold: clusters>=2, noise<0.3)")

    # Try MeanShift as secondary fallback
    logger.debug(f"Attempting MeanShift clustering (quantile={meanshift_quantile})")
    labels, centers, n_clusters = meanshift_cluster(colors, quantile=meanshift_quantile)

    if n_clusters >= 2:
        # MeanShift succeeded
        confidence = compute_cluster_confidence(
            colors, labels, centers, "meanshift"
        )
        logger.info(f"MeanShift succeeded: {n_clusters} clusters, confidence={confidence.overall:.3f}")

        if confidence.overall < confidence_threshold:
            logger.warning(f"MeanShift confidence ({confidence.overall:.3f}) below threshold ({confidence_threshold})")

        return ClusteringResult(
            labels=labels,
            centers=centers,
            method="meanshift",
            n_clusters=n_clusters,
            confidence=confidence
        )

    # MeanShift fallback triggered
    logger.warning(f"MeanShift fallback triggered: n_clusters={n_clusters} (threshold: clusters>=2)")

    # Fall back to KMeans as last resort
    logger.debug(f"Falling back to KMeans clustering (k={fallback_k})")
    labels, centers = kmeans_cluster(colors, k=fallback_k)
    confidence = compute_cluster_confidence(
        colors, labels, centers, "kmeans"
    )
    logger.info(f"KMeans fallback used: {fallback_k} clusters, confidence={confidence.overall:.3f}")

    if confidence.overall < confidence_threshold:
        logger.warning(f"KMeans confidence ({confidence.overall:.3f}) below threshold ({confidence_threshold}) - all methods exhausted")

    return ClusteringResult(
        labels=labels,
        centers=centers,
        method="kmeans",
        n_clusters=fallback_k,
        confidence=confidence
    )


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
    use_lab: bool = True,
    confidence_threshold: float = 0.5
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
        confidence_threshold: Minimum confidence score to accept without warning

    Returns:
        Dictionary with clustering results including confidence metrics
    """
    logger.info(f"Processing image: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        logger.error(f"Could not read image: {img_path}")
        raise FileNotFoundError(f"Could not read image: {img_path}")

    # Extract cells
    cells = extract_cells(img)
    if len(cells) == 0:
        logger.error("No cells detected in image")
        raise ValueError("No cells detected in image")

    logger.info(f"Extracted {len(cells)} cells from image")

    # Sample colors
    colors, coords = sample_cell_colors(img, cells, use_lab=use_lab)
    logger.debug(f"Sampled {len(colors)} colors (LAB={use_lab})")

    # Cluster using adaptive method with confidence scoring
    result = cluster_colors_with_confidence(
        colors,
        eps=eps,
        min_samples=min_samples,
        fallback_k=fallback_k,
        confidence_threshold=confidence_threshold
    )

    # Build regions
    regions = build_regions(coords, result.labels)

    # Convert LAB centers back to BGR for output
    centers = result.centers
    if use_lab and len(centers) > 0:
        centers_uint8 = np.clip(centers, 0, 255).astype(np.uint8)
        centers_reshaped = centers_uint8.reshape(-1, 1, 3)
        centers_bgr = cv2.cvtColor(centers_reshaped, cv2.COLOR_LAB2BGR)
        centers = centers_bgr.reshape(-1, 3).astype(np.float32)

    # Prepare output with confidence metrics
    out = {
        "method": result.method,
        "n_clusters": len(regions),
        "confidence": result.confidence.to_dict(),
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

    # Log final summary
    logger.info(
        f"Clustering complete: method={result.method}, "
        f"n_clusters={len(regions)}, confidence={result.confidence.overall:.3f}"
    )

    if result.confidence.overall < confidence_threshold:
        logger.warning(
            f"Final clustering confidence ({result.confidence.overall:.3f}) "
            f"below threshold ({confidence_threshold}) - results may be inaccurate"
        )

    return out


if __name__ == "__main__":
    main()
