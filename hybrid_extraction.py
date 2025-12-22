"""
Hybrid extraction module for grid-based puzzle detection.

This module provides adaptive grid line detection with RANSAC-style robust
fitting for grid parameters. It handles irregular grids, partial grids,
and non-standard grid patterns by robustly estimating grid spacing from
detected lines.

Key features:
- Hough line detection for grid line identification
- RANSAC-style robust fitting to filter outliers and estimate grid spacing
- Adaptive thresholding for varying lighting conditions
- Histogram analysis fallback for when Hough detection fails
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path


@dataclass
class GridLineResult:
    """
    Result of grid line detection and analysis.

    Attributes:
        horizontal_lines: Y-coordinates of detected horizontal grid lines
        vertical_lines: X-coordinates of detected vertical grid lines
        estimated_cell_width: Robustly estimated cell width (None if not determined)
        estimated_cell_height: Robustly estimated cell height (None if not determined)
        grid_dims: Estimated (rows, cols) of the grid (None if not determined)
        confidence: Confidence score for the grid detection (0.0 to 1.0)
        method: Detection method used ("hough_ransac", "histogram", "fallback")
    """
    horizontal_lines: np.ndarray
    vertical_lines: np.ndarray
    estimated_cell_width: Optional[float] = None
    estimated_cell_height: Optional[float] = None
    grid_dims: Optional[Tuple[int, int]] = None
    confidence: float = 0.0
    method: str = "unknown"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "horizontal_lines": self.horizontal_lines.tolist() if len(self.horizontal_lines) > 0 else [],
            "vertical_lines": self.vertical_lines.tolist() if len(self.vertical_lines) > 0 else [],
            "estimated_cell_width": self.estimated_cell_width,
            "estimated_cell_height": self.estimated_cell_height,
            "grid_dims": list(self.grid_dims) if self.grid_dims else None,
            "confidence": round(self.confidence, 4),
            "method": self.method
        }


def ransac_fit_spacing(
    positions: np.ndarray,
    min_spacing: float = 30.0,
    max_spacing: float = 200.0,
    n_iterations: int = 100,
    inlier_threshold: float = 10.0,
    min_inlier_ratio: float = 0.5
) -> Tuple[Optional[float], float, np.ndarray]:
    """
    Use RANSAC-style approach to robustly estimate grid spacing from line positions.

    This function handles outliers (spurious lines, merged cells) by:
    1. Sampling pairs of adjacent lines to hypothesize spacing values
    2. Counting how many inter-line gaps are consistent with each hypothesis
    3. Selecting the spacing with the most inliers

    Args:
        positions: Sorted array of line positions (x or y coordinates)
        min_spacing: Minimum valid spacing between grid lines
        max_spacing: Maximum valid spacing between grid lines
        n_iterations: Number of RANSAC iterations
        inlier_threshold: Maximum deviation to consider a gap as an inlier
        min_inlier_ratio: Minimum ratio of inliers required for valid fit

    Returns:
        Tuple of (estimated_spacing, confidence, inlier_mask):
        - estimated_spacing: Robustly estimated spacing (None if no valid fit)
        - confidence: Ratio of inliers to total gaps (0.0 to 1.0)
        - inlier_mask: Boolean mask indicating which gaps are inliers
    """
    if len(positions) < 2:
        return None, 0.0, np.array([], dtype=bool)

    positions = np.sort(positions)
    gaps = np.diff(positions)

    if len(gaps) == 0:
        return None, 0.0, np.array([], dtype=bool)

    # Filter gaps to valid range
    valid_mask = (gaps >= min_spacing) & (gaps <= max_spacing)
    valid_gaps = gaps[valid_mask]

    if len(valid_gaps) == 0:
        return None, 0.0, np.zeros(len(gaps), dtype=bool)

    best_spacing = None
    best_inlier_count = 0
    best_inlier_mask = np.zeros(len(gaps), dtype=bool)

    # RANSAC iterations
    for _ in range(n_iterations):
        # Sample a random valid gap as the hypothesis
        idx = np.random.randint(0, len(valid_gaps))
        hypothesis_spacing = valid_gaps[idx]

        # Count inliers: gaps that are approximately equal to hypothesis
        # or integer multiples (for merged cells)
        inlier_mask = np.zeros(len(gaps), dtype=bool)

        for i, gap in enumerate(gaps):
            if gap < min_spacing:
                continue

            # Check if gap is approximately n * hypothesis_spacing for n = 1, 2, 3
            for multiplier in [1, 2, 3]:
                expected = hypothesis_spacing * multiplier
                if abs(gap - expected) < inlier_threshold * multiplier:
                    inlier_mask[i] = True
                    break

        inlier_count = np.sum(inlier_mask)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_spacing = hypothesis_spacing
            best_inlier_mask = inlier_mask

    # Refine estimate using all inlier gaps
    if best_spacing is not None and best_inlier_count > 0:
        inlier_gaps = []
        for i, gap in enumerate(gaps):
            if best_inlier_mask[i]:
                # Normalize by multiplier to get base spacing
                for multiplier in [1, 2, 3]:
                    expected = best_spacing * multiplier
                    if abs(gap - expected) < inlier_threshold * multiplier:
                        inlier_gaps.append(gap / multiplier)
                        break

        if inlier_gaps:
            best_spacing = np.median(inlier_gaps)

    # Calculate confidence as ratio of inliers
    confidence = best_inlier_count / len(gaps) if len(gaps) > 0 else 0.0

    # Require minimum inlier ratio for valid result
    if confidence < min_inlier_ratio:
        return None, confidence, best_inlier_mask

    return best_spacing, confidence, best_inlier_mask


def detect_lines_hough(
    edges: np.ndarray,
    threshold: int = 50,
    min_line_length: int = 50,
    max_line_gap: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect horizontal and vertical lines using Hough Line Transform.

    Uses probabilistic Hough transform to find line segments, then filters
    by orientation to separate horizontal and vertical grid lines.

    Args:
        edges: Edge image (output of Canny or similar)
        threshold: Accumulator threshold for Hough transform
        min_line_length: Minimum line length to detect
        max_line_gap: Maximum gap between line segments to merge

    Returns:
        Tuple of (horizontal_positions, vertical_positions):
        - horizontal_positions: Y-coordinates of horizontal lines
        - vertical_positions: X-coordinates of vertical lines
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    horizontal_y = []
    vertical_x = []

    if lines is None:
        return np.array([]), np.array([])

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # Classify as horizontal or vertical based on angle
        if dx > 0 and dy / dx < 0.2:  # Nearly horizontal
            horizontal_y.append((y1 + y2) / 2)
        elif dy > 0 and dx / dy < 0.2:  # Nearly vertical
            vertical_x.append((x1 + x2) / 2)

    return np.array(horizontal_y), np.array(vertical_x)


def cluster_lines(
    positions: np.ndarray,
    min_distance: float = 15.0
) -> np.ndarray:
    """
    Cluster nearby line positions to reduce duplicates.

    Lines detected by Hough transform often have multiple detections for
    the same grid line. This function merges positions that are within
    min_distance of each other.

    Args:
        positions: Array of line positions
        min_distance: Minimum distance between distinct lines

    Returns:
        Array of clustered (deduplicated) line positions
    """
    if len(positions) == 0:
        return np.array([])

    positions = np.sort(positions)
    clusters = [[positions[0]]]

    for pos in positions[1:]:
        if pos - clusters[-1][-1] < min_distance:
            clusters[-1].append(pos)
        else:
            clusters.append([pos])

    # Return mean position of each cluster
    return np.array([np.mean(cluster) for cluster in clusters])


def detect_grid_lines_projection(
    gray: np.ndarray,
    min_distance: int = 30,
    rel_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect grid lines using edge projection analysis.

    Projects edge pixels along rows and columns to find positions
    where edges concentrate (grid line locations).

    Args:
        gray: Grayscale image
        min_distance: Minimum distance between detected lines
        rel_threshold: Relative threshold for peak detection (0.0 to 1.0)

    Returns:
        Tuple of (horizontal_positions, vertical_positions)
    """
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Project edges
    proj_y = edges.sum(axis=1).astype(np.float32)  # Sum along rows -> horizontal lines
    proj_x = edges.sum(axis=0).astype(np.float32)  # Sum along columns -> vertical lines

    def find_peaks(projection, min_dist, threshold_ratio):
        """Find peaks in 1D projection."""
        if len(projection) < 5:
            return np.array([])

        # Smooth projection
        kernel_size = min(21, len(projection) // 4 * 2 + 1)
        if kernel_size >= 3:
            smoothed = cv2.GaussianBlur(
                projection.reshape(1, -1),
                (1, kernel_size),
                0
            ).ravel()
        else:
            smoothed = projection

        max_val = smoothed.max()
        if max_val <= 0:
            return np.array([])

        threshold = max_val * threshold_ratio

        # Find local maxima above threshold
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > threshold:
                if smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
                    peaks.append(i)

        if not peaks:
            return np.array([])

        # Greedy selection to enforce minimum distance
        peaks = np.array(peaks)
        strengths = smoothed[peaks]
        order = np.argsort(-strengths)

        selected = []
        for idx in order:
            pos = peaks[idx]
            if all(abs(pos - s) >= min_dist for s in selected):
                selected.append(pos)

        return np.array(sorted(selected))

    h_lines = find_peaks(proj_y, min_distance, rel_threshold)
    v_lines = find_peaks(proj_x, min_distance, rel_threshold)

    return h_lines, v_lines


def detect_grid_lines_adaptive(
    image: np.ndarray,
    use_hough: bool = True,
    use_projection: bool = True,
    min_spacing: float = 30.0,
    max_spacing: float = 200.0,
    ransac_iterations: int = 100,
    debug_dir: Optional[str] = None
) -> GridLineResult:
    """
    Detect grid lines using adaptive method selection with RANSAC-style robust fitting.

    Combines multiple detection methods (Hough transform, edge projection) and
    uses RANSAC-style robust fitting to estimate grid spacing, handling outliers
    and irregular grids.

    The detection pipeline:
    1. Apply adaptive thresholding and edge detection
    2. Detect lines using Hough transform and/or projection analysis
    3. Cluster nearby lines to remove duplicates
    4. Use RANSAC to robustly estimate grid spacing from line positions
    5. Estimate grid dimensions based on detected lines and spacing

    Args:
        image: Input image (BGR or grayscale)
        use_hough: Whether to use Hough line detection
        use_projection: Whether to use projection-based detection
        min_spacing: Minimum valid cell spacing
        max_spacing: Maximum valid cell spacing
        ransac_iterations: Number of RANSAC iterations for spacing estimation
        debug_dir: Optional directory to save debug images

    Returns:
        GridLineResult with detected lines, estimated spacing, and confidence
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]

    # Apply adaptive thresholding for varying lighting
    block_size = max(11, min(h, w) // 20)
    if block_size % 2 == 0:
        block_size += 1

    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, -5
    )

    # Edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)

    # Combine adaptive threshold edges with Canny edges
    combined_edges = cv2.bitwise_or(edges, cv2.Canny(adaptive_thresh, 50, 150))

    # Save debug images if requested
    if debug_dir:
        out_dir = Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "adaptive_thresh.png"), adaptive_thresh)
        cv2.imwrite(str(out_dir / "edges.png"), edges)
        cv2.imwrite(str(out_dir / "combined_edges.png"), combined_edges)

    # Collect line positions from multiple methods
    all_h_lines = []
    all_v_lines = []

    # Method 1: Hough line detection
    if use_hough:
        h_hough, v_hough = detect_lines_hough(
            combined_edges,
            threshold=max(30, min(h, w) // 15),
            min_line_length=max(40, min(h, w) // 10),
            max_line_gap=15
        )
        all_h_lines.extend(h_hough)
        all_v_lines.extend(v_hough)

    # Method 2: Projection-based detection
    if use_projection:
        h_proj, v_proj = detect_grid_lines_projection(
            gray,
            min_distance=int(min_spacing * 0.8),
            rel_threshold=0.25
        )
        all_h_lines.extend(h_proj)
        all_v_lines.extend(v_proj)

    # Cluster to remove duplicates
    h_lines = cluster_lines(np.array(all_h_lines), min_distance=min_spacing * 0.4)
    v_lines = cluster_lines(np.array(all_v_lines), min_distance=min_spacing * 0.4)

    # RANSAC-style robust fitting for grid spacing
    h_spacing, h_confidence, h_inliers = ransac_fit_spacing(
        h_lines,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
        n_iterations=ransac_iterations
    )

    v_spacing, v_confidence, v_inliers = ransac_fit_spacing(
        v_lines,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
        n_iterations=ransac_iterations
    )

    # Determine overall confidence
    overall_confidence = (h_confidence + v_confidence) / 2 if h_confidence > 0 or v_confidence > 0 else 0.0

    # Use consistent spacing if one direction has higher confidence
    if h_spacing is not None and v_spacing is not None:
        if h_confidence > v_confidence * 1.3:
            v_spacing = h_spacing  # Use more confident estimate
        elif v_confidence > h_confidence * 1.3:
            h_spacing = v_spacing
    elif h_spacing is not None and v_spacing is None:
        v_spacing = h_spacing
    elif v_spacing is not None and h_spacing is None:
        h_spacing = v_spacing

    # Estimate grid dimensions
    grid_dims = None
    if h_spacing is not None and v_spacing is not None:
        n_rows = int(round(h / h_spacing)) if h_spacing > 0 else 0
        n_cols = int(round(w / v_spacing)) if v_spacing > 0 else 0
        if n_rows > 0 and n_cols > 0:
            grid_dims = (n_rows, n_cols)

    # Determine method used
    method = "hough_ransac" if use_hough else "projection_ransac"
    if overall_confidence < 0.3:
        method = "fallback"

    # Save debug visualization
    if debug_dir:
        debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        for y in h_lines:
            cv2.line(debug_img, (0, int(y)), (w - 1, int(y)), (0, 255, 0), 2)
        for x in v_lines:
            cv2.line(debug_img, (int(x), 0), (int(x), h - 1), (0, 255, 0), 2)
        cv2.imwrite(str(out_dir / "detected_grid.png"), debug_img)

    return GridLineResult(
        horizontal_lines=h_lines,
        vertical_lines=v_lines,
        estimated_cell_width=v_spacing,
        estimated_cell_height=h_spacing,
        grid_dims=grid_dims,
        confidence=overall_confidence,
        method=method
    )


def find_puzzle_roi(
    image: np.ndarray,
    lower_half_only: bool = True,
    min_saturation: int = 25,
    debug_dir: Optional[str] = None
) -> Optional[Tuple[int, int, int, int]]:
    """
    Find the puzzle region of interest (ROI) in an image.

    Uses saturation-based segmentation to identify the colorful puzzle
    area against a dark or neutral background.

    Args:
        image: Input BGR image
        lower_half_only: If True, only search in the lower half of the image
        min_saturation: Minimum saturation value for puzzle region
        debug_dir: Optional directory to save debug images

    Returns:
        Tuple (x, y, w, h) of the puzzle ROI, or None if not found
    """
    if image is None:
        return None

    h, w = image.shape[:2]

    # Convert to HSV for saturation analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    # Create saturation mask
    mask = (saturation > min_saturation).astype(np.uint8) * 255

    # Optionally restrict to lower half
    if lower_half_only:
        mask[:h // 2, :] = 0

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if n_labels <= 1:
        return None

    # Skip background (label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + np.argmax(areas)

    x = stats[largest_idx, cv2.CC_STAT_LEFT]
    y = stats[largest_idx, cv2.CC_STAT_TOP]
    bw = stats[largest_idx, cv2.CC_STAT_WIDTH]
    bh = stats[largest_idx, cv2.CC_STAT_HEIGHT]

    # Add padding
    pad = int(0.05 * max(bw, bh))
    x = max(0, x - pad)
    y = max(0, y - pad)
    bw = min(w - x, bw + 2 * pad)
    bh = min(h - y, bh + 2 * pad)

    if debug_dir:
        out_dir = Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "roi_mask.png"), mask)
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
        cv2.imwrite(str(out_dir / "roi_detected.png"), debug_img)

    return (x, y, bw, bh)


def extract_grid_cells(
    image: np.ndarray,
    grid_result: GridLineResult,
    roi: Optional[Tuple[int, int, int, int]] = None,
    min_cell_area: int = 400,
    debug_dir: Optional[str] = None
) -> List[Tuple[int, int, int, int]]:
    """
    Extract cell bounding boxes from detected grid lines.

    Creates cell rectangles from the intersections of horizontal and
    vertical grid lines, filtering by minimum area.

    Args:
        image: Input image (for dimensions)
        grid_result: Result from detect_grid_lines_adaptive
        roi: Optional ROI offset (x, y, w, h)
        min_cell_area: Minimum cell area to include
        debug_dir: Optional directory to save debug images

    Returns:
        List of (x, y, w, h) tuples for detected cells in image coordinates
    """
    h_lines = grid_result.horizontal_lines
    v_lines = grid_result.vertical_lines

    if len(h_lines) < 2 or len(v_lines) < 2:
        return []

    h, w = image.shape[:2]

    # Add image boundaries if not already present
    h_lines = np.sort(h_lines)
    v_lines = np.sort(v_lines)

    if h_lines[0] > 20:
        h_lines = np.concatenate([[0], h_lines])
    if h_lines[-1] < h - 20:
        h_lines = np.concatenate([h_lines, [h - 1]])

    if v_lines[0] > 20:
        v_lines = np.concatenate([[0], v_lines])
    if v_lines[-1] < w - 20:
        v_lines = np.concatenate([v_lines, [w - 1]])

    # ROI offset
    roi_x, roi_y = (roi[0], roi[1]) if roi else (0, 0)

    cells = []
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            y1 = int(h_lines[i])
            y2 = int(h_lines[i + 1])
            x1 = int(v_lines[j])
            x2 = int(v_lines[j + 1])

            cell_w = x2 - x1
            cell_h = y2 - y1

            if cell_w * cell_h >= min_cell_area:
                # Convert to global coordinates
                cells.append((roi_x + x1, roi_y + y1, cell_w, cell_h))

    # Sort by row then column
    cells.sort(key=lambda c: (c[1], c[0]))

    if debug_dir:
        out_dir = Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        debug_img = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for x, y, cw, ch in cells:
            # Adjust for ROI offset in visualization
            cv2.rectangle(debug_img, (x - roi_x, y - roi_y), (x - roi_x + cw, y - roi_y + ch), (0, 255, 0), 2)
        cv2.imwrite(str(out_dir / "extracted_cells.png"), debug_img)

    return cells


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect grid lines in puzzle images")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--debug-dir", default="debug_hybrid", help="Directory for debug output")
    parser.add_argument("--min-spacing", type=float, default=30.0, help="Minimum cell spacing")
    parser.add_argument("--max-spacing", type=float, default=200.0, help="Maximum cell spacing")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    # Find ROI
    roi = find_puzzle_roi(img, debug_dir=args.debug_dir)
    if roi:
        x, y, w, h = roi
        roi_img = img[y:y + h, x:x + w]
    else:
        roi_img = img
        roi = None

    # Detect grid lines
    result = detect_grid_lines_adaptive(
        roi_img,
        min_spacing=args.min_spacing,
        max_spacing=args.max_spacing,
        debug_dir=args.debug_dir
    )

    # Extract cells
    cells = extract_grid_cells(roi_img, result, roi=roi, debug_dir=args.debug_dir)

    # Output results
    print(f"Method: {result.method}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Horizontal lines: {len(result.horizontal_lines)}")
    print(f"Vertical lines: {len(result.vertical_lines)}")
    print(f"Estimated cell size: {result.estimated_cell_width:.1f} x {result.estimated_cell_height:.1f}" if result.estimated_cell_width else "Cell size: N/A")
    print(f"Grid dimensions: {result.grid_dims}" if result.grid_dims else "Grid dims: N/A")
    print(f"Cells detected: {len(cells)}")

    # Write cells to file
    with open("cells.txt", "w") as f:
        for (cx, cy, cw, ch) in cells:
            f.write(f"{cx},{cy},{cw},{ch}\n")
    print("Wrote cells.txt")
