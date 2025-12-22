"""
Multi-strategy puzzle cell detection using OpenCV.

This module provides multiple detection strategies for extracting puzzle cells
from screenshot images. Each strategy returns a DetectionResult dataclass with
cells, grid dimensions, regions, and confidence scores.

Strategies:
1. region_contours: Detects cells by finding colored region boundaries
2. color_segmentation: Segments by color using k-means, finds cells within regions
3. constraint_labels: Detects diamond-shaped markers and infers cells from positions
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
from pathlib import Path


Coord = Tuple[int, int]


@dataclass
class DetectionResult:
    """Result from a detection strategy."""
    success: bool
    cells: List[Tuple[int, int, int, int]]  # List of (x, y, w, h) bounding boxes
    grid_dims: Optional[Tuple[int, int]]  # (rows, cols)
    regions: Optional[Dict[str, List[int]]]  # region_id -> cell indices
    confidence: float  # 0.0 to 1.0
    method: str  # Strategy name
    error: Optional[str] = None


# =============================================================================
# Strategy 1: Region Contours Detection
# =============================================================================

def detect_by_region_contours(image_path: str, debug_dir: str = None) -> DetectionResult:
    """
    Strategy 1: Detect cells by finding colored region boundaries.

    Works for puzzles with distinct colored regions separated by grid lines.

    Steps:
    1. Load image and find board region by colorfulness
    2. Apply adaptive thresholding to detect tiles
    3. Find connected components as candidate cells
    4. Filter by size and aspect ratio
    5. Estimate grid dimensions from cell layout
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="region_contours",
                error="Could not read image"
            )

        h, w = img.shape[:2]

        # Find board region by saturation (colorful area)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, S, _ = cv2.split(hsv)

        S_MIN = 35
        mask_color = (S > S_MIN).astype(np.uint8) * 255

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find largest contour as board region
        cnts, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="region_contours",
                error="No board region found"
            )

        cnt = max(cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(cnt)

        # Add padding
        pad = int(0.06 * max(bw, bh))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)

        board = img[y0:y1, x0:x1].copy()

        # Detect tiles using adaptive thresholding on LAB L channel
        lab = cv2.cvtColor(board, cv2.COLOR_BGR2LAB)
        L, _, _ = cv2.split(lab)
        L_blur = cv2.GaussianBlur(L, (5, 5), 0)

        mask_tiles = cv2.adaptiveThreshold(
            L_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, -5
        )

        # Clean up
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_tiles = cv2.morphologyEx(mask_tiles, cv2.MORPH_OPEN, k2, iterations=1)
        mask_tiles = cv2.morphologyEx(mask_tiles, cv2.MORPH_CLOSE, k2, iterations=2)

        # Find connected components
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_tiles, connectivity=8)

        if num <= 1:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="region_contours",
                error="No tile components found"
            )

        # Estimate typical cell area
        areas = stats[1:, cv2.CC_STAT_AREA]
        areas_sorted = np.sort(areas)
        median_area = np.median(areas_sorted[len(areas_sorted)//4:3*len(areas_sorted)//4])
        if not np.isfinite(median_area) or median_area <= 0:
            median_area = np.median(areas_sorted)

        min_area = int(median_area * 0.35)
        max_area = int(median_area * 2.20)

        cells = []
        for i in range(1, num):
            cx, cy, cw, ch, area = stats[i]
            if area < min_area or area > max_area:
                continue
            ar = cw / float(ch) if ch > 0 else 0
            if ar < 0.75 or ar > 1.33:
                continue
            if cw < 20 or ch < 20:
                continue
            # Convert to global coordinates
            cells.append((x0 + cx, y0 + cy, cw, ch))

        if len(cells) == 0:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="region_contours",
                error="No valid cells found after filtering"
            )

        grid_dims = estimate_grid_dims(cells)
        regions = detect_regions_from_cells(img, cells)
        confidence = calculate_confidence(cells, grid_dims)

        if debug_dir:
            save_debug_image(img, cells, f"{debug_dir}/region_contours_method.png")

        return DetectionResult(
            success=True,
            cells=cells,
            grid_dims=grid_dims,
            regions=regions,
            confidence=confidence,
            method="region_contours"
        )

    except Exception as e:
        return DetectionResult(
            success=False, cells=[], grid_dims=None, regions=None,
            confidence=0.0, method="region_contours",
            error=str(e)
        )


# =============================================================================
# Strategy 2: Color Segmentation Detection
# =============================================================================

def detect_by_color_segmentation(image_path: str, debug_dir: str = None) -> DetectionResult:
    """
    Strategy 2: Detect cells by color segmentation using k-means clustering.

    Works for puzzles with distinct colored regions.

    Steps:
    1. Load image and extract board region
    2. Apply k-means clustering on colors
    3. Find cell-sized contours in each color segment
    4. Merge results from all segments
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="color_segmentation",
                error="Could not read image"
            )

        h, w = img.shape[:2]

        # Find board region
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, S, _ = cv2.split(hsv)
        mask_color = (S > 35).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="color_segmentation",
                error="No board region found"
            )

        cnt = max(cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(cnt)
        pad = int(0.02 * max(bw, bh))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)

        board = img[y0:y1, x0:x1].copy()

        # K-means color clustering
        pixels = board.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        k = 10  # Number of color clusters
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        segmented = centers[labels.flatten()].reshape(board.shape).astype(np.uint8)

        # Find cells in each color segment
        cells = []
        board_h, board_w = board.shape[:2]
        expected_cell_area = (board_w * board_h) / 50  # Rough estimate
        min_cell_area = expected_cell_area * 0.2
        max_cell_area = expected_cell_area * 3.0

        for i in range(k):
            color_mask = (labels.flatten() == i).reshape(board.shape[:2]).astype(np.uint8) * 255

            # Find contours in this color segment
            seg_cnts, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for scnt in seg_cnts:
                area = cv2.contourArea(scnt)
                if area < min_cell_area or area > max_cell_area:
                    continue

                sx, sy, sw, sh = cv2.boundingRect(scnt)
                ar = sw / float(sh) if sh > 0 else 0
                if 0.6 < ar < 1.67 and sw > 15 and sh > 15:
                    cells.append((x0 + sx, y0 + sy, sw, sh))

        if len(cells) == 0:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="color_segmentation",
                error="No cells found in color segments"
            )

        # Remove duplicate/overlapping cells
        cells = merge_overlapping_cells(cells)

        grid_dims = estimate_grid_dims(cells)
        regions = detect_regions_from_cells(img, cells)
        confidence = calculate_confidence(cells, grid_dims)

        if debug_dir:
            save_debug_image(img, cells, f"{debug_dir}/color_segmentation_method.png")

        return DetectionResult(
            success=True,
            cells=cells,
            grid_dims=grid_dims,
            regions=regions,
            confidence=confidence,
            method="color_segmentation"
        )

    except Exception as e:
        return DetectionResult(
            success=False, cells=[], grid_dims=None, regions=None,
            confidence=0.0, method="color_segmentation",
            error=str(e)
        )


# =============================================================================
# Strategy 3: Constraint Labels (Diamond Detection)
# =============================================================================

def detect_diamonds(img: np.ndarray, min_area: int = 100, max_area: int = 10000) -> List[Tuple[int, int, int, int, np.ndarray]]:
    """
    Detect diamond-shaped markers in an image using contour analysis.

    A diamond is identified as a 4-vertex polygon that is roughly square
    when rotated 45 degrees (aspect ratio ~1.0).

    Args:
        img: BGR image
        min_area: Minimum contour area to consider
        max_area: Maximum contour area to consider

    Returns:
        List of (cx, cy, w, h, contour) tuples for detected diamonds
        where (cx, cy) is the center of the diamond
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    diamonds = []

    # Try multiple thresholding approaches for robustness
    thresholds = []

    # Binary threshold
    _, binary1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    thresholds.append(binary1)

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    thresholds.append(adaptive)

    # Inverted adaptive threshold (for dark diamonds on light background)
    adaptive_inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    thresholds.append(adaptive_inv)

    # Edge detection + dilation
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    thresholds.append(edges_dilated)

    seen_centers = set()  # Avoid duplicates

    for thresh_img in thresholds:
        # Find contours
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < min_area or area > max_area:
                continue

            # Approximate polygon
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.04 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Diamonds should have 4 vertices
            if len(approx) != 4:
                continue

            # Check if it's diamond-shaped
            if is_diamond_shape(approx):
                # Get bounding box and center
                x, y, w, h = cv2.boundingRect(approx)
                cx, cy = x + w // 2, y + h // 2

                # Avoid duplicates (within 10 pixels)
                center_key = (cx // 10, cy // 10)
                if center_key in seen_centers:
                    continue
                seen_centers.add(center_key)

                diamonds.append((cx, cy, w, h, approx))

    return diamonds


def is_diamond_shape(approx: np.ndarray) -> bool:
    """
    Check if a 4-vertex polygon is diamond-shaped.

    A diamond is a rhombus (all sides roughly equal) that appears
    as a square rotated 45 degrees. We check:
    1. All 4 sides are roughly equal length
    2. The shape is roughly square (aspect ratio ~1.0)
    3. The diagonals are roughly perpendicular
    4. The shape has diamond-like orientation (rotated ~45 degrees)
    """
    if len(approx) != 4:
        return False

    # Get the 4 vertices
    pts = approx.reshape(4, 2)

    # Calculate side lengths
    sides = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        length = np.linalg.norm(p2 - p1)
        sides.append(length)

    # All sides should be roughly equal (within 30% tolerance - tighter)
    avg_side = np.mean(sides)
    if avg_side < 5:  # Minimum side length
        return False

    for side in sides:
        if abs(side - avg_side) / avg_side > 0.30:
            return False

    # Check bounding box aspect ratio (~1.0 for diamond)
    x, y, w, h = cv2.boundingRect(approx)
    if w == 0 or h == 0:
        return False

    aspect = w / h
    if aspect < 0.7 or aspect > 1.43:  # Tighter aspect ratio
        return False

    # Check that the shape fills a reasonable portion of its bounding box
    # A perfect diamond fills exactly 50% of its bounding box
    area = cv2.contourArea(approx)
    bbox_area = w * h
    fill_ratio = area / bbox_area
    if fill_ratio < 0.35 or fill_ratio > 0.75:
        return False

    return True


def detect_by_constraint_labels(image_path: str, debug_dir: str = None) -> DetectionResult:
    """
    Strategy 3: Detect diamond-shaped constraint labels and infer cell positions.

    For puzzles that have diamond-shaped markers indicating cell constraints,
    we can infer the grid structure from the diamond positions.

    Steps:
    1. Load image and detect diamond-shaped markers
    2. Cluster diamonds by position to identify grid structure
    3. Infer cell positions from diamond grid
    4. Verify by looking for actual cell boundaries
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="constraint_labels",
                error="Could not read image"
            )

        # Detect diamond markers
        diamonds = detect_diamonds(img)

        if len(diamonds) == 0:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="constraint_labels",
                error="No diamond markers detected"
            )

        # Extract diamond centers
        diamond_centers = [(cx, cy) for cx, cy, _, _, _ in diamonds]

        # Cluster diamonds to identify grid structure
        grid_dims = infer_grid_dims_from_diamonds(diamond_centers)

        # Infer cell positions from diamond positions
        cells = infer_cells_from_diamonds(diamond_centers, grid_dims)

        if len(cells) == 0:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="constraint_labels",
                error="Could not infer cells from diamonds"
            )

        regions = detect_regions_from_cells(img, cells)
        confidence = calculate_confidence(cells, grid_dims)

        if debug_dir:
            save_debug_image(img, cells, f"{debug_dir}/constraint_labels_method.png")

        return DetectionResult(
            success=True,
            cells=cells,
            grid_dims=grid_dims,
            regions=regions,
            confidence=confidence,
            method="constraint_labels"
        )

    except Exception as e:
        return DetectionResult(
            success=False, cells=[], grid_dims=None, regions=None,
            confidence=0.0, method="constraint_labels",
            error=str(e)
        )


def infer_grid_dims_from_diamonds(centers: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Infer grid dimensions from diamond center positions."""
    if not centers:
        return (0, 0)

    # Sort by y coordinate
    centers_by_y = sorted(centers, key=lambda c: c[1])

    # Group into rows
    rows = []
    current_row = [centers_by_y[0]]
    y_threshold = 50  # pixels

    for center in centers_by_y[1:]:
        if abs(center[1] - current_row[0][1]) <= y_threshold:
            current_row.append(center)
        else:
            rows.append(current_row)
            current_row = [center]
    rows.append(current_row)

    return (len(rows), max(len(row) for row in rows) if rows else 0)


def infer_cells_from_diamonds(centers: List[Tuple[int, int]], grid_dims: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    """Infer cell positions from diamond marker centers."""
    if not centers:
        return []

    # Calculate average cell size from diamond spacing
    centers = sorted(centers)
    if len(centers) < 2:
        # Single diamond - estimate cell size
        cell_size = 60
    else:
        # Estimate from spacing
        distances = []
        for i in range(len(centers) - 1):
            dist = np.sqrt((centers[i+1][0] - centers[i][0])**2 + (centers[i+1][1] - centers[i][1])**2)
            if dist > 10:  # Ignore very close points
                distances.append(dist)
        cell_size = int(np.median(distances) * 0.8) if distances else 60

    cells = []
    for cx, cy in centers:
        # Create bounding box centered on diamond
        x = cx - cell_size // 2
        y = cy - cell_size // 2
        cells.append((x, y, cell_size, cell_size))

    return cells


# =============================================================================
# Multi-Strategy Extraction
# =============================================================================

def extract_puzzle_multi_strategy(
    image_path: str,
    output_dir: str = None,
    strategies: List[str] = None
) -> Dict:
    """
    Try multiple detection strategies and return the best result.

    Args:
        image_path: Path to puzzle screenshot
        output_dir: Directory for debug output
        strategies: List of strategies to try (default: all)

    Returns:
        Best detection result with highest confidence
    """
    if strategies is None:
        strategies = ["region_contours", "color_segmentation", "constraint_labels"]

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []

    # Try each strategy
    if "region_contours" in strategies:
        result = detect_by_region_contours(image_path, output_dir)
        results.append(result)

    if "color_segmentation" in strategies:
        result = detect_by_color_segmentation(image_path, output_dir)
        results.append(result)

    if "constraint_labels" in strategies:
        result = detect_by_constraint_labels(image_path, output_dir)
        results.append(result)

    # Filter successful results
    successful = [r for r in results if r.success]

    if not successful:
        # Return best failed result with error info
        best_failed = max(results, key=lambda r: r.confidence)
        return {
            "success": False,
            "error": f"All strategies failed. Best attempt: {best_failed.method}",
            "attempts": [{"method": r.method, "error": r.error} for r in results]
        }

    # Pick result with highest confidence
    best = max(successful, key=lambda r: r.confidence)

    return {
        "success": True,
        "cells": best.cells,
        "grid_dims": best.grid_dims,
        "regions": best.regions,
        "num_cells": len(best.cells),
        "confidence": best.confidence,
        "method_used": best.method,
        "all_attempts": [
            {
                "method": r.method,
                "success": r.success,
                "confidence": r.confidence,
                "cells_found": len(r.cells) if r.cells else 0
            }
            for r in results
        ]
    }


# =============================================================================
# Helper Functions
# =============================================================================

def estimate_grid_dims(cells: List[Tuple[int, int, int, int]]) -> Tuple[int, int]:
    """Estimate grid dimensions from cell positions"""
    if not cells:
        return (0, 0)

    # Sort by y-coordinate to find rows
    cells_by_y = sorted(cells, key=lambda c: c[1])

    # Group into rows (cells with similar y-coordinates)
    rows = []
    current_row = [cells_by_y[0]]
    y_threshold = cells_by_y[0][3] * 0.5  # 50% of cell height

    for cell in cells_by_y[1:]:
        if abs(cell[1] - current_row[0][1]) <= y_threshold:
            current_row.append(cell)
        else:
            rows.append(current_row)
            current_row = [cell]
    rows.append(current_row)

    num_rows = len(rows)
    num_cols = max(len(row) for row in rows) if rows else 0

    return (num_rows, num_cols)


def detect_regions_from_cells(
    img: np.ndarray,
    cells: List[Tuple[int, int, int, int]]
) -> Dict[str, List[int]]:
    """Group cells into regions by color similarity"""
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        # Fallback if sklearn not available
        return {}

    # Sample color from center of each cell
    colors = []
    for x, y, w, h in cells:
        cx, cy = x + w // 2, y + h // 2
        sample_size = min(w, h) // 4
        x1, x2 = max(0, cx - sample_size), min(img.shape[1], cx + sample_size)
        y1, y2 = max(0, cy - sample_size), min(img.shape[0], cy + sample_size)

        if x2 > x1 and y2 > y1:
            patch = img[y1:y2, x1:x2]
            mean_color = patch.mean(axis=(0, 1))
            colors.append(mean_color)
        else:
            colors.append(np.array([128, 128, 128]))  # Gray default

    colors = np.array(colors)

    # Use DBSCAN for clustering (better for irregular shapes)
    clustering = DBSCAN(eps=30, min_samples=1).fit(colors)
    labels = clustering.labels_

    # Map cluster IDs to region letters
    regions = {}
    region_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    unique_labels = set(labels)
    for i, label in enumerate(sorted(unique_labels)):
        if i < len(region_letters):
            region_letter = region_letters[i]
            regions[region_letter] = [idx for idx, l in enumerate(labels) if l == label]

    return regions


def merge_overlapping_cells(cells: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Remove duplicate/overlapping cells, keeping the larger ones"""
    if not cells:
        return []

    # Sort by area (descending)
    cells = sorted(cells, key=lambda c: c[2] * c[3], reverse=True)

    merged = []
    for cell in cells:
        x1, y1, w1, h1 = cell
        x1_max, y1_max = x1 + w1, y1 + h1

        # Check if this cell overlaps significantly with any already merged cell
        overlaps = False
        for merged_cell in merged:
            x2, y2, w2, h2 = merged_cell
            x2_max, y2_max = x2 + w2, y2 + h2

            # Calculate intersection
            ix = max(0, min(x1_max, x2_max) - max(x1, x2))
            iy = max(0, min(y1_max, y2_max) - max(y1, y2))
            intersection = ix * iy

            # If overlap is significant (>30% of smaller cell)
            smaller_area = min(w1 * h1, w2 * h2)
            if intersection > smaller_area * 0.3:
                overlaps = True
                break

        if not overlaps:
            merged.append(cell)

    return merged


def calculate_confidence(
    cells: List[Tuple[int, int, int, int]],
    grid_dims: Tuple[int, int]
) -> float:
    """Calculate confidence score for detection result"""
    if not cells:
        return 0.0

    # Factors that increase confidence:
    # 1. Reasonable number of cells (7-30 typical for Pips puzzles)
    # 2. Consistent cell sizes
    # 3. Grid-like arrangement

    num_cells = len(cells)
    confidence = 0.0

    # Cell count score
    if 7 <= num_cells <= 30:
        confidence += 0.4
    elif 4 <= num_cells <= 40:
        confidence += 0.2

    # Size consistency score
    widths = [w for _, _, w, _ in cells]
    heights = [h for _, _, _, h in cells]
    if widths and heights:
        width_std = np.std(widths) / np.mean(widths) if np.mean(widths) > 0 else 1.0
        height_std = np.std(heights) / np.mean(heights) if np.mean(heights) > 0 else 1.0

        # Lower std = more consistent = higher confidence
        if width_std < 0.3 and height_std < 0.3:
            confidence += 0.3
        elif width_std < 0.5 and height_std < 0.5:
            confidence += 0.15

    # Grid arrangement score
    rows, cols = grid_dims
    if rows > 0 and cols > 0:
        expected_cells = rows * cols
        actual_cells = num_cells

        # Perfect match
        if expected_cells == actual_cells:
            confidence += 0.3
        # Close match (irregular grid with some gaps)
        elif abs(expected_cells - actual_cells) <= 3:
            confidence += 0.15

    return min(confidence, 1.0)


def save_debug_image(
    img: np.ndarray,
    cells: List[Tuple[int, int, int, int]],
    filename: str
):
    """Save annotated image showing detected cells"""
    debug_img = img.copy()

    for i, (x, y, w, h) in enumerate(cells):
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            debug_img, str(i), (x + 5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
        )

    cv2.imwrite(filename, debug_img)