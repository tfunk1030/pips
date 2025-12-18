"""
Enhanced Computer Vision Extraction (Version 2)

Improved extraction using multiple detection strategies for complex/irregular grids.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

Coord = Tuple[int, int]


@dataclass
class DetectionResult:
    """Result from a detection strategy"""
    success: bool
    cells: List[Tuple[int, int, int, int]]  # (x, y, w, h)
    grid_dims: Optional[Tuple[int, int]]
    regions: Optional[Dict[str, List[int]]]
    confidence: float
    method: str
    error: Optional[str] = None


def detect_by_region_contours(image_path: str, debug_dir: str = None) -> DetectionResult:
    """
    Strategy 1: Detect cells by finding colored region contours.

    Works better for irregular grids where cells are grouped by color.

    Steps:
    1. Load image and convert to different color spaces
    2. Segment by color to find distinct regions
    3. Find contours of each region
    4. Identify cell boundaries from region edges
    5. Reconstruct grid from boundaries
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="region_contours",
                error="Could not read image"
            )

        # Convert to LAB color space (better for color clustering)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Blur to reduce noise
        blurred = cv2.GaussianBlur(lab, (5, 5), 0)

        # Find edges (region boundaries)
        l, a, b = cv2.split(blurred)
        edges = cv2.Canny(l, 50, 150)

        # Dilate edges to connect nearby boundaries
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area (likely cells)
        cells = []
        min_area = 1000  # Minimum cell area in pixels
        max_area = 50000  # Maximum cell area

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)

                # Filter by aspect ratio (cells shouldn't be too elongated)
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 3.0:
                    cells.append((x, y, w, h))

        if len(cells) == 0:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="region_contours",
                error="No cells detected"
            )

        # Estimate grid dimensions
        grid_dims = estimate_grid_dims(cells)

        # Detect regions by color
        regions = detect_regions_from_cells(img, cells)

        # Calculate confidence based on cell count and regularity
        confidence = calculate_confidence(cells, grid_dims)

        # Save debug image if requested
        if debug_dir:
            save_debug_image(img, cells, f"{debug_dir}/contours_method.png")

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


def detect_by_constraint_labels(image_path: str, debug_dir: str = None) -> DetectionResult:
    """
    Strategy 2: Detect cells using constraint diamond labels as anchors.

    Diamond markers indicate region boundaries, use them to infer cell layout.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="constraint_labels",
                error="Could not read image"
            )

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect diamond shapes using shape detection
        diamonds = detect_diamond_shapes(img)

        if len(diamonds) == 0:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="constraint_labels",
                error="No constraint labels detected"
            )

        # Use diamond positions to infer grid structure
        # Diamonds are between regions, so cells are in the spaces
        cells = infer_cells_from_markers(img, diamonds)

        if len(cells) == 0:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="constraint_labels",
                error="Could not infer cells from markers"
            )

        grid_dims = estimate_grid_dims(cells)
        regions = detect_regions_from_cells(img, cells)
        confidence = calculate_confidence(cells, grid_dims)

        if debug_dir:
            save_debug_image(img, cells, f"{debug_dir}/constraint_method.png")

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


def detect_by_color_segmentation(image_path: str, debug_dir: str = None) -> DetectionResult:
    """
    Strategy 3: Segment by color first, then find cells within each color region.

    Better for puzzles with distinct colored regions.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="color_segmentation",
                error="Could not read image"
            )

        # Convert to LAB for better color segmentation
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Reshape for k-means clustering
        pixels = lab.reshape(-1, 3).astype(np.float32)

        # Use k-means to find dominant colors
        n_colors = 10  # Expect ~7 regions + background + borders
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        # Reshape labels back to image shape
        segmented = labels.reshape(img.shape[:2])

        # For each color cluster, find contiguous regions
        cells = []
        for color_idx in range(n_colors):
            mask = (segmented == color_idx).astype(np.uint8) * 255

            # Find connected components in this color
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 50000:  # Filter by size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.3 < aspect_ratio < 3.0:
                        cells.append((x, y, w, h))

        if len(cells) == 0:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="color_segmentation",
                error="No cells detected"
            )

        grid_dims = estimate_grid_dims(cells)
        regions = detect_regions_from_cells(img, cells)
        confidence = calculate_confidence(cells, grid_dims)

        if debug_dir:
            save_debug_image(img, cells, f"{debug_dir}/color_seg_method.png")

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


# Helper functions

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
    from sklearn.cluster import DBSCAN

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


def detect_diamond_shapes(img: np.ndarray) -> List[Tuple[int, int]]:
    """Detect diamond-shaped constraint labels"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    diamonds = []
    for contour in contours:
        # Approximate contour to polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Diamonds have 4 vertices
        if len(approx) == 4:
            # Check if it's roughly diamond-shaped (not rectangle)
            # Compute angle between edges
            # For now, just collect all quadrilaterals
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                diamonds.append((cx, cy))

    return diamonds


def infer_cells_from_markers(
    img: np.ndarray,
    markers: List[Tuple[int, int]]
) -> List[Tuple[int, int, int, int]]:
    """
    Infer cell positions from constraint marker positions.

    Markers are between cells, so cells are in the gaps.
    """
    # This is a complex heuristic - for now, return empty
    # TODO: Implement full marker-based cell inference
    return []


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
