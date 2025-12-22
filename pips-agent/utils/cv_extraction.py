"""
Computer Vision Extraction Utilities

Wraps the existing extract_board_cells_gridlines.py functionality to extract
puzzle structure from screenshots.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
from typing import List, Tuple, Dict, Set

# Add parent directory to path to import existing modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from existing CV extraction script
from extract_board_cells_gridlines import extract_cells_from_screenshot

Coord = Tuple[int, int]  # (row, col)


def extract_puzzle_structure(
    image_path: str,
    output_dir: str = None,
    lower_half_only: bool = False
) -> Dict:
    """
    Extract puzzle structure from screenshot.

    Args:
        image_path: Path to screenshot image
        output_dir: Directory for debug output (default: pips-agent/debug)
        lower_half_only: If True, only analyze lower half of image (default: False)

    Returns:
        Dictionary with:
        - cells: List of (x, y, w, h) tuples
        - grid_dims: (rows, cols) tuple
        - regions: Dict mapping region_id to list of cell indices
        - success: Boolean indicating if extraction succeeded
    """
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / "debug")

    try:
        # Run CV extraction
        extract_cells_from_screenshot(
            image_path,
            out_dir=output_dir,
            lower_half_only=lower_half_only
        )

        # Read cells.txt output
        cells_file = Path("cells.txt")
        if not cells_file.exists():
            return {
                "success": False,
                "error": "cells.txt not found after extraction"
            }

        cells = []
        with open(cells_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 4:
                    x, y, w, h = map(int, parts)
                    cells.append((x, y, w, h))

        if not cells:
            return {
                "success": False,
                "error": "No cells detected in screenshot"
            }

        # Infer grid dimensions
        grid_dims = infer_grid_layout(cells)

        # Detect regions by color
        regions = detect_regions_by_color(image_path, cells, grid_dims)

        return {
            "success": True,
            "cells": cells,
            "grid_dims": grid_dims,
            "regions": regions,
            "num_cells": len(cells)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def infer_grid_layout(cells: List[Tuple[int, int, int, int]]) -> Tuple[int, int]:
    """
    Infer grid dimensions (rows, cols) from cell positions.

    Args:
        cells: List of (x, y, w, h) tuples

    Returns:
        (rows, cols) tuple
    """
    if not cells:
        return (0, 0)

    # Sort by y-coordinate to find rows
    cells_by_y = sorted(cells, key=lambda c: c[1])

    # Group cells into rows (cells with similar y-coordinates)
    rows = []
    current_row = [cells_by_y[0]]
    y_threshold = cells_by_y[0][3] * 0.3  # 30% of cell height

    for cell in cells_by_y[1:]:
        if abs(cell[1] - current_row[0][1]) <= y_threshold:
            current_row.append(cell)
        else:
            rows.append(current_row)
            current_row = [cell]
    rows.append(current_row)

    num_rows = len(rows)

    # Count columns from first row
    num_cols = len(rows[0]) if rows else 0

    return (num_rows, num_cols)


def detect_regions_by_color(
    image_path: str,
    cells: List[Tuple[int, int, int, int]],
    grid_dims: Tuple[int, int]
) -> Dict[str, List[int]]:
    """
    Detect regions by clustering cell colors.

    Args:
        image_path: Path to original image
        cells: List of (x, y, w, h) tuples
        grid_dims: (rows, cols) tuple

    Returns:
        Dictionary mapping region letters (A, B, C...) to list of cell indices
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"A": list(range(len(cells)))}  # Fallback: all cells in one region

    # Sample color from center of each cell
    colors = []
    for x, y, w, h in cells:
        # Sample from center region (avoid borders)
        cx, cy = x + w // 2, y + h // 2
        sample_size = min(w, h) // 4
        x1, x2 = cx - sample_size, cx + sample_size
        y1, y2 = cy - sample_size, cy + sample_size

        patch = img[y1:y2, x1:x2]
        mean_color = patch.mean(axis=(0, 1))
        colors.append(mean_color)

    colors = np.array(colors, dtype=np.float32)

    # Use K-means clustering to group by color
    from sklearn.cluster import KMeans

    # Estimate number of clusters (regions)
    # Assume 3-8 regions typically
    n_clusters = min(max(3, len(cells) // 5), 8)

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(colors)
    except:
        # Fallback if sklearn not available or clustering fails
        labels = np.zeros(len(cells), dtype=int)

    # Map cluster IDs to region letters
    regions = {}
    region_letters = "ABCDEFGHIJKLMNOP"

    for i in range(n_clusters):
        region_letter = region_letters[i]
        regions[region_letter] = [idx for idx, label in enumerate(labels) if label == i]

    return regions
