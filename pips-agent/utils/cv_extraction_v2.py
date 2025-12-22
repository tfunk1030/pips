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

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
from pathlib import Path


@dataclass
class DetectionResult:
    """Result from a detection strategy."""
    success: bool
    cells: List[Tuple[int, int, int, int]]  # List of (x, y, w, h) bounding boxes
    grid_dims: Optional[Tuple[int, int]]  # (rows, cols)
    regions: Optional[Dict[int, List[Tuple[int, int]]]]  # region_id -> [(row, col), ...]
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
    if fill_ratio < 0.40 or fill_ratio > 0.65:  # Tighter range around 0.5
        return False

    # Check angles - diamond corners should have specific angles
    angles = []
    for i in range(4):
        p0 = pts[(i - 1) % 4]
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]

        v1 = p0 - p1
        v2 = p2 - p1

        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        angles.append(angle)

    # For a proper diamond, angles should sum to 360
    angle_sum = sum(angles)
    if abs(angle_sum - 360) > 20:
        return False

    # Diamonds have alternating acute and obtuse angles
    # Sort angles and check pattern
    sorted_angles = sorted(angles)

    # For a square diamond rotated 45 degrees, all angles should be ~90 degrees
    # For a stretched diamond, we should have pairs of similar angles
    angle_variance = np.std(angles)

    # If variance is low, all angles are similar (square diamond)
    if angle_variance < 15:
        # All angles should be close to 90 degrees
        for angle in angles:
            if angle < 60 or angle > 120:
                return False
    else:
        # Should have two pairs of opposite equal angles
        # Check that we have at least 2 angles in a reasonable range
        valid_angles = [a for a in angles if 50 < a < 130]
        if len(valid_angles) < 3:
            return False

    # Additional check: vertices should form a pattern where top/bottom or left/right
    # vertices are extreme (characteristic of a rotated square)
    sorted_by_y = sorted(range(4), key=lambda i: pts[i][1])
    sorted_by_x = sorted(range(4), key=lambda i: pts[i][0])

    # Check if extremes are single points (not edges)
    # This helps distinguish diamonds from rectangles
    top_idx = sorted_by_y[0]
    bottom_idx = sorted_by_y[-1]
    left_idx = sorted_by_x[0]
    right_idx = sorted_by_x[-1]

    # All four should be different vertices for a proper diamond
    extreme_vertices = {top_idx, bottom_idx, left_idx, right_idx}
    if len(extreme_vertices) < 3:  # Allow some tolerance
        return False

    return True


def detect_by_constraint_labels(image_path: str, debug_dir: str = None) -> DetectionResult:
    """
    Strategy 3: Detect cells by finding diamond-shaped constraint markers.

    Works for puzzles that use diamond markers at cell boundaries.

    Steps:
    1. Load image and preprocess
    2. Detect diamond shapes using contour analysis
    3. Infer cell grid from diamond positions
    4. Calculate cell bounding boxes
    5. Detect regions from inferred cells
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="constraint_labels",
                error="Could not read image"
            )

        h, w = img.shape[:2]

        # Scale area thresholds based on image size
        img_area = h * w
        min_diamond_area = max(50, int(img_area * 0.00001))  # Very small markers
        max_diamond_area = min(50000, int(img_area * 0.01))  # Up to 1% of image

        # Detect diamonds
        diamonds = detect_diamonds(img, min_diamond_area, max_diamond_area)

        if len(diamonds) < 4:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="constraint_labels",
                error=f"Too few diamonds detected ({len(diamonds)}), need at least 4 to infer grid"
            )

        # Extract diamond centers
        diamond_centers = [(d[0], d[1]) for d in diamonds]

        # Infer cells from diamond positions
        cells = infer_cells_from_diamonds(diamond_centers, img.shape)

        if len(cells) == 0:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="constraint_labels",
                error="No cells inferred from diamond markers"
            )

        grid_dims = estimate_grid_dims(cells)
        regions = detect_regions_from_cells(img, cells)
        confidence = calculate_confidence(cells, grid_dims)

        # Adjust confidence based on diamond detection quality
        diamond_confidence_factor = min(1.0, len(diamonds) / 10.0)  # More diamonds = more confidence
        confidence = confidence * diamond_confidence_factor

        if debug_dir:
            # Save debug image with diamonds and cells marked
            save_debug_image_with_diamonds(img, cells, diamonds, f"{debug_dir}/constraint_labels_method.png")

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


def infer_cells_from_diamonds(diamond_centers: List[Tuple[int, int]], img_shape: Tuple[int, ...]) -> List[Tuple[int, int, int, int]]:
    """
    Infer cell positions from detected diamond marker positions.

    Diamonds can mark either:
    1. Cell corners (grid intersections) - cells are between diamonds
    2. Cell centers - cells are centered on diamonds

    This function detects the pattern and calculates cell bounding boxes accordingly.

    Args:
        diamond_centers: List of (x, y) centers for detected diamonds
        img_shape: Shape of the image (h, w, ...)

    Returns:
        List of (x, y, w, h) cell bounding boxes
    """
    if len(diamond_centers) < 4:
        return []

    h, w = img_shape[:2]

    # Sort diamonds by position to find grid structure
    centers = np.array(diamond_centers)

    # Find unique X and Y positions (with tolerance for alignment)
    x_coords = sorted(centers[:, 0])
    y_coords = sorted(centers[:, 1])

    # Cluster X coordinates to find columns
    x_clusters = cluster_coordinates(x_coords)
    y_clusters = cluster_coordinates(y_coords)

    if len(x_clusters) < 2 or len(y_clusters) < 2:
        # Not enough structure to infer grid
        return []

    # Calculate spacing between clusters
    x_spacing = calculate_median_spacing(x_clusters)
    y_spacing = calculate_median_spacing(y_clusters)

    if x_spacing <= 0 or y_spacing <= 0:
        return []

    # Determine if diamonds mark corners or centers
    layout_pattern = detect_diamond_layout_pattern(
        diamond_centers, x_clusters, y_clusters, x_spacing, y_spacing
    )

    cells = []

    if layout_pattern == "corners":
        # Diamonds mark cell corners (grid intersections)
        # Cells are in the spaces between diamonds
        for i in range(len(y_clusters) - 1):
            for j in range(len(x_clusters) - 1):
                x1 = x_clusters[j]
                x2 = x_clusters[j + 1]
                y1 = y_clusters[i]
                y2 = y_clusters[i + 1]

                cell_x = int(x1)
                cell_y = int(y1)
                cell_w = int(x2 - x1)
                cell_h = int(y2 - y1)

                # Validate cell dimensions
                if cell_w > 10 and cell_h > 10:
                    cells.append((cell_x, cell_y, cell_w, cell_h))
    else:
        # Diamonds mark cell centers
        # Cells are centered on each diamond position
        cell_w = int(x_spacing)
        cell_h = int(y_spacing)

        # Create a grid of expected positions based on clusters
        for i, y_center in enumerate(y_clusters):
            for j, x_center in enumerate(x_clusters):
                # Check if there's actually a diamond near this grid position
                if has_diamond_near(diamond_centers, x_center, y_center, x_spacing, y_spacing):
                    # Cell is centered on the diamond
                    cell_x = int(x_center - cell_w / 2)
                    cell_y = int(y_center - cell_h / 2)

                    # Ensure cell is within image bounds
                    cell_x = max(0, cell_x)
                    cell_y = max(0, cell_y)

                    # Validate cell dimensions
                    if cell_w > 10 and cell_h > 10:
                        cells.append((cell_x, cell_y, cell_w, cell_h))

    return cells


def detect_diamond_layout_pattern(
    diamond_centers: List[Tuple[int, int]],
    x_clusters: List[float],
    y_clusters: List[float],
    x_spacing: float,
    y_spacing: float
) -> str:
    """
    Detect whether diamonds mark cell corners or cell centers.

    Corner pattern: Diamonds at grid intersections, (N+1) x (M+1) diamonds for N x M cells
    Center pattern: Diamonds at cell centers, exactly one diamond per cell

    The key heuristics:
    1. Corner patterns tend to have diamonds at the edges of the grid
    2. Center patterns have more evenly distributed diamonds within the grid area
    3. For corners, there should be ~1 more row/col of diamonds than cells
    4. For centers, diamond count should closely match expected cell count

    Args:
        diamond_centers: List of (x, y) diamond center positions
        x_clusters: Clustered x-coordinates of diamonds
        y_clusters: Clustered y-coordinates of diamonds
        x_spacing: Median horizontal spacing between clusters
        y_spacing: Median vertical spacing between clusters

    Returns:
        "corners" or "centers" indicating the detected pattern
    """
    num_diamonds = len(diamond_centers)
    num_x_clusters = len(x_clusters)
    num_y_clusters = len(y_clusters)

    # For corners pattern: We expect (rows+1) * (cols+1) diamonds
    # For centers pattern: We expect rows * cols diamonds

    # Expected diamonds if corners pattern
    corners_expected = num_x_clusters * num_y_clusters

    # If we have corner pattern, there would be (N-1)*(M-1) cells
    corners_cell_count = (num_x_clusters - 1) * (num_y_clusters - 1)

    # If we have center pattern, there would be N*M cells (same as diamond count per cluster)
    centers_cell_count = num_x_clusters * num_y_clusters

    # Check how many diamonds actually exist at cluster intersections
    # For a proper corner pattern, most cluster intersections should have diamonds
    diamonds_at_intersections = 0
    tolerance = min(x_spacing, y_spacing) * 0.4

    for x_c in x_clusters:
        for y_c in y_clusters:
            if has_diamond_near(diamond_centers, x_c, y_c, x_spacing, y_spacing, tolerance):
                diamonds_at_intersections += 1

    # Calculate fill ratios
    intersection_fill_ratio = diamonds_at_intersections / corners_expected if corners_expected > 0 else 0

    # Heuristic 1: Check if diamonds fill the grid intersections well
    # Corner patterns should have high fill ratio at intersections
    if intersection_fill_ratio > 0.7:
        # Most intersections have diamonds - likely corners pattern
        # Additional check: corners pattern should have at least 2x2 grid of diamonds
        if num_x_clusters >= 2 and num_y_clusters >= 2:
            # Check cell count reasonableness
            if corners_cell_count >= 1:
                return "corners"

    # Heuristic 2: Check diamond distribution relative to cluster grid
    # For center pattern, diamonds should be well-distributed within clusters
    # and there should be a reasonable number of cells
    if centers_cell_count >= 4 and intersection_fill_ratio > 0.5:
        # Moderate fill with reasonable cell count
        # Check if spacing suggests cells (not corners)
        # In corners pattern, spacing is cell size
        # In centers pattern, spacing is also cell size but diamonds are centered

        # Additional heuristic: check border behavior
        # Corner patterns often have diamonds at the very edges
        # Center patterns have diamonds set in from edges by half a cell

        centers = np.array(diamond_centers)
        min_x, max_x = centers[:, 0].min(), centers[:, 0].max()
        min_y, max_y = centers[:, 1].min(), centers[:, 1].max()

        # Calculate how close the outermost diamonds are to grid edges
        x_margin_ratio = (max_x - min_x) / (num_x_clusters * x_spacing) if num_x_clusters > 1 else 1
        y_margin_ratio = (max_y - min_y) / (num_y_clusters * y_spacing) if num_y_clusters > 1 else 1

        # For center patterns, the span should be about (N-1) * spacing
        # For corner patterns, the span should also be about (N-1) * spacing
        # This heuristic isn't very distinctive, so default to corners if ambiguous

    # Default to corners pattern as it's more common
    return "corners"


def has_diamond_near(
    diamond_centers: List[Tuple[int, int]],
    x: float,
    y: float,
    x_spacing: float,
    y_spacing: float,
    tolerance: float = None
) -> bool:
    """
    Check if there's a diamond near the given position.

    Args:
        diamond_centers: List of (x, y) diamond centers
        x, y: Position to check
        x_spacing, y_spacing: Grid spacing (used for default tolerance)
        tolerance: Maximum distance to consider "near" (default: 30% of spacing)

    Returns:
        True if a diamond exists near the position
    """
    if tolerance is None:
        tolerance = min(x_spacing, y_spacing) * 0.3

    for dx, dy in diamond_centers:
        dist = np.sqrt((dx - x) ** 2 + (dy - y) ** 2)
        if dist <= tolerance:
            return True

    return False


def cluster_coordinates(coords: List[float], tolerance: float = None) -> List[float]:
    """
    Cluster nearby coordinates to find discrete grid positions.

    Args:
        coords: Sorted list of coordinates
        tolerance: Max distance to consider coordinates as same cluster

    Returns:
        List of cluster center values
    """
    if len(coords) == 0:
        return []

    # Auto-calculate tolerance based on median spacing
    if tolerance is None:
        if len(coords) >= 2:
            diffs = [coords[i+1] - coords[i] for i in range(len(coords)-1)]
            median_diff = np.median(diffs) if diffs else 20
            tolerance = max(10, median_diff * 0.3)  # 30% of median spacing
        else:
            tolerance = 20

    clusters = []
    current_cluster = [coords[0]]

    for i in range(1, len(coords)):
        if coords[i] - current_cluster[-1] <= tolerance:
            current_cluster.append(coords[i])
        else:
            # Finalize current cluster
            clusters.append(np.mean(current_cluster))
            current_cluster = [coords[i]]

    # Don't forget the last cluster
    if current_cluster:
        clusters.append(np.mean(current_cluster))

    return clusters


def calculate_median_spacing(clusters: List[float]) -> float:
    """Calculate median spacing between cluster centers."""
    if len(clusters) < 2:
        return 0

    spacings = [clusters[i+1] - clusters[i] for i in range(len(clusters)-1)]
    return float(np.median(spacings))


# =============================================================================
# Helper Functions (shared across strategies)
# =============================================================================

def estimate_grid_dims(cells: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int]]:
    """
    Estimate grid dimensions (rows, cols) from detected cells.

    Uses cell positions to infer the grid layout.
    """
    if len(cells) == 0:
        return None

    # Get cell centers
    centers = [(x + w//2, y + h//2) for x, y, w, h in cells]

    # Cluster by Y to find rows
    y_coords = sorted([c[1] for c in centers])
    x_coords = sorted([c[0] for c in centers])

    # Simple clustering: count unique positions with tolerance
    def count_unique(coords, tolerance=None):
        if len(coords) == 0:
            return 0
        if tolerance is None:
            # Auto-calculate from median spacing
            if len(coords) >= 2:
                diffs = [coords[i+1] - coords[i] for i in range(len(coords)-1)]
                tolerance = max(10, np.median(diffs) * 0.4)
            else:
                tolerance = 20

        unique = [coords[0]]
        for c in coords[1:]:
            if c - unique[-1] > tolerance:
                unique.append(c)
        return len(unique)

    rows = count_unique(y_coords)
    cols = count_unique(x_coords)

    return (rows, cols) if rows > 0 and cols > 0 else None


def detect_regions_from_cells(img: np.ndarray, cells: List[Tuple[int, int, int, int]]) -> Dict[int, List[Tuple[int, int]]]:
    """
    Group cells into regions based on color similarity.

    Uses k-means clustering on average cell colors to determine regions.
    """
    if len(cells) == 0:
        return {}

    # Sample color from each cell center
    colors = []
    for x, y, w, h in cells:
        # Sample from center of cell
        cx, cy = x + w//2, y + h//2
        # Ensure within bounds
        cy = max(0, min(cy, img.shape[0] - 1))
        cx = max(0, min(cx, img.shape[1] - 1))

        # Sample a small region around center
        y1, y2 = max(0, cy - 5), min(img.shape[0], cy + 5)
        x1, x2 = max(0, cx - 5), min(img.shape[1], cx + 5)

        region = img[y1:y2, x1:x2]
        if region.size > 0:
            avg_color = region.mean(axis=(0, 1))
        else:
            avg_color = img[cy, cx]
        colors.append(avg_color)

    colors = np.array(colors, dtype=np.float32)

    # Determine number of clusters (regions)
    # Typically 5-15 regions for Pips puzzles
    k = min(max(5, len(cells) // 3), 15)
    k = min(k, len(cells))  # Can't have more clusters than cells

    if k < 2 or len(colors) < 2:
        # All cells in one region
        return {0: [(0, i) for i in range(len(cells))]}

    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, _ = cv2.kmeans(colors, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()

    # Build regions dict
    # First, estimate grid positions for cells
    grid_dims = estimate_grid_dims(cells)
    if grid_dims is None:
        # Fall back to linear indexing
        regions = {}
        for i, label in enumerate(labels):
            label = int(label)
            if label not in regions:
                regions[label] = []
            regions[label].append((0, i))  # (row, col) approximation
        return regions

    rows, cols = grid_dims

    # Sort cells by position to assign grid coordinates
    cell_info = [(x, y, w, h, i) for i, (x, y, w, h) in enumerate(cells)]
    cell_info.sort(key=lambda c: (c[1], c[0]))  # Sort by y, then x

    regions = {}
    for idx, (x, y, w, h, orig_idx) in enumerate(cell_info):
        row = idx // cols if cols > 0 else 0
        col = idx % cols if cols > 0 else idx
        label = int(labels[orig_idx])

        if label not in regions:
            regions[label] = []
        regions[label].append((row, col))

    return regions


def calculate_confidence(cells: List[Tuple[int, int, int, int]], grid_dims: Optional[Tuple[int, int]]) -> float:
    """
    Calculate confidence score (0.0-1.0) for detection quality.

    Higher confidence when:
    - Cell count is reasonable (7-30 for typical puzzles)
    - Cell sizes are consistent
    - Arrangement is grid-like
    """
    if len(cells) == 0:
        return 0.0

    confidence = 0.5  # Base confidence

    # Cell count factor
    count = len(cells)
    if 7 <= count <= 30:
        confidence += 0.2  # Ideal range
    elif 4 <= count <= 50:
        confidence += 0.1  # Acceptable range
    else:
        confidence -= 0.2  # Unusual count

    # Size consistency factor
    areas = [w * h for x, y, w, h in cells]
    if len(areas) >= 2:
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        cv = std_area / mean_area if mean_area > 0 else 1.0  # Coefficient of variation

        if cv < 0.2:
            confidence += 0.15  # Very consistent
        elif cv < 0.4:
            confidence += 0.05  # Reasonably consistent
        else:
            confidence -= 0.1  # Inconsistent sizes

    # Grid structure factor
    if grid_dims:
        rows, cols = grid_dims
        expected_count = rows * cols
        if abs(count - expected_count) / expected_count < 0.1:
            confidence += 0.15  # Grid structure matches cell count
        elif abs(count - expected_count) / expected_count < 0.3:
            confidence += 0.05  # Roughly matches

    # Clamp to valid range
    return max(0.0, min(1.0, confidence))


def merge_overlapping_cells(cells: List[Tuple[int, int, int, int]], overlap_thresh: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """
    Merge cells that significantly overlap.

    Uses non-maximum suppression approach.
    """
    if len(cells) <= 1:
        return cells

    # Convert to numpy for easier manipulation
    boxes = np.array(cells)

    # Calculate areas
    areas = boxes[:, 2] * boxes[:, 3]

    # Sort by area (descending)
    indices = np.argsort(-areas)

    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)

        if len(indices) == 1:
            break

        # Calculate IoU with remaining boxes
        remaining = indices[1:]

        # Get intersection
        x1 = np.maximum(boxes[i, 0], boxes[remaining, 0])
        y1 = np.maximum(boxes[i, 1], boxes[remaining, 1])
        x2 = np.minimum(boxes[i, 0] + boxes[i, 2], boxes[remaining, 0] + boxes[remaining, 2])
        y2 = np.minimum(boxes[i, 1] + boxes[i, 3], boxes[remaining, 1] + boxes[remaining, 3])

        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)
        intersection = w * h

        # Calculate IoU
        union = areas[i] + areas[remaining] - intersection
        iou = intersection / (union + 1e-6)

        # Keep boxes with low overlap
        indices = remaining[iou < overlap_thresh]

    return [tuple(boxes[i]) for i in keep]


def save_debug_image(img: np.ndarray, cells: List[Tuple[int, int, int, int]], output_path: str):
    """Save debug image with cells drawn."""
    debug_img = img.copy()

    for x, y, w, h in cells:
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, debug_img)


def save_debug_image_with_diamonds(img: np.ndarray, cells: List[Tuple[int, int, int, int]],
                                   diamonds: List[Tuple[int, int, int, int, np.ndarray]],
                                   output_path: str):
    """Save debug image with cells and diamond markers drawn."""
    debug_img = img.copy()

    # Draw cells in green
    for x, y, w, h in cells:
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw diamonds in red
    for cx, cy, w, h, contour in diamonds:
        cv2.drawContours(debug_img, [contour], 0, (0, 0, 255), 2)
        cv2.circle(debug_img, (cx, cy), 3, (255, 0, 0), -1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, debug_img)


# =============================================================================
# Multi-Strategy Extraction
# =============================================================================

def extract_puzzle_multi_strategy(image_path: str, output_dir: str = None,
                                  strategies: List[str] = None) -> Dict:
    """
    Extract puzzle cells using multiple detection strategies.

    Tries each strategy and returns the best result based on confidence.

    Args:
        image_path: Path to puzzle screenshot
        output_dir: Directory for debug output (optional)
        strategies: List of strategies to try (default: all)

    Returns:
        Dict with best result and all attempts:
        {
            "success": bool,
            "method_used": str,
            "cells": List[Tuple],
            "grid_dims": Tuple[int, int],
            "regions": Dict,
            "confidence": float,
            "num_cells": int,
            "all_attempts": List[Dict]
        }
    """
    if strategies is None:
        strategies = ["region_contours", "color_segmentation", "constraint_labels"]

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

    # Find best result by confidence
    successful = [r for r in results if r.success]

    if not successful:
        # All strategies failed
        all_attempts = [{
            "method": r.method,
            "success": r.success,
            "confidence": r.confidence,
            "error": r.error,
            "num_cells": len(r.cells)
        } for r in results]

        return {
            "success": False,
            "method_used": None,
            "cells": [],
            "grid_dims": None,
            "regions": None,
            "confidence": 0.0,
            "num_cells": 0,
            "all_attempts": all_attempts,
            "error": "All detection strategies failed"
        }

    # Select best by confidence
    best = max(successful, key=lambda r: r.confidence)

    all_attempts = [{
        "method": r.method,
        "success": r.success,
        "confidence": r.confidence,
        "error": r.error,
        "num_cells": len(r.cells)
    } for r in results]

    return {
        "success": True,
        "method_used": best.method,
        "cells": best.cells,
        "grid_dims": best.grid_dims,
        "regions": best.regions,
        "confidence": best.confidence,
        "num_cells": len(best.cells),
        "all_attempts": all_attempts
    }
