"""
CV Extraction V2 - Enhanced contour detection with polygon approximation.

This module provides enhanced contour detection capabilities using polygon
approximation with cv2.approxPolyDP for improved region boundary identification.
It handles irregular shapes, simplifies noisy contours, and provides shape
analysis for better region classification.

Key features:
- Polygon approximation with configurable epsilon for contour simplification
- Shape analysis (vertex count, corner detection)
- Contour filtering by area, aspect ratio, and shape characteristics
- Support for varying approximation accuracy based on contour perimeter
- Convex hull analysis for concave region detection and characterization
- Convexity defect analysis for identifying indentations and complex shapes
- Watershed algorithm for separating merged/touching regions
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path


@dataclass
class ApproximatedContour:
    """
    Result of polygon approximation for a single contour.

    Attributes:
        original_contour: The original contour points
        approximated_contour: Simplified contour after approxPolyDP
        epsilon: Epsilon value used for approximation
        original_vertex_count: Number of vertices in original contour
        approximated_vertex_count: Number of vertices after approximation
        area: Area of the approximated contour
        perimeter: Perimeter of the approximated contour
        bounding_rect: (x, y, w, h) bounding rectangle
        centroid: (cx, cy) centroid of the contour
        is_closed: Whether the contour is closed
        shape_type: Detected shape type ("triangle", "quadrilateral", "polygon", etc.)
        confidence: Confidence in the shape detection (0.0 to 1.0)
        hull_analysis: Optional convex hull analysis for concave region detection
        convexity_class: Classification based on convexity ("convex", "concave", etc.)
    """
    original_contour: np.ndarray
    approximated_contour: np.ndarray
    epsilon: float
    original_vertex_count: int
    approximated_vertex_count: int
    area: float
    perimeter: float
    bounding_rect: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    is_closed: bool = True
    shape_type: str = "polygon"
    confidence: float = 1.0
    hull_analysis: Optional["ConvexHullAnalysis"] = None
    convexity_class: str = "unknown"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "approximated_vertex_count": self.approximated_vertex_count,
            "original_vertex_count": self.original_vertex_count,
            "epsilon": round(self.epsilon, 4),
            "area": round(self.area, 2),
            "perimeter": round(self.perimeter, 2),
            "bounding_rect": list(self.bounding_rect),
            "centroid": [round(c, 2) for c in self.centroid],
            "is_closed": self.is_closed,
            "shape_type": self.shape_type,
            "confidence": round(self.confidence, 4),
            "convexity_class": self.convexity_class
        }
        if self.hull_analysis is not None:
            result["hull_analysis"] = self.hull_analysis.to_dict()
        return result


@dataclass
class ContourExtractionResult:
    """
    Result of contour extraction and approximation.

    Attributes:
        contours: List of ApproximatedContour objects
        total_contours_found: Total contours found before filtering
        contours_after_filtering: Number of contours after area/shape filtering
        method: Extraction method used
        epsilon_mode: Epsilon calculation mode ("adaptive", "fixed", "perimeter_ratio")
        confidence: Overall confidence in the extraction (0.0 to 1.0)
    """
    contours: List[ApproximatedContour]
    total_contours_found: int = 0
    contours_after_filtering: int = 0
    method: str = "approxPolyDP"
    epsilon_mode: str = "adaptive"
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "contours": [c.to_dict() for c in self.contours],
            "total_contours_found": self.total_contours_found,
            "contours_after_filtering": self.contours_after_filtering,
            "method": self.method,
            "epsilon_mode": self.epsilon_mode,
            "confidence": round(self.confidence, 4)
        }


@dataclass
class ConvexityDefect:
    """
    Represents a single convexity defect (indentation) in a contour.

    Convexity defects are points where the contour deviates inward from
    its convex hull. They help identify concave regions and complex shapes.

    Attributes:
        start_point: Starting point of the defect on the contour
        end_point: Ending point of the defect on the contour
        farthest_point: Point on the contour farthest from the hull
        depth: Distance from farthest point to the hull (in pixels)
        start_index: Index of start point in the contour
        end_index: Index of end point in the contour
        farthest_index: Index of farthest point in the contour
        angle: Angle of the defect (in degrees)
    """
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    farthest_point: Tuple[int, int]
    depth: float
    start_index: int
    end_index: int
    farthest_index: int
    angle: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_point": list(self.start_point),
            "end_point": list(self.end_point),
            "farthest_point": list(self.farthest_point),
            "depth": round(self.depth, 2),
            "angle": round(self.angle, 2)
        }


@dataclass
class ConvexHullAnalysis:
    """
    Result of convex hull analysis for a contour.

    Provides detailed information about the convex hull and any
    concavity characteristics of the original contour.

    Attributes:
        contour: Original contour analyzed
        hull: Convex hull of the contour
        hull_points: Hull as array of points (for drawing)
        is_convex: Whether the original contour is convex
        contour_area: Area of the original contour
        hull_area: Area of the convex hull
        solidity: Ratio of contour area to hull area (1.0 = perfectly convex)
        defects: List of convexity defects (indentations)
        defect_count: Number of significant defects
        max_defect_depth: Maximum depth of any defect
        avg_defect_depth: Average defect depth
        concavity_score: Overall concavity measure (0.0 = convex, 1.0 = highly concave)
        complexity_score: Measure of contour complexity
    """
    contour: np.ndarray
    hull: np.ndarray
    hull_points: np.ndarray
    is_convex: bool
    contour_area: float
    hull_area: float
    solidity: float
    defects: List[ConvexityDefect] = field(default_factory=list)
    defect_count: int = 0
    max_defect_depth: float = 0.0
    avg_defect_depth: float = 0.0
    concavity_score: float = 0.0
    complexity_score: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_convex": self.is_convex,
            "contour_area": round(self.contour_area, 2),
            "hull_area": round(self.hull_area, 2),
            "solidity": round(self.solidity, 4),
            "defect_count": self.defect_count,
            "max_defect_depth": round(self.max_defect_depth, 2),
            "avg_defect_depth": round(self.avg_defect_depth, 2),
            "concavity_score": round(self.concavity_score, 4),
            "complexity_score": round(self.complexity_score, 4),
            "defects": [d.to_dict() for d in self.defects]
        }


@dataclass
class WatershedSegment:
    """
    A single segment resulting from watershed segmentation.

    Attributes:
        label: Unique label for this segment (from watershed)
        contour: Contour of the segment
        area: Area of the segment
        centroid: (cx, cy) centroid of the segment
        bounding_rect: (x, y, w, h) bounding rectangle
        mask: Binary mask for this segment only
    """
    label: int
    contour: np.ndarray
    area: float
    centroid: Tuple[float, float]
    bounding_rect: Tuple[int, int, int, int]
    mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "label": self.label,
            "area": round(self.area, 2),
            "centroid": [round(c, 2) for c in self.centroid],
            "bounding_rect": list(self.bounding_rect)
        }


@dataclass
class WatershedResult:
    """
    Result of watershed segmentation for separating merged regions.

    The watershed algorithm treats the image as a topographic surface
    and floods from markers to find region boundaries. This is useful
    for separating touching or overlapping regions that appear merged.

    Attributes:
        segments: List of WatershedSegment objects
        markers: The marker image used for watershed
        labels: The final labeled image from watershed
        num_regions: Number of distinct regions found
        original_contour: The original merged contour that was split
        method: Method used for marker generation
        confidence: Confidence in the segmentation (0.0 to 1.0)
    """
    segments: List[WatershedSegment] = field(default_factory=list)
    markers: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    num_regions: int = 0
    original_contour: Optional[np.ndarray] = None
    method: str = "distance_transform"
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_regions": self.num_regions,
            "method": self.method,
            "confidence": round(self.confidence, 4),
            "segments": [s.to_dict() for s in self.segments]
        }


def approximate_contour(
    contour: np.ndarray,
    epsilon: Optional[float] = None,
    epsilon_ratio: float = 0.02,
    closed: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Approximate a contour using cv2.approxPolyDP with configurable epsilon.

    The Douglas-Peucker algorithm simplifies contours by removing vertices
    that don't significantly affect the contour shape. This is useful for:
    - Reducing noise in detected contours
    - Identifying geometric shapes (rectangles, polygons)
    - Simplifying complex region boundaries

    Args:
        contour: Input contour as numpy array of shape (N, 1, 2)
        epsilon: Fixed epsilon value for approximation. If None, computed
                 adaptively from contour perimeter using epsilon_ratio.
        epsilon_ratio: Ratio of perimeter to use as epsilon when epsilon is None.
                       Smaller values = more detail preserved (default 0.02 = 2%)
        closed: Whether to treat the contour as closed (default True)

    Returns:
        Tuple of (approximated_contour, epsilon_used):
        - approximated_contour: Simplified contour points
        - epsilon_used: The epsilon value that was used

    Example:
        >>> import cv2
        >>> import numpy as np
        >>> # Create a simple square contour
        >>> cnt = np.array([[[0,0]], [[10,0]], [[10,10]], [[0,10]]], dtype=np.int32)
        >>> approx, eps = approximate_contour(cnt, epsilon_ratio=0.01)
        >>> print(f"Vertices: {len(approx)}")
    """
    if contour is None or len(contour) < 3:
        return contour, 0.0

    # Calculate epsilon adaptively if not provided
    perimeter = cv2.arcLength(contour, closed)

    if epsilon is None:
        epsilon = perimeter * epsilon_ratio

    # Apply Douglas-Peucker algorithm
    approximated = cv2.approxPolyDP(contour, epsilon, closed)

    return approximated, epsilon


def detect_shape_type(
    contour: np.ndarray,
    approximated: np.ndarray
) -> Tuple[str, float]:
    """
    Detect the geometric shape type from an approximated contour.

    Analyzes vertex count and angles to classify the shape.

    Args:
        contour: Original contour
        approximated: Approximated contour from approxPolyDP

    Returns:
        Tuple of (shape_type, confidence):
        - shape_type: One of "triangle", "quadrilateral", "rectangle", "square",
                      "pentagon", "hexagon", "circle", "polygon"
        - confidence: Confidence in the shape classification (0.0 to 1.0)
    """
    vertex_count = len(approximated)

    if vertex_count < 3:
        return "line", 0.5

    if vertex_count == 3:
        return "triangle", 0.9

    if vertex_count == 4:
        # Check if it's a rectangle/square
        x, y, w, h = cv2.boundingRect(approximated)
        aspect_ratio = float(w) / h if h > 0 else 0

        # Calculate contour area vs bounding rect area
        contour_area = cv2.contourArea(approximated)
        rect_area = w * h
        fill_ratio = contour_area / rect_area if rect_area > 0 else 0

        # High fill ratio (> 0.85) suggests rectangle-like
        if fill_ratio > 0.85:
            if 0.9 <= aspect_ratio <= 1.1:
                return "square", 0.9
            else:
                return "rectangle", 0.85
        else:
            return "quadrilateral", 0.8

    if vertex_count == 5:
        return "pentagon", 0.8

    if vertex_count == 6:
        return "hexagon", 0.75

    if vertex_count > 8:
        # Check circularity: 4*pi*area / perimeter^2
        area = cv2.contourArea(approximated)
        perimeter = cv2.arcLength(approximated, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > 0.8:
                return "circle", circularity

    return "polygon", 0.7


def calculate_contour_centroid(contour: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the centroid of a contour using image moments.

    Args:
        contour: Input contour

    Returns:
        Tuple (cx, cy) centroid coordinates
    """
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        # Fallback to bounding rect center
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w / 2
        cy = y + h / 2

    return (cx, cy)


def compute_convex_hull(
    contour: np.ndarray,
    return_points: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the convex hull of a contour.

    The convex hull is the smallest convex polygon that encloses all points
    of the contour. It's useful for:
    - Detecting concave regions by comparing hull to original contour
    - Simplifying shape analysis
    - Finding convexity defects

    Args:
        contour: Input contour as numpy array
        return_points: If True, also return hull as array of points

    Returns:
        Tuple of (hull_indices, hull_points):
        - hull_indices: Convex hull as indices into the contour (for defects)
        - hull_points: Convex hull as array of points (for drawing)
    """
    if contour is None or len(contour) < 3:
        return np.array([]), np.array([])

    # Get hull as indices (needed for convexityDefects)
    hull_indices = cv2.convexHull(contour, returnPoints=False)

    # Get hull as points (needed for drawing and area calculation)
    hull_points = cv2.convexHull(contour, returnPoints=True)

    return hull_indices, hull_points


def calculate_defect_angle(
    start: Tuple[int, int],
    end: Tuple[int, int],
    farthest: Tuple[int, int]
) -> float:
    """
    Calculate the angle of a convexity defect.

    The angle is formed by the vectors from the farthest point to the
    start and end points of the defect.

    Args:
        start: Start point of the defect
        end: End point of the defect
        farthest: Farthest (deepest) point of the defect

    Returns:
        Angle in degrees (0-180)
    """
    # Vectors from farthest point to start and end
    v1 = np.array([start[0] - farthest[0], start[1] - farthest[1]])
    v2 = np.array([end[0] - farthest[0], end[1] - farthest[1]])

    # Calculate angle using dot product
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)

    if len1 == 0 or len2 == 0:
        return 0.0

    cos_angle = np.dot(v1, v2) / (len1 * len2)
    # Clamp to avoid numerical issues
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def find_convexity_defects(
    contour: np.ndarray,
    hull_indices: np.ndarray,
    min_depth: float = 5.0
) -> List[ConvexityDefect]:
    """
    Find convexity defects (indentations) in a contour.

    Convexity defects are regions where the contour deviates inward from
    its convex hull. They are essential for:
    - Detecting concave shapes
    - Identifying complex region boundaries
    - Splitting merged regions at concavity points

    Args:
        contour: Input contour
        hull_indices: Convex hull as indices (from cv2.convexHull with returnPoints=False)
        min_depth: Minimum depth (in pixels) to consider a defect significant

    Returns:
        List of ConvexityDefect objects sorted by depth (deepest first)
    """
    defects_list = []

    if contour is None or len(contour) < 4:
        return defects_list

    if hull_indices is None or len(hull_indices) < 3:
        return defects_list

    try:
        # Find convexity defects
        defects = cv2.convexityDefects(contour, hull_indices)

        if defects is None:
            return defects_list

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]

            # Depth is in fixed-point format (1/256 of a pixel)
            depth = d / 256.0

            if depth < min_depth:
                continue

            # Get the actual points
            start_point = tuple(contour[s][0])
            end_point = tuple(contour[e][0])
            farthest_point = tuple(contour[f][0])

            # Calculate angle at the defect
            angle = calculate_defect_angle(start_point, end_point, farthest_point)

            defect = ConvexityDefect(
                start_point=start_point,
                end_point=end_point,
                farthest_point=farthest_point,
                depth=depth,
                start_index=int(s),
                end_index=int(e),
                farthest_index=int(f),
                angle=angle
            )
            defects_list.append(defect)

    except cv2.error:
        # convexityDefects can fail on certain contour configurations
        pass

    # Sort by depth (deepest first)
    defects_list.sort(key=lambda d: d.depth, reverse=True)

    return defects_list


def analyze_convex_hull(
    contour: np.ndarray,
    min_defect_depth: float = 5.0
) -> ConvexHullAnalysis:
    """
    Perform comprehensive convex hull analysis on a contour.

    This function computes the convex hull and analyzes the contour's
    concavity characteristics, including:
    - Solidity (ratio of contour area to hull area)
    - Convexity defects (indentations)
    - Concavity and complexity scores

    Use cases:
    - Detecting irregular/concave puzzle regions
    - Identifying shapes that may need to be split
    - Classifying shape complexity

    Args:
        contour: Input contour as numpy array
        min_defect_depth: Minimum depth for defect detection

    Returns:
        ConvexHullAnalysis object with hull properties and defect analysis

    Example:
        >>> import cv2
        >>> import numpy as np
        >>> # Create an L-shaped contour (concave)
        >>> cnt = np.array([[[0,0]], [[100,0]], [[100,50]], [[50,50]],
        ...                 [[50,100]], [[0,100]]], dtype=np.int32)
        >>> analysis = analyze_convex_hull(cnt)
        >>> print(f"Solidity: {analysis.solidity:.2f}, Defects: {analysis.defect_count}")
    """
    if contour is None or len(contour) < 3:
        return ConvexHullAnalysis(
            contour=contour if contour is not None else np.array([]),
            hull=np.array([]),
            hull_points=np.array([]),
            is_convex=False,
            contour_area=0.0,
            hull_area=0.0,
            solidity=0.0
        )

    # Compute convex hull
    hull_indices, hull_points = compute_convex_hull(contour)

    # Check if contour is convex
    is_convex = cv2.isContourConvex(contour)

    # Calculate areas
    contour_area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(hull_points) if len(hull_points) > 0 else 0.0

    # Calculate solidity (1.0 = perfectly convex)
    solidity = contour_area / hull_area if hull_area > 0 else 0.0

    # Find convexity defects
    defects = []
    if not is_convex and len(hull_indices) >= 3:
        defects = find_convexity_defects(contour, hull_indices, min_defect_depth)

    # Calculate defect statistics
    defect_count = len(defects)
    max_defect_depth = max((d.depth for d in defects), default=0.0)
    avg_defect_depth = np.mean([d.depth for d in defects]) if defects else 0.0

    # Calculate concavity score (0.0 = convex, 1.0 = highly concave)
    # Based on solidity and defect characteristics
    if is_convex:
        concavity_score = 0.0
    else:
        # Factor 1: Inverse of solidity (lower solidity = more concave)
        solidity_factor = 1.0 - solidity

        # Factor 2: Normalized defect depth relative to contour size
        perimeter = cv2.arcLength(contour, True)
        depth_factor = min(1.0, max_defect_depth / (perimeter * 0.1)) if perimeter > 0 else 0.0

        # Factor 3: Number of defects (normalized)
        defect_factor = min(1.0, defect_count / 5.0)

        # Weighted combination
        concavity_score = (
            0.4 * solidity_factor +
            0.35 * depth_factor +
            0.25 * defect_factor
        )

    # Calculate complexity score
    # Based on vertex count, defects, and shape irregularity
    vertex_count = len(contour)
    complexity_score = min(1.0, (
        0.3 * min(1.0, vertex_count / 50.0) +
        0.3 * (1.0 - solidity) +
        0.2 * min(1.0, defect_count / 3.0) +
        0.2 * min(1.0, max_defect_depth / 20.0)
    ))

    return ConvexHullAnalysis(
        contour=contour,
        hull=hull_indices,
        hull_points=hull_points,
        is_convex=is_convex,
        contour_area=contour_area,
        hull_area=hull_area,
        solidity=solidity,
        defects=defects,
        defect_count=defect_count,
        max_defect_depth=max_defect_depth,
        avg_defect_depth=avg_defect_depth,
        concavity_score=concavity_score,
        complexity_score=complexity_score
    )


def split_concave_contour_at_defects(
    contour: np.ndarray,
    defects: List[ConvexityDefect],
    max_splits: int = 2,
    min_split_depth: float = 10.0
) -> List[np.ndarray]:
    """
    Split a concave contour at its deepest convexity defects.

    This is useful for separating merged regions that appear as a single
    concave shape. The function splits at the deepest defect points to
    create more convex sub-regions.

    Args:
        contour: Input contour to split
        defects: List of ConvexityDefect objects
        max_splits: Maximum number of splits to perform
        min_split_depth: Minimum defect depth to consider for splitting

    Returns:
        List of sub-contours (may be 1 if no splits performed)
    """
    if contour is None or len(contour) < 6:
        return [contour] if contour is not None else []

    # Filter defects by depth and limit count
    significant_defects = [
        d for d in defects
        if d.depth >= min_split_depth
    ][:max_splits]

    if not significant_defects:
        return [contour]

    # For simple implementation, we'll create a mask and use watershed-like
    # approach to split at defect points
    sub_contours = []

    # Sort defect indices for splitting
    defect_indices = sorted([d.farthest_index for d in significant_defects])

    # Split contour at defect points
    prev_idx = 0
    for idx in defect_indices:
        if idx > prev_idx + 2:  # Need at least 3 points for a valid contour
            segment = contour[prev_idx:idx + 1]
            if len(segment) >= 3:
                sub_contours.append(segment)
            prev_idx = idx

    # Add remaining segment
    if prev_idx < len(contour) - 2:
        segment = np.vstack([contour[prev_idx:], contour[:defect_indices[0] + 1]])
        if len(segment) >= 3:
            sub_contours.append(segment)

    # If splitting resulted in no valid contours, return original
    if not sub_contours:
        return [contour]

    return sub_contours


def classify_region_by_convexity(
    analysis: ConvexHullAnalysis
) -> Tuple[str, float]:
    """
    Classify a region based on its convex hull analysis.

    Returns a classification label and confidence based on the
    contour's convexity characteristics.

    Args:
        analysis: ConvexHullAnalysis result

    Returns:
        Tuple of (classification, confidence):
        - classification: One of "convex", "slightly_concave", "concave",
                         "highly_concave", "complex"
        - confidence: Confidence in the classification (0.0 to 1.0)
    """
    if analysis.is_convex:
        return "convex", 0.95

    solidity = analysis.solidity
    defect_count = analysis.defect_count
    concavity = analysis.concavity_score

    if solidity >= 0.95:
        return "convex", 0.9  # Nearly convex

    if solidity >= 0.85 and defect_count <= 1:
        return "slightly_concave", 0.85

    if solidity >= 0.7 and defect_count <= 3:
        return "concave", 0.8

    if solidity >= 0.5:
        return "highly_concave", 0.75

    return "complex", 0.7


def analyze_contours_batch_convexity(
    contours: List[np.ndarray],
    min_defect_depth: float = 5.0
) -> List[Tuple[np.ndarray, ConvexHullAnalysis, str]]:
    """
    Analyze convexity for a batch of contours.

    Args:
        contours: List of contours to analyze
        min_defect_depth: Minimum depth for defect detection

    Returns:
        List of tuples (contour, analysis, classification)
    """
    results = []

    for contour in contours:
        analysis = analyze_convex_hull(contour, min_defect_depth)
        classification, _ = classify_region_by_convexity(analysis)
        results.append((contour, analysis, classification))

    return results


# =============================================================================
# Watershed Algorithm for Separating Merged Regions
# =============================================================================


def generate_watershed_markers_distance(
    binary_mask: np.ndarray,
    distance_threshold: float = 0.5,
    min_marker_area: int = 50
) -> Tuple[np.ndarray, int]:
    """
    Generate watershed markers using distance transform.

    The distance transform calculates the distance from each foreground
    pixel to the nearest background pixel. Local maxima in the distance
    transform correspond to region centers, which become markers.

    Args:
        binary_mask: Binary mask of the region(s) to segment (255 = foreground)
        distance_threshold: Threshold for distance transform (0.0-1.0, fraction of max)
        min_marker_area: Minimum area for a marker to be considered valid

    Returns:
        Tuple of (markers, num_markers):
        - markers: Labeled marker image (0 = background, 1+ = marker labels)
        - num_markers: Number of distinct markers found
    """
    if binary_mask is None or binary_mask.size == 0:
        return np.zeros((1, 1), dtype=np.int32), 0

    # Ensure binary format
    if binary_mask.max() > 1:
        binary = (binary_mask > 127).astype(np.uint8) * 255
    else:
        binary = binary_mask.astype(np.uint8) * 255

    # Apply morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Compute distance transform
    dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)

    # Normalize distance transform
    if dist_transform.max() > 0:
        dist_normalized = dist_transform / dist_transform.max()
    else:
        return np.zeros_like(binary_mask, dtype=np.int32), 0

    # Threshold to find sure foreground (markers)
    sure_fg = (dist_normalized > distance_threshold).astype(np.uint8) * 255

    # Find connected components in sure foreground
    num_labels, markers = cv2.connectedComponents(sure_fg)

    # Filter out small markers
    if min_marker_area > 0:
        for label in range(1, num_labels):
            if np.sum(markers == label) < min_marker_area:
                markers[markers == label] = 0

        # Relabel to ensure consecutive labels
        unique_labels = np.unique(markers)
        unique_labels = unique_labels[unique_labels > 0]
        new_markers = np.zeros_like(markers)
        for new_label, old_label in enumerate(unique_labels, start=1):
            new_markers[markers == old_label] = new_label
        markers = new_markers
        num_labels = len(unique_labels) + 1

    # Add 1 to all labels (background becomes 1, markers become 2+)
    markers = markers + 1

    # Mark unknown region as 0 (for watershed)
    # Unknown = areas in original mask but not in sure foreground
    sure_bg = cv2.dilate(opened, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    markers[unknown == 255] = 0

    return markers.astype(np.int32), num_labels - 1


def generate_watershed_markers_peaks(
    binary_mask: np.ndarray,
    min_distance: int = 20,
    min_marker_area: int = 50
) -> Tuple[np.ndarray, int]:
    """
    Generate watershed markers using local maxima (peaks) in distance transform.

    This method finds peaks in the distance transform using morphological
    operations, which can handle regions with irregular shapes better
    than simple thresholding.

    Args:
        binary_mask: Binary mask of the region(s) to segment
        min_distance: Minimum distance between peaks (controls region separation)
        min_marker_area: Minimum area for a marker

    Returns:
        Tuple of (markers, num_markers)
    """
    if binary_mask is None or binary_mask.size == 0:
        return np.zeros((1, 1), dtype=np.int32), 0

    # Ensure binary format
    if binary_mask.max() > 1:
        binary = (binary_mask > 127).astype(np.uint8) * 255
    else:
        binary = binary_mask.astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Distance transform
    dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)

    # Find local maxima using dilated comparison
    # A pixel is a local max if it equals the max in its neighborhood
    kernel_size = max(3, min_distance)
    if kernel_size % 2 == 0:
        kernel_size += 1

    dilated = cv2.dilate(dist, np.ones((kernel_size, kernel_size)))
    local_max = (dist == dilated) & (dist > 0)

    # Convert to markers
    local_max_uint8 = local_max.astype(np.uint8) * 255

    # Find connected components
    num_labels, markers = cv2.connectedComponents(local_max_uint8)

    # Filter small markers
    if min_marker_area > 0:
        for label in range(1, num_labels):
            if np.sum(markers == label) < min_marker_area:
                markers[markers == label] = 0

        # Relabel
        unique_labels = np.unique(markers)
        unique_labels = unique_labels[unique_labels > 0]
        new_markers = np.zeros_like(markers)
        for new_label, old_label in enumerate(unique_labels, start=1):
            new_markers[markers == old_label] = new_label
        markers = new_markers
        num_labels = len(unique_labels) + 1

    # Adjust for watershed (background = 1, unknown = 0)
    markers = markers + 1
    sure_bg = cv2.dilate(cleaned, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, cleaned)
    markers[unknown == 255] = 0

    return markers.astype(np.int32), num_labels - 1


def apply_watershed_segmentation(
    image: np.ndarray,
    markers: np.ndarray
) -> np.ndarray:
    """
    Apply watershed algorithm to segment regions.

    The watershed algorithm treats the image as a topographic surface
    and floods from the marker positions. Boundaries are formed where
    different floods meet.

    Args:
        image: Input image (must be 3-channel BGR for cv2.watershed)
        markers: Marker image from generate_watershed_markers_*

    Returns:
        Labels image where each pixel is labeled with its region number.
        Boundary pixels are marked as -1.
    """
    if image is None or markers is None:
        return np.zeros((1, 1), dtype=np.int32)

    # Ensure image is 3-channel BGR
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        image_bgr = image.copy()

    # Ensure markers is int32
    markers_int32 = markers.astype(np.int32)

    # Apply watershed
    cv2.watershed(image_bgr, markers_int32)

    return markers_int32


def extract_segments_from_watershed(
    labels: np.ndarray,
    original_mask: np.ndarray,
    min_area: float = 100.0
) -> List[WatershedSegment]:
    """
    Extract individual segment information from watershed labels.

    Args:
        labels: Label image from watershed (1 = background, -1 = boundary, 2+ = segments)
        original_mask: Original binary mask for reference
        min_area: Minimum segment area to include

    Returns:
        List of WatershedSegment objects
    """
    segments = []

    # Get unique labels (exclude background=1 and boundary=-1)
    unique_labels = np.unique(labels)
    region_labels = [l for l in unique_labels if l > 1]

    for label in region_labels:
        # Create mask for this segment
        segment_mask = (labels == label).astype(np.uint8) * 255

        # Calculate area
        area = np.sum(segment_mask > 0)

        if area < min_area:
            continue

        # Find contour for this segment
        contours, _ = cv2.findContours(
            segment_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        # Use largest contour
        contour = max(contours, key=cv2.contourArea)

        # Calculate centroid
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w / 2, y + h / 2

        # Bounding rectangle
        bbox = cv2.boundingRect(contour)

        segment = WatershedSegment(
            label=int(label),
            contour=contour,
            area=float(area),
            centroid=(cx, cy),
            bounding_rect=bbox,
            mask=segment_mask
        )
        segments.append(segment)

    return segments


def separate_merged_regions(
    image: np.ndarray,
    binary_mask: Optional[np.ndarray] = None,
    contour: Optional[np.ndarray] = None,
    method: str = "distance",
    distance_threshold: float = 0.5,
    min_distance: int = 20,
    min_segment_area: float = 100.0,
    min_marker_area: int = 50,
    debug_dir: Optional[str] = None
) -> WatershedResult:
    """
    Separate merged/touching regions using the watershed algorithm.

    This is the main entry point for watershed-based region separation.
    It can work with either a binary mask or a contour to define the
    region to segment.

    Use cases:
    - Separating touching puzzle regions that appear as one contour
    - Splitting regions with overlapping colors
    - Handling merged cells in grid puzzles

    Args:
        image: Input image (BGR or grayscale)
        binary_mask: Binary mask of region(s) to segment. If None, generated from contour.
        contour: Contour to segment. Used if binary_mask is None.
        method: Marker generation method:
                - "distance": Distance transform with threshold (default)
                - "peaks": Local maxima in distance transform
        distance_threshold: Threshold for distance method (0.0-1.0)
        min_distance: Minimum distance between peaks for peaks method
        min_segment_area: Minimum area for resulting segments
        min_marker_area: Minimum area for markers
        debug_dir: Optional directory for debug images

    Returns:
        WatershedResult containing separated segments and metadata

    Example:
        >>> import cv2
        >>> import numpy as np
        >>> # Create image with two touching circles
        >>> img = np.zeros((200, 200, 3), dtype=np.uint8)
        >>> cv2.circle(img, (70, 100), 50, (255, 255, 255), -1)
        >>> cv2.circle(img, (130, 100), 50, (255, 255, 255), -1)
        >>> gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        >>> mask = (gray > 0).astype(np.uint8) * 255
        >>> result = separate_merged_regions(img, binary_mask=mask)
        >>> print(f"Found {result.num_regions} regions")
    """
    if image is None:
        return WatershedResult(confidence=0.0)

    # Ensure we have a 3-channel image for watershed
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image.copy()

    h, w = image_bgr.shape[:2]

    # Generate binary mask if not provided
    if binary_mask is None:
        if contour is not None:
            # Draw contour to create mask
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(binary_mask, [contour], -1, 255, -1)
        else:
            # Convert image to binary
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            _, binary_mask = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

    # Generate markers based on method
    if method == "peaks":
        markers, num_markers = generate_watershed_markers_peaks(
            binary_mask,
            min_distance=min_distance,
            min_marker_area=min_marker_area
        )
    else:  # distance (default)
        markers, num_markers = generate_watershed_markers_distance(
            binary_mask,
            distance_threshold=distance_threshold,
            min_marker_area=min_marker_area
        )

    # If only one or no markers found, return original as single region
    if num_markers <= 1:
        # Find contour from mask
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            moments = cv2.moments(largest)
            if moments["m00"] != 0:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
            else:
                x, y, bw, bh = cv2.boundingRect(largest)
                cx, cy = x + bw / 2, y + bh / 2

            segment = WatershedSegment(
                label=1,
                contour=largest,
                area=area,
                centroid=(cx, cy),
                bounding_rect=cv2.boundingRect(largest),
                mask=binary_mask
            )

            return WatershedResult(
                segments=[segment],
                markers=markers,
                labels=None,
                num_regions=1,
                original_contour=contour,
                method=method,
                confidence=0.5  # Lower confidence since no separation occurred
            )

        return WatershedResult(
            num_regions=0,
            method=method,
            confidence=0.0
        )

    # Apply watershed
    labels = apply_watershed_segmentation(image_bgr, markers)

    # Extract segments
    segments = extract_segments_from_watershed(
        labels,
        binary_mask,
        min_area=min_segment_area
    )

    # Calculate confidence based on segmentation quality
    if len(segments) > 1:
        # Higher confidence if we found multiple well-separated regions
        total_area = sum(s.area for s in segments)
        original_area = np.sum(binary_mask > 0)
        area_coverage = total_area / max(original_area, 1)

        # Check for reasonable segment sizes
        areas = [s.area for s in segments]
        avg_area = np.mean(areas) if areas else 0
        size_variance = np.std(areas) / max(avg_area, 1) if areas else 1

        confidence = min(1.0, (
            0.4 * area_coverage +
            0.3 * min(1.0, len(segments) / 5.0) +
            0.3 * max(0.0, 1.0 - size_variance)
        ))
    else:
        confidence = 0.3

    # Save debug images
    if debug_dir:
        out_dir = Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save binary mask
        cv2.imwrite(str(out_dir / "watershed_mask.png"), binary_mask)

        # Save markers visualization
        markers_vis = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(2, markers.max() + 1):
            color = (
                int((i * 73) % 256),
                int((i * 137) % 256),
                int((i * 199) % 256)
            )
            markers_vis[markers == i] = color
        cv2.imwrite(str(out_dir / "watershed_markers.png"), markers_vis)

        # Save labels visualization
        if labels is not None:
            labels_vis = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(2, labels.max() + 1):
                color = (
                    int((i * 73) % 256),
                    int((i * 137) % 256),
                    int((i * 199) % 256)
                )
                labels_vis[labels == i] = color
            # Mark boundaries in white
            labels_vis[labels == -1] = (255, 255, 255)
            cv2.imwrite(str(out_dir / "watershed_labels.png"), labels_vis)

        # Save segments with contours
        segments_vis = image_bgr.copy()
        for seg in segments:
            color = (
                int((seg.label * 73) % 256),
                int((seg.label * 137) % 256),
                int((seg.label * 199) % 256)
            )
            cv2.drawContours(segments_vis, [seg.contour], -1, color, 2)
            cv2.circle(segments_vis, (int(seg.centroid[0]), int(seg.centroid[1])), 3, color, -1)
        cv2.imwrite(str(out_dir / "watershed_segments.png"), segments_vis)

    return WatershedResult(
        segments=segments,
        markers=markers,
        labels=labels,
        num_regions=len(segments),
        original_contour=contour,
        method=method,
        confidence=confidence
    )


def separate_merged_regions_batch(
    image: np.ndarray,
    contours: List[np.ndarray],
    method: str = "distance",
    distance_threshold: float = 0.5,
    min_distance: int = 20,
    min_segment_area: float = 100.0,
    solidity_threshold: float = 0.85,
    debug_dir: Optional[str] = None
) -> List[WatershedResult]:
    """
    Apply watershed segmentation to a batch of potentially merged contours.

    Analyzes each contour and applies watershed only to those that
    appear to be merged (based on solidity and shape analysis).

    Args:
        image: Input image
        contours: List of contours to process
        method: Marker generation method ("distance" or "peaks")
        distance_threshold: Threshold for distance method
        min_distance: Minimum distance between peaks
        min_segment_area: Minimum area for resulting segments
        solidity_threshold: Contours with solidity below this are considered merged
        debug_dir: Optional directory for debug images

    Returns:
        List of WatershedResult objects (one per input contour)
    """
    results = []

    for i, contour in enumerate(contours):
        # Analyze contour to determine if it might be merged
        analysis = analyze_convex_hull(contour)

        # Only apply watershed if contour appears to be merged
        # (low solidity or multiple significant defects)
        should_watershed = (
            analysis.solidity < solidity_threshold or
            analysis.defect_count >= 2 or
            analysis.concavity_score > 0.3
        )

        if should_watershed:
            # Create debug subdirectory for this contour
            contour_debug_dir = None
            if debug_dir:
                contour_debug_dir = str(Path(debug_dir) / f"contour_{i}")

            result = separate_merged_regions(
                image,
                contour=contour,
                method=method,
                distance_threshold=distance_threshold,
                min_distance=min_distance,
                min_segment_area=min_segment_area,
                debug_dir=contour_debug_dir
            )
        else:
            # Return contour as single segment
            area = cv2.contourArea(contour)
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
            else:
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w / 2, y + h / 2

            segment = WatershedSegment(
                label=1,
                contour=contour,
                area=area,
                centroid=(cx, cy),
                bounding_rect=cv2.boundingRect(contour)
            )

            result = WatershedResult(
                segments=[segment],
                num_regions=1,
                original_contour=contour,
                method="none",
                confidence=0.9  # High confidence since it's a simple convex shape
            )

        results.append(result)

    return results


def filter_contours_by_area(
    contours: List[np.ndarray],
    min_area: float = 100.0,
    max_area: Optional[float] = None,
    image_area: Optional[float] = None
) -> List[np.ndarray]:
    """
    Filter contours by area constraints.

    Args:
        contours: List of contours to filter
        min_area: Minimum contour area to keep
        max_area: Maximum contour area to keep (None = no limit)
        image_area: Total image area for relative filtering

    Returns:
        Filtered list of contours
    """
    filtered = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_area:
            continue

        if max_area is not None and area > max_area:
            continue

        # If image_area provided, filter extreme sizes
        if image_area is not None:
            area_ratio = area / image_area
            # Skip contours that are too small (< 0.1%) or too large (> 90%)
            if area_ratio < 0.001 or area_ratio > 0.9:
                continue

        filtered.append(cnt)

    return filtered


def filter_contours_by_aspect_ratio(
    contours: List[np.ndarray],
    min_aspect: float = 0.2,
    max_aspect: float = 5.0
) -> List[np.ndarray]:
    """
    Filter contours by bounding rectangle aspect ratio.

    Args:
        contours: List of contours to filter
        min_aspect: Minimum aspect ratio (width/height)
        max_aspect: Maximum aspect ratio

    Returns:
        Filtered list of contours
    """
    filtered = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue

        aspect = float(w) / h

        if min_aspect <= aspect <= max_aspect:
            filtered.append(cnt)

    return filtered


def approximate_contours_batch(
    contours: List[np.ndarray],
    epsilon_mode: str = "adaptive",
    epsilon_ratio: float = 0.02,
    fixed_epsilon: float = 5.0,
    min_area: float = 100.0,
    max_area: Optional[float] = None,
    min_aspect: float = 0.2,
    max_aspect: float = 5.0,
    analyze_convexity: bool = False,
    min_defect_depth: float = 5.0
) -> List[ApproximatedContour]:
    """
    Approximate multiple contours with filtering.

    Applies polygon approximation to a batch of contours and returns
    structured results with shape analysis.

    Args:
        contours: List of input contours
        epsilon_mode: How to calculate epsilon:
                      - "adaptive": Based on perimeter * epsilon_ratio
                      - "fixed": Use fixed_epsilon value
                      - "perimeter_ratio": Same as adaptive
        epsilon_ratio: Ratio of perimeter for adaptive epsilon (default 0.02)
        fixed_epsilon: Fixed epsilon value when epsilon_mode is "fixed"
        min_area: Minimum area to keep contours
        max_area: Maximum area to keep contours (None = no limit)
        min_aspect: Minimum aspect ratio filter
        max_aspect: Maximum aspect ratio filter
        analyze_convexity: Whether to perform convex hull analysis
        min_defect_depth: Minimum defect depth for convexity analysis

    Returns:
        List of ApproximatedContour objects
    """
    results = []

    # Pre-filter by area
    filtered_contours = filter_contours_by_area(contours, min_area, max_area)

    # Filter by aspect ratio
    filtered_contours = filter_contours_by_aspect_ratio(
        filtered_contours, min_aspect, max_aspect
    )

    for cnt in filtered_contours:
        # Determine epsilon
        if epsilon_mode == "fixed":
            epsilon = fixed_epsilon
        else:  # adaptive or perimeter_ratio
            perimeter = cv2.arcLength(cnt, True)
            epsilon = perimeter * epsilon_ratio

        # Approximate contour
        approx, eps_used = approximate_contour(cnt, epsilon=epsilon)

        # Calculate properties
        area = cv2.contourArea(approx)
        perimeter = cv2.arcLength(approx, True)
        bbox = cv2.boundingRect(approx)
        centroid = calculate_contour_centroid(approx)

        # Detect shape type
        shape_type, shape_confidence = detect_shape_type(cnt, approx)

        # Perform convex hull analysis if requested
        hull_analysis = None
        convexity_class = "unknown"
        if analyze_convexity:
            hull_analysis = analyze_convex_hull(cnt, min_defect_depth)
            convexity_class, _ = classify_region_by_convexity(hull_analysis)

        # Create result object
        result = ApproximatedContour(
            original_contour=cnt,
            approximated_contour=approx,
            epsilon=eps_used,
            original_vertex_count=len(cnt),
            approximated_vertex_count=len(approx),
            area=area,
            perimeter=perimeter,
            bounding_rect=bbox,
            centroid=centroid,
            is_closed=True,
            shape_type=shape_type,
            confidence=shape_confidence,
            hull_analysis=hull_analysis,
            convexity_class=convexity_class
        )

        results.append(result)

    return results


def extract_contours_with_approximation(
    image: np.ndarray,
    epsilon_ratio: float = 0.02,
    epsilon_mode: str = "adaptive",
    fixed_epsilon: float = 5.0,
    min_area: float = 100.0,
    max_area: Optional[float] = None,
    min_aspect: float = 0.2,
    max_aspect: float = 5.0,
    use_canny: bool = True,
    canny_low: int = 50,
    canny_high: int = 150,
    use_threshold: bool = True,
    threshold_mode: str = "adaptive",
    analyze_convexity: bool = False,
    min_defect_depth: float = 5.0,
    debug_dir: Optional[str] = None
) -> ContourExtractionResult:
    """
    Extract and approximate contours from an image.

    This is the main entry point for contour extraction with polygon
    approximation. It handles:
    1. Preprocessing (grayscale conversion, edge detection)
    2. Contour detection with cv2.findContours
    3. Polygon approximation with cv2.approxPolyDP
    4. Filtering and shape analysis
    5. Optional convex hull analysis for concave region detection

    Args:
        image: Input image (BGR or grayscale)
        epsilon_ratio: Ratio of perimeter for adaptive epsilon
        epsilon_mode: Epsilon calculation mode ("adaptive", "fixed")
        fixed_epsilon: Fixed epsilon value when mode is "fixed"
        min_area: Minimum contour area to keep
        max_area: Maximum contour area to keep
        min_aspect: Minimum aspect ratio filter
        max_aspect: Maximum aspect ratio filter
        use_canny: Whether to use Canny edge detection
        canny_low: Canny low threshold
        canny_high: Canny high threshold
        use_threshold: Whether to apply thresholding
        threshold_mode: "adaptive" or "otsu" thresholding
        analyze_convexity: Whether to perform convex hull analysis
        min_defect_depth: Minimum defect depth for convexity analysis
        debug_dir: Optional directory for debug images

    Returns:
        ContourExtractionResult with approximated contours and metadata
    """
    if image is None:
        return ContourExtractionResult(
            contours=[],
            total_contours_found=0,
            contours_after_filtering=0,
            confidence=0.0
        )

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]
    image_area = h * w

    # Preprocessing: blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Generate binary image for contour detection
    binary = None

    if use_threshold:
        if threshold_mode == "adaptive":
            block_size = max(11, min(h, w) // 20)
            if block_size % 2 == 0:
                block_size += 1
            binary = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size, -5
            )
        else:  # otsu
            _, binary = cv2.threshold(
                blurred, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

    if use_canny:
        edges = cv2.Canny(blurred, canny_low, canny_high)
        if binary is not None:
            # Combine threshold and edges
            binary = cv2.bitwise_or(binary, edges)
        else:
            binary = edges

    if binary is None:
        # Fallback to simple thresholding
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    total_found = len(contours)

    # Apply polygon approximation with filtering
    approximated = approximate_contours_batch(
        list(contours),
        epsilon_mode=epsilon_mode,
        epsilon_ratio=epsilon_ratio,
        fixed_epsilon=fixed_epsilon,
        min_area=min_area,
        max_area=max_area if max_area else image_area * 0.9,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
        analyze_convexity=analyze_convexity,
        min_defect_depth=min_defect_depth
    )

    # Calculate overall confidence based on results
    if len(approximated) > 0:
        avg_confidence = np.mean([c.confidence for c in approximated])
        # Penalize if we filtered out many contours
        filter_ratio = len(approximated) / max(total_found, 1)
        confidence = avg_confidence * min(1.0, filter_ratio + 0.5)
    else:
        confidence = 0.0

    # Save debug images
    if debug_dir:
        out_dir = Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_dir / "cv_v2_binary.png"), binary)

        # Draw original contours
        debug_original = image.copy() if len(image.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_original, contours, -1, (0, 0, 255), 1)
        cv2.imwrite(str(out_dir / "cv_v2_contours_original.png"), debug_original)

        # Draw approximated contours
        debug_approx = image.copy() if len(image.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for ac in approximated:
            # Draw approximated contour
            cv2.drawContours(debug_approx, [ac.approximated_contour], -1, (0, 255, 0), 2)
            # Draw bounding rect
            x, y, bw, bh = ac.bounding_rect
            cv2.rectangle(debug_approx, (x, y), (x + bw, y + bh), (255, 0, 0), 1)
            # Label with shape type
            cv2.putText(
                debug_approx,
                f"{ac.shape_type} ({ac.approximated_vertex_count}v)",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1
            )
        cv2.imwrite(str(out_dir / "cv_v2_contours_approximated.png"), debug_approx)

        # Draw convex hull analysis if enabled
        if analyze_convexity:
            debug_hull = image.copy() if len(image.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for ac in approximated:
                if ac.hull_analysis is not None:
                    # Draw original contour in red
                    cv2.drawContours(debug_hull, [ac.original_contour], -1, (0, 0, 255), 1)
                    # Draw convex hull in green
                    if len(ac.hull_analysis.hull_points) > 0:
                        cv2.drawContours(debug_hull, [ac.hull_analysis.hull_points], -1, (0, 255, 0), 2)
                    # Draw defect points in blue
                    for defect in ac.hull_analysis.defects:
                        cv2.circle(debug_hull, defect.farthest_point, 4, (255, 0, 0), -1)
                    # Label with convexity class
                    x, y, bw, bh = ac.bounding_rect
                    cv2.putText(
                        debug_hull,
                        f"{ac.convexity_class} (s={ac.hull_analysis.solidity:.2f})",
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 0),
                        1
                    )
            cv2.imwrite(str(out_dir / "cv_v2_convex_hull.png"), debug_hull)

    return ContourExtractionResult(
        contours=approximated,
        total_contours_found=total_found,
        contours_after_filtering=len(approximated),
        method="approxPolyDP",
        epsilon_mode=epsilon_mode,
        confidence=confidence
    )


def refine_contour_approximation(
    contour: np.ndarray,
    target_vertices: int = 4,
    max_iterations: int = 20,
    tolerance: float = 0.1
) -> Tuple[np.ndarray, float]:
    """
    Iteratively refine approximation to achieve target vertex count.

    Useful when you need a specific shape (e.g., quadrilateral for puzzle cells)
    and want to find the optimal epsilon that produces that vertex count.

    Args:
        contour: Input contour
        target_vertices: Desired number of vertices
        max_iterations: Maximum refinement iterations
        tolerance: Tolerance ratio for binary search

    Returns:
        Tuple of (best_approximation, epsilon_used)
    """
    if contour is None or len(contour) < 3:
        return contour, 0.0

    perimeter = cv2.arcLength(contour, True)

    # Binary search for optimal epsilon
    eps_low = 0.001 * perimeter
    eps_high = 0.1 * perimeter
    best_approx = contour
    best_eps = eps_low
    best_diff = abs(len(contour) - target_vertices)

    for _ in range(max_iterations):
        eps_mid = (eps_low + eps_high) / 2
        approx = cv2.approxPolyDP(contour, eps_mid, True)
        vertex_count = len(approx)

        diff = abs(vertex_count - target_vertices)

        if diff < best_diff:
            best_diff = diff
            best_approx = approx
            best_eps = eps_mid

        if vertex_count == target_vertices:
            break
        elif vertex_count > target_vertices:
            # Need more simplification
            eps_low = eps_mid
        else:
            # Over-simplified
            eps_high = eps_mid

        # Check convergence
        if (eps_high - eps_low) / perimeter < tolerance:
            break

    return best_approx, best_eps


def extract_quadrilaterals(
    image: np.ndarray,
    min_area: float = 500.0,
    max_area: Optional[float] = None,
    require_convex: bool = False,
    debug_dir: Optional[str] = None
) -> List[ApproximatedContour]:
    """
    Extract quadrilateral shapes (4-vertex polygons) from an image.

    Specifically targets rectangular or near-rectangular regions like
    puzzle cells, cards, or document boundaries.

    Args:
        image: Input image
        min_area: Minimum area for detected quadrilaterals
        max_area: Maximum area (None = no limit)
        require_convex: If True, only return convex quadrilaterals
        debug_dir: Optional directory for debug output

    Returns:
        List of ApproximatedContour objects that are quadrilaterals
    """
    # Extract all contours with approximation
    result = extract_contours_with_approximation(
        image,
        epsilon_ratio=0.02,
        min_area=min_area,
        max_area=max_area,
        min_aspect=0.5,  # More strict for quads
        max_aspect=2.0,
        debug_dir=debug_dir
    )

    quadrilaterals = []

    for approx in result.contours:
        # Check if it's a quadrilateral
        if approx.approximated_vertex_count == 4:
            # Optionally check convexity
            if require_convex:
                if not cv2.isContourConvex(approx.approximated_contour):
                    continue

            quadrilaterals.append(approx)
        else:
            # Try to refine to 4 vertices
            refined, eps = refine_contour_approximation(
                approx.original_contour,
                target_vertices=4
            )
            if len(refined) == 4:
                # Recalculate properties
                area = cv2.contourArea(refined)
                if area >= min_area and (max_area is None or area <= max_area):
                    if require_convex and not cv2.isContourConvex(refined):
                        continue

                    quad = ApproximatedContour(
                        original_contour=approx.original_contour,
                        approximated_contour=refined,
                        epsilon=eps,
                        original_vertex_count=approx.original_vertex_count,
                        approximated_vertex_count=4,
                        area=area,
                        perimeter=cv2.arcLength(refined, True),
                        bounding_rect=cv2.boundingRect(refined),
                        centroid=calculate_contour_centroid(refined),
                        is_closed=True,
                        shape_type="quadrilateral",
                        confidence=0.85
                    )
                    quadrilaterals.append(quad)

    return quadrilaterals


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Extract and approximate contours from puzzle images"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--debug-dir",
        default="debug_cv_v2",
        help="Directory for debug output"
    )
    parser.add_argument(
        "--epsilon-ratio",
        type=float,
        default=0.02,
        help="Epsilon ratio for adaptive approximation"
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=100.0,
        help="Minimum contour area"
    )
    parser.add_argument(
        "--quads-only",
        action="store_true",
        help="Only extract quadrilaterals"
    )
    parser.add_argument(
        "--analyze-convexity",
        action="store_true",
        help="Perform convex hull analysis for concave region detection"
    )
    parser.add_argument(
        "--min-defect-depth",
        type=float,
        default=5.0,
        help="Minimum defect depth for convexity analysis (pixels)"
    )
    parser.add_argument(
        "--watershed",
        action="store_true",
        help="Apply watershed algorithm to separate merged regions"
    )
    parser.add_argument(
        "--watershed-method",
        choices=["distance", "peaks"],
        default="distance",
        help="Watershed marker generation method"
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.5,
        help="Distance threshold for watershed (0.0-1.0)"
    )
    parser.add_argument(
        "--min-distance",
        type=int,
        default=20,
        help="Minimum distance between peaks for watershed"
    )
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    if args.watershed:
        # Watershed mode - separate merged regions
        # First extract contours, then apply watershed to concave ones
        result = extract_contours_with_approximation(
            img,
            epsilon_ratio=args.epsilon_ratio,
            min_area=args.min_area,
            analyze_convexity=True,  # Need convexity analysis for watershed
            min_defect_depth=args.min_defect_depth,
            debug_dir=args.debug_dir
        )

        print(f"Extracted {len(result.contours)} contours")

        # Get original contours for watershed
        original_contours = [c.original_contour for c in result.contours]

        # Apply watershed to separate merged regions
        watershed_results = separate_merged_regions_batch(
            img,
            original_contours,
            method=args.watershed_method,
            distance_threshold=args.distance_threshold,
            min_distance=args.min_distance,
            min_segment_area=args.min_area,
            debug_dir=args.debug_dir
        )

        # Report results
        total_segments = sum(r.num_regions for r in watershed_results)
        print(f"Watershed segmentation: {len(original_contours)} contours -> {total_segments} segments")

        for i, ws_result in enumerate(watershed_results):
            if ws_result.num_regions > 1:
                print(f"  Contour {i+1}: Split into {ws_result.num_regions} segments "
                      f"(method={ws_result.method}, confidence={ws_result.confidence:.2f})")
                for j, seg in enumerate(ws_result.segments):
                    print(f"    Segment {j+1}: area={seg.area:.0f}, centroid={seg.centroid}")
            else:
                print(f"  Contour {i+1}: Single region (not split)")

        # Output JSON
        output_file = Path(args.debug_dir) / "watershed_results.json"
        Path(args.debug_dir).mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump([r.to_dict() for r in watershed_results], f, indent=2)
        print(f"Wrote {output_file}")

    elif args.quads_only:
        quads = extract_quadrilaterals(
            img,
            min_area=args.min_area,
            debug_dir=args.debug_dir
        )
        print(f"Found {len(quads)} quadrilaterals")
        for i, q in enumerate(quads):
            print(f"  {i+1}: {q.shape_type} area={q.area:.0f} "
                  f"bbox={q.bounding_rect}")
    else:
        result = extract_contours_with_approximation(
            img,
            epsilon_ratio=args.epsilon_ratio,
            min_area=args.min_area,
            analyze_convexity=args.analyze_convexity,
            min_defect_depth=args.min_defect_depth,
            debug_dir=args.debug_dir
        )

        print(f"Total contours found: {result.total_contours_found}")
        print(f"After filtering: {result.contours_after_filtering}")
        print(f"Confidence: {result.confidence:.3f}")

        for i, c in enumerate(result.contours):
            convexity_info = ""
            if args.analyze_convexity and c.hull_analysis is not None:
                convexity_info = f" [{c.convexity_class}, solidity={c.hull_analysis.solidity:.2f}, defects={c.hull_analysis.defect_count}]"
            print(f"  {i+1}: {c.shape_type} ({c.approximated_vertex_count} vertices) "
                  f"area={c.area:.0f} confidence={c.confidence:.2f}{convexity_info}")

        # Output JSON
        output_file = Path(args.debug_dir) / "contours.json"
        Path(args.debug_dir).mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Wrote {output_file}")
