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

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "approximated_vertex_count": self.approximated_vertex_count,
            "original_vertex_count": self.original_vertex_count,
            "epsilon": round(self.epsilon, 4),
            "area": round(self.area, 2),
            "perimeter": round(self.perimeter, 2),
            "bounding_rect": list(self.bounding_rect),
            "centroid": [round(c, 2) for c in self.centroid],
            "is_closed": self.is_closed,
            "shape_type": self.shape_type,
            "confidence": round(self.confidence, 4)
        }


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
    max_aspect: float = 5.0
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
            confidence=shape_confidence
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
        max_aspect=max_aspect
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
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    if args.quads_only:
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
            debug_dir=args.debug_dir
        )

        print(f"Total contours found: {result.total_contours_found}")
        print(f"After filtering: {result.contours_after_filtering}")
        print(f"Confidence: {result.confidence:.3f}")

        for i, c in enumerate(result.contours):
            print(f"  {i+1}: {c.shape_type} ({c.approximated_vertex_count} vertices) "
                  f"area={c.area:.0f} confidence={c.confidence:.2f}")

        # Output JSON
        output_file = Path(args.debug_dir) / "contours.json"
        Path(args.debug_dir).mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Wrote {output_file}")
