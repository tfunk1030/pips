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
- Intensity gradient analysis for edge-poor images
- Perspective correction for distorted mobile camera images
- Image quality validation (blur, noise, lighting, resolution)
- Pre-flight distortion detection for mobile camera captures
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from scipy import signal
from scipy.ndimage import uniform_filter1d
import logging

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class GridCell:
    """
    Represents a single cell in a grid.

    Attributes:
        row: Row index (0-based)
        col: Column index (0-based)
        x: X-coordinate of top-left corner
        y: Y-coordinate of top-left corner
        width: Cell width in pixels
        height: Cell height in pixels
        is_present: Whether the cell is present (for non-rectangular grids)
        confidence: Confidence that this cell exists (0.0 to 1.0)
    """
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    is_present: bool = True
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "row": self.row,
            "col": self.col,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "is_present": self.is_present,
            "confidence": round(self.confidence, 4)
        }


@dataclass
class PartialGridInfo:
    """
    Information about partial or non-rectangular grid detection.

    Attributes:
        is_rectangular: Whether the grid forms a complete rectangle
        is_partial: Whether the grid appears to be cropped/incomplete
        missing_cells: List of (row, col) tuples for cells that appear missing
        cell_presence_mask: 2D boolean array indicating cell presence
        visible_rows: Range of visible rows (start, end)
        visible_cols: Range of visible columns (start, end)
        extrapolated_dims: Estimated full grid dimensions if partial
        boundary_type: Type of grid boundary ("complete", "cropped_left", "cropped_right", etc.)
    """
    is_rectangular: bool = True
    is_partial: bool = False
    missing_cells: List[Tuple[int, int]] = field(default_factory=list)
    cell_presence_mask: Optional[np.ndarray] = None
    visible_rows: Optional[Tuple[int, int]] = None
    visible_cols: Optional[Tuple[int, int]] = None
    extrapolated_dims: Optional[Tuple[int, int]] = None
    boundary_type: str = "complete"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_rectangular": self.is_rectangular,
            "is_partial": self.is_partial,
            "missing_cells": self.missing_cells,
            "cell_presence_mask": self.cell_presence_mask.tolist() if self.cell_presence_mask is not None else None,
            "visible_rows": list(self.visible_rows) if self.visible_rows else None,
            "visible_cols": list(self.visible_cols) if self.visible_cols else None,
            "extrapolated_dims": list(self.extrapolated_dims) if self.extrapolated_dims else None,
            "boundary_type": self.boundary_type
        }


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
        partial_grid_info: Information about partial/non-rectangular grids
        cells: List of detected grid cells (for non-rectangular grids)
    """
    horizontal_lines: np.ndarray
    vertical_lines: np.ndarray
    estimated_cell_width: Optional[float] = None
    estimated_cell_height: Optional[float] = None
    grid_dims: Optional[Tuple[int, int]] = None
    confidence: float = 0.0
    method: str = "unknown"
    partial_grid_info: Optional[PartialGridInfo] = None
    cells: Optional[List[GridCell]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "horizontal_lines": self.horizontal_lines.tolist() if len(self.horizontal_lines) > 0 else [],
            "vertical_lines": self.vertical_lines.tolist() if len(self.vertical_lines) > 0 else [],
            "estimated_cell_width": self.estimated_cell_width,
            "estimated_cell_height": self.estimated_cell_height,
            "grid_dims": list(self.grid_dims) if self.grid_dims else None,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "partial_grid_info": self.partial_grid_info.to_dict() if self.partial_grid_info else None,
            "cells": [c.to_dict() for c in self.cells] if self.cells else None
        }


@dataclass
class PerspectiveCorrectionResult:
    """
    Result of perspective correction operation.

    Attributes:
        corrected_image: The perspective-corrected image
        transform_matrix: The 3x3 perspective transformation matrix
        source_points: Original corner points (top-left, top-right, bottom-right, bottom-left)
        destination_points: Target corner points after correction
        distortion_score: Measure of how distorted the original image was (0.0 = no distortion)
        was_corrected: Whether correction was actually applied
        original_size: Size of the original image (height, width)
        corrected_size: Size of the corrected image (height, width)
    """
    corrected_image: Optional[np.ndarray] = None
    transform_matrix: Optional[np.ndarray] = None
    source_points: Optional[np.ndarray] = None
    destination_points: Optional[np.ndarray] = None
    distortion_score: float = 0.0
    was_corrected: bool = False
    original_size: Optional[Tuple[int, int]] = None
    corrected_size: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_points": self.source_points.tolist() if self.source_points is not None else None,
            "destination_points": self.destination_points.tolist() if self.destination_points is not None else None,
            "distortion_score": round(self.distortion_score, 4),
            "was_corrected": self.was_corrected,
            "original_size": list(self.original_size) if self.original_size else None,
            "corrected_size": list(self.corrected_size) if self.corrected_size else None
        }


@dataclass
class ImageQualityResult:
    """
    Result of image quality validation.

    Provides comprehensive quality metrics to determine if an image
    is suitable for puzzle detection processing.

    Attributes:
        is_acceptable: Overall pass/fail for image quality
        blur_score: Measure of image sharpness (higher = sharper, 0-1 normalized)
        noise_score: Measure of image noise level (higher = less noise, 0-1)
        lighting_score: Measure of lighting quality (higher = better, 0-1)
        contrast_score: Measure of contrast quality (higher = better, 0-1)
        resolution_score: Measure of resolution adequacy (higher = better, 0-1)
        overall_score: Weighted combination of all scores (0-1)
        issues: List of detected quality issues
        recommendations: List of suggestions to improve image quality
        min_dimension: Smallest image dimension in pixels
        blur_variance: Raw Laplacian variance for blur detection
        brightness_mean: Mean brightness value (0-255)
        brightness_std: Standard deviation of brightness
    """
    is_acceptable: bool = True
    blur_score: float = 1.0
    noise_score: float = 1.0
    lighting_score: float = 1.0
    contrast_score: float = 1.0
    resolution_score: float = 1.0
    overall_score: float = 1.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    min_dimension: int = 0
    blur_variance: float = 0.0
    brightness_mean: float = 128.0
    brightness_std: float = 50.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_acceptable": self.is_acceptable,
            "blur_score": round(self.blur_score, 4),
            "noise_score": round(self.noise_score, 4),
            "lighting_score": round(self.lighting_score, 4),
            "contrast_score": round(self.contrast_score, 4),
            "resolution_score": round(self.resolution_score, 4),
            "overall_score": round(self.overall_score, 4),
            "issues": self.issues,
            "recommendations": self.recommendations,
            "min_dimension": self.min_dimension,
            "blur_variance": round(self.blur_variance, 2),
            "brightness_mean": round(self.brightness_mean, 2),
            "brightness_std": round(self.brightness_std, 2)
        }


@dataclass
class DistortionAnalysisResult:
    """
    Result of pre-flight distortion analysis.

    Analyzes an image for various types of distortion that may affect
    puzzle detection accuracy.

    Attributes:
        has_significant_distortion: Whether distortion exceeds acceptable threshold
        perspective_distortion: Measure of perspective/keystone distortion (0-1)
        barrel_distortion: Measure of lens barrel/pincushion distortion (0-1)
        rotation_angle: Detected rotation angle in degrees
        skew_angle: Detected skew/shear angle in degrees
        overall_distortion: Combined distortion metric (0-1)
        distortion_type: Primary distortion type detected
        correction_recommended: Whether correction should be applied
        detected_corners: Corner points if quadrilateral detected
        line_analysis: Results from line straightness analysis
    """
    has_significant_distortion: bool = False
    perspective_distortion: float = 0.0
    barrel_distortion: float = 0.0
    rotation_angle: float = 0.0
    skew_angle: float = 0.0
    overall_distortion: float = 0.0
    distortion_type: str = "none"
    correction_recommended: bool = False
    detected_corners: Optional[np.ndarray] = None
    line_analysis: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "has_significant_distortion": self.has_significant_distortion,
            "perspective_distortion": round(self.perspective_distortion, 4),
            "barrel_distortion": round(self.barrel_distortion, 4),
            "rotation_angle": round(self.rotation_angle, 2),
            "skew_angle": round(self.skew_angle, 2),
            "overall_distortion": round(self.overall_distortion, 4),
            "distortion_type": self.distortion_type,
            "correction_recommended": self.correction_recommended,
            "detected_corners": self.detected_corners.tolist() if self.detected_corners is not None else None,
            "line_analysis": self.line_analysis
        }


def validate_image_quality(
    image: np.ndarray,
    min_resolution: int = 200,
    blur_threshold: float = 100.0,
    min_brightness: float = 30.0,
    max_brightness: float = 225.0,
    min_contrast: float = 20.0,
    noise_threshold: float = 15.0,
    overall_threshold: float = 0.4,
    debug_dir: Optional[str] = None
) -> ImageQualityResult:
    """
    Validate image quality for puzzle detection suitability.

    Performs comprehensive quality analysis including:
    - Blur detection using Laplacian variance
    - Noise estimation using local variance analysis
    - Lighting assessment (brightness and uniformity)
    - Contrast measurement
    - Resolution adequacy check

    Args:
        image: Input BGR or grayscale image
        min_resolution: Minimum acceptable dimension in pixels
        blur_threshold: Laplacian variance threshold (lower = more blur)
        min_brightness: Minimum acceptable mean brightness (0-255)
        max_brightness: Maximum acceptable mean brightness (0-255)
        min_contrast: Minimum acceptable standard deviation of brightness
        noise_threshold: Maximum acceptable noise level
        overall_threshold: Minimum overall quality score (0-1)
        debug_dir: Optional directory to save debug images

    Returns:
        ImageQualityResult with quality metrics and recommendations
    """
    if image is None or image.size == 0:
        return ImageQualityResult(
            is_acceptable=False,
            overall_score=0.0,
            issues=["Image is empty or None"],
            recommendations=["Provide a valid image"]
        )

    result = ImageQualityResult()
    issues = []
    recommendations = []

    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]
    result.min_dimension = min(h, w)

    # ========== 1. Resolution Check ==========
    if result.min_dimension < min_resolution:
        result.resolution_score = result.min_dimension / min_resolution
        issues.append(f"Low resolution: {result.min_dimension}px (minimum: {min_resolution}px)")
        recommendations.append("Capture image at higher resolution or move closer to puzzle")
    else:
        # Score based on how much above minimum
        result.resolution_score = min(1.0, result.min_dimension / (min_resolution * 2))

    # ========== 2. Blur Detection (Laplacian Variance) ==========
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_variance = laplacian.var()
    result.blur_variance = blur_variance

    if blur_variance < blur_threshold:
        # Normalize to 0-1 scale (0 = very blurry, 1 = sharp)
        result.blur_score = min(blur_variance / blur_threshold, 1.0)
        issues.append(f"Image appears blurry (variance: {blur_variance:.1f}, threshold: {blur_threshold})")
        recommendations.append("Hold camera steady or use auto-focus before capture")
    else:
        # Score based on how much above threshold
        result.blur_score = min(1.0, blur_variance / (blur_threshold * 3))

    # ========== 3. Lighting Assessment ==========
    brightness_mean = np.mean(gray)
    brightness_std = np.std(gray)
    result.brightness_mean = float(brightness_mean)
    result.brightness_std = float(brightness_std)

    lighting_score = 1.0

    # Check for underexposure
    if brightness_mean < min_brightness:
        lighting_penalty = (min_brightness - brightness_mean) / min_brightness
        lighting_score -= lighting_penalty * 0.5
        issues.append(f"Image too dark (brightness: {brightness_mean:.1f})")
        recommendations.append("Increase lighting or adjust camera exposure")

    # Check for overexposure
    if brightness_mean > max_brightness:
        lighting_penalty = (brightness_mean - max_brightness) / (255 - max_brightness)
        lighting_score -= lighting_penalty * 0.5
        issues.append(f"Image too bright (brightness: {brightness_mean:.1f})")
        recommendations.append("Reduce lighting or adjust camera exposure")

    # Check for uneven lighting by analyzing quadrant brightness
    quadrant_means = [
        np.mean(gray[:h//2, :w//2]),      # Top-left
        np.mean(gray[:h//2, w//2:]),      # Top-right
        np.mean(gray[h//2:, :w//2]),      # Bottom-left
        np.mean(gray[h//2:, w//2:])       # Bottom-right
    ]
    quadrant_variation = max(quadrant_means) - min(quadrant_means)
    if quadrant_variation > 60:
        lighting_penalty = min((quadrant_variation - 60) / 100, 0.3)
        lighting_score -= lighting_penalty
        issues.append(f"Uneven lighting across image (variation: {quadrant_variation:.1f})")
        recommendations.append("Ensure uniform lighting across the puzzle")

    result.lighting_score = max(0.0, lighting_score)

    # ========== 4. Contrast Assessment ==========
    if brightness_std < min_contrast:
        result.contrast_score = brightness_std / min_contrast
        issues.append(f"Low contrast (std: {brightness_std:.1f})")
        recommendations.append("Ensure good lighting contrast between puzzle elements")
    else:
        result.contrast_score = min(1.0, brightness_std / (min_contrast * 3))

    # ========== 5. Noise Estimation ==========
    # Use local variance to estimate noise
    # Apply median filter and compare with original
    denoised = cv2.medianBlur(gray, 3)
    noise_estimate = np.std(gray.astype(float) - denoised.astype(float))

    if noise_estimate > noise_threshold:
        result.noise_score = max(0.0, 1.0 - (noise_estimate - noise_threshold) / noise_threshold)
        issues.append(f"High noise level detected ({noise_estimate:.1f})")
        recommendations.append("Use better lighting or reduce camera ISO")
    else:
        result.noise_score = 1.0 - (noise_estimate / noise_threshold) * 0.3

    # ========== Calculate Overall Score ==========
    # Weighted combination of all scores
    weights = {
        'blur': 0.30,      # Blur is critical for edge detection
        'lighting': 0.25,  # Lighting affects all processing
        'contrast': 0.20,  # Contrast needed for segmentation
        'noise': 0.15,     # Noise affects clustering
        'resolution': 0.10 # Resolution is a baseline requirement
    }

    result.overall_score = (
        weights['blur'] * result.blur_score +
        weights['lighting'] * result.lighting_score +
        weights['contrast'] * result.contrast_score +
        weights['noise'] * result.noise_score +
        weights['resolution'] * result.resolution_score
    )

    # Determine if image is acceptable
    result.is_acceptable = result.overall_score >= overall_threshold and result.min_dimension >= min_resolution
    result.issues = issues
    result.recommendations = recommendations

    # Log quality assessment
    logger.debug(f"Image quality: overall={result.overall_score:.2f}, blur={result.blur_score:.2f}, "
                f"lighting={result.lighting_score:.2f}, contrast={result.contrast_score:.2f}")

    # Save debug output
    if debug_dir:
        out_dir = Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create visualization
        debug_img = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Add quality info text
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0) if result.is_acceptable else (0, 0, 255)

        cv2.putText(debug_img, f"Quality: {result.overall_score:.2f}", (10, y_offset),
                   font, 0.7, color, 2)
        y_offset += 25
        cv2.putText(debug_img, f"Blur: {result.blur_score:.2f}", (10, y_offset),
                   font, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(debug_img, f"Lighting: {result.lighting_score:.2f}", (10, y_offset),
                   font, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(debug_img, f"Contrast: {result.contrast_score:.2f}", (10, y_offset),
                   font, 0.5, (255, 255, 255), 1)

        cv2.imwrite(str(out_dir / "image_quality.png"), debug_img)

        # Save detailed report
        with open(str(out_dir / "image_quality_report.txt"), "w") as f:
            f.write(f"Image Quality Report\n")
            f.write(f"====================\n\n")
            f.write(f"Overall Score: {result.overall_score:.4f}\n")
            f.write(f"Acceptable: {result.is_acceptable}\n\n")
            f.write(f"Individual Scores:\n")
            f.write(f"  - Blur: {result.blur_score:.4f} (variance: {result.blur_variance:.2f})\n")
            f.write(f"  - Noise: {result.noise_score:.4f}\n")
            f.write(f"  - Lighting: {result.lighting_score:.4f}\n")
            f.write(f"  - Contrast: {result.contrast_score:.4f}\n")
            f.write(f"  - Resolution: {result.resolution_score:.4f}\n\n")
            f.write(f"Image Stats:\n")
            f.write(f"  - Dimensions: {w}x{h}\n")
            f.write(f"  - Min dimension: {result.min_dimension}px\n")
            f.write(f"  - Brightness: {result.brightness_mean:.2f} (std: {result.brightness_std:.2f})\n\n")
            if issues:
                f.write(f"Issues:\n")
                for issue in issues:
                    f.write(f"  - {issue}\n")
            if recommendations:
                f.write(f"\nRecommendations:\n")
                for rec in recommendations:
                    f.write(f"  - {rec}\n")

    return result


def analyze_line_straightness(
    edges: np.ndarray,
    min_line_length: int = 50
) -> Dict:
    """
    Analyze straightness of detected lines to assess barrel/pincushion distortion.

    Barrel and pincushion distortion cause straight lines to appear curved,
    especially near image edges. This function detects such curvature.

    Args:
        edges: Edge image (output of Canny or similar)
        min_line_length: Minimum line segment length to analyze

    Returns:
        Dictionary with line straightness metrics
    """
    h, w = edges.shape[:2]

    # Detect line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=10
    )

    if lines is None or len(lines) < 4:
        return {
            "analyzed": False,
            "line_count": 0 if lines is None else len(lines),
            "curvature_score": 0.0
        }

    # Analyze lines in different image regions
    edge_margin = min(h, w) // 5
    edge_lines = []
    center_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Classify by position
        midx = (x1 + x2) / 2
        midy = (y1 + y2) / 2

        is_edge = (midx < edge_margin or midx > w - edge_margin or
                  midy < edge_margin or midy > h - edge_margin)

        if is_edge:
            edge_lines.append((x1, y1, x2, y2, line_len))
        else:
            center_lines.append((x1, y1, x2, y2, line_len))

    # Compare line segment consistency between edge and center
    # Barrel distortion causes edge lines to be shorter than expected
    avg_edge_len = np.mean([l[4] for l in edge_lines]) if edge_lines else 0
    avg_center_len = np.mean([l[4] for l in center_lines]) if center_lines else 0

    # Curvature score based on ratio (perfect = 1.0, distorted < 1.0)
    if avg_center_len > 0 and avg_edge_len > 0:
        length_ratio = avg_edge_len / avg_center_len
        curvature_score = min(length_ratio, 1.0 / length_ratio) if length_ratio > 0 else 0
    else:
        curvature_score = 1.0  # Can't assess

    return {
        "analyzed": True,
        "line_count": len(lines),
        "edge_line_count": len(edge_lines),
        "center_line_count": len(center_lines),
        "avg_edge_length": float(avg_edge_len),
        "avg_center_length": float(avg_center_len),
        "curvature_score": float(curvature_score)
    }


def detect_rotation_angle(
    edges: np.ndarray,
    angle_threshold: float = 2.0
) -> Tuple[float, float]:
    """
    Detect image rotation angle from dominant line orientations.

    Uses Hough line detection to find the dominant angle of lines
    in the image, which indicates rotation from horizontal/vertical.

    Args:
        edges: Edge image
        angle_threshold: Maximum angle to consider as valid rotation (degrees)

    Returns:
        Tuple of (rotation_angle, confidence):
        - rotation_angle: Detected rotation in degrees
        - confidence: Confidence in the detection (0-1)
    """
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is None or len(lines) < 3:
        return 0.0, 0.0

    # Collect angles
    angles = []
    for line in lines:
        rho, theta = line[0]
        # Convert to degrees and normalize to -90 to 90
        angle_deg = np.degrees(theta)
        # Normalize: 0-180 -> closest to 0 or 90
        if angle_deg > 90:
            angle_deg -= 180
        angles.append(angle_deg)

    angles = np.array(angles)

    # Find angles near 0 (horizontal) and near 90/-90 (vertical)
    horizontal_angles = angles[np.abs(angles) < 45]
    vertical_angles = angles[np.abs(np.abs(angles) - 90) < 45]

    # Calculate rotation from these
    h_rotation = np.median(horizontal_angles) if len(horizontal_angles) > 0 else 0
    v_rotation = np.median(vertical_angles) - 90 if len(vertical_angles) > 0 else 0

    # Average rotation estimate
    if len(horizontal_angles) > 0 and len(vertical_angles) > 0:
        rotation = (h_rotation + v_rotation) / 2
        confidence = min(len(horizontal_angles), len(vertical_angles)) / len(lines)
    elif len(horizontal_angles) > 0:
        rotation = h_rotation
        confidence = len(horizontal_angles) / len(lines) * 0.7
    elif len(vertical_angles) > 0:
        rotation = v_rotation
        confidence = len(vertical_angles) / len(lines) * 0.7
    else:
        rotation = 0.0
        confidence = 0.0

    return float(rotation), float(confidence)


def detect_distortion(
    image: np.ndarray,
    perspective_threshold: float = 0.05,
    barrel_threshold: float = 0.15,
    rotation_threshold: float = 3.0,
    debug_dir: Optional[str] = None
) -> DistortionAnalysisResult:
    """
    Perform pre-flight distortion analysis on an image.

    Detects various types of image distortion that may affect puzzle
    detection accuracy:
    - Perspective distortion (keystone effect from camera angle)
    - Barrel/pincushion distortion (lens distortion)
    - Image rotation
    - Skew/shear

    Args:
        image: Input BGR or grayscale image
        perspective_threshold: Threshold for significant perspective distortion (0-1)
        barrel_threshold: Threshold for significant barrel distortion (0-1)
        rotation_threshold: Threshold for significant rotation (degrees)
        debug_dir: Optional directory to save debug images

    Returns:
        DistortionAnalysisResult with distortion metrics and recommendations
    """
    if image is None or image.size == 0:
        return DistortionAnalysisResult(
            has_significant_distortion=False,
            distortion_type="error",
            overall_distortion=0.0
        )

    result = DistortionAnalysisResult()
    h, w = image.shape[:2]

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Edge detection for analysis
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # ========== 1. Perspective Distortion ==========
    # Try to detect quadrilateral corners
    corners = detect_quadrilateral_corners(image, debug_dir=None)

    if corners is not None:
        result.detected_corners = corners
        perspective_score = compute_distortion_score(corners)
        result.perspective_distortion = perspective_score
    else:
        # Fallback: analyze parallelism of detected lines
        result.perspective_distortion = 0.0

    # ========== 2. Barrel/Pincushion Distortion ==========
    line_analysis = analyze_line_straightness(edges)
    result.line_analysis = line_analysis

    if line_analysis.get("analyzed", False):
        curvature = line_analysis.get("curvature_score", 1.0)
        # Convert curvature score to distortion measure (1.0 = no distortion)
        result.barrel_distortion = max(0.0, 1.0 - curvature)
    else:
        result.barrel_distortion = 0.0

    # ========== 3. Rotation Detection ==========
    rotation_angle, rotation_conf = detect_rotation_angle(edges)
    result.rotation_angle = rotation_angle

    # ========== 4. Skew Detection ==========
    # Estimate skew from line angle variance
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

    if lines is not None and len(lines) > 5:
        h_angles = []
        v_angles = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > abs(y2 - y1):  # More horizontal
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                h_angles.append(angle)
            else:  # More vertical
                angle = np.degrees(np.arctan2(x2 - x1, y2 - y1))
                v_angles.append(angle)

        # Skew is the difference between horizontal and vertical deviations
        h_dev = np.std(h_angles) if h_angles else 0
        v_dev = np.std(v_angles) if v_angles else 0
        result.skew_angle = abs(h_dev - v_dev)
    else:
        result.skew_angle = 0.0

    # ========== Calculate Overall Distortion ==========
    # Weighted combination
    result.overall_distortion = (
        0.50 * result.perspective_distortion +
        0.25 * result.barrel_distortion +
        0.15 * min(abs(result.rotation_angle) / 10.0, 1.0) +
        0.10 * min(result.skew_angle / 5.0, 1.0)
    )

    # ========== Determine Distortion Type ==========
    distortion_types = []

    if result.perspective_distortion >= perspective_threshold:
        distortion_types.append("perspective")

    if result.barrel_distortion >= barrel_threshold:
        distortion_types.append("barrel")

    if abs(result.rotation_angle) >= rotation_threshold:
        distortion_types.append("rotation")

    if result.skew_angle >= 3.0:
        distortion_types.append("skew")

    if len(distortion_types) == 0:
        result.distortion_type = "none"
        result.has_significant_distortion = False
        result.correction_recommended = False
    elif len(distortion_types) == 1:
        result.distortion_type = distortion_types[0]
        result.has_significant_distortion = True
        result.correction_recommended = True
    else:
        result.distortion_type = "multiple"
        result.has_significant_distortion = True
        result.correction_recommended = True

    # Log distortion analysis
    logger.debug(f"Distortion analysis: type={result.distortion_type}, overall={result.overall_distortion:.3f}, "
                f"perspective={result.perspective_distortion:.3f}, barrel={result.barrel_distortion:.3f}")

    # Save debug output
    if debug_dir:
        out_dir = Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create visualization
        debug_img = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw detected corners if available
        if result.detected_corners is not None:
            pts = result.detected_corners.astype(np.int32)
            cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)
            for i, pt in enumerate(pts):
                cv2.circle(debug_img, tuple(pt), 5, (0, 0, 255), -1)
                cv2.putText(debug_img, str(i), (pt[0] + 5, pt[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add distortion info
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 255) if result.has_significant_distortion else (0, 255, 0)

        cv2.putText(debug_img, f"Distortion: {result.overall_distortion:.3f}", (10, y_offset),
                   font, 0.7, color, 2)
        y_offset += 25
        cv2.putText(debug_img, f"Type: {result.distortion_type}", (10, y_offset),
                   font, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(debug_img, f"Perspective: {result.perspective_distortion:.3f}", (10, y_offset),
                   font, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(debug_img, f"Barrel: {result.barrel_distortion:.3f}", (10, y_offset),
                   font, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(debug_img, f"Rotation: {result.rotation_angle:.2f}°", (10, y_offset),
                   font, 0.5, (255, 255, 255), 1)

        cv2.imwrite(str(out_dir / "distortion_analysis.png"), debug_img)
        cv2.imwrite(str(out_dir / "distortion_edges.png"), edges)

        # Save detailed report
        with open(str(out_dir / "distortion_report.txt"), "w") as f:
            f.write(f"Distortion Analysis Report\n")
            f.write(f"==========================\n\n")
            f.write(f"Overall Distortion: {result.overall_distortion:.4f}\n")
            f.write(f"Significant Distortion: {result.has_significant_distortion}\n")
            f.write(f"Correction Recommended: {result.correction_recommended}\n")
            f.write(f"Primary Type: {result.distortion_type}\n\n")
            f.write(f"Individual Metrics:\n")
            f.write(f"  - Perspective: {result.perspective_distortion:.4f}\n")
            f.write(f"  - Barrel/Pincushion: {result.barrel_distortion:.4f}\n")
            f.write(f"  - Rotation: {result.rotation_angle:.2f}°\n")
            f.write(f"  - Skew: {result.skew_angle:.2f}°\n\n")
            if result.line_analysis:
                f.write(f"Line Analysis:\n")
                for key, val in result.line_analysis.items():
                    f.write(f"  - {key}: {val}\n")

    return result


def validate_image_for_processing(
    image: np.ndarray,
    check_quality: bool = True,
    check_distortion: bool = True,
    auto_correct_distortion: bool = False,
    quality_threshold: float = 0.4,
    distortion_threshold: float = 0.05,
    debug_dir: Optional[str] = None
) -> Tuple[np.ndarray, ImageQualityResult, DistortionAnalysisResult, bool]:
    """
    Comprehensive pre-flight validation of image for puzzle detection.

    Combines image quality validation and distortion detection to determine
    if an image is suitable for processing. Optionally applies automatic
    perspective correction.

    Args:
        image: Input BGR image
        check_quality: Whether to perform quality validation
        check_distortion: Whether to perform distortion analysis
        auto_correct_distortion: Whether to automatically correct perspective distortion
        quality_threshold: Minimum quality score to pass validation
        distortion_threshold: Maximum distortion level before requiring correction
        debug_dir: Optional directory for debug output

    Returns:
        Tuple of (processed_image, quality_result, distortion_result, is_suitable):
        - processed_image: Original or corrected image
        - quality_result: ImageQualityResult with quality metrics
        - distortion_result: DistortionAnalysisResult with distortion metrics
        - is_suitable: Whether image passes all validation checks
    """
    if image is None or image.size == 0:
        return (
            image,
            ImageQualityResult(is_acceptable=False, issues=["Invalid image"]),
            DistortionAnalysisResult(distortion_type="error"),
            False
        )

    processed_image = image.copy()
    is_suitable = True

    # Quality validation
    quality_result = ImageQualityResult(is_acceptable=True)
    if check_quality:
        quality_result = validate_image_quality(
            image,
            overall_threshold=quality_threshold,
            debug_dir=debug_dir
        )
        if not quality_result.is_acceptable:
            is_suitable = False
            logger.warning(f"Image quality below threshold: {quality_result.overall_score:.2f}")

    # Distortion analysis
    distortion_result = DistortionAnalysisResult()
    if check_distortion:
        distortion_result = detect_distortion(
            image,
            perspective_threshold=distortion_threshold,
            debug_dir=debug_dir
        )

        # Auto-correct if requested and needed
        if auto_correct_distortion and distortion_result.correction_recommended:
            correction = correct_perspective(
                processed_image,
                source_corners=distortion_result.detected_corners,
                distortion_threshold=distortion_threshold,
                auto_detect_corners=(distortion_result.detected_corners is None),
                debug_dir=debug_dir
            )
            if correction.was_corrected:
                processed_image = correction.corrected_image
                logger.info(f"Applied perspective correction (distortion: {distortion_result.perspective_distortion:.3f})")
        elif distortion_result.has_significant_distortion and not auto_correct_distortion:
            # Significant distortion but not auto-correcting
            logger.warning(f"Significant distortion detected ({distortion_result.distortion_type}) but auto-correction disabled")

    return processed_image, quality_result, distortion_result, is_suitable


def order_corner_points(pts: np.ndarray) -> np.ndarray:
    """
    Order corner points in consistent order: top-left, top-right, bottom-right, bottom-left.

    This is essential for perspective transform to work correctly, as the points
    must be in a consistent order relative to the destination rectangle.

    Args:
        pts: Array of 4 corner points with shape (4, 2)

    Returns:
        Array of 4 points ordered as [top-left, top-right, bottom-right, bottom-left]
    """
    pts = pts.reshape(4, 2)

    # Sort by sum (x + y): top-left has smallest sum, bottom-right has largest
    s = pts.sum(axis=1)
    # Sort by difference (y - x): top-right has smallest diff, bottom-left has largest
    diff = np.diff(pts, axis=1).ravel()

    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]      # Top-left
    rect[2] = pts[np.argmax(s)]      # Bottom-right
    rect[1] = pts[np.argmin(diff)]   # Top-right
    rect[3] = pts[np.argmax(diff)]   # Bottom-left

    return rect


def detect_quadrilateral_corners(
    image: np.ndarray,
    min_area_ratio: float = 0.1,
    max_area_ratio: float = 0.98,
    approx_epsilon: float = 0.02,
    edge_margin: int = 5,
    debug_dir: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Detect the four corner points of a quadrilateral region (puzzle boundary).

    Uses edge detection and contour analysis to find the largest quadrilateral
    contour in the image, which typically corresponds to the puzzle boundary.

    Args:
        image: Input BGR image
        min_area_ratio: Minimum contour area as fraction of image area
        max_area_ratio: Maximum contour area as fraction of image area
        approx_epsilon: Epsilon parameter for polygon approximation (fraction of perimeter)
        edge_margin: Margin from image edges to consider corners valid
        debug_dir: Optional directory to save debug images

    Returns:
        Array of 4 corner points ordered as [TL, TR, BR, BL], or None if not found
    """
    if image is None:
        return None

    h, w = image.shape[:2]
    img_area = h * w
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate edges to close gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best_quad = None
    best_area = 0

    for contour in contours[:10]:  # Check top 10 largest contours
        area = cv2.contourArea(contour)

        if area < min_area or area > max_area:
            continue

        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        epsilon = approx_epsilon * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Look for quadrilaterals
        if len(approx) == 4:
            # Check if it's convex
            if cv2.isContourConvex(approx):
                # Verify corners are not too close to image edges
                points = approx.reshape(4, 2)
                if all(edge_margin < p[0] < w - edge_margin and
                       edge_margin < p[1] < h - edge_margin for p in points):
                    if area > best_area:
                        best_quad = points
                        best_area = area

    if best_quad is None:
        # Try alternative approach: find largest contour and fit bounding box
        if len(contours) > 0:
            largest = contours[0]
            if cv2.contourArea(largest) >= min_area:
                # Get minimum area rectangle
                rect = cv2.minAreaRect(largest)
                box = cv2.boxPoints(rect)
                best_quad = np.int32(box)

    if best_quad is not None:
        # Order points consistently
        best_quad = order_corner_points(best_quad.astype(np.float32))

        # Save debug visualization
        if debug_dir:
            out_dir = Path(debug_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            debug_img = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.polylines(debug_img, [best_quad.astype(np.int32)], True, (0, 255, 0), 3)

            # Draw corner labels
            labels = ["TL", "TR", "BR", "BL"]
            for i, (pt, label) in enumerate(zip(best_quad, labels)):
                cv2.circle(debug_img, tuple(pt.astype(int)), 8, (0, 0, 255), -1)
                cv2.putText(debug_img, label, (int(pt[0]) + 10, int(pt[1]) + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imwrite(str(out_dir / "detected_corners.png"), debug_img)
            cv2.imwrite(str(out_dir / "edges_for_corners.png"), edges)

    return best_quad


def compute_distortion_score(corners: np.ndarray) -> float:
    """
    Compute a score indicating how distorted the quadrilateral is from a rectangle.

    Measures distortion based on:
    1. Deviation of angles from 90 degrees
    2. Difference in opposite side lengths
    3. Non-parallelism of opposite sides

    Args:
        corners: Array of 4 corner points ordered as [TL, TR, BR, BL]

    Returns:
        Distortion score from 0.0 (perfect rectangle) to 1.0 (severely distorted)
    """
    if corners is None or len(corners) != 4:
        return 0.0

    corners = corners.reshape(4, 2).astype(np.float64)
    tl, tr, br, bl = corners

    # Calculate side lengths
    top = np.linalg.norm(tr - tl)
    right = np.linalg.norm(br - tr)
    bottom = np.linalg.norm(bl - br)
    left = np.linalg.norm(tl - bl)

    if min(top, right, bottom, left) < 1:
        return 0.0

    # 1. Ratio difference for opposite sides (should be 1.0 for rectangle)
    horizontal_ratio = min(top, bottom) / max(top, bottom) if max(top, bottom) > 0 else 1
    vertical_ratio = min(left, right) / max(left, right) if max(left, right) > 0 else 1
    ratio_score = 1.0 - (horizontal_ratio * vertical_ratio)

    # 2. Angle deviation from 90 degrees
    def angle_at_corner(p1, vertex, p2):
        """Calculate angle at vertex in degrees."""
        v1 = p1 - vertex
        v2 = p2 - vertex
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.degrees(np.arccos(cos_angle))

    angles = [
        angle_at_corner(bl, tl, tr),  # Top-left
        angle_at_corner(tl, tr, br),  # Top-right
        angle_at_corner(tr, br, bl),  # Bottom-right
        angle_at_corner(br, bl, tl),  # Bottom-left
    ]

    # Average deviation from 90 degrees
    angle_deviations = [abs(a - 90) / 90 for a in angles]
    angle_score = np.mean(angle_deviations)

    # 3. Parallelism check using cross product
    top_vec = tr - tl
    bottom_vec = br - bl
    left_vec = tl - bl
    right_vec = tr - br

    def parallelism(v1, v2):
        """Return 0 if parallel, higher if not."""
        n1 = v1 / (np.linalg.norm(v1) + 1e-8)
        n2 = v2 / (np.linalg.norm(v2) + 1e-8)
        cross = abs(np.cross(n1, n2))
        return cross

    h_parallel = parallelism(top_vec, bottom_vec)
    v_parallel = parallelism(left_vec, right_vec)
    parallel_score = (h_parallel + v_parallel) / 2

    # Combined distortion score (weighted average)
    distortion = 0.3 * ratio_score + 0.4 * angle_score + 0.3 * parallel_score

    return min(distortion, 1.0)


def correct_perspective(
    image: np.ndarray,
    source_corners: Optional[np.ndarray] = None,
    target_size: Optional[Tuple[int, int]] = None,
    distortion_threshold: float = 0.05,
    auto_detect_corners: bool = True,
    debug_dir: Optional[str] = None
) -> PerspectiveCorrectionResult:
    """
    Correct perspective distortion in an image using getPerspectiveTransform.

    This function handles images taken at an angle (e.g., mobile camera photos)
    by detecting the puzzle quadrilateral and warping it to a rectangular view.

    The correction process:
    1. Detect or use provided corner points of the distorted quadrilateral
    2. Calculate the distortion score to determine if correction is needed
    3. If distortion exceeds threshold, compute perspective transform matrix
    4. Apply warpPerspective to create corrected image

    Args:
        image: Input BGR or grayscale image
        source_corners: Optional pre-detected corner points (4x2 array).
                       If None and auto_detect_corners is True, corners are detected.
        target_size: Optional (width, height) for output. If None, computed from corners.
        distortion_threshold: Minimum distortion score to trigger correction (0.0 to 1.0).
                             Lower values mean more aggressive correction.
        auto_detect_corners: Whether to automatically detect corners if not provided
        debug_dir: Optional directory to save debug images

    Returns:
        PerspectiveCorrectionResult containing corrected image and metadata
    """
    if image is None:
        return PerspectiveCorrectionResult(was_corrected=False)

    h, w = image.shape[:2]
    result = PerspectiveCorrectionResult(
        original_size=(h, w),
        was_corrected=False
    )

    # Get source corners
    if source_corners is None:
        if auto_detect_corners:
            source_corners = detect_quadrilateral_corners(image, debug_dir=debug_dir)
        if source_corners is None:
            # No corners detected, return original
            result.corrected_image = image.copy()
            result.corrected_size = (h, w)
            return result

    # Ensure corners are properly ordered and float32
    source_corners = order_corner_points(source_corners.astype(np.float32))
    result.source_points = source_corners

    # Compute distortion score
    distortion = compute_distortion_score(source_corners)
    result.distortion_score = distortion

    # Check if correction is needed
    if distortion < distortion_threshold:
        result.corrected_image = image.copy()
        result.corrected_size = (h, w)
        return result

    # Calculate target rectangle dimensions from source corners
    tl, tr, br, bl = source_corners

    # Width: max of top and bottom edge lengths
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    target_width = int(max(width_top, width_bottom))

    # Height: max of left and right edge lengths
    height_left = np.linalg.norm(tl - bl)
    height_right = np.linalg.norm(tr - br)
    target_height = int(max(height_left, height_right))

    # Override with provided target size if given
    if target_size is not None:
        target_width, target_height = target_size

    # Ensure minimum dimensions
    target_width = max(target_width, 100)
    target_height = max(target_height, 100)

    # Define destination points (rectangular)
    destination_corners = np.float32([
        [0, 0],                          # Top-left
        [target_width - 1, 0],           # Top-right
        [target_width - 1, target_height - 1],  # Bottom-right
        [0, target_height - 1]           # Bottom-left
    ])
    result.destination_points = destination_corners

    # Compute perspective transformation matrix
    transform_matrix = cv2.getPerspectiveTransform(source_corners, destination_corners)
    result.transform_matrix = transform_matrix

    # Apply perspective correction
    corrected = cv2.warpPerspective(
        image,
        transform_matrix,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    result.corrected_image = corrected
    result.corrected_size = (target_height, target_width)
    result.was_corrected = True

    # Save debug visualization
    if debug_dir:
        out_dir = Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Draw source corners on original
        debug_original = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.polylines(debug_original, [source_corners.astype(np.int32)], True, (0, 255, 0), 3)

        # Add distortion score text
        cv2.putText(debug_original, f"Distortion: {distortion:.3f}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imwrite(str(out_dir / "perspective_original.png"), debug_original)
        cv2.imwrite(str(out_dir / "perspective_corrected.png"), corrected)

        # Save transformation info
        with open(str(out_dir / "perspective_info.txt"), "w") as f:
            f.write(f"Distortion score: {distortion:.4f}\n")
            f.write(f"Threshold: {distortion_threshold}\n")
            f.write(f"Was corrected: {result.was_corrected}\n")
            f.write(f"Original size: {w}x{h}\n")
            f.write(f"Corrected size: {target_width}x{target_height}\n")
            f.write(f"Source corners:\n{source_corners}\n")
            f.write(f"Destination corners:\n{destination_corners}\n")

    return result


def correct_perspective_simple(
    image: np.ndarray,
    corners: np.ndarray
) -> np.ndarray:
    """
    Simple perspective correction with minimal overhead.

    Convenience function for quick perspective correction when you already
    have the corner points and don't need detailed results.

    Args:
        image: Input image
        corners: 4 corner points in any order (will be auto-ordered)

    Returns:
        Perspective-corrected image
    """
    if image is None or corners is None or len(corners) != 4:
        return image

    result = correct_perspective(
        image,
        source_corners=corners,
        auto_detect_corners=False
    )

    return result.corrected_image if result.corrected_image is not None else image


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


def detect_grid_lines_histogram(
    gray: np.ndarray,
    min_spacing: int = 30,
    max_spacing: int = 200,
    smoothing_window: int = 5,
    gradient_threshold: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Detect grid lines using histogram/intensity gradient analysis.

    This is a fallback method when Hough and projection-based detection fail.
    It analyzes the intensity gradient along rows and columns to find grid lines
    where pixel intensity changes significantly (cell boundaries).

    The method works by:
    1. Computing row-wise and column-wise mean intensity profiles
    2. Calculating gradients (first derivative) of these profiles
    3. Finding local maxima in gradient magnitude (intensity transitions)
    4. Filtering peaks by spacing constraints

    Args:
        gray: Grayscale image
        min_spacing: Minimum distance between detected lines
        max_spacing: Maximum distance between detected lines
        smoothing_window: Window size for smoothing intensity profiles
        gradient_threshold: Minimum relative gradient magnitude (0.0 to 1.0)

    Returns:
        Tuple of (horizontal_positions, vertical_positions, confidence):
        - horizontal_positions: Y-coordinates of detected horizontal lines
        - vertical_positions: X-coordinates of detected vertical lines
        - confidence: Confidence score for the detection (0.0 to 1.0)
    """
    if gray is None or gray.size == 0:
        return np.array([]), np.array([]), 0.0

    h, w = gray.shape[:2]

    # Compute mean intensity profiles along rows and columns
    row_profile = gray.mean(axis=1).astype(np.float32)  # Horizontal lines (y positions)
    col_profile = gray.mean(axis=0).astype(np.float32)  # Vertical lines (x positions)

    def find_gradient_peaks(profile: np.ndarray, min_dist: int, max_dist: int,
                           smooth_win: int, grad_thresh: float) -> Tuple[np.ndarray, float]:
        """Find peaks in gradient magnitude of intensity profile."""
        if len(profile) < 10:
            return np.array([]), 0.0

        # Smooth the profile to reduce noise
        if smooth_win > 1:
            smoothed = uniform_filter1d(profile, size=smooth_win, mode='reflect')
        else:
            smoothed = profile

        # Compute gradient (first derivative)
        gradient = np.gradient(smoothed)

        # Use absolute gradient to detect both rising and falling edges
        abs_gradient = np.abs(gradient)

        # Normalize gradient
        max_grad = abs_gradient.max()
        if max_grad <= 0:
            return np.array([]), 0.0
        norm_gradient = abs_gradient / max_grad

        # Find local maxima above threshold
        # Use scipy's find_peaks if available for better peak detection
        try:
            peaks, properties = signal.find_peaks(
                norm_gradient,
                height=grad_thresh,
                distance=min_dist,
                prominence=grad_thresh * 0.5
            )
        except Exception:
            # Fallback to simple local maxima detection
            peaks = []
            for i in range(1, len(norm_gradient) - 1):
                if norm_gradient[i] > grad_thresh:
                    if norm_gradient[i] >= norm_gradient[i-1] and norm_gradient[i] >= norm_gradient[i+1]:
                        # Enforce minimum distance
                        if not peaks or (i - peaks[-1]) >= min_dist:
                            peaks.append(i)
            peaks = np.array(peaks)

        if len(peaks) == 0:
            return np.array([]), 0.0

        # Filter by maximum spacing (remove spurious peaks that create too-large cells)
        filtered_peaks = [peaks[0]]
        for p in peaks[1:]:
            gap = p - filtered_peaks[-1]
            if gap <= max_dist:
                filtered_peaks.append(p)
            elif gap <= max_dist * 2:
                # Gap is double the max - might be missing a line in between
                # Still accept this peak
                filtered_peaks.append(p)
        peaks = np.array(filtered_peaks)

        # Calculate confidence based on consistency of spacing
        if len(peaks) >= 2:
            gaps = np.diff(peaks)
            valid_gaps = gaps[(gaps >= min_dist) & (gaps <= max_dist)]
            if len(valid_gaps) > 0:
                spacing_std = np.std(valid_gaps) if len(valid_gaps) > 1 else 0
                mean_spacing = np.mean(valid_gaps)
                # Low std relative to mean = consistent spacing = higher confidence
                consistency = 1.0 - min(spacing_std / mean_spacing, 1.0) if mean_spacing > 0 else 0.0
                coverage = len(valid_gaps) / len(gaps) if len(gaps) > 0 else 0.0
                confidence = (consistency * 0.6 + coverage * 0.4)
            else:
                confidence = 0.2  # Some peaks found but spacing not in range
        else:
            confidence = 0.1 if len(peaks) > 0 else 0.0

        return peaks, confidence

    # Find peaks in both directions
    h_lines, h_conf = find_gradient_peaks(row_profile, min_spacing, max_spacing,
                                          smoothing_window, gradient_threshold)
    v_lines, v_conf = find_gradient_peaks(col_profile, min_spacing, max_spacing,
                                          smoothing_window, gradient_threshold)

    # Overall confidence is average of both directions
    overall_confidence = (h_conf + v_conf) / 2 if (h_conf > 0 or v_conf > 0) else 0.0

    return h_lines.astype(float), v_lines.astype(float), overall_confidence


def detect_grid_lines_autocorrelation(
    gray: np.ndarray,
    min_spacing: int = 30,
    max_spacing: int = 200
) -> Tuple[Optional[float], Optional[float], float]:
    """
    Estimate grid cell size using autocorrelation analysis.

    Autocorrelation can detect periodic patterns in the image,
    which is useful for grids with regular cell spacing. This method
    doesn't find individual lines but estimates the cell dimensions.

    Args:
        gray: Grayscale image
        min_spacing: Minimum cell spacing to detect
        max_spacing: Maximum cell spacing to detect

    Returns:
        Tuple of (cell_height, cell_width, confidence):
        - cell_height: Estimated cell height (vertical spacing)
        - cell_width: Estimated cell width (horizontal spacing)
        - confidence: Confidence in the estimation (0.0 to 1.0)
    """
    if gray is None or gray.size == 0:
        return None, None, 0.0

    h, w = gray.shape[:2]

    def find_autocorr_period(profile: np.ndarray, min_p: int, max_p: int) -> Tuple[Optional[float], float]:
        """Find dominant period in 1D profile using autocorrelation."""
        if len(profile) < max_p * 2:
            return None, 0.0

        # Normalize profile
        profile = profile.astype(np.float64)
        profile = profile - profile.mean()
        if profile.std() == 0:
            return None, 0.0
        profile = profile / profile.std()

        # Compute autocorrelation
        autocorr = np.correlate(profile, profile, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only

        # Normalize autocorrelation
        autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr

        # Find peaks in autocorrelation (periodic signals have peaks at lag = period)
        # Look for first significant peak after lag 0 within our spacing range
        search_start = min_p
        search_end = min(max_p, len(autocorr) - 1)

        if search_end <= search_start:
            return None, 0.0

        autocorr_segment = autocorr[search_start:search_end]

        try:
            peaks, properties = signal.find_peaks(
                autocorr_segment,
                height=0.2,  # Minimum correlation of 0.2
                distance=min_p // 2
            )
        except Exception:
            # Fallback: find local maximum
            if len(autocorr_segment) < 3:
                return None, 0.0
            local_max_idx = np.argmax(autocorr_segment)
            if autocorr_segment[local_max_idx] > 0.2:
                peaks = np.array([local_max_idx])
            else:
                return None, 0.0

        if len(peaks) == 0:
            return None, 0.0

        # Select the peak with highest correlation value
        peak_values = autocorr_segment[peaks]
        best_peak_idx = peaks[np.argmax(peak_values)]
        period = search_start + best_peak_idx
        confidence = float(autocorr_segment[best_peak_idx])

        return float(period), confidence

    # Compute mean profiles
    row_profile = gray.mean(axis=1)
    col_profile = gray.mean(axis=0)

    cell_height, h_conf = find_autocorr_period(row_profile, min_spacing, max_spacing)
    cell_width, v_conf = find_autocorr_period(col_profile, min_spacing, max_spacing)

    overall_confidence = (h_conf + v_conf) / 2 if (h_conf > 0 or v_conf > 0) else 0.0

    return cell_height, cell_width, overall_confidence


def generate_grid_lines_from_spacing(
    image_size: Tuple[int, int],
    cell_height: Optional[float],
    cell_width: Optional[float],
    offset_y: float = 0.0,
    offset_x: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate evenly-spaced grid lines from estimated cell dimensions.

    When we have cell size estimates but not actual line positions,
    this function generates regular grid lines at the estimated spacing.

    Args:
        image_size: (height, width) of the image
        cell_height: Estimated cell height
        cell_width: Estimated cell width
        offset_y: Starting Y offset for horizontal lines
        offset_x: Starting X offset for vertical lines

    Returns:
        Tuple of (horizontal_positions, vertical_positions)
    """
    h, w = image_size

    h_lines = np.array([])
    v_lines = np.array([])

    if cell_height is not None and cell_height > 0:
        n_h_lines = int(h / cell_height) + 1
        h_lines = np.array([offset_y + i * cell_height for i in range(n_h_lines)])
        h_lines = h_lines[(h_lines >= 0) & (h_lines < h)]

    if cell_width is not None and cell_width > 0:
        n_v_lines = int(w / cell_width) + 1
        v_lines = np.array([offset_x + i * cell_width for i in range(n_v_lines)])
        v_lines = v_lines[(v_lines >= 0) & (v_lines < w)]

    return h_lines, v_lines


def detect_grid_lines_adaptive(
    image: np.ndarray,
    use_hough: bool = True,
    use_projection: bool = True,
    use_histogram_fallback: bool = True,
    use_autocorrelation_fallback: bool = True,
    min_spacing: float = 30.0,
    max_spacing: float = 200.0,
    ransac_iterations: int = 100,
    fallback_threshold: float = 0.3,
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
    5. If confidence is low, fall back to histogram gradient analysis
    6. If still low confidence, try autocorrelation-based estimation
    7. Estimate grid dimensions based on detected lines and spacing

    Args:
        image: Input image (BGR or grayscale)
        use_hough: Whether to use Hough line detection
        use_projection: Whether to use projection-based detection
        use_histogram_fallback: Whether to use histogram analysis as fallback
        use_autocorrelation_fallback: Whether to use autocorrelation as final fallback
        min_spacing: Minimum valid cell spacing
        max_spacing: Maximum valid cell spacing
        ransac_iterations: Number of RANSAC iterations for spacing estimation
        fallback_threshold: Confidence threshold below which to trigger fallbacks
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

    # Determine overall confidence from primary methods
    overall_confidence = (h_confidence + v_confidence) / 2 if h_confidence > 0 or v_confidence > 0 else 0.0
    method = "hough_ransac" if use_hough else "projection_ransac"

    # ========== FALLBACK STAGE 1: Histogram Gradient Analysis ==========
    # If primary methods have low confidence, try histogram-based detection
    if overall_confidence < fallback_threshold and use_histogram_fallback:
        hist_h_lines, hist_v_lines, hist_confidence = detect_grid_lines_histogram(
            gray,
            min_spacing=int(min_spacing),
            max_spacing=int(max_spacing),
            smoothing_window=7,
            gradient_threshold=0.12
        )

        # Use histogram results if they're better
        if hist_confidence > overall_confidence:
            # Merge histogram lines with existing lines
            if len(hist_h_lines) > 0:
                all_h_merged = np.concatenate([h_lines, hist_h_lines])
                h_lines = cluster_lines(all_h_merged, min_distance=min_spacing * 0.4)
            if len(hist_v_lines) > 0:
                all_v_merged = np.concatenate([v_lines, hist_v_lines])
                v_lines = cluster_lines(all_v_merged, min_distance=min_spacing * 0.4)

            # Re-run RANSAC on merged lines
            h_spacing, h_confidence, _ = ransac_fit_spacing(
                h_lines,
                min_spacing=min_spacing,
                max_spacing=max_spacing,
                n_iterations=ransac_iterations
            )
            v_spacing, v_confidence, _ = ransac_fit_spacing(
                v_lines,
                min_spacing=min_spacing,
                max_spacing=max_spacing,
                n_iterations=ransac_iterations
            )
            overall_confidence = (h_confidence + v_confidence) / 2
            method = "histogram_ransac"

            if debug_dir:
                out_dir = Path(debug_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                # Save histogram fallback debug info
                debug_hist = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
                for y in hist_h_lines:
                    cv2.line(debug_hist, (0, int(y)), (w - 1, int(y)), (255, 0, 0), 1)  # Blue for histogram lines
                for x in hist_v_lines:
                    cv2.line(debug_hist, (int(x), 0), (int(x), h - 1), (255, 0, 0), 1)
                cv2.imwrite(str(out_dir / "histogram_lines.png"), debug_hist)

    # ========== FALLBACK STAGE 2: Autocorrelation Analysis ==========
    # If still low confidence, try autocorrelation to estimate cell size
    if overall_confidence < fallback_threshold and use_autocorrelation_fallback:
        auto_cell_h, auto_cell_w, auto_confidence = detect_grid_lines_autocorrelation(
            gray,
            min_spacing=int(min_spacing),
            max_spacing=int(max_spacing)
        )

        # Use autocorrelation results if they provide better spacing estimates
        if auto_confidence > 0.3 and (h_spacing is None or v_spacing is None or auto_confidence > overall_confidence):
            # Generate regular grid lines from autocorrelation spacing
            if auto_cell_h is not None and (h_spacing is None or auto_confidence > h_confidence):
                h_spacing = auto_cell_h
                gen_h_lines, _ = generate_grid_lines_from_spacing(
                    (h, w), auto_cell_h, None
                )
                if len(gen_h_lines) > len(h_lines):
                    h_lines = gen_h_lines

            if auto_cell_w is not None and (v_spacing is None or auto_confidence > v_confidence):
                v_spacing = auto_cell_w
                _, gen_v_lines = generate_grid_lines_from_spacing(
                    (h, w), None, auto_cell_w
                )
                if len(gen_v_lines) > len(v_lines):
                    v_lines = gen_v_lines

            overall_confidence = max(overall_confidence, auto_confidence * 0.8)  # Slight penalty for autocorr
            method = "autocorrelation"

            if debug_dir:
                out_dir = Path(debug_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                # Save autocorrelation debug info
                with open(str(out_dir / "autocorr_results.txt"), "w") as f:
                    f.write(f"Cell height: {auto_cell_h}\n")
                    f.write(f"Cell width: {auto_cell_w}\n")
                    f.write(f"Confidence: {auto_confidence}\n")

    # ========== Spacing Consistency ==========
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

    # ========== Grid Dimension Estimation ==========
    grid_dims = None
    if h_spacing is not None and v_spacing is not None:
        n_rows = int(round(h / h_spacing)) if h_spacing > 0 else 0
        n_cols = int(round(w / v_spacing)) if v_spacing > 0 else 0
        if n_rows > 0 and n_cols > 0:
            grid_dims = (n_rows, n_cols)

    # Mark as fallback if confidence still low after all attempts
    if overall_confidence < fallback_threshold:
        method = "low_confidence_" + method

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


def detect_partial_grid_boundaries(
    image: np.ndarray,
    h_lines: np.ndarray,
    v_lines: np.ndarray,
    cell_width: Optional[float],
    cell_height: Optional[float],
    edge_threshold: float = 0.15
) -> Tuple[str, bool]:
    """
    Detect if grid is partial (cropped) and determine boundary type.

    Analyzes edge positions relative to image boundaries to determine
    if the grid appears to be cropped on any side.

    Args:
        image: Input image
        h_lines: Horizontal line positions
        v_lines: Vertical line positions
        cell_width: Estimated cell width
        cell_height: Estimated cell height
        edge_threshold: Relative threshold for edge detection (fraction of cell size)

    Returns:
        Tuple of (boundary_type, is_partial):
        - boundary_type: One of "complete", "cropped_left", "cropped_right",
                        "cropped_top", "cropped_bottom", "cropped_multiple"
        - is_partial: Whether the grid appears to be partial
    """
    if image is None or len(h_lines) < 2 or len(v_lines) < 2:
        return "unknown", False

    h, w = image.shape[:2]
    h_lines = np.sort(h_lines)
    v_lines = np.sort(v_lines)

    # Use cell size or default estimate
    cw = cell_width if cell_width else (v_lines[-1] - v_lines[0]) / max(len(v_lines) - 1, 1)
    ch = cell_height if cell_height else (h_lines[-1] - h_lines[0]) / max(len(h_lines) - 1, 1)

    if cw <= 0 or ch <= 0:
        return "unknown", False

    # Check each edge
    cropped_edges = []

    # Left edge: first vertical line far from left boundary
    if v_lines[0] > cw * edge_threshold:
        # Check if the spacing suggests a cut-off cell
        if v_lines[0] < cw * 0.9:  # First line is less than a full cell from edge
            cropped_edges.append("left")

    # Right edge: last vertical line far from right boundary
    if (w - v_lines[-1]) > cw * edge_threshold:
        if (w - v_lines[-1]) < cw * 0.9:
            cropped_edges.append("right")

    # Top edge: first horizontal line far from top boundary
    if h_lines[0] > ch * edge_threshold:
        if h_lines[0] < ch * 0.9:
            cropped_edges.append("top")

    # Bottom edge: last horizontal line far from bottom boundary
    if (h - h_lines[-1]) > ch * edge_threshold:
        if (h - h_lines[-1]) < ch * 0.9:
            cropped_edges.append("bottom")

    if len(cropped_edges) == 0:
        return "complete", False
    elif len(cropped_edges) == 1:
        return f"cropped_{cropped_edges[0]}", True
    else:
        return "cropped_multiple", True


def detect_missing_cells(
    image: np.ndarray,
    h_lines: np.ndarray,
    v_lines: np.ndarray,
    saturation_threshold: int = 15,
    variance_threshold: float = 100.0,
    edge_density_threshold: float = 0.02
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Detect missing or empty cells in a grid for non-rectangular detection.

    Analyzes each cell region to determine if it contains meaningful content
    or appears to be empty/missing. Uses multiple cues:
    - Color saturation (colorful puzzles have saturated cells)
    - Intensity variance (uniform areas suggest empty cells)
    - Edge density (cells with content have more edges)

    Args:
        image: Input BGR image
        h_lines: Sorted horizontal line positions
        v_lines: Sorted vertical line positions
        saturation_threshold: Minimum saturation for a "present" cell
        variance_threshold: Minimum intensity variance for a "present" cell
        edge_density_threshold: Minimum edge pixel ratio for a "present" cell

    Returns:
        Tuple of (presence_mask, missing_cells):
        - presence_mask: 2D boolean array where True = cell is present
        - missing_cells: List of (row, col) tuples for missing cells
    """
    if image is None or len(h_lines) < 2 or len(v_lines) < 2:
        return np.array([[]]), []

    h_lines = np.sort(h_lines)
    v_lines = np.sort(v_lines)

    n_rows = len(h_lines) - 1
    n_cols = len(v_lines) - 1

    if n_rows <= 0 or n_cols <= 0:
        return np.array([[]]), []

    presence_mask = np.ones((n_rows, n_cols), dtype=bool)
    missing_cells = []

    # Convert to HSV for saturation analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if len(image.shape) == 3 else None

    # Compute edges for edge density analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)

    for row in range(n_rows):
        for col in range(n_cols):
            y1 = int(h_lines[row])
            y2 = int(h_lines[row + 1])
            x1 = int(v_lines[col])
            x2 = int(v_lines[col + 1])

            # Ensure valid region
            if y2 <= y1 or x2 <= x1:
                presence_mask[row, col] = False
                missing_cells.append((row, col))
                continue

            # Extract cell region
            cell_gray = gray[y1:y2, x1:x2]
            cell_edges = edges[y1:y2, x1:x2]

            # Skip very small cells
            if cell_gray.size < 100:
                continue

            # Check 1: Intensity variance (low variance = uniform/empty)
            intensity_var = np.var(cell_gray)

            # Check 2: Saturation (low saturation = possibly background)
            mean_saturation = 0
            if hsv is not None:
                cell_hsv = hsv[y1:y2, x1:x2]
                mean_saturation = np.mean(cell_hsv[:, :, 1])

            # Check 3: Edge density (few edges = empty area)
            edge_density = np.sum(cell_edges > 0) / cell_edges.size

            # Cell is considered missing if it fails multiple checks
            low_variance = intensity_var < variance_threshold
            low_saturation = mean_saturation < saturation_threshold
            low_edges = edge_density < edge_density_threshold

            # Require at least 2 of 3 indicators to mark as missing
            missing_indicators = sum([low_variance, low_saturation, low_edges])
            if missing_indicators >= 2:
                presence_mask[row, col] = False
                missing_cells.append((row, col))

    return presence_mask, missing_cells


def analyze_grid_shape(
    presence_mask: np.ndarray
) -> Tuple[bool, str]:
    """
    Analyze the shape of a grid based on cell presence mask.

    Determines if the grid is rectangular or has an irregular shape
    (L-shaped, T-shaped, etc.).

    Args:
        presence_mask: 2D boolean array where True = cell is present

    Returns:
        Tuple of (is_rectangular, shape_description):
        - is_rectangular: True if all cells are present (rectangular grid)
        - shape_description: Description of the grid shape
    """
    if presence_mask is None or presence_mask.size == 0:
        return True, "empty"

    n_rows, n_cols = presence_mask.shape
    total_cells = n_rows * n_cols
    present_cells = np.sum(presence_mask)

    # Fully rectangular
    if present_cells == total_cells:
        return True, "rectangular"

    # All cells missing
    if present_cells == 0:
        return True, "empty"

    # Analyze shape pattern
    row_counts = np.sum(presence_mask, axis=1)  # Cells per row
    col_counts = np.sum(presence_mask, axis=0)  # Cells per column

    # Check for L-shape (rows or columns with different counts)
    unique_row_counts = len(np.unique(row_counts[row_counts > 0]))
    unique_col_counts = len(np.unique(col_counts[col_counts > 0]))

    if unique_row_counts <= 2 and unique_col_counts <= 2:
        # Could be L-shaped or similar
        if unique_row_counts == 2 or unique_col_counts == 2:
            return False, "L_shaped"

    # Check for T-shape or cross
    middle_row = n_rows // 2
    middle_col = n_cols // 2
    if (row_counts[middle_row] == n_cols and
            np.mean(row_counts[row_counts < n_cols]) < n_cols / 2):
        return False, "T_shaped"

    # Check for corner-missing pattern
    corners = [
        presence_mask[0, 0],
        presence_mask[0, -1],
        presence_mask[-1, 0],
        presence_mask[-1, -1]
    ]
    if sum(corners) < 4 and present_cells > total_cells * 0.7:
        return False, "corners_missing"

    # Generic irregular shape
    if present_cells < total_cells * 0.95:
        return False, "irregular"

    return True, "nearly_rectangular"


def extrapolate_partial_grid(
    image_size: Tuple[int, int],
    h_lines: np.ndarray,
    v_lines: np.ndarray,
    cell_width: Optional[float],
    cell_height: Optional[float],
    boundary_type: str
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
    """
    Extrapolate grid lines for partial/cropped grids.

    Extends detected grid lines to estimate the full grid dimensions
    based on consistent cell spacing.

    Args:
        image_size: (height, width) of the image
        h_lines: Detected horizontal line positions
        v_lines: Detected vertical line positions
        cell_width: Estimated cell width
        cell_height: Estimated cell height
        boundary_type: Type of cropping ("cropped_left", etc.)

    Returns:
        Tuple of (extended_h_lines, extended_v_lines, extrapolated_dims):
        - extended_h_lines: Horizontal lines extended to image boundaries
        - extended_v_lines: Vertical lines extended to image boundaries
        - extrapolated_dims: Estimated full (rows, cols) including cropped regions
    """
    img_h, img_w = image_size
    h_lines = np.sort(h_lines.copy())
    v_lines = np.sort(v_lines.copy())

    if cell_width is None or cell_height is None:
        return h_lines, v_lines, None

    if cell_width <= 0 or cell_height <= 0:
        return h_lines, v_lines, None

    extended_h = list(h_lines)
    extended_v = list(v_lines)

    extra_rows = 0
    extra_cols = 0

    # Extend based on boundary type
    if "left" in boundary_type or boundary_type == "cropped_multiple":
        # Add lines to the left
        if len(v_lines) > 0:
            first_line = v_lines[0]
            while first_line > cell_width * 0.3:
                first_line -= cell_width
                if first_line >= 0:
                    extended_v.insert(0, first_line)
                    extra_cols += 1

    if "right" in boundary_type or boundary_type == "cropped_multiple":
        # Add lines to the right
        if len(v_lines) > 0:
            last_line = v_lines[-1]
            while last_line < img_w - cell_width * 0.3:
                last_line += cell_width
                if last_line <= img_w:
                    extended_v.append(last_line)
                    extra_cols += 1

    if "top" in boundary_type or boundary_type == "cropped_multiple":
        # Add lines to the top
        if len(h_lines) > 0:
            first_line = h_lines[0]
            while first_line > cell_height * 0.3:
                first_line -= cell_height
                if first_line >= 0:
                    extended_h.insert(0, first_line)
                    extra_rows += 1

    if "bottom" in boundary_type or boundary_type == "cropped_multiple":
        # Add lines to the bottom
        if len(h_lines) > 0:
            last_line = h_lines[-1]
            while last_line < img_h - cell_height * 0.3:
                last_line += cell_height
                if last_line <= img_h:
                    extended_h.append(last_line)
                    extra_rows += 1

    extended_h_arr = np.array(sorted(set(extended_h)))
    extended_v_arr = np.array(sorted(set(extended_v)))

    # Estimate full grid dimensions
    visible_rows = max(len(h_lines) - 1, 0)
    visible_cols = max(len(v_lines) - 1, 0)
    extrapolated_dims = (visible_rows + extra_rows, visible_cols + extra_cols)

    return extended_h_arr, extended_v_arr, extrapolated_dims


def extract_non_rectangular_cells(
    image: np.ndarray,
    h_lines: np.ndarray,
    v_lines: np.ndarray,
    presence_mask: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    min_cell_area: int = 400
) -> List[GridCell]:
    """
    Extract cells from a non-rectangular grid.

    Creates GridCell objects for each cell position, marking cells
    as present or missing based on the presence mask.

    Args:
        image: Input image (for dimensions)
        h_lines: Sorted horizontal line positions
        v_lines: Sorted vertical line positions
        presence_mask: 2D boolean array indicating cell presence
        roi: Optional ROI offset (x, y, w, h)
        min_cell_area: Minimum cell area to include

    Returns:
        List of GridCell objects representing the grid
    """
    cells = []

    if len(h_lines) < 2 or len(v_lines) < 2:
        return cells

    h_lines = np.sort(h_lines)
    v_lines = np.sort(v_lines)

    n_rows = len(h_lines) - 1
    n_cols = len(v_lines) - 1

    # Validate presence mask dimensions
    if presence_mask is None or presence_mask.shape != (n_rows, n_cols):
        presence_mask = np.ones((n_rows, n_cols), dtype=bool)

    roi_x, roi_y = (roi[0], roi[1]) if roi else (0, 0)

    for row in range(n_rows):
        for col in range(n_cols):
            y1 = int(h_lines[row])
            y2 = int(h_lines[row + 1])
            x1 = int(v_lines[col])
            x2 = int(v_lines[col + 1])

            cell_w = x2 - x1
            cell_h = y2 - y1
            is_present = presence_mask[row, col]

            # Calculate confidence based on cell characteristics
            confidence = 1.0 if is_present else 0.2

            if cell_w * cell_h >= min_cell_area:
                cells.append(GridCell(
                    row=row,
                    col=col,
                    x=roi_x + x1,
                    y=roi_y + y1,
                    width=cell_w,
                    height=cell_h,
                    is_present=is_present,
                    confidence=confidence
                ))

    return cells


def detect_grid_with_partial_support(
    image: np.ndarray,
    use_hough: bool = True,
    use_projection: bool = True,
    use_histogram_fallback: bool = True,
    use_autocorrelation_fallback: bool = True,
    min_spacing: float = 30.0,
    max_spacing: float = 200.0,
    ransac_iterations: int = 100,
    fallback_threshold: float = 0.3,
    detect_missing: bool = True,
    extrapolate_partial: bool = True,
    debug_dir: Optional[str] = None
) -> GridLineResult:
    """
    Detect grid lines with support for non-rectangular and partial grids.

    Enhanced version of detect_grid_lines_adaptive that also:
    - Detects missing cells in non-rectangular grids
    - Identifies partial/cropped grids
    - Extrapolates grid dimensions for incomplete grids

    Args:
        image: Input image (BGR or grayscale)
        use_hough: Whether to use Hough line detection
        use_projection: Whether to use projection-based detection
        use_histogram_fallback: Whether to use histogram analysis as fallback
        use_autocorrelation_fallback: Whether to use autocorrelation as final fallback
        min_spacing: Minimum valid cell spacing
        max_spacing: Maximum valid cell spacing
        ransac_iterations: Number of RANSAC iterations for spacing estimation
        fallback_threshold: Confidence threshold below which to trigger fallbacks
        detect_missing: Whether to detect missing cells for non-rectangular grids
        extrapolate_partial: Whether to extrapolate partial grids
        debug_dir: Optional directory to save debug images

    Returns:
        GridLineResult with detected lines, cells, and partial grid info
    """
    # First run standard grid detection
    result = detect_grid_lines_adaptive(
        image=image,
        use_hough=use_hough,
        use_projection=use_projection,
        use_histogram_fallback=use_histogram_fallback,
        use_autocorrelation_fallback=use_autocorrelation_fallback,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
        ransac_iterations=ransac_iterations,
        fallback_threshold=fallback_threshold,
        debug_dir=debug_dir
    )

    h, w = image.shape[:2] if len(image.shape) == 2 else image.shape[:2]

    # Initialize partial grid info
    partial_info = PartialGridInfo()

    # Detect partial/cropped boundaries
    boundary_type, is_partial = detect_partial_grid_boundaries(
        image,
        result.horizontal_lines,
        result.vertical_lines,
        result.estimated_cell_width,
        result.estimated_cell_height
    )
    partial_info.boundary_type = boundary_type
    partial_info.is_partial = is_partial

    # Extrapolate if partial grid detected
    h_lines = result.horizontal_lines
    v_lines = result.vertical_lines
    extrapolated_dims = None

    if extrapolate_partial and is_partial:
        h_lines, v_lines, extrapolated_dims = extrapolate_partial_grid(
            (h, w),
            result.horizontal_lines,
            result.vertical_lines,
            result.estimated_cell_width,
            result.estimated_cell_height,
            boundary_type
        )
        partial_info.extrapolated_dims = extrapolated_dims

        # Update result with extended lines
        result.horizontal_lines = h_lines
        result.vertical_lines = v_lines

    # Set visible range
    if len(h_lines) >= 2 and len(v_lines) >= 2:
        partial_info.visible_rows = (0, len(h_lines) - 1)
        partial_info.visible_cols = (0, len(v_lines) - 1)

    # Detect missing cells for non-rectangular grids
    presence_mask = None
    missing_cells = []

    if detect_missing and len(h_lines) >= 2 and len(v_lines) >= 2:
        # Convert to BGR if grayscale for saturation analysis
        if len(image.shape) == 2:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = image

        presence_mask, missing_cells = detect_missing_cells(
            img_bgr,
            h_lines,
            v_lines
        )

        partial_info.cell_presence_mask = presence_mask
        partial_info.missing_cells = missing_cells

        # Analyze grid shape
        is_rectangular, shape_desc = analyze_grid_shape(presence_mask)
        partial_info.is_rectangular = is_rectangular

        if not is_rectangular:
            result.method = f"{result.method}_non_rect"

    # Extract cells with presence information
    cells = extract_non_rectangular_cells(
        image,
        h_lines,
        v_lines,
        presence_mask if presence_mask is not None else np.ones((max(len(h_lines) - 1, 1), max(len(v_lines) - 1, 1)), dtype=bool)
    )

    # Update result with partial grid info and cells
    result.partial_grid_info = partial_info
    result.cells = cells

    # Save debug visualization for non-rectangular grids
    if debug_dir and (not partial_info.is_rectangular or partial_info.is_partial):
        out_dir = Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        debug_img = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw all cells, highlighting missing ones
        for cell in cells:
            x, y = cell.x, cell.y
            if cell.is_present:
                color = (0, 255, 0)  # Green for present
            else:
                color = (0, 0, 255)  # Red for missing
            cv2.rectangle(debug_img, (x, y), (x + cell.width, y + cell.height), color, 2)

        cv2.imwrite(str(out_dir / "non_rectangular_grid.png"), debug_img)

        # Save partial grid info
        with open(str(out_dir / "partial_grid_info.txt"), "w") as f:
            f.write(f"Is rectangular: {partial_info.is_rectangular}\n")
            f.write(f"Is partial: {partial_info.is_partial}\n")
            f.write(f"Boundary type: {partial_info.boundary_type}\n")
            f.write(f"Missing cells: {partial_info.missing_cells}\n")
            f.write(f"Extrapolated dims: {partial_info.extrapolated_dims}\n")

    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Detect grid lines in puzzle images")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--debug-dir", default="debug_hybrid", help="Directory for debug output")
    parser.add_argument("--min-spacing", type=float, default=30.0, help="Minimum cell spacing")
    parser.add_argument("--max-spacing", type=float, default=200.0, help="Maximum cell spacing")
    parser.add_argument("--partial", action="store_true", help="Enable partial/non-rectangular grid detection")
    parser.add_argument("--perspective", action="store_true", help="Apply perspective correction before grid detection")
    parser.add_argument("--distortion-threshold", type=float, default=0.05,
                       help="Minimum distortion score to trigger perspective correction (0.0-1.0)")
    parser.add_argument("--validate-quality", action="store_true",
                       help="Validate image quality before processing")
    parser.add_argument("--quality-threshold", type=float, default=0.4,
                       help="Minimum quality score to accept image (0.0-1.0)")
    parser.add_argument("--analyze-distortion", action="store_true",
                       help="Perform detailed distortion analysis")
    parser.add_argument("--auto-correct", action="store_true",
                       help="Automatically correct perspective distortion if detected")
    parser.add_argument("--json-output", action="store_true",
                       help="Output results in JSON format")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    # Track validation results for JSON output
    validation_results = {}

    # Image quality validation if requested
    quality_result = None
    if args.validate_quality:
        quality_result = validate_image_quality(
            img,
            overall_threshold=args.quality_threshold,
            debug_dir=args.debug_dir
        )
        validation_results["quality"] = quality_result.to_dict()

        if not args.json_output:
            print(f"\n=== Image Quality Validation ===")
            print(f"Overall Score: {quality_result.overall_score:.3f}")
            print(f"Acceptable: {quality_result.is_acceptable}")
            print(f"  - Blur: {quality_result.blur_score:.3f}")
            print(f"  - Lighting: {quality_result.lighting_score:.3f}")
            print(f"  - Contrast: {quality_result.contrast_score:.3f}")
            print(f"  - Noise: {quality_result.noise_score:.3f}")
            print(f"  - Resolution: {quality_result.resolution_score:.3f}")
            if quality_result.issues:
                print(f"Issues: {', '.join(quality_result.issues)}")
            if quality_result.recommendations:
                print(f"Recommendations: {', '.join(quality_result.recommendations)}")

        if not quality_result.is_acceptable and not args.auto_correct:
            if not args.json_output:
                print(f"\nWarning: Image quality below threshold ({quality_result.overall_score:.3f} < {args.quality_threshold})")
                print("Consider recapturing the image or use --auto-correct for best results.")

    # Distortion analysis if requested
    distortion_result = None
    if args.analyze_distortion:
        distortion_result = detect_distortion(
            img,
            perspective_threshold=args.distortion_threshold,
            debug_dir=args.debug_dir
        )
        validation_results["distortion"] = distortion_result.to_dict()

        if not args.json_output:
            print(f"\n=== Distortion Analysis ===")
            print(f"Overall Distortion: {distortion_result.overall_distortion:.3f}")
            print(f"Has Significant Distortion: {distortion_result.has_significant_distortion}")
            print(f"Distortion Type: {distortion_result.distortion_type}")
            print(f"  - Perspective: {distortion_result.perspective_distortion:.3f}")
            print(f"  - Barrel: {distortion_result.barrel_distortion:.3f}")
            print(f"  - Rotation: {distortion_result.rotation_angle:.2f}°")
            print(f"  - Skew: {distortion_result.skew_angle:.2f}°")
            print(f"Correction Recommended: {distortion_result.correction_recommended}")

    # Apply perspective correction if requested (either via --perspective or --auto-correct)
    perspective_result = None
    if args.perspective or (args.auto_correct and distortion_result and distortion_result.correction_recommended):
        # Use detected corners from distortion analysis if available
        source_corners = distortion_result.detected_corners if distortion_result else None
        perspective_result = correct_perspective(
            img,
            source_corners=source_corners,
            distortion_threshold=args.distortion_threshold,
            auto_detect_corners=(source_corners is None),
            debug_dir=args.debug_dir
        )
        if perspective_result.was_corrected:
            img = perspective_result.corrected_image
            if not args.json_output:
                print(f"\nPerspective correction applied (distortion: {perspective_result.distortion_score:.3f})")
        else:
            if not args.json_output:
                print(f"\nPerspective correction skipped (distortion: {perspective_result.distortion_score:.3f} < threshold: {args.distortion_threshold})")

    # Find ROI
    roi = find_puzzle_roi(img, debug_dir=args.debug_dir)
    if roi:
        x, y, w, h = roi
        roi_img = img[y:y + h, x:x + w]
    else:
        roi_img = img
        roi = None

    # Detect grid lines
    if args.partial:
        # Use enhanced detection with partial/non-rectangular support
        result = detect_grid_with_partial_support(
            roi_img,
            min_spacing=args.min_spacing,
            max_spacing=args.max_spacing,
            detect_missing=True,
            extrapolate_partial=True,
            debug_dir=args.debug_dir
        )
    else:
        # Use standard detection
        result = detect_grid_lines_adaptive(
            roi_img,
            min_spacing=args.min_spacing,
            max_spacing=args.max_spacing,
            debug_dir=args.debug_dir
        )

    # Extract cells (use cells from result if partial mode, otherwise extract)
    if args.partial and result.cells:
        cells = result.cells
    else:
        cell_tuples = extract_grid_cells(roi_img, result, roi=roi, debug_dir=args.debug_dir)
        cells = [GridCell(row=0, col=i, x=cx, y=cy, width=cw, height=ch)
                 for i, (cx, cy, cw, ch) in enumerate(cell_tuples)]

    # Output results
    if args.json_output:
        # Build JSON output
        output = {
            "grid_detection": result.to_dict(),
            "cells": [c.to_dict() if isinstance(c, GridCell) else {"x": c[0], "y": c[1], "width": c[2], "height": c[3]} for c in cells],
            "validation": validation_results
        }
        if perspective_result:
            output["perspective_correction"] = perspective_result.to_dict()
        print(json.dumps(output, indent=2))
    else:
        print(f"\n=== Grid Detection Results ===")
        print(f"Method: {result.method}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Horizontal lines: {len(result.horizontal_lines)}")
        print(f"Vertical lines: {len(result.vertical_lines)}")
        print(f"Estimated cell size: {result.estimated_cell_width:.1f} x {result.estimated_cell_height:.1f}" if result.estimated_cell_width else "Cell size: N/A")
        print(f"Grid dimensions: {result.grid_dims}" if result.grid_dims else "Grid dims: N/A")
        print(f"Cells detected: {len(cells)}")

        # Output perspective correction info if available
        if perspective_result:
            print(f"Perspective distortion: {perspective_result.distortion_score:.3f}")
            print(f"Perspective corrected: {perspective_result.was_corrected}")
            if perspective_result.was_corrected and perspective_result.corrected_size:
                print(f"Corrected size: {perspective_result.corrected_size[1]}x{perspective_result.corrected_size[0]}")

        # Output partial grid info if available
        if result.partial_grid_info:
            pgi = result.partial_grid_info
            print(f"Is rectangular: {pgi.is_rectangular}")
            print(f"Is partial: {pgi.is_partial}")
            print(f"Boundary type: {pgi.boundary_type}")
            if pgi.missing_cells:
                print(f"Missing cells: {len(pgi.missing_cells)}")
            if pgi.extrapolated_dims:
                print(f"Extrapolated dims: {pgi.extrapolated_dims}")

        # Write cells to file
        with open("cells.txt", "w") as f:
            for cell in cells:
                if isinstance(cell, GridCell):
                    f.write(f"{cell.x},{cell.y},{cell.width},{cell.height},{cell.row},{cell.col},{cell.is_present}\n")
                else:
                    f.write(f"{cell[0]},{cell[1]},{cell[2]},{cell[3]}\n")
        print("Wrote cells.txt")
