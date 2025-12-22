"""
Confidence Calibration Tests

Tests for the calibrated confidence scoring system to ensure:
1. Confidence scores are in valid range [0.0, 1.0]
2. Confidence levels map correctly to thresholds
3. Grid confidence calculation includes image quality factors
4. Borderline detection works correctly
"""

import pytest
import numpy as np
import cv2

from confidence_config import (
    CONFIDENCE_THRESHOLDS,
    get_confidence_level,
    is_borderline,
    get_threshold_values,
    get_all_components
)
from hybrid_extraction import _calculate_grid_confidence, find_puzzle_roi


def create_test_image(width: int = 400, height: int = 400,
                      saturation: int = 100, with_grid: bool = True) -> np.ndarray:
    """
    Create a synthetic test image for confidence testing.

    Args:
        width: Image width
        height: Image height
        saturation: Saturation level (0-255)
        with_grid: If True, add a colorful grid pattern
    """
    # Create base image (dark background)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)  # Dark gray background

    if with_grid:
        # Add a colorful grid region in the center
        grid_x, grid_y = width // 4, height // 4
        grid_w, grid_h = width // 2, height // 2

        # Create a colorful grid pattern (simulating puzzle cells)
        for row in range(5):
            for col in range(5):
                cell_x = grid_x + col * (grid_w // 5)
                cell_y = grid_y + row * (grid_h // 5)
                cell_w = grid_w // 5 - 4
                cell_h = grid_h // 5 - 4

                # Random color with specified saturation
                hue = (row * 5 + col * 7) * 10 % 180  # Varied hues
                color_hsv = np.array([[[hue, saturation, 200]]], dtype=np.uint8)
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]

                img[cell_y:cell_y+cell_h, cell_x:cell_x+cell_w] = color_bgr

    return img


def create_contour_from_bounds(x: int, y: int, w: int, h: int) -> np.ndarray:
    """Create a rectangular contour from bounds."""
    return np.array([
        [[x, y]],
        [[x + w, y]],
        [[x + w, y + h]],
        [[x, y + h]]
    ], dtype=np.int32)


def test_grid_confidence_range():
    """
    Test that _calculate_grid_confidence returns values in [0.0, 1.0].
    This is the main verification test for subtask-1-2.
    Module-level function for easy pytest discovery.
    """
    # Create test image with colorful grid
    img = create_test_image(400, 400, saturation=120, with_grid=True)

    # Define bounds for the grid region
    bounds = (100, 100, 200, 200)
    x, y, w, h = bounds
    contour = create_contour_from_bounds(x, y, w, h)

    # Calculate confidence
    confidence, breakdown = _calculate_grid_confidence(img, bounds, contour)

    # Verify overall confidence is in range
    assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range [0.0, 1.0]"

    # Verify all breakdown scores are in range
    for factor, score in breakdown.items():
        assert 0.0 <= score <= 1.0, f"Breakdown score {factor}={score} out of range"

    # Verify expected factors are present
    expected_factors = [
        "saturation",
        "area_ratio",
        "aspect_ratio",
        "relative_size",
        "edge_clarity",
        "contrast"
    ]
    for factor in expected_factors:
        assert factor in breakdown, f"Missing factor: {factor}"


class TestGridConfidenceRange:
    """Test that grid confidence scores are in valid range [0.0, 1.0]."""

    def test_grid_confidence_range_class(self):
        """
        Test that _calculate_grid_confidence returns values in [0.0, 1.0].
        This is the main verification test for subtask-1-2.
        """
        # Create test image with colorful grid
        img = create_test_image(400, 400, saturation=120, with_grid=True)

        # Define bounds for the grid region
        bounds = (100, 100, 200, 200)
        x, y, w, h = bounds
        contour = create_contour_from_bounds(x, y, w, h)

        # Calculate confidence
        confidence, breakdown = _calculate_grid_confidence(img, bounds, contour)

        # Verify overall confidence is in range
        assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range [0.0, 1.0]"

        # Verify all breakdown scores are in range
        for factor, score in breakdown.items():
            assert 0.0 <= score <= 1.0, f"Breakdown score {factor}={score} out of range"

    def test_grid_confidence_low_saturation(self):
        """Test confidence is lower for low saturation images."""
        # Low saturation image (grayscale-ish)
        img = create_test_image(400, 400, saturation=20, with_grid=True)
        bounds = (100, 100, 200, 200)
        contour = create_contour_from_bounds(100, 100, 200, 200)

        confidence, breakdown = _calculate_grid_confidence(img, bounds, contour)

        # Low saturation should result in lower confidence
        assert 0.0 <= confidence <= 1.0
        assert breakdown["saturation"] < 0.5, "Low saturation should have low saturation score"

    def test_grid_confidence_high_saturation(self):
        """Test confidence is higher for high saturation images."""
        # High saturation image
        img = create_test_image(400, 400, saturation=180, with_grid=True)
        bounds = (100, 100, 200, 200)
        contour = create_contour_from_bounds(100, 100, 200, 200)

        confidence, breakdown = _calculate_grid_confidence(img, bounds, contour)

        assert 0.0 <= confidence <= 1.0
        assert breakdown["saturation"] >= 0.7, "High saturation should have high saturation score"

    def test_grid_confidence_no_grid(self):
        """Test confidence is lower for images without clear grid."""
        # Image without grid pattern
        img = create_test_image(400, 400, saturation=50, with_grid=False)
        bounds = (100, 100, 200, 200)
        contour = create_contour_from_bounds(100, 100, 200, 200)

        confidence, breakdown = _calculate_grid_confidence(img, bounds, contour)

        # Should still be in valid range
        assert 0.0 <= confidence <= 1.0

    def test_grid_confidence_breakdown_factors(self):
        """Test that all expected factors are present in breakdown."""
        img = create_test_image(400, 400, saturation=120, with_grid=True)
        bounds = (100, 100, 200, 200)
        contour = create_contour_from_bounds(100, 100, 200, 200)

        confidence, breakdown = _calculate_grid_confidence(img, bounds, contour)

        # Check all expected factors are present
        expected_factors = [
            "saturation",
            "area_ratio",
            "aspect_ratio",
            "relative_size",
            "edge_clarity",
            "contrast"
        ]

        for factor in expected_factors:
            assert factor in breakdown, f"Missing factor: {factor}"

    def test_grid_confidence_aspect_ratio_square(self):
        """Test that square grids get good aspect ratio score."""
        img = create_test_image(400, 400, saturation=120, with_grid=True)
        # Square bounds
        bounds = (100, 100, 200, 200)
        contour = create_contour_from_bounds(100, 100, 200, 200)

        confidence, breakdown = _calculate_grid_confidence(img, bounds, contour)

        assert breakdown["aspect_ratio"] >= 0.9, "Square grid should have high aspect score"

    def test_grid_confidence_aspect_ratio_stretched(self):
        """Test that very stretched grids get lower aspect ratio score."""
        img = create_test_image(600, 200, saturation=120, with_grid=True)
        # Very wide bounds (3:1 aspect ratio)
        bounds = (50, 25, 500, 150)
        contour = create_contour_from_bounds(50, 25, 500, 150)

        confidence, breakdown = _calculate_grid_confidence(img, bounds, contour)

        assert breakdown["aspect_ratio"] < 0.8, "Stretched grid should have lower aspect score"


class TestConfidenceThresholds:
    """Test confidence threshold classification."""

    def test_get_confidence_level_high(self):
        """Test high confidence level classification."""
        # For puzzle_detection, high threshold is 0.80
        level = get_confidence_level(0.90, "puzzle_detection")
        assert level == "high"

        level = get_confidence_level(0.80, "puzzle_detection")
        assert level == "high"

    def test_get_confidence_level_medium(self):
        """Test medium confidence level classification."""
        # For puzzle_detection, medium threshold is 0.65
        level = get_confidence_level(0.75, "puzzle_detection")
        assert level == "medium"

        level = get_confidence_level(0.65, "puzzle_detection")
        assert level == "medium"

    def test_get_confidence_level_low(self):
        """Test low confidence level classification."""
        level = get_confidence_level(0.50, "puzzle_detection")
        assert level == "low"

        level = get_confidence_level(0.0, "puzzle_detection")
        assert level == "low"

    def test_is_borderline_near_high_threshold(self):
        """Test borderline detection near high threshold."""
        # High threshold for puzzle_detection is 0.80
        # Borderline margin is 0.05
        assert is_borderline(0.78, "puzzle_detection") == True
        assert is_borderline(0.82, "puzzle_detection") == True
        # 0.72 is outside the borderline range of both thresholds
        # (not within 0.05 of 0.80 high, not within 0.05 of 0.65 medium)
        assert is_borderline(0.72, "puzzle_detection") == False

    def test_is_borderline_near_medium_threshold(self):
        """Test borderline detection near medium threshold."""
        # Medium threshold for puzzle_detection is 0.65
        assert is_borderline(0.63, "puzzle_detection") == True
        assert is_borderline(0.67, "puzzle_detection") == True
        assert is_borderline(0.55, "puzzle_detection") == False


class TestComponentThresholds:
    """Test component-specific threshold configuration."""

    def test_all_components_have_thresholds(self):
        """Verify all required components have threshold definitions."""
        required_components = [
            "geometry_extraction",
            "ocr_detection",
            "puzzle_detection",
            "domino_detection"
        ]

        for component in required_components:
            assert component in CONFIDENCE_THRESHOLDS, f"Missing threshold for {component}"

    def test_threshold_values_structure(self):
        """Test that each component has high, medium, low thresholds."""
        for component in get_all_components():
            thresholds = get_threshold_values(component)
            assert "high" in thresholds
            assert "medium" in thresholds
            assert "low" in thresholds

    def test_threshold_ordering(self):
        """Test that thresholds are properly ordered (high > medium > low)."""
        for component in get_all_components():
            thresholds = get_threshold_values(component)
            assert thresholds["high"] > thresholds["medium"], f"{component}: high should be > medium"
            assert thresholds["medium"] > thresholds["low"], f"{component}: medium should be > low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
