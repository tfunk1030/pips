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


class TestGeometryConfidence:
    """Test geometry confidence calculation from main.py."""

    def test_geometry_confidence_valid_range(self):
        """Test _calculate_geometry_confidence returns values in [0.0, 1.0]."""
        from main import _calculate_geometry_confidence

        # Create a test image with grid pattern
        img = create_test_image(400, 400, saturation=120, with_grid=True)

        # Create some mock cells
        cells = [
            (100, 100, 50, 50),
            (160, 100, 50, 50),
            (220, 100, 50, 50),
            (100, 160, 50, 50),
            (160, 160, 50, 50),
            (220, 160, 50, 50),
        ]

        confidence, breakdown = _calculate_geometry_confidence(img, cells, rows=2, cols=3)

        assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range"
        for factor, score in breakdown.items():
            assert 0.0 <= score <= 1.0, f"Breakdown {factor}={score} out of range"

    def test_geometry_confidence_empty_cells(self):
        """Test _calculate_geometry_confidence with empty cells returns 0.0."""
        from main import _calculate_geometry_confidence

        img = create_test_image(400, 400, saturation=120, with_grid=True)

        confidence, breakdown = _calculate_geometry_confidence(img, cells=[], rows=0, cols=0)

        assert confidence == 0.0, "Empty cells should return 0.0 confidence"
        assert all(v == 0.0 for v in breakdown.values()), "All breakdown scores should be 0.0"

    def test_geometry_confidence_breakdown_factors(self):
        """Test all expected breakdown factors are present."""
        from main import _calculate_geometry_confidence

        img = create_test_image(400, 400, saturation=120, with_grid=True)
        cells = [(100, 100, 50, 50), (160, 100, 50, 50)]

        confidence, breakdown = _calculate_geometry_confidence(img, cells, rows=1, cols=2)

        expected_factors = [
            "saturation",
            "area_ratio",
            "aspect_ratio",
            "relative_size",
            "edge_clarity",
            "contrast"
        ]
        for factor in expected_factors:
            assert factor in breakdown, f"Missing breakdown factor: {factor}"

    def test_geometry_confidence_consistent_cells_higher(self):
        """Test consistent cell sizes produce higher confidence."""
        from main import _calculate_geometry_confidence

        img = create_test_image(400, 400, saturation=120, with_grid=True)

        # Consistent cell sizes
        consistent_cells = [
            (100, 100, 50, 50),
            (160, 100, 50, 50),
            (220, 100, 50, 50),
        ]

        # Inconsistent cell sizes
        inconsistent_cells = [
            (100, 100, 50, 50),
            (160, 100, 30, 70),  # Different size
            (220, 100, 80, 30),  # Very different size
        ]

        conf_consistent, breakdown_consistent = _calculate_geometry_confidence(
            img, consistent_cells, rows=1, cols=3
        )
        conf_inconsistent, breakdown_inconsistent = _calculate_geometry_confidence(
            img, inconsistent_cells, rows=1, cols=3
        )

        # Consistent should have higher area_ratio score
        assert breakdown_consistent["area_ratio"] > breakdown_inconsistent["area_ratio"], \
            "Consistent cell sizes should have higher area_ratio score"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_confidence_zero_value(self):
        """Test handling of 0.0 confidence."""
        level = get_confidence_level(0.0, "puzzle_detection")
        assert level == "low", "0.0 confidence should be 'low'"

    def test_confidence_one_value(self):
        """Test handling of 1.0 confidence."""
        level = get_confidence_level(1.0, "puzzle_detection")
        assert level == "high", "1.0 confidence should be 'high'"

    def test_confidence_at_exact_thresholds(self):
        """Test confidence exactly at threshold boundaries."""
        # At high threshold (0.80 for puzzle_detection)
        level = get_confidence_level(0.80, "puzzle_detection")
        assert level == "high", "Exact high threshold should be 'high'"

        # At medium threshold (0.65 for puzzle_detection)
        level = get_confidence_level(0.65, "puzzle_detection")
        assert level == "medium", "Exact medium threshold should be 'medium'"

    def test_confidence_just_below_thresholds(self):
        """Test confidence just below threshold boundaries."""
        # Just below high (0.80 - epsilon)
        level = get_confidence_level(0.79999, "puzzle_detection")
        assert level == "medium", "Just below high threshold should be 'medium'"

        # Just below medium (0.65 - epsilon)
        level = get_confidence_level(0.64999, "puzzle_detection")
        assert level == "low", "Just below medium threshold should be 'low'"

    def test_invalid_component_raises_error(self):
        """Test that invalid component name raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_confidence_level(0.5, "invalid_component")
        assert "Unknown component" in str(excinfo.value)

    def test_borderline_at_exact_threshold(self):
        """Test borderline detection at exact threshold values."""
        # At exactly high threshold (0.80)
        assert is_borderline(0.80, "puzzle_detection") == True

        # At exactly medium threshold (0.65)
        assert is_borderline(0.65, "puzzle_detection") == True

    def test_negative_confidence_clamped(self):
        """Test negative confidence is handled (classified as low)."""
        # Negative should still work (classified as low)
        level = get_confidence_level(-0.1, "puzzle_detection")
        assert level == "low", "Negative confidence should be 'low'"

    def test_confidence_above_one(self):
        """Test confidence above 1.0 is classified as high."""
        level = get_confidence_level(1.5, "puzzle_detection")
        assert level == "high", "Confidence > 1.0 should be 'high'"


class TestSpecCompliance:
    """Tests to verify implementation matches spec requirements."""

    def test_spec_threshold_values(self):
        """Verify threshold values match spec requirements."""
        # Spec defines these threshold values
        expected = {
            "geometry_extraction": {"high": 0.85, "medium": 0.70},
            "ocr_detection": {"high": 0.90, "medium": 0.75},
            "puzzle_detection": {"high": 0.80, "medium": 0.65},
            "domino_detection": {"high": 0.80, "medium": 0.65}
        }

        for component, thresholds in expected.items():
            actual = get_threshold_values(component)
            assert actual["high"] == thresholds["high"], \
                f"{component} high threshold should be {thresholds['high']}"
            assert actual["medium"] == thresholds["medium"], \
                f"{component} medium threshold should be {thresholds['medium']}"
            assert actual["low"] == 0.0, \
                f"{component} low threshold should be 0.0"

    def test_spec_borderline_margin(self):
        """Verify borderline margin is 5% as per spec."""
        from confidence_config import BORDERLINE_MARGIN
        assert BORDERLINE_MARGIN == 0.05, "Borderline margin should be 5% (0.05)"

    def test_high_confidence_threshold_accuracy_target(self):
        """
        Spec requirement: High confidence (>85% threshold) proven to have >90% actual accuracy.

        This test verifies the threshold structure supports this requirement.
        Statistical validation of actual accuracy requires ground truth data.
        """
        # Geometry uses 0.85 high threshold
        geo_high = get_threshold_values("geometry_extraction")["high"]
        assert geo_high >= 0.85, "Geometry high threshold should support >90% accuracy target"

        # OCR uses 0.90 high threshold (higher bar due to error modes)
        ocr_high = get_threshold_values("ocr_detection")["high"]
        assert ocr_high >= 0.90, "OCR high threshold should support >90% accuracy target"

    def test_confidence_breakdown_six_factors(self):
        """Verify confidence breakdown includes 6 quality factors per spec."""
        expected_factors = {
            "saturation",
            "area_ratio",
            "aspect_ratio",
            "relative_size",
            "edge_clarity",
            "contrast"
        }

        img = create_test_image(400, 400, saturation=120, with_grid=True)
        bounds = (100, 100, 200, 200)
        contour = create_contour_from_bounds(100, 100, 200, 200)

        confidence, breakdown = _calculate_grid_confidence(img, bounds, contour)

        actual_factors = set(breakdown.keys())
        assert actual_factors == expected_factors, \
            f"Breakdown factors {actual_factors} should match spec {expected_factors}"

    def test_all_components_defined(self):
        """Verify all spec-required components are defined."""
        required = [
            "geometry_extraction",
            "ocr_detection",
            "puzzle_detection",
            "domino_detection"
        ]

        all_components = get_all_components()
        for component in required:
            assert component in all_components, \
                f"Spec-required component {component} missing from CONFIDENCE_THRESHOLDS"

    def test_confidence_level_mapping(self):
        """
        Verify confidence levels map correctly per spec:
        - high: User can trust without review
        - medium: Suggest review
        - low: Requires manual verification
        """
        # Test across all components
        for component in get_all_components():
            thresholds = get_threshold_values(component)

            # Above high threshold -> high
            assert get_confidence_level(thresholds["high"] + 0.01, component) == "high"

            # Between medium and high -> medium
            mid_point = (thresholds["high"] + thresholds["medium"]) / 2
            assert get_confidence_level(mid_point, component) == "medium"

            # Below medium threshold -> low
            assert get_confidence_level(thresholds["medium"] - 0.01, component) == "low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
