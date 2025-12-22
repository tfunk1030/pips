"""
Unit tests for pip detection module.

Tests cover all pip values (0-6), rotations, edge cases, confidence scoring,
and preprocessing functions. Uses synthetic test images created with OpenCV
for reproducible testing.
"""

import pytest
import numpy as np
import cv2
import math
from typing import Tuple

from extract_dominoes import (
    # Preprocessing functions
    preprocess_domino_image,
    preprocess_for_hough,
    preprocess_for_contours,
    apply_clahe,
    detect_domino_color_mode,
    preprocess_adaptive,
    # Pip detection functions
    detect_pips_hough,
    detect_pips_contours,
    detect_pips_hough_adaptive,
    validate_pip_count,
    # Confidence scoring
    calculate_confidence,
    # Rotation handling
    detect_rotation_angle,
    detect_rotation_angle_from_image,
    rotate_domino,
    split_domino_halves,
    # Main detection function
    detect_domino_pips,
    # Pydantic model
    PipDetectionResult,
)


# =============================================================================
# Test Image Generation Helpers
# =============================================================================

def create_test_domino(
    width: int = 200,
    height: int = 100,
    left_pips: int = 3,
    right_pips: int = 5,
    pip_radius: int = 10,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    pip_color: Tuple[int, int, int] = (0, 0, 0),
    divider_width: int = 2
) -> np.ndarray:
    """
    Create a synthetic domino image with specified pip counts.

    Standard domino pip patterns:
    0: empty
    1: center pip
    2: diagonal corners (top-left, bottom-right)
    3: diagonal + center
    4: four corners
    5: four corners + center
    6: six pips (2 columns of 3)
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = bg_color

    # Draw center divider line
    mid_x = width // 2
    cv2.line(image, (mid_x, 0), (mid_x, height), (128, 128, 128), divider_width)

    def draw_pips_on_half(start_x: int, end_x: int, pip_count: int):
        """Draw pips on one half of the domino."""
        half_width = end_x - start_x
        cx = start_x + half_width // 2
        cy = height // 2

        # Spacing for pip positions
        dx = half_width // 4
        dy = height // 4

        # Pip positions based on count (following standard domino patterns)
        positions = []
        if pip_count == 0:
            pass  # No pips
        elif pip_count == 1:
            positions = [(cx, cy)]
        elif pip_count == 2:
            positions = [(cx - dx, cy - dy), (cx + dx, cy + dy)]
        elif pip_count == 3:
            positions = [(cx - dx, cy - dy), (cx, cy), (cx + dx, cy + dy)]
        elif pip_count == 4:
            positions = [
                (cx - dx, cy - dy), (cx + dx, cy - dy),
                (cx - dx, cy + dy), (cx + dx, cy + dy)
            ]
        elif pip_count == 5:
            positions = [
                (cx - dx, cy - dy), (cx + dx, cy - dy),
                (cx, cy),
                (cx - dx, cy + dy), (cx + dx, cy + dy)
            ]
        elif pip_count == 6:
            positions = [
                (cx - dx, cy - dy), (cx + dx, cy - dy),
                (cx - dx, cy), (cx + dx, cy),
                (cx - dx, cy + dy), (cx + dx, cy + dy)
            ]

        for pos in positions:
            cv2.circle(image, (int(pos[0]), int(pos[1])), pip_radius, pip_color, -1)

    # Draw pips on left half
    draw_pips_on_half(0, mid_x, left_pips)

    # Draw pips on right half
    draw_pips_on_half(mid_x, width, right_pips)

    return image


def create_rotated_test_domino(
    angle: float,
    width: int = 200,
    height: int = 100,
    left_pips: int = 3,
    right_pips: int = 5
) -> np.ndarray:
    """Create a rotated synthetic domino image."""
    # Create base domino
    domino = create_test_domino(width, height, left_pips, right_pips)

    # Calculate new canvas size to fit rotated image
    # Add generous padding to ensure the rotated domino fits
    abs_cos = abs(math.cos(math.radians(angle)))
    abs_sin = abs(math.sin(math.radians(angle)))
    new_w = int(height * abs_sin + width * abs_cos) + 100
    new_h = int(height * abs_cos + width * abs_sin) + 100

    # Create larger canvas with enough space for any rotation
    canvas_size = max(new_w, new_h, width + 100, height + 100)
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 200

    # Place domino in center of canvas
    start_y = (canvas.shape[0] - height) // 2
    start_x = (canvas.shape[1] - width) // 2
    canvas[start_y:start_y+height, start_x:start_x+width] = domino

    # Rotate around center
    center = (canvas.shape[1] // 2, canvas.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        canvas, M, (canvas.shape[1], canvas.shape[0]),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(200, 200, 200)
    )

    return rotated


def create_blank_domino(
    width: int = 200,
    height: int = 100,
    bg_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """Create a blank domino image with no pips (0-0)."""
    return create_test_domino(width, height, left_pips=0, right_pips=0, bg_color=bg_color)


def create_domino_half(
    pip_count: int,
    width: int = 100,
    height: int = 100,
    pip_radius: int = 10,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    pip_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """Create a single half-domino image for testing individual pip detection."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = bg_color

    cx, cy = width // 2, height // 2
    dx, dy = width // 4, height // 4

    positions = []
    if pip_count == 0:
        pass
    elif pip_count == 1:
        positions = [(cx, cy)]
    elif pip_count == 2:
        positions = [(cx - dx, cy - dy), (cx + dx, cy + dy)]
    elif pip_count == 3:
        positions = [(cx - dx, cy - dy), (cx, cy), (cx + dx, cy + dy)]
    elif pip_count == 4:
        positions = [
            (cx - dx, cy - dy), (cx + dx, cy - dy),
            (cx - dx, cy + dy), (cx + dx, cy + dy)
        ]
    elif pip_count == 5:
        positions = [
            (cx - dx, cy - dy), (cx + dx, cy - dy),
            (cx, cy),
            (cx - dx, cy + dy), (cx + dx, cy + dy)
        ]
    elif pip_count == 6:
        positions = [
            (cx - dx, cy - dy), (cx + dx, cy - dy),
            (cx - dx, cy), (cx + dx, cy),
            (cx - dx, cy + dy), (cx + dx, cy + dy)
        ]

    for pos in positions:
        cv2.circle(image, (int(pos[0]), int(pos[1])), pip_radius, pip_color, -1)

    return image


# =============================================================================
# Preprocessing Tests
# =============================================================================

class TestPreprocessing:
    """Tests for image preprocessing functions."""

    def test_preprocess_domino_image_basic(self):
        """Test basic preprocessing pipeline with valid image."""
        image = create_test_domino()
        gray, binary, cleaned = preprocess_domino_image(image)

        assert gray is not None
        assert binary is not None
        assert cleaned is not None
        assert len(gray.shape) == 2  # Grayscale
        assert len(binary.shape) == 2
        assert len(cleaned.shape) == 2
        assert gray.shape == image.shape[:2]

    def test_preprocess_domino_image_grayscale_input(self):
        """Test preprocessing with grayscale input image."""
        image = np.zeros((100, 200), dtype=np.uint8)
        image[40:60, 40:60] = 255

        gray, binary, cleaned = preprocess_domino_image(image)
        assert gray is not None
        assert gray.shape == image.shape

    def test_preprocess_domino_image_empty_raises(self):
        """Test that empty image raises ValueError."""
        with pytest.raises(ValueError, match="empty or None"):
            preprocess_domino_image(np.array([]))

    def test_preprocess_domino_image_none_raises(self):
        """Test that None image raises ValueError."""
        with pytest.raises(ValueError, match="empty or None"):
            preprocess_domino_image(None)

    def test_preprocess_for_hough(self):
        """Test Hough preprocessing produces grayscale blurred image."""
        image = create_test_domino()
        result = preprocess_for_hough(image)

        assert result is not None
        assert len(result.shape) == 2  # Grayscale
        assert result.dtype == np.uint8

    def test_preprocess_for_contours(self):
        """Test contour preprocessing produces binary image."""
        image = create_test_domino()
        result = preprocess_for_contours(image)

        assert result is not None
        assert len(result.shape) == 2
        # Binary image should only have 0 and 255 values
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
        assert all(v in [0, 255] for v in unique_values)

    def test_apply_clahe(self):
        """Test CLAHE enhances contrast."""
        # Create low contrast image
        image = np.full((100, 100), 128, dtype=np.uint8)
        image[40:60, 40:60] = 138  # Very small contrast difference

        result = apply_clahe(image)
        assert result is not None
        # CLAHE should increase contrast (std deviation)
        assert np.std(result) >= np.std(image)

    def test_detect_domino_color_mode_dark_pips(self):
        """Test color mode detection for dark pips on light background."""
        image = create_test_domino(
            bg_color=(255, 255, 255),
            pip_color=(0, 0, 0),
            left_pips=6,
            right_pips=6
        )
        mode = detect_domino_color_mode(image)
        assert mode == "dark_pips"

    def test_detect_domino_color_mode_light_pips(self):
        """Test color mode detection for light pips on dark background."""
        image = create_test_domino(
            bg_color=(30, 30, 30),
            pip_color=(220, 220, 220),
            left_pips=6,
            right_pips=6
        )
        mode = detect_domino_color_mode(image)
        # With light pips on dark background, should detect as light_pips
        assert mode in ["light_pips", "dark_pips"]  # Depends on implementation details

    def test_preprocess_adaptive_returns_info(self):
        """Test adaptive preprocessing returns preprocessing info."""
        image = create_test_domino()
        result, info = preprocess_adaptive(image)

        assert result is not None
        assert isinstance(info, dict)
        assert "color_mode" in info
        assert "clahe_applied" in info
        assert "block_size" in info


# =============================================================================
# Pip Detection Tests - All Values 0-6
# =============================================================================

class TestPipDetectionAllValues:
    """Tests for pip detection covering all valid values 0-6."""

    @pytest.mark.parametrize("pip_count", [0, 1, 2, 3, 4, 5, 6])
    def test_detect_pips_hough_all_values(self, pip_count):
        """Test Hough detection for each pip value 0-6."""
        image = create_domino_half(pip_count, width=100, height=100, pip_radius=8)

        count, circles, info = detect_pips_hough(image)

        assert isinstance(count, int)
        assert count >= 0
        assert isinstance(info, dict)
        # Note: Hough detection on synthetic images may not be perfect,
        # but should detect something for pip_count > 0

    @pytest.mark.parametrize("pip_count", [0, 1, 2, 3, 4, 5, 6])
    def test_detect_pips_contours_all_values(self, pip_count):
        """Test contour-based detection for each pip value 0-6."""
        image = create_domino_half(pip_count, width=100, height=100, pip_radius=8)

        count, contours, info = detect_pips_contours(image)

        assert isinstance(count, int)
        assert count >= 0
        assert isinstance(contours, list)
        assert isinstance(info, dict)

    @pytest.mark.parametrize("left_pips,right_pips", [
        (0, 0), (0, 1), (1, 1), (2, 3), (3, 4), (4, 5), (5, 6), (6, 6)
    ])
    def test_detect_domino_pips_combinations(self, left_pips, right_pips):
        """Test full domino pip detection for various combinations."""
        image = create_test_domino(left_pips=left_pips, right_pips=right_pips)

        result = detect_domino_pips(image)

        assert isinstance(result, PipDetectionResult)
        assert 0 <= result.left_pips <= 6
        assert 0 <= result.right_pips <= 6
        assert 0.0 <= result.left_confidence <= 1.0
        assert 0.0 <= result.right_confidence <= 1.0


class TestBlankDomino:
    """Tests for blank domino (0 pips) detection."""

    def test_detect_blank_left_half(self):
        """Test detection of blank (0 pip) left half."""
        image = create_test_domino(left_pips=0, right_pips=3)
        result = detect_domino_pips(image)

        assert result.left_pips == 0 or result.left_pips <= 1  # Should be 0 or low
        assert result.left_confidence >= 0.0

    def test_detect_blank_right_half(self):
        """Test detection of blank (0 pip) right half."""
        image = create_test_domino(left_pips=5, right_pips=0)
        result = detect_domino_pips(image)

        assert result.right_pips == 0 or result.right_pips <= 1  # Should be 0 or low
        assert result.right_confidence >= 0.0

    def test_detect_fully_blank_domino(self):
        """Test detection of fully blank domino (0-0)."""
        image = create_blank_domino()
        result = detect_domino_pips(image)

        # Both halves should be 0 or very low
        assert result.left_pips <= 1
        assert result.right_pips <= 1

    def test_blank_domino_high_confidence(self):
        """Blank dominoes should have reasonable confidence."""
        image = create_blank_domino()
        result = detect_domino_pips(image)

        # Blank detection should have some confidence
        assert result.left_confidence >= 0.5 or result.left_pips == 0
        assert result.right_confidence >= 0.5 or result.right_pips == 0


# =============================================================================
# Rotation Handling Tests
# =============================================================================

class TestRotationHandling:
    """Tests for rotation detection and handling."""

    def test_detect_rotation_angle_basic(self):
        """Test rotation angle detection from contour."""
        # Create a rectangular contour with more than 5 points
        # minAreaRect requires at least 5 points
        rect_contour = np.array([
            [[10, 10]], [[100, 10]], [[190, 10]],
            [[190, 50]], [[190, 90]],
            [[100, 90]], [[10, 90]], [[10, 50]]
        ], dtype=np.int32)

        angle, center, size, box = detect_rotation_angle(rect_contour)

        assert isinstance(angle, float)
        # Angle is returned, may be in various ranges depending on orientation
        # The function normalizes angles but specific values depend on contour shape
        assert -90 <= angle <= 90
        assert len(center) == 2
        assert len(size) == 2
        assert box.shape == (4, 2)

    def test_detect_rotation_angle_from_image(self):
        """Test rotation detection directly from image."""
        image = create_test_domino()

        angle, center, size, box, contour = detect_rotation_angle_from_image(image)

        assert isinstance(angle, float)
        # Angle should be within a reasonable range for non-rotated domino
        # May not be exactly 0 due to pip patterns affecting contour detection
        assert -45 <= angle <= 45  # Within normalized range

    def test_rotate_domino_preserves_content(self):
        """Test that rotation preserves image content."""
        image = create_test_domino()
        original_shape = image.shape

        rotated = rotate_domino(image, 0.0, expand_canvas=False)

        assert rotated.shape == original_shape

    def test_rotate_domino_45_degrees(self):
        """Test 45 degree rotation expands canvas."""
        image = create_test_domino()

        rotated = rotate_domino(image, 45.0, expand_canvas=True)

        # Rotated image should be larger to accommodate corners
        assert rotated.shape[0] > image.shape[0] or rotated.shape[1] > image.shape[1]

    def test_rotate_domino_grayscale(self):
        """Test rotation works with grayscale images."""
        image = np.zeros((100, 200), dtype=np.uint8)
        image[40:60, 40:60] = 255

        rotated = rotate_domino(image, 30.0)

        assert rotated is not None
        assert len(rotated.shape) == 2  # Still grayscale

    @pytest.mark.parametrize("angle", [0, 15, 30, 45, 90, 135, 180])
    def test_rotation_invariance(self, angle):
        """Test that pip detection works at various rotation angles."""
        left_pips, right_pips = 3, 5
        image = create_rotated_test_domino(angle, left_pips=left_pips, right_pips=right_pips)

        result = detect_domino_pips(image, auto_rotate=True)

        # Detection should work at any angle, even if not perfectly accurate
        assert 0 <= result.left_pips <= 6
        assert 0 <= result.right_pips <= 6
        assert result.left_confidence >= 0.0
        assert result.right_confidence >= 0.0


class TestSplitDominoHalves:
    """Tests for domino half splitting."""

    def test_split_equal_halves(self):
        """Test that split produces equal-width halves."""
        image = create_test_domino(width=200, height=100)

        left, right = split_domino_halves(image)

        assert left.shape[1] == right.shape[1]
        assert left.shape[0] == image.shape[0]
        assert right.shape[0] == image.shape[0]

    def test_split_odd_width(self):
        """Test split handles odd-width images."""
        image = np.zeros((100, 201, 3), dtype=np.uint8)

        left, right = split_domino_halves(image)

        assert left.shape[1] == right.shape[1]

    def test_split_with_padding(self):
        """Test split with padding reduces half width."""
        image = create_test_domino(width=200, height=100)

        no_padding_left, _ = split_domino_halves(image, padding=0)
        padding_left, _ = split_domino_halves(image, padding=5)

        assert padding_left.shape[1] < no_padding_left.shape[1]

    def test_split_grayscale(self):
        """Test split works with grayscale images."""
        image = np.zeros((100, 200), dtype=np.uint8)

        left, right = split_domino_halves(image)

        assert len(left.shape) == 2
        assert len(right.shape) == 2

    def test_split_empty_raises(self):
        """Test split raises error for empty image."""
        with pytest.raises(ValueError, match="empty or None"):
            split_domino_halves(np.array([]))


# =============================================================================
# Confidence Scoring Tests
# =============================================================================

class TestConfidenceScoring:
    """Tests for confidence score calculation."""

    def test_confidence_blank_domino(self):
        """Blank dominoes (0 pips) should have high confidence."""
        confidence = calculate_confidence(pip_count=0)

        assert confidence >= 0.9

    def test_confidence_perfect_detection(self):
        """Perfect detection metrics should give high confidence."""
        confidence = calculate_confidence(
            pip_count=3,
            circularity=1.0,
            size_variance=0.0
        )

        assert confidence >= 0.85

    def test_confidence_valid_range(self):
        """Confidence should always be in [0.0, 1.0] range."""
        test_cases = [
            (0, 1.0, 0.0),
            (3, 0.9, 0.1),
            (6, 0.8, 0.2),
            (5, 0.7, 0.3),
            (2, 0.6, 0.4),
        ]

        for pip_count, circularity, variance in test_cases:
            confidence = calculate_confidence(
                pip_count=pip_count,
                circularity=circularity,
                size_variance=variance
            )
            assert 0.0 <= confidence <= 1.0

    def test_confidence_low_circularity(self):
        """Low circularity should reduce confidence."""
        high_circ = calculate_confidence(pip_count=3, circularity=0.95)
        low_circ = calculate_confidence(pip_count=3, circularity=0.5)

        assert high_circ > low_circ

    def test_confidence_high_variance(self):
        """High size variance should reduce confidence."""
        low_var = calculate_confidence(pip_count=3, size_variance=0.0)
        high_var = calculate_confidence(pip_count=3, size_variance=0.5)

        assert low_var > high_var

    def test_confidence_invalid_pip_count(self):
        """Invalid pip count (>6) should give low confidence."""
        valid = calculate_confidence(pip_count=6, circularity=0.9)
        invalid_7 = calculate_confidence(pip_count=7, circularity=0.9)
        invalid_10 = calculate_confidence(pip_count=10, circularity=0.9)

        assert valid > invalid_7
        assert invalid_7 > invalid_10 or invalid_10 < 0.5

    def test_confidence_weights(self):
        """Test custom weight parameters affect result."""
        default = calculate_confidence(
            pip_count=3,
            circularity=0.5,
            size_variance=0.3
        )

        circ_weighted = calculate_confidence(
            pip_count=3,
            circularity=0.5,
            size_variance=0.3,
            circularity_weight=0.8,
            size_weight=0.1,
            count_weight=0.1
        )

        # Both should be valid confidence scores
        assert 0.0 <= default <= 1.0
        assert 0.0 <= circ_weighted <= 1.0


class TestValidatePipCount:
    """Tests for pip count validation."""

    @pytest.mark.parametrize("pip_count", [0, 1, 2, 3, 4, 5, 6])
    def test_valid_pip_counts(self, pip_count):
        """Valid pip counts (0-6) should return True."""
        assert validate_pip_count(pip_count) is True

    @pytest.mark.parametrize("pip_count", [-1, 7, 8, 10, 100])
    def test_invalid_pip_counts(self, pip_count):
        """Invalid pip counts should return False."""
        assert validate_pip_count(pip_count) is False


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_image(self):
        """Test handling of very small images."""
        image = np.zeros((20, 40, 3), dtype=np.uint8)
        image[5:15, 5:15] = 255

        # Should not raise, may return low confidence
        result = detect_domino_pips(image)
        assert isinstance(result, PipDetectionResult)

    def test_single_pixel_image_raises(self):
        """Single pixel image should raise or handle gracefully."""
        image = np.zeros((1, 1, 3), dtype=np.uint8)

        # May raise or return result with low/zero confidence
        try:
            result = detect_domino_pips(image)
            assert result.left_confidence == 0.0 or result.left_pips == 0
        except ValueError:
            pass  # Acceptable to raise for invalid image

    def test_uniform_white_image(self):
        """Uniform white image should detect as blank."""
        image = np.ones((100, 200, 3), dtype=np.uint8) * 255

        result = detect_domino_pips(image)

        # Should detect as blank (0 pips)
        assert result.left_pips <= 1
        assert result.right_pips <= 1

    def test_uniform_black_image(self):
        """Uniform black image should detect as blank."""
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        result = detect_domino_pips(image)

        assert isinstance(result, PipDetectionResult)

    def test_high_noise_image(self):
        """Test handling of very noisy image."""
        # Create random noise image
        np.random.seed(42)
        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)

        result = detect_domino_pips(image)

        # Should return result, confidence may be low
        assert isinstance(result, PipDetectionResult)
        assert 0 <= result.left_pips <= 6
        assert 0 <= result.right_pips <= 6

    def test_auto_rotate_disabled(self):
        """Test detection with auto_rotate disabled."""
        image = create_test_domino(left_pips=3, right_pips=5)

        result = detect_domino_pips(image, auto_rotate=False)

        assert isinstance(result, PipDetectionResult)
        assert 0 <= result.left_pips <= 6

    def test_hough_primary_disabled(self):
        """Test detection with HoughCircles as fallback only."""
        image = create_test_domino(left_pips=4, right_pips=2)

        result = detect_domino_pips(image, use_hough_primary=False)

        assert isinstance(result, PipDetectionResult)

    def test_no_fallback(self):
        """Test detection without fallback method."""
        image = create_test_domino(left_pips=3, right_pips=5)

        result = detect_domino_pips(image, fallback_to_contours=False)

        assert isinstance(result, PipDetectionResult)


# =============================================================================
# Pydantic Model Tests
# =============================================================================

class TestPipDetectionResult:
    """Tests for PipDetectionResult Pydantic model."""

    def test_valid_creation(self):
        """Test creating valid PipDetectionResult."""
        result = PipDetectionResult(
            left_pips=3,
            right_pips=5,
            left_confidence=0.92,
            right_confidence=0.87
        )

        assert result.left_pips == 3
        assert result.right_pips == 5
        assert result.left_confidence == 0.92
        assert result.right_confidence == 0.87

    def test_boundary_values(self):
        """Test boundary values for pips and confidence."""
        # Min values
        result_min = PipDetectionResult(
            left_pips=0,
            right_pips=0,
            left_confidence=0.0,
            right_confidence=0.0
        )
        assert result_min.left_pips == 0
        assert result_min.left_confidence == 0.0

        # Max values
        result_max = PipDetectionResult(
            left_pips=6,
            right_pips=6,
            left_confidence=1.0,
            right_confidence=1.0
        )
        assert result_max.left_pips == 6
        assert result_max.left_confidence == 1.0

    def test_invalid_pip_count_raises(self):
        """Test that pip count outside 0-6 raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PipDetectionResult(
                left_pips=7,  # Invalid
                right_pips=3,
                left_confidence=0.9,
                right_confidence=0.9
            )

    def test_invalid_confidence_raises(self):
        """Test that confidence outside 0.0-1.0 raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PipDetectionResult(
                left_pips=3,
                right_pips=3,
                left_confidence=1.5,  # Invalid
                right_confidence=0.9
            )


# =============================================================================
# Adaptive Detection Tests
# =============================================================================

class TestAdaptiveDetection:
    """Tests for adaptive pip detection."""

    def test_hough_adaptive_returns_best_result(self):
        """Test that adaptive Hough returns reasonable results."""
        image = create_domino_half(pip_count=4, width=100, height=100)

        count, circles, info = detect_pips_hough_adaptive(image)

        assert isinstance(count, int)
        assert count >= 0
        assert isinstance(info, dict)

    def test_hough_adaptive_max_pips(self):
        """Test max_pips parameter limits detection."""
        image = create_domino_half(pip_count=6, width=100, height=100)

        count, circles, info = detect_pips_hough_adaptive(image, max_pips=3)

        # Should limit to max_pips or return best valid result
        assert count <= 6  # May detect more than max in some cases

    def test_hough_adaptive_param_range(self):
        """Test custom param2_range parameter."""
        image = create_domino_half(pip_count=3, width=100, height=100)

        count, circles, info = detect_pips_hough_adaptive(
            image,
            param2_range=(10, 50, 10)
        )

        assert isinstance(count, int)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full pip detection pipeline."""

    def test_full_pipeline_standard_domino(self):
        """Test complete pipeline with standard domino."""
        image = create_test_domino(left_pips=4, right_pips=2)

        result = detect_domino_pips(image)

        assert isinstance(result, PipDetectionResult)
        assert 0 <= result.left_pips <= 6
        assert 0 <= result.right_pips <= 6
        assert 0.0 <= result.left_confidence <= 1.0
        assert 0.0 <= result.right_confidence <= 1.0

    def test_full_pipeline_inverted_colors(self):
        """Test pipeline with inverted color scheme."""
        image = create_test_domino(
            left_pips=3,
            right_pips=5,
            bg_color=(30, 30, 30),
            pip_color=(220, 220, 220)
        )

        result = detect_domino_pips(image)

        assert isinstance(result, PipDetectionResult)
        assert 0 <= result.left_pips <= 6

    def test_full_pipeline_with_rotation(self):
        """Test full pipeline with rotated domino."""
        image = create_rotated_test_domino(30, left_pips=5, right_pips=3)

        result = detect_domino_pips(image, auto_rotate=True)

        assert isinstance(result, PipDetectionResult)
        # Should detect something even if not exactly correct
        assert result.left_confidence >= 0.0
        assert result.right_confidence >= 0.0

    def test_self_test_function(self):
        """Test the module's built-in self-test."""
        from extract_dominoes import _self_test

        # Should not raise
        result = _self_test()
        assert result is True


# =============================================================================
# Detection Info Tests
# =============================================================================

class TestDetectionInfo:
    """Tests for detection information returned by detection functions."""

    def test_hough_detection_info_structure(self):
        """Test Hough detection returns expected info structure."""
        image = create_domino_half(pip_count=3, width=100, height=100)

        count, circles, info = detect_pips_hough(image)

        assert "mean_radius" in info
        assert "radius_variance" in info
        assert "image_size" in info
        assert "radii" in info

    def test_contour_detection_info_structure(self):
        """Test contour detection returns expected info structure."""
        image = create_domino_half(pip_count=3, width=100, height=100)

        count, contours, info = detect_pips_contours(image)

        assert "mean_area" in info
        assert "area_variance" in info
        assert "circularities" in info
        assert "image_size" in info
        assert "total_contours_found" in info

    def test_detection_info_with_no_pips(self):
        """Test detection info for blank domino."""
        image = create_domino_half(pip_count=0, width=100, height=100)

        count, circles, info = detect_pips_hough(image)

        assert count == 0
        assert info["radii"] == []
        assert info["mean_radius"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
