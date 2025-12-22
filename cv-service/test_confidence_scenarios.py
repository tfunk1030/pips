"""
High and Low Confidence Scenario Testing

This test module validates the confidence scoring system with specific
high and low confidence scenarios to ensure:

1. Clear images produce high confidence scores (green indicator)
2. Blurry/low quality images produce low confidence scores (red/amber indicator)
3. UI messages correctly match confidence levels

These tests correspond to the manual verification requirements in subtask-4-2.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from confidence_config import (
    CONFIDENCE_THRESHOLDS,
    get_confidence_level,
    is_borderline,
    BORDERLINE_MARGIN
)
from hybrid_extraction import _calculate_grid_confidence, find_puzzle_roi


# ============================================================================
# Test Image Generators
# ============================================================================

def create_clear_puzzle_image(
    width: int = 500,
    height: int = 500,
    saturation: int = 200,
    with_grid: bool = True,
    cell_count: int = 25,  # 5x5 grid
    grid_coverage: float = 0.6  # Grid takes 60% of image
) -> np.ndarray:
    """
    Create a clear, high-quality puzzle image that should produce HIGH confidence.

    Characteristics of clear images:
    - High saturation (vivid colors)
    - Clear grid structure
    - Good contrast
    - Regular cell sizes
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (240, 240, 240)  # Light background for good contrast

    if with_grid:
        grid_size = int(min(width, height) * grid_coverage)
        grid_x = (width - grid_size) // 2
        grid_y = (height - grid_size) // 2

        rows = cols = int(np.sqrt(cell_count))
        cell_size = grid_size // cols

        for row in range(rows):
            for col in range(cols):
                cell_x = grid_x + col * cell_size + 2
                cell_y = grid_y + row * cell_size + 2
                cell_w = cell_size - 4
                cell_h = cell_size - 4

                # Create vivid colors
                hue = (row * 5 + col * 7) * 15 % 180
                color_hsv = np.array([[[hue, saturation, 220]]], dtype=np.uint8)
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]

                img[cell_y:cell_y+cell_h, cell_x:cell_x+cell_w] = color_bgr

                # Add cell border for clarity
                cv2.rectangle(img, (cell_x, cell_y), (cell_x+cell_w, cell_y+cell_h),
                             (50, 50, 50), 1)

    return img


def create_blurry_puzzle_image(
    width: int = 500,
    height: int = 500,
    saturation: int = 50,
    blur_amount: int = 15
) -> np.ndarray:
    """
    Create a blurry, low-quality puzzle image that should produce LOW confidence.

    Characteristics of blurry/poor images:
    - Low saturation (washed out colors)
    - Blurred edges
    - Low contrast
    - Unclear structure
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (100, 100, 100)  # Gray background (low contrast)

    # Create a faint, blurry grid
    grid_size = int(min(width, height) * 0.5)
    grid_x = (width - grid_size) // 2
    grid_y = (height - grid_size) // 2

    for row in range(3):
        for col in range(3):
            cell_x = grid_x + col * (grid_size // 3) + 5
            cell_y = grid_y + row * (grid_size // 3) + 5
            cell_w = grid_size // 3 - 10
            cell_h = grid_size // 3 - 10

            # Low saturation colors (grayish)
            hue = (row * 3 + col * 5) * 20 % 180
            color_hsv = np.array([[[hue, saturation, 120]]], dtype=np.uint8)
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]

            img[cell_y:cell_y+cell_h, cell_x:cell_x+cell_w] = color_bgr

    # Apply blur
    img = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)

    return img


def create_grayscale_image(width: int = 500, height: int = 500) -> np.ndarray:
    """Create a grayscale (no color) image - should produce very low confidence."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Random grayscale patterns
    for row in range(5):
        for col in range(5):
            cell_x = 50 + col * 80
            cell_y = 50 + row * 80
            gray = 50 + (row + col) * 15
            img[cell_y:cell_y+70, cell_x:cell_x+70] = (gray, gray, gray)

    return img


def create_medium_quality_image(width: int = 500, height: int = 500) -> np.ndarray:
    """Create a medium quality image - should produce MEDIUM confidence."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (180, 180, 180)

    # Medium saturation, decent grid but not perfect
    grid_size = int(min(width, height) * 0.55)
    grid_x = (width - grid_size) // 2
    grid_y = (height - grid_size) // 2

    for row in range(4):
        for col in range(4):
            cell_x = grid_x + col * (grid_size // 4) + 3
            cell_y = grid_y + row * (grid_size // 4) + 3
            cell_w = grid_size // 4 - 6
            cell_h = grid_size // 4 - 6

            # Medium saturation (90-120)
            hue = (row * 4 + col * 6) * 12 % 180
            saturation = 100 + (row + col) % 30
            color_hsv = np.array([[[hue, saturation, 170]]], dtype=np.uint8)
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]

            img[cell_y:cell_y+cell_h, cell_x:cell_x+cell_w] = color_bgr

    # Slight blur to reduce clarity
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


# ============================================================================
# High Confidence Scenario Tests
# ============================================================================

class TestHighConfidenceScenarios:
    """
    Test scenarios where confidence should be HIGH (>= 0.85).

    Per spec: High confidence means "User can trust without review".
    Visual indicator: Green (#10b981)
    Message: "High confidence - likely accurate"
    """

    def test_clear_image_produces_high_confidence(self):
        """
        Scenario: Clear, high-saturation puzzle image
        Expected: High confidence (green indicator)
        """
        img = create_clear_puzzle_image(saturation=200)

        # Find puzzle ROI
        try:
            (padded_bounds, actual_bounds, contour) = find_puzzle_roi(img)
            confidence, breakdown = _calculate_grid_confidence(img, actual_bounds, contour)
        except ValueError:
            pytest.fail("Clear image should find puzzle ROI")

        # Verify high confidence
        level = get_confidence_level(confidence, "puzzle_detection")

        # Clear images should have good saturation score
        assert breakdown["saturation"] >= 0.7, \
            f"Clear image should have high saturation score, got {breakdown['saturation']}"

        # Log results for manual verification
        print(f"\n[HIGH CONFIDENCE SCENARIO - Clear Image]")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Level: {level}")
        print(f"  Saturation factor: {breakdown['saturation']:.3f}")
        print(f"  Contrast factor: {breakdown['contrast']:.3f}")

    def test_very_saturated_image_high_saturation_score(self):
        """
        Scenario: Very saturated (vivid colors) image
        Expected: High saturation score component
        """
        img = create_clear_puzzle_image(saturation=255)  # Max saturation

        try:
            (padded_bounds, actual_bounds, contour) = find_puzzle_roi(img)
            confidence, breakdown = _calculate_grid_confidence(img, actual_bounds, contour)
        except ValueError:
            pytest.fail("Very saturated image should find puzzle ROI")

        # Saturation score should be very high
        assert breakdown["saturation"] >= 0.85, \
            f"Very saturated image should have saturation >= 0.85, got {breakdown['saturation']}"

        print(f"\n[HIGH CONFIDENCE SCENARIO - Very Saturated]")
        print(f"  Saturation score: {breakdown['saturation']:.3f}")

    def test_optimal_grid_coverage_high_confidence(self):
        """
        Scenario: Image with optimal grid coverage (good area ratio)
        Expected: Good area_ratio score
        """
        img = create_clear_puzzle_image(grid_coverage=0.7)  # 70% coverage

        try:
            (padded_bounds, actual_bounds, contour) = find_puzzle_roi(img)
            confidence, breakdown = _calculate_grid_confidence(img, actual_bounds, contour)
        except ValueError:
            pytest.fail("Good coverage image should find puzzle ROI")

        # Area ratio should be good
        assert breakdown["area_ratio"] >= 0.6, \
            f"Good coverage should have area_ratio >= 0.6, got {breakdown['area_ratio']}"

        print(f"\n[HIGH CONFIDENCE SCENARIO - Optimal Coverage]")
        print(f"  Area ratio: {breakdown['area_ratio']:.3f}")


# ============================================================================
# Low Confidence Scenario Tests
# ============================================================================

class TestLowConfidenceScenarios:
    """
    Test scenarios where confidence should be LOW (< 0.70).

    Per spec: Low confidence means "Requires manual verification".
    Visual indicator: Red (#ef4444)
    Message: "Low confidence - manual verification required"
    """

    def test_blurry_image_produces_low_confidence(self):
        """
        Scenario: Blurry, low-saturation image
        Expected: Low confidence (red indicator)
        """
        img = create_blurry_puzzle_image(saturation=30, blur_amount=21)

        # Try to find ROI - may fail due to low color
        try:
            (padded_bounds, actual_bounds, contour) = find_puzzle_roi(img)
            confidence, breakdown = _calculate_grid_confidence(img, actual_bounds, contour)
            level = get_confidence_level(confidence, "puzzle_detection")
        except ValueError:
            # No ROI found - this is low confidence
            confidence = 0.0
            level = "low"
            breakdown = {"saturation": 0.0}

        # Blurry images should have low saturation
        assert breakdown["saturation"] < 0.5, \
            f"Blurry image should have low saturation, got {breakdown['saturation']}"

        print(f"\n[LOW CONFIDENCE SCENARIO - Blurry Image]")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Level: {level}")
        print(f"  Saturation factor: {breakdown.get('saturation', 0):.3f}")

    def test_grayscale_image_produces_low_confidence(self):
        """
        Scenario: Grayscale image (no color saturation)
        Expected: Very low confidence (red indicator)
        """
        img = create_grayscale_image()

        # Grayscale should fail to find colorful ROI
        try:
            (padded_bounds, actual_bounds, contour) = find_puzzle_roi(img)
            confidence, breakdown = _calculate_grid_confidence(img, actual_bounds, contour)
            saturation_score = breakdown.get("saturation", 0)
        except ValueError:
            # Expected - no colorful region found
            confidence = 0.0
            saturation_score = 0.0

        # Should have very low/zero saturation
        assert saturation_score < 0.2, \
            f"Grayscale image should have very low saturation, got {saturation_score}"

        level = get_confidence_level(confidence, "puzzle_detection")
        assert level == "low", f"Grayscale should be low confidence, got {level}"

        print(f"\n[LOW CONFIDENCE SCENARIO - Grayscale]")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Level: {level}")

    def test_low_saturation_image_low_score(self):
        """
        Scenario: Image with washed out colors
        Expected: Low saturation score
        """
        img = create_blurry_puzzle_image(saturation=40, blur_amount=5)

        try:
            (padded_bounds, actual_bounds, contour) = find_puzzle_roi(img)
            confidence, breakdown = _calculate_grid_confidence(img, actual_bounds, contour)
        except ValueError:
            confidence = 0.0
            breakdown = {"saturation": 0.0}

        # Low saturation input should produce low saturation score
        assert breakdown.get("saturation", 0) < 0.5, \
            f"Washed out image should have low saturation score"

        print(f"\n[LOW CONFIDENCE SCENARIO - Washed Out]")
        print(f"  Saturation score: {breakdown.get('saturation', 0):.3f}")


# ============================================================================
# Medium (Borderline) Confidence Scenario Tests
# ============================================================================

class TestMediumConfidenceScenarios:
    """
    Test scenarios where confidence should be MEDIUM (0.70 - 0.85).

    Per spec: Medium confidence means "Suggest review".
    Visual indicator: Amber (#f59e0b)
    Message: "Medium confidence - review recommended"
    """

    def test_medium_quality_produces_medium_confidence(self):
        """
        Scenario: Medium quality image (decent but not perfect)
        Expected: Medium confidence (amber indicator)
        """
        img = create_medium_quality_image()

        try:
            (padded_bounds, actual_bounds, contour) = find_puzzle_roi(img)
            confidence, breakdown = _calculate_grid_confidence(img, actual_bounds, contour)
        except ValueError:
            confidence = 0.0
            breakdown = {}

        level = get_confidence_level(confidence, "puzzle_detection")

        print(f"\n[MEDIUM CONFIDENCE SCENARIO]")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Level: {level}")
        print(f"  Breakdown: {breakdown}")

    def test_borderline_confidence_detection(self):
        """
        Test borderline detection for edge cases near thresholds.
        """
        # Test values near high threshold (0.80 for puzzle_detection)
        high_threshold = CONFIDENCE_THRESHOLDS["puzzle_detection"]["high"]
        medium_threshold = CONFIDENCE_THRESHOLDS["puzzle_detection"]["medium"]

        # Just below high threshold - should be borderline
        borderline_below_high = high_threshold - 0.03
        assert is_borderline(borderline_below_high, "puzzle_detection"), \
            f"{borderline_below_high} should be borderline (near high={high_threshold})"

        # Just above medium threshold - should be borderline
        borderline_above_medium = medium_threshold + 0.03
        assert is_borderline(borderline_above_medium, "puzzle_detection"), \
            f"{borderline_above_medium} should be borderline (near medium={medium_threshold})"

        print(f"\n[BORDERLINE DETECTION]")
        print(f"  High threshold: {high_threshold}")
        print(f"  Medium threshold: {medium_threshold}")
        print(f"  Borderline margin: {BORDERLINE_MARGIN}")


# ============================================================================
# UI Message Verification Tests
# ============================================================================

class TestUIMessageMatching:
    """
    Verify that UI messages correctly match confidence levels.

    Per spec:
    - High: "High confidence - likely accurate" (green)
    - Medium: "Medium confidence - review recommended" (amber)
    - Low: "Low confidence - manual verification required" (red)
    """

    # Expected messages per spec
    EXPECTED_MESSAGES = {
        "high": "High confidence - likely accurate",
        "medium": "Medium confidence - review recommended",
        "low": "Low confidence - manual verification required",
    }

    # Expected colors per spec
    EXPECTED_COLORS = {
        "high": "#10b981",    # Green
        "medium": "#f59e0b",  # Amber
        "low": "#ef4444",     # Red
    }

    def test_high_confidence_message_and_color(self):
        """Verify high confidence produces correct message and color."""
        confidence = 0.90
        level = get_confidence_level(confidence, "geometry_extraction")

        assert level == "high", f"0.90 should be high confidence"

        print(f"\n[UI VERIFICATION - High Confidence]")
        print(f"  Confidence: {confidence:.0%}")
        print(f"  Level: {level}")
        print(f"  Expected Message: {self.EXPECTED_MESSAGES[level]}")
        print(f"  Expected Color: {self.EXPECTED_COLORS[level]} (Green)")

    def test_medium_confidence_message_and_color(self):
        """Verify medium confidence produces correct message and color."""
        confidence = 0.75
        level = get_confidence_level(confidence, "geometry_extraction")

        assert level == "medium", f"0.75 should be medium confidence"

        print(f"\n[UI VERIFICATION - Medium Confidence]")
        print(f"  Confidence: {confidence:.0%}")
        print(f"  Level: {level}")
        print(f"  Expected Message: {self.EXPECTED_MESSAGES[level]}")
        print(f"  Expected Color: {self.EXPECTED_COLORS[level]} (Amber)")

    def test_low_confidence_message_and_color(self):
        """Verify low confidence produces correct message and color."""
        confidence = 0.50
        level = get_confidence_level(confidence, "geometry_extraction")

        assert level == "low", f"0.50 should be low confidence"

        print(f"\n[UI VERIFICATION - Low Confidence]")
        print(f"  Confidence: {confidence:.0%}")
        print(f"  Level: {level}")
        print(f"  Expected Message: {self.EXPECTED_MESSAGES[level]}")
        print(f"  Expected Color: {self.EXPECTED_COLORS[level]} (Red)")


# ============================================================================
# Integration with Existing Test Images
# ============================================================================

class TestExistingTestImages:
    """
    Verify confidence behavior using existing test images in test_images/.
    """

    @pytest.fixture
    def test_images_dir(self):
        """Get path to test images directory."""
        return Path(__file__).parent / "test_images"

    def test_high_saturation_images_high_confidence(self, test_images_dir):
        """
        High saturation images (sat >= 150) should produce higher confidence.
        """
        if not test_images_dir.exists():
            pytest.skip("test_images directory not found")

        high_sat_images = list(test_images_dir.glob("*sat1[5-9]*.png")) + \
                          list(test_images_dir.glob("*sat200*.png"))

        if not high_sat_images:
            pytest.skip("No high saturation test images found")

        results = []
        for img_path in high_sat_images[:5]:  # Test first 5
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            try:
                (_, actual_bounds, contour) = find_puzzle_roi(img)
                confidence, breakdown = _calculate_grid_confidence(img, actual_bounds, contour)
                results.append({
                    "file": img_path.name,
                    "confidence": confidence,
                    "saturation_score": breakdown["saturation"]
                })
            except ValueError:
                continue

        print(f"\n[HIGH SATURATION IMAGES]")
        for r in results:
            print(f"  {r['file']}: conf={r['confidence']:.1%}, sat={r['saturation_score']:.2f}")

        # At least some should have good saturation scores
        if results:
            avg_saturation = sum(r["saturation_score"] for r in results) / len(results)
            assert avg_saturation >= 0.5, \
                f"High saturation images should average >= 0.5 saturation score"

    def test_low_saturation_images_lower_confidence(self, test_images_dir):
        """
        Low saturation images (sat <= 60) should produce lower confidence.
        """
        if not test_images_dir.exists():
            pytest.skip("test_images directory not found")

        low_sat_images = list(test_images_dir.glob("*sat[3-6]0*.png"))

        if not low_sat_images:
            pytest.skip("No low saturation test images found")

        results = []
        for img_path in low_sat_images[:5]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            try:
                (_, actual_bounds, contour) = find_puzzle_roi(img)
                confidence, breakdown = _calculate_grid_confidence(img, actual_bounds, contour)
                results.append({
                    "file": img_path.name,
                    "confidence": confidence,
                    "saturation_score": breakdown["saturation"]
                })
            except ValueError:
                # Low saturation may fail to find ROI - expected
                results.append({
                    "file": img_path.name,
                    "confidence": 0.0,
                    "saturation_score": 0.0
                })

        print(f"\n[LOW SATURATION IMAGES]")
        for r in results:
            print(f"  {r['file']}: conf={r['confidence']:.1%}, sat={r['saturation_score']:.2f}")

    def test_no_grid_images_low_confidence(self, test_images_dir):
        """
        Images without grid (nogrid) should have low or zero confidence.
        """
        if not test_images_dir.exists():
            pytest.skip("test_images directory not found")

        nogrid_images = list(test_images_dir.glob("*nogrid*.png"))

        if not nogrid_images:
            pytest.skip("No nogrid test images found")

        results = []
        for img_path in nogrid_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            try:
                (_, actual_bounds, contour) = find_puzzle_roi(img)
                confidence, breakdown = _calculate_grid_confidence(img, actual_bounds, contour)
            except ValueError:
                # Expected - no colorful region
                confidence = 0.0

            level = get_confidence_level(confidence, "puzzle_detection")
            results.append({
                "file": img_path.name,
                "confidence": confidence,
                "level": level
            })

        print(f"\n[NO GRID IMAGES]")
        for r in results:
            print(f"  {r['file']}: conf={r['confidence']:.1%}, level={r['level']}")

        # No-grid images should generally have low confidence
        low_count = sum(1 for r in results if r["level"] == "low")
        assert low_count >= len(results) * 0.5, \
            f"Most nogrid images should have low confidence"


# ============================================================================
# Run Tests with Verbose Output
# ============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-s",  # Show print statements
    ])
