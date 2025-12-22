"""
Test suite for OCR confidence thresholds and classification.

Tests verify that OCR-specific confidence thresholds are correctly calibrated:
- High confidence threshold: 0.90 (OCR needs higher bar)
- Medium confidence threshold: 0.75
"""

import pytest
from tools.ocr_constraints import (
    OCR_CONFIDENCE_THRESHOLDS,
    ConfidenceLevel,
    get_ocr_confidence_level,
)


class TestOCRThresholds:
    """Test OCR confidence threshold values."""

    def test_ocr_thresholds_exist(self):
        """Test that OCR thresholds dictionary exists and has required keys."""
        assert OCR_CONFIDENCE_THRESHOLDS is not None
        assert "high" in OCR_CONFIDENCE_THRESHOLDS
        assert "medium" in OCR_CONFIDENCE_THRESHOLDS
        assert "low" in OCR_CONFIDENCE_THRESHOLDS

    def test_ocr_high_threshold(self):
        """Test that OCR high threshold is 0.90 (stricter than general detection)."""
        assert OCR_CONFIDENCE_THRESHOLDS["high"] == 0.90

    def test_ocr_medium_threshold(self):
        """Test that OCR medium threshold is 0.75."""
        assert OCR_CONFIDENCE_THRESHOLDS["medium"] == 0.75

    def test_ocr_low_threshold(self):
        """Test that OCR low threshold is 0.0."""
        assert OCR_CONFIDENCE_THRESHOLDS["low"] == 0.0

    def test_thresholds_are_ordered(self):
        """Test that thresholds are in descending order: high > medium > low."""
        high = OCR_CONFIDENCE_THRESHOLDS["high"]
        medium = OCR_CONFIDENCE_THRESHOLDS["medium"]
        low = OCR_CONFIDENCE_THRESHOLDS["low"]
        assert high > medium > low

    def test_thresholds_in_valid_range(self):
        """Test that all thresholds are in [0.0, 1.0] range."""
        for key, value in OCR_CONFIDENCE_THRESHOLDS.items():
            assert 0.0 <= value <= 1.0, f"Threshold {key} out of range: {value}"


class TestOCRConfidenceLevel:
    """Test OCR confidence level classification function."""

    def test_high_confidence_at_threshold(self):
        """Test that exactly 0.90 is classified as high."""
        assert get_ocr_confidence_level(0.90) == "high"

    def test_high_confidence_above_threshold(self):
        """Test that values above 0.90 are classified as high."""
        assert get_ocr_confidence_level(0.95) == "high"
        assert get_ocr_confidence_level(1.0) == "high"

    def test_medium_confidence_at_threshold(self):
        """Test that exactly 0.75 is classified as medium."""
        assert get_ocr_confidence_level(0.75) == "medium"

    def test_medium_confidence_range(self):
        """Test that values between 0.75 and 0.90 are classified as medium."""
        assert get_ocr_confidence_level(0.80) == "medium"
        assert get_ocr_confidence_level(0.85) == "medium"
        assert get_ocr_confidence_level(0.89) == "medium"

    def test_low_confidence_below_medium(self):
        """Test that values below 0.75 are classified as low."""
        assert get_ocr_confidence_level(0.74) == "low"
        assert get_ocr_confidence_level(0.50) == "low"
        assert get_ocr_confidence_level(0.0) == "low"

    def test_boundary_just_below_high(self):
        """Test boundary case just below high threshold."""
        assert get_ocr_confidence_level(0.8999) == "medium"

    def test_boundary_just_below_medium(self):
        """Test boundary case just below medium threshold."""
        assert get_ocr_confidence_level(0.7499) == "low"

    def test_return_type(self):
        """Test that return type is valid ConfidenceLevel."""
        result = get_ocr_confidence_level(0.85)
        assert result in ["high", "medium", "low"]


class TestOCRThresholdsVsSpec:
    """Test that OCR thresholds match the specification requirements."""

    def test_ocr_higher_than_general_high(self):
        """
        Test that OCR high threshold (0.90) is higher than general detection (0.85).
        OCR needs higher bar due to error modes (misread digits).
        """
        # OCR high threshold should be 0.90 per spec
        assert OCR_CONFIDENCE_THRESHOLDS["high"] >= 0.90

    def test_ocr_higher_than_general_medium(self):
        """
        Test that OCR medium threshold (0.75) is higher than general detection (0.70).
        """
        # OCR medium threshold should be 0.75 per spec
        assert OCR_CONFIDENCE_THRESHOLDS["medium"] >= 0.75

    def test_threshold_gap(self):
        """
        Test that there's reasonable gap between high and medium thresholds.
        This ensures meaningful differentiation between confidence levels.
        """
        high = OCR_CONFIDENCE_THRESHOLDS["high"]
        medium = OCR_CONFIDENCE_THRESHOLDS["medium"]
        gap = high - medium
        # Gap should be at least 10% (0.10)
        assert gap >= 0.10, f"Insufficient gap between thresholds: {gap}"


# Standalone test function for verification command compatibility
def test_ocr_thresholds():
    """
    Verification test for OCR confidence thresholds.

    This test validates that OCR thresholds are correctly calibrated:
    - High: 0.90 (OCR needs higher bar due to misread digit errors)
    - Medium: 0.75
    - Low: 0.0
    """
    # Test threshold values
    assert OCR_CONFIDENCE_THRESHOLDS["high"] == 0.90, "OCR high threshold should be 0.90"
    assert OCR_CONFIDENCE_THRESHOLDS["medium"] == 0.75, "OCR medium threshold should be 0.75"
    assert OCR_CONFIDENCE_THRESHOLDS["low"] == 0.0, "OCR low threshold should be 0.0"

    # Test threshold ordering
    assert OCR_CONFIDENCE_THRESHOLDS["high"] > OCR_CONFIDENCE_THRESHOLDS["medium"] > OCR_CONFIDENCE_THRESHOLDS["low"]

    # Test classification function
    assert get_ocr_confidence_level(0.95) == "high"
    assert get_ocr_confidence_level(0.90) == "high"
    assert get_ocr_confidence_level(0.80) == "medium"
    assert get_ocr_confidence_level(0.75) == "medium"
    assert get_ocr_confidence_level(0.50) == "low"
    assert get_ocr_confidence_level(0.0) == "low"


# Run tests with: cd pips-agent && python -m pytest test_ocr_confidence.py -v
