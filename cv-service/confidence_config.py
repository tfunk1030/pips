"""
Centralized Confidence Configuration Module

This module defines component-specific confidence thresholds for all detection
components in the CV service. Each component has calibrated thresholds based
on actual accuracy measurements.

Threshold meanings:
- high: User can trust detection without manual review (>90% actual accuracy)
- medium: Suggest review, detection is likely correct but may need verification
- low: Requires manual verification, confidence too low to trust automatically

Calibration methodology:
- Thresholds are set based on correlation between reported confidence and
  actual detection accuracy in validation datasets
- Target: confidence scores correlate with actual accuracy within +/-10%
"""

from typing import Literal, Dict

# Type alias for confidence level
ConfidenceLevel = Literal["high", "medium", "low"]


# Component-specific confidence thresholds
# Each component has independently calibrated thresholds based on accuracy measurement
CONFIDENCE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    # Geometry extraction: grid detection, cell positioning
    # Based on grid regularity analysis and cell detection accuracy
    "geometry_extraction": {
        "high": 0.85,   # User can trust without review
        "medium": 0.70, # Suggest review
        "low": 0.0      # Requires manual verification
    },

    # OCR detection: text recognition from domino pips
    # OCR needs higher threshold due to error modes (misread digits)
    "ocr_detection": {
        "high": 0.90,   # OCR needs higher bar
        "medium": 0.75, # Suggest review
        "low": 0.0      # Requires manual verification
    },

    # Puzzle detection: finding puzzle region in screenshot
    # Based on saturation mask and contour detection reliability
    "puzzle_detection": {
        "high": 0.80,   # User can trust without review
        "medium": 0.65, # Suggest review
        "low": 0.0      # Requires manual verification
    },

    # Domino detection: finding and cropping domino tray region
    # Based on edge detection and region analysis reliability
    "domino_detection": {
        "high": 0.80,   # User can trust without review
        "medium": 0.65, # Suggest review
        "low": 0.0      # Requires manual verification
    },
}


# Threshold boundary margin for "borderline" detection
# When confidence is within this margin of a threshold, flag as borderline
BORDERLINE_MARGIN = 0.05  # 5%


def get_confidence_level(
    confidence: float,
    component: str = "geometry_extraction"
) -> ConfidenceLevel:
    """
    Get categorical confidence level for a numeric confidence score.

    Args:
        confidence: Numeric confidence score (0.0 to 1.0)
        component: Detection component name (must be key in CONFIDENCE_THRESHOLDS)

    Returns:
        Confidence level: "high", "medium", or "low"

    Raises:
        ValueError: If component is not in CONFIDENCE_THRESHOLDS
    """
    if component not in CONFIDENCE_THRESHOLDS:
        raise ValueError(
            f"Unknown component: {component}. "
            f"Valid components: {list(CONFIDENCE_THRESHOLDS.keys())}"
        )

    thresholds = CONFIDENCE_THRESHOLDS[component]

    if confidence >= thresholds["high"]:
        return "high"
    elif confidence >= thresholds["medium"]:
        return "medium"
    else:
        return "low"


def is_borderline(
    confidence: float,
    component: str = "geometry_extraction"
) -> bool:
    """
    Check if confidence score is near a threshold boundary.

    When a score is within BORDERLINE_MARGIN of a threshold, users should
    be informed that the confidence is borderline and may need extra review.

    Args:
        confidence: Numeric confidence score (0.0 to 1.0)
        component: Detection component name

    Returns:
        True if confidence is within BORDERLINE_MARGIN of a threshold boundary
    """
    if component not in CONFIDENCE_THRESHOLDS:
        return False

    thresholds = CONFIDENCE_THRESHOLDS[component]

    # Check if near high threshold
    high_threshold = thresholds["high"]
    if abs(confidence - high_threshold) <= BORDERLINE_MARGIN:
        return True

    # Check if near medium threshold
    medium_threshold = thresholds["medium"]
    if abs(confidence - medium_threshold) <= BORDERLINE_MARGIN:
        return True

    return False


def get_threshold_values(component: str) -> Dict[str, float]:
    """
    Get threshold values for a specific component.

    Args:
        component: Detection component name

    Returns:
        Dictionary with "high", "medium", "low" threshold values

    Raises:
        ValueError: If component is not in CONFIDENCE_THRESHOLDS
    """
    if component not in CONFIDENCE_THRESHOLDS:
        raise ValueError(
            f"Unknown component: {component}. "
            f"Valid components: {list(CONFIDENCE_THRESHOLDS.keys())}"
        )

    return CONFIDENCE_THRESHOLDS[component].copy()


def get_all_components() -> list:
    """
    Get list of all registered detection components.

    Returns:
        List of component names
    """
    return list(CONFIDENCE_THRESHOLDS.keys())
