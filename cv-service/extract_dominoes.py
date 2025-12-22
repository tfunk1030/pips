"""
Domino pip detection module with preprocessing pipeline.

This module provides functions for preprocessing domino images
and detecting pips (dots) on domino tiles using OpenCV.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import os


# Debug output directory - set via environment variable
DEBUG_OUTPUT_DIR = os.environ.get('DEBUG_OUTPUT_DIR', None)


def save_debug_image(name: str, image: np.ndarray) -> None:
    """Save image to debug directory if DEBUG_OUTPUT_DIR is set."""
    if DEBUG_OUTPUT_DIR is not None:
        os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
        path = os.path.join(DEBUG_OUTPUT_DIR, name)
        cv2.imwrite(path, image)


def preprocess_domino_image(
    image: np.ndarray,
    block_size: int = 11,
    c_value: int = 2,
    morph_kernel_size: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess a domino image for pip detection.

    Applies grayscale conversion, adaptive thresholding, and morphological
    operations to prepare the image for pip detection.

    Args:
        image: Input BGR image of a domino tile.
        block_size: Block size for adaptive thresholding (must be odd).
        c_value: Constant subtracted from mean in adaptive thresholding.
        morph_kernel_size: Size of morphological kernel for noise reduction.

    Returns:
        Tuple of (grayscale, binary_threshold, cleaned) images.
        - grayscale: Grayscale version of input
        - binary_threshold: Adaptive threshold result
        - cleaned: Morphologically cleaned binary image
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")

    # Handle grayscale or BGR input
    if len(image.shape) == 2:
        gray = image.copy()
    elif len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    save_debug_image("01_grayscale.png", gray)

    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    save_debug_image("02_filtered.png", filtered)

    # Apply adaptive thresholding - handles lighting variations
    # THRESH_BINARY_INV makes pips white on black background
    binary = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_value
    )
    save_debug_image("03_binary_threshold.png", binary)

    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (morph_kernel_size, morph_kernel_size)
    )

    # Close operation to fill small gaps in pips
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Open operation to remove small noise specks
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    save_debug_image("04_cleaned.png", cleaned)

    return gray, binary, cleaned


def preprocess_for_hough(
    image: np.ndarray,
    blur_size: int = 5
) -> np.ndarray:
    """
    Preprocess image specifically for HoughCircles detection.

    HoughCircles works best on a blurred grayscale image with
    good edge definition.

    Args:
        image: Input BGR or grayscale image.
        blur_size: Size of Gaussian blur kernel (must be odd).

    Returns:
        Preprocessed grayscale image ready for HoughCircles.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")

    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    save_debug_image("05_hough_preprocessed.png", blurred)

    return blurred


def preprocess_for_contours(
    image: np.ndarray,
    block_size: int = 11,
    c_value: int = 5,
    invert: bool = True
) -> np.ndarray:
    """
    Preprocess image specifically for contour-based pip detection.

    Creates a clean binary image where pips are white blobs
    suitable for findContours.

    Args:
        image: Input BGR or grayscale image.
        block_size: Block size for adaptive thresholding.
        c_value: Constant for adaptive thresholding.
        invert: If True, pips become white on black background.

    Returns:
        Binary image ready for contour detection.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")

    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply bilateral filter for edge-preserving smoothing
    filtered = cv2.bilateralFilter(gray, 9, 50, 50)

    # Adaptive threshold
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    binary = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresh_type,
        block_size,
        c_value
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    save_debug_image("06_contour_preprocessed.png", cleaned)

    return cleaned


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Enhances contrast in images with uneven lighting.

    Args:
        image: Input BGR or grayscale image.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        Contrast-enhanced grayscale image.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")

    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Create CLAHE object and apply
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)

    save_debug_image("07_clahe_enhanced.png", enhanced)

    return enhanced


def detect_domino_color_mode(image: np.ndarray) -> str:
    """
    Detect whether pips are light-on-dark or dark-on-light.

    Analyzes the image to determine the color scheme for
    proper threshold direction.

    Args:
        image: Input BGR or grayscale domino image.

    Returns:
        "light_pips" if pips are lighter than background,
        "dark_pips" if pips are darker than background.
    """
    if image is None or image.size == 0:
        return "dark_pips"  # default assumption

    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Sample center region (where pips typically are)
    h, w = gray.shape[:2]
    center_region = gray[h//4:3*h//4, w//4:3*w//4]

    # Calculate statistics
    mean_val = np.mean(gray)
    center_mean = np.mean(center_region)

    # If center is darker than overall, pips are likely dark on light
    if center_mean < mean_val - 10:
        return "dark_pips"
    else:
        return "light_pips"


def preprocess_adaptive(
    image: np.ndarray,
    auto_detect_mode: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Adaptive preprocessing pipeline that adjusts to image characteristics.

    Automatically detects color mode and applies appropriate preprocessing.

    Args:
        image: Input BGR domino image.
        auto_detect_mode: If True, automatically detect pip color mode.

    Returns:
        Tuple of (preprocessed_image, preprocessing_info).
        preprocessing_info contains details about applied transformations.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")

    info = {
        "color_mode": None,
        "clahe_applied": False,
        "block_size": 11,
        "c_value": 2
    }

    # Detect color mode
    if auto_detect_mode:
        color_mode = detect_domino_color_mode(image)
        info["color_mode"] = color_mode
    else:
        color_mode = "dark_pips"
        info["color_mode"] = color_mode

    # Convert to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Check if CLAHE would help (low contrast image)
    img_std = np.std(gray)
    if img_std < 40:  # Low contrast
        gray = apply_clahe(gray)
        info["clahe_applied"] = True

    # Apply bilateral filter
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Determine threshold direction
    invert = (color_mode == "dark_pips")
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresh_type,
        info["block_size"],
        info["c_value"]
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    save_debug_image("08_adaptive_preprocessed.png", cleaned)

    return cleaned, info


# Module-level test function
def _self_test():
    """Quick self-test to verify module imports correctly."""
    # Create a simple test image
    test_img = np.zeros((100, 200, 3), dtype=np.uint8)
    test_img[40:60, 40:60] = 255  # White square

    # Test preprocessing
    gray, binary, cleaned = preprocess_domino_image(test_img)
    assert gray is not None
    assert binary is not None
    assert cleaned is not None

    # Test Hough preprocessing
    hough_ready = preprocess_for_hough(test_img)
    assert hough_ready is not None

    # Test contour preprocessing
    contour_ready = preprocess_for_contours(test_img)
    assert contour_ready is not None

    return True


if __name__ == "__main__":
    if _self_test():
        print("Self-test passed!")
    else:
        print("Self-test failed!")
