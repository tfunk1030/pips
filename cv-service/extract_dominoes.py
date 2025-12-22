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


def detect_pips_hough(
    image: np.ndarray,
    dp: float = 1.0,
    min_dist: Optional[int] = None,
    param1: int = 50,
    param2: int = 30,
    min_radius: int = 5,
    max_radius: int = 50
) -> Tuple[int, np.ndarray, dict]:
    """
    Detect pips on a domino image using Hough Circle Transform.

    Uses cv2.HoughCircles to detect circular pip shapes on a domino
    half image. This is the primary detection method for pip counting.

    Args:
        image: Input BGR or grayscale image of a domino half.
        dp: Inverse ratio of accumulator resolution to image resolution.
            dp=1 means same resolution, dp=2 means half resolution.
        min_dist: Minimum distance between detected circle centers.
            If None, defaults to 1/6 of the minimum image dimension.
        param1: Higher threshold for Canny edge detector (lower threshold
            is half of this value).
        param2: Accumulator threshold for circle centers at detection stage.
            Smaller values = more circles detected (including false positives).
        min_radius: Minimum circle radius to detect.
        max_radius: Maximum circle radius to detect.

    Returns:
        Tuple of (pip_count, circles, detection_info):
        - pip_count: Number of detected pips (circles)
        - circles: NumPy array of detected circles as (x, y, radius) or None
        - detection_info: Dict with detection metadata including:
            - radii: List of detected radii
            - mean_radius: Average radius of detected circles
            - radius_variance: Variance in radius sizes (for confidence)
            - image_size: Tuple of (height, width)
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")

    # Preprocess image for Hough detection
    preprocessed = preprocess_for_hough(image)

    # Calculate default min_dist based on image size
    h, w = preprocessed.shape[:2]
    if min_dist is None:
        # Default: circles should be at least 1/6 of min dimension apart
        min_dist = max(int(min(h, w) / 6), 10)

    # Scale radius parameters based on image size
    # For small domino images, we need to adjust radius ranges
    img_scale = min(h, w) / 100.0  # Normalize to 100px reference
    scaled_min_radius = max(int(min_radius * img_scale), 2)
    scaled_max_radius = min(int(max_radius * img_scale), min(h, w) // 3)

    # Ensure min_radius < max_radius
    if scaled_min_radius >= scaled_max_radius:
        scaled_min_radius = max(2, scaled_max_radius - 5)

    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(
        preprocessed,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=scaled_min_radius,
        maxRadius=scaled_max_radius
    )

    # Initialize detection info
    detection_info = {
        "radii": [],
        "mean_radius": 0.0,
        "radius_variance": 0.0,
        "image_size": (h, w),
        "scaled_min_radius": scaled_min_radius,
        "scaled_max_radius": scaled_max_radius,
        "min_dist": min_dist
    }

    # Process results
    if circles is None:
        save_debug_image("09_hough_no_circles.png", preprocessed)
        return 0, None, detection_info

    # Convert to integer coordinates
    circles = np.uint16(np.around(circles))
    detected_circles = circles[0, :]

    # Extract radii for statistics
    radii = [float(c[2]) for c in detected_circles]
    detection_info["radii"] = radii
    detection_info["mean_radius"] = float(np.mean(radii)) if radii else 0.0
    detection_info["radius_variance"] = float(np.var(radii)) if len(radii) > 1 else 0.0

    # Create debug visualization
    if DEBUG_OUTPUT_DIR is not None:
        debug_img = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        for circle in detected_circles:
            x, y, r = circle
            # Draw outer circle
            cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2)
            # Draw center point
            cv2.circle(debug_img, (x, y), 2, (0, 0, 255), 3)
        save_debug_image("09_hough_detected_circles.png", debug_img)

    pip_count = len(detected_circles)
    return pip_count, detected_circles, detection_info


def detect_pips_contours(
    image: np.ndarray,
    min_area: Optional[int] = None,
    max_area: Optional[int] = None,
    min_circularity: float = 0.5,
    block_size: int = 11,
    c_value: int = 5
) -> Tuple[int, list, dict]:
    """
    Detect pips on a domino image using contour-based analysis.

    This is a fallback detection method when HoughCircles fails to
    detect pips reliably. Uses adaptive thresholding and contour
    filtering based on area and circularity.

    Args:
        image: Input BGR or grayscale image of a domino half.
        min_area: Minimum contour area for valid pips. If None, scales
            automatically based on image size.
        max_area: Maximum contour area for valid pips. If None, scales
            automatically based on image size.
        min_circularity: Minimum circularity (4*pi*area/perimeter^2) for
            valid pips. Range 0-1, where 1 is a perfect circle.
        block_size: Block size for adaptive thresholding.
        c_value: Constant for adaptive thresholding.

    Returns:
        Tuple of (pip_count, contours, detection_info):
        - pip_count: Number of detected pips (valid contours)
        - contours: List of contour arrays for detected pips
        - detection_info: Dict with detection metadata including:
            - areas: List of contour areas
            - circularities: List of contour circularity values
            - mean_area: Average area of detected contours
            - area_variance: Variance in contour areas
            - image_size: Tuple of (height, width)
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")

    # Preprocess image for contour detection
    preprocessed = preprocess_for_contours(
        image,
        block_size=block_size,
        c_value=c_value,
        invert=True
    )

    # Get image dimensions
    h, w = preprocessed.shape[:2]

    # Calculate default area bounds based on image size
    # For a domino half, pips are typically 5-15% of the half width
    img_scale = (h * w) / 10000.0  # Normalize to 100x100 reference
    if min_area is None:
        min_area = max(int(10 * img_scale), 10)
    if max_area is None:
        max_area = min(int(500 * img_scale), h * w // 10)

    # Ensure min < max
    if min_area >= max_area:
        min_area = max(10, max_area - 50)

    # Find contours
    contours, _ = cv2.findContours(
        preprocessed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Initialize detection info
    detection_info = {
        "areas": [],
        "circularities": [],
        "mean_area": 0.0,
        "area_variance": 0.0,
        "image_size": (h, w),
        "min_area_threshold": min_area,
        "max_area_threshold": max_area,
        "min_circularity": min_circularity,
        "total_contours_found": len(contours)
    }

    # Filter contours by area and circularity
    valid_contours = []
    areas = []
    circularities = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Check area bounds
        if area < min_area or area > max_area:
            continue

        # Calculate circularity: 4*pi*area/perimeter^2
        # Perfect circle has circularity of 1.0
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Check circularity threshold
        if circularity < min_circularity:
            continue

        # Valid pip contour
        valid_contours.append(contour)
        areas.append(float(area))
        circularities.append(float(circularity))

    # Update detection info
    detection_info["areas"] = areas
    detection_info["circularities"] = circularities

    if areas:
        detection_info["mean_area"] = float(np.mean(areas))
        detection_info["area_variance"] = float(np.var(areas)) if len(areas) > 1 else 0.0
        detection_info["mean_circularity"] = float(np.mean(circularities))
    else:
        detection_info["mean_circularity"] = 0.0

    # Create debug visualization
    if DEBUG_OUTPUT_DIR is not None:
        debug_img = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        for i, contour in enumerate(valid_contours):
            # Draw contour
            cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 2)
            # Draw center point
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(debug_img, (cx, cy), 3, (0, 0, 255), -1)
        save_debug_image("10_contour_detected_pips.png", debug_img)

    pip_count = len(valid_contours)
    return pip_count, valid_contours, detection_info


def detect_pips_hough_adaptive(
    image: np.ndarray,
    param2_range: Tuple[int, int, int] = (20, 40, 5),
    max_pips: int = 6
) -> Tuple[int, np.ndarray, dict]:
    """
    Detect pips using adaptive HoughCircles parameters.

    Tries multiple param2 values to find the best detection result,
    since optimal parameters vary with image quality and pip size.

    Args:
        image: Input BGR or grayscale image of a domino half.
        param2_range: Tuple of (start, stop, step) for param2 values to try.
            Lower values detect more circles.
        max_pips: Maximum expected pips (default 6 for standard dominoes).

    Returns:
        Same as detect_pips_hough: (pip_count, circles, detection_info)
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")

    best_result = (0, None, {})
    best_score = -1

    # Try different param2 values
    for param2 in range(param2_range[0], param2_range[1], param2_range[2]):
        try:
            pip_count, circles, info = detect_pips_hough(
                image,
                param2=param2
            )

            # Score this detection
            # Prefer results with pip count in valid range
            if 0 <= pip_count <= max_pips:
                # Score based on consistency (low variance is good)
                variance_penalty = info.get("radius_variance", 0) / 100.0
                score = pip_count * (1 - min(variance_penalty, 0.5))

                if score > best_score:
                    best_score = score
                    best_result = (pip_count, circles, info)
                    info["param2_used"] = param2

            # If we get a valid count with low variance, stop searching
            if pip_count > 0 and pip_count <= max_pips and info.get("radius_variance", 999) < 10:
                break

        except Exception:
            continue

    return best_result


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

    # Test Hough circle detection
    pip_count, circles, info = detect_pips_hough(test_img)
    assert isinstance(pip_count, int)
    assert isinstance(info, dict)
    assert "mean_radius" in info

    # Test adaptive detection
    pip_count_adaptive, _, _ = detect_pips_hough_adaptive(test_img)
    assert isinstance(pip_count_adaptive, int)

    # Test contour-based pip detection
    pip_count_contours, contours, contour_info = detect_pips_contours(test_img)
    assert isinstance(pip_count_contours, int)
    assert isinstance(contours, list)
    assert isinstance(contour_info, dict)
    assert "mean_area" in contour_info
    assert "circularities" in contour_info

    return True


if __name__ == "__main__":
    if _self_test():
        print("Self-test passed!")
    else:
        print("Self-test failed!")
