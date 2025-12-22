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


def validate_pip_count(pip_count: int) -> bool:
    """
    Validate that a pip count is within valid domino range.

    Standard dominoes have pips ranging from 0 (blank) to 6 on each half.
    This function validates that a detected pip count is within this range.

    Args:
        pip_count: The detected number of pips to validate.

    Returns:
        True if pip_count is in range 0-6 (inclusive), False otherwise.
        Blank dominoes (0 pips) are considered valid.
    """
    return 0 <= pip_count <= 6


def calculate_confidence(
    pip_count: int,
    circularity: float = 1.0,
    size_variance: float = 0.0,
    circularity_weight: float = 0.4,
    size_weight: float = 0.3,
    count_weight: float = 0.3
) -> float:
    """
    Calculate confidence score for pip detection.

    Computes a confidence score (0.0-1.0) based on:
    - Pip circularity: How circular the detected pips are (1.0 = perfect circle)
    - Size consistency: Variance in pip sizes (lower = more consistent)
    - Count validation: Whether pip count is in valid range (0-6)

    High confidence (>0.85) indicates reliable detection.
    Low confidence (<0.7) indicates ambiguous or uncertain detection.

    Args:
        pip_count: Number of detected pips.
        circularity: Average circularity of detected pips (0.0-1.0).
            Circularity is calculated as 4*pi*area/perimeter^2.
            A perfect circle has circularity of 1.0.
            Good pip detection typically has circularity > 0.7.
        size_variance: Variance in pip sizes (normalized).
            Lower values indicate more consistent pip sizes.
            Good detection typically has variance < 0.3.
        circularity_weight: Weight for circularity factor (default 0.4).
        size_weight: Weight for size consistency factor (default 0.3).
        count_weight: Weight for count validation factor (default 0.3).

    Returns:
        Confidence score between 0.0 and 1.0.
        - > 0.85: High confidence, reliable detection
        - 0.7 - 0.85: Medium confidence, likely correct
        - < 0.7: Low confidence, uncertain detection

    Notes:
        For blank dominoes (0 pips), returns high confidence (0.95) since
        absence of pips is a valid and detectable state.
        For pip counts outside 0-6 range, applies a strong penalty.
    """
    # Handle blank dominoes (0 pips) - high confidence case
    if pip_count == 0:
        # Blank dominoes are valid and should have high confidence
        # when no circular shapes are detected
        return 0.95

    # 1. Circularity score (0.0 - 1.0)
    # Good pips have circularity > 0.7
    # Map circularity from [0.5, 1.0] range to [0.0, 1.0] score
    # Values below 0.5 get score 0, values above 0.9 get diminishing returns
    if circularity >= 0.9:
        circularity_score = 1.0
    elif circularity >= 0.5:
        # Linear mapping from [0.5, 0.9] to [0.0, 1.0]
        circularity_score = (circularity - 0.5) / 0.4
    else:
        # Poor circularity - likely not pips
        circularity_score = 0.0

    # 2. Size consistency score (0.0 - 1.0)
    # Lower variance is better - pips should be similar sizes
    # Map variance from [0.0, 0.5] to [1.0, 0.0]
    # Variance > 0.5 gets 0 score
    if size_variance <= 0.0:
        size_score = 1.0
    elif size_variance <= 0.5:
        # Linear mapping: 0.0 variance = 1.0 score, 0.5 variance = 0.0 score
        size_score = 1.0 - (size_variance / 0.5)
    else:
        # High variance - inconsistent pip sizes
        size_score = 0.0

    # 3. Count validation score (0.0 - 1.0)
    # Valid pip counts (0-6) get high score
    # Counts just outside range get partial score
    # Far outside range get zero
    if validate_pip_count(pip_count):
        count_score = 1.0
    elif pip_count == 7:
        # Just over the limit - might be noise
        count_score = 0.3
    elif pip_count == 8:
        # Likely over-detection
        count_score = 0.1
    else:
        # Far out of range (negative or > 8)
        count_score = 0.0

    # 4. Calculate weighted confidence
    # Ensure weights sum to 1.0
    total_weight = circularity_weight + size_weight + count_weight
    if total_weight <= 0:
        total_weight = 1.0

    confidence = (
        (circularity_weight * circularity_score +
         size_weight * size_score +
         count_weight * count_score) / total_weight
    )

    # Clamp to [0.0, 1.0]
    confidence = max(0.0, min(1.0, confidence))

    return confidence


def detect_rotation_angle(
    contour: np.ndarray
) -> Tuple[float, Tuple[float, float], Tuple[float, float], np.ndarray]:
    """
    Detect the rotation angle of a domino contour using minAreaRect.

    Uses cv2.minAreaRect to find the minimum area bounding rectangle
    of a contour and extracts the rotation angle. The angle is normalized
    to ensure dominoes are oriented with the longer side horizontal.

    Args:
        contour: Input contour points from cv2.findContours.
            Must be a numpy array of shape (N, 1, 2) or (N, 2).

    Returns:
        Tuple of (angle, center, size, box_points):
        - angle: Rotation angle in degrees. Positive values are counter-clockwise.
            Range is normalized to [-45, 45] degrees, assuming domino should
            be horizontal with longer side along x-axis.
        - center: Tuple (cx, cy) of the rectangle center point.
        - size: Tuple (width, height) of the rectangle dimensions.
            After normalization, width >= height (longer side is width).
        - box_points: NumPy array of 4 corner points of the rotated rectangle.

    Raises:
        ValueError: If contour is None, empty, or has invalid shape.

    Notes:
        cv2.minAreaRect returns angles in the range [-90, 0) degrees.
        This function normalizes the angle so that:
        - The longer side of the rectangle is considered the width
        - Angle is adjusted to keep the domino approximately horizontal
        - Output angle indicates rotation needed to straighten the domino
    """
    if contour is None or len(contour) == 0:
        raise ValueError("Input contour is empty or None")

    # Ensure contour has enough points for minAreaRect
    if len(contour) < 5:
        raise ValueError("Contour must have at least 5 points for minAreaRect")

    # Get minimum area bounding rectangle
    # Returns: ((cx, cy), (width, height), angle)
    # angle is in range [-90, 0)
    rect = cv2.minAreaRect(contour)

    center = rect[0]  # (cx, cy)
    size = rect[1]    # (width, height)
    angle = rect[2]   # angle in degrees

    width, height = size

    # Normalize angle to ensure longer side is horizontal
    # minAreaRect can return width < height depending on orientation
    # We want dominoes to be oriented with longer side horizontal
    if width < height:
        # Swap width and height, adjust angle by 90 degrees
        width, height = height, width
        angle = angle + 90

    # Normalize angle to range [-45, 45] for consistency
    # This ensures we don't flip the domino upside down unnecessarily
    if angle > 45:
        angle = angle - 90
    elif angle < -45:
        angle = angle + 90

    # Get the 4 corner points of the rotated rectangle
    # These are useful for visualization and cropping
    box_points = cv2.boxPoints(rect)
    box_points = np.intp(box_points)

    # Return normalized values
    normalized_size = (width, height)

    return angle, center, normalized_size, box_points


def detect_rotation_angle_from_image(
    image: np.ndarray,
    block_size: int = 11,
    c_value: int = 2
) -> Tuple[float, Tuple[float, float], Tuple[float, float], np.ndarray, np.ndarray]:
    """
    Detect the rotation angle of a domino from an image.

    Preprocesses the image, finds contours, and detects the rotation angle
    of the largest contour (assumed to be the domino).

    Args:
        image: Input BGR or grayscale image containing a domino.
        block_size: Block size for adaptive thresholding.
        c_value: Constant for adaptive thresholding.

    Returns:
        Tuple of (angle, center, size, box_points, contour):
        - angle: Rotation angle in degrees (see detect_rotation_angle).
        - center: Tuple (cx, cy) of the rectangle center point.
        - size: Tuple (width, height) of the rectangle dimensions.
        - box_points: NumPy array of 4 corner points.
        - contour: The largest contour found in the image.

    Raises:
        ValueError: If image is empty or no contours are found.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")

    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply bilateral filter for edge-preserving smoothing
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_value
    )

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No contours found in image")

    # Find the largest contour (assumed to be the domino)
    largest_contour = max(contours, key=cv2.contourArea)

    # Detect rotation angle from the largest contour
    angle, center, size, box_points = detect_rotation_angle(largest_contour)

    # Save debug visualization if enabled
    if DEBUG_OUTPUT_DIR is not None:
        debug_img = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_img, [box_points], 0, (0, 255, 0), 2)
        cv2.circle(debug_img, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
        # Add angle text
        cv2.putText(
            debug_img,
            f"Angle: {angle:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        save_debug_image("11_rotation_detected.png", debug_img)

    return angle, center, size, box_points, largest_contour


def rotate_domino(
    image: np.ndarray,
    angle: float,
    expand_canvas: bool = True,
    border_value: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Rotate a domino image by a specified angle using cv2.warpAffine.

    Applies rotation correction to straighten a domino before pip detection.
    Rotation is performed around the image center.

    Args:
        image: Input BGR or grayscale image to rotate.
        angle: Rotation angle in degrees. Positive values rotate
            counter-clockwise, negative values rotate clockwise.
        expand_canvas: If True, expand the canvas to fit the entire
            rotated image without cropping. If False, maintain original
            image dimensions (may crop corners).
        border_value: Color value for border pixels (default black).
            For grayscale images, only the first value is used.

    Returns:
        Rotated image as numpy array. Same dtype as input.

    Raises:
        ValueError: If image is None or empty.

    Notes:
        This function uses cv2.getRotationMatrix2D to create the
        transformation matrix and cv2.warpAffine to apply it.
        The rotation preserves the image center point.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")

    # Get image dimensions
    h, w = image.shape[:2]

    # Calculate center of image
    center = (w / 2.0, h / 2.0)

    # Create rotation matrix
    # getRotationMatrix2D(center, angle, scale)
    # angle: positive = counter-clockwise rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    if expand_canvas:
        # Calculate new image bounds to fit rotated image
        # Convert angle to radians
        abs_cos = abs(np.cos(np.radians(angle)))
        abs_sin = abs(np.sin(np.radians(angle)))

        # New width and height after rotation
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        # Adjust rotation matrix to account for new canvas size
        # Shift the center to the new image center
        M[0, 2] += (new_w / 2.0) - center[0]
        M[1, 2] += (new_h / 2.0) - center[1]

        output_size = (new_w, new_h)
    else:
        # Keep original dimensions
        output_size = (w, h)

    # Apply rotation using warpAffine
    # borderMode: use constant border with specified color
    if len(image.shape) == 2:
        # Grayscale image - use single channel border value
        border = border_value[0] if isinstance(border_value, tuple) else border_value
    else:
        border = border_value

    rotated = cv2.warpAffine(
        image,
        M,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border
    )

    # Save debug visualization if enabled
    if DEBUG_OUTPUT_DIR is not None:
        save_debug_image("12_rotated_domino.png", rotated)

    return rotated


def split_domino_halves(
    image: np.ndarray,
    padding: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a domino image into left and right halves.

    Divides a rotation-corrected domino image vertically down the middle
    to separate the two pip-containing halves for independent detection.

    Args:
        image: Input BGR or grayscale image of a domino tile.
            Should be rotation-corrected (horizontal) before splitting.
        padding: Optional padding in pixels to remove from the center
            of each half. Helps avoid the center divider line if present.

    Returns:
        Tuple of (left_half, right_half) images.
        - left_half: Left half of the domino (first pip value)
        - right_half: Right half of the domino (second pip value)
        Both halves will have equal width.

    Raises:
        ValueError: If image is None or empty.

    Notes:
        For best results, apply rotation correction using rotate_domino()
        before calling this function. The domino should be oriented with
        its longer side horizontal.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")

    # Get image dimensions
    if len(image.shape) == 3:
        h, w, c = image.shape
    else:
        h, w = image.shape

    # Calculate midpoint
    mid = w // 2

    # Ensure equal width by handling odd width images
    # For odd widths, left half gets the extra pixel
    left_end = mid
    right_start = mid

    # Apply padding if specified (to avoid center divider line)
    if padding > 0:
        # Don't let padding exceed reasonable bounds
        max_padding = min(w // 10, padding)
        left_end = max(0, mid - max_padding)
        right_start = min(w, mid + max_padding)

    # Split the image
    if len(image.shape) == 3:
        left_half = image[:, :left_end, :]
        right_half = image[:, right_start:, :]
    else:
        left_half = image[:, :left_end]
        right_half = image[:, right_start:]

    # Ensure equal widths by trimming if necessary
    left_w = left_half.shape[1]
    right_w = right_half.shape[1]

    if left_w != right_w:
        # Make both halves the minimum width
        min_w = min(left_w, right_w)
        if len(image.shape) == 3:
            left_half = left_half[:, :min_w, :]
            right_half = right_half[:, :min_w, :]
        else:
            left_half = left_half[:, :min_w]
            right_half = right_half[:, :min_w]

    # Save debug visualization if enabled
    if DEBUG_OUTPUT_DIR is not None:
        if len(image.shape) == 3:
            debug_img = image.copy()
            # Draw vertical line at split point
            cv2.line(debug_img, (mid, 0), (mid, h), (0, 255, 0), 2)
        else:
            debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.line(debug_img, (mid, 0), (mid, h), (0, 255, 0), 2)
        save_debug_image("13_split_halves.png", debug_img)
        save_debug_image("13_left_half.png", left_half)
        save_debug_image("13_right_half.png", right_half)

    return left_half, right_half


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

    # Test rotation function
    rotated = rotate_domino(test_img, 45.0)
    assert rotated is not None
    assert rotated.shape[2] == 3  # Still BGR

    # Test rotation without canvas expansion
    rotated_no_expand = rotate_domino(test_img, 30.0, expand_canvas=False)
    assert rotated_no_expand.shape[:2] == test_img.shape[:2]  # Same dimensions

    # Test grayscale rotation
    gray_img = np.zeros((100, 200), dtype=np.uint8)
    rotated_gray = rotate_domino(gray_img, 45.0)
    assert rotated_gray is not None
    assert len(rotated_gray.shape) == 2  # Still grayscale

    # Test split_domino_halves with BGR image
    domino_img = np.zeros((100, 200, 3), dtype=np.uint8)
    domino_img[:, :100] = [255, 0, 0]  # Blue left half
    domino_img[:, 100:] = [0, 0, 255]  # Red right half
    left, right = split_domino_halves(domino_img)
    assert left is not None
    assert right is not None
    assert left.shape[1] == right.shape[1]  # Equal widths
    assert left.shape[0] == domino_img.shape[0]  # Same height

    # Test split_domino_halves with grayscale image
    gray_domino = np.zeros((50, 100), dtype=np.uint8)
    gray_domino[:, :50] = 100
    gray_domino[:, 50:] = 200
    left_gray, right_gray = split_domino_halves(gray_domino)
    assert left_gray.shape[1] == right_gray.shape[1]
    assert len(left_gray.shape) == 2  # Still grayscale

    # Test split with odd width
    odd_img = np.zeros((100, 201, 3), dtype=np.uint8)
    left_odd, right_odd = split_domino_halves(odd_img)
    assert left_odd.shape[1] == right_odd.shape[1]

    # Test split with padding
    padded_left, padded_right = split_domino_halves(domino_img, padding=5)
    assert padded_left.shape[1] == padded_right.shape[1]
    assert padded_left.shape[1] < left.shape[1]  # Padding reduces width

    # Test calculate_confidence function
    # Test with valid pip count and good metrics
    conf_score = calculate_confidence(pip_count=3, circularity=0.9, size_variance=0.1)
    assert 0.0 <= conf_score <= 1.0
    assert conf_score >= 0.7  # Good detection should have reasonable confidence

    # Test blank domino
    conf_blank = calculate_confidence(pip_count=0)
    assert conf_blank >= 0.9  # Blank dominoes should have high confidence

    # Test edge case with max pips
    conf_max = calculate_confidence(pip_count=6, circularity=1.0, size_variance=0.0)
    assert conf_max >= 0.85  # Perfect detection should be high confidence

    # Test invalid pip count
    conf_invalid = calculate_confidence(pip_count=10, circularity=0.9, size_variance=0.1)
    assert conf_invalid < 0.7  # Invalid count should have low confidence

    return True


if __name__ == "__main__":
    if _self_test():
        print("Self-test passed!")
    else:
        print("Self-test failed!")
