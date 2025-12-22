"""
Image preprocessing functions for domino tray images.

This module provides preprocessing functions to improve domino pip detection
accuracy, especially in low-light or poorly balanced photos.
"""

import os
from pathlib import Path

import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance
    local contrast in an image.

    CLAHE is applied to the L channel of the LAB color space to improve
    contrast without affecting color balance.

    Args:
        image: Input BGR image (numpy array).
        clip_limit: Threshold for contrast limiting. Higher values give more
            contrast but may amplify noise. Default is 2.0.
        tile_grid_size: Size of grid for histogram equalization. Smaller tiles
            give more local contrast enhancement. Default is (8, 8).

    Returns:
        Enhanced BGR image with improved local contrast.

    Raises:
        ValueError: If image is None or empty.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is None or empty")

    # Convert BGR to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split into L, A, B channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Create CLAHE object and apply to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    # Merge channels back
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])

    # Convert back to BGR
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return result


def normalize_brightness(
    image: np.ndarray,
    target_brightness: float = 128.0
) -> np.ndarray:
    """
    Normalize image brightness to a target mean value.

    Adjusts the overall brightness of an image by scaling the V (value) channel
    in HSV color space to achieve the target mean brightness. Includes clipping
    prevention to avoid oversaturation.

    Args:
        image: Input BGR image (numpy array).
        target_brightness: Target mean brightness value (0-255). Default is 128.0
            which represents a medium brightness level.

    Returns:
        Brightness-normalized BGR image.

    Raises:
        ValueError: If image is None or empty.
        ValueError: If target_brightness is not in range [0, 255].
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is None or empty")

    if not 0 <= target_brightness <= 255:
        raise ValueError("target_brightness must be in range [0, 255]")

    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split into H, S, V channels
    h_channel, s_channel, v_channel = cv2.split(hsv)

    # Calculate current mean brightness from V channel
    current_brightness = np.mean(v_channel)

    # Avoid division by zero for completely black images
    if current_brightness < 1e-6:
        # Return original if image is essentially black
        return image.copy()

    # Calculate scaling factor
    scale_factor = target_brightness / current_brightness

    # Apply scaling to V channel with clipping prevention
    # Convert to float32 for accurate computation
    v_scaled = v_channel.astype(np.float32) * scale_factor

    # Clip to valid range [0, 255] to prevent overflow
    v_scaled = np.clip(v_scaled, 0, 255).astype(np.uint8)

    # Merge channels back
    hsv_normalized = cv2.merge([h_channel, s_channel, v_scaled])

    # Convert back to BGR
    result = cv2.cvtColor(hsv_normalized, cv2.COLOR_HSV2BGR)

    return result


def apply_white_balance(
    image: np.ndarray,
    method: str = "gray_world"
) -> np.ndarray:
    """
    Apply white balance correction to an image.

    Corrects color cast in images by adjusting channel gains. Supports multiple
    white balance algorithms.

    Args:
        image: Input BGR image (numpy array).
        method: White balance algorithm to use. Options:
            - "gray_world": Assumes average scene color should be gray. Scales
              each channel to have the same mean value. Works well for most
              natural scenes with varied colors.
            - "white_patch": Assumes the brightest pixels should be white.
              Scales channels based on the maximum values. Works well when
              there's a known white reference in the image.
            Default is "gray_world".

    Returns:
        Color-corrected BGR image.

    Raises:
        ValueError: If image is None or empty.
        ValueError: If method is not recognized.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is None or empty")

    valid_methods = ("gray_world", "white_patch")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    if method == "gray_world":
        return _white_balance_gray_world(image)
    else:  # white_patch
        return _white_balance_white_patch(image)


def _white_balance_gray_world(image: np.ndarray) -> np.ndarray:
    """
    Apply gray-world white balance assumption.

    The gray-world assumption states that the average color of a scene should
    be gray (neutral). This function scales each color channel so their means
    become equal.

    Args:
        image: Input BGR image (numpy array).

    Returns:
        Color-corrected BGR image.
    """
    # Convert to float32 for accurate computation
    img_float = image.astype(np.float32)

    # Calculate mean of each channel
    b_mean = np.mean(img_float[:, :, 0])
    g_mean = np.mean(img_float[:, :, 1])
    r_mean = np.mean(img_float[:, :, 2])

    # Calculate overall mean (target gray value)
    overall_mean = (b_mean + g_mean + r_mean) / 3.0

    # Avoid division by zero
    if b_mean < 1e-6 or g_mean < 1e-6 or r_mean < 1e-6:
        # Return original if any channel is essentially black
        return image.copy()

    # Calculate scaling factors for each channel
    b_scale = overall_mean / b_mean
    g_scale = overall_mean / g_mean
    r_scale = overall_mean / r_mean

    # Apply scaling to each channel
    result = img_float.copy()
    result[:, :, 0] = img_float[:, :, 0] * b_scale
    result[:, :, 1] = img_float[:, :, 1] * g_scale
    result[:, :, 2] = img_float[:, :, 2] * r_scale

    # Clip to valid range and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def _white_balance_white_patch(image: np.ndarray) -> np.ndarray:
    """
    Apply white-patch white balance assumption.

    The white-patch assumption states that the brightest pixels in an image
    should be white. This function scales each channel based on its maximum
    value to achieve white balance.

    Args:
        image: Input BGR image (numpy array).

    Returns:
        Color-corrected BGR image.
    """
    # Convert to float32 for accurate computation
    img_float = image.astype(np.float32)

    # Find maximum value in each channel (using 99th percentile to avoid outliers)
    b_max = np.percentile(img_float[:, :, 0], 99)
    g_max = np.percentile(img_float[:, :, 1], 99)
    r_max = np.percentile(img_float[:, :, 2], 99)

    # Target maximum (white point)
    target_max = 255.0

    # Avoid division by zero
    if b_max < 1e-6 or g_max < 1e-6 or r_max < 1e-6:
        # Return original if any channel has very low max value
        return image.copy()

    # Calculate scaling factors for each channel
    b_scale = target_max / b_max
    g_scale = target_max / g_max
    r_scale = target_max / r_max

    # Apply scaling to each channel
    result = img_float.copy()
    result[:, :, 0] = img_float[:, :, 0] * b_scale
    result[:, :, 1] = img_float[:, :, 1] * g_scale
    result[:, :, 2] = img_float[:, :, 2] * r_scale

    # Clip to valid range and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def preprocess_domino_tray(
    image: np.ndarray,
    enable_white_balance: bool = True,
    enable_brightness_normalize: bool = True,
    enable_clahe: bool = True,
    white_balance_method: str = "gray_world",
    target_brightness: float = 128.0,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple = (8, 8)
) -> tuple:
    """
    Apply full preprocessing pipeline to a domino tray image.

    Chains preprocessing steps in optimal order to improve domino pip detection:
    1. White balance correction (removes color cast)
    2. Brightness normalization (standardizes lighting)
    3. CLAHE contrast enhancement (improves local contrast for pip detection)

    Args:
        image: Input BGR image (numpy array).
        enable_white_balance: Whether to apply white balance correction.
            Default is True.
        enable_brightness_normalize: Whether to apply brightness normalization.
            Default is True.
        enable_clahe: Whether to apply CLAHE contrast enhancement.
            Default is True.
        white_balance_method: Method for white balance ("gray_world" or
            "white_patch"). Default is "gray_world".
        target_brightness: Target mean brightness for normalization (0-255).
            Default is 128.0.
        clahe_clip_limit: CLAHE clip limit for contrast limiting.
            Default is 2.0.
        clahe_tile_grid_size: CLAHE tile grid size for local histogram
            equalization. Default is (8, 8).

    Returns:
        A tuple containing:
            - result: Preprocessed BGR image
            - metrics: Dictionary with preprocessing metrics including:
                - steps_applied: List of preprocessing steps that were applied
                - original_brightness: Mean brightness before preprocessing
                - final_brightness: Mean brightness after preprocessing
                - brightness_change: Difference in brightness
                - original_contrast: Standard deviation of brightness before
                - final_contrast: Standard deviation of brightness after

    Raises:
        ValueError: If image is None or empty.

    Example:
        >>> import cv2
        >>> image = cv2.imread("domino_tray.jpg")
        >>> result, metrics = preprocess_domino_tray(image)
        >>> print(f"Applied steps: {metrics['steps_applied']}")
        >>> print(f"Brightness change: {metrics['brightness_change']:.1f}")
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is None or empty")

    # Initialize result and tracking
    result = image.copy()
    steps_applied = []

    # Calculate original image metrics
    original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_brightness = float(np.mean(original_gray))
    original_contrast = float(np.std(original_gray))

    # Step 1: White balance correction
    if enable_white_balance:
        result = apply_white_balance(result, method=white_balance_method)
        steps_applied.append(f"white_balance ({white_balance_method})")

    # Step 2: Brightness normalization
    if enable_brightness_normalize:
        result = normalize_brightness(result, target_brightness=target_brightness)
        steps_applied.append(f"brightness_normalize (target={target_brightness})")

    # Step 3: CLAHE contrast enhancement
    if enable_clahe:
        result = apply_clahe(
            result,
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size
        )
        steps_applied.append(f"clahe (clip={clahe_clip_limit})")

    # Calculate final image metrics
    final_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    final_brightness = float(np.mean(final_gray))
    final_contrast = float(np.std(final_gray))

    # Build metrics dictionary
    metrics = {
        "steps_applied": steps_applied,
        "original_brightness": original_brightness,
        "final_brightness": final_brightness,
        "brightness_change": final_brightness - original_brightness,
        "original_contrast": original_contrast,
        "final_contrast": final_contrast,
        "contrast_change": final_contrast - original_contrast,
    }

    return result, metrics


def preprocess_tile(
    tile: np.ndarray,
    enable_white_balance: bool = True,
    enable_brightness_normalize: bool = True,
    enable_clahe: bool = True,
    white_balance_method: str = "gray_world",
    target_brightness: float = 128.0,
    clahe_clip_limit: float = 3.0,
    clahe_tile_grid_size: tuple = (4, 4)
) -> np.ndarray:
    """
    Apply preprocessing pipeline to an individual domino tile.

    Similar to preprocess_domino_tray but with parameters optimized for small
    cropped domino tiles. Uses a smaller CLAHE tile grid size and higher clip
    limit by default since tiles are smaller than full images.

    Args:
        tile: Input BGR image of a cropped domino tile (numpy array).
        enable_white_balance: Whether to apply white balance correction.
            Default is True.
        enable_brightness_normalize: Whether to apply brightness normalization.
            Default is True.
        enable_clahe: Whether to apply CLAHE contrast enhancement.
            Default is True.
        white_balance_method: Method for white balance ("gray_world" or
            "white_patch"). Default is "gray_world".
        target_brightness: Target mean brightness for normalization (0-255).
            Default is 128.0.
        clahe_clip_limit: CLAHE clip limit for contrast limiting.
            Default is 3.0 (higher than full image to compensate for smaller area).
        clahe_tile_grid_size: CLAHE tile grid size for local histogram
            equalization. Default is (4, 4) (smaller than full image since
            tiles are already small).

    Returns:
        Preprocessed BGR image of the tile.

    Raises:
        ValueError: If tile is None or empty.

    Example:
        >>> import cv2
        >>> tile = cv2.imread("domino_tile.jpg")
        >>> processed_tile = preprocess_tile(tile)
    """
    if tile is None or tile.size == 0:
        raise ValueError("Input tile is None or empty")

    # Initialize result
    result = tile.copy()

    # Step 1: White balance correction
    if enable_white_balance:
        result = apply_white_balance(result, method=white_balance_method)

    # Step 2: Brightness normalization
    if enable_brightness_normalize:
        result = normalize_brightness(result, target_brightness=target_brightness)

    # Step 3: CLAHE contrast enhancement
    if enable_clahe:
        result = apply_clahe(
            result,
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size
        )

    return result


def create_histogram_image(image: np.ndarray, title: str = "") -> np.ndarray:
    """
    Create an image showing the color histogram of an input image.

    Args:
        image: Input BGR image (numpy array).
        title: Title to display on the histogram image.

    Returns:
        BGR image showing the histogram visualization.
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

    # Calculate histograms for each channel
    colors = ('b', 'g', 'r')
    labels = ('Blue', 'Green', 'Red')

    for i, (color, label) in enumerate(zip(colors, labels)):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=color, label=label, linewidth=1)

    ax.set_xlim([0, 256])
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.set_title(title if title else 'Color Histogram')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Convert matplotlib figure to numpy array
    fig.tight_layout()
    fig.canvas.draw()

    # Get the image from the figure using buffer_rgba (modern matplotlib API)
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)

    # Remove alpha channel (RGBA -> RGB) and convert to BGR for OpenCV
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

    plt.close(fig)

    return img_array


def save_debug_images(
    original: np.ndarray,
    after_white_balance: np.ndarray,
    after_brightness: np.ndarray,
    after_clahe: np.ndarray,
    output_dir: str = "debug_preprocess",
    base_name: str = "image"
) -> dict:
    """
    Save intermediate preprocessing images and histograms for debugging.

    Args:
        original: Original BGR image before preprocessing.
        after_white_balance: Image after white balance correction.
        after_brightness: Image after brightness normalization.
        after_clahe: Image after CLAHE contrast enhancement.
        output_dir: Directory to save debug images. Default is "debug_preprocess".
        base_name: Base name for output files. Default is "image".

    Returns:
        Dictionary with paths to all saved files.

    Example:
        >>> paths = save_debug_images(orig, wb, bright, clahe, base_name="IMG_2050")
        >>> print(f"Saved {len(paths)} debug images")
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # Define images and their labels
    stages = [
        ("01_original", original, "Original"),
        ("02_after_white_balance", after_white_balance, "After White Balance"),
        ("03_after_brightness", after_brightness, "After Brightness Normalization"),
        ("04_after_clahe", after_clahe, "After CLAHE"),
    ]

    # Save each intermediate image
    for stage_name, img, label in stages:
        # Save the image
        img_filename = f"{base_name}_{stage_name}.png"
        img_path = output_path / img_filename
        cv2.imwrite(str(img_path), img)
        saved_files[f"{stage_name}_image"] = str(img_path)

        # Save histogram for this stage
        hist_img = create_histogram_image(img, title=f"{label} Histogram")
        hist_filename = f"{base_name}_{stage_name}_histogram.png"
        hist_path = output_path / hist_filename
        cv2.imwrite(str(hist_path), hist_img)
        saved_files[f"{stage_name}_histogram"] = str(hist_path)

    # Create a combined histogram comparison image
    hist_images = []
    for stage_name, img, label in stages:
        hist_img = create_histogram_image(img, title=label)
        hist_images.append(hist_img)

    # Arrange histograms in a 2x2 grid
    top_row = np.hstack([hist_images[0], hist_images[1]])
    bottom_row = np.hstack([hist_images[2], hist_images[3]])
    combined_hist = np.vstack([top_row, bottom_row])

    combined_filename = f"{base_name}_histogram_comparison.png"
    combined_path = output_path / combined_filename
    cv2.imwrite(str(combined_path), combined_hist)
    saved_files["histogram_comparison"] = str(combined_path)

    return saved_files


def preprocess_domino_tray_debug(
    image: np.ndarray,
    output_dir: str = "debug_preprocess",
    base_name: str = "image",
    white_balance_method: str = "gray_world",
    target_brightness: float = 128.0,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple = (8, 8)
) -> tuple:
    """
    Apply preprocessing pipeline with debug output.

    Applies the full preprocessing pipeline and saves intermediate images
    at each step for debugging and visualization.

    Args:
        image: Input BGR image (numpy array).
        output_dir: Directory to save debug images. Default is "debug_preprocess".
        base_name: Base name for output files. Default is "image".
        white_balance_method: Method for white balance ("gray_world" or
            "white_patch"). Default is "gray_world".
        target_brightness: Target mean brightness for normalization (0-255).
            Default is 128.0.
        clahe_clip_limit: CLAHE clip limit for contrast limiting.
            Default is 2.0.
        clahe_tile_grid_size: CLAHE tile grid size for local histogram
            equalization. Default is (8, 8).

    Returns:
        A tuple containing:
            - result: Final preprocessed BGR image
            - metrics: Dictionary with preprocessing metrics
            - debug_files: Dictionary with paths to all saved debug files

    Raises:
        ValueError: If image is None or empty.

    Example:
        >>> import cv2
        >>> image = cv2.imread("domino_tray.jpg")
        >>> result, metrics, debug_files = preprocess_domino_tray_debug(
        ...     image, base_name="domino_tray"
        ... )
        >>> print(f"Saved debug files: {list(debug_files.keys())}")
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is None or empty")

    # Store original
    original = image.copy()

    # Calculate original metrics
    original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_brightness = float(np.mean(original_gray))
    original_contrast = float(np.std(original_gray))

    steps_applied = []

    # Step 1: White balance correction
    after_white_balance = apply_white_balance(original, method=white_balance_method)
    steps_applied.append(f"white_balance ({white_balance_method})")

    # Step 2: Brightness normalization
    after_brightness = normalize_brightness(
        after_white_balance,
        target_brightness=target_brightness
    )
    steps_applied.append(f"brightness_normalize (target={target_brightness})")

    # Step 3: CLAHE contrast enhancement
    after_clahe = apply_clahe(
        after_brightness,
        clip_limit=clahe_clip_limit,
        tile_grid_size=clahe_tile_grid_size
    )
    steps_applied.append(f"clahe (clip={clahe_clip_limit})")

    # Final result
    result = after_clahe

    # Calculate final metrics
    final_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    final_brightness = float(np.mean(final_gray))
    final_contrast = float(np.std(final_gray))

    # Build metrics dictionary
    metrics = {
        "steps_applied": steps_applied,
        "original_brightness": original_brightness,
        "final_brightness": final_brightness,
        "brightness_change": final_brightness - original_brightness,
        "original_contrast": original_contrast,
        "final_contrast": final_contrast,
        "contrast_change": final_contrast - original_contrast,
    }

    # Save debug images
    debug_files = save_debug_images(
        original=original,
        after_white_balance=after_white_balance,
        after_brightness=after_brightness,
        after_clahe=after_clahe,
        output_dir=output_dir,
        base_name=base_name
    )

    return result, metrics, debug_files
