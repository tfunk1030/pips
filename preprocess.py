"""
Image preprocessing functions for domino tray images.

This module provides preprocessing functions to improve domino pip detection
accuracy, especially in low-light or poorly balanced photos.
"""

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
