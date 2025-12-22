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
