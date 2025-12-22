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
