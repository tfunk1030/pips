"""
Extract dominoes from an image and count pips on each tile.

This script detects domino tiles in an image and counts the pips on each half.
Supports optional image preprocessing to improve detection accuracy in low-light
or poorly balanced photos.
"""

import argparse
import cv2
import numpy as np

from preprocess import preprocess_domino_tray, preprocess_tile


def count_pips(half: np.ndarray, threshold: int = 150) -> int:
    """
    Count the number of pips in a domino half.

    Args:
        half: BGR image of one half of a domino tile.
        threshold: Grayscale threshold for pip detection. Default is 150.

    Returns:
        Number of pips detected.
    """
    g = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
    _, t = cv2.threshold(g, threshold, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in cnts if 10 < cv2.contourArea(c) < 200)


def extract_dominoes(
    image: np.ndarray,
    detection_threshold: int = 200,
    tile_preprocess: bool = False
) -> list:
    """
    Extract domino tiles from an image and count pips.

    Args:
        image: BGR image containing domino tiles.
        detection_threshold: Grayscale threshold for domino detection.
            Default is 200.
        tile_preprocess: Whether to apply preprocessing to each individual
            cropped domino tile before pip counting. Uses optimized parameters
            for small images. Default is False.

    Returns:
        List of tuples (left_pips, right_pips) for each detected domino.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, detection_threshold, 255, cv2.THRESH_BINARY_INV)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dominoes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 120 < w < 240 and 50 < h < 100:
            tile = image[y:y+h, x:x+w]

            # Apply tile preprocessing if enabled
            if tile_preprocess:
                tile = preprocess_tile(tile)

            left = tile[:, :w//2]
            right = tile[:, w//2:]

            a = count_pips(left)
            b = count_pips(right)
            dominoes.append((a, b))

    return dominoes


def main():
    """Main entry point for domino extraction."""
    parser = argparse.ArgumentParser(
        description="Extract dominoes from an image and count pips."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="IMG_2050.png",
        help="Path to the image file (default: IMG_2050.png)"
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable image preprocessing (for A/B comparison)"
    )
    parser.add_argument(
        "--tile-preprocess",
        action="store_true",
        help="Apply preprocessing to each individual cropped domino tile"
    )

    args = parser.parse_args()

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not load image '{args.image}'")
        return 1

    # Apply preprocessing unless disabled
    if not args.no_preprocess:
        img, metrics = preprocess_domino_tray(img)
        print(f"Preprocessing applied: {', '.join(metrics['steps_applied'])}")
        print(f"  Brightness: {metrics['original_brightness']:.1f} -> {metrics['final_brightness']:.1f}")
        print(f"  Contrast: {metrics['original_contrast']:.1f} -> {metrics['final_contrast']:.1f}")
    else:
        print("Preprocessing disabled")

    # Report tile preprocessing status
    if args.tile_preprocess:
        print("Tile preprocessing: enabled (optimized parameters for small crops)")

    # Extract dominoes
    dominoes = extract_dominoes(img, tile_preprocess=args.tile_preprocess)
    print(f"Dominoes: {dominoes}")

    return 0


if __name__ == "__main__":
    exit(main())
