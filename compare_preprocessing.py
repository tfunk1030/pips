"""
Compare preprocessing effects on domino pip detection.

This script shows side-by-side comparison of images with and without
preprocessing, including pip detection results for each.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from preprocess import preprocess_domino_tray
from extract_dominoes import count_pips


def extract_dominoes_with_visualization(
    image: np.ndarray,
    detection_threshold: int = 200
) -> tuple:
    """
    Extract domino tiles from an image, count pips, and create visualization.

    Args:
        image: BGR image containing domino tiles.
        detection_threshold: Grayscale threshold for domino detection.
            Default is 200.

    Returns:
        Tuple of (dominoes_list, annotated_image) where:
            - dominoes_list: List of tuples (left_pips, right_pips)
            - annotated_image: Image with domino detections drawn
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, detection_threshold, 255, cv2.THRESH_BINARY_INV)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create annotated image copy
    annotated = image.copy()
    dominoes = []

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 120 < w < 240 and 50 < h < 100:
            tile = image[y:y+h, x:x+w]

            left = tile[:, :w//2]
            right = tile[:, w//2:]

            left_pips = count_pips(left)
            right_pips = count_pips(right)
            dominoes.append((left_pips, right_pips))

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Draw center divider
            cv2.line(
                annotated,
                (x + w//2, y),
                (x + w//2, y + h),
                (0, 255, 255),
                1
            )

            # Add pip count labels
            label = f"{left_pips}|{right_pips}"
            cv2.putText(
                annotated,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    return dominoes, annotated


def create_comparison_image(
    original: np.ndarray,
    preprocessed: np.ndarray,
    original_annotated: np.ndarray,
    preprocessed_annotated: np.ndarray,
    original_dominoes: list,
    preprocessed_dominoes: list,
    metrics: dict
) -> np.ndarray:
    """
    Create a side-by-side comparison image.

    Args:
        original: Original BGR image.
        preprocessed: Preprocessed BGR image.
        original_annotated: Original image with domino annotations.
        preprocessed_annotated: Preprocessed image with domino annotations.
        original_dominoes: List of (left_pips, right_pips) from original.
        preprocessed_dominoes: List of (left_pips, right_pips) from preprocessed.
        metrics: Preprocessing metrics dictionary.

    Returns:
        BGR comparison image with both versions side by side.
    """
    # Ensure images have the same height for horizontal stacking
    h1, w1 = original_annotated.shape[:2]
    h2, w2 = preprocessed_annotated.shape[:2]

    # Scale if needed to match heights
    if h1 != h2:
        scale = h1 / h2
        new_w = int(w2 * scale)
        preprocessed_annotated = cv2.resize(
            preprocessed_annotated,
            (new_w, h1)
        )
        w2 = new_w

    # Create header for comparison
    header_height = 100
    total_width = w1 + w2 + 10  # 10 pixels gap
    total_height = h1 + header_height

    # Create canvas
    canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

    # Add header labels
    cv2.putText(
        canvas,
        "ORIGINAL (No Preprocessing)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 200),
        2
    )
    cv2.putText(
        canvas,
        f"Detected: {original_dominoes}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )
    cv2.putText(
        canvas,
        f"Count: {len(original_dominoes)} dominoes",
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )

    cv2.putText(
        canvas,
        "PREPROCESSED (Enhanced)",
        (w1 + 20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 128, 0),
        2
    )
    cv2.putText(
        canvas,
        f"Detected: {preprocessed_dominoes}",
        (w1 + 20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )
    cv2.putText(
        canvas,
        f"Count: {len(preprocessed_dominoes)} dominoes | "
        f"Brightness: {metrics['original_brightness']:.0f} -> "
        f"{metrics['final_brightness']:.0f}",
        (w1 + 20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )

    # Place images
    canvas[header_height:header_height + h1, 0:w1] = original_annotated
    canvas[header_height:header_height + h1, w1 + 10:w1 + 10 + w2] = preprocessed_annotated

    # Draw separator line
    cv2.line(
        canvas,
        (w1 + 5, 0),
        (w1 + 5, total_height),
        (128, 128, 128),
        2
    )

    return canvas


def compare_preprocessing(
    image_path: str,
    output_dir: str = "debug_preprocess"
) -> dict:
    """
    Compare preprocessing effects on an image.

    Args:
        image_path: Path to the input image.
        output_dir: Directory to save comparison output. Default is
            "debug_preprocess".

    Returns:
        Dictionary with comparison results including:
            - original_dominoes: Dominoes detected without preprocessing
            - preprocessed_dominoes: Dominoes detected with preprocessing
            - metrics: Preprocessing metrics
            - output_path: Path to saved comparison image

    Raises:
        ValueError: If image cannot be loaded.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Process original (no preprocessing)
    original_dominoes, original_annotated = extract_dominoes_with_visualization(
        image
    )

    # Apply preprocessing
    preprocessed, metrics = preprocess_domino_tray(image)

    # Process preprocessed image
    preprocessed_dominoes, preprocessed_annotated = extract_dominoes_with_visualization(
        preprocessed
    )

    # Create comparison image
    comparison = create_comparison_image(
        original=image,
        preprocessed=preprocessed,
        original_annotated=original_annotated,
        preprocessed_annotated=preprocessed_annotated,
        original_dominoes=original_dominoes,
        preprocessed_dominoes=preprocessed_dominoes,
        metrics=metrics
    )

    # Save comparison image
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = Path(image_path).stem
    comparison_filename = f"{base_name}_comparison.png"
    comparison_path = output_path / comparison_filename
    cv2.imwrite(str(comparison_path), comparison)

    # Build results dictionary
    results = {
        "original_dominoes": original_dominoes,
        "preprocessed_dominoes": preprocessed_dominoes,
        "metrics": metrics,
        "output_path": str(comparison_path),
        "differences": {
            "count_original": len(original_dominoes),
            "count_preprocessed": len(preprocessed_dominoes),
            "count_diff": len(preprocessed_dominoes) - len(original_dominoes),
        }
    }

    return results


def main():
    """Main entry point for preprocessing comparison."""
    parser = argparse.ArgumentParser(
        description="Compare preprocessing effects on domino pip detection."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="IMG_2050.png",
        help="Path to the image file (default: IMG_2050.png)"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="debug_preprocess",
        help="Directory to save comparison output (default: debug_preprocess)"
    )

    args = parser.parse_args()

    try:
        results = compare_preprocessing(args.image, args.output_dir)

        print("=" * 60)
        print("PREPROCESSING COMPARISON RESULTS")
        print("=" * 60)

        print(f"\nOriginal (no preprocessing):")
        print(f"  Dominoes detected: {len(results['original_dominoes'])}")
        print(f"  Pip counts: {results['original_dominoes']}")

        print(f"\nPreprocessed:")
        print(f"  Dominoes detected: {len(results['preprocessed_dominoes'])}")
        print(f"  Pip counts: {results['preprocessed_dominoes']}")

        print(f"\nPreprocessing metrics:")
        metrics = results['metrics']
        print(f"  Steps applied: {', '.join(metrics['steps_applied'])}")
        print(
            f"  Brightness: {metrics['original_brightness']:.1f} -> "
            f"{metrics['final_brightness']:.1f} "
            f"({metrics['brightness_change']:+.1f})"
        )
        print(
            f"  Contrast: {metrics['original_contrast']:.1f} -> "
            f"{metrics['final_contrast']:.1f} "
            f"({metrics['contrast_change']:+.1f})"
        )

        print(f"\nComparison image saved to: {results['output_path']}")
        print("=" * 60)

        return 0

    except ValueError as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
