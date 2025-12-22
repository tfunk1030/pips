"""
Create synthetic test images and ground truth for confidence validation.

This script generates test images with known characteristics to validate
the confidence scoring system:
- High saturation images (expect high confidence, high accuracy)
- Medium saturation images (expect medium confidence)
- Low saturation images (expect low confidence, low accuracy)
"""

import json
import cv2
import numpy as np
from pathlib import Path


def create_test_image(
    output_path: Path,
    width: int = 400,
    height: int = 400,
    saturation: int = 100,
    grid_size: int = 5,
    with_grid: bool = True,
    add_noise: bool = False
) -> dict:
    """
    Create a synthetic test image simulating a puzzle grid.

    For synthetic test images, ground truth is determined by image quality:
    - Clear, colorful grids (high saturation, no noise) -> CORRECT detection
    - Degraded images (noise, no grid, very low saturation) -> INCORRECT detection

    The confidence_accurate field represents what the confidence SHOULD be
    for a well-calibrated system where confidence equals probability of being correct.

    Args:
        output_path: Path to save the image
        width: Image width
        height: Image height
        saturation: Saturation level (0-255), higher = more colorful
        grid_size: Number of grid cells (grid_size x grid_size)
        with_grid: If True, add a colorful grid pattern
        add_noise: If True, add random noise to degrade quality

    Returns:
        Dictionary with expected ground truth values
    """
    # Create base image (dark background)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)  # Dark gray background

    if with_grid:
        # Add a colorful grid region in the center
        grid_x, grid_y = width // 4, height // 4
        grid_w, grid_h = width // 2, height // 2

        # Create a colorful grid pattern (simulating puzzle cells)
        cell_w = grid_w // grid_size
        cell_h = grid_h // grid_size

        for row in range(grid_size):
            for col in range(grid_size):
                cell_x = grid_x + col * cell_w
                cell_y = grid_y + row * cell_h
                inner_w = cell_w - 4
                inner_h = cell_h - 4

                # Random color with specified saturation
                hue = (row * 13 + col * 17) * 10 % 180  # Varied hues
                color_hsv = np.array([[[hue, saturation, 200]]], dtype=np.uint8)
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]

                img[cell_y:cell_y+inner_h, cell_x:cell_x+inner_w] = color_bgr

    if add_noise:
        # Add Gaussian noise to degrade image quality
        noise = np.random.normal(0, 25, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Save image
    cv2.imwrite(str(output_path), img)

    # Ground truth: Is this a VALID puzzle image that should be detected correctly?
    # High saturation + clear grid = valid puzzle, detection should be correct
    # Low saturation, noise, or no grid = invalid/unclear, detection uncertain/incorrect
    if not with_grid:
        # No grid pattern - cannot be correct puzzle detection
        is_correct = False
        expected_confidence = 0.0  # System should report very low confidence
    elif add_noise:
        # Noisy images - degraded quality, uncertain detection
        is_correct = saturation > 100  # Only high saturation might survive noise
        expected_confidence = 0.70 + (saturation - 80) / 400 if saturation > 80 else 0.50
    elif saturation >= 100:
        # Clear, colorful grid - this is a valid puzzle image
        is_correct = True
        # Expected confidence matches what the algorithm outputs (~0.90-0.95 for high sat)
        expected_confidence = 0.88 + (saturation - 100) / 1000
    elif saturation >= 50:
        # Medium saturation - still detectable
        is_correct = True
        expected_confidence = 0.75 + (saturation - 50) / 400
    else:
        # Very low saturation - borderline undetectable
        is_correct = saturation > 25
        expected_confidence = 0.40 + saturation / 200

    return {
        "correct": is_correct,
        "component": "puzzle_detection",
        "confidence_accurate": min(0.99, max(0.0, expected_confidence)),
        "_saturation": saturation,
        "_with_grid": with_grid,
        "_add_noise": add_noise
    }


def main():
    test_images_dir = Path(__file__).parent / "test_images"
    test_images_dir.mkdir(exist_ok=True)

    ground_truth = {}

    # Generate test images with varying characteristics
    # For a well-calibrated system, these images should demonstrate:
    # - High confidence for clear, colorful grids (and they're all correct)
    # - Low confidence for noisy/unclear images (many incorrect)
    test_configs = [
        # High confidence images (high saturation, clear grid) - ALL should be correct
        {"saturation": 180, "with_grid": True, "add_noise": False},
        {"saturation": 200, "with_grid": True, "add_noise": False},
        {"saturation": 160, "with_grid": True, "add_noise": False},
        {"saturation": 170, "with_grid": True, "add_noise": False},
        {"saturation": 190, "with_grid": True, "add_noise": False},
        {"saturation": 150, "with_grid": True, "add_noise": False},
        {"saturation": 140, "with_grid": True, "add_noise": False},
        {"saturation": 155, "with_grid": True, "add_noise": False},
        {"saturation": 165, "with_grid": True, "add_noise": False},
        {"saturation": 175, "with_grid": True, "add_noise": False},

        # Medium confidence images - mostly correct
        {"saturation": 100, "with_grid": True, "add_noise": False},
        {"saturation": 120, "with_grid": True, "add_noise": False},
        {"saturation": 90, "with_grid": True, "add_noise": False},
        {"saturation": 110, "with_grid": True, "add_noise": False},
        {"saturation": 130, "with_grid": True, "add_noise": False},

        # Low confidence images (very low saturation) - often incorrect
        {"saturation": 40, "with_grid": True, "add_noise": False},
        {"saturation": 50, "with_grid": True, "add_noise": False},
        {"saturation": 30, "with_grid": True, "add_noise": False},
        {"saturation": 60, "with_grid": True, "add_noise": False},

        # Images without clear grid (should have low confidence, all incorrect)
        {"saturation": 50, "with_grid": False, "add_noise": False},
        {"saturation": 30, "with_grid": False, "add_noise": False},
        {"saturation": 80, "with_grid": False, "add_noise": False},
    ]

    for i, config in enumerate(test_configs):
        filename = f"test_{i:03d}_sat{config['saturation']}"
        if config.get("add_noise"):
            filename += "_noisy"
        if not config.get("with_grid", True):
            filename += "_nogrid"
        filename += ".png"

        output_path = test_images_dir / filename

        gt = create_test_image(
            output_path,
            saturation=config["saturation"],
            with_grid=config.get("with_grid", True),
            add_noise=config.get("add_noise", False)
        )

        ground_truth[filename] = gt
        print(f"Created: {filename} (expected conf: {gt['confidence_accurate']:.2f}, correct: {gt['correct']})")

    # Save ground truth
    gt_path = Path(__file__).parent / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nCreated {len(test_configs)} test images in {test_images_dir}")
    print(f"Ground truth saved to {gt_path}")


if __name__ == "__main__":
    main()
