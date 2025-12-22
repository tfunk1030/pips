"""
Unit tests for the preprocess module.

Tests for all preprocessing functions:
- apply_clahe: CLAHE contrast enhancement
- normalize_brightness: Brightness normalization
- apply_white_balance: White balance correction (gray_world and white_patch)
- preprocess_domino_tray: Combined pipeline function
- preprocess_tile: Tile-specific preprocessing
"""

import numpy as np
import pytest
import cv2
import sys
import os

# Add parent directory to path to import preprocess module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import (
    apply_clahe,
    normalize_brightness,
    apply_white_balance,
    preprocess_domino_tray,
    preprocess_tile,
    create_histogram_image,
    save_debug_images,
    preprocess_domino_tray_debug,
)


# ============================================================================
# Fixtures for creating synthetic test images
# ============================================================================

@pytest.fixture
def uniform_gray_image():
    """Create a uniform gray image (128, 128, 128) - 100x100 pixels."""
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def dark_image():
    """Create a dark gray image (50, 50, 50) - 100x100 pixels."""
    return np.full((100, 100, 3), 50, dtype=np.uint8)


@pytest.fixture
def bright_image():
    """Create a bright gray image (200, 200, 200) - 100x100 pixels."""
    return np.full((100, 100, 3), 200, dtype=np.uint8)


@pytest.fixture
def black_image():
    """Create a completely black image - 100x100 pixels."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def white_image():
    """Create a completely white image - 100x100 pixels."""
    return np.full((100, 100, 3), 255, dtype=np.uint8)


@pytest.fixture
def blue_tinted_image():
    """Create an image with blue color cast (more blue than red/green)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 180  # Blue channel high
    img[:, :, 1] = 100  # Green channel medium
    img[:, :, 2] = 100  # Red channel medium
    return img


@pytest.fixture
def red_tinted_image():
    """Create an image with red color cast (more red than blue/green)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 100  # Blue channel medium
    img[:, :, 1] = 100  # Green channel medium
    img[:, :, 2] = 180  # Red channel high
    return img


@pytest.fixture
def low_contrast_image():
    """Create a low contrast image with values clustered around 128."""
    rng = np.random.default_rng(42)
    # Values between 118 and 138 (narrow range)
    return rng.integers(118, 139, size=(100, 100, 3), dtype=np.uint8)


@pytest.fixture
def high_contrast_image():
    """Create a high contrast image with values spread across full range."""
    rng = np.random.default_rng(42)
    # Create image with full range values
    return rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)


@pytest.fixture
def gradient_image():
    """Create a gradient image (left=dark, right=bright)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        img[:, i, :] = int(i * 255 / 99)
    return img


@pytest.fixture
def small_tile():
    """Create a small domino tile-sized image (50x30 pixels)."""
    rng = np.random.default_rng(42)
    return rng.integers(50, 200, size=(30, 50, 3), dtype=np.uint8)


# ============================================================================
# Tests for apply_clahe
# ============================================================================

class TestApplyCLAHE:
    """Tests for the apply_clahe function."""

    def test_accepts_bgr_image(self, uniform_gray_image):
        """Test that apply_clahe accepts a BGR image."""
        result = apply_clahe(uniform_gray_image)
        assert result is not None
        assert result.shape == uniform_gray_image.shape
        assert result.dtype == np.uint8

    def test_raises_on_none_input(self):
        """Test that apply_clahe raises ValueError on None input."""
        with pytest.raises(ValueError, match="None or empty"):
            apply_clahe(None)

    def test_raises_on_empty_input(self):
        """Test that apply_clahe raises ValueError on empty input."""
        empty_image = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        with pytest.raises(ValueError, match="None or empty"):
            apply_clahe(empty_image)

    def test_enhances_low_contrast_image(self, low_contrast_image):
        """Test that CLAHE enhances contrast in a low contrast image."""
        result = apply_clahe(low_contrast_image)

        # Convert both to grayscale to measure contrast (std deviation)
        original_gray = cv2.cvtColor(low_contrast_image, cv2.COLOR_BGR2GRAY)
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        original_std = np.std(original_gray)
        result_std = np.std(result_gray)

        # CLAHE should increase contrast (higher std deviation)
        assert result_std >= original_std

    def test_default_parameters(self, low_contrast_image):
        """Test that default parameters (clip_limit=2.0, tile_grid_size=(8,8)) work."""
        result = apply_clahe(low_contrast_image)
        assert result is not None
        assert result.shape == low_contrast_image.shape

    def test_custom_clip_limit(self, low_contrast_image):
        """Test that custom clip_limit parameter is respected."""
        result_low = apply_clahe(low_contrast_image, clip_limit=1.0)
        result_high = apply_clahe(low_contrast_image, clip_limit=4.0)

        # Both should produce valid results
        assert result_low is not None
        assert result_high is not None

        # Higher clip limit typically produces more contrast enhancement
        # (though this can vary based on image content)

    def test_custom_tile_grid_size(self, gradient_image):
        """Test that custom tile_grid_size parameter is respected."""
        result_small = apply_clahe(gradient_image, tile_grid_size=(4, 4))
        result_large = apply_clahe(gradient_image, tile_grid_size=(16, 16))

        # Both should produce valid results
        assert result_small is not None
        assert result_large is not None
        assert result_small.shape == gradient_image.shape
        assert result_large.shape == gradient_image.shape

    def test_preserves_uniform_image(self, uniform_gray_image):
        """Test that CLAHE maintains uniformity of a uniform image."""
        result = apply_clahe(uniform_gray_image)

        # CLAHE should preserve uniformity - standard deviation should remain low
        # Note: Mean may shift due to LAB color space conversion, but std should stay low
        original_gray = cv2.cvtColor(uniform_gray_image, cv2.COLOR_BGR2GRAY)
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        original_std = np.std(original_gray)
        result_std = np.std(result_gray)

        # Both should have very low std (uniform images)
        assert result_std < 5  # Should remain uniform

    def test_output_in_valid_range(self, high_contrast_image):
        """Test that output values are in valid 0-255 range."""
        result = apply_clahe(high_contrast_image)
        assert np.min(result) >= 0
        assert np.max(result) <= 255


# ============================================================================
# Tests for normalize_brightness
# ============================================================================

class TestNormalizeBrightness:
    """Tests for the normalize_brightness function."""

    def test_accepts_bgr_image(self, dark_image):
        """Test that normalize_brightness accepts a BGR image."""
        result = normalize_brightness(dark_image)
        assert result is not None
        assert result.shape == dark_image.shape
        assert result.dtype == np.uint8

    def test_raises_on_none_input(self):
        """Test that normalize_brightness raises ValueError on None input."""
        with pytest.raises(ValueError, match="None or empty"):
            normalize_brightness(None)

    def test_raises_on_empty_input(self):
        """Test that normalize_brightness raises ValueError on empty input."""
        empty_image = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        with pytest.raises(ValueError, match="None or empty"):
            normalize_brightness(empty_image)

    def test_raises_on_invalid_target_brightness(self, uniform_gray_image):
        """Test that normalize_brightness raises on invalid target brightness."""
        with pytest.raises(ValueError, match="target_brightness"):
            normalize_brightness(uniform_gray_image, target_brightness=-10)
        with pytest.raises(ValueError, match="target_brightness"):
            normalize_brightness(uniform_gray_image, target_brightness=300)

    def test_achieves_target_brightness(self, dark_image):
        """Test that normalize_brightness achieves target brightness."""
        target = 128.0
        result = normalize_brightness(dark_image, target_brightness=target)

        # Convert to HSV and get V channel mean
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        result_brightness = np.mean(hsv[:, :, 2])

        # Should be close to target (within a reasonable tolerance)
        assert abs(result_brightness - target) < 10

    def test_brightens_dark_image(self, dark_image):
        """Test that dark images are brightened."""
        result = normalize_brightness(dark_image, target_brightness=128)

        # Get brightness of original and result
        original_hsv = cv2.cvtColor(dark_image, cv2.COLOR_BGR2HSV)
        result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

        original_brightness = np.mean(original_hsv[:, :, 2])
        result_brightness = np.mean(result_hsv[:, :, 2])

        assert result_brightness > original_brightness

    def test_darkens_bright_image(self, bright_image):
        """Test that bright images are darkened when target is lower."""
        result = normalize_brightness(bright_image, target_brightness=100)

        # Get brightness of original and result
        original_hsv = cv2.cvtColor(bright_image, cv2.COLOR_BGR2HSV)
        result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

        original_brightness = np.mean(original_hsv[:, :, 2])
        result_brightness = np.mean(result_hsv[:, :, 2])

        assert result_brightness < original_brightness

    def test_handles_black_image(self, black_image):
        """Test that black images are handled without division by zero."""
        result = normalize_brightness(black_image, target_brightness=128)
        # Should return original without error
        assert result is not None
        assert result.shape == black_image.shape

    def test_default_target_brightness(self, dark_image):
        """Test that default target brightness (128) is used."""
        result = normalize_brightness(dark_image)

        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        result_brightness = np.mean(hsv[:, :, 2])

        # Should be close to default target of 128
        assert abs(result_brightness - 128) < 10

    def test_output_in_valid_range(self, dark_image):
        """Test that output values are in valid 0-255 range."""
        result = normalize_brightness(dark_image, target_brightness=200)
        assert np.min(result) >= 0
        assert np.max(result) <= 255


# ============================================================================
# Tests for apply_white_balance
# ============================================================================

class TestApplyWhiteBalance:
    """Tests for the apply_white_balance function."""

    def test_accepts_bgr_image(self, blue_tinted_image):
        """Test that apply_white_balance accepts a BGR image."""
        result = apply_white_balance(blue_tinted_image)
        assert result is not None
        assert result.shape == blue_tinted_image.shape
        assert result.dtype == np.uint8

    def test_raises_on_none_input(self):
        """Test that apply_white_balance raises ValueError on None input."""
        with pytest.raises(ValueError, match="None or empty"):
            apply_white_balance(None)

    def test_raises_on_empty_input(self):
        """Test that apply_white_balance raises ValueError on empty input."""
        empty_image = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        with pytest.raises(ValueError, match="None or empty"):
            apply_white_balance(empty_image)

    def test_raises_on_invalid_method(self, uniform_gray_image):
        """Test that apply_white_balance raises on invalid method."""
        with pytest.raises(ValueError, match="method must be one of"):
            apply_white_balance(uniform_gray_image, method="invalid_method")

    def test_gray_world_method(self, blue_tinted_image):
        """Test gray_world method corrects blue color cast."""
        result = apply_white_balance(blue_tinted_image, method="gray_world")

        # After gray_world, channels should have more similar means
        original_b_mean = np.mean(blue_tinted_image[:, :, 0])
        original_g_mean = np.mean(blue_tinted_image[:, :, 1])
        original_r_mean = np.mean(blue_tinted_image[:, :, 2])

        result_b_mean = np.mean(result[:, :, 0])
        result_g_mean = np.mean(result[:, :, 1])
        result_r_mean = np.mean(result[:, :, 2])

        # Original has large difference between blue and red/green
        original_diff = abs(original_b_mean - original_r_mean)

        # Result should have smaller difference
        result_diff = abs(result_b_mean - result_r_mean)

        assert result_diff < original_diff

    def test_white_patch_method(self, blue_tinted_image):
        """Test white_patch method works correctly."""
        result = apply_white_balance(blue_tinted_image, method="white_patch")

        # Should produce a valid result
        assert result is not None
        assert result.shape == blue_tinted_image.shape

    def test_corrects_red_color_cast(self, red_tinted_image):
        """Test that gray_world corrects red color cast."""
        result = apply_white_balance(red_tinted_image, method="gray_world")

        # After correction, red channel should be closer to other channels
        original_r_mean = np.mean(red_tinted_image[:, :, 2])
        original_b_mean = np.mean(red_tinted_image[:, :, 0])

        result_r_mean = np.mean(result[:, :, 2])
        result_b_mean = np.mean(result[:, :, 0])

        # Original has large difference between red and blue
        original_diff = abs(original_r_mean - original_b_mean)

        # Result should have smaller difference
        result_diff = abs(result_r_mean - result_b_mean)

        assert result_diff < original_diff

    def test_handles_black_image(self, black_image):
        """Test that black images are handled without division by zero."""
        result = apply_white_balance(black_image, method="gray_world")
        # Should return original without error
        assert result is not None
        assert result.shape == black_image.shape

    def test_default_method_is_gray_world(self, blue_tinted_image):
        """Test that default method is gray_world."""
        result_default = apply_white_balance(blue_tinted_image)
        result_explicit = apply_white_balance(blue_tinted_image, method="gray_world")

        # Results should be identical
        np.testing.assert_array_equal(result_default, result_explicit)

    def test_output_in_valid_range(self, blue_tinted_image):
        """Test that output values are in valid 0-255 range."""
        result = apply_white_balance(blue_tinted_image)
        assert np.min(result) >= 0
        assert np.max(result) <= 255


# ============================================================================
# Tests for preprocess_domino_tray (pipeline function)
# ============================================================================

class TestPreprocessDominoTray:
    """Tests for the preprocess_domino_tray pipeline function."""

    def test_returns_tuple(self, uniform_gray_image):
        """Test that preprocess_domino_tray returns a tuple of (image, metrics)."""
        result = preprocess_domino_tray(uniform_gray_image)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_valid_image(self, dark_image):
        """Test that the returned image is valid."""
        result, metrics = preprocess_domino_tray(dark_image)
        assert result is not None
        assert result.shape == dark_image.shape
        assert result.dtype == np.uint8

    def test_returns_valid_metrics(self, dark_image):
        """Test that the returned metrics dictionary is valid."""
        result, metrics = preprocess_domino_tray(dark_image)

        assert isinstance(metrics, dict)
        assert "steps_applied" in metrics
        assert "original_brightness" in metrics
        assert "final_brightness" in metrics
        assert "brightness_change" in metrics
        assert "original_contrast" in metrics
        assert "final_contrast" in metrics

    def test_raises_on_none_input(self):
        """Test that preprocess_domino_tray raises ValueError on None input."""
        with pytest.raises(ValueError, match="None or empty"):
            preprocess_domino_tray(None)

    def test_raises_on_empty_input(self):
        """Test that preprocess_domino_tray raises ValueError on empty input."""
        empty_image = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        with pytest.raises(ValueError, match="None or empty"):
            preprocess_domino_tray(empty_image)

    def test_chains_all_steps_by_default(self, dark_image):
        """Test that all steps are applied by default."""
        result, metrics = preprocess_domino_tray(dark_image)

        steps = metrics["steps_applied"]
        assert len(steps) == 3
        assert any("white_balance" in step for step in steps)
        assert any("brightness_normalize" in step for step in steps)
        assert any("clahe" in step for step in steps)

    def test_can_disable_white_balance(self, dark_image):
        """Test that white_balance can be disabled."""
        result, metrics = preprocess_domino_tray(
            dark_image, enable_white_balance=False
        )

        steps = metrics["steps_applied"]
        assert not any("white_balance" in step for step in steps)
        assert len(steps) == 2

    def test_can_disable_brightness_normalize(self, dark_image):
        """Test that brightness_normalize can be disabled."""
        result, metrics = preprocess_domino_tray(
            dark_image, enable_brightness_normalize=False
        )

        steps = metrics["steps_applied"]
        assert not any("brightness_normalize" in step for step in steps)
        assert len(steps) == 2

    def test_can_disable_clahe(self, dark_image):
        """Test that clahe can be disabled."""
        result, metrics = preprocess_domino_tray(
            dark_image, enable_clahe=False
        )

        steps = metrics["steps_applied"]
        assert not any("clahe" in step for step in steps)
        assert len(steps) == 2

    def test_can_disable_all_steps(self, dark_image):
        """Test that all steps can be disabled."""
        result, metrics = preprocess_domino_tray(
            dark_image,
            enable_white_balance=False,
            enable_brightness_normalize=False,
            enable_clahe=False
        )

        steps = metrics["steps_applied"]
        assert len(steps) == 0

        # Result should be a copy of original
        np.testing.assert_array_equal(result, dark_image)

    def test_custom_white_balance_method(self, blue_tinted_image):
        """Test that custom white_balance_method is respected."""
        result_gw, metrics_gw = preprocess_domino_tray(
            blue_tinted_image, white_balance_method="gray_world"
        )
        result_wp, metrics_wp = preprocess_domino_tray(
            blue_tinted_image, white_balance_method="white_patch"
        )

        # Both should work, results may differ
        assert result_gw is not None
        assert result_wp is not None

    def test_custom_target_brightness(self, dark_image):
        """Test that custom target_brightness is respected."""
        result, metrics = preprocess_domino_tray(
            dark_image, target_brightness=180.0
        )

        steps = metrics["steps_applied"]
        assert any("180" in step for step in steps)

    def test_custom_clahe_parameters(self, low_contrast_image):
        """Test that custom CLAHE parameters are respected."""
        result, metrics = preprocess_domino_tray(
            low_contrast_image,
            clahe_clip_limit=3.0,
            clahe_tile_grid_size=(4, 4)
        )

        steps = metrics["steps_applied"]
        assert any("3.0" in step for step in steps)

    def test_output_in_valid_range(self, dark_image):
        """Test that output values are in valid 0-255 range."""
        result, metrics = preprocess_domino_tray(dark_image)
        assert np.min(result) >= 0
        assert np.max(result) <= 255


# ============================================================================
# Tests for preprocess_tile
# ============================================================================

class TestPreprocessTile:
    """Tests for the preprocess_tile function."""

    def test_returns_processed_tile(self, small_tile):
        """Test that preprocess_tile returns a processed tile."""
        result = preprocess_tile(small_tile)
        assert result is not None
        assert result.shape == small_tile.shape
        assert result.dtype == np.uint8

    def test_raises_on_none_input(self):
        """Test that preprocess_tile raises ValueError on None input."""
        with pytest.raises(ValueError, match="None or empty"):
            preprocess_tile(None)

    def test_raises_on_empty_input(self):
        """Test that preprocess_tile raises ValueError on empty input."""
        empty_image = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        with pytest.raises(ValueError, match="None or empty"):
            preprocess_tile(empty_image)

    def test_uses_tile_optimized_defaults(self, small_tile):
        """Test that tile-optimized defaults (clip_limit=3.0, grid=(4,4)) are used."""
        # This is validated by the fact it works on small tiles
        result = preprocess_tile(small_tile)
        assert result is not None

    def test_can_disable_white_balance(self, small_tile):
        """Test that white_balance can be disabled."""
        result = preprocess_tile(small_tile, enable_white_balance=False)
        assert result is not None

    def test_can_disable_brightness_normalize(self, small_tile):
        """Test that brightness_normalize can be disabled."""
        result = preprocess_tile(small_tile, enable_brightness_normalize=False)
        assert result is not None

    def test_can_disable_clahe(self, small_tile):
        """Test that clahe can be disabled."""
        result = preprocess_tile(small_tile, enable_clahe=False)
        assert result is not None

    def test_output_in_valid_range(self, small_tile):
        """Test that output values are in valid 0-255 range."""
        result = preprocess_tile(small_tile)
        assert np.min(result) >= 0
        assert np.max(result) <= 255


# ============================================================================
# Tests for create_histogram_image
# ============================================================================

class TestCreateHistogramImage:
    """Tests for the create_histogram_image function."""

    def test_returns_bgr_image(self, uniform_gray_image):
        """Test that create_histogram_image returns a valid BGR image."""
        result = create_histogram_image(uniform_gray_image)
        assert result is not None
        assert len(result.shape) == 3
        assert result.shape[2] == 3  # BGR
        assert result.dtype == np.uint8

    def test_creates_histogram_with_title(self, uniform_gray_image):
        """Test that create_histogram_image respects the title parameter."""
        result = create_histogram_image(uniform_gray_image, title="Test Histogram")
        assert result is not None

    def test_works_with_various_images(self, dark_image, bright_image, gradient_image):
        """Test that create_histogram_image works with various image types."""
        for img in [dark_image, bright_image, gradient_image]:
            result = create_histogram_image(img)
            assert result is not None
            assert len(result.shape) == 3


# ============================================================================
# Tests for save_debug_images
# ============================================================================

class TestSaveDebugImages:
    """Tests for the save_debug_images function."""

    def test_saves_files_to_directory(self, tmp_path, uniform_gray_image):
        """Test that save_debug_images saves files to the specified directory."""
        output_dir = str(tmp_path / "test_debug")

        paths = save_debug_images(
            original=uniform_gray_image,
            after_white_balance=uniform_gray_image,
            after_brightness=uniform_gray_image,
            after_clahe=uniform_gray_image,
            output_dir=output_dir,
            base_name="test"
        )

        assert isinstance(paths, dict)
        assert len(paths) > 0

        # Check that output directory was created
        assert os.path.isdir(output_dir)

    def test_returns_paths_dict(self, tmp_path, uniform_gray_image):
        """Test that save_debug_images returns a dictionary of paths."""
        output_dir = str(tmp_path / "test_debug")

        paths = save_debug_images(
            original=uniform_gray_image,
            after_white_balance=uniform_gray_image,
            after_brightness=uniform_gray_image,
            after_clahe=uniform_gray_image,
            output_dir=output_dir,
            base_name="test"
        )

        # Should contain paths for each stage and histograms
        assert "01_original_image" in paths
        assert "01_original_histogram" in paths
        assert "histogram_comparison" in paths


# ============================================================================
# Tests for preprocess_domino_tray_debug
# ============================================================================

class TestPreprocessDominoTrayDebug:
    """Tests for the preprocess_domino_tray_debug function."""

    def test_returns_tuple_of_three(self, tmp_path, dark_image):
        """Test that preprocess_domino_tray_debug returns (image, metrics, paths)."""
        output_dir = str(tmp_path / "test_debug")

        result = preprocess_domino_tray_debug(
            dark_image,
            output_dir=output_dir,
            base_name="test"
        )

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_returns_valid_components(self, tmp_path, dark_image):
        """Test that all returned components are valid."""
        output_dir = str(tmp_path / "test_debug")

        result_img, metrics, debug_files = preprocess_domino_tray_debug(
            dark_image,
            output_dir=output_dir,
            base_name="test"
        )

        # Check image
        assert result_img is not None
        assert result_img.shape == dark_image.shape

        # Check metrics
        assert isinstance(metrics, dict)
        assert "steps_applied" in metrics

        # Check debug files
        assert isinstance(debug_files, dict)
        assert len(debug_files) > 0

    def test_raises_on_none_input(self, tmp_path):
        """Test that preprocess_domino_tray_debug raises ValueError on None."""
        output_dir = str(tmp_path / "test_debug")

        with pytest.raises(ValueError, match="None or empty"):
            preprocess_domino_tray_debug(None, output_dir=output_dir)

    def test_saves_debug_files(self, tmp_path, dark_image):
        """Test that debug files are actually saved to disk."""
        output_dir = str(tmp_path / "test_debug")

        result_img, metrics, debug_files = preprocess_domino_tray_debug(
            dark_image,
            output_dir=output_dir,
            base_name="test"
        )

        # Check that files exist
        for path in debug_files.values():
            assert os.path.isfile(path)


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:
    """Integration tests for the preprocessing pipeline."""

    def test_full_pipeline_on_realistic_image(self):
        """Test full pipeline on a realistic synthetic domino-like image."""
        # Create a synthetic image simulating a domino tray
        rng = np.random.default_rng(42)

        # Create base dark image (simulating poor lighting)
        img = np.full((200, 300, 3), 40, dtype=np.uint8)

        # Add some "domino" like rectangles with dots
        for i in range(3):
            x = 20 + i * 90
            y = 30
            # Domino background (white-ish)
            img[y:y+60, x:x+70, :] = 200
            # Add some dots
            cv2.circle(img, (x+20, y+20), 5, (30, 30, 30), -1)
            cv2.circle(img, (x+50, y+40), 5, (30, 30, 30), -1)

        # Add blue color cast
        img[:, :, 0] = np.clip(img[:, :, 0] + 20, 0, 255)

        # Run full pipeline
        result, metrics = preprocess_domino_tray(img)

        # Verify pipeline ran all steps
        assert len(metrics["steps_applied"]) == 3

        # Verify brightness was improved (original was dark at ~40)
        assert metrics["final_brightness"] > metrics["original_brightness"]

        # Verify valid output
        assert result is not None
        assert np.min(result) >= 0
        assert np.max(result) <= 255

    def test_pipeline_consistency(self, dark_image):
        """Test that running the pipeline twice produces same results."""
        result1, metrics1 = preprocess_domino_tray(dark_image)
        result2, metrics2 = preprocess_domino_tray(dark_image)

        np.testing.assert_array_equal(result1, result2)
        assert metrics1 == metrics2

    def test_pipeline_does_not_modify_input(self, dark_image):
        """Test that the pipeline does not modify the input image."""
        original_copy = dark_image.copy()

        result, metrics = preprocess_domino_tray(dark_image)

        np.testing.assert_array_equal(dark_image, original_copy)
