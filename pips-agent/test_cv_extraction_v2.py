"""
Test suite for cv_extraction_v2 module.

Tests the multi-strategy puzzle cell detection system, with focus on
the diamond detection (constraint_labels) strategy.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path

# Import from the utils module
from utils.cv_extraction_v2 import (
    DetectionResult,
    detect_by_constraint_labels,
    detect_by_region_contours,
    detect_by_color_segmentation,
    extract_puzzle_multi_strategy,
    detect_diamonds,
    is_diamond_shape,
    infer_cells_from_diamonds,
    cluster_coordinates,
    calculate_median_spacing,
    estimate_grid_dims,
    calculate_confidence,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def blank_image_path(temp_dir):
    """Create a blank white image for testing error cases."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    path = os.path.join(temp_dir, "blank.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def diamond_grid_image_path(temp_dir):
    """
    Create a test image with a 3x3 grid of diamond markers.

    This simulates a puzzle with diamonds marking cell corners,
    which would create a 2x2 grid of cells.
    """
    # Create a white background
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255

    # Draw a 3x3 grid of diamonds (marking cell corners)
    # This creates a 2x2 grid of cells
    diamond_positions = [
        (100, 100), (200, 100), (300, 100),  # Top row
        (100, 200), (200, 200), (300, 200),  # Middle row
        (100, 300), (200, 300), (300, 300),  # Bottom row
    ]

    # Draw diamonds (rotated squares)
    diamond_size = 15
    for cx, cy in diamond_positions:
        pts = np.array([
            [cx, cy - diamond_size],  # Top
            [cx + diamond_size, cy],  # Right
            [cx, cy + diamond_size],  # Bottom
            [cx - diamond_size, cy],  # Left
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], (0, 0, 0))  # Black diamonds

    path = os.path.join(temp_dir, "diamond_grid.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def colored_puzzle_image_path(temp_dir):
    """
    Create a test image with colored regions (no diamonds).

    This tests that detect_by_constraint_labels correctly returns
    failure when no diamonds are present.
    """
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255

    # Draw colored rectangles representing puzzle cells
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
    ]

    # 2x2 grid of colored cells
    cell_size = 100
    for i, color in enumerate(colors):
        x = (i % 2) * cell_size + 100
        y = (i // 2) * cell_size + 100
        cv2.rectangle(img, (x, y), (x + cell_size, y + cell_size), color, -1)

    path = os.path.join(temp_dir, "colored_puzzle.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def single_diamond_image_path(temp_dir):
    """
    Create a test image with a single diamond (too few for grid inference).
    """
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255

    # Single diamond in center
    cx, cy = 100, 100
    diamond_size = 20
    pts = np.array([
        [cx, cy - diamond_size],
        [cx + diamond_size, cy],
        [cx, cy + diamond_size],
        [cx - diamond_size, cy],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 0))

    path = os.path.join(temp_dir, "single_diamond.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def sample_diamond_contour():
    """Create a sample diamond contour for unit testing is_diamond_shape()."""
    # Perfect diamond (square rotated 45 degrees)
    cx, cy = 50, 50
    size = 20
    pts = np.array([
        [[cx, cy - size]],      # Top
        [[cx + size, cy]],      # Right
        [[cx, cy + size]],      # Bottom
        [[cx - size, cy]],      # Left
    ], dtype=np.int32)
    return pts


@pytest.fixture
def sample_rectangle_contour():
    """Create a sample rectangle contour (not a diamond)."""
    pts = np.array([
        [[30, 30]],
        [[70, 30]],
        [[70, 50]],
        [[30, 50]],
    ], dtype=np.int32)
    return pts


# =============================================================================
# Basic Structure Tests
# =============================================================================

def test_basic_structure():
    """
    Verify that detect_by_constraint_labels exists and returns a DetectionResult.

    This is the most basic test to ensure the function is properly implemented
    and integrated into the module.
    """
    # Verify function exists and is callable
    assert callable(detect_by_constraint_labels), "detect_by_constraint_labels should be callable"

    # Verify DetectionResult dataclass exists with expected fields
    result = DetectionResult(
        success=False,
        cells=[],
        grid_dims=None,
        regions=None,
        confidence=0.0,
        method="test",
        error="test error"
    )

    assert hasattr(result, 'success')
    assert hasattr(result, 'cells')
    assert hasattr(result, 'grid_dims')
    assert hasattr(result, 'regions')
    assert hasattr(result, 'confidence')
    assert hasattr(result, 'method')
    assert hasattr(result, 'error')


def test_detection_result_dataclass():
    """Verify DetectionResult dataclass works correctly with all field types."""
    cells = [(10, 20, 30, 40), (50, 60, 70, 80)]
    grid_dims = (2, 3)
    regions = {0: [(0, 0), (0, 1)], 1: [(1, 0)]}

    result = DetectionResult(
        success=True,
        cells=cells,
        grid_dims=grid_dims,
        regions=regions,
        confidence=0.85,
        method="constraint_labels",
        error=None
    )

    assert result.success is True
    assert result.cells == cells
    assert result.grid_dims == grid_dims
    assert result.regions == regions
    assert result.confidence == 0.85
    assert result.method == "constraint_labels"
    assert result.error is None


def test_helper_functions_exist():
    """Verify all helper functions used by constraint_labels exist and are importable."""
    # These should all be importable without error
    assert callable(detect_diamonds)
    assert callable(is_diamond_shape)
    assert callable(infer_cells_from_diamonds)
    assert callable(cluster_coordinates)
    assert callable(calculate_median_spacing)
    assert callable(estimate_grid_dims)
    assert callable(calculate_confidence)


def test_all_strategies_exist():
    """Verify all three detection strategies exist and are callable."""
    assert callable(detect_by_region_contours)
    assert callable(detect_by_color_segmentation)
    assert callable(detect_by_constraint_labels)
    assert callable(extract_puzzle_multi_strategy)


# =============================================================================
# Error Handling Tests
# =============================================================================

def test_invalid_image_path():
    """Test that detect_by_constraint_labels handles invalid image path gracefully."""
    result = detect_by_constraint_labels("/nonexistent/path/image.png")

    assert isinstance(result, DetectionResult)
    assert result.success is False
    assert result.method == "constraint_labels"
    assert "Could not read image" in result.error
    assert result.confidence == 0.0
    assert result.cells == []


def test_blank_image_returns_failure(blank_image_path):
    """Test that a blank image (no diamonds) returns failure."""
    result = detect_by_constraint_labels(blank_image_path)

    assert isinstance(result, DetectionResult)
    assert result.success is False
    assert result.method == "constraint_labels"
    # Should fail due to too few diamonds or no cells inferred
    assert "Too few diamonds" in result.error or "No cells" in result.error


# =============================================================================
# Diamond Shape Detection Tests
# =============================================================================

def test_is_diamond_shape_valid(sample_diamond_contour):
    """Test that a proper diamond contour is recognized."""
    assert is_diamond_shape(sample_diamond_contour) is True


def test_is_diamond_shape_rectangle(sample_rectangle_contour):
    """Test that a rectangle is not recognized as a diamond."""
    assert is_diamond_shape(sample_rectangle_contour) is False


def test_is_diamond_shape_wrong_vertex_count():
    """Test that shapes with wrong vertex count are rejected."""
    # Triangle (3 vertices)
    triangle = np.array([
        [[50, 30]],
        [[70, 70]],
        [[30, 70]],
    ], dtype=np.int32)
    assert is_diamond_shape(triangle) is False

    # Pentagon (5 vertices)
    pentagon = np.array([
        [[50, 30]],
        [[70, 50]],
        [[60, 70]],
        [[40, 70]],
        [[30, 50]],
    ], dtype=np.int32)
    assert is_diamond_shape(pentagon) is False


# =============================================================================
# Coordinate Clustering Tests
# =============================================================================

def test_cluster_coordinates_basic():
    """Test basic coordinate clustering."""
    coords = [10, 12, 50, 52, 100, 102]
    clusters = cluster_coordinates(coords, tolerance=10)

    assert len(clusters) == 3
    assert abs(clusters[0] - 11) < 2  # Average of 10, 12
    assert abs(clusters[1] - 51) < 2  # Average of 50, 52
    assert abs(clusters[2] - 101) < 2  # Average of 100, 102


def test_cluster_coordinates_empty():
    """Test clustering with empty input."""
    clusters = cluster_coordinates([])
    assert clusters == []


def test_cluster_coordinates_single():
    """Test clustering with single coordinate."""
    clusters = cluster_coordinates([100])
    assert len(clusters) == 1
    assert clusters[0] == 100


def test_calculate_median_spacing():
    """Test median spacing calculation."""
    clusters = [100, 200, 300, 400]
    spacing = calculate_median_spacing(clusters)
    assert spacing == 100.0


def test_calculate_median_spacing_insufficient():
    """Test median spacing with insufficient data."""
    assert calculate_median_spacing([]) == 0
    assert calculate_median_spacing([100]) == 0


# =============================================================================
# Grid Dimension Estimation Tests
# =============================================================================

def test_estimate_grid_dims_basic():
    """Test grid dimension estimation from cells."""
    # 2x3 grid of cells
    cells = [
        (0, 0, 50, 50),    (60, 0, 50, 50),    (120, 0, 50, 50),
        (0, 60, 50, 50),   (60, 60, 50, 50),   (120, 60, 50, 50),
    ]
    dims = estimate_grid_dims(cells)

    assert dims is not None
    assert dims == (2, 3)  # 2 rows, 3 columns


def test_estimate_grid_dims_empty():
    """Test grid dimension estimation with no cells."""
    dims = estimate_grid_dims([])
    assert dims is None


# =============================================================================
# Confidence Calculation Tests
# =============================================================================

def test_calculate_confidence_ideal_range():
    """Test confidence is higher for ideal cell count (7-30)."""
    # Create 16 cells (4x4 grid) - ideal range
    cells = [(i * 50, j * 50, 45, 45) for i in range(4) for j in range(4)]
    grid_dims = (4, 4)

    confidence = calculate_confidence(cells, grid_dims)

    assert confidence > 0.5
    assert confidence <= 1.0


def test_calculate_confidence_no_cells():
    """Test confidence is 0 for no cells."""
    confidence = calculate_confidence([], None)
    assert confidence == 0.0


def test_calculate_confidence_range():
    """Test that confidence is always in valid range [0, 1]."""
    # Test various cell counts
    for count in [1, 5, 10, 20, 50, 100]:
        cells = [(i * 50, 0, 45, 45) for i in range(count)]
        confidence = calculate_confidence(cells, (1, count))

        assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range for {count} cells"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
