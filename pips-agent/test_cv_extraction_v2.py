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
# Diamond Detection Strategy Tests (detect_by_constraint_labels)
# =============================================================================

def test_detect_by_constraint_labels_basic(diamond_grid_image_path, temp_dir):
    """
    Test that detect_by_constraint_labels returns DetectionResult with correct structure.

    This validates the function exists, is callable, and returns the expected
    dataclass type with all required fields.
    """
    result = detect_by_constraint_labels(diamond_grid_image_path, temp_dir)

    # Verify return type
    assert isinstance(result, DetectionResult), "Should return DetectionResult dataclass"

    # Verify all required fields exist
    assert hasattr(result, 'success'), "Result should have 'success' field"
    assert hasattr(result, 'cells'), "Result should have 'cells' field"
    assert hasattr(result, 'grid_dims'), "Result should have 'grid_dims' field"
    assert hasattr(result, 'regions'), "Result should have 'regions' field"
    assert hasattr(result, 'confidence'), "Result should have 'confidence' field"
    assert hasattr(result, 'method'), "Result should have 'method' field"
    assert hasattr(result, 'error'), "Result should have 'error' field"

    # Verify method is correct
    assert result.method == "constraint_labels", "Method should be 'constraint_labels'"

    # Verify confidence is in valid range
    assert 0.0 <= result.confidence <= 1.0, "Confidence should be between 0 and 1"

    # Verify cells is a list
    assert isinstance(result.cells, list), "Cells should be a list"


def test_detect_by_constraint_labels_no_diamonds(colored_puzzle_image_path, temp_dir):
    """
    Test that detect_by_constraint_labels returns failure when no diamonds are present.

    When the image contains no diamond markers, the function should return a
    DetectionResult with success=False and an appropriate error message.
    """
    result = detect_by_constraint_labels(colored_puzzle_image_path, temp_dir)

    assert isinstance(result, DetectionResult)
    assert result.success is False, "Should fail when no diamonds present"
    assert result.method == "constraint_labels"
    assert result.confidence == 0.0, "Confidence should be 0 when detection fails"
    assert result.cells == [], "Cells should be empty list when detection fails"
    assert result.error is not None, "Error message should be set"
    assert "Too few diamonds" in result.error or "No cells" in result.error, \
        "Error should mention diamond detection failure"


def test_detect_by_constraint_labels_with_diamonds(diamond_grid_image_path, temp_dir):
    """
    Test that detect_by_constraint_labels successfully detects diamonds in test image.

    The diamond_grid_image_path fixture creates a 3x3 grid of diamonds that
    mark cell corners, which should result in a 2x2 grid of 4 cells.
    """
    result = detect_by_constraint_labels(diamond_grid_image_path, temp_dir)

    assert isinstance(result, DetectionResult)
    assert result.success is True, f"Detection should succeed. Error: {result.error}"
    assert result.method == "constraint_labels"

    # The 3x3 diamond grid should produce 4 cells (2x2 grid)
    assert len(result.cells) > 0, "Should detect cells from diamond markers"
    assert len(result.cells) == 4, f"Expected 4 cells for 3x3 diamond grid, got {len(result.cells)}"

    # Verify grid dimensions
    assert result.grid_dims is not None, "Grid dimensions should be calculated"
    assert result.grid_dims == (2, 2), f"Expected (2, 2) grid, got {result.grid_dims}"

    # Verify confidence is reasonable
    assert result.confidence > 0.3, f"Confidence {result.confidence} is too low"

    # Verify regions were detected
    assert result.regions is not None, "Regions should be calculated"

    # Verify each cell has valid bounding box format (x, y, w, h)
    for cell in result.cells:
        assert len(cell) == 4, "Each cell should have 4 values (x, y, w, h)"
        x, y, w, h = cell
        assert w > 0 and h > 0, "Cell width and height should be positive"


def test_detect_by_constraint_labels_inference(temp_dir):
    """
    Test that cell positions are correctly inferred from diamond positions.

    Creates a custom test image with known diamond positions and verifies
    that the inferred cells are positioned correctly.
    """
    # Create a test image with a 4x4 grid of diamonds (16 diamonds)
    # This should produce a 3x3 grid of 9 cells
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # Draw 4x4 grid of diamonds at regular intervals
    diamond_positions = []
    for row in range(4):
        for col in range(4):
            cx = 75 + col * 100  # Start at 75, space by 100 pixels
            cy = 75 + row * 100
            diamond_positions.append((cx, cy))

    # Draw diamonds
    diamond_size = 12
    for cx, cy in diamond_positions:
        pts = np.array([
            [cx, cy - diamond_size],  # Top
            [cx + diamond_size, cy],  # Right
            [cx, cy + diamond_size],  # Bottom
            [cx - diamond_size, cy],  # Left
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], (0, 0, 0))

    # Save test image
    test_image_path = os.path.join(temp_dir, "inference_test.png")
    cv2.imwrite(test_image_path, img)

    # Run detection
    result = detect_by_constraint_labels(test_image_path, temp_dir)

    assert result.success is True, f"Detection should succeed. Error: {result.error}"

    # Should produce 3x3 = 9 cells from 4x4 = 16 corner diamonds
    expected_cell_count = 9
    assert len(result.cells) == expected_cell_count, \
        f"Expected {expected_cell_count} cells from 4x4 diamond grid, got {len(result.cells)}"

    # Verify grid dimensions
    assert result.grid_dims == (3, 3), f"Expected (3, 3) grid, got {result.grid_dims}"

    # Verify cells are positioned correctly
    # With diamonds at 75, 175, 275, 375 pixels,
    # cells should span between adjacent diamond positions
    for x, y, w, h in result.cells:
        # Cell width should be approximately 100 pixels (diamond spacing)
        assert 80 <= w <= 120, f"Cell width {w} is unexpected"
        assert 80 <= h <= 120, f"Cell height {h} is unexpected"

        # Cell positions should be near diamond positions
        assert 50 <= x <= 350, f"Cell x position {x} is out of expected range"
        assert 50 <= y <= 350, f"Cell y position {y} is out of expected range"


def test_detect_by_constraint_labels_too_few_diamonds(single_diamond_image_path, temp_dir):
    """
    Test that detection fails gracefully with insufficient diamonds.

    When fewer than 4 diamonds are detected, the function should return
    failure since a grid cannot be inferred from too few points.
    """
    result = detect_by_constraint_labels(single_diamond_image_path, temp_dir)

    assert isinstance(result, DetectionResult)
    assert result.success is False, "Should fail with too few diamonds"
    assert result.method == "constraint_labels"
    assert result.confidence == 0.0
    assert "Too few diamonds" in result.error, \
        f"Error should mention too few diamonds, got: {result.error}"


def test_detect_by_constraint_labels_debug_output(diamond_grid_image_path, temp_dir):
    """
    Test that debug output is saved when debug_dir is provided.
    """
    result = detect_by_constraint_labels(diamond_grid_image_path, temp_dir)

    # Only check debug output if detection succeeded
    if result.success:
        debug_image_path = os.path.join(temp_dir, "constraint_labels_method.png")
        assert os.path.exists(debug_image_path), "Debug image should be saved"

        # Verify it's a valid image
        debug_img = cv2.imread(debug_image_path)
        assert debug_img is not None, "Debug image should be readable"


def test_detect_diamonds_function(diamond_grid_image_path):
    """
    Test the detect_diamonds() helper function directly.
    """
    img = cv2.imread(diamond_grid_image_path)
    assert img is not None

    diamonds = detect_diamonds(img)

    # Should detect 9 diamonds (3x3 grid)
    assert len(diamonds) == 9, f"Expected 9 diamonds, found {len(diamonds)}"

    # Each diamond should have (cx, cy, w, h, contour)
    for d in diamonds:
        assert len(d) == 5, "Diamond tuple should have 5 elements"
        cx, cy, w, h, contour = d
        assert w > 0 and h > 0, "Diamond dimensions should be positive"
        assert contour is not None, "Contour should not be None"


def test_infer_cells_from_diamonds_corner_pattern():
    """
    Test infer_cells_from_diamonds with a corner pattern.
    """
    # 3x3 grid of diamond centers at corners (100 pixel spacing)
    diamond_centers = [
        (100, 100), (200, 100), (300, 100),
        (100, 200), (200, 200), (300, 200),
        (100, 300), (200, 300), (300, 300),
    ]
    img_shape = (400, 400, 3)

    cells = infer_cells_from_diamonds(diamond_centers, img_shape)

    # Should produce 2x2 = 4 cells
    assert len(cells) == 4, f"Expected 4 cells, got {len(cells)}"

    # Each cell should be roughly 100x100 pixels
    for x, y, w, h in cells:
        assert 80 <= w <= 120, f"Cell width {w} unexpected"
        assert 80 <= h <= 120, f"Cell height {h} unexpected"


def test_infer_cells_from_diamonds_insufficient():
    """
    Test that infer_cells_from_diamonds returns empty list with too few diamonds.
    """
    # Only 3 diamonds - not enough to infer a grid
    diamond_centers = [(100, 100), (200, 100), (100, 200)]
    img_shape = (400, 400, 3)

    cells = infer_cells_from_diamonds(diamond_centers, img_shape)

    assert cells == [], "Should return empty list with fewer than 4 diamonds"


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


# =============================================================================
# Multi-Strategy Integration Tests
# =============================================================================

def test_multi_strategy_includes_constraint_labels(diamond_grid_image_path, temp_dir):
    """
    Test that the multi-strategy system includes constraint_labels strategy.

    This integration test verifies:
    1. constraint_labels strategy is included in the default strategy list
    2. All strategies run and contribute to results
    3. Best result is selected by confidence
    4. The result structure is correct
    """
    # Run multi-strategy extraction
    result = extract_puzzle_multi_strategy(diamond_grid_image_path, temp_dir)

    # Verify result structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "success" in result, "Result should have 'success' field"
    assert "method_used" in result, "Result should have 'method_used' field"
    assert "cells" in result, "Result should have 'cells' field"
    assert "grid_dims" in result, "Result should have 'grid_dims' field"
    assert "regions" in result, "Result should have 'regions' field"
    assert "confidence" in result, "Result should have 'confidence' field"
    assert "num_cells" in result, "Result should have 'num_cells' field"
    assert "all_attempts" in result, "Result should have 'all_attempts' field"

    # Verify all_attempts contains results from all strategies
    all_attempts = result["all_attempts"]
    assert isinstance(all_attempts, list), "all_attempts should be a list"
    assert len(all_attempts) == 3, f"Expected 3 strategy attempts, got {len(all_attempts)}"

    # Verify constraint_labels is in all_attempts
    methods = [attempt["method"] for attempt in all_attempts]
    assert "constraint_labels" in methods, "constraint_labels should be in all_attempts"
    assert "region_contours" in methods, "region_contours should be in all_attempts"
    assert "color_segmentation" in methods, "color_segmentation should be in all_attempts"

    # Verify each attempt has the expected structure
    for attempt in all_attempts:
        assert "method" in attempt, "Each attempt should have 'method' field"
        assert "success" in attempt, "Each attempt should have 'success' field"
        assert "confidence" in attempt, "Each attempt should have 'confidence' field"
        assert "error" in attempt, "Each attempt should have 'error' field"
        assert "num_cells" in attempt, "Each attempt should have 'num_cells' field"

    # Verify constraint_labels detected the diamonds in our test image
    constraint_labels_attempt = next(
        (a for a in all_attempts if a["method"] == "constraint_labels"), None
    )
    assert constraint_labels_attempt is not None
    assert constraint_labels_attempt["success"] is True, \
        f"constraint_labels should succeed on diamond image. Error: {constraint_labels_attempt['error']}"
    assert constraint_labels_attempt["num_cells"] == 4, \
        f"constraint_labels should detect 4 cells, got {constraint_labels_attempt['num_cells']}"

    # Verify that if successful, best method is selected by confidence
    if result["success"]:
        successful_attempts = [a for a in all_attempts if a["success"]]
        if len(successful_attempts) > 0:
            max_confidence = max(a["confidence"] for a in successful_attempts)
            assert result["confidence"] == max_confidence, \
                f"Best result confidence {result['confidence']} should match max {max_confidence}"


def test_multi_strategy_constraint_labels_wins_on_diamond_image(diamond_grid_image_path, temp_dir):
    """
    Test that constraint_labels strategy wins (is selected) on a diamond-marked puzzle.

    When provided with an image containing clear diamond markers and no colored
    regions, constraint_labels should produce the highest confidence result.
    """
    result = extract_puzzle_multi_strategy(diamond_grid_image_path, temp_dir)

    # On a pure diamond grid image, constraint_labels should win
    assert result["success"] is True, f"Multi-strategy should succeed. Error: {result.get('error')}"
    assert result["method_used"] == "constraint_labels", \
        f"constraint_labels should win on diamond image, but {result['method_used']} was selected"
    assert result["num_cells"] == 4, f"Expected 4 cells, got {result['num_cells']}"


def test_multi_strategy_all_fail(blank_image_path, temp_dir):
    """
    Test multi-strategy behavior when all strategies fail.

    On a blank image with no puzzle features, all strategies should fail
    and the result should indicate failure with appropriate error message.
    """
    result = extract_puzzle_multi_strategy(blank_image_path, temp_dir)

    assert result["success"] is False, "Should fail on blank image"
    assert result["method_used"] is None, "No method should be selected when all fail"
    assert result["confidence"] == 0.0, "Confidence should be 0 when all fail"
    assert result["num_cells"] == 0, "Should have 0 cells when all fail"
    assert "error" in result, "Should have error message"

    # Verify all strategies were attempted and failed
    all_attempts = result["all_attempts"]
    assert len(all_attempts) == 3, "Should have 3 attempts"
    for attempt in all_attempts:
        assert attempt["success"] is False, f"{attempt['method']} should have failed on blank image"


def test_multi_strategy_explicit_strategies_list(diamond_grid_image_path, temp_dir):
    """
    Test that providing an explicit strategies list works correctly.
    """
    # Test with only constraint_labels
    result = extract_puzzle_multi_strategy(
        diamond_grid_image_path,
        temp_dir,
        strategies=["constraint_labels"]
    )

    assert result["success"] is True
    assert len(result["all_attempts"]) == 1, "Should only have 1 attempt"
    assert result["all_attempts"][0]["method"] == "constraint_labels"
    assert result["method_used"] == "constraint_labels"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
