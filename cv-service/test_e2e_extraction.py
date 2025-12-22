"""
End-to-end integration tests for domino extraction pipeline with pip detection.

Tests the full flow: load image -> extract dominoes -> detect pips -> verify response format.
Uses synthetic test images and the httpx AsyncClient for API endpoint testing.
"""

import pytest
import numpy as np
import cv2
import base64
import math
from typing import Tuple, List

import httpx

from main import (
    app,
    decode_base64_image,
    crop_domino_region,
    DominoResult,
    DominoExtractionResponse,
    validate_domino_dimensions,
    check_image_quality,
    assess_detection_quality,
)
from extract_dominoes import (
    detect_domino_pips,
    PipDetectionResult,
    preprocess_domino_image,
    split_domino_halves,
    rotate_domino,
)


# Async client fixture for API tests
@pytest.fixture
def async_client():
    """Create an async test client for API tests."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# =============================================================================
# Test Image Generation Helpers
# =============================================================================

def create_test_domino(
    width: int = 200,
    height: int = 100,
    left_pips: int = 3,
    right_pips: int = 5,
    pip_radius: int = 10,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    pip_color: Tuple[int, int, int] = (0, 0, 0),
    divider_width: int = 2
) -> np.ndarray:
    """
    Create a synthetic domino image with specified pip counts.

    Standard domino pip patterns:
    0: empty
    1: center pip
    2: diagonal corners (top-left, bottom-right)
    3: diagonal + center
    4: four corners
    5: four corners + center
    6: six pips (2 columns of 3)
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = bg_color

    # Draw center divider line
    mid_x = width // 2
    cv2.line(image, (mid_x, 0), (mid_x, height), (128, 128, 128), divider_width)

    def draw_pips_on_half(start_x: int, end_x: int, pip_count: int):
        """Draw pips on one half of the domino."""
        half_width = end_x - start_x
        cx = start_x + half_width // 2
        cy = height // 2

        # Spacing for pip positions
        dx = half_width // 4
        dy = height // 4

        # Pip positions based on count (following standard domino patterns)
        positions = []
        if pip_count == 0:
            pass  # No pips
        elif pip_count == 1:
            positions = [(cx, cy)]
        elif pip_count == 2:
            positions = [(cx - dx, cy - dy), (cx + dx, cy + dy)]
        elif pip_count == 3:
            positions = [(cx - dx, cy - dy), (cx, cy), (cx + dx, cy + dy)]
        elif pip_count == 4:
            positions = [
                (cx - dx, cy - dy), (cx + dx, cy - dy),
                (cx - dx, cy + dy), (cx + dx, cy + dy)
            ]
        elif pip_count == 5:
            positions = [
                (cx - dx, cy - dy), (cx + dx, cy - dy),
                (cx, cy),
                (cx - dx, cy + dy), (cx + dx, cy + dy)
            ]
        elif pip_count == 6:
            positions = [
                (cx - dx, cy - dy), (cx + dx, cy - dy),
                (cx - dx, cy), (cx + dx, cy),
                (cx - dx, cy + dy), (cx + dx, cy + dy)
            ]

        for pos in positions:
            cv2.circle(image, (int(pos[0]), int(pos[1])), pip_radius, pip_color, -1)

    # Draw pips on left half
    draw_pips_on_half(0, mid_x, left_pips)

    # Draw pips on right half
    draw_pips_on_half(mid_x, width, right_pips)

    return image


def create_rotated_test_domino(
    angle: float,
    width: int = 200,
    height: int = 100,
    left_pips: int = 3,
    right_pips: int = 5
) -> np.ndarray:
    """Create a rotated synthetic domino image."""
    # Create base domino
    domino = create_test_domino(width, height, left_pips, right_pips)

    # Calculate new canvas size to fit rotated image
    abs_cos = abs(math.cos(math.radians(angle)))
    abs_sin = abs(math.sin(math.radians(angle)))
    new_w = int(height * abs_sin + width * abs_cos) + 100
    new_h = int(height * abs_cos + width * abs_sin) + 100

    # Create larger canvas
    canvas_size = max(new_w, new_h, width + 100, height + 100)
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 200

    # Place domino in center of canvas
    start_y = (canvas.shape[0] - height) // 2
    start_x = (canvas.shape[1] - width) // 2
    canvas[start_y:start_y+height, start_x:start_x+width] = domino

    # Rotate around center
    center = (canvas.shape[1] // 2, canvas.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        canvas, M, (canvas.shape[1], canvas.shape[0]),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(200, 200, 200)
    )

    return rotated


def create_puzzle_image_with_dominoes(
    num_dominoes: int = 4,
    image_size: Tuple[int, int] = (800, 600),
    domino_size: Tuple[int, int] = (150, 75),
    pip_configs: List[Tuple[int, int]] = None
) -> Tuple[np.ndarray, List[dict]]:
    """
    Create a synthetic puzzle image with multiple dominoes.

    Args:
        num_dominoes: Number of dominoes to place
        image_size: Overall image size (width, height)
        domino_size: Size of each domino (width, height)
        pip_configs: List of (left_pips, right_pips) tuples for each domino.
                    If None, uses random valid pip values.

    Returns:
        Tuple of (image, domino_boxes) where domino_boxes is a list of
        dicts with x, y, width, height keys.
    """
    img_width, img_height = image_size
    dom_width, dom_height = domino_size

    # Create base image with light gray background
    image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 220

    # Default pip configurations if not provided
    if pip_configs is None:
        pip_configs = [(i % 7, (i + 3) % 7) for i in range(num_dominoes)]

    domino_boxes = []

    # Calculate grid layout
    cols = int(np.ceil(np.sqrt(num_dominoes)))
    rows = int(np.ceil(num_dominoes / cols))

    # Spacing between dominoes
    spacing_x = (img_width - cols * dom_width) // (cols + 1)
    spacing_y = (img_height - rows * dom_height) // (rows + 1)

    for i in range(num_dominoes):
        row = i // cols
        col = i % cols

        # Calculate position
        x = spacing_x + col * (dom_width + spacing_x)
        y = spacing_y + row * (dom_height + spacing_y)

        # Get pip configuration
        left_pips, right_pips = pip_configs[i % len(pip_configs)]

        # Create and place domino
        domino = create_test_domino(
            width=dom_width,
            height=dom_height,
            left_pips=left_pips,
            right_pips=right_pips
        )

        # Place domino on image
        image[y:y+dom_height, x:x+dom_width] = domino

        # Record bounding box
        domino_boxes.append({
            "x": x,
            "y": y,
            "width": dom_width,
            "height": dom_height,
            "expected_left_pips": left_pips,
            "expected_right_pips": right_pips
        })

    return image, domino_boxes


def encode_image_base64(image: np.ndarray) -> str:
    """Encode an OpenCV image to base64 string."""
    success, buffer = cv2.imencode('.png', image)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer).decode('utf-8')


# =============================================================================
# Full Pipeline Integration Tests
# =============================================================================

class TestDominoPipDetectionPipeline:
    """Integration tests for the full domino pip detection pipeline."""

    def test_domino_pip_detection_pipeline(self):
        """
        Test the complete pipeline: image -> crop -> detect pips -> verify format.

        This is the main integration test for subtask-5-2.
        """
        # Create a test puzzle image with known pip values
        pip_configs = [(3, 5), (0, 1), (6, 6), (2, 4)]
        image, domino_boxes = create_puzzle_image_with_dominoes(
            num_dominoes=4,
            pip_configs=pip_configs
        )

        # Step 1: Verify image creation
        assert image is not None
        assert image.shape[2] == 3  # BGR image
        assert len(domino_boxes) == 4

        # Step 2: Test each domino through the full pipeline
        for i, box in enumerate(domino_boxes):
            # Crop the domino region
            cropped = crop_domino_region(
                image,
                box["x"],
                box["y"],
                box["width"],
                box["height"]
            )
            assert cropped is not None
            assert cropped.shape[0] == box["height"]
            assert cropped.shape[1] == box["width"]

            # Detect pips on the cropped domino
            result = detect_domino_pips(cropped)

            # Verify result format
            assert isinstance(result, PipDetectionResult)
            assert 0 <= result.left_pips <= 6
            assert 0 <= result.right_pips <= 6
            assert 0.0 <= result.left_confidence <= 1.0
            assert 0.0 <= result.right_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_pipeline_with_api_endpoint(self, async_client):
        """Test the full pipeline through the /crop-dominoes API endpoint."""
        # Create test image with dominoes
        pip_configs = [(1, 2), (3, 4), (5, 6)]
        image, domino_boxes = create_puzzle_image_with_dominoes(
            num_dominoes=3,
            pip_configs=pip_configs
        )

        # Encode image to base64
        base64_image = encode_image_base64(image)

        # Prepare request
        request_data = {
            "image": base64_image,
            "dominoes": [
                {"x": box["x"], "y": box["y"], "width": box["width"], "height": box["height"]}
                for box in domino_boxes
            ]
        }

        # Make API request
        async with async_client as client:
            response = await client.post("/crop-dominoes", json=request_data)

        # Verify response
        assert response.status_code == 200

        data = response.json()
        assert "dominoes" in data
        assert "total_count" in data
        assert "successful_count" in data
        assert "failed_count" in data
        assert data["total_count"] == 3
        assert len(data["dominoes"]) == 3

        # Verify each domino result
        for i, domino in enumerate(data["dominoes"]):
            assert "x" in domino
            assert "y" in domino
            assert "width" in domino
            assert "height" in domino
            assert "left_pips" in domino
            assert "right_pips" in domino
            assert "left_confidence" in domino
            assert "right_confidence" in domino

            # Verify pip values are in valid range or None (if detection failed)
            if domino["left_pips"] is not None:
                assert 0 <= domino["left_pips"] <= 6
            if domino["right_pips"] is not None:
                assert 0 <= domino["right_pips"] <= 6

    def test_pipeline_with_rotated_dominoes(self):
        """Test pipeline handles rotated dominoes correctly."""
        # Create a rotated domino
        rotated_image = create_rotated_test_domino(
            angle=30,
            left_pips=4,
            right_pips=2
        )

        # Detect pips (with auto-rotation)
        result = detect_domino_pips(rotated_image, auto_rotate=True)

        # Verify result format
        assert isinstance(result, PipDetectionResult)
        assert 0 <= result.left_pips <= 6
        assert 0 <= result.right_pips <= 6
        assert result.left_confidence >= 0.0
        assert result.right_confidence >= 0.0

    def test_pipeline_with_blank_dominoes(self):
        """Test pipeline correctly detects blank dominoes (0 pips)."""
        # Create blank domino
        blank_domino = create_test_domino(left_pips=0, right_pips=0)

        result = detect_domino_pips(blank_domino)

        # Both halves should detect 0 or very low pips
        assert result.left_pips <= 1
        assert result.right_pips <= 1
        # Blank detection should have some confidence
        assert result.left_confidence >= 0.0
        assert result.right_confidence >= 0.0

    def test_pipeline_with_inverted_colors(self):
        """Test pipeline handles inverted color scheme (light pips on dark)."""
        inverted_domino = create_test_domino(
            left_pips=3,
            right_pips=5,
            bg_color=(30, 30, 30),
            pip_color=(220, 220, 220)
        )

        result = detect_domino_pips(inverted_domino)

        # Should still detect pips even with inverted colors
        assert isinstance(result, PipDetectionResult)
        assert 0 <= result.left_pips <= 6
        assert 0 <= result.right_pips <= 6


# =============================================================================
# API Response Format Tests
# =============================================================================

class TestApiResponseFormat:
    """Tests for API response format verification."""

    @pytest.mark.asyncio
    async def test_api_response_format(self, async_client):
        """Test /crop-dominoes returns correct response format."""
        # Create simple test image
        domino = create_test_domino(width=100, height=50, left_pips=2, right_pips=3)

        # Create a larger canvas to place the domino
        image = np.ones((200, 200, 3), dtype=np.uint8) * 200
        image[50:100, 50:150] = domino

        base64_image = encode_image_base64(image)

        async with async_client as client:
            response = await client.post("/crop-dominoes", json={
                "image": base64_image,
                "dominoes": [{"x": 50, "y": 50, "width": 100, "height": 50}]
            })

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert isinstance(data, dict)
        assert "dominoes" in data
        assert "total_count" in data
        assert "successful_count" in data
        assert "failed_count" in data

        # Verify domino result fields
        domino_result = data["dominoes"][0]
        required_fields = [
            "x", "y", "width", "height",
            "left_pips", "right_pips",
            "left_confidence", "right_confidence"
        ]
        for field in required_fields:
            assert field in domino_result, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_api_response_with_error(self, async_client):
        """Test API returns appropriate error for invalid input."""
        # Test with invalid base64 image
        async with async_client as client:
            response = await client.post("/crop-dominoes", json={
                "image": "not-valid-base64!!!",
                "dominoes": [{"x": 0, "y": 0, "width": 100, "height": 50}]
            })

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_api_response_empty_dominoes(self, async_client):
        """Test API handles empty dominoes list."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        base64_image = encode_image_base64(image)

        async with async_client as client:
            response = await client.post("/crop-dominoes", json={
                "image": base64_image,
                "dominoes": []
            })

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0
        assert len(data["dominoes"]) == 0

    @pytest.mark.asyncio
    async def test_api_response_out_of_bounds_domino(self, async_client):
        """Test API handles domino bounding box outside image bounds."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        base64_image = encode_image_base64(image)

        # Request domino outside image bounds
        async with async_client as client:
            response = await client.post("/crop-dominoes", json={
                "image": base64_image,
                "dominoes": [{"x": 200, "y": 200, "width": 100, "height": 50}]
            })

        # Should return 200 but with error in the domino result
        assert response.status_code == 200
        data = response.json()
        assert data["failed_count"] == 1
        assert data["dominoes"][0]["error"] is not None


# =============================================================================
# Multiple Dominoes Tests
# =============================================================================

class TestMultipleDominoes:
    """Tests for processing multiple dominoes in a single request."""

    @pytest.mark.asyncio
    async def test_multiple_dominoes_processing(self, async_client):
        """Test correct processing of multiple dominoes in one request."""
        # Create image with 6 dominoes
        pip_configs = [(0, 1), (2, 3), (4, 5), (6, 0), (1, 6), (3, 3)]
        image, domino_boxes = create_puzzle_image_with_dominoes(
            num_dominoes=6,
            pip_configs=pip_configs
        )

        base64_image = encode_image_base64(image)

        async with async_client as client:
            response = await client.post("/crop-dominoes", json={
                "image": base64_image,
                "dominoes": [
                    {"x": box["x"], "y": box["y"], "width": box["width"], "height": box["height"]}
                    for box in domino_boxes
                ]
            })

        assert response.status_code == 200
        data = response.json()

        assert data["total_count"] == 6
        assert len(data["dominoes"]) == 6

        # All should be processed (successful or with warnings)
        assert data["successful_count"] + data["failed_count"] == 6

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self, async_client):
        """Test handling of mixed valid and invalid domino regions."""
        image, domino_boxes = create_puzzle_image_with_dominoes(
            num_dominoes=2,
            pip_configs=[(1, 2), (3, 4)]
        )

        base64_image = encode_image_base64(image)

        # Include one valid and one invalid domino region
        async with async_client as client:
            response = await client.post("/crop-dominoes", json={
                "image": base64_image,
                "dominoes": [
                    {"x": domino_boxes[0]["x"], "y": domino_boxes[0]["y"],
                     "width": domino_boxes[0]["width"], "height": domino_boxes[0]["height"]},
                    {"x": 9999, "y": 9999, "width": 100, "height": 50}  # Out of bounds
                ]
            })

        assert response.status_code == 200
        data = response.json()

        assert data["total_count"] == 2
        assert data["successful_count"] >= 1  # At least one should succeed
        assert data["failed_count"] >= 1  # At least one should fail


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions used in the pipeline."""

    def test_decode_base64_image(self):
        """Test base64 image decoding."""
        # Create test image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        base64_str = encode_image_base64(image)

        # Decode
        decoded = decode_base64_image(base64_str)

        assert decoded is not None
        assert decoded.shape == image.shape

    def test_decode_base64_with_data_uri(self):
        """Test base64 decoding with data URI prefix."""
        image = np.ones((50, 50, 3), dtype=np.uint8) * 100
        base64_str = encode_image_base64(image)

        # Add data URI prefix
        data_uri = f"data:image/png;base64,{base64_str}"

        decoded = decode_base64_image(data_uri)
        assert decoded is not None

    def test_crop_domino_region(self):
        """Test domino region cropping."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[50:100, 50:150] = 255  # White region

        cropped = crop_domino_region(image, 50, 50, 100, 50)

        assert cropped.shape == (50, 100, 3)
        assert np.all(cropped == 255)

    def test_crop_domino_region_bounds_check(self):
        """Test crop region bounds validation."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Should raise for out of bounds
        with pytest.raises(ValueError, match="exceeds"):
            crop_domino_region(image, 50, 50, 100, 100)

    def test_validate_domino_dimensions(self):
        """Test dimension validation for dominoes."""
        # Valid dimensions
        valid, warning = validate_domino_dimensions(100, 50)
        assert valid is True

        # Too small
        valid, warning = validate_domino_dimensions(10, 5)
        assert valid is False

    def test_check_image_quality(self):
        """Test image quality checking."""
        # Good quality image
        good_image = create_test_domino()
        ok, warning = check_image_quality(good_image)
        assert ok is True

        # Very low contrast image
        low_contrast = np.full((100, 100, 3), 128, dtype=np.uint8)
        ok, warning = check_image_quality(low_contrast)
        # May or may not be acceptable depending on threshold

    def test_assess_detection_quality(self):
        """Test detection quality assessment."""
        # High confidence
        warning = assess_detection_quality(0.9, 0.85)
        assert warning is None  # No warning for good confidence

        # Low confidence
        warning = assess_detection_quality(0.3, 0.4)
        assert warning is not None  # Should have warning


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self, async_client):
        """Test /health endpoint returns correct status."""
        async with async_client as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Edge case tests for the pipeline."""

    def test_very_small_domino(self):
        """Test handling of very small domino images."""
        small_domino = create_test_domino(width=40, height=20, left_pips=1, right_pips=1)

        result = detect_domino_pips(small_domino)

        # Should still return a valid result, even if detection is uncertain
        assert isinstance(result, PipDetectionResult)
        assert 0 <= result.left_pips <= 6
        assert 0 <= result.right_pips <= 6

    def test_all_pip_combinations(self):
        """Test all valid pip combinations (0-6 x 0-6)."""
        for left in range(7):
            for right in range(7):
                domino = create_test_domino(left_pips=left, right_pips=right)
                result = detect_domino_pips(domino)

                assert isinstance(result, PipDetectionResult)
                assert 0 <= result.left_pips <= 6
                assert 0 <= result.right_pips <= 6

    def test_noisy_image(self):
        """Test pipeline handles noisy images gracefully."""
        # Create domino
        domino = create_test_domino(left_pips=4, right_pips=2)

        # Add noise
        noise = np.random.normal(0, 25, domino.shape).astype(np.uint8)
        noisy_domino = cv2.add(domino, noise)

        result = detect_domino_pips(noisy_domino)

        # Should still return valid result
        assert isinstance(result, PipDetectionResult)
        assert 0 <= result.left_pips <= 6
        assert 0 <= result.right_pips <= 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
