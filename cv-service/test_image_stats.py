#!/usr/bin/env python3
"""
Tests for the /image-stats endpoint.

Can be run in two modes:
1. Standalone against running server: python test_image_stats.py [service_url]
2. With pytest (uses TestClient): pytest test_image_stats.py -v

Tests cover:
- Valid image with expected metrics
- Invalid image handling
- ROI functionality
- Metric range validation
"""

import base64
import sys
from typing import Optional

import cv2
import numpy as np


def create_test_image(
    width: int = 100,
    height: int = 100,
    color: tuple = (128, 128, 128),
    add_gradient: bool = False
) -> str:
    """Create a test image and return as base64 string."""
    if add_gradient:
        # Create a gradient image for testing dynamic range
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for x in range(width):
            val = int(255 * x / width)
            img[:, x] = [val, val, val]
    else:
        # Create solid color image
        img = np.full((height, width, 3), color, dtype=np.uint8)

    # Encode to PNG
    success, buffer = cv2.imencode('.png', img)
    if not success:
        raise ValueError("Failed to encode test image")

    return base64.b64encode(buffer.tobytes()).decode('utf-8')


def create_colored_test_image(
    width: int = 100,
    height: int = 100,
    rgb_color: tuple = (255, 0, 0)
) -> str:
    """Create a solid colored test image (RGB input, converted to BGR for OpenCV)."""
    # OpenCV uses BGR, so convert RGB to BGR
    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
    img = np.full((height, width, 3), bgr_color, dtype=np.uint8)

    success, buffer = cv2.imencode('.png', img)
    if not success:
        raise ValueError("Failed to encode test image")

    return base64.b64encode(buffer.tobytes()).decode('utf-8')


def create_two_region_image() -> str:
    """Create 200x200 image with dark top-left and bright bottom-right."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[100:200, 100:200] = [255, 255, 255]  # Bright bottom-right
    success, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


class ImageStatsEndpointTester:
    """Test cases for /image-stats endpoint (for use with running server)."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.endpoint = f"{base_url}/image-stats"

    def _call_endpoint(self, payload: dict) -> dict:
        """Make request to the endpoint."""
        import requests
        response = requests.post(self.endpoint, json=payload)
        return response.json()

    def test_valid_gray_image(self) -> bool:
        """Test with a solid gray image - should return expected brightness."""
        print("Test: Valid gray image...")

        image_b64 = create_test_image(color=(128, 128, 128))
        result = self._call_endpoint({"image": image_b64})

        if not result["success"]:
            print(f"  FAIL: {result.get('error', 'Unknown error')}")
            return False

        brightness = result["brightness"]
        if not (120 <= brightness <= 136):
            print(f"  FAIL: Expected brightness ~128, got {brightness}")
            return False

        contrast = result["contrast"]
        if contrast > 5:
            print(f"  FAIL: Expected low contrast for solid color, got {contrast}")
            return False

        if result["image_width"] != 100 or result["image_height"] != 100:
            print(f"  FAIL: Wrong dimensions: {result['image_width']}x{result['image_height']}")
            return False

        print(f"  PASS: brightness={brightness:.2f}, contrast={contrast:.2f}")
        return True

    def test_valid_gradient_image(self) -> bool:
        """Test with a gradient image - should show high dynamic range."""
        print("Test: Valid gradient image...")

        image_b64 = create_test_image(add_gradient=True)
        result = self._call_endpoint({"image": image_b64})

        if not result["success"]:
            print(f"  FAIL: {result.get('error', 'Unknown error')}")
            return False

        dr = result["dynamic_range"]
        if dr["min"] > 10:
            print(f"  FAIL: Expected dynamic_range.min near 0, got {dr['min']}")
            return False
        if dr["max"] < 245:
            print(f"  FAIL: Expected dynamic_range.max near 255, got {dr['max']}")
            return False

        if result["contrast"] < 50:
            print(f"  FAIL: Expected high contrast for gradient, got {result['contrast']}")
            return False

        print(f"  PASS: dynamic_range=({dr['min']}, {dr['max']}), contrast={result['contrast']:.2f}")
        return True

    def test_colored_image(self) -> bool:
        """Test with a saturated red image - should show color balance skew."""
        print("Test: Colored (red) image...")

        image_b64 = create_colored_test_image(rgb_color=(255, 0, 0))
        result = self._call_endpoint({"image": image_b64})

        if not result["success"]:
            print(f"  FAIL: {result.get('error', 'Unknown error')}")
            return False

        cb = result["color_balance"]
        if cb["r_mean"] < 200:
            print(f"  FAIL: Expected high r_mean for red image, got {cb['r_mean']}")
            return False
        if cb["g_mean"] > 10 or cb["b_mean"] > 10:
            print(f"  FAIL: Expected low g/b means, got g={cb['g_mean']}, b={cb['b_mean']}")
            return False
        if cb["r_ratio"] < 2.0:
            print(f"  FAIL: Expected r_ratio > 2 for red image, got {cb['r_ratio']}")
            return False

        sat = result["saturation"]
        if sat["mean"] < 200:
            print(f"  FAIL: Expected high saturation for pure red, got {sat['mean']}")
            return False

        print(f"  PASS: r_ratio={cb['r_ratio']:.3f}, saturation={sat['mean']:.2f}")
        return True

    def test_invalid_base64(self) -> bool:
        """Test with invalid base64 - should return error."""
        print("Test: Invalid base64...")

        result = self._call_endpoint({"image": "not_valid_base64!!!"})

        if result["success"]:
            print("  FAIL: Should have failed for invalid base64")
            return False

        if not result.get("error"):
            print("  FAIL: Should have error message")
            return False

        print(f"  PASS: Correctly rejected with: {result['error'][:50]}...")
        return True

    def test_invalid_image_data(self) -> bool:
        """Test with valid base64 but not an image - should return error."""
        print("Test: Invalid image data...")

        fake_data = base64.b64encode(b"This is not an image").decode('utf-8')
        result = self._call_endpoint({"image": fake_data})

        if result["success"]:
            print("  FAIL: Should have failed for non-image data")
            return False

        if not result.get("error"):
            print("  FAIL: Should have error message")
            return False

        print(f"  PASS: Correctly rejected with: {result['error']}")
        return True

    def test_roi_valid(self) -> bool:
        """Test with valid ROI - should apply region cropping."""
        print("Test: Valid ROI...")

        image_b64 = create_two_region_image()

        result = self._call_endpoint({
            "image": image_b64,
            "roi": {"x": 100, "y": 100, "width": 100, "height": 100}
        })

        if not result["success"]:
            print(f"  FAIL: {result.get('error', 'Unknown error')}")
            return False

        if not result["roi_applied"]:
            print("  FAIL: roi_applied should be True")
            return False

        if result["brightness"] < 250:
            print(f"  FAIL: Expected brightness ~255 for white region, got {result['brightness']}")
            return False

        print(f"  PASS: roi_applied=True, brightness={result['brightness']:.2f}")
        return True

    def test_roi_out_of_bounds(self) -> bool:
        """Test with ROI exceeding image bounds - should return error."""
        print("Test: ROI out of bounds...")

        image_b64 = create_test_image(width=100, height=100)

        result = self._call_endpoint({
            "image": image_b64,
            "roi": {"x": 50, "y": 50, "width": 100, "height": 100}
        })

        if result["success"]:
            print("  FAIL: Should have failed for out-of-bounds ROI")
            return False

        if "ROI bounds exceed" not in result.get("error", ""):
            print(f"  FAIL: Expected ROI bounds error, got: {result.get('error')}")
            return False

        print(f"  PASS: Correctly rejected with: {result['error']}")
        return True

    def test_roi_negative_coords(self) -> bool:
        """Test with negative ROI coordinates - should return error."""
        print("Test: ROI negative coordinates...")

        image_b64 = create_test_image(width=100, height=100)

        result = self._call_endpoint({
            "image": image_b64,
            "roi": {"x": -10, "y": 0, "width": 50, "height": 50}
        })

        if result["success"]:
            print("  FAIL: Should have failed for negative ROI coords")
            return False

        print(f"  PASS: Correctly rejected with: {result.get('error', 'error')}")
        return True

    def test_metric_ranges(self) -> bool:
        """Verify all metric values are in expected ranges."""
        print("Test: Metric range validation...")

        image_b64 = create_test_image(add_gradient=True)
        result = self._call_endpoint({"image": image_b64})

        if not result["success"]:
            print(f"  FAIL: {result.get('error', 'Unknown error')}")
            return False

        errors = []

        if not (0 <= result["brightness"] <= 255):
            errors.append(f"brightness {result['brightness']} not in [0, 255]")

        if result["contrast"] < 0:
            errors.append(f"contrast {result['contrast']} is negative")

        dr = result["dynamic_range"]
        if not (0 <= dr["min"] <= 255):
            errors.append(f"dynamic_range.min {dr['min']} not in [0, 255]")
        if not (0 <= dr["max"] <= 255):
            errors.append(f"dynamic_range.max {dr['max']} not in [0, 255]")
        if dr["min"] > dr["max"]:
            errors.append(f"dynamic_range.min > max: {dr['min']} > {dr['max']}")

        cb = result["color_balance"]
        for channel in ["r_mean", "g_mean", "b_mean"]:
            if not (0 <= cb[channel] <= 255):
                errors.append(f"color_balance.{channel} {cb[channel]} not in [0, 255]")

        for ratio in ["r_ratio", "g_ratio", "b_ratio"]:
            if cb[ratio] < 0:
                errors.append(f"color_balance.{ratio} {cb[ratio]} is negative")

        sat = result["saturation"]
        if not (0 <= sat["mean"] <= 255):
            errors.append(f"saturation.mean {sat['mean']} not in [0, 255]")
        if not (0 <= sat["min"] <= 255):
            errors.append(f"saturation.min {sat['min']} not in [0, 255]")
        if not (0 <= sat["max"] <= 255):
            errors.append(f"saturation.max {sat['max']} not in [0, 255]")

        if result["extraction_ms"] < 0:
            errors.append(f"extraction_ms {result['extraction_ms']} is negative")

        if errors:
            for e in errors:
                print(f"  FAIL: {e}")
            return False

        print("  PASS: All metrics in valid ranges")
        return True

    def test_timing_included(self) -> bool:
        """Verify extraction_ms is included in response."""
        print("Test: Timing metric included...")

        image_b64 = create_test_image()
        result = self._call_endpoint({"image": image_b64})

        if "extraction_ms" not in result:
            print("  FAIL: extraction_ms missing from response")
            return False

        if not isinstance(result["extraction_ms"], int):
            print(f"  FAIL: extraction_ms should be int, got {type(result['extraction_ms'])}")
            return False

        print(f"  PASS: extraction_ms={result['extraction_ms']}")
        return True

    def run_all_tests(self) -> bool:
        """Run all tests and return overall success."""
        print(f"\n{'='*50}")
        print(f"Testing /image-stats endpoint at {self.endpoint}")
        print(f"{'='*50}\n")

        tests = [
            self.test_valid_gray_image,
            self.test_valid_gradient_image,
            self.test_colored_image,
            self.test_invalid_base64,
            self.test_invalid_image_data,
            self.test_roi_valid,
            self.test_roi_out_of_bounds,
            self.test_roi_negative_coords,
            self.test_metric_ranges,
            self.test_timing_included,
        ]

        passed = 0
        failed = 0

        for test in tests:
            try:
                if test():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ERROR: {e}")
                failed += 1

        print(f"\n{'='*50}")
        print(f"Results: {passed} passed, {failed} failed")
        print(f"{'='*50}\n")

        return failed == 0


# ============================================================================
# Unit tests for _calculate_image_stats function (no server needed)
# ============================================================================
import pytest
import os

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

# Import the function directly for unit testing
try:
    from main import _calculate_image_stats, decode_image
    _function_available = True
except ImportError:
    _function_available = False


@pytest.fixture
def calc_stats():
    """Fixture to provide _calculate_image_stats function."""
    if not _function_available:
        pytest.skip("Could not import _calculate_image_stats from main")
    return _calculate_image_stats


def test_unit_gray_image_brightness(calc_stats):
    """Unit test: Gray image returns expected brightness."""
    img = np.full((100, 100, 3), (128, 128, 128), dtype=np.uint8)
    stats = calc_stats(img)

    assert 120 <= stats["brightness"] <= 136
    assert stats["contrast"] < 5  # Solid color has low contrast


def test_unit_gradient_dynamic_range(calc_stats):
    """Unit test: Gradient image has full dynamic range."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for x in range(100):
        val = int(255 * x / 100)
        img[:, x] = [val, val, val]

    stats = calc_stats(img)

    assert stats["dynamic_range"]["min"] <= 10
    assert stats["dynamic_range"]["max"] >= 245
    assert stats["contrast"] > 50  # Gradient has high contrast


def test_unit_red_image_color_balance(calc_stats):
    """Unit test: Red image shows red dominance."""
    # BGR format - red is (0, 0, 255)
    img = np.full((100, 100, 3), (0, 0, 255), dtype=np.uint8)
    stats = calc_stats(img)

    assert stats["color_balance"]["r_mean"] >= 250
    assert stats["color_balance"]["g_mean"] <= 5
    assert stats["color_balance"]["b_mean"] <= 5
    assert stats["color_balance"]["r_ratio"] > 2.5


def test_unit_saturated_image(calc_stats):
    """Unit test: Saturated color has high saturation."""
    # Pure red in BGR
    img = np.full((100, 100, 3), (0, 0, 255), dtype=np.uint8)
    stats = calc_stats(img)

    assert stats["saturation"]["mean"] >= 250
    assert stats["saturation"]["max"] == 255


def test_unit_grayscale_low_saturation(calc_stats):
    """Unit test: Grayscale image has zero saturation."""
    img = np.full((100, 100, 3), (128, 128, 128), dtype=np.uint8)
    stats = calc_stats(img)

    assert stats["saturation"]["mean"] == 0
    assert stats["saturation"]["min"] == 0
    assert stats["saturation"]["max"] == 0


def test_unit_black_image(calc_stats):
    """Unit test: Black image has zero brightness."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    stats = calc_stats(img)

    assert stats["brightness"] == 0
    assert stats["contrast"] == 0
    assert stats["dynamic_range"]["min"] == 0
    assert stats["dynamic_range"]["max"] == 0


def test_unit_white_image(calc_stats):
    """Unit test: White image has max brightness."""
    img = np.full((100, 100, 3), (255, 255, 255), dtype=np.uint8)
    stats = calc_stats(img)

    assert stats["brightness"] == 255
    assert stats["contrast"] == 0
    assert stats["dynamic_range"]["min"] == 255
    assert stats["dynamic_range"]["max"] == 255


# ============================================================================
# Integration tests using httpx with ASGITransport (for newer httpx versions)
# ============================================================================
# These tests use httpx ASGITransport to test the full endpoint.
# This approach works with httpx 0.28+ where TestClient API changed.

import httpx

# Try to import the app
_app = None
try:
    from main import app as _app
except Exception:
    pass


@pytest.fixture
def client():
    """Pytest fixture that provides an async httpx client."""
    if _app is None:
        pytest.skip("Could not import app from main")

    class AsyncClientWrapper:
        """Wrapper that runs async requests synchronously for tests."""

        def __init__(self, app):
            self.app = app

        def post(self, url: str, json: dict = None):
            """Make a POST request using async httpx."""
            import asyncio

            async def _request():
                transport = httpx.ASGITransport(app=self.app)
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    return await client.post(url, json=json)

            return asyncio.run(_request())

    return AsyncClientWrapper(_app)


def test_valid_image_returns_metrics(client):
    """Valid image returns success with all expected metrics."""
    image_b64 = create_test_image(color=(128, 128, 128))
    response = client.post("/image-stats", json={"image": image_b64})

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert "brightness" in result
    assert "contrast" in result
    assert "dynamic_range" in result
    assert "color_balance" in result
    assert "saturation" in result
    assert result["image_width"] == 100
    assert result["image_height"] == 100


def test_invalid_base64_returns_error(client):
    """Invalid base64 input returns error."""
    response = client.post("/image-stats", json={"image": "invalid!!!"})

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is False
    assert result["error"] is not None


def test_invalid_image_data_returns_error(client):
    """Valid base64 but not image data returns error."""
    fake_data = base64.b64encode(b"not an image").decode('utf-8')
    response = client.post("/image-stats", json={"image": fake_data})

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is False


def test_roi_applied_correctly(client):
    """Valid ROI is applied and affects calculations."""
    image_b64 = create_two_region_image()

    response = client.post("/image-stats", json={
        "image": image_b64,
        "roi": {"x": 100, "y": 100, "width": 100, "height": 100}
    })

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert result["roi_applied"] is True
    assert result["brightness"] >= 250  # White region


def test_roi_out_of_bounds_returns_error(client):
    """ROI exceeding image bounds returns error."""
    image_b64 = create_test_image(width=100, height=100)

    response = client.post("/image-stats", json={
        "image": image_b64,
        "roi": {"x": 50, "y": 50, "width": 100, "height": 100}
    })

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is False
    assert "ROI bounds exceed" in result["error"]


def test_brightness_in_valid_range(client):
    """Brightness value is in range 0-255."""
    image_b64 = create_test_image()
    response = client.post("/image-stats", json={"image": image_b64})

    result = response.json()
    assert result["success"] is True
    assert 0 <= result["brightness"] <= 255


def test_contrast_non_negative(client):
    """Contrast value is non-negative."""
    image_b64 = create_test_image()
    response = client.post("/image-stats", json={"image": image_b64})

    result = response.json()
    assert result["success"] is True
    assert result["contrast"] >= 0


def test_dynamic_range_valid(client):
    """Dynamic range has valid min <= max in 0-255."""
    image_b64 = create_test_image(add_gradient=True)
    response = client.post("/image-stats", json={"image": image_b64})

    result = response.json()
    assert result["success"] is True
    dr = result["dynamic_range"]
    assert 0 <= dr["min"] <= 255
    assert 0 <= dr["max"] <= 255
    assert dr["min"] <= dr["max"]


def test_color_balance_valid(client):
    """Color balance values are valid."""
    image_b64 = create_colored_test_image(rgb_color=(255, 0, 0))
    response = client.post("/image-stats", json={"image": image_b64})

    result = response.json()
    assert result["success"] is True
    cb = result["color_balance"]
    assert 0 <= cb["r_mean"] <= 255
    assert 0 <= cb["g_mean"] <= 255
    assert 0 <= cb["b_mean"] <= 255
    assert cb["r_ratio"] >= 0
    assert cb["g_ratio"] >= 0
    assert cb["b_ratio"] >= 0


def test_saturation_valid(client):
    """Saturation values are in valid range."""
    image_b64 = create_colored_test_image(rgb_color=(255, 0, 0))
    response = client.post("/image-stats", json={"image": image_b64})

    result = response.json()
    assert result["success"] is True
    sat = result["saturation"]
    assert 0 <= sat["mean"] <= 255
    assert 0 <= sat["min"] <= 255
    assert 0 <= sat["max"] <= 255


def test_extraction_ms_included(client):
    """Response includes extraction_ms timing metric."""
    image_b64 = create_test_image()
    response = client.post("/image-stats", json={"image": image_b64})

    result = response.json()
    assert "extraction_ms" in result
    assert isinstance(result["extraction_ms"], int)
    assert result["extraction_ms"] >= 0


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"

    tester = ImageStatsEndpointTester(base_url=url)
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)
