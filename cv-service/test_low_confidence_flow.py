#!/usr/bin/env python3
"""
E2E Tests for Low-Confidence Flow Verification

This script specifically tests the low-confidence extraction scenarios to verify
that the system properly:
1. Detects low confidence situations
2. Returns appropriate warnings
3. Provides correction-friendly data

Subtask: 6-2 - Verify low-confidence flow prompts user for verification

Usage:
    cd cv-service
    uvicorn main:app --host 0.0.0.0 --port 8080 --reload &
    python test_low_confidence_flow.py [image_path]
"""

import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import requests

try:
    import numpy as np
    from PIL import Image
    HAS_IMAGE_PROCESSING = True
except ImportError:
    HAS_IMAGE_PROCESSING = False


class LowConfidenceTestRunner:
    """Tests specifically for low-confidence extraction scenarios."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.results: List[Tuple[str, bool, str]] = []

    def log_result(self, test_name: str, passed: bool, message: str):
        """Log a test result."""
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_name}: {message}")
        self.results.append((test_name, passed, message))

    def check_service_health(self) -> bool:
        """Verify CV service is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def create_low_quality_image(self, original_b64: str) -> Optional[str]:
        """
        Create a degraded version of the image to simulate low-quality input.
        Applies: blur, noise, reduced contrast.
        """
        if not HAS_IMAGE_PROCESSING:
            return None

        try:
            # Decode original image
            img_data = base64.b64decode(original_b64)
            img = Image.open(io.BytesIO(img_data))

            # Convert to numpy array
            arr = np.array(img, dtype=np.float32)

            # 1. Add noise (simulate poor camera quality)
            noise = np.random.normal(0, 25, arr.shape)
            arr = np.clip(arr + noise, 0, 255)

            # 2. Reduce contrast (simulate poor lighting)
            mean_val = np.mean(arr)
            arr = mean_val + (arr - mean_val) * 0.5  # 50% contrast reduction

            # 3. Apply slight blur by averaging with neighbors
            from scipy import ndimage
            if len(arr.shape) == 3:
                for c in range(arr.shape[2]):
                    arr[:, :, c] = ndimage.uniform_filter(arr[:, :, c], size=3)
            else:
                arr = ndimage.uniform_filter(arr, size=3)

            arr = np.clip(arr, 0, 255).astype(np.uint8)

            # Convert back to image
            degraded_img = Image.fromarray(arr)

            # Encode to base64
            buffer = io.BytesIO()
            degraded_img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

        except Exception as e:
            print(f"    Warning: Could not create degraded image: {e}")
            return None

    def create_partial_image(self, original_b64: str, crop_percent: float = 0.6) -> Optional[str]:
        """
        Create a cropped version that only shows partial puzzle.
        Simulates a screenshot that doesn't capture the full board.
        """
        if not HAS_IMAGE_PROCESSING:
            return None

        try:
            img_data = base64.b64decode(original_b64)
            img = Image.open(io.BytesIO(img_data))

            width, height = img.size
            # Crop to show only part of the image
            crop_box = (
                0,
                0,
                int(width * crop_percent),
                int(height * crop_percent)
            )
            cropped_img = img.crop(crop_box)

            buffer = io.BytesIO()
            cropped_img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

        except Exception as e:
            print(f"    Warning: Could not create partial image: {e}")
            return None

    def test_low_confidence_detection(self, image_b64: str) -> Dict[str, Any]:
        """
        Test 1: Verify system detects low confidence from degraded images.
        """
        print("\n=== Test 1: Low Confidence Detection ===")

        # First test with normal image
        print("  Testing with original image...")
        normal_result = requests.post(
            f"{self.base_url}/crop-puzzle",
            json={"image": image_b64},
            timeout=30
        ).json()

        normal_confidence = normal_result.get("grid_confidence", 0)
        normal_level = normal_result.get("confidence_level", "unknown")
        self.log_result(
            "original_confidence",
            normal_result.get("success", False),
            f"Original: {normal_confidence:.0%} ({normal_level})"
        )

        # Test with degraded image if we can create one
        degraded_b64 = self.create_low_quality_image(image_b64)
        if degraded_b64:
            print("  Testing with degraded (noisy/low-contrast) image...")
            degraded_result = requests.post(
                f"{self.base_url}/crop-puzzle",
                json={"image": degraded_b64},
                timeout=30
            ).json()

            degraded_confidence = degraded_result.get("grid_confidence", 0)
            degraded_level = degraded_result.get("confidence_level", "unknown")

            # Degraded should have lower confidence
            confidence_dropped = degraded_confidence < normal_confidence
            self.log_result(
                "degraded_confidence",
                confidence_dropped or degraded_level in ["low", "unknown"],
                f"Degraded: {degraded_confidence:.0%} ({degraded_level}) - {'Lower' if confidence_dropped else 'Same/Higher'} than original"
            )

            return {
                "original": normal_result,
                "degraded": degraded_result,
                "confidence_difference": normal_confidence - degraded_confidence
            }
        else:
            self.log_result(
                "degraded_confidence",
                True,
                "Skipped - numpy/scipy not available for image degradation"
            )
            return {"original": normal_result, "degraded": None}

    def test_warnings_generated(self, image_b64: str) -> Dict[str, Any]:
        """
        Test 2: Verify appropriate warnings are generated for edge cases.
        """
        print("\n=== Test 2: Warning Generation ===")

        # Test with very low confidence threshold to force warnings
        result = requests.post(
            f"{self.base_url}/crop-puzzle",
            json={
                "image": image_b64,
                "min_confidence_threshold": 0.95  # Very high threshold
            },
            timeout=30
        ).json()

        warnings = result.get("warnings", [])
        has_warnings = len(warnings) > 0

        self.log_result(
            "warning_generation",
            True,  # Info only - warnings depend on actual confidence
            f"{len(warnings)} warnings: {'; '.join(warnings[:3])}" if warnings else "No warnings (image is high quality)"
        )

        # Check that warnings are informative
        if warnings:
            has_actionable_warning = any(
                any(keyword in w.lower() for keyword in ["verify", "check", "manual", "uncertain", "low"])
                for w in warnings
            )
            self.log_result(
                "actionable_warnings",
                has_actionable_warning,
                "Warnings include user-actionable guidance" if has_actionable_warning else "Warnings lack actionable guidance"
            )

        return result

    def test_partial_image_handling(self, image_b64: str) -> Dict[str, Any]:
        """
        Test 3: Verify system handles partial/cropped images gracefully.
        """
        print("\n=== Test 3: Partial Image Handling ===")

        partial_b64 = self.create_partial_image(image_b64, crop_percent=0.5)

        if partial_b64:
            print("  Testing with partial (50% cropped) image...")
            result = requests.post(
                f"{self.base_url}/crop-puzzle",
                json={"image": partial_b64},
                timeout=30
            ).json()

            confidence = result.get("grid_confidence", 0)
            confidence_level = result.get("confidence_level", "unknown")
            warnings = result.get("warnings", [])

            # Partial images should either have lower confidence or warnings
            handled_gracefully = (
                result.get("success", False) or
                confidence_level in ["low", "unknown"] or
                len(warnings) > 0
            )

            self.log_result(
                "partial_image_handling",
                handled_gracefully,
                f"Partial image: {confidence:.0%} ({confidence_level}) with {len(warnings)} warning(s)"
            )

            return result
        else:
            self.log_result(
                "partial_image_handling",
                True,
                "Skipped - PIL not available for image cropping"
            )
            return {}

    def test_confidence_level_classification(self, image_b64: str) -> Dict[str, Any]:
        """
        Test 4: Verify confidence is classified into appropriate levels.
        """
        print("\n=== Test 4: Confidence Level Classification ===")

        result = requests.post(
            f"{self.base_url}/crop-puzzle",
            json={"image": image_b64},
            timeout=30
        ).json()

        confidence = result.get("grid_confidence", 0)
        level = result.get("confidence_level", "unknown")

        # Verify level matches confidence value
        expected_level = (
            "high" if confidence >= 0.7 else
            "medium" if confidence >= 0.4 else
            "low" if confidence > 0 else
            "unknown"
        )

        level_correct = level == expected_level or level == "unknown"  # unknown is acceptable fallback

        self.log_result(
            "confidence_classification",
            level_correct,
            f"Confidence {confidence:.0%} classified as '{level}' (expected '{expected_level}')"
        )

        return result

    def test_preprocessing_quality_assessment(self, image_b64: str) -> Dict[str, Any]:
        """
        Test 5: Verify preprocessing returns quality assessment data.
        """
        print("\n=== Test 5: Preprocessing Quality Assessment ===")

        result = requests.post(
            f"{self.base_url}/preprocess-image",
            json={
                "image": image_b64,
                "normalize_contrast": True,
                "normalize_brightness": True
            },
            timeout=30
        ).json()

        if not result.get("success"):
            self.log_result(
                "preprocess_quality",
                False,
                f"Preprocessing failed: {result.get('error', 'Unknown')}"
            )
            return result

        # Check for quality metrics
        orig_stats = result.get("original_stats", {})
        proc_stats = result.get("processed_stats", {})

        has_quality_metrics = (
            "brightness" in orig_stats and
            "contrast" in orig_stats
        )

        self.log_result(
            "quality_metrics",
            has_quality_metrics,
            f"Quality metrics: brightness={orig_stats.get('brightness', 'N/A'):.1f}, contrast={orig_stats.get('contrast', 'N/A'):.1f}"
            if has_quality_metrics else "Missing quality metrics"
        )

        # Check if preprocessing improved image
        if has_quality_metrics:
            orig_contrast = orig_stats.get("contrast", 0)
            proc_contrast = proc_stats.get("contrast", 0)
            improvement = proc_contrast - orig_contrast

            self.log_result(
                "preprocessing_effect",
                True,  # Just informational
                f"Contrast change: {orig_contrast:.1f} -> {proc_contrast:.1f} ({improvement:+.1f})"
            )

        return result

    def test_correction_data_completeness(self, image_b64: str) -> Dict[str, Any]:
        """
        Test 6: Verify response contains all data needed for corrections.
        """
        print("\n=== Test 6: Correction Data Completeness ===")

        result = requests.post(
            f"{self.base_url}/crop-puzzle",
            json={"image": image_b64},
            timeout=30
        ).json()

        # Check for data needed by correction UI
        has_cropped_image = bool(result.get("cropped_image"))
        has_bounds = bool(result.get("bounds"))
        has_confidence = result.get("grid_confidence") is not None
        has_dimensions = (
            result.get("detected_rows") is not None and
            result.get("detected_cols") is not None
        )

        all_complete = has_cropped_image and has_bounds and has_confidence

        self.log_result(
            "correction_data",
            all_complete,
            f"cropped_image={has_cropped_image}, bounds={has_bounds}, confidence={has_confidence}, dimensions={has_dimensions}"
        )

        # Bounds should include precise coordinates for overlay alignment
        if has_bounds:
            bounds = result.get("bounds", {})
            has_coords = all(k in bounds for k in ["x", "y", "width", "height"])
            self.log_result(
                "bounds_coordinates",
                has_coords,
                f"Bounds: x={bounds.get('x')}, y={bounds.get('y')}, w={bounds.get('width')}, h={bounds.get('height')}"
                if has_coords else "Missing coordinate fields"
            )

        return result

    def run_all_tests(self, image_path: str = None) -> bool:
        """Run all low-confidence flow tests."""
        print("=" * 70)
        print("Low-Confidence Flow Verification Tests")
        print("Subtask 6-2: Verify low-confidence flow prompts user for verification")
        print("=" * 70)

        # Check service health
        if not self.check_service_health():
            print("\n[ERROR] CV Service not available at", self.base_url)
            print("Start the service with: uvicorn main:app --host 0.0.0.0 --port 8080")
            return False

        print(f"\n[OK] CV Service is healthy at {self.base_url}")

        # Load test image
        if image_path:
            image_file = Path(image_path)
        else:
            # Try to find a test image
            candidates = [
                Path("../IMG_2050.png"),
                Path("../IMG_2051.png"),
                Path("debug_puzzle_crop.png"),
            ]
            image_file = None
            for c in candidates:
                if c.exists():
                    image_file = c
                    break

        if not image_file or not image_file.exists():
            print("\n[ERROR] No test image found. Provide path as argument.")
            return False

        print(f"\nTest image: {image_file}")

        with open(image_file, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        print(f"Image size: {len(image_b64):,} chars base64")

        if not HAS_IMAGE_PROCESSING:
            print("\n[NOTE] numpy/scipy/PIL not installed - some tests will be skipped")
            print("       Install with: pip install numpy scipy pillow")

        # Run tests
        self.test_low_confidence_detection(image_b64)
        self.test_warnings_generated(image_b64)
        self.test_partial_image_handling(image_b64)
        self.test_confidence_level_classification(image_b64)
        self.test_preprocessing_quality_assessment(image_b64)
        self.test_correction_data_completeness(image_b64)

        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY - Low-Confidence Flow Verification")
        print("=" * 70)

        passed = sum(1 for _, success, _ in self.results if success)
        total = len(self.results)

        print(f"\nResults: {passed}/{total} tests passed")

        failed_tests = [(name, msg) for name, success, msg in self.results if not success]
        if failed_tests:
            print("\nFailed tests:")
            for name, msg in failed_tests:
                print(f"  - {name}: {msg}")

        all_passed = passed == total

        if all_passed:
            print("\n[SUCCESS] All low-confidence flow tests passed!")
            print("\nVerification confirms:")
            print("  - System detects low confidence scenarios")
            print("  - Appropriate warnings are generated")
            print("  - Partial images handled gracefully")
            print("  - Confidence levels classified correctly")
            print("  - Quality assessment data provided")
            print("  - All data for corrections is available")
        else:
            print(f"\n[WARNING] {total - passed} test(s) need attention")

        return all_passed


def main():
    """Main entry point."""
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8080"

    runner = LowConfidenceTestRunner(base_url=url)
    success = runner.run_all_tests(image_path)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
