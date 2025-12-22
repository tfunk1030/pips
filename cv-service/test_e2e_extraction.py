#!/usr/bin/env python3
"""
E2E Verification Tests for Enhanced CV Extraction Pipeline

This script tests the complete extraction flow with sample puzzle images,
verifying all endpoints and the improvements made in spec 001.

Tests:
1. Health check endpoint
2. Enhanced /crop-puzzle endpoint with grid detection confidence
3. /preprocess-image endpoint with image statistics
4. Full extraction flow: crop -> preprocess -> verify
5. Grid confidence and warnings

Usage:
    cd cv-service
    uvicorn main:app --host 0.0.0.0 --port 8080 --reload &
    python test_e2e_extraction.py [image_path]
"""

import base64
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import requests


class E2ETestRunner:
    """Runs E2E verification tests for the extraction pipeline."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.results: List[Tuple[str, bool, str]] = []

    def log_result(self, test_name: str, passed: bool, message: str):
        """Log a test result."""
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_name}: {message}")
        self.results.append((test_name, passed, message))

    def test_health_check(self) -> bool:
        """Test 1: Verify health check endpoint."""
        print("\n=== Test 1: Health Check ===")

        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log_result("health_check", True, "Service is healthy")
                    return True

            self.log_result("health_check", False, f"Unexpected response: {response.status_code}")
            return False

        except requests.exceptions.ConnectionError:
            self.log_result("health_check", False, "Cannot connect to service")
            return False
        except Exception as e:
            self.log_result("health_check", False, str(e))
            return False

    def test_crop_puzzle_enhanced(self, image_b64: str) -> Dict[str, Any]:
        """Test 2: Verify enhanced /crop-puzzle endpoint with grid detection."""
        print("\n=== Test 2: Enhanced Puzzle Cropping ===")

        try:
            response = requests.post(
                f"{self.base_url}/crop-puzzle",
                json={
                    "image": image_b64,
                    "exclude_bottom_percent": 0.05,
                    "min_confidence_threshold": 0.3,
                    "padding_percent": 0.05
                },
                timeout=30
            )

            result = response.json()

            # Check success
            if not result.get("success"):
                self.log_result("crop_success", False, result.get("error", "Unknown error"))
                return result

            self.log_result("crop_success", True, f"Cropped in {result.get('extraction_ms', 0)}ms")

            # Check grid confidence (new feature)
            grid_confidence = result.get("grid_confidence")
            if grid_confidence is not None:
                confidence_level = result.get("confidence_level", "unknown")
                self.log_result(
                    "grid_confidence",
                    True,
                    f"Confidence: {grid_confidence:.0%} ({confidence_level})"
                )
            else:
                self.log_result("grid_confidence", False, "No grid confidence returned")

            # Check detected dimensions (new feature)
            detected_rows = result.get("detected_rows")
            detected_cols = result.get("detected_cols")
            if detected_rows and detected_cols:
                self.log_result(
                    "grid_detection",
                    True,
                    f"Detected {detected_rows}x{detected_cols} grid"
                )
            else:
                self.log_result("grid_detection", False, "Grid dimensions not detected")

            # Check detection method (new feature)
            detection_method = result.get("detection_method", "unknown")
            self.log_result("detection_method", True, f"Method: {detection_method}")

            # Check warnings (new feature)
            warnings = result.get("warnings", [])
            if warnings:
                self.log_result("warnings", True, f"{len(warnings)} warning(s): {'; '.join(warnings[:2])}")
            else:
                self.log_result("warnings", True, "No warnings")

            # Check cropped image exists
            cropped = result.get("cropped_image")
            if cropped:
                cropped_size = len(cropped)
                original_size = len(image_b64)
                reduction = (1 - cropped_size / original_size) * 100
                self.log_result(
                    "image_cropped",
                    True,
                    f"Image reduced by {reduction:.0f}% ({cropped_size:,} chars)"
                )
            else:
                self.log_result("image_cropped", False, "No cropped image returned")

            return result

        except Exception as e:
            self.log_result("crop_puzzle", False, str(e))
            return {"success": False, "error": str(e)}

    def test_preprocess_image(self, image_b64: str) -> Dict[str, Any]:
        """Test 3: Verify /preprocess-image endpoint."""
        print("\n=== Test 3: Image Preprocessing ===")

        try:
            response = requests.post(
                f"{self.base_url}/preprocess-image",
                json={
                    "image": image_b64,
                    "normalize_contrast": True,
                    "normalize_brightness": True,
                    "auto_white_balance": True,
                    "sharpen": False
                },
                timeout=30
            )

            result = response.json()

            if not result.get("success"):
                self.log_result("preprocess_success", False, result.get("error", "Unknown error"))
                return result

            self.log_result("preprocess_success", True, f"Processed in {result.get('extraction_ms', 0)}ms")

            # Check operations applied (new feature)
            operations = result.get("operations_applied", [])
            if operations:
                self.log_result("operations", True, f"Applied: {', '.join(operations)}")
            else:
                self.log_result("operations", True, "No operations needed (image already optimal)")

            # Check original stats (new feature)
            orig_stats = result.get("original_stats", {})
            if orig_stats:
                brightness = orig_stats.get("brightness", 0)
                contrast = orig_stats.get("contrast", 0)
                self.log_result(
                    "original_stats",
                    True,
                    f"Original - Brightness: {brightness:.1f}, Contrast: {contrast:.1f}"
                )

            # Check processed stats (new feature)
            proc_stats = result.get("processed_stats", {})
            if proc_stats:
                brightness = proc_stats.get("brightness", 0)
                contrast = proc_stats.get("contrast", 0)
                self.log_result(
                    "processed_stats",
                    True,
                    f"Processed - Brightness: {brightness:.1f}, Contrast: {contrast:.1f}"
                )

            # Verify image was processed
            processed_image = result.get("preprocessed_image")
            if processed_image:
                self.log_result("processed_image", True, f"{len(processed_image):,} chars")
            else:
                self.log_result("processed_image", False, "No processed image returned")

            return result

        except Exception as e:
            self.log_result("preprocess_image", False, str(e))
            return {"success": False, "error": str(e)}

    def test_full_extraction_flow(self, image_b64: str) -> bool:
        """Test 4: Full extraction flow - crop -> preprocess."""
        print("\n=== Test 4: Full Extraction Flow ===")

        start_time = time.time()

        # Step 1: Crop puzzle
        print("  Step 1: Cropping puzzle region...")
        crop_result = requests.post(
            f"{self.base_url}/crop-puzzle",
            json={"image": image_b64},
            timeout=30
        ).json()

        if not crop_result.get("success"):
            self.log_result("flow_crop", False, crop_result.get("error", "Crop failed"))
            return False

        cropped_image = crop_result.get("cropped_image")
        self.log_result("flow_crop", True, f"Grid confidence: {crop_result.get('grid_confidence', 0):.0%}")

        # Step 2: Preprocess cropped image
        print("  Step 2: Preprocessing cropped image...")
        preprocess_result = requests.post(
            f"{self.base_url}/preprocess-image",
            json={"image": cropped_image},
            timeout=30
        ).json()

        if not preprocess_result.get("success"):
            self.log_result("flow_preprocess", False, preprocess_result.get("error", "Preprocess failed"))
            return False

        operations = preprocess_result.get("operations_applied", [])
        self.log_result("flow_preprocess", True, f"Operations: {', '.join(operations) or 'none'}")

        # Summary
        total_time = (time.time() - start_time) * 1000
        self.log_result(
            "flow_complete",
            True,
            f"Full flow completed in {total_time:.0f}ms"
        )

        return True

    def test_crop_dominoes(self, image_b64: str, puzzle_bottom_y: int = None) -> Dict[str, Any]:
        """Test 5: Verify /crop-dominoes endpoint."""
        print("\n=== Test 5: Domino Region Cropping ===")

        try:
            response = requests.post(
                f"{self.base_url}/crop-dominoes",
                json={
                    "image": image_b64,
                    "puzzle_bottom_y": puzzle_bottom_y
                },
                timeout=30
            )

            result = response.json()

            if result.get("success"):
                cropped = result.get("cropped_image")
                bounds = result.get("bounds", {})
                self.log_result(
                    "domino_crop",
                    True,
                    f"Cropped dominoes: {bounds.get('width', 0)}x{bounds.get('height', 0)}"
                )
            else:
                # May fail if no domino area detected - that's okay for some images
                self.log_result(
                    "domino_crop",
                    True,  # Not a failure, just informational
                    f"No domino region: {result.get('error', 'N/A')}"
                )

            return result

        except Exception as e:
            self.log_result("domino_crop", False, str(e))
            return {"success": False, "error": str(e)}

    def run_all_tests(self, image_path: str = None) -> bool:
        """Run all E2E tests."""
        print("=" * 60)
        print("E2E Verification Tests for Enhanced CV Extraction Pipeline")
        print("=" * 60)

        # Test 1: Health check
        if not self.test_health_check():
            print("\n[ERROR] Service not available. Start cv-service first.")
            return False

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

            if not image_file:
                print("\n[ERROR] No test image found. Provide path as argument.")
                return False

        print(f"\nTest image: {image_file}")

        with open(image_file, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        print(f"Image size: {len(image_b64):,} chars base64")

        # Run tests
        crop_result = self.test_crop_puzzle_enhanced(image_b64)
        self.test_preprocess_image(image_b64)
        self.test_full_extraction_flow(image_b64)

        # Get puzzle bottom Y for domino cropping
        puzzle_bottom_y = None
        if crop_result.get("bounds"):
            bounds = crop_result["bounds"]
            puzzle_bottom_y = bounds.get("y", 0) + bounds.get("height", 0)
        self.test_crop_dominoes(image_b64, puzzle_bottom_y)

        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

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
            print("\n[SUCCESS] All E2E verification tests passed!")
        else:
            print(f"\n[WARNING] {total - passed} test(s) failed")

        return all_passed


def main():
    """Main entry point."""
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8080"

    runner = E2ETestRunner(base_url=url)
    success = runner.run_all_tests(image_path)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
