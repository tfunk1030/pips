#!/usr/bin/env python3
"""Quick test script for CV service"""

import base64
import sys
import requests
from pathlib import Path

def test_extraction(image_path: str, url: str = "http://localhost:8080"):
    """Test CV extraction with a real image"""

    # Read and encode image
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Error: Image not found: {image_path}")
        return

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    b64 = base64.b64encode(img_bytes).decode("utf-8")
    print(f"Image loaded: {len(b64)} chars base64")

    # Call service
    print(f"Calling {url}/extract-geometry...")
    response = requests.post(
        f"{url}/extract-geometry",
        json={"image": b64, "lower_half_only": False}  # Scan full image for grid
    )

    result = response.json()

    if result["success"]:
        print(f"\n[OK] Success! Extracted in {result['extraction_ms']}ms")
        print(f"  Grid: {result['rows']}x{result['cols']}")
        print(f"  Cells: {len(result['cells'])}")
        print(f"  Shape:\n{result['shape']}")
        if result.get("grid_bounds"):
            bounds = result["grid_bounds"]
            print(f"  Bounds: ({bounds['left']},{bounds['top']}) to ({bounds['right']},{bounds['bottom']})")
    else:
        print(f"\n[FAIL] Error: {result['error']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_service.py <image_path> [service_url]")
        print("\nExample:")
        print("  python test_service.py ../debug/01_roi.png")
        print("  python test_service.py screenshot.png http://localhost:8080")
        sys.exit(1)

    image_path = sys.argv[1]
    url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8080"

    test_extraction(image_path, url)
