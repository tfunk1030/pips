#!/usr/bin/env python3
"""Test the /crop-puzzle endpoint"""

import base64
import sys
import requests
from pathlib import Path

def test_crop(image_path: str, url: str = "http://localhost:8080"):
    # Read and encode
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Error: Image not found: {image_path}")
        return

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    print(f"Original image: {len(b64):,} chars base64")

    # Call crop endpoint
    print(f"Calling {url}/crop-puzzle...")
    response = requests.post(
        f"{url}/crop-puzzle",
        json={"image": b64}
    )

    result = response.json()

    if result["success"]:
        cropped_len = len(result["cropped_image"])
        reduction = (1 - cropped_len / len(b64)) * 100

        print(f"\n[OK] Cropped in {result['extraction_ms']}ms")
        print(f"  Cropped image: {cropped_len:,} chars ({reduction:.0f}% smaller)")
        print(f"  Bounds: {result['bounds']}")

        # Save for inspection
        out_path = img_path.parent / f"{img_path.stem}_cropped.png"
        img_bytes = base64.b64decode(result["cropped_image"])
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        print(f"  Saved: {out_path}")
    else:
        print(f"\n[FAIL] {result['error']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_crop_endpoint.py <image_path>")
        sys.exit(1)

    test_crop(sys.argv[1])
