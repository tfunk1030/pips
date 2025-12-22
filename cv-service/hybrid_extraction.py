"""
Hybrid CV + AI Extraction

Strategy:
1. CV: Detect puzzle ROI using saturation mask (fast, reliable)
2. CV: Crop image to puzzle region only
3. Return cropped image + bounds for AI to analyze

This reduces AI's task from "find puzzle in full screenshot" to
"analyze this cropped puzzle image" - much higher accuracy.
"""

import base64
import io
import cv2
import numpy as np
from typing import Optional, Tuple
from pydantic import BaseModel


class CropResult(BaseModel):
    success: bool
    error: Optional[str] = None

    # Cropped image as base64 PNG
    cropped_image: Optional[str] = None

    # Bounds in original image coordinates (includes padding for cropped image)
    bounds: Optional[dict] = None

    # Actual grid bounds (without padding) - use this for overlay alignment
    grid_bounds: Optional[dict] = None

    # Timing
    extraction_ms: int = 0


def find_puzzle_roi(img: np.ndarray, s_min: int = 25) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """
    Find puzzle ROI using saturation mask.
    The puzzle has colorful cells that pop against the dark background.

    Returns:
        (padded_bounds, actual_bounds) where each is (x, y, width, height)
        - padded_bounds: includes padding for cropping (gives context)
        - actual_bounds: the actual puzzle grid (for overlay alignment)
    """
    H, W = img.shape[:2]

    # Convert to HSV, extract saturation channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]

    # Threshold saturation to find colorful regions
    mask = ((S > s_min) * 255).astype(np.uint8)

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours, get largest (the puzzle board)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No colorful region found - try lowering s_min")

    # Get bounding box of largest contour - this is the ACTUAL grid bounds
    largest = max(contours, key=cv2.contourArea)
    ax, ay, aw, ah = cv2.boundingRect(largest)
    actual_bounds = (ax, ay, aw, ah)

    # Add padding for cropped image (gives AI more context)
    pad = int(0.05 * max(aw, ah))
    px = max(0, ax - pad)
    py = max(0, ay - pad)
    pw = min(W - px, aw + 2 * pad)
    ph = min(H - py, ah + 2 * pad)
    padded_bounds = (px, py, pw, ph)

    return padded_bounds, actual_bounds


def find_domino_tray(img: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find the domino tray region using edge detection.
    Dominoes are white/gray rectangles with black dots.

    Returns (x, y, width, height) of the domino tray area.
    """
    H, W = img.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find edges (dominoes have strong rectangular edges)
    edges = cv2.Canny(gray, 50, 150)

    # Dilate to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: return bottom portion of image
        return 0, int(H * 0.6), W, int(H * 0.4)

    # Find contours that look like domino-sized rectangles
    # Dominoes are typically wider than tall, with aspect ratio ~2:1
    domino_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = w / max(h, 1)

        # Domino-like: reasonable size, horizontal orientation
        if area > 1000 and 0.3 < aspect < 5:
            domino_contours.append((x, y, w, h))

    if not domino_contours:
        # Fallback
        return 0, int(H * 0.6), W, int(H * 0.4)

    # Get bounding box of all domino-like contours
    min_x = min(c[0] for c in domino_contours)
    min_y = min(c[1] for c in domino_contours)
    max_x = max(c[0] + c[2] for c in domino_contours)
    max_y = max(c[1] + c[3] for c in domino_contours)

    # Add small padding
    pad = 20
    min_x = max(0, min_x - pad)
    min_y = max(0, min_y - pad)
    max_x = min(W, max_x + pad)
    max_y = min(H, max_y + pad)

    return min_x, min_y, max_x - min_x, max_y - min_y


def crop_domino_region(
    base64_image: str,
    puzzle_bottom_y: int = None
) -> CropResult:
    """
    Crop image to domino tray region only.
    Uses edge detection to find the actual domino tray, not just everything below puzzle.

    Args:
        base64_image: Full screenshot as base64
        puzzle_bottom_y: If provided, only search below this Y coordinate

    Returns:
        CropResult with cropped domino tray image and bounds
    """
    import time
    start = time.time()

    try:
        # Decode image
        if "base64," in base64_image:
            base64_image = base64_image.split("base64,")[1]

        img_bytes = base64.b64decode(base64_image)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return CropResult(
                success=False,
                error="Failed to decode image",
                extraction_ms=int((time.time() - start) * 1000)
            )

        H, W = img.shape[:2]

        # If puzzle_bottom_y provided, only look in that region
        if puzzle_bottom_y is not None:
            search_region = img[puzzle_bottom_y:, :]
            offset_y = puzzle_bottom_y
        else:
            # Search bottom 50% of image for dominoes
            search_start = int(H * 0.5)
            search_region = img[search_start:, :]
            offset_y = search_start

        # Find domino tray within search region
        dx, dy, dw, dh = find_domino_tray(search_region)

        # Convert to full image coordinates
        domino_x = dx
        domino_y = offset_y + dy
        domino_w = dw
        domino_h = dh

        # Crop the domino region
        cropped = img[domino_y:domino_y+domino_h, domino_x:domino_x+domino_w]

        if cropped.size == 0:
            return CropResult(
                success=False,
                error="Domino region is empty",
                extraction_ms=int((time.time() - start) * 1000)
            )

        # Encode cropped image as PNG base64
        _, buffer = cv2.imencode('.png', cropped)
        cropped_b64 = base64.b64encode(buffer).decode('utf-8')

        return CropResult(
            success=True,
            cropped_image=cropped_b64,
            bounds={
                "x": domino_x,
                "y": domino_y,
                "width": domino_w,
                "height": domino_h,
                "original_width": W,
                "original_height": H
            },
            extraction_ms=int((time.time() - start) * 1000)
        )

    except Exception as e:
        return CropResult(
            success=False,
            error=str(e),
            extraction_ms=int((time.time() - start) * 1000)
        )


def crop_puzzle_region(
    base64_image: str,
    exclude_bottom_percent: float = 0.05  # Exclude only a tiny bit (just buffer)
) -> CropResult:
    """
    Crop image to puzzle region only.

    Args:
        base64_image: Full screenshot as base64
        exclude_bottom_percent: Exclude bottom X% of ROI (domino tray)

    Returns:
        CropResult with cropped image, bounds (padded), and grid_bounds (actual)
    """
    import time
    start = time.time()

    try:
        # Decode image
        if "base64," in base64_image:
            base64_image = base64_image.split("base64,")[1]

        img_bytes = base64.b64decode(base64_image)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return CropResult(
                success=False,
                error="Failed to decode image",
                extraction_ms=int((time.time() - start) * 1000)
            )

        H, W = img.shape[:2]

        # Find puzzle ROI - returns both padded and actual bounds
        (x, y, w, h), (gx, gy, gw, gh) = find_puzzle_roi(img)

        # Exclude bottom portion (domino tray is often below puzzle)
        if exclude_bottom_percent > 0:
            h = int(h * (1 - exclude_bottom_percent))

        # Crop using padded bounds
        cropped = img[y:y+h, x:x+w]

        # Encode cropped image as PNG base64
        _, buffer = cv2.imencode('.png', cropped)
        cropped_b64 = base64.b64encode(buffer).decode('utf-8')

        return CropResult(
            success=True,
            cropped_image=cropped_b64,
            bounds={
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "original_width": W,
                "original_height": H
            },
            # Actual grid bounds (without padding) - use this for overlay alignment
            grid_bounds={
                "x": gx,
                "y": gy,
                "width": gw,
                "height": gh,
                "original_width": W,
                "original_height": H
            },
            extraction_ms=int((time.time() - start) * 1000)
        )

    except Exception as e:
        return CropResult(
            success=False,
            error=str(e),
            extraction_ms=int((time.time() - start) * 1000)
        )


# Test
if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python hybrid_extraction.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    # Read and encode
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    print(f"Image loaded: {len(b64)} chars")

    # Crop
    result = crop_puzzle_region(b64)

    if result.success:
        print(f"\n[OK] Cropped in {result.extraction_ms}ms")
        print(f"  Bounds: {result.bounds}")
        print(f"  Cropped image: {len(result.cropped_image)} chars base64")

        # Save cropped image for inspection
        out_path = image_path.parent / f"{image_path.stem}_cropped.png"
        img_bytes = base64.b64decode(result.cropped_image)
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        print(f"  Saved: {out_path}")
    else:
        print(f"\n[FAIL] {result.error}")
