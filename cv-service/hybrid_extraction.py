"""
Hybrid CV + AI Extraction

Strategy:
1. CV: Detect puzzle ROI using multiple techniques:
   - Saturation mask (for colorful cells)
   - Adaptive thresholding (for grid line detection)
   - Hough line detection (for precise grid boundaries)
2. CV: Crop image to puzzle region only
3. Return cropped image + bounds for AI to analyze

This reduces AI's task from "find puzzle in full screenshot" to
"analyze this cropped puzzle image" - much higher accuracy.
"""

import base64
import io
import cv2
import numpy as np
from typing import Optional, Tuple, List
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

    # Grid line detection confidence (0-1)
    grid_confidence: Optional[float] = None

    # Detected grid dimensions (if found via line detection)
    detected_rows: Optional[int] = None
    detected_cols: Optional[int] = None

    # Timing
    extraction_ms: int = 0


class GridLineResult:
    """Result from grid line detection"""
    def __init__(self):
        self.horizontal_lines: List[int] = []  # Y coordinates
        self.vertical_lines: List[int] = []    # X coordinates
        self.grid_bounds: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
        self.confidence: float = 0.0
        self.rows: int = 0
        self.cols: int = 0


def detect_grid_lines_adaptive(img: np.ndarray,
                               block_size: int = 15,
                               c_value: int = 5) -> GridLineResult:
    """
    Detect grid lines using adaptive thresholding.

    Adaptive thresholding works better than global thresholding because
    it handles varying lighting conditions across the puzzle screenshot.

    Args:
        img: Input image (BGR)
        block_size: Size of neighborhood for adaptive threshold (must be odd)
        c_value: Constant subtracted from mean

    Returns:
        GridLineResult with detected lines and bounds
    """
    result = GridLineResult()
    H, W = img.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding - this highlights edges and grid lines
    # ADAPTIVE_THRESH_GAUSSIAN_C uses weighted sum of neighborhood
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, c_value
    )

    # Also try mean-based for comparison
    adaptive_mean = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size + 4, c_value + 2
    )

    # Combine both adaptive results
    combined = cv2.bitwise_or(adaptive, adaptive_mean)

    # Morphological operations to enhance grid lines
    # Horizontal kernel to detect horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal = cv2.morphologyEx(combined, cv2.MORPH_OPEN, h_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, h_kernel, iterations=1)

    # Vertical kernel to detect vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    vertical = cv2.morphologyEx(combined, cv2.MORPH_OPEN, v_kernel, iterations=1)
    vertical = cv2.dilate(vertical, v_kernel, iterations=1)

    # Combine horizontal and vertical lines
    grid_lines = cv2.bitwise_or(horizontal, vertical)

    # Use Hough line detection for precise line positions
    lines = cv2.HoughLinesP(
        grid_lines,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=min(W, H) // 10,
        maxLineGap=10
    )

    if lines is not None:
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Determine if line is horizontal or vertical
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 15 or angle > 165:  # Horizontal (within 15 degrees)
                y_mid = (y1 + y2) // 2
                h_lines.append(y_mid)
            elif 75 < angle < 105:  # Vertical (within 15 degrees of 90)
                x_mid = (x1 + x2) // 2
                v_lines.append(x_mid)

        # Cluster lines to remove duplicates
        result.horizontal_lines = _cluster_lines(h_lines, threshold=15)
        result.vertical_lines = _cluster_lines(v_lines, threshold=15)

        # Calculate grid bounds from detected lines
        if len(result.horizontal_lines) >= 2 and len(result.vertical_lines) >= 2:
            min_x = min(result.vertical_lines)
            max_x = max(result.vertical_lines)
            min_y = min(result.horizontal_lines)
            max_y = max(result.horizontal_lines)
            result.grid_bounds = (min_x, min_y, max_x - min_x, max_y - min_y)

            # Estimate rows and columns from line spacing
            result.rows = max(1, len(result.horizontal_lines) - 1)
            result.cols = max(1, len(result.vertical_lines) - 1)

            # Calculate confidence based on grid regularity
            result.confidence = _calculate_grid_confidence(
                result.horizontal_lines,
                result.vertical_lines
            )

    return result


def _cluster_lines(lines: List[int], threshold: int = 15) -> List[int]:
    """
    Cluster nearby line positions and return representative values.

    Args:
        lines: List of line positions (x or y coordinates)
        threshold: Maximum distance to consider lines as same

    Returns:
        List of clustered line positions (sorted)
    """
    if not lines:
        return []

    lines = sorted(lines)
    clusters = [[lines[0]]]

    for line in lines[1:]:
        if line - clusters[-1][-1] <= threshold:
            clusters[-1].append(line)
        else:
            clusters.append([line])

    # Return mean of each cluster
    return sorted([int(np.mean(cluster)) for cluster in clusters])


def _calculate_grid_confidence(h_lines: List[int], v_lines: List[int]) -> float:
    """
    Calculate confidence score based on grid regularity.

    A regular grid should have evenly spaced lines.

    Args:
        h_lines: Horizontal line positions
        v_lines: Vertical line positions

    Returns:
        Confidence score 0.0-1.0
    """
    if len(h_lines) < 2 or len(v_lines) < 2:
        return 0.0

    # Calculate spacing regularity for horizontal lines
    h_spacings = np.diff(h_lines)
    h_std = np.std(h_spacings) / max(np.mean(h_spacings), 1)

    # Calculate spacing regularity for vertical lines
    v_spacings = np.diff(v_lines)
    v_std = np.std(v_spacings) / max(np.mean(v_spacings), 1)

    # Lower std means more regular = higher confidence
    # std of 0 = perfect regularity = confidence 1.0
    # std of 0.5+ = irregular = confidence approaching 0
    h_confidence = max(0, 1 - h_std * 2)
    v_confidence = max(0, 1 - v_std * 2)

    # Also factor in number of lines detected (more lines = more confident)
    line_count_bonus = min(1, (len(h_lines) + len(v_lines)) / 10)

    return (h_confidence + v_confidence) / 2 * (0.5 + 0.5 * line_count_bonus)


def detect_grid_with_canny(img: np.ndarray) -> GridLineResult:
    """
    Alternative grid detection using Canny edge detection.

    This is a fallback method when adaptive thresholding doesn't
    produce good results (e.g., low contrast images).

    Args:
        img: Input image (BGR)

    Returns:
        GridLineResult with detected lines and bounds
    """
    result = GridLineResult()
    H, W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Auto-calculate Canny thresholds using Otsu's method
    high_thresh, _ = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = high_thresh * 0.5

    # Apply Canny edge detection
    edges = cv2.Canny(filtered, low_thresh, high_thresh)

    # Dilate edges to connect nearby edge pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Use probabilistic Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=80,
        minLineLength=min(W, H) // 8,
        maxLineGap=15
    )

    if lines is not None:
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Weight by line length
            if angle < 10 or angle > 170:  # Horizontal
                h_lines.extend([(y1 + y2) // 2] * int(length // 50 + 1))
            elif 80 < angle < 100:  # Vertical
                v_lines.extend([(x1 + x2) // 2] * int(length // 50 + 1))

        result.horizontal_lines = _cluster_lines(h_lines, threshold=20)
        result.vertical_lines = _cluster_lines(v_lines, threshold=20)

        if len(result.horizontal_lines) >= 2 and len(result.vertical_lines) >= 2:
            min_x = min(result.vertical_lines)
            max_x = max(result.vertical_lines)
            min_y = min(result.horizontal_lines)
            max_y = max(result.horizontal_lines)
            result.grid_bounds = (min_x, min_y, max_x - min_x, max_y - min_y)
            result.rows = max(1, len(result.horizontal_lines) - 1)
            result.cols = max(1, len(result.vertical_lines) - 1)
            result.confidence = _calculate_grid_confidence(
                result.horizontal_lines,
                result.vertical_lines
            ) * 0.9  # Slightly lower confidence for Canny method

    return result


def find_puzzle_roi_saturation(img: np.ndarray, s_min: int = 25) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """
    Find puzzle ROI using saturation mask (original method).
    The puzzle has colorful cells that pop against the dark background.

    Returns:
        (padded_bounds, actual_bounds) where each is (x, y, width, height)
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


def find_puzzle_roi(img: np.ndarray, s_min: int = 25) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], Optional[GridLineResult]]:
    """
    Find puzzle ROI using multiple techniques for best accuracy.

    Strategy:
    1. Try adaptive thresholding for grid line detection
    2. Try Canny edge detection as fallback
    3. Fall back to saturation-based detection
    4. Combine results for best bounds

    Returns:
        (padded_bounds, actual_bounds, grid_line_result) where:
        - padded_bounds: includes padding for cropping (gives context)
        - actual_bounds: the actual puzzle grid (for overlay alignment)
        - grid_line_result: detailed grid line detection results (or None)
    """
    H, W = img.shape[:2]

    # Method 1: Adaptive thresholding for grid lines
    adaptive_result = detect_grid_lines_adaptive(img)

    # Method 2: Canny edge detection (fallback)
    canny_result = None
    if adaptive_result.confidence < 0.5:
        canny_result = detect_grid_with_canny(img)

    # Method 3: Saturation-based detection (always run as baseline)
    try:
        sat_padded, sat_actual = find_puzzle_roi_saturation(img, s_min)
    except ValueError:
        # If saturation fails, rely entirely on line detection
        sat_padded = None
        sat_actual = None

    # Determine best bounds based on confidence
    best_result = adaptive_result
    if canny_result and canny_result.confidence > adaptive_result.confidence:
        best_result = canny_result

    # If grid line detection found good bounds, use those
    if best_result.grid_bounds and best_result.confidence > 0.3:
        gx, gy, gw, gh = best_result.grid_bounds

        # Refine using saturation bounds if available (intersection)
        if sat_actual:
            sx, sy, sw, sh = sat_actual
            # Use intersection to get tighter bounds
            final_x = max(gx, sx)
            final_y = max(gy, sy)
            final_w = min(gx + gw, sx + sw) - final_x
            final_h = min(gy + gh, sy + sh) - final_y

            # Only use intersection if it's reasonably sized
            if final_w > 50 and final_h > 50:
                actual_bounds = (final_x, final_y, final_w, final_h)
            else:
                # Grid detection bounds are probably more accurate
                actual_bounds = (gx, gy, gw, gh)
        else:
            actual_bounds = (gx, gy, gw, gh)
    elif sat_actual:
        # Fall back to saturation-based bounds
        actual_bounds = sat_actual
    else:
        raise ValueError("Could not detect puzzle region with any method")

    # Calculate padded bounds
    ax, ay, aw, ah = actual_bounds
    pad = int(0.05 * max(aw, ah))
    px = max(0, ax - pad)
    py = max(0, ay - pad)
    pw = min(W - px, aw + 2 * pad)
    ph = min(H - py, ah + 2 * pad)
    padded_bounds = (px, py, pw, ph)

    return padded_bounds, actual_bounds, best_result


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

    Uses multiple detection techniques for best accuracy:
    1. Adaptive thresholding for grid line detection
    2. Canny edge detection (fallback)
    3. Saturation-based detection (baseline)

    Args:
        base64_image: Full screenshot as base64
        exclude_bottom_percent: Exclude bottom X% of ROI (domino tray)

    Returns:
        CropResult with cropped image, bounds (padded), grid_bounds (actual),
        plus grid detection confidence and estimated dimensions
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

        # Find puzzle ROI - returns padded bounds, actual bounds, and grid line result
        (x, y, w, h), (gx, gy, gw, gh), grid_result = find_puzzle_roi(img)

        # Exclude bottom portion (domino tray is often below puzzle)
        if exclude_bottom_percent > 0:
            h = int(h * (1 - exclude_bottom_percent))

        # Crop using padded bounds
        cropped = img[y:y+h, x:x+w]

        # Encode cropped image as PNG base64
        _, buffer = cv2.imencode('.png', cropped)
        cropped_b64 = base64.b64encode(buffer).decode('utf-8')

        # Extract grid detection info if available
        grid_confidence = grid_result.confidence if grid_result else None
        detected_rows = grid_result.rows if grid_result and grid_result.rows > 0 else None
        detected_cols = grid_result.cols if grid_result and grid_result.cols > 0 else None

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
            grid_confidence=grid_confidence,
            detected_rows=detected_rows,
            detected_cols=detected_cols,
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
        print(f"  Padded bounds: {result.bounds}")
        print(f"  Grid bounds: {result.grid_bounds}")
        print(f"  Grid confidence: {result.grid_confidence:.2f}" if result.grid_confidence else "  Grid confidence: N/A")
        print(f"  Detected dimensions: {result.detected_rows}x{result.detected_cols}" if result.detected_rows else "  Detected dimensions: N/A")
        print(f"  Cropped image: {len(result.cropped_image)} chars base64")

        # Save cropped image for inspection
        out_path = image_path.parent / f"{image_path.stem}_cropped.png"
        img_bytes = base64.b64decode(result.cropped_image)
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        print(f"  Saved: {out_path}")
    else:
        print(f"\n[FAIL] {result.error}")
