"""
CV Extraction Service
Exposes existing Python CV code via FastAPI for React Native app
"""

import base64
import io
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import existing CV code
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from extract_board_cells_gridlines import extract_cells_from_screenshot

app = FastAPI(title="Pips CV Service", version="1.0.0")

# Enable CORS for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    image: str  # base64 encoded
    lower_half_only: bool = True


class CellBounds(BaseModel):
    x: int
    y: int
    width: int
    height: int
    row: int
    col: int


class ExtractResponse(BaseModel):
    success: bool
    error: Optional[str] = None

    # Grid structure
    rows: int = 0
    cols: int = 0

    # Cell data
    cells: List[CellBounds] = []

    # Shape string (. = cell, # = hole)
    shape: str = ""

    # Grid bounds in image coordinates
    grid_bounds: Optional[dict] = None

    # Timing
    extraction_ms: int = 0


def decode_image(base64_str: str) -> np.ndarray:
    """Decode base64 image to OpenCV format"""
    # Strip data URL prefix if present
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]

    img_bytes = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image")

    return img


def cells_to_grid(cells: List[Tuple[int, int, int, int]]) -> Tuple[int, int, List[CellBounds], str]:
    """
    Convert cell bounding boxes to grid structure with hole detection.
    Returns (rows, cols, cell_bounds, shape_string)
    """
    if not cells:
        return 0, 0, [], ""

    # Sort cells by position
    cells_sorted = sorted(cells, key=lambda c: (c[1], c[0]))  # Sort by y, then x

    # Group into rows (cells with similar y-coordinates)
    rows_grouped = []
    current_row = [cells_sorted[0]]
    y_threshold = cells_sorted[0][3] * 0.5  # 50% of cell height

    for cell in cells_sorted[1:]:
        if abs(cell[1] - current_row[0][1]) <= y_threshold:
            current_row.append(cell)
        else:
            rows_grouped.append(sorted(current_row, key=lambda c: c[0]))  # Sort by x
            current_row = [cell]
    rows_grouped.append(sorted(current_row, key=lambda c: c[0]))

    num_rows = len(rows_grouped)
    num_cols = max(len(row) for row in rows_grouped)

    # Calculate expected cell positions
    # Find median cell size
    widths = [c[2] for c in cells]
    heights = [c[3] for c in cells]
    median_w = int(np.median(widths))
    median_h = int(np.median(heights))

    # Find grid origin (top-left cell)
    min_x = min(c[0] for c in cells)
    min_y = min(c[1] for c in cells)

    # Build grid with hole detection
    grid = [['#'] * num_cols for _ in range(num_rows)]
    cell_bounds = []

    for row_idx, row in enumerate(rows_grouped):
        for cell in row:
            x, y, w, h = cell
            # Determine column based on x position
            col_idx = round((x - min_x) / median_w)
            col_idx = max(0, min(col_idx, num_cols - 1))

            if grid[row_idx][col_idx] == '#':
                grid[row_idx][col_idx] = '.'
                cell_bounds.append(CellBounds(
                    x=x, y=y, width=w, height=h,
                    row=row_idx, col=col_idx
                ))

    # Generate shape string
    shape = '\n'.join(''.join(row) for row in grid)

    return num_rows, num_cols, cell_bounds, shape


@app.post("/extract-geometry", response_model=ExtractResponse)
async def extract_geometry(request: ExtractRequest):
    """
    Extract grid geometry from puzzle screenshot.
    Returns cell positions, holes, and grid dimensions.
    """
    import time
    start = time.time()

    try:
        # Decode image
        img = decode_image(request.image)

        # Save to temp file for CV processing
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            cv2.imwrite(f.name, img)
            temp_path = f.name

        # Run CV extraction
        temp_out = tempfile.mkdtemp()
        try:
            extract_cells_from_screenshot(
                temp_path,
                out_dir=temp_out,
                lower_half_only=request.lower_half_only
            )

            # Read cells.txt output
            cells = []
            cells_file = Path("cells.txt")
            if cells_file.exists():
                with open(cells_file, "r") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) == 4:
                            x, y, w, h = map(int, parts)
                            cells.append((x, y, w, h))

            if not cells:
                return ExtractResponse(
                    success=False,
                    error="No cells detected in image",
                    extraction_ms=int((time.time() - start) * 1000)
                )

            # Convert to grid structure
            rows, cols, cell_bounds, shape = cells_to_grid(cells)

            # Calculate grid bounds
            min_x = min(c.x for c in cell_bounds)
            min_y = min(c.y for c in cell_bounds)
            max_x = max(c.x + c.width for c in cell_bounds)
            max_y = max(c.y + c.height for c in cell_bounds)

            return ExtractResponse(
                success=True,
                rows=rows,
                cols=cols,
                cells=cell_bounds,
                shape=shape,
                grid_bounds={
                    "left": min_x,
                    "top": min_y,
                    "right": max_x,
                    "bottom": max_y,
                    "imageWidth": img.shape[1],
                    "imageHeight": img.shape[0]
                },
                extraction_ms=int((time.time() - start) * 1000)
            )

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_out, ignore_errors=True)
            Path(temp_path).unlink(missing_ok=True)
            Path("cells.txt").unlink(missing_ok=True)

    except Exception as e:
        return ExtractResponse(
            success=False,
            error=str(e),
            extraction_ms=int((time.time() - start) * 1000)
        )


class CropPuzzleRequest(BaseModel):
    """Request model for puzzle cropping with enhanced options."""
    image: str  # base64 encoded
    # Boundary detection options
    exclude_bottom_percent: float = 0.05  # Exclude bottom X% of ROI (domino tray buffer)
    min_confidence_threshold: float = 0.3  # Minimum confidence for grid detection
    # Advanced options
    use_adaptive_threshold: bool = True  # Use adaptive thresholding for grid lines
    use_canny_fallback: bool = True  # Use Canny edge detection as fallback
    padding_percent: float = 0.05  # Padding around detected grid (0.0 - 0.15)


class CropPuzzleResponse(BaseModel):
    """Response model with complete grid detection info."""
    success: bool
    error: Optional[str] = None

    # Cropped image
    cropped_image: Optional[str] = None

    # Bounds in original image coordinates (includes padding)
    bounds: Optional[dict] = None

    # Actual grid bounds (without padding) - for overlay alignment
    grid_bounds: Optional[dict] = None

    # Grid detection confidence and warnings
    grid_confidence: Optional[float] = None
    confidence_level: str = "unknown"  # "high", "medium", "low", "unknown"
    warnings: List[str] = []

    # Detected grid dimensions (if found via line detection)
    detected_rows: Optional[int] = None
    detected_cols: Optional[int] = None

    # Detection method used
    detection_method: str = "unknown"

    # Timing
    extraction_ms: int = 0


def _get_confidence_level(confidence: Optional[float]) -> str:
    """Convert numeric confidence to categorical level."""
    if confidence is None:
        return "unknown"
    elif confidence >= 0.7:
        return "high"
    elif confidence >= 0.4:
        return "medium"
    else:
        return "low"


def _generate_warnings(
    confidence: Optional[float],
    detected_rows: Optional[int],
    detected_cols: Optional[int],
    grid_bounds: Optional[dict],
    bounds: Optional[dict]
) -> List[str]:
    """Generate user-facing warnings based on detection quality."""
    warnings = []

    # Confidence-based warnings
    if confidence is None:
        warnings.append("Grid line detection failed - using fallback saturation-based detection")
    elif confidence < 0.3:
        warnings.append(f"Low grid detection confidence ({confidence:.0%}) - boundaries may be inaccurate")
    elif confidence < 0.5:
        warnings.append(f"Moderate grid detection confidence ({confidence:.0%}) - verify boundaries")

    # Dimension detection warnings
    if detected_rows is not None and detected_cols is not None:
        # Typical Pips puzzles are 4x4 to 8x8
        if detected_rows < 3 or detected_cols < 3:
            warnings.append(f"Unusually small grid detected ({detected_rows}x{detected_cols}) - may be partial detection")
        elif detected_rows > 10 or detected_cols > 10:
            warnings.append(f"Unusually large grid detected ({detected_rows}x{detected_cols}) - may include extra elements")
        # Check for aspect ratio issues (most Pips puzzles are square or near-square)
        ratio = detected_rows / detected_cols if detected_cols > 0 else 0
        if ratio < 0.5 or ratio > 2.0:
            warnings.append(f"Unusual grid aspect ratio ({detected_rows}:{detected_cols}) - verify detection")

    # Bounds validation warnings
    if grid_bounds and bounds:
        grid_area = grid_bounds.get("width", 0) * grid_bounds.get("height", 0)
        orig_area = grid_bounds.get("original_width", 1) * grid_bounds.get("original_height", 1)
        if grid_area > 0 and orig_area > 0:
            coverage = grid_area / orig_area
            if coverage < 0.1:
                warnings.append("Detected grid is very small relative to image - may be incorrect detection")
            elif coverage > 0.8:
                warnings.append("Detected grid covers most of image - may include non-puzzle areas")

    return warnings


@app.post("/crop-puzzle", response_model=CropPuzzleResponse)
async def crop_puzzle(request: CropPuzzleRequest):
    """
    Crop image to puzzle region only (excludes dominoes, UI).

    Uses multiple detection techniques for best accuracy:
    1. Adaptive thresholding for grid line detection (primary)
    2. Canny edge detection (fallback for low contrast)
    3. Saturation-based detection (baseline)

    Returns cropped image for AI analysis along with detection confidence
    and grid dimension estimates to help guide AI extraction.
    """
    import time
    from hybrid_extraction import crop_puzzle_region, CropResult

    start = time.time()

    # Validate padding_percent
    padding_pct = max(0.0, min(0.15, request.padding_percent))

    try:
        # Call the hybrid extraction with the exclude_bottom_percent option
        result = crop_puzzle_region(
            request.image,
            exclude_bottom_percent=request.exclude_bottom_percent
        )

        # Calculate elapsed time
        elapsed_ms = int((time.time() - start) * 1000)

        if not result.success:
            return CropPuzzleResponse(
                success=False,
                error=result.error or "Unknown error during crop",
                extraction_ms=elapsed_ms
            )

        # Determine detection method based on confidence
        if result.grid_confidence is not None and result.grid_confidence > 0.3:
            detection_method = "adaptive_threshold" if result.grid_confidence >= 0.5 else "canny_edge"
        else:
            detection_method = "saturation_fallback"

        # Get confidence level
        confidence_level = _get_confidence_level(result.grid_confidence)

        # Generate warnings
        warnings = _generate_warnings(
            result.grid_confidence,
            result.detected_rows,
            result.detected_cols,
            result.grid_bounds,
            result.bounds
        )

        # Add warning if confidence below threshold
        if result.grid_confidence is not None and result.grid_confidence < request.min_confidence_threshold:
            warnings.insert(0, f"Detection confidence ({result.grid_confidence:.0%}) below threshold ({request.min_confidence_threshold:.0%})")

        return CropPuzzleResponse(
            success=True,
            cropped_image=result.cropped_image,
            bounds=result.bounds,
            grid_bounds=result.grid_bounds,
            grid_confidence=result.grid_confidence,
            confidence_level=confidence_level,
            warnings=warnings,
            detected_rows=result.detected_rows,
            detected_cols=result.detected_cols,
            detection_method=detection_method,
            extraction_ms=elapsed_ms
        )

    except ValueError as e:
        # Handle specific detection errors
        return CropPuzzleResponse(
            success=False,
            error=str(e),
            warnings=["Grid detection failed - ensure image contains a visible puzzle grid"],
            extraction_ms=int((time.time() - start) * 1000)
        )
    except Exception as e:
        # Handle unexpected errors
        return CropPuzzleResponse(
            success=False,
            error=f"Unexpected error: {str(e)}",
            extraction_ms=int((time.time() - start) * 1000)
        )


class CropDominoRequest(BaseModel):
    image: str  # base64 encoded
    puzzle_bottom_y: Optional[int] = None  # Y coordinate where puzzle ends


@app.post("/crop-dominoes")
async def crop_dominoes(request: CropDominoRequest):
    """
    Crop image to domino tray region only (below the puzzle grid).
    Returns cropped image for AI domino extraction.
    """
    from hybrid_extraction import crop_domino_region

    result = crop_domino_region(request.image, request.puzzle_bottom_y)

    return {
        "success": result.success,
        "error": result.error,
        "cropped_image": result.cropped_image,
        "bounds": result.bounds,
        "extraction_ms": result.extraction_ms
    }


class PreprocessRequest(BaseModel):
    """Request model for image preprocessing before AI extraction."""
    image: str  # base64 encoded

    # Contrast/brightness normalization options
    normalize_contrast: bool = True  # Apply CLAHE for adaptive contrast
    normalize_brightness: bool = True  # Normalize brightness levels
    auto_white_balance: bool = True  # Apply automatic white balance
    sharpen: bool = False  # Apply mild sharpening (useful for blurry images)

    # Advanced options
    clahe_clip_limit: float = 2.0  # CLAHE contrast limit (1.0-4.0)
    clahe_grid_size: int = 8  # CLAHE tile grid size
    target_brightness: int = 128  # Target mean brightness (0-255)
    brightness_tolerance: int = 30  # Tolerance for brightness adjustment


class PreprocessResponse(BaseModel):
    """Response model for image preprocessing."""
    success: bool
    error: Optional[str] = None

    # Preprocessed image
    preprocessed_image: Optional[str] = None

    # Original and new image stats for comparison
    original_stats: Optional[dict] = None
    processed_stats: Optional[dict] = None

    # Operations applied
    operations_applied: List[str] = []

    # Timing
    extraction_ms: int = 0


def _calculate_image_stats(img: np.ndarray) -> dict:
    """
    Calculate image statistics for quality assessment.

    Args:
        img: Input image (BGR)

    Returns:
        Dictionary with brightness, contrast, and color balance stats
    """
    # Convert to different color spaces for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Brightness: mean of grayscale
    mean_brightness = float(np.mean(gray))

    # Contrast: standard deviation of grayscale
    contrast = float(np.std(gray))

    # Dynamic range: min-max of grayscale
    min_val = int(np.min(gray))
    max_val = int(np.max(gray))
    dynamic_range = max_val - min_val

    # Color balance: mean of each channel
    b, g, r = cv2.split(img)
    color_balance = {
        "red": float(np.mean(r)),
        "green": float(np.mean(g)),
        "blue": float(np.mean(b))
    }

    # Saturation: mean of saturation channel
    mean_saturation = float(np.mean(hsv[:, :, 1]))

    return {
        "brightness": round(mean_brightness, 2),
        "contrast": round(contrast, 2),
        "dynamic_range": dynamic_range,
        "min_value": min_val,
        "max_value": max_val,
        "color_balance": color_balance,
        "saturation": round(mean_saturation, 2)
    }


def _apply_clahe(img: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    CLAHE improves local contrast while limiting noise amplification.
    It's particularly effective for images with varying lighting.

    Args:
        img: Input image (BGR)
        clip_limit: Threshold for contrast limiting (higher = more contrast)
        grid_size: Size of grid for histogram equalization

    Returns:
        Image with enhanced contrast
    """
    # Convert to LAB color space (L channel for luminance)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to luminance channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_enhanced = clahe.apply(l)

    # Merge back and convert to BGR
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return result


def _normalize_brightness(img: np.ndarray, target: int = 128, tolerance: int = 30) -> Tuple[np.ndarray, bool]:
    """
    Normalize image brightness to target level.

    Args:
        img: Input image (BGR)
        target: Target mean brightness (0-255)
        tolerance: If current brightness is within tolerance, skip normalization

    Returns:
        (Normalized image, was_adjusted)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    current_brightness = np.mean(gray)

    # Skip if already within tolerance
    if abs(current_brightness - target) <= tolerance:
        return img.copy(), False

    # Calculate adjustment factor
    if current_brightness > 0:
        factor = target / current_brightness
    else:
        factor = 1.0

    # Clamp factor to reasonable range (0.5 to 2.0)
    factor = max(0.5, min(2.0, factor))

    # Apply brightness adjustment
    result = cv2.convertScaleAbs(img, alpha=factor, beta=0)

    return result, True


def _auto_white_balance(img: np.ndarray) -> np.ndarray:
    """
    Apply automatic white balance using the Gray World assumption.

    The Gray World algorithm assumes the average color of a scene is gray,
    and adjusts channel gains to achieve this.

    Args:
        img: Input image (BGR)

    Returns:
        White-balanced image
    """
    # Calculate average of each channel
    b, g, r = cv2.split(img)
    r_avg = np.mean(r)
    g_avg = np.mean(g)
    b_avg = np.mean(b)

    # Calculate overall average (gray point)
    overall_avg = (r_avg + g_avg + b_avg) / 3

    # Calculate scaling factors
    if r_avg > 0:
        r_scale = overall_avg / r_avg
    else:
        r_scale = 1.0
    if g_avg > 0:
        g_scale = overall_avg / g_avg
    else:
        g_scale = 1.0
    if b_avg > 0:
        b_scale = overall_avg / b_avg
    else:
        b_scale = 1.0

    # Clamp scales to reasonable range
    r_scale = max(0.5, min(2.0, r_scale))
    g_scale = max(0.5, min(2.0, g_scale))
    b_scale = max(0.5, min(2.0, b_scale))

    # Apply scaling and clip to valid range
    r_balanced = np.clip(r * r_scale, 0, 255).astype(np.uint8)
    g_balanced = np.clip(g * g_scale, 0, 255).astype(np.uint8)
    b_balanced = np.clip(b * b_scale, 0, 255).astype(np.uint8)

    return cv2.merge([b_balanced, g_balanced, r_balanced])


def _apply_sharpening(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Apply mild unsharp masking to sharpen the image.

    Args:
        img: Input image (BGR)
        strength: Sharpening strength (0.0 to 1.0)

    Returns:
        Sharpened image
    """
    # Create blurred version
    blurred = cv2.GaussianBlur(img, (0, 0), 3)

    # Unsharp mask: original + strength * (original - blurred)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)

    return sharpened


def _preprocess_image(
    img: np.ndarray,
    normalize_contrast: bool = True,
    normalize_brightness: bool = True,
    auto_white_balance: bool = True,
    sharpen: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: int = 8,
    target_brightness: int = 128,
    brightness_tolerance: int = 30
) -> Tuple[np.ndarray, List[str]]:
    """
    Apply preprocessing pipeline to image.

    Args:
        img: Input image (BGR)
        normalize_contrast: Apply CLAHE
        normalize_brightness: Normalize brightness levels
        auto_white_balance: Apply white balance correction
        sharpen: Apply sharpening
        clahe_clip_limit: CLAHE clip limit
        clahe_grid_size: CLAHE grid size
        target_brightness: Target brightness level
        brightness_tolerance: Tolerance for brightness adjustment

    Returns:
        (Preprocessed image, list of operations applied)
    """
    result = img.copy()
    operations = []

    # Order matters: white balance -> brightness -> contrast -> sharpen

    # Step 1: White balance (normalize color cast)
    if auto_white_balance:
        result = _auto_white_balance(result)
        operations.append("white_balance")

    # Step 2: Brightness normalization
    if normalize_brightness:
        result, was_adjusted = _normalize_brightness(
            result, target_brightness, brightness_tolerance
        )
        if was_adjusted:
            operations.append("brightness_normalization")

    # Step 3: Contrast enhancement (CLAHE)
    if normalize_contrast:
        result = _apply_clahe(result, clahe_clip_limit, clahe_grid_size)
        operations.append("clahe_contrast")

    # Step 4: Sharpening (optional, last to avoid amplifying noise)
    if sharpen:
        result = _apply_sharpening(result)
        operations.append("sharpening")

    return result, operations


@app.post("/preprocess-image", response_model=PreprocessResponse)
async def preprocess_image(request: PreprocessRequest):
    """
    Preprocess image with contrast/brightness normalization before AI extraction.

    This endpoint applies various image enhancement techniques to improve
    AI extraction accuracy:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Brightness normalization
    - Automatic white balance
    - Optional sharpening

    Returns the preprocessed image along with before/after statistics.
    """
    import time
    start = time.time()

    try:
        # Decode image
        img = decode_image(request.image)

        # Calculate original stats
        original_stats = _calculate_image_stats(img)

        # Validate CLAHE parameters
        clip_limit = max(1.0, min(4.0, request.clahe_clip_limit))
        grid_size = max(2, min(16, request.clahe_grid_size))

        # Apply preprocessing pipeline
        processed, operations = _preprocess_image(
            img,
            normalize_contrast=request.normalize_contrast,
            normalize_brightness=request.normalize_brightness,
            auto_white_balance=request.auto_white_balance,
            sharpen=request.sharpen,
            clahe_clip_limit=clip_limit,
            clahe_grid_size=grid_size,
            target_brightness=request.target_brightness,
            brightness_tolerance=request.brightness_tolerance
        )

        # Calculate processed stats
        processed_stats = _calculate_image_stats(processed)

        # Encode result as PNG base64
        _, buffer = cv2.imencode('.png', processed)
        processed_b64 = base64.b64encode(buffer).decode('utf-8')

        elapsed_ms = int((time.time() - start) * 1000)

        return PreprocessResponse(
            success=True,
            preprocessed_image=processed_b64,
            original_stats=original_stats,
            processed_stats=processed_stats,
            operations_applied=operations,
            extraction_ms=elapsed_ms
        )

    except ValueError as e:
        return PreprocessResponse(
            success=False,
            error=str(e),
            extraction_ms=int((time.time() - start) * 1000)
        )
    except Exception as e:
        return PreprocessResponse(
            success=False,
            error=f"Preprocessing failed: {str(e)}",
            extraction_ms=int((time.time() - start) * 1000)
        )


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "pips-cv"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
