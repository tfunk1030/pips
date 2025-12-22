"""
FastAPI service for computer vision operations on domino puzzle images.

This service provides endpoints for extracting geometry, cropping puzzles,
cropping individual dominoes, and detecting pip values.
"""

import base64
import logging

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from extract_dominoes import detect_domino_pips


# Configure logging
logger = logging.getLogger(__name__)


# FastAPI application instance
app = FastAPI(
    title="CV Service",
    description="Computer vision service for domino puzzle extraction and pip detection",
    version="1.0.0"
)

# CORS configuration for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Pydantic Response Models
# =============================================================================

class DominoResult(BaseModel):
    """
    Result for a single domino tile including position, size, and pip detection.

    Contains bounding box coordinates for the domino within the image,
    plus detected pip values and confidence scores for each half.
    """

    # Position and size fields
    x: int = Field(..., ge=0, description="X coordinate of domino bounding box")
    y: int = Field(..., ge=0, description="Y coordinate of domino bounding box")
    width: int = Field(..., gt=0, description="Width of domino bounding box")
    height: int = Field(..., gt=0, description="Height of domino bounding box")

    # Pip detection fields
    left_pips: Optional[int] = Field(
        None,
        ge=0,
        le=6,
        description="Detected pip count on left half (0-6), null if detection failed"
    )
    right_pips: Optional[int] = Field(
        None,
        ge=0,
        le=6,
        description="Detected pip count on right half (0-6), null if detection failed"
    )
    left_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score for left half detection (0.0-1.0)"
    )
    right_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score for right half detection (0.0-1.0)"
    )

    # Error tracking fields
    error: Optional[str] = Field(
        None,
        description="Error message if detection failed, null if successful"
    )
    warning: Optional[str] = Field(
        None,
        description="Warning message for partial or low-quality detections"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "x": 10,
                "y": 20,
                "width": 100,
                "height": 50,
                "left_pips": 3,
                "right_pips": 5,
                "left_confidence": 0.92,
                "right_confidence": 0.87,
                "error": None,
                "warning": None
            }
        }
    }


class DominoExtractionResponse(BaseModel):
    """
    Response for domino extraction endpoint containing all detected dominoes.
    """

    dominoes: List[DominoResult] = Field(
        ...,
        description="List of extracted domino results with pip detection"
    )
    total_count: int = Field(
        ...,
        ge=0,
        description="Total number of dominoes detected"
    )
    successful_count: int = Field(
        ...,
        ge=0,
        description="Number of dominoes with successful pip detection"
    )
    failed_count: int = Field(
        ...,
        ge=0,
        description="Number of dominoes with failed pip detection"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "dominoes": [
                    {
                        "x": 10,
                        "y": 20,
                        "width": 100,
                        "height": 50,
                        "left_pips": 3,
                        "right_pips": 5,
                        "left_confidence": 0.92,
                        "right_confidence": 0.87,
                        "error": None,
                        "warning": None
                    }
                ],
                "total_count": 1,
                "successful_count": 1,
                "failed_count": 0
            }
        }
    }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="Service version")


# =============================================================================
# Request Models
# =============================================================================

class DominoBoundingBox(BaseModel):
    """
    Bounding box coordinates for a single domino within the image.

    These coordinates define where to crop the domino from the source image
    for pip detection.
    """

    x: int = Field(..., ge=0, description="X coordinate of domino top-left corner")
    y: int = Field(..., ge=0, description="Y coordinate of domino top-left corner")
    width: int = Field(..., gt=0, description="Width of domino bounding box")
    height: int = Field(..., gt=0, description="Height of domino bounding box")

    model_config = {
        "json_schema_extra": {
            "example": {
                "x": 10,
                "y": 20,
                "width": 100,
                "height": 50
            }
        }
    }


class CropDominoesRequest(BaseModel):
    """
    Request model for the /crop-dominoes endpoint.

    Accepts a base64-encoded image and a list of domino bounding boxes.
    Each domino will be cropped and processed for pip detection.
    """

    image: str = Field(
        ...,
        description="Base64-encoded image data (PNG, JPG, or other supported format)"
    )
    dominoes: List[DominoBoundingBox] = Field(
        ...,
        description="List of domino bounding boxes to crop and analyze"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "image": "base64_encoded_image_data...",
                "dominoes": [
                    {"x": 10, "y": 20, "width": 100, "height": 50},
                    {"x": 120, "y": 20, "width": 100, "height": 50}
                ]
            }
        }
    }


# =============================================================================
# Helper Functions
# =============================================================================

def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode a base64-encoded image string to a numpy array.

    Args:
        base64_string: Base64-encoded image data. May include a data URI prefix
            like "data:image/png;base64," which will be stripped.

    Returns:
        Decoded image as a BGR numpy array (OpenCV format).

    Raises:
        ValueError: If the image cannot be decoded.
    """
    # Remove data URI prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)

        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Decode image using OpenCV
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image data")

        return image

    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")


def crop_domino_region(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int
) -> np.ndarray:
    """
    Crop a domino region from an image.

    Args:
        image: Source image as BGR numpy array.
        x: X coordinate of top-left corner.
        y: Y coordinate of top-left corner.
        width: Width of crop region.
        height: Height of crop region.

    Returns:
        Cropped domino image as BGR numpy array.

    Raises:
        ValueError: If crop region is out of bounds or invalid.
    """
    img_height, img_width = image.shape[:2]

    # Validate bounds
    if x < 0 or y < 0:
        raise ValueError(f"Negative coordinates: x={x}, y={y}")

    if x + width > img_width or y + height > img_height:
        raise ValueError(
            f"Crop region ({x}, {y}, {width}, {height}) exceeds "
            f"image bounds ({img_width}, {img_height})"
        )

    # Crop the region
    cropped = image[y:y + height, x:x + width]

    if cropped.size == 0:
        raise ValueError("Cropped region is empty")

    return cropped


# =============================================================================
# Pip Detection Quality Constants
# =============================================================================

# Minimum dimensions for reliable pip detection
# Dominoes smaller than this may have partial visibility or poor quality
MIN_DOMINO_WIDTH = 20   # Minimum width in pixels
MIN_DOMINO_HEIGHT = 10  # Minimum height in pixels

# Minimum area for pip detection (width * height)
MIN_DOMINO_AREA = 400   # 20x20 pixels minimum

# Confidence thresholds for quality warnings
LOW_CONFIDENCE_THRESHOLD = 0.5    # Below this, detection is unreliable
MEDIUM_CONFIDENCE_THRESHOLD = 0.7  # Below this, detection may be uncertain

# Aspect ratio bounds for valid dominoes (width/height)
# Dominoes are typically 2:1 ratio, but can vary
MIN_ASPECT_RATIO = 1.0   # Minimum width/height ratio
MAX_ASPECT_RATIO = 4.0   # Maximum width/height ratio


def validate_domino_dimensions(
    width: int,
    height: int
) -> tuple[bool, Optional[str]]:
    """
    Validate domino dimensions for pip detection quality.

    Checks if the domino bounding box meets minimum size requirements
    for reliable pip detection.

    Args:
        width: Width of domino bounding box in pixels.
        height: Height of domino bounding box in pixels.

    Returns:
        Tuple of (is_valid, warning_message):
        - is_valid: True if dimensions meet minimum requirements
        - warning_message: Warning string if dimensions are borderline, None otherwise
    """
    warning = None

    # Check minimum dimensions
    if width < MIN_DOMINO_WIDTH or height < MIN_DOMINO_HEIGHT:
        return False, f"Domino too small ({width}x{height}px), minimum is {MIN_DOMINO_WIDTH}x{MIN_DOMINO_HEIGHT}px"

    # Check minimum area
    area = width * height
    if area < MIN_DOMINO_AREA:
        return False, f"Domino area too small ({area}px²), minimum is {MIN_DOMINO_AREA}px²"

    # Check aspect ratio
    aspect_ratio = width / height if height > 0 else 0
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        warning = f"Unusual aspect ratio ({aspect_ratio:.2f}), may affect detection accuracy"

    # Check for very small dimensions (borderline quality)
    if width < MIN_DOMINO_WIDTH * 2 or height < MIN_DOMINO_HEIGHT * 2:
        warning = "Small domino size may reduce detection accuracy"

    return True, warning


def check_image_quality(
    image: np.ndarray
) -> tuple[bool, Optional[str]]:
    """
    Check image quality for pip detection.

    Analyzes the cropped domino image to detect quality issues
    that may affect pip detection accuracy.

    Args:
        image: Cropped domino image as BGR numpy array.

    Returns:
        Tuple of (is_acceptable, warning_message):
        - is_acceptable: True if image quality is sufficient for detection
        - warning_message: Warning string if quality issues detected, None otherwise
    """
    if image is None or image.size == 0:
        return False, "Image is empty or invalid"

    # Get image dimensions
    h, w = image.shape[:2]

    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Check contrast - low contrast makes pip detection difficult
    img_std = np.std(gray)
    if img_std < 15:
        return False, "Image has very low contrast, pip detection unreliable"
    if img_std < 25:
        return True, "Low contrast image may affect detection accuracy"

    # Check for mostly black or mostly white images (possible occlusion or overexposure)
    mean_val = np.mean(gray)
    if mean_val < 20:
        return False, "Image is too dark, possible occlusion"
    if mean_val > 235:
        return False, "Image is overexposed or blank"

    # Check for uniform images (no visible content)
    min_val, max_val = np.min(gray), np.max(gray)
    if max_val - min_val < 20:
        return True, "Image has limited dynamic range, may affect accuracy"

    return True, None


def assess_detection_quality(
    left_confidence: Optional[float],
    right_confidence: Optional[float]
) -> Optional[str]:
    """
    Assess overall detection quality and generate appropriate warning.

    Args:
        left_confidence: Confidence score for left half detection.
        right_confidence: Confidence score for right half detection.

    Returns:
        Warning message if quality is concerning, None if detection looks good.
    """
    if left_confidence is None or right_confidence is None:
        return None  # No detection to assess

    avg_confidence = (left_confidence + right_confidence) / 2
    min_confidence = min(left_confidence, right_confidence)

    if min_confidence < LOW_CONFIDENCE_THRESHOLD:
        return f"Low detection confidence ({min_confidence:.2f}), results may be unreliable"
    elif avg_confidence < MEDIUM_CONFIDENCE_THRESHOLD:
        return f"Detection confidence is moderate ({avg_confidence:.2f}), verify results"

    return None


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the service status and version information.
    """
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/crop-dominoes", response_model=DominoExtractionResponse)
async def crop_dominoes(request: CropDominoesRequest):
    """
    Crop dominoes from an image and detect pip values.

    This endpoint accepts a base64-encoded image and a list of domino bounding
    boxes. For each bounding box, it crops the domino region from the image
    and runs pip detection to identify the pip values on each half of the domino.

    The response includes the original bounding box coordinates plus the detected
    pip values (0-6 for each half) and confidence scores (0.0-1.0).

    Error Handling:
        - Partial dominoes: Returns error with null pip values
        - Poor quality images: Returns warning with low confidence scores
        - Invalid dimensions: Returns error with descriptive message
        - Detection failures: Returns warning with best-effort results

    Args:
        request: CropDominoesRequest containing:
            - image: Base64-encoded image data
            - dominoes: List of bounding boxes to process

    Returns:
        DominoExtractionResponse containing:
            - dominoes: List of DominoResult with pip detection results
            - total_count: Number of dominoes processed
            - successful_count: Number of successful detections
            - failed_count: Number of failed detections

    Raises:
        HTTPException: 400 if image cannot be decoded or is invalid
        HTTPException: 500 if processing fails unexpectedly
    """
    # Decode the base64 image
    try:
        image = decode_base64_image(request.image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Process each domino bounding box
    results: List[DominoResult] = []
    successful_count = 0
    failed_count = 0

    for i, domino_box in enumerate(request.dominoes):
        error_msg: Optional[str] = None
        warning_msg: Optional[str] = None

        try:
            # Step 1: Validate domino dimensions before processing
            is_valid, dimension_msg = validate_domino_dimensions(
                domino_box.width,
                domino_box.height
            )

            if not is_valid:
                # Dimensions too small for reliable detection
                logger.warning(
                    f"Domino {i} has invalid dimensions: {dimension_msg}"
                )
                result = DominoResult(
                    x=domino_box.x,
                    y=domino_box.y,
                    width=domino_box.width,
                    height=domino_box.height,
                    left_pips=None,
                    right_pips=None,
                    left_confidence=None,
                    right_confidence=None,
                    error=dimension_msg,
                    warning=None
                )
                results.append(result)
                failed_count += 1
                continue

            # Capture dimension warning if any
            if dimension_msg:
                warning_msg = dimension_msg

            # Step 2: Crop the domino region from the image
            cropped = crop_domino_region(
                image,
                domino_box.x,
                domino_box.y,
                domino_box.width,
                domino_box.height
            )

            # Step 3: Check image quality before pip detection
            quality_ok, quality_msg = check_image_quality(cropped)

            if not quality_ok:
                # Image quality too poor for detection
                logger.warning(
                    f"Domino {i} has poor image quality: {quality_msg}"
                )
                result = DominoResult(
                    x=domino_box.x,
                    y=domino_box.y,
                    width=domino_box.width,
                    height=domino_box.height,
                    left_pips=None,
                    right_pips=None,
                    left_confidence=None,
                    right_confidence=None,
                    error=quality_msg,
                    warning=warning_msg
                )
                results.append(result)
                failed_count += 1
                continue

            # Combine quality warning with dimension warning
            if quality_msg:
                warning_msg = quality_msg if not warning_msg else f"{warning_msg}; {quality_msg}"

            # Step 4: Run pip detection on the cropped domino
            pip_result = detect_domino_pips(cropped)

            # Step 5: Assess detection quality and add warnings
            detection_warning = assess_detection_quality(
                pip_result.left_confidence,
                pip_result.right_confidence
            )

            if detection_warning:
                warning_msg = detection_warning if not warning_msg else f"{warning_msg}; {detection_warning}"

            # Create result with bounding box and pip values
            result = DominoResult(
                x=domino_box.x,
                y=domino_box.y,
                width=domino_box.width,
                height=domino_box.height,
                left_pips=pip_result.left_pips,
                right_pips=pip_result.right_pips,
                left_confidence=pip_result.left_confidence,
                right_confidence=pip_result.right_confidence,
                error=None,
                warning=warning_msg
            )
            results.append(result)
            successful_count += 1

        except ValueError as e:
            # Handle crop errors and pip detection ValueError gracefully
            error_str = str(e)
            logger.warning(f"Failed to process domino {i}: {error_str}")

            # Categorize the error for better feedback
            if "bounds" in error_str.lower() or "coordinates" in error_str.lower():
                error_msg = f"Invalid crop region: {error_str}"
            elif "empty" in error_str.lower():
                error_msg = "Cropped region is empty, domino may be partially outside image"
            else:
                error_msg = f"Detection failed: {error_str}"

            result = DominoResult(
                x=domino_box.x,
                y=domino_box.y,
                width=domino_box.width,
                height=domino_box.height,
                left_pips=None,
                right_pips=None,
                left_confidence=None,
                right_confidence=None,
                error=error_msg,
                warning=warning_msg
            )
            results.append(result)
            failed_count += 1

        except cv2.error as e:
            # Handle OpenCV-specific errors
            error_str = str(e)
            logger.error(f"OpenCV error processing domino {i}: {error_str}")
            result = DominoResult(
                x=domino_box.x,
                y=domino_box.y,
                width=domino_box.width,
                height=domino_box.height,
                left_pips=None,
                right_pips=None,
                left_confidence=None,
                right_confidence=None,
                error=f"Image processing error: {error_str}",
                warning=warning_msg
            )
            results.append(result)
            failed_count += 1

        except Exception as e:
            # Log unexpected errors but continue processing other dominoes
            error_str = str(e)
            logger.error(f"Unexpected error processing domino {i}: {error_str}")
            result = DominoResult(
                x=domino_box.x,
                y=domino_box.y,
                width=domino_box.width,
                height=domino_box.height,
                left_pips=None,
                right_pips=None,
                left_confidence=None,
                right_confidence=None,
                error=f"Unexpected error: {error_str}",
                warning=warning_msg
            )
            results.append(result)
            failed_count += 1

    return DominoExtractionResponse(
        dominoes=results,
        total_count=len(results),
        successful_count=successful_count,
        failed_count=failed_count
    )


# Placeholder endpoints - to be implemented in subsequent subtasks
# POST /extract-geometry
# POST /crop-puzzle
# POST /preprocess-image
