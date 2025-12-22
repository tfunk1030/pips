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
                "right_confidence": 0.87
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
                        "right_confidence": 0.87
                    }
                ],
                "total_count": 1
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

    Args:
        request: CropDominoesRequest containing:
            - image: Base64-encoded image data
            - dominoes: List of bounding boxes to process

    Returns:
        DominoExtractionResponse containing:
            - dominoes: List of DominoResult with pip detection results
            - total_count: Number of dominoes processed

    Raises:
        HTTPException: 400 if image cannot be decoded or bounding boxes are invalid
        HTTPException: 500 if processing fails unexpectedly
    """
    # Decode the base64 image
    try:
        image = decode_base64_image(request.image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Process each domino bounding box
    results: List[DominoResult] = []

    for i, domino_box in enumerate(request.dominoes):
        try:
            # Crop the domino region from the image
            cropped = crop_domino_region(
                image,
                domino_box.x,
                domino_box.y,
                domino_box.width,
                domino_box.height
            )

            # Run pip detection on the cropped domino
            pip_result = detect_domino_pips(cropped)

            # Create result with bounding box and pip values
            result = DominoResult(
                x=domino_box.x,
                y=domino_box.y,
                width=domino_box.width,
                height=domino_box.height,
                left_pips=pip_result.left_pips,
                right_pips=pip_result.right_pips,
                left_confidence=pip_result.left_confidence,
                right_confidence=pip_result.right_confidence
            )
            results.append(result)

        except ValueError as e:
            # Handle crop errors gracefully - return result with null pip values
            logger.warning(f"Failed to process domino {i}: {str(e)}")
            result = DominoResult(
                x=domino_box.x,
                y=domino_box.y,
                width=domino_box.width,
                height=domino_box.height,
                left_pips=None,
                right_pips=None,
                left_confidence=None,
                right_confidence=None
            )
            results.append(result)

        except Exception as e:
            # Log unexpected errors but continue processing other dominoes
            logger.error(f"Unexpected error processing domino {i}: {str(e)}")
            result = DominoResult(
                x=domino_box.x,
                y=domino_box.y,
                width=domino_box.width,
                height=domino_box.height,
                left_pips=None,
                right_pips=None,
                left_confidence=None,
                right_confidence=None
            )
            results.append(result)

    return DominoExtractionResponse(
        dominoes=results,
        total_count=len(results)
    )


# Placeholder endpoints - to be implemented in subsequent subtasks
# POST /extract-geometry
# POST /crop-puzzle
# POST /preprocess-image
