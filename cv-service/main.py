"""
FastAPI service for computer vision operations on domino puzzle images.

This service provides endpoints for extracting geometry, cropping puzzles,
cropping individual dominoes, and detecting pip values.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional


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
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the service status and version information.
    """
    return HealthResponse(status="healthy", version="1.0.0")


# Placeholder endpoints - to be implemented in subsequent subtasks
# POST /extract-geometry
# POST /crop-puzzle
# POST /crop-dominoes
# POST /preprocess-image
