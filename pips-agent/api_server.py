"""
FastAPI server for pips-agent hint generation API.

This module provides HTTP endpoints for the graduated hint system,
allowing the frontend to request contextually-aware hints at 4 levels.
"""

from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Pydantic Models for Puzzle Specification
# =============================================================================

class PipsConfig(BaseModel):
    """Configuration for pip values in the puzzle."""
    pip_min: int = Field(ge=0, description="Minimum pip value")
    pip_max: int = Field(ge=0, description="Maximum pip value")


class DominoesConfig(BaseModel):
    """Configuration for domino tiles in the puzzle."""
    tiles: List[List[int]] = Field(description="List of domino tiles as [pip1, pip2] pairs")
    unique: Optional[bool] = Field(default=True, description="Whether tiles must be unique")


class BoardConfig(BaseModel):
    """Configuration for the puzzle board layout."""
    shape: List[str] = Field(description="Board shape where '.' is a cell and '#' is empty")
    regions: List[str] = Field(description="Region labels for each cell position")


class RegionConstraint(BaseModel):
    """A constraint on a puzzle region."""
    type: str = Field(description="Constraint type: 'sum' or 'all_equal'")
    op: Optional[str] = Field(default=None, description="Comparison operator for sum: '==', '!=', '<', '>'")
    value: Optional[int] = Field(default=None, description="Target value for sum constraints")


class PuzzleSpec(BaseModel):
    """Complete puzzle specification."""
    pips: PipsConfig
    dominoes: DominoesConfig
    board: BoardConfig
    region_constraints: Dict[str, RegionConstraint]


# =============================================================================
# Pydantic Models for Hint Request/Response
# =============================================================================

class HintRequest(BaseModel):
    """Request model for hint generation."""
    puzzle_spec: PuzzleSpec = Field(description="The current puzzle specification")
    level: int = Field(ge=1, le=4, description="Hint level (1-4)")
    current_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional current puzzle state with user placements"
    )

    @field_validator('level')
    @classmethod
    def validate_level(cls, v: int) -> int:
        if v < 1 or v > 4:
            raise ValueError('Hint level must be between 1 and 4')
        return v


class HintContent(BaseModel):
    """Content of a generated hint."""
    level: int = Field(description="The hint level (1-4)")
    content: str = Field(description="The hint text")
    type: str = Field(description="Hint type: 'strategy', 'direction', 'cell', 'partial_solution'")
    region: Optional[str] = Field(default=None, description="Target region for Level 2+ hints")
    cell: Optional[Dict[str, int]] = Field(
        default=None,
        description="Target cell coordinates for Level 3+ hints"
    )
    cells: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Multiple cell placements for Level 4 hints"
    )


class HintResponse(BaseModel):
    """Response model for hint generation."""
    success: bool = Field(description="Whether hint generation succeeded")
    hint: Optional[HintContent] = Field(default=None, description="The generated hint")
    error: Optional[str] = Field(default=None, description="Error message if generation failed")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Pips Agent API",
    description="API for graduated hint generation in the pips puzzle system",
    version="1.0.0",
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for service monitoring."""
    return {"status": "healthy", "service": "pips-agent"}


# =============================================================================
# Hint Generation Endpoint
# =============================================================================

# Hint type mapping by level
HINT_TYPES = {
    1: "strategy",
    2: "direction",
    3: "cell",
    4: "partial_solution",
}


@app.post("/generate-hint", response_model=HintResponse)
async def generate_hint(request: HintRequest) -> HintResponse:
    """
    Generate a hint for the puzzle at the specified level.

    Hint Levels:
    - Level 1: Strategic guidance (general approach, no specifics)
    - Level 2: Focused direction (identify specific region/constraint)
    - Level 3: Specific cell placement (one cell with correct value)
    - Level 4: Partial solution (multiple cell placements)
    """
    try:
        level = request.level
        puzzle_spec = request.puzzle_spec

        # Placeholder hint content - will be replaced by actual hint engine in phase 2
        hint_content = _generate_placeholder_hint(level, puzzle_spec)

        return HintResponse(
            success=True,
            hint=hint_content,
        )

    except ValueError as e:
        return HintResponse(
            success=False,
            error=str(e),
        )
    except Exception as e:
        return HintResponse(
            success=False,
            error=f"Hint generation failed: {str(e)}",
        )


def _generate_placeholder_hint(level: int, puzzle_spec: PuzzleSpec) -> HintContent:
    """
    Generate a placeholder hint based on level.

    This function will be replaced by actual hint generation logic
    from utils/hint_engine.py in phase 2.
    """
    hint_type = HINT_TYPES.get(level, "strategy")

    # Get region names from puzzle spec for context
    region_names = list(puzzle_spec.region_constraints.keys())
    first_region = region_names[0] if region_names else "A"

    if level == 1:
        return HintContent(
            level=1,
            content="Consider the constraints on each region and look for regions with limited possibilities.",
            type="strategy",
        )
    elif level == 2:
        return HintContent(
            level=2,
            content=f"Focus on region '{first_region}' - its constraints significantly limit possible placements.",
            type="direction",
            region=first_region,
        )
    elif level == 3:
        return HintContent(
            level=3,
            content="Place a domino at position (0, 0) to satisfy the nearby constraints.",
            type="cell",
            region=first_region,
            cell={"row": 0, "col": 0},
        )
    else:  # level == 4
        return HintContent(
            level=4,
            content="Here are several placements that form part of the solution:",
            type="partial_solution",
            cells=[
                {"row": 0, "col": 0, "value": 1},
                {"row": 0, "col": 1, "value": 2},
                {"row": 1, "col": 0, "value": 3},
            ],
        )
