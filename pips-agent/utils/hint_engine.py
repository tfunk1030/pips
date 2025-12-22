"""
Hint Engine for the Graduated Hint System.

This module provides hint generation functions for 4 escalating disclosure levels:
- Level 1: Strategic guidance (general approach, no specifics)
- Level 2: Focused direction (identify specific region/constraint)
- Level 3: Specific cell placement (one cell with correct value)
- Level 4: Partial solution (multiple cell placements)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import random


@dataclass
class HintResult:
    """Result of hint generation."""
    content: str
    hint_type: str
    region: Optional[str] = None
    cell: Optional[Dict[str, int]] = None
    cells: Optional[List[Dict[str, Any]]] = None


# =============================================================================
# Strategic Hint Templates for Level 1
# =============================================================================

# General strategic hints that don't reveal specific information
GENERAL_STRATEGY_HINTS = [
    "Start by identifying regions with the most restrictive constraints - they have fewer valid placements.",
    "Look for regions where the constraint value limits your domino choices significantly.",
    "Consider the pip range in your domino set - some constraint values may only be achievable one way.",
    "Focus on cells that have fewer adjacent neighbors - they're easier to reason about.",
    "Remember that each domino must be placed as a pair - think about which adjacent cells connect.",
]

# Hints based on constraint types present
SUM_CONSTRAINT_HINTS = [
    "For sum constraints, calculate the minimum and maximum possible totals from your available dominoes.",
    "When facing sum constraints, start with extreme values (very high or very low sums) as they have fewer solutions.",
    "Sum constraints with equality (==) are more restrictive - focus on those first.",
    "For strict inequality constraints (< or >), think about the boundary values.",
]

ALL_EQUAL_CONSTRAINT_HINTS = [
    "Regions with 'all equal' constraints need every cell to have the same pip value.",
    "For all-equal regions, identify which pip values appear on multiple domino halves.",
    "All-equal constraints dramatically limit domino placement - the same value must span the entire region.",
]

# Hints based on puzzle complexity
SMALL_PUZZLE_HINTS = [
    "For smaller puzzles, try working through the constraints systematically from most to least restrictive.",
    "With fewer cells to fill, you can often enumerate possible placements mentally.",
]

LARGE_PUZZLE_HINTS = [
    "For larger puzzles, identify clusters of interconnected regions and solve them together.",
    "Break the puzzle into sections and look for 'bottleneck' regions that connect different areas.",
]

# Hints based on domino availability
LIMITED_DOMINO_HINTS = [
    "With a limited domino set, track which tiles you've placed to narrow down remaining options.",
    "Some dominoes are 'doubles' (same pip on both ends) - note where they can fit.",
]

DOUBLE_DOMINO_HINTS = [
    "Double dominoes (where both halves have the same value) are special - plan their placement carefully.",
    "A double domino can satisfy an 'all equal' constraint within a 2-cell region entirely.",
]


def _count_constraint_types(region_constraints: Dict[str, Any]) -> Tuple[int, int]:
    """Count sum and all_equal constraints."""
    sum_count = 0
    all_equal_count = 0
    for constraint in region_constraints.values():
        constraint_type = constraint.get("type", "") if isinstance(constraint, dict) else getattr(constraint, "type", "")
        if constraint_type == "sum":
            sum_count += 1
        elif constraint_type == "all_equal":
            all_equal_count += 1
    return sum_count, all_equal_count


def _count_cells(board: Dict[str, Any]) -> int:
    """Count total number of playable cells in the board."""
    shape = board.get("shape", []) if isinstance(board, dict) else getattr(board, "shape", [])
    count = 0
    for row in shape:
        count += row.count(".")
    return count


def _has_double_dominoes(dominoes: Dict[str, Any]) -> bool:
    """Check if the domino set contains any doubles."""
    tiles = dominoes.get("tiles", []) if isinstance(dominoes, dict) else getattr(dominoes, "tiles", [])
    for tile in tiles:
        if len(tile) >= 2 and tile[0] == tile[1]:
            return True
    return False


def _analyze_puzzle_characteristics(puzzle_spec: Any) -> Dict[str, Any]:
    """
    Analyze the puzzle to determine its characteristics.

    Returns a dictionary with analysis results used to select relevant hints.
    """
    # Handle both dict-like and object-like puzzle specs
    if isinstance(puzzle_spec, dict):
        region_constraints = puzzle_spec.get("region_constraints", {})
        board = puzzle_spec.get("board", {})
        dominoes = puzzle_spec.get("dominoes", {})
    else:
        region_constraints = puzzle_spec.region_constraints
        board = puzzle_spec.board
        dominoes = puzzle_spec.dominoes

    sum_count, all_equal_count = _count_constraint_types(
        region_constraints if isinstance(region_constraints, dict) else {k: v for k, v in vars(region_constraints).items()} if hasattr(region_constraints, '__dict__') else {}
    )

    # Get region constraints as dict for iteration
    if not isinstance(region_constraints, dict):
        region_constraints = {k: v for k, v in region_constraints.items()} if hasattr(region_constraints, 'items') else {}

    cell_count = _count_cells(board if isinstance(board, dict) else board.__dict__ if hasattr(board, '__dict__') else {"shape": getattr(board, "shape", [])})
    has_doubles = _has_double_dominoes(dominoes if isinstance(dominoes, dict) else dominoes.__dict__ if hasattr(dominoes, '__dict__') else {"tiles": getattr(dominoes, "tiles", [])})

    return {
        "sum_constraints": sum_count,
        "all_equal_constraints": all_equal_count,
        "total_constraints": sum_count + all_equal_count,
        "cell_count": cell_count,
        "has_double_dominoes": has_doubles,
        "is_small_puzzle": cell_count <= 8,
        "is_large_puzzle": cell_count >= 20,
    }


def generate_hint_level_1(
    puzzle_spec: Any,
    previous_hints: Optional[List[str]] = None,
) -> HintResult:
    """
    Generate a Level 1 strategic hint.

    Level 1 hints provide general strategic guidance without revealing
    specific cells, regions, or values. They help the user think about
    the puzzle solving approach.

    Args:
        puzzle_spec: The puzzle specification (PuzzleSpec model or dict)
        previous_hints: List of previously given hints to avoid repetition

    Returns:
        HintResult with strategic guidance content
    """
    if previous_hints is None:
        previous_hints = []

    # Analyze the puzzle to select relevant hint categories
    characteristics = _analyze_puzzle_characteristics(puzzle_spec)

    # Build a pool of candidate hints based on puzzle characteristics
    candidate_hints: List[str] = []

    # Always include some general hints
    candidate_hints.extend(GENERAL_STRATEGY_HINTS)

    # Add constraint-type specific hints
    if characteristics["sum_constraints"] > 0:
        candidate_hints.extend(SUM_CONSTRAINT_HINTS)

    if characteristics["all_equal_constraints"] > 0:
        candidate_hints.extend(ALL_EQUAL_CONSTRAINT_HINTS)

    # Add puzzle-size specific hints
    if characteristics["is_small_puzzle"]:
        candidate_hints.extend(SMALL_PUZZLE_HINTS)
    elif characteristics["is_large_puzzle"]:
        candidate_hints.extend(LARGE_PUZZLE_HINTS)

    # Add domino-specific hints
    candidate_hints.extend(LIMITED_DOMINO_HINTS)
    if characteristics["has_double_dominoes"]:
        candidate_hints.extend(DOUBLE_DOMINO_HINTS)

    # Filter out previously given hints
    available_hints = [h for h in candidate_hints if h not in previous_hints]

    # If all hints have been used, reset and pick from general hints
    if not available_hints:
        available_hints = GENERAL_STRATEGY_HINTS.copy()

    # Select a hint (use random for variety, but could be made deterministic)
    selected_hint = random.choice(available_hints)

    return HintResult(
        content=selected_hint,
        hint_type="strategy",
    )
