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


@dataclass
class PuzzleStateValidation:
    """Result of puzzle state validation."""
    is_valid: bool
    is_solved: bool
    error_message: Optional[str] = None
    violations: Optional[List[Dict[str, Any]]] = None


# =============================================================================
# Puzzle State Validation Functions
# =============================================================================

def validate_puzzle_state(
    puzzle_spec: Any,
    current_state: Optional[Dict[str, Any]] = None,
) -> PuzzleStateValidation:
    """
    Validate the current puzzle state for edge cases.

    Checks for:
    1. Solved puzzle (all cells filled and constraints satisfied)
    2. Invalid state (constraint violations)

    Args:
        puzzle_spec: The puzzle specification
        current_state: Optional current puzzle state with user placements

    Returns:
        PuzzleStateValidation with is_valid, is_solved, and any error info
    """
    # Extract puzzle components
    if isinstance(puzzle_spec, dict):
        region_constraints = puzzle_spec.get("region_constraints", {})
        board = puzzle_spec.get("board", {})
        pips = puzzle_spec.get("pips", {})
    else:
        region_constraints = puzzle_spec.region_constraints
        board = puzzle_spec.board
        pips = puzzle_spec.pips

        # Convert to dict-like if needed
        if not isinstance(region_constraints, dict):
            if hasattr(region_constraints, 'items'):
                region_constraints = dict(region_constraints.items())
            elif hasattr(region_constraints, '__dict__'):
                region_constraints = region_constraints.__dict__

    board_dict = board if isinstance(board, dict) else {
        "regions": getattr(board, "regions", []),
        "shape": getattr(board, "shape", [])
    }
    pips_dict = pips if isinstance(pips, dict) else {
        "pip_min": getattr(pips, "pip_min", 0),
        "pip_max": getattr(pips, "pip_max", 6)
    }

    # If no current state provided, puzzle is not solved but valid to proceed
    if not current_state or "placements" not in current_state:
        return PuzzleStateValidation(is_valid=True, is_solved=False)

    placements = current_state.get("placements", {})

    # Get all cells and their regions
    shape = board_dict.get("shape", [])
    regions = board_dict.get("regions", [])
    pip_min = pips_dict.get("pip_min", 0)
    pip_max = pips_dict.get("pip_max", 6)

    # Collect cell values by region
    region_values: Dict[str, List[int]] = {}
    total_cells = 0
    filled_cells = 0
    violations: List[Dict[str, Any]] = []

    for row_idx, (shape_row, region_row) in enumerate(zip(shape, regions)):
        for col_idx, (cell_char, region_char) in enumerate(zip(shape_row, region_row)):
            if cell_char == '.':
                total_cells += 1
                cell_key = f"{row_idx},{col_idx}"

                if cell_key in placements:
                    filled_cells += 1
                    value = placements[cell_key]

                    # Check value is within valid pip range
                    if not isinstance(value, int) or value < pip_min or value > pip_max:
                        violations.append({
                            "type": "invalid_value",
                            "row": row_idx,
                            "col": col_idx,
                            "value": value,
                            "message": f"Value {value} at ({row_idx}, {col_idx}) is outside valid range [{pip_min}-{pip_max}]"
                        })
                    else:
                        # Add to region values
                        if region_char not in region_values:
                            region_values[region_char] = []
                        region_values[region_char].append(value)

    # Check constraint violations for filled regions
    for region_name, constraint in region_constraints.items():
        if isinstance(constraint, dict):
            constraint_type = constraint.get("type", "")
            op = constraint.get("op", "==")
            target_value = constraint.get("value", 0)
        else:
            constraint_type = getattr(constraint, "type", "")
            op = getattr(constraint, "op", "==")
            target_value = getattr(constraint, "value", 0)

        if region_name not in region_values:
            continue

        values = region_values[region_name]

        if constraint_type == "all_equal" and len(values) > 1:
            # All values must be equal
            if len(set(values)) > 1:
                violations.append({
                    "type": "constraint_violation",
                    "region": region_name,
                    "constraint": "all_equal",
                    "values": values,
                    "message": f"Region '{region_name}' has different values but requires all equal: {values}"
                })

        elif constraint_type == "sum" and values:
            # Check if sum constraint could be violated
            current_sum = sum(values)
            # Count expected cells in region
            expected_cells = sum(
                1 for r_row in regions for r_char in r_row if r_char == region_name
            )

            # If region is completely filled, check the constraint
            if len(values) == expected_cells:
                satisfied = _check_sum_constraint(current_sum, op, target_value)
                if not satisfied:
                    violations.append({
                        "type": "constraint_violation",
                        "region": region_name,
                        "constraint": "sum",
                        "op": op,
                        "target": target_value,
                        "actual": current_sum,
                        "message": f"Region '{region_name}' sum is {current_sum}, but constraint requires {op} {target_value}"
                    })

    # Determine result
    if violations:
        return PuzzleStateValidation(
            is_valid=False,
            is_solved=False,
            error_message="Puzzle state contains constraint violations",
            violations=violations
        )

    # Check if puzzle is solved (all cells filled with no violations)
    is_solved = (filled_cells == total_cells and total_cells > 0)

    return PuzzleStateValidation(
        is_valid=True,
        is_solved=is_solved
    )


def get_validation_hint(validation: PuzzleStateValidation) -> Optional[HintResult]:
    """
    Generate a hint based on puzzle state validation results.

    For invalid states, provides guidance on fixing the errors.
    For solved puzzles, indicates the puzzle is complete.

    Args:
        validation: The validation result from validate_puzzle_state

    Returns:
        HintResult if there's a special message, None otherwise
    """
    if validation.is_solved:
        return HintResult(
            content="🎉 Congratulations! This puzzle is already solved. All constraints are satisfied.",
            hint_type="info"
        )

    if not validation.is_valid and validation.violations:
        # Generate guidance based on the first violation
        violation = validation.violations[0]
        violation_type = violation.get("type", "")

        if violation_type == "invalid_value":
            return HintResult(
                content=f"⚠️ Invalid value detected: {violation.get('message', 'Check your placements.')} "
                        f"Remove or change the value at row {violation.get('row', '?')}, column {violation.get('col', '?')}.",
                hint_type="error"
            )

        elif violation_type == "constraint_violation":
            region = violation.get("region", "?")
            constraint = violation.get("constraint", "")

            if constraint == "all_equal":
                return HintResult(
                    content=f"⚠️ Constraint violation in region '{region}': {violation.get('message', '')} "
                            f"This region requires all cells to have the same value.",
                    hint_type="error",
                    region=region
                )
            elif constraint == "sum":
                return HintResult(
                    content=f"⚠️ Constraint violation in region '{region}': The current sum ({violation.get('actual', '?')}) "
                            f"doesn't satisfy the constraint ({violation.get('op', '?')} {violation.get('target', '?')}). "
                            f"Try adjusting the values in this region.",
                    hint_type="error",
                    region=region
                )

        # Generic error guidance
        return HintResult(
            content=f"⚠️ {validation.error_message}. Review your placements and check for errors.",
            hint_type="error"
        )

    return None


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


# =============================================================================
# Level 2 Hint Templates - Focused Direction to Region
# =============================================================================

# Templates for sum constraint regions
SUM_REGION_HINT_TEMPLATES = [
    "Focus on region '{region}' - its sum constraint ({op} {value}) significantly limits valid placements.",
    "Region '{region}' has a sum constraint of {op} {value}. Consider which domino combinations can achieve this.",
    "The sum constraint on region '{region}' ({op} {value}) makes it a good starting point.",
    "Examine region '{region}' closely - few domino pairs can satisfy its {op} {value} sum requirement.",
]

# Templates for all_equal constraint regions
ALL_EQUAL_REGION_HINT_TEMPLATES = [
    "Region '{region}' requires all cells to have equal pip values - this dramatically limits your options.",
    "Focus on region '{region}' with its all-equal constraint. Which pip values appear on multiple domino halves?",
    "The all-equal constraint on region '{region}' is very restrictive - start there.",
    "Region '{region}' needs identical values in each cell. Consider your available dominoes carefully.",
]

# Templates for small regions (fewer cells = more restrictive)
SMALL_REGION_HINT_TEMPLATES = [
    "Region '{region}' has only {cell_count} cell(s), making it easier to enumerate valid placements.",
    "Start with region '{region}' - with fewer cells, the constraint is more limiting.",
    "Consider region '{region}' first - smaller regions have fewer valid domino arrangements.",
]

# Templates for regions with extreme constraint values
EXTREME_CONSTRAINT_HINT_TEMPLATES = [
    "Region '{region}' has an extreme constraint value ({op} {value}) - fewer domino combinations can satisfy it.",
    "The {direction} constraint on region '{region}' ({op} {value}) limits your choices significantly.",
]

# Generic region focus templates
GENERIC_REGION_HINT_TEMPLATES = [
    "Take a closer look at region '{region}' and its constraints.",
    "Region '{region}' may be a productive place to focus your attention.",
    "Consider the constraints on region '{region}' and what placements they allow.",
]


def _get_region_cell_count(region_name: str, board: Dict[str, Any]) -> int:
    """Count how many cells belong to a given region."""
    regions = board.get("regions", []) if isinstance(board, dict) else getattr(board, "regions", [])
    count = 0
    for row in regions:
        count += row.count(region_name)
    return count


def _get_region_info(puzzle_spec: Any) -> List[Dict[str, Any]]:
    """
    Extract information about each region for hint selection.

    Returns a list of region info dicts with:
    - name: region identifier
    - constraint_type: 'sum' or 'all_equal'
    - op: comparison operator (for sum)
    - value: constraint value (for sum)
    - cell_count: number of cells in region
    - restrictiveness_score: estimated difficulty/restrictiveness
    """
    # Handle both dict-like and object-like puzzle specs
    if isinstance(puzzle_spec, dict):
        region_constraints = puzzle_spec.get("region_constraints", {})
        board = puzzle_spec.get("board", {})
    else:
        region_constraints = puzzle_spec.region_constraints
        board = puzzle_spec.board
        # Convert to dict if needed
        if not isinstance(region_constraints, dict):
            if hasattr(region_constraints, 'items'):
                region_constraints = dict(region_constraints.items())
            elif hasattr(region_constraints, '__dict__'):
                region_constraints = region_constraints.__dict__

    board_dict = board if isinstance(board, dict) else (
        {"regions": getattr(board, "regions", []), "shape": getattr(board, "shape", [])}
    )

    region_info_list = []

    for region_name, constraint in region_constraints.items():
        # Handle both dict and object constraints
        if isinstance(constraint, dict):
            constraint_type = constraint.get("type", "")
            op = constraint.get("op", "")
            value = constraint.get("value", 0)
        else:
            constraint_type = getattr(constraint, "type", "")
            op = getattr(constraint, "op", "")
            value = getattr(constraint, "value", 0)

        cell_count = _get_region_cell_count(region_name, board_dict)

        # Calculate restrictiveness score (higher = more restrictive = better hint target)
        restrictiveness = 0

        # All-equal constraints are very restrictive
        if constraint_type == "all_equal":
            restrictiveness += 50

        # Sum constraints with equality are more restrictive than inequalities
        if constraint_type == "sum":
            if op == "==":
                restrictiveness += 30
            elif op == "!=":
                restrictiveness += 10
            else:  # < or >
                restrictiveness += 20

        # Smaller regions are more restrictive
        if cell_count <= 2:
            restrictiveness += 25
        elif cell_count <= 4:
            restrictiveness += 15

        # Extreme values are more restrictive
        if value is not None:
            if value <= 2 or value >= 10:
                restrictiveness += 15

        region_info_list.append({
            "name": region_name,
            "constraint_type": constraint_type,
            "op": op,
            "value": value,
            "cell_count": cell_count,
            "restrictiveness_score": restrictiveness,
        })

    return region_info_list


def _select_target_region(
    region_info_list: List[Dict[str, Any]],
    previous_regions: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Select a region to focus the hint on.

    Prioritizes regions that:
    1. Haven't been hinted before (if previous_regions provided)
    2. Have higher restrictiveness scores
    """
    if not region_info_list:
        return None

    if previous_regions is None:
        previous_regions = set()

    # Filter out previously hinted regions
    available_regions = [r for r in region_info_list if r["name"] not in previous_regions]

    # If all regions have been hinted, reset
    if not available_regions:
        available_regions = region_info_list.copy()

    # Sort by restrictiveness (highest first) and pick from top candidates
    available_regions.sort(key=lambda r: r["restrictiveness_score"], reverse=True)

    # Pick randomly from top 3 most restrictive to add variety
    top_candidates = available_regions[:min(3, len(available_regions))]
    return random.choice(top_candidates)


def _generate_region_hint_text(region_info: Dict[str, Any]) -> str:
    """Generate hint text for a specific region based on its properties."""
    region_name = region_info["name"]
    constraint_type = region_info["constraint_type"]
    op = region_info.get("op", "")
    value = region_info.get("value", 0)
    cell_count = region_info.get("cell_count", 0)

    # Select appropriate template based on constraint type
    if constraint_type == "all_equal":
        template = random.choice(ALL_EQUAL_REGION_HINT_TEMPLATES)
        return template.format(region=region_name)

    elif constraint_type == "sum":
        # Check for extreme values
        if value is not None and (value <= 2 or value >= 10):
            direction = "low" if value <= 2 else "high"
            templates = EXTREME_CONSTRAINT_HINT_TEMPLATES + SUM_REGION_HINT_TEMPLATES
            template = random.choice(templates)
            return template.format(region=region_name, op=op, value=value, direction=direction)
        else:
            template = random.choice(SUM_REGION_HINT_TEMPLATES)
            return template.format(region=region_name, op=op, value=value)

    # For small regions regardless of constraint type
    if cell_count <= 2:
        template = random.choice(SMALL_REGION_HINT_TEMPLATES)
        return template.format(region=region_name, cell_count=cell_count)

    # Fallback to generic template
    template = random.choice(GENERIC_REGION_HINT_TEMPLATES)
    return template.format(region=region_name)


def generate_hint_level_2(
    puzzle_spec: Any,
    previous_hints: Optional[List[str]] = None,
    previous_regions: Optional[Set[str]] = None,
) -> HintResult:
    """
    Generate a Level 2 focused direction hint.

    Level 2 hints identify a specific region or constraint where the user
    should focus their attention. They don't reveal values but point to
    productive areas of the puzzle.

    Args:
        puzzle_spec: The puzzle specification (PuzzleSpec model or dict)
        previous_hints: List of previously given hint texts to avoid repetition
        previous_regions: Set of region names already hinted to avoid repetition

    Returns:
        HintResult with region-focused direction content
    """
    if previous_hints is None:
        previous_hints = []
    if previous_regions is None:
        previous_regions = set()

    # Extract region information from puzzle
    region_info_list = _get_region_info(puzzle_spec)

    # Handle edge case: no regions defined
    if not region_info_list:
        return HintResult(
            content="Examine the puzzle constraints carefully to find a starting point.",
            hint_type="direction",
        )

    # Select a target region
    target_region = _select_target_region(region_info_list, previous_regions)

    if target_region is None:
        return HintResult(
            content="Review each region's constraints to find the most restrictive one.",
            hint_type="direction",
        )

    # Generate hint text for the selected region
    hint_text = _generate_region_hint_text(target_region)

    # Avoid exact repetition of hint text
    attempts = 0
    while hint_text in previous_hints and attempts < 5:
        hint_text = _generate_region_hint_text(target_region)
        attempts += 1

    return HintResult(
        content=hint_text,
        hint_type="direction",
        region=target_region["name"],
    )


# =============================================================================
# Level 3 Hint Logic - Specific Cell Placement
# =============================================================================

# Templates for Level 3 cell placement hints
CELL_PLACEMENT_HINT_TEMPLATES = [
    "Place a {value} at row {row}, column {col} to satisfy the constraint on region '{region}'.",
    "Cell at position ({row}, {col}) should have the value {value}.",
    "Try placing {value} in the cell at row {row}, column {col}.",
    "The cell at ({row}, {col}) in region '{region}' needs to be {value}.",
    "Put a {value} in row {row}, column {col} - this satisfies the '{region}' constraint.",
]

# Templates when we can identify a domino placement
DOMINO_PLACEMENT_HINT_TEMPLATES = [
    "Place the [{pip1}, {pip2}] domino starting at row {row}, column {col}.",
    "The domino [{pip1}, {pip2}] fits well starting at position ({row}, {col}).",
    "Try the [{pip1}, {pip2}] domino at row {row}, column {col}.",
]

# Fallback when analysis can't determine specific placement
CELL_HINT_FALLBACK_TEMPLATES = [
    "Look at the cell at row {row}, column {col} in region '{region}' - consider what values satisfy the constraint.",
    "Focus on position ({row}, {col}) and determine what value fits the '{region}' constraint.",
    "The cell at ({row}, {col}) is key to solving region '{region}'.",
]


def _get_cell_positions_by_region(board: Dict[str, Any]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Extract cell positions grouped by region.

    Returns a dict mapping region names to lists of (row, col) tuples.
    """
    regions = board.get("regions", []) if isinstance(board, dict) else getattr(board, "regions", [])
    shape = board.get("shape", []) if isinstance(board, dict) else getattr(board, "shape", [])

    region_cells: Dict[str, List[Tuple[int, int]]] = {}

    for row_idx, (shape_row, region_row) in enumerate(zip(shape, regions)):
        for col_idx, (cell_char, region_char) in enumerate(zip(shape_row, region_row)):
            # Only consider playable cells
            if cell_char == '.':
                if region_char not in region_cells:
                    region_cells[region_char] = []
                region_cells[region_char].append((row_idx, col_idx))

    return region_cells


def _get_available_tiles(dominoes: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Extract available domino tiles as list of (pip1, pip2) tuples."""
    tiles = dominoes.get("tiles", []) if isinstance(dominoes, dict) else getattr(dominoes, "tiles", [])
    return [(t[0], t[1]) for t in tiles if len(t) >= 2]


def _get_pip_range(pips: Dict[str, Any]) -> Tuple[int, int]:
    """Extract pip range from pips config."""
    pip_min = pips.get("pip_min", 0) if isinstance(pips, dict) else getattr(pips, "pip_min", 0)
    pip_max = pips.get("pip_max", 6) if isinstance(pips, dict) else getattr(pips, "pip_max", 6)
    return pip_min, pip_max


def _find_valid_placements_for_region(
    region_name: str,
    region_cells: List[Tuple[int, int]],
    constraint: Dict[str, Any],
    available_tiles: List[Tuple[int, int]],
    pip_range: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """
    Find valid cell placements for a region based on its constraint.

    Returns a list of possible placements with cell, value, and confidence info.
    """
    constraint_type = constraint.get("type", "") if isinstance(constraint, dict) else getattr(constraint, "type", "")
    op = constraint.get("op", "==") if isinstance(constraint, dict) else getattr(constraint, "op", "==")
    value = constraint.get("value", 0) if isinstance(constraint, dict) else getattr(constraint, "value", 0)

    valid_placements = []
    pip_min, pip_max = pip_range

    # Get all pip values that appear on our available dominoes
    available_pips: Set[int] = set()
    for pip1, pip2 in available_tiles:
        available_pips.add(pip1)
        available_pips.add(pip2)

    if constraint_type == "all_equal":
        # For all_equal constraints, find which pip values can fill all cells
        # Each cell must have the same value
        cell_count = len(region_cells)

        # We need enough domino halves with the same pip value
        for pip_val in available_pips:
            # Count how many times this pip appears across all dominoes
            pip_count = sum(
                (1 if pip1 == pip_val else 0) + (1 if pip2 == pip_val else 0)
                for pip1, pip2 in available_tiles
            )

            if pip_count >= cell_count and region_cells:
                # This pip value could fill the region
                first_cell = region_cells[0]
                valid_placements.append({
                    "row": first_cell[0],
                    "col": first_cell[1],
                    "value": pip_val,
                    "region": region_name,
                    "confidence": "high" if pip_count == cell_count else "medium",
                    "reason": "all_equal",
                })

    elif constraint_type == "sum":
        # For sum constraints, find pip combinations that satisfy the sum
        cell_count = len(region_cells)

        if cell_count == 1:
            # Single cell: value must equal the sum target
            for pip_val in available_pips:
                if _check_sum_constraint(pip_val, op, value):
                    first_cell = region_cells[0]
                    valid_placements.append({
                        "row": first_cell[0],
                        "col": first_cell[1],
                        "value": pip_val,
                        "region": region_name,
                        "confidence": "high",
                        "reason": "sum_single_cell",
                    })

        elif cell_count == 2:
            # Two cells: find domino that satisfies sum constraint
            for pip1, pip2 in available_tiles:
                total = pip1 + pip2
                if _check_sum_constraint(total, op, value):
                    first_cell = region_cells[0]
                    valid_placements.append({
                        "row": first_cell[0],
                        "col": first_cell[1],
                        "value": pip1,
                        "region": region_name,
                        "confidence": "high",
                        "reason": "sum_two_cell",
                        "domino": (pip1, pip2),
                    })

        else:
            # Multiple cells: harder to determine, use heuristics
            # Find pip values that are likely to be part of the solution
            target_avg = value / cell_count if value else 0
            closest_pip = min(available_pips, key=lambda p: abs(p - target_avg)) if available_pips else 0

            if region_cells and closest_pip is not None:
                first_cell = region_cells[0]
                valid_placements.append({
                    "row": first_cell[0],
                    "col": first_cell[1],
                    "value": closest_pip,
                    "region": region_name,
                    "confidence": "low",
                    "reason": "sum_heuristic",
                })

    return valid_placements


def _check_sum_constraint(total: int, op: str, value: int) -> bool:
    """Check if a total satisfies a sum constraint."""
    if op == "==":
        return total == value
    elif op == "!=":
        return total != value
    elif op == "<":
        return total < value
    elif op == ">":
        return total > value
    elif op == "<=":
        return total <= value
    elif op == ">=":
        return total >= value
    return False


def _analyze_puzzle_for_cell_hint(puzzle_spec: Any) -> List[Dict[str, Any]]:
    """
    Analyze the puzzle to find determinable cell placements.

    Returns a list of placement suggestions sorted by confidence.
    """
    # Extract puzzle components
    if isinstance(puzzle_spec, dict):
        region_constraints = puzzle_spec.get("region_constraints", {})
        board = puzzle_spec.get("board", {})
        dominoes = puzzle_spec.get("dominoes", {})
        pips = puzzle_spec.get("pips", {})
    else:
        region_constraints = puzzle_spec.region_constraints
        board = puzzle_spec.board
        dominoes = puzzle_spec.dominoes
        pips = puzzle_spec.pips

        # Convert to dict-like if needed
        if not isinstance(region_constraints, dict):
            if hasattr(region_constraints, 'items'):
                region_constraints = dict(region_constraints.items())
            elif hasattr(region_constraints, '__dict__'):
                region_constraints = region_constraints.__dict__

    board_dict = board if isinstance(board, dict) else {
        "regions": getattr(board, "regions", []),
        "shape": getattr(board, "shape", [])
    }
    dominoes_dict = dominoes if isinstance(dominoes, dict) else {
        "tiles": getattr(dominoes, "tiles", [])
    }
    pips_dict = pips if isinstance(pips, dict) else {
        "pip_min": getattr(pips, "pip_min", 0),
        "pip_max": getattr(pips, "pip_max", 6)
    }

    # Get cell positions by region
    region_cells = _get_cell_positions_by_region(board_dict)

    # Get available tiles and pip range
    available_tiles = _get_available_tiles(dominoes_dict)
    pip_range = _get_pip_range(pips_dict)

    # Collect all valid placements
    all_placements = []

    for region_name, constraint in region_constraints.items():
        cells = region_cells.get(region_name, [])
        if not cells:
            continue

        constraint_dict = constraint if isinstance(constraint, dict) else {
            "type": getattr(constraint, "type", ""),
            "op": getattr(constraint, "op", ""),
            "value": getattr(constraint, "value", 0)
        }

        placements = _find_valid_placements_for_region(
            region_name,
            cells,
            constraint_dict,
            available_tiles,
            pip_range,
        )
        all_placements.extend(placements)

    # Sort by confidence (high first)
    confidence_order = {"high": 0, "medium": 1, "low": 2}
    all_placements.sort(key=lambda p: confidence_order.get(p.get("confidence", "low"), 3))

    return all_placements


def generate_hint_level_3(
    puzzle_spec: Any,
    previous_hints: Optional[List[str]] = None,
    previous_cells: Optional[Set[Tuple[int, int]]] = None,
) -> HintResult:
    """
    Generate a Level 3 specific cell placement hint.

    Level 3 hints reveal a specific cell coordinate with its correct value,
    giving the user a concrete placement to make.

    Args:
        puzzle_spec: The puzzle specification (PuzzleSpec model or dict)
        previous_hints: List of previously given hint texts to avoid repetition
        previous_cells: Set of (row, col) tuples already hinted to avoid repetition

    Returns:
        HintResult with specific cell placement information
    """
    if previous_hints is None:
        previous_hints = []
    if previous_cells is None:
        previous_cells = set()

    # Analyze puzzle to find determinable placements
    placements = _analyze_puzzle_for_cell_hint(puzzle_spec)

    # Filter out previously hinted cells
    available_placements = [
        p for p in placements
        if (p["row"], p["col"]) not in previous_cells
    ]

    # If all cells have been hinted, reset
    if not available_placements and placements:
        available_placements = placements

    # Handle edge case: no placements found
    if not available_placements:
        # Fallback: get any cell from the puzzle
        if isinstance(puzzle_spec, dict):
            board = puzzle_spec.get("board", {})
        else:
            board = puzzle_spec.board

        board_dict = board if isinstance(board, dict) else {
            "regions": getattr(board, "regions", []),
            "shape": getattr(board, "shape", [])
        }

        region_cells = _get_cell_positions_by_region(board_dict)

        # Get first available cell
        for region_name, cells in region_cells.items():
            if cells:
                first_cell = cells[0]
                template = random.choice(CELL_HINT_FALLBACK_TEMPLATES)
                hint_text = template.format(
                    row=first_cell[0],
                    col=first_cell[1],
                    region=region_name,
                )
                return HintResult(
                    content=hint_text,
                    hint_type="cell",
                    region=region_name,
                    cell={"row": first_cell[0], "col": first_cell[1]},
                )

        # Ultimate fallback
        return HintResult(
            content="Analyze the puzzle constraints to determine where to place the next domino.",
            hint_type="cell",
        )

    # Select the best placement (highest confidence)
    selected = available_placements[0]

    # Generate hint text
    row = selected["row"]
    col = selected["col"]
    value = selected["value"]
    region = selected["region"]
    confidence = selected.get("confidence", "medium")

    # Check if we have domino info for a more specific hint
    domino = selected.get("domino")
    if domino and confidence == "high":
        template = random.choice(DOMINO_PLACEMENT_HINT_TEMPLATES)
        hint_text = template.format(
            pip1=domino[0],
            pip2=domino[1],
            row=row,
            col=col,
        )
    elif confidence == "high" or confidence == "medium":
        template = random.choice(CELL_PLACEMENT_HINT_TEMPLATES)
        hint_text = template.format(
            value=value,
            row=row,
            col=col,
            region=region,
        )
    else:
        template = random.choice(CELL_HINT_FALLBACK_TEMPLATES)
        hint_text = template.format(
            row=row,
            col=col,
            region=region,
        )

    # Avoid exact repetition
    attempts = 0
    while hint_text in previous_hints and attempts < 5:
        if confidence == "high" or confidence == "medium":
            template = random.choice(CELL_PLACEMENT_HINT_TEMPLATES)
            hint_text = template.format(
                value=value,
                row=row,
                col=col,
                region=region,
            )
        else:
            template = random.choice(CELL_HINT_FALLBACK_TEMPLATES)
            hint_text = template.format(
                row=row,
                col=col,
                region=region,
            )
        attempts += 1

    return HintResult(
        content=hint_text,
        hint_type="cell",
        region=region,
        cell={"row": row, "col": col, "value": value},
    )


# =============================================================================
# Level 4 Hint Logic - Partial Solution (3-5 cells)
# =============================================================================

# Templates for Level 4 partial solution hints
PARTIAL_SOLUTION_INTRO_TEMPLATES = [
    "Here's a partial solution to help you make progress:",
    "These placements will get you started:",
    "Try these cell values to move forward:",
    "Here are several confirmed placements:",
    "Use these values to break through your stuck point:",
]

# Templates for individual cell descriptions within partial solution
PARTIAL_SOLUTION_CELL_TEMPLATES = [
    "• Row {row}, Column {col}: place {value}",
    "• ({row}, {col}) = {value}",
    "• Cell at row {row}, col {col} → {value}",
]

# Templates for domino-specific partial solutions
PARTIAL_SOLUTION_DOMINO_TEMPLATES = [
    "• Place domino [{pip1}, {pip2}] at ({row}, {col})",
    "• Domino [{pip1}, {pip2}] starting at row {row}, col {col}",
]

# Fallback when not enough cells can be determined
PARTIAL_SOLUTION_FALLBACK_TEMPLATES = [
    "Based on the puzzle constraints, here are the placements that can be determined:",
    "The following cells have determinable values:",
    "These are the most certain placements available:",
]


def _collect_unique_placements(
    placements: List[Dict[str, Any]],
    target_count: int = 5,
    min_count: int = 3,
) -> List[Dict[str, Any]]:
    """
    Collect unique cell placements, prioritizing high confidence ones.

    Args:
        placements: List of placement dictionaries from analysis
        target_count: Ideal number of placements to return (default 5)
        min_count: Minimum number to try to collect (default 3)

    Returns:
        List of unique placements (by cell position)
    """
    seen_cells: Set[Tuple[int, int]] = set()
    unique_placements: List[Dict[str, Any]] = []

    for placement in placements:
        cell_key = (placement["row"], placement["col"])
        if cell_key not in seen_cells:
            seen_cells.add(cell_key)
            unique_placements.append(placement)

            if len(unique_placements) >= target_count:
                break

    return unique_placements


def _format_partial_solution_text(
    placements: List[Dict[str, Any]],
) -> str:
    """
    Format multiple placements into a readable partial solution hint.

    Args:
        placements: List of placement dictionaries

    Returns:
        Formatted hint text with intro and cell list
    """
    if not placements:
        return "Unable to determine specific cell placements at this time."

    # Select intro template based on placement count
    if len(placements) >= 3:
        intro = random.choice(PARTIAL_SOLUTION_INTRO_TEMPLATES)
    else:
        intro = random.choice(PARTIAL_SOLUTION_FALLBACK_TEMPLATES)

    # Format each placement
    cell_lines = []
    for placement in placements:
        row = placement["row"]
        col = placement["col"]
        value = placement["value"]
        domino = placement.get("domino")

        if domino:
            template = random.choice(PARTIAL_SOLUTION_DOMINO_TEMPLATES)
            line = template.format(pip1=domino[0], pip2=domino[1], row=row, col=col)
        else:
            template = random.choice(PARTIAL_SOLUTION_CELL_TEMPLATES)
            line = template.format(row=row, col=col, value=value)

        cell_lines.append(line)

    return f"{intro}\n" + "\n".join(cell_lines)


def generate_hint_level_4(
    puzzle_spec: Any,
    previous_hints: Optional[List[str]] = None,
    previous_cells: Optional[Set[Tuple[int, int]]] = None,
) -> HintResult:
    """
    Generate a Level 4 partial solution hint.

    Level 4 hints reveal multiple cell placements (3-5 cells) to provide
    substantial progress for users who are truly stuck. This is the most
    revealing hint level, offering a partial solution without giving away
    the entire puzzle.

    Args:
        puzzle_spec: The puzzle specification (PuzzleSpec model or dict)
        previous_hints: List of previously given hint texts to avoid repetition
        previous_cells: Set of (row, col) tuples already hinted to avoid repetition

    Returns:
        HintResult with partial solution (multiple cell placements)
    """
    if previous_hints is None:
        previous_hints = []
    if previous_cells is None:
        previous_cells = set()

    # Analyze puzzle to find determinable placements (reuse Level 3 analysis)
    all_placements = _analyze_puzzle_for_cell_hint(puzzle_spec)

    # Filter out previously hinted cells if we have enough remaining
    available_placements = [
        p for p in all_placements
        if (p["row"], p["col"]) not in previous_cells
    ]

    # If not enough new cells, include previously hinted ones
    if len(available_placements) < 3:
        available_placements = all_placements

    # Collect 3-5 unique placements
    selected_placements = _collect_unique_placements(
        available_placements,
        target_count=5,
        min_count=3,
    )

    # Handle edge case: no or very few placements found
    if not selected_placements:
        # Try to get any cells from the puzzle as a fallback
        if isinstance(puzzle_spec, dict):
            board = puzzle_spec.get("board", {})
        else:
            board = puzzle_spec.board

        board_dict = board if isinstance(board, dict) else {
            "regions": getattr(board, "regions", []),
            "shape": getattr(board, "shape", [])
        }

        region_cells = _get_cell_positions_by_region(board_dict)

        # Collect first few cells from each region as fallback
        fallback_cells = []
        for region_name, cells in region_cells.items():
            for cell in cells[:2]:  # Take up to 2 cells per region
                fallback_cells.append({
                    "row": cell[0],
                    "col": cell[1],
                    "region": region_name,
                    "value": "?",  # Unknown value
                    "confidence": "low",
                })
                if len(fallback_cells) >= 5:
                    break
            if len(fallback_cells) >= 5:
                break

        if fallback_cells:
            return HintResult(
                content="The puzzle constraints are complex. Focus on these key cells:\n" +
                        "\n".join([f"• Row {c['row']}, Column {c['col']} in region '{c['region']}'"
                                   for c in fallback_cells[:5]]),
                hint_type="partial_solution",
                cells=[{"row": c["row"], "col": c["col"], "region": c.get("region")}
                       for c in fallback_cells[:5]],
            )

        # Ultimate fallback
        return HintResult(
            content="The puzzle requires careful analysis. Try working through each region's constraints systematically.",
            hint_type="partial_solution",
            cells=[],
        )

    # Format the partial solution text
    hint_text = _format_partial_solution_text(selected_placements)

    # Build the cells list for the response
    cells_list = [
        {
            "row": p["row"],
            "col": p["col"],
            "value": p["value"],
            "region": p.get("region"),
        }
        for p in selected_placements
    ]

    # Determine the primary region (most common in selections)
    region_counts: Dict[str, int] = {}
    for p in selected_placements:
        region = p.get("region", "")
        if region:
            region_counts[region] = region_counts.get(region, 0) + 1

    primary_region = max(region_counts, key=region_counts.get) if region_counts else None

    return HintResult(
        content=hint_text,
        hint_type="partial_solution",
        region=primary_region,
        cells=cells_list,
    )
