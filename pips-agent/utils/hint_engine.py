"""
Hint Generation Engine

Provides strategic hints for solving Pips puzzles without giving away the complete solution.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from solve_pips import Constraint

Coord = Tuple[int, int]


def analyze_constraints(
    region_map: Dict[str, List[Coord]],
    constraints: Dict[str, Constraint],
    pip_min: int,
    pip_max: int
) -> Dict[str, Dict]:
    """
    Analyze constraints to identify strategic insights.

    Args:
        region_map: Dict mapping region IDs to list of coordinates
        constraints: Dict mapping region IDs to Constraint objects
        pip_min: Minimum pip value
        pip_max: Maximum pip value

    Returns:
        Dict with analysis for each region:
        - difficulty: "easy", "medium", "hard"
        - num_cells: Number of cells in region
        - constraint_type: "sum", "all_equal", etc.
        - possible_combinations: Estimated number of valid combinations
    """
    analysis = {}

    for region_id, coords in region_map.items():
        if region_id not in constraints:
            continue

        constraint = constraints[region_id]
        num_cells = len(coords)

        # Analyze based on constraint type
        if constraint.type == "all_equal":
            # All equal is very constraining
            possible_combos = (pip_max - pip_min + 1)  # One value for all cells
            difficulty = "easy"

        elif constraint.type == "sum" and constraint.op == "==":
            # Exact sum constraint
            target = constraint.value
            avg_per_cell = target / num_cells

            # Very constrained if target is near min or max
            if avg_per_cell < pip_min + 1 or avg_per_cell > pip_max - 1:
                difficulty = "easy"
                possible_combos = 10
            elif num_cells == 2:
                difficulty = "easy"
                possible_combos = 20
            else:
                difficulty = "medium"
                possible_combos = 50

        elif constraint.type == "sum" and constraint.op in ["<", ">"]:
            # Inequality constraint - more flexible
            difficulty = "medium" if num_cells <= 3 else "hard"
            possible_combos = 100

        else:
            difficulty = "medium"
            possible_combos = 50

        analysis[region_id] = {
            "difficulty": difficulty,
            "num_cells": num_cells,
            "constraint_type": constraint.type,
            "constraint_op": constraint.op if constraint.type == "sum" else None,
            "constraint_value": constraint.value if constraint.type == "sum" else None,
            "possible_combinations": possible_combos
        }

    return analysis


def suggest_starting_region(
    region_map: Dict[str, List[Coord]],
    constraints: Dict[str, Constraint],
    pip_min: int,
    pip_max: int
) -> Tuple[str, str]:
    """
    Suggest which region to start with.

    Args:
        region_map: Dict mapping region IDs to list of coordinates
        constraints: Dict mapping region IDs to Constraint objects
        pip_min: Minimum pip value
        pip_max: Maximum pip value

    Returns:
        (region_id, reason) tuple
    """
    analysis = analyze_constraints(region_map, constraints, pip_min, pip_max)

    # Sort by difficulty and number of possibilities
    sorted_regions = sorted(
        analysis.items(),
        key=lambda x: (
            0 if x[1]["difficulty"] == "easy" else 1 if x[1]["difficulty"] == "medium" else 2,
            x[1]["possible_combinations"]
        )
    )

    if not sorted_regions:
        return None, "No regions found"

    best_region_id, info = sorted_regions[0]
    constraint = constraints[best_region_id]

    # Generate reason
    if constraint.type == "all_equal":
        reason = (
            f"Region {best_region_id} requires all cells to be equal - "
            f"this is very constraining! You'll need matching dominoes (like 2-2, 3-3, etc.)"
        )
    elif constraint.type == "sum" and constraint.op == "==":
        reason = (
            f"Region {best_region_id} has only {info['num_cells']} cells and must sum to {constraint.value} - "
            f"limited combinations available"
        )
    elif constraint.type == "sum" and constraint.op in ["<", ">"]:
        reason = (
            f"Region {best_region_id} has constraint {constraint.op} {constraint.value} - "
            f"this eliminates many possibilities"
        )
    else:
        reason = f"Region {best_region_id} appears most constrained"

    return best_region_id, reason


def generate_hints(
    region_map: Dict[str, List[Coord]],
    constraints: Dict[str, Constraint],
    dominoes: List[Tuple[int, int]],
    pip_min: int,
    pip_max: int
) -> List[str]:
    """
    Generate strategic hints for solving the puzzle.

    Args:
        region_map: Dict mapping region IDs to list of coordinates
        constraints: Dict mapping region IDs to Constraint objects
        dominoes: List of available dominoes
        pip_min: Minimum pip value
        pip_max: Maximum pip value

    Returns:
        List of hint strings
    """
    hints = []

    # Suggest starting region
    start_region, reason = suggest_starting_region(region_map, constraints, pip_min, pip_max)

    if start_region:
        hints.append(f"🎯 Start with Region {start_region}")
        hints.append(f"   {reason}")

    # Analyze all regions
    analysis = analyze_constraints(region_map, constraints, pip_min, pip_max)

    # Find easy regions
    easy_regions = [rid for rid, info in analysis.items() if info["difficulty"] == "easy"]

    if len(easy_regions) > 1:
        hints.append(f"\n💡 Other easy regions to tackle: {', '.join(easy_regions[1:3])}")

    # General strategy hints
    hints.append("\n🔍 General Strategy:")
    hints.append("   1. Work on most constrained regions first")
    hints.append("   2. Track which dominoes you've used")
    hints.append("   3. Check if placements violate neighboring region constraints")

    # Specific constraint type hints
    all_equal_regions = [rid for rid, c in constraints.items() if c.type == "all_equal"]
    if all_equal_regions:
        hints.append(f"\n⚠️  'All equal' regions ({', '.join(all_equal_regions)}) need matching dominoes")
        hints.append("   Look for doubles in your tray: 0-0, 1-1, 2-2, etc.")

    # Sum constraint hints
    tight_sum_regions = [
        rid for rid, info in analysis.items()
        if info["constraint_type"] == "sum" and
           info["constraint_op"] == "==" and
           info["num_cells"] == 2
    ]
    if tight_sum_regions:
        hints.append(f"\n📊 Two-cell sum regions are easier to solve")
        hints.append(f"   Focus on: {', '.join(tight_sum_regions)}")

    return hints


def explain_constraint_conflicts(
    region_map: Dict[str, List[Coord]],
    constraints: Dict[str, Constraint],
    pip_min: int,
    pip_max: int
) -> List[str]:
    """
    Explain potential constraint conflicts (useful when puzzle is unsolvable).

    Args:
        region_map: Dict mapping region IDs to list of coordinates
        constraints: Dict mapping region IDs to Constraint objects
        pip_min: Minimum pip value
        pip_max: Maximum pip value

    Returns:
        List of potential issues
    """
    issues = []

    for region_id, coords in region_map.items():
        if region_id not in constraints:
            issues.append(f"Region {region_id} has no constraint specified")
            continue

        constraint = constraints[region_id]
        num_cells = len(coords)

        # Check sum constraints for feasibility
        if constraint.type == "sum" and constraint.op == "==":
            target = constraint.value
            min_possible = num_cells * pip_min
            max_possible = num_cells * pip_max

            if target < min_possible:
                issues.append(
                    f"Region {region_id}: sum={target} is impossible "
                    f"(minimum possible is {min_possible} with {num_cells} cells)"
                )
            elif target > max_possible:
                issues.append(
                    f"Region {region_id}: sum={target} is impossible "
                    f"(maximum possible is {max_possible} with {num_cells} cells)"
                )

        elif constraint.type == "sum" and constraint.op == "<":
            min_possible = num_cells * pip_min
            if min_possible >= constraint.value:
                issues.append(
                    f"Region {region_id}: constraint < {constraint.value} is impossible "
                    f"(minimum possible sum is {min_possible})"
                )

        elif constraint.type == "sum" and constraint.op == ">":
            max_possible = num_cells * pip_max
            if max_possible <= constraint.value:
                issues.append(
                    f"Region {region_id}: constraint > {constraint.value} is impossible "
                    f"(maximum possible sum is {max_possible})"
                )

    return issues
