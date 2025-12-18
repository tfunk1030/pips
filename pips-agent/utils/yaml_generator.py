"""
YAML Puzzle Specification Generator

Auto-generates YAML puzzle specifications from extracted puzzle data.
"""

import yaml
from typing import Dict, List, Tuple


def generate_ascii_shape(
    cells: List[Tuple[int, int, int, int]],
    grid_dims: Tuple[int, int]
) -> str:
    """
    Generate ASCII shape grid from cell positions.

    Args:
        cells: List of (x, y, w, h) tuples
        grid_dims: (rows, cols) tuple

    Returns:
        Multi-line ASCII string with '.' for cells and '#' for empty spaces
    """
    rows, cols = grid_dims

    # Create grid mapping
    grid = [['#' for _ in range(cols)] for _ in range(rows)]

    # Sort cells by position to assign grid coordinates
    cells_sorted = sorted(cells, key=lambda c: (c[1], c[0]))  # Sort by y, then x

    # Assign cells to grid positions
    cell_idx = 0
    for r in range(rows):
        row_cells = []
        # Find cells in this row
        for cell in cells_sorted:
            if cell_idx < len(cells_sorted):
                row_cells.append(cell)
        # For simplicity, assume regular grid
        for c in range(min(len(row_cells), cols)):
            grid[r][c] = '.'
            cell_idx += 1

    # Convert to string
    lines = [''.join(row) for row in grid]
    return '\n'.join(lines)


def generate_ascii_regions(
    grid_dims: Tuple[int, int],
    regions: Dict[str, List[int]]
) -> str:
    """
    Generate ASCII region labels grid.

    Args:
        grid_dims: (rows, cols) tuple
        regions: Dict mapping region letters to list of cell indices

    Returns:
        Multi-line ASCII string with region labels
    """
    rows, cols = grid_dims

    # Create grid
    grid = [['#' for _ in range(cols)] for _ in range(rows)]

    # Map cell indices to grid positions
    cell_idx = 0
    for r in range(rows):
        for c in range(cols):
            # Find which region this cell belongs to
            for region_letter, cell_indices in regions.items():
                if cell_idx in cell_indices:
                    grid[r][c] = region_letter
                    break
            cell_idx += 1

    # Convert to string
    lines = [''.join(row) for row in grid]
    return '\n'.join(lines)


def create_puzzle_yaml(
    cells: List[Tuple[int, int, int, int]],
    grid_dims: Tuple[int, int],
    regions: Dict[str, List[int]],
    constraints: Dict[str, Dict],
    dominoes: List[Tuple[int, int]],
    pip_min: int = 0,
    pip_max: int = 6
) -> str:
    """
    Create complete YAML puzzle specification.

    Args:
        cells: List of (x, y, w, h) tuples
        grid_dims: (rows, cols) tuple
        regions: Dict mapping region letters to list of cell indices
        constraints: Dict mapping region letters to constraint specs
        dominoes: List of (pip1, pip2) tuples
        pip_min: Minimum pip value (default: 0)
        pip_max: Maximum pip value (default: 6)

    Returns:
        YAML string
    """
    # Generate ASCII grids
    shape = generate_ascii_shape(cells, grid_dims)
    region_labels = generate_ascii_regions(grid_dims, regions)

    # Build puzzle spec
    puzzle_spec = {
        "pips": {
            "pip_min": pip_min,
            "pip_max": pip_max
        },
        "dominoes": {
            "unique": True,
            "tiles": dominoes
        },
        "board": {
            "shape": shape,
            "regions": region_labels
        },
        "region_constraints": constraints
    }

    # Convert to YAML with proper formatting
    yaml_str = yaml.dump(puzzle_spec, default_flow_style=False, sort_keys=False)

    # Add pipe notation for multi-line strings
    yaml_str = yaml_str.replace("shape: '", "shape: |\\n    ")
    yaml_str = yaml_str.replace("regions: '", "regions: |\\n    ")

    return yaml_str


def validate_puzzle_spec(yaml_str: str) -> Tuple[bool, str]:
    """
    Validate YAML puzzle specification.

    Args:
        yaml_str: YAML string to validate

    Returns:
        (is_valid, error_message) tuple
    """
    try:
        data = yaml.safe_load(yaml_str)

        # Check required fields
        required_top_level = ["pips", "dominoes", "board", "region_constraints"]
        for field in required_top_level:
            if field not in data:
                return False, f"Missing required field: {field}"

        # Check pips
        if "pip_min" not in data["pips"] or "pip_max" not in data["pips"]:
            return False, "Missing pip_min or pip_max"

        # Check dominoes
        if "tiles" not in data["dominoes"]:
            return False, "Missing dominoes tiles"

        # Check board
        if "shape" not in data["board"] or "regions" not in data["board"]:
            return False, "Missing board shape or regions"

        # Check that shape and regions have same dimensions
        shape_lines = [line for line in data["board"]["shape"].split("\n") if line.strip()]
        regions_lines = [line for line in data["board"]["regions"].split("\n") if line.strip()]

        if len(shape_lines) != len(regions_lines):
            return False, "Shape and regions have different number of rows"

        for i, (shape_line, region_line) in enumerate(zip(shape_lines, regions_lines)):
            if len(shape_line) != len(region_line):
                return False, f"Shape and regions have different length at row {i}"

        return True, "Valid"

    except yaml.YAMLError as e:
        return False, f"YAML syntax error: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"
