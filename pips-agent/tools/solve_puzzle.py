"""
Solve Puzzle Tool

MCP tool for solving Pips puzzles using the CSP solver.
"""

from claude_agent_sdk import tool
from typing import Any, Dict
import sys
from pathlib import Path
import yaml

# Add parent directory to path to import solver
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from solve_pips import (
    parse_ascii_maps,
    build_adjacency,
    region_cells,
    solve,
    render_solution,
    Constraint
)


@tool(
    name="solve_puzzle",
    description="Solve the Pips puzzle completely using constraint satisfaction solver",
    input_schema={"yaml_specification": str}
)
async def solve_puzzle(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Solve puzzle from YAML specification.

    Args:
        args: Dict with 'yaml_specification' key

    Returns:
        Tool result dict with solution or error
    """
    yaml_spec = args.get("yaml_specification")

    if not yaml_spec:
        return {
            "content": [{
                "type": "text",
                "text": "Error: yaml_specification is required"
            }],
            "is_error": True
        }

    try:
        # Parse YAML
        data = yaml.safe_load(yaml_spec)

        # Extract components
        pips = data["pips"]
        pip_min = int(pips["pip_min"])
        pip_max = int(pips["pip_max"])

        dom = data["dominoes"]
        dominoes_raw = dom["tiles"]
        dominoes = [(int(x), int(y)) for x, y in dominoes_raw]

        board = data["board"]
        shape = board["shape"]
        regions_map = board["regions"]

        # Parse board
        cells_set, cell_region = parse_ascii_maps(shape, regions_map)
        cells = sorted(list(cells_set))
        adj = build_adjacency(cells_set)
        rmap = region_cells(cell_region)

        # Parse constraints
        constraints_raw = data["region_constraints"]
        constraints = {}

        for rid, obj in constraints_raw.items():
            ctype = obj["type"]
            if ctype == "sum":
                constraints[rid] = Constraint(
                    type="sum",
                    op=obj["op"],
                    value=int(obj["value"])
                )
            elif ctype == "all_equal":
                constraints[rid] = Constraint(type="all_equal")
            else:
                raise ValueError(f"Unknown constraint type: {ctype}")

        # Solve puzzle
        success, values = solve(
            cells=cells,
            adj=adj,
            cell_region=cell_region,
            region_map=rmap,
            constraints=constraints,
            dominoes=dominoes,
            pip_min=pip_min,
            pip_max=pip_max
        )

        if not success:
            response_text = """❌ No solution found!

The puzzle appears to be unsolvable with the given constraints and dominoes.

Possible reasons:
- Incorrect constraints
- Wrong domino list
- Conflicting region constraints
- Impossible constraint combinations

Would you like me to provide hints about what might be wrong?
"""
            return {
                "content": [{
                    "type": "text",
                    "text": response_text
                }],
                "is_error": True
            }

        # Render solution
        solution_grid = render_solution(shape, values)

        response_text = f"""✅ Puzzle solved successfully!

🎯 Solution:

```
{solution_grid}
```

📊 Verification:
- All cells filled with valid pip values
- All dominoes placed correctly
- All region constraints satisfied

🎉 Congratulations! The puzzle is complete.
"""

        return {
            "content": [{
                "type": "text",
                "text": response_text
            }],
            "solution": values  # Raw solution data
        }

    except KeyError as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error parsing YAML: Missing required field {str(e)}"
            }],
            "is_error": True
        }
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error solving puzzle: {str(e)}"
            }],
            "is_error": True
        }
