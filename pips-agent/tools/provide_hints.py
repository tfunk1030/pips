"""
Provide Hints Tool

MCP tool for providing strategic hints without solving the puzzle completely.
"""

from claude_agent_sdk import tool
from typing import Any, Dict
import sys
from pathlib import Path
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from solve_pips import (
    parse_ascii_maps,
    region_cells,
    Constraint
)
from utils.hint_engine import generate_hints, explain_constraint_conflicts


@tool(
    name="provide_hints",
    description="Provide strategic hints for solving the puzzle without giving away the complete solution",
    input_schema={"yaml_specification": str}
)
async def provide_hints(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate strategic hints for puzzle.

    Args:
        args: Dict with 'yaml_specification' key

    Returns:
        Tool result dict with hints
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

        # Check for potential issues first
        issues = explain_constraint_conflicts(rmap, constraints, pip_min, pip_max)

        if issues:
            response_text = "⚠️  Potential Issues Detected:\n\n"
            for issue in issues:
                response_text += f"- {issue}\n"
            response_text += "\n❌ These constraints may make the puzzle unsolvable."
            response_text += "\n\nPlease verify the constraints are correct."

            return {
                "content": [{
                    "type": "text",
                    "text": response_text
                }]
            }

        # Generate hints
        hints = generate_hints(rmap, constraints, dominoes, pip_min, pip_max)

        response_text = "💡 Strategic Hints for Solving Your Puzzle:\n\n"
        response_text += "\n".join(hints)

        response_text += "\n\n📝 Would you like:"
        response_text += "\n   - More specific hints for a particular region?"
        response_text += "\n   - The complete solution?"

        return {
            "content": [{
                "type": "text",
                "text": response_text
            }]
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
                "text": f"Error generating hints: {str(e)}"
            }],
            "is_error": True
        }
