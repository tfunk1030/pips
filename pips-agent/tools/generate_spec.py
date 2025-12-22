"""
Generate Puzzle Specification Tool

MCP tool for generating YAML puzzle specifications from extracted data.
"""

from claude_agent_sdk import tool
from typing import Any, Dict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.yaml_generator import create_puzzle_yaml, validate_puzzle_spec
from utils.ocr_helper import merge_constraints_with_user_input


@tool(
    name="generate_puzzle_spec",
    description="Generate YAML puzzle specification from extracted data, OCR constraints, and user input",
    input_schema={
        "cells": list,
        "grid_dims": list,
        "regions": dict,
        "ocr_constraints": dict,
        "user_constraints": dict,
        "dominoes": list,
        "pip_min": int,
        "pip_max": int
    }
)
async def generate_puzzle_spec(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate puzzle specification.

    Args:
        args: Dict with puzzle data

    Returns:
        Tool result dict with YAML specification
    """
    cells = args.get("cells", [])
    grid_dims_list = args.get("grid_dims", [])
    regions = args.get("regions", {})
    ocr_constraints = args.get("ocr_constraints", {})
    user_constraints = args.get("user_constraints", {})
    dominoes = args.get("dominoes", [])
    pip_min = args.get("pip_min", 0)
    pip_max = args.get("pip_max", 6)

    # Validate inputs
    if not cells:
        return {
            "content": [{
                "type": "text",
                "text": "Error: cells data is required"
            }],
            "is_error": True
        }

    if not regions:
        return {
            "content": [{
                "type": "text",
                "text": "Error: regions data is required"
            }],
            "is_error": True
        }

    if not dominoes:
        return {
            "content": [{
                "type": "text",
                "text": "Error: dominoes list is required"
            }],
            "is_error": True
        }

    # Convert grid_dims from list to tuple
    grid_dims = tuple(grid_dims_list) if isinstance(grid_dims_list, list) else grid_dims_list

    # Merge OCR and user constraints
    # OCR constraints come as dict of (constraint_dict, confidence) tuples
    ocr_only = {rid: constraint for rid, (constraint, conf) in ocr_constraints.items()}
    merged_constraints = merge_constraints_with_user_input(
        {rid: (constraint, conf) for rid, (constraint, conf) in ocr_constraints.items()},
        user_constraints,
        confidence_threshold=0.7
    )

    # Check if we have constraints for all regions
    all_region_ids = set(regions.keys())
    constrained_region_ids = set(merged_constraints.keys())
    missing_regions = all_region_ids - constrained_region_ids

    if missing_regions:
        return {
            "content": [{
                "type": "text",
                "text": f"❌ Missing constraints for regions: {', '.join(sorted(missing_regions))}\n\n"
                       f"Please provide constraints for these regions before generating specification."
            }],
            "is_error": True
        }

    try:
        # Generate YAML
        yaml_str = create_puzzle_yaml(
            cells=cells,
            grid_dims=grid_dims,
            regions=regions,
            constraints=merged_constraints,
            dominoes=dominoes,
            pip_min=pip_min,
            pip_max=pip_max
        )

        # Validate generated YAML
        is_valid, validation_msg = validate_puzzle_spec(yaml_str)

        if not is_valid:
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ Generated invalid YAML: {validation_msg}"
                }],
                "is_error": True
            }

        # Success!
        response_text = f"""✅ Puzzle specification generated successfully!

📄 YAML Specification:
```yaml
{yaml_str}
```

📊 Summary:
- Grid: {grid_dims[0]} rows × {grid_dims[1]} columns
- Regions: {len(regions)}
- Dominoes: {len(dominoes)}
- Constraints: {len(merged_constraints)}

🎯 Next step: Ask user if they want to:
1. Solve the puzzle completely
2. Get strategic hints
"""

        return {
            "content": [{
                "type": "text",
                "text": response_text
            }],
            "yaml_specification": yaml_str  # For next tool
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error generating YAML: {str(e)}"
            }],
            "is_error": True
        }
