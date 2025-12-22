"""
Extract Puzzle Tool

MCP tool for extracting puzzle structure from screenshots using CV.
"""

from claude_agent_sdk import tool
from typing import Any, Dict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cv_extraction import extract_puzzle_structure


@tool(
    name="extract_puzzle_from_screenshot",
    description="Extract puzzle grid structure, cells, and regions from a screenshot image using computer vision",
    input_schema={"image_path": str}
)
async def extract_puzzle_from_screenshot(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract puzzle structure from screenshot.

    Args:
        args: Dict with 'image_path' key

    Returns:
        Tool result dict with extracted puzzle data
    """
    image_path = args.get("image_path")

    if not image_path:
        return {
            "content": [{
                "type": "text",
                "text": "Error: image_path is required"
            }],
            "is_error": True
        }

    # Check if file exists
    if not Path(image_path).exists():
        return {
            "content": [{
                "type": "text",
                "text": f"Error: Image file not found: {image_path}"
            }],
            "is_error": True
        }

    # Run extraction
    result = extract_puzzle_structure(image_path)

    if not result.get("success"):
        error_msg = result.get("error", "Unknown error")
        return {
            "content": [{
                "type": "text",
                "text": f"Extraction failed: {error_msg}"
            }],
            "is_error": True
        }

    # Format successful result
    num_cells = result["num_cells"]
    rows, cols = result["grid_dims"]
    regions = result["regions"]
    num_regions = len(regions)

    response_text = f"""Puzzle extraction successful!

📊 Grid Information:
- Dimensions: {rows} rows × {cols} columns
- Total cells detected: {num_cells}
- Number of regions: {num_regions}

🎨 Regions detected:
"""

    for region_id, cell_indices in sorted(regions.items()):
        response_text += f"- Region {region_id}: {len(cell_indices)} cells\n"

    response_text += "\n✅ Puzzle structure extracted. Next steps:"
    response_text += "\n1. Use 'ocr_constraints_from_screenshot' to detect constraint text"
    response_text += "\n2. Or provide constraints manually if OCR fails"

    # Return with structured data
    return {
        "content": [{
            "type": "text",
            "text": response_text
        }],
        "puzzle_data": result  # Include raw data for next steps
    }
