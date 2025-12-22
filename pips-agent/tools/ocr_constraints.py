"""
OCR Constraints Tool

MCP tool for detecting constraint text from puzzle screenshots using OCR.
"""

from claude_agent_sdk import tool
from typing import Any, Dict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.ocr_helper import detect_constraints_from_image


@tool(
    name="ocr_constraints_from_screenshot",
    description="Use OCR to detect constraint text from the puzzle screenshot",
    input_schema={"image_path": str, "regions": dict, "cells": list}
)
async def ocr_constraints_from_screenshot(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect constraints using OCR.

    Args:
        args: Dict with 'image_path', 'regions', and 'cells' keys

    Returns:
        Tool result dict with detected constraints and confidence scores
    """
    image_path = args.get("image_path")
    regions = args.get("regions", {})
    cells = args.get("cells", [])

    if not image_path:
        return {
            "content": [{
                "type": "text",
                "text": "Error: image_path is required"
            }],
            "is_error": True
        }

    if not regions:
        return {
            "content": [{
                "type": "text",
                "text": "Error: regions data is required (from extract_puzzle_from_screenshot)"
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

    # Run OCR
    try:
        detected_constraints = detect_constraints_from_image(image_path, regions, cells)

        if not detected_constraints:
            return {
                "content": [{
                    "type": "text",
                    "text": "❌ No constraints detected via OCR. Please provide constraints manually."
                }]
            }

        # Format results with confidence scores
        response_text = "🔍 OCR Constraint Detection Results:\n\n"

        high_confidence = []
        medium_confidence = []
        low_confidence = []

        for region_id, (constraint, confidence) in detected_constraints.items():
            conf_pct = int(confidence * 100)

            # Format constraint string
            if constraint["type"] == "all_equal":
                constraint_str = "all equal"
            else:
                op = constraint["op"]
                value = constraint["value"]
                constraint_str = f"{op} {value}"

            entry = f"  Region {region_id}: \"{constraint_str}\" (confidence: {conf_pct}%)"

            if confidence >= 0.8:
                high_confidence.append(entry)
            elif confidence >= 0.6:
                medium_confidence.append(entry)
            else:
                low_confidence.append(entry)

        if high_confidence:
            response_text += "✅ High Confidence (≥80%):\n" + "\n".join(high_confidence) + "\n\n"

        if medium_confidence:
            response_text += "⚠️  Medium Confidence (60-79%):\n" + "\n".join(medium_confidence) + "\n\n"

        if low_confidence:
            response_text += "❓ Low Confidence (<60%):\n" + "\n".join(low_confidence) + "\n\n"

        # Check for missing regions
        all_region_ids = set(regions.keys())
        detected_region_ids = set(detected_constraints.keys())
        missing_regions = all_region_ids - detected_region_ids

        if missing_regions:
            response_text += f"\n⚠️  Could not detect constraints for regions: {', '.join(sorted(missing_regions))}"
            response_text += "\n   Please provide these constraints manually."

        response_text += "\n\n💡 Next step: Use 'generate_puzzle_spec' to create the puzzle specification."

        return {
            "content": [{
                "type": "text",
                "text": response_text
            }],
            "detected_constraints": detected_constraints  # Raw data for next step
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"OCR error: {str(e)}\n\nPlease provide constraints manually."
            }],
            "is_error": True
        }
