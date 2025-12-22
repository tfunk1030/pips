"""
OCR Helper Utilities

Uses Tesseract OCR to detect constraint text from puzzle screenshots.
"""

import cv2
import pytesseract
import re
from typing import Dict, List, Tuple, Optional
import numpy as np


def extract_text_from_image(image_path: str) -> List[Tuple[str, Tuple[int, int], float]]:
    """
    Extract all text from image with positions and confidence scores.

    Args:
        image_path: Path to image file

    Returns:
        List of (text, (x, y), confidence) tuples
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return []

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to preprocess
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Run OCR
        data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

        results = []
        n_boxes = len(data['text'])

        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])

            if text and conf > 0:  # Valid text with confidence
                x, y = int(data['left'][i]), int(data['top'][i])
                results.append((text, (x, y), conf))

        return results

    except Exception as e:
        print(f"OCR error: {e}")
        return []


def parse_constraint_from_text(text: str) -> Optional[Dict]:
    """
    Parse constraint specification from text.

    Args:
        text: Text string potentially containing constraint

    Returns:
        Constraint dict or None if no valid constraint found
    """
    text = text.lower().strip()

    # Pattern: "sum = N" or "sum: N" or "= N"
    sum_pattern = r'(?:sum\s*)?[=:]\s*(\d+)'
    match = re.search(sum_pattern, text)
    if match:
        value = int(match.group(1))
        return {"type": "sum", "op": "==", "value": value}

    # Pattern: "< N" or "less than N"
    lt_pattern = r'(?:less\s+than\s+)?<\s*(\d+)'
    match = re.search(lt_pattern, text)
    if match:
        value = int(match.group(1))
        return {"type": "sum", "op": "<", "value": value}

    # Pattern: "> N" or "greater than N"
    gt_pattern = r'(?:greater\s+than\s+)?>\s*(\d+)'
    match = re.search(gt_pattern, text)
    if match:
        value = int(match.group(1))
        return {"type": "sum", "op": ">", "value": value}

    # Pattern: "!= N" or "not equal N"
    ne_pattern = r'(?:not\s+equal\s+)?!=\s*(\d+)'
    match = re.search(ne_pattern, text)
    if match:
        value = int(match.group(1))
        return {"type": "sum", "op": "!=", "value": value}

    # Pattern: "all equal" or "same" or "equal"
    if any(phrase in text for phrase in ["all equal", "all same", "same value", "equal"]):
        return {"type": "all_equal"}

    return None


def detect_constraints_from_image(
    image_path: str,
    regions: Dict[str, List[int]],
    cells: List[Tuple[int, int, int, int]]
) -> Dict[str, Tuple[Dict, float]]:
    """
    Detect constraints for each region using OCR.

    Args:
        image_path: Path to image file
        regions: Dict mapping region letters to cell indices
        cells: List of (x, y, w, h) tuples

    Returns:
        Dict mapping region letters to (constraint_dict, confidence) tuples
    """
    # Extract all text from image
    text_items = extract_text_from_image(image_path)

    if not text_items:
        return {}

    # For each region, find nearest text and parse as constraint
    region_constraints = {}

    for region_letter, cell_indices in regions.items():
        if not cell_indices:
            continue

        # Calculate region centroid
        region_cells = [cells[i] for i in cell_indices if i < len(cells)]
        if not region_cells:
            continue

        region_x = np.mean([x + w/2 for x, y, w, h in region_cells])
        region_y = np.mean([y + h/2 for x, y, w, h in region_cells])

        # Find closest text to this region
        best_text = None
        best_distance = float('inf')
        best_confidence = 0.0

        for text, (tx, ty), conf in text_items:
            distance = np.sqrt((tx - region_x)**2 + (ty - region_y)**2)
            if distance < best_distance:
                constraint = parse_constraint_from_text(text)
                if constraint:  # Only consider valid constraints
                    best_distance = distance
                    best_text = text
                    best_confidence = conf

        # If we found a constraint near this region, add it
        if best_text:
            constraint = parse_constraint_from_text(best_text)
            if constraint:
                # Normalize confidence to 0-1 scale
                normalized_conf = min(best_confidence / 100.0, 1.0)
                region_constraints[region_letter] = (constraint, normalized_conf)

    return region_constraints


def merge_constraints_with_user_input(
    ocr_constraints: Dict[str, Tuple[Dict, float]],
    user_constraints: Dict[str, Dict],
    confidence_threshold: float = 0.7
) -> Dict[str, Dict]:
    """
    Merge OCR-detected constraints with user-provided constraints.

    Args:
        ocr_constraints: Dict from detect_constraints_from_image
        user_constraints: User-provided constraints dict
        confidence_threshold: Minimum confidence to trust OCR (default: 0.7)

    Returns:
        Merged constraints dict
    """
    merged = {}

    # Start with high-confidence OCR constraints
    for region_letter, (constraint, confidence) in ocr_constraints.items():
        if confidence >= confidence_threshold:
            merged[region_letter] = constraint

    # Override with user-provided constraints
    merged.update(user_constraints)

    return merged
