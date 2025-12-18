"""
Test CV Extraction v2 on User's Actual Puzzle

Compare detected structure to user-provided correct structure:
- Expected cells: 14
- Expected regions: 7
- Expected shape: ##.##\n.#...\n.....\n#....
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cv_extraction_v2 import extract_puzzle_multi_strategy

# User's correct puzzle structure
EXPECTED = {
    'total_cells': 14,
    'shape': '##.##\n.#...\n.....\n#....',
    'regions': 7,  # A, B, D, E, F, G (+ 3 unconstrained X, Y, Z)
    'constraints': 6,  # A, B, D, E, F, G (excluding unconstrained)
}

def test_user_puzzle():
    """Test CV extraction on user's puzzle image"""

    # Try different possible paths for the uploaded image
    possible_paths = [
        '../.artifacts/image_1734539806728_0.png',
        './.artifacts/image_1734539806728_0.png',
        'C:/Users/tfunk/pips/.artifacts/image_1734539806728_0.png',
        # Fallback to test images if user's not found
        '../IMG_2051.png',
        '../IMG_2050.png',
    ]

    image_path = None
    for path in possible_paths:
        test_path = Path(path)
        if test_path.exists():
            image_path = str(test_path)
            print(f"[INFO] Found image at: {image_path}")
            break

    if not image_path:
        print("[ERROR] Could not find user's puzzle image")
        print("[INFO] Tried paths:")
        for path in possible_paths:
            print(f"  - {path}")
        return

    # Run multi-strategy CV extraction
    print("\n" + "="*60)
    print("Testing CV Extraction v2 on User's Puzzle")
    print("="*60)

    result = extract_puzzle_multi_strategy(
        image_path,
        output_dir='debug/user_puzzle_validation',
        strategies=['region_contours', 'color_segmentation', 'constraint_labels']
    )

    # Display results
    print("\n[DETECTION RESULTS]")
    print(f"Success: {result['success']}")
    print(f"Method used: {result['method_used']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Cells detected: {result['num_cells']}")
    print(f"Grid dimensions: {result['grid_dims']}")
    print(f"Regions detected: {len(result.get('regions', {}))} color clusters")

    # Show all strategy attempts
    print("\n[ALL STRATEGY ATTEMPTS]")
    for attempt in result.get('all_attempts', []):
        status = "SUCCESS" if attempt['success'] else "FAILED"
        print(f"  {attempt['method']:20s} -> {status:7s} "
              f"({attempt['cells_found']} cells, {attempt['confidence']:.1%} confidence)")

    # Compare to expected structure
    print("\n" + "="*60)
    print("VALIDATION AGAINST USER'S CORRECT STRUCTURE")
    print("="*60)

    detected_cells = result['num_cells']
    expected_cells = EXPECTED['total_cells']

    print(f"\nCells:")
    print(f"  Expected: {expected_cells}")
    print(f"  Detected: {detected_cells}")
    print(f"  Difference: {abs(detected_cells - expected_cells)}")

    if detected_cells == expected_cells:
        print(f"  Accuracy: 100% [PERFECT MATCH]")
    else:
        accuracy = max(0, 100 - (abs(detected_cells - expected_cells) / expected_cells * 100))
        print(f"  Accuracy: {accuracy:.1f}%")

    detected_regions = len(result.get('regions', {}))
    expected_regions = EXPECTED['regions']

    print(f"\nRegions:")
    print(f"  Expected: {expected_regions}")
    print(f"  Detected: {detected_regions} color clusters")
    print(f"  Difference: {abs(detected_regions - expected_regions)}")

    if detected_regions == expected_regions:
        print(f"  Accuracy: 100% [PERFECT MATCH]")
    else:
        accuracy = max(0, 100 - (abs(detected_regions - expected_regions) / expected_regions * 100))
        print(f"  Accuracy: {accuracy:.1f}%")

    print(f"\nExpected Shape:")
    for line in EXPECTED['shape'].split('\\n'):
        print(f"  {line}")

    # Overall assessment
    print("\n" + "="*60)
    print("OVERALL ASSESSMENT")
    print("="*60)

    cell_accuracy = max(0, 100 - (abs(detected_cells - expected_cells) / expected_cells * 100))
    region_accuracy = max(0, 100 - (abs(detected_regions - expected_regions) / expected_regions * 100))

    avg_accuracy = (cell_accuracy + region_accuracy) / 2

    print(f"\nCell Detection: {cell_accuracy:.1f}%")
    print(f"Region Detection: {region_accuracy:.1f}%")
    print(f"Average Accuracy: {avg_accuracy:.1f}%")
    print(f"Detection Confidence: {result['confidence']:.1%}")

    if avg_accuracy >= 90:
        print("\n[STATUS] EXCELLENT - Detection is highly accurate")
    elif avg_accuracy >= 70:
        print("\n[STATUS] GOOD - Detection is reasonably accurate")
    elif avg_accuracy >= 50:
        print("\n[STATUS] MODERATE - Significant improvements still needed")
    else:
        print("\n[STATUS] POOR - Detection failed to capture structure")

    # Detailed comparison
    print("\n" + "="*60)
    print("DETAILED COMPARISON")
    print("="*60)

    print("\nWhat v2 improved:")
    print("  - Multi-strategy approach (tries 3 methods)")
    print("  - Region contour detection for irregular grids")
    print("  - Color segmentation for complex layouts")
    print("  - Confidence scoring to pick best result")

    print("\nWhat still needs work:")
    if detected_cells != expected_cells:
        print(f"  - Cell count detection (off by {abs(detected_cells - expected_cells)})")
    if detected_regions != expected_regions:
        print(f"  - Region identification (off by {abs(detected_regions - expected_regions)})")
    print("  - Constraint label -> cell inference (partial)")
    print("  - Domino pip counting (not implemented)")

    print("\n" + "="*60)

if __name__ == '__main__':
    test_user_puzzle()
