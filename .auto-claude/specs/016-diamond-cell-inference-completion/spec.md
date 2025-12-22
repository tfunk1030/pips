# Specification: Diamond Cell Inference Completion

## Overview

Complete the diamond marker detection strategy (Strategy 3) in the multi-strategy grid detection system. The function `detect_by_constraint_labels()` is documented but not implemented. Diamond-shaped markers are used in some Pips puzzles as visual anchors at cell boundaries—this strategy should detect these diamonds and use them to infer the positions and dimensions of the puzzle cells.

## Workflow Type

**Type**: feature

**Rationale**: This implements a documented but incomplete detection strategy that enhances grid detection accuracy for puzzles using diamond markers. It addresses explicit technical debt noted in IMPROVEMENTS_IMPLEMENTED.md and completes the multi-strategy detection system.

## Task Scope

### Services Involved
- **pips-agent** (primary) - Python CLI agent containing CV extraction utilities

### This Task Will:
- [ ] Implement `detect_by_constraint_labels()` function in `cv_extraction_v2.py`
- [ ] Detect diamond-shaped markers in puzzle images
- [ ] Infer cell positions and dimensions from detected diamond markers
- [ ] Calculate confidence scores for diamond-based detection
- [ ] Integrate diamond detection into `extract_puzzle_multi_strategy()`
- [ ] Add comprehensive test coverage for diamond marker puzzles

### Out of Scope:
- Other marker shapes (circles, squares, dots)
- Domino pip counting
- OCR constraint text detection
- UI for manual correction
- Changes to cv-service or pips-solver services

## Service Context

### pips-agent

**Tech Stack:**
- Language: Python 3.x
- Framework: None (CLI tool using Claude Agent SDK)
- Key libraries: OpenCV, NumPy, scikit-learn

**Key Directories:**
- `utils/` - Computer vision utilities (CV extraction logic lives here)
- `tools/` - MCP tools that expose agent capabilities

**Entry Point:** `main.py`

**How to Run:**
```bash
cd pips-agent
python main.py
```

**Key Dependencies:**
- `opencv-python` - Image processing and computer vision
- `numpy` - Array operations
- `scikit-learn` - Clustering for region detection

## Files to Modify

| File | Service | What to Change |
|------|---------|---------------|
| `pips-agent/utils/cv_extraction_v2.py` | pips-agent | Add `detect_by_constraint_labels()` function implementing diamond detection and cell inference |
| `pips-agent/utils/cv_extraction_v2.py` | pips-agent | Update `extract_puzzle_multi_strategy()` to include "constraint_labels" strategy |

## Files to Reference

These files show patterns to follow:

| File | Pattern to Copy |
|------|----------------|
| `pips-agent/utils/cv_extraction_v2.py` (lines 32-122) | `detect_by_region_contours()` - function structure, error handling, DetectionResult return |
| `pips-agent/utils/cv_extraction_v2.py` (lines 125-202) | `detect_by_color_segmentation()` - OpenCV operations, confidence calculation, debug output |
| `pips-agent/utils/cv_extraction_v2.py` (lines 273-341) | Helper functions: `estimate_grid_dims()`, `detect_regions_from_cells()` - reuse these |
| `pips-agent/utils/cv_extraction_v2.py` (lines 343-391) | `calculate_confidence()` - confidence scoring pattern |
| `pips-agent/utils/cv_extraction_v2.py` (lines 205-270) | `extract_puzzle_multi_strategy()` - how strategies are integrated |
| `pips-agent/test_user_puzzle_cv.py` | Test patterns for validating detection results |
| `pips-agent/IMPROVEMENTS_IMPLEMENTED.md` (lines 22-25) | Expected behavior documentation |

## Patterns to Follow

### Pattern 1: Detection Strategy Function Structure

From `detect_by_region_contours()` and `detect_by_color_segmentation()`:

```python
def detect_by_constraint_labels(image_path: str, debug_dir: str = None) -> DetectionResult:
    """
    Strategy 3: Detect cells by finding diamond-shaped constraint markers.

    Works for puzzles that use diamond markers at cell boundaries.

    Steps:
    1. Load image and preprocess
    2. Detect diamond shapes using contour analysis
    3. Infer cell grid from diamond positions
    4. Calculate cell bounding boxes
    5. Detect regions from inferred cells
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="constraint_labels",
                error="Could not read image"
            )

        # ... diamond detection logic ...

        if len(cells) == 0:
            return DetectionResult(
                success=False, cells=[], grid_dims=None, regions=None,
                confidence=0.0, method="constraint_labels",
                error="No cells inferred from markers"
            )

        grid_dims = estimate_grid_dims(cells)
        regions = detect_regions_from_cells(img, cells)
        confidence = calculate_confidence(cells, grid_dims)

        if debug_dir:
            save_debug_image(img, cells, f"{debug_dir}/constraint_labels_method.png")

        return DetectionResult(
            success=True,
            cells=cells,
            grid_dims=grid_dims,
            regions=regions,
            confidence=confidence,
            method="constraint_labels"
        )

    except Exception as e:
        return DetectionResult(
            success=False, cells=[], grid_dims=None, regions=None,
            confidence=0.0, method="constraint_labels",
            error=str(e)
        )
```

**Key Points:**
- Must return `DetectionResult` dataclass
- Handle image loading errors
- Use try/except for robustness
- Reuse helper functions: `estimate_grid_dims()`, `detect_regions_from_cells()`, `calculate_confidence()`
- Support optional debug output
- Set `method="constraint_labels"` for tracking

### Pattern 2: Integration into Multi-Strategy System

From `extract_puzzle_multi_strategy()`:

```python
# In extract_puzzle_multi_strategy():
if strategies is None:
    strategies = ["region_contours", "color_segmentation", "constraint_labels"]  # Add constraint_labels

# Try each strategy
if "region_contours" in strategies:
    result = detect_by_region_contours(image_path, output_dir)
    results.append(result)

if "color_segmentation" in strategies:
    result = detect_by_color_segmentation(image_path, output_dir)
    results.append(result)

if "constraint_labels" in strategies:
    result = detect_by_constraint_labels(image_path, output_dir)  # Add this block
    results.append(result)
```

**Key Points:**
- Add to default strategies list
- Follow same pattern as other strategies
- Results are automatically compared by confidence

### Pattern 3: Diamond Shape Detection

Diamond detection should use OpenCV contour analysis:

```python
# Pseudocode for diamond detection:
# 1. Convert to grayscale and threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 2. Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 3. Filter for diamond shapes
diamonds = []
for contour in contours:
    # Approximate polygon
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Diamonds typically have 4 vertices
    if len(approx) == 4:
        # Check if it's roughly diamond-shaped (square rotated 45°)
        # by analyzing vertex angles and aspect ratio
        diamonds.append(approx)

# 4. Infer grid from diamond positions
# Diamonds mark cell corners/boundaries
# Calculate cell positions from diamond layout
```

## Requirements

### Functional Requirements

1. **Diamond Marker Detection**
   - Description: Detect diamond-shaped markers in puzzle images using contour analysis
   - Acceptance: Function identifies diamonds with 4 vertices, appropriate size (avoid noise), and diamond-like geometry (aspect ratio ~1.0, rotated ~45°)

2. **Cell Position Inference**
   - Description: Calculate cell bounding boxes (x, y, w, h) from detected diamond positions
   - Acceptance: Inferred cells align with actual puzzle cells visible in image. Works when diamonds mark cell corners or cell centers.

3. **Robust Handling of Variations**
   - Description: Handle different diamond sizes, orientations, and layouts
   - Acceptance: Works with small/large diamonds, various rotation angles, regular/irregular grids

4. **Confidence Scoring**
   - Description: Calculate confidence score (0.0-1.0) for detection quality
   - Acceptance: Higher confidence when cell count is reasonable (7-30), sizes are consistent, and arrangement is grid-like

5. **Multi-Strategy Integration**
   - Description: Integrate seamlessly into `extract_puzzle_multi_strategy()`
   - Acceptance: Strategy runs alongside others, results are compared, best result is returned based on confidence

### Edge Cases

1. **No Diamonds Detected** - Return `DetectionResult` with `success=False` and descriptive error message
2. **Too Few Diamonds** - If <4 diamonds, cannot reliably infer grid. Return failure with appropriate error.
3. **Irregular Diamond Spacing** - Calculate cell dimensions from median diamond spacing to handle outliers
4. **Diamonds at Cell Centers vs Corners** - Detect layout pattern and adjust cell inference logic accordingly
5. **Partial Diamond Detection** - If some diamonds obscured/missing, attempt inference from detected subset with reduced confidence
6. **Background Noise** - Filter contours by size and shape criteria to avoid false positives

## Implementation Notes

### Algorithm Approach

The diamond detection should follow this high-level approach:

1. **Preprocessing**: Convert to grayscale, apply thresholding/edge detection
2. **Contour Detection**: Find all contours in the image
3. **Shape Filtering**: Filter for 4-vertex polygons with diamond-like properties
4. **Position Analysis**: Determine if diamonds mark corners or centers
5. **Grid Inference**: Calculate regular grid spacing from diamond positions
6. **Cell Calculation**: Compute cell bounding boxes based on inferred grid
7. **Region Detection**: Use existing `detect_regions_from_cells()` helper
8. **Confidence Calculation**: Use existing `calculate_confidence()` helper

### DO
- Follow the exact function signature pattern of other detection strategies
- Reuse existing helper functions (`estimate_grid_dims`, `detect_regions_from_cells`, `calculate_confidence`, `save_debug_image`)
- Return `DetectionResult` dataclass consistently
- Handle errors gracefully with try/except
- Provide descriptive error messages
- Support debug output when `debug_dir` is provided
- Test with various diamond sizes and orientations
- Filter contours by area and shape criteria to reduce false positives

### DON'T
- Create new data structures (use `DetectionResult`)
- Duplicate confidence/region detection logic (reuse helpers)
- Skip error handling (every code path should be safe)
- Assume diamond positions (detect whether they mark corners or centers)
- Ignore debug output option (helpful for troubleshooting)
- Hardcode thresholds without testing on real images

### Diamond Detection Hints

- Use `cv2.approxPolyDP()` with epsilon ~4% of contour perimeter to approximate polygons
- Check `len(approx) == 4` for quadrilaterals
- Calculate aspect ratio of bounding box: `w/h` should be ~0.8-1.2 for diamonds
- Consider rotation: Diamonds are squares rotated ~45°. Check vertex angles.
- Filter by area: Typical diamond markers are 100-10,000 pixels (adjust based on image size)
- Sort diamonds by position (x, y) to infer grid structure
- Use median spacing between diamonds to calculate cell size (robust to outliers)

## Development Environment

### Start Services

pips-agent is a standalone Python script:

```bash
# Run pips-agent
cd pips-agent
python main.py
```

For testing:

```bash
# Run test script
cd pips-agent
python test_user_puzzle_cv.py
```

### Service URLs
- pips-agent: Local CLI tool (no HTTP server)
- cv-service: http://localhost:8080 (separate service, not modified in this task)

### Required Environment Variables
- `ANTHROPIC_API_KEY`: API key for Claude (for pips-agent's main functionality, not for CV)
- `DEBUG_OUTPUT_DIR`: Directory for debug images (default: `pips-agent/debug`)

## Success Criteria

The task is complete when:

1. [ ] `detect_by_constraint_labels()` function is implemented in `cv_extraction_v2.py`
2. [ ] Function detects diamond-shaped markers using OpenCV contour analysis
3. [ ] Function infers cell positions from diamond locations
4. [ ] Function returns `DetectionResult` with cells, grid_dims, regions, confidence
5. [ ] Integration added to `extract_puzzle_multi_strategy()` with "constraint_labels" strategy
6. [ ] Strategy runs alongside existing strategies and competes based on confidence
7. [ ] Test coverage added (follow `test_user_puzzle_cv.py` patterns)
8. [ ] Manual testing shows diamond detection works with sample diamond-marked puzzle images
9. [ ] No console errors or exceptions
10. [ ] Existing tests still pass (no regressions)
11. [ ] Code follows existing style (function structure, error handling, helper reuse)
12. [ ] Debug output is saved when `debug_dir` is provided

## QA Acceptance Criteria

**CRITICAL**: These criteria must be verified by the QA Agent before sign-off.

### Unit Tests
| Test | File | What to Verify |
|------|------|----------------|
| test_detect_by_constraint_labels_basic | `pips-agent/tests/test_cv_extraction_v2.py` (new) | Function returns DetectionResult with correct structure |
| test_detect_by_constraint_labels_with_diamonds | `pips-agent/tests/test_cv_extraction_v2.py` (new) | Detects diamond markers in test image |
| test_detect_by_constraint_labels_no_diamonds | `pips-agent/tests/test_cv_extraction_v2.py` (new) | Returns failure when no diamonds present |
| test_detect_by_constraint_labels_inference | `pips-agent/tests/test_cv_extraction_v2.py` (new) | Correctly infers cell positions from diamonds |
| test_multi_strategy_includes_constraint_labels | `pips-agent/tests/test_cv_extraction_v2.py` (new) | Strategy is included and executed in multi-strategy detection |

### Integration Tests
| Test | Services | What to Verify |
|------|----------|----------------|
| test_extract_puzzle_multi_strategy_with_diamonds | pips-agent | `extract_puzzle_multi_strategy()` returns best result when constraint_labels succeeds |
| test_constraint_labels_vs_other_strategies | pips-agent | Confidence scoring correctly ranks strategies for diamond-marked puzzles |

### Manual Testing
| Test Case | Steps | Expected Outcome |
|-----------|-------|------------------|
| Diamond-marked puzzle detection | 1. Create/find puzzle image with diamond markers 2. Run `extract_puzzle_multi_strategy()` with "constraint_labels" strategy 3. Inspect debug output | Diamonds detected, cells inferred, confidence score reasonable (>0.3) |
| Integration with existing strategies | 1. Run with all strategies enabled 2. Check which strategy wins | Best strategy selected based on confidence |
| Error handling | 1. Run on image without diamonds 2. Check return value | Returns `DetectionResult` with `success=False` and descriptive error |

### Code Quality Verification
| Check | Tool/Method | Expected |
|-------|-------------|----------|
| Function signature matches pattern | Manual inspection | Same params/return as `detect_by_region_contours()` |
| Helper functions reused | Manual inspection | Uses `estimate_grid_dims()`, `detect_regions_from_cells()`, `calculate_confidence()` |
| Error handling present | Manual inspection | Try/except wraps main logic |
| Debug output supported | Manual inspection | Saves image when `debug_dir` provided |
| No hardcoded paths | Code review | All paths relative or passed as parameters |

### Regression Testing
| Test | Command | Expected |
|------|---------|----------|
| Existing strategies unaffected | `python test_user_puzzle_cv.py` | Test still passes, region_contours and color_segmentation work as before |
| No import errors | `python -c "from utils.cv_extraction_v2 import detect_by_constraint_labels"` | Imports successfully |

### QA Sign-off Requirements
- [ ] All unit tests pass (5 new tests)
- [ ] Integration test passes (multi-strategy includes constraint_labels)
- [ ] Manual testing confirms diamond detection works
- [ ] Code quality verified (follows patterns, reuses helpers, handles errors)
- [ ] No regressions in existing detection strategies
- [ ] Function integrated into `extract_puzzle_multi_strategy()` correctly
- [ ] Debug output saves correctly when enabled
- [ ] Confidence scoring produces reasonable values (0.0-1.0)
- [ ] Documentation updated (if needed)

---

## Additional Context

### Current State

**File**: `pips-agent/utils/cv_extraction_v2.py` (410 lines)

**Existing Strategies**:
- Strategy 1: `detect_by_region_contours()` - Detects cells by finding colored region boundaries
- Strategy 3: `detect_by_color_segmentation()` - Segments by color, finds cells within regions

**Missing**:
- Strategy 3 (documented as constraint label detection): `detect_by_constraint_labels()`

**Documentation Reference**:
`pips-agent/IMPROVEMENTS_IMPLEMENTED.md` lines 22-25:
```
#### Strategy 3: Constraint Label Detection (Partial)
- Detects diamond-shaped markers
- Uses markers as anchors for cell inference
- **Status:** Diamond detection implemented, cell inference TODO
```

**Note**: The documentation says "diamond detection implemented" but the function doesn't exist in the code. This task implements it from scratch.

### Diamond Marker Context

In Pips puzzles, diamond markers can be used as visual separators at:
- **Cell corners**: Diamonds mark the intersections of cell boundaries (grid points)
- **Cell centers**: Diamonds mark the center of each cell

The detection algorithm should:
1. Detect which pattern is present
2. Infer cell positions accordingly

### Testing Resources

**Existing Test**: `pips-agent/test_user_puzzle_cv.py`
- Shows how to test multi-strategy detection
- Validates cell count, region count, confidence
- Compares detected vs expected structure

**New Tests Needed**:
- Create `pips-agent/tests/test_cv_extraction_v2.py` (new file)
- Follow pytest patterns
- Mock/fixture test images with known diamond layouts
- Test both success and failure cases
- Test edge cases (few diamonds, irregular spacing, etc.)

### Performance Considerations

- Diamond detection should complete in <5 seconds for typical puzzle images
- No excessive memory usage (process images in-place where possible)
- Avoid redundant contour detection (run once, filter results)

### Future Enhancements (Out of Scope)

- Detection of other marker shapes (circles, squares)
- Machine learning-based marker detection
- Adaptive thresholding based on image brightness
- Multi-scale detection for varying image resolutions
