# Pips Agent - Known Limitations

## Computer Vision Extraction Limitations

### Current Issues

Based on real-world testing with user puzzles, the CV extraction has significant limitations:

#### 1. Grid Structure Detection

**Problem:** The current CV extraction only works reliably for simple rectangular grids.

**Example Failure:**
- **Expected:** 14 cells in 4×5 irregular grid
- **Detected:** Only 2 cells
- **Cause:** Edge detection and projection analysis fails with complex layouts

**What Works:**
- Simple rectangular grids with clear boundaries
- High-contrast, uniform cell spacing
- Regular grid patterns

**What Fails:**
- Irregular grid layouts with gaps (e.g., `##.##\n.#...\n.....\n#....`)
- Complex overlapping region colors
- Cells with varied sizes or spacing
- Grids with non-uniform backgrounds

#### 2. Region Detection

**Problem:** K-means color clustering doesn't accurately map to puzzle regions.

**Issues:**
- Detects arbitrary color clusters instead of logical regions
- Can't distinguish between overlapping visual elements
- Misses regions with subtle color differences
- Creates false regions from shadows/gradients

**Example:**
- Puzzle has 7 labeled regions (A-G)
- CV detected 3 arbitrary color clusters
- No correlation between detected clusters and actual puzzle regions

#### 3. Constraint Detection (OCR)

**Status:** Implemented but not fully tested with complex layouts.

**Known Issues:**
- OCR quality depends heavily on image resolution
- Diamond-shaped constraint labels may not be recognized
- Confidence scores need calibration
- Spatial mapping of text to regions is unreliable

#### 4. Domino Tile Detection

**Status:** Not implemented.

**Missing Capability:**
- No detection of domino tiles from the tray at bottom of screenshot
- User must manually list all available dominoes
- Would require:
  - Pip counting from domino images
  - Orientation detection
  - Accurate segmentation of individual dominoes

---

## Workarounds

### Current Best Practice

1. **Take screenshot** of puzzle
2. **Manually create JSON structure** with:
   - Grid dimensions and shape
   - Region mapping
   - Constraints
   - Available dominoes
3. **Provide JSON to agent**
4. **Agent solves accurately**

### Example Manual Structure

```json
{
  "board": {
    "rows": 4,
    "cols": 5,
    "shape": "##.##\\n.#...\\n.....\\n#...."
  },
  "regions": "##B##\\nA#BBC\\nDDDDG\\n#FEEG",
  "constraints": {
    "A": { "type": "sum", "op": ">", "value": 4 }
  },
  "dominoes": [[6,1], [3,3], ...]
}
```

---

## Impact Assessment

| Component | Status | Impact | Workaround Available |
|-----------|--------|--------|---------------------|
| Simple grid detection | ✅ Works | Low | N/A |
| Complex grid detection | ❌ Fails | High | Manual JSON |
| Region clustering | ⚠️ Unreliable | Medium | Manual regions |
| OCR constraints | ⚠️ Untested | Medium | Manual constraints |
| Domino detection | ❌ Missing | High | Manual list |
| Hint generation | ✅ Works | N/A | N/A |
| Puzzle solving | ✅ Works | N/A | N/A |

---

## User Experience Impact

**With API Credits:**
- Interactive agent works well for simple puzzles
- Falls back to asking user for manual input
- Graceful degradation to manual specification

**Without API Credits:**
- Tools can be called directly with manual JSON
- Full solving capability available offline
- Hint generation works perfectly

**Current UX Flow:**
1. User provides screenshot → CV fails
2. Agent asks for clarification → User provides JSON
3. Agent solves perfectly → Success!

**Ideal UX Flow (Not Yet Achieved):**
1. User provides screenshot → CV succeeds
2. Agent confirms structure → User approves
3. Agent solves → Success!

---

## Recommendations

### For Users

**Until CV is improved:**
1. Use manual JSON specification for complex puzzles
2. Test CV with simple rectangular puzzles first
3. Keep screenshots high-resolution and high-contrast
4. Provide feedback on detection failures

### For Developers

See `IMPROVEMENT_PLAN.md` for detailed enhancement roadmap.

---

## Test Cases

### Working Test Case
- **Puzzle:** `pips_puzzle.yaml` (sample 8×6 rectangular grid)
- **CV Result:** ✅ Success
- **Solver Result:** ✅ Correct solution

### Failing Test Case
- **Puzzle:** User's 4×5 irregular grid (Dec 2024)
- **CV Result:** ❌ Detected 2 cells instead of 14
- **Workaround:** ✅ Solved with manual JSON
- **Solver Result:** ✅ Correct solution

---

## Version Info

- **Pips Agent Version:** 1.0.0
- **Last Updated:** 2024-12-18
- **Tested With:** IMG_2050.png, IMG_2051.png, user puzzle screenshots
