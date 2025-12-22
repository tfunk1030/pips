# E2E Verification Guide for Enhanced AI Visual Pips Board Extraction

This document provides step-by-step verification procedures for the extraction flow improvements.

## Prerequisites

1. Python 3.12+ installed with dependencies:
   ```bash
   cd cv-service
   pip install -r requirements.txt
   ```

2. Node.js 18+ installed for pips-solver:
   ```bash
   cd pips-solver
   npm install
   ```

3. Test puzzle images available:
   - `IMG_2050.png` - Sample NYT Pips puzzle screenshot
   - `IMG_2051.png` - Alternative puzzle screenshot

## Verification Steps

### Step 1: Start CV Service

```bash
cd cv-service
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
INFO:     Started reloader process
```

Verify health check:
```bash
curl http://localhost:8080/health
```
Expected: `{"status":"healthy","service":"pips-cv"}`

### Step 2: Run Automated E2E Tests

```bash
cd cv-service
python test_e2e_extraction.py ../IMG_2050.png
```

Expected output includes:
- `[PASS] health_check: Service is healthy`
- `[PASS] crop_success: Cropped in XXXms`
- `[PASS] grid_confidence: Confidence: XX% (high/medium/low)`
- `[PASS] grid_detection: Detected NxM grid`
- `[PASS] detection_method: Method: adaptive_threshold/canny_edge`
- `[PASS] preprocess_success: Processed in XXXms`
- `[PASS] flow_complete: Full flow completed in XXXms`

### Step 3: Start Expo Dev Server

In a new terminal:
```bash
cd pips-solver
npm run start
```

Expected output:
```
Starting Metro Bundler
› Metro waiting on exp://...
› Press 'a' to open Android, 'i' to open iOS simulator
```

### Step 4: Manual App Testing (iOS Simulator/Android Emulator)

#### Test Flow: Happy Path Extraction

1. **Open app** - Launch in iOS simulator or Android emulator
2. **Navigate to New Puzzle** - Tap "+" or "New Puzzle" button
3. **Select puzzle image** - Choose `IMG_2050.png` from gallery
4. **Trigger AI extraction** - Tap "AI Extract" button
5. **Verify progress indicators**:
   - Stage 1: "Detecting grid dimensions..."
   - Stage 2: "Detecting cells and holes..."
   - Stage 3: "Mapping colored regions..."
   - Stage 4: "Extracting constraints..."
   - Stage 5: "Extracting dominoes..."
   - Each stage shows confidence percentage

6. **Verify extraction results**:
   - Grid dimensions displayed correctly
   - Regions are properly color-coded
   - Constraints are listed with values
   - Dominoes are shown with pip counts

7. **Test manual correction workflow**:
   - In AIVerificationModal, toggle between "Visual" and "Text" views
   - Tap a cell to edit (toggle hole/cell, change region)
   - Tap a domino to cycle pip values
   - Verify "Modified" badge appears on edited items
   - Tap "Reset" to undo changes
   - Tap "Accept" to apply corrections

#### Test Flow: Low Confidence (Subtask 6-2)

This flow verifies that users are properly prompted for verification when extraction confidence is low.

**Confidence Thresholds (from OverlayBuilderScreen.tsx):**
- `LOW_CONFIDENCE_THRESHOLD` = 70% - Shows "Review Recommended" alert
- `STAGE_CONFIDENCE_THRESHOLD` = 60% - Per-stage warning displayed
- `CRITICAL_CONFIDENCE_THRESHOLD` = 45% - Shows "Low Confidence Extraction" alert

**Test Steps:**

1. **Prepare a low-quality test image:**
   - Use a blurry or partially cropped puzzle screenshot
   - Or use the test script to create a degraded version:
     ```bash
     cd cv-service
     python test_low_confidence_flow.py ../IMG_2050.png
     ```

2. **Select low-quality image in app:**
   - Open app and navigate to "New Puzzle"
   - Select the degraded/partial image

3. **Trigger AI extraction:**
   - Tap "AI Extract" button
   - Watch the extraction progress

4. **Verify confidence indicators show low values:**
   - During extraction: Stage confidence % displayed in progress
   - After extraction: Overall confidence calculated and analyzed
   - Expected: At least one stage shows < 60% confidence

5. **Verify user is prompted to review/correct:**
   - **Critical confidence (< 45%):**
     - Alert: "Low Confidence Extraction"
     - Message includes confidence percentage and issues detected
     - Single button: "Review Results"
   - **Low confidence (< 70%):**
     - Alert: "Review Recommended"
     - Message shows confidence percentage
     - Lists low-confidence stages (e.g., "Please verify: Grid dimensions, Region mapping")
     - Single button: "Review Now"

6. **Verify AIVerificationModal opens:**
   - Modal displays automatically after alert is dismissed
   - Shows visual grid with region colors
   - Shows confidence bars at bottom (color-coded):
     - Green (>= 90%): High confidence
     - Amber (>= 80%): Medium confidence
     - Red (< 80%): Low confidence
   - Shows "Tap cells to correct errors" instruction

7. **Apply corrections and verify they persist:**
   - **Test cell editing:**
     - Tap a cell in the visual grid
     - Cell Edit Panel appears with options:
       - "Convert to Cell" / "Mark as Hole" toggle
       - Region letter buttons (A, B, C, etc.) to change region
     - Make a change and verify "Modified" badge appears in header

   - **Test domino editing:**
     - Scroll to Dominoes section
     - Tap a domino half to cycle pip value (0->1->2...->6->0)
     - Long-press to cycle backwards (6->5->4...->0->6)
     - Verify the pip dots update visually

   - **Test constraint editing:**
     - Tap a constraint in the Constraints section
     - Constraint Edit Panel appears with:
       - Type selection (Sum / All Equal)
       - Operator buttons (==, <, >, !=) for sum type
       - Value input field
       - Delete and Save buttons
     - Change value and tap Save

   - **Verify Reset functionality:**
     - Tap "Reset" button in header
     - All changes should revert to original AI extraction
     - "Modified" badge disappears

   - **Accept with corrections:**
     - Make some edits again
     - Tap "Accept with edits" button (changes to brass color when edited)
     - Verify app moves to Step 2 (Region Painting)
     - Navigate back to Step 1 and verify corrections are applied

**Automated Testing:**

Run the dedicated low-confidence flow test:
```bash
cd cv-service
python test_low_confidence_flow.py ../IMG_2050.png
```

This tests:
- Low confidence detection from degraded images
- Warning generation for uncertain extractions
- Partial image handling
- Confidence level classification (high/medium/low)
- Preprocessing quality assessment
- Correction data completeness

**Expected Test Output:**
```
[PASS] original_confidence: Original: XX% (high/medium)
[PASS] degraded_confidence: Degraded: XX% (medium/low)
[PASS] warning_generation: N warnings: ...
[PASS] actionable_warnings: Warnings include user-actionable guidance
[PASS] partial_image_handling: Handled gracefully
[PASS] confidence_classification: Classified correctly
[PASS] quality_metrics: Quality metrics available
[PASS] correction_data: All data for corrections available
```

### Step 5: Verify CV Service Endpoints

#### Test /crop-puzzle with enhanced response:

```bash
cd cv-service
python test_crop_endpoint.py ../IMG_2050.png
```

Verify response includes:
- `grid_confidence`: 0-1 float
- `confidence_level`: "high", "medium", "low", or "unknown"
- `detected_rows`: integer (if detected)
- `detected_cols`: integer (if detected)
- `detection_method`: "adaptive_threshold", "canny_edge", or "saturation_fallback"
- `warnings`: array of user-facing warnings

#### Test /preprocess-image:

```bash
# Using curl
curl -X POST http://localhost:8080/preprocess-image \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_image>", "normalize_contrast": true}'
```

Verify response includes:
- `original_stats`: brightness, contrast, dynamic_range, color_balance
- `processed_stats`: same fields as original_stats
- `operations_applied`: list of preprocessing operations

### Step 6: Verify Integration Points

#### CV-AI Integration:

1. Check that `preprocessForAIExtractionWithFallback()` in `cvExtraction.ts`:
   - Calls `/crop-puzzle` first
   - Calls `/preprocess-image` on cropped image
   - Falls back gracefully if CV service unavailable
   - Returns grid detection info for AI hints

2. Check that `extractPuzzle()` in `pipeline.ts`:
   - Reports progress with confidence for each stage
   - Validates outputs between stages
   - Generates review hints based on confidence

#### Frontend Integration:

1. Verify `OverlayBuilderScreen.tsx`:
   - Shows per-stage confidence during extraction
   - Displays warnings for low confidence
   - Triggers review prompt when confidence < 70%

2. Verify `AIVerificationModal.tsx`:
   - Shows visual diff with grid overlay
   - Allows cell-by-cell correction
   - Shows confidence bars for each section
   - Passes edited results to parent on accept

## Success Criteria

| Check | Expected Outcome |
|-------|------------------|
| CV health check | Returns 200 OK |
| Grid cropping | Successfully crops puzzle region |
| Grid confidence | Returns confidence 0-1 with level |
| Image preprocessing | Applies CLAHE, white balance, brightness normalization |
| Extraction pipeline | Completes all 5 stages with confidence |
| Progress indicators | Shows stage progress and confidence in app |
| Low confidence warning | Prompts user when confidence < 70% |
| Critical confidence warning | Shows strong warning when confidence < 45% |
| Low-confidence stages listed | Alert lists specific stages with low confidence |
| Manual correction | Can edit cells, regions, constraints, dominoes |
| Corrections persist | Edited values passed to builder state on Accept |
| Reset functionality | Reset button reverts all edits to original |
| Modified badge | "Modified" badge shows when edits made |
| Accept with edits | Button text changes and passes edited data |
| Fallback behavior | Uses raw image when CV service unavailable |

## Troubleshooting

### CV Service Won't Start

1. Check dependencies: `pip list | grep -E "fastapi|opencv|numpy"`
2. Check port not in use: `lsof -i :8080`
3. Check import errors: `python -c "import main"`

### Expo Won't Build

1. Clear cache: `npx expo start -c`
2. Reinstall dependencies: `rm -rf node_modules && npm install`
3. Check TypeScript: Note that bare `npx tsc --noEmit` produces Expo-related type errors - this is expected

### Extraction Fails

1. Check API keys in environment variables
2. Check CV service is running: `curl localhost:8080/health`
3. Check image is valid base64 PNG/JPG
4. Review console logs for specific error messages

## Files Modified in This Feature

### pips-solver

| File | Changes |
|------|---------|
| `src/services/extraction/stages/gridGeometry.ts` | Improved prompts with visual examples |
| `src/services/extraction/stages/cellDetection.ts` | Enhanced cell vs hole detection |
| `src/services/extraction/stages/regionMapping.ts` | Better color distinction |
| `src/services/extraction/stages/constraintExtraction.ts` | Clearer diamond label reading |
| `src/services/extraction/stages/dominoExtraction.ts` | Pip counting visual patterns |
| `src/services/extraction/validation/gridValidator.ts` | Cross-validation checks |
| `src/services/extraction/pipeline.ts` | Inter-stage validation, confidence aggregation |
| `src/services/cvExtraction.ts` | CV preprocessing, fallback logic |
| `src/app/screens/OverlayBuilderScreen.tsx` | Per-stage confidence, low-confidence warnings |
| `src/app/components/AIVerificationModal.tsx` | Visual diff, cell-by-cell correction |

### cv-service

| File | Changes |
|------|---------|
| `main.py` | Enhanced /crop-puzzle, new /preprocess-image endpoint |
| `hybrid_extraction.py` | Adaptive thresholding, grid line detection |

## Automated Test Scripts

Run the complete E2E test suite:

```bash
# Terminal 1: Start CV service
cd cv-service
uvicorn main:app --host 0.0.0.0 --port 8080

# Terminal 2: Run general E2E tests
cd cv-service
python test_e2e_extraction.py ../IMG_2050.png

# Terminal 3: Run low-confidence flow tests (Subtask 6-2)
cd cv-service
python test_low_confidence_flow.py ../IMG_2050.png
```

### test_e2e_extraction.py

The general E2E test script verifies:
1. Health check endpoint
2. Enhanced crop-puzzle with grid confidence
3. Image preprocessing with statistics
4. Full extraction flow (crop -> preprocess)
5. Domino region cropping

### test_low_confidence_flow.py (Subtask 6-2)

The low-confidence flow test script verifies:
1. Low confidence detection from degraded/partial images
2. Appropriate warning generation for uncertain extractions
3. Graceful handling of partial images
4. Correct confidence level classification (high/medium/low/unknown)
5. Quality assessment metrics in preprocessing
6. Completeness of data required for correction UI

**Optional dependencies for full testing:**
```bash
pip install numpy scipy pillow
```
These enable synthetic image degradation for more thorough low-confidence scenario testing.
