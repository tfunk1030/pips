# Specification: Complete Domino Pip Detection

## Overview

This task completes the incomplete domino pip detection implementation in the cv-service to accurately identify pip values (0-6) on domino tiles. Currently marked as incomplete in the codebase, this functionality is critical for the puzzle solver to generate correct solutions. The implementation will use OpenCV computer vision techniques to detect and count pips on each domino half, handling rotated dominoes and various visual styles.

## Workflow Type

**Type**: feature

**Rationale**: This is a feature implementation task that completes partially built functionality. While pip detection infrastructure may exist, the actual detection logic needs to be implemented or significantly enhanced to meet production requirements for accuracy, rotation handling, and visual robustness.

## Task Scope

### Services Involved
- **cv-service** (primary) - Python/FastAPI service responsible for all computer vision operations including domino extraction and pip detection
- **pips-solver** (integration) - React Native mobile app that consumes cv-service API for puzzle extraction workflow

### This Task Will:
- [x] Implement or complete pip detection algorithm to identify 0-6 pips on each domino half
- [x] Add rotation handling to detect pips on dominoes in any orientation
- [x] Implement preprocessing pipeline to handle different domino visual styles and colors
- [x] Create confidence scoring system that accurately reflects detection reliability
- [x] Add error handling for edge cases (blank dominoes, partial dominoes, occlusions)
- [x] Integrate pip detection into existing domino extraction pipeline

### Out of Scope:
- Training custom machine learning models (using classical CV techniques with OpenCV)
- Creating new API endpoints (using existing /crop-dominoes or similar)
- Modifying the puzzle solver logic or constraint handling
- Changes to the mobile app UI or user interaction flow
- Database schema changes or persistence layer modifications

## Service Context

### cv-service

**Tech Stack:**
- Language: Python
- Framework: FastAPI
- Key Libraries: opencv-python (≥4.8.0), numpy (≥1.24.0), pydantic (≥2.0.0)
- Key directories: cv-service/ (root), with main.py entry point

**Entry Point:** `cv-service/main.py`

**How to Run:**
```bash
# Development (local Python)
cd cv-service
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8080

# Production (Docker)
docker build -t cv-service ./cv-service
docker run -p 8080:8080 cv-service
```

**Port:** 8080

**Existing API Routes:**
- POST /extract-geometry - Extracts grid geometry from puzzle image
- POST /crop-puzzle - Crops puzzle region from full image
- POST /crop-dominoes - Crops individual dominoes from puzzle
- POST /preprocess-image - Applies preprocessing to enhance image quality
- GET /health - Health check endpoint

## Files to Modify

| File | Service | What to Change |
|------|---------|---------------|
| `cv-service/extract_dominoes.py` | cv-service | Complete pip detection implementation with robust circle detection, rotation handling, and confidence scoring |
| `cv-service/main.py` | cv-service | Update domino extraction endpoint to return pip values and confidence scores |
| `cv-service/requirements.txt` | cv-service | Verify OpenCV version ≥4.8.0 for latest detection algorithms |

## Files to Reference

These files show patterns to follow:

| File | Pattern to Copy |
|------|----------------|
| `cv-service/hybrid_extraction.py` | OpenCV usage patterns: adaptive thresholding, contour detection, morphological operations |
| `cv-service/main.py` | FastAPI route structure, error handling, response models with Pydantic v2 |
| `cv-service/Dockerfile` | Docker configuration for OpenCV dependencies and CORS setup |

## Patterns to Follow

### OpenCV Image Processing Pipeline

From research phase and `hybrid_extraction.py`:

```python
import cv2
import numpy as np

# 1. Load and convert to grayscale
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Apply adaptive thresholding (handles lighting variations)
binary = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11, 2
)

# 3. Noise reduction with morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 4. Detect circular contours (pips)
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=20,
    param1=50,
    param2=30,
    minRadius=5,
    maxRadius=50
)
```

**Key Points:**
- Always convert BGR to grayscale first (OpenCV loads as BGR, not RGB)
- Use adaptive thresholding instead of fixed thresholds for lighting robustness
- `cv2.HoughCircles()` is more robust than contour area filtering for pip detection
- Apply morphological operations to clean noise before detection

### Rotation Detection and Correction

Pattern for handling rotated dominoes:

```python
# Detect domino orientation
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    # Get minimum area rectangle (handles rotation)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    # Extract rotation angle
    angle = rect[2]

    # Rotate image to straighten domino
    center = rect[0]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
```

**Key Points:**
- Use `cv2.minAreaRect()` to detect rotation angle
- Rotate before splitting domino into halves for accurate pip detection
- Handle edge cases where angle calculation may need normalization

### FastAPI Response Models (Pydantic v2)

From `cv-service/main.py`:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class PipDetectionResult(BaseModel):
    left_pips: int = Field(..., ge=0, le=6, description="Pip count on left half")
    right_pips: int = Field(..., ge=0, le=6, description="Pip count on right half")
    left_confidence: float = Field(..., ge=0.0, le=1.0)
    right_confidence: float = Field(..., ge=0.0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "left_pips": 3,
                "right_pips": 5,
                "left_confidence": 0.92,
                "right_confidence": 0.87
            }
        }
    }
```

**Key Points:**
- Pydantic v2 uses `model_config` instead of `Config` class
- Use `Field()` for validation constraints (pip range 0-6, confidence 0-1)
- Include examples for API documentation

## Requirements

### Functional Requirements

1. **Pip Value Detection (0-6)**
   - Description: Accurately detect and count pips on each domino half, supporting values 0 (blank) through 6
   - Acceptance: Test with sample dominoes covering all values 0-6, verify correct counting with ≥90% accuracy

2. **Rotation Invariance**
   - Description: Detect pips correctly regardless of domino orientation (0-360 degrees)
   - Acceptance: Test with dominoes rotated at various angles (0°, 45°, 90°, 135°, 180°, etc.), verify consistent results

3. **Visual Style Robustness**
   - Description: Handle different domino appearances including colored pips, different backgrounds, varying sizes
   - Acceptance: Test with multiple domino styles (white pips on black, black on white, colored variants), verify detection works across styles

4. **Confidence Scoring**
   - Description: Provide meaningful confidence scores (0.0-1.0) that reflect actual detection reliability
   - Acceptance: Low confidence (<0.7) for ambiguous cases, high confidence (>0.85) for clear detections, scores correlate with actual accuracy

### Edge Cases

1. **Blank Dominoes (0 pips)** - Return 0 with high confidence when no circular contours detected in expected size range
2. **Partial Dominoes** - Return low confidence (<0.5) when domino is cut off or occluded, potentially skip detection
3. **Lighting Variations** - Use adaptive thresholding and CLAHE preprocessing to handle uneven lighting
4. **Overlapping Pips** - Apply morphological operations to separate touching pips, validate pip spacing
5. **Non-standard Pip Arrangements** - Validate detected pip positions match standard domino patterns (e.g., 6 pips should form 2x3 grid)

## Implementation Notes

### DO
- Follow the OpenCV preprocessing pipeline in `hybrid_extraction.py` for consistent image handling
- Use `cv2.HoughCircles()` as primary detection method (more robust than contour filtering alone)
- Implement `cv2.minAreaRect()` for rotation detection before splitting domino halves
- Apply bilateral filtering (`cv2.bilateralFilter()`) for noise reduction while preserving edges
- Use adaptive thresholding (`cv2.adaptiveThreshold()`) instead of fixed thresholds
- Validate pip counts against expected range (0-6) and return low confidence for out-of-range detections
- Add debug output options to save intermediate processing images (use DEBUG_OUTPUT_DIR from env)
- Handle OpenCV 4.x `findContours()` return signature (returns contours, hierarchy)

### DON'T
- Don't use fixed threshold values - lighting conditions vary significantly
- Don't rely solely on contour area filtering - circle detection is more accurate for pip shapes
- Don't skip rotation correction - pip counting fails on heavily rotated dominoes
- Don't create new API endpoints - integrate into existing domino extraction flow
- Don't assume RGB color space - OpenCV loads as BGR by default
- Don't use Pydantic v1 patterns - project uses v2 with breaking changes
- Don't return pip counts without confidence scores - downstream solver needs reliability metrics

## Development Environment

### Start Services

```bash
# cv-service (development)
cd cv-service
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8080

# cv-service (Docker - production mode)
docker build -t cv-service ./cv-service
docker run -p 8080:8080 cv-service

# pips-solver (for testing integration)
cd pips-solver
yarn install
yarn start
```

### Service URLs
- cv-service: http://localhost:8080
- cv-service docs: http://localhost:8080/docs
- pips-solver: http://localhost:3000 (Expo dev server)

### Required Environment Variables
- `DEBUG_OUTPUT_DIR`: Directory path for saving debug images (optional but recommended for development)
- `ANTHROPIC_API_KEY`: Required by pips-agent, not cv-service (optional for this task)

### Testing Pip Detection
```bash
# Test via API
curl -X POST http://localhost:8080/crop-dominoes \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_data", "dominoes": [...]}'

# View API documentation
open http://localhost:8080/docs
```

## Success Criteria

The task is complete when:

1. [x] Pip detection correctly identifies values 0-6 on test dominoes with ≥90% accuracy
2. [x] Detection works on dominoes rotated at various angles (0°, 45°, 90°, 180°, etc.)
3. [x] Handles at least 2 different domino visual styles (e.g., black/white and colored variants)
4. [x] Confidence scores range appropriately (low for ambiguous, high for clear detections)
5. [x] Edge cases handled gracefully (blank dominoes, partial visibility, poor lighting)
6. [x] API returns pip values and confidence scores in structured format
7. [x] No console errors or exceptions during normal operation
8. [x] Existing cv-service tests still pass
9. [x] Integration tested with sample puzzle images end-to-end

## QA Acceptance Criteria

**CRITICAL**: These criteria must be verified by the QA Agent before sign-off.

### Unit Tests
| Test | File | What to Verify |
|------|------|----------------|
| `test_pip_detection_all_values()` | `cv-service/test_extract_dominoes.py` | Correctly detects all pip values 0-6 on standard dominoes |
| `test_pip_detection_rotated()` | `cv-service/test_extract_dominoes.py` | Detection works at 0°, 45°, 90°, 180° rotations |
| `test_pip_confidence_scoring()` | `cv-service/test_extract_dominoes.py` | Confidence scores reflect detection reliability |
| `test_blank_domino()` | `cv-service/test_extract_dominoes.py` | Returns 0 pips with high confidence for blank halves |
| `test_invalid_pip_count()` | `cv-service/test_extract_dominoes.py` | Returns low confidence for detections outside 0-6 range |

### Integration Tests
| Test | Services | What to Verify |
|------|----------|----------------|
| `test_domino_extraction_pipeline()` | cv-service | Full pipeline from image → cropped dominoes → pip values |
| `test_api_response_format()` | cv-service | API returns pip values and confidence scores in correct Pydantic schema |
| `test_multiple_dominoes()` | cv-service | Correctly processes multiple dominoes in single request |

### End-to-End Tests
| Flow | Steps | Expected Outcome |
|------|-------|------------------|
| Full Puzzle Extraction | 1. Upload puzzle image 2. Extract geometry 3. Crop dominoes 4. Detect pips | All dominoes detected with pip values and confidence scores |
| Rotated Puzzle | 1. Upload rotated puzzle image 2. Process through pipeline | Pip detection handles rotation correctly |
| Different Domino Styles | 1. Test with white-on-black dominoes 2. Test with colored dominoes | Detection works across visual styles |

### Manual Verification Checklist
| Check | Command/Action | Expected Result |
|-------|----------------|-----------------|
| Test 0-pip detection | Process blank domino half | Returns 0 with confidence >0.8 |
| Test 6-pip detection | Process domino half with 6 pips | Returns 6 with confidence >0.85 |
| Test rotation handling | Process domino rotated 45° | Same pip count as 0° rotation |
| Test poor lighting | Process underexposed image | Adaptive thresholding compensates, returns reasonable confidence |
| Test API integration | Call /crop-dominoes with test image | Returns JSON with pip_values and confidence fields |
| Debug output | Set DEBUG_OUTPUT_DIR and process image | Intermediate processing images saved to debug directory |

### Docker Verification
| Check | Command | Expected |
|-------|---------|----------|
| Container builds | `docker build -t cv-service ./cv-service` | Build succeeds without errors |
| Container runs | `docker run -p 8080:8080 cv-service` | Service starts, /health returns 200 |
| OpenCV available | Check container logs | No "module not found" errors for cv2 |

### QA Sign-off Requirements
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All E2E tests pass
- [ ] Manual verification checklist complete (all items verified)
- [ ] Docker container builds and runs successfully
- [ ] API returns correct response format per Pydantic schema
- [ ] No regressions in existing cv-service functionality
- [ ] Code follows OpenCV patterns from hybrid_extraction.py
- [ ] No security vulnerabilities introduced (no eval(), pickle, or unsafe deserialization)
- [ ] Debug output disabled in production mode (or gated by environment variable)
- [ ] Confidence scoring correlates with actual detection accuracy (tested empirically)

## Implementation Strategy

### Phase 1: Core Pip Detection Algorithm
1. Implement or enhance pip detection in `extract_dominoes.py`
2. Add preprocessing pipeline (grayscale, adaptive threshold, morphological ops)
3. Implement circle detection using `cv2.HoughCircles()`
4. Add fallback to contour-based detection if circle detection fails
5. Validate pip counts (0-6 range)

### Phase 2: Rotation Handling
1. Add rotation detection using `cv2.minAreaRect()`
2. Implement image rotation correction
3. Split rotated domino into two halves correctly
4. Test with various rotation angles

### Phase 3: Confidence Scoring
1. Define confidence metrics (pip size consistency, circularity, count validation)
2. Implement confidence calculation per domino half
3. Return confidence scores with pip values
4. Test correlation between confidence and accuracy

### Phase 4: Integration & Testing
1. Update API response models to include pip values and confidence
2. Add unit tests covering all pip values and edge cases
3. Create integration tests for full pipeline
4. Manual testing with real puzzle images
5. Docker build verification

### Phase 5: QA & Refinement
1. Address QA feedback
2. Tune detection parameters (circle detection thresholds, contour filters)
3. Add debug output for troubleshooting
4. Final validation against acceptance criteria
