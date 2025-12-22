# Specification: Improved Region Detection for Complex Layouts

## Overview

This task enhances the region detection algorithms in the cv-service to handle complex puzzle layouts with irregular shapes, overlapping colors, and non-standard grid patterns. The current implementation fails on challenging puzzles exactly when users need help most. This enhancement targets 85%+ accuracy for region contour identification across diverse puzzle types, with specific focus on grid dimension estimation for irregular layouts, robust color segmentation, and graceful fallback strategies.

## Workflow Type

**Type**: feature

**Rationale**: This is a feature enhancement to existing computer vision capabilities. While the region detection system exists, we're adding new algorithmic approaches to handle previously unsupported edge cases (irregular shapes, color gradients, camera distortions). This expands the functional scope rather than fixing a bug or refactoring existing code.

## Task Scope

### Services Involved
- **cv-service** (primary) - Contains all region detection, color segmentation, and grid estimation logic

### This Task Will:
- [ ] Improve color segmentation to handle gradients and similar/overlapping colors
- [ ] Enhance grid dimension estimation for irregular and non-standard layouts
- [ ] Increase region contour identification accuracy to 85%+ across puzzle types
- [ ] Implement fallback strategies when primary detection methods fail
- [ ] Add robustness to handle image distortions from mobile camera angles
- [ ] Integrate advanced OpenCV techniques (watershed, DBSCAN, MeanShift clustering)

### Out of Scope:
- UI/UX changes in the pips-solver frontend
- API endpoint modifications or new route creation
- Changes to pips-agent service
- Complete rewrite of existing detection pipeline
- Real-time video processing
- OCR or text extraction improvements

## Service Context

### cv-service

**Tech Stack:**
- Language: Python
- Framework: FastAPI
- Core Libraries: OpenCV (opencv-python), NumPy, scikit-learn
- Package Manager: pip

**Entry Point:** `main.py`

**How to Run:**
```bash
cd cv-service
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```

**Port:** 8080

**Key Directories:**
- Root contains computer vision algorithms
- Dockerfile for containerized deployment

**API Endpoints:**
- `/extract-geometry` (POST) - Extracts puzzle geometry
- `/crop-puzzle` (POST) - Crops puzzle from image
- `/preprocess-image` (POST) - Image preprocessing
- `/health` (GET) - Health check

## Files to Modify

| File | Service | What to Change |
|------|---------|---------------|
| `cv-service/screenshot_to_regions.py` | cv-service | Replace k-means with adaptive clustering (DBSCAN/MeanShift) for automatic region count detection; enhance LAB color space processing with `pyrMeanShiftFiltering` for edge preservation |
| `cv-service/cells_to_regions.py` | cv-service | Replace KMeans with adaptive clustering algorithms; add gradient handling; implement color similarity thresholds |
| `cv-service/cv_extraction_v2.py` | cv-service | Enhance contour detection with polygon approximation (`approxPolyDP`), convex hull analysis, and watershed algorithm for overlapping regions |
| `cv-service/hybrid_extraction.py` | cv-service | Add perspective correction for distorted images; improve adaptive thresholding parameters; implement grid line detection with Hough transforms |

## Files to Reference

These files show patterns to follow:

| File | Pattern to Copy |
|------|----------------|
| `cv-service/screenshot_to_regions.py` | LAB color space conversion pattern; existing clustering structure |
| `cv-service/hybrid_extraction.py` | Adaptive thresholding implementation; error handling patterns |
| `cv-service/cv_extraction_v2.py` | Contour filtering by area/aspect ratio; OpenCV function usage patterns |
| `cv-service/cells_to_regions.py` | KMeans clustering structure (template for DBSCAN/MeanShift replacement) |

## Patterns to Follow

### LAB Color Space Conversion

From `screenshot_to_regions.py`:

```python
# Convert to LAB color space (perceptually uniform)
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
```

**Key Points:**
- LAB color space is optimal for color similarity comparisons
- Always convert from BGR (OpenCV default) not RGB
- LAB is perceptually uniform - euclidean distance matches human perception

### Adaptive Thresholding

From `hybrid_extraction.py`:

```python
# Adaptive thresholding for varying lighting conditions
adaptive_thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, blockSize, C
)
```

**Key Points:**
- Use adaptive over global thresholding for robustness
- GAUSSIAN_C handles varying illumination better than MEAN_C
- Block size must be odd number

### Contour Area Filtering

From `cv_extraction_v2.py`:

```python
# Filter contours by area and aspect ratio
contours = [cnt for cnt in contours
            if cv2.contourArea(cnt) > min_area
            and aspect_ratio_valid(cnt)]
```

**Key Points:**
- Filter noise by minimum area threshold
- Validate shape characteristics (aspect ratio, solidity)
- Sort contours by area for hierarchical processing

## Requirements

### Functional Requirements

1. **Adaptive Color Segmentation**
   - Description: Automatically determine number of color regions without pre-specifying cluster count; handle gradients and similar colors
   - Acceptance: Color segmentation handles gradients and similar colors across 85%+ of test puzzles
   - Implementation: Replace k-means with DBSCAN or MeanShift clustering; add `pyrMeanShiftFiltering` preprocessing

2. **Irregular Grid Dimension Estimation**
   - Description: Accurately detect grid dimensions even with non-standard patterns, irregular cell sizes, and partial grids
   - Acceptance: Grid dimensions accurately estimated for irregular layouts in 85%+ of cases
   - Implementation: Hough line detection + RANSAC-style robust fitting; histogram analysis as fallback; handle non-rectangular grids

3. **Enhanced Contour Identification**
   - Description: Precisely identify region boundaries for irregular shapes, concave regions, and overlapping areas
   - Acceptance: Region contours correctly identified for 85%+ of puzzles
   - Implementation: Polygon approximation with `approxPolyDP`; convex hull for concave regions; watershed algorithm for separating merged regions

4. **Distortion Handling**
   - Description: Correct for perspective distortions from mobile camera angles
   - Acceptance: Region detection works even with slight image distortions from camera angles
   - Implementation: Perspective transform detection and correction using `getPerspectiveTransform` + `warpPerspective`

5. **Fallback Strategies**
   - Description: Gracefully degrade when primary detection fails; try alternative methods before failing
   - Acceptance: Fallback strategies trigger when primary detection fails
   - Implementation: Multi-stage pipeline with confidence scoring; fallback from DBSCAN→MeanShift→k-means; grid estimation fallback from Hough→histogram→manual

### Edge Cases

1. **Monochromatic or near-monochromatic puzzles** - Use edge detection and contour analysis when color segmentation insufficient
2. **Extreme perspective distortion** - Pre-flight validation of image quality; request re-capture if distortion exceeds threshold
3. **Partial puzzle visibility** - Grid estimation should work with incomplete grids; extrapolate from visible structure
4. **Variable lighting conditions** - Adaptive preprocessing with CLAHE (Contrast Limited Adaptive Histogram Equalization)
5. **Overlapping region colors** - Watershed algorithm to separate touching regions; use gradient analysis at boundaries
6. **Non-grid layouts (Tetris-style)** - Disable grid estimation; rely purely on contour and color segmentation

## Implementation Notes

### DO
- Follow the LAB color space pattern from `screenshot_to_regions.py` for all color operations
- Reuse existing adaptive thresholding structure from `hybrid_extraction.py`
- Use area filtering pattern from `cv_extraction_v2.py` for noise reduction
- Add confidence scores to all detection results for fallback logic
- Preserve existing API contracts - internal algorithm changes only
- Use `cv2.pyrMeanShiftFiltering()` before clustering for edge-preserving smoothing
- Set `random_state` in clustering algorithms for reproducible results
- Validate all `cv2.imread()` results (returns None on failure)
- Consider image downscaling for initial analysis, refine on full resolution

### DON'T
- Change any existing API endpoint signatures or response formats
- Remove existing k-means implementation (keep as fallback)
- Assume RGB color order (OpenCV uses BGR)
- Use even numbers for kernel sizes in morphological operations
- Create new dependencies beyond: opencv-python, numpy, scikit-learn, scipy
- Hardcode cluster counts or grid dimensions
- Skip error handling for file I/O operations
- Mix NumPy `[y, x]` and OpenCV `(x, y)` coordinate conventions

## Development Environment

### Start Services

```bash
# Start cv-service
cd cv-service
pip install -r requirements.txt
uvicorn main:app --reload --port 8080

# Optional: Start frontend for integration testing
cd pips-solver
yarn install
yarn start  # Runs on port 3000
```

### Service URLs
- cv-service: http://localhost:8080
- cv-service docs: http://localhost:8080/docs
- pips-solver (optional): http://localhost:3000

### Required Environment Variables
None required for cv-service core functionality. Optional:
- `DEBUG_OUTPUT_DIR`: Directory for saving debug images (cv-service will create temp directory if not set)

### Testing Complex Puzzles
Create test dataset with:
- Irregular/non-rectangular regions
- Similar color regions (gradients, overlapping colors)
- Mobile camera captures with perspective distortion
- Varying lighting conditions
- Non-standard grid patterns

## Success Criteria

The task is complete when:

1. [ ] Region contours correctly identified for 85%+ of test puzzles (including complex layouts)
2. [ ] Color segmentation handles gradients and similar colors (validated with gradient test images)
3. [ ] Grid dimensions accurately estimated for irregular layouts (tested with non-standard grids)
4. [ ] Fallback strategies implemented and trigger appropriately (verified in logs/metrics)
5. [ ] Perspective distortion handling added (tested with angled mobile captures)
6. [ ] No regressions in existing simple puzzle detection (baseline tests pass)
7. [ ] All existing API endpoints return same response format
8. [ ] Code follows existing patterns from reference files
9. [ ] No console errors or unhandled exceptions
10. [ ] Performance acceptable (processing time < 5s per image)

## QA Acceptance Criteria

**CRITICAL**: These criteria must be verified by the QA Agent before sign-off.

### Unit Tests
| Test | File | What to Verify |
|------|------|----------------|
| `test_dbscan_clustering` | `test_region_detection.py` | DBSCAN correctly identifies variable region counts; handles edge cases with min_samples parameter |
| `test_meanshift_clustering` | `test_region_detection.py` | MeanShift automatic bandwidth estimation; converges on complex color gradients |
| `test_perspective_correction` | `test_image_preprocessing.py` | Perspective transform correctly undistorts angled images; handles extreme angles gracefully |
| `test_hough_line_detection` | `test_grid_estimation.py` | Hough line detection identifies grid lines; RANSAC fitting robust to outliers |
| `test_watershed_segmentation` | `test_region_detection.py` | Watershed algorithm separates overlapping regions; marker generation correct |
| `test_fallback_pipeline` | `test_detection_pipeline.py` | Fallback triggers when primary method fails; graceful degradation to k-means |
| `test_color_segmentation_gradients` | `test_color_processing.py` | Color segmentation handles gradients; similar color regions distinguished |

### Integration Tests
| Test | Services | What to Verify |
|------|----------|----------------|
| `test_extract_geometry_complex` | cv-service | POST `/extract-geometry` with complex puzzle images returns accurate region data; 85%+ accuracy on test set |
| `test_distorted_image_pipeline` | cv-service | Perspective-distorted images processed correctly end-to-end; correction applied before detection |
| `test_irregular_grid_estimation` | cv-service | Non-standard grid patterns estimated correctly; dimensions match ground truth |
| `test_api_response_format` | cv-service | All API responses maintain backward compatibility; schema unchanged |

### End-to-End Tests
| Flow | Steps | Expected Outcome |
|------|-------|------------------|
| Complex Puzzle Detection | 1. Upload irregular puzzle image via API 2. Process with enhanced algorithms 3. Validate region contours | 85%+ regions correctly identified; response includes confidence scores |
| Mobile Image Distortion | 1. Upload angled mobile capture 2. Perspective correction applied 3. Region detection runs | Image corrected; regions detected despite initial distortion |
| Gradient Color Handling | 1. Upload puzzle with color gradients 2. DBSCAN/MeanShift clustering 3. Validate segmentation | Similar colors distinguished; gradient regions separated correctly |
| Fallback Strategy | 1. Upload edge case image that fails DBSCAN 2. Observe fallback to MeanShift 3. Final fallback to k-means if needed | Pipeline doesn't fail; attempts all methods; logs fallback path |

### Manual Verification (Computer Vision)
| Scenario | Test Images | Checks |
|----------|-------------|--------|
| Irregular shapes | 5+ puzzles with concave/complex regions | Visual inspection: contours match region boundaries; no merged regions |
| Color gradients | 5+ puzzles with gradients/similar colors | Clustering correctly separates similar colors; no over-segmentation |
| Distorted images | 5+ angled mobile captures | Perspective correction visible; detection accuracy comparable to straight-on images |
| Non-standard grids | 5+ irregular grid patterns | Grid dimensions estimated; cell boundaries identified correctly |

### Performance Verification
| Check | Command/Tool | Expected |
|-------|--------------|----------|
| Processing time | Time API calls with complex images | < 5 seconds per image on average |
| Memory usage | Monitor process during batch processing | No memory leaks; stable memory consumption |
| Algorithm comparison | Log clustering method used per image | DBSCAN/MeanShift preferred; k-means only as fallback |

### Regression Testing
| Test Suite | Purpose | Expected |
|------------|---------|----------|
| `test_simple_puzzles.py` | Existing simple puzzle detection | All baseline tests pass; no accuracy degradation |
| `test_api_endpoints.py` | API contract validation | All endpoints respond correctly; schemas unchanged |
| `test_existing_fixtures.py` | Previously working test images | Same or better results; no regressions |

### QA Sign-off Requirements
- [ ] All unit tests pass (minimum 85% code coverage for new code)
- [ ] All integration tests pass
- [ ] All E2E tests pass with 85%+ accuracy on complex puzzle test set
- [ ] Manual verification complete for all scenarios
- [ ] Performance targets met (< 5s processing time, stable memory)
- [ ] Regression tests pass - no degradation in existing functionality
- [ ] Code follows established patterns from reference files
- [ ] No security vulnerabilities introduced (input validation, file handling)
- [ ] Logging adequate for debugging fallback strategies
- [ ] API documentation updated if any response fields added

### Test Data Requirements
QA must prepare test dataset including:
- 20+ complex puzzle images (irregular shapes, gradients, distortions)
- 10+ baseline simple puzzles (regression testing)
- 5+ edge cases per scenario (gradients, distortions, irregular grids, overlapping colors)
- Ground truth annotations for accuracy measurement

## Implementation Strategy

### Phase 1: Color Segmentation Enhancement
1. Add `pyrMeanShiftFiltering` preprocessing to `screenshot_to_regions.py`
2. Implement DBSCAN clustering as primary method
3. Add MeanShift as secondary fallback
4. Keep k-means as final fallback
5. Add confidence scoring to clustering results

### Phase 2: Grid Estimation Improvements
1. Implement Hough line detection in `hybrid_extraction.py`
2. Add RANSAC-style robust fitting for grid parameters
3. Implement histogram analysis fallback
4. Handle non-rectangular and partial grids

### Phase 3: Contour Enhancement
1. Add polygon approximation with `approxPolyDP` to `cv_extraction_v2.py`
2. Implement convex hull analysis for concave regions
3. Add watershed algorithm for separating merged regions
4. Enhance area/aspect ratio filtering

### Phase 4: Distortion Handling
1. Add perspective correction to `hybrid_extraction.py`
2. Implement image quality validation
3. Add pre-flight distortion detection
4. Request re-capture if distortion exceeds threshold

### Phase 5: Integration & Testing
1. Wire fallback pipeline across all modules
2. Add comprehensive logging and confidence metrics
3. Run accuracy evaluation on test dataset
4. Performance profiling and optimization
5. Documentation updates

## Risk Assessment

### Technical Risks
- **Risk**: DBSCAN/MeanShift may be slower than k-means
  - **Mitigation**: Image downscaling for initial analysis; benchmark performance; fall back if too slow

- **Risk**: Watershed algorithm requires marker generation which may be complex
  - **Mitigation**: Start with distance transform-based markers; iterative refinement

- **Risk**: Perspective correction may introduce artifacts
  - **Mitigation**: Only apply when distortion detected above threshold; validate correction quality

### Accuracy Risks
- **Risk**: 85% accuracy target may not be achievable for all puzzle types
  - **Mitigation**: Prioritize most common complex cases; document known limitations

- **Risk**: Fallback strategies may reduce accuracy
  - **Mitigation**: Add confidence scoring; log which method used for analysis

### Integration Risks
- **Risk**: Changes may break existing API consumers
  - **Mitigation**: Maintain API contract; only change internal algorithms; comprehensive regression testing

## Additional Technical Notes

### OpenCV Version Compatibility
- Target: OpenCV 4.5.0+
- Note: `findContours()` return signature differs between 3.x and 4.x
- Always check OpenCV version: `cv2.__version__`

### Color Space Considerations
- LAB: Best for color similarity (perceptually uniform)
- HSV: Good for hue-based segmentation (not needed here)
- BGR: OpenCV default (always convert to LAB for analysis)

### Clustering Algorithm Selection
- **DBSCAN**: Best for variable cluster counts; requires epsilon and min_samples tuning
- **MeanShift**: Automatic bandwidth estimation; slower but robust
- **k-means**: Fast but requires cluster count; use as fallback only

### Performance Optimization
- Use `MiniBatchKMeans` for large images if k-means needed
- Downscale images for initial analysis (e.g., max dimension 800px)
- Refine detection on full resolution only for selected regions
- Consider GPU acceleration (`cv2.cuda`) if available (optional)

### Debugging Support
- Save intermediate images to `DEBUG_OUTPUT_DIR` if set
- Log clustering method used, confidence scores, fallback triggers
- Include timing information for performance analysis
- Add verbose mode flag for detailed algorithm steps
