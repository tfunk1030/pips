# Specification: Accurate Confidence Scoring

## Overview

Calibrate the confidence scoring system across detection components to accurately reflect actual detection reliability within a 10% margin. Currently, confidence scores are overly conservative (showing ~40% when actual accuracy is higher), undermining user trust and creating confusion about when automated detection can be trusted versus when manual verification is needed. This feature will implement component-specific confidence thresholds and visual indicators to help users make informed decisions about reviewing extractions.

## Workflow Type

**Type**: feature

**Rationale**: This is a new capability that enhances the existing detection system by adding accurate confidence calibration and visual communication. It requires algorithmic changes to scoring logic, UI enhancements, and validation infrastructure - characteristics of a feature implementation rather than a bug fix or refactor.

## Task Scope

### Services Involved
- **cv-service** (primary) - Computer vision detection confidence scoring for geometry extraction, puzzle cropping, and domino detection
- **pips-agent** (primary) - OCR and detection algorithm confidence calculation
- **pips-solver** (integration) - Frontend display of confidence indicators and user communication

### This Task Will:
- [ ] Analyze current confidence calculation algorithms across all detection components
- [ ] Calibrate confidence scores to correlate with actual accuracy within ±10%
- [ ] Implement component-specific confidence thresholds (geometry extraction, OCR, puzzle detection, etc.)
- [ ] Add visual confidence indicators in the UI (pips-solver)
- [ ] Create validation infrastructure to measure actual accuracy vs reported confidence
- [ ] Document confidence score interpretation for users

### Out of Scope:
- Improving underlying detection algorithms (focus is on accurate scoring of existing algorithms)
- Changing the core detection/extraction logic
- Adding new detection features
- Historical data migration of old confidence scores

## Service Context

### cv-service

**Tech Stack:**
- Language: Python
- Framework: FastAPI
- Key directories: Root contains main.py
- Dependencies: opencv-python, numpy, fastapi, uvicorn, pydantic

**Entry Point:** `main.py`

**How to Run:**
```bash
cd cv-service
python main.py
```

**Port:** 8080

**API Endpoints Relevant to Confidence:**
- `/extract-geometry` (POST) - Returns geometry detection results with confidence
- `/crop-puzzle` (POST) - Returns cropped puzzle with detection confidence
- `/crop-dominoes` (POST) - Returns domino detection with confidence
- `/preprocess-image` (POST) - May include preprocessing quality confidence

### pips-agent

**Tech Stack:**
- Language: Python
- Framework: None (Agent/CLI tool)
- Key directories: utils/
- Dependencies: claude-agent-sdk, pytesseract, opencv-python, numpy, scikit-learn

**Entry Point:** `main.py`

**How to Run:**
```bash
cd pips-agent
python main.py
```

**Environment Variables:**
- `ANTHROPIC_API_KEY` (required, sensitive)
- `DEBUG_OUTPUT_DIR` (required)

**Confidence Sources:**
- OCR confidence from pytesseract
- ML model confidence from scikit-learn
- Pattern detection confidence

### pips-solver

**Tech Stack:**
- Language: TypeScript
- Framework: React Native / Expo
- Key directories: src/
- Package manager: yarn

**Entry Point:** `index.ts`

**How to Run:**
```bash
cd pips-solver
npm run start
```

**Port:** 3000

**Confidence Display:**
- Consumes confidence scores from cv-service API
- Consumes confidence scores from pips-agent API
- Displays visual indicators to users

## Files to Modify

### Discovery Required

The context gathering phase did not identify specific files. Implementation must begin with discovery to locate:

| Component | Expected Location | What to Change |
|-----------|-------------------|----------------|
| CV confidence calculation | `cv-service/main.py` or detection modules | Recalibrate confidence scoring algorithms for geometry/puzzle/domino detection |
| OCR confidence calculation | `pips-agent/utils/` or main processing logic | Adjust pytesseract confidence interpretation and thresholds |
| Agent confidence aggregation | `pips-agent/main.py` or processing pipeline | Implement component-specific threshold logic |
| Confidence API response | `cv-service/main.py` API routes | Ensure confidence scores are returned in responses |
| Frontend confidence display | `pips-solver/src/` components | Add visual confidence indicators (colors, badges, warnings) |
| Confidence thresholds config | Config files in cv-service and pips-agent | Define per-component confidence thresholds |

## Files to Reference

### Discovery Required

During implementation, identify and document:

| Pattern Type | Where to Look | What to Learn |
|--------------|---------------|---------------|
| Current confidence calculation | Existing detection code in cv-service | How confidence is currently computed |
| API response format | FastAPI route handlers in cv-service | Where confidence values are returned |
| OCR confidence handling | pytesseract usage in pips-agent | How OCR confidence is extracted and processed |
| ML model confidence | scikit-learn model usage in pips-agent | How model prediction confidence is calculated |
| Frontend API consumption | pips-solver API client code | How confidence data flows from backend to UI |
| UI component patterns | Existing pips-solver components | Design patterns for displaying status/quality indicators |

## Patterns to Follow

### Confidence Score Structure

Expected pattern for API responses:

```python
# FastAPI Response Model (cv-service)
from pydantic import BaseModel

class DetectionResult(BaseModel):
    result: dict
    confidence: float  # 0.0 to 1.0
    confidence_breakdown: dict  # Component-specific scores
    threshold: str  # "high" | "medium" | "low"
```

**Key Points:**
- Use 0.0-1.0 scale consistently across all services
- Include breakdown for transparency
- Provide categorical threshold interpretation

### Component-Specific Thresholds

Pattern for threshold configuration:

```python
# Confidence thresholds config
CONFIDENCE_THRESHOLDS = {
    "geometry_extraction": {
        "high": 0.85,  # User can trust without review
        "medium": 0.70,  # Suggest review
        "low": 0.0  # Requires manual verification
    },
    "ocr_detection": {
        "high": 0.90,  # OCR needs higher bar
        "medium": 0.75,
        "low": 0.0
    },
    "puzzle_detection": {
        "high": 0.80,
        "medium": 0.65,
        "low": 0.0
    }
}
```

**Key Points:**
- Each detection component has its own thresholds
- Thresholds calibrated based on actual accuracy measurement
- OCR typically needs higher confidence threshold due to error modes

### Visual Confidence Indicators (React Native)

Pattern for UI display:

```typescript
// pips-solver confidence component
interface ConfidenceIndicatorProps {
  confidence: number;
  threshold: 'high' | 'medium' | 'low';
  component: string;
}

const ConfidenceIndicator: React.FC<ConfidenceIndicatorProps> = ({
  confidence,
  threshold,
  component
}) => {
  const colors = {
    high: '#10b981',    // Green - trustworthy
    medium: '#f59e0b',  // Amber - review suggested
    low: '#ef4444'      // Red - verification required
  };

  const messages = {
    high: '✓ High confidence - likely accurate',
    medium: '⚠ Medium confidence - review recommended',
    low: '⚠ Low confidence - manual verification required'
  };

  return (
    <View>
      <Text style={{ color: colors[threshold] }}>
        {(confidence * 100).toFixed(0)}% confidence
      </Text>
      <Text>{messages[threshold]}</Text>
    </View>
  );
};
```

**Key Points:**
- Color-coded confidence levels (green/amber/red)
- Clear messaging about what confidence level means
- Display both numeric confidence and categorical interpretation

## Requirements

### Functional Requirements

1. **Accurate Confidence Correlation**
   - Description: Confidence scores must correlate with actual detection accuracy within ±10%
   - Acceptance: Statistical validation shows confidence-accuracy correlation coefficient > 0.9, with mean absolute error < 10%

2. **Component-Specific Thresholds**
   - Description: Different detection components (geometry, OCR, puzzle) have independently calibrated confidence thresholds
   - Acceptance: Each component has documented thresholds in config; thresholds validated against ground truth data

3. **Visual Confidence Communication**
   - Description: UI displays clear visual indicators (color, icons, messages) to communicate confidence levels
   - Acceptance: Users can immediately distinguish high/medium/low confidence detections in the UI

4. **High-Confidence Reliability**
   - Description: Detections marked as "high confidence" are proven reliable in testing
   - Acceptance: High-confidence detections have >90% actual accuracy in test dataset

### Edge Cases

1. **Conflicting Component Confidences** - When multiple detection components disagree (e.g., geometry extraction is high confidence but OCR is low), use the lowest confidence score and clearly indicate which component is uncertain

2. **Confidence Near Threshold Boundaries** - When confidence is within 5% of a threshold boundary (e.g., 82% when high threshold is 85%), add visual indicator that score is "borderline" to prevent over-confidence

3. **Zero/Missing Confidence** - If a detection component fails to produce a confidence score, default to "low" threshold and flag for manual review

4. **Image Quality Impact** - Low image quality may affect confidence; consider adding image quality score as input to confidence calculation

## Implementation Notes

### DO
- Start with data collection: gather ground truth labels for validation dataset
- Measure current accuracy for each detection component before recalibration
- Use statistical methods (precision/recall curves, calibration curves) to set thresholds
- Log confidence scores and actual outcomes for future refinement
- Make thresholds configurable via environment variables or config files
- Document the calibration methodology for future maintenance
- Add confidence score explanations in UI (tooltips, help text)

### DON'T
- Don't guess at confidence values without statistical backing
- Don't use a single threshold for all detection components
- Don't hide low confidence from users - transparency builds trust
- Don't over-inflate scores to "look better" - accuracy is paramount
- Don't skip validation step - must prove confidence correlates with accuracy
- Don't hardcode thresholds - make them adjustable

### Calibration Methodology

1. **Collect Ground Truth Data**: Create labeled validation dataset with correct/incorrect detections
2. **Measure Current Performance**: Run existing system and compare outputs to ground truth
3. **Analyze Confidence Distribution**: Plot confidence scores vs actual accuracy
4. **Identify Calibration Curve**: Determine if current confidence is consistently biased (e.g., always 40% lower than actual)
5. **Adjust Algorithm**: Apply calibration correction (e.g., Platt scaling, isotonic regression)
6. **Set Thresholds**: Use calibration curve to set high/medium/low boundaries
7. **Validate**: Re-test on held-out validation set to confirm ±10% correlation

## Development Environment

### Start Services

```bash
# Terminal 1: Start CV service
cd cv-service
python main.py
# Runs on http://localhost:8080

# Terminal 2: Start frontend (if testing UI)
cd pips-solver
npm run start
# Runs on http://localhost:3000

# Terminal 3: Test pips-agent (as needed)
cd pips-agent
python main.py
```

### Service URLs
- cv-service: http://localhost:8080
- cv-service docs: http://localhost:8080/docs
- pips-solver: http://localhost:3000

### Required Environment Variables
- `ANTHROPIC_API_KEY`: API key for pips-agent (in pips-agent/.env)
- `DEBUG_OUTPUT_DIR`: Debug output directory for pips-agent

### Testing Confidence Scores

```bash
# Example: Test geometry extraction confidence
curl -X POST http://localhost:8080/extract-geometry \
  -F "image=@test-image.jpg" \
  | jq '.confidence'

# Example: Test with multiple images
for img in test-images/*.jpg; do
  echo "Testing $img"
  curl -X POST http://localhost:8080/extract-geometry \
    -F "image=@$img" \
    | jq '.confidence'
done
```

## Success Criteria

The task is complete when:

1. [ ] Confidence scores correlate with actual detection accuracy within ±10% (validated statistically)
2. [ ] Component-specific confidence thresholds implemented for geometry extraction, OCR, puzzle detection, and domino detection
3. [ ] Visual indicators in pips-solver UI clearly communicate confidence levels with color coding and messages
4. [ ] High-confidence detections (>85% threshold) proven to have >90% actual accuracy in test dataset
5. [ ] Validation dataset created with ground truth labels for ongoing monitoring
6. [ ] Configuration files document threshold values and calibration methodology
7. [ ] No console errors or API failures
8. [ ] Existing tests still pass
9. [ ] New functionality verified via browser/API testing
10. [ ] Documentation added explaining confidence score interpretation for users

## QA Acceptance Criteria

**CRITICAL**: These criteria must be verified by the QA Agent before sign-off.

### Unit Tests
| Test | File | What to Verify |
|------|------|----------------|
| `test_confidence_calculation` | `cv-service/test_main.py` (create if needed) | Confidence calculation functions return values in [0.0, 1.0] range |
| `test_threshold_classification` | `cv-service/test_confidence.py` | Threshold classification correctly maps confidence to high/medium/low |
| `test_component_thresholds` | `cv-service/test_confidence.py` | Each detection component has defined thresholds |
| `test_confidence_response_format` | `cv-service/test_api.py` | API responses include confidence field with correct structure |
| `test_ocr_confidence_extraction` | `pips-agent/test_ocr.py` | OCR confidence correctly extracted from pytesseract results |
| `test_confidence_display` | `pips-solver/src/__tests__/ConfidenceIndicator.test.tsx` | UI component correctly displays confidence indicators |

### Integration Tests
| Test | Services | What to Verify |
|------|----------|----------------|
| End-to-end confidence flow | cv-service ↔ pips-solver | Confidence scores from cv-service correctly displayed in UI |
| Agent confidence aggregation | pips-agent ↔ cv-service | Combined confidence from multiple detection components calculated correctly |
| Low confidence handling | cv-service ↔ pips-solver | Low confidence detections trigger appropriate UI warnings |

### Validation Tests (Statistical)
| Test | Dataset | Expected Outcome |
|------|---------|------------------|
| Confidence-accuracy correlation | Ground truth validation set (50+ samples) | Pearson correlation > 0.9, MAE < 10% |
| High confidence accuracy | Detections with confidence > high threshold | Actual accuracy > 90% |
| Low confidence accuracy | Detections with confidence < medium threshold | Actual accuracy < 70% (correctly identified as uncertain) |
| Per-component calibration | Separate validation per component | Each component's confidence within ±10% of actual accuracy |

### End-to-End Tests
| Flow | Steps | Expected Outcome |
|------|-------|------------------|
| High confidence detection | 1. Upload clear puzzle image 2. Run extraction 3. Check confidence | Green indicator, >85% confidence, message "likely accurate" |
| Low confidence detection | 1. Upload blurry/poor image 2. Run extraction 3. Check confidence | Red/amber indicator, <70% confidence, message "review required" |
| Component breakdown visibility | 1. Run detection 2. View detailed results | UI shows confidence breakdown per component |

### Browser Verification (Frontend)
| Page/Component | URL | Checks |
|----------------|-----|--------|
| Detection Results | `http://localhost:3000/results` (or equivalent) | Confidence indicator visible, color-coded correctly |
| Confidence Tooltip | Hover over confidence indicator | Explanation of confidence level shown |
| High Confidence UI | Upload test image with expected high confidence | Green indicator, no warning messages |
| Low Confidence UI | Upload test image with expected low confidence | Red/amber indicator, clear warning to review |

### API Verification
| Endpoint | Test | Expected |
|----------|------|----------|
| `/extract-geometry` | POST with test image | Response includes `confidence` field (0.0-1.0) and `threshold` classification |
| `/crop-puzzle` | POST with test image | Response includes `confidence` field |
| `/crop-dominoes` | POST with test image | Response includes `confidence` field |

### Database Verification (if applicable)
| Check | Query/Command | Expected |
|-------|---------------|----------|
| Confidence logging | Check logs or database for stored confidence scores | Confidence scores logged for analysis and refinement |

### Configuration Verification
| Check | File | Expected |
|-------|------|----------|
| Thresholds documented | `cv-service/config.py` or similar | All components have defined high/medium thresholds |
| Calibration methodology | `README.md` or `docs/confidence-calibration.md` | Document explains how thresholds were determined |

### QA Sign-off Requirements
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All validation tests pass (statistical correlation verified)
- [ ] All E2E tests pass
- [ ] Browser verification complete - confidence indicators display correctly
- [ ] API verification complete - all endpoints return confidence scores
- [ ] Confidence-accuracy correlation validated (within ±10%)
- [ ] High-confidence detections validated (>90% accuracy)
- [ ] No regressions in existing functionality
- [ ] Code follows established patterns in cv-service and pips-agent
- [ ] No security vulnerabilities introduced
- [ ] Configuration files document thresholds and methodology
- [ ] User-facing documentation updated with confidence interpretation guide

## Appendix: Confidence Calibration Strategy

### Current State (Hypothesis)
- Confidence scores are systematically conservative (e.g., 40% reported when actual is 50-60%)
- Likely due to overly pessimistic confidence calculation or miscalibrated thresholds
- Users cannot distinguish reliable detections from unreliable ones

### Target State
- Confidence scores accurately predict detection quality
- High confidence (>85%) → >90% actual accuracy
- Medium confidence (70-85%) → 70-90% actual accuracy
- Low confidence (<70%) → <70% actual accuracy

### Calibration Process
1. **Data Collection**: Label 100+ detection samples as correct/incorrect
2. **Performance Measurement**: Run current system, record confidence vs accuracy
3. **Calibration**: Apply statistical calibration (Platt scaling or isotonic regression)
4. **Threshold Setting**: Use calibrated scores to set high/medium/low boundaries
5. **Validation**: Test on held-out set, verify ±10% correlation
6. **Monitoring**: Log confidence and outcomes for ongoing refinement

### Metrics to Track
- **Calibration Curve**: Plot predicted confidence vs actual accuracy (should be diagonal)
- **Expected Calibration Error (ECE)**: Average difference between confidence and accuracy
- **Brier Score**: Mean squared error of confidence predictions
- **Threshold Precision/Recall**: For each threshold, measure precision and recall
