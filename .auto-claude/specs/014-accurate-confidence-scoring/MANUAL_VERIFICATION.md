# Manual Verification Guide: High and Low Confidence Scenarios

This document provides step-by-step instructions for manually verifying the confidence scoring system behaves correctly for different image quality scenarios.

## Overview

The confidence scoring system should:
1. Display **green indicators** for high confidence (clear images)
2. Display **red/amber indicators** for low/medium confidence (blurry/poor images)
3. Show **appropriate messages** that match the confidence level

## Expected Behavior

| Image Quality | Confidence Range | Color | Message |
|---------------|------------------|-------|---------|
| Clear, vivid colors | >= 85% | Green (#10b981) | "High confidence - likely accurate" |
| Medium quality | 70-85% | Amber (#f59e0b) | "Medium confidence - review recommended" |
| Blurry, washed out | < 70% | Red (#ef4444) | "Low confidence - manual verification required" |

## Prerequisites

1. Start the cv-service backend:
   ```bash
   cd cv-service
   python main.py
   # Should start on http://localhost:8080
   ```

2. (Optional) Start the pips-solver frontend:
   ```bash
   cd pips-solver
   npm run start
   # Should start on http://localhost:3000
   ```

## Test Case 1: Clear Image (High Confidence)

### Description
Test with a clear, high-quality puzzle image with vivid colors.

### Test Images
- `cv-service/test_images/test_000_sat180.png` - High saturation (180)
- `cv-service/test_images/test_001_sat200.png` - Very high saturation (200)
- Any puzzle image with bright, vivid colors

### Steps
1. Via API:
   ```bash
   curl -X POST http://localhost:8080/crop-puzzle \
     -F "image=@cv-service/test_images/test_001_sat200.png" \
     | jq '.confidence, .threshold, .confidence_breakdown'
   ```

2. Via UI (if available):
   - Navigate to the puzzle upload page
   - Upload a clear, colorful puzzle image
   - Observe the confidence indicator

### Expected Results
- `confidence`: >= 0.85 (85%)
- `threshold`: "high"
- `confidence_breakdown.saturation`: >= 0.70
- UI indicator: **Green color**
- UI message: "High confidence - likely accurate"

### Pass Criteria
- [ ] Confidence score is >= 85%
- [ ] Threshold is classified as "high"
- [ ] Color indicator is green (#10b981)
- [ ] Message matches expected text

---

## Test Case 2: Blurry/Low Quality Image (Low Confidence)

### Description
Test with a blurry or low-saturation image that should have poor detection confidence.

### Test Images
- `cv-service/test_images/test_017_sat30.png` - Very low saturation (30)
- `cv-service/test_images/test_019_sat50_nogrid.png` - No grid, low saturation
- Any blurry or grayscale image

### Steps
1. Via API:
   ```bash
   curl -X POST http://localhost:8080/crop-puzzle \
     -F "image=@cv-service/test_images/test_017_sat30.png" \
     | jq '.confidence, .threshold, .confidence_breakdown'
   ```

2. Via UI (if available):
   - Navigate to the puzzle upload page
   - Upload a blurry or washed-out image
   - Observe the confidence indicator

### Expected Results
- `confidence`: < 0.70 (70%)
- `threshold`: "low"
- `confidence_breakdown.saturation`: < 0.50
- UI indicator: **Red color**
- UI message: "Low confidence - manual verification required"

### Pass Criteria
- [ ] Confidence score is < 70%
- [ ] Threshold is classified as "low"
- [ ] Color indicator is red (#ef4444)
- [ ] Message matches expected text

---

## Test Case 3: Medium Quality Image (Medium Confidence)

### Description
Test with an image that has moderate quality - not perfect but not terrible.

### Test Images
- `cv-service/test_images/test_012_sat90.png` - Medium saturation (90)
- `cv-service/test_images/test_010_sat100.png` - Medium saturation (100)

### Steps
1. Via API:
   ```bash
   curl -X POST http://localhost:8080/crop-puzzle \
     -F "image=@cv-service/test_images/test_012_sat90.png" \
     | jq '.confidence, .threshold, .confidence_breakdown'
   ```

### Expected Results
- `confidence`: 0.70 - 0.85 (70-85%)
- `threshold`: "medium"
- UI indicator: **Amber color**
- UI message: "Medium confidence - review recommended"

### Pass Criteria
- [ ] Confidence score is between 70-85%
- [ ] Threshold is classified as "medium"
- [ ] Color indicator is amber (#f59e0b)
- [ ] Message matches expected text

---

## Test Case 4: Borderline Confidence

### Description
Test that borderline confidence values (within 5% of thresholds) are properly flagged.

### Steps
1. Create or find an image that produces confidence near 0.85 or 0.70
2. Check if `is_borderline` is true in the response:
   ```bash
   curl -X POST http://localhost:8080/crop-puzzle \
     -F "image=@test_image.png" \
     | jq '.is_borderline'
   ```

### Expected Results
- For confidence 0.80-0.90 or 0.65-0.75: `is_borderline: true`
- UI should show additional "(borderline - near threshold)" text

### Pass Criteria
- [ ] Borderline flag is correctly set
- [ ] UI shows borderline indicator when applicable

---

## Test Case 5: No Grid Image

### Description
Test with an image that has no grid structure - should have very low or zero confidence.

### Test Images
- `cv-service/test_images/test_019_sat50_nogrid.png`
- `cv-service/test_images/test_020_sat30_nogrid.png`
- `cv-service/test_images/test_021_sat80_nogrid.png`

### Steps
```bash
curl -X POST http://localhost:8080/crop-puzzle \
  -F "image=@cv-service/test_images/test_019_sat50_nogrid.png" \
  | jq '.confidence, .threshold, .success'
```

### Expected Results
- `confidence`: Very low or 0
- `threshold`: "low"
- Detection may fail entirely (expected behavior)

### Pass Criteria
- [ ] No grid images produce low confidence
- [ ] System handles gracefully (no crashes)
- [ ] User is warned about low confidence

---

## Automated Test Verification

Before manual testing, ensure all automated tests pass:

### Backend Tests
```bash
cd cv-service
python -m pytest test_confidence_scenarios.py -v -s
```

### Frontend Tests
```bash
cd pips-solver
npm test -- ConfidenceScenarios
```

### Validation Script
```bash
cd cv-service
python validate_confidence.py --test-images test_images/ --ground-truth ground_truth.json
```

Expected output:
- Correlation > 0.9
- MAE < 10%
- High Confidence Accuracy > 90%

---

## Summary Checklist

### High Confidence Scenario
- [ ] Clear image produces >= 85% confidence
- [ ] Green indicator displayed
- [ ] Message: "High confidence - likely accurate"

### Low Confidence Scenario
- [ ] Blurry/poor image produces < 70% confidence
- [ ] Red indicator displayed
- [ ] Message: "Low confidence - manual verification required"

### Medium Confidence Scenario
- [ ] Medium quality image produces 70-85% confidence
- [ ] Amber indicator displayed
- [ ] Message: "Medium confidence - review recommended"

### Cross-Service Consistency
- [ ] Backend thresholds match frontend thresholds
- [ ] Color coding is consistent
- [ ] Messages are consistent

---

## Troubleshooting

### Issue: Confidence always low
- Check image has sufficient color saturation
- Ensure image contains a visible grid structure
- Verify image format is supported (PNG, JPG, WebP)

### Issue: Colors don't match expected
- Verify frontend thresholds match: high=0.85, medium=0.70
- Check CONFIDENCE_COLORS in ConfidenceIndicator.tsx

### Issue: Borderline not showing
- Borderline only shows within 5% of thresholds (0.80-0.90 or 0.65-0.75)
- Check isBorderlineConfidence function

---

## Sign-Off

Date: _______________

Tester: _______________

All test cases passed: [ ] Yes  [ ] No

Notes:
_______________________________________________
_______________________________________________
_______________________________________________
