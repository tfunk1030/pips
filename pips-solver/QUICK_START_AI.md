# Quick Start: AI Extraction Features

This guide helps developers understand and work with the new AI extraction features.

## 5-Minute Overview

### What Changed?

Added AI-powered puzzle extraction from screenshots with confidence scoring and self-verification.

### Key Files

- `src/services/aiExtraction.ts` - Main extraction logic
- `src/app/components/ConfidenceIndicator.tsx` - Confidence UI
- `src/model/overlayTypes.ts` - Type definitions with confidence

### API Flow

```
Screenshot → Board Extraction → (Verification?) → Domino Extraction → Builder State
                ↓                      ↓                   ↓              ↓
           Confidence <90%        Corrections          Confidence    UI Display
```

---

## For Frontend Developers

### Using the Extraction Function

```typescript
import { extractPuzzleFromImage } from '../services/aiExtraction';

// Basic usage
const result = await extractPuzzleFromImage(base64Image, apiKey, progress =>
  console.log(progress.message)
);

// With verification enabled
const result = await extractPuzzleFromImage(
  base64Image,
  apiKey,
  progress => console.log(progress.message),
  true // enable verification
);

// Handle result
if (result.success && result.result) {
  const { board, dominoes } = result.result;

  // Check confidence
  if (board.confidence?.grid < 0.8) {
    alert('Low grid confidence - please verify');
  }
}
```

### Displaying Confidence

```typescript
import ConfidenceIndicator from '../../components/ConfidenceIndicator';

// Compact mode (badge)
<ConfidenceIndicator
  label="Grid Layout"
  confidence={state.aiConfidence?.grid}
  compact
/>

// Full mode (progress bar)
<ConfidenceIndicator
  label="Region Boundaries"
  confidence={state.aiConfidence?.regions}
/>
```

### Color Coding

- **Green (≥90%):** High confidence
- **Orange (≥80%):** Medium confidence - verify recommended
- **Red (<80%):** Low confidence - manual check required

---

## For Backend/API Developers

### Prompt Structure

The extraction uses enhanced prompts with:

1. Explicit format examples
2. Critical formatting rules
3. Common mistakes section
4. Confidence score request

**Board Extraction Prompt:**

- Grid dimensions detection
- Shape with holes (# = hole, . = cell)
- Regions (A-J labels)
- Constraints (sum, <, >, ==, all_equal)
- Confidence scores

**Domino Extraction Prompt:**

- Pip counting tips (0-6 dots)
- Reference tray detection
- Confidence score

### Response Format

**Board Response:**

```json
{
  "rows": 4,
  "cols": 5,
  "shape": ".....\n.....\n.....\n.....",
  "regions": "AAABB\nACCBB\nACCDD\nEEEDD",
  "constraints": {
    "A": { "type": "sum", "op": "==", "value": 12 },
    "B": { "type": "all_equal" }
  },
  "confidence": {
    "grid": 0.95,
    "regions": 0.88,
    "constraints": 0.92
  },
  "reasoning": "Clear 4x5 grid..."
}
```

**Domino Response:**

```json
{
  "dominoes": [
    [6, 1],
    [5, 2],
    [4, 3]
  ],
  "confidence": 0.9,
  "reasoning": "Found 3 dominoes in tray..."
}
```

### Verification Prompt

When confidence < 90%, sends:

```
You previously extracted:
[initial extraction]

Looking at the image again, verify accuracy.
If errors found, provide corrections.
```

Response:

```json
{
  "is_correct": false,
  "corrections": {
    "regions": "AAABC\n..." // Only corrected fields
  },
  "issues_found": ["Region B boundaries incorrect"]
}
```

---

## For ML/AI Developers

### Improving Prompt Accuracy

Edit prompts in `src/services/aiExtraction.ts`:

```typescript
// Board extraction prompt
const BOARD_EXTRACTION_PROMPT = `...`;

// Domino extraction prompt
const DOMINO_EXTRACTION_PROMPT = `...`;

// Verification prompt
const VERIFICATION_PROMPT = `...`;
```

**Tips:**

1. Add more examples (especially edge cases)
2. Clarify ambiguous instructions
3. Include visual descriptions (e.g., "diamond-shaped markers")
4. Explicitly state what NOT to do

### Confidence Calibration

If confidence scores don't correlate with accuracy:

1. **Too High:** Add stricter confidence guidance in prompts
2. **Too Low:** Relax confidence criteria or improve prompt clarity
3. **Inconsistent:** Add examples of uncertain scenarios

### Model Selection

Models tried (in order):

```typescript
export const MODEL_CANDIDATES = [
  'claude-sonnet-4-20250514', // Preferred
  'claude-3-5-sonnet-20240620', // Fallback 1
  'claude-3-opus-20240229', // Fallback 2
];
```

Modify in `src/config/models.ts` to test other models.

---

## For QA/Testing

### Test Scenarios

**High Confidence:**

- Clear, well-lit screenshot
- Full puzzle visible (grid + tray)
- Standard grid size (4x4 to 6x6)

**Low Confidence:**

- Blurry image
- Partial cropping
- Poor lighting
- Unusual grid shapes

**Verification Triggers:**

- Medium confidence (80-89%)
- Ambiguous region boundaries
- Hard-to-read constraint text

### Expected Behavior

| Input Quality | Grid Conf | Region Conf | Constraint Conf | Verification? |
| ------------- | --------- | ----------- | --------------- | ------------- |
| High          | ≥90%      | ≥90%        | ≥85%            | No            |
| Medium        | 80-90%    | 80-90%      | 75-85%          | Yes           |
| Low           | <80%      | <80%        | <75%            | Yes + Warning |

### Debug Tips

Enable debug logging:

```typescript
// In aiExtraction.ts
console.log('[DEBUG] extractPuzzleFromImage - Starting...');
```

Check debug output:

- Extraction duration
- Model fallback attempts
- JSON parsing issues
- Confidence scores

---

## Common Issues & Solutions

### Issue: "Invalid JSON in board response"

**Cause:** Multi-line strings split incorrectly
**Solution:** Prompt explicitly forbids line breaks in JSON strings
**Fallback:** Regex-based JSON repair attempts 2-4 line splits

### Issue: Low confidence on good screenshots

**Cause:** Prompt too strict OR unusual puzzle layout
**Solution:**

1. Check if puzzle is atypical (non-rectangular, complex constraints)
2. Verify prompt examples cover this case
3. Add clarifying examples to prompt

### Issue: Verification pass not triggered

**Cause:** Confidence ≥ 90% OR verification disabled
**Solution:**

- Expected behavior if confidence is high
- Enable verification explicitly: `extractPuzzleFromImage(..., true)`

### Issue: API costs too high

**Cause:** Verification triggering too often
**Solution:**

1. Improve prompts to boost initial accuracy
2. Adjust verification threshold (currently <90%)
3. Consider caching similar puzzles

---

## Performance Benchmarks

### Extraction Time

- **Board:** ~2-3 seconds
- **Dominoes:** ~1-2 seconds
- **Verification:** ~2-4 seconds (when triggered)
- **Total (no verification):** 3-5 seconds
- **Total (with verification):** 6-10 seconds

### API Costs

- **Board extraction:** ~$0.008-0.015
- **Domino extraction:** ~$0.003-0.010
- **Verification:** ~$0.005-0.015
- **Total (no verification):** ~$0.011-0.025
- **Total (with verification):** ~$0.016-0.040

### Memory Usage

- Image encoding: ~2-5 MB (depends on resolution)
- API response: ~1-3 KB
- State storage: ~5-10 KB

---

## Future Enhancements

### Ready to Implement

1. Add confidence to Steps 3 & 4 (2 hours)
2. User-configurable verification threshold (1 hour)
3. API usage dashboard (3-4 hours)

### Deferred (Phase 3)

1. Bounding box coordinates (5-8 hours)
2. Overlay visualization (4-6 hours)
3. Click-to-correct interaction (4-6 hours)

### Potential Optimizations

1. Image preprocessing (resize, enhance)
2. Batch extraction for multiple puzzles
3. Local caching of similar puzzles
4. Model fine-tuning (advanced)

---

## Resources

### Documentation

- `IMPLEMENTATION_SUMMARY.md` - Technical details
- `TESTING_GUIDE.md` - Test scenarios
- `COMPREHENSIVE_REPORT.md` - Full implementation report
- `plans/structured-seeking-ladybug.md` - Original analysis

### External Links

- [Claude Vision API Docs](https://docs.anthropic.com/claude/docs/vision)
- [Anthropic Console](https://console.anthropic.com/)
- [Zod Schema Validation](https://zod.dev/)

### Support

- For bugs: Create issue with screenshot + confidence scores
- For improvements: See `IMPLEMENTATION_STATUS.md` for planned features
- For questions: Check CLAUDE.md for project context

---

**Last Updated:** December 19, 2025
**Quick Start Version:** 1.0
