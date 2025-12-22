# Specification: Enhance AI Visual Pips Board Extraction Accuracy

## Overview

This task enhances the NYT Pips puzzle extraction system to improve accuracy of AI-based data extraction from puzzle screenshots. The current 5-stage multi-model extraction pipeline (`aiExtraction.ts`, `pipeline.ts`) is producing inaccurate results, particularly in grid detection, region mapping, constraint reading, and domino pip counting. This specification defines improvements to the extraction pipeline, CV preprocessing integration, validation logic, and UI/UX workflow to achieve reliable, accurate puzzle extraction.

## Workflow Type

**Type**: feature

**Rationale**: This is a feature enhancement that improves existing functionality (AI extraction accuracy) while adding new capabilities (enhanced validation, better UI feedback, improved preprocessing). It involves changes across multiple services and requires coordinated improvements to the extraction pipeline, CV service, and frontend UI.

## Task Scope

### Services Involved
- **pips-solver** (primary) - React Native/Expo frontend containing AI extraction logic, UI components, and user workflow
- **cv-service** (integration) - Python/FastAPI backend for image preprocessing, grid detection, and puzzle cropping
- **pips-agent** (reference) - Python agent with OCR and constraint extraction tools to reference patterns

### This Task Will:
- [ ] Improve grid geometry detection accuracy in Stage 1 (rows/cols counting)
- [ ] Enhance cell detection and hole identification in Stage 2
- [ ] Improve region color mapping accuracy in Stage 3
- [ ] Fix constraint extraction (sum values, operators, all_equal detection) in Stage 4
- [ ] Improve domino pip counting accuracy in Stage 5
- [ ] Add confidence-based validation with user review prompts
- [ ] Enhance CV preprocessing to provide cleaner images to AI
- [ ] Improve AI verification modal with visual diff and correction UI
- [ ] Add progress indicators with per-stage confidence feedback

### Out of Scope:
- Complete rewrite of the extraction pipeline architecture
- Adding new AI providers beyond existing (OpenRouter, Google, Anthropic, OpenAI)
- Mobile app UI redesign beyond extraction-related screens
- Changes to the puzzle solving logic (`solver.ts`)

## Service Context

### pips-solver (Primary)

**Tech Stack:**
- Language: TypeScript
- Framework: React Native + Expo SDK v54.0.30
- Key libraries: expo-image-picker, expo-file-system, react-native-svg, Zod
- Key directories: `src/services/extraction/`, `src/app/components/`, `src/app/screens/`

**Entry Point:** `index.ts`

**How to Run:**
```bash
cd pips-solver
npm run start
# or
npx expo start
```

**Port:** 8081 (Metro), 19000/19001 (Expo)

### cv-service (Integration)

**Tech Stack:**
- Language: Python 3.x
- Framework: FastAPI
- Key libraries: OpenCV, NumPy, Pydantic
- Key directories: Root level

**Entry Point:** `main.py`

**How to Run:**
```bash
cd cv-service
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

**Port:** 8080

**API Routes:**
- `POST /extract-geometry` - Extract grid geometry from screenshot
- `POST /crop-puzzle` - Crop image to puzzle region only
- `POST /crop-dominoes` - Crop image to domino tray region
- `GET /health` - Health check

## Files to Modify

| File | Service | What to Change |
|------|---------|---------------|
| `pips-solver/src/services/aiExtraction.ts` | pips-solver | Improve prompt engineering for board/domino extraction, enhance error handling |
| `pips-solver/src/services/extraction/pipeline.ts` | pips-solver | Add validation between stages, improve confidence aggregation |
| `pips-solver/src/services/extraction/stages/gridGeometry.ts` | pips-solver | Improve grid dimension detection prompts |
| `pips-solver/src/services/extraction/stages/cellDetection.ts` | pips-solver | Enhance cell vs hole detection |
| `pips-solver/src/services/extraction/stages/regionMapping.ts` | pips-solver | Improve color region identification |
| `pips-solver/src/services/extraction/stages/constraintExtraction.ts` | pips-solver | Fix constraint value and operator extraction |
| `pips-solver/src/services/extraction/stages/dominoExtraction.ts` | pips-solver | Improve pip counting accuracy |
| `pips-solver/src/services/extraction/validation/gridValidator.ts` | pips-solver | Add cross-validation checks |
| `pips-solver/src/services/cvExtraction.ts` | pips-solver | Enhance CV service integration |
| `pips-solver/src/app/components/AIVerificationModal.tsx` | pips-solver | Add visual diff, cell-by-cell correction UI |
| `pips-solver/src/app/screens/OverlayBuilderScreen.tsx` | pips-solver | Improve extraction workflow and feedback |
| `cv-service/main.py` | cv-service | Enhance grid detection and cropping accuracy |
| `cv-service/hybrid_extraction.py` | cv-service | Improve puzzle region detection |

## Files to Reference

These files show patterns to follow:

| File | Pattern to Copy |
|------|----------------|
| `pips-solver/src/services/extractionSchemas.ts` | Zod schema validation patterns for AI responses |
| `pips-solver/src/services/jsonParsingUtils.ts` | JSON parsing with fallback strategies |
| `pips-solver/src/services/ensembleExtraction.ts` | Multi-model consensus voting pattern |
| `pips-solver/src/model/overlayTypes.ts` | Type definitions for extraction results |
| `pips-agent/tools/extract_puzzle.py` | CV-based extraction approach (reference) |
| `pips-agent/utils/ocr_helper.py` | Tesseract OCR integration for constraint reading |

## Patterns to Follow

### Zod Schema Validation

From `pips-solver/src/services/extractionSchemas.ts`:

```typescript
import { z } from 'zod';

export const BoardExtractionSchema = z.object({
  rows: z.number().int().positive(),
  cols: z.number().int().positive(),
  shape: z.string(),
  regions: z.string(),
  constraints: z.record(z.object({
    type: z.enum(['sum', 'all_equal', 'all_different']),
    op: z.enum(['==', '<', '>', '!=']).optional(),
    value: z.number().optional(),
  })).optional(),
  confidence: z.object({
    grid: z.number().min(0).max(1),
    regions: z.number().min(0).max(1),
    constraints: z.number().min(0).max(1),
  }).optional(),
});
```

**Key Points:**
- All AI responses must be validated with Zod schemas
- Use `safeParse()` for non-throwing validation
- Include confidence scores in all extraction results

### Multi-Model Consensus Pattern

From `pips-solver/src/services/ensembleExtraction.ts`:

```typescript
// Run extraction with multiple models
const responses = await Promise.all(
  models.map(model => extractWithModel(image, model))
);

// Find consensus among responses
const consensus = findConsensus(responses, {
  gridThreshold: 0.8,  // At least 80% must agree on grid
  regionThreshold: 0.7, // At least 70% must agree on regions
});
```

**Key Points:**
- Query multiple models in parallel for critical extraction stages
- Use confidence-weighted voting for consensus
- Fall back to single model if consensus fails

### Stage-Based Extraction Pipeline

From `pips-solver/src/services/extraction/pipeline.ts`:

```typescript
// Each stage builds on previous results
const gridResult = await extractGridGeometry(image, config);
const cellResult = await extractCellDetection(image, gridResult, config);
const regionResult = await extractRegionMapping(image, gridResult, cellResult, config);
const constraintResult = await extractConstraints(image, regionResult, config);
const dominoResult = await extractDominoes(image, cellResult, config);
```

**Key Points:**
- Each stage receives context from previous stages
- Validate stage output before passing to next stage
- Track per-stage confidence for overall quality assessment

## Requirements

### Functional Requirements

1. **Accurate Grid Dimension Detection**
   - Description: Correctly identify rows and columns from puzzle screenshot, including irregular shapes
   - Acceptance: Grid dimensions match actual puzzle ±0 tolerance for standard puzzles

2. **Reliable Cell/Hole Detection**
   - Description: Distinguish between valid cells and holes in irregular puzzle shapes
   - Acceptance: Shape string accurately represents puzzle structure with no false positives/negatives

3. **Accurate Region Color Mapping**
   - Description: Correctly identify and label colored regions with consistent letter assignments
   - Acceptance: All regions correctly identified with proper boundaries

4. **Correct Constraint Extraction**
   - Description: Extract constraint types (sum, all_equal) and values from diamond labels
   - Acceptance: Constraints match visual labels with ≥95% accuracy

5. **Reliable Domino Pip Counting**
   - Description: Count pips (0-6) on each domino half in the tray
   - Acceptance: Domino values correct for ≥90% of dominoes per puzzle

6. **User Verification Workflow**
   - Description: Allow users to review and correct AI extraction before applying
   - Acceptance: Modal shows clear visual representation, allows cell-by-cell correction

7. **Progressive Confidence Feedback**
   - Description: Show per-stage confidence during extraction with clear indicators
   - Acceptance: User sees extraction progress with confidence percentages

### Edge Cases

1. **Low Image Quality** - Reduce confidence scores, prompt for manual verification
2. **Unusual Grid Shapes** - Handle L-shapes, crosses, scattered holes correctly
3. **Similar Region Colors** - Distinguish between adjacent similar-colored regions
4. **Small/Rotated Text** - Handle constraint labels that are small or at angles
5. **Cropped/Partial Images** - Detect when puzzle is partially visible, warn user
6. **HEIC Format Images** - Convert to JPEG before processing (already handled)

## Implementation Notes

### DO
- Follow the 5-stage extraction pattern in `pipeline.ts`
- Reuse `parseJSONWithFallback()` for all AI response parsing
- Use Zod schemas for validation of all extraction results
- Add descriptive logging at each extraction stage
- Show confidence percentages to users after extraction
- Allow users to correct individual cells/regions in verification modal
- Use CV service for initial cropping when available (hybrid mode)

### DON'T
- Create new extraction approaches when existing patterns work
- Skip validation between extraction stages
- Show raw AI responses to users (format appropriately)
- Ignore low confidence scores (prompt for review)
- Make API calls without proper error handling

## Development Environment

### Start Services

```bash
# Terminal 1: Start CV Service
cd cv-service
uvicorn main:app --host 0.0.0.0 --port 8080 --reload

# Terminal 2: Start Expo Frontend
cd pips-solver
npm run start
```

### Service URLs
- CV Service: http://localhost:8080
- Expo Dev Server: http://localhost:8081
- Expo Dev Tools: http://localhost:19000

### Required Environment Variables
- `OPENROUTER_API_KEY`: OpenRouter API key (access to all models)
- `GOOGLE_API_KEY`: Google/Gemini API key (optional if using OpenRouter)
- `ANTHROPIC_API_KEY`: Anthropic/Claude API key (optional if using OpenRouter)
- `OPENAI_API_KEY`: OpenAI/GPT API key (optional if using OpenRouter)

## Success Criteria

The task is complete when:

1. [ ] Grid dimension detection achieves ≥98% accuracy on test puzzle set
2. [ ] Cell/hole detection correctly identifies puzzle shape
3. [ ] Region mapping correctly identifies all colored regions
4. [ ] Constraint extraction achieves ≥95% accuracy
5. [ ] Domino pip counting achieves ≥90% accuracy per puzzle
6. [ ] Verification modal allows cell-by-cell correction
7. [ ] Confidence indicators show per-stage progress
8. [ ] No console errors during extraction workflow
9. [ ] Existing tests still pass
10. [ ] New functionality verified via simulator/device testing

## QA Acceptance Criteria

**CRITICAL**: These criteria must be verified by the QA Agent before sign-off.

### Unit Tests
| Test | File | What to Verify |
|------|------|----------------|
| Grid schema validation | `extractionSchemas.test.ts` | BoardExtractionSchema validates correct data, rejects invalid |
| JSON parsing fallback | `jsonParsingUtils.test.ts` | parseJSONWithFallback handles malformed JSON gracefully |
| Grid validation | `gridValidator.test.ts` | Validates shape/regions dimensions match |
| Constraint parsing | `constraintParser.test.ts` | Parses sum, all_equal, comparison operators |

### Integration Tests
| Test | Services | What to Verify |
|------|----------|----------------|
| CV cropping flow | pips-solver ↔ cv-service | Image cropped correctly before AI extraction |
| Hybrid extraction | pips-solver ↔ cv-service | CV bounds used for overlay alignment |
| Multi-model consensus | pips-solver ↔ AI APIs | Responses from multiple models aggregated correctly |

### End-to-End Tests
| Flow | Steps | Expected Outcome |
|------|-------|------------------|
| Happy path extraction | 1. Select puzzle image 2. AI Extract 3. Verify 4. Accept | Puzzle created with accurate data |
| Low confidence flow | 1. Select poor quality image 2. AI Extract | User prompted to verify/correct |
| Manual correction | 1. Extract 2. Verify 3. Correct cells 4. Accept | Corrections applied to final state |

### Browser/Device Verification
| Page/Component | URL/Screen | Checks |
|----------------|-----|--------|
| OverlayBuilderScreen | App → New Puzzle | AI extraction starts, progress shown |
| AIVerificationModal | App → New Puzzle → Extract | Shows grid, regions, constraints, dominoes |
| Step1GridAlignment | Builder Step 1 | Grid overlay aligns with image |

### Database Verification (if applicable)
| Check | Query/Command | Expected |
|-------|---------------|----------|
| Draft auto-save | `AsyncStorage.getItem('drafts')` | Draft saved after image selection |
| Puzzle save | `AsyncStorage.getItem('puzzles')` | Puzzle saved with complete spec |

### QA Sign-off Requirements
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All E2E tests pass
- [ ] Device verification complete (iOS simulator or Android emulator)
- [ ] Extraction accuracy verified on 5+ test puzzles
- [ ] No regressions in existing functionality
- [ ] Code follows established patterns (Zod validation, error handling)
- [ ] No security vulnerabilities introduced (API keys not exposed)
