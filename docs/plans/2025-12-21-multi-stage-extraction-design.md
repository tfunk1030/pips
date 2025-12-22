# Multi-Stage AI Extraction Pipeline Design

**Date:** December 21, 2025
**Status:** Approved
**Goal:** Improve puzzle extraction accuracy from <25% to 80%+ for NYT Pips screenshots

---

## Executive Summary

Replace the current single/dual-pass AI extraction with a 5-stage pipeline where each stage:
- Queries 3 models in parallel (Gemini 3 Pro, GPT-5.2, Claude Opus 4.5)
- Uses confidence-weighted consensus with majority fallback
- Has stage-specific validation rules
- Can retry with clarifying prompts on failure

**Key constraints:**
- App Store deployable (no local CV dependencies)
- Accuracy priority over speed (30-45s acceptable)
- OpenRouter API support for unified key management

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-STAGE EXTRACTION PIPELINE              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Screenshot ──►  Stage 1  ──►  Stage 2  ──►  Stage 3  ──►  ... │
│                  (Grid)       (Cells)       (Regions)           │
│                    │            │             │                 │
│              ┌─────┴─────┐ ┌────┴────┐  ┌─────┴─────┐          │
│              │ Gemini 3  │ │ Gemini 3│  │ Gemini 3  │          │
│              │ GPT-5.2   │ │ GPT-5.2 │  │ GPT-5.2   │          │
│              │ Opus 4.5  │ │ Opus 4.5│  │ Opus 4.5  │          │
│              └─────┬─────┘ └────┬────┘  └─────┬─────┘          │
│                    │            │             │                 │
│              ┌─────▼─────┐ ┌────▼────┐  ┌─────▼─────┐          │
│              │ Consensus │ │Consensus│  │ Consensus │          │
│              │ + Validate│ │+Validate│  │ +Validate │          │
│              └─────┬─────┘ └────┬────┘  └─────┴─────┘          │
│                    │            │             │                 │
│                    ▼            ▼             ▼                 │
│              [Grid Dims]   [Shape]      [Regions]               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5 Stages with Validation

| Stage | Output | Validation Rules |
|-------|--------|------------------|
| 1. Grid Geometry | rows, cols | 4 ≤ rows,cols ≤ 8 (NYT range) |
| 2. Cell/Hole Map | shape string | length = rows × cols, only `.#` chars |
| 3. Region Labels | regions string | same dims as shape, A-Z only, covers all `.` cells |
| 4. Constraints | region→constraint map | regions exist, values 0-42 (max sum), valid operators |
| 5. Dominoes | array of [pip1, pip2] | count = (cells - holes) / 2, pips 0-6, no duplicates |

---

## Model Selection (December 2025)

Based on current benchmarks:

| Model | Strengths | Use Case |
|-------|-----------|----------|
| **Gemini 3 Pro** | #1 vision, spatial understanding, derendering | All stages |
| **GPT-5.2** | Chart/UI reasoning, fine visual detail | All stages |
| **Claude Opus 4.5** | Instruction following, structured output | All stages |

All 3 models used for every stage to maximize accuracy through ensemble consensus.

---

## Stage-by-Stage Prompts

### Stage 1: Grid Geometry

```
Prompt:
"Count the puzzle grid dimensions in this NYT Pips screenshot.
- Count ROWS: horizontal lines of cells from top to bottom
- Count COLS: vertical columns from left to right
- Include holes (empty positions) in your count
- NYT Pips grids are typically 4-8 rows and 4-8 columns

Return JSON: { "rows": N, "cols": M, "confidence": 0.0-1.0 }
Confidence: 0.95+ if grid lines are clear, lower if obscured/cropped"
```

**Validation:**
- `4 ≤ rows ≤ 8` and `4 ≤ cols ≤ 8`
- All 3 models must return valid integers
- Cross-check: if 2 models say 6x5 and 1 says 5x6, flag for rotation check

### Stage 2: Cell/Hole Detection

```
Prompt (uses Stage 1 output):
"Given this {rows}x{cols} NYT Pips grid, identify which positions are cells vs holes.
- Cell (.): colored background where a domino pip can be placed
- Hole (#): empty/dark space, no cell exists there

Build a shape string: {rows} lines, {cols} characters each.
Use '.' for cells, '#' for holes.

Example 6x5 with corners missing:
##...#
......
......
......
#...##

Return JSON: { "shape": "multiline string", "confidence": 0.0-1.0 }"
```

**Validation:**
- Exactly `rows` lines, each with exactly `cols` characters
- Only `.` and `#` characters
- Count of `.` must be even (dominoes cover 2 cells each)
- At least 8 cells (minimum viable puzzle)

### Stage 3: Region Mapping

```
Prompt (uses Stage 1 & 2 output):
"Given this NYT Pips grid shape:
{shape}

Identify distinct colored regions. Each region has a unique background color.
Common NYT colors: pink, coral, orange, peach, teal, cyan, gray, olive, green, purple.

Assign labels A, B, C, D... to each region.
Build a regions string matching the shape dimensions.
Use '#' for holes (same as shape), letters for cells.

Example:
Shape:    Regions:
##...#    ##AAB#
......    CCAABB
......    CCDDBB
......    EEDDFF
#...##    #EE###

Return JSON: { "regions": "multiline string", "confidence": 0.0-1.0 }"
```

**Validation:**
- Dimensions match shape exactly
- `#` positions match shape's `#` positions
- Only A-Z letters for cell positions
- Each region has ≥2 cells (must fit at least 1 domino)
- Regions are contiguous (cells with same label must be adjacent)

### Stage 4: Constraint Extraction

```
Prompt (uses Stage 3 output):
"Given these regions: {region_labels}

Find the constraint for each region. Constraints appear as:
- Diamond shapes with numbers (e.g., ◇12 means sum=12)
- Operators: = (equals), < (less than), > (greater than)
- 'E' or '=' symbol means all_equal (all pips same value)

For each region, extract:
- type: 'sum' or 'all_equal'
- op: '==', '<', '>' (for sum constraints)
- value: the target number (for sum constraints)

Return JSON: {
  "constraints": { "A": {"type":"sum","op":"==","value":12}, "B": {...} },
  "confidence": 0.0-1.0
}"
```

**Validation:**
- All constraint regions exist in regions string
- Sum values: 0-42 range (max possible: 6+6+6+6+6+6+6)
- Operators must be `==`, `<`, or `>`
- all_equal constraints have no value/op

### Stage 5: Domino Extraction

```
Prompt:
"Count the dominoes in the tray/reference area of this NYT Pips screenshot.
Each domino has two halves with 0-6 pips (dots) each.

Count pips on each half carefully:
- 0 pips = blank
- 1-6 pips = count the dots

Expected domino count: {expected_count} (calculated from grid)

Return JSON: {
  "dominoes": [[pip1, pip2], [pip1, pip2], ...],
  "confidence": 0.0-1.0
}"
```

**Validation:**
- Domino count = (cell_count) / 2
- Each pip value 0-6
- No duplicate dominoes (NYT uses unique set)
- Dominoes sorted for comparison: [min, max] ordering

---

## Consensus Algorithm

```typescript
function resolveConsensus(responses: ModelResponse[]): ConsensusResult {
  // responses = [{ model, answer, confidence }, ...]

  // Step 1: Check for high-confidence winner
  const sorted = responses.sort((a, b) => b.confidence - a.confidence);
  const top = sorted[0];
  const second = sorted[1];

  if (top.confidence - second.confidence > 0.10) {
    // Clear winner by confidence
    return { answer: top.answer, source: 'confidence', confident: true };
  }

  // Step 2: Confidence within 10%, use majority vote
  const votes = groupByAnswer(responses);
  const majority = votes.find(v => v.count >= 2);

  if (majority) {
    return { answer: majority.answer, source: 'majority', confident: true };
  }

  // Step 3: All 3 differ, no consensus
  return { answer: top.answer, source: 'best-effort', confident: false };
}
```

### Retry Strategy

| Scenario | Action |
|----------|--------|
| Consensus reached | Proceed to next stage |
| No consensus, retry < 2 | Re-prompt all 3 with clarifying hints |
| No consensus, retry = 2 | Use highest confidence + flag `needsReview: true` |
| Validation fails | Re-prompt with specific error feedback |

### Clarifying Retry Prompts

```
Grid retry: "Previous attempts got {A}, {B}, {C}.
             Look again - count grid lines carefully, not cells."

Shape retry: "Previous attempts differ. Focus on:
              - Dark/black areas = holes (#)
              - Colored areas = cells (.)
              The grid is {rows}x{cols}."

Region retry: "Previous attempts: {A}, {B}, {C}.
               Key: each DISTINCT background color = one region.
               Similar colors (coral vs pink) are DIFFERENT regions."
```

---

## API Integration

### Supported Configurations

**Option 1: OpenRouter (Recommended)**
- Single API key for all models
- Unified billing
- Easy model switching

**Option 2: Individual Keys**
- Direct API access
- Lower latency
- 3 keys to manage

### Configuration Interface

```typescript
interface ExtractionConfig {
  // API Keys - OpenRouter OR individual
  apiKeys: {
    openrouter?: string;        // If set, used for all models
    google?: string;            // Override for Gemini
    openai?: string;            // Override for GPT
    anthropic?: string;         // Override for Claude
  };

  // Model identifiers (OpenRouter format)
  models: {
    spatial: string;    // default: "google/gemini-3-pro"
    detail: string;     // default: "openai/gpt-5.2"
    validator: string;  // default: "anthropic/claude-opus-4.5"
  };

  // Behavior tuning
  maxRetries: number;           // default: 2
  confidenceThreshold: number;  // default: 0.10 (for majority fallback)
  timeoutMs: number;            // default: 30000 per stage

  // NYT-specific validation bounds
  validation: {
    minGridSize: number;        // default: 4
    maxGridSize: number;        // default: 8
    pipRange: [number, number]; // default: [0, 6]
    uniqueDominoes: boolean;    // default: true
  };

  // Debug/review options
  flagLowConfidence: boolean;   // default: true (< 0.7)
  saveDebugResponses: boolean;  // default: false
  enableLegacyFallback: boolean; // default: true
}
```

### API Routing Logic

```typescript
function getApiConfig(model: string, config: ExtractionConfig) {
  // Check for model-specific key first
  if (model.startsWith('google/') && config.apiKeys.google) {
    return { endpoint: 'googleapis.com', key: config.apiKeys.google };
  }
  if (model.startsWith('openai/') && config.apiKeys.openai) {
    return { endpoint: 'api.openai.com', key: config.apiKeys.openai };
  }
  if (model.startsWith('anthropic/') && config.apiKeys.anthropic) {
    return { endpoint: 'api.anthropic.com', key: config.apiKeys.anthropic };
  }

  // Fall back to OpenRouter
  if (config.apiKeys.openrouter) {
    return {
      endpoint: 'openrouter.ai/api/v1',
      key: config.apiKeys.openrouter,
      model: model
    };
  }

  throw new Error(`No API key configured for ${model}`);
}
```

### Error Handling

| Error Type | Handling |
|------------|----------|
| API key missing | Abort with clear message: "Missing {provider} API key in Settings" |
| API rate limit | Retry with exponential backoff (1s, 2s, 4s) |
| API timeout | Skip model for this stage, continue with 2-model consensus |
| Invalid JSON response | Parse with fallback strategies, retry if still fails |
| Validation failure | Retry stage with error feedback in prompt |
| All retries exhausted | Return partial result + `needsReview: true` |

---

## Performance Estimates

### Timing

| Stage | Parallel API calls | ~Time |
|-------|-------------------|-------|
| 1. Grid | 3 calls | 3-5s |
| 2. Cells | 3 calls | 3-5s |
| 3. Regions | 3 calls | 3-5s |
| 4. Constraints | 3 calls | 3-5s |
| 5. Dominoes | 3 calls | 3-5s |
| **Total** | 15 calls | **15-25s** |

With retries (worst case): 30-45s

### Cost (per extraction)

| Model | Calls | ~Input tokens | ~Output tokens | ~Cost |
|-------|-------|---------------|----------------|-------|
| Gemini 3 Pro | 5 | 5 × 2000 | 5 × 200 | ~$0.03 |
| GPT-5.2 | 5 | 5 × 2000 | 5 × 200 | ~$0.08 |
| Opus 4.5 | 5 | 5 × 2000 | 5 × 200 | ~$0.15 |
| **Total** | 15 | ~30k | ~3k | **~$0.26** |

With retries (worst case): ~$0.50 per extraction

---

## Data Structures

### Extraction Result

```typescript
interface ExtractionResult {
  // Core puzzle data
  grid: {
    rows: number;
    cols: number;
    shape: string;      // multiline, '.' and '#'
    regions: string;    // multiline, letters and '#'
  };
  constraints: Record<string, Constraint>;
  dominoes: [number, number][];

  // Confidence & review flags
  confidence: {
    overall: number;    // min of all stages
    perStage: {
      grid: number;
      cells: number;
      regions: number;
      constraints: number;
      dominoes: number;
    };
  };
  needsReview: boolean;
  reviewHints: string[];  // e.g., ["Region boundaries low confidence"]

  // Debug info (if enabled)
  debug?: {
    rawResponses: Record<string, ModelResponse[]>;
    consensusDetails: Record<string, ConsensusResult>;
    retryCount: number;
  };
}

interface Constraint {
  type: 'sum' | 'all_equal';
  op?: '==' | '<' | '>';
  value?: number;
}

interface ModelResponse {
  model: string;
  answer: any;
  confidence: number;
  latencyMs: number;
}

interface ConsensusResult {
  answer: any;
  source: 'confidence' | 'majority' | 'best-effort';
  confident: boolean;
}
```

---

## File Structure

```
pips-solver/src/
├── services/
│   ├── extraction/
│   │   ├── index.ts                 # Main entry point
│   │   ├── types.ts                 # Interfaces & types
│   │   ├── config.ts                # Configuration defaults
│   │   ├── apiClient.ts             # OpenRouter + direct API calls
│   │   ├── consensus.ts             # Voting & retry logic
│   │   ├── stages/
│   │   │   ├── gridGeometry.ts      # Stage 1
│   │   │   ├── cellDetection.ts     # Stage 2
│   │   │   ├── regionMapping.ts     # Stage 3
│   │   │   ├── constraintExtraction.ts  # Stage 4
│   │   │   └── dominoExtraction.ts  # Stage 5
│   │   └── validation/
│   │       ├── gridValidator.ts     # NYT-specific rules
│   │       ├── regionValidator.ts   # Contiguity checks
│   │       └── dominoValidator.ts   # Count & uniqueness
│   ├── aiExtraction.ts              # DEPRECATED - keep for fallback
│   └── ensembleExtraction.ts        # DEPRECATED - keep for fallback
├── app/
│   └── screens/
│       └── SettingsScreen.tsx       # Add OpenRouter key option
```

---

## Implementation Order

| Phase | Tasks | Dependency |
|-------|-------|------------|
| 1 | Types, config, API client | None |
| 2 | Stage 1-2 (grid + cells) | Phase 1 |
| 3 | Stage 3-4 (regions + constraints) | Phase 2 |
| 4 | Stage 5 (dominoes) | Phase 3 |
| 5 | Consensus + retry logic | Phase 4 |
| 6 | Validation layer | Phase 5 |
| 7 | Integration + Settings UI | Phase 6 |
| 8 | Testing + tuning | Phase 7 |

---

## Fallback Strategy

```typescript
async function extractPuzzle(image: string, config: ExtractionConfig) {
  try {
    // Try new multi-stage pipeline
    return await multiStageExtract(image, config);
  } catch (error) {
    if (config.enableLegacyFallback) {
      // Fall back to old single-pass extraction
      console.warn('Multi-stage failed, using legacy extraction');
      return await legacyExtract(image, config);
    }
    throw error;
  }
}
```

---

## Future Enhancements

If accuracy still insufficient after implementation:

1. **Add Cloud CV Service** - Deploy Python CV pipeline for geometric grid detection
2. **On-device preprocessing** - React Native frame processors for image enhancement
3. **Fine-tuned models** - Train specialized model on NYT Pips screenshots
4. **User feedback loop** - Collect corrections to improve prompts over time

---

## Acceptance Criteria

- [ ] Extraction accuracy ≥80% on NYT Pips screenshots (measured on 20 test images)
- [ ] All 5 stages implemented with 3-model ensemble
- [ ] Consensus + retry logic working
- [ ] OpenRouter API integration working
- [ ] Settings UI updated with API key options
- [ ] Legacy fallback working
- [ ] Solver successfully solves extracted puzzles
