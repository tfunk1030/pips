# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a NYT Pips puzzle solver project with two main components:

1. **Python Backend** (root directory): Computer vision and constraint satisfaction solver for extracting and solving puzzles from screenshots
2. **React Native Mobile App** (`pips-solver/`): Interactive mobile app for solving Pips puzzles with CSP techniques

## React Native Mobile App (`pips-solver/`)

### Development Commands

```bash
cd pips-solver

# Install dependencies
npm install
# or
yarn install

# Start development server
npm start

# Run on specific platforms
npm run android   # Android device/emulator
npm run ios       # iOS device/simulator
npm run web       # Web browser

# Scan QR code with Expo Go app or press:
# - 'a' for Android
# - 'i' for iOS
# - 'w' for web
```

### Architecture

The mobile app follows a clear modular architecture:

- **`src/model/`**: Core data structures, YAML/JSON parsing, puzzle normalization

  - `types.ts`: TypeScript interfaces for puzzles, solutions, constraints
  - `parser.ts`: YAML/JSON puzzle specification parser
  - `normalize.ts`: Precomputes adjacency lists, region cells, valid edges

- **`src/solver/`**: CSP solver engine (backtracking with constraint propagation)

  - `solver.ts`: Main backtracking algorithm with async/non-blocking support
  - `heuristics.ts`: MRV (Minimum Remaining Values) heuristic for variable ordering
  - `propagate.ts`: Forward checking and constraint propagation
  - `explain.ts`: Generates explanations for unsatisfiable puzzles

- **`src/validator/`**: Validation modules

  - `validateSpec.ts`: Ensures puzzle specification is well-formed
  - `validateSolution.ts`: Verifies solution correctness against all constraints

- **`src/app/`**: UI layer

  - `screens/`: React Native screens (Home, PuzzleViewer, Solve, Settings)
  - `components/`: Reusable components (GridRenderer with SVG-based rendering)
  - `navigation/`: Navigation setup using React Navigation

- **`src/storage/`**: AsyncStorage wrapper for persisting puzzles and settings

- **`src/samples/`**: Sample puzzle YAML files

### CSP Solver Algorithm

The solver uses:

- **Backtracking** with constraint propagation
- **MRV Heuristic**: Chooses most constrained edge first
- **Forward Checking**: Prunes domains after each assignment
- **Non-blocking execution**: Yields to event loop every N iterations to keep UI responsive

Constraint types supported:

- `sum`: Region values must sum to N (with operators: ==, !=, <, >)
- `all_equal`: All values in region must be identical

### Important Development Notes

- Always validate solver output before displaying "Solved" - validation is mandatory for correctness
- The solver runs incrementally/non-blocking to keep UI responsive
- Puzzle specifications use YAML format (see `src/samples/` for examples)
- TypeScript strict mode is enabled

### AI Extraction Features

The mobile app includes AI-powered puzzle extraction from screenshots.

#### Multi-Stage Extraction Pipeline (Dec 2025)

A 5-stage pipeline with 3-model ensemble for maximum accuracy:

**Models Used (via OpenRouter or direct API):**
- **Gemini 3 Pro** (`google/gemini-3-pro`): Best for grid geometry and spatial understanding
- **GPT-5.2** (`openai/gpt-5.2`): Best for OCR, pip counting, and fine visual detail
- **Claude Opus 4.5** (`anthropic/claude-opus-4.5`): Best for instruction following and structured output

**5 Extraction Stages:**
1. **Grid Geometry** - Extract rows/cols dimensions
2. **Cell Detection** - Identify cells (`.`) vs holes (`#`)
3. **Region Mapping** - Map colored regions to labels (A-Z)
4. **Constraint Extraction** - Extract sum/all_equal constraints per region
5. **Domino Extraction** - Count pip values on each domino tile

**Consensus Algorithm:**
- All 3 models run in parallel for each stage
- Confidence-weighted voting with majority fallback
- If top confidence exceeds second by >10%, use highest confidence
- Otherwise, majority vote determines result
- Automatic retry with clarifying prompts on low confidence (<70%)

**NYT Pips Validation Rules:**
- Grid size: 4-8 rows/cols
- Pip values: 0-6
- Unique dominoes required
- Region contiguity check (BFS connectivity)
- Sum feasibility validation

**File Structure:**
```
src/services/extraction/
├── index.ts           # Re-exports all modules
├── types.ts           # Core interfaces (ExtractionConfig, ExtractionResult)
├── config.ts          # Default config, NYT_VALIDATION constants
├── apiClient.ts       # Unified API client (OpenRouter + direct providers)
├── consensus.ts       # Confidence-weighted voting algorithm
├── pipeline.ts        # Main 5-stage orchestrator
├── stages/            # Individual stage implementations
│   ├── gridGeometry.ts
│   ├── cellDetection.ts
│   ├── regionMapping.ts
│   ├── constraintExtraction.ts
│   └── dominoExtraction.ts
└── validation/        # NYT-specific validators
    ├── gridValidator.ts
    ├── regionValidator.ts
    └── dominoValidator.ts
```

#### Configuration

**API Key Modes:**
- `openrouter`: Single OpenRouter API key for all models (recommended)
- `individual`: Separate API keys per provider (Google, OpenAI, Anthropic)

**Extraction Strategies:**
- `fast`: Single model (~5s)
- `balanced`: With verification (~15s)
- `accurate`: Multi-stage (~25s)
- `ensemble`: 3-model consensus (~45s) - **DEFAULT for maximum accuracy**

**Settings (in `src/storage/puzzles.ts`):**
```typescript
interface AppSettings {
  apiKeyMode?: 'openrouter' | 'individual';
  openrouterApiKey?: string;
  anthropicApiKey?: string;
  googleApiKey?: string;
  openaiApiKey?: string;
  extractionStrategy?: 'fast' | 'balanced' | 'accurate' | 'ensemble';
  useMultiStagePipeline?: boolean;  // default: true
  saveDebugResponses?: boolean;     // for troubleshooting
}
```

#### Design Document

Full architecture details in `docs/plans/2025-12-21-multi-stage-extraction-design.md`

## Python Backend (Root Directory)

### Development Commands

```bash
# Solve a puzzle from YAML specification
python solve_pips.py pips_puzzle.yaml

# Extract board cells from screenshot using gridline detection
python extract_board_cells_gridlines.py IMG_2050.png

# Extract board cells (other versions available for tuning)
python extract_board_cells.py
python extract_board_cells_v2.py
python extract_board_cells_v3_autotune.py
python extract_board_cells_v4_edges_plus_masks.py
```

### Architecture

- **`solve_pips.py`**: Main CSP solver for Pips puzzles

  - Parses YAML puzzle specifications with ASCII art board layouts
  - Implements backtracking with constraint propagation
  - Uses MRV-ish heuristic for cell selection
  - Supports sum and all_equal constraints with comparison operators

- **`extract_board_cells_gridlines.py`**: Computer vision pipeline for extracting puzzle grid from screenshots

  - Uses edge detection and projection analysis to find grid lines
  - Filters candidates by interior brightness to distinguish real cells from background
  - Outputs cell coordinates to `cells.txt`
  - Debug images saved to `debug_gridlines/` directory

- **`pips_puzzle.yaml`**: YAML puzzle specification format
  - ASCII art board layout with `.` for cells and `#` for non-cells
  - Region labels map (same dimensions as shape)
  - Domino tiles list
  - Region constraints (sum with operators, all_equal)

### YAML Puzzle Format

The Python solver expects this structure:

```yaml
pips:
  pip_min: 0
  pip_max: 6

dominoes:
  unique: true
  tiles:
    - [6, 1]
    - [6, 2]
    # ... more domino pairs

board:
  # '.' = cell exists, '#' = no cell
  shape: |
    ##....####
    ##....####

  # same dimensions, each '.' becomes a region label
  regions: |
    ##BBCC####
    ##HGFD####

region_constraints:
  A: { type: sum, op: '==', value: 12 }
  B: { type: sum, op: '<', value: 2 }
  E: { type: all_equal }
```

### Computer Vision Pipeline Notes

- ROI detection uses saturation mask to find board vs dark background
- Grid line detection via edge projection and peak finding
- Cell validation by interior brightness and texture variance
- Multiple extraction versions available for different screenshot types
- Debug output includes visualization of ROI, edges, grid lines, and detected cells

## Code Quality Standards

### AI Extraction Module Architecture

**New Multi-Stage Pipeline** (Dec 2025):
- `src/services/extraction/`: Complete 5-stage extraction with 3-model ensemble
- `src/services/extraction/types.ts`: Core interfaces (ExtractionConfig, ExtractionResult, etc.)
- `src/services/extraction/config.ts`: NYT_VALIDATION constants, model configs
- `src/services/extraction/consensus.ts`: Confidence-weighted voting algorithm

**Legacy Modules** (still present for fallback):
- `src/services/extractionSchemas.ts`: Zod schemas for AI response validation
- `src/services/jsonParsingUtils.ts`: JSON parsing with LLM-specific fallback strategies
- `aiExtraction.ts` and `ensembleExtraction.ts`: Single/dual model extraction

**Pattern**: New extraction work should go in `src/services/extraction/`. Legacy modules remain for backward compatibility.

### LLM JSON Response Handling

**Problem**: Vision LLMs often return multiline strings incorrectly formatted:
```json
"shape": "....."
         "....."   // <- LLM splits across lines
```

**Solution**: Use `parseJSONWithFallback()` from `jsonParsingUtils.ts` which:
1. Attempts standard JSON.parse
2. Falls back to multiline field fixing (2, 3, 4+ line patterns)
3. Validates against Zod schema

**Anti-pattern**: Don't inline JSON parsing logic per-file. Always use shared utilities.

### Prompt Engineering Guidelines

**DRY for Prompts**: Prompt templates containing guidance (domino counts, grid rules) must follow DRY principles. Duplicated prompts drift over time.

**Magic Numbers**: Extract repeated values to shared constants or configuration:
```typescript
// Good: Centralized guidance
const DOMINO_COUNTS = { small: '7-8', medium: '9-12', large: '12-14+' };

// Bad: Hardcoded in multiple prompt strings
"Small puzzles: ~7-8"  // in file A
"Small puzzles: 7-8"   // in file B (subtle drift)
```

## Development Guidelines

### When Refactoring AI Extraction Code

**Verification Checklist**:
- [ ] Run `npx tsc --noEmit` after changes to catch import/type errors
- [ ] New extraction features go in `src/services/extraction/`
- [ ] Stage-specific prompts in `src/services/extraction/stages/`
- [ ] Validation logic in `src/services/extraction/validation/`
- [ ] Constants in `src/services/extraction/config.ts` (NYT_VALIDATION)

**Legacy modules still exist** for backward compatibility:
- `aiExtraction.ts`, `ensembleExtraction.ts`, `extractionSchemas.ts`, `jsonParsingUtils.ts`

### Multi-Stage Pipeline Testing

**Key files to verify after changes:**
```bash
# Type check the entire extraction module
npx tsc --noEmit --strict

# Verify exports match imports
grep -n "export" src/services/extraction/index.ts
```

**Consensus algorithm** (`consensus.ts`):
- Confidence threshold: 0.70 for retry trigger
- High confidence gap: 0.10 (if top > second by this, use top)
- Otherwise: majority vote

### Code Duplication Detection

**Files that commonly drift**:
- Legacy: `aiExtraction.ts` ↔ `ensembleExtraction.ts`
- New pipeline stages: prompts in `stages/*.ts` should share constants from `config.ts`

**Pattern**: When modifying extraction, check both legacy and new modules:
```bash
grep -rn "pattern" src/services/extraction/ src/services/aiExtraction.ts
```
