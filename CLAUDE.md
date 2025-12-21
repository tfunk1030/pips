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

The mobile app includes AI-powered puzzle extraction from screenshots using Claude Vision API:

#### Enhanced Extraction (Implemented Dec 2025)

- **Confidence Scores:** AI returns confidence metrics (0.0-1.0) for grid, regions, and constraints
- **Visual Feedback:** Confidence indicators in UI with color coding (green/orange/red)
- **Smart Prompts:** Explicit format examples and common mistake avoidance in prompts
- **Self-Verification:** Optional verification pass where Claude checks its own extraction (triggered on low confidence)
- **Better Error Messages:** Context-aware error messages with actionable next steps

#### Extraction Flow

1. **Pass 1: Board Structure** - Extracts grid dimensions, holes, regions, constraints with confidence scores
2. **Optional Verification** - If confidence < 90%, Claude verifies and corrects its extraction
3. **Pass 2: Dominoes** - Extracts domino tiles from reference tray with confidence score
4. **User Review** - Confidence indicators help users identify areas needing verification

#### Configuration

- API key stored in app Settings
- Verification pass optional (default: disabled, auto-enabled on low confidence)
- See `IMPLEMENTATION_SUMMARY.md` for detailed implementation notes

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

**Shared Modules** (as of Dec 2025 refactor):
- `src/services/extractionSchemas.ts`: Centralized Zod schemas for AI response validation
- `src/services/jsonParsingUtils.ts`: JSON parsing with LLM-specific fallback strategies
- `aiExtraction.ts` and `ensembleExtraction.ts` import from shared modules

**Pattern**: When adding new AI extraction schemas or parsing logic, add to shared modules first to prevent duplication drift.

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
- [ ] Check both `aiExtraction.ts` AND `ensembleExtraction.ts` for duplicated logic
- [ ] Verify prompt guidance is consistent between single-model and ensemble paths
- [ ] Zod schemas should be in `extractionSchemas.ts`, not inline

### Code Duplication Detection

**Files that commonly drift**:
- `aiExtraction.ts` ↔ `ensembleExtraction.ts` (prompts, schemas, parsing)

**Pattern**: When modifying one extraction file, grep for similar patterns in the other:
```bash
grep -n "pattern" src/services/aiExtraction.ts src/services/ensembleExtraction.ts
```
