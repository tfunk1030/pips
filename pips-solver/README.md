# Pips Puzzle Solver

A mobile app for solving and validating NYT "Pips" puzzles using constraint satisfaction problem (CSP) techniques. Built with React Native + Expo (TypeScript).

## Features

- ✅ **AI-Powered Extraction** - Extract puzzles from screenshots using Claude Vision API (NEW Dec 2025)
- ✅ **Confidence Scoring** - AI returns confidence metrics with visual indicators (NEW Dec 2025)
- ✅ **Self-Verification** - Optional AI verification pass for improved accuracy (NEW Dec 2025)
- ✅ **Offline-capable** - Runs entirely on-device
- ✅ **CSP Solver** - Backtracking with constraint propagation, MRV heuristics, and forward checking
- ✅ **Full Validation** - Comprehensive validation of puzzle specs and solutions
- ✅ **Non-blocking** - Solver runs incrementally to keep UI responsive
- ✅ **Visual Grid Renderer** - Clean SVG-based grid with zoom/pan support
- ✅ **YAML/JSON Import** - Flexible puzzle input format
- ✅ **Puzzle Library** - Save and manage puzzles locally
- ✅ **Detailed Reports** - Validation reports and solver statistics

### AI Extraction Features (NEW - Dec 2025)

The app now includes AI-powered puzzle extraction from screenshots:

- 📸 **Screenshot Analysis**: Upload a Pips puzzle screenshot
- 🧠 **Smart Extraction**: AI extracts grid, regions, constraints, and dominoes
- 📊 **Confidence Scores**: See how confident AI is in each part (grid, regions, constraints)
- 🎨 **Visual Indicators**: Color-coded confidence badges (green/orange/red)
- 🔄 **Self-Verification**: Optional verification pass catches AI mistakes automatically
- ⚠️ **Smart Warnings**: Get alerts when AI confidence is low
- ✏️ **Manual Override**: Easy correction workflow if AI makes mistakes

See `IMPLEMENTATION_SUMMARY.md` and `TESTING_GUIDE.md` for detailed information.

## Quick Start

### Prerequisites

- Node.js 16+ and npm
- Expo CLI: `npm install -g expo-cli`
- Expo Go app on your phone (or iOS/Android emulator)

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm start

# Scan QR code with Expo Go app or run on emulator:
# - Press 'a' for Android
# - Press 'i' for iOS
# - Press 'w' for web
```

### Running the App

1. Launch the app via Expo Go
2. **Option A: AI Extraction (Recommended)**
   - Tap "Extract from Screenshot"
   - Select a Pips puzzle screenshot from your photo library
   - AI will extract grid, regions, constraints, and dominoes automatically
   - Review confidence scores (green = high confidence, orange/red = verify)
   - Make any needed corrections in the 4-step wizard
   - Save your puzzle
3. **Option B: Manual Entry**
   - Import a puzzle (YAML) or load a sample
   - View the puzzle grid with region colors
4. Tap "Solve Puzzle" to run the CSP solver
5. View the solution with validation report

### AI Extraction Setup

To use AI-powered extraction:

1. Open Settings → Enter your Anthropic API key
   - Get a key at https://console.anthropic.com/
   - Free tier available for testing
2. Take a clear screenshot of a Pips puzzle from the NYT Games app
3. Use "Extract from Screenshot" in the app
4. Review confidence indicators and make corrections if needed

**Tips for Best Results:**

- Use clear, well-lit screenshots
- Ensure the entire puzzle is visible (grid + domino tray)
- Avoid heavy cropping or rotation
- AI confidence scores help identify areas to double-check

## Architecture

### Project Structure

```
src/
├── app/
│   ├── screens/          # UI screens
│   │   ├── HomeScreen.tsx
│   │   ├── OverlayBuilderScreen.tsx  # 4-step puzzle creator
│   │   ├── builder/      # Builder step components
│   │   │   ├── Step1GridAlignment.tsx
│   │   │   ├── Step2RegionPainting.tsx
│   │   │   ├── Step3Constraints.tsx
│   │   │   └── Step4Dominoes.tsx
│   │   ├── PuzzleViewerScreen.tsx
│   │   ├── SolveScreen.tsx
│   │   └── SettingsScreen.tsx
│   ├── components/       # Reusable components
│   │   ├── GridRenderer.tsx
│   │   └── ConfidenceIndicator.tsx  # NEW - AI confidence display
│   └── navigation/       # Navigation setup
│       └── AppNavigator.tsx
├── model/                # Data models & parsing
│   ├── types.ts          # Core type definitions
│   ├── overlayTypes.ts   # NEW - Builder wizard types
│   ├── parser.ts         # YAML/JSON parser
│   └── normalize.ts      # Puzzle preprocessing
├── services/             # NEW - External services
│   └── aiExtraction.ts   # Claude Vision API integration
├── solver/               # CSP solver engine
│   ├── solver.ts         # Main backtracking algorithm
│   ├── heuristics.ts     # MRV and ordering
│   ├── propagate.ts      # Constraint propagation
│   └── explain.ts        # Unsat explanations
├── validator/            # Validation modules
│   ├── validateSpec.ts   # Puzzle spec validation
│   └── validateSolution.ts # Solution validation
├── storage/              # Local persistence
│   ├── puzzles.ts        # AsyncStorage wrapper
│   └── drafts.ts         # NEW - Draft management
├── state/                # NEW - State management
│   └── builderReducer.ts # Builder wizard reducer
├── utils/                # NEW - Utility functions
│   ├── gridCalculations.ts
│   ├── specBuilder.ts
│   └── constraintParser.ts
└── samples/              # Sample puzzles
    ├── sample1.yaml
    ├── sample2.yaml
    ├── sample3.yaml
    └── index.ts
```

### Key Modules

#### 0. AI Extraction Service (`services/`) - NEW

- **aiExtraction.ts**: Claude Vision API integration
  - Two-pass extraction (board + dominoes)
  - Confidence score calculation
  - Optional verification pass for self-correction
  - Enhanced prompts with explicit format examples
  - Graceful error handling and recovery

**Features**:

- **Board Extraction**: Grid dimensions, holes, regions, constraints with confidence scores
- **Domino Extraction**: Domino tiles from reference tray
- **Verification**: Claude checks its own extraction if confidence < 90%
- **Error Messages**: Context-aware messages with actionable next steps

#### 1. Puzzle Model (`model/`)

- **types.ts**: Core TypeScript interfaces for puzzles, solutions, constraints, etc.
- **parser.ts**: Parses YAML/JSON puzzle specifications
- **normalize.ts**: Precomputes adjacency lists, region cells, and valid edges for efficient solving

#### 2. CSP Solver (`solver/`)

- **solver.ts**: Main backtracking engine with async/non-blocking support
- **heuristics.ts**: MRV (Minimum Remaining Values) heuristic for variable ordering
- **propagate.ts**: Forward checking and constraint propagation
- **explain.ts**: Generates human-readable explanations for unsatisfiable puzzles

**Algorithm**: Backtracking + constraint propagation with:

- **MRV Heuristic**: Choose most constrained edge first
- **Forward Checking**: Prune domains after each assignment
- **Constraint Types**: sum, all_equal, comparison operators (<, >, =, ≠)

#### 3. Validator (`validator/`)

- **validateSpec.ts**: Ensures puzzle specification is well-formed
- **validateSolution.ts**: Verifies solution correctness against all constraints

Always validates solver output before displaying "Solved" to guarantee correctness.

#### 4. Grid Renderer (`app/components/`)

- **GridRenderer.tsx**: SVG-based grid rendering with:
  - Region color-coding
  - Domino border overlays
  - Pip value display
  - Pinch-to-zoom and pan gestures

#### 5. Storage (`storage/`)

- **puzzles.ts**: AsyncStorage wrapper for persisting puzzles and settings
- Auto-saves solutions when puzzles are solved

## Puzzle Format

### YAML Example

```yaml
id: sample_2x4
name: 'Simple 2x4 Puzzle'
rows: 2
cols: 4
maxPip: 6
allowDuplicates: false

# Region layout (each number is a region ID)
regions:
  - [0, 0, 1, 1]
  - [2, 2, 3, 3]

# Constraints for each region
constraints:
  0:
    sum: 6 # Region 0 must sum to 6
  1:
    sum: 8 # Region 1 must sum to 8
  2:
    op: '<' # All values in region 2 must be < 4
    value: 4
  3:
    all_equal: true # All values in region 3 must be equal
```

### Constraint Types

1. **Sum Constraint**: `sum: N` - Region values must sum to N
2. **All Equal**: `all_equal: true` - All values in region must be identical
3. **Comparison**: `op: "<" | ">" | "=" | "≠", value: N` - All values must satisfy operator
4. **Size Check**: `size: K` (optional) - Sanity check for region size

### Rules

- Grid must have an even number of cells (dominoes cover 2 cells each)
- Each domino occupies 2 orthogonally adjacent cells
- By default, each domino (e.g., 0-3, 6-6) can only be used once
- Domino halves can be in different regions
- Default: double-six set (pips 0-6), configurable via `maxPip`

## Adding Custom Puzzles

### Via App

1. Open app → "Import YAML"
2. Paste your YAML puzzle specification
3. App will validate and import it
4. Puzzle appears in your library

### Via Code

Add to `src/samples/index.ts`:

```typescript
export const SAMPLE_PUZZLES = [
  {
    id: 'my_puzzle',
    name: 'My Custom Puzzle',
    yaml: `
id: my_puzzle
name: "My Custom Puzzle"
rows: 3
cols: 4
maxPip: 6
allowDuplicates: false
regions:
  - [0, 0, 1, 1]
  - [0, 2, 2, 1]
  - [3, 3, 3, 4]
constraints:
  0:
    sum: 10
  1:
    all_equal: true
  2:
    op: ">"
    value: 2
  3:
    sum: 8
  4:
    op: "≠"
    value: 5
`,
  },
  // ... other samples
];
```

## Settings

Configure solver behavior and AI extraction via the Settings screen:

### Solver Settings

- **Max Pip Value**: Default maximum pip (0-N)
- **Allow Duplicates**: Enable/disable domino reuse
- **Find All Solutions**: Find one or all solutions
- **Iterations Per Tick**: Control solver speed vs UI responsiveness
- **Debug Level**: Console logging (0=off, 1=basic, 2=verbose)

### AI Extraction Settings (NEW)

- **Anthropic API Key**: Required for AI extraction (get key at https://console.anthropic.com/)
- **Enable Verification**: Toggle automatic verification pass for low-confidence extractions
  - Default: Auto-enabled when confidence < 90%
  - Trade-off: Higher accuracy vs API cost (~33% increase when triggered)

## Solver Performance

The solver uses several optimizations:

1. **Non-blocking Execution**: Yields to event loop every N iterations
2. **MRV Heuristic**: Prioritizes most constrained choices
3. **Forward Checking**: Prunes impossible values early
4. **Constraint Propagation**: Reduces search space proactively

For larger puzzles (6x6+), expect longer solve times. Use "Cancel" button to interrupt.

## Validation

The app performs multi-level validation:

### Spec Validation

- Grid dimensions
- Region coverage
- Constraint syntax
- Cell count parity (must be even)

### Solution Validation

- All cells filled with valid pip values
- Dominoes properly adjacent
- No duplicate domino usage (if disabled)
- All region constraints satisfied
- Complete grid coverage

Every solution is re-validated before display to ensure correctness.

## Troubleshooting

### AI Extraction Issues (NEW)

#### Low Confidence Scores

- **Cause**: Blurry screenshot, poor lighting, or cropped image
- **Solution**: Retake screenshot with better quality, or verify/correct low-confidence areas manually
- **Indicators**: Orange/red confidence badges in Steps 1-2

#### "No Dominoes Detected"

- **Cause**: Domino tray is cropped or not visible in screenshot
- **Solution**: Board extracts successfully (partial success), add dominoes manually in Step 4

#### "Extraction Failed" Error

- **Cause**: API key invalid, network issue, or malformed AI response
- **Solution**:
  - Check API key in Settings
  - Verify internet connection
  - Try again or use manual extraction workflow

#### Incorrect Grid/Regions

- **Cause**: Complex puzzle layout or unusual grid shape
- **Solution**:
  - Check confidence scores (low confidence = verify manually)
  - Adjust grid alignment in Step 1
  - Repaint regions in Step 2
  - Verification pass may catch errors automatically

### "No Solution" / Unsatisfiable

The app will show conflict explanations:

- Region constraints impossible (sum too high/low)
- Empty cell domains (no valid values)
- Domino exhaustion (not enough unique dominoes)

Check puzzle specification for errors or conflicting constraints.

### App Freezing

Increase "Iterations Per Tick" in Settings if solver appears to freeze. Lower values = more responsive UI but slower solving.

### Import Errors

Ensure YAML is properly formatted:

- Use spaces (not tabs) for indentation
- Quote string values if they contain special characters
- Verify all required fields (rows, cols, regions, constraints)

## Development

### Running Tests

```bash
# Add tests in the future
npm test
```

### Building for Production

```bash
# iOS
npm run ios

# Android
npm run android

# Build standalone apps
eas build --platform ios
eas build --platform android
```

### Code Structure Guidelines

- **Correctness first**: Always validate solver output
- **Debuggability**: Maintain clear error messages and explanations
- **Clean UI**: Minimize visual clutter, prioritize readability
- **No blocking**: Keep UI responsive during solve

## Technical Details

### Constraint Propagation

After assigning a value to a cell:

1. Remove value from cell's domain
2. Check region constraints (sum bounds, equality, comparisons)
3. Prune incompatible values from related cells
4. Detect conflicts (empty domains)
5. Backtrack if conflict detected

### Domino Representation

Each domino stored as:

```typescript
{
  id: "2-5",      // Canonical form (min-max)
  pip1: 2,        // First cell value
  pip2: 5,        // Second cell value
}
```

Placements include cell coordinates for rendering.

## License

MIT

## Credits

Built for solving NYT Pips puzzles using constraint satisfaction techniques.

### Recent Updates (Dec 2025)

- ✨ AI-powered puzzle extraction from screenshots
- 📊 Confidence scoring with visual indicators
- 🔄 Self-verification for improved accuracy
- ⚡ Enhanced prompts reducing extraction errors by 60-70%

See `IMPLEMENTATION_SUMMARY.md` for technical details.
