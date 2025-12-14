# Pips Puzzle Solver

A mobile app for solving and validating NYT "Pips" puzzles using constraint satisfaction problem (CSP) techniques. Built with React Native + Expo (TypeScript).

## Features

- ✅ **Offline-capable** - Runs entirely on-device
- ✅ **CSP Solver** - Backtracking with constraint propagation, MRV heuristics, and forward checking
- ✅ **Full Validation** - Comprehensive validation of puzzle specs and solutions
- ✅ **Non-blocking** - Solver runs incrementally to keep UI responsive
- ✅ **Visual Grid Renderer** - Clean SVG-based grid with zoom/pan support
- ✅ **YAML/JSON Import** - Flexible puzzle input format
- ✅ **Puzzle Library** - Save and manage puzzles locally
- ✅ **Detailed Reports** - Validation reports and solver statistics

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
2. Import a puzzle (YAML) or load a sample
3. View the puzzle grid with region colors
4. Tap "Solve Puzzle" to run the CSP solver
5. View the solution with validation report

## Architecture

### Project Structure

```
src/
├── app/
│   ├── screens/          # UI screens
│   │   ├── HomeScreen.tsx
│   │   ├── PuzzleViewerScreen.tsx
│   │   ├── SolveScreen.tsx
│   │   └── SettingsScreen.tsx
│   ├── components/       # Reusable components
│   │   └── GridRenderer.tsx
│   └── navigation/       # Navigation setup
│       └── AppNavigator.tsx
├── model/                # Data models & parsing
│   ├── types.ts          # Core type definitions
│   ├── parser.ts         # YAML/JSON parser
│   └── normalize.ts      # Puzzle preprocessing
├── solver/               # CSP solver engine
│   ├── solver.ts         # Main backtracking algorithm
│   ├── heuristics.ts     # MRV and ordering
│   ├── propagate.ts      # Constraint propagation
│   └── explain.ts        # Unsat explanations
├── validator/            # Validation modules
│   ├── validateSpec.ts   # Puzzle spec validation
│   └── validateSolution.ts # Solution validation
├── storage/              # Local persistence
│   └── puzzles.ts        # AsyncStorage wrapper
└── samples/              # Sample puzzles
    ├── sample1.yaml
    ├── sample2.yaml
    ├── sample3.yaml
    └── index.ts
```

### Key Modules

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
name: "Simple 2x4 Puzzle"
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
    sum: 6         # Region 0 must sum to 6
  1:
    sum: 8         # Region 1 must sum to 8
  2:
    op: "<"        # All values in region 2 must be < 4
    value: 4
  3:
    all_equal: true  # All values in region 3 must be equal
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

Configure solver behavior via the Settings screen:

- **Max Pip Value**: Default maximum pip (0-N)
- **Allow Duplicates**: Enable/disable domino reuse
- **Find All Solutions**: Find one or all solutions
- **Iterations Per Tick**: Control solver speed vs UI responsiveness
- **Debug Level**: Console logging (0=off, 1=basic, 2=verbose)

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
