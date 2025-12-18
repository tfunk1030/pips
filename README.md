# Pips (NYT Pips assistant)

An offline-first Expo + React Native reference app for importing, validating, and solving NYT “Pips” domino puzzles using a constraint satisfaction approach. The solver runs entirely on-device and yields frequently so the UI stays responsive.

## Features
- YAML/JSON import screen with a built-in sample puzzle.
- Local library stored in AsyncStorage for offline use.
- React Native + SVG grid viewer with pan/zoom support and region coloring.
- CSP-style solver with MRV, forward checking, and region-bound propagation.
- Validation pipeline for both puzzle specs and solver outputs.
- Deterministic progress reporting and cancel-friendly async loop.

## Project structure
```
src/
  app/
    navigation/RootNavigator.tsx
    screens/ (Home, Editor, Viewer, Solve, Settings)
    components/PuzzleGrid.tsx
  model/
    types.ts
    parser.ts
    normalize.ts
  solver/
    solver.ts
    heuristics.ts
    propagate.ts
    explain.ts
  validator/
    validateSpec.ts
    validateSolution.ts
  storage/puzzles.ts
  samples/sample1.yaml
```

## Running
1. Install dependencies (requires Expo SDK 51):
   ```bash
   npm install
   ```
2. Start the Metro server:
   ```bash
   npm run start
   ```
3. Open the project in Expo Go or an emulator. All logic runs offline once bundled.

## Puzzle format
Example (`src/samples/sample1.yaml`):
```yaml
name: Tiny Example
rows: 2
cols: 2
regions:
  - [1, 1]
  - [2, 2]
regionConstraints:
  1:
    sum: 3
  2:
    sum: 3
maxPip: 3
allowDuplicates: false
```

### Supported fields
- `rows`, `cols`: grid dimensions
- `regions`: 2D array of region ids matching the grid
- `regionConstraints`: per-region rule: `sum`, `op` with value, or `all_equal`
- `maxPip`: highest pip value (default double-six)
- `allowDuplicates`: whether dominoes can repeat

## Solver and validator
- **solver.ts**: MRV backtracking with region-bound pruning and tick-based yielding. Produces grid pips, domino pairings, stats, and unsat explanations.
- **validateSpec.ts**: confirms region coverage and constraints are well-formed.
- **validateSolution.ts**: re-checks domino adjacency/uniqueness and region math before displaying a result.

## Adding puzzles
1. Tap **Import YAML** and paste a new puzzle, or drop a YAML file into `src/samples/` for bundling.
2. Save it to the library; it remains available offline.

## Debugging notes
- Solver progress is reported through callbacks and never blocks the UI thread.
- Validation always runs on solver outputs to avoid displaying inconsistent solutions.
