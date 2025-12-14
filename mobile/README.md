# Pips CSP (Expo / React Native)

An **offline** mobile app that:
- Parses a Pips puzzle spec (YAML/JSON)
- Solves it on-device as a **CSP** (chunked async search so UI stays responsive)
- Validates the solver output with an independent validator
- Renders the grid + domino pairings using `react-native-svg` (pan/zoom)

## Run

```bash
npm install
npm run start
```

## Core flow

1. Open **Library**
2. Tap **Load Sample** (or **Paste YAML**)
3. Tap **Solve**
4. App shows:
   - **Pip grid** + **domino links**
   - **Solver stats** (nodes/backtracks/prunes/time/yields)
   - **Validation report** (regions + domino usage)
   - If unsat: a **human-readable explanation** (dead-end / region bound contradiction, etc.)

## Puzzle spec

The parser supports:
- **Formal schema** (recommended)
- **Legacy ASCII schema** (compatible with the `nyt_sample.yaml` in this repo)

### Formal schema (recommended)

`grid.cells` is optional. Use `.` for a cell and `#` for “no cell” (holes).

```yaml
grid:
  rows: 2
  cols: 2
  cells: |
    ..
    ..
regions: |
  AA
  BB
regionConstraints:
  A: { sum: 1 }
  B: { sum: 2, all_equal: true }
dominoes:
  maxPip: 1
  unique: true
```

**Region constraints supported**
- `sum: N` (shorthand for `op: "=" value: N`)
- `op: "=" | "<" | ">" | "≠"` with `value: N` (applies to the **region sum**)
- `all_equal: true`
- `size: k` (sanity check)

**Domino rules**
- Orthogonal adjacency only
- Each cell belongs to exactly one domino (perfect matching)
- Domino domain defaults to a full double-`maxPip` set unless `dominoes.tiles` is provided

### Legacy ASCII schema

See `src/samples/nyt_sample.yaml` (also embedded as a button in the app).

## Solver output format

The solver returns:
- `solution.gridPips[r][c]` (with `-1` for missing cells)
- `solution.mateCellIdByCellId[cellId]`
- `solution.dominoes[]` (endpoints + pip values + domino key)
- `stats` (nodes/backtracks/prunes/time/yields)

## Architecture

Required layout is under `src/`:

```
src/
  app/
    screens/         // Library, Editor, Puzzle, Settings
    components/      // SVG grid + report views
    navigation/      // React Navigation stack
  model/
    types.ts
    parser.ts        // YAML/JSON -> raw puzzle
    normalize.ts     // region maps, adjacency, domino domain
  solver/
    solver.ts        // CSP engine (MRV + forward checking + yielding)
    heuristics.ts
    propagate.ts
    explain.ts
  validator/
    validateSpec.ts
    validateSolution.ts
  storage/
    puzzles.ts       // AsyncStorage
  samples/
    sample1.yaml
    nyt_sample.yaml
    samples.ts       // embedded sample strings for Metro
```



