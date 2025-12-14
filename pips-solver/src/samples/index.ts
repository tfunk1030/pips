/**
 * Sample puzzles for the app
 */

export const SAMPLE_PUZZLES = [
  {
    id: 'sample_2x2_tiny',
    name: 'Tiny 2x2 Test',
    yaml: `# Tiny 2x2 Pips puzzle for quick testing

id: sample_2x2_tiny
name: "Tiny 2x2 Test"
rows: 2
cols: 2
maxPip: 6
allowDuplicates: false

# Region layout
regions:
  - [0, 1]
  - [0, 1]

# Constraints
constraints:
  0:
    sum: 7
  1:
    sum: 5
`,
  },
  {
    id: 'sample_2x4_simple',
    name: 'Simple 2x4 Puzzle',
    yaml: `# Simple 2x4 Pips puzzle
# 4 dominoes needed to fill the grid

id: sample_2x4_simple
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
    sum: 6    # Region 0 must sum to 6
  1:
    sum: 8    # Region 1 must sum to 8
  2:
    sum: 5    # Region 2 must sum to 5
  3:
    all_equal: true  # Region 3 must have all equal values
`,
  },
  {
    id: 'sample_3x4_medium',
    name: 'Medium 3x4 Puzzle',
    yaml: `# Medium 3x4 Pips puzzle with various constraints

id: sample_3x4_medium
name: "Medium 3x4 Puzzle"
rows: 3
cols: 4
maxPip: 6
allowDuplicates: false

# Region layout
regions:
  - [0, 0, 1, 1]
  - [0, 2, 2, 1]
  - [3, 3, 3, 4]

# Constraints
constraints:
  0:
    sum: 9
  1:
    sum: 12
    size: 3
  2:
    op: "<"
    value: 4
  3:
    all_equal: true
  4:
    op: ">"
    value: 3
`,
  },
];
