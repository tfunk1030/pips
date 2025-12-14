// NOTE: Metro doesn't load raw YAML files by default; we embed sample text for offline use.

export const SAMPLE_1_YAML = `grid:
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
`;

export const SAMPLE_NYT_YAML = `pips:
  pip_min: 0
  pip_max: 6

dominoes:
  unique: true
  tiles:
    - [6,1]
    - [6,2]
    - [6,0]
    - [6,3]
    - [6,4]
    - [5,3]
    - [2,2]
    - [2,1]
    - [1,0]
    - [1,3]
    - [0,4]
    - [3,4]
    - [4,4]
    - [4,2]
    - [3,3]

board:
  shape: |
    ##....####
    ##....####
    ....###..#
    ....##..##
    ##..###.##
    ##..###.##
    ##..######
    ##..######
  regions: |
    ##BBCC####
    ##HGFD####
    AAHG###MN#
    EEEI##OM##
    ##EI###M##
    ##LJ###M##
    ##KJ######
    ##KJ######

region_constraints:
  A: { type: sum, op: "==", value: 12 }
  B: { type: sum, op: "<",  value: 2 }
  C: { type: sum, op: "==", value: 10 }
  D: { type: sum, op: ">",  value: 2 }
  E: { type: all_equal }
  F: { type: sum, op: "==",  value: 2 }
  G: { type: sum, op: "==",  value: 2 }
  H: { type: sum, op: "==",  value: 6 }
  I: { type: sum, op: "==", value: 12 }
  J: { type: sum, op: "==",  value: 6 }
  K: { type: all_equal }
  L: { type: sum, op: "==",  value: 2 }
  M: { type: sum, op: "==",  value: 4 }
  N: { type: sum, op: "==",  value: 4 }
  O: { type: sum, op: ">",  value: 4 }
`;



