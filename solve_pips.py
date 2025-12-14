# solve_pips.py
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import yaml

Coord = Tuple[int, int]  # (row, col)
Domino = Tuple[int, int]


@dataclass(frozen=True)
class Constraint:
    type: str  # "sum" or "all_equal"
    op: Optional[str] = None  # for sum: ==, !=, <, >
    value: Optional[int] = None


def parse_ascii_maps(shape_str: str, regions_str: str) -> Tuple[Set[Coord], Dict[Coord, str]]:
    shape_lines = [line.rstrip("\n") for line in shape_str.splitlines() if line.strip() != ""]
    region_lines = [line.rstrip("\n") for line in regions_str.splitlines() if line.strip() != ""]
    if len(shape_lines) != len(region_lines):
        raise ValueError("shape and regions must have same number of lines")

    cells: Set[Coord] = set()
    cell_region: Dict[Coord, str] = {}

    for r, (sline, rline) in enumerate(zip(shape_lines, region_lines)):
        if len(sline) != len(rline):
            raise ValueError(f"Line length mismatch at row {r}: shape vs regions")
        for c, (ch_s, ch_r) in enumerate(zip(sline, rline)):
            if ch_s == ".":
                if ch_r == "#" or ch_r == "." or ch_r == " ":
                    raise ValueError(f"Region label missing at ({r},{c}); got '{ch_r}'")
                cells.add((r, c))
                cell_region[(r, c)] = ch_r
            elif ch_s == "#":
                # no cell; region map should match
                continue
            else:
                raise ValueError(f"Invalid char in shape at ({r},{c}): '{ch_s}' (use '.' or '#')")

    return cells, cell_region


def build_adjacency(cells: Set[Coord]) -> Dict[Coord, List[Coord]]:
    adj: Dict[Coord, List[Coord]] = {cell: [] for cell in cells}
    for (r, c) in cells:
        for nr, nc in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]:
            if (nr, nc) in cells:
                adj[(r, c)].append((nr, nc))
    return adj


def region_cells(cell_region: Dict[Coord, str]) -> Dict[str, List[Coord]]:
    out: Dict[str, List[Coord]] = {}
    for coord, rid in cell_region.items():
        out.setdefault(rid, []).append(coord)
    return out


def check_constraint_partial(
    rid: str,
    coords: List[Coord],
    values: Dict[Coord, Optional[int]],
    cons: Constraint,
    pip_min: int,
    pip_max: int,
) -> bool:
    """Conservative partial checking: prune only when impossible."""
    filled = [values[c] for c in coords if values[c] is not None]
    unfilled_count = sum(1 for c in coords if values[c] is None)

    if cons.type == "all_equal":
        if not filled:
            return True
        first = filled[0]
        return all(v == first for v in filled)

    if cons.type != "sum":
        raise ValueError(f"Unknown constraint type: {cons.type}")

    assert cons.op is not None and cons.value is not None
    s = sum(filled)
    target = cons.value

    # bounds for remaining cells (conservative, ignores domino inventory)
    min_possible = s + unfilled_count * pip_min
    max_possible = s + unfilled_count * pip_max

    if cons.op == "==":
        return (min_possible <= target <= max_possible)
    if cons.op == "!=":
        # only fail when fully assigned and equals
        if unfilled_count == 0:
            return s != target
        return True
    if cons.op == "<":
        # If even the minimum already violates (>= target), impossible
        # For "< target": must end strictly below target
        if min_possible >= target:
            return False
        # If fully assigned, enforce strict
        if unfilled_count == 0:
            return s < target
        return True
    if cons.op == ">":
        # For "> target": must end strictly above target
        if max_possible <= target:
            return False
        if unfilled_count == 0:
            return s > target
        return True

    raise ValueError(f"Unknown op: {cons.op}")


def all_constraints_ok(
    region_map: Dict[str, List[Coord]],
    constraints: Dict[str, Constraint],
    values: Dict[Coord, Optional[int]],
    pip_min: int,
    pip_max: int,
    touched_regions: Optional[Set[str]] = None,
) -> bool:
    regs = touched_regions if touched_regions is not None else set(region_map.keys())
    for rid in regs:
        if rid not in constraints:
            raise ValueError(f"Region '{rid}' has no constraint entry")
        if not check_constraint_partial(
            rid, region_map[rid], values, constraints[rid], pip_min, pip_max
        ):
            return False
    return True


def pick_next_cell(cells: List[Coord], values: Dict[Coord, Optional[int]], adj: Dict[Coord, List[Coord]]) -> Optional[Coord]:
    """MRV-ish: pick an unfilled cell with fewest unfilled neighbors."""
    unfilled = [c for c in cells if values[c] is None]
    if not unfilled:
        return None
    def score(c: Coord) -> int:
        return sum(1 for n in adj[c] if values[n] is None)
    return min(unfilled, key=score)


def solve(
    cells: List[Coord],
    adj: Dict[Coord, List[Coord]],
    cell_region: Dict[Coord, str],
    region_map: Dict[str, List[Coord]],
    constraints: Dict[str, Constraint],
    dominoes: List[Domino],
    pip_min: int,
    pip_max: int,
):
    # state
    values: Dict[Coord, Optional[int]] = {c: None for c in cells}
    used: List[bool] = [False] * len(dominoes)

    # Pre-group identical dominoes so we don't try symmetric duplicates as much
    # We'll just keep list as is; duplicates are okay with used mask.

    def backtrack() -> bool:
        cell = pick_next_cell(cells, values, adj)
        if cell is None:
            return True  # solved

        # choose a neighbor to pair with
        for nb in adj[cell]:
            if values[nb] is not None:
                continue

            # try each remaining domino, both orientations
            for i, (a, b) in enumerate(dominoes):
                if used[i]:
                    continue

                for (va, vb) in [(a, b), (b, a)] if a != b else [(a, b)]:
                    # place
                    values[cell] = va
                    values[nb] = vb
                    used[i] = True

                    touched = {cell_region[cell], cell_region[nb]}
                    if all_constraints_ok(region_map, constraints, values, pip_min, pip_max, touched_regions=touched):
                        if backtrack():
                            return True

                    # undo
                    values[cell] = None
                    values[nb] = None
                    used[i] = False

            # If pairing with this neighbor fails, try a different neighbor
        return False

    ok = backtrack()
    return ok, values


def render_solution(shape_str: str, values: Dict[Coord, Optional[int]]) -> str:
    shape_lines = [line.rstrip("\n") for line in shape_str.splitlines() if line.strip() != ""]
    out_lines = []
    for r, line in enumerate(shape_lines):
        row_chars = []
        for c, ch in enumerate(line):
            if ch == ".":
                v = values.get((r, c), None)
                row_chars.append(str(v) if v is not None else "?")
            else:
                row_chars.append("#")
        out_lines.append(" ".join(row_chars))
    return "\n".join(out_lines)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_file", help="Path to pips_puzzle.yaml")
    args = ap.parse_args()

    with open(args.yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    pips = data["pips"]
    pip_min = int(pips["pip_min"])
    pip_max = int(pips["pip_max"])

    dom = data["dominoes"]
    unique = bool(dom.get("unique", True))
    tiles_raw = dom["tiles"]
    dominoes: List[Domino] = [(int(x), int(y)) for x, y in tiles_raw]

    board = data["board"]
    shape = board["shape"]
    regions = board["regions"]

    cells_set, cell_region = parse_ascii_maps(shape, regions)
    cells = sorted(list(cells_set))
    adj = build_adjacency(cells_set)

    rmap = region_cells(cell_region)

    constraints_raw = data["region_constraints"]
    constraints: Dict[str, Constraint] = {}
    for rid, obj in constraints_raw.items():
        ctype = obj["type"]
        if ctype == "sum":
            constraints[rid] = Constraint(type="sum", op=obj["op"], value=int(obj["value"]))
        elif ctype == "all_equal":
            constraints[rid] = Constraint(type="all_equal")
        else:
            raise ValueError(f"Unknown constraint type in YAML for {rid}: {ctype}")

    # sanity check: every region in map has a constraint
    for rid in rmap.keys():
        if rid not in constraints:
            raise ValueError(f"Region '{rid}' appears in board but missing from region_constraints")

    ok, values = solve(
        cells=cells,
        adj=adj,
        cell_region=cell_region,
        region_map=rmap,
        constraints=constraints,
        dominoes=dominoes,
        pip_min=pip_min,
        pip_max=pip_max,
    )

    if not ok:
        print("NO SOLUTION with current YAML (most likely a region label mismatch or domino list mismatch).")
        print("Filled grid (unknowns as '?'):\n")
        print(render_solution(shape, values))
        return

    print("SOLVED.\n")
    print("Pip grid:\n")
    print(render_solution(shape, values))


if __name__ == "__main__":
    main()
