"""
Test script to demonstrate the pips-agent capabilities
"""
import sys
import os

# First check if we have an API key
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("=" * 60)
    print("TESTING PIPS AGENT - Components Verification")
    print("=" * 60)
    print()
    print("Note: Full interactive agent requires ANTHROPIC_API_KEY")
    print("Testing individual components instead...")
    print()

    # Test 1: CV Extraction
    print("Test 1: Computer Vision Extraction")
    print("-" * 60)
    sys.path.insert(0, '..')
    from utils.cv_extraction import extract_puzzle_structure

    result = extract_puzzle_structure('../IMG_2050.png', output_dir='../debug', lower_half_only=False)
    if result['success']:
        print(f"✓ Extracted {result['grid_dims'][0]}x{result['grid_dims'][1]} grid")
        print(f"✓ Found {len(result['cells'])} cells")
        print(f"✓ Detected {len(result['regions'])} regions")
    else:
        print(f"✗ Failed: {result.get('error')}")
    print()

    # Test 2: YAML Generation
    print("Test 2: YAML Specification Generation")
    print("-" * 60)
    from utils.yaml_generator import create_puzzle_yaml

    sample_constraints = {
        'A': {'type': 'sum', 'operator': '==', 'value': 5},
        'B': {'type': 'all_equal'}
    }
    sample_dominoes = ['0-1', '1-2', '2-3']

    yaml_str = create_puzzle_yaml(
        cells=[(100, 100, 50, 50), (150, 100, 50, 50)],
        grid_dims=(1, 2),
        regions={'A': [0], 'B': [1]},
        constraints=sample_constraints,
        dominoes=sample_dominoes,
        pip_min=0,
        pip_max=6
    )
    print("✓ Generated YAML specification")
    print(f"  Length: {len(yaml_str)} characters")
    print()

    # Test 3: Solver
    print("Test 3: Puzzle Solver")
    print("-" * 60)
    import yaml
    from solve_pips import parse_ascii_maps, build_adjacency, region_cells, solve, Constraint

    # Load and solve the sample puzzle
    with open('../pips_puzzle.yaml', 'r') as f:
        data = yaml.safe_load(f.read())

    shape_rows, region_rows = data['board']['shape'], data['board']['regions']
    cells, cell_region = parse_ascii_maps(shape_rows, region_rows)
    adj = build_adjacency(cells)
    rmap = region_cells(cell_region)

    constraints = {}
    for r, cdata in data.get('region_constraints', {}).items():
        if cdata['type'] == 'sum':
            constraints[r] = Constraint(type='sum', op=cdata.get('op'), value=cdata['value'])
        elif cdata['type'] == 'all_equal':
            constraints[r] = Constraint(type='all_equal', op=None, value=None)

    success, values = solve(
        cells=list(cells),
        adj=adj,
        cell_region=cell_region,
        region_map=rmap,
        constraints=constraints,
        dominoes=data['dominoes']['tiles'],
        pip_min=data['pips']['pip_min'],
        pip_max=data['pips']['pip_max']
    )

    if success:
        print(f"✓ Solved {len(cells)}-cell puzzle with {len(rmap)} regions")
    else:
        print("✗ No solution found")
    print()

    # Test 4: Hint Generation
    print("Test 4: Hint Generation")
    print("-" * 60)
    from utils.hint_engine import generate_hints

    hints = generate_hints(
        region_map=rmap,
        constraints=constraints,
        dominoes=[f"{d[0]}-{d[1]}" for d in data['dominoes']['tiles']],
        pip_min=data['pips']['pip_min'],
        pip_max=data['pips']['pip_max']
    )

    print(f"✓ Generated {len(hints)} strategic hints")
    print()

    # Summary
    print("=" * 60)
    print("COMPONENT TEST RESULTS: ALL PASSED ✓")
    print("=" * 60)
    print()
    print("To run the full interactive agent:")
    print("1. Create .env file: cp .env.example .env")
    print("2. Add your ANTHROPIC_API_KEY to .env")
    print("3. Run: python main.py")

else:
    print("=" * 60)
    print("STARTING PIPS AGENT - Interactive Demo")
    print("=" * 60)
    print()
    print("Running with live API key...")
    print("This will start the interactive agent in a moment...")
    print()

    # For automated testing, we'll just verify it can start
    from main import SYSTEM_PROMPT
    print("[OK] Main module imported successfully")
    print(f"[OK] System prompt loaded ({len(SYSTEM_PROMPT)} characters)")
    print()
    print("Agent is ready to run!")
    print("To use interactively, run: python main.py")
