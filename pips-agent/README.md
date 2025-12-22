# Pips Puzzle Agent

A Python Agent SDK application for analyzing and solving NYT Pips puzzles from screenshots using Claude Code.

## Features

- 📸 **Screenshot Analysis**: Extract puzzle structure from images using computer vision
- 🔍 **OCR Constraint Detection**: Automatically read constraint text from puzzles
- 🧩 **CSP Solver**: Solve puzzles completely using constraint satisfaction
- 💡 **Strategic Hints**: Get educational hints without spoiling the solution
- 🤖 **Interactive Agent**: Conversational interface powered by Claude

## Prerequisites

- Python 3.10+
- Tesseract OCR (for constraint detection)
  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
  - Mac: `brew install tesseract`
  - Linux: `sudo apt-get install tesseract-ocr`

## Installation

1. Install dependencies:

```bash
cd pips-agent
pip install -r requirements.txt
```

2. Set up your API key:

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Anthropic API key
# Get your key from: https://console.anthropic.com/
```

3. (Optional) If Tesseract is not in your PATH, set `TESSERACT_CMD` in `.env`

## Usage

Start the interactive agent:

```bash
python main.py
```

### Example Session

```
You: Analyze ../IMG_2050.png

Claude: I'll analyze your puzzle screenshot...
[Extracts grid structure, detects regions]

I've detected a 6x5 grid with 15 cells and 5 color-coded regions.
Let me try to read the constraint text...

[Runs OCR, detects constraints with confidence scores]

Region A: "sum = 12" (confidence: 95%)
Region B: "all equal" (confidence: 89%)
...

Would you like me to solve it completely or provide hints?

You: Give me hints

Claude: [Provides strategic hints without spoiling the solution]

🎯 Start with Region B (all equal constraint)
- If all cells must be equal, you need matching dominoes
- Look for doubles in your tray (0-0, 1-1, etc.)
...
```

## How It Works

### 1. Computer Vision Extraction

Uses `extract_board_cells_gridlines.py` from the parent directory to:
- Detect grid lines via edge detection
- Identify individual cells
- Cluster cells into color-coded regions

### 2. OCR Constraint Detection

Uses Tesseract OCR to:
- Extract text from the screenshot
- Parse constraint patterns (sum, <, >, all equal, etc.)
- Map constraints to regions based on spatial proximity
- Provide confidence scores for each detection

### 3. YAML Specification Generation

Automatically generates YAML puzzle specifications compatible with `solve_pips.py`:
- ASCII shape and region grids
- Region constraints
- Domino tiles list
- Pip value ranges

### 4. CSP Solving

Uses the existing `solve_pips.py` solver:
- Backtracking with constraint propagation
- MRV (Minimum Remaining Values) heuristic
- Forward checking
- Validates all solutions

### 5. Hint Generation

Strategic hint engine that:
- Analyzes constraint difficulty
- Suggests starting regions
- Explains strategic approaches
- Helps with unsolvable puzzles

## Project Structure

```
pips-agent/
├── main.py                    # Entry point with ClaudeSDKClient
├── tools/                     # Custom MCP tools
│   ├── extract_puzzle.py      # CV extraction tool
│   ├── ocr_constraints.py     # OCR constraint detection
│   ├── generate_spec.py       # YAML generation
│   ├── solve_puzzle.py        # CSP solver wrapper
│   └── provide_hints.py       # Hint generation
├── utils/                     # Utility modules
│   ├── cv_extraction.py       # CV pipeline wrapper
│   ├── yaml_generator.py      # YAML spec generation
│   ├── ocr_helper.py          # OCR utilities
│   └── hint_engine.py         # Hint generation logic
├── requirements.txt
├── .env.example
└── README.md
```

## Agent Workflow

1. **Extract**: User provides screenshot → Agent extracts puzzle structure
2. **OCR**: Agent attempts OCR constraint detection (with confidence scores)
3. **Clarify**: Agent asks user for missing/low-confidence constraints
4. **Generate**: Agent creates YAML puzzle specification
5. **Ask**: Agent always asks: "Solve completely or provide hints?"
6. **Execute**: Based on user choice, either solves or provides strategic hints

## Integration with Existing Code

The agent integrates with existing Python scripts from the parent directory:

- `solve_pips.py` - CSP solver functions (imported directly)
- `extract_board_cells_gridlines.py` - CV extraction (imported directly)
- `pips_puzzle.yaml` - YAML format reference

No modifications to existing files required!

## Troubleshooting

### OCR Not Working

- Ensure Tesseract is installed and in PATH
- Or set `TESSERACT_CMD` in `.env` to the full path

### CV Extraction Fails

- Ensure screenshot has good contrast
- Puzzle should be in lower half of image (default)
- Check debug output in `debug/` directory

### Solver Says "No Solution"

- Verify constraints are correct
- Check domino tiles list matches puzzle tray
- Use hint mode to identify potential constraint conflicts

### Import Errors

Make sure you're running from the `pips-agent/` directory:

```bash
cd pips-agent
python main.py
```

## Development

To modify the agent:

1. **Add new tools**: Create tool files in `tools/` using `@tool` decorator
2. **Modify system prompt**: Edit `SYSTEM_PROMPT` in `main.py`
3. **Enhance utilities**: Update modules in `utils/` for better extraction/solving
4. **Test changes**: Use sample screenshots from `../IMG_2050.png`

## License

Part of the Pips puzzle solver project.
