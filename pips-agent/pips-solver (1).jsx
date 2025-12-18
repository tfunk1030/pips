import React, { useState, useCallback, useRef } from 'react';
import { Upload, Lightbulb, Eye, Target, RotateCcw, Play, ChevronRight, ChevronLeft, Sparkles, Loader2, Check, X, Zap, Plus, Minus, Palette, Bug, Grid3X3, Camera, Edit3, Wand2 } from 'lucide-react';

// ═══════════════════════════════════════════════════════════════════════════════
// PIPS SOLVER ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

class PipsSolverEngine {
  constructor(puzzle, debug = false) {
    this.puzzle = puzzle;
    this.debug = debug;
    this.logs = [];
    this.grid = new Map();
    this.usedDominoes = new Set();
    this.adjacency = this.buildAdjacency();
    this.regionCells = this.buildRegionCells();
    this.cellToRegion = this.buildCellToRegion();
    for (const cell of puzzle.board.cells) this.grid.set(cell, null);
  }

  log(msg) { if (this.debug) this.logs.push(msg); }

  buildAdjacency() {
    const adj = new Map();
    for (const cell of this.puzzle.board.cells) {
      const [r, c] = cell.split(',').map(Number);
      const neighbors = [];
      for (const [dr, dc] of [[0, 1], [1, 0], [0, -1], [-1, 0]]) {
        const key = `${r + dr},${c + dc}`;
        if (this.puzzle.board.cells.has(key)) neighbors.push(key);
      }
      adj.set(cell, neighbors);
    }
    return adj;
  }

  buildRegionCells() {
    const rc = new Map();
    for (const [rid, coords] of this.puzzle.regions) {
      rc.set(rid, coords.map(([r, c]) => `${r},${c}`));
    }
    return rc;
  }

  buildCellToRegion() {
    const ctr = new Map();
    for (const [rid, coords] of this.puzzle.regions) {
      for (const [r, c] of coords) ctr.set(`${r},${c}`, rid);
    }
    return ctr;
  }

  solve() {
    this.log(`Starting solve: ${this.puzzle.dominoes.length} dominoes, ${this.puzzle.board.cells.size} cells`);
    const expectedCells = this.puzzle.dominoes.length * 2;
    if (this.puzzle.board.cells.size !== expectedCells) {
      return { placements: [], solved: false, grid: new Map(), logs: this.logs, 
        error: `Cell mismatch: ${this.puzzle.board.cells.size} cells but need ${expectedCells} for ${this.puzzle.dominoes.length} dominoes` };
    }
    const placements = [];
    const solved = this.backtrack(placements, 0);
    return { placements, solved, grid: new Map(this.grid), logs: this.logs };
  }

  backtrack(placements, depth) {
    const emptyCell = this.findEmptyCell();
    if (!emptyCell) return this.checkAllConstraintsFinal();

    const emptyNeighbors = (this.adjacency.get(emptyCell) || []).filter(n => this.grid.get(n) === null);
    if (emptyNeighbors.length === 0) return false;

    for (let di = 0; di < this.puzzle.dominoes.length; di++) {
      if (this.usedDominoes.has(di)) continue;
      const domino = this.puzzle.dominoes[di];

      for (const neighbor of emptyNeighbors) {
        const orientations = domino[0] === domino[1] ? [[domino[0], domino[1]]] : [[domino[0], domino[1]], [domino[1], domino[0]]];

        for (const [v1, v2] of orientations) {
          this.grid.set(emptyCell, v1);
          this.grid.set(neighbor, v2);
          this.usedDominoes.add(di);

          if (this.checkPartialConstraints([emptyCell, neighbor])) {
            const [r1, c1] = emptyCell.split(',').map(Number);
            const [r2, c2] = neighbor.split(',').map(Number);
            placements.push({ domino, dominoIndex: di, coords: [[r1, c1], [r2, c2]], values: [v1, v2] });
            if (this.backtrack(placements, depth + 1)) return true;
            placements.pop();
          }

          this.grid.set(emptyCell, null);
          this.grid.set(neighbor, null);
          this.usedDominoes.delete(di);
        }
      }
    }
    return false;
  }

  findEmptyCell() {
    let bestCell = null, minNeighbors = Infinity;
    for (const [cell, value] of this.grid) {
      if (value !== null) continue;
      const emptyNeighbors = (this.adjacency.get(cell) || []).filter(n => this.grid.get(n) === null).length;
      if (emptyNeighbors < minNeighbors) { minNeighbors = emptyNeighbors; bestCell = cell; }
    }
    return bestCell;
  }

  checkPartialConstraints(changedCells) {
    const affected = new Set();
    for (const cell of changedCells) {
      const rid = this.cellToRegion.get(cell);
      if (rid) affected.add(rid);
    }
    for (const rid of affected) if (!this.checkConstraintPartial(rid)) return false;
    return true;
  }

  checkConstraintPartial(regionId) {
    const constraint = this.puzzle.constraints.get(regionId);
    if (!constraint || constraint.type === 'any') return true;
    const cells = this.regionCells.get(regionId) || [];
    const values = []; let emptyCount = 0;
    for (const cell of cells) { const v = this.grid.get(cell); if (v === null) emptyCount++; else values.push(v); }
    const [pipMin, pipMax] = this.puzzle.pipRange;

    if (constraint.type === 'sum') {
      const sum = values.reduce((a, b) => a + b, 0);
      const minP = sum + emptyCount * pipMin, maxP = sum + emptyCount * pipMax;
      switch (constraint.op) {
        case '==': return minP <= constraint.value && maxP >= constraint.value;
        case '!=': return emptyCount > 0 || sum !== constraint.value;
        case '<': return minP < constraint.value;
        case '>': return maxP > constraint.value;
      }
    }
    if (constraint.type === 'all_equal') return values.length < 2 || values.every(v => v === values[0]);
    if (constraint.type === 'all_different') return values.length === new Set(values).size;
    return true;
  }

  checkAllConstraintsFinal() {
    for (const [rid, constraint] of this.puzzle.constraints) {
      if (!constraint || constraint.type === 'any') continue;
      const cells = this.regionCells.get(rid) || [];
      const values = cells.map(c => this.grid.get(c));
      if (constraint.type === 'sum') {
        const sum = values.reduce((a, b) => a + b, 0);
        switch (constraint.op) {
          case '==': if (sum !== constraint.value) return false; break;
          case '!=': if (sum === constraint.value) return false; break;
          case '<': if (sum >= constraint.value) return false; break;
          case '>': if (sum <= constraint.value) return false; break;
        }
      }
      if (constraint.type === 'all_equal' && !values.every(v => v === values[0])) return false;
      if (constraint.type === 'all_different' && values.length !== new Set(values).size) return false;
    }
    return true;
  }
}

function solvePips(puzzle, debug = false) { return new PipsSolverEngine(puzzle, debug).solve(); }

function createPuzzleFromJSON(json) {
  let shapeLines = typeof json.board.shape === 'string' ? json.board.shape.trim().split('\n') : json.board.shape;
  const cells = new Set();
  const rows = shapeLines.length;
  const cols = Math.max(...shapeLines.map(l => l.length));
  for (let r = 0; r < rows; r++) for (let c = 0; c < shapeLines[r].length; c++) if (shapeLines[r][c] === '.') cells.add(`${r},${c}`);

  let regionLines = typeof json.regions === 'string' ? json.regions.trim().split('\n') : json.regions;
  const regions = new Map();
  for (let r = 0; r < regionLines.length; r++) {
    for (let c = 0; c < regionLines[r].length; c++) {
      const ch = regionLines[r][c];
      if (ch !== '#' && ch !== ' ' && ch !== '.') {
        if (!regions.has(ch)) regions.set(ch, []);
        regions.get(ch).push([r, c]);
      }
    }
  }
  const constraints = new Map(Object.entries(json.constraints || {}));
  return { pipRange: json.pipRange || [0, 6], dominoes: json.dominoes, board: { rows, cols, cells }, regions, constraints };
}

// ═══════════════════════════════════════════════════════════════════════════════
// TWO-PASS EXTRACTION: Board first, then Dominoes with intense focus
// ═══════════════════════════════════════════════════════════════════════════════

const BOARD_EXTRACTION_PROMPT = `Analyze this Pips puzzle screenshot. Extract the BOARD, REGIONS, and CONSTRAINTS only (ignore dominoes for now).

STEP 1: COUNT THE GRID
- Count rows top to bottom
- Count columns left to right  
- Note any holes (missing/black cells)
- Say: "Grid is X rows by Y columns. Holes at: ..."

STEP 2: MAP REGIONS
- Each colored area is a region
- Assign letters A, B, C... to each distinct color
- Map every cell to its region letter
- Use '#' for holes, '.' for cells without a region
- Say: "Region A is [color], Region B is [color]..."

STEP 3: READ CONSTRAINTS
- Look for small badges/diamonds on each region showing:
  * Number (6, 8, etc.) = sum equals that number
  * "=" = all values must match
  * "X" = all values must differ
  * ">N" or "<N" = sum greater/less than N
  * Nothing = no constraint
- Say: "Region A shows 8, Region B shows =, ..."

Output JSON:
\`\`\`json
{
  "board": {"rows": N, "cols": N, "shape": "....\\n...."},
  "regions": "AABB\\nCCDD",
  "constraints": {"A": {"type": "sum", "op": "==", "value": 8}, "B": {"type": "all_equal"}}
}
\`\`\``;

const DOMINO_EXTRACTION_PROMPT = `Look at the DOMINO TRAY in this puzzle screenshot. Count the pips on each domino VERY carefully.

DOMINO TRAY LOCATION: Usually at the bottom of the screen, sometimes on the side. It shows a row of dominoes.

EACH DOMINO has TWO halves separated by a line. Count dots on each half:
- 0 dots = blank
- 1 dot = center only
- 2 dots = diagonal corners
- 3 dots = diagonal line
- 4 dots = four corners
- 5 dots = four corners + center
- 6 dots = two columns of three

COUNT OUT LOUD - go through each domino one by one:
"Domino 1: Left side has __ dots, right side has __ dots → [_, _]
Domino 2: Left side has __ dots, right side has __ dots → [_, _]
..."

Continue until you've counted ALL dominoes in the tray.

Then output ONLY the dominoes array:
\`\`\`json
{"dominoes": [[6,1], [3,3], [5,2], [0,4], [2,2]]}
\`\`\`

IMPORTANT: Count carefully. Look at each domino individually. Don't guess.`;

async function extractBoardFromImage(imageBase64, mediaType) {
  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 2048,
      messages: [{
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: mediaType, data: imageBase64 }},
          { type: 'text', text: BOARD_EXTRACTION_PROMPT }
        ]
      }]
    })
  });
  
  const data = await response.json();
  if (data.error) throw new Error(data.error.message);
  const text = data.content?.[0]?.text || '';
  const jsonMatch = text.match(/```json\n?([\s\S]*?)\n?```/) || text.match(/(\{[\s\S]*\})/);
  if (!jsonMatch) throw new Error('No JSON found');
  return { data: JSON.parse(jsonMatch[1] || jsonMatch[0]), reasoning: text.split('```')[0] };
}

async function extractDominoesFromImage(imageBase64, mediaType) {
  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 2048,
      messages: [{
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: mediaType, data: imageBase64 }},
          { type: 'text', text: DOMINO_EXTRACTION_PROMPT }
        ]
      }]
    })
  });
  
  const data = await response.json();
  if (data.error) throw new Error(data.error.message);
  const text = data.content?.[0]?.text || '';
  const jsonMatch = text.match(/```json\n?([\s\S]*?)\n?```/) || text.match(/(\{[\s\S]*\})/);
  if (!jsonMatch) throw new Error('No JSON found');
  return { data: JSON.parse(jsonMatch[1] || jsonMatch[0]), reasoning: text.split('```')[0] };
}

async function extractPuzzleFromImage(imageBase64, mediaType, onProgress) {
  // Pass 1: Board
  onProgress?.('Extracting grid, regions, constraints...');
  const boardResult = await extractBoardFromImage(imageBase64, mediaType);
  
  // Pass 2: Dominoes (separate focused call)
  onProgress?.('Counting dominoes...');
  const dominoResult = await extractDominoesFromImage(imageBase64, mediaType);
  
  return {
    puzzleData: {
      ...boardResult.data,
      dominoes: dominoResult.data.dominoes || []
    },
    reasoning: `=== BOARD ===\n${boardResult.reasoning}\n\n=== DOMINOES ===\n${dominoResult.reasoning}`
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// VISUAL COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const REGION_COLORS = ['#FF9800', '#009688', '#9C27B0', '#E91E63', '#4CAF50', '#2196F3', '#FF5722', '#607D8B', '#795548', '#00BCD4', '#8BC34A', '#FFC107'];

// ═══════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════════
// SMART OVERLAY BUILDER - AI crops board, no distortion
// ═══════════════════════════════════════════════════════════════════════════════

const CROP_PROMPT = `Analyze this Pips puzzle screenshot. Find the puzzle BOARD (colored grid of cells, NOT the dominoes).

Return JSON:
\`\`\`json
{"rows": <num>, "cols": <num>, "bounds": {"top": <% from top>, "left": <% from left>, "bottom": <% from top>, "right": <% from left>}}
\`\`\``;

async function analyzeBoard(imageBase64, mediaType) {
  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 500,
        messages: [{ role: 'user', content: [
          { type: 'image', source: { type: 'base64', media_type: mediaType, data: imageBase64 }},
          { type: 'text', text: CROP_PROMPT }
        ]}]
      })
    });
    const data = await response.json();
    const text = data.content?.[0]?.text || '';
    const jsonMatch = text.match(/```json\n?([\s\S]*?)\n?```/) || text.match(/(\{[\s\S]*\})/);
    if (jsonMatch) return JSON.parse(jsonMatch[1] || jsonMatch[0]);
  } catch (e) { console.log('Board analysis failed:', e); }
  return null;
}

async function cropImage(imageUrl, bounds) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      try {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const pad = 0.03; // 3% padding
        
        const left = Math.max(0, bounds.left / 100 - pad);
        const top = Math.max(0, bounds.top / 100 - pad);
        const right = Math.min(1, bounds.right / 100 + pad);
        const bottom = Math.min(1, bounds.bottom / 100 + pad);
        
        const x = left * img.width;
        const y = top * img.height;
        const w = (right - left) * img.width;
        const h = (bottom - top) * img.height;
        
        canvas.width = Math.max(1, Math.round(w));
        canvas.height = Math.max(1, Math.round(h));
        ctx.drawImage(img, x, y, w, h, 0, 0, canvas.width, canvas.height);
        
        const result = canvas.toDataURL('image/png');
        resolve(result);
      } catch (e) {
        console.log('Canvas crop error:', e);
        resolve(imageUrl);
      }
    };
    img.onerror = () => resolve(imageUrl);
    img.src = imageUrl;
  });
}

const OverlayBuilder = ({ imageUrl, onComplete, onCancel, savedState }) => {
  const [rows, setRows] = useState(savedState?.rows || 4);
  const [cols, setCols] = useState(savedState?.cols || 5);
  const [displayImage, setDisplayImage] = useState(imageUrl);
  // Simple rectangle bounds (percentages)
  const [bounds, setBounds] = useState(savedState?.bounds || { left: 10, top: 10, right: 90, bottom: 80 });
  const [step, setStep] = useState(savedState?.step || 0);
  const [holes, setHoles] = useState(savedState?.holes || {});
  const [regions, setRegions] = useState(savedState?.regions || {});
  const [currentRegion, setCurrentRegion] = useState(0);
  const [constraints, setConstraints] = useState(savedState?.constraints || {});
  const [dominoes, setDominoes] = useState(savedState?.dominoes || []);
  const [quickDominoes, setQuickDominoes] = useState('');
  const [isPainting, setIsPainting] = useState(false);
  const [draggingEdge, setDraggingEdge] = useState(null); // 'left', 'right', 'top', 'bottom'
  const containerRef = useRef(null);

  React.useEffect(() => {
    if (savedState?.step > 0) { 
      setDisplayImage(savedState.displayImage || imageUrl);
      if (savedState.bounds) setBounds(savedState.bounds);
      setStep(savedState.step); 
      return; 
    }
    setDisplayImage(imageUrl);
    
    (async () => {
      try {
        const base64 = imageUrl.split(',')[1];
        const mediaType = imageUrl.split(';')[0].split(':')[1];
        const result = await analyzeBoard(base64, mediaType);
        if (result) {
          setRows(result.rows || 4);
          setCols(result.cols || 5);
          if (result.bounds) {
            setBounds({
              left: result.bounds.left,
              top: result.bounds.top,
              right: result.bounds.right,
              bottom: result.bounds.bottom
            });
          }
        }
      } catch (e) { console.log('Analysis failed:', e); }
      setStep(1);
    })();
  }, [imageUrl, savedState]);

  const saveState = useCallback(async () => {
    try { await window.storage?.set('pips-draft', JSON.stringify({ rows, cols, displayImage, bounds, step, holes, regions, constraints, dominoes, imageUrl, timestamp: Date.now() })); } catch (e) {}
  }, [rows, cols, displayImage, bounds, step, holes, regions, constraints, dominoes, imageUrl]);
  React.useEffect(() => { if (step > 0) saveState(); }, [step, holes, regions, constraints, dominoes, bounds, saveState]);

  // Edge dragging
  const handleMouseMove = (e) => {
    if (!draggingEdge || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
    const y = Math.max(0, Math.min(100, ((e.clientY - rect.top) / rect.height) * 100));
    
    setBounds(prev => {
      if (draggingEdge === 'left') return { ...prev, left: Math.min(x, prev.right - 10) };
      if (draggingEdge === 'right') return { ...prev, right: Math.max(x, prev.left + 10) };
      if (draggingEdge === 'top') return { ...prev, top: Math.min(y, prev.bottom - 10) };
      if (draggingEdge === 'bottom') return { ...prev, bottom: Math.max(y, prev.top + 10) };
      return prev;
    });
  };
  
  const handleTouchMove = (e) => {
    if (!draggingEdge || !containerRef.current) return;
    const touch = e.touches[0];
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(100, ((touch.clientX - rect.left) / rect.width) * 100));
    const y = Math.max(0, Math.min(100, ((touch.clientY - rect.top) / rect.height) * 100));
    
    setBounds(prev => {
      if (draggingEdge === 'left') return { ...prev, left: Math.min(x, prev.right - 10) };
      if (draggingEdge === 'right') return { ...prev, right: Math.max(x, prev.left + 10) };
      if (draggingEdge === 'top') return { ...prev, top: Math.min(y, prev.bottom - 10) };
      if (draggingEdge === 'bottom') return { ...prev, bottom: Math.max(y, prev.top + 10) };
      return prev;
    });
  };

  // Get cell position from rectangular bounds
  const getCellStyle = (r, c) => {
    const cellWidth = (bounds.right - bounds.left) / cols;
    const cellHeight = (bounds.bottom - bounds.top) / rows;
    return {
      position: 'absolute',
      left: `${bounds.left + c * cellWidth}%`,
      top: `${bounds.top + r * cellHeight}%`,
      width: `${cellWidth}%`,
      height: `${cellHeight}%`
    };
  };

  const handleCell = (r, c, isDrag = false) => {
    const key = `${r},${c}`;
    if (step === 1) setHoles(prev => ({ ...prev, [key]: !prev[key] }));
    else if (step === 2 && !holes[key]) setRegions(prev => ({ ...prev, [key]: currentRegion }));
  };

  const getUniqueRegions = () => {
    const unique = new Set();
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) if (!holes[`${r},${c}`]) unique.add(regions[`${r},${c}`] ?? 0);
    return [...unique].sort((a, b) => a - b);
  };

  const cellCount = rows * cols - Object.values(holes).filter(Boolean).length;
  const isValid = cellCount === dominoes.length * 2 && cellCount > 0;

  const cycleDomino = (idx, half) => {
    const d = [...dominoes]; d[idx] = [...d[idx]]; d[idx][half] = (d[idx][half] + 1) % 7; setDominoes(d);
  };
  const parseQuickDominoes = () => {
    const t = quickDominoes.replace(/[^0-6]/g, ''), pairs = [];
    for (let i = 0; i < t.length - 1; i += 2) pairs.push([+t[i], +t[i+1]]);
    if (pairs.length) setDominoes(pairs); setQuickDominoes('');
  };
  const autoFill = () => setDominoes(Array(Math.floor(cellCount/2)).fill([0,0]));

  const buildPuzzle = async () => {
    const shape = Array(rows).fill(0).map((_,r) => Array(cols).fill(0).map((_,c) => holes[`${r},${c}`] ? '#' : '.').join('')).join('\n');
    const regStr = Array(rows).fill(0).map((_,r) => Array(cols).fill(0).map((_,c) => holes[`${r},${c}`] ? '#' : String.fromCharCode(65+(regions[`${r},${c}`]??0))).join('')).join('\n');
    try { await window.storage?.delete('pips-draft'); } catch(e) {}
    onComplete({ pipRange: [0,6], board: { rows, cols, shape }, regions: regStr, constraints, dominoes });
  };

  const titles = ['Analyzing...', '1. Align Grid', '2. Regions', '3. Constraints', '4. Dominoes'];
  const hints = ['Finding puzzle...', 'Drag edges to fit grid. Tap holes. Adjust rows/cols.', 'Pick color, tap/drag to paint', 'Set each region constraint', 'Type "61 33 36" or tap'];

  return (
    <div className="space-y-2 text-white" onMouseUp={() => { setDraggingEdge(null); setIsPainting(false); }} onMouseLeave={() => { setDraggingEdge(null); setIsPainting(false); }}>
      <div className="flex items-center justify-between">
        <button onClick={onCancel} className="text-purple-300 text-sm flex items-center"><ChevronLeft className="w-4 h-4"/>Back</button>
        <span className="font-medium text-sm">{titles[step]}</span>
        <span className="text-purple-300 text-xs">{step}/4</span>
      </div>
      <div className="flex gap-1">{[1,2,3,4].map(s => <div key={s} className={`flex-1 h-1 rounded ${step===s?'bg-purple-500':step>s?'bg-green-500':'bg-white/20'}`}/>)}</div>
      <p className="text-xs text-purple-300 text-center">{hints[step]}</p>

      {step === 0 && (
        <div className="relative rounded-xl overflow-hidden bg-gray-900">
          <img src={displayImage} alt="Puzzle" className="w-full h-auto opacity-50" style={{maxHeight:'45vh'}}/>
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <Loader2 className="w-8 h-8 text-purple-500 animate-spin mx-auto"/>
              <p className="text-purple-300 text-sm mt-2">Finding puzzle board...</p>
            </div>
          </div>
        </div>
      )}

      {step > 0 && displayImage && (
        <div 
          ref={containerRef}
          className="relative rounded-xl overflow-hidden bg-gray-900"
          onMouseMove={handleMouseMove}
          onMouseUp={() => { setDraggingEdge(null); setIsPainting(false); }}
          onMouseLeave={() => { setDraggingEdge(null); setIsPainting(false); }}
          onTouchMove={handleTouchMove}
          onTouchEnd={() => { setDraggingEdge(null); setIsPainting(false); }}
        >
          <img src={displayImage} alt="Puzzle" className="w-full h-auto" style={{maxHeight:'45vh'}}/>
          
          {/* Grid outline - thick prominent border */}
          {step === 1 && (
            <div 
              className="absolute pointer-events-none"
              style={{
                left: `${bounds.left}%`,
                top: `${bounds.top}%`,
                width: `${bounds.right - bounds.left}%`,
                height: `${bounds.bottom - bounds.top}%`,
                border: '3px solid white',
                boxShadow: '0 0 8px rgba(0,0,0,0.8)',
                zIndex: 15
              }}
            />
          )}
          
          {/* Grid lines - visible in steps 1, 2, 3 */}
          {step >= 1 && step <= 3 && (
            <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{zIndex: 20}}>
              {/* Vertical lines */}
              {Array(cols + 1).fill(0).map((_, c) => {
                const x = bounds.left + (bounds.right - bounds.left) * c / cols;
                return <line key={`v${c}`} x1={`${x}%`} y1={`${bounds.top}%`} x2={`${x}%`} y2={`${bounds.bottom}%`} 
                  stroke="white" strokeWidth="2"/>;
              })}
              {/* Horizontal lines */}
              {Array(rows + 1).fill(0).map((_, r) => {
                const y = bounds.top + (bounds.bottom - bounds.top) * r / rows;
                return <line key={`h${r}`} x1={`${bounds.left}%`} y1={`${y}%`} x2={`${bounds.right}%`} y2={`${y}%`} 
                  stroke="white" strokeWidth="2"/>;
              })}
            </svg>
          )}
          
          {/* Grid cells - VERY LOW opacity overlays */}
          {step >= 2 && Array(rows).fill(0).map((_,r) => Array(cols).fill(0).map((_,c) => {
            const key = `${r},${c}`, isHole = holes[key], ridx = regions[key] ?? 0;
            const cellStyle = getCellStyle(r, c);
            return (
              <div key={key}
                style={{
                  ...cellStyle,
                  backgroundColor: isHole ? 'rgba(0,0,0,0.7)' : REGION_COLORS[ridx],
                  opacity: isHole ? 1 : 0.2,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  zIndex: 5
                }}
                onClick={() => handleCell(r,c)}
                onMouseDown={() => step===2 && setIsPainting(true)}
                onMouseEnter={() => isPainting && step===2 && !holes[key] && setRegions(p => ({...p,[key]:currentRegion}))}
              >
                {isHole && <X className="w-4 h-4 text-red-400"/>}
                {!isHole && <span className="text-white text-sm font-bold" style={{textShadow:'0 0 3px black, 0 0 3px black, 0 0 6px black'}}>{String.fromCharCode(65+ridx)}</span>}
              </div>
            );
          }))}
          
          {/* Holes markers in step 1 */}
          {step === 1 && Array(rows).fill(0).map((_,r) => Array(cols).fill(0).map((_,c) => {
            const key = `${r},${c}`, isHole = holes[key];
            const cellStyle = getCellStyle(r, c);
            return (
              <div key={key}
                style={{
                  ...cellStyle,
                  backgroundColor: isHole ? 'rgba(0,0,0,0.8)' : 'transparent',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer'
                }}
                onClick={() => handleCell(r,c)}
              >
                {isHole && <X className="w-5 h-5 text-red-500"/>}
              </div>
            );
          }))}
          
          {/* Edge drag handles */}
          {step === 1 && (
            <>
              {/* Left edge */}
              <div
                className="absolute w-3 bg-white cursor-ew-resize rounded-r shadow-lg"
                style={{ left: `calc(${bounds.left}% - 6px)`, top: `${bounds.top}%`, height: `${bounds.bottom - bounds.top}%`, zIndex: 25 }}
                onMouseDown={(e) => { e.preventDefault(); setDraggingEdge('left'); }}
                onTouchStart={(e) => { e.preventDefault(); setDraggingEdge('left'); }}
              />
              {/* Right edge */}
              <div
                className="absolute w-3 bg-white cursor-ew-resize rounded-l shadow-lg"
                style={{ left: `calc(${bounds.right}% - 6px)`, top: `${bounds.top}%`, height: `${bounds.bottom - bounds.top}%`, zIndex: 25 }}
                onMouseDown={(e) => { e.preventDefault(); setDraggingEdge('right'); }}
                onTouchStart={(e) => { e.preventDefault(); setDraggingEdge('right'); }}
              />
              {/* Top edge */}
              <div
                className="absolute h-3 bg-white cursor-ns-resize rounded-b shadow-lg"
                style={{ left: `${bounds.left}%`, top: `calc(${bounds.top}% - 6px)`, width: `${bounds.right - bounds.left}%`, zIndex: 25 }}
                onMouseDown={(e) => { e.preventDefault(); setDraggingEdge('top'); }}
                onTouchStart={(e) => { e.preventDefault(); setDraggingEdge('top'); }}
              />
              {/* Bottom edge */}
              <div
                className="absolute h-3 bg-white cursor-ns-resize rounded-t shadow-lg"
                style={{ left: `${bounds.left}%`, top: `calc(${bounds.bottom}% - 6px)`, width: `${bounds.right - bounds.left}%`, zIndex: 25 }}
                onMouseDown={(e) => { e.preventDefault(); setDraggingEdge('bottom'); }}
                onTouchStart={(e) => { e.preventDefault(); setDraggingEdge('bottom'); }}
              />
            </>
          )}
        </div>
      )}

      {step === 1 && (
        <div className="flex justify-center gap-3">
          <div className="flex items-center gap-1 text-sm"><span className="text-purple-300">Rows</span>
            <button onClick={()=>setRows(Math.max(2,rows-1))} className="w-6 h-6 bg-white/20 rounded">-</button>
            <span className="w-4 text-center">{rows}</span>
            <button onClick={()=>setRows(Math.min(8,rows+1))} className="w-6 h-6 bg-white/20 rounded">+</button>
          </div>
          <div className="flex items-center gap-1 text-sm"><span className="text-purple-300">Cols</span>
            <button onClick={()=>setCols(Math.max(2,cols-1))} className="w-6 h-6 bg-white/20 rounded">-</button>
            <span className="w-4 text-center">{cols}</span>
            <button onClick={()=>setCols(Math.min(8,cols+1))} className="w-6 h-6 bg-white/20 rounded">+</button>
          </div>
        </div>
      )}

      {step === 2 && (
        <div className="flex justify-center gap-1 flex-wrap">
          {REGION_COLORS.slice(0,10).map((col,i) => (
            <button key={i} onClick={()=>setCurrentRegion(i)} className={`w-7 h-7 rounded text-xs font-bold text-white ${currentRegion===i?'ring-2 ring-white scale-110':''}`} style={{backgroundColor:col}}>{String.fromCharCode(65+i)}</button>
          ))}
        </div>
      )}

      {step === 3 && (
        <div className="space-y-1 max-h-32 overflow-y-auto">
          {getUniqueRegions().map(ri => {
            const L = String.fromCharCode(65+ri), c = constraints[L] || {type:'any'};
            return (
              <div key={L} className="flex items-center gap-2 bg-white/10 rounded p-1">
                <div className="w-5 h-5 rounded" style={{backgroundColor:REGION_COLORS[ri]}}/>
                <span className="font-bold w-4">{L}</span>
                <select value={c.type} onChange={e => {
                  const t = e.target.value;
                  setConstraints(p => ({...p, [L]: t==='sum' ? {type:'sum',op:'==',value:6} : {type:t}}));
                }} className="bg-gray-800 rounded px-1 py-0.5 text-sm flex-1">
                  <option value="any">None</option><option value="sum">Sum</option><option value="all_equal">= Equal</option><option value="all_different">✕ Diff</option>
                </select>
                {c.type==='sum' && <>
                  <select value={c.op||'=='} onChange={e=>setConstraints(p=>({...p,[L]:{...c,op:e.target.value}}))} className="bg-gray-800 rounded px-1 py-0.5 text-sm w-10">
                    <option value="==">=</option><option value=">">&gt;</option><option value="<">&lt;</option>
                  </select>
                  <input type="number" value={c.value??''} onChange={e=>setConstraints(p=>({...p,[L]:{...c,value:+e.target.value||0}}))} className="bg-gray-800 rounded px-1 py-0.5 text-sm w-12"/>
                </>}
              </div>
            );
          })}
        </div>
      )}

      {step === 4 && (
        <div className="space-y-2">
          <div className="flex gap-1">
            <input value={quickDominoes} onChange={e=>setQuickDominoes(e.target.value)} onKeyDown={e=>e.key==='Enter'&&parseQuickDominoes()} placeholder="61 33 36 43" className="flex-1 bg-gray-800 rounded px-2 py-1 text-sm font-mono"/>
            <button onClick={parseQuickDominoes} className="px-2 py-1 bg-purple-500 rounded text-sm">Set</button>
            <button onClick={autoFill} className="px-2 py-1 bg-gray-700 rounded text-sm">#{Math.floor(cellCount/2)}</button>
          </div>
          <div className="grid grid-cols-4 gap-1 max-h-24 overflow-y-auto">
            {dominoes.map((d,i) => (
              <div key={i} className="flex items-center bg-white/5 rounded p-0.5">
                <div className="flex bg-white rounded overflow-hidden">
                  <div onClick={()=>cycleDomino(i,0)} className="w-6 h-6 flex items-center justify-center border-r border-gray-300 cursor-pointer"><PipDisplay value={d[0]} size={18}/></div>
                  <div onClick={()=>cycleDomino(i,1)} className="w-6 h-6 flex items-center justify-center cursor-pointer"><PipDisplay value={d[1]} size={18}/></div>
                </div>
                <button onClick={()=>setDominoes(dominoes.filter((_,j)=>j!==i))} className="text-red-400 ml-auto px-1"><X className="w-3 h-3"/></button>
              </div>
            ))}
          </div>
          <p className={`text-center text-sm ${isValid?'text-green-400':'text-yellow-400'}`}>{dominoes.length}/{Math.floor(cellCount/2)} {isValid?'✓':''}</p>
        </div>
      )}

      {step > 0 && (
        <div className="flex gap-2">
          {step > 1 && <button onClick={()=>setStep(step-1)} className="flex-1 py-2 bg-white/20 rounded-lg text-sm">← Back</button>}
          {step < 4 ? (
            <button onClick={()=>setStep(step+1)} className="flex-1 py-2 bg-purple-500 rounded-lg text-sm font-medium">Next →</button>
          ) : (
            <button onClick={buildPuzzle} disabled={!isValid} className={`flex-1 py-2 rounded-lg text-sm font-medium ${isValid?'bg-green-500':'bg-gray-600 text-gray-400'}`}>Solve →</button>
          )}
        </div>
      )}
    </div>
  );
};

const PipDisplay = ({ value, size = 36, color = '#1a1a1a' }) => {
  const pip = size * 0.15, margin = size * 0.22, center = size / 2;
  const positions = {
    0: [], 1: [[center, center]], 2: [[margin, margin], [size - margin, size - margin]],
    3: [[margin, margin], [center, center], [size - margin, size - margin]],
    4: [[margin, margin], [size - margin, margin], [margin, size - margin], [size - margin, size - margin]],
    5: [[margin, margin], [size - margin, margin], [center, center], [margin, size - margin], [size - margin, size - margin]],
    6: [[margin, margin], [size - margin, margin], [margin, center], [size - margin, center], [margin, size - margin], [size - margin, size - margin]],
  };
  return (
    <div style={{ width: size, height: size, position: 'relative' }}>
      {(positions[value] || []).map(([x, y], i) => (
        <div key={i} style={{ position: 'absolute', left: x - pip/2, top: y - pip/2, width: pip, height: pip, borderRadius: '50%', backgroundColor: color }} />
      ))}
    </div>
  );
};

const ConstraintBadge = ({ constraint }) => {
  if (!constraint) return null;
  let label = '';
  if (constraint.type === 'sum') {
    const ops = { '==': '', '!=': '≠', '<': '<', '>': '>' };
    label = `${ops[constraint.op] || ''}${constraint.value}`;
  } else if (constraint.type === 'all_equal') label = '=';
  else if (constraint.type === 'all_different') label = '✕';
  else if (constraint.type === 'any') return null;

  return (
    <div className="absolute -top-2 -right-2 w-6 h-6 bg-white rounded-full shadow-md flex items-center justify-center text-xs font-bold text-gray-700 border border-gray-200 transform rotate-45 z-10">
      <span className="transform -rotate-45">{label}</span>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// PUZZLE TEMPLATES
// ═══════════════════════════════════════════════════════════════════════════════

const TEMPLATES = [
  { name: '4×4', rows: 4, cols: 4, holes: [] },
  { name: '4×5', rows: 4, cols: 5, holes: [] },
  { name: '5×5', rows: 5, cols: 5, holes: [] },
  { name: '4×5 corners', rows: 4, cols: 5, holes: [[0,0],[0,4],[3,0],[3,4]] },
  { name: '5×5 center', rows: 5, cols: 5, holes: [[2,2]] },
  { name: '4×5 cross', rows: 4, cols: 5, holes: [[0,0],[0,1],[0,3],[0,4],[3,0],[3,4]] },
];

// ═══════════════════════════════════════════════════════════════════════════════
// ENHANCED PUZZLE BUILDER WITH EXTRACTION INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

const PuzzleBuilder = ({ onPuzzleReady, initialData, onStepChange, referenceImage }) => {
  // Initialize from extraction data or defaults
  const [rows, setRows] = useState(initialData?.rows || 4);
  const [cols, setCols] = useState(initialData?.cols || 4);
  const [grid, setGrid] = useState(() => {
    if (initialData?.grid) return initialData.grid;
    return Array(4).fill(null).map(() => Array(4).fill(0));
  });
  const [holes, setHoles] = useState(() => {
    if (initialData?.holes) return initialData.holes;
    return Array(4).fill(null).map(() => Array(4).fill(false));
  });
  const [currentRegion, setCurrentRegion] = useState(0);
  const [selectedRegion, setSelectedRegion] = useState(null);
  const [constraints, setConstraints] = useState(initialData?.constraints || {});
  const [dominoes, setDominoes] = useState(initialData?.dominoes || [[0, 0]]);
  const [step, setStep] = useState(1);
  const [isPainting, setIsPainting] = useState(false);
  const [constraintShorthand, setConstraintShorthand] = useState('');
  const [showOverlay, setShowOverlay] = useState(!!referenceImage);

  // Update state when initialData changes (e.g., from extraction)
  React.useEffect(() => {
    if (initialData) {
      setRows(initialData.rows || 4);
      setCols(initialData.cols || 4);
      setGrid(initialData.grid || Array(4).fill(null).map(() => Array(4).fill(0)));
      setHoles(initialData.holes || Array(4).fill(null).map(() => Array(4).fill(false)));
      setConstraints(initialData.constraints || {});
      setDominoes(initialData.dominoes || [[0, 0]]);
      setStep(1); // Start at step 1 to review
    }
  }, [initialData]);

  // Notify parent of step changes
  React.useEffect(() => { onStepChange?.(step); }, [step, onStepChange]);

  const updateGridSize = (newRows, newCols, holesArr = null) => {
    setRows(newRows);
    setCols(newCols);
    setGrid(Array(newRows).fill(null).map((_, r) => Array(newCols).fill(null).map((_, c) => grid[r]?.[c] ?? 0)));
    setHoles(holesArr || Array(newRows).fill(null).map((_, r) => Array(newCols).fill(null).map((_, c) => holes[r]?.[c] ?? false)));
  };

  const applyTemplate = (template) => {
    const holesArr = Array(template.rows).fill(null).map(() => Array(template.cols).fill(false));
    template.holes.forEach(([r, c]) => { if (holesArr[r]) holesArr[r][c] = true; });
    setRows(template.rows);
    setCols(template.cols);
    setGrid(Array(template.rows).fill(null).map(() => Array(template.cols).fill(0)));
    setHoles(holesArr);
  };

  const toggleHole = (r, c) => {
    const newHoles = holes.map(row => [...row]);
    newHoles[r][c] = !newHoles[r][c];
    setHoles(newHoles);
  };

  const paintRegion = (r, c) => {
    if (holes[r]?.[c]) return;
    const newGrid = grid.map(row => [...row]);
    newGrid[r][c] = currentRegion;
    setGrid(newGrid);
  };

  // Flood fill on drag
  const handleCellMouseDown = (r, c) => {
    if (step === 2) { setIsPainting(true); paintRegion(r, c); }
  };
  const handleCellMouseEnter = (r, c) => {
    if (step === 2 && isPainting) paintRegion(r, c);
  };
  const handleMouseUp = () => setIsPainting(false);

  const getRegionLetters = () => {
    const used = new Set();
    grid.forEach((row, r) => row.forEach((cell, c) => { if (!holes[r]?.[c]) used.add(cell); }));
    return [...used].sort((a, b) => a - b).map(i => String.fromCharCode(65 + i));
  };

  const setConstraint = (regionIdx, constraint) => {
    setConstraints(prev => ({ ...prev, [String.fromCharCode(65 + regionIdx)]: constraint }));
  };

  // Parse shorthand like "A=8 B>4 C= D✕" 
  const parseConstraintShorthand = () => {
    const parts = constraintShorthand.trim().split(/\s+/);
    const newConstraints = { ...constraints };
    parts.forEach(part => {
      const match = part.match(/^([A-Z])(=|>|<|≠|!=|✕|x)?(\d+)?$/i);
      if (match) {
        const [, letter, op, value] = match;
        const upperLetter = letter.toUpperCase();
        if (op === '✕' || op === 'x') newConstraints[upperLetter] = { type: 'all_different' };
        else if (op === '=' && !value) newConstraints[upperLetter] = { type: 'all_equal' };
        else if (value) {
          const opMap = { '=': '==', '>': '>', '<': '<', '≠': '!=', '!=': '!=' };
          newConstraints[upperLetter] = { type: 'sum', op: opMap[op] || '==', value: parseInt(value) };
        }
      }
    });
    setConstraints(newConstraints);
    setConstraintShorthand('');
  };

  const addDomino = () => setDominoes([...dominoes, [0, 0]]);
  const removeDomino = (idx) => setDominoes(dominoes.filter((_, i) => i !== idx));
  const updateDomino = (idx, half, value) => {
    const newDominoes = [...dominoes];
    newDominoes[idx] = [...newDominoes[idx]];
    newDominoes[idx][half] = value;
    setDominoes(newDominoes);
  };
  
  // Tap to cycle pip value (0→1→2→...→6→0)
  const cycleDomino = (idx, half, direction = 1) => {
    const newDominoes = [...dominoes];
    newDominoes[idx] = [...newDominoes[idx]];
    newDominoes[idx][half] = (newDominoes[idx][half] + direction + 7) % 7;
    setDominoes(newDominoes);
  };
  
  // Auto-fill correct number of blank dominoes
  const autoFillDominoCount = () => {
    const needed = Math.floor(cellCount / 2);
    if (needed > 0) {
      setDominoes(Array(needed).fill(null).map(() => [0, 0]));
    }
  };

  // Quick add multiple dominoes - supports "61 33 36 43" or "6-1 3-3" or "6,1 3,3"
  const [quickDominoes, setQuickDominoes] = useState('');
  const parseQuickDominoes = () => {
    // Match pairs of digits with optional separator
    const text = quickDominoes.replace(/[^0-6]/g, ''); // Keep only 0-6
    const pairs = [];
    for (let i = 0; i < text.length - 1; i += 2) {
      pairs.push([parseInt(text[i]), parseInt(text[i + 1])]);
    }
    if (pairs.length > 0) setDominoes(pairs);
    setQuickDominoes('');
  };

  const buildPuzzle = () => {
    const shape = grid.map((row, r) => row.map((_, c) => holes[r][c] ? '#' : '.').join('')).join('\n');
    const regions = grid.map((row, r) => row.map((cell, c) => holes[r][c] ? '#' : String.fromCharCode(65 + cell)).join('')).join('\n');
    onPuzzleReady({ pipRange: [0, 6], board: { rows, cols, shape }, regions, constraints, dominoes });
  };

  const cellCount = grid.flat().filter((_, i) => !holes[Math.floor(i / cols)]?.[i % cols]).length;
  const dominoCount = dominoes.length;
  const isValid = cellCount === dominoCount * 2;

  return (
    <div className="bg-white/10 rounded-2xl p-4 space-y-4" onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
      {/* Step Indicator */}
      <div className="flex justify-center items-center gap-2 mb-2">
        {[1, 2, 3, 4].map(s => (
          <React.Fragment key={s}>
            <button onClick={() => setStep(s)}
              className={`w-8 h-8 rounded-full font-bold transition-all ${step === s ? 'bg-purple-500 text-white scale-110' : 'bg-white/20 text-purple-200 hover:bg-white/30'}`}>
              {s}
            </button>
            {s < 4 && <div className={`w-8 h-0.5 ${step > s ? 'bg-purple-500' : 'bg-white/20'}`} />}
          </React.Fragment>
        ))}
      </div>

      {/* Step 1: Grid Size & Holes */}
      {step === 1 && (
        <div className="space-y-4">
          <h3 className="text-white font-semibold text-center">Step 1: Grid Shape</h3>
          
          {/* Reference image toggle */}
          {referenceImage && (
            <div className="flex justify-center">
              {showOverlay ? (
                <div className="flex flex-col items-center gap-1">
                  <img src={referenceImage} alt="Reference" className="max-h-32 rounded-lg shadow border border-purple-500/50" />
                  <button onClick={() => setShowOverlay(false)} className="text-purple-400 text-xs hover:text-white">Hide</button>
                </div>
              ) : (
                <button onClick={() => setShowOverlay(true)} className="text-purple-400 text-xs hover:text-white">Show reference image</button>
              )}
            </div>
          )}
          
          {/* Templates */}
          <div className="flex flex-wrap justify-center gap-2">
            {TEMPLATES.map((t, i) => (
              <button key={i} onClick={() => applyTemplate(t)}
                className="px-3 py-1.5 bg-white/10 hover:bg-white/20 text-purple-200 rounded-lg text-sm flex items-center gap-1">
                <Grid3X3 className="w-3 h-3" /> {t.name}
              </button>
            ))}
          </div>

          <div className="flex justify-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-purple-200 text-sm">Rows:</span>
              <button onClick={() => updateGridSize(Math.max(2, rows - 1), cols)} className="w-7 h-7 bg-white/20 rounded text-white"><Minus className="w-4 h-4 mx-auto" /></button>
              <span className="text-white w-6 text-center">{rows}</span>
              <button onClick={() => updateGridSize(Math.min(8, rows + 1), cols)} className="w-7 h-7 bg-white/20 rounded text-white"><Plus className="w-4 h-4 mx-auto" /></button>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-purple-200 text-sm">Cols:</span>
              <button onClick={() => updateGridSize(rows, Math.max(2, cols - 1))} className="w-7 h-7 bg-white/20 rounded text-white"><Minus className="w-4 h-4 mx-auto" /></button>
              <span className="text-white w-6 text-center">{cols}</span>
              <button onClick={() => updateGridSize(rows, Math.min(8, cols + 1))} className="w-7 h-7 bg-white/20 rounded text-white"><Plus className="w-4 h-4 mx-auto" /></button>
            </div>
          </div>

          <p className="text-purple-200 text-center text-sm">Click cells to toggle holes</p>
          <div className="flex justify-center">
            <div className="inline-block bg-gray-800 p-2 rounded-xl">
              {Array(rows).fill(null).map((_, r) => (
                <div key={r} className="flex">
                  {Array(cols).fill(null).map((_, c) => (
                    <div key={c} onClick={() => toggleHole(r, c)}
                      className={`w-10 h-10 m-0.5 rounded cursor-pointer transition-all flex items-center justify-center text-xs font-bold
                        ${holes[r]?.[c] ? 'bg-gray-900 border-2 border-dashed border-gray-600 text-gray-600' : 'bg-gray-400 text-gray-600'}`}>
                      {holes[r]?.[c] ? '#' : '.'}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
          <p className="text-purple-300 text-center text-sm">{cellCount} cells</p>
          <button onClick={() => setStep(2)} className="w-full py-2 bg-purple-500 text-white rounded-lg font-medium">Next: Paint Regions →</button>
        </div>
      )}

      {/* Step 2: Paint Regions (with drag support) */}
      {step === 2 && (
        <div className="space-y-4">
          <h3 className="text-white font-semibold text-center">Step 2: Paint Regions</h3>
          
          <div className="flex flex-col lg:flex-row gap-4 items-start justify-center">
            {/* Reference image */}
            {referenceImage && showOverlay && (
              <div className="flex flex-col items-center gap-1">
                <img src={referenceImage} alt="Reference" className="max-h-48 rounded-lg shadow border border-purple-500/50" />
                <button onClick={() => setShowOverlay(false)} className="text-purple-400 text-xs hover:text-white">Hide</button>
              </div>
            )}
            
            <div className="flex flex-col items-center gap-3">
              {/* Color picker */}
              <div className="flex justify-center gap-1 flex-wrap">
                {REGION_COLORS.slice(0, 10).map((color, i) => (
                  <button key={i} onClick={() => setCurrentRegion(i)}
                    className={`w-8 h-8 rounded-lg transition-all flex items-center justify-center ${currentRegion === i ? 'ring-2 ring-white scale-110' : ''}`}
                    style={{ backgroundColor: color }}>
                    <span className="text-white font-bold text-xs drop-shadow-lg">{String.fromCharCode(65 + i)}</span>
                  </button>
                ))}
              </div>
              <p className="text-purple-200 text-center text-xs">Click or drag to paint • Match colors to reference</p>
              
              {/* Grid */}
              <div className="inline-block bg-gray-800 p-2 rounded-xl select-none">
                {grid.map((row, r) => (
                  <div key={r} className="flex">
                    {row.map((cell, c) => (
                      <div key={c}
                        onMouseDown={() => handleCellMouseDown(r, c)}
                        onMouseEnter={() => handleCellMouseEnter(r, c)}
                        className={`w-10 h-10 m-0.5 rounded cursor-pointer transition-all flex items-center justify-center ${holes[r]?.[c] ? 'bg-gray-900' : ''}`}
                        style={{ backgroundColor: holes[r]?.[c] ? undefined : REGION_COLORS[cell] }}>
                        {!holes[r]?.[c] && <span className="text-white font-bold text-sm drop-shadow-lg">{String.fromCharCode(65 + cell)}</span>}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
              
              {referenceImage && !showOverlay && (
                <button onClick={() => setShowOverlay(true)} className="text-purple-400 text-xs hover:text-white">Show reference</button>
              )}
            </div>
          </div>
          
          <div className="flex gap-2">
            <button onClick={() => setStep(1)} className="flex-1 py-2 bg-white/20 text-white rounded-lg">← Back</button>
            <button onClick={() => setStep(3)} className="flex-1 py-2 bg-purple-500 text-white rounded-lg">Next: Constraints →</button>
          </div>
        </div>
      )}

      {/* Step 3: Constraints with visual grid */}
      {step === 3 && (
        <div className="space-y-4">
          <h3 className="text-white font-semibold text-center">Step 3: Set Constraints</h3>
          
          {/* Shorthand input */}
          <div className="flex gap-2">
            <input
              type="text"
              value={constraintShorthand}
              onChange={e => setConstraintShorthand(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && parseConstraintShorthand()}
              placeholder="Quick: A=8 B>4 C= Dx"
              className="flex-1 bg-gray-800 text-white rounded px-3 py-2 text-sm placeholder-gray-500"
            />
            <button onClick={parseConstraintShorthand} className="px-3 py-2 bg-purple-500 text-white rounded text-sm">Apply</button>
          </div>
          <p className="text-purple-300 text-xs text-center">Shorthand: A=8 (sum=8), A&gt;4, A&lt;4, A= (all equal), Ax (all different)</p>

          <div className="flex flex-col lg:flex-row gap-4 items-start justify-center">
            {/* Grid */}
            <div className="flex justify-center">
              <div className="inline-block bg-gray-800 p-2 rounded-xl">
                {grid.map((row, r) => (
                  <div key={r} className="flex">
                    {row.map((cell, c) => {
                      const isHole = holes[r]?.[c];
                      return (
                        <div key={c} onClick={() => !isHole && setSelectedRegion(cell)}
                          className={`w-10 h-10 m-0.5 rounded flex items-center justify-center cursor-pointer transition-all
                            ${isHole ? 'bg-gray-900' : ''} ${!isHole && selectedRegion === cell ? 'ring-2 ring-white scale-105' : ''}`}
                          style={{ backgroundColor: isHole ? undefined : REGION_COLORS[cell] }}>
                          {!isHole && <span className="text-white font-bold text-sm drop-shadow-lg">{String.fromCharCode(65 + cell)}</span>}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>

            {/* Constraint editor */}
            <div className="flex-1 max-w-xs">
              {selectedRegion !== null ? (
                <div className="bg-white/10 rounded-lg p-3 space-y-3">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded" style={{ backgroundColor: REGION_COLORS[selectedRegion] }} />
                    <span className="text-white font-bold text-lg">Region {String.fromCharCode(65 + selectedRegion)}</span>
                  </div>
                  {(() => {
                    const letter = String.fromCharCode(65 + selectedRegion);
                    const constraint = constraints[letter] || { type: 'any' };
                    return (
                      <div className="space-y-2">
                        <select value={constraint.type} onChange={e => {
                          const type = e.target.value;
                          if (type === 'sum') setConstraint(selectedRegion, { type: 'sum', op: '==', value: 6 });
                          else setConstraint(selectedRegion, { type });
                        }} className="w-full bg-gray-800 text-white rounded px-3 py-2">
                          <option value="any">Any (no constraint)</option>
                          <option value="sum">Sum equals/less/greater</option>
                          <option value="all_equal">All Equal (=)</option>
                          <option value="all_different">All Different (✕)</option>
                        </select>
                        {constraint.type === 'sum' && (
                          <div className="flex gap-2">
                            <select value={constraint.op || '=='} onChange={e => setConstraint(selectedRegion, { ...constraint, op: e.target.value })}
                              className="bg-gray-800 text-white rounded px-3 py-2">
                              <option value="==">= equals</option>
                              <option value="<">&lt; less than</option>
                              <option value=">">&gt; greater</option>
                              <option value="!=">≠ not equal</option>
                            </select>
                            <input type="number" value={constraint.value || 0} onChange={e => setConstraint(selectedRegion, { ...constraint, value: parseInt(e.target.value) || 0 })}
                              className="bg-gray-800 text-white rounded px-3 py-2 w-20" min="0" max="99" />
                          </div>
                        )}
                      </div>
                    );
                  })()}
                </div>
              ) : (
                <div className="bg-white/5 rounded-lg p-4 text-center text-purple-300 text-sm">Click a region to edit</div>
              )}
              
              {/* Summary */}
              <div className="mt-3 space-y-1 max-h-32 overflow-y-auto">
                {getRegionLetters().map(letter => {
                  const regionIdx = letter.charCodeAt(0) - 65;
                  const constraint = constraints[letter] || { type: 'any' };
                  let label = '—';
                  if (constraint.type === 'sum') {
                    const ops = { '==': '=', '!=': '≠', '<': '<', '>': '>' };
                    label = `${ops[constraint.op] || '='} ${constraint.value}`;
                  } else if (constraint.type === 'all_equal') label = 'All =';
                  else if (constraint.type === 'all_different') label = 'All ✕';
                  return (
                    <div key={letter} onClick={() => setSelectedRegion(regionIdx)}
                      className={`flex items-center gap-2 text-xs cursor-pointer p-1 rounded ${selectedRegion === regionIdx ? 'bg-white/20' : 'hover:bg-white/10'}`}>
                      <div className="w-4 h-4 rounded" style={{ backgroundColor: REGION_COLORS[regionIdx] }} />
                      <span className="text-white font-medium w-4">{letter}</span>
                      <span className="text-purple-200">{label}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="flex gap-2">
            <button onClick={() => setStep(2)} className="flex-1 py-2 bg-white/20 text-white rounded-lg">← Back</button>
            <button onClick={() => setStep(4)} className="flex-1 py-2 bg-purple-500 text-white rounded-lg">Next: Dominoes →</button>
          </div>
        </div>
      )}

      {/* Step 4: Dominoes */}
      {step === 4 && (
        <div className="space-y-4">
          <h3 className="text-white font-semibold text-center">Step 4: Enter Dominoes</h3>

          {/* Quick entry - simplified format */}
          <div className="space-y-2">
            <div className="flex gap-2">
              <input type="text" value={quickDominoes} onChange={e => setQuickDominoes(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && parseQuickDominoes()}
                placeholder="Type: 61 33 36 43 14 10 14"
                className="flex-1 bg-gray-800 text-white rounded px-3 py-2 text-sm placeholder-gray-500 font-mono" />
              <button onClick={parseQuickDominoes} className="px-3 py-2 bg-purple-500 text-white rounded text-sm">Set</button>
            </div>
            <p className="text-purple-400 text-xs text-center">Just type digits: 61 = [6,1] domino. Tap dominoes below to adjust.</p>
          </div>

          <div className="flex flex-col lg:flex-row gap-4 items-start justify-center">
            {/* Reference image if available */}
            {referenceImage && showOverlay && (
              <div className="flex flex-col items-center gap-2">
                <img src={referenceImage} alt="Reference" className="max-h-40 rounded-lg shadow border border-purple-500/50" />
                <button onClick={() => setShowOverlay(false)} className="text-purple-400 text-xs hover:text-white">Hide reference</button>
              </div>
            )}
            {referenceImage && !showOverlay && (
              <button onClick={() => setShowOverlay(true)} className="text-purple-400 text-xs hover:text-white">Show reference</button>
            )}

            {/* Grid preview */}
            <div className="flex flex-col items-center gap-2">
              <div className="inline-block bg-gray-800 p-2 rounded-xl">
                {grid.map((row, r) => (
                  <div key={r} className="flex">
                    {row.map((cell, c) => {
                      const isHole = holes[r]?.[c];
                      const letter = String.fromCharCode(65 + cell);
                      const constraint = constraints[letter];
                      let badge = '';
                      if (constraint?.type === 'sum') {
                        const ops = { '==': '', '!=': '≠', '<': '<', '>': '>' };
                        badge = `${ops[constraint.op] || ''}${constraint.value}`;
                      } else if (constraint?.type === 'all_equal') badge = '=';
                      else if (constraint?.type === 'all_different') badge = '✕';
                      return (
                        <div key={c} className={`w-8 h-8 m-0.5 rounded flex items-center justify-center ${isHole ? 'bg-gray-900' : ''}`}
                          style={{ backgroundColor: isHole ? undefined : REGION_COLORS[cell] }}>
                          {!isHole && badge && <span className="text-white font-bold text-[10px] drop-shadow-lg">{badge}</span>}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
              <p className={`text-sm font-medium ${isValid ? 'text-green-400' : 'text-yellow-400'}`}>
                Need {Math.floor(cellCount / 2)}, have {dominoCount}
              </p>
              <button onClick={autoFillDominoCount} className="text-purple-400 text-xs hover:text-white">
                Auto-fill {Math.floor(cellCount / 2)} blank dominoes
              </button>
            </div>

            {/* Domino list with TAP TO CYCLE */}
            <div className="flex-1 max-w-sm space-y-2">
              <div className="grid grid-cols-2 gap-2 max-h-64 overflow-y-auto pr-1">
                {dominoes.map((domino, i) => (
                  <div key={i} className="flex items-center gap-1 bg-white/5 rounded-lg p-1">
                    <span className="text-purple-300 text-xs w-4">{i + 1}</span>
                    
                    {/* Tap-to-cycle domino */}
                    <div className="flex bg-white rounded shadow overflow-hidden cursor-pointer select-none">
                      <div 
                        onClick={() => cycleDomino(i, 0, 1)}
                        onContextMenu={(e) => { e.preventDefault(); cycleDomino(i, 0, -1); }}
                        className="w-9 h-9 flex items-center justify-center border-r border-gray-200 hover:bg-gray-100 active:bg-gray-200"
                        title="Click to increase, right-click to decrease"
                      >
                        <PipDisplay value={domino[0]} size={28} />
                      </div>
                      <div 
                        onClick={() => cycleDomino(i, 1, 1)}
                        onContextMenu={(e) => { e.preventDefault(); cycleDomino(i, 1, -1); }}
                        className="w-9 h-9 flex items-center justify-center hover:bg-gray-100 active:bg-gray-200"
                        title="Click to increase, right-click to decrease"
                      >
                        <PipDisplay value={domino[1]} size={28} />
                      </div>
                    </div>
                    
                    <button onClick={() => removeDomino(i)} className="p-0.5 text-red-400 hover:text-red-300 ml-auto">
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
              
              <button onClick={addDomino} className="w-full py-1.5 bg-white/20 text-white rounded-lg flex items-center justify-center gap-2 text-sm">
                <Plus className="w-4 h-4" /> Add Domino
              </button>
            </div>
          </div>

          <div className="flex gap-2">
            <button onClick={() => setStep(3)} className="flex-1 py-2 bg-white/20 text-white rounded-lg">← Back</button>
            <button onClick={buildPuzzle} disabled={!isValid}
              className={`flex-1 py-2 rounded-lg font-semibold flex items-center justify-center gap-2 ${isValid ? 'bg-green-500 text-white' : 'bg-gray-600 text-gray-400 cursor-not-allowed'}`}>
              <Play className="w-4 h-4" /> Build & Solve
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// AUTO EXTRACTION COMPONENT (runs automatically, shows reasoning)
// ═══════════════════════════════════════════════════════════════════════════════

const AutoExtractor = ({ imageData, mediaType, onComplete, onCancel, onSkip }) => {
  const [status, setStatus] = useState('extracting'); // extracting, done, error
  const [progressText, setProgressText] = useState('Starting extraction...');
  const [reasoning, setReasoning] = useState('');
  const [puzzleData, setPuzzleData] = useState(null);
  const [error, setError] = useState(null);

  // Auto-run extraction on mount
  React.useEffect(() => {
    let cancelled = false;
    
    (async () => {
      try {
        setStatus('extracting');
        const result = await extractPuzzleFromImage(imageData, mediaType, (msg) => {
          if (!cancelled) setProgressText(msg);
        });
        if (cancelled) return;
        
        setPuzzleData(result.puzzleData);
        setReasoning(result.reasoning);
        setStatus('done');
      } catch (e) {
        if (cancelled) return;
        setError(e.message);
        setStatus('error');
      }
    })();
    
    return () => { cancelled = true; };
  }, [imageData, mediaType]);

  const handleContinue = () => {
    // Convert to builder format
    const json = puzzleData;
    const shapeLines = json.board.shape.split('\n');
    const regionLines = json.regions.split('\n');
    const rows = shapeLines.length;
    const cols = Math.max(...shapeLines.map(l => l.length));

    const holes = Array(rows).fill(null).map((_, r) =>
      Array(cols).fill(null).map((_, c) => shapeLines[r]?.[c] !== '.')
    );

    // Map region letters to indices
    const letterToIdx = {};
    let nextIdx = 0;
    regionLines.forEach(line => {
      [...line].forEach(ch => {
        if (ch !== '#' && ch !== '.' && ch !== ' ' && !(ch in letterToIdx)) {
          letterToIdx[ch] = nextIdx++;
        }
      });
    });

    const grid = Array(rows).fill(null).map((_, r) =>
      Array(cols).fill(null).map((_, c) => {
        const ch = regionLines[r]?.[c];
        if (ch === '#' || ch === '.' || ch === ' ' || ch === undefined) return 0;
        return letterToIdx[ch] ?? 0;
      })
    );

    // Remap constraints to new letter indices
    const constraints = {};
    Object.entries(json.constraints || {}).forEach(([letter, constraint]) => {
      if (letter in letterToIdx) {
        constraints[String.fromCharCode(65 + letterToIdx[letter])] = constraint;
      }
    });

    onComplete({
      rows, cols, holes, grid, constraints,
      dominoes: json.dominoes || [[0, 0]]
    });
  };

  return (
    <div className="bg-white/10 rounded-2xl p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-white font-semibold flex items-center gap-2">
          <Wand2 className="w-5 h-5" /> AI Extraction
        </h3>
        <button onClick={onCancel} className="text-purple-300 hover:text-white"><X className="w-5 h-5" /></button>
      </div>

      {/* Image preview */}
      <div className="flex justify-center">
        <img src={`data:${mediaType};base64,${imageData}`} alt="Puzzle" className="max-h-32 rounded-lg shadow" />
      </div>

      {/* Status */}
      {status === 'extracting' && (
        <div className="text-center py-4">
          <Loader2 className="w-10 h-10 text-purple-400 animate-spin mx-auto mb-3" />
          <p className="text-purple-200">{progressText}</p>
          <button onClick={onSkip} className="mt-3 text-purple-400 hover:text-white text-sm underline">
            Skip AI → Enter manually with reference
          </button>
        </div>
      )}

      {status === 'error' && (
        <div className="space-y-3">
          <div className="p-3 bg-red-500/20 border border-red-400 rounded-lg text-red-200 text-sm">
            {error}
          </div>
          <div className="flex gap-2">
            <button onClick={onCancel} className="flex-1 py-2 bg-white/20 text-white rounded-lg">Cancel</button>
            <button onClick={onSkip} className="flex-1 py-2 bg-purple-500 text-white rounded-lg">Enter manually</button>
          </div>
        </div>
      )}

      {status === 'done' && puzzleData && (
        <div className="space-y-3">
          {/* Reasoning (collapsible) */}
          <details className="bg-white/5 rounded-lg">
            <summary className="px-3 py-2 text-purple-300 text-sm cursor-pointer hover:text-white">
              View AI reasoning...
            </summary>
            <pre className="px-3 pb-3 text-xs text-gray-400 whitespace-pre-wrap max-h-48 overflow-y-auto">
              {reasoning}
            </pre>
          </details>

          {/* Extracted Data Summary */}
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="bg-white/5 rounded-lg p-2">
              <p className="text-purple-400 text-xs">Grid</p>
              <p className="text-white font-mono">{puzzleData.board.rows}×{puzzleData.board.cols}</p>
            </div>
            <div className="bg-white/5 rounded-lg p-2">
              <p className="text-purple-400 text-xs">Dominoes</p>
              <p className="text-white font-mono">{puzzleData.dominoes?.length || 0}</p>
            </div>
          </div>

          {/* Visual preview of extracted data */}
          <div className="bg-white/5 rounded-lg p-2">
            <p className="text-purple-400 text-xs mb-1">Regions & Constraints</p>
            <div className="flex gap-2 overflow-x-auto">
              {Object.entries(puzzleData.constraints || {}).map(([letter, c]) => {
                let label = '—';
                if (c.type === 'sum') {
                  const ops = { '==': '=', '!=': '≠', '<': '<', '>': '>' };
                  label = `${ops[c.op] || '='}${c.value}`;
                } else if (c.type === 'all_equal') label = '=';
                else if (c.type === 'all_different') label = '✕';
                return (
                  <div key={letter} className="flex items-center gap-1 bg-white/10 rounded px-2 py-1">
                    <span className="text-white font-bold text-xs">{letter}</span>
                    <span className="text-purple-200 text-xs">{label}</span>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="bg-white/5 rounded-lg p-2">
            <p className="text-purple-400 text-xs mb-1">Dominoes</p>
            <p className="text-white text-xs font-mono">
              {puzzleData.dominoes?.map(d => `[${d[0]}-${d[1]}]`).join(' ')}
            </p>
          </div>

          {/* Validation */}
          {(() => {
            const shapeLines = puzzleData.board.shape.split('\n');
            const cellCount = shapeLines.join('').split('').filter(c => c === '.').length;
            const dominoCount = puzzleData.dominoes?.length || 0;
            const isValid = cellCount === dominoCount * 2;
            return (
              <div className={`p-2 rounded-lg text-sm ${isValid ? 'bg-green-500/20 text-green-300' : 'bg-yellow-500/20 text-yellow-300'}`}>
                {isValid 
                  ? `✓ Valid: ${cellCount} cells, ${dominoCount} dominoes`
                  : `⚠ Mismatch: ${cellCount} cells but ${dominoCount} dominoes (need ${cellCount/2})`
                }
              </div>
            );
          })()}

          <div className="flex gap-2">
            <button onClick={onSkip} className="flex-1 py-2 bg-white/20 text-white rounded-lg">Start fresh</button>
            <button onClick={handleContinue} className="flex-1 py-2 bg-green-500 text-white rounded-lg font-medium flex items-center justify-center gap-2">
              <Edit3 className="w-4 h-4" /> Edit & Fix
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// PUZZLE DISPLAY COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const PuzzleBoard = ({ puzzle, solution, revealedSet, onCellClick, hintMode }) => {
  const { board, regions, constraints } = puzzle;
  const cellSize = 48;

  const getCellValue = (r, c) => solution?.solved ? solution.grid.get(`${r},${c}`) : null;
  const isRevealed = (r, c) => {
    if (!solution?.solved) return false;
    const placement = solution.placements.find(p => {
      const [[r1, c1], [r2, c2]] = p.coords;
      return (r1 === r && c1 === c) || (r2 === r && c2 === c);
    });
    return placement && revealedSet.has(placement.dominoIndex);
  };

  const getRegionInfo = (r, c) => {
    for (const [rid, coords] of regions) {
      if (coords.some(([rr, cc]) => rr === r && cc === c)) {
        const idx = rid.charCodeAt(0) - 65;
        return { id: rid, color: REGION_COLORS[idx % REGION_COLORS.length] };
      }
    }
    return { id: null, color: '#666' };
  };

  const isRegionTopLeft = (r, c, rid) => {
    const coords = regions.get(rid);
    if (!coords) return false;
    const minR = Math.min(...coords.map(([rr]) => rr));
    const minC = Math.min(...coords.filter(([rr]) => rr === minR).map(([, cc]) => cc));
    return r === minR && c === minC;
  };

  return (
    <div className="inline-block bg-gray-800 p-2 rounded-xl shadow-2xl">
      {Array.from({ length: board.rows }, (_, r) => (
        <div key={r} className="flex">
          {Array.from({ length: board.cols }, (_, c) => {
            const key = `${r},${c}`;
            if (!board.cells.has(key)) return <div key={key} style={{ width: cellSize, height: cellSize, margin: 1 }} />;
            
            const value = getCellValue(r, c);
            const revealed = isRevealed(r, c);
            const { id: regionId, color } = getRegionInfo(r, c);
            const showConstraint = regionId && isRegionTopLeft(r, c, regionId);
            const constraint = regionId ? constraints.get(regionId) : null;

            return (
              <div key={key} onClick={() => hintMode && onCellClick?.(r, c)}
                className={`relative rounded-lg transition-all ${hintMode ? 'cursor-pointer hover:scale-105' : ''}`}
                style={{ width: cellSize, height: cellSize, margin: 1, backgroundColor: color, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {showConstraint && <ConstraintBadge constraint={constraint} />}
                {revealed && value !== null && (
                  <div className="bg-white rounded-md shadow p-0.5"><PipDisplay value={value} size={32} /></div>
                )}
                {!revealed && hintMode && <Target className="w-4 h-4 text-white/50" />}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
};

const DominoTray = ({ dominoes, usedIndices }) => (
  <div className="flex flex-wrap gap-2 justify-center p-3 bg-gray-100 rounded-xl">
    {dominoes.map((pips, i) => {
      const used = usedIndices.has(i);
      return (
        <div key={i} className={`flex bg-white rounded-lg shadow relative ${used ? 'opacity-30' : ''}`}>
          <div className="p-1.5 border-r border-gray-200"><PipDisplay value={pips[0]} size={24} /></div>
          <div className="p-1.5"><PipDisplay value={pips[1]} size={24} /></div>
          {used && <div className="absolute inset-0 flex items-center justify-center bg-green-500/20 rounded-lg"><Check className="w-5 h-5 text-green-600" /></div>}
        </div>
      );
    })}
  </div>
);

// ═══════════════════════════════════════════════════════════════════════════════
// SAMPLE PUZZLES
// ═══════════════════════════════════════════════════════════════════════════════

const SAMPLE_PUZZLES = [
  { name: "4×4 Easy", data: { pipRange: [0, 6], dominoes: [[6,1],[2,2],[5,3],[2,1],[0,4],[6,2],[1,0],[3,4]], board: { rows: 4, cols: 4, shape: "....\n....\n....\n...." }, regions: "AABB\nCCDD\nEEFF\nGGHH", constraints: { A:{type:"sum",op:"==",value:7},B:{type:"sum",op:"==",value:4},C:{type:"sum",op:"==",value:8},D:{type:"sum",op:"==",value:3},E:{type:"sum",op:"==",value:4},F:{type:"sum",op:"==",value:8},G:{type:"sum",op:"==",value:1},H:{type:"sum",op:"==",value:7}}}},
  { name: "Mixed", data: { pipRange: [0, 6], dominoes: [[5,5],[2,3],[4,4],[3,0],[6,1],[0,0]], board: { rows: 3, cols: 4, shape: "....\n....\n...." }, regions: "AABB\nCCDD\nEEFF", constraints: { A:{type:"sum",op:"==",value:10},B:{type:"all_equal"},C:{type:"sum",op:"<",value:6},D:{type:"sum",op:">",value:4},E:{type:"all_different"},F:{type:"sum",op:"==",value:7}}}},
  { name: "Dec 18", data: { 
    pipRange: [0, 6], 
    board: { rows: 4, cols: 5, shape: "##.##\n.#...\n.....\n#...." }, 
    regions: "##B##\nA#BB.\n.DDDG\n#FEE.", 
    constraints: { 
      A: { type: "sum", op: ">", value: 4 },
      B: { type: "sum", op: "==", value: 8 },
      D: { type: "sum", op: "==", value: 3 },
      E: { type: "sum", op: "==", value: 8 },
      F: { type: "sum", op: "==", value: 6 },
      G: { type: "sum", op: ">", value: 4 }
    }, 
    dominoes: [[6,1], [3,3], [3,6], [4,3], [1,5], [2,0], [1,4]]
  }}
];

// ═══════════════════════════════════════════════════════════════════════════════
// PUZZLE ARCHIVE - Save and load puzzles by date
// ═══════════════════════════════════════════════════════════════════════════════

const PuzzleArchive = ({ onSelect, onClose }) => {
  const [puzzles, setPuzzles] = useState([]);
  const [loading, setLoading] = useState(true);

  React.useEffect(() => {
    (async () => {
      try {
        const result = await window.storage?.list('puzzle:', true);
        if (result?.keys) {
          const loaded = [];
          for (const key of result.keys.slice(0, 30)) { // Load last 30
            try {
              const data = await window.storage.get(key, true);
              if (data?.value) {
                const puzzle = JSON.parse(data.value);
                loaded.push({ key, ...puzzle });
              }
            } catch (e) {}
          }
          loaded.sort((a, b) => b.date.localeCompare(a.date));
          setPuzzles(loaded);
        }
      } catch (e) { console.log('Archive load error:', e); }
      setLoading(false);
    })();
  }, []);

  const formatDate = (dateStr) => {
    const d = new Date(dateStr + 'T12:00:00');
    return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
  };

  return (
    <div className="bg-white/10 rounded-2xl p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-white font-semibold">Puzzle Archive</h3>
        <button onClick={onClose} className="text-purple-300 hover:text-white"><X className="w-5 h-5" /></button>
      </div>

      {loading ? (
        <div className="text-center py-8">
          <Loader2 className="w-8 h-8 text-purple-400 animate-spin mx-auto" />
        </div>
      ) : puzzles.length === 0 ? (
        <div className="text-center py-8 text-purple-300">
          <p>No saved puzzles yet.</p>
          <p className="text-sm mt-1">Solved puzzles will appear here.</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {puzzles.map((p) => (
            <button key={p.key} onClick={() => onSelect(p.puzzle)}
              className="w-full p-3 bg-white/5 hover:bg-white/10 rounded-lg text-left flex items-center gap-3">
              <div className="text-2xl">📅</div>
              <div className="flex-1">
                <div className="text-white font-medium">{formatDate(p.date)}</div>
                <div className="text-purple-300 text-sm">{p.puzzle?.dominoes?.length || '?'} dominoes</div>
              </div>
              <ChevronRight className="w-5 h-5 text-purple-300" />
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

// Save puzzle to archive
async function savePuzzleToArchive(puzzle, date = null) {
  const dateStr = date || new Date().toISOString().split('T')[0];
  const key = `puzzle:${dateStr}`;
  try {
    await window.storage?.set(key, JSON.stringify({ date: dateStr, puzzle, savedAt: Date.now() }), true);
    return true;
  } catch (e) {
    console.log('Save failed:', e);
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════════════════════

export default function PipsSolverApp() {
  const [puzzle, setPuzzle] = useState(null);
  const [solution, setSolution] = useState(null);
  const [revealedSet, setRevealedSet] = useState(new Set());
  const [mode, setMode] = useState('progressive');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [view, setView] = useState('home'); // home, builder, extraction, solve, overlay, archive
  const [extractionImage, setExtractionImage] = useState(null);
  const [extractionMediaType, setExtractionMediaType] = useState(null);
  const [referenceImageUrl, setReferenceImageUrl] = useState(null);
  const [builderInitialData, setBuilderInitialData] = useState(null);
  const [debugLogs, setDebugLogs] = useState([]);
  const [showDebug, setShowDebug] = useState(false);
  const [hasDraft, setHasDraft] = useState(false);
  const [draftData, setDraftData] = useState(null);
  const [currentPuzzleJson, setCurrentPuzzleJson] = useState(null);
  const fileInputRef = useRef(null);

  // Check for saved draft on mount
  React.useEffect(() => {
    (async () => {
      try {
        const result = await window.storage?.get('pips-draft');
        if (result?.value) {
          const draft = JSON.parse(result.value);
          // Only show if less than 24 hours old
          if (Date.now() - draft.timestamp < 24 * 60 * 60 * 1000) {
            setHasDraft(true);
            setDraftData(draft);
          }
        }
      } catch (e) {}
    })();
  }, []);

  const loadPuzzle = useCallback((puzzleJson) => {
    try {
      const p = createPuzzleFromJSON(puzzleJson);
      setPuzzle({ ...p, dominoes: puzzleJson.dominoes });
      setCurrentPuzzleJson(puzzleJson);
      setSolution(null);
      setRevealedSet(new Set());
      setError(null);
      setDebugLogs([]);
      setView('solve');
    } catch (e) {
      setError(`Failed to load: ${e.message}`);
    }
  }, []);

  const handleImageUpload = (file, useOverlay = false) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      const base64 = dataUrl.split(',')[1];
      setReferenceImageUrl(dataUrl);
      
      if (useOverlay) {
        setView('overlay');
      } else {
        setExtractionImage(base64);
        setExtractionMediaType(file.type);
        setView('extraction');
      }
    };
    reader.readAsDataURL(file);
  };
  
  const handleSkipExtraction = () => {
    setBuilderInitialData(null);
    setView('builder');
  };

  const handleExtractionComplete = (data) => {
    setBuilderInitialData(data);
    setView('builder');
  };
  
  const handleOverlayComplete = (puzzleJson) => {
    loadPuzzle(puzzleJson);
  };
  
  const resumeDraft = () => {
    if (draftData?.imageUrl) {
      setReferenceImageUrl(draftData.imageUrl);
      setView('overlay');
    }
  };
  
  const clearDraft = async () => {
    try { await window.storage?.delete('pips-draft'); } catch (e) {}
    setHasDraft(false);
    setDraftData(null);
  };

  const handleSolve = useCallback(async () => {
    if (!puzzle) return;
    setLoading(true);
    setTimeout(async () => {
      const sol = solvePips(puzzle, true);
      setSolution(sol);
      setDebugLogs(sol.logs || []);
      if (mode === 'complete' && sol.solved) setRevealedSet(new Set(sol.placements.map(p => p.dominoIndex)));
      setError(sol.solved ? null : (sol.error || 'No solution found'));
      setLoading(false);
      
      // Save to archive if solved
      if (sol.solved && currentPuzzleJson) {
        await savePuzzleToArchive(currentPuzzleJson);
      }
    }, 50);
  }, [puzzle, mode, currentPuzzleJson]);

  const revealNext = useCallback(() => {
    if (!solution?.solved) return;
    const next = solution.placements.find(p => !revealedSet.has(p.dominoIndex));
    if (next) setRevealedSet(prev => new Set([...prev, next.dominoIndex]));
  }, [solution, revealedSet]);

  const revealAll = useCallback(() => {
    if (solution?.solved) setRevealedSet(new Set(solution.placements.map(p => p.dominoIndex)));
  }, [solution]);

  const handleCellClick = useCallback((r, c) => {
    if (!solution?.solved || mode !== 'hint') return;
    const placement = solution.placements.find(p => {
      const [[r1, c1], [r2, c2]] = p.coords;
      return (r1 === r && c1 === c) || (r2 === r && c2 === c);
    });
    if (placement && !revealedSet.has(placement.dominoIndex)) {
      setRevealedSet(prev => new Set([...prev, placement.dominoIndex]));
    }
  }, [solution, mode, revealedSet]);

  const reset = () => { setRevealedSet(new Set()); setSolution(null); setDebugLogs([]); };
  const goHome = () => { 
    setView('home'); 
    setPuzzle(null); 
    setSolution(null); 
    setBuilderInitialData(null); 
    setReferenceImageUrl(null);
    setExtractionImage(null);
    setError(null);
    setCurrentPuzzleJson(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900 p-4">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="text-center mb-4">
          <h1 className="text-2xl font-bold text-white flex items-center justify-center gap-2 cursor-pointer" onClick={goHome}>
            <Sparkles className="w-6 h-6 text-yellow-400" /> Pips Solver
          </h1>
        </div>

        {/* Home View */}
        {view === 'home' && (
          <div className="space-y-4">
            {/* Resume Draft */}
            {hasDraft && draftData && (
              <div className="bg-yellow-500/20 border border-yellow-500/50 rounded-xl p-3 flex items-center gap-3">
                <div className="text-2xl">📝</div>
                <div className="flex-1">
                  <p className="text-white font-medium">Continue where you left off?</p>
                  <p className="text-yellow-200 text-sm">Step {draftData.step}/5 • {new Date(draftData.timestamp).toLocaleTimeString()}</p>
                </div>
                <button onClick={resumeDraft} className="px-3 py-1.5 bg-yellow-500 text-black rounded-lg text-sm font-medium">Resume</button>
                <button onClick={clearDraft} className="text-yellow-300 hover:text-white"><X className="w-5 h-5" /></button>
              </div>
            )}
          
            {/* Main Actions */}
            <div className="space-y-2">
              <p className="text-purple-300 text-sm text-center">Upload puzzle screenshot:</p>
              <div className="grid grid-cols-2 gap-3">
                <label className="p-4 bg-white/10 hover:bg-white/20 rounded-2xl text-white flex flex-col items-center gap-2 transition-all cursor-pointer">
                  <input type="file" accept="image/*" className="hidden" onChange={e => handleImageUpload(e.target.files?.[0], true)} />
                  <Camera className="w-8 h-8 text-purple-300" />
                  <span className="font-medium text-sm">Overlay Trace</span>
                  <span className="text-xs text-purple-300 text-center">Tap on image</span>
                </label>
                <label className="p-4 bg-white/10 hover:bg-white/20 rounded-2xl text-white flex flex-col items-center gap-2 transition-all cursor-pointer">
                  <input type="file" accept="image/*" className="hidden" onChange={e => handleImageUpload(e.target.files?.[0], false)} />
                  <Wand2 className="w-8 h-8 text-purple-300" />
                  <span className="font-medium text-sm">AI Extract</span>
                  <span className="text-xs text-purple-300 text-center">Auto + edit</span>
                </label>
              </div>
            </div>
            
            <div className="flex gap-2">
              <button onClick={() => { setReferenceImageUrl(null); setBuilderInitialData(null); setView('builder'); }}
                className="flex-1 py-3 bg-white/10 hover:bg-white/20 rounded-xl text-white flex items-center justify-center gap-2">
                <Palette className="w-5 h-5" /> Manual Build
              </button>
              <button onClick={() => setView('archive')}
                className="flex-1 py-3 bg-white/10 hover:bg-white/20 rounded-xl text-white flex items-center justify-center gap-2">
                📅 Archive
              </button>
            </div>

            {/* Quick JSON Paste */}
            <details className="bg-white/5 rounded-xl">
              <summary className="p-3 text-purple-300 text-sm cursor-pointer hover:text-white">
                Paste JSON directly...
              </summary>
              <div className="px-3 pb-3 space-y-2">
                <textarea 
                  id="jsonPaste"
                  className="w-full h-32 bg-gray-900 text-green-400 font-mono text-xs rounded-lg p-2 border border-purple-500/30"
                  placeholder='{"board": {...}, "regions": "...", "constraints": {...}, "dominoes": [...]}'
                />
                <button 
                  onClick={() => {
                    try {
                      const json = JSON.parse(document.getElementById('jsonPaste').value);
                      loadPuzzle(json);
                    } catch (e) {
                      setError('Invalid JSON: ' + e.message);
                    }
                  }}
                  className="w-full py-2 bg-purple-500 text-white rounded-lg text-sm"
                >
                  Load JSON
                </button>
              </div>
            </details>

            {/* Sample Puzzles */}
            <div className="bg-white/5 rounded-xl p-4">
              <p className="text-purple-300 text-sm font-medium mb-2 text-center">Samples:</p>
              <div className="flex flex-wrap gap-2 justify-center">
                {SAMPLE_PUZZLES.map((s, i) => (
                  <button key={i} onClick={() => loadPuzzle(s.data)}
                    className="px-3 py-1.5 bg-white/10 hover:bg-white/20 text-white rounded-lg flex items-center gap-1 text-sm">
                    <Zap className="w-3 h-3 text-yellow-400" /> {s.name}
                  </button>
                ))}
              </div>
            </div>
            
            {error && (
              <div className="p-3 bg-red-500/20 border border-red-400 rounded-xl text-red-200 text-sm">
                {error}
              </div>
            )}
          </div>
        )}
        
        {/* Archive View */}
        {view === 'archive' && (
          <PuzzleArchive onSelect={loadPuzzle} onClose={goHome} />
        )}
        
        {/* Overlay Tracing View */}
        {view === 'overlay' && (
          referenceImageUrl ? (
            <OverlayBuilder 
              imageUrl={referenceImageUrl} 
              onComplete={handleOverlayComplete}
              onCancel={goHome}
              savedState={draftData?.imageUrl === referenceImageUrl ? draftData : null}
            />
          ) : (
            <div className="bg-red-500/20 p-4 rounded-xl text-red-200">
              Error: No image loaded. <button onClick={goHome} className="underline">Go back</button>
            </div>
          )
        )}

        {/* Extraction View */}
        {view === 'extraction' && extractionImage && (
          <AutoExtractor
            imageData={extractionImage}
            mediaType={extractionMediaType}
            onComplete={handleExtractionComplete}
            onCancel={goHome}
            onSkip={handleSkipExtraction}
          />
        )}

        {/* Builder View */}
        {view === 'builder' && (
          <div className="space-y-4">
            <button onClick={goHome} className="text-purple-300 hover:text-white flex items-center gap-1 text-sm">
              <ChevronLeft className="w-4 h-4" /> Back
            </button>
            <PuzzleBuilder
              onPuzzleReady={loadPuzzle}
              initialData={builderInitialData}
              onStepChange={() => {}}
              referenceImage={referenceImageUrl}
            />
          </div>
        )}

        {/* Solve View */}
        {view === 'solve' && puzzle && (
          <div className="space-y-4">
            <button onClick={goHome} className="text-purple-300 hover:text-white flex items-center gap-1 text-sm">
              <ChevronLeft className="w-4 h-4" /> Back
            </button>

            {error && (
              <div className="p-3 bg-red-500/20 border border-red-400 rounded-xl text-red-200 text-sm flex items-start gap-2">
                <X className="w-5 h-5 flex-shrink-0" /> {error}
              </div>
            )}

            <div className="bg-white/10 rounded-2xl p-4 space-y-4">
              {/* Mode */}
              <div className="flex justify-center gap-2">
                {[{ id: 'complete', label: 'Full', icon: Eye }, { id: 'hint', label: 'Tap', icon: Target }, { id: 'progressive', label: 'Step', icon: ChevronRight }].map(({ id, label, icon: Icon }) => (
                  <button key={id} onClick={() => setMode(id)}
                    className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 text-sm ${mode === id ? 'bg-purple-500 text-white' : 'bg-white/10 text-purple-200'}`}>
                    <Icon className="w-4 h-4" /> {label}
                  </button>
                ))}
              </div>

              {/* Board */}
              <div className="flex justify-center overflow-x-auto">
                <PuzzleBoard puzzle={puzzle} solution={solution} revealedSet={revealedSet} onCellClick={handleCellClick} hintMode={mode === 'hint' && solution?.solved} />
              </div>

              {/* Dominoes */}
              <DominoTray dominoes={puzzle.dominoes} usedIndices={revealedSet} />

              {/* Controls */}
              <div className="flex flex-wrap justify-center gap-2">
                {!solution && (
                  <button onClick={handleSolve} disabled={loading}
                    className="px-4 py-2 bg-green-500 text-white rounded-xl font-semibold flex items-center gap-2">
                    {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5" />} Solve
                  </button>
                )}
                {solution?.solved && mode === 'progressive' && (
                  <button onClick={revealNext} disabled={revealedSet.size === solution.placements.length}
                    className="px-4 py-2 bg-blue-500 text-white rounded-xl font-semibold flex items-center gap-2 disabled:opacity-50">
                    <Lightbulb className="w-5 h-5" /> Next ({revealedSet.size}/{solution.placements.length})
                  </button>
                )}
                {solution?.solved && (
                  <button onClick={revealAll} className="px-4 py-2 bg-purple-500 text-white rounded-xl font-semibold flex items-center gap-2">
                    <Eye className="w-5 h-5" /> Show All
                  </button>
                )}
                <button onClick={reset} className="px-4 py-2 bg-white/20 text-white rounded-xl font-semibold flex items-center gap-2">
                  <RotateCcw className="w-5 h-5" /> Reset
                </button>
                <button onClick={() => setShowDebug(!showDebug)} className="px-3 py-2 bg-white/10 text-white rounded-xl">
                  <Bug className="w-5 h-5" />
                </button>
              </div>

              {solution?.solved && revealedSet.size === solution.placements.length && (
                <div className="text-center">
                  <span className="inline-flex items-center gap-2 px-4 py-2 bg-yellow-400 text-black rounded-xl font-bold">
                    <Check className="w-5 h-5" /> Solved!
                  </span>
                </div>
              )}

              {showDebug && debugLogs.length > 0 && (
                <div className="bg-gray-900 rounded-lg p-3 max-h-32 overflow-y-auto">
                  {debugLogs.map((log, i) => <p key={i} className="text-green-400 text-xs font-mono">{log}</p>)}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
