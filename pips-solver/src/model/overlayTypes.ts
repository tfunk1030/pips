/**
 * Types for the OverlayBuilder workflow
 * Used to create puzzles from screenshots via a 4-step wizard
 */

// ════════════════════════════════════════════════════════════════════════════
// Grid State (Step 1)
// ════════════════════════════════════════════════════════════════════════════

export interface GridBounds {
  /** Percentage from left edge of image (0-100) */
  left: number;
  /** Percentage from top edge of image (0-100) */
  top: number;
  /** Percentage from left edge of image (0-100) */
  right: number;
  /** Percentage from top edge of image (0-100) */
  bottom: number;
}

export interface GridState {
  rows: number;
  cols: number;
  bounds: GridBounds;
  /** 2D array where true = hole (no cell), false = valid cell */
  holes: boolean[][];
}

// ════════════════════════════════════════════════════════════════════════════
// Region State (Step 2)
// ════════════════════════════════════════════════════════════════════════════

export interface RegionPalette {
  colors: string[];
  labels: string[];
  selectedIndex: number;
}

export interface RegionState {
  /** 2D array where null = hole, 0-9 = region index */
  regionGrid: (number | null)[][];
  palette: RegionPalette;
}

// ════════════════════════════════════════════════════════════════════════════
// Constraint State (Step 3)
// ════════════════════════════════════════════════════════════════════════════

export type ConstraintType = 'none' | 'sum' | 'all_equal' | 'all_different';
export type ConstraintOp = '==' | '<' | '>' | '!=';

export interface ConstraintDef {
  type: ConstraintType;
  op?: ConstraintOp;
  value?: number;
}

export interface ConstraintState {
  /** Map from region index to constraint definition */
  regionConstraints: Record<number, ConstraintDef>;
  selectedRegion: number | null;
}

// ════════════════════════════════════════════════════════════════════════════
// Domino State (Step 4)
// ════════════════════════════════════════════════════════════════════════════

export type DominoPair = [number, number];

export interface DominoState {
  dominoes: DominoPair[];
  /** Expected count based on cell count / 2 */
  expectedCount: number;
}

// ════════════════════════════════════════════════════════════════════════════
// Overall Builder State
// ════════════════════════════════════════════════════════════════════════════

export type BuilderStep = 1 | 2 | 3 | 4;

export interface ImageInfo {
  uri: string;
  width: number;
  height: number;
  base64?: string;
}

export interface OverlayBuilderState {
  step: BuilderStep;
  image: ImageInfo | null;
  grid: GridState;
  regions: RegionState;
  constraints: ConstraintState;
  dominoes: DominoState;
  draftId: string;
  draftUpdatedAt: number;
  /** AI extraction status */
  aiStatus: 'idle' | 'extracting' | 'done' | 'error';
  aiError?: string;
  aiReasoning?: string;
}

// ════════════════════════════════════════════════════════════════════════════
// Builder Actions (for useReducer)
// ════════════════════════════════════════════════════════════════════════════

export type BuilderAction =
  // Navigation
  | { type: 'SET_STEP'; step: BuilderStep }
  | { type: 'RESET' }
  | { type: 'LOAD_DRAFT'; state: OverlayBuilderState }

  // Image
  | { type: 'SET_IMAGE'; image: ImageInfo }

  // Grid (Step 1)
  | { type: 'SET_GRID_BOUNDS'; bounds: GridBounds }
  | { type: 'SET_ROWS'; rows: number }
  | { type: 'SET_COLS'; cols: number }
  | { type: 'TOGGLE_HOLE'; row: number; col: number }
  | { type: 'RESIZE_GRID'; rows: number; cols: number }

  // Regions (Step 2)
  | { type: 'SELECT_PALETTE_COLOR'; index: number }
  | { type: 'PAINT_CELL'; row: number; col: number }
  | { type: 'CLEAR_REGION'; regionIndex: number }

  // Constraints (Step 3)
  | { type: 'SELECT_REGION'; regionIndex: number | null }
  | { type: 'SET_CONSTRAINT'; regionIndex: number; constraint: ConstraintDef }
  | { type: 'APPLY_CONSTRAINT_SHORTHAND'; shorthand: string }

  // Dominoes (Step 4)
  | { type: 'ADD_DOMINO'; pip1: number; pip2: number }
  | { type: 'REMOVE_DOMINO'; index: number }
  | { type: 'UPDATE_DOMINO'; index: number; pip1: number; pip2: number }
  | { type: 'CYCLE_DOMINO_PIP'; dominoIndex: number; half: 0 | 1; direction: 1 | -1 }
  | { type: 'SET_DOMINOES'; dominoes: DominoPair[] }
  | { type: 'AUTO_FILL_DOMINOES' }

  // AI Extraction
  | { type: 'AI_START' }
  | { type: 'AI_SUCCESS'; grid: Partial<GridState>; regions: Partial<RegionState>; constraints: Partial<ConstraintState>; dominoes: DominoPair[]; reasoning: string }
  | { type: 'AI_ERROR'; error: string };

// ════════════════════════════════════════════════════════════════════════════
// Draft Recovery
// ════════════════════════════════════════════════════════════════════════════

export interface DraftMeta {
  draftId: string;
  imageUri: string;
  step: BuilderStep;
  updatedAt: number;
  rows: number;
  cols: number;
}

// ════════════════════════════════════════════════════════════════════════════
// AI Extraction Types
// ════════════════════════════════════════════════════════════════════════════

export interface BoardExtractionResult {
  rows: number;
  cols: number;
  shape: string;
  regions: string;
  constraints: Record<string, { type: string; op?: string; value?: number }>;
}

export interface DominoExtractionResult {
  dominoes: DominoPair[];
}

export interface AIExtractionResult {
  board: BoardExtractionResult;
  dominoes: DominoExtractionResult;
  reasoning: string;
}

// ════════════════════════════════════════════════════════════════════════════
// Default Values
// ════════════════════════════════════════════════════════════════════════════

export const DEFAULT_PALETTE: RegionPalette = {
  colors: [
    '#FF9800', // A - Orange
    '#009688', // B - Teal
    '#9C27B0', // C - Purple
    '#E91E63', // D - Pink
    '#4CAF50', // E - Green
    '#2196F3', // F - Blue
    '#FF5722', // G - Deep Orange
    '#607D8B', // H - Blue Gray
    '#795548', // I - Brown
    '#00BCD4', // J - Cyan
  ],
  labels: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
  selectedIndex: 0,
};

export const DEFAULT_GRID_BOUNDS: GridBounds = {
  left: 10,
  top: 10,
  right: 90,
  bottom: 80,
};

export function createDefaultGridState(rows = 4, cols = 4): GridState {
  return {
    rows,
    cols,
    bounds: { ...DEFAULT_GRID_BOUNDS },
    holes: Array(rows).fill(null).map(() => Array(cols).fill(false)),
  };
}

export function createDefaultRegionState(rows: number, cols: number): RegionState {
  return {
    regionGrid: Array(rows).fill(null).map(() => Array(cols).fill(0)),
    palette: { ...DEFAULT_PALETTE },
  };
}

export function createDefaultConstraintState(): ConstraintState {
  return {
    regionConstraints: {},
    selectedRegion: null,
  };
}

export function createDefaultDominoState(): DominoState {
  return {
    dominoes: [],
    expectedCount: 0,
  };
}

export function createInitialBuilderState(): OverlayBuilderState {
  return {
    step: 1,
    image: null,
    grid: createDefaultGridState(),
    regions: createDefaultRegionState(4, 4),
    constraints: createDefaultConstraintState(),
    dominoes: createDefaultDominoState(),
    draftId: `draft_${Date.now()}`,
    draftUpdatedAt: Date.now(),
    aiStatus: 'idle',
  };
}
