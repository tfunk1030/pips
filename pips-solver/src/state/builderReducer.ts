/**
 * Builder Reducer
 * State management for the OverlayBuilder wizard
 * Extracted from OverlayBuilderScreen for maintainability and testability
 */

import {
  BuilderAction,
  DominoPair,
  OverlayBuilderState,
  createInitialBuilderState,
} from '../model/overlayTypes';
import { parseConstraintShorthand } from '../utils/constraintParser';

// ════════════════════════════════════════════════════════════════════════════
// Main Reducer
// ════════════════════════════════════════════════════════════════════════════

export function builderReducer(
  state: OverlayBuilderState,
  action: BuilderAction
): OverlayBuilderState {
  switch (action.type) {
    // Navigation
    case 'SET_STEP':
      return { ...state, step: action.step };

    case 'RESET':
      return createInitialBuilderState();

    case 'LOAD_DRAFT':
      return action.state;

    // Image
    case 'SET_IMAGE':
      return { ...state, image: action.image };

    // Grid (Step 1)
    case 'SET_GRID_BOUNDS':
      return { ...state, grid: { ...state.grid, bounds: action.bounds } };

    case 'SET_ROWS': {
      // Allow up to 12 rows to match AI extraction capability
      const newRows = Math.max(2, Math.min(12, action.rows));
      return {
        ...state,
        grid: {
          ...state.grid,
          rows: newRows,
          holes: resizeArray2D(state.grid.holes, newRows, state.grid.cols, false),
        },
        regions: {
          ...state.regions,
          regionGrid: resizeArray2D(state.regions.regionGrid, newRows, state.grid.cols, 0),
        },
      };
    }

    case 'SET_COLS': {
      // Allow up to 12 cols to match AI extraction capability
      const newCols = Math.max(2, Math.min(12, action.cols));
      return {
        ...state,
        grid: {
          ...state.grid,
          cols: newCols,
          holes: resizeArray2D(state.grid.holes, state.grid.rows, newCols, false),
        },
        regions: {
          ...state.regions,
          regionGrid: resizeArray2D(state.regions.regionGrid, state.grid.rows, newCols, 0),
        },
      };
    }

    case 'RESIZE_GRID': {
      // Allow up to 12x12 to match AI extraction capability
      const rows = Math.max(2, Math.min(12, action.rows));
      const cols = Math.max(2, Math.min(12, action.cols));
      return {
        ...state,
        grid: {
          ...state.grid,
          rows,
          cols,
          holes: resizeArray2D(state.grid.holes, rows, cols, false),
        },
        regions: {
          ...state.regions,
          regionGrid: resizeArray2D(state.regions.regionGrid, rows, cols, 0),
        },
      };
    }

    case 'TOGGLE_HOLE': {
      const newHoles = state.grid.holes.map((row, r) =>
        row.map((cell, c) => (r === action.row && c === action.col ? !cell : cell))
      );
      // Also update region grid - holes become null
      const newRegionGrid = state.regions.regionGrid.map((row, r) =>
        row.map((cell, c) => {
          if (r === action.row && c === action.col) {
            return newHoles[r][c] ? null : cell ?? 0;
          }
          return cell;
        })
      );
      return {
        ...state,
        grid: { ...state.grid, holes: newHoles },
        regions: { ...state.regions, regionGrid: newRegionGrid },
      };
    }

    // Regions (Step 2)
    case 'SELECT_PALETTE_COLOR':
      return {
        ...state,
        regions: {
          ...state.regions,
          palette: { ...state.regions.palette, selectedIndex: action.index },
        },
      };

    case 'PAINT_CELL': {
      // Don't paint holes
      if (state.grid.holes[action.row]?.[action.col]) {
        return state;
      }
      const newRegionGrid = state.regions.regionGrid.map((row, r) =>
        row.map((cell, c) =>
          r === action.row && c === action.col ? state.regions.palette.selectedIndex : cell
        )
      );
      return {
        ...state,
        regions: { ...state.regions, regionGrid: newRegionGrid },
      };
    }

    case 'CLEAR_REGION': {
      const newRegionGrid = state.regions.regionGrid.map(row =>
        row.map(cell => (cell === action.regionIndex ? 0 : cell))
      );
      return {
        ...state,
        regions: { ...state.regions, regionGrid: newRegionGrid },
      };
    }

    // Constraints (Step 3)
    case 'SELECT_REGION':
      return {
        ...state,
        constraints: { ...state.constraints, selectedRegion: action.regionIndex },
      };

    case 'SET_CONSTRAINT':
      return {
        ...state,
        constraints: {
          ...state.constraints,
          regionConstraints: {
            ...state.constraints.regionConstraints,
            [action.regionIndex]: action.constraint,
          },
        },
      };

    case 'APPLY_CONSTRAINT_SHORTHAND': {
      const parsed = parseConstraintShorthand(action.shorthand);
      return {
        ...state,
        constraints: {
          ...state.constraints,
          regionConstraints: {
            ...state.constraints.regionConstraints,
            ...parsed,
          },
        },
      };
    }

    // Dominoes (Step 4)
    case 'ADD_DOMINO':
      return {
        ...state,
        dominoes: {
          ...state.dominoes,
          dominoes: [...state.dominoes.dominoes, [action.pip1, action.pip2]],
        },
      };

    case 'REMOVE_DOMINO':
      return {
        ...state,
        dominoes: {
          ...state.dominoes,
          dominoes: state.dominoes.dominoes.filter((_, i) => i !== action.index),
        },
      };

    case 'UPDATE_DOMINO': {
      const newDominoes = [...state.dominoes.dominoes];
      newDominoes[action.index] = [action.pip1, action.pip2];
      return {
        ...state,
        dominoes: { ...state.dominoes, dominoes: newDominoes },
      };
    }

    case 'CYCLE_DOMINO_PIP': {
      const newDominoes = [...state.dominoes.dominoes];
      const domino = [...newDominoes[action.dominoIndex]] as DominoPair;
      domino[action.half] = (domino[action.half] + action.direction + 7) % 7;
      newDominoes[action.dominoIndex] = domino;
      return {
        ...state,
        dominoes: { ...state.dominoes, dominoes: newDominoes },
      };
    }

    case 'SET_DOMINOES':
      return {
        ...state,
        dominoes: { ...state.dominoes, dominoes: action.dominoes },
      };

    case 'AUTO_FILL_DOMINOES': {
      const cellCount = countValidCells(state.grid.holes);
      const needed = Math.floor(cellCount / 2);

      // Only auto-fill if we don't already have extracted dominoes
      // Check if existing dominoes are "real" (not all [0,0])
      const existingDominoes = state.dominoes.dominoes;
      const hasRealDominoes =
        existingDominoes.length > 0 && existingDominoes.some(([a, b]) => a !== 0 || b !== 0);

      if (hasRealDominoes && existingDominoes.length === needed) {
        // Keep existing dominoes, just update expected count
        return {
          ...state,
          dominoes: { ...state.dominoes, expectedCount: needed },
        };
      }

      // Fill with empty dominoes only if we don't have real ones
      const dominoes: DominoPair[] = Array(needed)
        .fill(null)
        .map(() => [0, 0]);
      return {
        ...state,
        dominoes: { ...state.dominoes, dominoes, expectedCount: needed },
      };
    }

    // AI Extraction
    case 'AI_START':
      return { ...state, aiStatus: 'extracting', aiError: undefined };

    case 'AI_SUCCESS': {
      console.log('[DEBUG] builderReducer AI_SUCCESS:', {
        gridData: action.grid,
        regionsData: action.regions,
        constraintsData: action.constraints,
        dominoCount: action.dominoes.length,
      });

      // Validate dimension consistency between grid and regions
      const aiRows = action.grid.rows || 0;
      const aiCols = action.grid.cols || 0;
      const regionGridRows = action.regions.regionGrid?.length || 0;
      const regionGridCols = action.regions.regionGrid?.[0]?.length || 0;

      if (regionGridRows !== aiRows || regionGridCols !== aiCols) {
        console.warn('[AI_SUCCESS] Dimension mismatch detected:', {
          gridDims: `${aiRows}x${aiCols}`,
          regionGridDims: `${regionGridRows}x${regionGridCols}`,
        });
        // Resize regionGrid to match grid dimensions
        if (action.regions.regionGrid) {
          const resizedRegionGrid = resizeArray2D(
            action.regions.regionGrid,
            aiRows,
            aiCols,
            0
          );
          action.regions.regionGrid = resizedRegionGrid;
          console.log('[AI_SUCCESS] Resized regionGrid to match grid dimensions');
        }
      }

      // Merge AI results into state
      // Calculate optimal bounds for the new grid if we have image dimensions
      let gridBounds = action.grid.bounds || state.grid.bounds;
      if (state.image && action.grid.rows && action.grid.cols) {
        const { calculateOptimalBounds } = require('../utils/gridCalculations');
        gridBounds = calculateOptimalBounds(action.grid.rows, action.grid.cols, {
          width: state.image.width,
          height: state.image.height,
        });
        console.log('[DEBUG] Calculated optimal bounds:', gridBounds);
      }

      const newGrid = { ...state.grid, ...action.grid, bounds: gridBounds };
      const newRegions = { ...state.regions, ...action.regions };
      const newConstraints = { ...state.constraints, ...action.constraints };

      console.log('[DEBUG] New state after AI_SUCCESS:', {
        gridRows: newGrid.rows,
        gridCols: newGrid.cols,
        regionGridSize: newRegions.regionGrid?.length,
        constraintCount: Object.keys(newConstraints.regionConstraints || {}).length,
        dominoCount: action.dominoes.length,
      });

      return {
        ...state,
        grid: newGrid,
        regions: newRegions,
        constraints: newConstraints,
        dominoes: { ...state.dominoes, dominoes: action.dominoes },
        aiStatus: 'done',
        aiReasoning: action.reasoning,
        aiConfidence: action.confidence,
      };
    }

    case 'AI_ERROR':
      return { ...state, aiStatus: 'error', aiError: action.error };

    default:
      return state;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ════════════════════════════════════════════════════════════════════════════

/**
 * Resize a 2D array preserving existing values
 */
export function resizeArray2D<T>(
  arr: T[][],
  newRows: number,
  newCols: number,
  defaultValue: T
): T[][] {
  return Array(newRows)
    .fill(null)
    .map((_, r) =>
      Array(newCols)
        .fill(null)
        .map((_, c) => arr[r]?.[c] ?? defaultValue)
    );
}

/**
 * Count non-hole cells in the grid
 */
export function countValidCells(holes: boolean[][]): number {
  return holes.flat().filter(h => !h).length;
}

/**
 * Get initial state for the builder
 */
export function getInitialBuilderState(): OverlayBuilderState {
  return createInitialBuilderState();
}
