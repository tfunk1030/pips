/**
 * Grid calculation utilities
 * Handles cell positioning, hit testing, and coordinate transforms
 */

import { GridBounds } from '../model/overlayTypes';

export interface CellPosition {
  row: number;
  col: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ImageDimensions {
  width: number;
  height: number;
}

/**
 * Calculate pixel positions for all cells in the grid
 */
export function calculateCellPositions(
  bounds: GridBounds,
  rows: number,
  cols: number,
  imageDims: ImageDimensions
): CellPosition[][] {
  const { width: imgWidth, height: imgHeight } = imageDims;

  // Convert percentage bounds to pixels
  const left = (bounds.left / 100) * imgWidth;
  const top = (bounds.top / 100) * imgHeight;
  const right = (bounds.right / 100) * imgWidth;
  const bottom = (bounds.bottom / 100) * imgHeight;

  const gridWidth = right - left;
  const gridHeight = bottom - top;

  const cellWidth = gridWidth / cols;
  const cellHeight = gridHeight / rows;

  const positions: CellPosition[][] = [];

  for (let r = 0; r < rows; r++) {
    const row: CellPosition[] = [];
    for (let c = 0; c < cols; c++) {
      row.push({
        row: r,
        col: c,
        x: left + c * cellWidth,
        y: top + r * cellHeight,
        width: cellWidth,
        height: cellHeight,
      });
    }
    positions.push(row);
  }

  return positions;
}

/**
 * Hit test to determine which cell was tapped
 * Returns null if outside grid
 */
export function hitTestCell(
  x: number,
  y: number,
  bounds: GridBounds,
  rows: number,
  cols: number,
  imageDims: ImageDimensions
): { row: number; col: number } | null {
  const { width: imgWidth, height: imgHeight } = imageDims;

  // Convert percentage bounds to pixels
  const left = (bounds.left / 100) * imgWidth;
  const top = (bounds.top / 100) * imgHeight;
  const right = (bounds.right / 100) * imgWidth;
  const bottom = (bounds.bottom / 100) * imgHeight;

  // Check if point is inside grid
  if (x < left || x > right || y < top || y > bottom) {
    return null;
  }

  const gridWidth = right - left;
  const gridHeight = bottom - top;

  const cellWidth = gridWidth / cols;
  const cellHeight = gridHeight / rows;

  const col = Math.floor((x - left) / cellWidth);
  const row = Math.floor((y - top) / cellHeight);

  // Clamp to valid range
  if (row < 0 || row >= rows || col < 0 || col >= cols) {
    return null;
  }

  return { row, col };
}

/**
 * Get cell center point in pixels
 */
export function getCellCenter(
  row: number,
  col: number,
  bounds: GridBounds,
  rows: number,
  cols: number,
  imageDims: ImageDimensions
): { x: number; y: number } {
  const positions = calculateCellPositions(bounds, rows, cols, imageDims);
  const cell = positions[row]?.[col];

  if (!cell) {
    return { x: 0, y: 0 };
  }

  return {
    x: cell.x + cell.width / 2,
    y: cell.y + cell.height / 2,
  };
}

/**
 * Convert pixel coordinates to percentage bounds
 */
export function pixelsToBounds(
  left: number,
  top: number,
  right: number,
  bottom: number,
  imageDims: ImageDimensions
): GridBounds {
  return {
    left: (left / imageDims.width) * 100,
    top: (top / imageDims.height) * 100,
    right: (right / imageDims.width) * 100,
    bottom: (bottom / imageDims.height) * 100,
  };
}

/**
 * Constrain bounds to valid range
 */
export function constrainBounds(bounds: GridBounds): GridBounds {
  return {
    left: Math.max(0, Math.min(bounds.left, bounds.right - 10)),
    top: Math.max(0, Math.min(bounds.top, bounds.bottom - 10)),
    right: Math.min(100, Math.max(bounds.right, bounds.left + 10)),
    bottom: Math.min(100, Math.max(bounds.bottom, bounds.top + 10)),
  };
}

/**
 * Calculate optimal bounds for a grid given an aspect ratio
 */
export function calculateOptimalBounds(
  rows: number,
  cols: number,
  imageDims: ImageDimensions,
  padding: number = 10
): GridBounds {
  const imageAspect = imageDims.width / imageDims.height;
  const gridAspect = cols / rows;

  let bounds: GridBounds;

  if (gridAspect > imageAspect) {
    // Grid is wider than image - fit width
    const height = ((100 - 2 * padding) * imageAspect) / gridAspect;
    const topPad = (100 - height) / 2;
    bounds = {
      left: padding,
      top: topPad,
      right: 100 - padding,
      bottom: 100 - topPad,
    };
  } else {
    // Grid is taller than image - fit height
    const width = ((100 - 2 * padding) * gridAspect) / imageAspect;
    const leftPad = (100 - width) / 2;
    bounds = {
      left: leftPad,
      top: padding,
      right: 100 - leftPad,
      bottom: 100 - padding,
    };
  }

  return constrainBounds(bounds);
}

/**
 * Snap bounds to create even cell sizes
 */
export function snapBoundsToGrid(
  bounds: GridBounds,
  rows: number,
  cols: number
): GridBounds {
  // Ensure grid dimensions are multiples of cell count for clean rendering
  const width = bounds.right - bounds.left;
  const height = bounds.bottom - bounds.top;

  // Round to nearest cell boundary
  const cellWidth = width / cols;
  const cellHeight = height / rows;

  return {
    left: bounds.left,
    top: bounds.top,
    right: bounds.left + cellWidth * cols,
    bottom: bounds.top + cellHeight * rows,
  };
}
