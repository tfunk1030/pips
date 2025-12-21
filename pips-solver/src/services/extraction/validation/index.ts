/**
 * Validation Layer
 *
 * Re-exports all validation functions.
 */

export {
  validateGridGeometry,
  validateCellDetection,
  checkRotationMismatch,
} from './gridValidator';

export {
  validateRegionMapping,
  validateConstraints,
} from './regionValidator';

export {
  validateDominoes,
  checkDominoFeasibility,
  getDominoStats,
} from './dominoValidator';

// Re-export types
export type { ValidationResult, ValidationError } from '../types';

import { ExtractionResult, ValidationResult, ValidationError } from '../types';
import { validateGridGeometry, validateCellDetection } from './gridValidator';
import { validateRegionMapping, validateConstraints } from './regionValidator';
import { validateDominoes, checkDominoFeasibility } from './dominoValidator';

/**
 * Validate complete extraction result
 */
export function validateExtractionResult(result: ExtractionResult): ValidationResult {
  const allErrors: ValidationError[] = [];

  // Validate grid geometry
  const gridResult = validateGridGeometry({
    rows: result.grid.rows,
    cols: result.grid.cols,
    confidence: result.confidence.perStage.grid,
  });
  allErrors.push(...gridResult.errors);

  // Validate cell detection
  const cellResult = validateCellDetection(
    { shape: result.grid.shape, confidence: result.confidence.perStage.cells },
    { rows: result.grid.rows, cols: result.grid.cols, confidence: result.confidence.perStage.grid }
  );
  allErrors.push(...cellResult.errors);

  // Validate region mapping
  const regionResult = validateRegionMapping(
    { regions: result.grid.regions, confidence: result.confidence.perStage.regions },
    { rows: result.grid.rows, cols: result.grid.cols, confidence: result.confidence.perStage.grid },
    { shape: result.grid.shape, confidence: result.confidence.perStage.cells }
  );
  allErrors.push(...regionResult.errors);

  // Validate constraints
  const constraintResult = validateConstraints(
    { constraints: result.constraints, confidence: result.confidence.perStage.constraints },
    { regions: result.grid.regions, confidence: result.confidence.perStage.regions }
  );
  allErrors.push(...constraintResult.errors);

  // Validate dominoes
  const dominoResult = validateDominoes(
    { dominoes: result.dominoes, confidence: result.confidence.perStage.dominoes },
    { shape: result.grid.shape, confidence: result.confidence.perStage.cells }
  );
  allErrors.push(...dominoResult.errors);

  // Check feasibility
  const cellCount = (result.grid.shape.match(/\./g) || []).length;
  const feasibility = checkDominoFeasibility(
    result.dominoes,
    cellCount,
    result.constraints
  );

  if (!feasibility.feasible) {
    for (const issue of feasibility.issues) {
      allErrors.push({
        stage: 'dominoes',
        field: 'feasibility',
        message: issue,
      });
    }
  }

  return {
    valid: allErrors.length === 0,
    errors: allErrors,
  };
}
