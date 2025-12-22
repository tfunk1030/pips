/**
 * Comprehensive tests for gridValidator.ts
 *
 * Tests the grid comparison logic with various disagreement scenarios including:
 * - Utility functions (cellKey, parseCellKey, generateDisagreementId)
 * - Severity classification functions
 * - Main comparison function with different types of disagreements
 */

import {
  // Types
  DisagreementSeverity,
  DisagreementType,
  BoardComparisonInput,
  DominoComparisonInput,
  GridDimensionDisagreement,

  // Utility functions
  cellKey,
  parseCellKey,
  generateDisagreementId,
  createEmptyComparisonResult,

  // Severity functions
  classifyDisagreementSeverity,
  isSeverityAtLeast,
  compareSeverities,
  sortDisagreementsBySeverity,
  filterDisagreementsBySeverity,
  getHighestSeverity,
  getSeverityReason,
  DISAGREEMENT_SEVERITY_MAP,
  SEVERITY_ORDER,
  SEVERITY_PRIORITY,

  // Main comparison function
  compareCellDetections,
} from '../gridValidator';

import type {
  BoardExtractionResult,
  DominoExtractionResult,
  DominoPair,
} from '../../../../model/overlayTypes';

// ════════════════════════════════════════════════════════════════════════════
// Test Fixtures
// ════════════════════════════════════════════════════════════════════════════

/**
 * Helper to create a BoardComparisonInput item
 */
function createBoardResponse(
  model: string,
  data: Partial<BoardExtractionResult>,
  parseSuccess = true
): BoardComparisonInput[0] {
  const fullData: BoardExtractionResult = {
    rows: data.rows ?? 3,
    cols: data.cols ?? 3,
    shape: data.shape ?? '...\n...\n...',
    regions: data.regions ?? 'AAB\nAAB\nCCB',
    constraints: data.constraints ?? {
      A: { type: 'sum', value: 10, op: '==' },
      B: { type: 'sum', value: 8, op: '==' },
      C: { type: 'all_different' },
    },
    confidence: data.confidence ?? { grid: 0.9, regions: 0.85, constraints: 0.8 },
    ...(data.gridLocation && { gridLocation: data.gridLocation }),
  };

  return {
    model,
    parsedData: parseSuccess ? fullData : null,
    parseSuccess,
    confidence: fullData.confidence
      ? (fullData.confidence.grid + fullData.confidence.regions + fullData.confidence.constraints) / 3
      : undefined,
  };
}

/**
 * Helper to create a DominoComparisonInput item
 */
function createDominoResponse(
  model: string,
  dominoes: DominoPair[],
  parseSuccess = true,
  confidence = 0.9
): DominoComparisonInput[0] {
  const data: DominoExtractionResult = { dominoes, confidence };

  return {
    model,
    parsedData: parseSuccess ? data : null,
    parseSuccess,
    confidence,
  };
}

// ════════════════════════════════════════════════════════════════════════════
// Utility Function Tests
// ════════════════════════════════════════════════════════════════════════════

describe('Utility Functions', () => {
  describe('cellKey', () => {
    it('generates a key from row and column', () => {
      expect(cellKey(0, 0)).toBe('0,0');
      expect(cellKey(2, 5)).toBe('2,5');
      expect(cellKey(10, 20)).toBe('10,20');
    });

    it('handles edge cases', () => {
      expect(cellKey(0, 0)).toBe('0,0');
      expect(cellKey(999, 999)).toBe('999,999');
    });
  });

  describe('parseCellKey', () => {
    it('parses a key back to coordinates', () => {
      expect(parseCellKey('0,0')).toEqual({ row: 0, col: 0 });
      expect(parseCellKey('2,5')).toEqual({ row: 2, col: 5 });
      expect(parseCellKey('10,20')).toEqual({ row: 10, col: 20 });
    });

    it('roundtrips with cellKey', () => {
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          const key = cellKey(r, c);
          const coord = parseCellKey(key);
          expect(coord).toEqual({ row: r, col: c });
        }
      }
    });
  });

  describe('generateDisagreementId', () => {
    it('creates unique IDs for different types', () => {
      const id1 = generateDisagreementId('grid_dimensions', 'rows');
      const id2 = generateDisagreementId('grid_dimensions', 'cols');
      expect(id1).toBe('grid_dimensions:rows');
      expect(id2).toBe('grid_dimensions:cols');
      expect(id1).not.toBe(id2);
    });

    it('includes multiple identifiers', () => {
      const id = generateDisagreementId('hole_position', 2, 3);
      expect(id).toBe('hole_position:2:3');
    });

    it('handles numeric and string identifiers', () => {
      const id1 = generateDisagreementId('constraint_type', 'A');
      const id2 = generateDisagreementId('domino_value', 1, 2);
      expect(id1).toBe('constraint_type:A');
      expect(id2).toBe('domino_value:1:2');
    });
  });

  describe('createEmptyComparisonResult', () => {
    it('creates result with zero disagreements', () => {
      const result = createEmptyComparisonResult(['model1', 'model2']);

      expect(result.summary).toEqual({ total: 0, critical: 0, warning: 0, info: 0 });
      expect(result.allDisagreements).toHaveLength(0);
      expect(result.isUnanimous).toBe(true);
      expect(result.modelsCompared).toEqual(['model1', 'model2']);
    });

    it('includes empty disagreement collections', () => {
      const result = createEmptyComparisonResult(['a', 'b']);

      expect(result.disagreementsByType.gridDimensions).toEqual([]);
      expect(result.disagreementsByType.holePositions).toEqual([]);
      expect(result.disagreementsByType.regionAssignments).toEqual([]);
      expect(result.disagreementsByType.constraints).toEqual([]);
      expect(result.disagreementsByType.dominoes).toEqual([]);
    });

    it('includes timestamp', () => {
      const before = Date.now();
      const result = createEmptyComparisonResult([]);
      const after = Date.now();

      expect(result.comparedAt).toBeGreaterThanOrEqual(before);
      expect(result.comparedAt).toBeLessThanOrEqual(after);
    });
  });
});

// ════════════════════════════════════════════════════════════════════════════
// Severity Classification Tests
// ════════════════════════════════════════════════════════════════════════════

describe('Severity Classification', () => {
  describe('DISAGREEMENT_SEVERITY_MAP', () => {
    it('classifies critical disagreements correctly', () => {
      expect(DISAGREEMENT_SEVERITY_MAP.grid_dimensions).toBe('critical');
      expect(DISAGREEMENT_SEVERITY_MAP.hole_position).toBe('critical');
      expect(DISAGREEMENT_SEVERITY_MAP.domino_count).toBe('critical');
    });

    it('classifies warning disagreements correctly', () => {
      expect(DISAGREEMENT_SEVERITY_MAP.region_assignment).toBe('warning');
      expect(DISAGREEMENT_SEVERITY_MAP.constraint_type).toBe('warning');
      expect(DISAGREEMENT_SEVERITY_MAP.constraint_value).toBe('warning');
      expect(DISAGREEMENT_SEVERITY_MAP.constraint_operator).toBe('warning');
      expect(DISAGREEMENT_SEVERITY_MAP.domino_value).toBe('warning');
    });

    it('covers all disagreement types', () => {
      const allTypes: DisagreementType[] = [
        'grid_dimensions',
        'hole_position',
        'region_assignment',
        'constraint_type',
        'constraint_value',
        'constraint_operator',
        'domino_count',
        'domino_value',
      ];

      for (const type of allTypes) {
        expect(DISAGREEMENT_SEVERITY_MAP[type]).toBeDefined();
      }
    });
  });

  describe('classifyDisagreementSeverity', () => {
    it('returns critical for grid dimension mismatches', () => {
      expect(classifyDisagreementSeverity('grid_dimensions')).toBe('critical');
    });

    it('returns critical for hole position mismatches', () => {
      expect(classifyDisagreementSeverity('hole_position')).toBe('critical');
    });

    it('returns warning for region assignment differences', () => {
      expect(classifyDisagreementSeverity('region_assignment')).toBe('warning');
    });

    it('returns warning for constraint differences', () => {
      expect(classifyDisagreementSeverity('constraint_type')).toBe('warning');
      expect(classifyDisagreementSeverity('constraint_value')).toBe('warning');
      expect(classifyDisagreementSeverity('constraint_operator')).toBe('warning');
    });

    it('returns critical for domino count mismatches', () => {
      expect(classifyDisagreementSeverity('domino_count')).toBe('critical');
    });

    it('returns warning for individual domino differences', () => {
      expect(classifyDisagreementSeverity('domino_value')).toBe('warning');
    });
  });

  describe('SEVERITY_ORDER', () => {
    it('orders severities from highest to lowest priority', () => {
      expect(SEVERITY_ORDER).toEqual(['critical', 'warning', 'info']);
    });
  });

  describe('SEVERITY_PRIORITY', () => {
    it('assigns lower numbers to higher priorities', () => {
      expect(SEVERITY_PRIORITY.critical).toBeLessThan(SEVERITY_PRIORITY.warning);
      expect(SEVERITY_PRIORITY.warning).toBeLessThan(SEVERITY_PRIORITY.info);
    });

    it('has consistent values', () => {
      expect(SEVERITY_PRIORITY.critical).toBe(0);
      expect(SEVERITY_PRIORITY.warning).toBe(1);
      expect(SEVERITY_PRIORITY.info).toBe(2);
    });
  });

  describe('isSeverityAtLeast', () => {
    it('returns true when severity meets threshold', () => {
      expect(isSeverityAtLeast('critical', 'critical')).toBe(true);
      expect(isSeverityAtLeast('critical', 'warning')).toBe(true);
      expect(isSeverityAtLeast('critical', 'info')).toBe(true);
      expect(isSeverityAtLeast('warning', 'warning')).toBe(true);
      expect(isSeverityAtLeast('warning', 'info')).toBe(true);
      expect(isSeverityAtLeast('info', 'info')).toBe(true);
    });

    it('returns false when severity is below threshold', () => {
      expect(isSeverityAtLeast('warning', 'critical')).toBe(false);
      expect(isSeverityAtLeast('info', 'critical')).toBe(false);
      expect(isSeverityAtLeast('info', 'warning')).toBe(false);
    });
  });

  describe('compareSeverities', () => {
    it('returns negative when first is more severe', () => {
      expect(compareSeverities('critical', 'warning')).toBeLessThan(0);
      expect(compareSeverities('critical', 'info')).toBeLessThan(0);
      expect(compareSeverities('warning', 'info')).toBeLessThan(0);
    });

    it('returns positive when second is more severe', () => {
      expect(compareSeverities('warning', 'critical')).toBeGreaterThan(0);
      expect(compareSeverities('info', 'critical')).toBeGreaterThan(0);
      expect(compareSeverities('info', 'warning')).toBeGreaterThan(0);
    });

    it('returns zero when equal', () => {
      expect(compareSeverities('critical', 'critical')).toBe(0);
      expect(compareSeverities('warning', 'warning')).toBe(0);
      expect(compareSeverities('info', 'info')).toBe(0);
    });
  });

  describe('sortDisagreementsBySeverity', () => {
    it('sorts by severity with critical first', () => {
      const items = [
        { severity: 'info' as DisagreementSeverity },
        { severity: 'critical' as DisagreementSeverity },
        { severity: 'warning' as DisagreementSeverity },
      ];

      const sorted = sortDisagreementsBySeverity(items);

      expect(sorted[0].severity).toBe('critical');
      expect(sorted[1].severity).toBe('warning');
      expect(sorted[2].severity).toBe('info');
    });

    it('does not mutate original array', () => {
      const items = [
        { severity: 'info' as DisagreementSeverity },
        { severity: 'critical' as DisagreementSeverity },
      ];

      const sorted = sortDisagreementsBySeverity(items);

      expect(items[0].severity).toBe('info');
      expect(sorted).not.toBe(items);
    });

    it('handles empty array', () => {
      expect(sortDisagreementsBySeverity([])).toEqual([]);
    });

    it('handles single item', () => {
      const items = [{ severity: 'warning' as DisagreementSeverity }];
      expect(sortDisagreementsBySeverity(items)).toEqual(items);
    });
  });

  describe('filterDisagreementsBySeverity', () => {
    const items = [
      { severity: 'critical' as DisagreementSeverity, id: 1 },
      { severity: 'warning' as DisagreementSeverity, id: 2 },
      { severity: 'info' as DisagreementSeverity, id: 3 },
      { severity: 'warning' as DisagreementSeverity, id: 4 },
    ];

    it('filters to critical only when threshold is critical', () => {
      const filtered = filterDisagreementsBySeverity(items, 'critical');
      expect(filtered.map(d => d.id)).toEqual([1]);
    });

    it('filters to critical and warning when threshold is warning', () => {
      const filtered = filterDisagreementsBySeverity(items, 'warning');
      expect(filtered.map(d => d.id)).toEqual([1, 2, 4]);
    });

    it('returns all when threshold is info', () => {
      const filtered = filterDisagreementsBySeverity(items, 'info');
      expect(filtered).toHaveLength(4);
    });
  });

  describe('getHighestSeverity', () => {
    it('returns undefined for empty array', () => {
      expect(getHighestSeverity([])).toBeUndefined();
    });

    it('returns the only severity for single item', () => {
      expect(getHighestSeverity([{ severity: 'warning' }])).toBe('warning');
    });

    it('returns critical when present', () => {
      const items = [
        { severity: 'info' as DisagreementSeverity },
        { severity: 'critical' as DisagreementSeverity },
        { severity: 'warning' as DisagreementSeverity },
      ];
      expect(getHighestSeverity(items)).toBe('critical');
    });

    it('returns warning when no critical', () => {
      const items = [
        { severity: 'info' as DisagreementSeverity },
        { severity: 'warning' as DisagreementSeverity },
      ];
      expect(getHighestSeverity(items)).toBe('warning');
    });

    it('returns info when only info present', () => {
      const items = [
        { severity: 'info' as DisagreementSeverity },
        { severity: 'info' as DisagreementSeverity },
      ];
      expect(getHighestSeverity(items)).toBe('info');
    });
  });

  describe('getSeverityReason', () => {
    it('returns meaningful explanation for each type', () => {
      const types: DisagreementType[] = [
        'grid_dimensions',
        'hole_position',
        'domino_count',
        'region_assignment',
        'constraint_type',
        'constraint_value',
        'constraint_operator',
        'domino_value',
      ];

      for (const type of types) {
        const reason = getSeverityReason(type);
        expect(typeof reason).toBe('string');
        expect(reason.length).toBeGreaterThan(10);
      }
    });
  });
});

// ════════════════════════════════════════════════════════════════════════════
// Main Comparison Function Tests
// ════════════════════════════════════════════════════════════════════════════

describe('compareCellDetections', () => {
  describe('Edge Cases', () => {
    it('returns empty result for single model', () => {
      const board = [createBoardResponse('gemini', {})];
      const domino = [createDominoResponse('gemini', [[1, 2]])];

      const result = compareCellDetections(board, domino);

      expect(result.isUnanimous).toBe(true);
      expect(result.allDisagreements).toHaveLength(0);
      expect(result.modelsCompared).toEqual(['gemini']);
    });

    it('returns empty result when no models provided', () => {
      const result = compareCellDetections([], []);

      expect(result.isUnanimous).toBe(true);
      expect(result.allDisagreements).toHaveLength(0);
    });

    it('handles failed parses gracefully', () => {
      const board = [
        createBoardResponse('gemini', {}, true),
        createBoardResponse('claude', {}, false),
      ];
      const domino = [
        createDominoResponse('gemini', [[1, 2]], true),
        createDominoResponse('claude', [], false),
      ];

      const result = compareCellDetections(board, domino);

      // Should still work with at least one successful parse
      expect(result.modelsCompared).toContain('gemini');
      expect(result.modelsCompared).toContain('claude');
    });
  });

  describe('Identical Inputs (No Disagreements)', () => {
    it('returns no disagreements when all models agree on everything', () => {
      const shape = '...\n...\n...';
      const regions = 'AAB\nAAB\nCCB';
      const constraints = {
        A: { type: 'sum', value: 10, op: '==' },
        B: { type: 'sum', value: 8, op: '==' },
        C: { type: 'all_different' },
      };
      const dominoes: DominoPair[] = [
        [1, 2],
        [3, 4],
        [5, 6],
      ];

      const board = [
        createBoardResponse('gemini', { rows: 3, cols: 3, shape, regions, constraints }),
        createBoardResponse('claude', { rows: 3, cols: 3, shape, regions, constraints }),
        createBoardResponse('gpt', { rows: 3, cols: 3, shape, regions, constraints }),
      ];
      const domino = [
        createDominoResponse('gemini', dominoes),
        createDominoResponse('claude', dominoes),
        createDominoResponse('gpt', dominoes),
      ];

      const result = compareCellDetections(board, domino);

      expect(result.isUnanimous).toBe(true);
      expect(result.summary.total).toBe(0);
      expect(result.summary.critical).toBe(0);
      expect(result.summary.warning).toBe(0);
      expect(result.summary.info).toBe(0);
      expect(result.allDisagreements).toHaveLength(0);
      expect(result.modelsCompared).toHaveLength(3);
    });

    it('handles two identical models', () => {
      const board = [
        createBoardResponse('model1', {}),
        createBoardResponse('model2', {}),
      ];
      const domino = [
        createDominoResponse('model1', [[1, 2]]),
        createDominoResponse('model2', [[1, 2]]),
      ];

      const result = compareCellDetections(board, domino);

      expect(result.isUnanimous).toBe(true);
      expect(result.allDisagreements).toHaveLength(0);
    });
  });

  describe('Dimension Mismatches', () => {
    it('detects row count disagreements', () => {
      const board = [
        createBoardResponse('gemini', { rows: 3, cols: 3, shape: '...\n...\n...' }),
        createBoardResponse('claude', { rows: 4, cols: 3, shape: '...\n...\n...\n...' }),
      ];
      const domino = [
        createDominoResponse('gemini', []),
        createDominoResponse('claude', []),
      ];

      const result = compareCellDetections(board, domino);

      expect(result.isUnanimous).toBe(false);
      expect(result.summary.critical).toBeGreaterThan(0);

      const dimDisagreements = result.disagreementsByType.gridDimensions as GridDimensionDisagreement[];
      expect(dimDisagreements.some(d => d.dimension === 'rows')).toBe(true);

      const rowDisagreement = dimDisagreements.find(d => d.dimension === 'rows');
      expect(rowDisagreement?.values['gemini']).toBe(3);
      expect(rowDisagreement?.values['claude']).toBe(4);
      expect(rowDisagreement?.severity).toBe('critical');
    });

    it('detects column count disagreements', () => {
      const board = [
        createBoardResponse('gemini', { rows: 3, cols: 3, shape: '...\n...\n...' }),
        createBoardResponse('claude', { rows: 3, cols: 4, shape: '....\n....\n....' }),
      ];
      const domino = [
        createDominoResponse('gemini', []),
        createDominoResponse('claude', []),
      ];

      const result = compareCellDetections(board, domino);

      const dimDisagreements = result.disagreementsByType.gridDimensions as GridDimensionDisagreement[];
      expect(dimDisagreements.some(d => d.dimension === 'cols')).toBe(true);
    });

    it('detects both row and column disagreements', () => {
      const board = [
        createBoardResponse('gemini', { rows: 3, cols: 3, shape: '...\n...\n...' }),
        createBoardResponse('claude', { rows: 4, cols: 5, shape: '.....\n.....\n.....\n.....' }),
      ];
      const domino = [
        createDominoResponse('gemini', []),
        createDominoResponse('claude', []),
      ];

      const result = compareCellDetections(board, domino);

      const dimDisagreements = result.disagreementsByType.gridDimensions;
      expect(dimDisagreements).toHaveLength(2);
    });
  });

  describe('Hole Position Disagreements', () => {
    it('detects when one model sees a hole and another does not', () => {
      const board = [
        createBoardResponse('gemini', {
          rows: 3,
          cols: 3,
          shape: '...\n...\n...',
          regions: 'AAA\nAAA\nAAA',
        }),
        createBoardResponse('claude', {
          rows: 3,
          cols: 3,
          shape: '...\n.#.\n...',
          regions: 'AAA\nA#A\nAAA',
        }),
      ];
      const domino = [
        createDominoResponse('gemini', []),
        createDominoResponse('claude', []),
      ];

      const result = compareCellDetections(board, domino);

      expect(result.isUnanimous).toBe(false);
      const holeDisagreements = result.disagreementsByType.holePositions;
      expect(holeDisagreements.length).toBeGreaterThan(0);

      const centerHole = holeDisagreements.find(
        d => d.coordinate.row === 1 && d.coordinate.col === 1
      );
      expect(centerHole).toBeDefined();
      expect(centerHole?.severity).toBe('critical');
    });

    it('detects multiple hole disagreements', () => {
      const board = [
        createBoardResponse('gemini', {
          rows: 3,
          cols: 3,
          shape: '...\n...\n...',
          regions: 'AAA\nAAA\nAAA',
        }),
        createBoardResponse('claude', {
          rows: 3,
          cols: 3,
          shape: '#..\n...\n..#',
          regions: '#AA\nAAA\nAA#',
        }),
      ];
      const domino = [
        createDominoResponse('gemini', []),
        createDominoResponse('claude', []),
      ];

      const result = compareCellDetections(board, domino);

      const holeDisagreements = result.disagreementsByType.holePositions;
      expect(holeDisagreements.length).toBe(2);
    });

    it('includes cell in disagreement map', () => {
      const board = [
        createBoardResponse('gemini', {
          rows: 2,
          cols: 2,
          shape: '..\n..',
          regions: 'AA\nAA',
        }),
        createBoardResponse('claude', {
          rows: 2,
          cols: 2,
          shape: '.#\n..',
          regions: 'A#\nAA',
        }),
      ];
      const domino = [
        createDominoResponse('gemini', []),
        createDominoResponse('claude', []),
      ];

      const result = compareCellDetections(board, domino);

      expect(result.cellDisagreementMap.has('0,1')).toBe(true);
    });
  });

  describe('Region Assignment Disagreements', () => {
    it('detects when models assign different regions to same cell', () => {
      const board = [
        createBoardResponse('gemini', {
          rows: 3,
          cols: 3,
          shape: '...\n...\n...',
          regions: 'AAB\nAAB\nCCB',
        }),
        createBoardResponse('claude', {
          rows: 3,
          cols: 3,
          shape: '...\n...\n...',
          regions: 'AAA\nAAA\nCCC',
        }),
      ];
      const domino = [
        createDominoResponse('gemini', []),
        createDominoResponse('claude', []),
      ];

      const result = compareCellDetections(board, domino);

      expect(result.isUnanimous).toBe(false);
      const regionDisagreements = result.disagreementsByType.regionAssignments;
      expect(regionDisagreements.length).toBeGreaterThan(0);

      // Check that differences at column 2 are detected (B vs A/C)
      const col2Disagreements = regionDisagreements.filter(d => d.coordinate.col === 2);
      expect(col2Disagreements.length).toBeGreaterThan(0);

      // Region disagreements should be warnings
      expect(col2Disagreements[0].severity).toBe('warning');
    });

    it('does not report disagreements for hole cells', () => {
      const board = [
        createBoardResponse('gemini', {
          rows: 2,
          cols: 2,
          shape: '.#\n..',
          regions: 'A#\nAA',
        }),
        createBoardResponse('claude', {
          rows: 2,
          cols: 2,
          shape: '.#\n..',
          regions: 'A#\nAA',
        }),
      ];
      const domino = [
        createDominoResponse('gemini', []),
        createDominoResponse('claude', []),
      ];

      const result = compareCellDetections(board, domino);

      // Should not have disagreements since the hole cell is excluded
      expect(result.disagreementsByType.regionAssignments).toHaveLength(0);
    });
  });

  describe('Constraint Disagreements', () => {
    it('detects constraint type disagreements', () => {
      const board = [
        createBoardResponse('gemini', {
          rows: 2,
          cols: 2,
          shape: '..\n..',
          regions: 'AA\nAA',
          constraints: { A: { type: 'sum', value: 10 } },
        }),
        createBoardResponse('claude', {
          rows: 2,
          cols: 2,
          shape: '..\n..',
          regions: 'AA\nAA',
          constraints: { A: { type: 'all_different' } },
        }),
      ];
      const domino = [
        createDominoResponse('gemini', []),
        createDominoResponse('claude', []),
      ];

      const result = compareCellDetections(board, domino);

      const constraintDisagreements = result.disagreementsByType.constraints;
      const typeDisagreement = constraintDisagreements.find(d => d.type === 'constraint_type');
      expect(typeDisagreement).toBeDefined();
      expect(typeDisagreement?.region).toBe('A');
      expect(typeDisagreement?.severity).toBe('warning');
    });

    it('detects constraint value disagreements', () => {
      const board = [
        createBoardResponse('gemini', {
          rows: 2,
          cols: 2,
          shape: '..\n..',
          regions: 'AA\nAA',
          constraints: { A: { type: 'sum', value: 10, op: '==' } },
        }),
        createBoardResponse('claude', {
          rows: 2,
          cols: 2,
          shape: '..\n..',
          regions: 'AA\nAA',
          constraints: { A: { type: 'sum', value: 15, op: '==' } },
        }),
      ];
      const domino = [
        createDominoResponse('gemini', []),
        createDominoResponse('claude', []),
      ];

      const result = compareCellDetections(board, domino);

      const constraintDisagreements = result.disagreementsByType.constraints;
      const valueDisagreement = constraintDisagreements.find(d => d.type === 'constraint_value');
      expect(valueDisagreement).toBeDefined();
      expect(valueDisagreement?.severity).toBe('warning');
    });

    it('detects constraint operator disagreements', () => {
      const board = [
        createBoardResponse('gemini', {
          rows: 2,
          cols: 2,
          shape: '..\n..',
          regions: 'AA\nAA',
          constraints: { A: { type: 'sum', value: 10, op: '==' } },
        }),
        createBoardResponse('claude', {
          rows: 2,
          cols: 2,
          shape: '..\n..',
          regions: 'AA\nAA',
          constraints: { A: { type: 'sum', value: 10, op: '<' } },
        }),
      ];
      const domino = [
        createDominoResponse('gemini', []),
        createDominoResponse('claude', []),
      ];

      const result = compareCellDetections(board, domino);

      const constraintDisagreements = result.disagreementsByType.constraints;
      const opDisagreement = constraintDisagreements.find(d => d.type === 'constraint_operator');
      expect(opDisagreement).toBeDefined();
      expect(opDisagreement?.severity).toBe('warning');
    });

    it('handles missing constraints in one model', () => {
      const board = [
        createBoardResponse('gemini', {
          rows: 2,
          cols: 2,
          shape: '..\n..',
          regions: 'AA\nAA',
          constraints: {
            A: { type: 'sum', value: 10 },
            B: { type: 'all_different' },
          },
        }),
        createBoardResponse('claude', {
          rows: 2,
          cols: 2,
          shape: '..\n..',
          regions: 'AA\nAA',
          constraints: { A: { type: 'sum', value: 10 } },
        }),
      ];
      const domino = [
        createDominoResponse('gemini', []),
        createDominoResponse('claude', []),
      ];

      const result = compareCellDetections(board, domino);

      // Should detect that region B has constraint in one model but not the other
      const constraintDisagreements = result.disagreementsByType.constraints;
      const regionBDisagreement = constraintDisagreements.find(d => d.region === 'B');
      expect(regionBDisagreement).toBeDefined();
    });
  });

  describe('Domino Disagreements', () => {
    it('detects domino count disagreements', () => {
      const board = [
        createBoardResponse('gemini', {}),
        createBoardResponse('claude', {}),
      ];
      const domino = [
        createDominoResponse('gemini', [[1, 2], [3, 4]]),
        createDominoResponse('claude', [[1, 2], [3, 4], [5, 6]]),
      ];

      const result = compareCellDetections(board, domino);

      const dominoDisagreements = result.disagreementsByType.dominoes;
      const countDisagreement = dominoDisagreements.find(d => d.type === 'domino_count');
      expect(countDisagreement).toBeDefined();
      expect(countDisagreement?.severity).toBe('critical');
      expect(countDisagreement?.values['gemini']).toBe(2);
      expect(countDisagreement?.values['claude']).toBe(3);
    });

    it('detects individual domino value disagreements (order-independent)', () => {
      const board = [
        createBoardResponse('gemini', {}),
        createBoardResponse('claude', {}),
      ];
      const domino = [
        createDominoResponse('gemini', [[1, 2], [3, 4]]),
        createDominoResponse('claude', [[1, 2], [5, 6]]),
      ];

      const result = compareCellDetections(board, domino);

      const dominoDisagreements = result.disagreementsByType.dominoes;
      const valueDisagreements = dominoDisagreements.filter(d => d.type === 'domino_value');
      expect(valueDisagreements.length).toBeGreaterThan(0);
      expect(valueDisagreements[0].severity).toBe('warning');
    });

    it('treats [1,2] and [2,1] as the same domino', () => {
      const board = [
        createBoardResponse('gemini', {}),
        createBoardResponse('claude', {}),
      ];
      const domino = [
        createDominoResponse('gemini', [[1, 2], [3, 4]]),
        createDominoResponse('claude', [[2, 1], [4, 3]]),  // Same dominoes, different order
      ];

      const result = compareCellDetections(board, domino);

      // Should not have value disagreements since they're the same dominoes
      const dominoDisagreements = result.disagreementsByType.dominoes;
      const valueDisagreements = dominoDisagreements.filter(d => d.type === 'domino_value');
      expect(valueDisagreements).toHaveLength(0);
    });

    it('handles empty domino lists', () => {
      const board = [
        createBoardResponse('gemini', {}),
        createBoardResponse('claude', {}),
      ];
      const domino = [
        createDominoResponse('gemini', []),
        createDominoResponse('claude', []),
      ];

      const result = compareCellDetections(board, domino);

      expect(result.disagreementsByType.dominoes).toHaveLength(0);
    });
  });

  describe('Summary Statistics', () => {
    it('correctly counts disagreements by severity', () => {
      const board = [
        createBoardResponse('gemini', {
          rows: 3,
          cols: 3,
          shape: '...\n...\n...',
          regions: 'AAB\nAAB\nCCB',
        }),
        createBoardResponse('claude', {
          rows: 4,  // row disagreement (critical)
          cols: 3,
          shape: '...\n.#.\n...\n...',  // hole disagreement (critical)
          regions: 'AAA\nA#A\nCCC\nCCC',  // region disagreement (warning)
        }),
      ];
      const domino = [
        createDominoResponse('gemini', [[1, 2]]),
        createDominoResponse('claude', [[1, 2], [3, 4]]),  // count disagreement (critical)
      ];

      const result = compareCellDetections(board, domino);

      expect(result.summary.total).toBeGreaterThan(0);
      expect(result.summary.critical).toBeGreaterThan(0);
      // We may have warnings from region differences
      expect(result.summary.critical + result.summary.warning + result.summary.info).toBe(result.summary.total);
    });

    it('sets isUnanimous to false when disagreements exist', () => {
      const board = [
        createBoardResponse('gemini', { rows: 3, cols: 3 }),
        createBoardResponse('claude', { rows: 4, cols: 3 }),
      ];
      const domino = [
        createDominoResponse('gemini', []),
        createDominoResponse('claude', []),
      ];

      const result = compareCellDetections(board, domino);

      expect(result.isUnanimous).toBe(false);
    });
  });

  describe('Model Results', () => {
    it('includes normalized results for each model', () => {
      const board = [
        createBoardResponse('gemini', { rows: 3, cols: 3 }),
        createBoardResponse('claude', { rows: 3, cols: 3 }),
      ];
      const domino = [
        createDominoResponse('gemini', [[1, 2]]),
        createDominoResponse('claude', [[1, 2]]),
      ];

      const result = compareCellDetections(board, domino);

      expect(result.modelResults).toHaveLength(2);
      expect(result.modelResults[0].model).toBe('gemini');
      expect(result.modelResults[1].model).toBe('claude');
    });

    it('includes dimensions in normalized results', () => {
      const board = [
        createBoardResponse('gemini', { rows: 4, cols: 5 }),
        createBoardResponse('claude', { rows: 4, cols: 5 }),
      ];
      const domino = [
        createDominoResponse('gemini', []),
        createDominoResponse('claude', []),
      ];

      const result = compareCellDetections(board, domino);

      expect(result.modelResults[0].dimensions).toEqual({ rows: 4, cols: 5 });
    });

    it('includes dominoes from domino responses', () => {
      const board = [
        createBoardResponse('gemini', {}),
        createBoardResponse('claude', {}),
      ];
      const domino = [
        createDominoResponse('gemini', [[1, 2], [3, 4]]),
        createDominoResponse('claude', [[1, 2], [3, 4]]),
      ];

      const result = compareCellDetections(board, domino);

      expect(result.modelResults[0].dominoes).toEqual([[1, 2], [3, 4]]);
    });
  });

  describe('Comparison Options', () => {
    it('respects referenceModel option', () => {
      const board = [
        createBoardResponse('gemini', {}),
        createBoardResponse('claude', {}),
      ];
      const domino = [
        createDominoResponse('gemini', []),
        createDominoResponse('claude', []),
      ];

      const result = compareCellDetections(board, domino, { referenceModel: 'claude' });

      expect(result.selectedModel).toBe('claude');
    });

    it('can filter out info-level disagreements', () => {
      // Note: Current implementation doesn't generate info-level disagreements
      // This test verifies the option exists and is processed
      const board = [
        createBoardResponse('gemini', {}),
        createBoardResponse('claude', {}),
      ];
      const domino = [
        createDominoResponse('gemini', []),
        createDominoResponse('claude', []),
      ];

      const result = compareCellDetections(board, domino, { includeInfoLevel: false });

      // Should not include any info-level disagreements
      expect(result.allDisagreements.every(d => d.severity !== 'info')).toBe(true);
    });
  });

  describe('Three or More Models', () => {
    it('handles three models with varying agreements', () => {
      const board = [
        createBoardResponse('gemini', { rows: 3, cols: 3 }),
        createBoardResponse('claude', { rows: 3, cols: 3 }),
        createBoardResponse('gpt', { rows: 4, cols: 3 }),  // GPT disagrees on rows
      ];
      const domino = [
        createDominoResponse('gemini', [[1, 2]]),
        createDominoResponse('claude', [[1, 2]]),
        createDominoResponse('gpt', [[1, 2]]),
      ];

      const result = compareCellDetections(board, domino);

      expect(result.modelsCompared).toHaveLength(3);
      expect(result.isUnanimous).toBe(false);

      // Should detect row disagreement
      const dimDisagreements = result.disagreementsByType.gridDimensions;
      expect(dimDisagreements.length).toBeGreaterThan(0);
    });

    it('records values from all models in disagreement', () => {
      const board = [
        createBoardResponse('gemini', { rows: 3, cols: 3 }),
        createBoardResponse('claude', { rows: 4, cols: 3 }),
        createBoardResponse('gpt', { rows: 5, cols: 3 }),
      ];
      const domino = [
        createDominoResponse('gemini', []),
        createDominoResponse('claude', []),
        createDominoResponse('gpt', []),
      ];

      const result = compareCellDetections(board, domino);

      const dimDisagreements = result.disagreementsByType.gridDimensions as GridDimensionDisagreement[];
      const rowDisagreement = dimDisagreements.find(d => d.dimension === 'rows');
      expect(rowDisagreement?.values['gemini']).toBe(3);
      expect(rowDisagreement?.values['claude']).toBe(4);
      expect(rowDisagreement?.values['gpt']).toBe(5);
    });
  });

  describe('Complex Scenarios', () => {
    it('handles multiple types of disagreements simultaneously', () => {
      const board = [
        createBoardResponse('gemini', {
          rows: 3,
          cols: 3,
          shape: '...\n...\n...',
          regions: 'AAB\nAAB\nCCB',
          constraints: { A: { type: 'sum', value: 10 }, B: { type: 'sum', value: 5 } },
        }),
        createBoardResponse('claude', {
          rows: 3,
          cols: 4,  // column disagreement
          shape: '....\n.#..\n....',  // hole disagreement
          regions: 'AAAA\nA#BB\nCCCC',  // region disagreement
          constraints: { A: { type: 'all_different' } },  // constraint disagreement
        }),
      ];
      const domino = [
        createDominoResponse('gemini', [[1, 2], [3, 4]]),
        createDominoResponse('claude', [[5, 6]]),  // domino disagreements
      ];

      const result = compareCellDetections(board, domino);

      expect(result.isUnanimous).toBe(false);
      expect(result.disagreementsByType.gridDimensions.length).toBeGreaterThan(0);
      expect(result.disagreementsByType.holePositions.length).toBeGreaterThan(0);
      expect(result.disagreementsByType.constraints.length).toBeGreaterThan(0);
      expect(result.disagreementsByType.dominoes.length).toBeGreaterThan(0);
    });

    it('handles realistic extraction results', () => {
      // Simulate a typical scenario where two models mostly agree but have minor differences
      const board = [
        createBoardResponse('gemini', {
          rows: 6,
          cols: 7,
          shape: '.......\n.......\n.......\n.......\n.......\n.......',
          regions: 'AAAABBB\nAAAABBB\nCCCDDDD\nCCCDDDD\nEEEFFFF\nEEEFFFF',
          constraints: {
            A: { type: 'sum', value: 10, op: '==' },
            B: { type: 'sum', value: 12, op: '==' },
            C: { type: 'all_different' },
            D: { type: 'all_equal' },
            E: { type: 'sum', value: 8, op: '==' },
            F: { type: 'none' },
          },
          confidence: { grid: 0.95, regions: 0.90, constraints: 0.85 },
        }),
        createBoardResponse('claude', {
          rows: 6,
          cols: 7,
          shape: '.......\n.......\n.......\n.......\n.......\n.......',
          regions: 'AAAABBB\nAAAABBB\nCCCDDDD\nCCCDDDD\nEEEFFFF\nEEEFFFF',
          constraints: {
            A: { type: 'sum', value: 10, op: '==' },
            B: { type: 'sum', value: 12, op: '==' },
            C: { type: 'all_different' },
            D: { type: 'all_equal' },
            E: { type: 'sum', value: 9, op: '==' },  // Minor disagreement on E's value
            F: { type: 'none' },
          },
          confidence: { grid: 0.92, regions: 0.88, constraints: 0.87 },
        }),
      ];
      const domino = [
        createDominoResponse('gemini', [[0, 1], [2, 3], [4, 5], [6, 0], [1, 2], [3, 4], [5, 6]], true, 0.93),
        createDominoResponse('claude', [[0, 1], [2, 3], [4, 5], [6, 0], [1, 2], [3, 4], [5, 6]], true, 0.91),
      ];

      const result = compareCellDetections(board, domino);

      // Should have one constraint value disagreement for region E
      expect(result.disagreementsByType.constraints.length).toBeGreaterThan(0);
      const eConstraint = result.disagreementsByType.constraints.find(d => d.region === 'E');
      expect(eConstraint).toBeDefined();
      expect(eConstraint?.type).toBe('constraint_value');
    });
  });
});
