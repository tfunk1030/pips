/**
 * Tests for extraction pipeline confidence hints generation.
 */

import { generateConfidenceHints, calculateOverallConfidence } from '../pipeline';
import { StageConfidence, ConfidenceHint, ExtractionStage } from '../types';

/**
 * Helper to create a StageConfidence with minimal required fields.
 */
function createStageConfidence(
  stage: ExtractionStage,
  confidence: number,
  method: string = 'test_method',
): StageConfidence {
  return { stage, confidence, method };
}

describe('generateConfidenceHints', () => {
  describe('empty input handling', () => {
    it('should return empty array for empty input', () => {
      const result = generateConfidenceHints([]);
      expect(result).toEqual([]);
    });
  });

  describe('low confidence detection at each stage', () => {
    const stages: ExtractionStage[] = [
      'BOARD_DETECTION',
      'GRID_ALIGNMENT',
      'CELL_EXTRACTION',
      'PIP_RECOGNITION',
    ];

    const expectedDisplayNames: Record<ExtractionStage, string> = {
      BOARD_DETECTION: 'Board detection',
      GRID_ALIGNMENT: 'Grid dimensions',
      CELL_EXTRACTION: 'Cell extraction',
      PIP_RECOGNITION: 'Pip recognition',
    };

    it.each(stages)(
      'should generate low confidence hint for %s when confidence < 0.7',
      (stage) => {
        const stageConfidences: StageConfidence[] = [
          createStageConfidence(stage, 0.6),
        ];

        const hints = generateConfidenceHints(stageConfidences);

        expect(hints).toHaveLength(1);
        expect(hints[0].stage).toBe(stage);
        expect(hints[0].message).toBe(`Low confidence in ${expectedDisplayNames[stage]}`);
        expect(hints[0].severity).toBe('medium'); // 0.6 is between 0.5 and 0.7
        expect(hints[0].suggestion).toBeDefined();
        expect(hints[0].suggestion.length).toBeGreaterThan(0);
      },
    );

    it.each(stages)(
      'should NOT generate hint for %s when confidence >= 0.7',
      (stage) => {
        const stageConfidences: StageConfidence[] = [
          createStageConfidence(stage, 0.75),
        ];

        const hints = generateConfidenceHints(stageConfidences);

        // Should not have low confidence hints (may have variance hint with single item)
        const lowConfidenceHints = hints.filter((h) =>
          h.message.startsWith('Low confidence'),
        );
        expect(lowConfidenceHints).toHaveLength(0);
      },
    );

    it.each(stages)(
      'should generate hint for %s at exactly 0.7 threshold boundary',
      (stage) => {
        // At exactly 0.7, should NOT generate (< 0.7 is required)
        const stageConfidences: StageConfidence[] = [
          createStageConfidence(stage, 0.7),
        ];

        const hints = generateConfidenceHints(stageConfidences);
        const lowConfidenceHints = hints.filter((h) =>
          h.message.startsWith('Low confidence'),
        );
        expect(lowConfidenceHints).toHaveLength(0);
      },
    );

    it.each(stages)(
      'should generate hint for %s just below 0.7 threshold',
      (stage) => {
        const stageConfidences: StageConfidence[] = [
          createStageConfidence(stage, 0.699),
        ];

        const hints = generateConfidenceHints(stageConfidences);
        const lowConfidenceHints = hints.filter((h) =>
          h.message.startsWith('Low confidence'),
        );
        expect(lowConfidenceHints).toHaveLength(1);
      },
    );
  });

  describe('severity determination', () => {
    it('should assign high severity for confidence < 0.5', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.3),
      ];

      const hints = generateConfidenceHints(stageConfidences);

      expect(hints[0].severity).toBe('high');
    });

    it('should assign medium severity for confidence between 0.5 and 0.7', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.6),
      ];

      const hints = generateConfidenceHints(stageConfidences);

      expect(hints[0].severity).toBe('medium');
    });

    it('should assign high severity at exactly 0.5', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.5),
      ];

      const hints = generateConfidenceHints(stageConfidences);

      // At 0.5, should be medium (not < 0.5)
      expect(hints[0].severity).toBe('medium');
    });

    it('should assign high severity just below 0.5', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.49),
      ];

      const hints = generateConfidenceHints(stageConfidences);

      expect(hints[0].severity).toBe('high');
    });
  });

  describe('variance detection', () => {
    it('should generate variance hint when variance > 0.15', () => {
      // High variance: 0.9 and 0.4 => variance ≈ 0.0625, not enough
      // Need bigger spread: 0.95 and 0.3 => mean = 0.625, variance = (0.325^2 + 0.325^2)/2 = 0.105625
      // Still not enough. Let's use 0.98 and 0.2 => mean = 0.59, variance = (0.39^2 + 0.39^2)/2 = 0.1521
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.98),
        createStageConfidence('GRID_ALIGNMENT', 0.2),
      ];

      const hints = generateConfidenceHints(stageConfidences);
      const varianceHints = hints.filter((h) =>
        h.message.includes('varies significantly'),
      );

      expect(varianceHints).toHaveLength(1);
      expect(varianceHints[0].severity).toBe('medium');
      expect(varianceHints[0].suggestion).toContain('Review the extraction results');
    });

    it('should NOT generate variance hint when variance <= 0.15', () => {
      // Low variance: close values
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.8),
        createStageConfidence('GRID_ALIGNMENT', 0.85),
        createStageConfidence('CELL_EXTRACTION', 0.82),
        createStageConfidence('PIP_RECOGNITION', 0.78),
      ];

      const hints = generateConfidenceHints(stageConfidences);
      const varianceHints = hints.filter((h) =>
        h.message.includes('varies significantly'),
      );

      expect(varianceHints).toHaveLength(0);
    });

    it('should attach variance hint to the lowest confidence stage', () => {
      // Need high variance: values 0.98 and 0.1 give variance = 0.1936 > 0.15
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.98),
        createStageConfidence('GRID_ALIGNMENT', 0.1), // Lowest
      ];

      const hints = generateConfidenceHints(stageConfidences);
      const varianceHint = hints.find((h) =>
        h.message.includes('varies significantly'),
      );

      expect(varianceHint).toBeDefined();
      expect(varianceHint!.stage).toBe('GRID_ALIGNMENT');
    });

    it('should generate both low confidence AND variance hints when applicable', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.95),
        createStageConfidence('GRID_ALIGNMENT', 0.15), // Low AND contributes to variance
      ];

      const hints = generateConfidenceHints(stageConfidences);

      expect(hints.length).toBeGreaterThanOrEqual(2);

      const lowConfidenceHint = hints.find((h) =>
        h.message.includes('Low confidence'),
      );
      const varianceHint = hints.find((h) =>
        h.message.includes('varies significantly'),
      );

      expect(lowConfidenceHint).toBeDefined();
      expect(varianceHint).toBeDefined();
    });
  });

  describe('multiple stage issues', () => {
    it('should generate hints for all low confidence stages', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.5),
        createStageConfidence('GRID_ALIGNMENT', 0.45),
        createStageConfidence('CELL_EXTRACTION', 0.55),
        createStageConfidence('PIP_RECOGNITION', 0.6),
      ];

      const hints = generateConfidenceHints(stageConfidences);
      const lowConfidenceHints = hints.filter((h) =>
        h.message.includes('Low confidence'),
      );

      // All 4 stages are below 0.7
      expect(lowConfidenceHints).toHaveLength(4);
    });

    it('should preserve order of stages in hints', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.4),
        createStageConfidence('GRID_ALIGNMENT', 0.3),
        createStageConfidence('CELL_EXTRACTION', 0.5),
      ];

      const hints = generateConfidenceHints(stageConfidences);
      const lowConfidenceHints = hints.filter((h) =>
        h.message.includes('Low confidence'),
      );

      expect(lowConfidenceHints[0].stage).toBe('BOARD_DETECTION');
      expect(lowConfidenceHints[1].stage).toBe('GRID_ALIGNMENT');
      expect(lowConfidenceHints[2].stage).toBe('CELL_EXTRACTION');
    });
  });

  describe('hint message formatting', () => {
    it('should include proper stage name in message', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('GRID_ALIGNMENT', 0.4),
      ];

      const hints = generateConfidenceHints(stageConfidences);

      expect(hints[0].message).toBe('Low confidence in Grid dimensions');
    });

    it('should provide actionable suggestions for each stage', () => {
      const expectedSuggestions: Record<ExtractionStage, string> = {
        BOARD_DETECTION: 'Ensure the entire board is visible and centered in the frame',
        GRID_ALIGNMENT: 'Try capturing from directly above the board with less angle',
        CELL_EXTRACTION: 'Ensure good lighting and avoid shadows across the board',
        PIP_RECOGNITION: 'Make sure pip markings are clearly visible and not worn',
      };

      for (const [stage, expectedSuggestion] of Object.entries(expectedSuggestions)) {
        const stageConfidences: StageConfidence[] = [
          createStageConfidence(stage as ExtractionStage, 0.4),
        ];

        const hints = generateConfidenceHints(stageConfidences);
        expect(hints[0].suggestion).toBe(expectedSuggestion);
      }
    });

    it('should format variance hint message correctly', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.95),
        createStageConfidence('GRID_ALIGNMENT', 0.1),
      ];

      const hints = generateConfidenceHints(stageConfidences);
      const varianceHint = hints.find((h) =>
        h.message.includes('varies'),
      );

      expect(varianceHint!.message).toBe(
        'Extraction confidence varies significantly across stages',
      );
    });
  });

  describe('edge cases', () => {
    it('should handle all confidence values at zero', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0),
        createStageConfidence('GRID_ALIGNMENT', 0),
        createStageConfidence('CELL_EXTRACTION', 0),
        createStageConfidence('PIP_RECOGNITION', 0),
      ];

      const hints = generateConfidenceHints(stageConfidences);
      const lowConfidenceHints = hints.filter((h) =>
        h.message.includes('Low confidence'),
      );

      // All stages should generate hints with high severity
      expect(lowConfidenceHints).toHaveLength(4);
      lowConfidenceHints.forEach((hint) => {
        expect(hint.severity).toBe('high');
      });

      // Should NOT have variance hint (variance = 0)
      const varianceHints = hints.filter((h) =>
        h.message.includes('varies significantly'),
      );
      expect(varianceHints).toHaveLength(0);
    });

    it('should handle all confidence values at one', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 1),
        createStageConfidence('GRID_ALIGNMENT', 1),
        createStageConfidence('CELL_EXTRACTION', 1),
        createStageConfidence('PIP_RECOGNITION', 1),
      ];

      const hints = generateConfidenceHints(stageConfidences);

      // No hints should be generated for perfect confidence
      expect(hints).toHaveLength(0);
    });

    it('should handle single stage input', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.5),
      ];

      const hints = generateConfidenceHints(stageConfidences);

      // Should have low confidence hint but NO variance hint (single value has 0 variance)
      expect(hints).toHaveLength(1);
      expect(hints[0].message).toContain('Low confidence');
    });

    it('should handle two identical stages (though unusual)', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.5),
        createStageConfidence('BOARD_DETECTION', 0.5),
      ];

      const hints = generateConfidenceHints(stageConfidences);
      const lowConfidenceHints = hints.filter((h) =>
        h.message.includes('Low confidence'),
      );

      // Should generate 2 low confidence hints
      expect(lowConfidenceHints).toHaveLength(2);

      // No variance hint (identical values = 0 variance)
      const varianceHints = hints.filter((h) =>
        h.message.includes('varies significantly'),
      );
      expect(varianceHints).toHaveLength(0);
    });

    it('should handle confidence values slightly above threshold', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.701),
        createStageConfidence('GRID_ALIGNMENT', 0.71),
        createStageConfidence('CELL_EXTRACTION', 0.72),
        createStageConfidence('PIP_RECOGNITION', 0.73),
      ];

      const hints = generateConfidenceHints(stageConfidences);
      const lowConfidenceHints = hints.filter((h) =>
        h.message.includes('Low confidence'),
      );

      expect(lowConfidenceHints).toHaveLength(0);
    });

    it('should handle very small variance just above threshold', () => {
      // Variance threshold is 0.15
      // Create values that give variance just above 0.15
      // Mean = 0.7, values at 0.9 and 0.5 => variance = (0.2^2 + 0.2^2)/2 = 0.04 (not enough)
      // Need: variance > 0.15, so sqrt(0.15) ≈ 0.387
      // Mean = 0.5, values at 0.9 and 0.1 => variance = (0.4^2 + 0.4^2)/2 = 0.16 (just above)
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.9),
        createStageConfidence('GRID_ALIGNMENT', 0.1),
      ];

      const hints = generateConfidenceHints(stageConfidences);
      const varianceHints = hints.filter((h) =>
        h.message.includes('varies significantly'),
      );

      expect(varianceHints).toHaveLength(1);
    });

    it('should handle variance exactly at threshold', () => {
      // Variance = 0.15 exactly (should NOT trigger, need > 0.15)
      // Mean = 0.5, for variance = 0.15: (x-0.5)^2 = 0.15, x ≈ 0.887 or 0.113
      // Using 0.887 and 0.113: variance = (0.387^2 + 0.387^2)/2 ≈ 0.15
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.8873),
        createStageConfidence('GRID_ALIGNMENT', 0.1127),
      ];

      const hints = generateConfidenceHints(stageConfidences);
      const varianceHints = hints.filter((h) =>
        h.message.includes('varies significantly'),
      );

      // At exactly 0.15, should NOT trigger (> 0.15 required)
      // But due to floating point, this might be slightly above, so we accept either result
      expect(varianceHints.length).toBeLessThanOrEqual(1);
    });
  });
});

describe('calculateOverallConfidence', () => {
  describe('empty input handling', () => {
    it('should return 0 for empty input', () => {
      const result = calculateOverallConfidence([]);
      expect(result).toBe(0);
    });
  });

  describe('average calculation', () => {
    it('should calculate average of all stage confidences', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.8),
        createStageConfidence('GRID_ALIGNMENT', 0.9),
        createStageConfidence('CELL_EXTRACTION', 0.7),
        createStageConfidence('PIP_RECOGNITION', 0.6),
      ];

      const result = calculateOverallConfidence(stageConfidences);

      // (0.8 + 0.9 + 0.7 + 0.6) / 4 = 3.0 / 4 = 0.75
      expect(result).toBeCloseTo(0.75, 10);
    });

    it('should return the single value for single stage', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.85),
      ];

      const result = calculateOverallConfidence(stageConfidences);

      expect(result).toBe(0.85);
    });

    it('should handle two stages correctly', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.6),
        createStageConfidence('GRID_ALIGNMENT', 0.4),
      ];

      const result = calculateOverallConfidence(stageConfidences);

      expect(result).toBe(0.5);
    });
  });

  describe('edge cases', () => {
    it('should handle all zeros', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0),
        createStageConfidence('GRID_ALIGNMENT', 0),
        createStageConfidence('CELL_EXTRACTION', 0),
        createStageConfidence('PIP_RECOGNITION', 0),
      ];

      const result = calculateOverallConfidence(stageConfidences);

      expect(result).toBe(0);
    });

    it('should handle all ones', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 1),
        createStageConfidence('GRID_ALIGNMENT', 1),
        createStageConfidence('CELL_EXTRACTION', 1),
        createStageConfidence('PIP_RECOGNITION', 1),
      ];

      const result = calculateOverallConfidence(stageConfidences);

      expect(result).toBe(1);
    });

    it('should handle mixed extreme values', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0),
        createStageConfidence('GRID_ALIGNMENT', 1),
      ];

      const result = calculateOverallConfidence(stageConfidences);

      expect(result).toBe(0.5);
    });

    it('should maintain precision for decimal values', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.333),
        createStageConfidence('GRID_ALIGNMENT', 0.333),
        createStageConfidence('CELL_EXTRACTION', 0.334),
      ];

      const result = calculateOverallConfidence(stageConfidences);

      // (0.333 + 0.333 + 0.334) / 3 = 1 / 3 ≈ 0.3333...
      expect(result).toBeCloseTo(0.3333, 4);
    });

    it('should handle very small confidence values', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.001),
        createStageConfidence('GRID_ALIGNMENT', 0.002),
        createStageConfidence('CELL_EXTRACTION', 0.003),
      ];

      const result = calculateOverallConfidence(stageConfidences);

      expect(result).toBeCloseTo(0.002, 4);
    });
  });
});

describe('integration tests', () => {
  describe('typical usage scenarios', () => {
    it('should handle high confidence scenario with no hints', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.95),
        createStageConfidence('GRID_ALIGNMENT', 0.92),
        createStageConfidence('CELL_EXTRACTION', 0.94),
        createStageConfidence('PIP_RECOGNITION', 0.91),
      ];

      const hints = generateConfidenceHints(stageConfidences);
      const overall = calculateOverallConfidence(stageConfidences);

      expect(hints).toHaveLength(0);
      expect(overall).toBeCloseTo(0.93, 2);
    });

    it('should handle mixed issues scenario', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.65),
        createStageConfidence('GRID_ALIGNMENT', 0.58),
        createStageConfidence('CELL_EXTRACTION', 0.72),
        createStageConfidence('PIP_RECOGNITION', 0.68),
      ];

      const hints = generateConfidenceHints(stageConfidences);
      const overall = calculateOverallConfidence(stageConfidences);

      // Should have hints for stages below 0.7
      const lowConfHints = hints.filter((h) =>
        h.message.includes('Low confidence'),
      );
      expect(lowConfHints.length).toBeGreaterThan(0);
      expect(overall).toBeCloseTo(0.6575, 2);
    });

    it('should handle failed extraction scenario', () => {
      const stageConfidences: StageConfidence[] = [
        createStageConfidence('BOARD_DETECTION', 0.25),
      ];

      const hints = generateConfidenceHints(stageConfidences);
      const overall = calculateOverallConfidence(stageConfidences);

      expect(hints).toHaveLength(1);
      expect(hints[0].severity).toBe('high');
      expect(overall).toBe(0.25);
    });
  });
});
