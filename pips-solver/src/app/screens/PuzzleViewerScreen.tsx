/**
 * Puzzle Viewer Screen
 *
 * Displays the puzzle interface with hint functionality.
 * Integrates the HintModal for graduated hint assistance.
 */

import React, { useCallback, useState } from 'react';
import { SafeAreaView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { colors, radii, spacing } from '../../theme';
import { generateHint, HintServiceError, PuzzleSpec } from '../../services/hintService';
import { HintContent, HintLevel, HintModal } from '../components/HintModal';

// ════════════════════════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════════════════════════

/** Props for PuzzleViewerScreen */
export interface PuzzleViewerScreenProps {
  /** The puzzle specification to display and solve */
  puzzleSpec?: PuzzleSpec;
  /** Current user placements on the puzzle */
  currentState?: Record<string, unknown>;
}

// ════════════════════════════════════════════════════════════════════════════
// Default Puzzle (for demonstration)
// ════════════════════════════════════════════════════════════════════════════

const DEFAULT_PUZZLE_SPEC: PuzzleSpec = {
  pips: {
    pip_min: 0,
    pip_max: 6,
  },
  dominoes: {
    tiles: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
    ],
  },
  board: {
    shape: ['####', '####'],
    regions: ['AABB', 'CCDD'],
  },
  region_constraints: {
    A: { type: 'sum', op: '==', value: 5 },
    B: { type: 'sum', op: '==', value: 7 },
    C: { type: 'all_equal' },
    D: { type: 'sum', op: '==', value: 4 },
  },
};

// ════════════════════════════════════════════════════════════════════════════
// Component
// ════════════════════════════════════════════════════════════════════════════

export function PuzzleViewerScreen({
  puzzleSpec = DEFAULT_PUZZLE_SPEC,
  currentState,
}: PuzzleViewerScreenProps) {
  // State for hint modal
  const [hintModalVisible, setHintModalVisible] = useState(false);
  const [revealedLevels, setRevealedLevels] = useState<HintLevel[]>([]);

  /**
   * Open the hint modal
   */
  const handleHintPress = useCallback(() => {
    setHintModalVisible(true);
  }, []);

  /**
   * Close the hint modal
   */
  const handleHintModalClose = useCallback(() => {
    setHintModalVisible(false);
  }, []);

  /**
   * Request a hint at the specified level from the API
   */
  const handleRequestHint = useCallback(
    async (level: HintLevel): Promise<HintContent> => {
      try {
        // Call the hint service API
        const hint = await generateHint(puzzleSpec, level, currentState);

        // Track revealed levels for UI indication
        if (!revealedLevels.includes(level)) {
          setRevealedLevels((prev) => [...prev, level]);
        }

        return hint;
      } catch (error) {
        // Re-throw with user-friendly message
        if (error instanceof HintServiceError) {
          throw error;
        }
        throw new Error('Failed to get hint. Please try again.');
      }
    },
    [puzzleSpec, currentState, revealedLevels]
  );

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Puzzle Solver</Text>
      </View>

      {/* Puzzle Area (Placeholder) */}
      <View style={styles.puzzleContainer}>
        <View style={styles.puzzlePlaceholder}>
          <Text style={styles.placeholderIcon}>🧩</Text>
          <Text style={styles.placeholderText}>Puzzle Grid</Text>
          <Text style={styles.placeholderSubtext}>
            {Object.keys(puzzleSpec.region_constraints).length} regions with constraints
          </Text>
        </View>
      </View>

      {/* Action Bar */}
      <View style={styles.actionBar}>
        <TouchableOpacity
          style={styles.hintButton}
          onPress={handleHintPress}
          activeOpacity={0.7}
        >
          <Text style={styles.hintButtonIcon}>💡</Text>
          <Text style={styles.hintButtonText}>Hint</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.actionButton} activeOpacity={0.7}>
          <Text style={styles.actionButtonText}>Check</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.actionButton} activeOpacity={0.7}>
          <Text style={styles.actionButtonText}>Reset</Text>
        </TouchableOpacity>
      </View>

      {/* Hint Modal */}
      <HintModal
        visible={hintModalVisible}
        onClose={handleHintModalClose}
        onRequestHint={handleRequestHint}
        revealedLevels={revealedLevels}
      />
    </SafeAreaView>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Styles
// ════════════════════════════════════════════════════════════════════════════

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.surface.charcoal,
  },
  header: {
    paddingHorizontal: spacing[5],
    paddingVertical: spacing[4],
    borderBottomWidth: 1,
    borderBottomColor: colors.surface.ash,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.text.primary,
  },
  puzzleContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing[5],
  },
  puzzlePlaceholder: {
    width: '100%',
    aspectRatio: 1,
    maxWidth: 350,
    backgroundColor: colors.surface.graphite,
    borderRadius: radii.xl,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: colors.surface.ash,
  },
  placeholderIcon: {
    fontSize: 60,
    marginBottom: spacing[3],
  },
  placeholderText: {
    fontSize: 20,
    fontWeight: '600',
    color: colors.text.primary,
    marginBottom: spacing[1],
  },
  placeholderSubtext: {
    fontSize: 14,
    color: colors.text.secondary,
  },
  actionBar: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: spacing[5],
    paddingVertical: spacing[4],
    gap: spacing[3],
    borderTopWidth: 1,
    borderTopColor: colors.surface.ash,
  },
  hintButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.accent.brass,
    paddingHorizontal: spacing[5],
    paddingVertical: spacing[3],
    borderRadius: radii.lg,
    gap: spacing[2],
  },
  hintButtonIcon: {
    fontSize: 18,
  },
  hintButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.inverse,
  },
  actionButton: {
    backgroundColor: colors.surface.slate,
    paddingHorizontal: spacing[4],
    paddingVertical: spacing[3],
    borderRadius: radii.md,
  },
  actionButtonText: {
    fontSize: 14,
    fontWeight: '500',
    color: colors.text.primary,
  },
});

export default PuzzleViewerScreen;
