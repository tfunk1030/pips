/**
 * Hint Modal Component
 *
 * Displays a graduated 4-level hint system for puzzle solving assistance.
 * Users can request progressively more specific hints:
 * - Level 1: Strategic guidance
 * - Level 2: Focused direction (region/constraint)
 * - Level 3: Specific cell placement
 * - Level 4: Partial solution (3-5 cells)
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import React, { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Modal,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { colors, radii, spacing } from '../../theme';

// ════════════════════════════════════════════════════════════════════════════
// Storage Keys
// ════════════════════════════════════════════════════════════════════════════

/** Storage key for persisting the last used hint level */
const HINT_LEVEL_STORAGE_KEY = '@pips_solver/last_hint_level';

// ════════════════════════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════════════════════════

/** Hint levels from 1 (general) to 4 (specific) */
export type HintLevel = 1 | 2 | 3 | 4;

/** Types of hints returned by the API */
export type HintType = 'strategy' | 'direction' | 'cell' | 'partial_solution';

/** Individual cell placement info for Level 3/4 hints */
export interface CellPlacement {
  row: number;
  col: number;
  value: number;
  region?: string;
}

/** Hint content returned from the API */
export interface HintContent {
  level: HintLevel;
  type: HintType;
  content: string;
  cells?: CellPlacement[];
}

/** Props for the HintModal component */
export interface HintModalProps {
  /** Whether the modal is visible */
  visible: boolean;
  /** Callback when modal is closed */
  onClose: () => void;
  /** Callback when user requests a hint at a specific level */
  onRequestHint: (level: HintLevel) => Promise<HintContent>;
  /** Optional: previously revealed hint levels to show progression */
  revealedLevels?: HintLevel[];
}

// ════════════════════════════════════════════════════════════════════════════
// Constants
// ════════════════════════════════════════════════════════════════════════════

const HINT_LEVEL_INFO: Record<HintLevel, { title: string; description: string; icon: string }> = {
  1: {
    title: 'Strategy Hint',
    description: 'General solving strategy without revealing specifics',
    icon: '💡',
  },
  2: {
    title: 'Direction Hint',
    description: 'Identifies a specific region or constraint to examine',
    icon: '🔍',
  },
  3: {
    title: 'Cell Hint',
    description: 'Reveals a single specific cell placement',
    icon: '📍',
  },
  4: {
    title: 'Partial Solution',
    description: 'Shows multiple cell placements to help you progress',
    icon: '🗺️',
  },
};

// ════════════════════════════════════════════════════════════════════════════
// Component
// ════════════════════════════════════════════════════════════════════════════

export function HintModal({
  visible,
  onClose,
  onRequestHint,
  revealedLevels = [],
}: HintModalProps) {
  const [currentHint, setCurrentHint] = useState<HintContent | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedLevel, setSelectedLevel] = useState<HintLevel | null>(null);

  /**
   * Load the last used hint level from AsyncStorage when modal becomes visible
   */
  useEffect(() => {
    if (!visible) return;

    const loadLastHintLevel = async () => {
      try {
        const storedLevel = await AsyncStorage.getItem(HINT_LEVEL_STORAGE_KEY);
        if (storedLevel) {
          const parsedLevel = parseInt(storedLevel, 10);
          if (parsedLevel >= 1 && parsedLevel <= 4) {
            setSelectedLevel(parsedLevel as HintLevel);
          }
        }
      } catch {
        // Silently ignore storage read errors - use default behavior
      }
    };

    loadLastHintLevel();
  }, [visible]);

  /**
   * Persist the hint level to AsyncStorage
   */
  const persistHintLevel = useCallback(async (level: HintLevel) => {
    try {
      await AsyncStorage.setItem(HINT_LEVEL_STORAGE_KEY, level.toString());
    } catch {
      // Silently ignore storage write errors
    }
  }, []);

  /**
   * Handle hint level button press
   */
  const handleLevelPress = useCallback(
    async (level: HintLevel) => {
      setSelectedLevel(level);
      setLoading(true);
      setError(null);

      // Persist the selected level asynchronously (don't await)
      persistHintLevel(level);

      try {
        const hint = await onRequestHint(level);
        setCurrentHint(hint);
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : 'Failed to get hint. Please try again.';
        setError(errorMessage);
        setCurrentHint(null);
      } finally {
        setLoading(false);
      }
    },
    [onRequestHint, persistHintLevel]
  );

  /**
   * Handle modal close - reset state
   */
  const handleClose = useCallback(() => {
    setCurrentHint(null);
    setError(null);
    setSelectedLevel(null);
    onClose();
  }, [onClose]);

  /**
   * Render a hint level button
   */
  const renderLevelButton = useCallback(
    (level: HintLevel) => {
      const info = HINT_LEVEL_INFO[level];
      const isSelected = selectedLevel === level;
      const isRevealed = revealedLevels.includes(level);
      const isLoading = loading && selectedLevel === level;

      return (
        <TouchableOpacity
          key={level}
          style={[
            styles.levelButton,
            isSelected && styles.levelButtonSelected,
            isRevealed && styles.levelButtonRevealed,
          ]}
          onPress={() => handleLevelPress(level)}
          disabled={loading}
          activeOpacity={0.7}
        >
          <View style={styles.levelButtonContent}>
            <View style={styles.levelHeader}>
              <Text style={styles.levelIcon}>{info.icon}</Text>
              <View style={styles.levelTitleContainer}>
                <Text style={[styles.levelTitle, isSelected && styles.levelTitleSelected]}>
                  Level {level}: {info.title}
                </Text>
                {isRevealed && <Text style={styles.revealedBadge}>Used</Text>}
              </View>
            </View>
            <Text style={[styles.levelDescription, isSelected && styles.levelDescriptionSelected]}>
              {info.description}
            </Text>
            {isLoading && (
              <ActivityIndicator
                size="small"
                color={colors.accent.brass}
                style={styles.levelLoading}
              />
            )}
          </View>
        </TouchableOpacity>
      );
    },
    [selectedLevel, revealedLevels, loading, handleLevelPress]
  );

  /**
   * Render the hint content display
   */
  const renderHintContent = useCallback(() => {
    if (loading) {
      return (
        <View style={styles.hintContentContainer}>
          <ActivityIndicator size="large" color={colors.accent.brass} />
          <Text style={styles.loadingText}>Analyzing puzzle...</Text>
        </View>
      );
    }

    if (error) {
      return (
        <View style={styles.hintContentContainer}>
          <Text style={styles.errorIcon}>⚠️</Text>
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity
            style={styles.retryButton}
            onPress={() => selectedLevel && handleLevelPress(selectedLevel)}
          >
            <Text style={styles.retryButtonText}>Try Again</Text>
          </TouchableOpacity>
        </View>
      );
    }

    if (!currentHint) {
      return (
        <View style={styles.hintContentContainer}>
          <Text style={styles.placeholderIcon}>🎯</Text>
          <Text style={styles.placeholderText}>
            Select a hint level above to get puzzle-solving assistance
          </Text>
        </View>
      );
    }

    const levelInfo = HINT_LEVEL_INFO[currentHint.level];

    return (
      <View style={styles.hintContentContainer}>
        <View style={styles.hintHeader}>
          <Text style={styles.hintIcon}>{levelInfo.icon}</Text>
          <Text style={styles.hintTitle}>{levelInfo.title}</Text>
        </View>
        <Text style={styles.hintText}>{currentHint.content}</Text>

        {/* Render cell placements for Level 3/4 hints */}
        {currentHint.cells && currentHint.cells.length > 0 && (
          <View style={styles.cellsContainer}>
            <Text style={styles.cellsTitle}>Cell Placements:</Text>
            {currentHint.cells.map((cell, index) => (
              <View key={index} style={styles.cellItem}>
                <Text style={styles.cellCoord}>
                  R{cell.row + 1}C{cell.col + 1}
                </Text>
                <Text style={styles.cellValue}>{cell.value}</Text>
                {cell.region && <Text style={styles.cellRegion}>Region {cell.region}</Text>}
              </View>
            ))}
          </View>
        )}
      </View>
    );
  }, [loading, error, currentHint, selectedLevel, handleLevelPress]);

  return (
    <Modal visible={visible} animationType="slide" presentationStyle="pageSheet">
      <View style={styles.container}>
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.headerContent}>
            <Text style={styles.title}>Puzzle Hints</Text>
            <Text style={styles.subtitle}>
              Choose a hint level - more specific hints reveal more information
            </Text>
          </View>
          <TouchableOpacity style={styles.closeButton} onPress={handleClose}>
            <Text style={styles.closeText}>×</Text>
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.content} contentContainerStyle={styles.contentContainer}>
          {/* Hint Level Buttons */}
          <View style={styles.levelsSection}>
            <Text style={styles.sectionTitle}>Select Hint Level</Text>
            {([1, 2, 3, 4] as HintLevel[]).map(renderLevelButton)}
          </View>

          {/* Hint Content Display */}
          <View style={styles.hintSection}>
            <Text style={styles.sectionTitle}>Hint</Text>
            {renderHintContent()}
          </View>
        </ScrollView>

        {/* Footer with close button */}
        <View style={styles.footer}>
          <TouchableOpacity style={styles.footerButton} onPress={handleClose}>
            <Text style={styles.footerButtonText}>Close</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
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
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    padding: spacing[5],
    borderBottomWidth: 1,
    borderBottomColor: colors.surface.ash,
  },
  headerContent: {
    flex: 1,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.text.primary,
    marginBottom: spacing[1],
  },
  subtitle: {
    fontSize: 14,
    color: colors.text.secondary,
  },
  closeButton: {
    width: 32,
    height: 32,
    borderRadius: radii.full,
    backgroundColor: colors.surface.ash,
    alignItems: 'center',
    justifyContent: 'center',
  },
  closeText: {
    fontSize: 24,
    color: colors.text.primary,
    marginTop: -2,
  },
  content: {
    flex: 1,
  },
  contentContainer: {
    padding: spacing[5],
  },
  levelsSection: {
    marginBottom: spacing[6],
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.primary,
    marginBottom: spacing[3],
  },
  levelButton: {
    backgroundColor: colors.surface.slate,
    borderRadius: radii.lg,
    padding: spacing[4],
    marginBottom: spacing[3],
    borderWidth: 2,
    borderColor: 'transparent',
  },
  levelButtonSelected: {
    borderColor: colors.accent.brass,
    backgroundColor: colors.surface.graphite,
  },
  levelButtonRevealed: {
    opacity: 0.8,
  },
  levelButtonContent: {
    position: 'relative',
  },
  levelHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing[2],
  },
  levelIcon: {
    fontSize: 20,
    marginRight: spacing[2],
  },
  levelTitleContainer: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  levelTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.primary,
  },
  levelTitleSelected: {
    color: colors.accent.brass,
  },
  revealedBadge: {
    fontSize: 10,
    fontWeight: '600',
    color: colors.text.tertiary,
    backgroundColor: colors.surface.ash,
    paddingHorizontal: spacing[2],
    paddingVertical: spacing[1],
    borderRadius: radii.sm,
  },
  levelDescription: {
    fontSize: 13,
    color: colors.text.secondary,
    lineHeight: 18,
  },
  levelDescriptionSelected: {
    color: colors.text.primary,
  },
  levelLoading: {
    position: 'absolute',
    right: 0,
    top: 0,
  },
  hintSection: {
    flex: 1,
  },
  hintContentContainer: {
    backgroundColor: colors.surface.graphite,
    borderRadius: radii.lg,
    padding: spacing[5],
    minHeight: 150,
    alignItems: 'center',
    justifyContent: 'center',
  },
  placeholderIcon: {
    fontSize: 40,
    marginBottom: spacing[3],
  },
  placeholderText: {
    fontSize: 14,
    color: colors.text.tertiary,
    textAlign: 'center',
  },
  loadingText: {
    fontSize: 14,
    color: colors.text.secondary,
    marginTop: spacing[3],
  },
  errorIcon: {
    fontSize: 40,
    marginBottom: spacing[3],
  },
  errorText: {
    fontSize: 14,
    color: colors.semantic.coral,
    textAlign: 'center',
    marginBottom: spacing[3],
  },
  retryButton: {
    backgroundColor: colors.surface.slate,
    paddingHorizontal: spacing[4],
    paddingVertical: spacing[2],
    borderRadius: radii.md,
  },
  retryButtonText: {
    fontSize: 14,
    fontWeight: '500',
    color: colors.text.primary,
  },
  hintHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    marginBottom: spacing[3],
  },
  hintIcon: {
    fontSize: 24,
    marginRight: spacing[2],
  },
  hintTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: colors.accent.brass,
  },
  hintText: {
    fontSize: 16,
    color: colors.text.primary,
    lineHeight: 24,
    alignSelf: 'flex-start',
  },
  cellsContainer: {
    marginTop: spacing[4],
    alignSelf: 'stretch',
    backgroundColor: colors.surface.slate,
    borderRadius: radii.md,
    padding: spacing[3],
  },
  cellsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text.secondary,
    marginBottom: spacing[2],
  },
  cellItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing[2],
    borderBottomWidth: 1,
    borderBottomColor: colors.surface.ash,
  },
  cellCoord: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text.primary,
    width: 60,
  },
  cellValue: {
    fontSize: 18,
    fontWeight: '700',
    color: colors.accent.brass,
    width: 40,
    textAlign: 'center',
  },
  cellRegion: {
    fontSize: 12,
    color: colors.text.tertiary,
    flex: 1,
  },
  footer: {
    padding: spacing[4],
    borderTopWidth: 1,
    borderTopColor: colors.surface.ash,
  },
  footerButton: {
    backgroundColor: colors.surface.graphite,
    paddingVertical: spacing[4],
    borderRadius: radii.md,
    alignItems: 'center',
  },
  footerButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.primary,
  },
});

export default HintModal;
