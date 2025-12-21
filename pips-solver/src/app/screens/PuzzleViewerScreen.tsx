/**
 * Puzzle Viewer Screen
 * Displays puzzle grid and allows navigation to solve
 */

import React, { useEffect, useState } from 'react';
import { Alert, ScrollView, StyleSheet, View } from 'react-native';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import { normalizePuzzle } from '../../model/normalize';
import { Cell, StoredPuzzle } from '../../model/types';
import { getPuzzle } from '../../storage/puzzles';
import { colors, spacing, radii, shadows } from '../../theme';
import { fontFamilies, textStyles } from '../../theme/fonts';
import { Button, Card, Badge, Heading, Body, Label, Mono, Screen } from '../components/ui';
import GridRenderer from '../components/GridRenderer';

export default function PuzzleViewerScreen({ route, navigation }: any) {
  const { puzzleId } = route.params;
  const [puzzle, setPuzzle] = useState<StoredPuzzle | null>(null);
  const [selectedCell, setSelectedCell] = useState<Cell | undefined>(undefined);

  useEffect(() => {
    loadPuzzle();
  }, [puzzleId]);

  const loadPuzzle = async () => {
    const loaded = await getPuzzle(puzzleId);
    if (loaded) {
      setPuzzle(loaded);
    } else {
      Alert.alert('Error', 'Puzzle not found');
      navigation.goBack();
    }
  };

  const handleCellPress = (cell: Cell) => {
    setSelectedCell(cell);
  };

  const getCellInfo = (cell: Cell) => {
    if (!puzzle) return null;

    const regionId = puzzle.spec.regions[cell.row][cell.col];
    if (regionId === -1) {
      return {
        position: `(${cell.row}, ${cell.col})`,
        regionId: '—',
        constraint: 'Hole (no cell)',
      };
    }

    const constraint = puzzle.spec.constraints[regionId];

    let constraintText = '';
    if (constraint) {
      if (constraint.sum !== undefined) {
        constraintText += `Sum = ${constraint.sum}`;
      } else if (constraint.op && constraint.value !== undefined) {
        constraintText += `Sum ${constraint.op} ${constraint.value}`;
      }
      if (constraint.all_equal) {
        constraintText += (constraintText ? ', ' : '') + 'All Equal';
      }
      if (constraint.all_different) {
        constraintText += (constraintText ? ', ' : '') + 'All Different';
      }
    }

    return {
      position: `(${cell.row}, ${cell.col})`,
      regionId,
      constraint: constraintText || 'None',
    };
  };

  if (!puzzle || !puzzle.spec) {
    return (
      <Screen>
        <View style={styles.loadingContainer}>
          <Body>Loading...</Body>
        </View>
      </Screen>
    );
  }

  const normalized = normalizePuzzle(puzzle.spec);
  const cellInfo = selectedCell ? getCellInfo(selectedCell) : null;

  return (
    <Screen>
      {/* Header */}
      <Animated.View entering={FadeInDown.duration(400)} style={styles.header}>
        <Button
          variant="ghost"
          size="small"
          title="Back"
          onPress={() => navigation.goBack()}
        />
        <Heading size="medium" style={styles.title}>{puzzle.name}</Heading>
        <View style={styles.placeholder} />
      </Animated.View>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {/* Puzzle Info Card */}
        <Animated.View entering={FadeInUp.delay(100).duration(400)}>
          <Card variant="elevated" style={styles.infoCard}>
            <View style={styles.cardHeader}>
              <Heading size="small">Puzzle Info</Heading>
              <Badge
                variant={puzzle.solved ? 'success' : 'info'}
                label={puzzle.solved ? 'Solved' : 'Unsolved'}
              />
            </View>
            <View style={styles.infoGrid}>
              <View style={styles.infoItem}>
                <Label size="small">Size</Label>
                <Mono>{puzzle.spec.rows}x{puzzle.spec.cols}</Mono>
              </View>
              <View style={styles.infoItem}>
                <Label size="small">Max Pip</Label>
                <Mono>{puzzle.spec.maxPip || 6}</Mono>
              </View>
              <View style={styles.infoItem}>
                <Label size="small">Duplicates</Label>
                <Mono>{puzzle.spec.allowDuplicates ? 'Yes' : 'No'}</Mono>
              </View>
            </View>
          </Card>
        </Animated.View>

        {/* Grid Display */}
        <Animated.View entering={FadeInUp.delay(200).duration(400)}>
          <Card variant="default" style={styles.gridCard}>
            <View style={styles.gridContainer}>
              <GridRenderer
                puzzle={normalized}
                solution={puzzle.solution}
                onCellPress={handleCellPress}
                highlightCell={selectedCell}
              />
            </View>
          </Card>
        </Animated.View>

        {/* Selected Cell Info */}
        {cellInfo && (
          <Animated.View entering={FadeInUp.duration(300)}>
            <Card variant="accent" style={styles.cellInfoCard}>
              <Heading size="small" style={styles.cellInfoTitle}>Selected Cell</Heading>
              <View style={styles.cellInfoGrid}>
                <View style={styles.cellInfoItem}>
                  <Label size="small">Position</Label>
                  <Mono>{cellInfo.position}</Mono>
                </View>
                <View style={styles.cellInfoItem}>
                  <Label size="small">Region</Label>
                  <Mono>{cellInfo.regionId}</Mono>
                </View>
              </View>
              <View style={styles.cellInfoConstraint}>
                <Label size="small">Constraint</Label>
                <Body size="small" color="secondary">{cellInfo.constraint}</Body>
              </View>
            </Card>
          </Animated.View>
        )}

        {/* Solution Stats */}
        {puzzle.solution && (
          <Animated.View entering={FadeInUp.delay(300).duration(400)}>
            <Card variant="elevated" style={styles.solutionCard}>
              <View style={styles.cardHeader}>
                <Heading size="small">Solution Stats</Heading>
                <Badge variant="success" label="Valid" />
              </View>
              <View style={styles.statsGrid}>
                <View style={styles.statItem}>
                  <Mono style={styles.statValue}>{puzzle.solution.dominoes.length}</Mono>
                  <Label size="small">Dominoes</Label>
                </View>
                <View style={styles.statItem}>
                  <Mono style={styles.statValue}>{puzzle.solution.stats.nodes}</Mono>
                  <Label size="small">Nodes</Label>
                </View>
                <View style={styles.statItem}>
                  <Mono style={styles.statValue}>{puzzle.solution.stats.backtracks}</Mono>
                  <Label size="small">Backtracks</Label>
                </View>
                <View style={styles.statItem}>
                  <Mono style={styles.statValue}>{puzzle.solution.stats.timeMs}ms</Mono>
                  <Label size="small">Time</Label>
                </View>
              </View>
            </Card>
          </Animated.View>
        )}

        {/* Bottom spacing */}
        <View style={{ height: spacing[6] }} />
      </ScrollView>

      {/* Footer Action */}
      <Animated.View entering={FadeInUp.delay(400).duration(400)} style={styles.footer}>
        <Button
          variant="primary"
          size="large"
          title={puzzle.solved ? 'Re-Solve Puzzle' : 'Solve Puzzle'}
          onPress={() => navigation.navigate('Solve', { puzzleId: puzzle.id })}
          style={styles.solveButton}
        />
      </Animated.View>
    </Screen>
  );
}

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing[4],
    paddingVertical: spacing[3],
    borderBottomWidth: 1,
    borderBottomColor: colors.surface.slate,
  },
  title: {
    flex: 1,
    textAlign: 'center',
  },
  placeholder: {
    width: 70,
  },
  content: {
    flex: 1,
    paddingHorizontal: spacing[4],
  },
  infoCard: {
    marginTop: spacing[4],
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing[4],
  },
  infoGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  infoItem: {
    alignItems: 'center',
    gap: spacing[1],
  },
  gridCard: {
    marginTop: spacing[4],
    padding: 0,
    overflow: 'hidden',
  },
  gridContainer: {
    height: 380,
    backgroundColor: colors.surface.charcoal,
    borderRadius: radii.lg,
  },
  cellInfoCard: {
    marginTop: spacing[4],
  },
  cellInfoTitle: {
    marginBottom: spacing[3],
  },
  cellInfoGrid: {
    flexDirection: 'row',
    gap: spacing[6],
    marginBottom: spacing[3],
  },
  cellInfoItem: {
    gap: spacing[1],
  },
  cellInfoConstraint: {
    gap: spacing[1],
  },
  solutionCard: {
    marginTop: spacing[4],
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statItem: {
    alignItems: 'center',
    gap: spacing[1],
  },
  statValue: {
    fontSize: 18,
    fontFamily: fontFamilies.monoMedium,
    color: colors.accent.brass,
  },
  footer: {
    padding: spacing[4],
    borderTopWidth: 1,
    borderTopColor: colors.surface.slate,
    backgroundColor: colors.surface.charcoal,
  },
  solveButton: {
    width: '100%',
  },
});
