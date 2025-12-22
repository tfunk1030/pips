/**
 * Solve Screen
 * Runs the solver and displays results with validation
 */

import React, { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  ScrollView,
  StyleSheet,
  Switch,
  TouchableOpacity,
  View,
} from 'react-native';
import Animated, { FadeInDown, FadeInUp, FadeIn } from 'react-native-reanimated';
import { normalizePuzzle } from '../../model/normalize';
import { Cell, SolverProgress, StoredPuzzle, ValidationResult } from '../../model/types';
import { solvePuzzleAsync } from '../../solver/solver';
import { getPuzzle, getSettings, updatePuzzleSolution } from '../../storage/puzzles';
import { validateSolution } from '../../validator/validateSolution';
import { colors, spacing, radii } from '../../theme';
import { fontFamilies } from '../../theme/fonts';
import { Button, Card, Badge, Heading, Body, Label, Mono, Screen } from '../components/ui';
import GridRenderer from '../components/GridRenderer';

export default function SolveScreen({ route, navigation }: any) {
  const { puzzleId } = route.params;
  const [puzzle, setPuzzle] = useState<StoredPuzzle | null>(null);
  const [solving, setSolving] = useState(false);
  const [progress, setProgress] = useState<SolverProgress | null>(null);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [showValidation, setShowValidation] = useState(false);
  const cancelSignal = useRef({ cancelled: false });
  const [stepByStep, setStepByStep] = useState(false);
  const [revealed, setRevealed] = useState<boolean[][]>([]);
  const [highlightCell, setHighlightCell] = useState<Cell | null>(null);

  useEffect(() => {
    loadPuzzle();
  }, [puzzleId]);

  const initRevealState = (loaded: StoredPuzzle) => {
    if (!loaded.solution) {
      setRevealed([]);
      return;
    }
    const rows = loaded.spec.rows;
    const cols = loaded.spec.cols;
    setRevealed(Array.from({ length: rows }, () => Array.from({ length: cols }, () => false)));
  };

  const loadPuzzle = async () => {
    const loaded = await getPuzzle(puzzleId);
    if (loaded) {
      setPuzzle(loaded);
      initRevealState(loaded);
      if (loaded.solution) {
        const normalized = normalizePuzzle(loaded.spec);
        const result = validateSolution(normalized, loaded.solution);
        setValidation(result);
      }
    } else {
      Alert.alert('Error', 'Puzzle not found');
      navigation.goBack();
    }
  };

  const doSolve = async (opts?: { ignoreTray?: boolean }) => {
    if (!puzzle) return;

    setSolving(true);
    setProgress(null);
    setValidation(null);
    cancelSignal.current = { cancelled: false };

    try {
      const settings = await getSettings();
      const requiredDominoes = Math.floor(
        puzzle.spec.regions.flat().filter(regionId => regionId !== -1).length / 2
      );

      const specForSolve = opts?.ignoreTray
        ? { ...puzzle.spec, dominoes: undefined, allowDuplicates: true }
        : puzzle.spec;

      const normalized = normalizePuzzle(specForSolve);

      console.log(
        '[SOLVE] Puzzle spec:',
        JSON.stringify(
          {
            rows: specForSolve.rows,
            cols: specForSolve.cols,
            regions: specForSolve.regions,
            constraints: specForSolve.constraints,
            dominoes: specForSolve.dominoes,
          },
          null,
          2
        )
      );
      console.log('[SOLVE] Normalized regions:', normalized.regionCells.size, 'regions');
      console.log('[SOLVE] Normalized edges:', normalized.edges.length, 'edges');

      const config = {
        maxPip: puzzle.spec.maxPip || settings.defaultMaxPip,
        allowDuplicates:
          (opts?.ignoreTray ? true : puzzle.spec.allowDuplicates) ||
          settings.defaultAllowDuplicates,
        findAll: settings.defaultFindAll,
        maxIterationsPerTick: settings.maxIterationsPerTick,
        debugLevel: settings.defaultDebugLevel as 0 | 1 | 2,
      };

      const result = await solvePuzzleAsync(
        normalized,
        config,
        prog => setProgress(prog),
        cancelSignal.current
      );

      if (result.success && result.solutions.length > 0) {
        const solution = result.solutions[0];
        const validationResult = validateSolution(normalized, solution);
        setValidation(validationResult);

        if (validationResult.valid) {
          await updatePuzzleSolution(puzzle.id, solution);
          const updated = { ...puzzle, solution, solved: true };
          setPuzzle(updated);
          initRevealState(updated);

          Alert.alert(
            'Success!',
            `Puzzle solved in ${solution.stats.timeMs}ms\n\nNodes: ${solution.stats.nodes}\nBacktracks: ${solution.stats.backtracks}\nPrunes: ${solution.stats.prunes}`
          );
        } else {
          Alert.alert(
            'Validation Failed',
            `Solver returned an invalid solution!\n\nErrors:\n${validationResult.errors.join('\n')}`,
            [{ text: 'OK' }]
          );
        }
      } else {
        Alert.alert(
          'No Solution',
          `Puzzle is unsatisfiable.\n\n${result.explanation.message}\n\n${result.explanation.details.join('\n')}`,
          [{ text: 'OK' }]
        );
      }
    } catch (error) {
      Alert.alert('Error', `Solver error: ${error}`);
    } finally {
      setSolving(false);
      setProgress(null);
    }
  };

  const handleSolve = async () => {
    if (!puzzle) return;

    const requiredDominoes = Math.floor(
      puzzle.spec.regions.flat().filter(regionId => regionId !== -1).length / 2
    );
    const trayCount = puzzle.spec.dominoes?.length;

    if (trayCount !== undefined && trayCount !== requiredDominoes) {
      Alert.alert(
        'Domino tray incomplete',
        `This puzzle has ${requiredDominoes} domino slots (cells/2) but only ${trayCount} tray dominoes were entered.\n\nYou can still solve by ignoring the tray constraint (duplicates allowed), or go back and enter the full tray.`,
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Solve ignoring tray', onPress: () => doSolve({ ignoreTray: true }) },
          { text: 'Solve anyway', onPress: () => doSolve() },
        ]
      );
      return;
    }

    await doSolve();
  };

  const handleCancel = () => {
    cancelSignal.current.cancelled = true;
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

  const effectiveSolution = (() => {
    if (!puzzle.solution) return undefined;
    if (!stepByStep) return puzzle.solution;

    const gridPips = puzzle.solution.gridPips.map((row, r) =>
      row.map((v, c) => {
        if (normalized.spec.regions[r]?.[c] === -1) return null;
        return revealed[r]?.[c] ? v : null;
      })
    );

    return { ...puzzle.solution, gridPips };
  })();

  const revealCell = (cell: Cell) => {
    if (!puzzle.solution) return;
    if (normalized.spec.regions[cell.row]?.[cell.col] === -1) return;

    setRevealed(prev => {
      const next =
        prev.length === normalized.spec.rows
          ? prev.map(r => [...r])
          : Array.from({ length: normalized.spec.rows }, () =>
              Array.from({ length: normalized.spec.cols }, () => false)
            );
      next[cell.row][cell.col] = !next[cell.row][cell.col];
      return next;
    });

    setHighlightCell(cell);
    setTimeout(() => setHighlightCell(null), 600);
  };

  const revealNext = () => {
    if (!puzzle.solution) return;
    for (let r = 0; r < normalized.spec.rows; r++) {
      for (let c = 0; c < normalized.spec.cols; c++) {
        if (normalized.spec.regions[r]?.[c] === -1) continue;
        if (!revealed[r]?.[c]) {
          revealCell({ row: r, col: c });
          return;
        }
      }
    }
  };

  const revealAll = () => {
    if (!puzzle.solution) return;
    setRevealed(
      Array.from({ length: normalized.spec.rows }, () =>
        Array.from({ length: normalized.spec.cols }, () => true)
      )
    );
    setHighlightCell(null);
  };

  const resetReveal = () => {
    if (!puzzle.solution) return;
    initRevealState(puzzle);
    setHighlightCell(null);
  };

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
        <Heading size="small" style={styles.title}>Solve: {puzzle.name}</Heading>
        <View style={styles.placeholder} />
      </Animated.View>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {/* Grid View */}
        <Animated.View entering={FadeInUp.delay(100).duration(400)}>
          <Card variant="default" style={styles.gridCard}>
            <View style={styles.gridContainer}>
              <GridRenderer
                puzzle={normalized}
                solution={effectiveSolution}
                onCellPress={stepByStep ? revealCell : undefined}
                highlightCell={highlightCell || undefined}
              />
            </View>
          </Card>
        </Animated.View>

        {/* Step by Step Controls */}
        {puzzle.solution && (
          <Animated.View entering={FadeInUp.delay(200).duration(400)}>
            <Card variant="outlined" style={styles.stepCard}>
              <View style={styles.stepRow}>
                <Label size="medium">Step-by-step reveal</Label>
                <Switch
                  value={stepByStep}
                  onValueChange={setStepByStep}
                  trackColor={{ false: colors.surface.graphite, true: colors.accent.brass }}
                  thumbColor={stepByStep ? colors.accent.brassLight : colors.surface.ash}
                />
              </View>

              {stepByStep && (
                <Animated.View entering={FadeIn.duration(300)}>
                  <View style={styles.stepButtons}>
                    <Button
                      variant="secondary"
                      size="small"
                      title="Reveal Next"
                      onPress={revealNext}
                      style={styles.stepButton}
                    />
                    <Button
                      variant="secondary"
                      size="small"
                      title="Reveal All"
                      onPress={revealAll}
                      style={styles.stepButton}
                    />
                    <Button
                      variant="ghost"
                      size="small"
                      title="Reset"
                      onPress={resetReveal}
                      style={styles.stepButton}
                    />
                  </View>
                  <Body size="small" color="tertiary" style={styles.stepHint}>
                    Tap a cell to reveal/hide its value
                  </Body>
                </Animated.View>
              )}
            </Card>
          </Animated.View>
        )}

        {/* Progress Display */}
        {solving && progress && (
          <Animated.View entering={FadeIn.duration(300)}>
            <Card variant="accent" style={styles.progressCard}>
              <View style={styles.progressHeader}>
                <Heading size="small" style={styles.progressTitle}>Solving...</Heading>
                <ActivityIndicator size="small" color={colors.accent.brass} />
              </View>
              <View style={styles.progressStats}>
                <View style={styles.progressStat}>
                  <Mono style={styles.progressValue}>{progress.nodes}</Mono>
                  <Label size="small">Nodes</Label>
                </View>
                <View style={styles.progressStat}>
                  <Mono style={styles.progressValue}>{progress.backtracks}</Mono>
                  <Label size="small">Backtracks</Label>
                </View>
                <View style={styles.progressStat}>
                  <Mono style={styles.progressValue}>{progress.prunes}</Mono>
                  <Label size="small">Prunes</Label>
                </View>
                <View style={styles.progressStat}>
                  <Mono style={styles.progressValue}>{progress.currentDepth}</Mono>
                  <Label size="small">Depth</Label>
                </View>
              </View>
            </Card>
          </Animated.View>
        )}

        {/* Validation Report */}
        {validation && (
          <Animated.View entering={FadeInUp.delay(300).duration(400)}>
            <Card variant="elevated" style={styles.validationCard}>
              <TouchableOpacity
                style={styles.validationHeader}
                onPress={() => setShowValidation(!showValidation)}
              >
                <View style={styles.validationTitleRow}>
                  <Heading size="small">Validation Report</Heading>
                  <Badge
                    variant={validation.valid ? 'success' : 'error'}
                    label={validation.valid ? 'Valid' : 'Invalid'}
                  />
                </View>
                <Mono style={styles.expandIcon}>{showValidation ? '▼' : '▶'}</Mono>
              </TouchableOpacity>

              {showValidation && (
                <Animated.View entering={FadeIn.duration(200)} style={styles.validationContent}>
                  {validation.errors.length > 0 && (
                    <View style={styles.errorSection}>
                      <Label size="medium" style={styles.errorTitle}>Errors</Label>
                      {validation.errors.map((error, i) => (
                        <Body key={i} size="small" style={styles.errorText}>
                          • {error}
                        </Body>
                      ))}
                    </View>
                  )}

                  {validation.regionChecks && validation.regionChecks.length > 0 && (
                    <View style={styles.checksSection}>
                      <Label size="medium">Region Checks</Label>
                      {validation.regionChecks.map((check, i) => (
                        <Body
                          key={i}
                          size="small"
                          style={check.valid ? styles.checkValid : styles.checkInvalid}
                        >
                          {check.message}
                        </Body>
                      ))}
                    </View>
                  )}

                  {validation.dominoChecks && validation.dominoChecks.length > 0 && (
                    <View style={styles.checksSection}>
                      <Label size="medium">
                        Domino Checks ({validation.dominoChecks.length})
                      </Label>
                      {validation.dominoChecks.filter(c => !c.valid).length > 0 ? (
                        validation.dominoChecks
                          .filter(c => !c.valid)
                          .map((check, i) => (
                            <Body key={i} size="small" style={styles.checkInvalid}>
                              {check.message}
                            </Body>
                          ))
                      ) : (
                        <Body size="small" style={styles.checkValid}>All dominoes valid</Body>
                      )}
                    </View>
                  )}
                </Animated.View>
              )}
            </Card>
          </Animated.View>
        )}

        {/* Bottom spacing */}
        <View style={{ height: spacing[6] }} />
      </ScrollView>

      {/* Footer Action */}
      <Animated.View entering={FadeInUp.delay(400).duration(400)} style={styles.footer}>
        {solving ? (
          <Button
            variant="danger"
            size="large"
            title="Cancel"
            onPress={handleCancel}
            style={styles.actionButton}
          />
        ) : (
          <Button
            variant="success"
            size="large"
            title={puzzle.solved ? 'Re-Solve' : 'Solve Puzzle'}
            onPress={handleSolve}
            style={styles.actionButton}
          />
        )}
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
  stepCard: {
    marginTop: spacing[4],
  },
  stepRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  stepButtons: {
    flexDirection: 'row',
    gap: spacing[2],
    marginTop: spacing[3],
  },
  stepButton: {
    flex: 1,
  },
  stepHint: {
    marginTop: spacing[3],
    textAlign: 'center',
  },
  progressCard: {
    marginTop: spacing[4],
  },
  progressHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing[4],
  },
  progressTitle: {
    color: colors.accent.brass,
  },
  progressStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  progressStat: {
    alignItems: 'center',
    gap: spacing[1],
  },
  progressValue: {
    fontSize: 18,
    fontFamily: fontFamilies.monoMedium,
    color: colors.text.primary,
  },
  validationCard: {
    marginTop: spacing[4],
  },
  validationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  validationTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing[3],
  },
  expandIcon: {
    color: colors.text.tertiary,
  },
  validationContent: {
    marginTop: spacing[4],
    paddingTop: spacing[4],
    borderTopWidth: 1,
    borderTopColor: colors.surface.slate,
  },
  errorSection: {
    marginBottom: spacing[4],
  },
  errorTitle: {
    color: colors.semantic.coral,
    marginBottom: spacing[2],
  },
  errorText: {
    color: colors.semantic.coral,
    marginBottom: spacing[1],
  },
  checksSection: {
    marginBottom: spacing[4],
  },
  checkValid: {
    color: colors.semantic.jade,
    marginTop: spacing[1],
  },
  checkInvalid: {
    color: colors.semantic.coral,
    marginTop: spacing[1],
  },
  footer: {
    padding: spacing[4],
    borderTopWidth: 1,
    borderTopColor: colors.surface.slate,
    backgroundColor: colors.surface.charcoal,
  },
  actionButton: {
    width: '100%',
  },
});
