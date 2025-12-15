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
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { normalizePuzzle } from '../../model/normalize';
import { SolverProgress, StoredPuzzle, ValidationResult } from '../../model/types';
import { solvePuzzleAsync } from '../../solver/solver';
import { getPuzzle, getSettings, updatePuzzleSolution } from '../../storage/puzzles';
import { validateSolution } from '../../validator/validateSolution';
import GridRenderer from '../components/GridRenderer';

export default function SolveScreen({ route, navigation }: any) {
  const { puzzleId } = route.params;
  const [puzzle, setPuzzle] = useState<StoredPuzzle | null>(null);
  const [solving, setSolving] = useState(false);
  const [progress, setProgress] = useState<SolverProgress | null>(null);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [showValidation, setShowValidation] = useState(false);
  const cancelSignal = useRef({ cancelled: false });

  useEffect(() => {
    loadPuzzle();
  }, [puzzleId]);

  const loadPuzzle = async () => {
    const loaded = await getPuzzle(puzzleId);
    if (loaded) {
      setPuzzle(loaded);
      if (loaded.solution) {
        // Validate existing solution
        const normalized = normalizePuzzle(loaded.spec);
        const result = validateSolution(normalized, loaded.solution);
        setValidation(result);
      }
    } else {
      Alert.alert('Error', 'Puzzle not found');
      navigation.goBack();
    }
  };

  const handleSolve = async () => {
    if (!puzzle) return;

    setSolving(true);
    setProgress(null);
    setValidation(null);
    cancelSignal.current = { cancelled: false };

    try {
      const settings = await getSettings();
      const normalized = normalizePuzzle(puzzle.spec);

      const config = {
        maxPip: puzzle.spec.maxPip || settings.defaultMaxPip,
        allowDuplicates: puzzle.spec.allowDuplicates || settings.defaultAllowDuplicates,
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

        // Validate solution
        const validationResult = validateSolution(normalized, solution);
        setValidation(validationResult);

        if (validationResult.valid) {
          // Save solution
          await updatePuzzleSolution(puzzle.id, solution);

          // Update local state
          setPuzzle({ ...puzzle, solution, solved: true });

          Alert.alert(
            'Success!',
            `Puzzle solved in ${solution.stats.timeMs}ms\n\nNodes: ${solution.stats.nodes}\nBacktracks: ${solution.stats.backtracks}\nPrunes: ${solution.stats.prunes}`
          );
        } else {
          Alert.alert(
            'Validation Failed',
            `Solver returned an invalid solution!\n\nErrors:\n${validationResult.errors.join(
              '\n'
            )}`,
            [{ text: 'OK' }]
          );
        }
      } else {
        Alert.alert(
          'No Solution',
          `Puzzle is unsatisfiable.\n\n${
            result.explanation.message
          }\n\n${result.explanation.details.join('\n')}`,
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

  const handleCancel = () => {
    cancelSignal.current.cancelled = true;
  };

  if (!puzzle || !puzzle.spec) {
    return (
      <View style={styles.container}>
        <Text>Loading...</Text>
      </View>
    );
  }

  const normalized = normalizePuzzle(puzzle.spec);

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Solve: {puzzle.name}</Text>
        <View style={styles.placeholder} />
      </View>

      <ScrollView style={styles.content}>
        {/* Grid View */}
        <View style={styles.gridContainer}>
          <GridRenderer puzzle={normalized} solution={puzzle.solution} />
        </View>

        {/* Progress Display */}
        {solving && progress && (
          <View style={styles.progressCard}>
            <Text style={styles.progressTitle}>Solving...</Text>
            <Text style={styles.progressText}>Nodes: {progress.nodes}</Text>
            <Text style={styles.progressText}>Backtracks: {progress.backtracks}</Text>
            <Text style={styles.progressText}>Prunes: {progress.prunes}</Text>
            <Text style={styles.progressText}>Depth: {progress.currentDepth}</Text>
            <ActivityIndicator size="large" color="#007AFF" style={styles.spinner} />
          </View>
        )}

        {/* Validation Report */}
        {validation && (
          <View style={styles.validationCard}>
            <TouchableOpacity
              style={styles.validationHeader}
              onPress={() => setShowValidation(!showValidation)}
            >
              <Text style={styles.validationTitle}>
                Validation Report {validation.valid ? '✓' : '✗'}
              </Text>
              <Text style={styles.expandIcon}>{showValidation ? '▼' : '▶'}</Text>
            </TouchableOpacity>

            {showValidation && (
              <View style={styles.validationContent}>
                {validation.errors.length > 0 && (
                  <View style={styles.errorSection}>
                    <Text style={styles.errorTitle}>Errors:</Text>
                    {validation.errors.map((error, i) => (
                      <Text key={i} style={styles.errorText}>
                        • {error}
                      </Text>
                    ))}
                  </View>
                )}

                {validation.regionChecks && validation.regionChecks.length > 0 && (
                  <View style={styles.checksSection}>
                    <Text style={styles.checksTitle}>Region Checks:</Text>
                    {validation.regionChecks.map((check, i) => (
                      <Text key={i} style={check.valid ? styles.checkValid : styles.checkInvalid}>
                        {check.message}
                      </Text>
                    ))}
                  </View>
                )}

                {validation.dominoChecks && validation.dominoChecks.length > 0 && (
                  <View style={styles.checksSection}>
                    <Text style={styles.checksTitle}>
                      Domino Checks: {validation.dominoChecks.length} total
                    </Text>
                    {validation.dominoChecks.filter(c => !c.valid).length > 0 ? (
                      validation.dominoChecks
                        .filter(c => !c.valid)
                        .map((check, i) => (
                          <Text key={i} style={styles.checkInvalid}>
                            {check.message}
                          </Text>
                        ))
                    ) : (
                      <Text style={styles.checkValid}>All dominoes valid ✓</Text>
                    )}
                  </View>
                )}
              </View>
            )}
          </View>
        )}
      </ScrollView>

      {/* Action Buttons */}
      <View style={styles.footer}>
        {solving ? (
          <TouchableOpacity style={styles.cancelButton} onPress={handleCancel}>
            <Text style={styles.cancelButtonText}>Cancel</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity style={styles.solveButton} onPress={handleSolve}>
            <Text style={styles.solveButtonText}>{puzzle.solved ? 'Re-Solve' : 'Solve'}</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    paddingTop: 60,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
  },
  backButton: {
    padding: 8,
  },
  backButtonText: {
    fontSize: 16,
    color: '#007AFF',
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    flex: 1,
    textAlign: 'center',
  },
  placeholder: {
    width: 70,
  },
  content: {
    flex: 1,
  },
  gridContainer: {
    backgroundColor: '#fff',
    margin: 16,
    borderRadius: 8,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    height: 400,
  },
  progressCard: {
    backgroundColor: '#E3F2FD',
    padding: 16,
    margin: 16,
    marginTop: 0,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#2196F3',
  },
  progressTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1976D2',
    marginBottom: 12,
  },
  progressText: {
    fontSize: 14,
    color: '#1976D2',
    marginBottom: 4,
  },
  spinner: {
    marginTop: 12,
  },
  validationCard: {
    backgroundColor: '#fff',
    margin: 16,
    marginTop: 0,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  validationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
  },
  validationTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  expandIcon: {
    fontSize: 16,
    color: '#666',
  },
  validationContent: {
    paddingHorizontal: 16,
    paddingBottom: 16,
  },
  errorSection: {
    marginBottom: 12,
  },
  errorTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#F44336',
    marginBottom: 8,
  },
  errorText: {
    fontSize: 14,
    color: '#F44336',
    marginBottom: 4,
  },
  checksSection: {
    marginBottom: 12,
  },
  checksTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  checkValid: {
    fontSize: 14,
    color: '#4CAF50',
    marginBottom: 4,
  },
  checkInvalid: {
    fontSize: 14,
    color: '#F44336',
    marginBottom: 4,
  },
  footer: {
    padding: 16,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#ddd',
  },
  solveButton: {
    backgroundColor: '#4CAF50',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  solveButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  cancelButton: {
    backgroundColor: '#F44336',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  cancelButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
});
