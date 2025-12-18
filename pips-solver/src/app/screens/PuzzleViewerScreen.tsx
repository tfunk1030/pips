/**
 * Puzzle Viewer Screen
 * Displays puzzle grid and allows navigation to solve
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Alert,
} from 'react-native';
import { StoredPuzzle, Cell } from '../../model/types';
import { getPuzzle } from '../../storage/puzzles';
import { normalizePuzzle } from '../../model/normalize';
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
    const constraint = puzzle.spec.constraints[regionId];

    let constraintText = '';
    if (constraint) {
      if (constraint.sum !== undefined) {
        constraintText += `Sum: ${constraint.sum}`;
      }
      if (constraint.all_equal) {
        constraintText += (constraintText ? ', ' : '') + 'All Equal';
      }
      if (constraint.op && constraint.value !== undefined) {
        constraintText += (constraintText ? ', ' : '') + `All ${constraint.op} ${constraint.value}`;
      }
    }

    return {
      position: `(${cell.row}, ${cell.col})`,
      regionId,
      constraint: constraintText || 'None',
    };
  };

  if (!puzzle) {
    return (
      <View style={styles.container}>
        <Text>Loading...</Text>
      </View>
    );
  }

  const normalized = normalizePuzzle(puzzle.spec);
  const cellInfo = selectedCell ? getCellInfo(selectedCell) : null;

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.title}>{puzzle.name}</Text>
        <View style={styles.placeholder} />
      </View>

      <ScrollView style={styles.content}>
        <View style={styles.infoCard}>
          <Text style={styles.infoTitle}>Puzzle Info</Text>
          <Text style={styles.infoText}>Size: {puzzle.spec.rows}x{puzzle.spec.cols}</Text>
          <Text style={styles.infoText}>Max Pip: {puzzle.spec.maxPip || 6}</Text>
          <Text style={styles.infoText}>
            Duplicates: {puzzle.spec.allowDuplicates ? 'Allowed' : 'Not Allowed'}
          </Text>
          <Text style={styles.infoText}>
            Status: {puzzle.solved ? 'Solved ✓' : 'Unsolved'}
          </Text>
        </View>

        <View style={styles.gridContainer}>
          <GridRenderer
            puzzle={normalized}
            solution={puzzle.solution}
            onCellPress={handleCellPress}
            highlightCell={selectedCell}
          />
        </View>

        {cellInfo && (
          <View style={styles.cellInfoCard}>
            <Text style={styles.cellInfoTitle}>Selected Cell Info</Text>
            <Text style={styles.cellInfoText}>Position: {cellInfo.position}</Text>
            <Text style={styles.cellInfoText}>Region: {cellInfo.regionId}</Text>
            <Text style={styles.cellInfoText}>Constraint: {cellInfo.constraint}</Text>
          </View>
        )}

        {puzzle.solution && (
          <View style={styles.solutionCard}>
            <Text style={styles.solutionTitle}>Solution</Text>
            <Text style={styles.solutionText}>
              Dominoes: {puzzle.solution.dominoes.length}
            </Text>
            <Text style={styles.solutionText}>
              Nodes Searched: {puzzle.solution.stats.nodes}
            </Text>
            <Text style={styles.solutionText}>
              Backtracks: {puzzle.solution.stats.backtracks}
            </Text>
            <Text style={styles.solutionText}>
              Time: {puzzle.solution.stats.timeMs}ms
            </Text>
          </View>
        )}
      </ScrollView>

      <View style={styles.footer}>
        <TouchableOpacity
          style={styles.solveButton}
          onPress={() => navigation.navigate('Solve', { puzzleId: puzzle.id })}
        >
          <Text style={styles.solveButtonText}>
            {puzzle.solved ? 'Re-Solve' : 'Solve Puzzle'}
          </Text>
        </TouchableOpacity>
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
    fontSize: 20,
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
  infoCard: {
    backgroundColor: '#fff',
    padding: 16,
    margin: 16,
    marginBottom: 0,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  infoTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
  },
  infoText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
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
  cellInfoCard: {
    backgroundColor: '#fff',
    padding: 16,
    margin: 16,
    marginTop: 0,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cellInfoTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  cellInfoText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  solutionCard: {
    backgroundColor: '#E8F5E9',
    padding: 16,
    margin: 16,
    marginTop: 0,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#4CAF50',
  },
  solutionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2E7D32',
    marginBottom: 8,
  },
  solutionText: {
    fontSize: 14,
    color: '#2E7D32',
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
});
