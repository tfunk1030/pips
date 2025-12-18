import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, ActivityIndicator } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/RootNavigator';
import PuzzleGrid from '../components/PuzzleGrid';
import { solvePuzzle } from '../../solver/solver';
import { validateSolution } from '../../validator/validateSolution';
import { SolveResult } from '../../model/types';


type Props = NativeStackScreenProps<RootStackParamList, 'Solve'>;

export default function SolveScreen({ route, navigation }: Props) {
  const { puzzle, sourceText } = route.params;
  const [result, setResult] = useState<SolveResult | null>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    const run = async () => {
      setRunning(true);
      const res = await solvePuzzle(puzzle, { allowDuplicates: puzzle.allowDuplicates ?? false, findAll: false, maxPip: puzzle.maxPip ?? 6, progressInterval: 2500 }, (progress) => {
        setResult((prev) => (prev ? { ...prev, stats: { ...prev.stats, nodes: progress.nodes, backtracks: progress.backtracks } } : prev));
      });
      const validationReport = res.solution ? validateSolution(puzzle, res.solution) : res.validationReport;
      setResult({ ...res, validationReport });
      setRunning(false);
    };
    run();
  }, [puzzle]);

  const stats = result?.stats;

  return (
    <ScrollView style={styles.container} contentContainerStyle={{ paddingBottom: 24 }}>
      <Text style={styles.header}>Solving…</Text>
      {running && <ActivityIndicator color="#4a60c4" />}
      {result && (
        <>
          <Text style={styles.status}>Status: {result.status}</Text>
          {result.solution && <PuzzleGrid puzzle={puzzle} solution={result.solution} />}
          <Text style={styles.section}>Stats</Text>
          {stats && (
            <Text style={styles.text}>
              nodes={stats.nodes} backtracks={stats.backtracks} prunes={stats.prunes} time={stats.elapsedMs.toFixed(1)}ms
            </Text>
          )}
          <Text style={styles.section}>Validation</Text>
          {result.validationReport.issues.map((issue, idx) => (
            <Text key={idx} style={{ color: issue.level === 'error' ? '#f78c6c' : '#9aa5ce' }}>
              • {issue.message}
            </Text>
          ))}
          {result.explanation && (
            <Text style={{ color: '#f78c6c', marginTop: 8 }}>Reason: {result.explanation}</Text>
          )}
          <TouchableOpacity style={styles.secondary} onPress={() => navigation.navigate('Viewer', { puzzle, sourceText })}>
            <Text style={styles.secondaryText}>Back</Text>
          </TouchableOpacity>
        </>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0c1021', padding: 16 },
  header: { color: '#fff', fontSize: 20, marginBottom: 8 },
  status: { color: '#e6e6e6', marginBottom: 8 },
  section: { color: '#9aa5ce', marginTop: 16, marginBottom: 4 },
  text: { color: '#e6e6e6' },
  secondary: { marginTop: 16, borderWidth: 1, borderColor: '#4a60c4', padding: 12, borderRadius: 8 },
  secondaryText: { color: '#4a60c4', textAlign: 'center', fontWeight: 'bold' },
});
