import * as React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import type { SolveResult } from '../../solver/solver';
import { theme } from '../theme';

export function SolverStatsView({ result }: { result: SolveResult }) {
  const stats = (result as any).stats as { nodes: number; backtracks: number; prunes: number; timeMs: number; yields: number } | undefined;
  if (!stats) return null;

  return (
    <View style={styles.card}>
      <Text style={styles.title}>Solver</Text>
      <Text style={styles.row}>status: {result.kind}</Text>
      <Text style={styles.row}>nodes: {stats.nodes.toLocaleString()}</Text>
      <Text style={styles.row}>backtracks: {stats.backtracks.toLocaleString()}</Text>
      <Text style={styles.row}>prunes: {stats.prunes.toLocaleString()}</Text>
      <Text style={styles.row}>time: {stats.timeMs.toLocaleString()} ms</Text>
      <Text style={styles.row}>yields: {stats.yields.toLocaleString()}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: { marginTop: 12, backgroundColor: theme.colors.card, borderRadius: 12, padding: 12, borderWidth: 1, borderColor: theme.colors.border },
  title: { color: theme.colors.text, fontWeight: '900', fontSize: 16, marginBottom: 6 },
  row: { color: theme.colors.muted, fontSize: 12, marginTop: 2 },
});



